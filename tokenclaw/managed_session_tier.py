from __future__ import annotations

import copy
import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from tokenclaw.http_client import async_client

from tokenclaw.action_executor import ActionExecutor
from tokenclaw.managed_egress import (
    ManagedEgressBlocked,
    assert_managed_egress_safe,
    managed_egress_blocked_meta,
)
from tokenclaw.managed_mode import managed_product_mode
from tokenclaw.pricing import MODEL_ALIASES
from tokenclaw.recommendations import (
    _managed_headers,
    managed_auth_configured,
    managed_auth_source,
    managed_loopback_auth_allowed,
    recommendation_server_configured,
    recommendation_server_url,
    recommendation_timeout_seconds,
    recommendations_enabled,
)
from tokenclaw.router import HAIKU_DEFAULT, OPUS_DEFAULT, SONNET_DEFAULT


SESSION_TIER_PATH = "/v1/session-tier"
# Wire schemas the managed server (tokenclaw_server) enforces. The server's
# SessionTier request model is strict (extra="forbid") and still uses the
# agentflow.* wire vocabulary, so the request literal must match it exactly.
# Decisions are dual-accepted (agentflow.* is emitted today; tokenclaw.* is
# tolerated) to mirror the migration pattern used for policy-decision responses.
SESSION_TIER_REQUEST_SCHEMA = "agentflow.session_tier_request.v1"
SESSION_TIER_DECISION_SCHEMA = "agentflow.session_tier_decision.v1"
SESSION_TIER_DECISION_SCHEMAS = (
    "agentflow.session_tier_decision.v1",
    "tokenclaw.session_tier_decision.v1",
)
SESSION_TIER_ENABLED_ENV = "TOKENCLAW_SESSION_TIER_ENABLED"
SESSION_TIER_CANARY_SALT_ENV = "TOKENCLAW_SESSION_TIER_CANARY_SALT"

_SESSION_TIER_CACHE: dict[str, dict[str, Any]] = {}


def clear_session_tier_cache() -> None:
    _SESSION_TIER_CACHE.clear()


def session_tier_enabled() -> bool:
    configured = os.getenv(SESSION_TIER_ENABLED_ENV)
    if configured is not None:
        return configured.strip().lower() not in {"", "0", "false", "no", "off"}
    return recommendations_enabled()


def count_tool_definitions(body: dict[str, Any]) -> int:
    tools = body.get("tools")
    if isinstance(tools, list):
        return len([tool for tool in tools if isinstance(tool, dict)])
    return 0


def _base_meta() -> dict[str, Any]:
    return {
        "schema": "tokenclaw.local_session_tier_decision_meta.v1",
        "endpoint": SESSION_TIER_PATH,
        "enabled": session_tier_enabled(),
        "auth_configured": managed_auth_configured(),
        "auth_source": managed_auth_source(),
        "loopback_unauthenticated_allowed": managed_loopback_auth_allowed(),
        "fallback": "local-policy",
        "applied": False,
        "raw_payload_included": False,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "session_ids_included": False,
        "metadata_only": True,
    }


def _target_model_for_tier(tier: str | None, decision: dict[str, Any]) -> str | None:
    explicit = decision.get("target_model") or decision.get("routed_model")
    if isinstance(explicit, str) and explicit:
        return MODEL_ALIASES.get(explicit, explicit)
    if tier == "haiku":
        return HAIKU_DEFAULT
    if tier == "sonnet":
        return SONNET_DEFAULT
    if tier == "opus":
        return OPUS_DEFAULT
    return None


def _target_model_for_canary(canary: dict[str, Any], *, fallback_tier: str | None) -> str | None:
    explicit = canary.get("target_model") or canary.get("routed_model")
    if isinstance(explicit, str) and explicit:
        return MODEL_ALIASES.get(explicit, explicit)
    target_tier = str(canary.get("target_tier") or fallback_tier or "").strip().lower()
    family = str(canary.get("target_model_family") or "").strip().lower()
    if target_tier == "haiku" or family == "claude-haiku":
        return HAIKU_DEFAULT
    if target_tier == "sonnet" or family == "claude-sonnet":
        return SONNET_DEFAULT
    if target_tier == "opus" or family == "claude-opus":
        return OPUS_DEFAULT
    return None


def _bounded_fraction(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return default
    return default


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expired(value: Any, *, now: datetime | None = None) -> bool:
    expires_at = _parse_time(value)
    if expires_at is None:
        return False
    return expires_at <= (now or datetime.now(timezone.utc))


def _canary_assignment(
    *,
    session_id: str | None,
    session_tier_meta: dict[str, Any],
    current_model: str,
) -> dict[str, Any] | None:
    canary = session_tier_meta.get("session_tier_canary")
    if not isinstance(canary, dict) or canary.get("status") != "recommended":
        return None
    existing = session_tier_meta.get("local_canary_assignment")
    if isinstance(existing, dict) and existing.get("treatment") in {"canary", "holdout", "hold", "veto"}:
        return dict(existing)
    target_model = _target_model_for_canary(canary, fallback_tier=session_tier_meta.get("tier"))
    if not target_model:
        return {
            "schema": "tokenclaw.local_session_tier_canary_assignment.v1",
            "status": "vetoed",
            "treatment": "veto",
            "reason": "missing-canary-target-model",
            "selected": False,
            "target_model": None,
            "metadata_only": True,
        }
    if _expired(canary.get("expires_at")):
        return {
            "schema": "tokenclaw.local_session_tier_canary_assignment.v1",
            "status": "vetoed",
            "treatment": "veto",
            "reason": "expired-session-tier-canary",
            "selected": False,
            "target_model": target_model,
            "metadata_only": True,
        }
    canary_fraction = _bounded_fraction(canary.get("canary_fraction"), 0.0)
    holdout_fraction = _bounded_fraction(canary.get("holdout_fraction"), 0.0)
    salt = os.getenv(SESSION_TIER_CANARY_SALT_ENV, "tokenclaw-session-tier-canary-v1")
    basis = {
        "session_id": session_id or "missing-session",
        "policy_id": canary.get("policy_id"),
        "cohort_bucket": canary.get("cohort_bucket"),
        "target_model": target_model,
        "current_model": current_model,
    }
    digest = hashlib.sha256(f"{salt}:{basis}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if bucket < holdout_fraction:
        treatment = "holdout"
        selected = False
        status = "heldout"
        reason = "session-tier-canary-holdout"
    elif bucket < min(1.0, holdout_fraction + canary_fraction):
        treatment = "canary"
        selected = True
        status = "selected"
        reason = "session-tier-canary-selected"
    else:
        treatment = "hold"
        selected = False
        status = "held"
        reason = "session-tier-canary-not-selected"
    return {
        "schema": "tokenclaw.local_session_tier_canary_assignment.v1",
        "status": status,
        "treatment": treatment,
        "reason": reason,
        "selected": selected,
        "bucket": round(bucket, 8),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "cohort_key_hash": f"sha256:{digest}",
        "target_model": target_model,
        "metadata_only": True,
        "raw_session_ids_included": False,
        "raw_prompts_included": False,
        "raw_responses_included": False,
    }


def _session_tier_payload(unit: dict[str, Any], *, tool_count: int) -> dict[str, Any]:
    input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
    tool_features = unit.get("tool_features") if isinstance(unit.get("tool_features"), dict) else {}
    grouping = unit.get("grouping_identifiers") if isinstance(unit.get("grouping_identifiers"), dict) else {}
    input_tokens = input_features.get("input_tokens_est")
    payload = {
        "schema": SESSION_TIER_REQUEST_SCHEMA,
        "source_surface": unit.get("source_surface"),
        "app_family": unit.get("app_family"),
        "requested_model": unit.get("requested_model"),
        "category": input_features.get("category") or tool_features.get("category"),
        "workflow_phase": input_features.get("workflow_phase") or tool_features.get("workflow_phase"),
        "text_chars": input_features.get("text_chars"),
        "input_tokens": input_tokens,
        "has_tools": bool(tool_features.get("has_tools")),
        "tool_count": max(0, int(tool_count or 0)),
        "grouping_identifiers": {
            "session_id_hash": grouping.get("session_id_hash"),
        },
    }
    return {
        key: value
        for key, value in payload.items()
        if value is not None and not (key == "grouping_identifiers" and not value.get("session_id_hash"))
    }


def _normalize_decision(body: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(body, dict):
        return None, "decision-not-object"
    if body.get("schema") not in SESSION_TIER_DECISION_SCHEMAS:
        return None, "unsupported-schema"
    if body.get("feature_only") is not True:
        return None, "feature-only-required"
    if body.get("locally_executed") is not True:
        return None, "local-execution-required"
    if body.get("provider_forwarding") is not False:
        return None, "provider-forwarding-not-allowed"
    if body.get("server_content_processing") is not False:
        return None, "server-content-processing-not-allowed"
    tier = body.get("tier")
    if tier not in {"haiku", "sonnet", "opus"}:
        return None, "unsupported-tier"
    target_model = _target_model_for_tier(tier, body)
    if not target_model:
        return None, "missing-target-model"
    reason_codes = body.get("reason_codes")
    if not isinstance(reason_codes, list):
        reason_codes = []
    canary = body.get("session_tier_canary")
    if not isinstance(canary, dict):
        canary = None
    omitted_actions = body.get("omitted_actions")
    if not isinstance(omitted_actions, list):
        omitted_actions = []
    return {
        "status": "received",
        "session_tier_source": body.get("session_tier_source") or "managed",
        "policy_source": "managed-recommended",
        "policy_id": (canary or {}).get("policy_id") or body.get("policy_id"),
        "decision_id": body.get("decision_id"),
        "tier": tier,
        "target_model": target_model,
        "confidence": body.get("confidence"),
        "session_type": body.get("session_type"),
        "reason_codes": [str(item) for item in reason_codes if isinstance(item, (str, int, float))],
        "session_tier_canary": copy.deepcopy(canary),
        "omitted_actions": [copy.deepcopy(item) for item in omitted_actions if isinstance(item, dict)],
        "hold_tier_for_session": bool(body.get("hold_tier_for_session", True)),
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "raw_payload_included": False,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "session_ids_included": False,
        "metadata_only": True,
    }, None


async def fetch_or_get_session_tier(
    unit: dict[str, Any],
    *,
    session_id: str | None,
    tool_count: int = 0,
) -> dict[str, Any]:
    meta = _base_meta()
    if not session_id:
        meta.update({"status": "skipped", "reason": "missing-session-id", "cache_status": "none"})
        return meta

    cached = _SESSION_TIER_CACHE.get(session_id)
    if cached is not None:
        result = copy.deepcopy(cached)
        result.update({"cache_status": "hit", "turn_role": "subsequent-turn"})
        return result

    if not session_tier_enabled():
        meta.update({"status": "skipped", "reason": "disabled", "cache_status": "miss"})
        return meta
    if not recommendation_server_configured():
        meta.update({"status": "skipped", "reason": "server-url-not-configured", "cache_status": "miss"})
        _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
        return meta
    if not managed_auth_configured():
        meta.update({"status": "skipped", "reason": "managed-auth-not-configured", "cache_status": "miss"})
        _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
        return meta

    try:
        payload = _session_tier_payload(unit, tool_count=tool_count)
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update(managed_egress_blocked_meta(endpoint=SESSION_TIER_PATH, violations=exc.violations))
        meta["cache_status"] = "miss"
        _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
        return meta

    started = time.time()
    try:
        async with async_client(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + SESSION_TIER_PATH,
                json=payload,
                headers=_managed_headers(),
            )
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        if response.status_code >= 400:
            meta.update({
                "status": "error",
                "reason": "server-error",
                "error": response.text[:500],
                "cache_status": "stored-error",
            })
            _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
            return meta
        try:
            body = response.json()
        except Exception as exc:
            meta.update({
                "status": "invalid",
                "reason": "invalid-json",
                "error": repr(exc),
                "cache_status": "stored-error",
            })
            _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
            return meta
        decision, error = _normalize_decision(body)
        if decision is None:
            meta.update({
                "status": "invalid",
                "reason": "invalid-schema",
                "schema_error": error,
                "cache_status": "stored-error",
            })
            _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
            return meta
        meta.update(decision)
        meta.update({"cache_status": "stored", "turn_role": "first-turn"})
        _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
        return meta
    except Exception as exc:
        meta.update({
            "status": "error",
            "reason": "fetch-error",
            "error": repr(exc),
            "cache_status": "stored-error",
        })
        _SESSION_TIER_CACHE[session_id] = copy.deepcopy(meta)
        return meta


def apply_session_tier_to_body(
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    session_tier_meta: dict[str, Any],
    *,
    session_id: str | None = None,
    stream: bool = False,
) -> str | None:
    routing_meta["managed_session_tier"] = session_tier_meta
    if session_tier_meta.get("session_tier_source") == "managed":
        routing_meta["session_tier_source"] = "managed"
        routing_meta["session_type"] = session_tier_meta.get("session_type")
        routing_meta["session_tier"] = session_tier_meta.get("tier")
        routing_meta["session_tier_confidence"] = session_tier_meta.get("confidence")
        routing_meta["session_tier_reason_codes"] = session_tier_meta.get("reason_codes") or []

    if session_tier_meta.get("status") != "received":
        session_tier_meta["applied"] = False
        session_tier_meta.setdefault("apply_reason", "no-managed-session-tier")
        session_tier_meta.setdefault("local_result", "noop")
        return None

    current_model = str(body.get("model") or routing_meta.get("routed_model") or routing_meta.get("requested_model") or "")
    assignment = _canary_assignment(
        session_id=session_id,
        session_tier_meta=session_tier_meta,
        current_model=current_model,
    )
    if assignment is not None:
        return _apply_session_tier_canary_to_body(
            body,
            routing_meta,
            session_tier_meta,
            assignment,
            session_id=session_id,
            stream=stream,
        )

    if session_tier_meta.get("cache_status") != "hit":
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "first-turn-classification-only"
        session_tier_meta["local_result"] = "held"
        return None

    target_model = session_tier_meta.get("target_model")
    if not isinstance(target_model, str) or not target_model:
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "missing-target-model"
        session_tier_meta["local_result"] = "vetoed"
        return None
    if (
        target_model == HAIKU_DEFAULT
        and isinstance(routing_meta.get("thinking_gate"), dict)
        and routing_meta["thinking_gate"].get("status") == "blocked"
    ):
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "local-thinking-safety-guard"
        session_tier_meta["local_result"] = "vetoed"
        return None

    body["model"] = target_model
    routing_meta["routed_model"] = target_model
    routing_meta["policy_source"] = "managed-recommended"
    session_tier_meta["applied"] = True
    session_tier_meta["changed_model"] = target_model != current_model
    session_tier_meta["apply_reason"] = "cached-session-tier"
    session_tier_meta["local_result"] = "applied" if session_tier_meta["changed_model"] else "noop"
    session_tier_meta["local_action_taken"] = "session_tier"
    return target_model


def _apply_session_tier_canary_to_body(
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    session_tier_meta: dict[str, Any],
    assignment: dict[str, Any],
    *,
    session_id: str | None,
    stream: bool,
) -> str | None:
    canary = session_tier_meta.get("session_tier_canary")
    if not isinstance(canary, dict):
        return None
    session_tier_meta["local_canary_assignment"] = assignment
    target_model = assignment.get("target_model")
    current_model = str(body.get("model") or routing_meta.get("routed_model") or routing_meta.get("requested_model") or "")
    phase_canary = {
        "enabled": True,
        "policy_id": canary.get("policy_id"),
        "policy_source": "managed-recommended",
        "target_model": target_model,
        "target_tier": canary.get("target_tier"),
        "category": routing_meta.get("category"),
        "cohort": canary.get("cohort_bucket"),
        "cohort_hash": assignment.get("cohort_key_hash"),
        "canary_fraction": assignment.get("canary_fraction"),
        "holdout_fraction": assignment.get("holdout_fraction"),
        "has_tools": routing_meta.get("has_tools"),
        "workflow_phase": routing_meta.get("workflow_phase"),
        "metadata_only": True,
    }
    routing_meta["phase_canary"] = phase_canary

    if not stream:
        session_tier_meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "session-tier-canary-streaming-only",
            "local_result": "held",
            "would_route_model": target_model,
        })
        phase_canary.update({"status": "holdout", "reason": "session-tier-canary-streaming-only"})
        _cache_session_tier_assignment(session_id, session_tier_meta)
        return None
    if assignment.get("treatment") == "veto":
        session_tier_meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": assignment.get("reason"),
            "local_result": "vetoed",
            "would_route_model": target_model,
        })
        phase_canary.update({"status": "safety_stopped", "reason": "safety-stop-tripped"})
        _cache_session_tier_assignment(session_id, session_tier_meta)
        return None
    if assignment.get("treatment") != "canary":
        local_result = "heldout" if assignment.get("treatment") == "holdout" else "held"
        session_tier_meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": assignment.get("reason"),
            "local_result": local_result,
            "would_route_model": target_model,
        })
        phase_canary.update({"status": "holdout", "reason": "selected-holdout"})
        routing_meta["local_result"] = local_result
        routing_meta["routing_outcome_label"] = local_result
        _cache_session_tier_assignment(session_id, session_tier_meta)
        return None
    if (
        target_model == HAIKU_DEFAULT
        and isinstance(routing_meta.get("thinking_gate"), dict)
        and routing_meta["thinking_gate"].get("status") == "blocked"
    ):
        session_tier_meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "local-thinking-safety-guard",
            "local_result": "vetoed",
            "would_route_model": target_model,
        })
        phase_canary.update({"status": "safety_stopped", "reason": "safety-stop-tripped"})
        routing_meta["local_result"] = "vetoed"
        routing_meta["routing_outcome_label"] = "vetoed"
        _cache_session_tier_assignment(session_id, session_tier_meta)
        return None

    decision = {
        "enabled": True,
        "status": "received",
        "policy_id": canary.get("policy_id") or session_tier_meta.get("policy_id"),
        "decision_id": session_tier_meta.get("decision_id"),
        "source_surface": "anthropic_messages",
        "target_model": target_model,
        "reason": "managed-session-tier-canary",
        "routing": {
            "status": "recommended",
            "target_model": target_model,
            "route_proposal": {
                "target_model": target_model,
                "traffic_treatment": "canary",
                "route_selected": True,
                "server_selected_canary_membership": True,
                "canary_fraction": assignment.get("canary_fraction"),
                "holdout_fraction": assignment.get("holdout_fraction"),
            },
        },
    }
    execution = ActionExecutor(provider="anthropic").execute(
        body=body,
        routing_meta=routing_meta,
        decision=decision,
        application_enabled=managed_product_mode().local_application_enabled,
        source_surface="anthropic_messages",
    )
    session_tier_meta["action_executor"] = execution
    session_tier_meta["local_actions"] = execution
    status = str(execution.get("status") or "held")
    changed_model = bool(execution.get("changed_model"))
    local_result = status if status in {"applied", "held", "heldout", "vetoed", "fallback", "noop"} else "held"
    session_tier_meta.update({
        "applied": bool(execution.get("applied") and changed_model),
        "changed_model": changed_model,
        "apply_reason": execution.get("apply_reason"),
        "fallback": execution.get("fallback"),
        "local_result": local_result,
        "would_route_model": target_model,
    })
    routing_meta["local_result"] = local_result
    routing_meta["routing_outcome_label"] = local_result
    if local_result == "applied":
        phase_canary.update({"status": "applied", "reason": "selected-canary"})
        session_tier_meta["local_action_taken"] = "session_tier_canary"
        routed = str(body.get("model") or target_model)
        _cache_session_tier_assignment(session_id, session_tier_meta)
        return routed if routed != current_model else None
    if local_result == "vetoed":
        phase_canary.update({"status": "safety_stopped", "reason": "safety-stop-tripped"})
    else:
        phase_canary.update({"status": "holdout", "reason": "selected-holdout"})
    _cache_session_tier_assignment(session_id, session_tier_meta)
    return None


def _cache_session_tier_assignment(session_id: str | None, session_tier_meta: dict[str, Any]) -> None:
    if not session_id:
        return
    cached = copy.deepcopy(session_tier_meta)
    cached["cache_status"] = "hit"
    cached["turn_role"] = "subsequent-turn"
    _SESSION_TIER_CACHE[session_id] = cached
