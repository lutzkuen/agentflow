from __future__ import annotations

import copy
import os
import time
from typing import Any

import httpx

from agentflow_proxy.managed_egress import (
    ManagedEgressBlocked,
    assert_managed_egress_safe,
    managed_egress_blocked_meta,
)
from agentflow_proxy.pricing import MODEL_ALIASES
from agentflow_proxy.recommendations import (
    _managed_headers,
    managed_auth_configured,
    managed_auth_source,
    managed_loopback_auth_allowed,
    recommendation_server_configured,
    recommendation_server_url,
    recommendation_timeout_seconds,
    recommendations_enabled,
)
from agentflow_proxy.router import HAIKU_DEFAULT, OPUS_DEFAULT, SONNET_DEFAULT


SESSION_TIER_PATH = "/v1/session-tier"
SESSION_TIER_REQUEST_SCHEMA = "agentflow.session_tier_request.v1"
SESSION_TIER_DECISION_SCHEMA = "agentflow.session_tier_decision.v1"
SESSION_TIER_ENABLED_ENV = "AGENTFLOW_SESSION_TIER_ENABLED"

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
        "schema": "agentflow.local_session_tier_decision_meta.v1",
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
        "phase": input_features.get("workflow_phase") or tool_features.get("workflow_phase"),
        "text_chars": input_features.get("text_chars"),
        "input_tokens": input_tokens,
        "input_tokens_est": input_tokens,
        "has_tools": bool(tool_features.get("has_tools")),
        "tool_count": max(0, int(tool_count or 0)),
        "grouping_identifiers": {
            "session_id_hash": grouping.get("session_id_hash"),
        },
        "privacy_summary": {
            "metadata_only": True,
            "raw_body_storage": False,
            "raw_payload_included": False,
            "provider_bodies_included": False,
            "raw_prompts_included": False,
            "session_ids_included": False,
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
    if body.get("schema") != SESSION_TIER_DECISION_SCHEMA:
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
    return {
        "status": "received",
        "session_tier_source": body.get("session_tier_source") or "managed",
        "policy_source": "managed-recommended",
        "tier": tier,
        "target_model": target_model,
        "confidence": body.get("confidence"),
        "session_type": body.get("session_type"),
        "reason_codes": [str(item) for item in reason_codes if isinstance(item, (str, int, float))],
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
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
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
        return None
    if session_tier_meta.get("cache_status") != "hit":
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "first-turn-classification-only"
        return None

    target_model = session_tier_meta.get("target_model")
    if not isinstance(target_model, str) or not target_model:
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "missing-target-model"
        return None
    if (
        target_model == HAIKU_DEFAULT
        and isinstance(routing_meta.get("thinking_gate"), dict)
        and routing_meta["thinking_gate"].get("status") == "blocked"
    ):
        session_tier_meta["applied"] = False
        session_tier_meta["apply_reason"] = "local-thinking-safety-guard"
        return None

    current_model = str(body.get("model") or routing_meta.get("routed_model") or routing_meta.get("requested_model") or "")
    body["model"] = target_model
    routing_meta["routed_model"] = target_model
    routing_meta["policy_source"] = "managed-recommended"
    session_tier_meta["applied"] = True
    session_tier_meta["changed_model"] = target_model != current_model
    session_tier_meta["apply_reason"] = "cached-session-tier"
    session_tier_meta["local_action_taken"] = "session_tier"
    return target_model
