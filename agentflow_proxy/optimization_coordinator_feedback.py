from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import utc_now


FEEDBACK_SCHEMA = "agentflow.optimization_coordinator_lifecycle_feedback.v1"
SOURCE_SURFACE = "optimization_coordinator_lifecycle"
RAW_REASON_HINTS = {
    "api",
    "apikey",
    "authorization",
    "body",
    "cache-key",
    "cache_key",
    "content",
    "file",
    "message",
    "path",
    "payload",
    "prompt",
    "provider-body",
    "provider_body",
    "request",
    "response",
    "secret",
    "session",
    "tenant",
    "thread",
    "tool-payload",
    "tool_payload",
}


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback)


def _safe_id(value: Any, *, prefix: str, fallback: str = "unknown") -> str:
    return public_id(value, prefix=prefix, fallback=fallback) or fallback


def _reason_codes(*values: Any) -> list[str]:
    codes: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                codes.extend(_reason_codes(item))
            continue
        text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
        if text:
            if any(hint.replace("_", "-") in text for hint in RAW_REASON_HINTS):
                public = public_id(text, prefix="reason", fallback="redacted-reason")
                codes.append(public or "redacted-reason")
            else:
                codes.append(public_label(text, "redacted-reason"))
    return sorted(set(codes))


def _status_bucket(status_code: Any) -> str:
    code = _as_int(status_code)
    if code is None or code <= 0:
        return "unknown"
    if code < 200:
        return "lt_2xx"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _retry_bucket(retry_count: Any) -> str:
    retries = _as_int(retry_count)
    if retries is None or retries <= 0:
        return "none"
    if retries == 1:
        return "1"
    if retries == 2:
        return "2"
    return "gte_3"


def _money_bucket(value: Any) -> str:
    amount = _as_float(value)
    if amount <= 0:
        return "none"
    if amount < 0.001:
        return "lt_0_001"
    if amount < 0.01:
        return "0_001_0_01"
    if amount < 0.10:
        return "0_01_0_10"
    return "gte_0_10"


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_responses_included": False,
        "raw_provider_bodies_included": False,
        "raw_tool_payloads_included": False,
        "raw_terminal_text_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "policy_file_contents_included": False,
        "secrets_included": False,
        "tenant_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _lifecycle_status_for_suppression(reasons: list[str]) -> str:
    reason_set = set(reasons)
    if "coordinator-holdout" in reason_set:
        return "holdout"
    if reason_set & {"rollback", "rollback-required"}:
        return "rollback"
    if reason_set & {"safety-stop", "safety-stopped", "safety-stop-tripped", "quality-regression"}:
        return "safety_stop"
    return "suppressed"


def _family_event_from_selected(selected: dict[str, Any], family: str) -> dict[str, Any]:
    event = {
        "action_family": _safe_label(family),
        "lifecycle_status": "selected",
        "cohort": "selected",
        "selected": True,
        "eligible": True,
        "policy_source": _safe_label(selected.get("policy_source"), "unknown"),
        "status": _safe_label(selected.get("status"), "unknown"),
        "candidate_id": _safe_id(selected.get("candidate_id") or selected.get("target_candidate_id"), prefix="candidate", fallback="unknown"),
        "rule_id": _safe_id(selected.get("rule_id") or selected.get("target_rule_id"), prefix="rule", fallback="unknown"),
        "action_id": _safe_id(selected.get("action_id") or selected.get("promotion_action_id"), prefix="action", fallback="unknown"),
        "reason_codes": _reason_codes(selected.get("reason_codes"), "coordinator-selected"),
    }
    return {key: value for key, value in event.items() if value not in (None, "", [], "unknown")}


def _family_event_from_suppressed(item: dict[str, Any]) -> dict[str, Any]:
    reasons = _reason_codes(item.get("reason_codes"), item.get("reason"), item.get("status"))
    event = {
        "action_family": _safe_label(item.get("family")),
        "lifecycle_status": _lifecycle_status_for_suppression(reasons),
        "cohort": _lifecycle_status_for_suppression(reasons),
        "selected": False,
        "eligible": True,
        "status": _safe_label(item.get("status"), "unknown"),
        "candidate_id": _safe_id(item.get("candidate_id") or item.get("target_candidate_id"), prefix="candidate", fallback="unknown"),
        "rule_id": _safe_id(item.get("rule_id") or item.get("target_rule_id"), prefix="rule", fallback="unknown"),
        "action_id": _safe_id(item.get("action_id") or item.get("promotion_action_id"), prefix="action", fallback="unknown"),
        "reason_codes": reasons,
    }
    return {key: value for key, value in event.items() if value not in (None, "", [], "unknown")}


def _public_family_events(decision: dict[str, Any], extra_events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    selected_family = str(decision.get("selected_action_family") or decision.get("selected_family") or "none")
    selected = decision.get("selected_candidate") if isinstance(decision.get("selected_candidate"), dict) else {}
    if selected_family != "none":
        events.append(_family_event_from_selected(selected, selected_family))
    for item in decision.get("suppressed_families") or []:
        if isinstance(item, dict):
            events.append(_family_event_from_suppressed(item))
    for item in extra_events or []:
        if not isinstance(item, dict):
            continue
        status = _safe_label(item.get("lifecycle_status") or item.get("status"), "unknown")
        events.append({
            key: value
            for key, value in {
                "action_family": _safe_label(item.get("action_family") or item.get("family")),
                "lifecycle_status": status,
                "cohort": _safe_label(item.get("cohort") or status),
                "selected": bool(item.get("selected")),
                "eligible": bool(item.get("eligible", True)),
                "candidate_id": _safe_id(item.get("candidate_id") or item.get("target_candidate_id"), prefix="candidate", fallback="unknown"),
                "rule_id": _safe_id(item.get("rule_id") or item.get("target_rule_id"), prefix="rule", fallback="unknown"),
                "action_id": _safe_id(item.get("action_id"), prefix="action", fallback="unknown"),
                "reason_codes": _reason_codes(item.get("reason_codes"), item.get("reason")),
            }.items()
            if value not in (None, "", [], "unknown")
        })
    return [event for event in events if event.get("action_family")]


def _event_type(decision: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return _safe_label(explicit, "runtime-selected")
    selected_family = str(decision.get("selected_action_family") or decision.get("selected_family") or "none")
    reasons = set(_reason_codes(decision.get("reason_codes")))
    if selected_family != "none":
        return "runtime-selected"
    if "coordinator-holdout" in reasons:
        return "runtime-holdout"
    return "runtime-suppressed"


def build_optimization_coordinator_lifecycle_feedback(
    decision: dict[str, Any],
    *,
    enforcement: dict[str, Any] | None = None,
    event_type: str | None = None,
    event_phase: str = "runtime",
    status_code: int | None = None,
    retry_count: int | None = None,
    cost_est_usd: float | None = None,
    cost_baseline_usd: float | None = None,
    extra_family_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(decision, dict) or decision.get("schema") != "agentflow.optimization_coordinator.v1":
        return None
    family_events = _public_family_events(decision, extra_family_events)
    if not family_events:
        return None

    selected_family = _safe_label(decision.get("selected_action_family") or decision.get("selected_family"), "none")
    suppressed_families = sorted({
        _safe_label(item.get("action_family") or item.get("family"))
        for item in family_events
        if item.get("lifecycle_status") in {"suppressed", "holdout", "rollback", "safety_stop"}
    })
    reason_codes = sorted({
        str(reason)
        for item in family_events
        for reason in item.get("reason_codes", [])
        if reason
    } | set(_reason_codes(decision.get("reason_codes"))))
    savings = max(0.0, _as_float(cost_baseline_usd) - _as_float(cost_est_usd))
    lifecycle_type = _event_type(decision, event_type)
    digest_basis = {
        "decision_hash": decision.get("decision_hash"),
        "event_type": lifecycle_type,
        "event_phase": event_phase,
        "family_events": family_events,
    }
    digest = hashlib.sha256(json.dumps(digest_basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    metadata = {
        "schema": FEEDBACK_SCHEMA,
        "lifecycle_kind": "optimization_coordinator",
        "command": f"optimization-coordinator-{event_phase}",
        "event_type": lifecycle_type,
        "event_phase": _safe_label(event_phase, "runtime"),
        "local_result_status": "error" if (enforcement or {}).get("status") == "error" else "ok",
        "selected_family": selected_family,
        "suppressed_families": suppressed_families,
        "suppressed_family_count": len(suppressed_families),
        "coordinator_decision_hash": _safe_id(decision.get("decision_hash"), prefix="coordinator-decision"),
        "source_surface": _safe_label(decision.get("source_surface"), "provider_request"),
        "provider_family": _safe_label(decision.get("provider_family"), "unknown"),
        "endpoint": _safe_label(decision.get("endpoint"), "unknown"),
        "category": _safe_label(decision.get("category"), "unknown") if decision.get("category") else None,
        "phase": _safe_label(decision.get("phase"), "unknown") if decision.get("phase") else None,
        "text_bucket": _safe_label(decision.get("text_bucket"), "unknown"),
        "input_token_bucket": _safe_label(decision.get("input_token_bucket"), "unknown"),
        "candidate_count": _as_int(decision.get("candidate_count")) or 0,
        "entry_count": _as_int(decision.get("entry_count")) or 0,
        "family_event_count": len(family_events),
        "status_bucket": _status_bucket(status_code),
        "error_bucket": "status_error" if (_as_int(status_code) or 0) >= 400 else "none",
        "retry_bucket": _retry_bucket(retry_count),
        "cost_bucket": _money_bucket(cost_est_usd),
        "savings_bucket": _money_bucket(savings),
        "reason_codes": reason_codes,
        "family_events": family_events,
        "privacy": _privacy_summary(),
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    event = {
        "event_type": lifecycle_type,
        "occurred_at": utc_now(),
        "recommendation_id": f"optimization-coordinator:{digest[:24]}",
        "bundle_hash": f"sha256:{digest}",
        "policy_sections": ["routing", "crunch", "cache"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "metadata": metadata,
    }
    assert_managed_egress_safe(event)
    return event


def optimization_coordinator_lifecycle_feedback_public_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(meta.get("enabled")),
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "status_code": meta.get("status_code"),
        "latency_ms": meta.get("latency_ms"),
        "payload_included": False,
    }


async def queue_optimization_coordinator_lifecycle_feedback(
    store_obj: Any,
    decision: dict[str, Any],
    *,
    enforcement: dict[str, Any] | None = None,
    event_type: str | None = None,
    event_phase: str = "runtime",
    status_code: int | None = None,
    retry_count: int | None = None,
    cost_est_usd: float | None = None,
    cost_baseline_usd: float | None = None,
    extra_family_events: list[dict[str, Any]] | None = None,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    from agentflow_proxy import recommendations

    payload = build_optimization_coordinator_lifecycle_feedback(
        decision,
        enforcement=enforcement,
        event_type=event_type,
        event_phase=event_phase,
        status_code=status_code,
        retry_count=retry_count,
        cost_est_usd=cost_est_usd,
        cost_baseline_usd=cost_baseline_usd,
        extra_family_events=extra_family_events,
    )
    if payload is None:
        return {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-optimization-coordinator-lifecycle-event",
            "auth_configured": recommendations.managed_auth_configured(),
            "payload_included": False,
        }
    return await recommendations.queue_policy_event_feedback(
        store_obj,
        payload,
        source_surface=SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=flush_immediately,
    )
