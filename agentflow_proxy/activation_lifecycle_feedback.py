from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.openai_optimization_governor import LIFECYCLE_SCHEMA, LIFECYCLE_SOURCE_SURFACE
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import utc_now


QUEUE_META_SCHEMA = "agentflow.activation_staged_lifecycle_feedback_queue_meta.v1"
SUMMARY_SCHEMA = "agentflow.activation_staged_lifecycle_feedback_summary.v1"

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


def _privacy_summary() -> dict[str, Any]:
    return {
        "telemetry_profile": "metadata-only",
        "local_only": True,
        "metadata_only": True,
        "aggregate_only": False,
        "raw_payload_included": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_responses_included": False,
        "raw_provider_bodies_included": False,
        "raw_tool_payloads_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "tenant_ids_included": False,
        "file_paths_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback)


def _safe_id(value: Any, *, prefix: str, fallback: str = "unknown") -> str:
    return public_id(value, prefix=prefix, fallback=fallback) or fallback


def _public_ref(value: Any, *, prefix: str, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _reason_code(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    if any(hint.replace("_", "-") in text for hint in RAW_REASON_HINTS):
        return public_id(text, prefix="reason", fallback="redacted-reason") or "redacted-reason"
    return public_label(text, "redacted-reason")


def _reason_codes(*values: Any) -> list[str]:
    codes: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                code = _reason_code(item)
                if code:
                    codes.append(code)
        else:
            code = _reason_code(value)
            if code:
                codes.append(code)
    return sorted(set(codes))


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


def _savings_estimate(value: dict[str, Any]) -> float:
    for key in (
        "savings_estimate_usd",
        "estimated_savings_usd",
        "projected_savings_usd",
        "observed_savings_usd",
        "net_savings_usd",
        "expected_net_savings_usd",
    ):
        amount = _as_float(value.get(key))
        if amount > 0:
            return round(amount, 8)
    return 0.0


def _status_bucket(status: Any) -> str:
    text = str(status or "").strip().lower().replace("_", "-")
    if text in {"applied", "apply", "wrote", "written"}:
        return "applied"
    if text in {"dry-run", "dry_run", "preview", "projected", "staged"}:
        return "dry_run"
    if text in {"holdout", "canary-holdout"}:
        return "holdout"
    if text in {"rollback", "rollback-required", "rollback_required"}:
        return "rollback_required"
    if text in {"suppressed", "blocked", "rejected", "omitted", "safety-stopped", "safety_stop", "safety-stopped"}:
        return "suppressed"
    if text in {"not-selected", "not_selected", "skipped", "bypassed", "disabled"}:
        return "suppressed"
    return text or "unknown"


def _cohort_for_action(action: dict[str, Any], *, event_phase: str, default: str) -> str:
    raw = (
        action.get("cohort")
        or action.get("canary_cohort")
        or action.get("lifecycle_status")
        or action.get("status")
        or default
    )
    status = _status_bucket(raw)
    if status == "dry_run":
        return "dry_run" if event_phase in {"dry_run", "stage"} else default
    return status


def _family_for_section(section: Any, action_family: Any = None) -> str:
    text = str(action_family or section or "unknown").strip().lower().replace("-", "_")
    if text in {"crunch", "old_context_summary", "old_context_summarization", "summary"}:
        return "old_context_summary" if text != "crunch" else "crunch"
    if text in {"cache", "cache_replay"}:
        return "cache_replay"
    if text == "routing":
        return "routing"
    return _safe_label(text, "unknown")


def _event_from_staged_draft(item: dict[str, Any], *, event_phase: str) -> dict[str, Any]:
    family = _family_for_section(item.get("section"), item.get("action_family"))
    cohort = "staged" if event_phase == "stage" else _cohort_for_action(item, event_phase=event_phase, default="dry_run")
    return {
        "action_family": family,
        "cohort": cohort,
        "selected": False,
        "eligible": bool(item.get("ok", True)),
        "status": "staged" if event_phase == "stage" else _status_bucket(item.get("status") or "dry_run"),
        "candidate_id": _safe_id(item.get("candidate_id") or item.get("target_candidate_id"), prefix="candidate"),
        "rule_id": _safe_id(item.get("rule_id") or item.get("policy_id") or item.get("target_rule_id"), prefix="rule", fallback="unknown"),
        "action_id": _safe_id(item.get("action_id") or item.get("promotion_action_id") or item.get("draft_id"), prefix="action", fallback="unknown"),
        "policy_source": _safe_label(item.get("policy_source") or "local-manual", "local-manual"),
        "savings_estimate_usd": _savings_estimate(item),
        "reason_codes": _reason_codes(item.get("reason_codes"), "activation-staged-action"),
    }


def _event_from_action(item: dict[str, Any], *, event_phase: str) -> dict[str, Any]:
    family = _family_for_section(item.get("policy_section"), item.get("action_family"))
    status = _status_bucket(item.get("status") or ("applied" if event_phase == "apply" else "dry_run"))
    cohort = _cohort_for_action(item, event_phase=event_phase, default=status)
    return {
        "action_family": family,
        "cohort": cohort,
        "selected": status == "applied",
        "eligible": status not in {"suppressed", "rollback_required"},
        "status": status,
        "candidate_id": _safe_id(item.get("candidate_id") or item.get("target_candidate_id"), prefix="candidate"),
        "rule_id": _safe_id(item.get("rule_id") or item.get("target_rule_id") or item.get("policy_id"), prefix="rule", fallback="unknown"),
        "action_id": _safe_id(item.get("action_id") or item.get("promotion_action_id"), prefix="action", fallback="unknown"),
        "policy_source": _safe_label(item.get("policy_source") or "local-manual", "local-manual"),
        "savings_estimate_usd": _savings_estimate(item),
        "reason_codes": _reason_codes(item.get("reason_codes"), item.get("reason"), f"activation-{event_phase}"),
    }


def _event_from_omission(item: dict[str, Any], *, event_phase: str) -> dict[str, Any]:
    family = _family_for_section(item.get("section") or item.get("policy_section") or "routing", item.get("action_family"))
    return {
        "action_family": family,
        "cohort": "suppressed",
        "selected": False,
        "eligible": False,
        "status": "suppressed",
        "candidate_id": _safe_id(item.get("candidate_id") or item.get("target_candidate_id"), prefix="candidate"),
        "rule_id": _safe_id(item.get("rule_id") or item.get("target_rule_id"), prefix="rule", fallback="unknown"),
        "policy_source": _safe_label(item.get("policy_source") or "local-manual", "local-manual"),
        "savings_estimate_usd": _savings_estimate(item),
        "reason_codes": _reason_codes(item.get("reason_codes"), item.get("reason"), f"activation-{event_phase}-suppressed"),
    }


def _family_events(result: dict[str, Any], *, event_phase: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in result.get("staged_drafts") or []:
        if isinstance(item, dict):
            events.append(_event_from_staged_draft(item, event_phase=event_phase))
    for item in result.get("actions") or []:
        if isinstance(item, dict):
            events.append(_event_from_action(item, event_phase=event_phase))
    for item in result.get("omitted") or []:
        if isinstance(item, dict):
            events.append(_event_from_omission(item, event_phase=event_phase))
    return [
        {key: value for key, value in event.items() if value not in (None, "", [], "unknown")}
        for event in events
        if event.get("action_family")
    ]


def build_activation_staged_lifecycle_feedback(
    result: dict[str, Any],
    *,
    event_phase: str,
    command: str,
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    phase = _safe_label(event_phase, "stage")
    family_events = _family_events(result, event_phase=phase)
    if not family_events:
        return None
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    all_reasons = sorted({
        str(reason)
        for event in family_events
        for reason in event.get("reason_codes", [])
        if reason
    })
    digest_basis = {
        "schema": result.get("schema"),
        "phase": phase,
        "command": command,
        "events": family_events,
    }
    digest = hashlib.sha256(json.dumps(digest_basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    payload = {
        "schema": LIFECYCLE_SCHEMA,
        "event_type": "activation_staged_optimization_lifecycle",
        "occurred_at": result.get("generated_at") or utc_now(),
        "provider": "openai",
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "endpoint": _safe_label(result.get("endpoint") or "responses", "responses"),
        "category": "activation_staged_optimization",
        "stream": False,
        "selected_action_family": "none",
        "family_events": family_events,
        "event_phase": phase,
        "command_name": _safe_label(command, "activation-lifecycle"),
        "lifecycle_state": lifecycle_feedback_state_from_events(family_events),
        "candidate_count": len({
            event.get("candidate_id")
            for event in family_events
            if event.get("candidate_id")
        }),
        "family_event_count": len(family_events),
        "status_bucket": "2xx" if result.get("ok") else "blocked",
        "retry_bucket": "none",
        "cost_bucket": _money_bucket(summary.get("estimated_cost_usd") or summary.get("projected_cost_usd")),
        "savings_bucket": _money_bucket(summary.get("projected_savings_usd") or summary.get("expected_net_savings_usd")),
        "reason_codes": all_reasons,
        "feedback_public_id": f"activation-lifecycle:{digest[:16]}",
        "privacy": _privacy_summary(),
    }
    assert_managed_egress_safe(payload)
    return payload


def lifecycle_feedback_state_from_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return "missing_feedback"
    cohorts = {str(event.get("cohort") or event.get("status") or "") for event in events}
    statuses = {str(event.get("status") or "") for event in events}
    if cohorts & {"rollback", "rollback_required"} or statuses & {"rollback", "rollback_required"}:
        return "rollback_required"
    if cohorts & {"safety_stop", "safety_stopped", "suppressed"} or statuses & {"safety_stop", "safety_stopped", "suppressed"}:
        return "suppressed"
    if cohorts & {"applied", "canary_applied"} and cohorts & {"holdout", "canary_holdout"}:
        return "healthy_canary"
    if cohorts & {"holdout", "canary_holdout"} and not (cohorts & {"applied", "canary_applied"}):
        return "holdout_only"
    if cohorts & {"applied", "canary_applied", "staged", "dry_run"}:
        return "healthy_canary"
    return "missing_feedback"


def _empty_lifecycle_group(policy_ref: str, cohort: str, family: str) -> dict[str, Any]:
    return {
        "policy_ref": policy_ref,
        "candidate_id": None,
        "cohort_label": cohort,
        "action_family": family,
        "event_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "fallback_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "safety_stop_count": 0,
        "savings_estimate_usd": 0.0,
    }


def _event_policy_ref(event: dict[str, Any]) -> str:
    return _public_ref(
        event.get("policy_ref")
        or event.get("policy_id")
        or event.get("rule_id")
        or event.get("candidate_id"),
        prefix="policy",
        fallback="unknown",
    )


def _payload_error_observed(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status_bucket") or "").strip().lower()
    return status in {"4xx", "5xx", "blocked", "error", "failed", "retryable-error"}


def _payload_retry_observed(payload: dict[str, Any]) -> bool:
    retry = str(payload.get("retry_bucket") or "").strip().lower()
    return retry not in {"", "none", "0", "zero"}


def _add_lifecycle_group_event(
    groups: dict[tuple[str, str, str], dict[str, Any]],
    *,
    event: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    family = _safe_label(event.get("action_family"), "unknown")
    cohort = _safe_label(event.get("cohort") or event.get("status"), "unknown")
    policy_ref = _event_policy_ref(event)
    group = groups.setdefault((policy_ref, cohort, family), _empty_lifecycle_group(policy_ref, cohort, family))
    if not group.get("candidate_id") and event.get("candidate_id"):
        group["candidate_id"] = _safe_id(event.get("candidate_id"), prefix="candidate")
    group["event_count"] += 1
    if cohort in {"applied", "canary_applied"}:
        group["applied_count"] += 1
    if cohort in {"holdout", "canary_holdout"}:
        group["holdout_count"] += 1
    if cohort in {"fallback", "fallback_applied", "rate_limit_fallback"}:
        group["fallback_count"] += 1
    if _payload_error_observed(payload):
        group["error_count"] += 1
    if _payload_retry_observed(payload):
        group["retry_count"] += 1
    if cohort in {"safety_stop", "safety_stopped"}:
        group["safety_stop_count"] += 1
    group["savings_estimate_usd"] += _savings_estimate(event)


def _finalize_lifecycle_group(group: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(group.get("event_count"))
    errors = _as_int(group.get("error_count"))
    return {
        "policy_ref": group["policy_ref"],
        "candidate_id": group.get("candidate_id"),
        "cohort_label": group["cohort_label"],
        "action_family": group["action_family"],
        "event_count": count,
        "applied_count": _as_int(group.get("applied_count")),
        "holdout_count": _as_int(group.get("holdout_count")),
        "fallback_count": _as_int(group.get("fallback_count")),
        "error_count": errors,
        "retry_count": _as_int(group.get("retry_count")),
        "safety_stop_count": _as_int(group.get("safety_stop_count")),
        "error_rate": round(errors / count, 6) if count > 0 else 0.0,
        "savings_estimate_usd": round(_as_float(group.get("savings_estimate_usd")), 8),
    }


async def queue_activation_staged_lifecycle_feedback(
    store_obj: Any,
    result: dict[str, Any],
    *,
    event_phase: str,
    command: str,
) -> dict[str, Any]:
    from agentflow_proxy.recommendations import queue_policy_event_feedback

    payload = build_activation_staged_lifecycle_feedback(result, event_phase=event_phase, command=command)
    if payload is None:
        return {
            "schema": QUEUE_META_SCHEMA,
            "source_surface": LIFECYCLE_SOURCE_SURFACE,
            "event_phase": event_phase,
            "status": "skipped",
            "reason": "no-activation-lifecycle-events",
            "payload_included": False,
        }
    meta = await queue_policy_event_feedback(
        store_obj,
        payload,
        source_surface=LIFECYCLE_SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=False,
    )
    return {
        "schema": QUEUE_META_SCHEMA,
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "event_phase": event_phase,
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "payload_included": False,
    }


def activation_lifecycle_feedback_summary(store_obj: Any, *, limit: int = 10000) -> dict[str, Any]:
    rows = (
        store_obj.managed_outcome_feedback_payload_rows(source_surface=LIFECYCLE_SOURCE_SURFACE, limit=limit)
        if hasattr(store_obj, "managed_outcome_feedback_payload_rows")
        else []
    )
    queue_rows = 0
    family_event_count = 0
    state_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    family_state_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    lifecycle_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        try:
            payload = json.loads(str(row.get("payload_json") or "{}"))
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("schema") != LIFECYCLE_SCHEMA:
            continue
        if payload.get("event_type") != "activation_staged_optimization_lifecycle":
            continue
        events = [event for event in payload.get("family_events") or [] if isinstance(event, dict)]
        if not events:
            continue
        queue_rows += 1
        phase = str(payload.get("event_phase") or "unknown")
        state = str(payload.get("lifecycle_state") or lifecycle_feedback_state_from_events(events))
        phase_counts[phase] += 1
        state_counts[state] += 1
        for event in events:
            family_event_count += 1
            family = str(event.get("action_family") or "unknown")
            cohort = str(event.get("cohort") or "unknown")
            cohort_counts[cohort] += 1
            family_state_counts[f"{family}:{state}"] += 1
            candidate_id = event.get("candidate_id")
            if candidate_id:
                candidate_counts[str(candidate_id)] += 1
            _add_lifecycle_group_event(lifecycle_groups, event=event, payload=payload)
    cohort_lifecycle_metadata = [
        _finalize_lifecycle_group(group)
        for group in lifecycle_groups.values()
    ]
    cohort_lifecycle_metadata.sort(
        key=lambda item: (
            -_as_int(item.get("event_count")),
            str(item.get("action_family") or ""),
            str(item.get("policy_ref") or ""),
            str(item.get("cohort_label") or ""),
        )
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "queue_rows": queue_rows,
        "family_event_count": family_event_count,
        "state_breakdown": [{"value": key, "count": value} for key, value in sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))],
        "event_phase_breakdown": [{"value": key, "count": value} for key, value in sorted(phase_counts.items(), key=lambda item: (-item[1], item[0]))],
        "cohort_breakdown": [{"value": key, "count": value} for key, value in sorted(cohort_counts.items(), key=lambda item: (-item[1], item[0]))],
        "family_state_breakdown": [{"value": key, "count": value} for key, value in sorted(family_state_counts.items(), key=lambda item: (-item[1], item[0]))],
        "candidate_id_breakdown": [{"value": key, "count": value} for key, value in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))[:50]],
        "cohort_lifecycle_metadata": cohort_lifecycle_metadata[:50],
        "payload_json_included": False,
        "privacy": _privacy_summary(),
    }
