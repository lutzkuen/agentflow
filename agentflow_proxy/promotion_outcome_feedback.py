from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.promotion_outcome_feedback_ledger.v1"
ENTRY_SCHEMA = "agentflow.promotion_outcome_feedback_entry.v1"
SUMMARY_SCHEMA = "agentflow.promotion_outcome_feedback_summary.v1"

_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cache_key",
    "command",
    "content",
    "credential",
    "file_path",
    "generated_summary",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_payload",
    "raw_request",
    "raw_response",
    "raw_context",
    "request_id",
    "secret",
    "session_id",
    "summary_text",
    "tool_payload",
    "transcript",
)
_ALLOWED_RAW_LIKE_KEYS = {
    "cache_keys_included",
    "content_free",
    "raw_content_included",
    "raw_messages_included",
    "raw_params_included",
    "raw_provider_bodies_included",
    "raw_prompts_included",
    "raw_request_bodies_included",
    "raw_responses_included",
    "raw_session_ids_included",
    "request_ids_included",
    "session_id_hash",
    "tool_payloads_included",
}


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "local_only": True,
        "append_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "file_paths_included": False,
    }


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: Any, places: int = 8) -> float:
    return round(_as_float(value), places)


def _scan_raw_like(value: Any, path: str, violations: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_LIKE_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if item not in (None, False, 0, "", [], {}):
                    violations.append({"path": child, "message": "raw or local-identifier outcome feedback is not accepted"})
                    continue
            _scan_raw_like(item, child, violations)
    elif isinstance(value, list):
        for index, item in enumerate(value[:500]):
            _scan_raw_like(item, f"{path}[{index}]", violations)


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _counts(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return [{"value": value, "count": count} for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def _action_status(action: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, bool]:
    actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
    next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
    verdict = str(next_step.get("verdict") or "unknown")
    recommendation = str(next_step.get("recommendation") or "")
    error_delta = _as_float(actual.get("applied_minus_holdout_error_rate"))
    safety_stops = _as_int(actual.get("actual_safety_stopped_count"))
    rollback_needed = (
        verdict == "rollback"
        or recommendation == "rollback"
        or safety_stops > 0
        or error_delta > _as_float(thresholds.get("max_error_rate_delta"), 0.05)
    )
    if rollback_needed:
        return "rollback-needed", True
    if verdict in {"needs_more_samples", "needs-more-samples"}:
        return "needs-more-samples", False
    if verdict in {"hold", "keep-canary"} or recommendation == "keep-canary":
        return "regression-flagged", False
    if recommendation == "promote" or verdict == "widen":
        return "positive", False
    return "needs-review", False


def build_promotion_outcome_feedback_entries(impact_report: dict[str, Any], *, recorded_at: str | None = None) -> dict[str, Any]:
    recorded = recorded_at or utc_now()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "status": "invalid",
        "generated_at": recorded,
        "append_only": True,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "entries": [],
        "summary": {},
        "privacy": _privacy_summary(),
    }
    if not isinstance(impact_report, dict):
        result["error"] = {"type": "invalid_impact_report", "message": "impact report must be a JSON object"}
        return result
    violations: list[dict[str, str]] = []
    _scan_raw_like(impact_report, "$", violations)
    if violations:
        result.update({
            "status": "privacy-blocked",
            "error": {"type": "privacy_blocked", "message": "promotion outcome feedback contains raw-like fields"},
            "privacy": {**_privacy_summary(), "input_privacy_violations": violations[:20]},
        })
        return result
    actions = impact_report.get("actions")
    if not isinstance(actions, list):
        result["error"] = {"type": "invalid_impact_report", "message": "impact report actions must be a list"}
        return result

    entries: list[dict[str, Any]] = []
    source_bundle = impact_report.get("source_action_bundle") if isinstance(impact_report.get("source_action_bundle"), dict) else {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
        projection = action.get("projection") if isinstance(action.get("projection"), dict) else {}
        thresholds = action.get("thresholds") if isinstance(action.get("thresholds"), dict) else {}
        next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
        cohorts = actual.get("cohorts") if isinstance(actual.get("cohorts"), dict) else {}
        applied = cohorts.get("canary_applied") if isinstance(cohorts.get("canary_applied"), dict) else {}
        holdout = cohorts.get("canary_holdout") if isinstance(cohorts.get("canary_holdout"), dict) else {}
        skipped = cohorts.get("skipped") if isinstance(cohorts.get("skipped"), dict) else {}
        bypassed = cohorts.get("bypassed_or_disabled") if isinstance(cohorts.get("bypassed_or_disabled"), dict) else {}
        status, rollback_needed = _action_status(action, thresholds)
        policy_id = str(action.get("policy_id") or action.get("target_rule_id") or action.get("target_candidate_id") or action.get("action_id") or "unknown")
        entry = {
            "schema": ENTRY_SCHEMA,
            "id": _stable_id("promotion-outcome", recorded, policy_id, action.get("action_id"), action.get("path")),
            "created_at": recorded,
            "impact_generated_at": impact_report.get("generated_at"),
            "policy_id": policy_id,
            "action_family": str(action.get("action_family") or "unknown"),
            "policy_section": str(action.get("policy_section") or action.get("action_family") or "unknown"),
            "rule_source": action.get("rule_source") or "unknown",
            "rule_id": action.get("target_rule_id"),
            "candidate_id": action.get("target_candidate_id"),
            "action_id": action.get("action_id"),
            "source_evidence_schema": action.get("source_evidence_schema") or source_bundle.get("schema"),
            "status": status,
            "recommendation": next_step.get("recommendation"),
            "rollback_needed": rollback_needed,
            "reason_codes": list(next_step.get("reason_codes") or []),
            "warning_codes": list(next_step.get("warning_codes") or []),
            "observed_savings_usd": _rounded(actual.get("observed_savings_usd")),
            "projected_savings_usd": _rounded(projection.get("projected_savings_usd")),
            "projection_realization_ratio": next_step.get("projected_vs_observed_savings_ratio"),
            "applied_count": _as_int(actual.get("actual_canary_applied_count")),
            "holdout_count": _as_int(actual.get("actual_canary_holdout_count")),
            "skipped_count": _as_int(actual.get("actual_skipped_count")),
            "bypassed_count": _as_int(actual.get("actual_bypassed_or_disabled_count")),
            "safety_stop_count": _as_int(actual.get("actual_safety_stopped_count")),
            "error_rate_delta": _rounded(actual.get("applied_minus_holdout_error_rate"), 6),
            "retry_rate_delta": _rounded(actual.get("applied_minus_holdout_retry_rate"), 6),
            "latency_delta_ms": actual.get("applied_minus_holdout_latency_avg_ms"),
            "cohort_metrics": {
                "canary_applied": applied,
                "canary_holdout": holdout,
                "skipped": skipped,
                "bypassed_or_disabled": bypassed,
                "safety_stopped": cohorts.get("safety_stopped") if isinstance(cohorts.get("safety_stopped"), dict) else {},
            },
            "thresholds": thresholds,
            "privacy": _privacy_summary(),
        }
        entries.append({key: value for key, value in entry.items() if value not in (None, "", [], {})})

    result.update({
        "ok": True,
        "status": "recordable" if entries else "no-actions",
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "rollback_needed_count": sum(1 for entry in entries if entry.get("rollback_needed")),
            "observed_savings_usd": round(sum(_as_float(entry.get("observed_savings_usd")) for entry in entries), 8),
            "projected_savings_usd": round(sum(_as_float(entry.get("projected_savings_usd")) for entry in entries), 8),
            "status_counts": _counts(entries, "status"),
            "action_family_counts": _counts(entries, "action_family"),
            "recommendation_counts": _counts(entries, "recommendation"),
        },
    })
    return result


def record_promotion_outcome_feedback(impact_report: dict[str, Any], *, store_obj: Any, recorded_at: str | None = None) -> dict[str, Any]:
    ledger = build_promotion_outcome_feedback_entries(impact_report, recorded_at=recorded_at)
    if not ledger.get("ok"):
        return ledger
    rows_written = 0
    for entry in ledger.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        store_obj.log_promotion_outcome_feedback(
            id=entry.get("id"),
            created_at=entry.get("created_at"),
            impact_generated_at=entry.get("impact_generated_at"),
            policy_id=entry.get("policy_id"),
            action_family=entry.get("action_family"),
            policy_section=entry.get("policy_section"),
            rule_source=entry.get("rule_source"),
            rule_id=entry.get("rule_id"),
            candidate_id=entry.get("candidate_id"),
            action_id=entry.get("action_id"),
            source_evidence_schema=entry.get("source_evidence_schema"),
            status=entry.get("status"),
            recommendation=entry.get("recommendation"),
            rollback_needed=1 if entry.get("rollback_needed") else 0,
            observed_savings_usd=entry.get("observed_savings_usd"),
            projected_savings_usd=entry.get("projected_savings_usd"),
            projection_realization_ratio=entry.get("projection_realization_ratio"),
            applied_count=entry.get("applied_count"),
            holdout_count=entry.get("holdout_count"),
            skipped_count=entry.get("skipped_count"),
            bypassed_count=entry.get("bypassed_count"),
            safety_stop_count=entry.get("safety_stop_count"),
            error_rate_delta=entry.get("error_rate_delta"),
            retry_rate_delta=entry.get("retry_rate_delta"),
            latency_delta_ms=entry.get("latency_delta_ms"),
            feedback_json=stable_json(entry),
        )
        rows_written += 1
    ledger["wrote_store"] = rows_written > 0
    ledger["summary"]["rows_written"] = rows_written
    ledger["status"] = "recorded" if rows_written else ledger.get("status")
    return ledger


def promotion_outcome_feedback_summary(store_obj: Any, *, limit: int = 1000) -> dict[str, Any]:
    rows = store_obj.promotion_outcome_feedback_rows(limit=limit)
    entries: list[dict[str, Any]] = []
    for row in rows:
        try:
            entry = json.loads(row.get("feedback_json") or "{}")
        except Exception:
            entry = {}
        if not isinstance(entry, dict):
            entry = {}
        entries.append({**entry, "id": row.get("id"), "created_at": row.get("created_at")})

    candidates: list[dict[str, Any]] = []
    for entry in entries:
        cohorts = entry.get("cohort_metrics") if isinstance(entry.get("cohort_metrics"), dict) else {}
        candidates.append({
            "schema": "agentflow.promotion_outcome_feedback_candidate.v1",
            "candidate_id": entry.get("candidate_id") or entry.get("policy_id"),
            "target_candidate_id": entry.get("candidate_id"),
            "action_id": entry.get("action_id"),
            "rule_id": entry.get("rule_id"),
            "action_family": entry.get("action_family"),
            "policy_section": entry.get("policy_section"),
            "verdict": "rollback" if entry.get("rollback_needed") else entry.get("recommendation"),
            "reason_codes": entry.get("reason_codes") or [],
            "observed_savings_usd": entry.get("observed_savings_usd"),
            "projected_savings_usd": entry.get("projected_savings_usd"),
            "cohort_metrics": cohorts,
            "evidence_source": "promotion_outcome_feedback",
            "privacy": _privacy_summary(),
        })

    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "entry_count": len(entries),
        "entries": entries,
        "candidates": candidates,
        "summary": {
            "entry_count": len(entries),
            "rollback_needed_count": sum(1 for entry in entries if entry.get("rollback_needed")),
            "observed_savings_usd": round(sum(_as_float(entry.get("observed_savings_usd")) for entry in entries), 8),
            "projected_savings_usd": round(sum(_as_float(entry.get("projected_savings_usd")) for entry in entries), 8),
            "status_counts": _counts(entries, "status"),
            "action_family_counts": _counts(entries, "action_family"),
            "recommendation_counts": _counts(entries, "recommendation"),
        },
        "privacy": _privacy_summary(),
    }
