from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from tokenclaw.store import stable_json, utc_now


SCHEMA = "agentflow.promotion_outcome_feedback_ledger.v1"
ENTRY_SCHEMA = "agentflow.promotion_outcome_feedback_entry.v1"
SUMMARY_SCHEMA = "agentflow.promotion_outcome_feedback_summary.v1"
POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE = "post_promotion_action_outcomes"
POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ENDPOINT = "/v1/promotion-blocker-action-outcome-rollups"
POST_PROMOTION_ACTION_OUTCOME_ROLLUP_INGEST_SCHEMA = "agentflow.promotion_blocker_action_outcome_rollup_ingest.v1"
POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ROW_SCHEMA = "agentflow.promotion_blocker_action_outcome_rollup_row.v1"

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


def _count_dict(rows: list[dict[str, Any]], key: str, *, weight_key: str | None = None) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key) or "unknown")
        weight = max(1, _as_int(row.get(weight_key))) if weight_key else 1
        counter[value] += weight
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _increment_count(counts: dict[str, int], value: Any, amount: int = 1) -> None:
    key = str(value or "unknown")
    counts[key] = counts.get(key, 0) + max(0, amount)


def _rollup_privacy_summary() -> dict[str, Any]:
    privacy = _privacy_summary()
    privacy.update({
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "tenant_secrets_included": False,
    })
    return privacy


def _entry_next_action(entry: dict[str, Any]) -> str:
    recommendation = str(entry.get("recommendation") or "").replace("_", "-")
    status = str(entry.get("status") or "").replace("_", "-")
    if entry.get("rollback_needed") or recommendation == "rollback" or status == "rollback-needed":
        return "rollback-local-policy"
    if recommendation in {"promote", "widen", "widen-local-policy"} or status == "positive":
        return "widen-local-policy"
    return "keep-blocked"


def _entry_outcome_count(entry: dict[str, Any]) -> int:
    counted = (
        _as_int(entry.get("applied_count"))
        + _as_int(entry.get("holdout_count"))
        + _as_int(entry.get("skipped_count"))
        + _as_int(entry.get("bypassed_count"))
        + _as_int(entry.get("safety_stop_count"))
    )
    return max(1, counted)


def _load_feedback_entry(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        entry = json.loads(row.get("feedback_json") or "{}")
    except Exception:
        entry = {}
    if not isinstance(entry, dict):
        entry = {}
    entry = {**entry, "id": row.get("id"), "created_at": row.get("created_at")}
    violations: list[dict[str, str]] = []
    _scan_raw_like(entry, "$", violations)
    return entry, violations


def _row_source_surface(entry: dict[str, Any]) -> str:
    source = entry.get("source_surface")
    if source:
        return str(source)
    evidence = str(entry.get("source_evidence_schema") or "")
    if "openai" in evidence:
        return "openai_responses"
    if "anthropic" in evidence or "claude" in evidence:
        return "anthropic_messages"
    return "unknown"


def build_post_promotion_action_outcome_rollups(
    store_obj: Any,
    *,
    limit: int = 1000,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now()
    capped = max(1, min(int(limit or 1), 10000))
    result: dict[str, Any] = {
        "schema": "agentflow.post_promotion_action_outcome_rollups.v1",
        "ok": False,
        "status": "invalid",
        "generated_at": generated,
        "source_surface": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE,
        "endpoint": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ENDPOINT,
        "rollups": [],
        "payload": None,
        "summary": {},
        "privacy": _rollup_privacy_summary(),
    }
    if not hasattr(store_obj, "promotion_outcome_feedback_rows"):
        result.update({
            "status": "unsupported-store",
            "error": {"type": "unsupported_store", "message": "store does not expose promotion outcome feedback rows"},
        })
        return result

    entries: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for row in store_obj.promotion_outcome_feedback_rows(limit=capped):
        entry, entry_violations = _load_feedback_entry(row)
        if entry_violations:
            violations.extend(entry_violations)
            continue
        if entry:
            entries.append(entry)
    if violations:
        result.update({
            "status": "privacy-blocked",
            "error": {"type": "privacy_blocked", "message": "stored promotion outcome feedback contains raw-like fields"},
            "privacy": {**_rollup_privacy_summary(), "input_privacy_violations": violations[:20]},
        })
        return result
    if not entries:
        result.update({
            "ok": True,
            "status": "no-outcomes",
            "summary": {"entry_count": 0, "rollup_count": 0},
        })
        return result

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (
            _row_source_surface(entry),
            str(entry.get("action_family") or "unknown"),
            str(entry.get("policy_section") or entry.get("action_family") or "unknown"),
            str(entry.get("rule_source") or "unknown"),
        )
        groups.setdefault(key, []).append(entry)

    rollups: list[dict[str, Any]] = []
    for (source_surface, action_family, policy_section, policy_source), group in groups.items():
        outcome_status_counts: dict[str, int] = {}
        reason_code_counts: dict[str, int] = {}
        source_outcome_counts: dict[str, int] = {}
        next_action_counts: dict[str, int] = {}
        applied_count = holdout_count = skipped_count = bypassed_count = safety_stopped_count = 0
        observed = projected = 0.0
        for entry in group:
            outcome_count = _entry_outcome_count(entry)
            _increment_count(outcome_status_counts, entry.get("status"), outcome_count)
            _increment_count(source_outcome_counts, entry.get("recommendation") or entry.get("status"), outcome_count)
            _increment_count(next_action_counts, _entry_next_action(entry), outcome_count)
            for reason in entry.get("reason_codes") or []:
                _increment_count(reason_code_counts, reason, outcome_count)
            applied_count += _as_int(entry.get("applied_count"))
            holdout_count += _as_int(entry.get("holdout_count"))
            skipped_count += _as_int(entry.get("skipped_count"))
            bypassed_count += _as_int(entry.get("bypassed_count"))
            safety_stopped_count += _as_int(entry.get("safety_stop_count"))
            observed += _as_float(entry.get("observed_savings_usd"))
            projected += _as_float(entry.get("projected_savings_usd"))
        outcome_count_total = sum(outcome_status_counts.values()) or len(group)
        top_outcome = max(outcome_status_counts.items(), key=lambda item: (item[1], item[0]))[0]
        top_next_action = max(next_action_counts.items(), key=lambda item: (item[1], item[0]))[0]
        blocker_reason = (
            max(reason_code_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if reason_code_counts
            else top_outcome
        )
        rollups.append({
            "schema": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ROW_SCHEMA,
            "source_surface": source_surface,
            "local_action_family": action_family,
            "action_family": action_family,
            "recommendation_type": "post-promotion-action-outcome",
            "blocker_reason": blocker_reason,
            "outcome_status": top_outcome,
            "next_action": top_next_action,
            "policy_source": policy_source,
            "observed_savings_usd": round(observed, 8),
            "projected_savings_usd": round(max(0.0, projected), 8),
            "row_count": len(group),
            "outcome_count": outcome_count_total,
            "executed_count": applied_count + holdout_count,
            "applied_count": applied_count,
            "skipped_count": skipped_count + bypassed_count,
            "safety_stopped_count": safety_stopped_count,
            "noop_count": max(0, outcome_count_total - applied_count - holdout_count - skipped_count - bypassed_count - safety_stopped_count),
            "outcome_status_counts": outcome_status_counts,
            "reason_code_counts": reason_code_counts,
            "recommendation_type_counts": {"post-promotion-action-outcome": outcome_count_total},
            "local_action_family_counts": {action_family: outcome_count_total},
            "policy_source_counts": {policy_source: outcome_count_total},
            "source_outcome_counts": source_outcome_counts,
            "local_executor_compatibility": {
                "status": "blocked" if top_next_action == "rollback-local-policy" else "compatible",
                "local_action_family": action_family,
                "reason_codes": sorted(reason_code_counts),
            },
            "metadata": {
                "aggregate_only": True,
                "policy_section": policy_section,
                "holdout_count": holdout_count,
                "bypassed_count": bypassed_count,
                "savings_delta_usd": round(observed - projected, 8),
                "source_entry_count": len(group),
            },
            "privacy": _rollup_privacy_summary(),
        })
    rollups.sort(key=lambda row: (-_as_int(row.get("outcome_count")), str(row.get("local_action_family")), str(row.get("source_surface"))))
    payload = {
        "schema": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_INGEST_SCHEMA,
        "generated_at": generated,
        "run_id": _stable_id("post-promotion-outcomes", generated, len(entries), len(rollups)),
        "window": {
            "source": "local-promotion-outcome-feedback",
            "entry_count": len(entries),
        },
        "status": "ready",
        "top_next_action": rollups[0].get("next_action") if rollups else None,
        "summary": {
            "entry_count": len(entries),
            "rollup_count": len(rollups),
            "outcome_count": sum(_as_int(row.get("outcome_count")) for row in rollups),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in rollups), 8),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in rollups), 8),
            "outcome_status_counts": _count_dict(entries, "status", weight_key=None),
            "action_family_counts": _count_dict(entries, "action_family", weight_key=None),
        },
        "rollups": rollups,
        "privacy": _rollup_privacy_summary(),
    }
    result.update({
        "ok": True,
        "status": "ready",
        "rollups": rollups,
        "payload": payload,
        "summary": payload["summary"],
    })
    return result


async def queue_post_promotion_action_outcome_rollups(
    store_obj: Any,
    *,
    limit: int = 1000,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    from tokenclaw import recommendations
    from tokenclaw.managed_egress import ManagedEgressBlocked, assert_managed_egress_safe, managed_egress_blocked_meta

    built = build_post_promotion_action_outcome_rollups(store_obj, limit=limit)
    meta: dict[str, Any] = {
        "schema": "agentflow.post_promotion_action_outcome_rollup_flush_status.v1",
        "enabled": recommendations.recommendations_enabled(),
        "server_url": recommendations.recommendation_server_url(),
        "endpoint": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ENDPOINT,
        "source_surface": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE,
        "rollup_status": built.get("status"),
        "rollup_count": len(built.get("rollups") or []),
        "payload_included": False,
        "privacy": _rollup_privacy_summary(),
    }
    if not built.get("ok"):
        meta.update({
            "status": "rejected",
            "reason": built.get("status") or "invalid-rollups",
            "error": built.get("error"),
        })
        return meta
    if built.get("status") == "no-outcomes":
        meta.update({"status": "skipped", "reason": "no-post-promotion-outcomes"})
        return meta
    if not recommendations.recommendations_enabled():
        meta.update({"status": "disabled", "reason": "managed-feedback-disabled"})
        return meta
    if not recommendations.recommendation_server_configured():
        meta.update({"status": "no-managed-config", "reason": "server-url-not-configured"})
        return meta
    if not hasattr(store_obj, "enqueue_managed_outcome_feedback"):
        meta.update({"status": "skipped", "reason": "store-does-not-support-managed-feedback-queue"})
        return meta

    payload = built.get("payload") if isinstance(built.get("payload"), dict) else {}
    try:
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update({
            "status": "rejected",
            "reason": "unsafe-egress-payload",
            **managed_egress_blocked_meta(endpoint=POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ENDPOINT, violations=exc.violations),
        })
        return meta

    queue_id = _stable_id("post-promotion-action-outcomes", stable_json(payload))
    if hasattr(store_obj, "get_managed_outcome_feedback"):
        existing = store_obj.get_managed_outcome_feedback(queue_id)
        if existing:
            meta.update({
                "status": existing.get("status") if existing.get("status") == "sent" else "pending-flush",
                "reason": "already-queued",
                "queue_id": queue_id,
                "attempts": existing.get("attempts") or 0,
            })
            return meta

    row = {
        "id": queue_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "source_surface": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE,
        "endpoint": POST_PROMOTION_ACTION_OUTCOME_ROLLUP_ENDPOINT,
        "optimization_unit_id": 0,
        "payload_json": stable_json(payload),
        "status": "queued",
        "attempts": 0,
        "next_attempt_at": utc_now(),
    }
    store_obj.enqueue_managed_outcome_feedback(**row)
    meta.update({
        "status": "pending-flush",
        "reason": "queued",
        "queue_id": queue_id,
        "attempts": 0,
    })
    if flush_immediately:
        results = await recommendations.flush_queued_outcome_feedback(
            store_obj,
            limit=1,
            source_surface=POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE,
        )
        first = results[0] if results else {}
        meta.update({
            "status": "flushed" if first.get("status") == "sent" else first.get("status") or "pending-flush",
            "reason": first.get("reason") or meta.get("reason"),
            "flush_result": {key: value for key, value in first.items() if key != "payload_json"},
        })
    return meta


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
