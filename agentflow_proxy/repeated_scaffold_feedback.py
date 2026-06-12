from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.store import utc_now


FEEDBACK_SCHEMA = "agentflow.repeated_scaffold_lifecycle_feedback.v1"
SOURCE_SURFACE = "repeated_scaffold_lifecycle"
STATUS_SCHEMA = "agentflow.repeated_scaffold_lifecycle_feedback_queue_status.v1"
FLUSH_SCHEMA = "agentflow.repeated_scaffold_lifecycle_feedback_flush.v1"

_ERROR_CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_utc_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _seconds_since(raw: Any, now: datetime) -> int | None:
    parsed = _parse_utc_iso(raw)
    if parsed is None:
        return None
    return max(0, int((now - parsed).total_seconds()))


def _breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _breakdown_counts(rows: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("value") or "unknown")
        count = _as_int(row.get("count"))
        if count:
            counts[value] = counts.get(value, 0) + count
    return dict(sorted(counts.items()))


def _token_bucket(tokens: Any) -> str:
    value = _as_int(tokens)
    if value <= 0:
        return "none"
    if value < 1_000:
        return "lt_1k_tokens"
    if value < 4_000:
        return "1k_4k_tokens"
    if value < 16_000:
        return "4k_16k_tokens"
    if value < 64_000:
        return "16k_64k_tokens"
    return "gte_64k_tokens"


def _cost_bucket(usd: Any) -> str:
    value = _as_float(usd)
    if value <= 0:
        return "none"
    if value < 0.001:
        return "lt_0_001_usd"
    if value < 0.01:
        return "0_001_0_01_usd"
    if value < 0.10:
        return "0_01_0_10_usd"
    if value < 1.0:
        return "0_10_1_usd"
    return "gte_1_usd"


def _latency_bucket(ms: Any) -> str:
    value = _as_float(ms, -1.0)
    if value < 0:
        return "unknown"
    if value < 1_000:
        return "lt_1s"
    if value < 2_000:
        return "1s_2s"
    if value < 10_000:
        return "2s_10s"
    if value < 30_000:
        return "10s_30s"
    return "gte_30s"


def _retry_bucket(rate: Any) -> str:
    value = _as_float(rate)
    if value <= 0:
        return "none"
    if value <= 0.05:
        return "lte_5pct"
    if value <= 0.20:
        return "5pct_20pct"
    return "gt_20pct"


def _error_class(row: dict[str, Any]) -> str | None:
    status_code = _as_int(row.get("last_status_code"), -1)
    if status_code >= 500:
        return "http-5xx"
    if status_code >= 400:
        return "http-4xx"
    if status_code >= 300:
        return "http-3xx"
    error = str(row.get("last_error") or "").strip()
    if not error:
        return None
    candidate = error.split(":", 1)[0].split("(", 1)[0].strip()
    if _ERROR_CLASS_RE.match(candidate):
        return candidate
    return "error"


def _queue_state(status: Any) -> str:
    raw_status = str(status or "unknown")
    if raw_status == "sent":
        return "sent"
    if raw_status in {"queued", "sending", "retryable-error"}:
        return "pending"
    if raw_status in {"error", "dropped-after-limit"}:
        return "error"
    return raw_status


def _public_queue_row(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "status": row.get("status"),
        "queue_state": _queue_state(row.get("status")),
        "attempts": _as_int(row.get("attempts")),
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "last_error_class": _error_class(row),
        "sent_at": row.get("sent_at"),
        "age_seconds": _seconds_since(row.get("created_at"), now),
        "payload_included": False,
    }


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_instructions_included": False,
        "raw_responses_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "cache_keys_included": False,
        "pattern_hashes_included": False,
        "request_fingerprints_included": False,
        "payload_json_included": False,
    }


def build_repeated_scaffold_lifecycle_feedback_status(
    store_obj: Any,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    sample_cap = max(0, min(_as_int(sample_limit, 20), 100))
    rows = (
        store_obj.managed_outcome_feedback_rows(source_surface=SOURCE_SURFACE, limit=10000)
        if hasattr(store_obj, "managed_outcome_feedback_rows")
        else []
    )
    due_rows = (
        store_obj.due_managed_outcome_feedback(limit=max(1, sample_cap or 1), source_surface=SOURCE_SURFACE)
        if hasattr(store_obj, "due_managed_outcome_feedback")
        else []
    )
    status_counts: dict[str, int] = {}
    queue_state_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    error_class_counts: dict[str, int] = {}
    last_success_at: str | None = None
    last_error_at: str | None = None
    last_error_class: str | None = None
    pending_rows: list[dict[str, Any]] = []

    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        state = _queue_state(status)
        queue_state_counts[state] = queue_state_counts.get(state, 0) + 1
        source = str(row.get("source_surface") or "unknown")
        endpoint = str(row.get("endpoint") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        if status in {"queued", "retryable-error"}:
            pending_rows.append(row)
        if status == "sent" and row.get("sent_at"):
            sent_at = str(row.get("sent_at"))
            if last_success_at is None or sent_at > last_success_at:
                last_success_at = sent_at
        klass = _error_class(row)
        if klass:
            error_class_counts[klass] = error_class_counts.get(klass, 0) + 1
            updated_at = str(row.get("updated_at") or row.get("created_at") or "")
            if last_error_at is None or updated_at > last_error_at:
                last_error_at = updated_at
                last_error_class = klass

    oldest_pending = min(
        pending_rows,
        key=lambda row: _parse_utc_iso(row.get("created_at")) or now,
        default=None,
    )
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "due": len(due_rows),
        "sent": status_counts.get("sent", 0),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "error": status_counts.get("error", 0),
        "dropped_after_limit": status_counts.get("dropped-after-limit", 0),
        "pending": queue_state_counts.get("pending", 0),
        "last_success_at": last_success_at,
        "last_error_at": last_error_at,
        "last_error_class": last_error_class,
        "oldest_pending_age_seconds": _seconds_since(oldest_pending.get("created_at"), now) if oldest_pending else None,
    }
    return {
        "schema": STATUS_SCHEMA,
        "source_surface": SOURCE_SURFACE,
        "generated_at": now.isoformat(),
        "summary": summary,
        "status_breakdown": _breakdown_from_counts(status_counts),
        "queue_state_breakdown": _breakdown_from_counts(queue_state_counts),
        "source_surface_breakdown": _breakdown_from_counts(source_counts),
        "endpoint_breakdown": _breakdown_from_counts(endpoint_counts),
        "last_error_class_breakdown": _breakdown_from_counts(error_class_counts),
        "oldest_pending": _public_queue_row(oldest_pending, now=now) if oldest_pending else None,
        "due_samples": [
            _public_queue_row(row, now=now)
            for row in due_rows[:sample_cap]
        ],
        "privacy": _privacy_summary(),
    }


async def flush_repeated_scaffold_lifecycle_feedback(
    store_obj: Any,
    *,
    limit: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    capped = max(1, min(_as_int(limit, 5), 100))
    before = build_repeated_scaffold_lifecycle_feedback_status(store_obj, sample_limit=capped)
    if dry_run:
        results = [
            {**row, "status": "would-send"}
            for row in before.get("due_samples", [])
            if isinstance(row, dict)
        ]
        flush_status = "dry-run"
        reason = "dry-run"
    else:
        from agentflow_proxy import recommendations

        if recommendations.recommendations_enabled():
            raw_results = await recommendations.flush_queued_outcome_feedback(
                store_obj,
                limit=capped,
                source_surface=SOURCE_SURFACE,
            )
            results = [_public_flush_result(item) for item in raw_results]
            flush_status = "completed"
            reason = "ok"
        else:
            results = []
            flush_status = "skipped"
            reason = "managed-feedback-disabled"
    after = build_repeated_scaffold_lifecycle_feedback_status(store_obj, sample_limit=capped)

    result_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        result_counts[status] = result_counts.get(status, 0) + 1
    return {
        "schema": FLUSH_SCHEMA,
        "ok": True,
        "dry_run": bool(dry_run),
        "source_surface": SOURCE_SURFACE,
        "limit": capped,
        "flush": {
            "status": flush_status,
            "reason": reason,
            "attempted": len(results) if not dry_run else 0,
            "would_attempt": len(results) if dry_run else 0,
            "sent": result_counts.get("sent", 0),
            "retryable_error": result_counts.get("retryable-error", 0),
            "dropped_after_limit": result_counts.get("dropped-after-limit", 0),
        },
        "before": before["summary"],
        "after": after["summary"],
        "result_breakdown": _breakdown_from_counts(result_counts),
        "results": results,
        "privacy": _privacy_summary(),
    }


def _public_flush_result(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": item.get("queue_id"),
        "source_surface": SOURCE_SURFACE,
        "endpoint": item.get("endpoint"),
        "status": item.get("status"),
        "attempts": item.get("attempts"),
        "last_status_code": item.get("status_code"),
        "last_error": item.get("error"),
    }
    return {
        "queue_id": item.get("queue_id"),
        "source_surface": SOURCE_SURFACE,
        "endpoint": item.get("endpoint"),
        "status": item.get("status"),
        "queue_state": _queue_state(item.get("status")),
        "reason": item.get("reason"),
        "attempts": _as_int(item.get("attempts")),
        "last_status_code": item.get("status_code"),
        "last_error_class": _error_class(row),
        "payload_included": False,
        "server_url_included": False,
    }


def _candidate_feedback(candidate: dict[str, Any]) -> dict[str, Any]:
    cohorts = candidate.get("cohort_metrics") if isinstance(candidate.get("cohort_metrics"), dict) else {}
    applied = cohorts.get("applied") if isinstance(cohorts.get("applied"), dict) else {}
    holdout = cohorts.get("holdout") if isinstance(cohorts.get("holdout"), dict) else {}
    cohort_counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    reason_codes = [str(item) for item in (candidate.get("reason_codes") or []) if item]
    warning_codes = [str(item) for item in (candidate.get("warning_codes") or []) if item]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "rule_id": candidate.get("rule_id"),
        "provider": candidate.get("provider"),
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "model_tier": candidate.get("model_tier"),
        "policy_source": candidate.get("policy_source"),
        "action_family": "crunch",
        "optimization_family": "repeated_provider_scaffolding",
        "verdict": candidate.get("verdict"),
        "rollout_verdict": candidate.get("rollout_verdict"),
        "next_action": candidate.get("next_action"),
        "reason_codes": reason_codes,
        "warning_codes": warning_codes,
        "rollback_reason_codes": reason_codes if candidate.get("verdict") == "rollback" else [],
        "safety_stop_reason_counts": _breakdown_counts(candidate.get("safety_stop_reason_counts")),
        "skip_reason_counts": _breakdown_counts(candidate.get("skip_reason_counts")),
        "canary_cohort_counts": {
            "applied": _as_int(cohort_counts.get("applied")),
            "holdout": _as_int(cohort_counts.get("holdout")),
            "skipped": _as_int(cohort_counts.get("skipped")),
            "safety_stop": _as_int(cohort_counts.get("safety_stop")),
            "unknown": _as_int(cohort_counts.get("unknown")),
        },
        "status_class_counts": _breakdown_counts(candidate.get("status_buckets")),
        "reason_bucket_counts": _breakdown_counts(candidate.get("reason_buckets")),
        "saved_tokens_bucket": _token_bucket(candidate.get("estimated_saved_tokens")),
        "cost_savings_bucket": _cost_bucket(candidate.get("estimated_savings_usd")),
        "applied_retry_rate_bucket": _retry_bucket(applied.get("retry_rate")),
        "holdout_retry_rate_bucket": _retry_bucket(holdout.get("retry_rate")),
        "applied_latency_bucket": _latency_bucket(applied.get("latency_avg_ms")),
        "holdout_latency_bucket": _latency_bucket(holdout.get("latency_avg_ms")),
        "applied_error_rate": applied.get("error_rate"),
        "holdout_error_rate": holdout.get("error_rate"),
        "applied_retry_rate": applied.get("retry_rate"),
        "holdout_retry_rate": holdout.get("retry_rate"),
        "estimated_saved_tokens": _as_int(candidate.get("estimated_saved_tokens")),
        "estimated_savings_usd": round(_as_float(candidate.get("estimated_savings_usd")), 8),
        "sample_count": _as_int(candidate.get("sample_count")),
        "oldest_observed_at": candidate.get("oldest_observed_at"),
        "latest_observed_at": candidate.get("latest_observed_at"),
        "stale": bool((candidate.get("stale_evidence") or {}).get("stale")) if isinstance(candidate.get("stale_evidence"), dict) else False,
        "privacy": _privacy_summary(),
    }


def build_repeated_scaffold_lifecycle_feedback(report: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [item for item in (report.get("candidates") or []) if isinstance(item, dict)]
    if not candidates:
        return None

    from agentflow_proxy import __version__

    feedback_items = [_candidate_feedback(item) for item in candidates]
    candidate_ids = sorted({str(item.get("candidate_id")) for item in feedback_items if item.get("candidate_id")})
    rule_ids = sorted({str(item.get("rule_id")) for item in feedback_items if item.get("rule_id")})
    basis = {
        "schema": report.get("schema"),
        "generated_at": report.get("generated_at"),
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "status": report.get("status"),
    }
    basis_json = json.dumps(basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(basis_json.encode("utf-8")).hexdigest()
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metadata = {
        "schema": FEEDBACK_SCHEMA,
        "lifecycle_kind": "repeated_scaffold_crunch",
        "command": "repeated-scaffold-impact",
        "local_result_status": report.get("status"),
        "read_only": bool(report.get("read_only", True)),
        "wrote_local_files": bool(report.get("wrote_local_files")),
        "wrote_store": bool(report.get("wrote_store")),
        "candidate_count": len(feedback_items),
        "sampled_call_count": _as_int(summary.get("sampled_call_count")),
        "observed_metadata_row_count": _as_int(summary.get("observed_repeated_scaffold_metadata_row_count")),
        "applied_count": _as_int(summary.get("applied_count")),
        "holdout_count": _as_int(summary.get("holdout_count")),
        "safety_stop_count": _as_int(summary.get("safety_stop_count")),
        "skipped_count": _as_int(summary.get("skipped_count")),
        "estimated_saved_tokens_bucket": _token_bucket(summary.get("estimated_saved_tokens")),
        "estimated_savings_bucket": _cost_bucket(summary.get("estimated_savings_usd")),
        "verdict_counts": _breakdown_counts(summary.get("verdict_counts")),
        "reason_code_counts": _breakdown_counts(summary.get("reason_code_counts")),
        "candidate_feedback": feedback_items,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "privacy": _privacy_summary(),
    }
    event = {
        "event_type": "impact",
        "occurred_at": utc_now(),
        "recommendation_id": candidate_ids[0] if len(candidate_ids) == 1 else f"repeated-scaffold:{digest[:24]}",
        "bundle_hash": f"sha256:{digest}",
        "policy_sections": ["crunch"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }
    assert_managed_egress_safe(event)
    return event
