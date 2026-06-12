from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.store import utc_now


FEEDBACK_SCHEMA = "agentflow.repeated_scaffold_lifecycle_feedback.v1"
SOURCE_SURFACE = "repeated_scaffold_lifecycle"


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
