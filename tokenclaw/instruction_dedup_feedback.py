from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.managed_egress import assert_managed_egress_safe
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


FEEDBACK_SCHEMA = "tokenclaw.instruction_dedup_lifecycle_feedback.v1"
SOURCE_SURFACE = "instruction_dedup_lifecycle"


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_id(value: Any, *, prefix: str, fallback: str = "unknown") -> str:
    return public_id(value, prefix=prefix, fallback=fallback) or fallback


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback)


def _breakdown_dict(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "unknown")
        count = _as_int(item.get("count"))
        if count:
            counts[value] = counts.get(value, 0) + count
    return dict(sorted(counts.items()))


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_instruction_text_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "terminal_output_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "tenant_ids_included": False,
        "policy_file_contents_included": False,
        "instruction_section_fingerprints_included": False,
        "secrets_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _event_type(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    if not result.get("ok", True):
        return "rejected"
    if _as_int(summary.get("rollback_action_count")) > 0:
        return "rollback"
    if _as_int(summary.get("safety_stop_count")) > 0:
        return "safety-stop"
    if _as_int(summary.get("applied_count")) > 0:
        return "runtime-selected"
    if _as_int(summary.get("holdout_count")) > 0:
        return "holdout"
    if _as_int(summary.get("blocked_count")) > 0:
        return "runtime-suppressed"
    if _as_int(summary.get("candidate_group_count")) > 0:
        return "reviewed"
    return "rejected"


def _snapshots(result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for item in result.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        cohorts = item.get("cohorts") if isinstance(item.get("cohorts"), dict) else {}
        cohort_counts = {
            str(name): _as_int(value.get("count")) if isinstance(value, dict) else 0
            for name, value in cohorts.items()
        }
        deltas = item.get("deltas") if isinstance(item.get("deltas"), dict) else {}
        snapshot = {
            "candidate_id": _safe_id(item.get("candidate_id"), prefix="instruction-dedup-candidate"),
            "rule_id": _safe_label(item.get("rule_id") or "unknown", "unknown"),
            "lifecycle_status": _safe_label(item.get("next_action") or "unknown", "unknown"),
            "decision_status": _safe_label(item.get("next_action") or "unknown", "unknown"),
            "policy_source": _safe_label(item.get("policy_source") or "unknown", "unknown"),
            "provider": _safe_label(item.get("provider") or "unknown", "unknown"),
            "source_surface": _safe_label(item.get("source_surface") or "unknown", "unknown"),
            "endpoint": _safe_label(item.get("endpoint") or "unknown", "unknown"),
            "category": _safe_label(item.get("category") or "unknown", "unknown"),
            "workflow_phase": _safe_label(item.get("workflow_phase") or "unknown", "unknown"),
            "requested_model_family": _safe_label(item.get("requested_model_family") or "unknown", "unknown"),
            "routed_model_family": _safe_label(item.get("routed_model_family") or "unknown", "unknown"),
            "stream": bool(item.get("stream")),
            "cohort_counts": cohort_counts,
            "actual_cohort_counts": {
                "canary_applied": cohort_counts.get("applied", 0),
                "canary_holdout": cohort_counts.get("holdout", 0),
                "runtime_suppressed": cohort_counts.get("blocked", 0),
                "safety_stop": cohort_counts.get("safety_stop", 0),
                "skipped": cohort_counts.get("skipped", 0),
            },
            "coordinator_conflict_count": _as_int(item.get("coordinator_conflict_count")),
            "reason_codes": [_safe_label(value, "unknown") for value in item.get("reason_codes") or []],
            "error_rate_delta": deltas.get("error_rate_delta"),
            "retry_rate_delta": deltas.get("retry_rate_delta"),
            "latency_avg_ms_delta": deltas.get("latency_avg_ms_delta"),
            "cost_avg_usd_delta": deltas.get("cost_avg_usd_delta"),
            "net_savings_usd": round(_as_float(item.get("net_savings_usd")), 8),
            "projected_holdout_savings_usd": round(_as_float(item.get("projected_holdout_savings_usd")), 8),
        }
        snapshots.append({
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        })
    return snapshots


def build_instruction_dedup_lifecycle_feedback(result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    snapshots = _snapshots(result)
    if not snapshots:
        return None

    from tokenclaw import __version__

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    event_type = _event_type(result)
    candidate_ids = sorted({str(item.get("candidate_id")) for item in snapshots if item.get("candidate_id")})
    rule_ids = sorted({str(item.get("rule_id")) for item in snapshots if item.get("rule_id")})
    cohort_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for item in snapshots:
        for key, value in (item.get("actual_cohort_counts") or item.get("cohort_counts") or {}).items():
            cohort_counts[str(key)] = cohort_counts.get(str(key), 0) + _as_int(value)
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

    basis = {
        "schema": result.get("schema"),
        "event_type": event_type,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "generated_at": result.get("generated_at"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    metadata = {
        "schema": FEEDBACK_SCHEMA,
        "lifecycle_kind": "instruction_section_deduplication",
        "command": "instruction-dedup-impact",
        "event_type": event_type,
        "local_result_status": "ok" if result.get("ok", True) else "error",
        "dry_run": False,
        "read_only": bool(result.get("read_only", True)),
        "wrote_local_files": bool(result.get("wrote_local_files")),
        "wrote_store": bool(result.get("wrote_store")),
        "action_count": len(snapshots),
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "cohort_counts": cohort_counts,
        "reason_code_counts": reason_counts,
        "reviewed_count": _as_int(summary.get("candidate_group_count")),
        "applied_count": _as_int(summary.get("applied_count")),
        "holdout_count": _as_int(summary.get("holdout_count")),
        "runtime_suppressed_count": _as_int(summary.get("blocked_count")),
        "safety_stop_count": _as_int(summary.get("safety_stop_count")),
        "rollback_action_count": _as_int(summary.get("rollback_action_count")),
        "saved_tokens_est": _as_int(summary.get("saved_tokens_est")),
        "projected_saved_usd": round(_as_float(summary.get("projected_saved_usd")), 8),
        "net_savings_usd": round(_as_float(summary.get("net_savings_usd")), 8),
        "projected_holdout_savings_usd": round(_as_float(summary.get("projected_holdout_savings_usd")), 8),
        "next_action_counts": _breakdown_dict(summary.get("next_action_counts")),
        "status_breakdown": _breakdown_dict(summary.get("status_breakdown")),
        "action_snapshots": snapshots,
        "privacy": _privacy_summary(),
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    event = {
        "event_type": event_type,
        "occurred_at": utc_now(),
        "recommendation_id": candidate_ids[0] if len(candidate_ids) == 1 else f"instruction-dedup:{digest[:24]}",
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


async def queue_instruction_dedup_lifecycle_feedback(
    store_obj: Any,
    result: dict[str, Any],
    *,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    from tokenclaw import recommendations

    payload = build_instruction_dedup_lifecycle_feedback(result)
    if payload is None:
        return {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-instruction-dedup-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        }
    return await recommendations.queue_policy_event_feedback(
        store_obj,
        payload,
        source_surface=SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=flush_immediately,
    )
