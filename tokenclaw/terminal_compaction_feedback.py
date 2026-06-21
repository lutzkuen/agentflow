from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.managed_egress import assert_managed_egress_safe
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


FEEDBACK_SCHEMA = "tokenclaw.terminal_output_compaction_lifecycle_feedback.v1"
SOURCE_SURFACE = "terminal_output_compaction_lifecycle"


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
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_terminal_text_included": False,
        "raw_terminal_lines_included": False,
        "raw_tool_payloads_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "policy_file_contents_included": False,
        "secrets_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _event_type(command: str, result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    if not result.get("ok", True):
        return "rejected"
    if command == "apply":
        return "applied"
    if command == "impact":
        if _as_int(summary.get("rollback_action_count")) > 0:
            return "rollback"
        if _as_int(summary.get("safety_stop_count")) > 0:
            return "safety-stop"
        if _as_int(summary.get("applied_count")) > 0:
            return "canary-applied"
        if _as_int(summary.get("holdout_count")) > 0:
            return "holdout"
    planned = _as_int(summary.get("planned_call_count") or result.get("planned_action_count"))
    changed = _as_int(result.get("changed_action_count"))
    return "reviewed" if planned > 0 or changed > 0 else "rejected"


def _rule_id_from_result(result: dict[str, Any]) -> str:
    policy = result.get("policy") if isinstance(result.get("policy"), dict) else {}
    return _safe_label(policy.get("rule_id") or "terminal-output-compaction", "terminal-output-compaction")


def _dry_run_snapshots(result: dict[str, Any]) -> list[dict[str, Any]]:
    policy = result.get("policy") if isinstance(result.get("policy"), dict) else {}
    rule_id = _safe_label(policy.get("rule_id") or "local-terminal-output-compaction-dry-run", "unknown")
    snapshots: list[dict[str, Any]] = []
    for item in result.get("plans") or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        snapshot = {
            "candidate_id": _safe_id(item.get("candidate_id"), prefix="terminal-compaction-candidate"),
            "rule_id": rule_id,
            "lifecycle_status": "reviewed" if status == "planned" else "rejected",
            "decision_status": status,
            "policy_source": _safe_label(item.get("policy_source") or policy.get("policy_source") or "unknown"),
            "source_surface": _safe_label(item.get("source_surface") or "unknown"),
            "category": _safe_label(item.get("category") or "unknown"),
            "model_family": _safe_label(item.get("model_family") or "unknown"),
            "stream": bool(item.get("stream")),
            "status_code_bucket": _safe_label(item.get("status_code_bucket") or "unknown"),
            "blockers": [_safe_label(value, "unknown") for value in item.get("blockers") or []],
            "projected_saved_tokens": _as_int(item.get("projected_saved_tokens")),
            "projected_saved_chars": _as_int(item.get("projected_saved_chars")),
            "projected_saved_usd": round(_as_float(item.get("projected_saved_usd")), 8),
            "target_count": _as_int(item.get("target_count")),
            "preservation_flags": {
                str(key): bool(value)
                for key, value in (item.get("preservation_flags") or {}).items()
                if isinstance(key, str)
            } if isinstance(item.get("preservation_flags"), dict) else {},
        }
        snapshots.append({
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        })
    return snapshots


def _impact_snapshots(result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for item in result.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        cohorts = item.get("cohorts") if isinstance(item.get("cohorts"), dict) else {}
        cohort_counts = {
            name: _as_int(value.get("count")) if isinstance(value, dict) else 0
            for name, value in cohorts.items()
        }
        deltas = item.get("deltas") if isinstance(item.get("deltas"), dict) else {}
        snapshot = {
            "candidate_id": _safe_id(item.get("candidate_id"), prefix="terminal-compaction-candidate"),
            "rule_id": _safe_label(item.get("rule_id") or "unknown", "unknown"),
            "lifecycle_status": _safe_label(item.get("verdict") or "unknown", "unknown"),
            "decision_status": _safe_label(item.get("verdict") or "unknown", "unknown"),
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
                "skipped": cohort_counts.get("skipped", 0),
                "safety_stop": cohort_counts.get("safety_stop", 0),
            },
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


def _rollout_snapshots(result: dict[str, Any], *, command: str) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for item in result.get("actions") or []:
        if not isinstance(item, dict):
            continue
        family = str(item.get("rule_collection") or item.get("candidate_family") or item.get("policy_section") or "")
        proposed = item.get("proposed_edit") if isinstance(item.get("proposed_edit"), dict) else {}
        rule = proposed.get("rule") if isinstance(proposed.get("rule"), dict) else proposed
        if "terminal" not in family and not (
            isinstance(rule, dict)
            and str((rule.get("action") or {}).get("type") if isinstance(rule.get("action"), dict) else "").replace("-", "_")
            in {"compact_terminal_output", "terminal_output_compaction"}
        ):
            continue
        canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
        safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
        snapshot = {
            "action_id": _safe_id(item.get("action_id"), prefix="terminal-compaction-action", fallback="unknown"),
            "candidate_id": _safe_id(
                item.get("target_candidate_id") or rule.get("candidate_id"),
                prefix="terminal-compaction-candidate",
                fallback="unknown",
            ),
            "rule_id": _safe_label(item.get("target_rule_id") or rule.get("id") or rule.get("rule_id"), "unknown"),
            "lifecycle_status": "applied" if command == "apply" and result.get("ok") and not result.get("dry_run") else "reviewed",
            "decision_status": _safe_label(item.get("status") or ("planned" if result.get("ok") else "rejected"), "unknown"),
            "action_type": _safe_label(item.get("action_type") or "unknown", "unknown"),
            "policy_source": _safe_label(rule.get("policy_source") or item.get("policy_source") or "managed-recommended", "unknown"),
            "canary_fraction": canary.get("canary_fraction") or canary.get("fraction"),
            "holdout_fraction": canary.get("holdout_fraction"),
            "safety_stop_enabled": bool(safety.get("enabled")) if safety else None,
            "changed": bool(proposed.get("changed") or item.get("changed")),
            "blockers": [_safe_label(value, "unknown") for value in item.get("blockers") or []],
        }
        snapshots.append({
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", [], {})
        })
    return snapshots


def _action_snapshots(result: dict[str, Any], *, command: str) -> list[dict[str, Any]]:
    schema = str(result.get("schema") or "")
    if schema == "tokenclaw.terminal_output_compaction_dry_run.v1":
        return _dry_run_snapshots(result)
    if schema == "tokenclaw.terminal_output_compaction_impact.v1":
        return _impact_snapshots(result)
    return _rollout_snapshots(result, command=command)


def build_terminal_output_compaction_lifecycle_feedback(
    result: dict[str, Any],
    *,
    command: str,
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    snapshots = _action_snapshots(result, command=command)
    if not snapshots:
        return None

    from tokenclaw import __version__

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    event_type = _event_type(command, result)
    candidate_ids = sorted({str(item.get("candidate_id")) for item in snapshots if item.get("candidate_id")})
    rule_ids = sorted({str(item.get("rule_id")) for item in snapshots if item.get("rule_id")})
    action_ids = sorted({str(item.get("action_id")) for item in snapshots if item.get("action_id")})
    cohort_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for item in snapshots:
        for key, value in (item.get("actual_cohort_counts") or item.get("cohort_counts") or {}).items():
            cohort_counts[str(key)] = cohort_counts.get(str(key), 0) + _as_int(value)
        for reason in item.get("reason_codes") or item.get("blockers") or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

    basis = {
        "schema": result.get("schema"),
        "command": command,
        "event_type": event_type,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "action_ids": action_ids,
        "generated_at": result.get("generated_at"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    metadata = {
        "schema": FEEDBACK_SCHEMA,
        "lifecycle_kind": "terminal_output_compaction",
        "command": f"terminal-output-compaction-{command}",
        "event_type": event_type,
        "local_result_status": "ok" if result.get("ok", True) else "error",
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only", command in {"review", "dry-run", "impact"})),
        "wrote_local_files": bool(result.get("wrote_local_files") or result.get("wrote_policy_files")),
        "wrote_store": bool(result.get("wrote_store")),
        "rule_id": _rule_id_from_result(result),
        "action_count": len(snapshots),
        "candidate_count": len(candidate_ids),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "cohort_counts": cohort_counts,
        "reason_code_counts": reason_counts,
        "reviewed_count": sum(1 for item in snapshots if item.get("lifecycle_status") == "reviewed"),
        "applied_count": _as_int(summary.get("applied_count")) or sum(1 for item in snapshots if item.get("lifecycle_status") == "applied"),
        "holdout_count": _as_int(summary.get("holdout_count")),
        "canary_applied_count": _as_int(summary.get("applied_count")),
        "safety_stop_count": _as_int(summary.get("safety_stop_count")),
        "rollback_action_count": _as_int(summary.get("rollback_action_count")),
        "projected_saved_tokens": _as_int(summary.get("projected_saved_tokens")),
        "projected_saved_usd": round(_as_float(summary.get("projected_saved_usd")), 8),
        "net_savings_usd": round(_as_float(summary.get("net_savings_usd")), 8),
        "projected_holdout_savings_usd": round(_as_float(summary.get("projected_holdout_savings_usd")), 8),
        "verdict_counts": _breakdown_dict(summary.get("verdict_counts")),
        "status_breakdown": _breakdown_dict(summary.get("status_breakdown")),
        "action_snapshots": snapshots,
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "privacy": _privacy_summary(),
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    event = {
        "event_type": event_type,
        "occurred_at": utc_now(),
        "recommendation_id": candidate_ids[0] if len(candidate_ids) == 1 else f"terminal-output-compaction:{digest[:24]}",
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


async def queue_terminal_output_compaction_lifecycle_feedback(
    store_obj: Any,
    result: dict[str, Any],
    *,
    command: str,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    from tokenclaw import recommendations

    payload = build_terminal_output_compaction_lifecycle_feedback(result, command=command)
    if payload is None:
        return {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-terminal-output-compaction-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        }
    return await recommendations.queue_policy_event_feedback(
        store_obj,
        payload,
        source_surface=SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=flush_immediately,
    )
