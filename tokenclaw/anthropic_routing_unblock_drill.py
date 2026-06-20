from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any

from tokenclaw.activation_lifecycle_feedback import (
    SAFETY_STOP_BURNDOWN_SCHEMA,
    build_activation_safety_stop_burndown,
)
from tokenclaw.orchestrator_research import sanitize_value
from tokenclaw.public_metadata import public_id


SCHEMA = "agentflow.anthropic_routing_safety_stop_unblock_drill.v1"
ENTRY_SCHEMA = "agentflow.anthropic_routing_safety_stop_unblock_drill_entry.v1"
ACCEPTANCE_SCHEMA = "agentflow.anthropic_routing_safety_stop_unblock_drill_acceptance.v1"
PRIVACY_SCHEMA = "agentflow.anthropic_routing_safety_stop_unblock_drill_privacy.v1"

REQUIRED_CRITERIA = [
    "safety_stop_reason_review",
    "safer_threshold_or_executor_guard",
    "rollback_proof",
    "applied_coverage",
    "holdout_coverage",
]


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "read_only": True,
        "local_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _source_to_burndown(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("schema") == SAFETY_STOP_BURNDOWN_SCHEMA:
        return source
    if source.get("schema") == "agentflow.pass_through_routing_activation_candidates.v1":
        source = {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {"stats_summary": {"pass_through_routing_report": source}},
        }
    return build_activation_safety_stop_burndown(research_plan=source)


def _passed_criteria(row: dict[str, Any]) -> dict[str, bool]:
    criteria = row.get("unblock_criteria") if isinstance(row.get("unblock_criteria"), dict) else {}
    results = criteria.get("criterion_results") if isinstance(criteria.get("criterion_results"), dict) else {}
    return {
        field: bool((results.get(field) or {}).get("passed"))
        for field in REQUIRED_CRITERIA
    }


def _review_field(row: dict[str, Any], field: str) -> dict[str, Any]:
    value = row.get(field)
    if isinstance(value, dict):
        return sanitize_value(value)
    passed = _passed_criteria(row).get(field, False)
    return {
        "schema": f"agentflow.anthropic_routing_safety_stop_{field}_drill_review.v1",
        "status": "present" if passed else "missing",
        "present": passed,
        "passed": passed,
        "stale": False,
        "reason_codes": [f"{field.replace('_', '-')}-present" if passed else f"{field.replace('_', '-')}-missing"],
        "metadata_only": True,
        "aggregate_only": True,
    }


def _rollback_metadata(row: dict[str, Any], *, stage_allowed: bool) -> dict[str, Any]:
    metadata = row.get("rollback_metadata") if isinstance(row.get("rollback_metadata"), dict) else {}
    result = {
        "schema": "agentflow.anthropic_routing_safety_stop_rollback_metadata.v1",
        "rollback_action_type": "keep_anthropic_routing_policy_disabled",
        "rollback_action": "keep-routing-policy-disabled",
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "disabled_policy_state": "anthropic-routing-canary-disabled",
        "keep_disabled_action": "do-not-stage-or-widen-until-unblock-criteria-pass",
        "active_policy_changed": False,
        "wrote_active_policy_files": False,
        "promotion_allowed": False,
        "stage_allowed": stage_allowed,
        "metadata_only": True,
        "aggregate_only": True,
        "policy_file_contents_included": False,
    }
    for key in (
        "rollback_action_type",
        "rollback_action",
        "target_local_policy_section",
        "target_local_rule_file",
        "disabled_policy_state",
        "keep_disabled_action",
    ):
        if metadata.get(key):
            result[key] = sanitize_value(metadata.get(key))
    return result


def _entry_fingerprint(row: dict[str, Any], stage_allowed: bool) -> str:
    material = {
        "source": row.get("fingerprint") or row.get("policy_ref"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "requested_model": row.get("requested_model"),
        "target_model": row.get("target_model") or row.get("candidate_target_model"),
        "safety_stop_count": _as_int(row.get("safety_stop_count")),
        "stage_allowed": stage_allowed,
    }
    return public_id(json.dumps(material, sort_keys=True), prefix="anthropic-routing-drill") or "anthropic-routing-drill:unknown"


def _drill_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("action_family") != "routing" or row.get("provider") != "anthropic":
        return None
    criteria = _passed_criteria(row)
    safety_stop_count = _as_int(row.get("safety_stop_count"))
    all_criteria_passed = all(criteria.values())
    stage_allowed = safety_stop_count <= 0 and all_criteria_passed
    drill_status = "recovery-ready" if stage_allowed else "keep-blocked"
    acceptance = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "met",
        "reports_all_required_criteria": set(REQUIRED_CRITERIA) <= set(criteria),
        "keeps_canary_disabled": True,
        "promotion_allowed": False,
        "stage_allowed": stage_allowed,
        "active_policy_changed": False,
        "wrote_active_policy_files": False,
        "rollback_metadata_present": True,
        "disabled_policy_rollback_proof": True,
        "metadata_only": True,
        "aggregate_only": True,
    }
    return {
        "schema": ENTRY_SCHEMA,
        "rank": _as_int(row.get("rank")),
        "fingerprint": _entry_fingerprint(row, stage_allowed),
        "source_schema": sanitize_value(row.get("source_schema") or row.get("evidence_schema")),
        "source_rank": _as_int(row.get("rank")),
        "status": drill_status,
        "next_action": "mark-anthropic-routing-recovery-ready" if stage_allowed else "keep-anthropic-routing-blocked-until-safety-stop-burndown",
        "provider": "anthropic",
        "source_surface": sanitize_value(row.get("source_surface")),
        "endpoint": sanitize_value(row.get("endpoint")),
        "category": sanitize_value(row.get("category")),
        "workflow_phase": sanitize_value(row.get("workflow_phase")),
        "requested_model": sanitize_value(row.get("requested_model")),
        "candidate_target_model": sanitize_value(row.get("target_model") or row.get("candidate_target_model")),
        "required_local_executor": sanitize_value(row.get("required_local_executor") or "anthropic-routing-rules"),
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "sample_count": _as_int(row.get("sample_count")),
        "matched_count": _as_int(row.get("matched_count")),
        "observed_count": _as_int(row.get("observed_count")),
        "safety_stop_count": safety_stop_count,
        "applied_count": _as_int(row.get("applied_count")),
        "holdout_count": _as_int(row.get("holdout_count")),
        "projected_savings_usd": round(_as_float(row.get("savings_estimate_usd") or row.get("projected_savings_usd")), 8),
        "criteria_passed": criteria,
        "criterion_results": sanitize_value((row.get("unblock_criteria") or {}).get("criterion_results") if isinstance(row.get("unblock_criteria"), dict) else {}),
        "needed_resolution": sanitize_value(row.get("needed_resolution") or []),
        "evidence_freshness_status": sanitize_value(row.get("evidence_freshness_status")),
        "evidence_age_hours": row.get("evidence_age_hours"),
        "max_evidence_age_hours": row.get("max_evidence_age_hours"),
        "safety_stop_reason_review": _review_field(row, "safety_stop_reason_review"),
        "safer_threshold_or_executor_guard": _review_field(row, "safer_threshold_or_executor_guard"),
        "rollback_proof": _review_field(row, "rollback_proof"),
        "applied_coverage": _review_field(row, "applied_coverage"),
        "holdout_coverage": _review_field(row, "holdout_coverage"),
        "rollback_metadata": _rollback_metadata(row, stage_allowed=stage_allowed),
        "local_file_backed_representation": sanitize_value(row.get("local_file_backed_representation") or {
            "exists": True,
            "policy_section": "routing",
            "policy_source": "local-file-backed",
            "rule_file": "routing_rules.yaml",
            "metadata_only": True,
            "aggregate_only": True,
        }),
        "promotion_allowed": False,
        "stage_allowed": stage_allowed,
        "review_only": True,
        "dry_run_only": True,
        "active_policy_changed": False,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "acceptance": acceptance,
        "privacy": _privacy(),
    }


def build_anthropic_routing_safety_stop_unblock_drill(
    source: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a no-traffic Anthropic routing safety-stop unblock drill."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    source = source if isinstance(source, dict) else {}
    burndown = _source_to_burndown(source)
    rows = [
        entry
        for row in burndown.get("groups") or []
        if isinstance(row, dict)
        for entry in [_drill_entry(row)]
        if isinstance(entry, dict)
    ]
    rows.sort(key=lambda item: (_as_int(item.get("rank")) or 999999, str(item.get("fingerprint") or "")))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    stage_ready = [row for row in rows if row.get("stage_allowed") is True]
    blocked = [row for row in rows if row.get("stage_allowed") is not True]
    all_report_criteria = all(bool((row.get("acceptance") or {}).get("reports_all_required_criteria")) for row in rows) if rows else False
    no_writes = all(row.get("wrote_active_policy_files") is False and row.get("active_policy_changed") is False for row in rows)
    report = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "status": "ranked" if rows else "no-anthropic-routing-safety-stop-evidence",
        "source_schema": sanitize_value(burndown.get("schema")),
        "summary": {
            "drill_entry_count": len(rows),
            "blocked_count": len(blocked),
            "recovery_ready_count": len(stage_ready),
            "stage_ready_count": len(stage_ready),
            "promotion_allowed_count": 0,
            "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in rows),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in rows),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in rows),
            "top_status": rows[0].get("status") if rows else None,
            "top_next_action": rows[0].get("next_action") if rows else None,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "status_counts": [
                {"value": key, "count": count} for key, count in sorted(status_counts.items())
            ],
        },
        "acceptance": {
            "schema": ACCEPTANCE_SCHEMA,
            "status": "met" if rows and all_report_criteria and no_writes else ("not-applicable" if not rows else "failed"),
            "reports_all_required_criteria": all_report_criteria,
            "blocked_rows_keep_stage_and_promotion_disabled": all(row.get("stage_allowed") is False and row.get("promotion_allowed") is False for row in blocked),
            "stage_ready_rows_require_all_criteria": all(all((row.get("criteria_passed") or {}).values()) and _as_int(row.get("safety_stop_count")) == 0 for row in stage_ready),
            "promotion_never_allowed_by_drill": all(row.get("promotion_allowed") is False for row in rows),
            "no_active_policy_write": no_writes,
            "rollback_metadata_present": all(isinstance(row.get("rollback_metadata"), dict) for row in rows) if rows else False,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "entries": rows,
        "source_burndown_summary": sanitize_value(burndown.get("summary") or {}),
        "privacy": _privacy(),
    }
    return sanitize_value(report)
