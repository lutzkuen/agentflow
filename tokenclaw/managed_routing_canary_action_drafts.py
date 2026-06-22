from __future__ import annotations

from collections import Counter
from typing import Any

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import stable_json, utc_now


SCHEMA = "tokenclaw.managed_routing_canary_action_drafts.v1"
ACTION_SCHEMA = "tokenclaw.managed_routing_canary_action_draft.v1"
BLOCKED_SCHEMA = "tokenclaw.managed_routing_canary_blocked_action.v1"
PRIVACY_SCHEMA = "tokenclaw.managed_routing_canary_action_drafts_privacy.v1"
TARGET_LOCAL_RULE_FILE = "routing_canary_policy.yaml"
TARGET_LOCAL_POLICY_SECTION = "routing.canaries"

ACCEPTED_DECISIONS = {"canary", "widen"}
BLOCKED_DECISIONS = {"hold", "rollback", "no-evidence"}
STALE_REASON_CODES = {"routing-lifecycle-stale-evidence", "stale-routing-pathway-matrix"}
MISSING_HOLDOUT_REASON_CODES = {
    "routing-lifecycle-missing-holdout-coverage",
    "missing-holdout-coverage",
}
MISSING_APPLIED_REASON_CODES = {
    "routing-lifecycle-missing-applied-coverage",
    "missing-applied-coverage",
}
LOW_SAVINGS_REASON_CODES = {
    "routing-lifecycle-savings-not-positive",
    "low-savings",
    "missing-positive-savings",
}
UNSAFE_REASON_CODES = {
    "routing-lifecycle-local-executor-incompatible",
    "routing-lifecycle-semantic-regression",
    "routing-lifecycle-safety-stop",
    "routing-lifecycle-rollback",
    "routing-lifecycle-error-observed",
    "routing-lifecycle-error-rate-regression",
    "routing-lifecycle-retry-rate-regression",
    "routing-lifecycle-fallback-rate-regression",
    "semantic-quality-regression-observed",
    "safety-stop-observed",
    "error-observed",
    "fallback-observed",
}


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "feature_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
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


def _label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback=fallback)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reason_codes(*values: Any) -> list[str]:
    reasons: list[str] = []
    for value in values:
        if isinstance(value, list):
            for item in value:
                text = _label(item, fallback="")
                if text and text not in reasons:
                    reasons.append(text)
        else:
            text = _label(value, fallback="")
            if text and text not in reasons:
                reasons.append(text)
    return reasons


def _rows_from_rollups(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollup in _list(source.get("rollups")):
        if not isinstance(rollup, dict):
            continue
        metadata = _dict(rollup.get("metadata"))
        outcome = dict(_dict(metadata.get("pathway_outcome")))
        decision = _dict(metadata.get("pathway_decision"))
        if not outcome and not decision:
            continue
        outcome.setdefault("candidate_fingerprint", rollup.get("recommendation_id"))
        outcome.setdefault("pathway_id", rollup.get("policy_id"))
        outcome.setdefault("source_surface", rollup.get("source_surface"))
        outcome.setdefault("app_family", rollup.get("app_family"))
        outcome.setdefault("requested_model", rollup.get("requested_model"))
        outcome.setdefault("target_model", rollup.get("candidate_target_model"))
        outcome.setdefault("workflow_phase", rollup.get("phase"))
        outcome.setdefault("category", rollup.get("category"))
        outcome.setdefault("text_bucket", rollup.get("text_bucket"))
        outcome.setdefault("token_bucket", rollup.get("token_bucket"))
        outcome["managed_pathway_decision"] = decision
        outcome["managed_rollup_ref"] = public_id(rollup.get("recommendation_id"), prefix="routing-pathway-rollup")
        rows.append(outcome)
    return rows


def _rows_from_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _rows_from_rollups(source)
    if rows:
        return rows
    for key in ("outcomes", "rows", "actions"):
        values = source.get(key)
        if isinstance(values, list):
            return [row for row in values if isinstance(row, dict)]
    metadata = _dict(source.get("metadata"))
    if metadata.get("pathway_outcome") or metadata.get("pathway_decision"):
        row = dict(_dict(metadata.get("pathway_outcome")))
        row["managed_pathway_decision"] = _dict(metadata.get("pathway_decision"))
        return [row]
    return []


def _decision_obj(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("managed_pathway_decision", "pathway_lifecycle_decision"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    value = row.get("pathway_decision")
    if isinstance(value, dict):
        return value
    decision = str(value or row.get("decision") or "").strip().lower()
    if decision in ACCEPTED_DECISIONS | BLOCKED_DECISIONS:
        return {
            "schema": "agentflow.routing_pathway_lifecycle_decision.v1",
            "decision": decision,
            "next_action": row.get("pathway_decision_next_action") or row.get("next_action"),
            "reason_codes": row.get("pathway_decision_reason_codes") or row.get("reason_codes") or [],
            "inputs": row.get("decision_inputs") or {},
        }
    return {}


def _candidate_ref(row: dict[str, Any]) -> str:
    value = row.get("candidate_fingerprint") or row.get("recommendation_id")
    if isinstance(value, str) and value.strip():
        return public_id(value.strip(), prefix="routing-pathway-candidate") or "routing-pathway-candidate:unknown"
    material = {
        "source_surface": row.get("source_surface"),
        "app_family": row.get("app_family"),
        "category": row.get("category"),
        "workflow_phase": row.get("workflow_phase") or row.get("phase"),
        "requested_model": row.get("requested_model") or row.get("requested_model_family"),
        "target_model": row.get("target_model") or row.get("candidate_target_model") or row.get("routed_model"),
    }
    return public_id(stable_json(material), prefix="routing-pathway-candidate") or "routing-pathway-candidate:unknown"


def _draft_ref(row: dict[str, Any], decision: dict[str, Any]) -> str:
    material = {
        "schema": ACTION_SCHEMA,
        "candidate": _candidate_ref(row),
        "decision": decision.get("decision"),
        "next_action": decision.get("next_action"),
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
    }
    return public_id(stable_json(material), prefix="routing-canary-draft") or "routing-canary-draft:unknown"


def _narrow_reason(decision_name: str, reason_codes: list[str]) -> str:
    reasons = set(reason_codes)
    if reasons & STALE_REASON_CODES:
        return "stale-managed-routing-score"
    if reasons & UNSAFE_REASON_CODES or decision_name == "rollback":
        return "unsafe-managed-routing-score"
    if reasons & MISSING_HOLDOUT_REASON_CODES:
        return "missing-holdout-coverage"
    if reasons & MISSING_APPLIED_REASON_CODES:
        return "missing-applied-coverage"
    if reasons & LOW_SAVINGS_REASON_CODES:
        return "low-savings-routing-score"
    if decision_name in BLOCKED_DECISIONS:
        return f"managed-routing-decision-{decision_name}"
    return "managed-routing-score-not-actionable"


def _base_fields(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    inputs = _dict(decision.get("inputs"))
    target_model = row.get("target_model") or row.get("candidate_target_model") or row.get("routed_model")
    return {
        "source_schema": _label(row.get("schema")),
        "managed_decision_schema": _label(decision.get("schema")),
        "source_policy": "managed-recommended",
        "candidate_fingerprint": _candidate_ref(row),
        "pathway_id": public_id(row.get("pathway_id") or row.get("rule_id"), prefix="routing-pathway"),
        "source_surface": _label(row.get("source_surface")),
        "app_family": _label(row.get("app_family")),
        "provider_family": _label(row.get("provider_family") or row.get("provider")),
        "endpoint": _label(row.get("endpoint")),
        "category": _label(row.get("category")),
        "workflow_phase": _label(row.get("workflow_phase") or row.get("phase")),
        "requested_model": _label(row.get("requested_model") or row.get("requested_model_family")),
        "target_model": _label(target_model or row.get("target_model_family")),
        "text_bucket": _label(row.get("text_bucket") or row.get("text_chars_bucket")),
        "token_bucket": _label(row.get("token_bucket")),
        "local_action_family": "routing",
        "local_executor": _label(row.get("required_local_executor") or row.get("local_executor"), fallback="routing-canary-policy"),
        "decision": _label(decision.get("decision")),
        "managed_next_action": _label(decision.get("next_action")),
        "confidence": decision.get("confidence"),
        "applied_count": _as_int(row.get("applied_count") or inputs.get("applied_count")),
        "holdout_count": _as_int(row.get("holdout_count") or inputs.get("holdout_count")),
        "safety_stop_count": _as_int(row.get("safety_stop_count") or inputs.get("safety_stop_count")),
        "rollback_count": _as_int(row.get("rollback_count") or inputs.get("rollback_count")),
        "error_count": _as_int(row.get("error_count") or inputs.get("error_count")),
        "fallback_count": _as_int(row.get("fallback_count") or inputs.get("fallback_count")),
        "retry_count": _as_int(row.get("retry_count") or inputs.get("retry_count")),
        "observed_savings_usd": _as_float(
            row.get("observed_savings_usd") or row.get("observed_saved_usd") or inputs.get("observed_savings_usd")
        ),
        "projected_savings_usd": _as_float(
            row.get("projected_savings_usd") or row.get("projected_saved_usd") or inputs.get("projected_savings_usd")
        ),
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
        "target_local_policy_section": TARGET_LOCAL_POLICY_SECTION,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "routing_apply_action_count": 0,
        "privacy": _privacy(),
    }


def _rollback_metadata(reason_codes: list[str]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.managed_routing_canary_rollback_metadata.v1",
        "rollback_action_type": "disable-routing-canary-draft",
        "rollback_canary_fraction": 0.0,
        "rollback_holdout_fraction": 0.0,
        "reason_codes": _reason_codes(
            reason_codes,
            [
                "operator-requested",
                "safety-stop-observed",
                "error-rate-regression",
                "retry-or-fallback-regression",
            ],
        ),
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
        "policy_files_written": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _action_draft(row: dict[str, Any], decision: dict[str, Any], reason_codes: list[str]) -> dict[str, Any]:
    decision_name = _label(decision.get("decision"))
    canary = _dict(row.get("canary"))
    default_canary = 0.25 if decision_name == "widen" else 0.10
    canary_fraction = _bounded_fraction(canary.get("canary_fraction"), default_canary)
    holdout_fraction = _bounded_fraction(canary.get("holdout_fraction"), 0.10)
    base = _base_fields(row, decision)
    draft = {
        **base,
        "schema": ACTION_SCHEMA,
        "status": "drafted",
        "action_type": "widen-routing-canary-draft" if decision_name == "widen" else "stage-routing-canary-draft",
        "reason": "managed-routing-score-accepted",
        "reason_codes": _reason_codes(reason_codes, "managed-routing-score-accepted"),
        "draft_fingerprint": _draft_ref(row, decision),
        "active_policy_write": False,
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "routing_canary_policy_patch": {
            "schema": "tokenclaw.managed_routing_canary_policy_patch_draft.v1",
            "patch_type": "stage_or_widen_routing_canary",
            "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
            "target_local_policy_section": TARGET_LOCAL_POLICY_SECTION,
            "source_policy": "managed-recommended",
            "candidate_fingerprint": base["candidate_fingerprint"],
            "requested_model": base["requested_model"],
            "target_model": base["target_model"],
            "source_surface": base["source_surface"],
            "category": base["category"],
            "workflow_phase": base["workflow_phase"],
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "enabled": False,
            "review_required": True,
            "policy_files_written": False,
        },
        "rollback_metadata": _rollback_metadata(reason_codes),
    }
    return {key: value for key, value in draft.items() if value not in (None, "", [])}


def _blocked_action(row: dict[str, Any], decision: dict[str, Any], reason_codes: list[str]) -> dict[str, Any]:
    decision_name = _label(decision.get("decision"))
    reason = _narrow_reason(decision_name, reason_codes)
    blocked = {
        **_base_fields(row, decision),
        "schema": BLOCKED_SCHEMA,
        "status": "blocked",
        "reason": reason,
        "reason_codes": _reason_codes(reason_codes, reason),
        "blocked_action_type": "keep-routing-canary-draft-blocked",
        "draft_fingerprint": _draft_ref(row, decision),
        "active_policy_write": False,
        "canary_fraction": 0.0,
        "holdout_fraction": 0.0,
        "rollback_metadata": _rollback_metadata(reason_codes),
    }
    return {key: value for key, value in blocked.items() if value not in (None, "", [])}


def _row_to_action(row: dict[str, Any]) -> dict[str, Any]:
    decision = _decision_obj(row)
    reason_codes = _reason_codes(
        row.get("reason_codes"),
        row.get("blocker_codes"),
        decision.get("reason_codes"),
        row.get("pathway_decision_reason_codes"),
    )
    decision_name = _label(decision.get("decision"), fallback="unknown")
    if decision_name in ACCEPTED_DECISIONS and not (set(reason_codes) & (STALE_REASON_CODES | UNSAFE_REASON_CODES | LOW_SAVINGS_REASON_CODES)):
        return _action_draft(row, decision, reason_codes)
    return _blocked_action(row, decision or {"decision": decision_name, "reason_codes": reason_codes}, reason_codes)


def build_managed_routing_canary_action_drafts(source: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_from_source(source)
    actions = [_row_to_action(row) for row in rows]
    drafted = [row for row in actions if row.get("status") == "drafted"]
    blocked = [row for row in actions if row.get("status") == "blocked"]
    reason_counts = Counter(str(row.get("reason") or "unknown") for row in actions)
    decision_counts = Counter(str(row.get("decision") or "unknown") for row in actions)
    source_counts = Counter(str(row.get("source_surface") or "unknown") for row in actions)
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "drafted" if drafted else "blocked" if blocked else "empty",
        "source_schema": _label(source.get("schema")),
        "managed_dependency": "optional",
        "review_only": True,
        "authoritative_for_active_policy": False,
        "feature_only": True,
        "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
        "target_local_policy_section": TARGET_LOCAL_POLICY_SECTION,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "routing_apply_action_count": 0,
        "summary": {
            "scored_row_count": len(rows),
            "action_count": len(actions),
            "action_draft_count": len(drafted),
            "blocked_action_count": len(blocked),
            "routing_apply_action_count": 0,
            "policy_files_written": False,
            "target_local_rule_file": TARGET_LOCAL_RULE_FILE,
            "reason_counts": [{"value": key, "count": value} for key, value in sorted(reason_counts.items())],
            "decision_counts": [{"value": key, "count": value} for key, value in sorted(decision_counts.items())],
            "source_surface_counts": [{"value": key, "count": value} for key, value in sorted(source_counts.items())],
        },
        "actions": drafted,
        "blocked_actions": blocked,
        "privacy": _privacy(),
    }
    violations = managed_egress_violations(result)
    result["egress_guard"] = {
        "schema": "tokenclaw.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        result["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return result
