from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any

from tokenclaw.orchestrator_research import (
    build_activation_burndown_report,
    sanitize_value,
)
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.public_metadata import public_id


SCHEMA = "agentflow.local_activation_executor_plan.v1"
ENTRY_SCHEMA = "agentflow.local_activation_executor_plan_entry.v1"
PRIVACY_SCHEMA = "agentflow.local_activation_executor_privacy.v1"
HANDOFF_SCHEMA = "agentflow.local_activation_managed_handoff.v1"
HANDOFF_ROW_SCHEMA = "agentflow.local_activation_managed_handoff_row.v1"
HANDOFF_PRIVACY_SCHEMA = "agentflow.local_activation_managed_handoff_privacy.v1"
PREVIEW_REQUEST_SCHEMA = "agentflow.managed_activation_preview_request.v1"
PREVIEW_RESULT_SCHEMA = "agentflow.managed_activation_preview_result.v1"
PREVIEW_DECISION_SCHEMA = "agentflow.managed_activation_preview_decision.v1"

SAFE_SELECTABLE_CLASSES = {"draft-local-policy", "review-only", "retire"}
SUPPORTED_PREVIEW_LOCAL_ACTION_FAMILIES = ["routing", "crunch", "cache", "activation-feedback"]
PREVIEW_REQUIRED_LOCAL_ACTION_FAMILIES = {"routing", "cache", "activation-feedback"}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
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


def _handoff_privacy() -> dict[str, Any]:
    privacy = _privacy()
    privacy.update(
        {
            "schema": HANDOFF_PRIVACY_SCHEMA,
            "feature_only": True,
            "locally_executed": True,
            "content_free": True,
            "provider_forwarding": False,
            "server_content_processing": False,
            "managed_enforced": False,
            "server_ingestion_required": False,
        }
    )
    return privacy


def _input_to_burndown(source: dict[str, Any], *, now: datetime | None) -> dict[str, Any]:
    schema = str(source.get("schema") or "")
    if schema == "agentflow.activation_burndown.v1":
        return source
    if schema == "agentflow.local_activation_next_action_queue.v1":
        source = {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {"stats_summary": {"local_activation_next_action_queue": source}},
        }
    elif schema == "agentflow.evidence_to_activation_next_action_ledger.v1":
        source = {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {"stats_summary": {"evidence_to_activation_next_action_ledger": source}},
        }
    return build_activation_burndown_report(source, now=now)


def _has_any(row: dict[str, Any], *needles: str) -> bool:
    text = " ".join(
        [
            str(row.get("local_action_family") or ""),
            str(row.get("successor_status") or ""),
            str(row.get("current_status") or ""),
            str(row.get("current_state") or ""),
            str(row.get("next_action") or ""),
            str(row.get("unblock_reason") or ""),
            str(row.get("evidence_schema") or ""),
            " ".join(str(item) for item in row.get("blocker_codes") or []),
        ]
    ).lower()
    return any(needle.lower() in text for needle in needles)


def _nested_dict(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _managed_preview_gate(row: dict[str, Any]) -> dict[str, Any]:
    return _nested_dict(row, "managed_preview_gate")


def _managed_preview_required(row: dict[str, Any]) -> bool:
    if row.get("managed_preview_required") is not None:
        return bool(row.get("managed_preview_required"))
    gate = _managed_preview_gate(row)
    if gate.get("required") is not None:
        return bool(gate.get("required"))
    family = str(row.get("local_action_family") or row.get("lever") or "").strip()
    current_status = str(row.get("current_status") or "").strip()
    duplicate_status = str(row.get("duplicate_suppression_status") or "").strip()
    if duplicate_status == "suppressed" or current_status in {"full-rollout", "superseded"}:
        return False
    return family in PREVIEW_REQUIRED_LOCAL_ACTION_FAMILIES


def _preview_status(row: dict[str, Any], gate: dict[str, Any]) -> str:
    return str(
        row.get("preview_verification_status")
        or gate.get("status")
        or ("preview-required" if _managed_preview_required(row) else "preview-optional")
    ).strip()


def _preview_decision_value(row: dict[str, Any], gate: dict[str, Any]) -> str:
    return str(row.get("preview_verification_decision") or gate.get("decision") or "").strip()


def _full_rollout_outcome(row: dict[str, Any]) -> dict[str, Any]:
    return _nested_dict(row, "full_rollout_activation_outcome", "durable_full_rollout_outcome")


def _full_rollout_gate(row: dict[str, Any]) -> dict[str, Any]:
    gate = _nested_dict(row, "keep_active_regression_gate")
    if gate:
        return gate
    outcome = _full_rollout_outcome(row)
    return outcome.get("keep_active_regression_gate") if isinstance(outcome.get("keep_active_regression_gate"), dict) else {}


def _is_full_rollout_crunch(row: dict[str, Any]) -> bool:
    family = str(row.get("local_action_family") or row.get("lever") or "").strip()
    current_status = str(row.get("current_status") or "").strip()
    current_state = str(row.get("current_state") or row.get("state") or "").strip()
    if family != "crunch":
        return False
    return bool(
        current_status == "full-rollout"
        or current_state == "full-rollout-active"
        or row.get("measured_full_rollout_activation")
        or _full_rollout_outcome(row)
    )


def _executor_action_class(row: dict[str, Any]) -> str:
    status = str(row.get("successor_status") or "").strip()
    current_status = str(row.get("current_status") or "").strip()
    current_state = str(row.get("current_state") or "").strip()
    if _is_full_rollout_crunch(row) or status == "keep-current-rule" or current_status == "full-rollout":
        gate_state = str(_full_rollout_gate(row).get("state") or "").strip()
        outcome_value = str(row.get("full_rollout_outcome") or _full_rollout_outcome(row).get("outcome") or "").strip()
        if gate_state == "rollback-required" or outcome_value == "rollback-required":
            return "rollback-required"
        if gate_state == "review-stale-evidence" or outcome_value == "review-stale-evidence":
            return "review-only"
        if gate_state == "keep-blocked" or outcome_value == "keep-blocked":
            return "keep-blocked"
        return "keep-current-rule"
    if status == "suppress-duplicate" or str(row.get("duplicate_suppression_status") or "") == "suppressed":
        return "suppress-duplicate"
    if current_status == "superseded" or current_state in {"retired-no-repeat", "superseded"} or _has_any(row, "retire-cache-replay"):
        return "retire"
    if _has_any(row, "semantic-quality-regression-observed", "review-openai-routing-canary-blockers"):
        return "review-only"
    if _has_any(
        row,
        "safety-stop",
        "missing-applied-coverage",
        "missing-holdout-coverage",
        "rollback-proof-missing",
        "unsafe-tool-calls-without-invalidation",
        "invalidation-evidence-missing",
    ):
        return "keep-blocked"
    if status == "ready" and row.get("target_local_rule_file") and row.get("target_local_policy_section"):
        return "draft-local-policy"
    if status in {"review", "review-only"}:
        return "review-only"
    if status == "keep-blocked":
        return "keep-blocked"
    return "review-only"


def _executor_next_action(row: dict[str, Any], action_class: str) -> str:
    original = str(row.get("next_action") or "inspect-local-evidence").strip()
    if action_class == "keep-current-rule":
        outcome = _full_rollout_outcome(row)
        return str(
            row.get("full_rollout_successor_next_action")
            or outcome.get("successor_next_action")
            or "keep-current-rule-only"
        ).strip()
    if action_class in {"review-only", "rollback-required", "keep-blocked"} and _is_full_rollout_crunch(row):
        gate = _full_rollout_gate(row)
        gate_next = str(gate.get("deterministic_next_action") or gate.get("next_action") or "").strip()
        if gate_next:
            return gate_next
    if action_class == "suppress-duplicate":
        return "suppress-duplicate-successor"
    if action_class == "keep-blocked" and not original:
        return "keep-blocked-until-safety-or-evidence-clears"
    return original or "inspect-local-evidence"


def _executor_reason_codes(row: dict[str, Any], action_class: str) -> list[str]:
    codes = {str(item) for item in row.get("blocker_codes") or [] if str(item or "").strip()}
    gate = _full_rollout_gate(row)
    if isinstance(gate.get("reason_codes"), list):
        codes.update(str(item) for item in gate.get("reason_codes") or [] if str(item or "").strip())
    outcome = _full_rollout_outcome(row)
    for key in ("outcome", "next_action", "successor_decision", "successor_next_action", "successor_no_op_reason"):
        value = str(outcome.get(key) or row.get(f"full_rollout_{key}") or "").strip()
        if value:
            codes.add(value)
    for key in ("unblock_reason", "duplicate_suppression_reason", "current_status", "current_state"):
        value = str(row.get(key) or "").strip()
        if value and value != "unknown":
            codes.add(value)
    codes.add(action_class)
    return sorted(sanitize_value(list(codes)))


def _entry_fingerprint(row: dict[str, Any], action_class: str, next_action: str) -> str:
    material = {
        "source": row.get("fingerprint") or row.get("source_fingerprint"),
        "class": action_class,
        "next_action": next_action,
        "target": row.get("target_local_rule_file"),
    }
    return public_id(json.dumps(material, sort_keys=True), prefix="executor") or "executor:unknown"


def _executor_entry(row: dict[str, Any]) -> dict[str, Any]:
    action_class = _executor_action_class(row)
    next_action = _executor_next_action(row, action_class)
    review_only = action_class in {"review-only", "keep-blocked", "retire", "rollback-required"}
    outcome = _full_rollout_outcome(row)
    gate = _full_rollout_gate(row)
    duplicate_suppression_status = str(row.get("duplicate_suppression_status") or "").strip()
    entry = {
        "schema": ENTRY_SCHEMA,
        "rank": _as_int(row.get("rank")),
        "selected": False,
        "executor_action_class": action_class,
        "executor_status": "selectable" if action_class in SAFE_SELECTABLE_CLASSES else "blocked-or-terminal",
        "executor_next_action": next_action,
        "fingerprint": _entry_fingerprint(row, action_class, next_action),
        "source_successor_fingerprint": sanitize_value(row.get("fingerprint")),
        "source_fingerprint": sanitize_value(row.get("source_fingerprint") or row.get("fingerprint")),
        "source_rank": _as_int(row.get("source_rank") or row.get("rank")),
        "source_queue_rank": _as_int(row.get("source_queue_rank")),
        "source_ledger_rank": _as_int(row.get("source_ledger_rank")),
        "lever": sanitize_value(row.get("lever") or "unknown"),
        "local_action_family": sanitize_value(row.get("local_action_family") or row.get("lever") or "unknown"),
        "provider_scope": sanitize_value(row.get("provider_scope")),
        "current_status": sanitize_value(row.get("current_status") or "unknown"),
        "current_state": sanitize_value(row.get("current_state") or "unknown"),
        "successor_status": sanitize_value(row.get("successor_status") or "review"),
        "target_local_policy_section": sanitize_value(row.get("target_local_policy_section")),
        "target_local_rule_file": sanitize_value(row.get("target_local_rule_file")),
        "blocker_codes": sanitize_value([str(item) for item in row.get("blocker_codes") or [] if str(item or "").strip()]),
        "reason_codes": _executor_reason_codes(row, action_class),
        "duplicate_suppression_status": sanitize_value(row.get("duplicate_suppression_status")),
        "duplicate_suppression_reason": sanitize_value(row.get("duplicate_suppression_reason")),
        "new_activation_issue_recommended": False
        if action_class == "keep-current-rule" or duplicate_suppression_status == "suppressed"
        else None,
        "sample_count": _as_int(row.get("sample_count")),
        "applied_count": _as_int(row.get("applied_count")),
        "holdout_count": _as_int(row.get("holdout_count")),
        "skipped_count": _as_int(row.get("skipped_count")),
        "fallback_count": _as_int(row.get("fallback_count")),
        "retry_count": _as_int(row.get("retry_count")),
        "rollback_count": _as_int(row.get("rollback_count")),
        "safety_stop_count": _as_int(row.get("safety_stop_count")),
        "error_rate_delta": round(_as_float(row.get("error_rate_delta")), 8),
        "retry_rate_delta": round(_as_float(row.get("retry_rate_delta")), 8),
        "fallback_rate_delta": round(_as_float(row.get("fallback_rate_delta")), 8),
        "observed_saved_tokens": _as_int(row.get("observed_saved_tokens")),
        "projected_saved_tokens": _as_int(row.get("projected_saved_tokens")),
        "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
        "realized_savings_usd": round(_as_float(row.get("realized_savings_usd")), 8),
        "observed_savings_usd": round(_as_float(row.get("observed_savings_usd") or row.get("realized_savings_usd")), 8),
        "full_rollout_outcome": sanitize_value(row.get("full_rollout_outcome") or outcome.get("outcome")),
        "full_rollout_outcome_next_action": sanitize_value(
            row.get("full_rollout_outcome_next_action") or outcome.get("next_action")
        ),
        "full_rollout_successor_decision": sanitize_value(
            row.get("full_rollout_successor_decision") or outcome.get("successor_decision")
        ),
        "full_rollout_successor_next_action": sanitize_value(
            row.get("full_rollout_successor_next_action") or outcome.get("successor_next_action")
        ),
        "full_rollout_successor_no_op_reason": sanitize_value(
            row.get("full_rollout_successor_no_op_reason") or outcome.get("successor_no_op_reason")
        ),
        "full_rollout_activation_outcome": sanitize_value(outcome) if outcome else None,
        "keep_active_regression_gate": sanitize_value(gate) if gate else None,
        "measured_full_rollout_activation": bool(row.get("measured_full_rollout_activation"))
        if row.get("measured_full_rollout_activation") is not None
        else None,
        "durable_outcome_ledger_entry": bool(row.get("durable_outcome_ledger_entry"))
        if row.get("durable_outcome_ledger_entry") is not None
        else None,
        "acceptance_metric": sanitize_value(row.get("acceptance_metric")),
        "expected_savings_path": sanitize_value(row.get("expected_savings_path")),
        "review_only": review_only,
        "dry_run": True,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }
    gate = _managed_preview_gate(row)
    managed_preview_required = _managed_preview_required(row)
    entry["managed_preview_required"] = managed_preview_required
    entry["preview_verified"] = bool(row.get("preview_verified") or gate.get("verified"))
    entry["preview_verification_status"] = sanitize_value(_preview_status(row, gate))
    entry["preview_verification_decision"] = sanitize_value(_preview_decision_value(row, gate))
    if gate:
        entry["managed_preview_gate"] = sanitize_value(gate)
    for key in (
        "evidence_schema",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
    ):
        if row.get(key) is not None:
            entry[key] = sanitize_value(row.get(key))
    preserved = {
        "rank",
        "source_rank",
        "source_queue_rank",
        "source_ledger_rank",
        "sample_count",
        "projected_savings_usd",
        "realized_savings_usd",
        "observed_savings_usd",
        "applied_count",
        "holdout_count",
        "skipped_count",
        "fallback_count",
        "retry_count",
        "rollback_count",
        "safety_stop_count",
        "error_rate_delta",
        "retry_rate_delta",
        "fallback_rate_delta",
        "observed_saved_tokens",
        "projected_saved_tokens",
        "new_activation_issue_recommended",
        "selected",
        "review_only",
        "dry_run",
        "policy_files_written",
        "provider_calls_made",
        "managed_server_calls_made",
        "managed_preview_required",
        "preview_verified",
    }
    return {key: value for key, value in entry.items() if value not in (None, "", [], 0) or key in preserved}


def _full_rollout_collapse_key(row: dict[str, Any]) -> tuple[str, str, str, int, int] | None:
    if not _is_full_rollout_crunch(row):
        return None
    return (
        str(row.get("target_local_policy_section") or "crunch.rules"),
        str(row.get("target_local_rule_file") or "crunch_rules.yaml"),
        str(row.get("duplicate_suppression_reason") or "repeated-context-crunch-full-rollout-active"),
        _as_int(row.get("applied_count")),
        _as_int(row.get("holdout_count")),
    )


def _full_rollout_row_score(row: dict[str, Any]) -> tuple[int, int, int, float, int]:
    return (
        1 if _full_rollout_outcome(row) else 0,
        1 if _full_rollout_gate(row) else 0,
        1 if row.get("durable_outcome_ledger_entry") else 0,
        _as_float(row.get("realized_savings_usd")),
        _as_int(row.get("sample_count")),
    )


def _collapse_full_rollout_crunch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, int, int], int] = {}
    for row in rows:
        key = _full_rollout_collapse_key(row)
        if key is None:
            collapsed.append(row)
            continue
        existing_index = by_key.get(key)
        if existing_index is None:
            by_key[key] = len(collapsed)
            collapsed.append(row)
            continue
        if _full_rollout_row_score(row) > _full_rollout_row_score(collapsed[existing_index]):
            collapsed[existing_index] = row
    return collapsed


def build_local_activation_executor_plan(
    source: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a dry-run executor/review plan from local activation successor evidence."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    burndown = _input_to_burndown(source, now=now)
    rows = _collapse_full_rollout_crunch_rows([row for row in burndown.get("rows") or [] if isinstance(row, dict)])
    entries = [_executor_entry(row) for row in rows]
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    selectable = [entry for entry in entries if entry.get("executor_action_class") in SAFE_SELECTABLE_CLASSES]
    selected = selectable[0] if selectable else None
    if selected is not None:
        selected["selected"] = True
        selected["executor_status"] = "selected"

    class_counts = Counter(str(entry.get("executor_action_class") or "unknown") for entry in entries)
    family_counts = Counter(str(entry.get("local_action_family") or "unknown") for entry in entries)
    result = {
        "schema": SCHEMA,
        "generated_at": now.isoformat(),
        "status": "ranked" if entries else "empty",
        "source_schema": sanitize_value(burndown.get("schema")),
        "source_generated_at": sanitize_value(burndown.get("generated_at")),
        "summary": {
            "executor_entry_count": len(entries),
            "selectable_action_count": len(selectable),
            "selected_action_count": 1 if selected else 0,
            "selected_executor_action_class": selected.get("executor_action_class") if selected else None,
            "selected_next_action": selected.get("executor_next_action") if selected else None,
            "selected_local_action_family": selected.get("local_action_family") if selected else None,
            "selected_target_local_rule_file": selected.get("target_local_rule_file") if selected else None,
            "selected_fingerprint": selected.get("fingerprint") if selected else None,
            "blocked_or_terminal_count": len(entries) - len(selectable),
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "executor_action_class_counts": [
                {"value": key, "count": count} for key, count in sorted(class_counts.items())
            ],
            "local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(family_counts.items())
            ],
        },
        "entries": entries,
        "selected_action": selected,
        "source_activation_burndown_summary": sanitize_value(burndown.get("summary") or {}),
        "privacy": _privacy(),
    }
    return sanitize_value(result)


def _handoff_ref(entry: dict[str, Any], *, prefix: str = "handoff") -> str:
    material = {
        "executor": entry.get("fingerprint"),
        "source": entry.get("source_fingerprint"),
        "family": entry.get("local_action_family"),
        "action": entry.get("executor_action_class"),
        "next_action": entry.get("executor_next_action"),
        "target": entry.get("target_local_rule_file"),
    }
    return public_id(json.dumps(material, sort_keys=True), prefix=prefix) or f"{prefix}:unknown"


def _public_ref(value: Any, *, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return public_id(text, prefix=prefix, fallback=None)


def _handoff_row(entry: dict[str, Any]) -> dict[str, Any]:
    coverage = {
        "metadata_only": True,
        "aggregate_only": True,
        "sample_count": _as_int(entry.get("sample_count")),
        "applied_count": _as_int(entry.get("applied_count")),
        "holdout_count": _as_int(entry.get("holdout_count")),
        "skipped_count": _as_int(entry.get("skipped_count")),
        "fallback_count": _as_int(entry.get("fallback_count")),
        "retry_count": _as_int(entry.get("retry_count")),
        "rollback_count": _as_int(entry.get("rollback_count")),
        "safety_stop_count": _as_int(entry.get("safety_stop_count")),
        "error_rate_delta": round(_as_float(entry.get("error_rate_delta")), 8),
        "retry_rate_delta": round(_as_float(entry.get("retry_rate_delta")), 8),
        "fallback_rate_delta": round(_as_float(entry.get("fallback_rate_delta")), 8),
    }
    savings = {
        "metadata_only": True,
        "aggregate_only": True,
        "projected_saved_tokens": _as_int(entry.get("projected_saved_tokens")),
        "observed_saved_tokens": _as_int(entry.get("observed_saved_tokens")),
        "projected_savings_usd": round(_as_float(entry.get("projected_savings_usd")), 8),
        "realized_savings_usd": round(_as_float(entry.get("realized_savings_usd")), 8),
        "observed_savings_usd": round(_as_float(entry.get("observed_savings_usd")), 8),
    }
    row = {
        "schema": HANDOFF_ROW_SCHEMA,
        "rank": _as_int(entry.get("rank")),
        "handoff_ref": _handoff_ref(entry),
        "source_executor_ref": _public_ref(entry.get("fingerprint"), prefix="executor-ref"),
        "source_activation_ref": _public_ref(entry.get("source_fingerprint"), prefix="activation-ref"),
        "source_successor_ref": _public_ref(entry.get("source_successor_fingerprint"), prefix="successor-ref"),
        "managed_dependency": "optional",
        "locally_executed": True,
        "feature_only": True,
        "server_content_processing": False,
        "provider_forwarding": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_preview_required": bool(entry.get("managed_preview_required")),
        "preview_verified": bool(entry.get("preview_verified")),
        "preview_verification_status": sanitize_value(entry.get("preview_verification_status")),
        "preview_verification_decision": sanitize_value(entry.get("preview_verification_decision")),
        "local_action_family": sanitize_value(entry.get("local_action_family") or "unknown"),
        "lever": sanitize_value(entry.get("lever") or entry.get("local_action_family") or "unknown"),
        "executor_action_class": sanitize_value(entry.get("executor_action_class") or "review-only"),
        "executor_status": sanitize_value(entry.get("executor_status") or "unknown"),
        "executor_next_action": sanitize_value(entry.get("executor_next_action") or "inspect-local-evidence"),
        "successor_status": sanitize_value(entry.get("successor_status") or "review"),
        "current_status": sanitize_value(entry.get("current_status") or "unknown"),
        "current_state": sanitize_value(entry.get("current_state") or "unknown"),
        "activation_outcome": sanitize_value(entry.get("full_rollout_outcome")),
        "activation_outcome_next_action": sanitize_value(entry.get("full_rollout_outcome_next_action")),
        "successor_decision": sanitize_value(entry.get("full_rollout_successor_decision")),
        "successor_next_action": sanitize_value(entry.get("full_rollout_successor_next_action")),
        "successor_no_op_reason": sanitize_value(entry.get("full_rollout_successor_no_op_reason")),
        "target_local_policy_section": sanitize_value(entry.get("target_local_policy_section")),
        "target_local_rule_file": sanitize_value(entry.get("target_local_rule_file")),
        "evidence_schema": sanitize_value(entry.get("evidence_schema")),
        "source_surface": sanitize_value(entry.get("source_surface")),
        "endpoint": sanitize_value(entry.get("endpoint")),
        "category": sanitize_value(entry.get("category")),
        "workflow_phase": sanitize_value(entry.get("workflow_phase")),
        "requested_model": sanitize_value(entry.get("requested_model")),
        "candidate_target_model": sanitize_value(entry.get("candidate_target_model")),
        "required_local_executor": sanitize_value(entry.get("required_local_executor")),
        "blocker_codes": sanitize_value(entry.get("blocker_codes") or []),
        "reason_codes": sanitize_value(entry.get("reason_codes") or []),
        "duplicate_suppression_status": sanitize_value(entry.get("duplicate_suppression_status")),
        "duplicate_suppression_reason": sanitize_value(entry.get("duplicate_suppression_reason")),
        "new_activation_issue_recommended": entry.get("new_activation_issue_recommended"),
        "measured_full_rollout_activation": entry.get("measured_full_rollout_activation"),
        "durable_outcome_ledger_entry": entry.get("durable_outcome_ledger_entry"),
        "coverage": coverage,
        "savings": savings,
        "privacy": _handoff_privacy(),
    }
    if isinstance(entry.get("managed_preview_gate"), dict):
        row["managed_preview_gate"] = sanitize_value(entry.get("managed_preview_gate"))
    preserved = {
        "rank",
        "policy_files_written",
        "provider_calls_made",
        "managed_server_calls_made",
        "managed_preview_required",
        "preview_verified",
        "provider_forwarding",
        "server_content_processing",
        "managed_enforced",
        "locally_executed",
        "feature_only",
        "coverage",
        "savings",
        "privacy",
    }
    return {key: value for key, value in row.items() if value not in (None, "", [], 0) or key in preserved}


def build_local_activation_executor_managed_handoff(
    source: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Export local executor outcomes as feature-only rows for optional managed previews."""
    plan = source if source.get("schema") == SCHEMA else build_local_activation_executor_plan(source, now=now)
    entries = [entry for entry in plan.get("entries") or [] if isinstance(entry, dict)]
    rows = [_handoff_row(entry) for entry in entries]
    class_counts = Counter(str(row.get("executor_action_class") or "unknown") for row in rows)
    family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in rows)
    status_counts = Counter(str(row.get("successor_status") or "unknown") for row in rows)
    preview_required_rows = [row for row in rows if bool(row.get("managed_preview_required"))]
    preview_required_family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in preview_required_rows)
    privacy = _handoff_privacy()
    result = {
        "schema": HANDOFF_SCHEMA,
        "generated_at": plan.get("generated_at") or (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "status": "exported" if rows else "empty",
        "source_schema": sanitize_value(plan.get("schema")),
        "source_generated_at": sanitize_value(plan.get("generated_at")),
        "managed_dependency": "optional",
        "server_ingestion_required": False,
        "locally_executed": True,
        "feature_only": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "supported_local_action_families": SUPPORTED_PREVIEW_LOCAL_ACTION_FAMILIES,
        "rows": rows,
        "summary": {
            "handoff_row_count": len(rows),
            "preview_required_row_count": len(preview_required_rows),
            "local_action_family_count": len(family_counts),
            "executor_action_class_counts": [
                {"value": key, "count": count} for key, count in sorted(class_counts.items())
            ],
            "local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(family_counts.items())
            ],
            "preview_required_local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(preview_required_family_counts.items())
            ],
            "successor_status_counts": [
                {"value": key, "count": count} for key, count in sorted(status_counts.items())
            ],
            "projected_savings_usd": round(
                sum(_as_float(row.get("savings", {}).get("projected_savings_usd")) for row in rows),
                8,
            ),
            "realized_savings_usd": round(
                sum(_as_float(row.get("savings", {}).get("realized_savings_usd")) for row in rows),
                8,
            ),
            "observed_savings_usd": round(
                sum(_as_float(row.get("savings", {}).get("observed_savings_usd")) for row in rows),
                8,
            ),
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "provider_forwarding": False,
            "server_content_processing": False,
            "managed_enforced": False,
        },
        "privacy": privacy,
    }
    result = sanitize_value(result)
    violations = managed_egress_violations(result)
    result["egress_guard"] = {
        "schema": "agentflow.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        result["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return result


def build_managed_activation_preview_request(
    source: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the feature-only batch request for an opt-in managed activation preview."""
    handoff = (
        source
        if source.get("schema") == HANDOFF_SCHEMA
        else build_local_activation_executor_managed_handoff(source, now=now)
    )
    rows = [row for row in handoff.get("rows") or [] if isinstance(row, dict)]
    family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in rows)
    preview_required_rows = [row for row in rows if bool(row.get("managed_preview_required"))]
    preview_required_family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in preview_required_rows)
    result = {
        "schema": PREVIEW_REQUEST_SCHEMA,
        "generated_at": handoff.get("generated_at")
        or (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "mode": "review-only",
        "dry_run": True,
        "managed_dependency": "optional",
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "source_handoff_schema": sanitize_value(handoff.get("schema")),
        "source_handoff_status": sanitize_value(handoff.get("status")),
        "source_handoff_summary": sanitize_value(handoff.get("summary") or {}),
        "supported_local_action_families": sanitize_value(
            handoff.get("supported_local_action_families") or SUPPORTED_PREVIEW_LOCAL_ACTION_FAMILIES
        ),
        "rows": rows,
        "summary": {
            "handoff_row_count": len(rows),
            "preview_required_row_count": len(preview_required_rows),
            "local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(family_counts.items())
            ],
            "preview_required_local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(preview_required_family_counts.items())
            ],
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "privacy": _handoff_privacy(),
    }
    result = sanitize_value(result)
    violations = managed_egress_violations(result)
    result["egress_guard"] = {
        "schema": "agentflow.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        result["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return result


def _preview_decision_rows(response_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(response_payload, dict):
        return []
    for key in ("decisions", "preview_decisions", "results", "rows"):
        value = response_payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    preview = response_payload.get("preview")
    if isinstance(preview, dict):
        return _preview_decision_rows(preview)
    return []


def _decision_bool(row: dict[str, Any], *keys: str) -> bool:
    return any(bool(row.get(key)) for key in keys)


def _preview_decision(row: dict[str, Any]) -> dict[str, Any]:
    decision = str(
        row.get("preview_decision")
        or row.get("policy_decision")
        or row.get("decision")
        or row.get("status")
        or "review-only"
    ).strip()
    omitted_reason = str(row.get("omitted_reason") or "").strip()
    reason_codes = row.get("reason_codes") if isinstance(row.get("reason_codes"), list) else []
    result = {
        "schema": PREVIEW_DECISION_SCHEMA,
        "handoff_ref": sanitize_value(row.get("handoff_ref") or row.get("source_handoff_ref")),
        "preview_ref": _public_ref(
            row.get("preview_ref") or row.get("fingerprint") or row.get("decision_id") or row.get("handoff_ref"),
            prefix="preview",
        ),
        "local_action_family": sanitize_value(row.get("local_action_family") or row.get("family")),
        "executor_action_class": sanitize_value(row.get("executor_action_class")),
        "classification": sanitize_value(row.get("classification")),
        "decision": sanitize_value(decision),
        "status": sanitize_value(row.get("status") or decision),
        "recommended_next_action": sanitize_value(row.get("recommended_next_action") or row.get("next_action")),
        "local_next_action": sanitize_value(row.get("local_next_action")),
        "agreement_status": sanitize_value(row.get("agreement_status")),
        "agrees_with_local_next_action": bool(row.get("agrees_with_local_next_action")),
        "omitted_reason": sanitize_value(omitted_reason) if omitted_reason else None,
        "no_op_reason": sanitize_value(row.get("no_op_reason")),
        "reason_codes": sanitize_value([str(item) for item in reason_codes if str(item or "").strip()]),
        "review_only": True if row.get("review_only") is None else bool(row.get("review_only")),
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": bool(row.get("managed_enforced")),
        "policy_files_written": bool(row.get("policy_files_written")),
        "provider_calls_made": bool(row.get("provider_calls_made")),
    }
    for key in (
        "cohort_class",
        "rollup_outcome_status",
        "crunch_preview_decision",
        "crunch_preview_confidence",
        "quality_risk_reason_codes",
        "projected_saved_tokens",
        "projected_saved_usd",
        "projected_savings_usd",
        "observed_saved_tokens",
        "observed_saved_usd",
        "observed_crunch_ratio",
        "sample_count",
        "applied_count",
        "holdout_count",
        "rollback_count",
        "safety_stop_count",
        "successor_action_fingerprint",
        "successor_decision_fingerprint",
        "target_local_policy_section",
        "target_local_rule_file",
        "source_fingerprint",
        "source_successor_fingerprint",
        "source_queue_rank",
        "source_ledger_rank",
    ):
        if row.get(key) is not None:
            result[key] = sanitize_value(row.get(key))
    preserved = {
        "review_only",
        "feature_only",
        "locally_executed",
        "provider_forwarding",
        "server_content_processing",
        "managed_enforced",
        "policy_files_written",
        "provider_calls_made",
    }
    return {key: value for key, value in result.items() if value not in (None, "", []) or key in preserved}


def build_managed_activation_preview_result(
    request_payload: dict[str, Any],
    *,
    response_payload: Any | None = None,
    fetch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize an optional managed preview response without retaining raw server payloads."""
    rows = [row for row in request_payload.get("rows") or [] if isinstance(row, dict)]
    decisions = [_preview_decision(row) for row in _preview_decision_rows(response_payload)]
    handoff_refs = {str(row.get("handoff_ref") or "") for row in rows if row.get("handoff_ref")}
    decision_refs = {str(row.get("handoff_ref") or "") for row in decisions if row.get("handoff_ref")}
    matched_refs = sorted(handoff_refs & decision_refs)
    no_op_count = sum(
        1
        for row in decisions
        if str(row.get("decision") or "").lower() in {"no-op", "noop", "omitted"}
        or bool(row.get("no_op_reason"))
        or bool(row.get("omitted_reason"))
    )
    omitted_count = sum(1 for row in decisions if bool(row.get("omitted_reason")))
    review_only_count = sum(1 for row in decisions if bool(row.get("review_only")))
    active_policy_write_count = sum(
        1
        for row in decisions
        if _decision_bool(row, "policy_files_written", "managed_enforced", "provider_calls_made")
    )
    managed_calls_made = bool((fetch or {}).get("managed_server_calls_made"))
    fetch_status = str((fetch or {}).get("status") or "").strip()
    status = "previewed" if managed_calls_made and fetch_status == "ok" else fetch_status or "skipped"
    result = {
        "schema": PREVIEW_RESULT_SCHEMA,
        "status": status,
        "mode": "review-only",
        "preview_request": request_payload,
        "fetch": sanitize_value(fetch or {
            "status": "skipped",
            "reason": "managed-preview-url-not-configured",
            "managed_server_calls_made": False,
        }),
        "preview": {
            "schema": "agentflow.managed_activation_preview_decisions.v1",
            "decision_count": len(decisions),
            "decisions": decisions,
        },
        "coverage": {
            "schema": "agentflow.managed_activation_preview_coverage.v1",
            "handoff_row_count": len(rows),
            "preview_decision_count": len(decisions),
            "matched_handoff_ref_count": len(matched_refs),
            "missing_preview_decision_count": max(0, len(rows) - len(matched_refs)),
            "omitted_count": omitted_count,
            "no_op_count": no_op_count,
            "review_only_count": review_only_count,
            "active_policy_write_count": active_policy_write_count,
            "policy_files_written": active_policy_write_count > 0,
            "provider_calls_made": any(bool(row.get("provider_calls_made")) for row in decisions),
            "managed_server_calls_made": managed_calls_made,
        },
        "summary": {
            "submitted_row_count": len(rows),
            "handoff_row_count": len(rows),
            "preview_row_count": len(decisions),
            "preview_decision_count": len(decisions),
            "matched_handoff_ref_count": len(matched_refs),
            "missing_preview_decision_count": max(0, len(rows) - len(matched_refs)),
            "omission_count": omitted_count,
            "omitted_count": omitted_count,
            "no_op_count": no_op_count,
            "review_only_count": review_only_count,
            "active_policy_write_count": active_policy_write_count,
            "managed_server_calls_made": managed_calls_made,
            "provider_calls_made": any(bool(row.get("provider_calls_made")) for row in decisions),
            "policy_files_written": active_policy_write_count > 0,
        },
        "privacy": {
            **_handoff_privacy(),
            "managed_server_calls_made": managed_calls_made,
            "policy_files_written": active_policy_write_count > 0,
        },
    }
    result = sanitize_value(result)
    violations = managed_egress_violations(result)
    result["egress_guard"] = {
        "schema": "agentflow.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        result["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return result
