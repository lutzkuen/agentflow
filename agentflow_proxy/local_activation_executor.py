from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any

from agentflow_proxy.orchestrator_research import (
    build_activation_burndown_report,
    sanitize_value,
)
from agentflow_proxy.public_metadata import public_id


SCHEMA = "agentflow.local_activation_executor_plan.v1"
ENTRY_SCHEMA = "agentflow.local_activation_executor_plan_entry.v1"
PRIVACY_SCHEMA = "agentflow.local_activation_executor_privacy.v1"

SAFE_SELECTABLE_CLASSES = {"draft-local-policy", "review-only", "retire"}


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
