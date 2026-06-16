from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.local_activation_outcome_summary.v1"
OUTCOME_SCHEMA = "agentflow.local_activation_outcome_summary_row.v1"
PRIVACY_SCHEMA = "agentflow.local_activation_outcome_summary_privacy.v1"
CRUNCH_POLICY_DECISION_SCHEMA = "agentflow.request_shape_crunch_policy_decision.v1"
CRUNCH_ACTIVATION_EVIDENCE_SCHEMA = "agentflow.request_shape_crunch_activation_evidence.v1"
CACHE_REPLAY_POLICY_DECISION_SCHEMA = "agentflow.request_shape_cache_replay_policy_decision.v1"

RULE_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
}

RAW_REASON_HINTS = (
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "content",
    "file_path",
    "message",
    "payload",
    "prompt",
    "provider_body",
    "raw",
    "request_id",
    "response",
    "secret",
    "session_id",
    "tenant_id",
    "tool_payload",
)


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "feature_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "individual_candidate_ids_included": False,
        "absolute_paths_included": False,
        "policy_file_contents_included": False,
    }


def _safe_code(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    if len(text) > 96 or any(hint in text for hint in RAW_REASON_HINTS) or "/" in text or "\\" in text:
        return public_id(text, prefix="reason", fallback="redacted-reason")
    return public_label(text, "unknown")


def _reason_codes(*values: Any) -> list[str]:
    codes: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            for key in ("reason", "status", "no_op_reason", "skip_reason", "fallback_reason", "blocker_reason"):
                code = _safe_code(value.get(key))
                if code and code != "unknown":
                    codes.add(code)
            blockers = value.get("blocker_codes")
            if isinstance(blockers, list):
                for blocker in blockers:
                    code = _safe_code(blocker)
                    if code and code != "unknown":
                        codes.add(code)
        elif isinstance(value, list):
            for item in value:
                code = _safe_code(item)
                if code and code != "unknown":
                    codes.add(code)
        else:
            code = _safe_code(value)
            if code and code != "unknown":
                codes.add(code)
    return sorted(codes)


def _nested_dict(meta: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _cohort(meta: dict[str, Any], *nested_keys: str) -> str:
    nested = _nested_dict(meta, *nested_keys)
    for source in (nested, meta):
        for key in ("cohort", "canary_cohort", "lifecycle_event", "status", "reason"):
            value = str(source.get(key) or "").strip().lower().replace("-", "_")
            if value:
                return value
    return ""


def _is_holdout(meta: dict[str, Any], *nested_keys: str) -> bool:
    cohort = _cohort(meta, *nested_keys)
    return "holdout" in cohort or meta.get("holdout") is True


def _is_safety_stopped(meta: dict[str, Any], *nested_keys: str) -> bool:
    cohort = _cohort(meta, *nested_keys)
    return "safety_stopped" in cohort or "safety_stop" in cohort or meta.get("safety_stopped") is True


def _is_applied(meta: dict[str, Any], *nested_keys: str) -> bool:
    nested = _nested_dict(meta, *nested_keys)
    if meta.get("applied") is True or nested.get("applied") is True:
        return True
    cohort = _cohort(meta, *nested_keys)
    return "applied" in cohort or "hit" == cohort or "replayed" in cohort


def _rule_file_state(section: str, config_dir: str | Path | None) -> dict[str, Any]:
    rule_file = RULE_FILES[section]
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(Path(config_dir).expanduser() / rule_file)
    candidates.append(Path(__file__).resolve().parent / rule_file)
    exists = any(path.exists() for path in candidates)
    source = "local-file-backed" if exists else "missing-local-policy-file"
    return {
        "policy_section": section,
        "rule_file": rule_file,
        "exists": exists,
        "policy_source": source,
        "path_included": False,
        "policy_file_contents_included": False,
    }


def _empty_row(section: str, config_dir: str | Path | None) -> dict[str, Any]:
    return {
        "schema": OUTCOME_SCHEMA,
        "policy_section": section,
        "local_action_family": section,
        "local_file_backed_representation": _rule_file_state(section, config_dir),
        "row_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "safety_stopped_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "observed_savings_usd": 0.0,
        "projected_savings_usd": 0.0,
        "projected_saved_tokens": 0,
        "projected_saved_chars": 0,
        "blocker_codes": [],
        "next_action": "review-local-activation-outcome",
        "managed_dependency": "optional",
    }


def _row_savings(row: dict[str, Any]) -> float:
    if _as_int(row.get("status_code")) >= 400:
        return 0.0
    return max(0.0, _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")))


def _meta_savings(meta: dict[str, Any]) -> float:
    for key in (
        "observed_savings_usd",
        "projected_savings_usd",
        "projected_saved_usd",
        "estimated_saved_cost_usd",
        "savings_usd",
        "crunch_savings_usd",
    ):
        value = _as_float(meta.get(key))
        if value > 0:
            return value
    return 0.0


def _meta_tokens(meta: dict[str, Any]) -> int:
    for key in ("projected_saved_tokens", "tokens_saved_est", "saved_tokens", "crunch_tokens_saved"):
        value = _as_int(meta.get(key))
        if value > 0:
            return value
    return 0


def _meta_chars(meta: dict[str, Any]) -> int:
    for key in ("projected_saved_chars", "saved_chars", "chars_saved", "crunch_chars_saved"):
        value = _as_int(meta.get(key))
        if value > 0:
            return value
    before = _as_int(meta.get("before_chars") or meta.get("original_chars"))
    after = _as_int(meta.get("after_chars") or meta.get("result_chars"))
    return max(0, before - after)


def _record_common(row_summary: dict[str, Any], row: dict[str, Any], meta: dict[str, Any], *, applied: bool, holdout: bool, safety_stopped: bool, fallback: bool) -> None:
    row_summary["row_count"] += 1
    row_summary["applied_count"] += int(applied)
    row_summary["holdout_count"] += int(holdout)
    row_summary["skipped_count"] += int(not applied and not holdout)
    row_summary["safety_stopped_count"] += int(safety_stopped)
    row_summary["error_count"] += int(_as_int(row.get("status_code")) >= 400)
    row_summary["retry_count"] += _as_int(row.get("retry_count"))
    row_summary["fallback_count"] += int(fallback)
    if applied:
        row_summary["observed_savings_usd"] += _row_savings(row)
    row_summary["projected_savings_usd"] += _meta_savings(meta)
    row_summary["projected_saved_tokens"] += _meta_tokens(meta)
    row_summary["projected_saved_chars"] += _meta_chars(meta)


def _finalize_row(row: dict[str, Any], blockers: Counter[str]) -> dict[str, Any]:
    total = max(1, _as_int(row.get("row_count")))
    row["observed_savings_usd"] = round(_as_float(row.get("observed_savings_usd")), 8)
    row["projected_savings_usd"] = round(_as_float(row.get("projected_savings_usd")), 8)
    row["applied_rate"] = round(_as_int(row.get("applied_count")) / total, 6) if row.get("row_count") else 0.0
    row["holdout_rate"] = round(_as_int(row.get("holdout_count")) / total, 6) if row.get("row_count") else 0.0
    row["error_rate"] = round(_as_int(row.get("error_count")) / total, 6) if row.get("row_count") else 0.0
    row["retry_rate"] = round(_as_int(row.get("retry_count")) / total, 6) if row.get("row_count") else 0.0
    row["blocker_codes"] = [
        {"code": code, "count": count}
        for code, count in blockers.most_common(8)
    ]
    current_action = str(row.get("next_action") or "").strip()
    if current_action and current_action != "review-local-activation-outcome":
        return row
    if row["applied_count"]:
        row["next_action"] = "compare-applied-holdout-and-promote-or-keep-blocked"
    elif row["holdout_count"]:
        row["next_action"] = "collect-applied-activation-coverage"
    elif row["row_count"]:
        row["next_action"] = "stage-or-review-local-policy-activation"
    return row


def _activation_keep_active_duplicate_suppression(
    *,
    decision_id: str | None,
    active_rule_ref: str | None,
    target_local_rule_file: str,
    target_local_policy_section: str,
) -> dict[str, Any]:
    fingerprint_source = "|".join(
        part
        for part in (
            "crunch",
            decision_id or "",
            active_rule_ref or "",
            target_local_rule_file,
            target_local_policy_section,
        )
        if part
    )
    return {
        "schema": "agentflow.local_activation_keep_active_duplicate_suppression.v1",
        "suppresses_new_activation_issue": True,
        "suppresses_generic_crunch_activation_issue": True,
        "reason": "repeated-context-crunch-active-at-max-rollout",
        "activation_ref": public_id(fingerprint_source or "crunch-keep-active", prefix="activation", fallback="activation:unknown"),
        "matching_local_policy": "crunch_rules",
        "target_local_rule_file": target_local_rule_file,
        "target_local_policy_section": target_local_policy_section,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _safe_local_duplicate_suppression(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if not value:
        return {}
    result: dict[str, Any] = {
        "schema": public_label(value.get("schema"), "agentflow.local_activation_keep_active_duplicate_suppression.v1"),
        "suppresses_new_activation_issue": bool(value.get("suppresses_new_activation_issue")),
        "suppresses_generic_crunch_activation_issue": bool(value.get("suppresses_generic_crunch_activation_issue")),
        "reason": public_label(value.get("reason"), "unknown"),
        "matching_local_policy": public_label(value.get("matching_local_policy"), "unknown"),
        "target_local_rule_file": public_label(value.get("target_local_rule_file"), "crunch_rules.yaml"),
        "target_local_policy_section": public_label(value.get("target_local_policy_section"), "crunch.rules"),
        "metadata_only": True,
        "aggregate_only": True,
    }
    activation_ref = public_label(value.get("activation_ref"), "") or public_label(value.get("fingerprint"), "")
    if activation_ref:
        result["activation_ref"] = activation_ref
    return result


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _report_decision_entry(report: dict[str, Any]) -> dict[str, Any]:
    top = report.get("top_decision")
    if isinstance(top, dict):
        return top
    decisions = report.get("decisions")
    if isinstance(decisions, list):
        for item in decisions:
            if isinstance(item, dict):
                return item
    return {}


def _decision_count(report: dict[str, Any]) -> int:
    decisions = report.get("decisions")
    if isinstance(decisions, list):
        return len([item for item in decisions if isinstance(item, dict)])
    return 1 if _report_decision_entry(report) else 0


def _apply_crunch_policy_decision_report(row: dict[str, Any], blockers: Counter[str], report: dict[str, Any]) -> bool:
    if report.get("schema") != CRUNCH_POLICY_DECISION_SCHEMA:
        return False
    top = _report_decision_entry(report)
    summary = _first_dict(report.get("summary"))
    metrics = _first_dict(top.get("metrics"))
    coverage = _first_dict(top.get("coverage"), summary.get("coverage"), metrics.get("coverage"))
    applied = _as_int(metrics.get("applied_count") or coverage.get("applied_count") or summary.get("applied_count"))
    holdout = _as_int(metrics.get("holdout_count") or coverage.get("holdout_count") or summary.get("holdout_count"))
    skipped = _as_int(coverage.get("skipped_count") or summary.get("skipped_count"))
    safety_stopped = _as_int(metrics.get("safety_stop_count") or coverage.get("safety_stop_count") or summary.get("safety_stop_count"))
    error_count = _as_int(metrics.get("applied_error_count")) + _as_int(metrics.get("holdout_error_count"))
    retry_count = _as_int(metrics.get("applied_retry_count")) + _as_int(metrics.get("holdout_retry_count"))
    fallback_count = _as_int(metrics.get("fallback_count") or coverage.get("fallback_count") or summary.get("fallback_count"))
    observed_tokens = _as_int(top.get("observed_saved_tokens") or metrics.get("observed_saved_tokens") or summary.get("observed_saved_tokens"))
    observed_savings = _as_float(top.get("observed_saved_usd") or metrics.get("observed_saved_usd") or summary.get("observed_saved_usd"))
    row_count = _as_int(coverage.get("observed_count") or coverage.get("matched_count") or (applied + holdout + skipped))
    if row_count <= 0:
        row_count = applied + holdout + skipped

    row["row_count"] = max(_as_int(row.get("row_count")), row_count)
    row["applied_count"] = max(_as_int(row.get("applied_count")), applied)
    row["holdout_count"] = max(_as_int(row.get("holdout_count")), holdout)
    row["skipped_count"] = max(_as_int(row.get("skipped_count")), skipped)
    row["safety_stopped_count"] = max(_as_int(row.get("safety_stopped_count")), safety_stopped)
    row["error_count"] = max(_as_int(row.get("error_count")), error_count)
    row["retry_count"] = max(_as_int(row.get("retry_count")), retry_count)
    row["fallback_count"] = max(_as_int(row.get("fallback_count")), fallback_count)
    row["observed_savings_usd"] = max(_as_float(row.get("observed_savings_usd")), observed_savings)
    row["observed_saved_tokens"] = max(_as_int(row.get("observed_saved_tokens")), observed_tokens)
    row["projected_savings_usd"] = max(_as_float(row.get("projected_savings_usd")), _as_float(summary.get("projected_savings_usd")))
    row["source_evidence_schema"] = CRUNCH_POLICY_DECISION_SCHEMA
    row["source_decision_id"] = str(top.get("decision_id") or report.get("decision_id") or "") or None
    row["source_decision"] = public_label(top.get("decision") or report.get("decision"), "unknown")
    row["graduation_decision"] = public_label(top.get("graduation_decision") or report.get("graduation_decision"), "unknown")
    row["safety_stop_state"] = public_label(top.get("safety_stop_state") or summary.get("safety_stop_state"), "none")
    row["coverage"] = {
        "schema": "agentflow.local_activation_outcome_decision_coverage.v1",
        "source_schema": public_label(coverage.get("schema"), "unknown"),
        "metadata_only": True,
        "aggregate_only": True,
        "observed_count": row_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": skipped,
        "safety_stop_count": safety_stopped,
        "error_count": error_count,
        "retry_count": retry_count,
        "fallback_count": fallback_count,
    }
    row["decision_count"] = _decision_count(report)
    row["target_local_rule_file"] = "crunch_rules.yaml"
    row["target_local_policy_section"] = "crunch.rules"
    row["next_action"] = public_label(top.get("source_recommended_next_action") or summary.get("source_impact_recommendation") or top.get("decision"), row.get("next_action") or "review")

    reason_codes = _reason_codes(top.get("reason_codes"), top, summary)
    if row["source_decision"] not in {"widen", "promote", "apply", "recommended", "unknown"}:
        reason_codes.append(f"decision-{row['source_decision']}")
    for code in reason_codes:
        blockers[code] += 1
    row["source_report"] = {
        "schema": CRUNCH_POLICY_DECISION_SCHEMA,
        "status": public_label(report.get("status"), "unknown"),
        "decision": row["source_decision"],
        "decision_count": _decision_count(report),
        "metadata_only": True,
        "aggregate_only": True,
    }
    return True


def _apply_crunch_activation_evidence_report(row: dict[str, Any], blockers: Counter[str], report: dict[str, Any]) -> bool:
    if report.get("schema") != CRUNCH_ACTIVATION_EVIDENCE_SCHEMA:
        return False
    summary = _first_dict(report.get("summary"))
    rules = report.get("rules") if isinstance(report.get("rules"), list) else []
    top_rule = next((item for item in rules if isinstance(item, dict)), {})
    decision_id = str(report.get("decision_id") or summary.get("decision_id") or "") or None
    applied = _as_int(summary.get("applied_count"))
    holdout = _as_int(summary.get("holdout_count"))
    skipped = _as_int(summary.get("skipped_count"))
    safety_stop_count = _as_int(summary.get("safety_stop_count"))
    rollback_count = _as_int(summary.get("rollback_count"))
    fallback_count = _as_int(summary.get("fallback_count"))
    observed_tokens = _as_int(summary.get("observed_saved_tokens") or summary.get("projected_saved_tokens"))
    observed_savings = _as_float(summary.get("observed_saved_usd") or summary.get("projected_saved_usd"))
    row_count = applied + holdout + skipped
    if row_count <= 0:
        row_count = _as_int(summary.get("matched_count") or summary.get("rows_considered"))

    target_rule_file = public_label(summary.get("target_local_rule_file") or "crunch_rules.yaml", "crunch_rules.yaml")
    target_policy_section = public_label(summary.get("target_local_policy_section") or "crunch.rules", "crunch.rules")
    post_status = public_label(summary.get("post_widening_status"), "unknown")
    post_next_action = public_label(
        summary.get("post_widening_next_action") or summary.get("next_action") or report.get("next_action"),
        "review-local-activation-outcome",
    )
    keep_active = post_next_action == "keep-active" and post_status == "post-widening-active-at-max-rollout"
    active_rule_ref = str(top_rule.get("rule_ref") or top_rule.get("rule_id") or "") or None

    row["row_count"] = max(_as_int(row.get("row_count")), row_count)
    row["applied_count"] = max(_as_int(row.get("applied_count")), applied)
    row["holdout_count"] = max(_as_int(row.get("holdout_count")), holdout)
    row["skipped_count"] = max(_as_int(row.get("skipped_count")), skipped)
    row["safety_stopped_count"] = max(_as_int(row.get("safety_stopped_count")), safety_stop_count)
    row["rollback_count"] = max(_as_int(row.get("rollback_count")), rollback_count)
    row["fallback_count"] = max(_as_int(row.get("fallback_count")), fallback_count)
    row["observed_savings_usd"] = max(_as_float(row.get("observed_savings_usd")), observed_savings)
    row["projected_savings_usd"] = max(_as_float(row.get("projected_savings_usd")), observed_savings)
    row["observed_saved_tokens"] = max(_as_int(row.get("observed_saved_tokens")), observed_tokens)
    row["projected_saved_tokens"] = max(_as_int(row.get("projected_saved_tokens")), observed_tokens)
    row["source_evidence_schema"] = CRUNCH_ACTIVATION_EVIDENCE_SCHEMA
    row["source_decision_id"] = decision_id
    row["source_decision"] = public_label(report.get("decision") or summary.get("decision"), "unknown")
    row["graduation_decision"] = public_label(report.get("graduation_decision") or summary.get("graduation_decision"), "unknown")
    row["safety_stop_state"] = public_label(summary.get("safety_stop_state"), "none")
    row["target_local_rule_file"] = target_rule_file
    row["target_local_policy_section"] = target_policy_section
    row["active_rule_count"] = _as_int(summary.get("active_rule_count"))
    row["widened_rule_count"] = _as_int(summary.get("widened_rule_count"))
    row["active_rule_ref"] = public_label(active_rule_ref, "unknown") if active_rule_ref else None
    row["active_rule_source"] = public_label(top_rule.get("policy_source") or summary.get("policy_source"), "unknown")
    row["active_rule_decision_id"] = public_label(top_rule.get("decision_id") or decision_id, "unknown")
    source_reports = report.get("source_reports") if isinstance(report.get("source_reports"), dict) else {}
    row["active_rule_source_evidence_schema"] = public_label(
        top_rule.get("source_evidence_schema") or source_reports.get("active_rule_source_evidence_schema"),
        "unknown",
    )
    row["post_widening_status"] = post_status
    row["post_widening_next_action"] = post_next_action
    row["post_widening_reason_codes"] = [
        public_label(item, "unknown")
        for item in (summary.get("post_widening_reason_codes") or [])
        if public_label(item, "unknown") != "unknown"
    ]
    row["outcome"] = "keep-active" if keep_active else post_next_action
    row["next_action"] = post_next_action
    row["error_rate_delta"] = round(_as_float(summary.get("error_rate_delta")), 6)
    row["retry_rate_delta"] = round(_as_float(summary.get("retry_rate_delta")), 6)
    row["fallback_rate_delta"] = round(_as_float(summary.get("fallback_rate_delta")), 6)
    row["coverage"] = {
        "schema": "agentflow.local_activation_outcome_decision_coverage.v1",
        "source_schema": CRUNCH_ACTIVATION_EVIDENCE_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "observed_count": row_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": skipped,
        "safety_stop_count": safety_stop_count,
        "rollback_count": rollback_count,
        "fallback_count": fallback_count,
        "error_rate_delta": row["error_rate_delta"],
        "retry_rate_delta": row["retry_rate_delta"],
        "fallback_rate_delta": row["fallback_rate_delta"],
        "canary_fraction": round(_as_float(summary.get("canary_fraction")), 6),
        "max_rollout_fraction": round(_as_float(summary.get("max_rollout_fraction")), 6),
    }
    duplicate_suppression = report.get("duplicate_suppression")
    if not isinstance(duplicate_suppression, dict):
        follow_up = report.get("activation_follow_up") if isinstance(report.get("activation_follow_up"), dict) else {}
        duplicate_suppression = follow_up.get("duplicate_suppression") if isinstance(follow_up.get("duplicate_suppression"), dict) else {}
    row["duplicate_suppression"] = _safe_local_duplicate_suppression(duplicate_suppression) or _activation_keep_active_duplicate_suppression(
        decision_id=decision_id,
        active_rule_ref=active_rule_ref,
        target_local_rule_file=target_rule_file,
        target_local_policy_section=target_policy_section,
    )
    row["source_report"] = {
        "schema": CRUNCH_ACTIVATION_EVIDENCE_SCHEMA,
        "status": public_label(report.get("status"), "unknown"),
        "decision": row["source_decision"],
        "post_widening_status": post_status,
        "metadata_only": True,
        "aggregate_only": True,
    }
    for code in _reason_codes(row["post_widening_reason_codes"], summary):
        blockers[code] += 1
    return True


def _apply_cache_policy_decision_report(row: dict[str, Any], blockers: Counter[str], report: dict[str, Any]) -> bool:
    if report.get("schema") != CACHE_REPLAY_POLICY_DECISION_SCHEMA:
        return False
    top = _report_decision_entry(report)
    summary = _first_dict(report.get("summary"))
    metrics = _first_dict(top.get("metrics"))
    coverage = _first_dict(top.get("coverage"))
    applied = _as_int(metrics.get("applied_count") or summary.get("applied_count"))
    holdout = _as_int(metrics.get("holdout_count") or summary.get("holdout_count"))
    row_count = _as_int(metrics.get("observed_row_count") or summary.get("observed_row_count") or (applied + holdout))
    retry_count = _as_int(metrics.get("retry_count") or summary.get("retry_count"))
    fallback_count = _as_int(metrics.get("fallback_count") or summary.get("fallback_count"))
    error_count = _as_int(metrics.get("error_count") or summary.get("error_count"))
    observed_hits = _as_int(metrics.get("observed_hits") or summary.get("observed_hits"))
    exact_hit_count = _as_int(metrics.get("exact_hit_count") or summary.get("exact_hit_count"))
    projected_hits = _as_int(metrics.get("projected_hits") or summary.get("projected_hits"))
    observed_savings = _as_float(metrics.get("observed_savings_usd") or summary.get("observed_savings_usd"))
    projected_savings = _as_float(metrics.get("projected_savings_usd") or summary.get("projected_savings_usd"))

    row["row_count"] = max(_as_int(row.get("row_count")), row_count)
    row["applied_count"] = max(_as_int(row.get("applied_count")), applied)
    row["holdout_count"] = max(_as_int(row.get("holdout_count")), holdout)
    row["skipped_count"] = max(_as_int(row.get("skipped_count")), max(0, row_count - applied - holdout))
    row["error_count"] = max(_as_int(row.get("error_count")), error_count)
    row["retry_count"] = max(_as_int(row.get("retry_count")), retry_count)
    row["fallback_count"] = max(_as_int(row.get("fallback_count")), fallback_count)
    row["observed_savings_usd"] = max(_as_float(row.get("observed_savings_usd")), observed_savings)
    row["projected_savings_usd"] = max(_as_float(row.get("projected_savings_usd")), projected_savings)
    row["source_evidence_schema"] = CACHE_REPLAY_POLICY_DECISION_SCHEMA
    row["source_decision_id"] = str(top.get("decision_id") or "") or None
    row["source_decision"] = public_label(top.get("decision") or report.get("decision"), "unknown")
    row["graduation_decision"] = row["source_decision"]
    row["coverage"] = {
        "schema": "agentflow.local_activation_outcome_decision_coverage.v1",
        "source_schema": public_label(coverage.get("schema"), "unknown"),
        "metadata_only": True,
        "aggregate_only": True,
        "observed_count": row_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "observed_hits": observed_hits,
        "exact_hit_count": exact_hit_count,
        "projected_hits": projected_hits,
        "retry_count": retry_count,
        "fallback_count": fallback_count,
        "error_count": error_count,
        "invalidation_skipped_count": _as_int(metrics.get("invalidation_skipped_count") or summary.get("invalidation_skipped_count")),
        "unsupported_shape_count": _as_int(metrics.get("unsupported_shape_count") or summary.get("unsupported_shape_count")),
    }
    row["decision_count"] = _decision_count(report)
    row["target_local_rule_file"] = "cache_rules.yaml"
    row["target_local_policy_section"] = "cache.pattern_rules"
    row["next_action"] = public_label(top.get("recommended_next_action") or top.get("next_action") or summary.get("decision"), row.get("next_action") or "review")
    row["observed_hits"] = observed_hits
    row["exact_hit_count"] = exact_hit_count
    row["projected_hits"] = projected_hits

    reason_codes = _reason_codes(report.get("reason_codes"), top.get("reason_codes"), top, summary)
    if row["source_decision"] not in {"widen", "promote", "apply", "recommended", "unknown"}:
        reason_codes.append(f"decision-{row['source_decision']}")
    for code in reason_codes:
        blockers[code] += 1
    row["source_report"] = {
        "schema": CACHE_REPLAY_POLICY_DECISION_SCHEMA,
        "status": public_label(report.get("status"), "unknown"),
        "decision": row["source_decision"],
        "decision_count": _decision_count(report),
        "metadata_only": True,
        "aggregate_only": True,
    }
    return True


def _apply_policy_decision_reports(
    summaries: dict[str, dict[str, Any]],
    blockers: dict[str, Counter[str]],
    activation_reports: list[dict[str, Any]],
) -> int:
    applied = 0
    for report in activation_reports:
        if not isinstance(report, dict):
            continue
        if _apply_crunch_activation_evidence_report(summaries["crunch"], blockers["crunch"], report):
            applied += 1
            continue
        if _apply_crunch_policy_decision_report(summaries["crunch"], blockers["crunch"], report):
            applied += 1
            continue
        if _apply_cache_policy_decision_report(summaries["cache"], blockers["cache"], report):
            applied += 1
    return applied


def _fetch_rows(store_obj: Any, *, limit: int) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 1), 20_000))
    return [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, provider, source_surface, endpoint, path,
                   requested_model, routed_model, status_code, retry_count,
                   cache_hit, cost_est_usd, cost_baseline_usd,
                   routing_json, crunch_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]


def build_local_activation_outcome_summary(
    store_obj: Any,
    *,
    limit: int = 1000,
    config_dir: str | Path | None = None,
    activation_reports: list[dict[str, Any]] | None = None,
    managed_activation_bundle_apply_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = _fetch_rows(store_obj, limit=limit)
    summaries = {section: _empty_row(section, config_dir) for section in ("routing", "crunch", "cache")}
    blockers = {section: Counter() for section in ("routing", "crunch", "cache")}

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))

        requested_model = str(row.get("requested_model") or "")
        routed_model = str(row.get("routed_model") or "")
        routing_applied = bool(requested_model and routed_model and requested_model != routed_model) or _is_applied(
            routing,
            "openai_routing_canary",
            "anthropic_routing_canary",
            "routing_canary",
            "managed_policy_decision",
        )
        routing_holdout = _is_holdout(routing, "openai_routing_canary", "anthropic_routing_canary", "routing_canary")
        routing_safety = _is_safety_stopped(routing, "openai_routing_canary", "anthropic_routing_canary", "routing_canary")
        routing_fallback = bool(routing.get("fallback_reason") or _nested_dict(routing, "openai_routing_canary", "anthropic_routing_canary", "routing_canary").get("fallback_reason"))
        _record_common(
            summaries["routing"],
            row,
            routing,
            applied=routing_applied,
            holdout=routing_holdout,
            safety_stopped=routing_safety,
            fallback=routing_fallback,
        )
        for code in _reason_codes(routing, _nested_dict(routing, "openai_routing_canary", "anthropic_routing_canary", "routing_canary")):
            blockers["routing"][code] += 1

        crunch_applied = _is_applied(crunch, "repeated_context_crunch_canary", "old_context_summarization") or bool(
            crunch.get("changed") or _meta_chars(crunch) > 0 or _meta_tokens(crunch) > 0
        )
        crunch_holdout = _is_holdout(crunch, "repeated_context_crunch_canary", "old_context_summarization")
        crunch_safety = _is_safety_stopped(crunch, "repeated_context_crunch_canary", "old_context_summarization")
        crunch_fallback = bool(crunch.get("fallback_reason"))
        _record_common(
            summaries["crunch"],
            row,
            crunch,
            applied=crunch_applied,
            holdout=crunch_holdout,
            safety_stopped=crunch_safety,
            fallback=crunch_fallback,
        )
        for code in _reason_codes(crunch, _nested_dict(crunch, "repeated_context_crunch_canary", "old_context_summarization")):
            blockers["crunch"][code] += 1

        cache_applied = bool(row.get("cache_hit")) or _is_applied(cache, "cache_replay_canary", "exact_cache_replay")
        cache_holdout = _is_holdout(cache, "cache_replay_canary", "exact_cache_replay")
        cache_safety = _is_safety_stopped(cache, "cache_replay_canary", "exact_cache_replay")
        cache_fallback = bool(cache.get("fallback_reason"))
        _record_common(
            summaries["cache"],
            row,
            cache,
            applied=cache_applied,
            holdout=cache_holdout,
            safety_stopped=cache_safety,
            fallback=cache_fallback,
        )
        for code in _reason_codes(cache, _nested_dict(cache, "cache_replay_canary", "exact_cache_replay")):
            blockers["cache"][code] += 1

    activation_reports = [item for item in (activation_reports or []) if isinstance(item, dict)]
    policy_decision_report_count = _apply_policy_decision_reports(summaries, blockers, activation_reports)

    outcome_summaries = [_finalize_row(summaries[section], blockers[section]) for section in ("routing", "crunch", "cache")]
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "tracked" if rows or policy_decision_report_count else "no-local-traffic",
        "read_only": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_dependency": "optional",
        "summary": {
            "rows_considered": len(rows),
            "lookback_limit": max(1, min(int(limit or 1), 20_000)),
            "local_action_family_count": len(outcome_summaries),
            "policy_decision_report_count": policy_decision_report_count,
            "policy_decision_families": sorted(
                {
                    row["local_action_family"]
                    for row in outcome_summaries
                    if row.get("source_evidence_schema")
                    in {CRUNCH_ACTIVATION_EVIDENCE_SCHEMA, CRUNCH_POLICY_DECISION_SCHEMA, CACHE_REPLAY_POLICY_DECISION_SCHEMA}
                }
            ),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in outcome_summaries),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in outcome_summaries),
            "error_count": sum(_as_int(row.get("error_count")) for row in outcome_summaries),
            "retry_count": sum(_as_int(row.get("retry_count")) for row in outcome_summaries),
            "fallback_count": sum(_as_int(row.get("fallback_count")) for row in outcome_summaries),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in outcome_summaries), 8),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in outcome_summaries), 8),
        },
        "outcome_summaries": outcome_summaries,
        "local_policy_handoff": {
            "source": "local-activation-outcome-summary",
            "supported_local_action_families": ["routing", "crunch", "cache"],
            "source_policy_decision_schemas": [
                CRUNCH_ACTIVATION_EVIDENCE_SCHEMA,
                CRUNCH_POLICY_DECISION_SCHEMA,
                CACHE_REPLAY_POLICY_DECISION_SCHEMA,
            ],
            "managed_dependency": "optional",
            "server_ingestion_required": False,
        },
        "privacy": _privacy(),
    }
    if managed_activation_bundle_apply_events is not None:
        from agentflow_proxy.managed_activation_bundle_apply_outcomes import (
            build_managed_activation_bundle_apply_outcomes,
        )

        result["managed_activation_bundle_apply_outcomes"] = build_managed_activation_bundle_apply_outcomes(
            managed_activation_bundle_apply_events
        )
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
