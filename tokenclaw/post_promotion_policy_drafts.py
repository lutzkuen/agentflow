from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from tokenclaw.post_promotion_priority_delta_review import SCHEMA as PRIORITY_REVIEW_SCHEMA
from tokenclaw.store import utc_now


SCHEMA = "agentflow.post_promotion_policy_draft_dry_run.v1"
DRAFT_SCHEMA = "agentflow.post_promotion_policy_draft.v1"
OMISSION_SCHEMA = "agentflow.post_promotion_policy_draft_omission.v1"
IMPACT_GATE_SCHEMA = "agentflow.post_promotion_policy_draft_impact_gate.v1"

_VALID_ACTIONS = {"widen-local-policy", "collect-holdout-evidence", "rollback-local-policy", "keep-blocked"}
_VALID_FAMILIES = {"routing", "cache", "crunch"}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}$")
_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")

_FORBIDDEN_KEYS = {
    "api_key",
    "api_keys",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "content",
    "contents",
    "file_content",
    "file_contents",
    "file_path",
    "file_paths",
    "message",
    "messages",
    "password",
    "passwords",
    "prompt",
    "prompts",
    "provider_body",
    "raw_context",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_fingerprint",
    "request_fingerprints",
    "request_id",
    "request_ids",
    "response",
    "secret",
    "secrets",
    "session_id",
    "session_ids",
    "system_prompt",
    "thread_id",
    "thread_ids",
    "tool_input",
    "tool_inputs",
    "tool_payload",
    "tool_payloads",
    "tool_result",
    "tool_results",
    "transcript",
    "transcripts",
}
_ALLOWED_RAW_FLAG_KEYS = {
    "raw_content_included",
    "raw_messages_included",
    "raw_payload_included",
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
    "raw_request_bodies_included",
    "raw_request_ids_included",
    "raw_session_ids_included",
    "raw_source_reports_included",
    "raw_tool_payloads_included",
    "raw_transcripts_included",
}

_FAMILY_POLICY = {
    "routing": {
        "target_local_rule_file": "routing_rules.yaml",
        "target_local_policy_section": "routing.rules",
        "review_command": "agentflow-post-promotion-policy-draft-dry-run",
        "family_review_command": "agentflow-routing-promotion-draft-dry-run",
        "apply_preview_command": "agentflow-policy-draft-stage --section routing <reviewed-draft.json>",
        "widen_operation": "widen_existing_routing_rule",
        "rollback_action_type": "disable_rule",
    },
    "cache": {
        "target_local_rule_file": "cache_rules.yaml",
        "target_local_policy_section": "cache.pattern_rules",
        "review_command": "agentflow-post-promotion-policy-draft-dry-run",
        "family_review_command": "agentflow-cache-promotion-draft-dry-run",
        "apply_preview_command": "agentflow-policy-draft-stage --section cache <reviewed-draft.json>",
        "widen_operation": "widen_existing_cache_pattern_rule",
        "rollback_action_type": "disable_rule",
    },
    "crunch": {
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "anthropic_thinking_history_compaction.rules",
        "review_command": "agentflow-post-promotion-policy-draft-dry-run",
        "family_review_command": "agentflow-crunch-promotion-draft-dry-run",
        "apply_preview_command": "agentflow-policy-draft-stage --section crunch <reviewed-draft.json>",
        "widen_operation": "widen_existing_crunch_rule",
        "rollback_action_type": "disable_rule",
    },
}


def _privacy() -> dict[str, Any]:
    return {
        "local_only": True,
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "content_free": True,
        "read_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "file_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "wrote_local_policy_files": False,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _label(value: Any, *, default: str = "unknown", max_length: int = 200) -> str:
    text = str(value or "").strip().replace("_", "-")
    if not text:
        return default
    text = text[:max_length]
    return text if _LABEL_RE.match(text) else "unsanitized-label"


def _reason(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if not text:
        return None
    return text if _REASON_RE.match(text) else "unsanitized-reason-code"


def _reason_list(*values: Any) -> list[str]:
    reasons: set[str] = set()
    for value in values:
        if isinstance(value, list):
            for item in value:
                reason = _reason(item)
                if reason:
                    reasons.add(reason)
        else:
            reason = _reason(value)
            if reason:
                reasons.add(reason)
    return sorted(reasons)


def _privacy_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).strip()
                lowered = key_text.lower()
                child_path = f"{item_path}.{key_text}"
                if lowered in _ALLOWED_RAW_FLAG_KEYS:
                    continue
                if lowered in _FORBIDDEN_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "post-promotion policy drafts accept metadata only, not raw prompts, provider bodies, identifiers, cache keys, file paths, or secrets",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "status": "invalid",
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": 0,
            "draft_count": 0,
            "widen_draft_count": 0,
            "rollback_draft_count": 0,
            "omitted_count": 0,
        },
        "drafts": [],
        "omitted": [],
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_enforced": False,
        "privacy": _privacy(),
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = str(candidate.get("delta_id") or "").strip()
    if explicit:
        cleaned = "".join(char for char in explicit if char.isalnum() or char in {"-", "_", ":"}).strip("-_:")
        if cleaned:
            return cleaned[:120]
    return _stable_id(
        "post-promotion-delta",
        candidate.get("action_family"),
        candidate.get("next_action"),
        candidate.get("source_surface"),
        candidate.get("recommendation_type"),
        candidate.get("rank"),
    )


def _rule_id(candidate: dict[str, Any], action: str) -> str:
    candidate_id = _candidate_id(candidate)
    suffix = "".join(char if char.isalnum() else "-" for char in candidate_id.lower()).strip("-")
    suffix = suffix[:64] or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"post-promotion-{action.replace('-local-policy', '')}-{suffix}"


def _family(candidate: dict[str, Any]) -> str:
    family = _label(candidate.get("action_family"), default="unknown")
    if family in _VALID_FAMILIES:
        return family
    section = _label(candidate.get("policy_section"), default="unknown")
    return section if section in _VALID_FAMILIES else family


def _policy_meta(family: str) -> dict[str, Any] | None:
    return _FAMILY_POLICY.get(family)


def _evidence_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    return {
        "schema": "agentflow.post_promotion_policy_draft_evidence_summary.v1",
        "delta_id": _candidate_id(candidate),
        "source_review_schema": PRIORITY_REVIEW_SCHEMA,
        "rank": _as_int(candidate.get("rank")),
        "action_family": _family(candidate),
        "source_surface": _label(candidate.get("source_surface"), default="unknown"),
        "recommendation_type": _label(candidate.get("recommendation_type"), default="unknown"),
        "feedback_window": candidate.get("feedback_window"),
        "stability_score_label": candidate.get("stability_score_label"),
        "record_count": _as_int(source.get("record_count")),
        "candidate_count": _as_int(source.get("candidate_count")),
        "rollup_count": _as_int(source.get("rollup_count")),
        "affected_call_count": _first_positive_int(source, candidate, "affected_call_count", "record_count", "sample_count", "candidate_count", "rollup_count"),
        "affected_row_count": _first_positive_int(source, candidate, "affected_row_count", "record_count", "sample_count", "candidate_count", "rollup_count"),
        "promotion_status": source.get("promotion_status"),
        "rank_score": round(_as_float(source.get("rank_score")), 8),
        "savings_delta_usd": round(_as_float(candidate.get("savings_delta_usd") or source.get("savings_delta_usd")), 8),
        "confidence": _bounded_fraction(candidate.get("confidence"), 0.0),
        "no_op_reasons": _reason_list(candidate.get("no_op_reasons")),
        "privacy": _privacy(),
    }


def _candidate_value(candidate: dict[str, Any], *keys: str) -> Any:
    source = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    for key in keys:
        if key in candidate:
            return candidate.get(key)
        if key in source:
            return source.get(key)
    return None


def _first_positive_int(source: dict[str, Any], candidate: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = source.get(key)
        if value is None:
            value = candidate.get(key)
        number = _as_int(value)
        if number > 0:
            return number
    return 0


def _candidate_bool(candidate: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        safe = _as_bool(_candidate_value(candidate, key))
        if safe is not None:
            return safe
    return None


def _candidate_fraction(candidate: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _candidate_value(candidate, key)
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        number = _as_float(value, -1.0)
        if number >= 0.0:
            return _bounded_fraction(number, 0.0)
    return None


def _candidate_count(candidate: dict[str, Any], *keys: str) -> int:
    for key in keys:
        number = _as_int(_candidate_value(candidate, key))
        if number > 0:
            return number
    return 0


def _stale_evidence(candidate: dict[str, Any]) -> bool:
    value = _candidate_value(candidate, "stale_evidence", "stale")
    if isinstance(value, dict):
        return bool(value.get("stale"))
    return bool(_as_bool(value))


def _safety_stop_active(candidate: dict[str, Any]) -> bool:
    if _candidate_count(candidate, "safety_stop_count", "safety_stopped_count") > 0:
        return True
    if _candidate_bool(candidate, "safety_stop_active", "safety_stop_tripped") is True:
        return True
    status = _label(_candidate_value(candidate, "safety_stop_status", "promotion_status"), default="")
    if status in {"safety-stop", "safety-stopped", "safety-stop-active", "safety-stop-tripped"}:
        return True
    return any("safety-stop" in reason for reason in _reason_list(candidate.get("no_op_reasons")))


def _has_holdout_coverage(candidate: dict[str, Any]) -> bool:
    if _candidate_count(candidate, "current_holdout_count", "holdout_count", "canary_holdout_count") > 0:
        return True
    fraction = _candidate_fraction(candidate, "current_holdout_fraction", "holdout_fraction")
    return bool(fraction and fraction > 0.0)


def _has_preserved_previous_rule(candidate: dict[str, Any]) -> bool:
    return _candidate_bool(
        candidate,
        "preserved_previous_rule",
        "previous_rule_preserved",
        "previous_rule_available",
    ) is True


def _impact_gate(
    candidate: dict[str, Any],
    *,
    action: str,
    rule_id: str,
    widen_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    affected_call_count = _first_positive_int(
        evidence,
        candidate,
        "affected_call_count",
        "record_count",
        "sample_count",
        "candidate_count",
        "rollup_count",
    )
    affected_row_count = _first_positive_int(
        evidence,
        candidate,
        "affected_row_count",
        "record_count",
        "sample_count",
        "candidate_count",
        "rollup_count",
    )
    current_canary = _candidate_fraction(candidate, "current_canary_fraction", "canary_fraction") or 0.0
    projected_canary = 0.0 if action == "rollback-local-policy" else _bounded_fraction(current_canary + widen_fraction, widen_fraction)
    observed_holdout = _candidate_fraction(candidate, "current_holdout_fraction", "holdout_fraction")
    required_holdout = 0.0 if action == "rollback-local-policy" else _bounded_fraction(holdout_fraction, 0.10)

    blockers: list[str] = []
    if action == "widen-local-policy" and not _has_holdout_coverage(candidate):
        blockers.append("missing-holdout-coverage")
    if _stale_evidence(candidate):
        blockers.append("stale-evidence")
    if _safety_stop_active(candidate):
        blockers.append("safety-stop-active")
    if action == "rollback-local-policy" and not _has_preserved_previous_rule(candidate):
        blockers.append("missing-preserved-previous-rule")

    return {
        "schema": IMPACT_GATE_SCHEMA,
        "status": "blocked" if blockers else "passed",
        "reason": blockers[0] if blockers else "impact-gate-passed",
        "blocker_reasons": blockers,
        "draft_action": action,
        "rule_id": rule_id,
        "action_family": _family(candidate),
        "source_surface": _label(candidate.get("source_surface"), default="unknown"),
        "recommendation_type": _label(candidate.get("recommendation_type"), default="unknown"),
        "affected_call_count": affected_call_count,
        "affected_row_count": affected_row_count,
        "current_canary_fraction": current_canary,
        "projected_canary_fraction": projected_canary,
        "required_holdout_fraction": required_holdout,
        "observed_holdout_fraction": observed_holdout if observed_holdout is not None else 0.0,
        "holdout_coverage_present": _has_holdout_coverage(candidate),
        "stale_evidence": _stale_evidence(candidate),
        "safety_stop_active": _safety_stop_active(candidate),
        "preserved_previous_rule": _has_preserved_previous_rule(candidate),
        "expected_savings_delta_usd": round(_as_float(candidate.get("savings_delta_usd") or evidence.get("savings_delta_usd")), 8),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }


def _review_metadata(candidate: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.post_promotion_policy_draft_review_metadata.v1",
        "required": True,
        "source": "post-promotion-priority-delta-review",
        "recommendation_source": "managed-recommended",
        "local_execution_required": True,
        "target_local_rule_file": policy["target_local_rule_file"],
        "target_local_policy_section": policy["target_local_policy_section"],
        "review_command": policy["review_command"],
        "family_review_command": policy["family_review_command"],
        "apply_preview_command": policy["apply_preview_command"],
        "policy_source": "local-manual",
        "delta_id": _candidate_id(candidate),
        "privacy": _privacy(),
    }


def _widen_draft(
    candidate: dict[str, Any],
    policy: dict[str, Any],
    *,
    widen_fraction: float,
    holdout_fraction: float,
    impact_gate: dict[str, Any],
) -> dict[str, Any]:
    evidence = _evidence_summary(candidate)
    activation_delta = _bounded_fraction(widen_fraction, 0.05)
    holdout = _bounded_fraction(holdout_fraction, 0.10)
    rule_id = _rule_id(candidate, "widen-local-policy")
    patch = {
        "schema": "agentflow.post_promotion_local_policy_patch.v1",
        "operation": policy["widen_operation"],
        "requires_existing_compatible_rule": True,
        "target_rule_selector": {
            "delta_id": _candidate_id(candidate),
            "recommendation_type": evidence["recommendation_type"],
            "source_surface": evidence["source_surface"],
            "policy_section": policy["target_local_policy_section"],
        },
        "bounded_changes": {
            "activation_fraction_delta_lte": activation_delta,
            "preserve_holdout_fraction_gte": holdout,
            "preserve_safety_stop": True,
            "preserve_rate_limit_fallback": True,
            "preserve_requested_model_fallback": True,
            "preserve_operator_overrides": True,
        },
        "review_required": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    return {
        "schema": DRAFT_SCHEMA,
        "status": "drafted",
        "draft_action": "widen-local-policy",
        "draft_id": rule_id,
        "rule_id": rule_id,
        "target_candidate_id": _candidate_id(candidate),
        "action_family": evidence["action_family"],
        "target_local_rule_file": policy["target_local_rule_file"],
        "target_local_policy_section": policy["target_local_policy_section"],
        "source": "post-promotion-priority-delta-review",
        "review": _review_metadata(candidate, policy),
        "proposed_policy_patch": patch,
        "evidence_summary": evidence,
        "dry_run_impact_estimate": {
            "schema": "agentflow.post_promotion_policy_draft_impact_estimate.v1",
            "affected_call_count": impact_gate["affected_call_count"],
            "affected_row_count": impact_gate["affected_row_count"],
            "savings_delta_usd": evidence["savings_delta_usd"],
            "confidence": evidence["confidence"],
            "activation_fraction_delta_lte": activation_delta,
            "holdout_fraction_gte": holdout,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "dry_run_impact_gate": impact_gate,
        "rollback_metadata": {
            "schema": "agentflow.post_promotion_policy_draft_rollback_metadata.v1",
            "rollback_action_type": policy["rollback_action_type"],
            "preserve_previous_rule_required": True,
            "preserve_operator_rule_history": True,
            "rollback_reason_codes": [
                "safety-stop-observed",
                "error-rate-regression",
                "negative-savings-regression",
                "operator-requested",
            ],
        },
        "privacy": _privacy(),
    }


def _rollback_draft(
    candidate: dict[str, Any],
    policy: dict[str, Any],
    *,
    impact_gate: dict[str, Any],
) -> dict[str, Any]:
    evidence = _evidence_summary(candidate)
    rule_id = _rule_id(candidate, "rollback-local-policy")
    patch = {
        "schema": "agentflow.post_promotion_local_policy_patch.v1",
        "operation": "rollback_existing_rule",
        "rollback_action_type": policy["rollback_action_type"],
        "requires_existing_compatible_rule": True,
        "target_rule_selector": {
            "delta_id": _candidate_id(candidate),
            "recommendation_type": evidence["recommendation_type"],
            "source_surface": evidence["source_surface"],
            "policy_section": policy["target_local_policy_section"],
        },
        "bounded_changes": {
            "enabled": False,
            "activation_fraction": 0.0,
            "preserve_previous_rule_required": True,
            "preserve_operator_rule_history": True,
            "delete_rule": False,
        },
        "review_required": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    return {
        "schema": DRAFT_SCHEMA,
        "status": "drafted",
        "draft_action": "rollback-local-policy",
        "draft_id": rule_id,
        "rule_id": rule_id,
        "target_candidate_id": _candidate_id(candidate),
        "action_family": evidence["action_family"],
        "target_local_rule_file": policy["target_local_rule_file"],
        "target_local_policy_section": policy["target_local_policy_section"],
        "source": "post-promotion-priority-delta-review",
        "review": _review_metadata(candidate, policy),
        "proposed_policy_patch": patch,
        "evidence_summary": evidence,
        "dry_run_impact_estimate": {
            "schema": "agentflow.post_promotion_policy_draft_impact_estimate.v1",
            "affected_call_count": impact_gate["affected_call_count"],
            "affected_row_count": impact_gate["affected_row_count"],
            "savings_delta_usd": evidence["savings_delta_usd"],
            "confidence": evidence["confidence"],
            "projected_policy_risk_removed": True,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "dry_run_impact_gate": impact_gate,
        "rollback_metadata": {
            "schema": "agentflow.post_promotion_policy_draft_rollback_metadata.v1",
            "rollback_action_type": policy["rollback_action_type"],
            "preserve_previous_rule_required": True,
            "preserve_operator_rule_history": True,
            "rollback_reason_codes": _reason_list(candidate.get("no_op_reasons"), "post-promotion-negative-delta", "operator-requested"),
        },
        "privacy": _privacy(),
    }


def _omission(
    candidate: dict[str, Any],
    reason: str,
    *,
    path: str | None = None,
    impact_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = _family(candidate)
    policy = _policy_meta(family) or {
        "target_local_rule_file": f"{family}_rules.yaml",
        "target_local_policy_section": f"{family}.rules",
        "review_command": "agentflow-post-promotion-policy-draft-dry-run",
        "family_review_command": "",
        "apply_preview_command": "",
    }
    result = {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "path": path,
        "target_candidate_id": _candidate_id(candidate),
        "action_family": family,
        "next_action": _label(candidate.get("next_action"), default="keep-blocked"),
        "target_local_rule_file": policy["target_local_rule_file"],
        "target_local_policy_section": policy["target_local_policy_section"],
        "source": "post-promotion-priority-delta-review",
        "review": _review_metadata(candidate, policy),
        "no_op_reasons": _reason_list(candidate.get("no_op_reasons"), reason),
        "privacy": _privacy(),
    }
    if impact_gate is not None:
        result["dry_run_impact_gate"] = impact_gate
    return result


def _draft_for_candidate(
    candidate: dict[str, Any],
    *,
    path: str,
    widen_fraction: float,
    holdout_fraction: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    family = _family(candidate)
    policy = _policy_meta(family)
    if policy is None:
        return None, _omission(candidate, "unsupported-action-family", path=path)
    action = _label(candidate.get("next_action"), default="keep-blocked")
    if action not in _VALID_ACTIONS:
        return None, _omission(candidate, "unsupported-next-action", path=path)
    if candidate.get("status") != "recommended" or action in {"keep-blocked", "collect-holdout-evidence"}:
        reason = "keep-blocked" if action == "keep-blocked" else "not-recommended"
        if action == "collect-holdout-evidence":
            reason = "collect-holdout-evidence"
        return None, _omission(candidate, reason, path=path)
    rule_id = _rule_id(candidate, action)
    impact_gate = _impact_gate(
        candidate,
        action=action,
        rule_id=rule_id,
        widen_fraction=widen_fraction,
        holdout_fraction=holdout_fraction,
    )
    if impact_gate["status"] == "blocked":
        return None, _omission(candidate, str(impact_gate["reason"]), path=path, impact_gate=impact_gate)
    if action == "widen-local-policy":
        return _widen_draft(
            candidate,
            policy,
            widen_fraction=widen_fraction,
            holdout_fraction=holdout_fraction,
            impact_gate=impact_gate,
        ), None
    if action == "rollback-local-policy":
        return _rollback_draft(candidate, policy, impact_gate=impact_gate), None
    return None, _omission(candidate, "unsupported-next-action", path=path)


def _count_rows(values: list[str]) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count} for value, count in Counter(values).items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def build_post_promotion_policy_drafts(
    priority_review: Any,
    *,
    widen_fraction: float = 0.05,
    holdout_fraction: float = 0.10,
    max_drafts: int = 20,
) -> dict[str, Any]:
    if not isinstance(priority_review, dict):
        return _error_result("invalid_report", "post-promotion priority review must be a JSON object")
    raw_errors = _privacy_errors(priority_review)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "post-promotion priority review contains raw prompt, response, provider body, identifier, file path, cache key, or secret fields",
            errors=raw_errors,
        )
    candidates = priority_review.get("candidates")
    if not isinstance(candidates, list):
        return _error_result("invalid_report", "post-promotion priority review must include a candidates list")

    draft_limit = max(1, min(_as_int(max_drafts, 20), 100))
    drafts: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            omitted.append({
                "schema": OMISSION_SCHEMA,
                "status": "omitted",
                "reason": "invalid-candidate",
                "path": f"$.candidates[{index}]",
                "target_candidate_id": None,
                "privacy": _privacy(),
            })
            continue
        if len(drafts) >= draft_limit and candidate.get("next_action") != "keep-blocked":
            omitted.append(_omission(candidate, "max-drafts-reached", path=f"$.candidates[{index}]"))
            continue
        draft, omission = _draft_for_candidate(
            candidate,
            path=f"$.candidates[{index}]",
            widen_fraction=widen_fraction,
            holdout_fraction=holdout_fraction,
        )
        if draft is not None:
            drafts.append(draft)
        if omission is not None:
            omitted.append(omission)

    draft_actions = [str(item.get("draft_action") or "unknown") for item in drafts]
    omitted_reasons = [str(item.get("reason") or "unknown") for item in omitted]
    impact_gates = [
        item["dry_run_impact_gate"]
        for item in drafts + omitted
        if isinstance(item.get("dry_run_impact_gate"), dict)
    ]
    blocked_impact_gates = [gate for gate in impact_gates if gate.get("status") == "blocked"]
    ok = bool(drafts or omitted) and not blocked_impact_gates
    result = {
        "schema": SCHEMA,
        "ok": ok,
        "status": "blocked" if blocked_impact_gates else "drafted" if drafts else "no-op",
        "generated_at": utc_now(),
        "source_report_schema": priority_review.get("schema"),
        "source_report_generated_at": priority_review.get("generated_at"),
        "summary": {
            "candidate_count": len(candidates),
            "draft_count": len(drafts),
            "widen_draft_count": sum(1 for item in drafts if item.get("draft_action") == "widen-local-policy"),
            "rollback_draft_count": sum(1 for item in drafts if item.get("draft_action") == "rollback-local-policy"),
            "omitted_count": len(omitted),
            "draft_action_counts": _count_rows(draft_actions),
            "omission_reason_counts": _count_rows(omitted_reasons),
            "impact_gate_count": len(impact_gates),
            "impact_gate_pass_count": sum(1 for gate in impact_gates if gate.get("status") == "passed"),
            "impact_gate_blocked_count": len(blocked_impact_gates),
            "impact_gate_blocker_reason_counts": _count_rows([
                str(reason)
                for gate in blocked_impact_gates
                for reason in gate.get("blocker_reasons", [])
            ]),
            "projected_affected_call_count": sum(_as_int(gate.get("affected_call_count")) for gate in impact_gates),
            "projected_affected_row_count": sum(_as_int(gate.get("affected_row_count")) for gate in impact_gates),
            "widen_fraction": _bounded_fraction(widen_fraction, 0.05),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "target_local_rule_files": sorted({str(item.get("target_local_rule_file")) for item in drafts + omitted if item.get("target_local_rule_file")}),
            "target_local_policy_sections": sorted({str(item.get("target_local_policy_section")) for item in drafts + omitted if item.get("target_local_policy_section")}),
        },
        "drafts": drafts,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "wrote_local_files": False,
        "wrote_local_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "managed_enforced": False,
        "privacy": _privacy(),
        "error": (
            {
                "type": "impact_gate_blocked",
                "message": "one or more post-promotion policy drafts were blocked by local dry-run impact gates",
                "blocker_reason_counts": _count_rows([
                    str(reason)
                    for gate in blocked_impact_gates
                    for reason in gate.get("blocker_reasons", [])
                ]),
            }
            if blocked_impact_gates
            else None
            if ok
            else {"type": "no_candidates", "message": "no post-promotion priority candidates were available"}
        ),
    }
    json.dumps(result, sort_keys=True)
    return result
