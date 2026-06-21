from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.openai_routing_narrow_canary_review.v1"
DRAFT_SCHEMA = "tokenclaw.openai_routing_narrow_canary_draft.v1"
OMISSION_SCHEMA = "tokenclaw.openai_routing_narrow_canary_omission.v1"
RECOVERY_PLAN_SCHEMA = "tokenclaw.openai_routing_recovery_plan.v1"
RECOVERY_OPTION_SCHEMA = "tokenclaw.openai_routing_recovery_option.v1"
RECOVERY_COVERAGE_SCHEMA = "tokenclaw.openai_routing_recovery_coverage.v1"
RECOVERY_ROLLBACK_SCHEMA = "tokenclaw.openai_routing_recovery_rollback_no_write.v1"
RECOVERY_STALE_EVIDENCE_SCHEMA = "tokenclaw.openai_routing_recovery_stale_evidence.v1"
MANAGED_PREVIEW_AGREEMENT_SCHEMA = "tokenclaw.openai_routing_managed_preview_agreement.v1"
MANAGED_PREVIEW_HEALTH_GATE_SCHEMA = "tokenclaw.openai_routing_managed_preview_health_gate.v1"
RECOVERY_SIZING_SCHEMA = "tokenclaw.openai_routing_recovery_sizing.v1"
PATHWAY_OUTCOME_SCHEMA = "tokenclaw.local_routing_pathway_outcome_feedback.v1"
PATHWAY_OUTCOME_ROW_SCHEMA = "tokenclaw.local_routing_pathway_outcome_feedback_row.v1"
ROUTING_PREVIEW_EVIDENCE_SCHEMAS = {
    "tokenclaw.openai_routing_promotion_decision_report.v1",
    "tokenclaw.pass_through_routing_activation_candidates.v1",
    PATHWAY_OUTCOME_SCHEMA,
    PATHWAY_OUTCOME_ROW_SCHEMA,
}
PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "aggregate_only": True,
    "raw_prompts_included": False,
    "raw_messages_included": False,
    "raw_request_bodies_included": False,
    "raw_response_bodies_included": False,
    "provider_bodies_included": False,
    "raw_provider_bodies_included": False,
    "request_ids_included": False,
    "raw_request_ids_included": False,
    "session_ids_included": False,
    "raw_session_ids_included": False,
    "cache_keys_included": False,
    "individual_candidate_ids_included": False,
    "file_paths_included": False,
    "secrets_included": False,
    "policy_file_contents_included": False,
    "policy_files_written": False,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
}

RAW_KEYS = {
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "content",
    "file_path",
    "messages",
    "password",
    "prompt",
    "provider_body",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_id",
    "response",
    "secret",
    "session_id",
    "system_prompt",
    "thread_id",
    "tool_input",
    "tool_payload",
    "tool_result",
    "transcript",
}
ALLOWED_RAW_FLAG_KEYS = {
    "raw_prompts_included",
    "raw_messages_included",
    "raw_provider_bodies_included",
    "raw_request_ids_included",
    "raw_session_ids_included",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


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


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item or "").strip()})
    text = _string(value)
    return [text] if text else []


def _privacy_errors(value: Any, *, path: str = "$") -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def walk(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key).strip()
                lowered = key_text.lower()
                child_path = f"{item_path}.{key_text}"
                if lowered in ALLOWED_RAW_FLAG_KEYS or (lowered.endswith("_included") and isinstance(child, bool)):
                    continue
                if lowered in RAW_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "OpenAI routing narrow-canary review accepts metadata only, not raw prompts, provider bodies, identifiers, file paths, cache keys, or secrets",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _safe_model_pair(cohort: dict[str, Any]) -> tuple[str, str]:
    requested = _string(
        cohort.get("requested_model")
        or cohort.get("model_pattern")
        or (cohort.get("target") or {}).get("requested_model")
    )
    target = _string(
        cohort.get("target_model")
        or cohort.get("candidate_target_model")
        or cohort.get("routed_model")
        or (cohort.get("target") or {}).get("target_model")
    )
    return requested, target


def _is_codex_surface(cohort: dict[str, Any]) -> bool:
    surface = _string(cohort.get("source_surface")).lower()
    app_family = _string(cohort.get("app_family")).lower()
    return surface in {"codex_turn", "codex_app_turn"} or app_family == "codex"


def _policy_target(cohort: dict[str, Any]) -> dict[str, str]:
    if _is_codex_surface(cohort):
        return {
            "policy_section": "codex_app.summary_model_hint",
            "rule_file": "codex_app_rules.yaml",
            "rollback_action_type": "disable_codex_app_routing_narrow_canary",
            "policy_id_prefix": "local-codex-routing-narrow-canary-",
            "fingerprint_prefix": "codex-routing-narrow-canary",
        }
    return {
        "policy_section": "routing.rules",
        "rule_file": "routing_rules.yaml",
        "rollback_action_type": "disable_openai_routing_narrow_canary",
        "policy_id_prefix": "local-openai-routing-narrow-canary-",
        "fingerprint_prefix": "openai-routing-narrow-canary",
    }


def _cohort_counts(cohort: dict[str, Any]) -> dict[str, int]:
    lifecycle = cohort.get("openai_canary_lifecycle_evidence")
    if not isinstance(lifecycle, dict):
        lifecycle = cohort.get("lifecycle") if isinstance(cohort.get("lifecycle"), dict) else {}
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    applied = (
        _as_int(cohort.get("applied_count"))
        or _as_int(counts.get("canary_applied"))
        or _as_int(lifecycle.get("applied_count"))
    )
    holdout = (
        _as_int(cohort.get("holdout_count"))
        or _as_int(counts.get("canary_holdout"))
        or _as_int(lifecycle.get("holdout_count"))
    )
    safety_stop = (
        _as_int(cohort.get("safety_stop_count"))
        or _as_int(counts.get("safety_stopped"))
        or _as_int(lifecycle.get("safety_stop_count"))
    )
    skipped = _as_int(cohort.get("skipped_count")) or _as_int(counts.get("skipped")) or _as_int(lifecycle.get("skipped_count"))
    unknown = _as_int(cohort.get("unknown_count")) or _as_int(counts.get("unknown")) or _as_int(lifecycle.get("unknown_count"))
    matched = (
        _as_int(cohort.get("matched_count"))
        or _as_int(cohort.get("sample_count"))
        or _as_int(cohort.get("blocked_count"))
        or _as_int(lifecycle.get("matched_count"))
        or applied + holdout + safety_stop + skipped + unknown
    )
    return {
        "matched": matched,
        "applied": applied,
        "holdout": holdout,
        "safety_stop": safety_stop,
        "skipped": skipped,
        "unknown": unknown,
    }


def _lifecycle(cohort: dict[str, Any]) -> dict[str, Any]:
    lifecycle = cohort.get("openai_canary_lifecycle_evidence")
    if isinstance(lifecycle, dict):
        return lifecycle
    lifecycle = cohort.get("lifecycle")
    return lifecycle if isinstance(lifecycle, dict) else {}


def _stale_evidence(cohort: dict[str, Any]) -> dict[str, Any]:
    stale = cohort.get("stale_evidence")
    if not isinstance(stale, dict):
        lifecycle = _lifecycle(cohort)
        stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
    is_stale = bool(stale.get("stale")) if stale else False
    latest = stale.get("latest_observed_at") if stale else None
    if latest is None:
        latest = _lifecycle(cohort).get("latest_observed_at")
    age_hours = stale.get("age_hours") if stale else None
    max_age_hours = stale.get("max_age_hours") if stale else 72.0
    if stale:
        status = "stale" if is_stale else "fresh"
    elif latest:
        status = "fresh-or-active"
    else:
        status = "unknown"
    return {
        "schema": RECOVERY_STALE_EVIDENCE_SCHEMA,
        "status": status,
        "stale": is_stale,
        "latest_observed_at": latest,
        "age_hours": age_hours,
        "max_age_hours": max_age_hours,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _coverage(cohort: dict[str, Any]) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    lifecycle = _lifecycle(cohort)
    return {
        "schema": RECOVERY_COVERAGE_SCHEMA,
        "matched_count": counts["matched"],
        "applied_count": counts["applied"],
        "holdout_count": counts["holdout"],
        "safety_stop_count": counts["safety_stop"],
        "skipped_count": counts["skipped"],
        "unknown_count": counts["unknown"],
        "error_count": _as_int(cohort.get("error_count") or lifecycle.get("error_count")),
        "fallback_count": _as_int(cohort.get("fallback_count") or lifecycle.get("fallback_count")),
        "retry_count": _as_int(cohort.get("retry_count") or lifecycle.get("retry_count")),
        "has_applied_coverage": counts["applied"] > 0,
        "has_holdout_coverage": counts["holdout"] > 0,
        "has_no_safety_stops": counts["safety_stop"] == 0,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _rollback_no_write(selected_option: str, cohort: dict[str, Any] | None = None) -> dict[str, Any]:
    target = _policy_target(cohort or {})
    return {
        "schema": RECOVERY_ROLLBACK_SCHEMA,
        "selected_option": selected_option,
        "active_policy_changed": False,
        "policy_files_written": False,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "rollback_action_type": target["rollback_action_type"],
        "rollback_required_before_activation": True,
        "target_local_policy_section": target["policy_section"],
        "target_local_rule_file": target["rule_file"],
        "metadata_only": True,
        "aggregate_only": True,
    }


def _option(name: str, *, selected: str, allowed: bool, reason: str, next_action: str) -> dict[str, Any]:
    return {
        "schema": RECOVERY_OPTION_SCHEMA,
        "option": name,
        "selected": name == selected,
        "allowed": allowed,
        "review_only": True,
        "reason": reason,
        "next_action": next_action,
        "active_policy_changed": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _recovery_plan(cohort: dict[str, Any], *, omission_reason: str | None) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    reasons = set(_reason_codes(cohort))
    requested, target_model = _safe_model_pair(cohort)
    stale = _stale_evidence(cohort)
    coverage = _coverage(cohort)
    semantic = _semantic_recovery_classification(cohort)
    semantic_blocked = (
        semantic["observed"]
        and semantic["classification"] != "narrower-canary"
        and (
            omission_reason is None
            or omission_reason
            in {
                "semantic-quality-regression-observed",
                "semantic-regression-rollback-required",
                "openai-routing-recovery-blocked",
            }
        )
    )
    missing_coverage = counts["applied"] <= 0 or counts["holdout"] <= 0
    safety_blocked = counts["safety_stop"] > 0 or "safety-stop-observed" in reasons
    stale_blocked = bool(stale.get("stale"))
    clean_for_review = not semantic_blocked and not missing_coverage and not safety_blocked and not stale_blocked
    selected_option = "restage-review-only" if clean_for_review else "keep-blocked"
    blocker_reason = omission_reason or ("none" if clean_for_review else "openai-routing-recovery-blocked")
    blocker_status = "cleared" if clean_for_review else "active"
    restage_reason = "blocker-cleared-fresh-applied-holdout-coverage" if clean_for_review else blocker_reason
    policy_target = _policy_target(cohort)
    return {
        "schema": RECOVERY_PLAN_SCHEMA,
        "fingerprint": _stable_id("openai-routing-recovery", _fingerprint(cohort), selected_option, blocker_reason),
        "selected_option": selected_option,
        "decision": "draft-narrower-canary" if clean_for_review else "keep-blocked",
        "status": "review-only" if clean_for_review else "keep-blocked",
        "next_action": "operator-review-narrower-openai-routing-canary" if clean_for_review else "keep-openai-routing-blocked",
        "target_local_policy_section": policy_target["policy_section"],
        "target_local_rule_file": policy_target["rule_file"],
        "provider": "openai",
        "source_surface": cohort.get("source_surface"),
        "app_family": cohort.get("app_family"),
        "endpoint": cohort.get("endpoint"),
        "requested_model": requested,
        "target_model": target_model,
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "blocker_status": blocker_status,
        "blocker_reason": blocker_reason,
        "blocker_codes": sorted(reasons),
        "semantic_regression_recovery": semantic,
        "coverage": coverage,
        "stale_evidence": stale,
        "rollback_no_write": _rollback_no_write(selected_option, cohort),
        "options": [
            _option(
                "keep-blocked",
                selected=selected_option,
                allowed=True,
                reason=blocker_reason,
                next_action="keep-openai-routing-blocked",
            ),
            _option(
                "retire-disabled-rule",
                selected=selected_option,
                allowed=semantic_blocked,
                reason="manual-retirement-review-for-disabled-semantic-regression-rule" if semantic_blocked else "no-disabled-semantic-regression-rule",
                next_action="operator-review-disabled-openai-routing-rule-retirement",
            ),
            _option(
                "narrow-threshold",
                selected=selected_option,
                allowed=clean_for_review,
                reason=restage_reason,
                next_action="operator-review-narrow-openai-routing-threshold",
            ),
            _option(
                "restage-review-only",
                selected=selected_option,
                allowed=clean_for_review,
                reason=restage_reason,
                next_action="operator-review-narrower-openai-routing-canary",
            ),
        ],
        "privacy": PRIVACY,
    }


def _semantic_regression_action(cohort: dict[str, Any]) -> dict[str, Any]:
    action = cohort.get("semantic_regression_action")
    if isinstance(action, dict):
        return action
    lifecycle_review = cohort.get("routing_lifecycle_review")
    if isinstance(lifecycle_review, dict) and isinstance(lifecycle_review.get("semantic_regression_action"), dict):
        return lifecycle_review["semantic_regression_action"]
    promotion_decision = cohort.get("promotion_decision")
    if isinstance(promotion_decision, dict) and isinstance(promotion_decision.get("semantic_regression_action"), dict):
        return promotion_decision["semantic_regression_action"]
    return {}


def _semantic_recovery_classification(cohort: dict[str, Any]) -> dict[str, Any]:
    reasons = set(_reason_codes(cohort))
    action = _semantic_regression_action(cohort)
    action_classification = _string(action.get("action_classification"))
    action_next = _string(action.get("deterministic_next_action") or action.get("next_action"))
    observed = bool(action.get("observed")) or "semantic-quality-regression-observed" in reasons
    if not observed:
        classification = "not-applicable"
        next_action = "use-existing-openai-routing-promotion-decision"
    elif action_classification == "narrow-canary-shape":
        classification = "narrower-canary"
        next_action = action_next or "draft-narrow-openai-routing-canary-shape"
    elif action_classification == "rollback-required":
        classification = "rollback"
        next_action = action_next or "draft-openai-routing-rollback"
    else:
        classification = "keep-blocked"
        next_action = action_next or "keep-openai-routing-blocked"
    return {
        "schema": "tokenclaw.openai_routing_semantic_regression_recovery_classification.v1",
        "observed": observed,
        "classification": classification,
        "action_classification": action_classification or None,
        "next_action": next_action,
        "reason_codes": sorted(reasons),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _normalized_recovery_action(action: Any) -> str:
    text = _string(action).lower()
    if text in {
        "draft-codex-routing-recovery-canary",
        "draft-openai-routing-recovery-canary",
        "draft-narrow-openai-routing-canary-shape",
        "draft-narrower-openai-routing-canary",
        "stage-narrow-routing-canary",
        "narrow-openai-routing-canary-shape",
        "operator-review-codex-routing-canary",
        "operator-review-narrower-openai-routing-canary",
        "operator-review-narrow-openai-routing-threshold",
    }:
        return "draft-recovery-canary"
    if text in {"draft-openai-routing-rollback", "rollback-required", "rollback-local-routing-rule"}:
        return "rollback"
    if text in {"keep-openai-routing-blocked", "keep-blocked", "review-openai-routing-canary-blockers"}:
        return "keep-blocked"
    return text or "unknown"


def _local_recovery_next_action(cohort: dict[str, Any]) -> str:
    semantic = _semantic_recovery_classification(cohort)
    if semantic["classification"] == "narrower-canary":
        return str(semantic["next_action"])
    if semantic["classification"] == "rollback":
        return str(semantic["next_action"])
    if semantic["classification"] == "keep-blocked":
        return str(semantic["next_action"])
    if _is_codex_surface(cohort):
        return "draft-codex-routing-recovery-canary"
    return "draft-openai-routing-recovery-canary"


def _managed_preview_outcomes(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows = report.get("outcomes")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _preview_age_hours(report: dict[str, Any]) -> float | None:
    for key in ("latest_preview_age_hours", "preview_age_hours", "age_hours"):
        if report.get(key) is not None:
            return _as_float(report.get(key))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    for key in ("latest_preview_age_hours", "preview_age_hours", "age_hours"):
        if summary.get(key) is not None:
            return _as_float(summary.get(key))
    return None


def _managed_preview_health_gate(
    managed_preview_health: dict[str, Any] | None,
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any]:
    report = managed_preview_health if isinstance(managed_preview_health, dict) else {}
    if not report and isinstance(managed_preview_outcomes, dict):
        report = managed_preview_outcomes
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    status_text = _string(report.get("status") or summary.get("status")).lower()
    outcomes = _managed_preview_outcomes(managed_preview_outcomes)
    stale_after_hours = _as_float(
        report.get("stale_after_hours")
        or summary.get("stale_after_hours")
        or (managed_preview_outcomes or {}).get("stale_after_hours")
        or 72.0,
        72.0,
    )
    age_hours = _preview_age_hours(report)
    stale = bool(report.get("stale") or summary.get("stale") or (age_hours is not None and age_hours > stale_after_hours))
    accepted_batch_count = _as_int(
        report.get("accepted_batch_count")
        or summary.get("accepted_batch_count")
        or (1 if status_text in {"tracked", "previewed", "ok"} else 0)
    )
    previewed_row_count = _as_int(
        report.get("previewed_row_count")
        or summary.get("previewed_row_count")
        or summary.get("stored_preview_outcome_count")
        or len(outcomes)
    )
    rejected_batch_count = _as_int(report.get("rejected_batch_count") or summary.get("rejected_batch_count"))
    rejected_row_count = _as_int(report.get("rejected_row_count") or summary.get("rejected_row_count"))
    validation_error_count = _as_int(report.get("validation_error_count") or summary.get("validation_error_count"))
    if not report:
        status = "missing-preview-health"
        reason = "managed-preview-health-missing"
        next_action = "refresh-managed-activation-preview"
    elif status_text in {"no-data-preview-health", "no-data-preview", "no-data", "missing-preview"}:
        status = "no-data-preview-health"
        reason = _string(report.get("reason") or summary.get("reason")) or "managed-preview-health-no-data"
        next_action = _string(report.get("next_action") or summary.get("next_action")) or "refresh-managed-activation-preview"
    elif status_text in {"blocked", "error", "unavailable"} or rejected_batch_count > 0 or rejected_row_count > 0 or validation_error_count > 0:
        status = "rejected-preview-health"
        reason = "managed-preview-health-rejected"
        next_action = "review-managed-activation-preview-rejection"
    elif stale:
        status = "stale-preview-health"
        reason = "managed-preview-health-stale"
        next_action = "refresh-managed-activation-preview"
    elif accepted_batch_count <= 0 or previewed_row_count <= 0:
        status = "incomplete-preview-health"
        reason = "managed-preview-health-incomplete"
        next_action = "refresh-managed-activation-preview"
    else:
        status = "fresh-preview-health"
        reason = "managed-preview-health-fresh"
        next_action = "use-managed-preview-gate"
    return {
        "schema": MANAGED_PREVIEW_HEALTH_GATE_SCHEMA,
        "status": status,
        "passed": status == "fresh-preview-health",
        "reason": reason,
        "next_action": next_action,
        "accepted_batch_count": accepted_batch_count,
        "previewed_row_count": previewed_row_count,
        "rejected_batch_count": rejected_batch_count,
        "rejected_row_count": rejected_row_count,
        "validation_error_count": validation_error_count,
        "latest_preview_age_hours": age_hours,
        "stale_after_hours": stale_after_hours,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": bool(report.get("managed_server_calls_made") or summary.get("managed_server_calls_made")),
        "metadata_only": True,
        "aggregate_only": True,
    }


def _embedded_managed_preview_health(cohort: dict[str, Any]) -> dict[str, Any] | None:
    gate = cohort.get("managed_preview_gate")
    if not isinstance(gate, dict):
        return None
    health = gate.get("health_gate")
    return health if isinstance(health, dict) else gate


def _cohort_match_refs(cohort: dict[str, Any]) -> set[str]:
    refs = {
        _string(cohort.get("fingerprint")),
        _string(cohort.get("source_fingerprint")),
        _string(cohort.get("candidate_fingerprint")),
        _string(cohort.get("target_candidate_id")),
        _fingerprint(cohort),
    }
    candidate_set = cohort.get("candidate_set") if isinstance(cohort.get("candidate_set"), dict) else {}
    refs.add(_string(candidate_set.get("candidate_fingerprint")))
    return {ref for ref in refs if ref}


def _outcome_match_score(cohort: dict[str, Any], outcome: dict[str, Any]) -> int:
    if _string(outcome.get("local_action_family")).lower() != "routing":
        return -1
    evidence_schema = _string(outcome.get("evidence_schema"))
    if evidence_schema and evidence_schema not in ROUTING_PREVIEW_EVIDENCE_SCHEMAS:
        return -1
    cohort_refs = _cohort_match_refs(cohort)
    outcome_candidate_refs = {
        _string(outcome.get("candidate_fingerprint")),
        _string(outcome.get("source_candidate_fingerprint")),
    }
    outcome_source_refs = {
        _string(outcome.get("source_fingerprint")),
        _string(outcome.get("source_activation_fingerprint")),
        _string(outcome.get("activation_fingerprint")),
        _string(outcome.get("local_activation_fingerprint")),
    }
    outcome_candidate_refs = {ref for ref in outcome_candidate_refs if ref}
    outcome_source_refs = {ref for ref in outcome_source_refs if ref}
    if outcome_candidate_refs and cohort_refs.isdisjoint(outcome_candidate_refs):
        return -1
    score = 10
    if outcome_candidate_refs and not cohort_refs.isdisjoint(outcome_candidate_refs):
        score += 30
    elif outcome_source_refs and not cohort_refs.isdisjoint(outcome_source_refs):
        score += 30
    requested, target = _safe_model_pair(cohort)
    cohort_values = {
        "source_surface": _string(cohort.get("source_surface")),
        "endpoint": _string(cohort.get("endpoint")),
        "category": _string(cohort.get("category")),
        "requested_model": requested,
        "target_model": target,
    }
    for key in ("source_surface", "endpoint", "category", "requested_model", "target_model"):
        outcome_value = _string(outcome.get(key))
        cohort_value = cohort_values[key]
        if outcome_value and cohort_value and outcome_value == cohort_value:
            score += 1
    return score


def _managed_preview_outcome_for_cohort(
    cohort: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
) -> dict[str, Any]:
    best: tuple[int, float, dict[str, Any]] | None = None
    for row in _managed_preview_outcomes(managed_preview_outcomes):
        score = _outcome_match_score(cohort, row)
        if score < 0:
            continue
        age = _as_float(row.get("preview_age_hours"))
        rank_key = (score, -age, row)
        if best is None or rank_key[:2] > best[:2]:
            best = rank_key
    return best[2] if best else {}


def _managed_preview_agreement(
    cohort: dict[str, Any],
    managed_preview_outcomes: dict[str, Any] | None,
    managed_preview_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_next_action = _local_recovery_next_action(cohort)
    normalized_local = _normalized_recovery_action(local_next_action)
    health_gate = _managed_preview_health_gate(
        managed_preview_health if isinstance(managed_preview_health, dict) else _embedded_managed_preview_health(cohort),
        managed_preview_outcomes,
    )
    selected = _managed_preview_outcome_for_cohort(cohort, managed_preview_outcomes)
    if not health_gate["passed"]:
        reason = str(health_gate["reason"])
        normalized_managed = "blocked"
        agreed = False
    elif not selected:
        reason = "missing-managed-preview-outcome"
        normalized_managed = "missing"
        agreed = False
    elif bool(selected.get("failed_closed")):
        reason = "managed-preview-failed-closed"
        normalized_managed = _normalized_recovery_action(selected.get("next_action"))
        agreed = False
    elif bool(selected.get("stale")) or _string(selected.get("classification")) == "stale-preview":
        reason = "stale-managed-preview-outcome"
        normalized_managed = _normalized_recovery_action(selected.get("next_action"))
        agreed = False
    elif bool(selected.get("missing_preview_decision")):
        reason = "missing-managed-preview-decision"
        normalized_managed = _normalized_recovery_action(selected.get("next_action"))
        agreed = False
    elif bool(selected.get("disagrees_with_local_evidence")) or _string(selected.get("classification")) == "managed-local-disagreement":
        reason = "managed-local-disagreement"
        normalized_managed = _normalized_recovery_action(selected.get("next_action"))
        agreed = False
    else:
        normalized_managed = _normalized_recovery_action(selected.get("next_action"))
        agreed = normalized_local == normalized_managed and normalized_local == "draft-recovery-canary"
        reason = "local-managed-preview-agree" if agreed else "managed-preview-action-disagreement"
    return {
        "schema": MANAGED_PREVIEW_AGREEMENT_SCHEMA,
        "required": True,
        "agreed": agreed,
        "reason": reason,
        "local_next_action": local_next_action,
        "managed_next_action": selected.get("next_action") if selected else None,
        "normalized_local_action": normalized_local,
        "normalized_managed_action": normalized_managed,
        "managed_classification": selected.get("classification") if selected else None,
        "managed_decision": selected.get("decision") if selected else None,
        "managed_preview_age_hours": selected.get("preview_age_hours") if selected else None,
        "handoff_ref": selected.get("handoff_ref") if selected else None,
        "preview_ref": selected.get("preview_ref") if selected else None,
        "health_gate": health_gate,
        "review_only": True,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _reason_codes(cohort: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("reason_codes", "blocker_codes", "blockers"):
        reasons.extend(_string_list(cohort.get(key)))
    lifecycle = cohort.get("openai_canary_lifecycle_evidence")
    if not isinstance(lifecycle, dict):
        lifecycle = cohort.get("lifecycle") if isinstance(cohort.get("lifecycle"), dict) else {}
    reasons.extend(_string_list(lifecycle.get("blocker_codes")))
    semantic = cohort.get("semantic_quality") if isinstance(cohort.get("semantic_quality"), dict) else {}
    reasons.extend(_string_list(semantic.get("reason_codes")))
    disabled = cohort.get("disabled_local_policy_rule") if isinstance(cohort.get("disabled_local_policy_rule"), dict) else {}
    if disabled.get("reason"):
        reasons.append(str(disabled["reason"]))
    return sorted(set(reasons))


def _cohort_from_pass_through_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    lifecycle = bucket.get("openai_canary_lifecycle_evidence") if isinstance(bucket.get("openai_canary_lifecycle_evidence"), dict) else {}
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    return {
        "source_schema": "tokenclaw.pass_through_routing_activation_candidates.v1",
        "rank": bucket.get("rank"),
        "provider": bucket.get("provider"),
        "source_surface": bucket.get("source_surface") or "openai_responses",
        "endpoint": bucket.get("endpoint") or "responses",
        "requested_model": bucket.get("requested_model"),
        "target_model": bucket.get("candidate_target_model") or bucket.get("target_model"),
        "category": bucket.get("category") or "unknown",
        "workflow_phase": bucket.get("workflow_phase"),
        "matched_count": bucket.get("sample_count"),
        "applied_count": counts.get("canary_applied"),
        "holdout_count": counts.get("canary_holdout"),
        "safety_stop_count": counts.get("safety_stopped"),
        "reason_codes": lifecycle.get("blocker_codes") or [],
        "estimated_savings_per_1000_calls_usd": bucket.get("estimated_savings_per_1000_calls_usd"),
        "projected_savings_usd": bucket.get("projected_savings_usd"),
        "openai_canary_lifecycle_evidence": lifecycle,
        "candidate_fingerprint": bucket.get("candidate_fingerprint"),
        "candidate_set": bucket.get("candidate_set") if isinstance(bucket.get("candidate_set"), dict) else {},
    }


def _cohort_from_promotion_decision(decision: dict[str, Any]) -> dict[str, Any]:
    target = decision.get("target") if isinstance(decision.get("target"), dict) else {}
    lifecycle = decision.get("lifecycle") if isinstance(decision.get("lifecycle"), dict) else {}
    candidate_set = decision.get("candidate_set") if isinstance(decision.get("candidate_set"), dict) else {}
    return {
        "source_schema": "tokenclaw.openai_routing_promotion_decision.v1",
        "provider": "openai",
        "source_surface": target.get("source_surface"),
        "endpoint": target.get("endpoint"),
        "requested_model": target.get("requested_model"),
        "target_model": target.get("target_model"),
        "category": target.get("category"),
        "matched_count": decision.get("matched_count"),
        "applied_count": lifecycle.get("applied_count"),
        "holdout_count": lifecycle.get("holdout_count"),
        "safety_stop_count": lifecycle.get("safety_stop_count"),
        "skipped_count": lifecycle.get("skipped_count"),
        "unknown_count": lifecycle.get("unknown_count"),
        "reason_codes": decision.get("reason_codes") or [],
        "estimated_savings_per_1000_calls_usd": decision.get("savings_per_1000_calls_usd"),
        "projected_savings_usd": decision.get("projected_savings_usd"),
        "semantic_quality": decision.get("semantic_quality") if isinstance(decision.get("semantic_quality"), dict) else {},
        "semantic_regression_action": decision.get("semantic_regression_action")
        if isinstance(decision.get("semantic_regression_action"), dict)
        else {},
        "routing_lifecycle_review": decision.get("routing_lifecycle_review")
        if isinstance(decision.get("routing_lifecycle_review"), dict)
        else {},
        "disabled_local_policy_rule": decision.get("disabled_local_policy_rule") if isinstance(decision.get("disabled_local_policy_rule"), dict) else {},
        "lifecycle": lifecycle,
        "candidate_fingerprint": decision.get("candidate_fingerprint") or candidate_set.get("candidate_fingerprint"),
        "candidate_set": candidate_set,
    }


def _cohort_from_pathway_outcome(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_schema": row.get("schema") or PATHWAY_OUTCOME_ROW_SCHEMA,
        "provider": row.get("provider") or "openai",
        "source_surface": row.get("source_surface"),
        "app_family": row.get("app_family"),
        "endpoint": row.get("endpoint"),
        "requested_model": row.get("requested_model"),
        "target_model": row.get("target_model") or row.get("candidate_target_model"),
        "requested_model_family": row.get("requested_model_family"),
        "target_model_family": row.get("target_model_family"),
        "category": row.get("category"),
        "workflow_phase": row.get("workflow_phase"),
        "matched_count": row.get("matched_count") or row.get("sample_count"),
        "applied_count": row.get("applied_count"),
        "holdout_count": row.get("holdout_count"),
        "safety_stop_count": row.get("safety_stop_count"),
        "skipped_count": row.get("skipped_count"),
        "unknown_count": row.get("unknown_count"),
        "error_count": row.get("error_count"),
        "fallback_count": row.get("fallback_count"),
        "retry_count": row.get("retry_count"),
        "reason_codes": _string_list(row.get("blocker_status")) + _string_list(row.get("reason_codes")),
        "estimated_savings_per_1000_calls_usd": row.get("savings_per_1000_calls_usd") or row.get("estimated_savings_per_1000_calls_usd"),
        "projected_savings_usd": row.get("projected_savings_usd"),
        "semantic_quality": row.get("semantic_quality") if isinstance(row.get("semantic_quality"), dict) else {},
        "status": row.get("status"),
        "blocker_status": row.get("blocker_status"),
        "recommended_next_action": row.get("recommended_next_action"),
        "candidate_suggested_next_action": row.get("candidate_suggested_next_action"),
        "candidate_fingerprint": row.get("candidate_fingerprint"),
        "coverage": row.get("coverage") if isinstance(row.get("coverage"), dict) else {},
    }


def _successor_actions(report: dict[str, Any]) -> list[dict[str, Any]]:
    queue = report.get("local_activation_next_action_queue")
    if not isinstance(queue, dict):
        return []
    rows = queue.get("successor_actions")
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and _string(row.get("local_action_family")).lower() == "routing"
        and _string(row.get("evidence_schema")) == "tokenclaw.openai_routing_promotion_decision_report.v1"
    ]


def _enrich_cohort_with_successor(cohort: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(cohort)
    reasons = set(_reason_codes(enriched))
    reasons.update(_string_list(action.get("blocker_codes")))
    if action.get("unblock_reason"):
        reasons.add(_string(action.get("unblock_reason")))
    enriched.update({
        "source_fingerprint": action.get("source_fingerprint"),
        "successor_action_fingerprint": action.get("fingerprint"),
        "successor_status": action.get("successor_status") or action.get("current_status"),
        "preview_verified": bool(action.get("preview_verified")),
        "preview_verification_status": action.get("preview_verification_status"),
        "preview_verification_decision": action.get("preview_verification_decision"),
        "managed_preview_gate": action.get("managed_preview_gate") if isinstance(action.get("managed_preview_gate"), dict) else {},
        "acceptance_metric": action.get("acceptance_metric"),
        "expected_savings_path": action.get("expected_savings_path"),
        "recommended_next_action": action.get("recommended_next_action"),
        "reason_codes": sorted(reason for reason in reasons if reason),
    })
    for key in ("applied_count", "holdout_count", "sample_count", "projected_savings_usd", "savings_per_1000_calls_usd"):
        if action.get(key) is not None:
            enriched[key] = action.get(key)
    return enriched


def _cohorts_from_stats_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    pathway = report.get("local_routing_pathway_outcome_feedback")
    if isinstance(pathway, dict):
        cohorts = _cohorts_from_report(pathway)
        if cohorts:
            return cohorts
    promotion_report = report.get("openai_routing_promotion_decision")
    cohorts: list[dict[str, Any]] = []
    if isinstance(promotion_report, dict):
        cohorts.extend(_cohorts_from_report(promotion_report))
    actions = _successor_actions(report)
    if actions and cohorts:
        return [_enrich_cohort_with_successor(cohort, actions[0]) for cohort in cohorts]
    if actions:
        return [_enrich_cohort_with_successor({}, action) for action in actions]
    return cohorts


def _cohorts_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("evidence"), dict) and isinstance(report["evidence"].get("stats_summary"), dict):
        return _cohorts_from_stats_summary(report["evidence"]["stats_summary"])
    if isinstance(report.get("stats_summary"), dict):
        return _cohorts_from_stats_summary(report["stats_summary"])
    if "openai_routing_promotion_decision" in report or "local_activation_next_action_queue" in report:
        return _cohorts_from_stats_summary(report)
    if isinstance(report.get("cohorts"), list):
        return [item for item in report["cohorts"] if isinstance(item, dict)]
    if report.get("schema") == PATHWAY_OUTCOME_SCHEMA and isinstance(report.get("outcomes"), list):
        return [_cohort_from_pathway_outcome(item) for item in report["outcomes"] if isinstance(item, dict)]
    if isinstance(report.get("decisions"), list):
        return [_cohort_from_promotion_decision(item) for item in report["decisions"] if isinstance(item, dict)]
    if isinstance(report.get("promotion_decision"), dict):
        return [_cohort_from_promotion_decision(report["promotion_decision"])]
    if report.get("schema") == "tokenclaw.pass_through_routing_activation_candidates.v1" and isinstance(report.get("buckets"), list):
        return [
            _cohort_from_pass_through_bucket(item)
            for item in report["buckets"]
            if isinstance(item, dict) and _string(item.get("provider")).lower() == "openai"
        ]
    if isinstance(report.get("buckets"), list):
        return [item for item in report["buckets"] if isinstance(item, dict)]
    if isinstance(report.get("candidates"), list):
        return [item for item in report["candidates"] if isinstance(item, dict)]
    return []


def _fingerprint(cohort: dict[str, Any]) -> str:
    requested, target = _safe_model_pair(cohort)
    prefix = _policy_target(cohort)["fingerprint_prefix"]
    return _stable_id(
        prefix,
        cohort.get("source_surface"),
        cohort.get("app_family"),
        cohort.get("endpoint"),
        requested,
        target,
        cohort.get("category"),
        cohort.get("workflow_phase"),
    )


def _omission(cohort: dict[str, Any], reason: str, *, path: str | None = None) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    requested, target = _safe_model_pair(cohort)
    policy_target = _policy_target(cohort)
    return {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "path": path,
        "fingerprint": _fingerprint(cohort),
        "provider": "openai",
        "source_surface": cohort.get("source_surface"),
        "app_family": cohort.get("app_family"),
        "endpoint": cohort.get("endpoint"),
        "requested_model": requested,
        "target_model": target,
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "matched_count": counts["matched"],
        "applied_count": counts["applied"],
        "holdout_count": counts["holdout"],
        "reason_codes": _reason_codes(cohort),
        "coverage": _coverage(cohort),
        "stale_evidence": _stale_evidence(cohort),
        "rollback_no_write": _rollback_no_write("keep-blocked", cohort),
        "recovery_plan": _recovery_plan(cohort, omission_reason=reason),
        "recovery_sizing": _recovery_sizing(None, None, reason=reason),
        "semantic_regression_recovery": _semantic_recovery_classification(cohort),
        "target_local_policy_section": policy_target["policy_section"],
        "target_local_rule_file": policy_target["rule_file"],
        "privacy": PRIVACY,
    }


def _recovery_sizing(
    canary_fraction: float | None,
    holdout_fraction: float | None,
    *,
    reason: str,
) -> dict[str, Any]:
    available = canary_fraction is not None and holdout_fraction is not None
    return {
        "schema": RECOVERY_SIZING_SCHEMA,
        "status": "available" if available else "not-available",
        "reason": "review-only-sizing-available" if available else reason,
        "canary_fraction": _bounded_fraction(canary_fraction, 0.05) if available else None,
        "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10) if available else None,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _omission_reason(cohort: dict[str, Any]) -> str | None:
    provider = _string(cohort.get("provider") or "openai").lower()
    source_surface = _string(cohort.get("source_surface")).lower()
    requested, target = _safe_model_pair(cohort)
    reasons = set(_reason_codes(cohort))
    counts = _cohort_counts(cohort)
    if provider and provider != "openai":
        return "unsupported-provider"
    if source_surface and source_surface not in {
        "openai",
        "openai_responses",
        "openai_chat",
        "openai_chat_completions",
        "codex_turn",
        "codex_app_turn",
        "unknown",
    }:
        return "unsupported-source-surface"
    if not requested or not target:
        return "missing-model-pair"
    if counts["safety_stop"] > 0 or "safety-stop-observed" in reasons:
        return "safety-stop-observed"
    if counts["applied"] <= 0:
        return "missing-applied-coverage"
    if counts["holdout"] <= 0:
        return "missing-holdout-coverage"
    if _stale_evidence(cohort).get("stale"):
        return "stale-openai-routing-evidence"
    if target.lower() == requested.lower():
        return "not-a-downgrade"
    semantic = _semantic_recovery_classification(cohort)
    pathway_ready = (
        _string(cohort.get("status")).lower() in {"ready", "review-only", "preview-agreed"}
        or _string(cohort.get("blocker_status")).lower() == "applied-and-holdout-coverage-present"
        or _normalized_recovery_action(cohort.get("recommended_next_action")) == "draft-recovery-canary"
    )
    if pathway_ready and not semantic["observed"]:
        return None
    if not semantic["observed"]:
        return "not-semantic-regression-row"
    if semantic["classification"] == "narrower-canary":
        return None
    if semantic["classification"] == "rollback":
        return "semantic-regression-rollback-required"
    return "semantic-quality-regression-observed"


def _draft_for_cohort(
    cohort: dict[str, Any],
    *,
    canary_fraction: float,
    holdout_fraction: float,
    managed_preview_agreement: dict[str, Any],
) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    requested, target = _safe_model_pair(cohort)
    category = _string(cohort.get("category") or "unknown")
    endpoint = _string(cohort.get("endpoint") or "responses")
    source_surface = _string(cohort.get("source_surface") or "openai_responses")
    fingerprint = _fingerprint(cohort)
    allow_tools = category == "tool-light"
    policy_target = _policy_target(cohort)
    policy_id = fingerprint.replace(f"{policy_target['fingerprint_prefix']}:", policy_target["policy_id_prefix"])
    bounded_canary = _bounded_fraction(canary_fraction, 0.05)
    bounded_holdout = _bounded_fraction(holdout_fraction, 0.10)
    generic_canary = {
        "enabled": False,
        "review_only": True,
        "policy_id": policy_id,
        "policy_source": "local-manual",
        "model_pattern": requested,
        "target_model": target,
        "eligible_categories": [category],
        "allow_tools": allow_tools,
        "allow_stream": False,
        "canary_fraction": bounded_canary,
        "holdout_fraction": bounded_holdout,
        "cohort_unit": "session" if not _is_codex_surface(cohort) else "turn",
        "salt": _stable_id("routing-narrow-canary-salt", fingerprint),
    }
    draft = {
        "schema": DRAFT_SCHEMA,
        "status": "review-only",
        "review_only": True,
        "active_policy_changed": False,
        "policy_files_written": False,
        "fingerprint": fingerprint,
        "policy_id": policy_id,
        "target_local_policy_section": policy_target["policy_section"],
        "target_local_rule_file": policy_target["rule_file"],
        "provider": "openai",
        "source_surface": source_surface,
        "app_family": cohort.get("app_family"),
        "endpoint": endpoint,
        "requested_model": requested,
        "target_model": target,
        "category": category,
        "workflow_phase": cohort.get("workflow_phase"),
        "matched_count": counts["matched"],
        "applied_count": counts["applied"],
        "holdout_count": counts["holdout"],
        "canary_fraction": bounded_canary,
        "holdout_fraction": bounded_holdout,
        "estimated_savings_per_1000_calls_usd": round(_as_float(cohort.get("estimated_savings_per_1000_calls_usd") or cohort.get("savings_per_1000_calls_usd")), 6),
        "projected_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 6),
        "proposed_rule_conditions": {
            "provider": "openai",
            "source_surface": source_surface,
            "app_family": cohort.get("app_family"),
            "endpoint": endpoint,
            "model_pattern": requested,
            "category": category,
            "has_tools": allow_tools,
            "stream": False,
        },
        "proposed_routing_canary": generic_canary,
        "rollback_condition": {
            "schema": "tokenclaw.openai_routing_narrow_canary_rollback_condition.v1",
            "rollback_action_type": policy_target["rollback_action_type"],
            "rollback_canary_fraction": 0.0,
            "rollback_holdout_fraction": 0.0,
            "reason_codes": [
                "semantic-quality-regression-observed",
                "safety-stop-observed",
                "error-rate-regression",
                "retry-or-fallback-regression",
                "operator-requested",
            ],
        },
        "coverage": _coverage(cohort),
        "stale_evidence": _stale_evidence(cohort),
        "rollback_no_write": _rollback_no_write("restage-review-only", cohort),
        "recovery_plan": _recovery_plan(cohort, omission_reason=None),
        "recovery_sizing": _recovery_sizing(bounded_canary, bounded_holdout, reason="review-only-sizing-available"),
        "semantic_regression_recovery": _semantic_recovery_classification(cohort),
        "managed_preview_agreement": managed_preview_agreement,
        "privacy": PRIVACY,
    }
    if _is_codex_surface(cohort):
        draft["proposed_codex_app_canary"] = {
            **generic_canary,
            "section": "summary_model_hint",
            "target_local_policy_section": policy_target["policy_section"],
            "target_local_rule_file": policy_target["rule_file"],
            "source_surface": source_surface,
            "app_family": cohort.get("app_family") or "codex",
        }
    else:
        draft["proposed_openai_canary"] = generic_canary
    return draft


def _normalized_review_report(report: dict[str, Any]) -> dict[str, Any]:
    def relevant_stats(stats: dict[str, Any], source_schema: Any) -> dict[str, Any]:
        return {
            "schema": "tokenclaw.openai_routing_narrow_canary_review_input.v1",
            "source_report_schema": source_schema,
            "openai_routing_promotion_decision": stats.get("openai_routing_promotion_decision")
            if isinstance(stats.get("openai_routing_promotion_decision"), dict)
            else {},
            "local_routing_pathway_outcome_feedback": stats.get("local_routing_pathway_outcome_feedback")
            if isinstance(stats.get("local_routing_pathway_outcome_feedback"), dict)
            else {},
            "local_activation_next_action_queue": stats.get("local_activation_next_action_queue")
            if isinstance(stats.get("local_activation_next_action_queue"), dict)
            else {},
            "privacy": stats.get("privacy") if isinstance(stats.get("privacy"), dict) else {},
        }

    evidence = report.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("stats_summary"), dict):
        return relevant_stats(evidence["stats_summary"], report.get("schema"))
    if isinstance(report.get("stats_summary"), dict):
        return relevant_stats(report["stats_summary"], report.get("schema"))
    if "openai_routing_promotion_decision" in report or "local_activation_next_action_queue" in report:
        return relevant_stats(report, report.get("schema"))
    return report


def build_openai_routing_narrow_canary_review(
    report: dict[str, Any],
    *,
    managed_preview_outcomes: dict[str, Any] | None = None,
    managed_preview_health: dict[str, Any] | None = None,
    canary_fraction: float = 0.05,
    holdout_fraction: float = 0.10,
    top_candidates: int = 1,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return _error_result("invalid_report", "OpenAI routing narrow-canary review requires a JSON report object")
    review_report = _normalized_review_report(report)
    raw_errors = _privacy_errors(review_report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "OpenAI routing narrow-canary review received raw request content or identifiers",
            errors=raw_errors,
        )

    cohorts = _cohorts_from_report(review_report)
    omitted: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    managed_agreements: list[dict[str, Any]] = []
    for index, cohort in enumerate(cohorts):
        reason = _omission_reason(cohort)
        if reason is not None:
            omitted_item = _omission(cohort, reason, path=f"$.cohorts[{index}]")
            if reason == "semantic-quality-regression-observed" and (
                isinstance(managed_preview_health, dict)
                or isinstance(managed_preview_outcomes, dict)
                or _embedded_managed_preview_health(cohort) is not None
            ):
                agreement = _managed_preview_agreement(cohort, managed_preview_outcomes, managed_preview_health)
                managed_agreements.append(agreement)
                omitted_item["managed_preview_agreement"] = agreement
                health_gate = agreement.get("health_gate") if isinstance(agreement.get("health_gate"), dict) else {}
                if health_gate.get("reason"):
                    omitted_item["recovery_sizing"] = _recovery_sizing(None, None, reason=str(health_gate["reason"]))
            omitted.append(omitted_item)
            continue
        agreement = _managed_preview_agreement(cohort, managed_preview_outcomes, managed_preview_health)
        managed_agreements.append(agreement)
        if not agreement["agreed"]:
            omitted_item = _omission(cohort, str(agreement["reason"]), path=f"$.cohorts[{index}]")
            omitted_item["managed_preview_agreement"] = agreement
            omitted.append(omitted_item)
            continue
        cohort = dict(cohort)
        cohort["managed_preview_agreement"] = agreement
        eligible.append(cohort)

    eligible.sort(
        key=lambda item: (
            _as_float(item.get("estimated_savings_per_1000_calls_usd") or item.get("savings_per_1000_calls_usd")),
            _cohort_counts(item)["matched"],
            _cohort_counts(item)["applied"] + _cohort_counts(item)["holdout"],
        ),
        reverse=True,
    )
    limit = max(1, _as_int(top_candidates, 1))
    drafts = [
        _draft_for_cohort(
            cohort,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
            managed_preview_agreement=cohort.get("managed_preview_agreement")
            if isinstance(cohort.get("managed_preview_agreement"), dict)
            else _managed_preview_agreement(cohort, managed_preview_outcomes, managed_preview_health),
        )
        for cohort in eligible[:limit]
    ]
    for cohort in eligible[limit:]:
        omitted.append(_omission(cohort, "lower-ranked-clean-cohort-not-drafted"))

    reason_counts: dict[str, int] = {}
    for item in omitted:
        reason = str(item.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    omission_breakdown = [
        {"value": key, "count": value}
        for key, value in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    semantic_regression_row_count = sum(1 for item in cohorts if _semantic_recovery_classification(item)["observed"])
    regressed = [item for item in omitted if item.get("reason") == "semantic-quality-regression-observed"]
    preview_health_gates = [
        agreement.get("health_gate")
        for agreement in managed_agreements
        if isinstance(agreement.get("health_gate"), dict)
    ]
    top_preview_health_gate = preview_health_gates[0] if preview_health_gates else {}
    recovery_plan = drafts[0].get("recovery_plan") if drafts else None
    if recovery_plan is None and regressed:
        recovery_plan = regressed[0].get("recovery_plan")
    if recovery_plan is None and omitted:
        recovery_plan = omitted[0].get("recovery_plan")
    decision = "draft-narrower-canary" if drafts else "keep-blocked"
    if not drafts and regressed and len(regressed) == len(omitted):
        reason = "semantic-quality-regression-observed"
    elif not drafts:
        reason = "no-clean-covered-openai-routing-cohort"
    else:
        reason = "clean-openai-routing-cohort-found"

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "source_report_schema": review_report.get("source_report_schema") or review_report.get("schema"),
        "decision": decision,
        "status": "review-only" if drafts else "keep-blocked",
        "next_action": "operator-review-narrower-openai-routing-canary" if drafts else "keep-openai-routing-blocked",
        "reason": reason,
        "reason_codes": [reason],
        "summary": {
            "cohort_count": len(cohorts),
            "eligible_clean_cohort_count": len(eligible),
            "draft_count": len(drafts),
            "omitted_count": len(omitted),
            "regressed_cohort_count": semantic_regression_row_count,
            "blocked_regressed_cohort_count": len(regressed),
            "managed_preview_agreement_count": sum(1 for item in managed_agreements if item.get("agreed")),
            "managed_preview_disagreement_count": sum(1 for item in managed_agreements if not item.get("agreed")),
            "managed_preview_health_status": top_preview_health_gate.get("status"),
            "managed_preview_health_reason": top_preview_health_gate.get("reason"),
            "managed_preview_required": True,
            "canary_fraction": _bounded_fraction(canary_fraction, 0.05),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "top_draft_fingerprint": drafts[0]["fingerprint"] if drafts else None,
            "recovery_selected_option": recovery_plan.get("selected_option") if isinstance(recovery_plan, dict) else None,
            "recovery_blocker_status": recovery_plan.get("blocker_status") if isinstance(recovery_plan, dict) else None,
            "omission_reason_breakdown": omission_breakdown,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "recovery_plan": recovery_plan,
        "managed_preview_agreements": managed_agreements,
        "drafts": drafts,
        "regressed_cohorts": regressed,
        "omitted": omitted,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
    }


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "decision": "error",
        "status": "error",
        "next_action": "fix-input-report",
        "reason": error_type,
        "reason_codes": [error_type],
        "summary": {
            "cohort_count": 0,
            "eligible_clean_cohort_count": 0,
            "draft_count": 0,
            "omitted_count": 0,
            "regressed_cohort_count": 0,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "drafts": [],
        "regressed_cohorts": [],
        "omitted": [],
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }
