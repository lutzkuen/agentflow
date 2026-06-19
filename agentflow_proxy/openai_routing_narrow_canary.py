from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_routing_narrow_canary_review.v1"
DRAFT_SCHEMA = "agentflow.openai_routing_narrow_canary_draft.v1"
OMISSION_SCHEMA = "agentflow.openai_routing_narrow_canary_omission.v1"
PRIVACY = {
    "local_only": True,
    "metadata_only": True,
    "aggregate_only": True,
    "raw_prompts_included": False,
    "raw_messages_included": False,
    "provider_bodies_included": False,
    "raw_provider_bodies_included": False,
    "request_ids_included": False,
    "raw_request_ids_included": False,
    "session_ids_included": False,
    "raw_session_ids_included": False,
    "cache_keys_included": False,
    "file_paths_included": False,
    "secrets_included": False,
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
        "source_schema": "agentflow.pass_through_routing_activation_candidates.v1",
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
    }


def _cohort_from_promotion_decision(decision: dict[str, Any]) -> dict[str, Any]:
    target = decision.get("target") if isinstance(decision.get("target"), dict) else {}
    lifecycle = decision.get("lifecycle") if isinstance(decision.get("lifecycle"), dict) else {}
    return {
        "source_schema": "agentflow.openai_routing_promotion_decision.v1",
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
        "disabled_local_policy_rule": decision.get("disabled_local_policy_rule") if isinstance(decision.get("disabled_local_policy_rule"), dict) else {},
        "lifecycle": lifecycle,
    }


def _cohorts_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("cohorts"), list):
        return [item for item in report["cohorts"] if isinstance(item, dict)]
    if isinstance(report.get("decisions"), list):
        return [_cohort_from_promotion_decision(item) for item in report["decisions"] if isinstance(item, dict)]
    if isinstance(report.get("promotion_decision"), dict):
        return [_cohort_from_promotion_decision(report["promotion_decision"])]
    if report.get("schema") == "agentflow.pass_through_routing_activation_candidates.v1" and isinstance(report.get("buckets"), list):
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
    return _stable_id(
        "openai-routing-narrow-canary",
        cohort.get("source_surface"),
        cohort.get("endpoint"),
        requested,
        target,
        cohort.get("category"),
        cohort.get("workflow_phase"),
    )


def _omission(cohort: dict[str, Any], reason: str, *, path: str | None = None) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    requested, target = _safe_model_pair(cohort)
    return {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "path": path,
        "fingerprint": _fingerprint(cohort),
        "provider": "openai",
        "source_surface": cohort.get("source_surface"),
        "endpoint": cohort.get("endpoint"),
        "requested_model": requested,
        "target_model": target,
        "category": cohort.get("category"),
        "workflow_phase": cohort.get("workflow_phase"),
        "matched_count": counts["matched"],
        "applied_count": counts["applied"],
        "holdout_count": counts["holdout"],
        "reason_codes": _reason_codes(cohort),
        "privacy": PRIVACY,
    }


def _omission_reason(cohort: dict[str, Any]) -> str | None:
    provider = _string(cohort.get("provider") or "openai").lower()
    source_surface = _string(cohort.get("source_surface")).lower()
    requested, target = _safe_model_pair(cohort)
    reasons = set(_reason_codes(cohort))
    counts = _cohort_counts(cohort)
    if provider and provider != "openai":
        return "unsupported-provider"
    if source_surface and source_surface not in {"openai", "openai_responses", "openai_chat", "openai_chat_completions", "unknown"}:
        return "unsupported-source-surface"
    if not requested or not target:
        return "missing-model-pair"
    if "semantic-quality-regression-observed" in reasons:
        return "semantic-quality-regression-observed"
    if counts["safety_stop"] > 0 or "safety-stop-observed" in reasons:
        return "safety-stop-observed"
    if counts["applied"] <= 0:
        return "missing-applied-coverage"
    if counts["holdout"] <= 0:
        return "missing-holdout-coverage"
    if target.lower() == requested.lower():
        return "not-a-downgrade"
    return None


def _draft_for_cohort(cohort: dict[str, Any], *, canary_fraction: float, holdout_fraction: float) -> dict[str, Any]:
    counts = _cohort_counts(cohort)
    requested, target = _safe_model_pair(cohort)
    category = _string(cohort.get("category") or "unknown")
    endpoint = _string(cohort.get("endpoint") or "responses")
    source_surface = _string(cohort.get("source_surface") or "openai_responses")
    fingerprint = _fingerprint(cohort)
    allow_tools = category == "tool-light"
    policy_id = fingerprint.replace("openai-routing-narrow-canary:", "local-openai-routing-narrow-canary-")
    bounded_canary = _bounded_fraction(canary_fraction, 0.05)
    bounded_holdout = _bounded_fraction(holdout_fraction, 0.10)
    return {
        "schema": DRAFT_SCHEMA,
        "status": "review-only",
        "review_only": True,
        "active_policy_changed": False,
        "policy_files_written": False,
        "fingerprint": fingerprint,
        "policy_id": policy_id,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "provider": "openai",
        "source_surface": source_surface,
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
            "endpoint": endpoint,
            "model_pattern": requested,
            "category": category,
            "has_tools": allow_tools,
            "stream": False,
        },
        "proposed_openai_canary": {
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
            "cohort_unit": "session",
            "salt": _stable_id("openai-routing-narrow-canary-salt", fingerprint),
        },
        "rollback_condition": {
            "schema": "agentflow.openai_routing_narrow_canary_rollback_condition.v1",
            "rollback_action_type": "disable_openai_routing_narrow_canary",
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
        "privacy": PRIVACY,
    }


def build_openai_routing_narrow_canary_review(
    report: dict[str, Any],
    *,
    canary_fraction: float = 0.05,
    holdout_fraction: float = 0.10,
    top_candidates: int = 1,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return _error_result("invalid_report", "OpenAI routing narrow-canary review requires a JSON report object")
    raw_errors = _privacy_errors(report)
    if raw_errors:
        return _error_result(
            "raw_payload_rejected",
            "OpenAI routing narrow-canary review received raw request content or identifiers",
            errors=raw_errors,
        )

    cohorts = _cohorts_from_report(report)
    omitted: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for index, cohort in enumerate(cohorts):
        reason = _omission_reason(cohort)
        if reason is not None:
            omitted.append(_omission(cohort, reason, path=f"$.cohorts[{index}]"))
            continue
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
        _draft_for_cohort(cohort, canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
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
    regressed = [item for item in omitted if item.get("reason") == "semantic-quality-regression-observed"]
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
        "source_report_schema": report.get("schema"),
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
            "regressed_cohort_count": len(regressed),
            "canary_fraction": _bounded_fraction(canary_fraction, 0.05),
            "holdout_fraction": _bounded_fraction(holdout_fraction, 0.10),
            "top_draft_fingerprint": drafts[0]["fingerprint"] if drafts else None,
            "omission_reason_breakdown": omission_breakdown,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
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
