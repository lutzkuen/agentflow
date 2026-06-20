from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "agentflow.anthropic_routing_canary_stage.v1"
OMISSION_SCHEMA = "agentflow.anthropic_routing_canary_stage_omission.v1"
STAGED_SCHEMA = "agentflow.anthropic_routing_canary_staged_draft.v1"
PROJECTED_LIFECYCLE_SCHEMA = "agentflow.anthropic_routing_canary_projected_lifecycle_coverage.v1"
BLOCKED_REVIEW_SCHEMA = "agentflow.anthropic_routing_canary_blocked_review.v1"
ACCEPTANCE_SCHEMA = "agentflow.anthropic_routing_canary_stage_acceptance.v1"
EXECUTOR_GUARD_DRY_RUN_SCHEMA = "agentflow.anthropic_routing_executor_guard_dry_run.v1"
PASS_THROUGH_SCHEMA = "agentflow.pass_through_routing_activation_candidates.v1"
SUPPORTED_EXECUTORS = {"anthropic-routing-rules", "claude-phase-routing-canary", "phase-canary"}

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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_id(prefix: str, *items: Any) -> str:
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")[:72] or "anthropic-routing-canary"


def _string(value: Any) -> str:
    return str(value or "").strip()


def _is_unknown(value: Any) -> bool:
    return _string(value).lower() in {"", "unknown", "none", "null"}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    return round(max(0.0, min(1.0, _as_float(value, default))), 4)


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
                if lowered.endswith("_included") and isinstance(child, bool):
                    continue
                if lowered in RAW_KEYS or lowered.startswith("raw_"):
                    errors.append({
                        "path": child_path,
                        "message": "Anthropic canary staging accepts aggregate metadata only, not raw prompts, provider bodies, request identifiers, file paths, cache keys, or secrets",
                    })
                    continue
                walk(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{item_path}[{index}]")

    walk(value, path)
    return errors


def _candidate_id(candidate: dict[str, Any]) -> str:
    explicit = _string(candidate.get("candidate_id") or candidate.get("target_candidate_id"))
    if explicit:
        safe = "".join(char for char in explicit if char.isalnum() or char in {"-", "_", ":"}).strip("-_:")
        if safe:
            return safe[:120]
    return _stable_id(
        "anthropic-route-candidate",
        candidate.get("rank"),
        candidate.get("provider"),
        candidate.get("source_surface"),
        candidate.get("endpoint"),
        candidate.get("requested_model"),
        candidate.get("candidate_target_model") or candidate.get("target_model"),
        candidate.get("category"),
    )


def _metadata_value(*values: Any) -> str:
    for value in values:
        text = _string(value)
        if text and not _is_unknown(text):
            return text
    return ""


def _candidate_from_pass_through_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    provider = _string(bucket.get("provider")).lower()
    required_executor = _string(bucket.get("required_local_executor") or "anthropic-routing-rules")
    category = _metadata_value(bucket.get("category"), "tool-result")
    source_surface = _metadata_value(bucket.get("source_surface"), bucket.get("surface"), "anthropic_messages")
    endpoint = _metadata_value(bucket.get("endpoint"), "messages")
    requested = _metadata_value(bucket.get("requested_model"))
    target = _metadata_value(bucket.get("candidate_target_model"), bucket.get("target_model"))
    sample_count = _as_int(bucket.get("sample_count") or bucket.get("count"))
    savings_per_1000 = _as_float(bucket.get("estimated_savings_per_1000_calls_usd"))
    actionability = _string(bucket.get("actionability"))
    blockers: list[str] = []
    blockers.extend(_string_list(bucket.get("blocker_codes")))
    blockers.extend(_string_list(bucket.get("blockers")))
    lifecycle = bucket.get("anthropic_canary_lifecycle_evidence")
    if isinstance(lifecycle, dict):
        lifecycle_blockers = _lifecycle_blockers(lifecycle)
        blockers.extend(lifecycle_blockers)
    if actionability and actionability != "actionable":
        blockers.append(actionability)
    if provider != "anthropic":
        blockers.append("unsupported-provider")
    if required_executor not in {"", *SUPPORTED_EXECUTORS}:
        blockers.append("unsupported-local-executor")
    if category != "tool-result":
        blockers.append("category-not-enabled")
    if "sonnet" not in requested.lower() or "haiku" not in target.lower():
        blockers.append("not-sonnet-to-haiku")
    if bucket.get("no_op_reason"):
        reason = _string(bucket.get("no_op_reason")).lower().replace(" ", "-")
        if "thinking" in reason:
            blockers.append("thinking-routing-guard")
        elif "stream" in reason:
            blockers.append("unsupported-streaming-shape")
        elif "routed-model" in reason and "missing" in reason:
            blockers.append("missing-routed-model")
        elif "routed-model-metadata" in reason:
            blockers.append("missing-routed-model")
    if bucket.get("routed_model") in (None, "") and _string(bucket.get("no_op_reason")):
        blockers.append("missing-routed-model")
    candidate = {
        "candidate_id": _candidate_id(bucket),
        "provider": "anthropic",
        "source_surface": source_surface,
        "endpoint": endpoint,
        "requested_model": requested,
        "target_model": target,
        "category": category,
        "workflow_phase": "tool-execution" if category == "tool-result" else "unknown",
        "workflow_phase_confidence": "high" if category == "tool-result" else "low",
        "matched_count": sample_count,
        "estimated_savings_per_1000_calls_usd": savings_per_1000,
        "projected_savings_usd": round((savings_per_1000 * sample_count) / 1000.0, 6),
        "actionability": actionability,
        "expected_local_executor": required_executor,
        "stream": True,
        "blockers": sorted(set(blockers)),
        "next_action": _metadata_value(bucket.get("next_action")),
        "aggregate_inference": {
            "source": "pass_through_routing_report",
            "input_source_surface_label": bucket.get("source_surface"),
            "input_endpoint_label": bucket.get("endpoint"),
            "source_surface_inferred": _is_unknown(bucket.get("source_surface")),
            "endpoint_inferred": _is_unknown(bucket.get("endpoint")),
            "metadata_only": True,
        },
    }
    if isinstance(lifecycle, dict):
        candidate["anthropic_canary_lifecycle_evidence"] = lifecycle
    return candidate


def _candidate_list_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema") == PASS_THROUGH_SCHEMA and isinstance(report.get("buckets"), list):
        return [
            _candidate_from_pass_through_bucket(bucket)
            for bucket in report.get("buckets") or []
            if isinstance(bucket, dict) and _string(bucket.get("provider")).lower() == "anthropic"
        ]
    candidates = report.get("candidates")
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    return []


def _safety_blockers(candidate: dict[str, Any]) -> list[str]:
    blockers = set(_string_list(candidate.get("blockers")) + _string_list(candidate.get("blocker_codes")))
    safety = {
        "thinking-routing-guard",
        "thinking-history-blocked",
        "top-level-thinking-blocked",
        "unsupported-streaming-shape",
        "unsafe-tool-call-context",
        "tool-call-cache-disabled",
        "needs-lifecycle-evidence",
        "safety-stop-observed",
    }
    return sorted(blockers & safety)


def _lifecycle_blockers(lifecycle: dict[str, Any]) -> list[str]:
    blockers = set(_string_list(lifecycle.get("blocker_codes")))
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    coverage = lifecycle.get("coverage") if isinstance(lifecycle.get("coverage"), dict) else {}
    stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
    if _as_int(counts.get("canary_applied")) <= 0 or _as_float(coverage.get("applied_rate")) <= 0:
        blockers.add("missing-applied-coverage")
    if _as_int(counts.get("canary_holdout")) <= 0 or _as_float(coverage.get("holdout_rate")) <= 0:
        blockers.add("missing-holdout-coverage")
    if _as_int(counts.get("safety_stopped")) > 0:
        blockers.add("safety-stop-observed")
    if bool(stale.get("stale")):
        blockers.add("stale-lifecycle-evidence")
    return sorted(blockers)


def _existing_lifecycle(candidate: dict[str, Any]) -> dict[str, Any] | None:
    lifecycle = candidate.get("anthropic_canary_lifecycle_evidence")
    return lifecycle if isinstance(lifecycle, dict) else None


def _local_file_backed_representation() -> dict[str, Any]:
    return {
        "exists": True,
        "policy_section": "routing",
        "policy_source": "local-file-backed",
        "reason": "file-backed-local-policy",
        "rule_file": "routing_rules.yaml",
    }


def _first_safety_breakdown_value(lifecycle: dict[str, Any], *keys: str) -> Any:
    breakdown = lifecycle.get("safety_stop_breakdown")
    if not isinstance(breakdown, list):
        return None
    for row in breakdown:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
    return None


def _criterion_result(passed: bool, passed_code: str, failed_code: str) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "status": "passed" if passed else "failed",
        "reason_codes": [passed_code if passed else failed_code],
    }


def _executor_guard_dry_run(candidate: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    coverage = lifecycle.get("coverage") if isinstance(lifecycle.get("coverage"), dict) else {}
    stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
    blockers = set(_string_list(lifecycle.get("blocker_codes")))
    safety_stop_count = _as_int(counts.get("safety_stopped"))
    applied_count = _as_int(counts.get("canary_applied"))
    holdout_count = _as_int(counts.get("canary_holdout"))
    required_executor = _metadata_value(
        candidate.get("expected_local_executor"),
        candidate.get("required_local_executor"),
        _first_safety_breakdown_value(lifecycle, "expected_local_executor", "required_local_executor"),
        "anthropic-routing-rules",
    )
    executor_compatible = required_executor in SUPPORTED_EXECUTORS
    breakdown = lifecycle.get("safety_stop_breakdown") if isinstance(lifecycle.get("safety_stop_breakdown"), list) else []
    if breakdown:
        executor_compatible = executor_compatible or any(bool(row.get("executor_compatible")) for row in breakdown if isinstance(row, dict))

    missing_applied = applied_count <= 0 or "missing-applied-coverage" in blockers
    missing_holdout = holdout_count <= 0 or "missing-holdout-coverage" in blockers
    stale_evidence = bool(stale.get("stale"))
    safety_stop_clear = safety_stop_count <= 0 and "safety-stop-observed" not in blockers
    applied_present = applied_count > 0 and not missing_applied and not stale_evidence
    holdout_present = holdout_count > 0 and not missing_holdout and not stale_evidence
    safer_guard_present = safety_stop_clear and executor_compatible and not stale_evidence
    rollback_proof_present = safety_stop_clear and not stale_evidence

    criteria = {
        "safety_stop_reason_review": _criterion_result(
            safety_stop_clear,
            "safety-stop-count-zero",
            "safety-stop-observed",
        ),
        "safer_threshold_or_executor_guard": _criterion_result(
            safer_guard_present,
            "executor-guard-present",
            "safer-threshold-or-executor-guard-missing",
        ),
        "rollback_proof": _criterion_result(
            rollback_proof_present,
            "rollback-proof-present",
            "rollback-proof-missing",
        ),
        "applied_coverage": _criterion_result(
            applied_present,
            "applied-coverage-present",
            "missing-applied-coverage",
        ),
        "holdout_coverage": _criterion_result(
            holdout_present,
            "holdout-coverage-present",
            "missing-holdout-coverage",
        ),
    }
    failed = [field for field, result in criteria.items() if not result["passed"]]
    reason_codes = sorted({code for field in failed for code in criteria[field]["reason_codes"]})
    guard_ready = not failed
    return {
        "schema": EXECUTOR_GUARD_DRY_RUN_SCHEMA,
        "status": "guard-ready" if guard_ready else "keep-blocked",
        "decision": "guard-ready" if guard_ready else "keep-blocked",
        "target_candidate_id": _candidate_id(candidate),
        "provider": "anthropic",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase") or ("tool-execution" if candidate.get("category") == "tool-result" else "unknown"),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
        "required_local_executor": required_executor,
        "executor_compatible": executor_compatible,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "matched_count": _as_int(lifecycle.get("matched_count")),
        "observed_count": _as_int(lifecycle.get("observed_count")),
        "safety_stop_count": safety_stop_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "coverage": {
            "matched_count": _as_int(coverage.get("matched_count") or lifecycle.get("matched_count")),
            "applied_rate": _as_float(coverage.get("applied_rate")),
            "holdout_rate": _as_float(coverage.get("holdout_rate")),
            "observed_rate": _as_float(coverage.get("observed_rate")),
        },
        "criterion_results": criteria,
        "needed_resolution": failed,
        "reason_codes": reason_codes,
        "next_action": "operator-review-anthropic-routing-guard-ready" if guard_ready else "keep-anthropic-routing-blocked-until-safety-stop-burndown",
        "stage_allowed": guard_ready,
        "promotion_allowed": False,
        "guard_ready": guard_ready,
        "dry_run_only": True,
        "active_policy_changed": False,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "local_file_backed_representation": _local_file_backed_representation(),
        "rollback_action": "keep-routing-policy-disabled" if not guard_ready else "keep-disabled-until-operator-stages-reviewed-canary",
        "privacy": PRIVACY,
    }


def _next_action_for_blockers(candidate: dict[str, Any], blockers: list[str]) -> str:
    explicit = _string(candidate.get("next_action"))
    if explicit:
        return explicit
    blocker_set = set(blockers)
    if "safety-stop-observed" in blocker_set:
        return "review-anthropic-routing-safety-stop-before-canary"
    if {"missing-applied-coverage", "missing-holdout-coverage"} & blocker_set:
        return "collect-anthropic-routing-applied-holdout-coverage"
    if "stale-lifecycle-evidence" in blocker_set:
        return "refresh-anthropic-routing-lifecycle-evidence"
    return "stage-anthropic-routing-canary-review"


def _blocked_needed_resolution(candidate: dict[str, Any], blockers: list[str], lifecycle: dict[str, Any]) -> list[str]:
    needed: list[str] = []
    blocker_set = set(blockers)
    aggregate_inference = candidate.get("aggregate_inference") if isinstance(candidate.get("aggregate_inference"), dict) else {}
    if aggregate_inference.get("source_surface_inferred") or aggregate_inference.get("endpoint_inferred"):
        needed.append("narrower_cohort")
    if "missing-applied-coverage" in blocker_set:
        needed.append("applied_coverage")
    if "missing-holdout-coverage" in blocker_set:
        needed.append("holdout_coverage")
    if "stale-lifecycle-evidence" in blocker_set:
        needed.append("fresh_lifecycle_evidence")
    if "safety-stop-observed" in blocker_set or _as_int((lifecycle.get("cohort_counts") or {}).get("safety_stopped")) > 0:
        needed.extend(["safety_stop_reason_review", "safer_threshold_or_executor_guard", "rollback_proof"])
    return sorted(set(needed))


def _keep_blocked_reason(needed_resolution: list[str], blockers: list[str]) -> str:
    needed = set(needed_resolution)
    if "safety_stop_reason_review" not in needed and "safety-stop-observed" not in set(blockers):
        return "anthropic-routing-blocked-until-lifecycle-coverage"
    parts = ["anthropic-routing-safety-stop-needs"]
    if "narrower_cohort" in needed:
        parts.append("narrower-cohort")
    if {"applied_coverage", "holdout_coverage"} & needed:
        parts.append("applied-holdout-coverage")
    if "fresh_lifecycle_evidence" in needed:
        parts.append("fresh-lifecycle-evidence")
    if "safer_threshold_or_executor_guard" in needed:
        parts.append("safer-threshold-or-executor-guard")
    if "rollback_proof" in needed:
        parts.append("rollback-proof")
    return "-".join(parts)


def _next_state_reason(needed_resolution: list[str], blockers: list[str]) -> str:
    blocker_set = set(blockers)
    if "stale-lifecycle-evidence" in blocker_set:
        return "safety-stop-awaits-fresh-lifecycle-or-holdout-evidence"
    if "safety-stop-observed" in blocker_set:
        return "safety-stop-requires-safer-threshold-or-executor-guard-and-rollback-proof"
    return "missing-applied-or-holdout-coverage"


def _normalize_lifecycle_evidence(candidate: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    coverage = lifecycle.get("coverage") if isinstance(lifecycle.get("coverage"), dict) else {}
    stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
    blockers = _lifecycle_blockers(lifecycle)
    matched = _as_int(lifecycle.get("matched_count") or coverage.get("matched_count") or candidate.get("matched_count"))
    observed = _as_int(lifecycle.get("observed_count"))
    if observed <= 0:
        observed = sum(
            _as_int(counts.get(key))
            for key in ("canary_applied", "canary_holdout", "safety_stopped", "skipped", "bypassed_or_disabled", "unknown")
        )
    return {
        "schema": lifecycle.get("schema") or "agentflow.anthropic_routing_canary_lifecycle_evidence.v1",
        "status": lifecycle.get("status") or ("blocked" if blockers else "matched"),
        "matched_count": matched,
        "observed_count": observed,
        "cohort_counts": {
            "canary_applied": _as_int(counts.get("canary_applied")),
            "canary_holdout": _as_int(counts.get("canary_holdout")),
            "safety_stopped": _as_int(counts.get("safety_stopped")),
            "skipped": _as_int(counts.get("skipped")),
            "bypassed_or_disabled": _as_int(counts.get("bypassed_or_disabled")),
            "unknown": _as_int(counts.get("unknown")),
        },
        "coverage": {
            "matched_count": matched,
            "observed_rate": _as_float(coverage.get("observed_rate")),
            "applied_rate": _as_float(coverage.get("applied_rate")),
            "holdout_rate": _as_float(coverage.get("holdout_rate")),
        },
        "error_count": _as_int(lifecycle.get("error_count")),
        "retry_count": _as_int(lifecycle.get("retry_count")),
        "fallback_count": _as_int(lifecycle.get("fallback_count")),
        "latest_observed_at": lifecycle.get("latest_observed_at"),
        "stale_evidence": {
            "stale": bool(stale.get("stale")),
            "age_hours": stale.get("age_hours") if isinstance(stale.get("age_hours"), (int, float, type(None))) else None,
            "max_age_hours": stale.get("max_age_hours") if isinstance(stale.get("max_age_hours"), (int, float, type(None))) else 72.0,
        },
        "blocker_codes": blockers,
        "blocker_reason_breakdown": lifecycle.get("blocker_reason_breakdown") if isinstance(lifecycle.get("blocker_reason_breakdown"), list) else [],
        "safety_stop_breakdown": lifecycle.get("safety_stop_breakdown") if isinstance(lifecycle.get("safety_stop_breakdown"), list) else [],
        "durable_blocked_reason": lifecycle.get("durable_blocked_reason"),
        "next_action": _next_action_for_blockers(candidate, blockers),
        "local_file_backed_representation": _local_file_backed_representation(),
        "privacy": PRIVACY,
    }


def _candidate_omission_reason(candidate: dict[str, Any], *, min_samples: int) -> str | None:
    blockers = set(_string_list(candidate.get("blockers")) + _string_list(candidate.get("blocker_codes")))
    if "safety-stop-observed" in blockers:
        return "safety-stop-observed"
    if "missing-applied-coverage" in blockers:
        return "missing-applied-coverage"
    if "missing-holdout-coverage" in blockers:
        return "missing-holdout-coverage"
    if "stale-lifecycle-evidence" in blockers:
        return "stale-lifecycle-evidence"
    safety_blockers = _safety_blockers(candidate)
    if safety_blockers:
        return safety_blockers[0]
    if _as_int(candidate.get("matched_count")) < min_samples:
        return "insufficient-samples"
    if _string(candidate.get("provider")).lower() != "anthropic":
        return "unsupported-provider"
    if _string(candidate.get("category")) != "tool-result":
        return "category-not-enabled"
    if "sonnet" not in _string(candidate.get("requested_model")).lower():
        return "not-sonnet-request"
    if "haiku" not in _string(candidate.get("target_model")).lower():
        return "missing-haiku-target"
    if blockers:
        return sorted(blockers)[0]
    if _as_float(candidate.get("estimated_savings_per_1000_calls_usd")) <= 0:
        return "missing-savings-estimate"
    return None


def _projected_lifecycle(
    *,
    matched_count: int,
    canary_fraction: float,
    holdout_fraction: float,
    savings_per_1000_calls_usd: float,
    safety_blockers: list[str] | None = None,
) -> dict[str, Any]:
    matched = max(0, matched_count)
    blockers = sorted(set(safety_blockers or []))
    if blockers:
        return {
            "schema": PROJECTED_LIFECYCLE_SCHEMA,
            "status": "safety_stopped",
            "matched_count": matched,
            "observed_count": matched,
            "cohort_counts": {
                "canary_applied": 0,
                "canary_holdout": 0,
                "safety_stopped": matched,
                "skipped": 0,
                "bypassed_or_disabled": 0,
                "unknown": 0,
            },
            "coverage": {"matched_count": matched, "observed_rate": 1.0 if matched else 0.0, "applied_rate": 0.0, "holdout_rate": 0.0},
            "estimated_savings_per_1000_calls_usd": round(savings_per_1000_calls_usd, 6),
            "error_count": 0,
            "retry_count": 0,
            "fallback_count": 0,
            "stale_evidence": {"stale": False, "max_age_hours": 72.0},
            "blocker_codes": blockers,
            "privacy": PRIVACY,
        }

    projected_holdout_count = min(matched, int(math.ceil(matched * holdout_fraction)))
    projected_canary_count = min(max(0, matched - projected_holdout_count), int(math.ceil(matched * canary_fraction)))
    observed = projected_holdout_count + projected_canary_count
    missing: list[str] = []
    if projected_canary_count <= 0:
        missing.append("missing-projected-applied-coverage")
    if projected_holdout_count <= 0:
        missing.append("missing-projected-holdout-coverage")
    return {
        "schema": PROJECTED_LIFECYCLE_SCHEMA,
        "status": "projected" if observed else "no-projected-coverage",
        "matched_count": matched,
        "observed_count": observed,
        "cohort_counts": {
            "canary_applied": projected_canary_count,
            "canary_holdout": projected_holdout_count,
            "safety_stopped": 0,
            "skipped": 0,
            "bypassed_or_disabled": max(0, matched - observed),
            "unknown": 0,
        },
        "coverage": {
            "matched_count": matched,
            "observed_rate": round(observed / matched, 6) if matched else 0.0,
            "applied_rate": round(projected_canary_count / matched, 6) if matched else 0.0,
            "holdout_rate": round(projected_holdout_count / matched, 6) if matched else 0.0,
        },
        "estimated_savings_per_1000_calls_usd": round(savings_per_1000_calls_usd, 6),
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "stale_evidence": {"stale": False, "max_age_hours": 72.0},
        "blocker_codes": missing,
        "privacy": PRIVACY,
    }


def _blocked_review(
    candidate: dict[str, Any],
    reason: str,
    lifecycle: dict[str, Any],
    executor_guard_dry_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = _string_list(lifecycle.get("blocker_codes"))
    next_action = _next_action_for_blockers(candidate, blockers)
    needed_resolution = _blocked_needed_resolution(candidate, blockers, lifecycle)
    keep_blocked_reason = _keep_blocked_reason(needed_resolution, blockers)
    return {
        "schema": BLOCKED_REVIEW_SCHEMA,
        "status": "blocked-review",
        "reason": reason,
        "next_state": "keep-blocked",
        "next_state_reason": _next_state_reason(needed_resolution, blockers),
        "keep_blocked_reason": keep_blocked_reason,
        "needed_resolution": needed_resolution,
        "target_candidate_id": _candidate_id(candidate),
        "provider": "anthropic",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
        "category": candidate.get("category"),
        "matched_count": _as_int(lifecycle.get("matched_count")),
        "observed_count": _as_int(lifecycle.get("observed_count")),
        "cohort_counts": lifecycle.get("cohort_counts"),
        "coverage": lifecycle.get("coverage"),
        "stale_evidence": lifecycle.get("stale_evidence"),
        "blocker_codes": blockers,
        "safety_stop_breakdown": lifecycle.get("safety_stop_breakdown") if isinstance(lifecycle.get("safety_stop_breakdown"), list) else [],
        "next_action": next_action,
        "durable_next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
        "local_file_backed_representation": _local_file_backed_representation(),
        "target_local_rule_file": "routing_rules.yaml",
        "promotion_allowed": False,
        "stage_allowed": False,
        "active_policy_changed": False,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "executor_guard_dry_run": executor_guard_dry_run,
        "lifecycle_evidence": lifecycle,
        "privacy": PRIVACY,
    }


def _omission(
    candidate: dict[str, Any],
    reason: str,
    lifecycle: dict[str, Any] | None = None,
    blocked_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": OMISSION_SCHEMA,
        "status": "blocked-review" if blocked_review else "safety_stopped" if lifecycle and lifecycle.get("status") == "safety_stopped" else "omitted",
        "reason": reason,
        "target_candidate_id": _candidate_id(candidate),
        "provider": "anthropic",
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
        "category": candidate.get("category"),
        "blocker_codes": _string_list(candidate.get("blockers") or candidate.get("blocker_codes")),
        "projected_lifecycle_evidence": lifecycle,
        "blocked_review": blocked_review,
        "privacy": PRIVACY,
    }


def _stage_review_intent(canary: dict[str, Any]) -> dict[str, Any]:
    promotion = canary.get("promotion") if isinstance(canary.get("promotion"), dict) else {}
    lifecycle = promotion.get("projected_anthropic_canary_lifecycle_evidence") if isinstance(promotion.get("projected_anthropic_canary_lifecycle_evidence"), dict) else {}
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    return {
        "schema": "agentflow.anthropic_routing_canary_stage_review_intent.v1",
        "status": "ready-for-operator-review",
        "routing_change_mode": "draft-only",
        "active_policy_changed": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "requested_model": canary.get("requested_model"),
        "target_model": canary.get("target_model"),
        "matched_count": _as_int(promotion.get("matched_count")),
        "intended_canary_fraction": canary.get("canary_fraction"),
        "intended_holdout_fraction": canary.get("holdout_fraction"),
        "projected_canary_applied_count": _as_int(counts.get("canary_applied")),
        "projected_canary_holdout_count": _as_int(counts.get("canary_holdout")),
        "projected_safety_stopped_count": _as_int(counts.get("safety_stopped")),
        "safety_stop_enabled": bool((canary.get("safety_stop") or {}).get("enabled")) if isinstance(canary.get("safety_stop"), dict) else False,
        "privacy_proof": PRIVACY,
    }


def _acceptance_report(
    staged: list[dict[str, Any]],
    *,
    projected_applied: int,
    projected_holdout: int,
    blocked_reviews: list[dict[str, Any]] | None = None,
    executor_guard_dry_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reported_tool_result_canary = False
    lifecycle_counts_present = False
    safety_gates_present = False
    blocked_review_recorded = False
    blocked_review_has_rule_file = False
    blocked_review_prevents_promotion = False
    blocked_review_has_keep_blocked_reason = False
    executor_guard_blocks_safety_stops = False
    executor_guard_reports_all_criteria = False
    executor_guard_ready_without_activation = False

    for draft in staged:
        routing = draft.get("policies", {}).get("routing", {}) if isinstance(draft.get("policies"), dict) else {}
        canary = routing.get("phase_canary") if isinstance(routing, dict) else None
        if not isinstance(canary, dict):
            continue
        lifecycle = (canary.get("promotion") or {}).get("projected_anthropic_canary_lifecycle_evidence")
        counts = lifecycle.get("cohort_counts") if isinstance(lifecycle, dict) else {}
        gates = canary.get("safety_gates") if isinstance(canary.get("safety_gates"), dict) else {}
        requested = _string(canary.get("requested_model")).lower()
        target = _string(canary.get("target_model")).lower()
        reported_tool_result_canary = reported_tool_result_canary or (
            canary.get("provider") == "anthropic"
            and "sonnet" in requested
            and "haiku" in target
            and "tool-result" in _string_list(canary.get("eligible_categories"))
            and "tool-execution" in _string_list(canary.get("eligible_workflow_phases"))
            and canary.get("enabled") is False
            and canary.get("review_only") is True
        )
        lifecycle_counts_present = lifecycle_counts_present or (
            isinstance(lifecycle, dict)
            and "canary_applied" in counts
            and "canary_holdout" in counts
            and "skipped" in counts
            and "safety_stopped" in counts
            and "error_count" in lifecycle
            and "fallback_count" in lifecycle
            and "retry_count" in lifecycle
        )
        safety_gates_present = safety_gates_present or (
            bool(gates.get("block_thinking_history"))
            and bool(gates.get("block_top_level_thinking"))
            and bool(gates.get("block_unsafe_tool_call_context"))
            and bool(gates.get("fallback_to_requested_on_rate_limit"))
            and bool(gates.get("content_free"))
        )

    for blocked in blocked_reviews or []:
        counts = blocked.get("cohort_counts") if isinstance(blocked.get("cohort_counts"), dict) else {}
        coverage = blocked.get("coverage") if isinstance(blocked.get("coverage"), dict) else {}
        representation = (
            blocked.get("local_file_backed_representation")
            if isinstance(blocked.get("local_file_backed_representation"), dict)
            else {}
        )
        blockers = set(_string_list(blocked.get("blocker_codes")))
        blocked_review_recorded = blocked_review_recorded or (
            blocked.get("status") == "blocked-review"
            and _as_int(blocked.get("matched_count")) > 0
            and "missing-applied-coverage" in blockers
            and "missing-holdout-coverage" in blockers
            and "safety-stop-observed" in blockers
            and "next_action" in blocked
            and "canary_applied" in counts
            and "canary_holdout" in counts
            and "safety_stopped" in counts
            and "applied_rate" in coverage
            and "holdout_rate" in coverage
        )
        blocked_review_has_rule_file = blocked_review_has_rule_file or (
            representation.get("exists") is True
            and representation.get("policy_section") == "routing"
            and representation.get("rule_file") == "routing_rules.yaml"
            and blocked.get("target_local_rule_file") == "routing_rules.yaml"
        )
        blocked_review_prevents_promotion = blocked_review_prevents_promotion or (
            blocked.get("promotion_allowed") is False
            and blocked.get("stage_allowed") is False
            and blocked.get("active_policy_changed") is False
            and blocked.get("wrote_active_policy_files") is False
        )
        needed = _string_list(blocked.get("needed_resolution"))
        blocked_review_has_keep_blocked_reason = blocked_review_has_keep_blocked_reason or (
            blocked.get("next_state") == "keep-blocked"
            and _string(blocked.get("keep_blocked_reason")).startswith("anthropic-routing-safety-stop-needs")
            and "safety_stop_reason_review" in needed
            and "rollback_proof" in needed
        )

    required_criteria = {
        "safety_stop_reason_review",
        "safer_threshold_or_executor_guard",
        "rollback_proof",
        "applied_coverage",
        "holdout_coverage",
    }
    for dry_run in executor_guard_dry_runs or []:
        criteria = dry_run.get("criterion_results") if isinstance(dry_run.get("criterion_results"), dict) else {}
        criteria_keys = set(criteria)
        executor_guard_reports_all_criteria = executor_guard_reports_all_criteria or required_criteria <= criteria_keys
        if dry_run.get("status") == "keep-blocked" and _as_int(dry_run.get("safety_stop_count")) > 0:
            executor_guard_blocks_safety_stops = executor_guard_blocks_safety_stops or (
                dry_run.get("stage_allowed") is False
                and dry_run.get("promotion_allowed") is False
                and dry_run.get("active_policy_changed") is False
                and dry_run.get("wrote_active_policy_files") is False
                and required_criteria <= criteria_keys
            )
        if dry_run.get("status") == "guard-ready":
            executor_guard_ready_without_activation = executor_guard_ready_without_activation or (
                dry_run.get("guard_ready") is True
                and dry_run.get("active_policy_changed") is False
                and dry_run.get("wrote_active_policy_files") is False
                and dry_run.get("provider_calls_made") is False
                and dry_run.get("managed_server_calls_made") is False
            )

    holdout_coverage = projected_holdout > 0
    privacy_clean = (
        bool(PRIVACY["metadata_only"])
        and bool(PRIVACY["aggregate_only"])
        and not bool(PRIVACY["raw_prompts_included"])
        and not bool(PRIVACY["provider_bodies_included"])
        and not bool(PRIVACY["request_ids_included"])
        and not bool(PRIVACY["session_ids_included"])
        and not bool(PRIVACY["provider_calls_made"])
        and not bool(PRIVACY["managed_server_calls_made"])
    )
    staged_acceptance_met = all(
        [
            reported_tool_result_canary,
            holdout_coverage,
            lifecycle_counts_present,
            safety_gates_present,
            privacy_clean,
        ]
    )
    blocked_acceptance_met = all(
        [
            blocked_review_recorded,
            blocked_review_has_rule_file,
            blocked_review_prevents_promotion,
            blocked_review_has_keep_blocked_reason,
            privacy_clean,
        ]
    )
    acceptance_met = staged_acceptance_met or blocked_acceptance_met
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "met" if acceptance_met else "not_met",
        "acceptance_met": acceptance_met,
        "tool_result_sonnet_to_haiku_candidate_reported": reported_tool_result_canary,
        "projected_canary_applied_count": projected_applied,
        "projected_canary_holdout_count": projected_holdout,
        "holdout_coverage_projected": holdout_coverage,
        "lifecycle_counts_include_applied_holdout_skipped_safety_error_retry_fallback": lifecycle_counts_present,
        "thinking_and_tool_safety_gates_present": safety_gates_present,
        "blocked_review_recorded": blocked_review_recorded,
        "blocked_review_has_local_rule_file_representation": blocked_review_has_rule_file,
        "no_automatic_promotion_while_blocked": blocked_review_prevents_promotion,
        "blocked_review_has_keep_blocked_reason": blocked_review_has_keep_blocked_reason,
        "executor_guard_dry_run_reports_all_criteria": executor_guard_reports_all_criteria,
        "executor_guard_blocks_safety_stopped_cohorts": executor_guard_blocks_safety_stops,
        "executor_guard_ready_without_activation": executor_guard_ready_without_activation,
        "metadata_only_privacy_proof": privacy_clean,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
    }


def _candidate_payload(
    report: dict[str, Any],
    candidate: dict[str, Any],
    *,
    canary_fraction: float,
    holdout_fraction: float,
    executor_guard_dry_run: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    candidate_id = _candidate_id(candidate)
    bounded_canary_fraction = _bounded_fraction(canary_fraction, 0.05)
    bounded_holdout_fraction = _bounded_fraction(holdout_fraction, 0.10)
    matched = max(1, _as_int(candidate.get("matched_count"), 1))
    savings_per_1000 = _as_float(candidate.get("estimated_savings_per_1000_calls_usd"))
    lifecycle = _projected_lifecycle(
        matched_count=matched,
        canary_fraction=bounded_canary_fraction,
        holdout_fraction=bounded_holdout_fraction,
        savings_per_1000_calls_usd=savings_per_1000,
    )
    policy_id = f"local-anthropic-routing-canary-{_safe_id(candidate_id)}"
    canary = {
        "enabled": False,
        "review_only": True,
        "policy_id": policy_id,
        "target_candidate_id": candidate_id,
        "policy_source": "local-manual",
        "provider": "anthropic",
        "source_surface": _metadata_value(candidate.get("source_surface"), "anthropic_messages"),
        "app_family": "anthropic",
        "model_pattern": _string(candidate.get("requested_model")),
        "requested_model": _string(candidate.get("requested_model")),
        "target_model": _string(candidate.get("target_model")),
        "routed_model": _string(candidate.get("target_model")),
        "stream": True,
        "eligible_workflow_phases": ["tool-execution"],
        "excluded_workflow_phases": ["planning", "thinking", "verification", "summary", "unknown"],
        "eligible_categories": ["tool-result"],
        "excluded_categories": ["tool-heavy", "tool-light", "code-gen", "long-context", "chat", "short-completion"],
        "min_workflow_phase_confidence": "high",
        "min_text_chars": 0,
        "max_text_chars": 0,
        "canary_fraction": bounded_canary_fraction,
        "holdout_fraction": bounded_holdout_fraction,
        "salt": _stable_id("anthropic-routing-canary-salt", candidate_id, candidate.get("requested_model"), candidate.get("target_model")),
        "cohort_unit": "session",
        "safety_gates": {
            "block_thinking_history": True,
            "block_top_level_thinking": True,
            "block_unsupported_streaming_shape": True,
            "block_unsafe_tool_call_context": True,
            "strip_model_incompatible_params": True,
            "fallback_to_requested_on_rate_limit": True,
            "content_free": True,
            "provider_calls_made_by_stage": False,
        },
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 10,
            "min_holdout_samples": 5,
            "max_error_rate": 0.05,
            "max_retry_rate": 0.20,
            "max_fallback_rate": 0.20,
            "max_latency_regression_ratio": 1.50,
            "limit": 500,
        },
        "fallback": {
            "enabled": True,
            "fallback_model": _string(candidate.get("requested_model")),
            "reason_codes": ["rate_limited", "upstream_error", "local-canary-safety-stop", "operator-rollback"],
        },
        "promotion": {
            "schema": "agentflow.anthropic_routing_canary_stage_metadata.v1",
            "source": "pass_through_routing_report" if report.get("schema") == PASS_THROUGH_SCHEMA else "anthropic_routing_report",
            "source_report_schema": report.get("schema"),
            "source_report_generated_at": report.get("generated_at"),
            "candidate_id": candidate_id,
            "provider": "anthropic",
            "source_surface": candidate.get("source_surface"),
            "endpoint": candidate.get("endpoint"),
            "requested_model": candidate.get("requested_model"),
            "target_model": candidate.get("target_model"),
            "category": candidate.get("category"),
            "workflow_phase": candidate.get("workflow_phase"),
            "workflow_phase_confidence": candidate.get("workflow_phase_confidence"),
            "matched_count": matched,
            "source_actionability": candidate.get("actionability"),
            "expected_local_executor": candidate.get("expected_local_executor") or "anthropic-routing-rules",
            "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 6),
            "estimated_savings_per_1000_calls_usd": round(savings_per_1000, 6),
            "projected_cohort_counts": {
                "matched": matched,
                "canary_applied": _as_int(lifecycle["cohort_counts"].get("canary_applied")),
                "canary_holdout": _as_int(lifecycle["cohort_counts"].get("canary_holdout")),
                "safety_stopped": _as_int(lifecycle["cohort_counts"].get("safety_stopped")),
                "bypassed_or_disabled": _as_int(lifecycle["cohort_counts"].get("bypassed_or_disabled")),
            },
            "projected_anthropic_canary_lifecycle_evidence": lifecycle,
            "aggregate_inference": candidate.get("aggregate_inference") or {},
            "reason_codes": ["eligible-anthropic-sonnet-to-haiku", "tool-result-phase-canary"],
            "rollback_metadata": {
                "schema": "agentflow.anthropic_routing_canary_rollback.v1",
                "rollback_action_type": "disable_phase_canary",
                "rollback_canary_fraction": 0.0,
                "rollback_holdout_fraction": 0.0,
                "rollback_reason_codes": [
                    "thinking-history-blocked",
                    "safety-stop-observed",
                    "error-rate-regression",
                    "retry-or-fallback-regression",
                    "latency-regression",
                    "operator-requested",
                ],
                "preserve_previous_rule_required": True,
            },
            "executor_guard_dry_run": executor_guard_dry_run,
            "privacy": PRIVACY,
        },
    }
    return candidate_id, {"phase_canary": canary}


def _error_result(error_type: str, message: str, *, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "summary": {"candidate_count": 0, "eligible_candidate_count": 0, "staged_count": 0, "omitted_count": 0, "acceptance_met": False},
        "staged_drafts": [],
        "omitted": [],
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
        "acceptance": _acceptance_report([], projected_applied=0, projected_holdout=0),
        "error": {"type": error_type, "message": message, "errors": errors or []},
    }


def build_anthropic_routing_canary_stage_report(
    report: dict[str, Any],
    *,
    canary_fraction: float = 0.05,
    holdout_fraction: float = 0.10,
    min_samples: int = 5,
    top_candidates: int = 1,
    draft_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return _error_result("invalid_report", "Anthropic routing canary staging requires a JSON report object")
    errors = _privacy_errors(report)
    if errors:
        return _error_result(
            "raw_payload_rejected",
            "Anthropic routing canary staging received raw request content or identifiers",
            errors=errors,
        )

    candidates = _candidate_list_from_report(report)
    staged: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    blocked_reviews: list[dict[str, Any]] = []
    executor_guard_dry_runs: list[dict[str, Any]] = []
    executor_guard_by_candidate_id: dict[str, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        existing_lifecycle = _existing_lifecycle(candidate)
        normalized_lifecycle = _normalize_lifecycle_evidence(candidate, existing_lifecycle) if existing_lifecycle else None
        executor_guard_dry_run = _executor_guard_dry_run(candidate, normalized_lifecycle) if normalized_lifecycle else None
        if executor_guard_dry_run:
            executor_guard_dry_runs.append(executor_guard_dry_run)
            executor_guard_by_candidate_id[_candidate_id(candidate)] = executor_guard_dry_run
        reason = _candidate_omission_reason(candidate, min_samples=min_samples)
        if reason:
            safety_blockers = _safety_blockers(candidate)
            lifecycle = None
            blocked_review = None
            if normalized_lifecycle:
                lifecycle = normalized_lifecycle
                if _string_list(lifecycle.get("blocker_codes")):
                    blocked_review = _blocked_review(candidate, reason, lifecycle, executor_guard_dry_run)
                    blocked_reviews.append(blocked_review)
            if safety_blockers and lifecycle is None:
                lifecycle = _projected_lifecycle(
                    matched_count=_as_int(candidate.get("matched_count")),
                    canary_fraction=0.0,
                    holdout_fraction=0.0,
                    savings_per_1000_calls_usd=_as_float(candidate.get("estimated_savings_per_1000_calls_usd")),
                    safety_blockers=safety_blockers,
                )
            omitted.append(_omission(candidate, reason, lifecycle, blocked_review))
            continue
        eligible.append(candidate)

    eligible.sort(key=lambda item: (-_as_float(item.get("estimated_savings_per_1000_calls_usd")), -_as_int(item.get("matched_count")), _candidate_id(item)))
    for index, candidate in enumerate(eligible[: max(1, int(top_candidates or 1))]):
        candidate_id, routing_policy = _candidate_payload(
            report,
            candidate,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
            executor_guard_dry_run=executor_guard_by_candidate_id.get(_candidate_id(candidate)),
        )
        canary = routing_policy["phase_canary"]
        suffix = "" if index == 0 else f"-{index + 1}"
        staged.append({
            "schema": STAGED_SCHEMA,
            "status": "staged",
            "draft_id": draft_id or f"{_safe_id(candidate_id)}{suffix}",
            "target_candidate_id": candidate_id,
            "target_local_policy": "routing.phase_canary",
            "active_policy_changed": False,
            "wrote_active_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policies": {"routing": routing_policy},
            "review_intent": _stage_review_intent(canary),
            "privacy": PRIVACY,
        })

    projected_applied = 0
    projected_holdout = 0
    projected_safety = 0
    for draft in staged:
        canary = draft["policies"]["routing"]["phase_canary"]
        counts = canary["promotion"]["projected_anthropic_canary_lifecycle_evidence"]["cohort_counts"]
        projected_applied += _as_int(counts.get("canary_applied"))
        projected_holdout += _as_int(counts.get("canary_holdout"))
        projected_safety += _as_int(counts.get("safety_stopped"))
    for item in omitted:
        lifecycle = item.get("projected_lifecycle_evidence")
        if isinstance(lifecycle, dict):
            projected_safety += _as_int((lifecycle.get("cohort_counts") or {}).get("safety_stopped"))
    acceptance = _acceptance_report(
        staged,
        projected_applied=projected_applied,
        projected_holdout=projected_holdout,
        blocked_reviews=blocked_reviews,
        executor_guard_dry_runs=executor_guard_dry_runs,
    )
    top_blocked = blocked_reviews[0] if blocked_reviews else {}
    top_guard = executor_guard_dry_runs[0] if executor_guard_dry_runs else {}

    return {
        "schema": SCHEMA,
        "ok": True,
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": len(candidates),
            "eligible_candidate_count": len(eligible),
            "staged_count": len(staged),
            "omitted_count": len(omitted),
            "blocked_review_count": len(blocked_reviews),
            "projected_canary_applied_count": projected_applied,
            "projected_canary_holdout_count": projected_holdout,
            "projected_safety_stopped_count": projected_safety,
            "acceptance_met": bool(acceptance["acceptance_met"]),
            "top_next_state": top_blocked.get("next_state"),
            "top_keep_blocked_reason": top_blocked.get("keep_blocked_reason"),
            "top_needed_resolution": top_blocked.get("needed_resolution") or [],
            "executor_guard_dry_run_count": len(executor_guard_dry_runs),
            "executor_guard_ready_count": sum(1 for item in executor_guard_dry_runs if item.get("status") == "guard-ready"),
            "executor_guard_keep_blocked_count": sum(1 for item in executor_guard_dry_runs if item.get("status") == "keep-blocked"),
            "top_executor_guard_status": top_guard.get("status"),
            "top_executor_guard_reason_codes": top_guard.get("reason_codes") or [],
            "top_required_local_executor": top_guard.get("required_local_executor"),
            "estimated_savings_per_1000_calls_usd": round(
                max((_as_float(draft["policies"]["routing"]["phase_canary"]["promotion"].get("estimated_savings_per_1000_calls_usd")) for draft in staged), default=0.0),
                6,
            ),
        },
        "staged_drafts": staged,
        "omitted": omitted,
        "blocked_reviews": blocked_reviews,
        "executor_guard_dry_runs": executor_guard_dry_runs,
        "wrote_active_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": PRIVACY,
        "acceptance": acceptance,
    }
