from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from agentflow_proxy.promotion_safety import classify_family_safety_stop_reason
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.optimization_promotion_rollout_actions.v1"
ACTION_SCHEMA = "agentflow.optimization_promotion_rollout_action.v1"
OMISSION_SCHEMA = "agentflow.optimization_promotion_rollout_omission.v1"
OMISSION_BUCKET_SCHEMA = "agentflow.optimization_promotion_rollout_omission_bucket.v1"

ACTIONABLE_VERDICTS = {"widen", "hold", "rollback"}
CACHE_STALE_DEPENDENCY_REASONS = {
    "dependency-cap-exceeded",
    "dependency-changed",
    "dependency-created",
    "dependency-deleted",
    "file-dependency-changed",
    "file-dependency-invalidated",
    "stale-dependency-blocker",
    "stale-risk-blockers",
}
CACHE_MISSING_INVALIDATION_REASONS = {
    "dependency-audit-missing",
    "dependency-freshness-missing",
    "dependency-missing",
    "file-dependency-evidence-absent",
    "file-dependency-missing",
    "file-watch-disabled",
    "missing-safe-invalidation-evidence",
    "safe-invalidation-required",
    "tool-call-cache-disabled",
}
CACHE_STABLE_DEPENDENCY_STATUSES = {"fresh", "stable", "dependency-stable", "safe", "valid"}
CACHE_UNSTABLE_DEPENDENCY_STATUSES = {"invalidated", "stale", "stale-risk", "changed", "unsafe"}
CACHE_MISSING_DEPENDENCY_STATUSES = {"missing", "unknown-missing", "absent", "unavailable"}
CACHE_REPLAY_FEEDBACK_OUTCOMES = [
    "cache-hit",
    "cache-miss",
    "canary-holdout",
    "invalidated",
    "stale-risk",
    "safety-stopped",
    "noop",
]
LOCAL_POLICY_SECTIONS = {
    "routing": {
        "policy_section": "routing",
        "target_local_policy_section": "routing.rules",
        "rule_prefix": "promotion-routing",
        "review_command": "agentflow-policy-review",
        "apply_command": "agentflow-optimization-promotion-canaries-apply --dry-run",
    },
    "cache": {
        "policy_section": "cache",
        "target_local_policy_section": "cache.rules",
        "rule_prefix": "promotion-cache",
        "review_command": "agentflow-managed-rollout-actions-review",
        "apply_command": "agentflow-optimization-promotion-canaries-apply --dry-run",
    },
    "crunch": {
        "policy_section": "crunch",
        "target_local_policy_section": "crunch.rules",
        "rule_prefix": "promotion-crunch",
        "review_command": "agentflow-managed-rollout-actions-review",
        "apply_command": "agentflow-optimization-promotion-canaries-apply --dry-run",
    },
}

_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    return f"{prefix}:{_stable_hash(parts)[:length]}"


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


def _bounded_fraction(value: Any, default: float = 0.0) -> float:
    return round(min(1.0, max(0.0, _as_float(value, default))), 6)


def _reason(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text if _REASON_RE.match(text) else "unsanitized-reason-code"


def _reason_codes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    reasons = {_reason(value) for value in values}
    return sorted(reason for reason in reasons if reason)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "safe", "stable", "fresh"}
    return bool(value)


def _cache_dependency_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "file_dependency_audit",
        "dependency_audit",
        "dependency_freshness",
        "dependency_evidence",
        "local_dependency_freshness",
    ):
        value = candidate.get(key)
        if isinstance(value, dict):
            return value
    cacheability = candidate.get("cacheability") if isinstance(candidate.get("cacheability"), dict) else {}
    for key in ("file_dependency_audit", "dependency_audit", "dependency_freshness"):
        value = cacheability.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _cache_reason_set(candidate: dict[str, Any]) -> set[str]:
    reasons: set[str] = set(_reason_codes(candidate.get("reason_codes")))
    for key in ("blockers", "warning_codes", "cache_replay_blocker_reasons"):
        reasons.update(_reason_codes(candidate.get(key)))
    for key in ("blocker_reason_breakdown", "invalidation_reason_breakdown", "stale_dependency_breakdown"):
        rows = candidate.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    reason = _reason(row.get("value") or row.get("reason"))
                    if reason:
                        reasons.add(reason)
    audit = _cache_dependency_audit(candidate)
    for key in ("invalidation_reason", "dependency_capture_reason", "status", "reason"):
        reason = _reason(audit.get(key))
        if reason:
            reasons.add(reason)
    return reasons


def _cache_dependency_status(candidate: dict[str, Any]) -> str | None:
    for key in ("file_dependency_status", "dependency_status", "dependency_health", "dependency_gate_status"):
        value = str(candidate.get(key) or "").strip().lower().replace("_", "-")
        if value:
            return value
    audit = _cache_dependency_audit(candidate)
    for key in ("status", "dependency_status", "dependency_health"):
        value = str(audit.get(key) or "").strip().lower().replace("_", "-")
        if value:
            return value
    return None


def _cache_safe_invalidation(candidate: dict[str, Any]) -> bool:
    audit = _cache_dependency_audit(candidate)
    cacheability = candidate.get("cacheability") if isinstance(candidate.get("cacheability"), dict) else {}
    return any(
        _truthy(value)
        for value in (
            candidate.get("safe_invalidation_evidence"),
            candidate.get("safe_invalidation"),
            candidate.get("file_dependency_evidence_available"),
            candidate.get("file_dependency_fingerprint_available"),
            cacheability.get("safe_invalidation_evidence"),
            cacheability.get("file_dependency_evidence_available"),
            audit.get("safe_invalidation_evidence"),
            audit.get("file_dependency_evidence_available"),
        )
    )


def _cache_tool_or_stream_related(candidate: dict[str, Any]) -> bool:
    category = str(candidate.get("category") or "").strip().lower().replace("_", "-")
    return bool(candidate.get("has_tools") or candidate.get("stream") or category.startswith("tool-"))


def _cache_dependency_gate(candidate: dict[str, Any]) -> dict[str, Any]:
    reasons = _cache_reason_set(candidate)
    status = _cache_dependency_status(candidate)
    audit = _cache_dependency_audit(candidate)
    safe = _cache_safe_invalidation(candidate)
    explicit_dependency_signal = bool(
        status
        or audit
        or safe
        or reasons & (CACHE_STALE_DEPENDENCY_REASONS | CACHE_MISSING_INVALIDATION_REASONS)
    )
    requires_dependency = bool(
        _cache_tool_or_stream_related(candidate)
        or explicit_dependency_signal
        or candidate.get("requires_dependency_evidence")
        or candidate.get("safe_invalidation_required")
    )
    stale = sorted(reasons & CACHE_STALE_DEPENDENCY_REASONS)
    missing = sorted(reasons & CACHE_MISSING_INVALIDATION_REASONS)
    if status in CACHE_UNSTABLE_DEPENDENCY_STATUSES or stale or audit.get("cap_exceeded") or audit.get("invalidation_reason"):
        return {
            "status": "blocked",
            "reason": stale[0] if stale else str(audit.get("invalidation_reason") or status or "stale-dependency-blocker"),
            "safety_outcome": "stale-risk",
            "requires_dependency_evidence": requires_dependency,
            "safe_invalidation_evidence": False,
        }
    if status in CACHE_MISSING_DEPENDENCY_STATUSES or missing or (requires_dependency and not safe):
        return {
            "status": "blocked",
            "reason": missing[0] if missing else "missing-safe-invalidation-evidence",
            "safety_outcome": "noop",
            "requires_dependency_evidence": requires_dependency,
            "safe_invalidation_evidence": False,
        }
    if safe or status in CACHE_STABLE_DEPENDENCY_STATUSES:
        return {
            "status": "ready",
            "reason": "dependency-stable",
            "safety_outcome": "canary",
            "requires_dependency_evidence": requires_dependency,
            "safe_invalidation_evidence": True,
        }
    return {
        "status": "ready",
        "reason": "no-dependency-required",
        "safety_outcome": "canary",
        "requires_dependency_evidence": False,
        "safe_invalidation_evidence": False,
    }


def _cache_dependency_omission_reason(candidate: dict[str, Any]) -> str | None:
    if _local_policy(candidate) != LOCAL_POLICY_SECTIONS["cache"]:
        return None
    if str(candidate.get("verdict") or "") == "rollback":
        return None
    gate = _cache_dependency_gate(candidate)
    if gate["status"] == "ready":
        return None
    reason = str(gate.get("reason") or "")
    if gate.get("safety_outcome") == "stale-risk" or "stale" in reason or "changed" in reason or "invalidated" in reason:
        return "cache-replay-stale-dependency-risk"
    return "cache-replay-missing-invalidation-evidence"


def _projected_cache_hit_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    matched = _as_int(candidate.get("matched_count") or candidate.get("sample_count"))
    projected_hits = _as_int(
        candidate.get("projected_hit_count")
        or candidate.get("duplicate_fingerprint_rows")
        or candidate.get("session_pattern_repeated_rows")
        or candidate.get("safety_eligible_count")
    )
    hit_rate_value = candidate.get("projected_hit_rate")
    if hit_rate_value is None and matched > 0 and projected_hits > 0:
        hit_rate_value = projected_hits / matched
    return {
        "matched_count": matched,
        "projected_hit_count": projected_hits,
        "projected_hit_rate": _bounded_fraction(hit_rate_value, 0.0),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
    }


def _normalize_pattern_hash(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    digest = text.removeprefix("sha256:") if text.startswith("sha256:") else text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _collect_pattern_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if (
                key_l in {"pattern_hash", "normalized_pattern_hash", "crunch_pattern_hash", "cache_pattern_hash", "pattern_hashes", "hashes"}
                or key_l.endswith(("_pattern_hash", "_pattern_sha256"))
                or ("pattern" in key_l and key_l.endswith(("_hash", "_hashes", "_sha256")))
            ):
                if isinstance(item, list):
                    hashes.extend(hash_value for nested in item if (hash_value := _normalize_pattern_hash(nested)))
                elif (hash_value := _normalize_pattern_hash(item)) is not None:
                    hashes.append(hash_value)
            hashes.extend(_collect_pattern_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_collect_pattern_hashes(item))
    return sorted(set(hashes))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _counter_rows(values: list[str]) -> list[dict[str, Any]]:
    counts = Counter(values)
    rows = [{"value": value, "count": count} for value, count in counts.items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _sum_counter_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count} for value, count in counts.items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "local_only": True,
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("candidate_id") or "").strip()
    if value:
        return value[:160]
    return _stable_id("promotion-candidate", candidate)


def _action_family(candidate: dict[str, Any]) -> str:
    return str(candidate.get("action_family") or "unknown").strip().lower().replace("_", "-")


def _local_policy(candidate: dict[str, Any]) -> dict[str, str] | None:
    family = _action_family(candidate)
    if family in {"routing", "model-routing", "phase-routing"}:
        return LOCAL_POLICY_SECTIONS["routing"]
    if family in {"cache", "cache-replay", "cache-replayability"}:
        return LOCAL_POLICY_SECTIONS["cache"]
    if family in {"crunch", "pattern", "old-context-summarization", "old-context-summary"}:
        return LOCAL_POLICY_SECTIONS["crunch"]
    optimization_family = str(candidate.get("optimization_family") or "").strip().lower().replace("_", "-")
    if "routing" in optimization_family:
        return LOCAL_POLICY_SECTIONS["routing"]
    if "cache" in optimization_family:
        return LOCAL_POLICY_SECTIONS["cache"]
    if "crunch" in optimization_family or "summary" in optimization_family or "summarization" in optimization_family:
        return LOCAL_POLICY_SECTIONS["crunch"]
    return None


def _is_old_context_summary_candidate(candidate: dict[str, Any]) -> bool:
    family = _action_family(candidate)
    optimization_family = str(candidate.get("optimization_family") or "").strip().lower().replace("_", "-")
    candidate_family = str(candidate.get("candidate_family") or "").strip().lower().replace("_", "-")
    return any(
        "old-context" in value or "old-context-summary" in value or "old-context-summarization" in value
        for value in (family, optimization_family, candidate_family)
    ) or any("summarization" in value or "summary" in value for value in (family, optimization_family, candidate_family))


def _cohort_counts(candidate: dict[str, Any]) -> dict[str, int]:
    counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    return {
        "canary_applied": _as_int(counts.get("canary_applied")),
        "canary_holdout": _as_int(counts.get("canary_holdout")),
        "bypassed_or_disabled": _as_int(counts.get("bypassed_or_disabled")),
    }


def _current_canary_fraction(candidate: dict[str, Any]) -> float:
    counts = _cohort_counts(candidate)
    total = counts["canary_applied"] + counts["canary_holdout"]
    return _bounded_fraction(counts["canary_applied"] / total) if total else 0.0


def _recommended_canary_fraction(
    candidate: dict[str, Any],
    *,
    initial_canary_fraction: float,
    widen_step: float,
    max_canary_fraction: float,
) -> float:
    verdict = str(candidate.get("verdict") or "")
    current = _current_canary_fraction(candidate)
    if verdict == "rollback":
        return 0.0
    if verdict == "hold":
        return current
    if current <= 0:
        return _bounded_fraction(initial_canary_fraction)
    return _bounded_fraction(min(max_canary_fraction, current + widen_step))


def _evidence_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    eval_evidence = candidate.get("eval_evidence") if isinstance(candidate.get("eval_evidence"), dict) else {}
    score_summary = eval_evidence.get("score_summary") if isinstance(eval_evidence.get("score_summary"), dict) else {}
    result = {
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "sample_count": _as_int(candidate.get("sample_count")),
        "cohort_counts": _cohort_counts(candidate),
        "eval_result_count": _as_int(eval_evidence.get("result_count")),
        "eval_pass_count": _as_int(eval_evidence.get("pass_count")),
        "eval_fail_count": _as_int(eval_evidence.get("fail_count")),
        "eval_blocked_count": _as_int(eval_evidence.get("blocked_count")),
        "latest_eval_result_at": eval_evidence.get("latest_result_at"),
        "eval_evidence_stale": bool(eval_evidence.get("stale")),
        "avg_output_similarity": score_summary.get("avg_output_similarity"),
        "avg_quality_score": score_summary.get("avg_quality_score"),
        "reason_codes": _reason_codes(candidate.get("reason_codes")),
    }
    if _local_policy(candidate) == LOCAL_POLICY_SECTIONS["cache"]:
        gate = _cache_dependency_gate(candidate)
        result["cache_replay_dependency_gate"] = {
            "status": gate["status"],
            "reason": gate["reason"],
            "requires_dependency_evidence": gate["requires_dependency_evidence"],
            "safe_invalidation_evidence": gate["safe_invalidation_evidence"],
            "file_paths_included": False,
            "dependency_path_values_included": False,
            "dependency_fingerprints_included": False,
        }
        result["cache_replay_projection"] = _projected_cache_hit_metadata(candidate)
        result["cache_replay_feedback_outcomes"] = CACHE_REPLAY_FEEDBACK_OUTCOMES
    return result


def _privacy_blocked(candidate: dict[str, Any]) -> bool:
    privacy = candidate.get("privacy") if isinstance(candidate.get("privacy"), dict) else {}
    unsafe_flags = (
        "raw_prompts_included",
        "raw_provider_bodies_included",
        "raw_responses_included",
        "raw_transcripts_included",
        "tool_payloads_included",
        "cache_keys_included",
        "request_ids_included",
        "raw_session_ids_included",
        "filesystem_paths_included",
        "api_keys_included",
    )
    return any(bool(privacy.get(flag)) for flag in unsafe_flags)


def _omission_reason(candidate: dict[str, Any], *, policy: dict[str, str] | None = None) -> str | None:
    if _privacy_blocked(candidate):
        return "privacy-blocked"
    if policy is None:
        return "unsupported-local-policy-section"
    verdict = str(candidate.get("verdict") or "")
    reasons = set(_reason_codes(candidate.get("reason_codes")))
    if verdict == "needs_eval" or "eval-results-missing" in reasons or any(reason.startswith("insufficient-") for reason in reasons):
        return "insufficient-eval-evidence"
    if dependency_reason := _cache_dependency_omission_reason(candidate):
        return dependency_reason
    if verdict not in ACTIONABLE_VERDICTS:
        return "unsupported-promotion-verdict"
    return None


def _target_rule_id(policy: dict[str, str], candidate: dict[str, Any]) -> str:
    candidate_id = _candidate_id(candidate)
    digest = _stable_hash([policy["policy_section"], candidate_id])[:12]
    return f"{policy['rule_prefix']}-{digest}"


def _action_type(candidate: dict[str, Any]) -> str:
    verdict = str(candidate.get("verdict") or "")
    return "rollback" if verdict == "rollback" else "hold" if verdict == "hold" else "widen"


def _action(
    candidate: dict[str, Any],
    *,
    policy: dict[str, str],
    initial_canary_fraction: float,
    widen_step: float,
    max_canary_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    action_type = _action_type(candidate)
    canary_fraction = _recommended_canary_fraction(
        candidate,
        initial_canary_fraction=initial_canary_fraction,
        widen_step=widen_step,
        max_canary_fraction=max_canary_fraction,
    )
    target_rule_id = _target_rule_id(policy, candidate)
    local_policy_update: dict[str, Any] = {
        "kind": "yaml-rule-canary",
        "policy_source": "managed-recommended",
        "managed_enforced": False,
        "required_local_review": True,
        "candidate_target_model": candidate.get("candidate_target_model"),
        "candidate_profile": candidate.get("candidate_profile"),
    }
    if policy["policy_section"] == "crunch" and _is_old_context_summary_candidate(candidate):
        local_policy_update["kind"] = "old-context-summarization-canary"
        thresholds = candidate.get("thresholds") if isinstance(candidate.get("thresholds"), dict) else {}
        conditions = candidate.get("conditions") if isinstance(candidate.get("conditions"), dict) else {}
        action_fields = candidate.get("action") if isinstance(candidate.get("action"), dict) else {}
        canary = candidate.get("canary") if isinstance(candidate.get("canary"), dict) else {}
        safety = (
            candidate.get("safety_stop")
            if isinstance(candidate.get("safety_stop"), dict)
            else candidate.get("safety_gates")
            if isinstance(candidate.get("safety_gates"), dict)
            else {}
        )
        summary_update: dict[str, Any] = {
            "enabled": True,
            "rule_id": target_rule_id,
            "candidate_id": candidate_id,
            "model": (
                candidate.get("summary_model")
                or candidate.get("model_hint")
                or candidate.get("candidate_target_model")
                or action_fields.get("model")
            ),
            "profile": candidate.get("candidate_profile") or candidate.get("profile"),
            "placement": candidate.get("placement") or action_fields.get("placement") or "system",
        }
        for key in (
            "min_request_chars",
            "min_summarized_chars",
            "max_turns",
            "keep_recent_turns",
            "max_summary_chars",
            "max_source_chars",
            "max_summary_cost_usd",
            "excluded_categories",
            "block_tool_protocol",
            "block_thinking",
        ):
            value = candidate.get(key, conditions.get(key, thresholds.get(key, action_fields.get(key))))
            if value is not None:
                summary_update[key] = value
        local_policy_update["old_context_summarization"] = summary_update
        local_policy_update["canary"] = {
            "enabled": action_type != "rollback",
            "fraction": canary_fraction,
            "holdout_fraction": 0.0 if action_type == "rollback" else _bounded_fraction(
                canary.get("holdout_fraction", holdout_fraction)
            ),
            "salt": str(canary.get("salt") or canary.get("canary_salt") or target_rule_id),
            "unit": str(canary.get("unit") or canary.get("canary_unit") or "source_hash"),
        }
        local_policy_update["safety_stop"] = {
            "enabled": True,
            "min_outcome_samples": _as_int(safety.get("min_outcome_samples", safety.get("min_samples", 5)), 5),
            "window": _as_int(safety.get("window", 500), 500),
            "max_error_rate": _as_float(safety.get("max_error_rate"), 0.1),
            "max_retry_rate": _as_float(safety.get("max_retry_rate"), 0.25),
            "max_negative_net_savings_rate": _as_float(safety.get("max_negative_net_savings_rate"), 0.5),
            "max_summary_failure_rate": _as_float(safety.get("max_summary_failure_rate"), 0.1),
            "max_error_rate_delta": _as_float(safety.get("max_error_rate_delta"), 0.05),
        }
    if policy["policy_section"] == "cache":
        dependency_gate = _cache_dependency_gate(candidate)
        projection = _projected_cache_hit_metadata(candidate)
        pattern_hashes = _collect_pattern_hashes(candidate)
        conditions: dict[str, Any] = {}
        if pattern_hashes:
            conditions["pattern_hashes"] = pattern_hashes
        for key in (
            "source_surface",
            "app_family",
            "category",
            "workflow_phase",
            "text_bucket",
            "token_bucket",
            "cacheability_bucket",
        ):
            value = candidate.get(key)
            if value is not None:
                conditions[key] = str(value)
        cacheability = candidate.get("cacheability") if isinstance(candidate.get("cacheability"), dict) else {}
        if "cacheability_bucket" not in conditions and cacheability.get("cacheability_bucket") is not None:
            conditions["cacheability_bucket"] = str(cacheability["cacheability_bucket"])
        for key in ("has_tools", "stream"):
            if key in candidate:
                conditions[key] = bool(candidate.get(key))
        replayability_levels = _string_list(candidate.get("replayability_levels") or candidate.get("replayability_level"))
        if replayability_levels:
            conditions["replayability_levels"] = replayability_levels
        for key in (
            "static_information_hint",
            "time_sensitive_hint",
            "user_specific_hint",
            "exact_cache_candidate_hint",
        ):
            if key in candidate:
                conditions[key] = bool(candidate.get(key))
            elif key in cacheability:
                conditions[key] = bool(cacheability.get(key))
        safe_invalidation = bool(dependency_gate.get("safe_invalidation_evidence"))
        local_policy_update["conditions"] = conditions
        local_policy_update["action"] = {
            "type": "exact_cache_pattern",
            "allow_tool_calls": bool(candidate.get("has_tools")),
            "safe_invalidation": safe_invalidation,
            "safe_invalidation_evidence": safe_invalidation,
            "streaming": bool(candidate.get("stream")),
            "estimated_saved_cost_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
            "projected_hit_count": projection["projected_hit_count"],
            "projected_hit_rate": projection["projected_hit_rate"],
        }
        local_policy_update["cache_replay_canary"] = {
            "schema": "agentflow.cache_replay_dependency_gated_canary.v1",
            "dependency_gate": dependency_gate,
            "projection": projection,
            "canary_fraction": canary_fraction,
            "holdout_fraction": 0.0 if action_type == "rollback" else _bounded_fraction(holdout_fraction),
            "feedback_outcomes": CACHE_REPLAY_FEEDBACK_OUTCOMES,
            "records_hit_miss_holdout_invalidated_stale_risk_safety_stop_noop": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "dependency_path_values_included": False,
            "individual_candidate_ids_included": False,
        }
        local_policy_update["safety_stop"] = {
            "min_outcome_samples": 5,
            "rollback_threshold": 0.2,
        }
    action = {
        "schema": ACTION_SCHEMA,
        "action_id": _stable_id("promotion-rollout-action", candidate_id, action_type, policy["policy_section"], target_rule_id),
        "status": "planned",
        "action_type": action_type,
        "verdict": str(candidate.get("verdict") or ""),
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "action_family": _action_family(candidate),
        "optimization_family": str(candidate.get("optimization_family") or "unknown"),
        "source_surface": str(candidate.get("source_surface") or "unknown"),
        "app_family": str(candidate.get("app_family") or "unknown"),
        "policy_section": policy["policy_section"],
        "target_local_policy_section": policy["target_local_policy_section"],
        "local_policy_update": local_policy_update,
        "current_canary_fraction": _current_canary_fraction(candidate),
        "canary_fraction": canary_fraction,
        "holdout_fraction": 0.0 if action_type == "rollback" else _bounded_fraction(holdout_fraction),
        "evidence_summary": _evidence_summary(candidate),
        "rollback_metadata": {
            "rollback_action_type": "rollback",
            "rollback_canary_fraction": 0.0,
            "rollback_reason_codes": [
                "eval-failed",
                "safety-stop-observed",
                "rollback-error-rate",
                "operator-requested",
            ],
            "preserve_previous_rule_required": True,
        },
        "local_review": {
            "required": True,
            "review_command": policy["review_command"],
            "apply_preview_command": policy["apply_command"],
        },
        "privacy": _privacy_summary(),
    }
    return action


def _omission(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    reason_codes = _reason_codes(candidate.get("reason_codes"))
    family_safety = classify_family_safety_stop_reason(
        action_family=_action_family(candidate),
        reason=reason,
        reason_codes=reason_codes,
        file_backed_policy_exists=_local_policy(candidate) is not None,
    )
    result = {
        "schema": OMISSION_SCHEMA,
        "status": "omitted",
        "reason": reason,
        "target_candidate_id": _candidate_id(candidate),
        "action_family": _action_family(candidate),
        "optimization_family": str(candidate.get("optimization_family") or "unknown"),
        "source_surface": str(candidate.get("source_surface") or "unknown"),
        "app_family": str(candidate.get("app_family") or "unknown"),
        "verdict": str(candidate.get("verdict") or "unknown"),
        "evidence_summary": _evidence_summary(candidate),
        "privacy": _privacy_summary(),
    }
    if family_safety:
        result["safety_stop_reason"] = family_safety
        result["safety_stop_reason_code"] = family_safety["code"]
        result["recommended_blocker_state"] = family_safety["blocked_state"]
        result["recommended_unblock_action"] = family_safety["local_unblock_action"]
    return result


def _omission_bucket_next_action(reason: str, reason_codes: list[str], *, action_family: str) -> str:
    family_safety = classify_family_safety_stop_reason(
        action_family=action_family,
        reason=reason,
        reason_codes=reason_codes,
    )
    if family_safety:
        return str(family_safety["next_action"])
    reasons = {str(item or "").strip().lower().replace("_", "-") for item in reason_codes if str(item or "").strip()}
    reason_l = str(reason or "").strip().lower().replace("_", "-")
    haystack = " ".join(sorted(reasons | {reason_l}))
    family_l = str(action_family or "").strip().lower().replace("_", "-")
    if "privacy" in haystack or "unsupported-local-policy-section" in haystack:
        return "keep-blocked"
    if "safety" in haystack or "rollback" in haystack or "regression" in haystack or "eval-failed" in haystack:
        return "review-safety-stop"
    if "dependency" in haystack or "freshness" in haystack or "stale-risk" in haystack or "stale-evidence" in haystack:
        return "fix-dependency-freshness"
    if "insufficient-canary-holdout" in haystack or "missing-holdout" in haystack:
        return "collect-canary-holdout"
    if "insufficient-canary-applied" in haystack or "missing-applied" in haystack or "missing-canary-lifecycle" in haystack:
        return "collect-canary-applied"
    if "eval-results-missing" in haystack or "insufficient-eval" in haystack or "eval-queued" in haystack or reason_l == "insufficient-eval-evidence":
        return "run-local-shadow-eval"
    if "cache" in family_l and ("invalidation" in haystack or "stale" in haystack):
        return "fix-dependency-freshness"
    if reason_l.startswith("unsupported-"):
        return "keep-blocked"
    return "inspect-promotion-evidence"


def _omission_buckets(omitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in omitted:
        if not isinstance(row, dict):
            continue
        evidence = row.get("evidence_summary") if isinstance(row.get("evidence_summary"), dict) else {}
        family_safety = row.get("safety_stop_reason") if isinstance(row.get("safety_stop_reason"), dict) else None
        reason = _reason((family_safety or {}).get("code") or row.get("reason")) or "unknown"
        reason_codes = _reason_codes(evidence.get("reason_codes"))
        action_family = str(row.get("action_family") or "unknown").strip().lower().replace("_", "-") or "unknown"
        optimization_family = str(row.get("optimization_family") or "unknown").strip().lower().replace("_", "-") or "unknown"
        source_surface = str(row.get("source_surface") or "unknown").strip().lower().replace("_", "-") or "unknown"
        app_family = str(row.get("app_family") or "unknown").strip().lower().replace("_", "-") or "unknown"
        next_action = _omission_bucket_next_action(reason, reason_codes, action_family=action_family)
        key = (action_family, optimization_family, source_surface, app_family, reason, next_action)
        bucket = grouped.setdefault(
            key,
            {
                "schema": OMISSION_BUCKET_SCHEMA,
                "status": "omitted",
                "bucket_id": _stable_id("promotion-omission-bucket", key),
                "action_family": action_family,
                "optimization_family": optimization_family,
                "source_surface": source_surface,
                "app_family": app_family,
                "reason": reason,
                "next_action": next_action,
                "safety_stop_reason": family_safety,
                "recommended_blocker_state": (family_safety or {}).get("blocked_state"),
                "recommended_unblock_action": (family_safety or {}).get("local_unblock_action"),
                "candidate_count": 0,
                "sample_count": 0,
                "projected_savings_usd": 0.0,
                "reason_code_counts": {},
                "verdict_counts": {},
                "privacy": _privacy_summary(),
            },
        )
        bucket["candidate_count"] += 1
        bucket["sample_count"] += _as_int(evidence.get("sample_count"))
        bucket["projected_savings_usd"] += _as_float(evidence.get("projected_savings_usd"))
        _counter = bucket["reason_code_counts"]
        if not reason_codes:
            reason_codes = [reason]
        for code in reason_codes:
            _counter[code] = _counter.get(code, 0) + 1
        verdict = str(row.get("verdict") or "unknown")
        bucket["verdict_counts"][verdict] = bucket["verdict_counts"].get(verdict, 0) + 1

    buckets: list[dict[str, Any]] = []
    for bucket in grouped.values():
        reason_counts = _sum_counter_rows(bucket.pop("reason_code_counts"))
        verdict_counts = _sum_counter_rows(bucket.pop("verdict_counts"))
        bucket["projected_savings_usd"] = round(_as_float(bucket["projected_savings_usd"]), 8)
        bucket["reason_code_counts"] = reason_counts
        bucket["top_reason_codes"] = [str(row["value"]) for row in reason_counts[:5]]
        bucket["verdict_counts"] = verdict_counts
        buckets.append(bucket)
    buckets.sort(
        key=lambda row: (
            -_as_float(row.get("projected_savings_usd")),
            -_as_int(row.get("candidate_count")),
            str(row.get("next_action")),
            str(row.get("action_family")),
            str(row.get("optimization_family")),
            str(row.get("source_surface")),
            str(row.get("reason")),
        )
    )
    for index, bucket in enumerate(buckets, start=1):
        bucket["rank"] = index
    return buckets


def _safety_stop_reason_buckets(omitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in omitted:
        safety = row.get("safety_stop_reason") if isinstance(row.get("safety_stop_reason"), dict) else None
        if not safety:
            continue
        key = (
            str(safety.get("action_family") or row.get("action_family") or "unknown"),
            str(safety.get("code") or "unknown"),
            str(safety.get("next_action") or "inspect-promotion-evidence"),
            str(safety.get("blocked_state") or "unknown"),
        )
        bucket = counts.setdefault(
            key,
            {
                "action_family": key[0],
                "reason_code": key[1],
                "next_action": key[2],
                "recommended_blocker_state": key[3],
                "candidate_count": 0,
                "projected_savings_usd": 0.0,
                "privacy": _privacy_summary(),
            },
        )
        evidence = row.get("evidence_summary") if isinstance(row.get("evidence_summary"), dict) else {}
        bucket["candidate_count"] += 1
        bucket["projected_savings_usd"] += _as_float(evidence.get("projected_savings_usd"))
    rows = list(counts.values())
    for row in rows:
        row["projected_savings_usd"] = round(_as_float(row.get("projected_savings_usd")), 8)
    rows.sort(key=lambda row: (-_as_int(row["candidate_count"]), -_as_float(row["projected_savings_usd"]), str(row["action_family"]), str(row["reason_code"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_optimization_promotion_actions(
    promotion_report: dict[str, Any],
    *,
    initial_canary_fraction: float = 0.10,
    widen_step: float = 0.25,
    max_canary_fraction: float = 1.0,
    holdout_fraction: float = 0.10,
) -> dict[str, Any]:
    candidates = promotion_report.get("candidates") if isinstance(promotion_report, dict) else []
    if not isinstance(candidates, list):
        candidates = promotion_report.get("verdicts") if isinstance(promotion_report, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    actions: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        policy = _local_policy(candidate)
        omission_reason = _omission_reason(candidate, policy=policy)
        if omission_reason:
            omitted.append(_omission(candidate, reason=omission_reason))
            continue
        if policy is None:
            omitted.append(_omission(candidate, reason="unsupported-local-policy-section"))
            continue
        actions.append(
            _action(
                candidate,
                policy=policy,
                initial_canary_fraction=initial_canary_fraction,
                widen_step=widen_step,
                max_canary_fraction=max_canary_fraction,
                holdout_fraction=holdout_fraction,
            )
        )

    actions.sort(key=lambda row: (str(row.get("policy_section")), str(row.get("target_candidate_id")), str(row.get("action_type"))))
    omitted.sort(key=lambda row: (str(row.get("reason")), str(row.get("target_candidate_id"))))
    action_families = [str(row.get("action_family") or "unknown") for row in actions]
    policy_sections = [str(row.get("policy_section") or "unknown") for row in actions]
    omission_reasons = [str(row.get("reason") or "unknown") for row in omitted]
    omission_buckets = _omission_buckets(omitted)
    safety_stop_reason_buckets = _safety_stop_reason_buckets(omitted)
    top_omission_bucket = omission_buckets[0] if omission_buckets else None
    top_safety_stop_reason = safety_stop_reason_buckets[0] if safety_stop_reason_buckets else None
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "read_only": True,
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "source_report_schema": promotion_report.get("schema") if isinstance(promotion_report, dict) else None,
        "summary": {
            "candidate_count": len(candidates),
            "action_count": len(actions),
            "omitted_count": len(omitted),
            "action_family_counts": _counter_rows(action_families),
            "policy_section_counts": _counter_rows(policy_sections),
            "omission_reason_counts": _counter_rows(omission_reasons),
            "omission_bucket_count": len(omission_buckets),
            "top_omission_next_action": top_omission_bucket.get("next_action") if top_omission_bucket else None,
            "top_omission_bucket": top_omission_bucket,
            "safety_stop_reason_bucket_count": len(safety_stop_reason_buckets),
            "top_safety_stop_reason": top_safety_stop_reason,
        },
        "actions": actions,
        "omission_buckets": omission_buckets,
        "safety_stop_reason_buckets": safety_stop_reason_buckets,
        "omitted": omitted,
        "privacy": _privacy_summary(),
    }
