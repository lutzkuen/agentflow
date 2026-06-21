from __future__ import annotations

from collections import Counter
from typing import Any

from tokenclaw.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from tokenclaw.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report
from tokenclaw.openai_cache_replay_report import _as_float, _as_int, build_openai_cache_replay_report
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.openai_cache_replay_blocker_outcomes.v1"

READY_REASONS = {
    "replay-rule-required",
    "exact-miss",
}
STALE_DEPENDENCY_REASONS = {
    "dependency-cap-exceeded",
    "dependency-changed",
    "dependency-created",
    "dependency-deleted",
    "file-dependency-changed",
    "file-dependency-invalidated",
    "stale-dependency-blocker",
    "stale-risk-blockers",
}
MISSING_INVALIDATION_REASONS = {
    "dependency-audit-missing",
    "dependency-missing",
    "file-dependency-missing",
    "file-watch-disabled",
    "invalidation-evidence-missing",
    "safe-invalidation-required",
    "tool-call-cache-disabled",
    "unsafe-tool-calls-without-invalidation",
}
UNSAFE_DEPENDENCY_REASONS = {
    "unsafe-dependency-evidence",
    "unsafe-tool-calls-without-invalidation",
}
NOOP_REASONS = {
    "already-cache-hit",
    "canary-holdout-only",
    "streaming-replay-not-supported",
    "no-openai-cache-replay-opportunity-observed",
    "no-openai-calls-observed",
    "no-runnable-canary-traffic",
    "staged-canary-policy-missing",
    "unsupported-streaming-shape",
}
SUPPORTED_REPLAY_ENDPOINTS = {"responses", "chat_completions"}
OUTCOME_PRIORITY = {
    "replay-ready": 4,
    "stale-dependency": 3,
    "unsafe-dependency": 3,
    "unknown-dependency": 2,
    "missing-invalidation": 2,
    "noop": 1,
}


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "tool_payloads_included": False,
        "absolute_paths_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "request_fingerprints_included": False,
        "file_dependency_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _counter_rows(counter: Counter[str] | dict[str, int], *, key: str = "value") -> list[dict[str, Any]]:
    return [
        {key: name, "count": count}
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count
    ]


def _count_rows(rows: Any, *, key: str = "value") -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if not isinstance(row, dict):
            continue
        counts[str(row.get(key) or "unknown")] += _as_int(row.get("count"))
    return counts


def _candidate_reasons(candidate: dict[str, Any]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for key in ("blocker_reason_breakdown", "invalidation_reason_breakdown", "reason_code_breakdown"):
        reasons.update(_count_rows(candidate.get(key)))
    for key in ("blockers", "reason_codes", "warning_codes"):
        values = candidate.get(key)
        if isinstance(values, list):
            for value in values:
                reasons[str(value or "unknown")] += 1
    for key in ("cache_reason", "verdict"):
        value = candidate.get(key)
        if value:
            reasons[str(value)] += 1
    return reasons


def _outcome_for_opportunity(candidate: dict[str, Any]) -> tuple[str, str]:
    reasons = set(_candidate_reasons(candidate))
    safety_eligible = _as_int(candidate.get("safety_eligible_count"))
    projected = _as_float(candidate.get("projected_savings_usd"))
    has_tools = bool(candidate.get("has_tools"))
    stream = bool(candidate.get("stream"))
    endpoint = str(candidate.get("endpoint") or "")
    dependency_status = str(candidate.get("file_dependency_status") or "")
    audit = candidate.get("file_dependency_audit") if isinstance(candidate.get("file_dependency_audit"), dict) else {}
    safe_dependency = bool(audit.get("safe_invalidation_evidence") or dependency_status == "stable")
    invalidation_reason = str(audit.get("invalidation_reason") or "")
    if stream or endpoint and endpoint not in SUPPORTED_REPLAY_ENDPOINTS:
        if stream:
            return "noop", "unsupported-streaming-shape"
        return "noop", "unsupported-endpoint"
    if has_tools:
        if dependency_status == "invalidated" or invalidation_reason in STALE_DEPENDENCY_REASONS:
            return "stale-dependency", invalidation_reason or "stale-dependency-evidence"
        if safe_dependency:
            return "replay-ready", "safe-invalidation-evidence-present"
        if dependency_status == "unsafe" or reasons & UNSAFE_DEPENDENCY_REASONS:
            return "unsafe-dependency", "unsafe-tool-calls-without-invalidation"
        if dependency_status == "unknown":
            return "unknown-dependency", "dependency-evidence-unknown"
        if dependency_status == "missing" or reasons & MISSING_INVALIDATION_REASONS:
            return "missing-invalidation", "invalidation-evidence-missing"
        return "unknown-dependency", "dependency-evidence-unknown"
    if reasons & STALE_DEPENDENCY_REASONS:
        return "stale-dependency", sorted(reasons & STALE_DEPENDENCY_REASONS)[0]
    if reasons & MISSING_INVALIDATION_REASONS:
        return "missing-invalidation", sorted(reasons & MISSING_INVALIDATION_REASONS)[0]
    if safety_eligible > 0 or projected > 0:
        return "replay-ready", "safe-replay-candidate"
    if reasons & NOOP_REASONS:
        return "noop", sorted(reasons & NOOP_REASONS)[0]
    if reasons <= READY_REASONS and _as_int(candidate.get("matched_count")):
        return "replay-ready", "exact-replay-shape-observed"
    return "noop", "no-runnable-cache-replay-action"


def _outcome_for_impact(candidate: dict[str, Any]) -> tuple[str, str]:
    reasons = set(_candidate_reasons(candidate))
    verdict = str(candidate.get("verdict") or "")
    cohorts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    if reasons & STALE_DEPENDENCY_REASONS or _as_int(cohorts.get("invalidated")):
        return "stale-dependency", sorted((reasons & STALE_DEPENDENCY_REASONS) or {"invalidated-cohort-observed"})[0]
    if reasons & UNSAFE_DEPENDENCY_REASONS:
        return "unsafe-dependency", "unsafe-tool-calls-without-invalidation"
    if reasons & MISSING_INVALIDATION_REASONS:
        return "missing-invalidation", sorted(reasons & MISSING_INVALIDATION_REASONS)[0]
    if verdict in {"widen", "promote"} or _as_int(cohorts.get("applied")) or _as_int(cohorts.get("holdout")):
        return "replay-ready", verdict or "cache-replay-lifecycle-observed"
    if verdict == "rollback" or _as_int(cohorts.get("safety_stop")):
        return "noop", "cache-replay-safety-stop"
    return "noop", verdict or "no-cache-replay-lifecycle-action"


def _top_next_action(outcome_counts: Counter[str], reason_counts: Counter[str]) -> str:
    if outcome_counts.get("replay-ready"):
        return "stage-local-cache-replay-canary"
    if outcome_counts.get("stale-dependency"):
        return "refresh-cache-replay-dependency-evidence"
    if outcome_counts.get("unknown-dependency"):
        return "collect-safe-invalidation-evidence"
    if outcome_counts.get("missing-invalidation"):
        return "collect-safe-invalidation-evidence"
    if outcome_counts.get("noop"):
        if reason_counts.get("unsupported-streaming-shape"):
            return "keep-streaming-cache-replay-noop"
        return "keep-cache-replay-noop"
    return "keep-cache-replay-observing"


def _next_action_for_outcome(outcome: str, reason: str) -> str:
    if outcome == "replay-ready":
        return "stage-local-cache-replay-canary"
    if outcome == "stale-dependency":
        return "refresh-cache-replay-dependency-evidence"
    if outcome == "unsafe-dependency":
        return "collect-safe-invalidation-evidence"
    if outcome == "unknown-dependency":
        return "collect-safe-invalidation-evidence"
    if outcome == "missing-invalidation":
        return "collect-safe-invalidation-evidence"
    if reason in {"unsupported-streaming-shape", "streaming-replay-not-supported"}:
        return "keep-streaming-cache-replay-noop"
    return "keep-cache-replay-noop"


def _safe_audit(candidate: dict[str, Any]) -> dict[str, Any] | None:
    audit = candidate.get("file_dependency_audit")
    if not isinstance(audit, dict):
        return None
    allowed = (
        "schema",
        "file_watch_enabled",
        "snapshot_root_policy",
        "root_path_included",
        "snapshot_count",
        "snapshot_count_bucket",
        "candidate_path_count_bucket",
        "raw_candidate_path_count_bucket",
        "distinct_candidate_path_count_bucket",
        "max_paths",
        "cap_exceeded",
        "cap_trimmed",
        "dependency_capture_reason",
        "present_path_count",
        "missing_path_count",
        "changed_path_count",
        "deleted_path_count",
        "created_path_count",
        "invalidation_reason",
        "safe_invalidation_evidence",
        "file_dependency_evidence_available",
        "paths_included",
    )
    sanitized = {key: audit.get(key) for key in allowed if key in audit}
    sanitized["paths_included"] = False
    sanitized["root_path_included"] = False
    return sanitized


def _cohort_row(*, source: str, candidate: dict[str, Any], outcome: str, reason: str, amount: int) -> dict[str, Any]:
    audit = _safe_audit(candidate)
    return {
        "schema": "tokenclaw.openai_cache_replay_blocker_outcome_cohort.v1",
        "rank": 0,
        "source": source,
        "outcome": outcome,
        "reason": reason,
        "next_action": _next_action_for_outcome(outcome, reason),
        "sample_count": max(0, _as_int(amount)),
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "stream": bool(candidate.get("stream")),
        "has_tools": bool(candidate.get("has_tools")),
        "text_bucket": candidate.get("text_bucket"),
        "token_bucket": candidate.get("token_bucket"),
        "cache_status": candidate.get("cache_status"),
        "cache_reason": candidate.get("cache_reason"),
        "replayability_level": candidate.get("replayability_level"),
        "file_dependency_status": candidate.get("file_dependency_status"),
        "file_dependency_fingerprint_available": bool(candidate.get("file_dependency_fingerprint_available")),
        "file_dependency_audit": audit,
        "safe_invalidation_evidence": bool((audit or {}).get("safe_invalidation_evidence")),
        "projected_hits": _as_int(candidate.get("projected_hits")),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "observed_savings_usd": (
            round(_as_float(candidate.get("observed_savings_usd")), 8)
            if candidate.get("observed_savings_usd") is not None
            else None
        ),
        "blocker_codes": sorted(_candidate_reasons(candidate)),
        "tool_cache_replay_enabled": False,
        "streaming_replay_enabled": False,
        "emits_cache_apply_action": False,
        "policy_files_written": False,
        "aggregate_only": True,
        "metadata_only": True,
        "privacy": _privacy_summary(),
    }


def _add_outcome(
    *,
    outcome_counts: Counter[str],
    reason_counts: Counter[str],
    source_counts: Counter[str],
    outcome: str,
    reason: str,
    source: str,
    amount: int,
) -> None:
    count = max(0, int(amount or 0))
    if not count:
        return
    outcome_counts[outcome] += count
    reason_counts[reason] += count
    source_counts[f"{source}:{outcome}"] += count


def build_openai_cache_replay_blocker_outcomes_report(
    store_obj: Any,
    *,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
) -> dict[str, Any]:
    opportunity = build_openai_cache_replay_report(store_obj, limit=opportunity_limit)
    impact = build_openai_cache_replay_impact_report(store_obj, limit=impact_limit)
    readiness = build_openai_cache_replay_readiness_report(
        store_obj,
        opportunity_limit=opportunity_limit,
        impact_limit=impact_limit,
    )
    opp_summary = opportunity.get("summary") if isinstance(opportunity.get("summary"), dict) else {}
    impact_summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}

    outcome_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    cohorts: list[dict[str, Any]] = []

    for candidate in opportunity.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        outcome, reason = _outcome_for_opportunity(candidate)
        amount = _as_int(candidate.get("matched_count"))
        _add_outcome(
            outcome_counts=outcome_counts,
            reason_counts=reason_counts,
            source_counts=source_counts,
            outcome=outcome,
            reason=reason,
            source="opportunity",
            amount=amount,
        )
        if amount:
            cohorts.append(
                _cohort_row(
                    source="opportunity",
                    candidate=candidate,
                    outcome=outcome,
                    reason=reason,
                    amount=amount,
                )
            )

    for candidate in impact.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        outcome, reason = _outcome_for_impact(candidate)
        amount = _as_int(candidate.get("sample_count"))
        _add_outcome(
            outcome_counts=outcome_counts,
            reason_counts=reason_counts,
            source_counts=source_counts,
            outcome=outcome,
            reason=reason,
            source="impact",
            amount=amount,
        )
        if amount:
            cohorts.append(
                _cohort_row(
                    source="impact",
                    candidate=candidate,
                    outcome=outcome,
                    reason=reason,
                    amount=amount,
                )
            )

    staged_policy = (
        ((readiness.get("lifecycle_diagnostics") or {}).get("staged_canary_policy") or {})
        if isinstance(readiness.get("lifecycle_diagnostics"), dict)
        else {}
    )
    staged_status = str(staged_policy.get("status") or "")
    if staged_status and staged_status != "staged-policy-can-run":
        for reason in staged_policy.get("blockers") or [staged_status]:
            reason_text = str(reason or staged_status)
            _add_outcome(
                outcome_counts=outcome_counts,
                reason_counts=reason_counts,
                source_counts=source_counts,
                outcome="noop" if reason_text in NOOP_REASONS else "missing-invalidation",
                reason=reason_text,
                source="staged-policy",
                amount=1,
            )
            cohorts.append(
                _cohort_row(
                    source="staged-policy",
                    candidate={
                        "source_surface": "local-policy",
                        "endpoint": "cache_rules",
                        "category": "cache-replay",
                        "workflow_phase": "policy-readiness",
                    },
                    outcome="noop" if reason_text in NOOP_REASONS else "missing-invalidation",
                    reason=reason_text,
                    amount=1,
                )
            )

    cohorts.sort(
        key=lambda row: (
            OUTCOME_PRIORITY.get(str(row.get("outcome")), 0),
            _as_float(row.get("projected_savings_usd")),
            _as_int(row.get("projected_hits")),
            _as_int(row.get("sample_count")),
            str(row.get("source_surface") or ""),
            str(row.get("endpoint") or ""),
            str(row.get("category") or ""),
        ),
        reverse=True,
    )
    for rank, row in enumerate(cohorts, start=1):
        row["rank"] = rank
    top_next_action = str(cohorts[0].get("next_action")) if cohorts else _top_next_action(outcome_counts, reason_counts)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "status": "matched" if sum(outcome_counts.values()) else "no-cache-replay-outcomes",
        "top_next_action": top_next_action,
        "summary": {
            "openai_call_count": _as_int(opp_summary.get("openai_call_count")),
            "opportunity_candidate_count": _as_int(opp_summary.get("candidate_count")),
            "impact_candidate_count": _as_int(impact_summary.get("candidate_count")),
            "observed_replay_metadata_rows": _as_int(impact_summary.get("observed_openai_cache_replay_metadata_row_count")),
            "outcome_count": sum(outcome_counts.values()),
            "replay_ready_count": outcome_counts.get("replay-ready", 0),
            "stale_dependency_count": outcome_counts.get("stale-dependency", 0),
            "unsafe_dependency_count": outcome_counts.get("unsafe-dependency", 0),
            "unknown_dependency_count": outcome_counts.get("unknown-dependency", 0),
            "missing_invalidation_count": outcome_counts.get("missing-invalidation", 0),
            "noop_count": outcome_counts.get("noop", 0),
            "staged_canary_count": 1 if staged_status == "staged-policy-can-run" else 0,
            "staged_canary_policy_status": staged_status or None,
            "applied_count": _as_int(impact_summary.get("applied_count")),
            "holdout_count": _as_int(impact_summary.get("holdout_count")),
            "exact_hit_count": _as_int(impact_summary.get("actual_hits")),
            "safety_stop_count": _as_int(impact_summary.get("safety_stop_count")),
            "invalidated_count": _as_int(impact_summary.get("invalidated_count")),
            "projected_savings_usd": round(
                max(_as_float(opp_summary.get("projected_savings_usd")), _as_float(impact_summary.get("projected_savings_usd"))),
                8,
            ),
            "observed_savings_usd": impact_summary.get("observed_savings_usd"),
            "top_next_action": top_next_action,
            "ranked_cohort_count": len(cohorts),
        },
        "cohorts": cohorts,
        "outcome_breakdown": _counter_rows(outcome_counts, key="outcome"),
        "reason_breakdown": _counter_rows(reason_counts),
        "source_outcome_breakdown": _counter_rows(source_counts),
        "acceptance": {
            "emits_ranked_replay_ready_stale_and_missing_cohorts": all(
                outcome in {row.get("outcome") for row in cohorts}
                for outcome in ("replay-ready", "stale-dependency", "missing-invalidation")
            ),
            "emits_ranked_dependency_evidence_classes": all(
                outcome in {row.get("outcome") for row in cohorts}
                for outcome in ("replay-ready", "stale-dependency", "unsafe-dependency", "missing-invalidation")
            ),
            "safe_rows_stage_local_cache_replay_canary": all(
                row.get("next_action") == "stage-local-cache-replay-canary"
                for row in cohorts
                if row.get("outcome") == "replay-ready" and row.get("safe_invalidation_evidence")
            ),
            "stale_rows_refresh_dependency_evidence": all(
                row.get("next_action") == "refresh-cache-replay-dependency-evidence"
                for row in cohorts
                if row.get("outcome") == "stale-dependency"
            ),
            "unsafe_rows_collect_safe_invalidation_evidence": all(
                row.get("next_action") == "collect-safe-invalidation-evidence"
                for row in cohorts
                if row.get("outcome") == "unsafe-dependency"
            ),
            "unknown_rows_collect_safe_invalidation_evidence": all(
                row.get("next_action") == "collect-safe-invalidation-evidence"
                for row in cohorts
                if row.get("outcome") == "unknown-dependency"
            ),
            "missing_rows_collect_safe_invalidation_evidence": all(
                row.get("next_action") == "collect-safe-invalidation-evidence"
                for row in cohorts
                if row.get("outcome") == "missing-invalidation"
            ),
            "distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence": all(
                outcome in {row.get("outcome") for row in cohorts}
                for outcome in (
                    "replay-ready",
                    "stale-dependency",
                    "unsafe-dependency",
                    "unknown-dependency",
                    "missing-invalidation",
                )
            ),
            "no_policy_files_written": True,
            "metadata_only": True,
            "aggregate_only": True,
        },
        "source_reports": {
            "opportunity_schema": opportunity.get("schema"),
            "impact_schema": impact.get("schema"),
            "readiness_schema": readiness.get("schema"),
            "opportunity_limit": opportunity.get("limit"),
            "impact_limit": impact.get("lookback_limit"),
            "raw_source_reports_included": False,
            "individual_candidate_ids_included": False,
        },
        "privacy": {
            **_privacy_summary(),
            "basis": "aggregate cache replay opportunity, readiness, impact, dependency, and no-op metadata only",
        },
    }
