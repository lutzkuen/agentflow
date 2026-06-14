from __future__ import annotations

from collections import Counter
from typing import Any

from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from agentflow_proxy.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report
from agentflow_proxy.openai_cache_replay_report import _as_float, _as_int, build_openai_cache_replay_report
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_blocker_outcomes.v1"

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
    "safe-invalidation-required",
    "tool-call-cache-disabled",
}
NOOP_REASONS = {
    "already-cache-hit",
    "canary-holdout-only",
    "no-openai-cache-replay-opportunity-observed",
    "no-openai-calls-observed",
    "no-runnable-canary-traffic",
    "staged-canary-policy-missing",
    "unsupported-streaming-shape",
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
    if reasons & MISSING_INVALIDATION_REASONS:
        return "missing-invalidation", sorted(reasons & MISSING_INVALIDATION_REASONS)[0]
    if verdict in {"widen", "promote"} or _as_int(cohorts.get("applied")) or _as_int(cohorts.get("holdout")):
        return "replay-ready", verdict or "cache-replay-lifecycle-observed"
    if verdict == "rollback" or _as_int(cohorts.get("safety_stop")):
        return "noop", "cache-replay-safety-stop"
    return "noop", verdict or "no-cache-replay-lifecycle-action"


def _top_next_action(outcome_counts: Counter[str], reason_counts: Counter[str]) -> str:
    if outcome_counts.get("stale-dependency"):
        return "refresh-cache-replay-dependency-evidence"
    if outcome_counts.get("missing-invalidation"):
        return "collect-safe-invalidation-evidence"
    if outcome_counts.get("replay-ready"):
        return "stage-local-cache-replay-canary"
    if outcome_counts.get("noop"):
        if reason_counts.get("unsupported-streaming-shape"):
            return "keep-streaming-cache-replay-noop"
        return "keep-cache-replay-noop"
    return "keep-cache-replay-observing"


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

    for candidate in opportunity.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        outcome, reason = _outcome_for_opportunity(candidate)
        _add_outcome(
            outcome_counts=outcome_counts,
            reason_counts=reason_counts,
            source_counts=source_counts,
            outcome=outcome,
            reason=reason,
            source="opportunity",
            amount=_as_int(candidate.get("matched_count")),
        )

    for candidate in impact.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        outcome, reason = _outcome_for_impact(candidate)
        _add_outcome(
            outcome_counts=outcome_counts,
            reason_counts=reason_counts,
            source_counts=source_counts,
            outcome=outcome,
            reason=reason,
            source="impact",
            amount=_as_int(candidate.get("sample_count")),
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

    top_next_action = _top_next_action(outcome_counts, reason_counts)
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
            "missing_invalidation_count": outcome_counts.get("missing-invalidation", 0),
            "noop_count": outcome_counts.get("noop", 0),
            "projected_savings_usd": round(
                max(_as_float(opp_summary.get("projected_savings_usd")), _as_float(impact_summary.get("projected_savings_usd"))),
                8,
            ),
            "observed_savings_usd": impact_summary.get("observed_savings_usd"),
            "top_next_action": top_next_action,
        },
        "outcome_breakdown": _counter_rows(outcome_counts, key="outcome"),
        "reason_breakdown": _counter_rows(reason_counts),
        "source_outcome_breakdown": _counter_rows(source_counts),
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
