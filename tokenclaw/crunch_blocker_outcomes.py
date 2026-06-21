from __future__ import annotations

from collections import Counter
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.crunch_blocker_outcomes.v1"

CRUNCH_LOCAL_ACTION_FAMILIES = {
    "crunch",
    "pattern",
    "old-context-summary",
    "old-context-summarization",
    "repeated-context-crunch",
    "repeated-context",
    "thinking-compaction",
    "instruction-dedup",
    "terminal-output-compaction",
    "anthropic-thinking-history-compaction",
    "anthropic-thinking-compaction",
}

_THINKING_FAMILIES = {
    "anthropic-thinking-history-compaction",
    "anthropic-thinking-compaction",
    "thinking-compaction",
    "thinking-deduplication",
}

_INSTRUCTION_DEDUP_FAMILIES = {
    "instruction-dedup",
    "instruction-deduplication",
    "old-context-summary",
    "old-context-summarization",
    "terminal-output-compaction",
}

_REPEATED_CONTEXT_FAMILIES = {
    "repeated-context",
    "repeated-context-crunch",
    "repeated-scaffold-crunch",
    "repeated-scaffold",
    "crunch",
    "pattern",
}


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "absolute_paths_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


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


def _counter_rows(counter: Counter[str], *, key: str = "value") -> list[dict[str, Any]]:
    return [
        {key: name, "count": count}
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count
    ]


def _crunch_sub_family(candidate_family: str, recommendation_type: str = "") -> str:
    f = str(candidate_family or "").lower().replace("_", "-")
    r = str(recommendation_type or "").lower().replace("_", "-")
    for text in (f, r):
        if text in _THINKING_FAMILIES or "thinking" in text:
            return "thinking-compaction"
        if text in _INSTRUCTION_DEDUP_FAMILIES or "instruction" in text or "dedup" in text or "old-context" in text or "terminal" in text:
            return "instruction-dedup"
        if text in _REPEATED_CONTEXT_FAMILIES or "repeated" in text:
            return "repeated-context"
    return "unsupported"


def _is_crunch_candidate(candidate: dict[str, Any]) -> bool:
    family = str(candidate.get("local_action_family") or "").lower().replace("_", "-")
    return family in CRUNCH_LOCAL_ACTION_FAMILIES or family == "unknown" and "crunch" in str(candidate.get("candidate_family") or "").lower()


def _lifecycle_from_candidate(candidate: dict[str, Any]) -> str:
    recommendation_type = str(candidate.get("recommendation_type") or "").lower().replace("_", "-")
    next_action = str(candidate.get("next_action") or "").lower().replace("_", "-")
    status = str(candidate.get("status") or "noop").lower()
    blocker_codes = set(str(b) for b in (candidate.get("blocker_reason_codes") or []))

    if "safety-stop" in recommendation_type or "safety-stop" in next_action or "safety-stop-observed" in blocker_codes:
        return "safety-stop"
    if "apply-local-rule" in recommendation_type or "apply-local-rule" in next_action:
        return "applied-local-rule"
    if "canary" in recommendation_type or "canary" in next_action:
        return "canary"
    if "dry-run" in recommendation_type or "dry-run" in next_action:
        return "dry-run"
    if status == "recommended":
        return "dry-run"
    return "no-op"


def _add_outcome(
    *,
    outcome_counts: Counter[str],
    reason_counts: Counter[str],
    family_counts: Counter[str],
    source_counts: Counter[str],
    lifecycle: str,
    blocker_reasons: list[str],
    sub_family: str,
    source: str,
    weight: int = 1,
) -> None:
    count = max(0, int(weight or 0))
    if not count:
        return
    outcome_counts[lifecycle] += count
    family_counts[sub_family] += count
    source_counts[f"{source}:{lifecycle}"] += count
    for reason in blocker_reasons:
        reason_counts[reason] += count


def _top_next_action(outcome_counts: Counter[str], reason_counts: Counter[str]) -> str:
    if outcome_counts.get("safety-stop"):
        return "investigate-crunch-safety-stop"
    if outcome_counts.get("applied-local-rule"):
        return "monitor-applied-crunch-rule"
    if outcome_counts.get("canary"):
        return "monitor-crunch-canary-lifecycle"
    if outcome_counts.get("dry-run"):
        return "stage-local-crunch-dry-run"
    if reason_counts.get("non-positive-projection"):
        return "collect-crunch-savings-evidence"
    return "keep-crunch-observing"


def build_crunch_blocker_outcomes_report(
    store_obj: Any,
    *,
    rollup_limit: int = 1000,
    promotion_blocker_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from tokenclaw.request_shape_rollups import build_request_shape_rollups_report

    rollup_report = build_request_shape_rollups_report(store_obj, limit=rollup_limit, persist=False)
    crunch_dry_run = rollup_report.get("crunch_opportunity_dry_run") if isinstance(rollup_report.get("crunch_opportunity_dry_run"), dict) else {}
    dry_run_summary = crunch_dry_run.get("summary") if isinstance(crunch_dry_run.get("summary"), dict) else {}

    outcome_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    projected_savings = 0.0
    observed_savings = 0.0

    for cohort in crunch_dry_run.get("cohorts") or []:
        if not isinstance(cohort, dict):
            continue
        lifecycle = "dry-run" if str(cohort.get("readiness") or "") == "measurement-ready" else "no-op"
        blocker_reasons = [str(b) for b in (cohort.get("blockers") or [])]
        if lifecycle == "no-op":
            reason_str = str(cohort.get("reason") or "")
            if reason_str and reason_str not in blocker_reasons:
                blocker_reasons = [reason_str] + blocker_reasons
        row_count = max(1, _as_int(cohort.get("row_count")))
        _add_outcome(
            outcome_counts=outcome_counts,
            reason_counts=reason_counts,
            family_counts=family_counts,
            source_counts=source_counts,
            lifecycle=lifecycle,
            blocker_reasons=blocker_reasons,
            sub_family="repeated-context",
            source="crunch-opportunity",
            weight=row_count,
        )
        if lifecycle == "dry-run":
            projected_savings += _as_float(cohort.get("projected_saved_usd"))
            observed_savings += _as_float(cohort.get("current_conservative_savings_usd"))

    blocker_review_candidate_count = 0
    blocker_review_schema = None
    if isinstance(promotion_blocker_review, dict):
        blocker_review_schema = promotion_blocker_review.get("schema")
        for candidate in promotion_blocker_review.get("candidates") or []:
            if not isinstance(candidate, dict) or not _is_crunch_candidate(candidate):
                continue
            blocker_review_candidate_count += 1
            lifecycle = _lifecycle_from_candidate(candidate)
            blocker_reasons = [str(b) for b in (candidate.get("blocker_reason_codes") or [])]
            sub_family = _crunch_sub_family(
                str(candidate.get("candidate_family") or ""),
                str(candidate.get("recommendation_type") or ""),
            )
            _add_outcome(
                outcome_counts=outcome_counts,
                reason_counts=reason_counts,
                family_counts=family_counts,
                source_counts=source_counts,
                lifecycle=lifecycle,
                blocker_reasons=blocker_reasons,
                sub_family=sub_family,
                source="promotion-blocker-review",
                weight=1,
            )
            proj = _as_float(candidate.get("projected_savings_usd"))
            if lifecycle in {"dry-run", "canary", "applied-local-rule"} and proj > 0:
                projected_savings += proj

    top_next_action = _top_next_action(outcome_counts, reason_counts)
    total_outcomes = sum(outcome_counts.values())

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "status": "matched" if total_outcomes else "no-crunch-outcomes",
        "top_next_action": top_next_action,
        "summary": {
            "crunch_opportunity_cohort_count": _as_int(dry_run_summary.get("candidate_count")),
            "promotion_blocker_crunch_candidate_count": blocker_review_candidate_count,
            "outcome_count": total_outcomes,
            "dry_run_count": outcome_counts.get("dry-run", 0),
            "canary_count": outcome_counts.get("canary", 0),
            "safety_stop_count": outcome_counts.get("safety-stop", 0),
            "applied_local_rule_count": outcome_counts.get("applied-local-rule", 0),
            "no_op_count": outcome_counts.get("no-op", 0),
            "projected_savings_usd": round(projected_savings, 8),
            "observed_savings_usd": round(
                observed_savings + _as_float(dry_run_summary.get("current_conservative_savings_usd")),
                8,
            ),
            "top_next_action": top_next_action,
        },
        "lifecycle_breakdown": _counter_rows(outcome_counts, key="lifecycle"),
        "reason_breakdown": _counter_rows(reason_counts),
        "family_breakdown": _counter_rows(family_counts, key="crunch_family"),
        "source_lifecycle_breakdown": _counter_rows(source_counts),
        "source_reports": {
            "crunch_opportunity_dry_run_schema": crunch_dry_run.get("schema"),
            "promotion_blocker_review_schema": blocker_review_schema,
            "rollup_limit": rollup_report.get("limit"),
            "raw_source_reports_included": False,
            "individual_candidate_ids_included": False,
        },
        "privacy": {
            **_privacy_summary(),
            "basis": "aggregate crunch opportunity cohort and promotion blocker recommendation metadata only",
        },
    }
