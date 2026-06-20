from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "agentflow.local_promotion_candidates.v1"
CANDIDATE_SCHEMA = "agentflow.local_promotion_candidate.v1"

READY_VERDICTS = {"promote", "promotion-ready", "widen", "widen-ready"}
NON_BLOCKING_REASON_CODES = {"target-savings-met", "canary-full-coverage", "promotion-ready", "widen-ready"}
LOCAL_TARGETS = {
    "cache": {
        "target_local_rule_file": "cache_rules.yaml",
        "target_local_policy_section": "cache.rules",
        "required_local_executor": "openai-cache-replay-canary",
    },
    "crunch": {
        "target_local_rule_file": "crunch_rules.yaml",
        "target_local_policy_section": "crunch.rules",
        "required_local_executor": "request-shape-crunch-canary",
    },
    "routing": {
        "target_local_rule_file": "routing_rules.yaml",
        "target_local_policy_section": "routing.rules",
        "required_local_executor": "routing-canary",
    },
}

_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "local_only": True,
        "read_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "request_fingerprints_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
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


def _round(value: Any, places: int = 8) -> float:
    return round(_as_float(value), places)


def _reason(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", "-")
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
        elif isinstance(value, dict):
            for row in value.values():
                reason = _reason(row)
                if reason:
                    reasons.add(reason)
        else:
            reason = _reason(value)
            if reason:
                reasons.add(reason)
    return sorted(reasons)


def _breakdown(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _cohort_count(candidate: dict[str, Any], *names: str) -> int:
    for name in names:
        if name in candidate:
            return _as_int(candidate.get(name))
    counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
    for name in names:
        if name in counts:
            return _as_int(counts.get(name))
    metrics = candidate.get("cohort_metrics") if isinstance(candidate.get("cohort_metrics"), dict) else {}
    for name in names:
        value = metrics.get(name)
        if isinstance(value, dict):
            return _as_int(value.get("count"))
    return 0


def _blockers(
    *,
    candidate: dict[str, Any],
    applied_count: int,
    holdout_count: int,
    safety_stop_count: int,
    extra: list[str] | None = None,
) -> list[str]:
    reasons = set(extra or [])
    reasons.update(_reason_list(candidate.get("reason_codes"), candidate.get("blocker_codes")))
    top = _reason(candidate.get("top_blocker") or candidate.get("top_remaining_blocker"))
    if top:
        reasons.add(top)
    for key in ("blocker_reason_breakdown", "remaining_blocker_breakdown", "reason_code_breakdown"):
        rows = candidate.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    reason = _reason(row.get("value") or row.get("reason"))
                    if reason:
                        reasons.add(reason)
    if applied_count <= 0:
        reasons.add("missing-applied-evidence")
    if holdout_count <= 0:
        reasons.add("missing-holdout-evidence")
    if safety_stop_count > 0:
        reasons.add("safety-stop-observed")
    stale = candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {}
    if stale.get("stale"):
        reasons.add("stale-evidence")
    return sorted(reason for reason in reasons if reason not in NON_BLOCKING_REASON_CODES)


def _promotion_ready(verdict: str, *, applied_count: int, holdout_count: int, safety_stop_count: int, blockers: list[str]) -> bool:
    if verdict not in READY_VERDICTS:
        return False
    if applied_count <= 0 or holdout_count <= 0 or safety_stop_count > 0:
        return False
    blocking_markers = (
        "missing-",
        "insufficient-",
        "safety-stop",
        "rollback",
        "stale-evidence",
        "error-rate",
        "retry-rate",
        "latency-regression",
        "negative-",
    )
    return not any(any(reason.startswith(marker) or marker in reason for marker in blocking_markers) for reason in blockers)


def _readiness_state(promotion_ready: bool, *, applied_count: int, holdout_count: int, safety_stop_count: int, projected: float) -> str:
    if promotion_ready:
        return "promotion-ready"
    if applied_count or holdout_count or safety_stop_count:
        return "blocked"
    if projected > 0:
        return "projected"
    return "missing-evidence"


def _next_action(action_family: str, promotion_ready: bool, blockers: list[str], fallback: str | None = None) -> str:
    if promotion_ready:
        return {
            "cache": "promote-openai-cache-replay-rule-draft",
            "crunch": "promote-repeated-context-crunch-rule-draft",
            "routing": "promote-routing-rule-draft",
        }.get(action_family, "promote-local-policy-rule-draft")
    if any("safety-stop" in reason for reason in blockers):
        return f"review-{action_family}-safety-stop"
    if any("holdout" in reason for reason in blockers):
        return f"collect-{action_family}-holdout-evidence"
    if any("applied" in reason or "sample" in reason for reason in blockers):
        return f"collect-{action_family}-canary-evidence"
    return fallback or f"stage-{action_family}-promotion-canary"


def _candidate(
    *,
    action_family: str,
    source_schema: str | None,
    source_status: str | None,
    source_rank: int,
    candidate: dict[str, Any],
    observed_savings_usd: float,
    projected_savings_usd: float,
    sample_count: int,
    applied_count: int,
    holdout_count: int,
    safety_stop_count: int,
    verdict: str,
    next_action_fallback: str | None = None,
    extra_blockers: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = _blockers(
        candidate=candidate,
        applied_count=applied_count,
        holdout_count=holdout_count,
        safety_stop_count=safety_stop_count,
        extra=extra_blockers,
    )
    ready = _promotion_ready(
        verdict,
        applied_count=applied_count,
        holdout_count=holdout_count,
        safety_stop_count=safety_stop_count,
        blockers=blockers,
    )
    target = LOCAL_TARGETS[action_family]
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "rank": 0,
        "action_family": action_family,
        "source_evidence_schema": source_schema,
        "source_evidence_status": source_status,
        "source_rank": source_rank,
        "promotion_ready": ready,
        "readiness_state": _readiness_state(
            ready,
            applied_count=applied_count,
            holdout_count=holdout_count,
            safety_stop_count=safety_stop_count,
            projected=projected_savings_usd,
        ),
        "verdict": verdict or "unknown",
        "next_action": _next_action(action_family, ready, blockers, next_action_fallback),
        "no_op_reason": None if ready else (blockers[0] if blockers else "promotion-evidence-not-ready"),
        "target_local_rule_file": target["target_local_rule_file"],
        "target_local_policy_section": target["target_local_policy_section"],
        "required_local_executor": target["required_local_executor"],
        "provider": candidate.get("provider"),
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "requested_model": candidate.get("requested_model") or candidate.get("original_model"),
        "target_model": candidate.get("target_model") or candidate.get("candidate_target_model"),
        "sample_count": sample_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "safety_stop_count": safety_stop_count,
        "observed_savings_usd": round(observed_savings_usd, 8),
        "projected_savings_usd": round(projected_savings_usd, 8),
        "blocker_codes": blockers,
        "privacy": _privacy_summary(),
    }
    if extras:
        payload.update(extras)
    return payload


def _cache_candidates(cache_impact: dict[str, Any], rollups: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_schema = cache_impact.get("schema")
    source_status = str(cache_impact.get("status") or "")
    for source_rank, item in enumerate(cache_impact.get("candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        applied = _cohort_count(item, "applied_count", "applied")
        holdout = _cohort_count(item, "holdout_count", "holdout")
        safety = _cohort_count(item, "safety_stop_count", "safety_stop")
        projected_hits = _as_int(item.get("projected_hits") or item.get("projected_hit_count"))
        actual_hits = _as_int(item.get("actual_hits") or item.get("actual_hit_count"))
        measurement = item.get("canary_hit_measurement") if isinstance(item.get("canary_hit_measurement"), dict) else {}
        candidates.append(
            _candidate(
                action_family="cache",
                source_schema=str(source_schema or ""),
                source_status=source_status,
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=_as_float(item.get("actual_saved_cost_usd"), _as_float(item.get("observed_savings_usd"))),
                projected_savings_usd=_as_float(item.get("projected_saved_usd"), _as_float(item.get("projected_savings_usd"))),
                sample_count=_as_int(item.get("sample_count"), applied + holdout),
                applied_count=applied,
                holdout_count=holdout,
                safety_stop_count=safety,
                verdict=str(item.get("verdict") or ""),
                next_action_fallback=str(item.get("next_action") or "collect_more_applied_and_holdout_cache_replay_evidence"),
                extras={
                    "projected_hits": projected_hits,
                    "actual_hits": actual_hits,
                    "invalidated_count": _as_int(item.get("invalidated_count")),
                    "miss_count": _as_int(item.get("miss_count")),
                    "stream": item.get("stream"),
                    "has_tools": item.get("has_tools"),
                    "text_bucket": item.get("text_bucket"),
                    "token_bucket": item.get("token_bucket"),
                    "replay_source_schema": item.get("replay_source_schema"),
                    "replay_ready": item.get("replay_ready"),
                    "readiness": item.get("readiness"),
                    "dry_run_projected_savings_usd": _round(item.get("dry_run_projected_savings_usd"), 8),
                    "canary_hit_measurement": {
                        "hit_realization_rate": measurement.get("hit_realization_rate"),
                        "savings_realization_rate": measurement.get("savings_realization_rate"),
                    },
                    "first_observed_at": item.get("oldest_observed_at") or item.get("first_observed_at"),
                    "last_observed_at": item.get("latest_observed_at") or item.get("last_observed_at"),
                },
            )
        )

    replay = rollups.get("cache_replayability_dry_run") if isinstance(rollups.get("cache_replayability_dry_run"), dict) else {}
    for source_rank, item in enumerate(replay.get("cohorts") or [], start=1):
        if not isinstance(item, dict):
            continue
        projected = _as_float(item.get("projected_savings_usd"))
        if projected <= 0:
            continue
        candidates.append(
            _candidate(
                action_family="cache",
                source_schema=str(replay.get("schema") or ""),
                source_status=str(replay.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=0.0,
                projected_savings_usd=projected,
                sample_count=_as_int(item.get("row_count")),
                applied_count=0,
                holdout_count=0,
                safety_stop_count=0,
                verdict="projected",
                next_action_fallback="stage-cache-replay-canary",
                extra_blockers=["missing-measured-cache-canary-impact", *_reason_list(item.get("blockers"))],
                extras={
                    "projected_hits": _as_int(item.get("projected_hits")),
                    "stream": item.get("stream"),
                    "has_tools": item.get("has_tools"),
                    "text_bucket": item.get("text_bucket"),
                    "token_bucket": item.get("token_bucket"),
                    "replay_source_schema": replay.get("schema"),
                    "replay_ready": item.get("readiness") == "replay-ready",
                    "readiness": item.get("readiness"),
                },
            )
        )
    return candidates


def _crunch_candidates(rollups: dict[str, Any], thinking_impact: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    impact_report = thinking_impact if isinstance(thinking_impact, dict) else {}
    for source_rank, item in enumerate(impact_report.get("candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        cohorts = item.get("cohorts") if isinstance(item.get("cohorts"), dict) else {}
        applied = _as_int((cohorts.get("applied") or {}).get("count"))
        holdout = _as_int((cohorts.get("holdout") or {}).get("count"))
        safety = _as_int((cohorts.get("safety_stop") or {}).get("count"))
        verdict = str(item.get("canary_impact_decision") or item.get("verdict") or "")
        candidates.append(
            _candidate(
                action_family="crunch",
                source_schema=str(impact_report.get("schema") or ""),
                source_status=str(impact_report.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=_as_float(item.get("observed_saved_usd")),
                projected_savings_usd=_as_float(item.get("projected_saved_usd")),
                sample_count=applied + holdout + safety + _as_int((cohorts.get("skipped") or {}).get("count")),
                applied_count=applied,
                holdout_count=holdout,
                safety_stop_count=safety,
                verdict=verdict,
                next_action_fallback="review-repeated-context-crunch-canary-impact-blocker",
                extras={
                    "projected_saved_tokens": _as_int(item.get("projected_saved_tokens")),
                    "observed_saved_tokens": _as_int(item.get("observed_saved_tokens")),
                    "avg_crunch_ratio": _round(item.get("avg_crunch_ratio"), 6),
                    "stream": item.get("stream"),
                    "requested_model_family": item.get("requested_model_family"),
                    "routed_model_family": item.get("routed_model_family"),
                    "first_observed_at": item.get("first_observed_at"),
                    "last_observed_at": item.get("last_observed_at"),
                    "canary_impact_decision": item.get("canary_impact_decision"),
                    "budget_governor_action": ((item.get("budget_governor_feedback") or {}).get("recommended_budget_action"))
                    if isinstance(item.get("budget_governor_feedback"), dict)
                    else None,
                },
            )
        )

    if candidates:
        return candidates

    impact = rollups.get("crunch_canary_impact") if isinstance(rollups.get("crunch_canary_impact"), dict) else {}
    for source_rank, item in enumerate(impact.get("candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        applied = _cohort_count(item, "applied_count", "canary_applied")
        holdout = _cohort_count(item, "holdout_count", "canary_holdout")
        safety = _cohort_count(item, "safety_stop_count", "safety_stopped")
        candidates.append(
            _candidate(
                action_family="crunch",
                source_schema=str(impact.get("schema") or ""),
                source_status=str(impact.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=_as_float(item.get("saved_usd")),
                projected_savings_usd=_as_float(item.get("projected_saved_usd"), _as_float(item.get("saved_usd"))),
                sample_count=_as_int(item.get("observed_count"), applied + holdout),
                applied_count=applied,
                holdout_count=holdout,
                safety_stop_count=safety,
                verdict=str(item.get("verdict") or ""),
                next_action_fallback=str(item.get("next_action") or "review-repeated-context-crunch-canary-impact-blocker"),
                extras={
                    "projected_saved_tokens": _as_int(item.get("projected_saved_tokens"), _as_int(item.get("saved_tokens"))),
                    "observed_saved_tokens": _as_int(item.get("saved_tokens")),
                    "fallback_count": _as_int(item.get("fallback_count")),
                    "rollback_count": _as_int(item.get("rollback_count")),
                },
            )
        )

    if candidates:
        return candidates

    dry_run = rollups.get("crunch_opportunity_dry_run") if isinstance(rollups.get("crunch_opportunity_dry_run"), dict) else {}
    for source_rank, item in enumerate(dry_run.get("cohorts") or [], start=1):
        if not isinstance(item, dict):
            continue
        projected = _as_float(item.get("projected_savings_usd"), _as_float(item.get("projected_saved_usd")))
        if projected <= 0:
            continue
        candidates.append(
            _candidate(
                action_family="crunch",
                source_schema=str(dry_run.get("schema") or ""),
                source_status=str(dry_run.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=_as_float(item.get("current_conservative_savings_usd")),
                projected_savings_usd=projected,
                sample_count=_as_int(item.get("row_count"), _as_int(item.get("matched_count"))),
                applied_count=0,
                holdout_count=0,
                safety_stop_count=0,
                verdict="projected",
                next_action_fallback="stage-repeated-context-crunch-canary",
                extra_blockers=["missing-measured-crunch-canary-impact"],
                extras={"projected_saved_tokens": _as_int(item.get("projected_saved_tokens"))},
            )
        )
    return candidates


def _routing_candidates(claude_impact: dict[str, Any], openai_routing: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_rank, item in enumerate(claude_impact.get("candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        applied = _cohort_count(item, "canary_applied_count", "canary_applied")
        holdout = _cohort_count(item, "canary_holdout_count", "canary_holdout")
        safety = _cohort_count(item, "safety_stopped_count", "safety_stopped")
        candidates.append(
            _candidate(
                action_family="routing",
                source_schema=str(claude_impact.get("schema") or ""),
                source_status=str(claude_impact.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=_as_float(item.get("observed_savings_usd")),
                projected_savings_usd=_as_float(item.get("projected_savings_usd"), _as_float(item.get("observed_savings_usd"))),
                sample_count=_as_int(item.get("sample_count"), applied + holdout),
                applied_count=applied,
                holdout_count=holdout,
                safety_stop_count=safety,
                verdict=str(item.get("verdict") or ""),
                next_action_fallback="collect-routing-canary-evidence",
                extras={
                    "policy_id": item.get("policy_id"),
                    "workflow_phase_confidence": item.get("workflow_phase_confidence"),
                    "stream": item.get("stream"),
                    "canary_fraction": item.get("canary_fraction"),
                    "holdout_fraction": item.get("holdout_fraction"),
                    "oldest_observed_at": item.get("oldest_observed_at"),
                    "latest_observed_at": item.get("latest_observed_at"),
                    "last_observed_at": item.get("latest_observed_at"),
                    "stale_evidence": item.get("stale_evidence"),
                    "reason_codes": item.get("reason_codes"),
                    "stripped_param_counts": item.get("stripped_param_counts"),
                    "safety_skip_counts": item.get("safety_skip_counts"),
                    "cohort_counts": item.get("cohort_counts"),
                    "cohort_metrics": item.get("cohort_metrics"),
                    "applied_vs_holdout_deltas": item.get("applied_vs_holdout_deltas"),
                    "fallback_count": _cohort_count(item, "fallback_count"),
                    "retry_count": _cohort_count(item, "retry_count"),
                    "error_count": sum(
                        _as_int((row or {}).get("error_count"))
                        for row in (item.get("cohort_metrics") or {}).values()
                        if isinstance(row, dict)
                    )
                    if isinstance(item.get("cohort_metrics"), dict)
                    else 0,
                },
            )
        )

    for source_rank, item in enumerate(openai_routing.get("candidates") or [], start=1):
        if not isinstance(item, dict):
            continue
        lifecycle = item.get("openai_canary_lifecycle_evidence") if isinstance(item.get("openai_canary_lifecycle_evidence"), dict) else {}
        counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
        applied = _as_int(counts.get("canary_applied"))
        holdout = _as_int(counts.get("canary_holdout"))
        safety = _as_int(counts.get("safety_stopped"))
        blockers = [str(code) for code in lifecycle.get("blocker_codes") or []]
        verdict = "widen" if applied and holdout and not safety and not blockers else "projected"
        candidates.append(
            _candidate(
                action_family="routing",
                source_schema=str(openai_routing.get("schema") or ""),
                source_status=str(openai_routing.get("status") or ""),
                source_rank=source_rank,
                candidate=item,
                observed_savings_usd=0.0,
                projected_savings_usd=_as_float(item.get("projected_savings_usd")),
                sample_count=_as_int(item.get("matched_count"), applied + holdout),
                applied_count=applied,
                holdout_count=holdout,
                safety_stop_count=safety,
                verdict=verdict,
                next_action_fallback="stage-openai-routing-canary",
                extra_blockers=blockers,
                extras={
                    "estimated_savings_per_1000_calls_usd": _round(item.get("estimated_savings_per_1000_calls_usd"), 6),
                    "current_routed_count": _as_int(item.get("current_routed_count")),
                },
            )
        )
    return candidates


def build_local_promotion_candidates_from_reports(source_reports: dict[str, Any]) -> dict[str, Any]:
    cache_impact = source_reports.get("cache_impact") if isinstance(source_reports.get("cache_impact"), dict) else {}
    rollups = source_reports.get("request_shape_rollups") if isinstance(source_reports.get("request_shape_rollups"), dict) else {}
    thinking_impact = (
        source_reports.get("anthropic_thinking_compaction_impact")
        if isinstance(source_reports.get("anthropic_thinking_compaction_impact"), dict)
        else {}
    )
    claude_impact = source_reports.get("claude_routing_impact") if isinstance(source_reports.get("claude_routing_impact"), dict) else {}
    openai_routing = source_reports.get("openai_routing_report") if isinstance(source_reports.get("openai_routing_report"), dict) else {}

    candidates = (
        _cache_candidates(cache_impact, rollups)
        + _crunch_candidates(rollups, thinking_impact)
        + _routing_candidates(claude_impact, openai_routing)
    )
    candidates.sort(
        key=lambda item: (
            bool(item.get("promotion_ready")),
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("sample_count")),
        ),
        reverse=True,
    )
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank

    family_counts: Counter[str] = Counter(str(item.get("action_family") or "unknown") for item in candidates)
    readiness_counts: Counter[str] = Counter(str(item.get("readiness_state") or "unknown") for item in candidates)
    blocker_counts: Counter[str] = Counter()
    for item in candidates:
        for reason in item.get("blocker_codes") or []:
            blocker_counts[str(reason)] += 1

    top = candidates[0] if candidates else {}
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "status": "ranked" if candidates else "no-local-promotion-candidates",
        "summary": {
            "candidate_count": len(candidates),
            "promotion_ready_count": sum(1 for item in candidates if item.get("promotion_ready")),
            "blocked_count": sum(1 for item in candidates if item.get("readiness_state") == "blocked"),
            "projected_count": sum(1 for item in candidates if item.get("readiness_state") == "projected"),
            "cache_candidate_count": family_counts.get("cache", 0),
            "crunch_candidate_count": family_counts.get("crunch", 0),
            "routing_candidate_count": family_counts.get("routing", 0),
            "observed_savings_usd": round(sum(_as_float(item.get("observed_savings_usd")) for item in candidates), 8),
            "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in candidates), 8),
            "top_action_family": top.get("action_family"),
            "top_next_action": top.get("next_action"),
            "top_readiness_state": top.get("readiness_state"),
        },
        "family_breakdown": _breakdown(family_counts),
        "readiness_breakdown": _breakdown(readiness_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "source_reports": {
            "cache_impact_schema": cache_impact.get("schema"),
            "request_shape_rollups_schema": rollups.get("schema"),
            "anthropic_thinking_compaction_impact_schema": thinking_impact.get("schema"),
            "claude_routing_impact_schema": claude_impact.get("schema"),
            "openai_routing_report_schema": openai_routing.get("schema"),
            "raw_source_reports_included": False,
        },
        "candidates": candidates,
        "privacy": {
            **_privacy_summary(),
            "basis": "sanitized aggregate cache, crunch, and routing canary reports only",
        },
    }


def build_local_promotion_candidates_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    since: str | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    from tokenclaw.claude_canary_impact import build_claude_canary_impact_report
    from tokenclaw.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report
    from tokenclaw.openai_cache_replay_impact import build_openai_cache_replay_impact_report
    from tokenclaw.openai_routing_report import build_openai_routing_report
    from tokenclaw.request_shape_rollups import build_request_shape_rollups_report

    source_reports = {
        "cache_impact": build_openai_cache_replay_impact_report(store_obj, limit=capped_limit, since=since),
        "request_shape_rollups": build_request_shape_rollups_report(store_obj, limit=capped_limit, persist=False),
        "anthropic_thinking_compaction_impact": build_anthropic_thinking_compaction_impact_report(store_obj, limit=capped_limit, since=since),
        "claude_routing_impact": build_claude_canary_impact_report(store_obj, limit=capped_limit, since=since),
        "openai_routing_report": build_openai_routing_report(store_obj, limit=capped_limit),
    }
    result = build_local_promotion_candidates_from_reports(source_reports)
    result["lookback_limit"] = capped_limit
    result["since"] = since
    return result
