from __future__ import annotations

from collections import Counter
from typing import Any

from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report
from agentflow_proxy.openai_cache_replay_report import _as_float, _as_int, build_openai_cache_replay_report
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_readiness.v1"


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
        "tool_payloads_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "pattern_hashes_included": False,
        "request_fingerprints_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _breakdown_map(rows: Any, key: str = "value") -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get(key) or "unknown")
        result[value] = result.get(value, 0) + _as_int(row.get("count"))
    return result


def _counter_rows(counter: Counter[str] | dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count
    ]


def _verdict_counts(impact: dict[str, Any]) -> dict[str, int]:
    gate = impact.get("quality_gate") if isinstance(impact.get("quality_gate"), dict) else {}
    counts = _breakdown_map(gate.get("verdict_counts"))
    if counts:
        return counts
    return dict(Counter(str(row.get("verdict") or "unknown") for row in impact.get("candidates") or [] if isinstance(row, dict)))


def _top_state(opportunity: dict[str, Any], impact: dict[str, Any]) -> tuple[str, str]:
    opp = opportunity.get("summary") if isinstance(opportunity.get("summary"), dict) else {}
    imp = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    verdicts = _verdict_counts(impact)

    if verdicts.get("rollback") or _as_int(imp.get("safety_stop_count")):
        return "blocked", "safety-gate-blocked"
    if _as_float(imp.get("observed_savings_usd")) > 0:
        return "saving", "observed-openai-cache-replay-savings"
    if _as_int(imp.get("applied_count")) or _as_int(imp.get("holdout_count")):
        return "canarying", "applied-or-holdout-cohort-observed"
    if _as_int(imp.get("observed_openai_cache_replay_metadata_row_count")):
        return "blocked", "cache-replay-metadata-without-healthy-cohorts"
    if _as_int(opp.get("openai_call_count")) <= 0:
        return "disabled", "no-openai-calls-observed"
    if _as_int(opp.get("safety_eligible_count")) > 0 or _as_float(opp.get("projected_savings_usd")) > 0:
        return "blocked", "replay-opportunity-needs-rule-and-canary-evidence"
    return "disabled", "no-openai-cache-replay-opportunity-observed"


def _state_counts(state: str, opportunity: dict[str, Any], impact: dict[str, Any]) -> list[dict[str, Any]]:
    imp = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    opp = opportunity.get("summary") if isinstance(opportunity.get("summary"), dict) else {}
    counts: Counter[str] = Counter()
    counts[state] += 1
    if _as_int(imp.get("applied_count")) or _as_int(imp.get("holdout_count")):
        counts["canarying"] += 1
    if _as_float(imp.get("observed_savings_usd")) > 0:
        counts["saving"] += 1
    if _as_int(imp.get("blocked_count")) or _as_int(imp.get("invalidated_count")) or _as_int(imp.get("safety_stop_count")):
        counts["blocked"] += 1
    if _as_int(opp.get("openai_call_count")) and not _as_int(imp.get("observed_openai_cache_replay_metadata_row_count")):
        counts["disabled"] += 1
    return _counter_rows(counts)


def _impact_candidate_rows(impact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in impact.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "unknown")
        if verdict == "promote" and _as_float(item.get("observed_savings_usd")) > 0:
            readiness = "saving"
        elif verdict in {"promote", "need-more-samples"}:
            readiness = "canarying"
        elif verdict == "rollback":
            readiness = "blocked"
        elif verdict == "hold":
            readiness = "blocked"
        else:
            readiness = "canarying"
        cohorts = item.get("cohort_counts") if isinstance(item.get("cohort_counts"), dict) else {}
        rows.append(
            {
                "kind": "impact",
                "readiness": readiness,
                "candidate_id": item.get("candidate_id"),
                "rule_id": item.get("rule_id"),
                "verdict": verdict,
                "reason_codes": item.get("reason_codes") or [],
                "warning_codes": item.get("warning_codes") or [],
                "sample_count": _as_int(item.get("sample_count")),
                "applied_count": _as_int(cohorts.get("applied")),
                "holdout_count": _as_int(cohorts.get("holdout")),
                "blocked_count": _as_int(cohorts.get("blocked")),
                "invalidated_count": _as_int(cohorts.get("invalidated")),
                "safety_stop_count": _as_int(cohorts.get("safety_stop")),
                "observed_savings_usd": item.get("observed_savings_usd"),
                "projected_savings_usd": item.get("projected_savings_usd"),
                "latest_observed_at": item.get("latest_observed_at"),
                "endpoint": item.get("endpoint"),
                "category": item.get("category"),
                "workflow_phase": item.get("workflow_phase"),
                "privacy": _privacy_summary(),
            }
        )
    return rows


def _opportunity_candidate_rows(opportunity: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in opportunity.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        blockers = [str(value) for value in item.get("blockers") or []]
        safety_eligible = _as_int(item.get("safety_eligible_count"))
        projected = _as_float(item.get("projected_savings_usd"))
        readiness = "blocked" if safety_eligible or projected > 0 else "disabled"
        rows.append(
            {
                "kind": "opportunity",
                "readiness": readiness,
                "candidate_id": item.get("candidate_id"),
                "rule_id": None,
                "verdict": "no-canary-evidence",
                "reason_codes": blockers[:8] or ["no-replay-rule-evidence"],
                "warning_codes": [],
                "sample_count": _as_int(item.get("matched_count")),
                "applied_count": 0,
                "holdout_count": 0,
                "blocked_count": _as_int(item.get("blocked_count")),
                "invalidated_count": sum(
                    _as_int(row.get("count"))
                    for row in item.get("invalidation_reason_breakdown") or []
                    if isinstance(row, dict)
                ),
                "safety_stop_count": 0,
                "observed_savings_usd": 0.0,
                "projected_savings_usd": item.get("projected_savings_usd"),
                "latest_observed_at": None,
                "endpoint": item.get("endpoint"),
                "category": item.get("category"),
                "workflow_phase": item.get("workflow_phase"),
                "privacy": _privacy_summary(),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_openai_cache_replay_readiness_report(
    store_obj: Any,
    *,
    opportunity_limit: int = 1000,
    impact_limit: int = 500,
) -> dict[str, Any]:
    opportunity = build_openai_cache_replay_report(store_obj, limit=opportunity_limit)
    impact = build_openai_cache_replay_impact_report(store_obj, limit=impact_limit)
    opp = opportunity.get("summary") if isinstance(opportunity.get("summary"), dict) else {}
    imp = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    state, state_reason = _top_state(opportunity, impact)
    impact_rows = _impact_candidate_rows(impact)
    opportunity_rows = _opportunity_candidate_rows(opportunity) if not impact_rows else []
    verdict_counts = _verdict_counts(impact)
    blocker_counts = _breakdown_map(opportunity.get("blocker_reason_breakdown"))

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "state": state,
        "state_reason": state_reason,
        "summary": {
            "openai_call_count": _as_int(opp.get("openai_call_count")),
            "opportunity_candidate_count": _as_int(opp.get("candidate_count")),
            "impact_candidate_count": _as_int(imp.get("candidate_count")),
            "observed_replay_metadata_rows": _as_int(imp.get("observed_openai_cache_replay_metadata_row_count")),
            "applied_count": _as_int(imp.get("applied_count")),
            "holdout_count": _as_int(imp.get("holdout_count")),
            "blocked_count": _as_int(imp.get("blocked_count")),
            "invalidated_count": _as_int(imp.get("invalidated_count")),
            "safety_stop_count": _as_int(imp.get("safety_stop_count")),
            "safety_eligible_count": _as_int(opp.get("safety_eligible_count")),
            "request_fingerprint_rows": _as_int(opp.get("request_fingerprint_rows")),
            "file_dependency_fingerprint_rows": _as_int(opp.get("file_dependency_fingerprint_rows")),
            "estimated_cost_usd": opp.get("estimated_cost_usd"),
            "projected_savings_usd": round(
                max(_as_float(opp.get("projected_savings_usd")), _as_float(imp.get("projected_savings_usd"))),
                8,
            ),
            "observed_savings_usd": imp.get("observed_savings_usd"),
            "top_blockers": _counter_rows(Counter(blocker_counts))[:6],
        },
        "state_breakdown": _state_counts(state, opportunity, impact),
        "cohort_breakdown": impact.get("cohort_breakdown") or [],
        "blocker_reason_breakdown": opportunity.get("blocker_reason_breakdown") or [],
        "cache_outcome_breakdown": opportunity.get("cache_outcome_breakdown") or [],
        "quality_gate_verdict_breakdown": _counter_rows(verdict_counts),
        "quality_gate_reason_breakdown": (
            impact.get("quality_gate", {}).get("reason_code_counts")
            if isinstance(impact.get("quality_gate"), dict)
            else []
        ) or [],
        "candidates": impact_rows + opportunity_rows,
        "source_reports": {
            "opportunity_schema": opportunity.get("schema"),
            "impact_schema": impact.get("schema"),
            "opportunity_limit": opportunity.get("limit"),
            "impact_limit": impact.get("lookback_limit"),
            "raw_source_reports_included": False,
        },
        "privacy": {
            **_privacy_summary(),
            "basis": "sanitized OpenAI cache replay opportunity and impact metadata only",
        },
    }
