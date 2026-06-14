from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.provider_adoption_gate import (
    build_provider_adoption_gate,
    provider_adoption_thresholds,
    provider_adoption_windows_by_call,
)
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.claude_canary_impact.v1"
VERDICT_SCHEMA = "agentflow.claude_canary_promotion_verdict.v1"
ANTHROPIC_LIFECYCLE_REPORT_SCHEMA = "agentflow.anthropic_routing_canary_lifecycle_report.v1"
ANTHROPIC_LIFECYCLE_SCHEMA = "agentflow.anthropic_routing_canary_lifecycle_evidence.v1"

_REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


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


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_transcripts_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "api_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "local_only": True,
    }


def _reason_code(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text if _REASON_CODE_RE.match(text) else fallback


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "rate_limit_fallback_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "cost_est_usd_total": 0.0,
        "cost_baseline_usd_total": 0.0,
        "observed_savings_usd": 0.0,
        "requested_model_fallback_cost_usd": 0.0,
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    fallback_count = _as_int(raw.get("fallback_count"))
    rate_limit_fallback_count = _as_int(raw.get("rate_limit_fallback_count"))
    return {
        "count": count,
        "error_count": _as_int(raw.get("error_count")),
        "retry_count": _as_int(raw.get("retry_count")),
        "fallback_count": fallback_count,
        "rate_limit_fallback_count": rate_limit_fallback_count,
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_count")) / count, 6) if count else 0.0,
        "fallback_rate": round(fallback_count / count, 6) if count else 0.0,
        "rate_limit_fallback_rate": round(rate_limit_fallback_count / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd_total")), 8),
        "cost_baseline_usd": round(_as_float(raw.get("cost_baseline_usd_total")), 8),
        "observed_savings_usd": round(_as_float(raw.get("observed_savings_usd")), 8),
        "requested_model_fallback_cost_usd": round(_as_float(raw.get("requested_model_fallback_cost_usd")), 8),
    }


def _status_bucket(status_code: Any) -> str:
    code = _as_int(status_code, -1)
    if code < 0:
        return "unknown"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _cohort_name(canary: dict[str, Any]) -> str:
    cohort = str(canary.get("cohort") or "").strip()
    status = str(canary.get("status") or "").strip()
    reason = str(canary.get("reason") or "").strip()
    safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
    if status == "applied" or cohort == "canary_applied":
        return "canary_applied"
    if status == "holdout" or cohort == "canary_holdout":
        return "canary_holdout"
    if status == "safety_stopped" or safety.get("tripped") or "safety-stop" in reason:
        return "safety_stopped"
    if status in {"disabled", "ineligible", "noop"} or cohort == "bypassed_or_disabled":
        return "bypassed_or_disabled"
    if status in {"not_selected", "skipped"} or cohort == "skipped":
        return "skipped"
    return "unknown"


def _observed_savings(row: dict[str, Any], cohort: str) -> float:
    if cohort != "canary_applied":
        return 0.0
    if row.get("cost_baseline_usd") is not None and row.get("cost_est_usd") is not None:
        return _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd"))
    return 0.0


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where_since = "and created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    rows = store_obj.conn.execute(
        f"""
        select id, created_at, coalesce(provider, 'anthropic') as provider, source_surface,
               endpoint, requested_model, routed_model, stream, status_code, latency_ms,
               retry_count, cost_est_usd, cost_baseline_usd, routing_json, crunch_json,
               cache_json, category
        from calls
        where routing_json is not null
          and coalesce(provider, 'anthropic') = 'anthropic'
          {where_since}
        order by created_at desc
        limit ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _candidate_key(canary: dict[str, Any]) -> str:
    return str(
        canary.get("candidate_id")
        or canary.get("target_candidate_id")
        or canary.get("promotion_action_id")
        or canary.get("rule_id")
        or canary.get("policy_id")
        or "unknown-claude-canary"
    )


def _group_key(canary: dict[str, Any]) -> str:
    parts = [
        canary.get("provider") or "anthropic",
        canary.get("source_surface") or "anthropic_messages",
        canary.get("requested_model") or canary.get("original_model") or "unknown-requested",
        canary.get("target_model") or "unknown-target",
        canary.get("category") or "unknown-category",
        canary.get("workflow_phase") or "unknown-phase",
        "stream" if bool(canary.get("stream")) else "nonstream",
        canary.get("rule_id") or canary.get("policy_id") or "unknown-rule",
        _candidate_key(canary),
    ]
    return "|".join(_reason_code(part, "unknown") for part in parts)


def _new_aggregate(group_key: str, canary: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_key": group_key,
        "candidate_id": _candidate_key(canary),
        "rule_id": canary.get("rule_id") or canary.get("policy_id"),
        "promotion_action_id": canary.get("promotion_action_id"),
        "target_candidate_id": canary.get("target_candidate_id"),
        "policy_id": canary.get("policy_id"),
        "policy_source": canary.get("policy_source"),
        "provider": canary.get("provider") or "anthropic",
        "source_surface": canary.get("source_surface") or "anthropic_messages",
        "app_family": canary.get("app_family") or "anthropic",
        "original_model": canary.get("original_model") or canary.get("requested_model"),
        "target_model": canary.get("target_model"),
        "category": canary.get("category"),
        "workflow_phase": canary.get("workflow_phase"),
        "workflow_phase_confidence": canary.get("workflow_phase_confidence"),
        "stream": bool(canary.get("stream")),
        "canary_fraction": canary.get("canary_fraction"),
        "holdout_fraction": canary.get("holdout_fraction"),
        "cohorts": {
            "canary_applied": _empty_cohort(),
            "canary_holdout": _empty_cohort(),
            "skipped": _empty_cohort(),
            "bypassed_or_disabled": _empty_cohort(),
            "safety_stopped": _empty_cohort(),
            "unknown": _empty_cohort(),
        },
        "status_buckets": Counter(),
        "reason_buckets": Counter(),
        "category_buckets": Counter(),
        "workflow_phase_buckets": Counter(),
        "stream_buckets": Counter(),
        "cache_interactions": Counter(),
        "crunch_interactions": Counter(),
        "safety_skip_counts": Counter(),
        "stripped_param_counts": Counter(),
        "fallback_reason_counts": Counter(),
        "provider_adoption_observations": [],
        "latest_observed_at": None,
        "oldest_observed_at": None,
    }


def _add_row(
    aggregate: dict[str, Any],
    row: dict[str, Any],
    routing: dict[str, Any],
    canary: dict[str, Any],
    provider_adoption_windows: list[dict[str, Any]] | None = None,
) -> None:
    cohort = _cohort_name(canary)
    bucket = aggregate["cohorts"].get(cohort, aggregate["cohorts"]["unknown"])
    bucket["count"] += 1
    status_code = _as_int(row.get("status_code"), -1)
    if status_code >= 400:
        bucket["error_count"] += 1
    if _as_int(row.get("retry_count")) > 0:
        bucket["retry_count"] += 1

    fallback_reason = str(canary.get("fallback_reason") or routing.get("fallback_reason") or "")
    if fallback_reason:
        bucket["fallback_count"] += 1
        aggregate["fallback_reason_counts"][_reason_code(fallback_reason)] += 1
        if fallback_reason == "rate_limited":
            bucket["rate_limit_fallback_count"] += 1
        bucket["requested_model_fallback_cost_usd"] += _as_float(row.get("cost_est_usd"))

    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        bucket["latency_ms_total"] += latency_ms
        bucket["latency_sample_count"] += 1
    bucket["cost_est_usd_total"] += _as_float(row.get("cost_est_usd"))
    bucket["cost_baseline_usd_total"] += _as_float(row.get("cost_baseline_usd"))
    bucket["observed_savings_usd"] += _observed_savings(row, cohort)

    reason = _reason_code(canary.get("reason"))
    aggregate["status_buckets"][_status_bucket(row.get("status_code"))] += 1
    aggregate["reason_buckets"][reason] += 1
    aggregate["category_buckets"][_reason_code(canary.get("category") or row.get("category"), "unknown")] += 1
    aggregate["workflow_phase_buckets"][_reason_code(canary.get("workflow_phase"), "unknown")] += 1
    aggregate["stream_buckets"]["stream" if bool(row.get("stream")) else "nonstream"] += 1

    if reason in {"workflow-phase-not-enabled", "thinking-history-not-enabled", "thinking-request-not-enabled"}:
        if str(canary.get("workflow_phase") or "") == "thinking" or "thinking" in reason:
            aggregate["safety_skip_counts"]["thinking-history"] += 1
    if reason in {"source-surface-not-supported", "stream-scope-not-enabled", "request-too-large", "category-not-enabled"}:
        aggregate["safety_skip_counts"][reason] += 1
    safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
    if safety.get("tripped"):
        for code in safety.get("reason_codes") or []:
            aggregate["safety_skip_counts"][_reason_code(code)] += 1

    stripped = routing.get("stripped_params")
    if isinstance(stripped, list):
        for value in stripped:
            aggregate["stripped_param_counts"][_reason_code(value, "unknown-param")] += 1

    cache = _json_obj(row.get("cache_json"))
    cache_status = _reason_code(cache.get("status") or ("hit" if cache.get("hit") else None), "unknown")
    aggregate["cache_interactions"][cache_status] += 1
    crunch = _json_obj(row.get("crunch_json"))
    crunch_status = "changed" if crunch.get("changed") else "unchanged"
    aggregate["crunch_interactions"][crunch_status] += 1

    created_at = row.get("created_at")
    if created_at:
        if aggregate["latest_observed_at"] is None or str(created_at) > str(aggregate["latest_observed_at"]):
            aggregate["latest_observed_at"] = str(created_at)
        if aggregate["oldest_observed_at"] is None or str(created_at) < str(aggregate["oldest_observed_at"]):
            aggregate["oldest_observed_at"] = str(created_at)
    aggregate["provider_adoption_observations"].append({
        "cohort": cohort,
        "provider_adoption_windows": provider_adoption_windows or [],
    })


def _stale_evidence(latest_observed_at: str | None, *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    latest = _parse_time(latest_observed_at)
    if latest is None:
        return {"stale": False, "age_hours": None, "max_age_hours": round(float(max_age_hours), 3)}
    age_hours = (now.astimezone(timezone.utc) - latest).total_seconds() / 3600.0
    return {"stale": age_hours > max_age_hours, "age_hours": round(age_hours, 3), "max_age_hours": round(float(max_age_hours), 3)}


def _decide_verdict(
    *,
    cohorts: dict[str, dict[str, Any]],
    deltas: dict[str, Any],
    stale: dict[str, Any],
    thresholds: dict[str, Any],
    provider_adoption_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    safety = cohorts["safety_stopped"]
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if _as_int(applied.get("count")) < _as_int(thresholds["min_canary_applied_samples"]):
        reason_codes.append("insufficient-applied-samples")
    if _as_int(holdout.get("count")) < _as_int(thresholds["min_canary_holdout_samples"]):
        reason_codes.append("insufficient-holdout-samples")
    if reason_codes:
        return {"verdict": "needs_more_samples", "reason_codes": reason_codes, "warning_codes": warning_codes}

    observed_savings = _as_float(applied.get("observed_savings_usd"))
    if _as_int(safety.get("count")) > 0:
        reason_codes.append("safety-stop-observed")
    if observed_savings < 0:
        reason_codes.append("negative-observed-savings")
    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        reason_codes.append("rollback-error-rate")
    if _as_float(applied.get("fallback_rate")) >= _as_float(thresholds["rollback_fallback_rate"]):
        reason_codes.append("rollback-fallback-rate")
    if reason_codes:
        return {"verdict": "rollback", "reason_codes": reason_codes, "warning_codes": warning_codes}

    if stale.get("stale"):
        reason_codes.append("stale-evidence")
    if _as_float(applied.get("error_rate")) > _as_float(thresholds["max_error_rate"]):
        reason_codes.append("applied-error-rate-above-threshold")
    if _as_float(deltas.get("applied_minus_holdout_error_rate")) > _as_float(thresholds["max_error_rate_delta"]):
        reason_codes.append("error-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_retry_rate")) > _as_float(thresholds["max_retry_rate_delta"]):
        reason_codes.append("retry-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_fallback_rate")) > _as_float(thresholds["max_fallback_rate_delta"]):
        reason_codes.append("fallback-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_rate_limit_fallback_rate")) > _as_float(thresholds["max_rate_limit_fallback_rate_delta"]):
        reason_codes.append("rate-limit-fallback-regression")
    latency_delta = deltas.get("applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and _as_float(latency_delta) > _as_float(thresholds["max_latency_regression_ms"]):
        reason_codes.append("latency-regression")
    gate = provider_adoption_gate or {}
    if gate.get("blocking"):
        reason_codes.extend(str(code) for code in gate.get("reason_codes") or [])
    else:
        warning_codes.extend(str(code) for code in gate.get("warning_codes") or [])
    if observed_savings <= 0:
        reason_codes.append("non-positive-observed-savings")

    if reason_codes:
        return {"verdict": "hold", "reason_codes": sorted(set(reason_codes)), "warning_codes": sorted(set(warning_codes))}
    return {"verdict": "widen", "reason_codes": ["target-savings-met"], "warning_codes": warning_codes}


def _finalize_candidate(
    aggregate: dict[str, Any],
    *,
    now: datetime,
    max_evidence_age_hours: float,
    min_applied_samples: int,
    min_holdout_samples: int,
    max_error_rate: float,
    max_error_rate_delta: float,
    max_retry_rate_delta: float,
    max_fallback_rate_delta: float,
    max_rate_limit_fallback_rate_delta: float,
    max_latency_regression_ms: int,
    rollback_error_rate: float,
    rollback_fallback_rate: float,
) -> dict[str, Any]:
    cohorts = {key: _finalize_cohort(value) for key, value in aggregate["cohorts"].items()}
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    deltas = {
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_fallback_rate": round(_as_float(applied["fallback_rate"]) - _as_float(holdout["fallback_rate"]), 6),
        "applied_minus_holdout_rate_limit_fallback_rate": round(
            _as_float(applied["rate_limit_fallback_rate"]) - _as_float(holdout["rate_limit_fallback_rate"]), 6
        ),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
        "applied_minus_holdout_cost_est_usd": round(_as_float(applied["cost_est_usd"]) - _as_float(holdout["cost_est_usd"]), 8),
    }
    stale = _stale_evidence(aggregate.get("latest_observed_at"), now=now, max_age_hours=max_evidence_age_hours)
    thresholds = {
        "min_canary_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_canary_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_fallback_rate_delta": round(float(max_fallback_rate_delta), 6),
        "max_rate_limit_fallback_rate_delta": round(float(max_rate_limit_fallback_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
        "rollback_fallback_rate": round(float(rollback_fallback_rate), 6),
        "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
    }
    thresholds.update(provider_adoption_thresholds())
    provider_adoption_gate = build_provider_adoption_gate(
        aggregate.get("provider_adoption_observations") or [],
        thresholds={
            key: value
            for key, value in thresholds.items()
            if key.startswith("min_provider_adoption") or key.startswith("max_applied")
        },
    )
    decision = _decide_verdict(
        cohorts=cohorts,
        deltas=deltas,
        stale=stale,
        thresholds=thresholds,
        provider_adoption_gate=provider_adoption_gate,
    )
    canary_fraction = _as_float(aggregate.get("canary_fraction"))
    holdout_fraction = _as_float(aggregate.get("holdout_fraction"))
    if decision.get("verdict") == "widen" and canary_fraction + holdout_fraction >= 1.0:
        decision = {
            "verdict": "promote",
            "reason_codes": ["target-savings-met", "canary-full-coverage"],
            "warning_codes": decision.get("warning_codes", []),
        }
    observed_total = _as_float(applied.get("observed_savings_usd"))
    return {
        "schema": VERDICT_SCHEMA,
        "group_key": aggregate["group_key"],
        "candidate_id": aggregate["candidate_id"],
        "rule_id": aggregate.get("rule_id"),
        "promotion_action_id": aggregate.get("promotion_action_id"),
        "target_candidate_id": aggregate.get("target_candidate_id"),
        "policy_id": aggregate.get("policy_id"),
        "policy_source": aggregate.get("policy_source"),
        "optimization_family": "claude_phase_routing",
        "action_family": "routing",
        "provider": aggregate.get("provider"),
        "source_surface": aggregate.get("source_surface"),
        "app_family": aggregate.get("app_family"),
        "original_model": aggregate.get("original_model"),
        "candidate_target_model": aggregate.get("target_model"),
        "category": aggregate.get("category"),
        "workflow_phase": aggregate.get("workflow_phase"),
        "workflow_phase_confidence": aggregate.get("workflow_phase_confidence"),
        "stream": aggregate.get("stream"),
        "canary_fraction": aggregate.get("canary_fraction"),
        "holdout_fraction": aggregate.get("holdout_fraction"),
        "sample_count": sum(_as_int(cohort.get("count")) for cohort in cohorts.values()),
        "cohort_counts": {key: value["count"] for key, value in cohorts.items()},
        "cohort_metrics": cohorts,
        "applied_vs_holdout_deltas": deltas,
        "observed_savings_usd": round(observed_total, 8),
        "requested_model_fallback_cost_usd": round(
            sum(_as_float(cohort.get("requested_model_fallback_cost_usd")) for cohort in cohorts.values()), 8
        ),
        "cache_interaction_counts": _counter_rows(aggregate["cache_interactions"]),
        "crunch_interaction_counts": _counter_rows(aggregate["crunch_interactions"]),
        "status_buckets": _counter_rows(aggregate["status_buckets"]),
        "reason_buckets": _counter_rows(aggregate["reason_buckets"]),
        "category_buckets": _counter_rows(aggregate["category_buckets"]),
        "workflow_phase_buckets": _counter_rows(aggregate["workflow_phase_buckets"]),
        "stream_buckets": _counter_rows(aggregate["stream_buckets"]),
        "safety_skip_counts": _counter_rows(aggregate["safety_skip_counts"]),
        "stripped_param_counts": _counter_rows(aggregate["stripped_param_counts"]),
        "fallback_reason_counts": _counter_rows(aggregate["fallback_reason_counts"]),
        "oldest_observed_at": aggregate.get("oldest_observed_at"),
        "latest_observed_at": aggregate.get("latest_observed_at"),
        "stale_evidence": stale,
        "thresholds": thresholds,
        "provider_adoption_gate": provider_adoption_gate,
        "verdict": decision["verdict"],
        "reason_codes": decision["reason_codes"],
        "warning_codes": decision["warning_codes"],
        "next_action": {
            "promote": "promote_claude_canary_to_permanent_local_rule",
            "widen": "widen_local_claude_canary",
            "hold": "keep_current_claude_canary_fraction",
            "rollback": "rollback_or_disable_claude_canary",
            "needs_more_samples": "collect_claude_canary_applied_and_holdout_evidence",
        }[decision["verdict"]],
        "privacy": _privacy_summary(),
    }


def build_claude_canary_impact_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_evidence_age_hours: float = 72.0,
    max_error_rate: float = 0.05,
    max_error_rate_delta: float = 0.05,
    max_retry_rate_delta: float = 0.10,
    max_fallback_rate_delta: float = 0.10,
    max_rate_limit_fallback_rate_delta: float = 0.05,
    max_latency_regression_ms: int = 2000,
    rollback_error_rate: float = 0.20,
    rollback_fallback_rate: float = 0.50,
    now: datetime | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    now_dt = now or datetime.now(timezone.utc)
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    adoption_by_call = provider_adoption_windows_by_call(store_obj, [row.get("id") for row in rows])
    aggregates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
        if not canary:
            continue
        group_key = _group_key(canary)
        aggregate = aggregates.setdefault(group_key, _new_aggregate(group_key, canary))
        _add_row(aggregate, row, routing, canary, adoption_by_call.get(str(row.get("id") or ""), []))
        observed_rows += 1

    candidates = [
        _finalize_candidate(
            aggregate,
            now=now_dt,
            max_evidence_age_hours=max_evidence_age_hours,
            min_applied_samples=min_applied_samples,
            min_holdout_samples=min_holdout_samples,
            max_error_rate=max_error_rate,
            max_error_rate_delta=max_error_rate_delta,
            max_retry_rate_delta=max_retry_rate_delta,
            max_fallback_rate_delta=max_fallback_rate_delta,
            max_rate_limit_fallback_rate_delta=max_rate_limit_fallback_rate_delta,
            max_latency_regression_ms=max_latency_regression_ms,
            rollback_error_rate=rollback_error_rate,
            rollback_fallback_rate=rollback_fallback_rate,
        )
        for aggregate in aggregates.values()
    ]
    candidates.sort(key=lambda item: (str(item.get("verdict")), str(item.get("group_key"))))

    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in candidates)
    reason_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    stripped_counts: Counter[str] = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        for row in item.get("safety_skip_counts") or []:
            safety_counts[str(row.get("value") or "unknown")] += _as_int(row.get("count"))
        for row in item.get("stripped_param_counts") or []:
            stripped_counts[str(row.get("value") or "unknown")] += _as_int(row.get("count"))

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "since": since,
        "status": "matched" if observed_rows else "no-claude-canary-metadata",
        "summary": {
            "sampled_call_count": len(rows),
            "observed_claude_canary_metadata_row_count": observed_rows,
            "candidate_group_count": len(candidates),
            "canary_applied_count": sum(_as_int(item["cohort_counts"].get("canary_applied")) for item in candidates),
            "canary_holdout_count": sum(_as_int(item["cohort_counts"].get("canary_holdout")) for item in candidates),
            "safety_stopped_count": sum(_as_int(item["cohort_counts"].get("safety_stopped")) for item in candidates),
            "rate_limit_fallback_count": sum(
                _as_int(item["cohort_metrics"]["canary_applied"].get("rate_limit_fallback_count")) for item in candidates
            ),
            "requested_model_fallback_cost_usd": round(
                sum(_as_float(item.get("requested_model_fallback_cost_usd")) for item in candidates), 8
            ),
            "observed_savings_usd": round(sum(_as_float(item.get("observed_savings_usd")) for item in candidates), 8),
            "verdict_counts": _counter_rows(verdict_counts),
            "reason_code_counts": _counter_rows(reason_counts),
            "safety_skip_counts": _counter_rows(safety_counts),
            "stripped_param_counts": _counter_rows(stripped_counts),
        },
        "candidates": candidates,
        "privacy": _privacy_summary(),
    }


def _sum_cohort_metric(candidate: dict[str, Any], key: str) -> int:
    metrics = candidate.get("cohort_metrics") if isinstance(candidate.get("cohort_metrics"), dict) else {}
    total = 0
    for value in metrics.values():
        if isinstance(value, dict):
            total += _as_int(value.get(key))
    return total


def _lifecycle_blocker_counts(candidate: dict[str, Any], *, observed: int, applied: int, holdout: int) -> Counter[str]:
    blockers: Counter[str] = Counter()
    if observed == 0:
        blockers["missing-anthropic-canary-lifecycle-evidence"] += 1
    if applied == 0:
        blockers["missing-applied-coverage"] += max(1, observed)
    if holdout == 0:
        blockers["missing-holdout-coverage"] += max(1, observed)

    safety_stopped = _as_int((candidate.get("cohort_counts") or {}).get("safety_stopped"))
    if safety_stopped:
        blockers["safety-stop-observed"] += safety_stopped
    error_count = _sum_cohort_metric(candidate, "error_count")
    retry_count = _sum_cohort_metric(candidate, "retry_count")
    fallback_count = _sum_cohort_metric(candidate, "fallback_count")
    if error_count:
        blockers["error-observed"] += error_count
    if retry_count:
        blockers["retry-observed"] += retry_count
    if fallback_count:
        blockers["fallback-observed"] += fallback_count

    stale = candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {}
    if stale.get("stale"):
        blockers["stale-evidence"] += max(1, observed)

    for row in candidate.get("safety_skip_counts") or []:
        if not isinstance(row, dict):
            continue
        reason = _reason_code(row.get("value"), "unknown-safety-skip")
        count = max(1, _as_int(row.get("count")))
        blockers[reason] += count
        if "thinking" in reason:
            blockers["thinking-routing-guard"] += count
        if reason == "stream-scope-not-enabled":
            blockers["unsupported-streaming-shape"] += count

    for reason in candidate.get("reason_codes") or []:
        text = _reason_code(reason, "unknown")
        if text not in {"target-savings-met", "canary-full-coverage"}:
            blockers[text] += 1
    return blockers


def _anthropic_lifecycle_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "canary_applied": _as_int((candidate.get("cohort_counts") or {}).get("canary_applied")),
        "canary_holdout": _as_int((candidate.get("cohort_counts") or {}).get("canary_holdout")),
        "safety_stopped": _as_int((candidate.get("cohort_counts") or {}).get("safety_stopped")),
        "skipped": _as_int((candidate.get("cohort_counts") or {}).get("skipped")),
        "bypassed_or_disabled": _as_int((candidate.get("cohort_counts") or {}).get("bypassed_or_disabled")),
        "unknown": _as_int((candidate.get("cohort_counts") or {}).get("unknown")),
    }
    observed = sum(counts.values())
    applied = counts["canary_applied"]
    holdout = counts["canary_holdout"]
    blockers = _lifecycle_blocker_counts(candidate, observed=observed, applied=applied, holdout=holdout)
    return {
        "schema": ANTHROPIC_LIFECYCLE_SCHEMA,
        "status": "matched" if observed else "no-anthropic-canary-metadata",
        "observed_count": observed,
        "cohort_counts": counts,
        "coverage": {
            "matched_count": observed,
            "observed_rate": 1.0 if observed else 0.0,
            "applied_rate": round(applied / observed, 6) if observed else 0.0,
            "holdout_rate": round(holdout / observed, 6) if observed else 0.0,
        },
        "error_count": _sum_cohort_metric(candidate, "error_count"),
        "retry_count": _sum_cohort_metric(candidate, "retry_count"),
        "fallback_count": _sum_cohort_metric(candidate, "fallback_count"),
        "oldest_observed_at": candidate.get("oldest_observed_at"),
        "latest_observed_at": candidate.get("latest_observed_at"),
        "stale_evidence": candidate.get("stale_evidence") if isinstance(candidate.get("stale_evidence"), dict) else {},
        "reason_breakdown": candidate.get("reason_buckets") or [],
        "blocker_codes": sorted(blockers),
        "blocker_reason_breakdown": _counter_rows(blockers),
        "privacy": _privacy_summary(),
    }


def _anthropic_lifecycle_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    lifecycle = _anthropic_lifecycle_from_candidate(candidate)
    return {
        "schema": "agentflow.anthropic_routing_canary_lifecycle_candidate.v1",
        "candidate_id": candidate.get("candidate_id"),
        "target_candidate_id": candidate.get("target_candidate_id"),
        "policy_id": candidate.get("policy_id"),
        "policy_source": candidate.get("policy_source"),
        "provider": candidate.get("provider") or "anthropic",
        "source_surface": candidate.get("source_surface") or "anthropic_messages",
        "app_family": candidate.get("app_family"),
        "requested_model": candidate.get("original_model"),
        "target_model": candidate.get("candidate_target_model"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "workflow_phase_confidence": candidate.get("workflow_phase_confidence"),
        "stream": candidate.get("stream"),
        "canary_fraction": candidate.get("canary_fraction"),
        "holdout_fraction": candidate.get("holdout_fraction"),
        "observed_savings_usd": candidate.get("observed_savings_usd"),
        "requested_model_fallback_cost_usd": candidate.get("requested_model_fallback_cost_usd"),
        "next_action": candidate.get("next_action"),
        "anthropic_canary_lifecycle_evidence": lifecycle,
        "privacy": _privacy_summary(),
    }


def build_anthropic_routing_canary_lifecycle_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    max_evidence_age_hours: float = 72.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    impact = build_claude_canary_impact_report(
        store_obj,
        limit=limit,
        since=since,
        min_applied_samples=0,
        min_holdout_samples=0,
        max_evidence_age_hours=max_evidence_age_hours,
        now=now,
    )
    candidates = [_anthropic_lifecycle_candidate(candidate) for candidate in impact.get("candidates") or []]
    blocker_totals: Counter[str] = Counter()
    for candidate in candidates:
        lifecycle = candidate.get("anthropic_canary_lifecycle_evidence") or {}
        for row in lifecycle.get("blocker_reason_breakdown") or []:
            if isinstance(row, dict):
                blocker_totals[str(row.get("value") or "unknown")] += _as_int(row.get("count"))

    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    return {
        "schema": ANTHROPIC_LIFECYCLE_REPORT_SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": max(1, min(int(limit or 500), 10_000)),
        "since": since,
        "status": "matched" if candidates else "no-anthropic-canary-metadata",
        "summary": {
            "sampled_call_count": summary.get("sampled_call_count", 0),
            "observed_anthropic_canary_metadata_row_count": summary.get("observed_claude_canary_metadata_row_count", 0),
            "candidate_group_count": len(candidates),
            "canary_applied_count": summary.get("canary_applied_count", 0),
            "canary_holdout_count": summary.get("canary_holdout_count", 0),
            "safety_stopped_count": summary.get("safety_stopped_count", 0),
            "fallback_count": sum(
                _as_int((candidate["anthropic_canary_lifecycle_evidence"] or {}).get("fallback_count"))
                for candidate in candidates
            ),
            "retry_count": sum(
                _as_int((candidate["anthropic_canary_lifecycle_evidence"] or {}).get("retry_count"))
                for candidate in candidates
            ),
            "error_count": sum(
                _as_int((candidate["anthropic_canary_lifecycle_evidence"] or {}).get("error_count"))
                for candidate in candidates
            ),
            "stale_evidence_count": sum(
                _as_int((candidate["anthropic_canary_lifecycle_evidence"] or {}).get("observed_count"))
                for candidate in candidates
                if ((candidate["anthropic_canary_lifecycle_evidence"] or {}).get("stale_evidence") or {}).get("stale")
            ),
            "blocker_reason_breakdown": _counter_rows(blocker_totals),
        },
        "candidates": candidates,
        "source_report": {
            "schema": impact.get("schema"),
            "status": impact.get("status"),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "privacy": _privacy_summary(),
    }
