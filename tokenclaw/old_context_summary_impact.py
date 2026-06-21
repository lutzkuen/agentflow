from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from tokenclaw.old_context_summary_dry_run import OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA
from tokenclaw.provider_adoption_gate import (
    build_provider_adoption_gate,
    provider_adoption_thresholds,
    provider_adoption_windows_by_call,
)
from tokenclaw.store import utc_now


OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA = "tokenclaw.old_context_summary_impact.v1"

OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA = "tokenclaw.old_context_summary_quality_gate.v1"


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_surface(provider: Any, path: Any) -> str:
    provider_text = str(provider or "anthropic")
    path_text = str(path or "")
    if provider_text == "anthropic" and path_text.endswith("/v1/messages"):
        return "anthropic_messages"
    if provider_text == "openai":
        return "openai"
    return provider_text


def _model_tier(model: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    if "gpt" in text:
        return "openai"
    return "unknown"


def _bucket_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [{"value": key, "count": counts[key]} for key in sorted(counts)]


def _increment(counts: dict[str, int], value: Any) -> None:
    key = str(value or "unknown")
    counts[key] = counts.get(key, 0) + 1


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


def _latency_bucket(latency_ms: Any) -> str:
    latency = _as_int(latency_ms, -1)
    if latency < 0:
        return "unknown"
    if latency < 1_000:
        return "lt_1s"
    if latency < 2_000:
        return "1s_2s"
    if latency < 10_000:
        return "2s_10s"
    return "gte_10s"


def _cohort(summary: dict[str, Any]) -> str:
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    cohort = str(canary.get("cohort") or canary.get("status") or "")
    status = str(summary.get("status") or "")
    reason = str(summary.get("reason") or "")
    if cohort == "canary_applied" or status == "applied":
        return "canary_applied"
    if cohort == "canary_holdout" or reason == "canary_holdout":
        return "canary_holdout"
    if status in {"bypass", "skipped", "disabled"} or "disabled" in reason or "safety-stop" in reason:
        return "bypassed_or_disabled"
    return "unknown"


def _summary_failed(summary: dict[str, Any]) -> bool:
    status_code = summary.get("summary_status_code")
    return (
        _as_int(status_code, 0) >= 400
        or bool(summary.get("summary_error"))
        or str(summary.get("status") or "") in {"summary_failed", "error"}
        or str(summary.get("reason") or "") in {"summary-error", "summary-model-error"}
    )


def _dry_run_from_report(report: Any) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    if report.get("schema") == OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA:
        return report
    impact = report.get("impact_summary") if isinstance(report.get("impact_summary"), dict) else {}
    sections = impact.get("sections") if isinstance(impact.get("sections"), dict) else {}
    crunch = sections.get("crunch") if isinstance(sections.get("crunch"), dict) else {}
    dry_run = crunch.get("old_context_summary_dry_run")
    if isinstance(dry_run, dict) and dry_run.get("schema") == OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA:
        return dry_run
    dry_run = report.get("old_context_summary_dry_run")
    if isinstance(dry_run, dict) and dry_run.get("schema") == OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA:
        return dry_run
    return None


def _eligible_group_filters(dry_run: dict[str, Any]) -> set[tuple[str, str, str, bool]]:
    filters: set[tuple[str, str, str, bool]] = set()
    for group in dry_run.get("groups") or []:
        if not isinstance(group, dict) or group.get("blocker") != "eligible":
            continue
        filters.add((
            str(group.get("source_surface") or "unknown"),
            str(group.get("category") or "unknown"),
            str(group.get("model_tier") or "unknown"),
            bool(group.get("stream")),
        ))
    return filters


def _projection(dry_run: dict[str, Any]) -> dict[str, Any]:
    policy = dry_run.get("policy") if isinstance(dry_run.get("policy"), dict) else {}
    summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
    eligible = _as_int(summary.get("eligible_call_count"))
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    canary_enabled = bool(canary.get("enabled"))
    fraction = _as_float(canary.get("fraction"), 1.0 if not canary_enabled else 0.0)
    projected_applied = int(round(eligible * fraction)) if canary_enabled else eligible
    projected_holdout = max(0, eligible - projected_applied) if canary_enabled else 0
    return {
        "eligible_call_count": eligible,
        "projected_affected_metadata_row_count": eligible,
        "projected_canary_applied_count": projected_applied,
        "projected_canary_holdout_count": projected_holdout,
        "projected_bypass_or_disabled_count": 0,
        "projected_saved_chars": _as_int(summary.get("projected_saved_chars")),
        "projected_saved_tokens": _as_int(summary.get("projected_saved_tokens")),
        "estimated_summary_cost_usd": round(_as_float(summary.get("estimated_summary_cost_usd")), 8),
        "projected_gross_savings_usd": round(_as_float(summary.get("projected_gross_savings_usd")), 8),
        "projected_net_savings_usd": round(_as_float(summary.get("projected_net_savings_usd")), 8),
    }


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    params: tuple[Any, ...]
    where = ""
    if since:
        where = "where created_at >= ?"
        params = (since, max(1, int(limit)))
    else:
        params = (max(1, int(limit)),)
    sql = f"""
        select created_at, provider, path, requested_model, routed_model, stream,
               id, status_code, latency_ms, actual_input_tokens, actual_output_tokens,
               cost_est_usd, cost_baseline_usd, crunch_json, routing_json, cache_json,
               category, retry_count
        from calls
        {where}
        order by created_at desc
        limit ?
    """
    return [dict(row) for row in store_obj.conn.execute(sql, params).fetchall()]


def _row_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    crunch = _json_obj(row.get("crunch_json"))
    summary = crunch.get("old_context_summarization") if isinstance(crunch, dict) else None
    if not isinstance(summary, dict):
        return None
    routing = _json_obj(row.get("routing_json"))
    return {
        "source_surface": _source_surface(row.get("provider"), row.get("path")),
        "call_id": row.get("id"),
        "category": str(summary.get("category") or row.get("category") or routing.get("category") or "unknown"),
        "model_tier": _model_tier(row.get("requested_model")),
        "stream": bool(row.get("stream")),
        "status_code": row.get("status_code"),
        "latency_ms": row.get("latency_ms"),
        "retry_count": row.get("retry_count"),
        "summary": summary,
        "provider_adoption_windows": row.get("provider_adoption_windows") if isinstance(row.get("provider_adoption_windows"), list) else [],
    }


def _matches(summary: dict[str, Any], *, policy: dict[str, Any], filters: set[tuple[str, str, str, bool]]) -> bool:
    meta = summary["summary"]
    rule_id = policy.get("rule_id")
    candidate_id = policy.get("candidate_id")
    if rule_id and meta.get("rule_id") != rule_id:
        return False
    if candidate_id and meta.get("candidate_id") != candidate_id:
        return False
    if filters:
        key = (
            str(summary.get("source_surface") or "unknown"),
            str(summary.get("category") or "unknown"),
            str(summary.get("model_tier") or "unknown"),
            bool(summary.get("stream")),
        )
        if key not in filters:
            return False
    return True


def _actual(matched: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    latency_counts: dict[str, int] = {}
    cache_counts: dict[str, int] = {}
    safety_counts: dict[str, int] = {}
    applied_latencies: list[int] = []
    holdout_latencies: list[int] = []
    applied = 0
    holdout = 0
    bypass = 0
    summary_failures = 0
    status_errors = 0
    retries = 0
    saved_chars = 0
    saved_tokens = 0
    gross = 0.0
    summary_cost = 0.0
    net = 0.0
    for item in matched:
        meta = item["summary"]
        cohort = _cohort(meta)
        latency = _as_int(item.get("latency_ms"), -1)
        if cohort == "canary_applied":
            applied += 1
            if latency >= 0:
                applied_latencies.append(latency)
        elif cohort == "canary_holdout":
            holdout += 1
            if latency >= 0:
                holdout_latencies.append(latency)
        elif cohort == "bypassed_or_disabled":
            bypass += 1
        summary_failures += int(_summary_failed(meta))
        status_errors += int(_as_int(item.get("status_code"), 0) >= 400)
        retries += int(_as_int(item.get("retry_count"), 0) > 0)
        saved_chars += _as_int(meta.get("saved_chars"))
        saved_tokens += _as_int(meta.get("tokens_saved_est"))
        gross += _as_float(meta.get("estimated_gross_savings_usd"))
        summary_cost += _as_float(meta.get("summary_cost_est_usd"))
        net += _as_float(meta.get("estimated_net_savings_usd"))
        _increment(status_counts, _status_bucket(item.get("status_code")))
        _increment(decision_counts, meta.get("status"))
        _increment(reason_counts, meta.get("reason"))
        _increment(latency_counts, _latency_bucket(item.get("latency_ms")))
        _increment(cache_counts, "summary-cache-hit" if meta.get("summary_cache_hit") else "summary-cache-miss-or-unused")
        safety_state = meta.get("safety_stop_state")
        safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
        if not safety_state and safety:
            safety_state = "stopped" if safety.get("stopped") else "clear"
        _increment(safety_counts, safety_state or "not-reported")

    def avg(values: list[int]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    applied_avg = avg(applied_latencies)
    holdout_avg = avg(holdout_latencies)
    return {
        "matched_metadata_row_count": len(matched),
        "actual_canary_applied_count": applied,
        "actual_canary_holdout_count": holdout,
        "actual_bypassed_or_disabled_count": bypass,
        "summary_failure_count": summary_failures,
        "error_count": status_errors,
        "retry_count": retries,
        "error_rate": round(status_errors / len(matched), 6) if matched else 0.0,
        "retry_rate": round(retries / len(matched), 6) if matched else 0.0,
        "actual_saved_chars": saved_chars,
        "actual_tokens_saved_est": saved_tokens,
        "actual_gross_savings_usd": round(gross, 8),
        "actual_summary_model_cost_usd": round(summary_cost, 8),
        "actual_net_savings_usd": round(net, 8),
        "latency": {
            "applied_avg_ms": applied_avg,
            "holdout_avg_ms": holdout_avg,
            "applied_minus_holdout_avg_ms": round(applied_avg - holdout_avg, 2)
            if applied_avg is not None and holdout_avg is not None
            else None,
        },
        "status_buckets": _bucket_counts(status_counts),
        "summary_decision_status_buckets": _bucket_counts(decision_counts),
        "summary_reason_buckets": _bucket_counts(reason_counts),
        "latency_buckets": _bucket_counts(latency_counts),
        "summary_cache_buckets": _bucket_counts(cache_counts),
        "safety_stop_buckets": _bucket_counts(safety_counts),
    }


def _empty_cohort_metrics() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_count": 0,
        "summary_failure_count": 0,
        "safety_stop_count": 0,
        "bypass_or_disabled_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "tokens_saved_est": 0,
        "gross_savings_usd": 0.0,
        "summary_model_cost_usd": 0.0,
        "net_savings_usd": 0.0,
    }


def _finalize_cohort_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(metrics.get("count"))
    latency_samples = _as_int(metrics.get("latency_sample_count"))
    result = {
        "count": count,
        "error_count": _as_int(metrics.get("error_count")),
        "retry_count": _as_int(metrics.get("retry_count")),
        "summary_failure_count": _as_int(metrics.get("summary_failure_count")),
        "safety_stop_count": _as_int(metrics.get("safety_stop_count")),
        "bypass_or_disabled_count": _as_int(metrics.get("bypass_or_disabled_count")),
        "error_rate": round(_as_int(metrics.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(metrics.get("retry_count")) / count, 6) if count else 0.0,
        "summary_failure_rate": round(_as_int(metrics.get("summary_failure_count")) / count, 6) if count else 0.0,
        "safety_stop_rate": round(_as_int(metrics.get("safety_stop_count")) / count, 6) if count else 0.0,
        "bypass_or_disabled_rate": round(_as_int(metrics.get("bypass_or_disabled_count")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(metrics.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "tokens_saved_est": _as_int(metrics.get("tokens_saved_est")),
        "gross_savings_usd": round(_as_float(metrics.get("gross_savings_usd")), 8),
        "summary_model_cost_usd": round(_as_float(metrics.get("summary_model_cost_usd")), 8),
        "net_savings_usd": round(_as_float(metrics.get("net_savings_usd")), 8),
    }
    return result


def _cohort_metrics(matched: list[dict[str, Any]]) -> dict[str, Any]:
    cohorts = {
        "canary_applied": _empty_cohort_metrics(),
        "canary_holdout": _empty_cohort_metrics(),
        "bypassed_or_disabled": _empty_cohort_metrics(),
        "unknown": _empty_cohort_metrics(),
    }
    for item in matched:
        meta = item["summary"]
        cohort = _cohort(meta)
        bucket = cohorts.get(cohort, cohorts["unknown"])
        bucket["count"] += 1
        bucket["error_count"] += int(_as_int(item.get("status_code"), 0) >= 400)
        bucket["retry_count"] += int(_as_int(item.get("retry_count"), 0) > 0)
        bucket["summary_failure_count"] += int(_summary_failed(meta))
        bucket["bypass_or_disabled_count"] += int(cohort == "bypassed_or_disabled")
        safety_state = meta.get("safety_stop_state")
        safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
        if not safety_state and safety:
            safety_state = "stopped" if safety.get("stopped") else "clear"
        bucket["safety_stop_count"] += int(str(safety_state or "") == "stopped")
        latency = _as_int(item.get("latency_ms"), -1)
        if latency >= 0:
            bucket["latency_ms_total"] += latency
            bucket["latency_sample_count"] += 1
        bucket["tokens_saved_est"] += _as_int(meta.get("tokens_saved_est"))
        bucket["gross_savings_usd"] += _as_float(meta.get("estimated_gross_savings_usd"))
        bucket["summary_model_cost_usd"] += _as_float(meta.get("summary_cost_est_usd"))
        bucket["net_savings_usd"] += _as_float(meta.get("estimated_net_savings_usd"))

    return {key: _finalize_cohort_metrics(value) for key, value in cohorts.items()}


def _quality_gate_thresholds(policy: dict[str, Any], projection: dict[str, Any]) -> dict[str, Any]:
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    safety_stop = policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}
    raw_safety_gates = policy.get("safety_gates") if isinstance(policy.get("safety_gates"), dict) else {}
    min_samples = _as_int(raw_safety_gates.get("min_outcome_samples"), 0)
    min_applied = _as_int(raw_safety_gates.get("min_canary_applied_samples"), 0)
    min_holdout = _as_int(raw_safety_gates.get("min_canary_holdout_samples"), 0)
    if not min_samples:
        min_samples = max(10, _as_int(canary.get("holdout_sample_count")), _as_int(projection.get("projected_affected_metadata_row_count")))
    if not min_applied:
        min_applied = max(1, min_samples // 2)
    if not min_holdout:
        min_holdout = max(1, min_samples // 4) if bool(canary.get("enabled")) else 0
    thresholds = {
        "schema": "tokenclaw.old_context_summary_quality_gate_thresholds.v1",
        "min_matched_samples": min_samples,
        "min_canary_applied_samples": min_applied,
        "min_canary_holdout_samples": min_holdout,
        "min_net_savings_usd": round(_as_float(raw_safety_gates.get("min_net_savings_usd"), 0.0), 8),
        "min_payback_ratio": round(_as_float(raw_safety_gates.get("min_payback_ratio"), 1.0), 6),
        "min_projection_realization_ratio": round(_as_float(raw_safety_gates.get("min_projection_realization_ratio"), 0.5), 6),
        "max_error_rate": round(_as_float(raw_safety_gates.get("max_error_rate"), _as_float(safety_stop.get("max_error_rate"), 0.05)), 6),
        "max_error_rate_delta": round(_as_float(raw_safety_gates.get("max_error_rate_delta"), 0.0), 6),
        "max_retry_rate": round(_as_float(raw_safety_gates.get("max_retry_rate"), 0.10), 6),
        "max_retry_rate_delta": round(_as_float(raw_safety_gates.get("max_retry_rate_delta"), 0.05), 6),
        "max_summary_failure_rate": round(_as_float(raw_safety_gates.get("max_summary_failure_rate"), _as_float(safety_stop.get("max_summary_failure_rate"), 0.02)), 6),
        "max_bypass_or_disabled_rate": round(_as_float(raw_safety_gates.get("max_bypass_or_disabled_rate"), 0.10), 6),
        "max_safety_stop_count": _as_int(raw_safety_gates.get("max_safety_stop_count"), 0),
        "max_latency_regression_ms": _as_int(raw_safety_gates.get("max_latency_regression_ms"), 2_000),
        "rollback_error_rate": round(_as_float(raw_safety_gates.get("rollback_error_rate"), 0.10), 6),
        "rollback_summary_failure_rate": round(_as_float(raw_safety_gates.get("rollback_summary_failure_rate"), 0.05), 6),
        "rollback_safety_stop_count": _as_int(raw_safety_gates.get("rollback_safety_stop_count"), 1),
        "rollback_negative_net_savings_usd": round(_as_float(raw_safety_gates.get("rollback_negative_net_savings_usd"), 0.0), 8),
    }
    thresholds.update(provider_adoption_thresholds(
        min_fulfilled_samples=_as_int(raw_safety_gates.get("min_provider_adoption_fulfilled_samples"), 1),
        max_applied_abandonment_rate=_as_float(raw_safety_gates.get("max_provider_adoption_abandonment_rate"), 0.02),
        max_applied_orphan_result_rate=_as_float(raw_safety_gates.get("max_provider_adoption_orphan_result_rate"), 0.02),
        max_applied_risk_rate=_as_float(raw_safety_gates.get("max_provider_adoption_risk_rate"), 0.05),
        max_applied_vs_holdout_risk_rate_delta=_as_float(raw_safety_gates.get("max_provider_adoption_risk_rate_delta"), 0.02),
    ))
    return thresholds


def decide_old_context_summary_quality_gate(
    *,
    policy: dict[str, Any],
    projection: dict[str, Any],
    actual: dict[str, Any],
    matched: list[dict[str, Any]],
) -> dict[str, Any]:
    thresholds = _quality_gate_thresholds(policy, projection)
    cohorts = _cohort_metrics(matched)
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    bypassed = cohorts["bypassed_or_disabled"]
    matched_count = _as_int(actual.get("matched_metadata_row_count"))
    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    summary_failure_count = _as_int(actual.get("summary_failure_count"))
    bypass_count = _as_int(actual.get("actual_bypassed_or_disabled_count"))
    net_savings = _as_float(actual.get("actual_net_savings_usd"))
    gross_savings = _as_float(actual.get("actual_gross_savings_usd"))
    summary_cost = _as_float(actual.get("actual_summary_model_cost_usd"))
    projected_net = _as_float(projection.get("projected_net_savings_usd"))
    payback_ratio = round(gross_savings / summary_cost, 6) if summary_cost > 0 else (None if gross_savings <= 0 else None)
    projection_realization_ratio = round(net_savings / projected_net, 6) if projected_net > 0 else None
    latency_delta = None
    if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
        latency_delta = round(_as_float(applied.get("latency_avg_ms")) - _as_float(holdout.get("latency_avg_ms")), 2)
    error_rate_delta = round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6)
    retry_rate_delta = round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6)
    provider_adoption_gate = build_provider_adoption_gate(
        (
            {
                "cohort": _cohort(item["summary"]),
                "provider_adoption_windows": item.get("provider_adoption_windows") or [],
            }
            for item in matched
        ),
        thresholds={
            key: value
            for key, value in thresholds.items()
            if key.startswith("min_provider_adoption") or key.startswith("max_applied")
        },
    )

    blockers: list[str] = []
    warnings: list[str] = []
    rollback_reasons: list[str] = []
    if matched_count < _as_int(thresholds["min_matched_samples"]):
        blockers.append("insufficient-matched-samples")
    if applied_count < _as_int(thresholds["min_canary_applied_samples"]):
        blockers.append("insufficient-canary-applied-samples")
    if bool((policy.get("canary") or {}).get("enabled")) and holdout_count < _as_int(thresholds["min_canary_holdout_samples"]):
        blockers.append("insufficient-canary-holdout-samples")
    if net_savings <= _as_float(thresholds["min_net_savings_usd"]):
        blockers.append("non-positive-net-savings")
    if payback_ratio is not None and payback_ratio < _as_float(thresholds["min_payback_ratio"]):
        blockers.append("summary-cost-payback-below-threshold")
    if projection_realization_ratio is not None and projection_realization_ratio < _as_float(thresholds["min_projection_realization_ratio"]):
        warnings.append("net-savings-below-projection")
    if _as_float(applied.get("error_rate")) > _as_float(thresholds["max_error_rate"]):
        blockers.append("applied-error-rate-above-threshold")
    if error_rate_delta > _as_float(thresholds["max_error_rate_delta"]):
        blockers.append("applied-error-rate-regression")
    if _as_float(applied.get("retry_rate")) > _as_float(thresholds["max_retry_rate"]):
        blockers.append("applied-retry-rate-above-threshold")
    if retry_rate_delta > _as_float(thresholds["max_retry_rate_delta"]):
        blockers.append("applied-retry-rate-regression")
    if applied_count and _as_float(applied.get("summary_failure_rate")) > _as_float(thresholds["max_summary_failure_rate"]):
        blockers.append("summary-failure-rate-above-threshold")
    if matched_count and (bypass_count / matched_count) > _as_float(thresholds["max_bypass_or_disabled_rate"]):
        blockers.append("bypass-or-disabled-rate-above-threshold")
    if _as_int(applied.get("safety_stop_count")) + _as_int(bypassed.get("safety_stop_count")) > _as_int(thresholds["max_safety_stop_count"]):
        blockers.append("safety-stop-events-present")
    if latency_delta is not None and latency_delta > _as_int(thresholds["max_latency_regression_ms"]):
        warnings.append("latency-regression-above-threshold")
    if provider_adoption_gate.get("blocking"):
        blockers.extend(str(code) for code in provider_adoption_gate.get("reason_codes") or [])
    else:
        warnings.extend(str(code) for code in provider_adoption_gate.get("warning_codes") or [])

    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        rollback_reasons.append("rollback-error-rate")
    if applied_count and _as_float(applied.get("summary_failure_rate")) >= _as_float(thresholds["rollback_summary_failure_rate"]):
        rollback_reasons.append("rollback-summary-failure-rate")
    if _as_int(applied.get("safety_stop_count")) + _as_int(bypassed.get("safety_stop_count")) >= _as_int(thresholds["rollback_safety_stop_count"]):
        rollback_reasons.append("rollback-safety-stop")
    if net_savings < -abs(_as_float(thresholds["rollback_negative_net_savings_usd"])):
        rollback_reasons.append("rollback-negative-net-savings")

    if rollback_reasons:
        verdict = "rollback"
        reason_codes = rollback_reasons
    elif any(code.startswith("insufficient-") for code in blockers):
        verdict = "hold"
        reason_codes = blockers
    elif blockers:
        verdict = "hold"
        reason_codes = blockers
    else:
        verdict = "promote"
        reason_codes = ["quality-gate-passed"]

    return {
        "schema": OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "warning_codes": warnings,
        "thresholds": thresholds,
        "metrics": {
            "matched_metadata_row_count": matched_count,
            "canary_applied_count": applied_count,
            "canary_holdout_count": holdout_count,
            "bypassed_or_disabled_count": bypass_count,
            "summary_failure_count": summary_failure_count,
            "net_savings_usd": round(net_savings, 8),
            "gross_savings_usd": round(gross_savings, 8),
            "summary_model_cost_usd": round(summary_cost, 8),
            "payback_ratio": payback_ratio,
            "projection_realization_ratio": projection_realization_ratio,
            "applied_minus_holdout_error_rate": error_rate_delta,
            "applied_minus_holdout_retry_rate": retry_rate_delta,
            "applied_minus_holdout_latency_avg_ms": latency_delta,
        },
        "provider_adoption_gate": provider_adoption_gate,
        "cohorts": cohorts,
        "read_only": True,
        "wrote_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
        },
    }


def build_old_context_summary_quality_gate(
    dry_run_or_review_report: Any,
    *,
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    impact = measure_old_context_summary_impact(
        dry_run_or_review_report,
        store_obj=store_obj,
        limit=limit,
        since=since,
    )
    gate = impact.get("quality_gate") if isinstance(impact.get("quality_gate"), dict) else {}
    dry_run = impact.get("dry_run") if isinstance(impact.get("dry_run"), dict) else {}
    policy = dry_run.get("policy") if isinstance(dry_run.get("policy"), dict) else {}
    projection = dry_run.get("projection") if isinstance(dry_run.get("projection"), dict) else {}
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    actual = impact.get("actual") if isinstance(impact.get("actual"), dict) else {}
    delta = impact.get("delta") if isinstance(impact.get("delta"), dict) else {}
    privacy = gate.get("privacy") if isinstance(gate.get("privacy"), dict) else {}
    if not privacy:
        privacy = impact.get("privacy") if isinstance(impact.get("privacy"), dict) else {}
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA,
        "ok": bool(gate),
        "generated_at": impact.get("generated_at") or utc_now(),
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "impact_schema": impact.get("schema"),
        "impact_ok": bool(impact.get("ok")),
        "impact_status": impact.get("status"),
        "source_error_type": (impact.get("error") or {}).get("type") if isinstance(impact.get("error"), dict) else None,
        "candidate_id": policy.get("candidate_id"),
        "rule_id": policy.get("rule_id"),
        "policy_source": policy.get("policy_source"),
        "model": policy.get("model"),
        "canary": policy.get("canary") if isinstance(policy.get("canary"), dict) else {},
        "safety_gates": policy.get("safety_gates") if isinstance(policy.get("safety_gates"), dict) else {},
        "projection": projection,
        "summary": {
            "sampled_call_count": summary.get("sampled_call_count"),
            "old_context_summary_metadata_row_count": summary.get("old_context_summary_metadata_row_count"),
            "projected_affected_metadata_row_count": summary.get("projected_affected_metadata_row_count"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "error_rate": summary.get("error_rate"),
            "retry_rate": summary.get("retry_rate"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_gross_savings_usd": summary.get("actual_gross_savings_usd"),
            "actual_summary_model_cost_usd": summary.get("actual_summary_model_cost_usd"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "net_savings_vs_projection_delta_usd": summary.get("net_savings_vs_projection_delta_usd"),
        },
        "actual": actual,
        "delta": delta,
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": bool(privacy.get("raw_old_context_included", False)),
            "generated_summaries_included": bool(privacy.get("generated_summaries_included", False)),
            "raw_prompts_included": bool(privacy.get("raw_prompts_included", False)),
            "raw_messages_included": bool(privacy.get("raw_messages_included", False)),
            "raw_request_bodies_included": bool(privacy.get("raw_request_bodies_included", False)),
            "raw_responses_included": bool(privacy.get("raw_responses_included", False)),
            "raw_transcripts_included": bool(privacy.get("raw_transcripts_included", False)),
            "provider_bodies_included": bool(privacy.get("provider_bodies_included", False)),
            "tool_payloads_included": bool(privacy.get("tool_payloads_included", False)),
            "cache_keys_included": bool(privacy.get("cache_keys_included", False)),
            "request_ids_included": bool(privacy.get("request_ids_included", False)),
            "tenant_ids_included": bool(privacy.get("tenant_ids_included", False)),
            "local_session_ids_included": bool(privacy.get("local_session_ids_included", False)),
            "file_paths_included": bool(privacy.get("file_paths_included", False)),
            "basis": "dry-run aggregate projections plus post-apply old-context summarization metadata, status buckets, latency buckets, and size-derived savings only",
        },
    }
    if gate:
        result.update({
            "verdict": gate.get("verdict"),
            "reason_codes": gate.get("reason_codes") if isinstance(gate.get("reason_codes"), list) else [],
            "warning_codes": gate.get("warning_codes") if isinstance(gate.get("warning_codes"), list) else [],
            "thresholds": gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {},
            "metrics": gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {},
            "cohorts": gate.get("cohorts") if isinstance(gate.get("cohorts"), dict) else {},
            "provider_adoption_gate": gate.get("provider_adoption_gate") if isinstance(gate.get("provider_adoption_gate"), dict) else {},
            "quality_gate": {
                "schema": gate.get("schema"),
                "verdict": gate.get("verdict"),
                "reason_codes": gate.get("reason_codes") if isinstance(gate.get("reason_codes"), list) else [],
                "warning_codes": gate.get("warning_codes") if isinstance(gate.get("warning_codes"), list) else [],
                "thresholds": gate.get("thresholds") if isinstance(gate.get("thresholds"), dict) else {},
                "metrics": gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {},
                "cohorts": gate.get("cohorts") if isinstance(gate.get("cohorts"), dict) else {},
                "provider_adoption_gate": gate.get("provider_adoption_gate") if isinstance(gate.get("provider_adoption_gate"), dict) else {},
                "privacy": result["privacy"],
            },
        })
    else:
        result["verdict"] = "hold"
        result["reason_codes"] = ["quality-gate-not-computed"]
        result["warning_codes"] = []
        result["thresholds"] = {}
        result["metrics"] = {}
        result["cohorts"] = {}
        if isinstance(impact.get("error"), dict):
            result["error"] = impact["error"]
    return result


def _delta(projection: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "matched_vs_projected_affected_delta": _as_int(actual.get("matched_metadata_row_count")) - _as_int(projection.get("projected_affected_metadata_row_count")),
        "applied_vs_projected_delta": _as_int(actual.get("actual_canary_applied_count")) - _as_int(projection.get("projected_canary_applied_count")),
        "holdout_vs_projected_delta": _as_int(actual.get("actual_canary_holdout_count")) - _as_int(projection.get("projected_canary_holdout_count")),
        "bypass_or_disabled_vs_projected_delta": _as_int(actual.get("actual_bypassed_or_disabled_count")) - _as_int(projection.get("projected_bypass_or_disabled_count")),
        "saved_tokens_vs_projection_delta": _as_int(actual.get("actual_tokens_saved_est")) - _as_int(projection.get("projected_saved_tokens")),
        "saved_chars_vs_projection_delta": _as_int(actual.get("actual_saved_chars")) - _as_int(projection.get("projected_saved_chars")),
        "net_savings_vs_projection_delta_usd": round(
            _as_float(actual.get("actual_net_savings_usd")) - _as_float(projection.get("projected_net_savings_usd")),
            8,
        ),
    }


def measure_old_context_summary_impact(
    dry_run_or_review_report: Any,
    *,
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 5000))
    result: dict[str, Any] = {
        "schema": OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "post_apply_since": since,
        "summary": {},
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "basis": "dry-run aggregate projections plus post-apply old-context summarization metadata, status buckets, latency buckets, and size-derived savings only",
        },
        "warnings": [],
    }
    dry_run = _dry_run_from_report(dry_run_or_review_report)
    if dry_run is None:
        result["error"] = {"type": "invalid_dry_run_report", "message": f"expected {OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA} or a policy review containing it"}
        return result
    if not dry_run.get("ok"):
        result["error"] = {"type": "dry_run_not_ok", "message": "old-context summary dry-run report was not successful"}
        return result

    since_value = since or dry_run.get("generated_at")
    if since_value and _parse_utc_datetime(since_value) is None:
        result["error"] = {"type": "invalid_since", "message": "post-apply since timestamp must be ISO-8601"}
        return result
    result["post_apply_since"] = since_value

    policy = dry_run.get("policy") if isinstance(dry_run.get("policy"), dict) else {}
    filters = _eligible_group_filters(dry_run)
    projection = _projection(dry_run)
    rows = _call_rows(store_obj, limit=lookback_limit, since=since_value)
    adoption_by_call = provider_adoption_windows_by_call(store_obj, [row.get("id") for row in rows])
    enriched_rows = [
        {**row, "provider_adoption_windows": adoption_by_call.get(str(row.get("id") or ""), [])}
        for row in rows
    ]
    summaries = [summary for row in enriched_rows if (summary := _row_summary(row)) is not None]
    matched = [summary for summary in summaries if _matches(summary, policy=policy, filters=filters)]
    actual = _actual(matched)
    quality_gate = decide_old_context_summary_quality_gate(
        policy=policy,
        projection=projection,
        actual=actual,
        matched=matched,
    )
    if not matched:
        result.update({
            "status": "no-post-apply-matches",
            "dry_run": {
                "generated_at": dry_run.get("generated_at"),
                "policy": {
                    "rule_id": policy.get("rule_id"),
                    "candidate_id": policy.get("candidate_id"),
                    "policy_source": policy.get("policy_source"),
                    "safety_gates": policy.get("safety_gates") if isinstance(policy.get("safety_gates"), dict) else {},
                },
                "projection": projection,
            },
            "summary": {
                "sampled_call_count": len(rows),
                "old_context_summary_metadata_row_count": len(summaries),
                "projected_affected_metadata_row_count": projection["projected_affected_metadata_row_count"],
                "actual_matched_metadata_row_count": 0,
            },
            "quality_gate": quality_gate,
            "match_filters": {
                "rule_id": policy.get("rule_id"),
                "candidate_id": policy.get("candidate_id"),
                "eligible_group_count": len(filters),
            },
            "error": {"type": "no_post_apply_matches", "message": "no local old-context summarization metadata matched the dry-run projection"},
        })
        return result

    result.update({
        "ok": True,
        "status": "matched",
        "dry_run": {
            "generated_at": dry_run.get("generated_at"),
            "policy": {
                "rule_id": policy.get("rule_id"),
                "candidate_id": policy.get("candidate_id"),
                "policy_source": policy.get("policy_source"),
                "model": policy.get("model"),
                "canary": policy.get("canary") if isinstance(policy.get("canary"), dict) else {},
                "safety_stop": policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {},
                "safety_gates": policy.get("safety_gates") if isinstance(policy.get("safety_gates"), dict) else {},
            },
            "projection": projection,
        },
        "actual": actual,
        "delta": _delta(projection, actual),
        "quality_gate": quality_gate,
        "summary": {
            "sampled_call_count": len(rows),
            "old_context_summary_metadata_row_count": len(summaries),
            "eligible_group_filter_count": len(filters),
            "projected_affected_metadata_row_count": projection["projected_affected_metadata_row_count"],
            "actual_matched_metadata_row_count": actual["matched_metadata_row_count"],
            "actual_canary_applied_count": actual["actual_canary_applied_count"],
            "actual_canary_holdout_count": actual["actual_canary_holdout_count"],
            "actual_bypassed_or_disabled_count": actual["actual_bypassed_or_disabled_count"],
            "summary_failure_count": actual["summary_failure_count"],
            "error_rate": actual["error_rate"],
            "retry_rate": actual["retry_rate"],
            "actual_tokens_saved_est": actual["actual_tokens_saved_est"],
            "actual_gross_savings_usd": actual["actual_gross_savings_usd"],
            "actual_summary_model_cost_usd": actual["actual_summary_model_cost_usd"],
            "actual_net_savings_usd": actual["actual_net_savings_usd"],
            "net_savings_vs_projection_delta_usd": _delta(projection, actual)["net_savings_vs_projection_delta_usd"],
            "quality_gate_verdict": quality_gate["verdict"],
        },
        "match_filters": {
            "rule_id": policy.get("rule_id"),
            "candidate_id": policy.get("candidate_id"),
            "eligible_groups": [
                {"source_surface": source, "category": category, "model_tier": tier, "stream": stream}
                for source, category, tier, stream in sorted(filters)
            ],
        },
    })
    return result
