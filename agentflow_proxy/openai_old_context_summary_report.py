from __future__ import annotations

import os
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_old_context_summary_opportunity.v1"
QUALITY_GATE_SCHEMA = "agentflow.openai_old_context_summary_quality_gate.v1"
DEFAULT_MIN_REQUEST_CHARS = 32_000
DEFAULT_MIN_OLDER_CONTEXT_CHARS = 8_000
DEFAULT_MAX_SUMMARY_CHARS = 4_000
DEFAULT_SUMMARY_COMPRESSION_RATIO = 0.125
DEFAULT_SUMMARY_MODEL = "gpt-5-mini"
DEFAULT_MIN_APPLIED_SAMPLES = 2
DEFAULT_MIN_HOLDOUT_SAMPLES = 1
DEFAULT_MAX_ERROR_RATE_DELTA = 0.05
DEFAULT_MAX_RETRY_RATE_DELTA = 0.10
DEFAULT_MAX_LATENCY_REGRESSION_MS = 2_000
DEFAULT_MAX_SUMMARY_FAILURE_RATE = 0.02
DEFAULT_MAX_NEGATIVE_NET_SAVINGS_RATE = 0.0
DEFAULT_MAX_EVIDENCE_AGE_HOURS = 72.0

_CHAR_BUCKET_ESTIMATES = {
    "none": 0,
    "unknown": 0,
    "lt_2k_chars": 1_000,
    "2k_8k_chars": 5_000,
    "8k_32k_chars": 16_000,
    "32k_128k_chars": 64_000,
    "gte_128k_chars": 128_000,
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _summary_provider_configured() -> bool:
    if _env_bool("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED", False):
        return True
    return bool(os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _summary_model() -> str:
    return os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MODEL") or DEFAULT_SUMMARY_MODEL


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = str(key or "unknown")
    counter[text] = counter.get(text, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _text_bucket(chars: int) -> str:
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "unknown"
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 4_000:
        return "1k_4k_tokens"
    if tokens < 16_000:
        return "4k_16k_tokens"
    if tokens < 64_000:
        return "16k_64k_tokens"
    return "gte_64k_tokens"


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


def _source_surface(row: dict[str, Any]) -> str:
    return str(row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any]) -> str:
    return str(row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _routing_feature_summary(routing: dict[str, Any]) -> dict[str, Any]:
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        value = routing.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _row_text_chars(row: dict[str, Any], routing: dict[str, Any], feature: dict[str, Any]) -> int:
    chars = _as_int(routing.get("text_chars"))
    if chars > 0:
        return chars
    bucket = str(feature.get("text_bucket") or "")
    if bucket in _CHAR_BUCKET_ESTIMATES:
        return _CHAR_BUCKET_ESTIMATES[bucket]
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * 4)


def _input_tokens(row: dict[str, Any], text_chars: int) -> int:
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    if tokens > 0:
        return tokens
    return max(0, text_chars // 4)


def _cache_status(row: dict[str, Any], cache: dict[str, Any]) -> str:
    return str(cache.get("status") or ("hit" if _as_int(row.get("cache_hit")) else "missing"))


def _estimated_older_context_chars(old_context: dict[str, Any], text_chars: int) -> int:
    bucket = str(old_context.get("older_context_text_bucket") or "unknown")
    estimated = _CHAR_BUCKET_ESTIMATES.get(bucket, 0)
    if estimated <= 0:
        return 0
    return min(estimated, max(0, text_chars))


def _project_summary(
    *,
    requested_model: str,
    older_context_chars: int,
    summary_model: str,
) -> dict[str, Any]:
    summary_chars = min(
        DEFAULT_MAX_SUMMARY_CHARS,
        max(400, int(older_context_chars * DEFAULT_SUMMARY_COMPRESSION_RATIO)),
    )
    summarized_chars = max(0, older_context_chars)
    saved_chars = max(0, summarized_chars - summary_chars)
    saved_tokens = max(0, saved_chars // 4)
    summary_input_tokens = max(1, summarized_chars // 4)
    summary_output_tokens = max(1, summary_chars // 4)
    summary_cost = estimate_cost(
        summary_model,
        summary_input_tokens,
        summary_output_tokens,
        provider="openai",
    ) or 0.0
    basis = pricing_basis(requested_model or "", provider="openai")
    input_price = _as_float(basis.get("input_usd_per_million"))
    gross = (saved_tokens / 1_000_000.0) * input_price
    return {
        "projected_summarized_chars": summarized_chars,
        "projected_summary_chars": summary_chars,
        "projected_saved_chars": saved_chars,
        "projected_saved_tokens": saved_tokens,
        "estimated_summary_cost_usd": summary_cost,
        "projected_gross_savings_usd": gross,
        "projected_net_savings_usd": gross - summary_cost,
    }


def _group_key(
    *,
    source_surface: str,
    endpoint: str,
    model_family: str,
    category: str,
    workflow_phase: str,
    stream: bool,
    text_bucket: str,
    token_bucket: str,
    has_tools: bool,
    cache_status: str,
    plateau_status: str,
    blocker: str,
) -> tuple[Any, ...]:
    return (
        source_surface,
        endpoint,
        model_family,
        category,
        workflow_phase,
        stream,
        text_bucket,
        token_bucket,
        has_tools,
        cache_status,
        plateau_status,
        blocker,
    )


def _empty_group(key: tuple[Any, ...]) -> dict[str, Any]:
    (
        source_surface,
        endpoint,
        model_family,
        category,
        workflow_phase,
        stream,
        text_bucket,
        token_bucket,
        has_tools,
        cache_status,
        plateau_status,
        blocker,
    ) = key
    return {
        "source_surface": source_surface,
        "endpoint": endpoint,
        "requested_model_family": model_family,
        "category": category,
        "workflow_phase": workflow_phase,
        "stream": stream,
        "text_bucket": text_bucket,
        "input_token_bucket": token_bucket,
        "has_tools": has_tools,
        "cache_status": cache_status,
        "context_plateau_status": plateau_status,
        "blocker": blocker,
        "call_count": 0,
        "eligible_count": 0,
        "blocked_count": 0,
        "projected_summarized_chars": 0,
        "projected_summary_chars": 0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "estimated_summary_cost_usd": 0.0,
        "projected_gross_savings_usd": 0.0,
        "projected_net_savings_usd": 0.0,
        "_status_counts": {},
    }


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    status_counts = group.pop("_status_counts", {})
    group["estimated_summary_cost_usd"] = round(_as_float(group["estimated_summary_cost_usd"]), 8)
    group["projected_gross_savings_usd"] = round(_as_float(group["projected_gross_savings_usd"]), 8)
    group["projected_net_savings_usd"] = round(_as_float(group["projected_net_savings_usd"]), 8)
    group["status_breakdown"] = _breakdown(status_counts)
    group["privacy"] = {
        "metadata_only": True,
        "raw_body_used": False,
        "group_id_derived_from_raw_body": False,
    }
    return group


def _summary_meta(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    if summary:
        return summary
    summary = crunch.get("openai_old_context_summarization") if isinstance(crunch.get("openai_old_context_summarization"), dict) else {}
    return summary


def _summary_cohort(summary: dict[str, Any]) -> str:
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    cohort = str(canary.get("cohort") or "").strip()
    status = str(summary.get("status") or "").strip()
    if status == "disabled" or not bool(summary.get("enabled", True)):
        return "disabled"
    if cohort == "canary_applied" or (status == "applied" and bool(summary.get("applied"))):
        return "canary_applied"
    if cohort in {"holdout", "canary_holdout"} or status == "holdout":
        return "canary_holdout"
    if status in {"skipped", "bypass", "blocked"}:
        return "bypassed_or_disabled"
    return "unknown"


def _summary_failed(summary: dict[str, Any]) -> bool:
    status_code = summary.get("summary_status_code")
    reasons = {str(item) for item in summary.get("reason_codes") or []}
    return (
        _as_int(status_code, 0) >= 400
        or bool(summary.get("summary_error"))
        or str(summary.get("status") or "") in {"summary_failed", "error"}
        or bool({"summary_empty_or_malformed", "summary-error", "summary-model-error"} & reasons)
    )


def _quality_key(
    row: dict[str, Any],
    routing: dict[str, Any],
    feature: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[Any, ...]:
    requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
    return (
        str(summary.get("rule_id") or "unknown-openai-old-context-summary-rule"),
        str(summary.get("summary_model") or DEFAULT_SUMMARY_MODEL),
        str(summary.get("endpoint") or feature.get("endpoint") or _endpoint(row)),
        str(
            feature.get("requested_model_family")
            or row.get("requested_model_family")
            or openai_model_family(requested_model)
            or "unknown"
        ),
        str(row.get("category") or summary.get("category") or feature.get("category") or routing.get("category") or "unknown"),
        str(summary.get("workflow_phase") or feature.get("workflow_phase") or routing.get("workflow_phase") or "unknown"),
    )


def _empty_quality_gate(key: tuple[Any, ...]) -> dict[str, Any]:
    rule_id, summary_model, endpoint, model_family, category, workflow_phase = key
    return {
        "schema": QUALITY_GATE_SCHEMA,
        "rule_id": rule_id,
        "summary_model": summary_model,
        "endpoint": endpoint,
        "requested_model_family": model_family,
        "category": category,
        "workflow_phase": workflow_phase,
        "actual_matched_metadata_row_count": 0,
        "candidate_id_count": 0,
        "candidate_ids_included": False,
        "cohort_counts": {
            "canary_applied": 0,
            "canary_holdout": 0,
            "bypassed_or_disabled": 0,
            "disabled": 0,
            "unknown": 0,
        },
        "cohort_metrics": {
            "canary_applied": _empty_quality_cohort(),
            "canary_holdout": _empty_quality_cohort(),
            "bypassed_or_disabled": _empty_quality_cohort(),
            "disabled": _empty_quality_cohort(),
            "unknown": _empty_quality_cohort(),
        },
        "totals": {
            "estimated_tokens_saved": 0,
            "estimated_gross_savings_usd": 0.0,
            "summary_cost_est_usd": 0.0,
            "estimated_net_savings_usd": 0.0,
            "summary_failure_count": 0,
            "negative_net_savings_count": 0,
        },
        "_candidate_ids": set(),
        "_status_counts": Counter(),
        "_summary_status_counts": Counter(),
        "_summary_provider_status_counts": Counter(),
        "_reason_counts": Counter(),
        "_oldest_observed_at": None,
        "_latest_observed_at": None,
    }


def _empty_quality_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_count": 0,
        "summary_failure_count": 0,
        "negative_net_savings_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "estimated_tokens_saved": 0,
        "estimated_gross_savings_usd": 0.0,
        "summary_cost_est_usd": 0.0,
        "estimated_net_savings_usd": 0.0,
    }


def _add_quality_row(gate: dict[str, Any], row: dict[str, Any], summary: dict[str, Any]) -> None:
    cohort = _summary_cohort(summary)
    if cohort not in gate["cohort_counts"]:
        cohort = "unknown"
    bucket = gate["cohort_metrics"][cohort]
    status_code = _as_int(row.get("status_code"), 0)
    retry_count = _as_int(row.get("retry_count"), 0)
    latency_ms = _as_int(row.get("latency_ms"), -1)
    net = _as_float(summary.get("estimated_net_savings_usd"))
    failed = _summary_failed(summary)
    negative = net < 0

    gate["actual_matched_metadata_row_count"] += 1
    gate["cohort_counts"][cohort] += 1
    candidate_id = str(summary.get("candidate_id") or "")
    if candidate_id:
        gate["_candidate_ids"].add(candidate_id)
    bucket["count"] += 1
    bucket["error_count"] += int(status_code >= 400)
    bucket["retry_count"] += int(retry_count > 0)
    bucket["summary_failure_count"] += int(failed)
    bucket["negative_net_savings_count"] += int(negative)
    if latency_ms >= 0:
        bucket["latency_ms_total"] += latency_ms
        bucket["latency_sample_count"] += 1
    for key, source_key in (
        ("estimated_tokens_saved", "estimated_tokens_saved"),
        ("estimated_gross_savings_usd", "estimated_gross_savings_usd"),
        ("summary_cost_est_usd", "summary_cost_est_usd"),
        ("estimated_net_savings_usd", "estimated_net_savings_usd"),
    ):
        value = _as_int(summary.get(source_key)) if key == "estimated_tokens_saved" else _as_float(summary.get(source_key))
        bucket[key] += value
        gate["totals"][key] += value
    gate["totals"]["summary_failure_count"] += int(failed)
    gate["totals"]["negative_net_savings_count"] += int(negative)
    gate["_status_counts"][_status_bucket(row.get("status_code"))] += 1
    gate["_summary_status_counts"][str(summary.get("status") or "unknown")] += 1
    if summary.get("summary_status_code") is not None:
        gate["_summary_provider_status_counts"][_status_bucket(summary.get("summary_status_code"))] += 1
    for reason in summary.get("reason_codes") or []:
        gate["_reason_counts"][str(reason or "unknown")] += 1
    created_at = row.get("created_at")
    if created_at:
        if gate["_oldest_observed_at"] is None or str(created_at) < str(gate["_oldest_observed_at"]):
            gate["_oldest_observed_at"] = str(created_at)
        if gate["_latest_observed_at"] is None or str(created_at) > str(gate["_latest_observed_at"]):
            gate["_latest_observed_at"] = str(created_at)


def _finalize_quality_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "count": count,
        "error_count": _as_int(raw.get("error_count")),
        "retry_count": _as_int(raw.get("retry_count")),
        "summary_failure_count": _as_int(raw.get("summary_failure_count")),
        "negative_net_savings_count": _as_int(raw.get("negative_net_savings_count")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_count")) / count, 6) if count else 0.0,
        "summary_failure_rate": round(_as_int(raw.get("summary_failure_count")) / count, 6) if count else 0.0,
        "negative_net_savings_rate": round(_as_int(raw.get("negative_net_savings_count")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "estimated_tokens_saved": _as_int(raw.get("estimated_tokens_saved")),
        "estimated_gross_savings_usd": round(_as_float(raw.get("estimated_gross_savings_usd")), 8),
        "summary_cost_est_usd": round(_as_float(raw.get("summary_cost_est_usd")), 8),
        "estimated_net_savings_usd": round(_as_float(raw.get("estimated_net_savings_usd")), 8),
    }


def _freshness(latest_observed_at: str | None, *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    latest = _parse_time(latest_observed_at)
    if latest is None:
        return {"stale": False, "age_hours": None, "max_age_hours": round(float(max_age_hours), 3)}
    age_hours = (now.astimezone(timezone.utc) - latest).total_seconds() / 3600.0
    return {"stale": age_hours > max_age_hours, "age_hours": round(age_hours, 3), "max_age_hours": round(float(max_age_hours), 3)}


def _finalize_quality_gate(
    gate: dict[str, Any],
    *,
    now: datetime,
    min_applied_samples: int,
    min_holdout_samples: int,
    max_error_rate_delta: float,
    max_retry_rate_delta: float,
    max_latency_regression_ms: int,
    max_summary_failure_rate: float,
    max_negative_net_savings_rate: float,
    max_evidence_age_hours: float,
) -> dict[str, Any]:
    gate["candidate_id_count"] = len(gate.pop("_candidate_ids", set()))
    cohorts = {
        key: _finalize_quality_cohort(value)
        for key, value in gate["cohort_metrics"].items()
    }
    gate["cohort_metrics"] = cohorts
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    deltas = {
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
    }
    thresholds = {
        "min_canary_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_canary_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "max_summary_failure_rate": round(float(max_summary_failure_rate), 6),
        "max_negative_net_savings_rate": round(float(max_negative_net_savings_rate), 6),
        "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
    }
    freshness = _freshness(gate.get("_latest_observed_at"), now=now, max_age_hours=max_evidence_age_hours)
    reason_codes: list[str] = []
    rollback_reasons: list[str] = []
    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    active_count = applied_count + holdout_count + _as_int(gate["cohort_counts"].get("bypassed_or_disabled"))
    if active_count == 0 and _as_int(gate["cohort_counts"].get("disabled")) > 0:
        verdict = "disabled"
        reason_codes = ["summary-policy-disabled"]
    else:
        if applied_count < thresholds["min_canary_applied_samples"]:
            reason_codes.append("insufficient-applied-samples")
        if holdout_count < thresholds["min_canary_holdout_samples"]:
            reason_codes.append("insufficient-holdout-samples")
        if reason_codes:
            verdict = "needs_more_samples"
        else:
            if _as_float(applied.get("summary_failure_rate")) > thresholds["max_summary_failure_rate"]:
                rollback_reasons.append("summary-failure-rate")
            if _as_float(applied.get("negative_net_savings_rate")) > thresholds["max_negative_net_savings_rate"]:
                rollback_reasons.append("negative-net-savings-rate")
            if rollback_reasons:
                verdict = "rollback"
                reason_codes = rollback_reasons
            else:
                if freshness.get("stale"):
                    reason_codes.append("stale-evidence")
                if _as_float(deltas["applied_minus_holdout_error_rate"]) > thresholds["max_error_rate_delta"]:
                    reason_codes.append("error-rate-regression")
                if _as_float(deltas["applied_minus_holdout_retry_rate"]) > thresholds["max_retry_rate_delta"]:
                    reason_codes.append("retry-rate-regression")
                if latency_delta is not None and latency_delta > thresholds["max_latency_regression_ms"]:
                    reason_codes.append("latency-regression")
                verdict = "hold" if reason_codes else "promote"
                if not reason_codes:
                    reason_codes = ["quality-gate-passed"]

    status_counts = gate.pop("_status_counts", Counter())
    summary_status_counts = gate.pop("_summary_status_counts", Counter())
    summary_provider_status_counts = gate.pop("_summary_provider_status_counts", Counter())
    reason_counts = gate.pop("_reason_counts", Counter())
    gate["oldest_observed_at"] = gate.pop("_oldest_observed_at")
    gate["latest_observed_at"] = gate.pop("_latest_observed_at")
    for key in ("estimated_gross_savings_usd", "summary_cost_est_usd", "estimated_net_savings_usd"):
        gate["totals"][key] = round(_as_float(gate["totals"][key]), 8)
    gate.update({
        "verdict": verdict,
        "reason_codes": reason_codes,
        "thresholds": thresholds,
        "applied_vs_holdout_deltas": deltas,
        "freshness": freshness,
        "status_code_buckets": _breakdown(dict(status_counts)),
        "summary_status_buckets": _breakdown(dict(summary_status_counts)),
        "summary_provider_status_buckets": _breakdown(dict(summary_provider_status_counts)),
        "summary_reason_buckets": _breakdown(dict(reason_counts)),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "generated_summaries_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "candidate_ids_included": False,
        },
    })
    return gate


def _row_blocker(
    *,
    feature: dict[str, Any],
    old_context: dict[str, Any],
    endpoint: str,
    text_chars: int,
    older_context_chars: int,
    has_tools: bool,
    cache_status: str,
    summary_provider_configured: bool,
) -> str:
    if not feature or not old_context:
        return "blocked_missing_body_or_feature"
    if endpoint not in {"responses", "chat_completions"}:
        return "unsupported_endpoint"
    if not summary_provider_configured:
        return "summary_provider_not_configured"
    if cache_status == "hit":
        return "cache_hit_no_upstream_savings"
    if has_tools:
        return "tool_protocol_risk"
    if _as_int(old_context.get("older_context_item_count")) <= 0:
        return "no_older_context"
    if text_chars < DEFAULT_MIN_REQUEST_CHARS:
        return "request_below_min_chars"
    if older_context_chars < DEFAULT_MIN_OLDER_CONTEXT_CHARS:
        return "older_context_below_min_chars"
    return "eligible"


def build_openai_old_context_summary_report(
    store_obj: Any,
    limit: int = 1000,
    *,
    summary_provider_configured: bool | None = None,
    summary_model: str | None = None,
    min_applied_samples: int = DEFAULT_MIN_APPLIED_SAMPLES,
    min_holdout_samples: int = DEFAULT_MIN_HOLDOUT_SAMPLES,
    max_error_rate_delta: float = DEFAULT_MAX_ERROR_RATE_DELTA,
    max_retry_rate_delta: float = DEFAULT_MAX_RETRY_RATE_DELTA,
    max_latency_regression_ms: int = DEFAULT_MAX_LATENCY_REGRESSION_MS,
    max_summary_failure_rate: float = DEFAULT_MAX_SUMMARY_FAILURE_RATE,
    max_negative_net_savings_rate: float = DEFAULT_MAX_NEGATIVE_NET_SAVINGS_RATE,
    max_evidence_age_hours: float = DEFAULT_MAX_EVIDENCE_AGE_HOURS,
    now: datetime | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    provider_ready = _summary_provider_configured() if summary_provider_configured is None else bool(summary_provider_configured)
    model = summary_model or _summary_model()
    now_dt = now or datetime.now(timezone.utc)
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, crunch_json, routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    blocker_totals: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    model_family_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    quality_gates: dict[tuple[Any, ...], dict[str, Any]] = {}
    openai_count = 0
    eligible_count = 0
    blocked_count = 0
    feature_rows = 0
    summary_metadata_rows = 0
    totals = {
        "projected_summarized_chars": 0,
        "projected_summary_chars": 0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "estimated_summary_cost_usd": 0.0,
        "projected_gross_savings_usd": 0.0,
        "projected_net_savings_usd": 0.0,
    }

    for row in rows:
        if str(row.get("provider") or "").lower() != "openai":
            continue
        openai_count += 1
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        feature = _routing_feature_summary(routing)
        summary_meta = _summary_meta(row)
        if feature:
            feature_rows += 1
        if summary_meta:
            summary_metadata_rows += 1
        old_context = feature.get("old_context") if isinstance(feature.get("old_context"), dict) else {}
        source_surface = str(feature.get("source_surface") or _source_surface(row))
        endpoint = str(feature.get("endpoint") or _endpoint(row))
        requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
        model_family = str(
            feature.get("requested_model_family")
            or row.get("requested_model_family")
            or openai_model_family(requested_model)
            or "unknown"
        )
        category = str(row.get("category") or feature.get("category") or routing.get("category") or "unknown")
        workflow_phase = str(feature.get("workflow_phase") or routing.get("workflow_phase") or "unknown")
        stream = bool(_as_int(row.get("stream")) or feature.get("stream"))
        has_tools = bool(feature.get("has_tools") or routing.get("has_tools"))
        text_chars = _row_text_chars(row, routing, feature)
        input_tokens = _input_tokens(row, text_chars)
        text_bucket = _text_bucket(text_chars)
        token_bucket = _token_bucket(input_tokens)
        cache_state = _cache_status(row, cache)
        plateau_status = str(routing.get("context_plateau_status") or "unknown")
        older_context_chars = _estimated_older_context_chars(old_context, text_chars)
        blocker = _row_blocker(
            feature=feature,
            old_context=old_context,
            endpoint=endpoint,
            text_chars=text_chars,
            older_context_chars=older_context_chars,
            has_tools=has_tools,
            cache_status=cache_state,
            summary_provider_configured=provider_ready,
        )
        projection = {}
        if blocker == "eligible":
            projection = _project_summary(
                requested_model=requested_model,
                older_context_chars=older_context_chars,
                summary_model=model,
            )
            eligible_count += 1
            for key, value in projection.items():
                totals[key] += value
        else:
            blocked_count += 1
            _increment(blocker_totals, blocker)

        _increment(endpoint_counts, endpoint)
        _increment(surface_counts, source_surface)
        _increment(category_counts, category)
        _increment(model_family_counts, model_family)
        _increment(cache_status_counts, cache_state)

        key = _group_key(
            source_surface=source_surface,
            endpoint=endpoint,
            model_family=model_family,
            category=category,
            workflow_phase=workflow_phase,
            stream=stream,
            text_bucket=text_bucket,
            token_bucket=token_bucket,
            has_tools=has_tools,
            cache_status=cache_state,
            plateau_status=plateau_status,
            blocker=blocker,
        )
        group = groups.setdefault(key, _empty_group(key))
        group["call_count"] += 1
        if blocker == "eligible":
            group["eligible_count"] += 1
            for key_name, value in projection.items():
                group[key_name] += value
        else:
            group["blocked_count"] += 1
        _increment(group["_status_counts"], _status_bucket(row.get("status_code")))

        if summary_meta:
            quality_key = _quality_key(row, routing, feature, summary_meta)
            quality_gate = quality_gates.setdefault(quality_key, _empty_quality_gate(quality_key))
            _add_quality_row(quality_gate, row, summary_meta)

    output_groups = [_finalize_group(group) for group in groups.values()]
    output_groups.sort(
        key=lambda item: (
            _as_int(item.get("eligible_count")),
            _as_int(item.get("call_count")),
            _as_float(item.get("projected_net_savings_usd")),
        ),
        reverse=True,
    )

    for key in ("estimated_summary_cost_usd", "projected_gross_savings_usd", "projected_net_savings_usd"):
        totals[key] = round(_as_float(totals[key]), 8)

    output_quality_gates = [
        _finalize_quality_gate(
            gate,
            now=now_dt,
            min_applied_samples=min_applied_samples,
            min_holdout_samples=min_holdout_samples,
            max_error_rate_delta=max_error_rate_delta,
            max_retry_rate_delta=max_retry_rate_delta,
            max_latency_regression_ms=max_latency_regression_ms,
            max_summary_failure_rate=max_summary_failure_rate,
            max_negative_net_savings_rate=max_negative_net_savings_rate,
            max_evidence_age_hours=max_evidence_age_hours,
        )
        for gate in quality_gates.values()
    ]
    output_quality_gates.sort(
        key=lambda item: (
            str(item.get("verdict") or ""),
            -_as_int(item.get("actual_matched_metadata_row_count")),
            str(item.get("rule_id") or ""),
            str(item.get("endpoint") or ""),
        )
    )
    verdict_counts: dict[str, int] = {}
    quality_reason_counts: dict[str, int] = {}
    status_code_counts: dict[str, int] = {}
    quality_totals = {
        "estimated_tokens_saved": 0,
        "estimated_gross_savings_usd": 0.0,
        "summary_cost_est_usd": 0.0,
        "estimated_net_savings_usd": 0.0,
        "summary_failure_count": 0,
        "negative_net_savings_count": 0,
    }
    for gate in output_quality_gates:
        _increment(verdict_counts, gate.get("verdict"))
        for reason in gate.get("reason_codes") or []:
            _increment(quality_reason_counts, reason)
        for bucket in gate.get("status_code_buckets") or []:
            _increment(status_code_counts, bucket.get("value"), _as_int(bucket.get("count")))
        gate_totals = gate.get("totals") if isinstance(gate.get("totals"), dict) else {}
        for key in quality_totals:
            quality_totals[key] += _as_int(gate_totals.get(key)) if key.endswith("_count") or key == "estimated_tokens_saved" else _as_float(gate_totals.get(key))
    for key in ("estimated_gross_savings_usd", "summary_cost_est_usd", "estimated_net_savings_usd"):
        quality_totals[key] = round(_as_float(quality_totals[key]), 8)

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": openai_count,
            "feature_row_count": feature_rows,
            "openai_old_context_summary_metadata_row_count": summary_metadata_rows,
            "eligible_count": eligible_count,
            "blocked_count": blocked_count,
            "projected_summarized_chars": totals["projected_summarized_chars"],
            "projected_summary_chars": totals["projected_summary_chars"],
            "projected_saved_chars": totals["projected_saved_chars"],
            "projected_saved_tokens": totals["projected_saved_tokens"],
            "estimated_summary_cost_usd": totals["estimated_summary_cost_usd"],
            "projected_gross_savings_usd": totals["projected_gross_savings_usd"],
            "projected_net_savings_usd": totals["projected_net_savings_usd"],
        },
        "quality_gate_summary": {
            "schema": "agentflow.openai_old_context_summary_quality_gate_summary.v1",
            "quality_gate_count": len(output_quality_gates),
            "actual_matched_metadata_row_count": sum(_as_int(item.get("actual_matched_metadata_row_count")) for item in output_quality_gates),
            "canary_applied_count": sum(_as_int((item.get("cohort_counts") or {}).get("canary_applied")) for item in output_quality_gates),
            "canary_holdout_count": sum(_as_int((item.get("cohort_counts") or {}).get("canary_holdout")) for item in output_quality_gates),
            "summary_failure_count": _as_int(quality_totals["summary_failure_count"]),
            "negative_net_savings_count": _as_int(quality_totals["negative_net_savings_count"]),
            "estimated_tokens_saved": _as_int(quality_totals["estimated_tokens_saved"]),
            "estimated_gross_savings_usd": quality_totals["estimated_gross_savings_usd"],
            "summary_cost_est_usd": quality_totals["summary_cost_est_usd"],
            "estimated_net_savings_usd": quality_totals["estimated_net_savings_usd"],
            "verdict_counts": _breakdown(verdict_counts),
            "reason_code_counts": _breakdown(quality_reason_counts),
            "status_code_buckets": _breakdown(status_code_counts),
        },
        "measurement_policy": {
            "schema": "agentflow.openai_old_context_summary_measurement_policy.v1",
            "read_only": True,
            "provider_calls_made": False,
            "summary_provider_configured": provider_ready,
            "summary_model": model,
            "min_request_chars": DEFAULT_MIN_REQUEST_CHARS,
            "min_older_context_chars": DEFAULT_MIN_OLDER_CONTEXT_CHARS,
            "max_summary_chars": DEFAULT_MAX_SUMMARY_CHARS,
            "summary_compression_ratio": DEFAULT_SUMMARY_COMPRESSION_RATIO,
            "raw_body_required_for_report_output": False,
            "missing_feature_blocker": "blocked_missing_body_or_feature",
            "quality_gate_thresholds": {
                "min_canary_applied_samples": max(0, _as_int(min_applied_samples)),
                "min_canary_holdout_samples": max(0, _as_int(min_holdout_samples)),
                "max_error_rate_delta": round(float(max_error_rate_delta), 6),
                "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
                "max_latency_regression_ms": _as_int(max_latency_regression_ms),
                "max_summary_failure_rate": round(float(max_summary_failure_rate), 6),
                "max_negative_net_savings_rate": round(float(max_negative_net_savings_rate), 6),
                "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
            },
        },
        "endpoint_breakdown": _breakdown(endpoint_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "category_breakdown": _breakdown(category_counts),
        "requested_model_family_breakdown": _breakdown(model_family_counts),
        "cache_status_breakdown": _breakdown(cache_status_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "groups": output_groups,
        "quality_gates": output_quality_gates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "generated_summaries_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "session_ids_included": False,
            "candidate_ids_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local calls table metadata plus sanitized OpenAI feature/cache/routing summaries only",
        },
    }
