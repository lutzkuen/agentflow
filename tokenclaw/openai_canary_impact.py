from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.provider_adoption_gate import (
    build_provider_adoption_gate,
    provider_adoption_thresholds,
    provider_adoption_windows_by_call,
)
from tokenclaw.store import utc_now


SCHEMA = "agentflow.openai_canary_impact.v1"
VERDICT_SCHEMA = "agentflow.openai_canary_promotion_verdict.v1"
ROUTING_PROMOTION_VERDICT_SCHEMA = "agentflow.openai_routing_promotion_verdict.v1"
DEFAULT_MIN_HOLDOUT_VOLUME = 10

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


def _counter_total(counter: Counter[str]) -> int:
    return sum(_as_int(value) for value in counter.values())


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "observed_savings_usd": 0.0,
        "projected_savings_usd": 0.0,
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "count": count,
        "error_count": _as_int(raw.get("error_count")),
        "retry_count": _as_int(raw.get("retry_count")),
        "fallback_count": _as_int(raw.get("fallback_count")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_count")) / count, 6) if count else 0.0,
        "fallback_rate": round(_as_int(raw.get("fallback_count")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "observed_savings_usd": round(_as_float(raw.get("observed_savings_usd")), 8),
        "projected_savings_usd": round(_as_float(raw.get("projected_savings_usd")), 8),
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


def _canary_reason(canary: dict[str, Any]) -> str:
    return str(canary.get("reason") or canary.get("status") or canary.get("cohort") or "unknown").strip() or "unknown"


def _durable_unknown_lifecycle_reason(canary: dict[str, Any]) -> tuple[str, str]:
    reason = _canary_reason(canary).lower()
    status = str(canary.get("status") or "").strip().lower()
    cohort = str(canary.get("cohort") or "").strip().lower()

    safe_bypass_markers = {
        "disabled",
        "noop",
        "canary-disabled",
        "openai-routing-disabled",
        "target-model-already-selected",
        "outside-canary-fraction",
        "outside-canary-and-holdout",
        "not-selected",
        "not_selected",
    }
    if status in {"disabled", "noop"} or reason in safe_bypass_markers:
        return "safe_bypass", reason

    unsupported_markers = {
        "category-not-enabled",
        "request-too-small",
        "request-too-large",
        "requested-model-not-enabled",
        "streaming-not-enabled",
        "tool-request-not-enabled",
        "token-estimate-too-small",
        "token-estimate-too-large",
        "workflow-phase-not-enabled",
    }
    if status == "ineligible" or reason in unsupported_markers:
        return "unsupported_shape", reason

    if "stale" in reason or "stale" in status or "stale" in cohort:
        return "promotion_blocker", "stale-canary-metadata"
    if "shape" in reason and ("mismatch" in reason or "miss" in reason):
        return "promotion_blocker", "shape-mismatch"
    if "marker" in reason or (not status and not cohort):
        return "promotion_blocker", "missing-canary-marker"
    if reason in {"missing-status", "unknown", "mystery"} or status in {"", "unknown", "mystery"} or cohort in {"", "unknown", "mystery", "none"}:
        return "promotion_blocker", "canary-lifecycle-logging-gap"
    return "promotion_blocker", "unrecognized-canary-lifecycle-state"


def _classify_canary_lifecycle_reason(cohort: str, canary: dict[str, Any]) -> tuple[str, str]:
    reason = _canary_reason(canary)
    status = str(canary.get("status") or "").strip()
    reason_l = reason.lower()
    if cohort in {"canary_applied", "canary_holdout"}:
        return "covered", reason
    if cohort == "safety_stopped" or "safety-stop" in reason_l:
        return "promotion_blocker", reason
    if cohort == "unknown":
        return _durable_unknown_lifecycle_reason(canary)
    if cohort == "skipped" and reason_l in {
        "outside-canary-fraction",
        "outside-canary-and-holdout",
        "not-selected",
        "not_selected",
    }:
        return "safe_bypass", reason
    if status in {"disabled", "noop"} or reason_l in {"disabled", "noop", "canary-disabled", "openai-routing-disabled"}:
        return "safe_bypass", reason
    unsupported_markers = {
        "category-not-enabled",
        "request-too-small",
        "request-too-large",
        "requested-model-not-enabled",
        "streaming-not-enabled",
        "tool-request-not-enabled",
        "token-estimate-too-small",
        "token-estimate-too-large",
        "workflow-phase-not-enabled",
    }
    if status == "ineligible" or reason_l in unsupported_markers:
        return "unsupported_shape", reason
    return "promotion_blocker", reason


def _observed_savings(row: dict[str, Any], canary: dict[str, Any], cohort: str) -> float:
    if cohort != "canary_applied":
        return 0.0
    if row.get("cost_baseline_usd") is not None and row.get("cost_est_usd") is not None:
        return _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd"))
    return _as_float(canary.get("projected_input_savings_usd"))


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    if since:
        rows = store_obj.conn.execute(
            """
            select id, created_at, coalesce(provider, 'anthropic') as provider, source_surface,
                   endpoint, requested_model, routed_model, stream, status_code, latency_ms,
                   retry_count, cost_est_usd, cost_baseline_usd, routing_json, crunch_json,
                   cache_json, category
            from calls
            where created_at >= ?
              and routing_json is not null
              and coalesce(provider, 'anthropic') = 'openai'
            order by created_at desc
            limit ?
            """,
            (since, capped),
        ).fetchall()
    else:
        rows = store_obj.conn.execute(
            """
            select id, created_at, coalesce(provider, 'anthropic') as provider, source_surface,
                   endpoint, requested_model, routed_model, stream, status_code, latency_ms,
                   retry_count, cost_est_usd, cost_baseline_usd, routing_json, crunch_json,
                   cache_json, category
            from calls
            where routing_json is not null
              and coalesce(provider, 'anthropic') = 'openai'
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    return [dict(row) for row in rows]


def _candidate_key(canary: dict[str, Any]) -> str:
    return str(
        canary.get("candidate_id")
        or canary.get("target_candidate_id")
        or canary.get("promotion_action_id")
        or canary.get("rule_id")
        or canary.get("policy_id")
        or "unknown-openai-canary"
    )


def _new_aggregate(candidate_id: str, canary: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "rule_id": canary.get("rule_id") or canary.get("policy_id"),
        "promotion_action_id": canary.get("promotion_action_id"),
        "target_candidate_id": canary.get("target_candidate_id"),
        "policy_id": canary.get("policy_id"),
        "policy_source": canary.get("policy_source"),
        "source_surface": canary.get("source_surface") or "openai_provider_request",
        "app_family": canary.get("app_family") or "generic_openai",
        "original_model": canary.get("original_model") or canary.get("requested_model"),
        "target_model": canary.get("target_model"),
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
        "skipped_reason_buckets": Counter(),
        "unknown_reason_buckets": Counter(),
        "safe_bypass_reason_buckets": Counter(),
        "unsupported_shape_reason_buckets": Counter(),
        "promotion_blocker_reason_buckets": Counter(),
        "unclassified_reason_buckets": Counter(),
        "category_buckets": Counter(),
        "endpoint_buckets": Counter(),
        "cache_interactions": Counter(),
        "crunch_interactions": Counter(),
        "provider_adoption_observations": [],
        "latest_observed_at": None,
        "oldest_observed_at": None,
    }


def _add_row(
    aggregate: dict[str, Any],
    row: dict[str, Any],
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
    if canary.get("fallback_reason"):
        bucket["fallback_count"] += 1
    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        bucket["latency_ms_total"] += latency_ms
        bucket["latency_sample_count"] += 1
    bucket["observed_savings_usd"] += _observed_savings(row, canary, cohort)
    bucket["projected_savings_usd"] += _as_float(canary.get("projected_input_savings_usd"))

    aggregate["status_buckets"][_status_bucket(row.get("status_code"))] += 1
    aggregate["reason_buckets"][_reason_code(canary.get("reason"))] += 1
    classification, classified_reason = _classify_canary_lifecycle_reason(cohort, canary)
    if cohort == "skipped":
        aggregate["skipped_reason_buckets"][classified_reason] += 1
    if cohort == "unknown":
        aggregate["unknown_reason_buckets"][classified_reason] += 1
    if cohort in {"skipped", "unknown", "bypassed_or_disabled"}:
        if classification == "safe_bypass":
            aggregate["safe_bypass_reason_buckets"][classified_reason] += 1
        elif classification == "unsupported_shape":
            aggregate["unsupported_shape_reason_buckets"][classified_reason] += 1
        elif classification == "promotion_blocker":
            aggregate["promotion_blocker_reason_buckets"][classified_reason] += 1
        elif classification == "unclassified":
            aggregate["unclassified_reason_buckets"][classified_reason] += 1
    aggregate["category_buckets"][_reason_code(canary.get("category") or row.get("category"), "unknown")] += 1
    aggregate["endpoint_buckets"][_reason_code(row.get("endpoint") or row.get("source_surface"), "unknown")] += 1

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
    observed_savings = _as_float(applied.get("observed_savings_usd"))
    projected_savings = _as_float(applied.get("projected_savings_usd"))
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if _as_int(applied.get("count")) < _as_int(thresholds["min_canary_applied_samples"]):
        reason_codes.append("insufficient-applied-samples")
    if _as_int(holdout.get("count")) < _as_int(thresholds["min_canary_holdout_samples"]):
        reason_codes.append("insufficient-holdout-samples")
    if reason_codes:
        return {"verdict": "needs_eval", "reason_codes": reason_codes, "warning_codes": warning_codes}

    if _as_int(safety.get("count")) > 0:
        reason_codes.append("safety-stop-observed")
    if observed_savings < 0:
        reason_codes.append("negative-observed-savings")
    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        reason_codes.append("rollback-error-rate")
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
    latency_delta = deltas.get("applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and _as_float(latency_delta) > _as_float(thresholds["max_latency_regression_ms"]):
        reason_codes.append("latency-regression")
    gate = provider_adoption_gate or {}
    if gate.get("blocking"):
        reason_codes.extend(str(code) for code in gate.get("reason_codes") or [])
    else:
        warning_codes.extend(str(code) for code in gate.get("warning_codes") or [])
    if projected_savings > 0:
        ratio = observed_savings / projected_savings
        if ratio < _as_float(thresholds["min_projection_realization_ratio"]):
            reason_codes.append("savings-below-projection")
        elif observed_savings > 0:
            warning_codes.append("target-savings-met")
    elif observed_savings <= 0:
        reason_codes.append("non-positive-observed-savings")

    verdict = "hold" if reason_codes else "widen"
    if not reason_codes:
        reason_codes = ["target-savings-met"]
    return {"verdict": verdict, "reason_codes": sorted(set(reason_codes)), "warning_codes": sorted(set(warning_codes))}


def _skipped_unknown_classification(aggregate: dict[str, Any]) -> dict[str, Any]:
    safe_bypass_count = _counter_total(aggregate["safe_bypass_reason_buckets"])
    unsupported_shape_count = _counter_total(aggregate["unsupported_shape_reason_buckets"])
    promotion_blocker_count = _counter_total(aggregate["promotion_blocker_reason_buckets"])
    unclassified_count = _counter_total(aggregate["unclassified_reason_buckets"])
    return {
        "schema": "agentflow.openai_routing_canary_skipped_unknown_classification.v1",
        "safe_bypass_count": safe_bypass_count,
        "unsupported_shape_count": unsupported_shape_count,
        "promotion_blocker_count": promotion_blocker_count,
        "unclassified_count": unclassified_count,
        "classified_count": safe_bypass_count + unsupported_shape_count + promotion_blocker_count,
        "requires_operator_review": bool(unsupported_shape_count or promotion_blocker_count or unclassified_count),
        "safe_bypass_reason_breakdown": _counter_rows(aggregate["safe_bypass_reason_buckets"]),
        "unsupported_shape_reason_breakdown": _counter_rows(aggregate["unsupported_shape_reason_buckets"]),
        "promotion_blocker_reason_breakdown": _counter_rows(aggregate["promotion_blocker_reason_buckets"]),
        "unclassified_reason_breakdown": _counter_rows(aggregate["unclassified_reason_buckets"]),
    }


def _routing_promotion_verdict(
    aggregate: dict[str, Any],
    *,
    cohorts: dict[str, dict[str, Any]],
    stale: dict[str, Any],
) -> dict[str, Any]:
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    safety = cohorts["safety_stopped"]
    skipped = cohorts["skipped"]
    bypassed = cohorts["bypassed_or_disabled"]
    unknown = cohorts["unknown"]
    classification = _skipped_unknown_classification(aggregate)
    reasons: Counter[str] = Counter()

    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    observed_count = sum(_as_int(cohort.get("count")) for cohort in cohorts.values())
    total_error_count = sum(_as_int(cohort.get("error_count")) for cohort in cohorts.values())
    total_fallback_count = sum(_as_int(cohort.get("fallback_count")) for cohort in cohorts.values())
    total_retry_count = sum(_as_int(cohort.get("retry_count")) for cohort in cohorts.values())
    if observed_count <= 0:
        reasons["missing-canary-lifecycle-evidence"] += 1
    if applied_count <= 0:
        reasons["missing-applied-coverage"] += 1
    if holdout_count <= 0:
        reasons["missing-holdout-coverage"] += 1
        if 0 < observed_count < DEFAULT_MIN_HOLDOUT_VOLUME:
            reasons["insufficient-volume-for-holdout"] += max(observed_count, 1)
    if _as_int(safety.get("count")):
        reasons["safety-stop-observed"] += _as_int(safety.get("count"))
    if total_error_count:
        reasons["error-observed"] += total_error_count
    if total_fallback_count:
        reasons["fallback-observed"] += total_fallback_count
    if total_retry_count:
        reasons["retry-observed"] += total_retry_count
    if _as_int(classification.get("unclassified_count")):
        reasons["unclassified-canary-lifecycle-rows"] += _as_int(classification.get("unclassified_count"))
    if _as_int(classification.get("unsupported_shape_count")):
        reasons["skipped-canary-unsupported-shape"] += _as_int(classification.get("unsupported_shape_count"))
    if _as_int(classification.get("promotion_blocker_count")):
        reasons["skipped-canary-promotion-blocker"] += _as_int(classification.get("promotion_blocker_count"))
    if stale.get("stale"):
        reasons["stale-evidence"] += observed_count
    if _as_float(applied.get("observed_savings_usd")) <= 0 and _as_float(applied.get("projected_savings_usd")) <= 0:
        reasons["non-positive-observed-or-projected-savings"] += 1

    hard_blockers = {
        "safety-stop-observed",
        "error-observed",
        "fallback-observed",
        "retry-observed",
        "skipped-canary-promotion-blocker",
        "non-positive-observed-or-projected-savings",
    }
    keep_staged_reasons = {
        "missing-canary-lifecycle-evidence",
        "missing-applied-coverage",
        "missing-holdout-coverage",
        "insufficient-volume-for-holdout",
        "unclassified-canary-lifecycle-rows",
        "skipped-canary-unsupported-shape",
        "stale-evidence",
    }
    reason_codes = sorted(reasons)
    if any(reason in hard_blockers for reason in reason_codes):
        verdict = "blocked"
        next_action = "review-openai-routing-canary-blockers"
    elif any(reason in keep_staged_reasons for reason in reason_codes):
        verdict = "keep-staged"
        if "skipped-canary-unsupported-shape" in reasons:
            next_action = "narrow-openai-routing-canary-shape"
        elif "unclassified-canary-lifecycle-rows" in reasons:
            next_action = "classify-openai-routing-canary-skipped-unknown"
        else:
            next_action = "collect-openai-routing-canary-evidence"
    else:
        verdict = "promotion-ready"
        next_action = "draft-openai-routing-rule"
        reason_codes = ["promotion-ready"]
        reasons["promotion-ready"] += 1

    return {
        "schema": ROUTING_PROMOTION_VERDICT_SCHEMA,
        "optimization_family": "openai_local_routing",
        "action_family": "routing",
        "verdict": verdict,
        "promotion_ready": verdict == "promotion-ready",
        "next_action": next_action,
        "reason_codes": reason_codes,
        "reason_code_breakdown": _counter_rows(reasons),
        "skipped_reason_breakdown": _counter_rows(aggregate["skipped_reason_buckets"]),
        "unknown_reason_breakdown": _counter_rows(aggregate["unknown_reason_buckets"]),
        "skipped_unknown_classification": classification,
        "evidence": {
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": _as_int(skipped.get("count")),
            "bypassed_or_disabled_count": _as_int(bypassed.get("count")),
            "unknown_count": _as_int(unknown.get("count")),
            "safety_stop_count": _as_int(safety.get("count")),
            "error_count": total_error_count,
            "fallback_count": total_fallback_count,
            "retry_count": total_retry_count,
            "observed_count": observed_count,
            "observed_savings_usd": round(_as_float(applied.get("observed_savings_usd")), 8),
            "projected_savings_usd": round(sum(_as_float(cohort.get("projected_savings_usd")) for cohort in cohorts.values()), 8),
            "latest_observed_at": aggregate.get("latest_observed_at"),
            "oldest_observed_at": aggregate.get("oldest_observed_at"),
            "stale_evidence": stale,
        },
        "quality_gates": {
            "requires_applied_coverage": True,
            "requires_holdout_coverage": True,
            "requires_classified_skipped_unknown_rows": True,
            "requires_zero_safety_stops": True,
            "requires_zero_errors": True,
            "requires_zero_fallbacks": True,
            "requires_zero_retries": True,
            "requires_positive_savings": True,
        },
        "privacy": _privacy_summary(),
    }


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
    max_latency_regression_ms: int,
    rollback_error_rate: float,
    min_projection_realization_ratio: float,
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
        "applied_minus_holdout_latency_avg_ms": latency_delta,
    }
    stale = _stale_evidence(
        aggregate.get("latest_observed_at"),
        now=now,
        max_age_hours=max_evidence_age_hours,
    )
    thresholds = {
        "min_canary_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_canary_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_fallback_rate_delta": round(float(max_fallback_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
        "min_projection_realization_ratio": round(float(min_projection_realization_ratio), 6),
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
    routing_verdict = _routing_promotion_verdict(aggregate, cohorts=cohorts, stale=stale)
    projected_total = sum(_as_float(cohort.get("projected_savings_usd")) for cohort in cohorts.values())
    observed_total = _as_float(applied.get("observed_savings_usd"))
    return {
        "schema": VERDICT_SCHEMA,
        "candidate_id": aggregate["candidate_id"],
        "rule_id": aggregate.get("rule_id"),
        "promotion_action_id": aggregate.get("promotion_action_id"),
        "target_candidate_id": aggregate.get("target_candidate_id"),
        "policy_id": aggregate.get("policy_id"),
        "policy_source": aggregate.get("policy_source"),
        "optimization_family": "openai_local_routing",
        "action_family": "routing",
        "source_surface": aggregate.get("source_surface"),
        "app_family": aggregate.get("app_family"),
        "original_model": aggregate.get("original_model"),
        "candidate_target_model": aggregate.get("target_model"),
        "canary_fraction": aggregate.get("canary_fraction"),
        "holdout_fraction": aggregate.get("holdout_fraction"),
        "sample_count": sum(_as_int(cohort.get("count")) for cohort in cohorts.values()),
        "cohort_counts": {key: value["count"] for key, value in cohorts.items()},
        "cohort_metrics": cohorts,
        "applied_vs_holdout_deltas": deltas,
        "observed_savings_usd": round(observed_total, 8),
        "projected_savings_usd": round(projected_total, 8),
        "cache_interaction_counts": _counter_rows(aggregate["cache_interactions"]),
        "crunch_interaction_counts": _counter_rows(aggregate["crunch_interactions"]),
        "status_buckets": _counter_rows(aggregate["status_buckets"]),
        "reason_buckets": _counter_rows(aggregate["reason_buckets"]),
        "skipped_reason_buckets": _counter_rows(aggregate["skipped_reason_buckets"]),
        "unknown_reason_buckets": _counter_rows(aggregate["unknown_reason_buckets"]),
        "skipped_unknown_classification": routing_verdict["skipped_unknown_classification"],
        "category_buckets": _counter_rows(aggregate["category_buckets"]),
        "endpoint_buckets": _counter_rows(aggregate["endpoint_buckets"]),
        "oldest_observed_at": aggregate.get("oldest_observed_at"),
        "latest_observed_at": aggregate.get("latest_observed_at"),
        "stale_evidence": stale,
        "thresholds": thresholds,
        "provider_adoption_gate": provider_adoption_gate,
        "verdict": decision["verdict"],
        "reason_codes": decision["reason_codes"],
        "warning_codes": decision["warning_codes"],
        "routing_promotion_verdict": routing_verdict,
        "next_action": {
            "widen": "widen_local_openai_canary",
            "hold": "keep_current_openai_canary_fraction",
            "rollback": "rollback_or_disable_openai_canary",
            "needs_eval": "collect_openai_canary_holdout_evidence_or_run_eval",
        }[decision["verdict"]],
        "privacy": _privacy_summary(),
    }


def build_openai_canary_impact_report(
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
    max_latency_regression_ms: int = 2000,
    rollback_error_rate: float = 0.20,
    min_projection_realization_ratio: float = 0.50,
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
        canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
        if not canary:
            continue
        candidate_id = _candidate_key(canary)
        aggregate = aggregates.setdefault(candidate_id, _new_aggregate(candidate_id, canary))
        _add_row(aggregate, row, canary, adoption_by_call.get(str(row.get("id") or ""), []))
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
            max_latency_regression_ms=max_latency_regression_ms,
            rollback_error_rate=rollback_error_rate,
            min_projection_realization_ratio=min_projection_realization_ratio,
        )
        for aggregate in aggregates.values()
    ]
    candidates.sort(key=lambda item: (str(item.get("verdict")), str(item.get("candidate_id"))))

    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in candidates)
    routing_verdict_counts = Counter(
        str((item.get("routing_promotion_verdict") or {}).get("verdict") or "unknown")
        for item in candidates
    )
    reason_counts: Counter[str] = Counter()
    routing_reason_counts: Counter[str] = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        for reason in (item.get("routing_promotion_verdict") or {}).get("reason_codes") or []:
            routing_reason_counts[str(reason)] += 1
    try:
        from tokenclaw.activation_lifecycle_feedback import activation_lifecycle_feedback_summary

        lifecycle_feedback = activation_lifecycle_feedback_summary(store_obj, limit=lookback_limit)
    except Exception:
        lifecycle_feedback = {
            "schema": "agentflow.activation_staged_lifecycle_feedback_summary.v1",
            "queue_rows": 0,
            "family_event_count": 0,
            "state_breakdown": [],
            "event_phase_breakdown": [],
            "cohort_breakdown": [],
            "family_state_breakdown": [],
            "candidate_id_breakdown": [],
            "payload_json_included": False,
        }

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
        "status": "matched" if observed_rows else "no-openai-canary-metadata",
        "summary": {
            "sampled_call_count": len(rows),
            "observed_openai_canary_metadata_row_count": observed_rows,
            "candidate_count": len(candidates),
            "canary_applied_count": sum(_as_int(item["cohort_counts"].get("canary_applied")) for item in candidates),
            "canary_holdout_count": sum(_as_int(item["cohort_counts"].get("canary_holdout")) for item in candidates),
            "safety_stopped_count": sum(_as_int(item["cohort_counts"].get("safety_stopped")) for item in candidates),
            "observed_savings_usd": round(sum(_as_float(item.get("observed_savings_usd")) for item in candidates), 8),
            "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in candidates), 8),
            "verdict_counts": _counter_rows(verdict_counts),
            "reason_code_counts": _counter_rows(reason_counts),
            "routing_promotion_verdict_counts": _counter_rows(routing_verdict_counts),
            "routing_promotion_reason_code_counts": _counter_rows(routing_reason_counts),
        },
        "candidates": candidates,
        "activation_lifecycle_feedback": lifecycle_feedback,
        "privacy": _privacy_summary(),
    }
