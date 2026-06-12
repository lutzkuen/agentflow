from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.limiter import model_tier
from agentflow_proxy.optimization.openai_features import openai_model_family, openai_source_surface
from agentflow_proxy.pricing import estimate_blended_input_savings, estimate_cost
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.repeated_scaffold_impact.v1"
VERDICT_SCHEMA = "agentflow.repeated_scaffold_promotion_verdict.v1"
TOKEN_CHARS = 4

_REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_RAW_ID_HINT_RE = re.compile(
    r"[/\\]|\s|raw|secret|token|cache[-_]?key|request[-_]?id|session[-_]?id|thread[-_]?id|sha256:[0-9a-f]{24,}",
    re.IGNORECASE,
)


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


def _reason_code(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text if _REASON_CODE_RE.match(text) else fallback


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _public_id(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if text and _PUBLIC_ID_RE.match(text) and not _RAW_ID_HINT_RE.search(text):
        return text
    digest = hashlib.sha256((text or "unknown").encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


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
        "local_only": True,
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


def _source_surface(row: dict[str, Any], routing: dict[str, Any]) -> str:
    value = row.get("source_surface") or routing.get("source_surface")
    if value:
        return str(value)
    provider = str(row.get("provider") or "anthropic").lower()
    if provider == "openai":
        return openai_source_surface(str(row.get("path") or ""))
    if provider == "anthropic":
        return "anthropic_messages"
    return "unknown"


def _workflow_phase(row: dict[str, Any], routing: dict[str, Any]) -> str:
    for key in ("workflow_phase", "phase", "session_phase"):
        value = routing.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        unit = routing.get(key)
        if isinstance(unit, dict):
            input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
            value = unit.get("workflow_phase") or input_features.get("workflow_phase") or input_features.get("phase")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(row.get("category") or routing.get("category") or "unknown")


def _model_tier(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "anthropic").lower()
    stored = row.get("routed_model_family") or row.get("requested_model_family")
    if stored:
        return str(stored)
    model = str(row.get("routed_model") or row.get("requested_model") or "")
    if provider == "openai":
        return openai_model_family(model) or "unknown"
    tier = model_tier(model)
    return tier if tier != "other" else "unknown"


def _cohort(rule: dict[str, Any], provider_meta: dict[str, Any]) -> str:
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
    status = str(provider_meta.get("status") or rule.get("status") or "").strip().lower()
    reason = str(provider_meta.get("reason") or rule.get("reason") or "").strip().lower()
    canary_status = str(canary.get("status") or "").strip().lower()
    canary_cohort = str(canary.get("cohort") or "").strip().lower()
    skip_reasons = [
        str(item.get("reason") or "").strip().lower()
        for item in (rule.get("skip_reasons") if isinstance(rule.get("skip_reasons"), list) else [])
        if isinstance(item, dict)
    ]
    if safety.get("tripped") or "safety-stop" in reason or any("safety-stop" in item for item in skip_reasons):
        return "safety_stop"
    if canary_status == "holdout" or canary_cohort == "canary_holdout" or _as_int(rule.get("holdout_count")) > 0:
        return "holdout"
    if canary_status == "applied" or canary_cohort == "canary_applied" or _as_int(rule.get("applied_count")) > 0 or status == "applied":
        return "applied"
    if status in {"disabled", "skipped"} or skip_reasons:
        return "skipped"
    return "unknown"


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_rows": 0,
        "retry_attempts": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "estimated_saved_chars": 0,
        "estimated_saved_tokens": 0,
        "estimated_savings_usd": 0.0,
        "non_positive_savings_count": 0,
        "negative_savings_count": 0,
        "safety_stop_count": 0,
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    non_positive = _as_int(raw.get("non_positive_savings_count"))
    negative = _as_int(raw.get("negative_savings_count"))
    return {
        "count": count,
        "error_count": _as_int(raw.get("error_count")),
        "retry_rows": _as_int(raw.get("retry_rows")),
        "retry_attempts": _as_int(raw.get("retry_attempts")),
        "safety_stop_count": _as_int(raw.get("safety_stop_count")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_rows")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "estimated_saved_chars": _as_int(raw.get("estimated_saved_chars")),
        "estimated_saved_tokens": _as_int(raw.get("estimated_saved_tokens")),
        "estimated_savings_usd": round(_as_float(raw.get("estimated_savings_usd")), 8),
        "non_positive_savings_count": non_positive,
        "negative_savings_count": negative,
        "non_positive_savings_rate": round(non_positive / count, 6) if count else 0.0,
        "negative_savings_rate": round(negative / count, 6) if count else 0.0,
    }


def _estimated_savings_usd(row: dict[str, Any], provider_meta: dict[str, Any]) -> float:
    for key in ("estimated_savings_usd", "estimated_saved_cost_usd"):
        if provider_meta.get(key) is not None:
            return _as_float(provider_meta.get(key))
    tokens_saved = _as_int(provider_meta.get("tokens_saved_est"))
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    cache_read_tokens = _as_int(row.get("cache_read_input_tokens"))
    model = str(row.get("routed_model") or row.get("requested_model") or "")
    provider = str(row.get("provider") or "anthropic").lower()
    blended = estimate_blended_input_savings(
        model,
        tokens_saved=tokens_saved,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        provider=provider,
    )
    if blended is not None:
        return blended
    return estimate_cost(model, tokens_saved, 0, provider=provider) or 0.0


def _group_key_parts(row: dict[str, Any], routing: dict[str, Any], provider_meta: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    provider = str(row.get("provider") or "anthropic").lower()
    category = str(row.get("category") or provider_meta.get("category") or routing.get("category") or "unknown")
    return {
        "provider": provider,
        "source_surface": _source_surface(row, routing),
        "endpoint": str(row.get("endpoint") or routing.get("endpoint") or "unknown"),
        "category": category,
        "workflow_phase": _workflow_phase(row, routing),
        "model_tier": _model_tier(row),
        "rule_id": _public_id(rule.get("rule_id") or rule.get("id") or "unknown-repeated-scaffold-rule", "rule-id"),
        "candidate_id": _public_id(rule.get("candidate_id") or rule.get("rule_id") or "unknown-repeated-scaffold-candidate", "candidate-id"),
        "policy_source": str(rule.get("policy_source") or provider_meta.get("policy_source") or "unknown"),
    }


def _new_aggregate(parts: dict[str, Any]) -> dict[str, Any]:
    group_key = "|".join(_reason_code(parts.get(key), "unknown") for key in (
        "provider",
        "source_surface",
        "category",
        "workflow_phase",
        "model_tier",
        "rule_id",
        "candidate_id",
    ))
    return {
        "group_key": group_key,
        **parts,
        "cohorts": {
            "applied": _empty_cohort(),
            "holdout": _empty_cohort(),
            "skipped": _empty_cohort(),
            "safety_stop": _empty_cohort(),
            "unknown": _empty_cohort(),
        },
        "status_buckets": Counter(),
        "reason_buckets": Counter(),
        "category_buckets": Counter(),
        "workflow_phase_buckets": Counter(),
        "model_tier_buckets": Counter(),
        "safety_stop_reason_counts": Counter(),
        "skip_reason_counts": Counter(),
        "latest_observed_at": None,
        "oldest_observed_at": None,
        "canary": {},
    }


def _add_row(aggregate: dict[str, Any], row: dict[str, Any], provider_meta: dict[str, Any], rule: dict[str, Any], cohort: str) -> None:
    bucket = aggregate["cohorts"].get(cohort, aggregate["cohorts"]["unknown"])
    bucket["count"] += 1
    status_code = _as_int(row.get("status_code"), -1)
    if status_code >= 400:
        bucket["error_count"] += 1
    retry_count = _as_int(row.get("retry_count"))
    if retry_count > 0:
        bucket["retry_rows"] += 1
        bucket["retry_attempts"] += retry_count
    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        bucket["latency_ms_total"] += latency_ms
        bucket["latency_sample_count"] += 1

    tokens_saved = _as_int(provider_meta.get("tokens_saved_est")) if cohort == "applied" else 0
    chars_saved = _as_int(provider_meta.get("saved_chars")) if cohort == "applied" else 0
    savings = _estimated_savings_usd(row, provider_meta) if cohort == "applied" else 0.0
    bucket["estimated_saved_chars"] += chars_saved
    bucket["estimated_saved_tokens"] += tokens_saved
    bucket["estimated_savings_usd"] += savings
    if cohort == "applied" and savings <= 0:
        bucket["non_positive_savings_count"] += 1
    if cohort == "applied" and savings < 0:
        bucket["negative_savings_count"] += 1
    if cohort == "safety_stop":
        bucket["safety_stop_count"] += 1

    aggregate["status_buckets"][_status_bucket(row.get("status_code"))] += 1
    aggregate["reason_buckets"][_reason_code(provider_meta.get("reason"))] += 1
    aggregate["category_buckets"][_reason_code(aggregate.get("category"))] += 1
    aggregate["workflow_phase_buckets"][_reason_code(aggregate.get("workflow_phase"))] += 1
    aggregate["model_tier_buckets"][_reason_code(aggregate.get("model_tier"))] += 1

    for item in rule.get("skip_reasons") or []:
        if isinstance(item, dict):
            reason = _reason_code(item.get("reason"))
            aggregate["skip_reason_counts"][reason] += _as_int(item.get("count"), 1) or 1
            if "safety-stop" in reason:
                aggregate["safety_stop_reason_counts"][reason] += _as_int(item.get("count"), 1) or 1
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
    if safety.get("tripped"):
        for reason in safety.get("reason_codes") or ["safety-stop-observed"]:
            aggregate["safety_stop_reason_counts"][_reason_code(reason)] += 1

    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    if canary and not aggregate["canary"]:
        aggregate["canary"] = {
            key: canary.get(key)
            for key in ("enabled", "selected", "cohort", "fraction", "threshold", "unit", "reason", "status")
            if canary.get(key) is not None
        } | {"pattern_hashes_included": False, "request_fingerprints_included": False}

    created_at = row.get("created_at")
    if created_at:
        if aggregate["latest_observed_at"] is None or str(created_at) > str(aggregate["latest_observed_at"]):
            aggregate["latest_observed_at"] = str(created_at)
        if aggregate["oldest_observed_at"] is None or str(created_at) < str(aggregate["oldest_observed_at"]):
            aggregate["oldest_observed_at"] = str(created_at)


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
) -> dict[str, Any]:
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    safety = cohorts["safety_stop"]
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if _as_int(applied.get("count")) < _as_int(thresholds["min_applied_samples"]):
        reason_codes.append("insufficient-applied-samples")
    if _as_int(holdout.get("count")) < _as_int(thresholds["min_holdout_samples"]):
        reason_codes.append("insufficient-holdout-samples")
    if reason_codes:
        return {"verdict": "need-more-samples", "reason_codes": reason_codes, "warning_codes": warning_codes}

    rollback_reasons: list[str] = []
    if _as_int(safety.get("count")):
        rollback_reasons.append("safety-stop-observed")
    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        rollback_reasons.append("rollback-error-rate")
    if _as_float(deltas.get("applied_minus_holdout_error_rate")) > _as_float(thresholds["max_error_rate_delta"]):
        rollback_reasons.append("error-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_retry_rate")) > _as_float(thresholds["max_retry_rate_delta"]):
        rollback_reasons.append("retry-rate-regression")
    if rollback_reasons:
        return {"verdict": "rollback", "reason_codes": rollback_reasons, "warning_codes": warning_codes}

    if stale.get("stale"):
        reason_codes.append("stale-evidence")
    if _as_float(applied.get("error_rate")) > _as_float(thresholds["max_error_rate"]):
        reason_codes.append("applied-error-rate-above-threshold")
    latency_delta = deltas.get("applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and _as_float(latency_delta) > _as_float(thresholds["max_latency_regression_ms"]):
        reason_codes.append("latency-regression")
    if _as_float(applied.get("estimated_savings_usd")) <= 0:
        reason_codes.append("non-positive-estimated-savings")
    if _as_float(applied.get("non_positive_savings_rate")) > _as_float(thresholds["max_non_positive_savings_rate"]):
        reason_codes.append("non-positive-savings-rate-above-threshold")
    if reason_codes:
        return {"verdict": "hold", "reason_codes": reason_codes, "warning_codes": warning_codes}
    return {"verdict": "promote", "reason_codes": ["target-savings-met"], "warning_codes": warning_codes}


def _finalize_candidate(aggregate: dict[str, Any], *, now: datetime, thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {key: _finalize_cohort(value) for key, value in aggregate["cohorts"].items()}
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    deltas = {
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
        "applied_minus_holdout_estimated_savings_usd": round(
            _as_float(applied["estimated_savings_usd"]) - _as_float(holdout["estimated_savings_usd"]), 8
        ),
    }
    stale = _stale_evidence(
        aggregate.get("latest_observed_at"),
        now=now,
        max_age_hours=_as_float(thresholds["max_evidence_age_hours"]),
    )
    decision = _decide_verdict(cohorts=cohorts, deltas=deltas, stale=stale, thresholds=thresholds)
    rollout_verdict = {
        "promote": "widen",
        "hold": "hold",
        "rollback": "rollback",
        "need-more-samples": "need-more-samples",
    }[decision["verdict"]]
    return {
        "schema": VERDICT_SCHEMA,
        "group_key": aggregate["group_key"],
        "candidate_id": aggregate["candidate_id"],
        "rule_id": aggregate.get("rule_id"),
        "policy_source": aggregate.get("policy_source"),
        "optimization_family": "repeated_provider_scaffolding",
        "action_family": "crunch",
        "provider": aggregate.get("provider"),
        "source_surface": aggregate.get("source_surface"),
        "endpoint": aggregate.get("endpoint"),
        "category": aggregate.get("category"),
        "workflow_phase": aggregate.get("workflow_phase"),
        "model_tier": aggregate.get("model_tier"),
        "sample_count": sum(_as_int(cohort.get("count")) for cohort in cohorts.values()),
        "cohort_counts": {key: value["count"] for key, value in cohorts.items()},
        "cohort_metrics": cohorts,
        "applied_vs_holdout_deltas": deltas,
        "estimated_saved_chars": _as_int(applied.get("estimated_saved_chars")),
        "estimated_saved_tokens": _as_int(applied.get("estimated_saved_tokens")),
        "estimated_savings_usd": _as_float(applied.get("estimated_savings_usd")),
        "negative_savings_rate": _as_float(applied.get("negative_savings_rate")),
        "non_positive_savings_rate": _as_float(applied.get("non_positive_savings_rate")),
        "safety_stop_count": _as_int(cohorts["safety_stop"].get("count")),
        "status_buckets": _counter_rows(aggregate["status_buckets"]),
        "reason_buckets": _counter_rows(aggregate["reason_buckets"]),
        "category_buckets": _counter_rows(aggregate["category_buckets"]),
        "workflow_phase_buckets": _counter_rows(aggregate["workflow_phase_buckets"]),
        "model_tier_buckets": _counter_rows(aggregate["model_tier_buckets"]),
        "skip_reason_counts": _counter_rows(aggregate["skip_reason_counts"]),
        "safety_stop_reason_counts": _counter_rows(aggregate["safety_stop_reason_counts"]),
        "canary": aggregate.get("canary") or {"pattern_hashes_included": False, "request_fingerprints_included": False},
        "oldest_observed_at": aggregate.get("oldest_observed_at"),
        "latest_observed_at": aggregate.get("latest_observed_at"),
        "stale_evidence": stale,
        "thresholds": thresholds,
        "verdict": decision["verdict"],
        "rollout_verdict": rollout_verdict,
        "reason_codes": decision["reason_codes"],
        "warning_codes": decision["warning_codes"],
        "next_action": {
            "promote": "widen_repeated_scaffold_crunch_canary",
            "hold": "keep_current_repeated_scaffold_crunch_fraction",
            "rollback": "rollback_or_disable_repeated_scaffold_crunch_rule",
            "need-more-samples": "collect_repeated_scaffold_applied_and_holdout_evidence",
        }[decision["verdict"]],
        "privacy": _privacy_summary(),
    }


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where_since = "and created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    rows = store_obj.conn.execute(
        f"""
        select created_at, path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint, requested_model, routed_model,
               requested_model_family, routed_model_family, stream, status_code,
               latency_ms, input_tokens_est, actual_input_tokens,
               cache_read_input_tokens, retry_count, category, routing_json, crunch_json
        from calls
        where crunch_json is not null
          {where_since}
        order by created_at desc
        limit ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def build_repeated_scaffold_impact_report(
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
    max_latency_regression_ms: int = 2000,
    max_non_positive_savings_rate: float = 0.0,
    rollback_error_rate: float = 0.20,
    now: datetime | None = None,
) -> dict[str, Any]:
    from agentflow_proxy.repeated_scaffold_feedback import build_repeated_scaffold_lifecycle_feedback_status

    lookback_limit = max(1, min(int(limit or 500), 10_000))
    thresholds = {
        "min_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "max_non_positive_savings_rate": round(float(max_non_positive_savings_rate), 6),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
    }
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    aggregates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    cohort_counts: Counter[str] = Counter()

    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        provider_meta = crunch.get("repeated_provider_scaffolding") if isinstance(crunch.get("repeated_provider_scaffolding"), dict) else {}
        if not provider_meta:
            continue
        rules = provider_meta.get("rules") if isinstance(provider_meta.get("rules"), list) else []
        if not rules:
            continue
        routing = _json_obj(row.get("routing_json"))
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            parts = _group_key_parts(row, routing, provider_meta, rule)
            group_key = "|".join(_reason_code(parts.get(key), "unknown") for key in (
                "provider",
                "source_surface",
                "category",
                "workflow_phase",
                "model_tier",
                "rule_id",
                "candidate_id",
            ))
            aggregate = aggregates.setdefault(group_key, _new_aggregate(parts))
            cohort = _cohort(rule, provider_meta)
            _add_row(aggregate, row, provider_meta, rule, cohort)
            observed_rows += 1
            cohort_counts[cohort] += 1

    now_dt = now or datetime.now(timezone.utc)
    candidates = [
        _finalize_candidate(aggregate, now=now_dt, thresholds=thresholds)
        for aggregate in aggregates.values()
    ]
    candidates.sort(key=lambda item: (str(item.get("verdict")), str(item.get("group_key"))))
    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in candidates)
    reason_counts: Counter[str] = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1

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
        "status": "matched" if observed_rows else "no-repeated-scaffold-canary-metadata",
        "summary": {
            "sampled_call_count": len(rows),
            "observed_repeated_scaffold_metadata_row_count": observed_rows,
            "candidate_group_count": len(candidates),
            "applied_count": cohort_counts.get("applied", 0),
            "holdout_count": cohort_counts.get("holdout", 0),
            "safety_stop_count": cohort_counts.get("safety_stop", 0),
            "skipped_count": cohort_counts.get("skipped", 0),
            "estimated_saved_chars": sum(_as_int(item.get("estimated_saved_chars")) for item in candidates),
            "estimated_saved_tokens": sum(_as_int(item.get("estimated_saved_tokens")) for item in candidates),
            "estimated_savings_usd": round(sum(_as_float(item.get("estimated_savings_usd")) for item in candidates), 8),
            "verdict_counts": _counter_rows(verdict_counts),
            "reason_code_counts": _counter_rows(reason_counts),
        },
        "managed_lifecycle_feedback_queue": build_repeated_scaffold_lifecycle_feedback_status(
            store_obj,
            sample_limit=20,
        ),
        "candidates": candidates,
        "privacy": _privacy_summary(),
    }
