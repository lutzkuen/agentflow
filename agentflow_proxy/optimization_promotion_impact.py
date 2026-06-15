from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization_promotion_actions import SCHEMA as PROMOTION_ACTIONS_SCHEMA
from agentflow_proxy.optimization_promotion_canary import normalize_promotion_canary_bundle
from agentflow_proxy.provider_adoption_gate import (
    build_provider_adoption_gate,
    provider_adoption_thresholds,
    provider_adoption_windows_by_call,
)
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.optimization_promotion_impact.v1"

_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cache_key",
    "command",
    "content",
    "credential",
    "file_path",
    "generated_summary",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_payload",
    "raw_request",
    "raw_response",
    "raw_context",
    "request_id",
    "secret",
    "session_id",
    "summary_text",
    "tool_payload",
    "transcript",
)
_ALLOWED_RAW_LIKE_KEYS = {
    "apply_preview_command",
    "cache_keys_included",
    "content_free",
    "filesystem_paths_included",
    "raw_content_included",
    "raw_provider_bodies_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_session_ids_included",
    "request_ids_included",
    "review_command",
    "session_id_hash",
    "tool_payloads_included",
}


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


def _counter_rows(values: list[str]) -> list[dict[str, Any]]:
    counts = Counter(values)
    return [{"value": value, "count": count} for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _increment(counter: dict[str, int], value: Any) -> None:
    key = str(value or "unknown")
    counter[key] = counter.get(key, 0) + 1


def _public_recommendation(verdict: Any) -> str:
    text = str(verdict or "").strip()
    if text == "widen":
        return "promote"
    if text == "hold":
        return "keep-canary"
    if text == "rollback":
        return "rollback"
    return "needs-more-evidence"


def _action_family(action: dict[str, Any]) -> str:
    text = str(action.get("action_family") or action.get("optimization_family") or action.get("policy_section") or "").strip().lower()
    text = text.replace("_", "-")
    if text in {"route", "routing", "phase-routing", "provider-routing"}:
        return "routing"
    if text in {"cache", "cache-replay", "exact-cache", "semantic-cache"}:
        return "cache"
    if text in {
        "crunch",
        "old-context-summary",
        "old-context-summarization",
        "repeated-scaffold-crunch",
        "terminal-output-compaction",
        "anthropic-thinking-history-compaction",
    }:
        return "crunch"
    if text in {"eval", "evaluation", "evaluation-only", "shadow-eval", "shadow-evaluation", "optimization-eval"}:
        return "evaluation-only"
    return text or "unknown"


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "local_only": True,
    }


def _scan_raw_like(value: Any, path: str, violations: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_LIKE_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if item not in (None, False, 0, "", [], {}):
                    violations.append({"path": child, "message": "raw or local-identifier promotion impact input is not accepted"})
                    continue
            _scan_raw_like(item, child, violations)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _scan_raw_like(item, f"{path}[{index}]", violations)


def _source_surface(provider: Any, path: Any) -> str:
    provider_text = str(provider or "anthropic")
    path_text = str(path or "")
    if provider_text == "anthropic" and path_text.endswith("/v1/messages"):
        return "anthropic_messages"
    if provider_text == "openai":
        return "openai"
    return provider_text


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


def _error_bucket(error: Any, status_code: Any) -> str:
    code = _as_int(status_code, -1)
    if 0 <= code < 400:
        return "none"
    text = str(error or "").lower()
    if "unsupported" in text and "model" in text:
        return "unsupported_model"
    if "rate" in text or code == 429:
        return "rate_limited"
    if code >= 500:
        return "provider_5xx"
    if code >= 400:
        return "provider_4xx"
    return "unknown"


def _row_savings(row: dict[str, Any], meta: dict[str, Any]) -> float:
    if row.get("cost_baseline_usd") is not None and row.get("cost_est_usd") is not None:
        return _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd"))
    for key in ("estimated_cost_savings_usd", "estimated_net_savings_usd", "actual_net_savings_usd"):
        if meta.get(key) is not None:
            return _as_float(meta.get(key))
    return 0.0


def _cohort_from_meta(meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    status = str(meta.get("status") or canary.get("status") or "").strip()
    cohort = str(meta.get("cohort") or canary.get("cohort") or "").strip()
    reason = str(meta.get("reason") or canary.get("reason") or "").strip()
    safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
    if status == "applied" or cohort in {"applied", "canary_applied"}:
        return "canary_applied"
    if status == "holdout" or cohort in {"holdout", "canary_holdout"} or reason == "canary_holdout":
        return "canary_holdout"
    if status == "safety_stopped" or cohort == "bypassed_or_disabled" or "safety-stop" in reason or safety.get("tripped"):
        return "safety_stopped"
    if status in {"skipped", "not_selected", "ineligible"} or cohort == "skipped":
        return "skipped"
    if status in {"disabled", "bypass", "bypassed"} or "disabled" in reason:
        return "bypassed_or_disabled"
    return "unknown"


def _promotion_meta_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("phase_canary", "openai_canary", "promotion_canary", "optimization_promotion_canary"):
        value = decision.get(key)
        if isinstance(value, dict) and (value.get("target_candidate_id") or value.get("promotion_action_id") or value.get("action_id")):
            return value
    value = decision.get("canary")
    if isinstance(value, dict) and (value.get("target_candidate_id") or value.get("promotion_action_id") or value.get("action_id")):
        return value
    return None


def _pattern_rule_candidates(meta: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if isinstance(meta.get("pattern_rule"), dict):
        rules.append(meta["pattern_rule"])
    pattern_rules = meta.get("pattern_rules") if isinstance(meta.get("pattern_rules"), dict) else {}
    for key in ("rules", "skip_reasons"):
        for item in pattern_rules.get(key) or []:
            if isinstance(item, dict) and (item.get("candidate_id") or item.get("rule_id") or item.get("promotion_action_id") or item.get("canary")):
                rules.append(item)
    return rules


def _observation_from_meta(row: dict[str, Any], meta: dict[str, Any], *, policy_section: str) -> dict[str, Any] | None:
    candidate_id = str(meta.get("target_candidate_id") or meta.get("candidate_id") or "").strip()
    action_id = str(meta.get("promotion_action_id") or meta.get("action_id") or "").strip()
    rule_id = str(meta.get("target_rule_id") or meta.get("rule_id") or meta.get("policy_id") or "").strip()
    if not candidate_id and not action_id and not rule_id:
        return None
    cohort = _cohort_from_meta(meta)
    reason = str(meta.get("reason") or (meta.get("canary") if isinstance(meta.get("canary"), dict) else {}).get("reason") or "unknown")
    return {
        "created_at": row.get("created_at"),
        "call_id": row.get("id"),
        "policy_section": policy_section,
        "action_id": action_id or None,
        "target_candidate_id": candidate_id or None,
        "target_rule_id": rule_id or None,
        "policy_source": meta.get("policy_source"),
        "status": str(meta.get("status") or (meta.get("canary") if isinstance(meta.get("canary"), dict) else {}).get("status") or "unknown"),
        "cohort": cohort,
        "reason": reason,
        "status_code": row.get("status_code"),
        "retry_count": _as_int(row.get("retry_count")),
        "fallback_count": _as_int(meta.get("fallback_count")),
        "latency_ms": row.get("latency_ms"),
        "source_surface": row.get("source_surface"),
        "savings_usd": _row_savings(row, meta) if cohort == "canary_applied" else 0.0,
        "error_bucket": _error_bucket(row.get("error"), row.get("status_code")),
        "safety_stop": meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {},
        "provider_adoption_windows": row.get("provider_adoption_windows") if isinstance(row.get("provider_adoption_windows"), list) else [],
    }


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    if since:
        rows = store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, requested_model, routed_model, stream, status_code,
                   latency_ms, retry_count, cost_est_usd, cost_baseline_usd,
                   error, routing_json, crunch_json, cache_json, category
            from calls
            where created_at >= ?
            order by created_at desc
            limit ?
            """,
            (since, capped),
        ).fetchall()
    else:
        rows = store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, requested_model, routed_model, stream, status_code,
                   latency_ms, retry_count, cost_est_usd, cost_baseline_usd,
                   error, routing_json, crunch_json, cache_json, category
            from calls
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        row["source_surface"] = row.get("source_surface") or _source_surface(row.get("provider"), row.get("path"))
    return result


def _observations(rows: list[dict[str, Any]], adoption_by_call: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        call_id = str(row.get("id") or "")
        if adoption_by_call and call_id:
            row = {**row, "provider_adoption_windows": adoption_by_call.get(call_id, [])}
        routing = _json_obj(row.get("routing_json"))
        routing_meta = _promotion_meta_from_decision(routing)
        if routing_meta:
            item = _observation_from_meta(row, routing_meta, policy_section="routing")
            if item:
                observations.append(item)
        for policy_section, decision in (("crunch", _json_obj(row.get("crunch_json"))), ("cache", _json_obj(row.get("cache_json")))):
            for rule in _pattern_rule_candidates(decision):
                item = _observation_from_meta(row, rule, policy_section=policy_section)
                if item:
                    observations.append(item)
    return observations


def _matches_action(observation: dict[str, Any], action: dict[str, Any]) -> bool:
    if str(observation.get("policy_section") or "") != str(action.get("policy_section") or ""):
        return False
    action_id = str(action.get("action_id") or "").strip()
    if action_id and observation.get("action_id") and str(observation["action_id"]) != action_id:
        return False
    candidate_id = str(action.get("target_candidate_id") or "").strip()
    if candidate_id and observation.get("target_candidate_id") and str(observation["target_candidate_id"]) != candidate_id:
        return False
    rule_id = str(action.get("target_rule_id") or "").strip()
    if rule_id and observation.get("target_rule_id") and str(observation["target_rule_id"]) != rule_id:
        return False
    return bool(action_id or candidate_id or rule_id)


def _empty_cohort() -> dict[str, Any]:
    return {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0, "savings_usd": 0.0}


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
        "observed_savings_usd": round(_as_float(raw.get("savings_usd")), 8),
    }


def _actual(matched: list[dict[str, Any]]) -> dict[str, Any]:
    cohorts = {
        "canary_applied": _empty_cohort(),
        "canary_holdout": _empty_cohort(),
        "skipped": _empty_cohort(),
        "bypassed_or_disabled": _empty_cohort(),
        "safety_stopped": _empty_cohort(),
        "unknown": _empty_cohort(),
    }
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    latency_counts: dict[str, int] = {}
    savings = 0.0
    latest_observed_at = None
    for item in matched:
        cohort = str(item.get("cohort") or "unknown")
        bucket = cohorts.get(cohort, cohorts["unknown"])
        bucket["count"] += 1
        errored = _as_int(item.get("status_code")) >= 400
        retried = _as_int(item.get("retry_count")) > 0
        bucket["error_count"] += int(errored)
        bucket["retry_count"] += int(retried)
        bucket["fallback_count"] += _as_int(item.get("fallback_count"))
        latency = _as_int(item.get("latency_ms"), -1)
        if latency >= 0:
            bucket["latency_ms_total"] += latency
            bucket["latency_sample_count"] += 1
        if cohort == "canary_applied":
            value = _as_float(item.get("savings_usd"))
            bucket["savings_usd"] += value
            savings += value
        _increment(status_counts, _status_bucket(item.get("status_code")))
        _increment(reason_counts, item.get("reason"))
        _increment(error_counts, item.get("error_bucket"))
        _increment(latency_counts, _latency_bucket(item.get("latency_ms")))
        safety = item.get("safety_stop") if isinstance(item.get("safety_stop"), dict) else {}
        for code in safety.get("reason_codes") or []:
            _increment(reason_counts, code)
        created = item.get("created_at")
        if created and (not latest_observed_at or str(created) > str(latest_observed_at)):
            latest_observed_at = str(created)
    finalized = {key: _finalize_cohort(value) for key, value in cohorts.items()}
    applied = finalized["canary_applied"]
    holdout = finalized["canary_holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    return {
        "matched_metadata_row_count": len(matched),
        "actual_canary_applied_count": finalized["canary_applied"]["count"],
        "actual_canary_holdout_count": finalized["canary_holdout"]["count"],
        "actual_skipped_count": finalized["skipped"]["count"],
        "actual_bypassed_or_disabled_count": finalized["bypassed_or_disabled"]["count"],
        "actual_safety_stopped_count": finalized["safety_stopped"]["count"],
        "actual_unknown_count": finalized["unknown"]["count"],
        "observed_savings_usd": round(savings, 8),
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_fallback_rate": round(_as_float(applied["fallback_rate"]) - _as_float(holdout["fallback_rate"]), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
        "latest_observed_at": latest_observed_at,
        "cohorts": finalized,
        "status_buckets": _counter_rows([key for key, count in status_counts.items() for _ in range(count)]),
        "reason_buckets": _counter_rows([key for key, count in reason_counts.items() for _ in range(count)]),
        "error_buckets": _counter_rows([key for key, count in error_counts.items() for _ in range(count)]),
        "latency_buckets": _counter_rows([key for key, count in latency_counts.items() for _ in range(count)]),
    }


def _projected(action: dict[str, Any]) -> dict[str, Any]:
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    counts = evidence.get("cohort_counts") if isinstance(evidence.get("cohort_counts"), dict) else {}
    affected = _as_int(evidence.get("sample_count")) or sum(_as_int(counts.get(key)) for key in ("canary_applied", "canary_holdout", "bypassed_or_disabled"))
    return {
        "affected_metadata_row_count": affected,
        "projected_savings_usd": round(_as_float(evidence.get("projected_savings_usd")), 8),
        "current_canary_applied_count": _as_int(counts.get("canary_applied")),
        "current_canary_holdout_count": _as_int(counts.get("canary_holdout")),
        "current_bypassed_or_disabled_count": _as_int(counts.get("bypassed_or_disabled")),
        "target_canary_fraction": round(_as_float(action.get("canary_fraction")), 6),
        "target_holdout_fraction": round(_as_float(action.get("holdout_fraction")), 6),
    }


def _thresholds(action: dict[str, Any], *, min_applied_samples: int, min_holdout_samples: int, max_evidence_age_hours: float) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    safety = local_update.get("safety_stop") if isinstance(local_update.get("safety_stop"), dict) else {}
    thresholds = {
        "min_canary_applied_samples": max(0, _as_int(safety.get("min_applied_samples"), min_applied_samples)),
        "min_canary_holdout_samples": max(0, _as_int(safety.get("min_holdout_samples"), min_holdout_samples)),
        "max_error_rate": round(_as_float(safety.get("max_error_rate"), 0.05), 6),
        "rollback_error_rate": round(_as_float(safety.get("rollback_error_rate"), 0.20), 6),
        "max_error_rate_delta": round(_as_float(safety.get("max_error_rate_delta"), 0.05), 6),
        "max_retry_rate_delta": round(_as_float(safety.get("max_retry_rate_delta"), 0.10), 6),
        "max_latency_regression_ms": _as_int(safety.get("max_latency_regression_ms"), 2000),
        "min_projection_realization_ratio": round(_as_float(safety.get("min_projection_realization_ratio"), 0.50), 6),
        "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
    }
    thresholds.update(provider_adoption_thresholds(
        min_fulfilled_samples=_as_int(safety.get("min_provider_adoption_fulfilled_samples"), 1),
        max_applied_abandonment_rate=_as_float(safety.get("max_provider_adoption_abandonment_rate"), 0.02),
        max_applied_orphan_result_rate=_as_float(safety.get("max_provider_adoption_orphan_result_rate"), 0.02),
        max_applied_risk_rate=_as_float(safety.get("max_provider_adoption_risk_rate"), 0.05),
        max_applied_vs_holdout_risk_rate_delta=_as_float(safety.get("max_provider_adoption_risk_rate_delta"), 0.02),
    ))
    return thresholds


def _stale_evidence(actual: dict[str, Any], *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    latest = _parse_utc_datetime(actual.get("latest_observed_at"))
    if latest is None:
        return {"stale": False, "age_hours": None, "max_age_hours": round(float(max_age_hours), 3)}
    age_hours = (now.astimezone(timezone.utc) - latest).total_seconds() / 3600.0
    return {"stale": age_hours > max_age_hours, "age_hours": round(age_hours, 3), "max_age_hours": round(float(max_age_hours), 3)}


def _next_step(
    *,
    action: dict[str, Any],
    actual: dict[str, Any],
    projection: dict[str, Any],
    thresholds: dict[str, Any],
    stale: dict[str, Any],
    provider_adoption_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = actual["cohorts"]["canary_applied"]
    holdout = actual["cohorts"]["canary_holdout"]
    projected_savings = _as_float(projection.get("projected_savings_usd"))
    observed_savings = _as_float(actual.get("observed_savings_usd"))
    projection_ratio = round(observed_savings / projected_savings, 6) if projected_savings > 0 else None
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if _as_int(applied.get("count")) < _as_int(thresholds.get("min_canary_applied_samples")):
        reason_codes.append("insufficient-canary-applied-samples")
    if _as_float(action.get("holdout_fraction")) > 0 and _as_int(holdout.get("count")) < _as_int(thresholds.get("min_canary_holdout_samples")):
        reason_codes.append("insufficient-canary-holdout-samples")
    if reason_codes:
        verdict = "needs_more_samples"
    else:
        if _as_int(actual.get("actual_safety_stopped_count")) > 0:
            reason_codes.append("safety-stop-observed")
        if observed_savings < 0:
            reason_codes.append("negative-observed-savings")
        if _as_float(applied.get("error_rate")) >= _as_float(thresholds.get("rollback_error_rate")):
            reason_codes.append("rollback-error-rate")
        if reason_codes:
            verdict = "rollback"
        else:
            if stale.get("stale"):
                reason_codes.append("stale-evidence")
            if _as_float(applied.get("error_rate")) > _as_float(thresholds.get("max_error_rate")):
                reason_codes.append("applied-error-rate-above-threshold")
            if _as_float(actual.get("applied_minus_holdout_error_rate")) > _as_float(thresholds.get("max_error_rate_delta")):
                reason_codes.append("applied-error-rate-regression")
            if _as_float(actual.get("applied_minus_holdout_retry_rate")) > _as_float(thresholds.get("max_retry_rate_delta")):
                reason_codes.append("applied-retry-rate-regression")
            latency_delta = actual.get("applied_minus_holdout_latency_avg_ms")
            if latency_delta is not None and _as_float(latency_delta) > _as_int(thresholds.get("max_latency_regression_ms")):
                reason_codes.append("latency-regression")
            if projection_ratio is not None and projection_ratio < _as_float(thresholds.get("min_projection_realization_ratio")):
                warning_codes.append("observed-savings-below-projection")
                reason_codes.append("projection-realization-below-threshold")
            gate = provider_adoption_gate or {}
            if gate.get("blocking"):
                reason_codes.extend(str(code) for code in gate.get("reason_codes") or [])
            else:
                warning_codes.extend(str(code) for code in gate.get("warning_codes") or [])
            verdict = "hold" if reason_codes else "widen"
            if not reason_codes:
                reason_codes = ["promotion-impact-positive"]

    return {
        "verdict": verdict,
        "recommendation": _public_recommendation(verdict),
        "reason_codes": reason_codes,
        "warning_codes": warning_codes,
        "projected_vs_observed_savings_ratio": projection_ratio,
        "recommended_local_next_step": verdict,
    }


def _family_empty() -> dict[str, Any]:
    return {
        "action_count": 0,
        "matched_metadata_row_count": 0,
        "projected_savings_usd": 0.0,
        "observed_savings_usd": 0.0,
        "stale_evidence_action_count": 0,
        "cohorts": {
            "canary_applied": _empty_cohort(),
            "canary_holdout": _empty_cohort(),
            "skipped": _empty_cohort(),
            "bypassed_or_disabled": _empty_cohort(),
            "safety_stopped": _empty_cohort(),
            "unknown": _empty_cohort(),
        },
        "verdicts": [],
        "recommendations": [],
        "reason_counts": {},
        "warning_counts": {},
    }


def _merge_finalized_cohort(target: dict[str, Any], source: dict[str, Any]) -> None:
    count = _as_int(source.get("count"))
    target["count"] += count
    target["error_count"] += _as_int(source.get("error_count"))
    target["retry_count"] += _as_int(source.get("retry_count"))
    target["fallback_count"] += _as_int(source.get("fallback_count"))
    target["savings_usd"] += _as_float(source.get("observed_savings_usd"))
    latency = source.get("latency_avg_ms")
    if latency is not None and count:
        target["latency_ms_total"] += _as_float(latency) * count
        target["latency_sample_count"] += count


def _top_blocker(reason_counts: dict[str, int]) -> str | None:
    blockers = {
        key: count
        for key, count in reason_counts.items()
        if key not in {"promotion-impact-positive"}
    }
    if not blockers:
        return None
    return sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _family_recommendation(recommendations: list[str]) -> str:
    if not recommendations:
        return "needs-more-evidence"
    if "rollback" in recommendations:
        return "rollback"
    if "keep-canary" in recommendations:
        return "keep-canary"
    if "needs-more-evidence" in recommendations:
        return "needs-more-evidence"
    return "promote"


def _cohort_blocker_count(cohort: dict[str, Any]) -> int:
    return (
        _as_int(cohort.get("error_count"))
        + _as_int(cohort.get("retry_count"))
        + _as_int(cohort.get("fallback_count"))
    )


def _missing_family_measurements(
    *,
    applied_count: int,
    holdout_or_bypass_count: int,
    projected_savings: float,
    observed_savings: float,
) -> list[str]:
    missing: list[str] = []
    if applied_count <= 0:
        missing.append("missing-applied-promotion-metadata")
    if holdout_or_bypass_count <= 0:
        missing.append("missing-holdout-or-bypass-promotion-metadata")
    if applied_count <= 0 or (projected_savings > 0 and observed_savings == 0):
        missing.append("missing-observed-savings-delta")
    return missing


def _family_impacts(action_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for action in action_results:
        family = _action_family(action)
        row = families.setdefault(family, _family_empty())
        actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
        projection = action.get("projection") if isinstance(action.get("projection"), dict) else {}
        next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
        cohorts = actual.get("cohorts") if isinstance(actual.get("cohorts"), dict) else {}
        row["action_count"] += 1
        row["matched_metadata_row_count"] += _as_int(actual.get("matched_metadata_row_count"))
        row["projected_savings_usd"] += _as_float(projection.get("projected_savings_usd"))
        row["observed_savings_usd"] += _as_float(actual.get("observed_savings_usd"))
        if isinstance(action.get("stale_evidence"), dict) and action["stale_evidence"].get("stale"):
            row["stale_evidence_action_count"] += 1
        for cohort_name, cohort in cohorts.items():
            if isinstance(cohort, dict):
                bucket = row["cohorts"].setdefault(str(cohort_name), _empty_cohort())
                _merge_finalized_cohort(bucket, cohort)
        verdict = str(next_step.get("verdict") or "unknown")
        recommendation = str(next_step.get("recommendation") or _public_recommendation(verdict))
        row["verdicts"].append(verdict)
        row["recommendations"].append(recommendation)
        for reason in next_step.get("reason_codes") or []:
            _increment(row["reason_counts"], reason)
        for warning in next_step.get("warning_codes") or []:
            _increment(row["warning_counts"], warning)

    results: list[dict[str, Any]] = []
    for family, row in sorted(families.items()):
        finalized = {key: _finalize_cohort(value) for key, value in row["cohorts"].items()}
        applied = finalized.get("canary_applied", _finalize_cohort(_empty_cohort()))
        holdout = finalized.get("canary_holdout", _finalize_cohort(_empty_cohort()))
        bypass = finalized.get("bypassed_or_disabled", _finalize_cohort(_empty_cohort()))
        safety_stopped = finalized.get("safety_stopped", _finalize_cohort(_empty_cohort()))
        applied_count = _as_int(applied.get("count"))
        holdout_count = _as_int(holdout.get("count"))
        bypass_count = _as_int(bypass.get("count"))
        holdout_or_bypass_count = holdout_count + bypass_count
        applied_blocker_count = _cohort_blocker_count(applied)
        holdout_or_bypass_blocker_count = _cohort_blocker_count(holdout) + _cohort_blocker_count(bypass) + _as_int(safety_stopped.get("count"))
        blocker_delta = applied_blocker_count - holdout_or_bypass_blocker_count
        savings_delta = round(
            _as_float(applied.get("observed_savings_usd"))
            - _as_float(holdout.get("observed_savings_usd"))
            - _as_float(bypass.get("observed_savings_usd")),
            8,
        )
        latency_delta = None
        if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
            latency_delta = round(_as_float(applied.get("latency_avg_ms")) - _as_float(holdout.get("latency_avg_ms")), 2)
        recommendation = _family_recommendation(row["recommendations"])
        missing_measurements = _missing_family_measurements(
            applied_count=applied_count,
            holdout_or_bypass_count=holdout_or_bypass_count,
            projected_savings=_as_float(row["projected_savings_usd"]),
            observed_savings=_as_float(row["observed_savings_usd"]),
        )
        result = {
            "action_family": family,
            "action_count": row["action_count"],
            "matched_metadata_row_count": row["matched_metadata_row_count"],
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "bypass_count": bypass_count,
            "holdout_or_bypass_count": holdout_or_bypass_count,
            "applied_blocker_count": applied_blocker_count,
            "holdout_or_bypass_blocker_count": holdout_or_bypass_blocker_count,
            "blocker_delta": blocker_delta,
            "projected_savings_usd": round(_as_float(row["projected_savings_usd"]), 8),
            "observed_savings_usd": round(_as_float(row["observed_savings_usd"]), 8),
            "savings_delta_usd": savings_delta,
            "missing_measurements": missing_measurements,
            "cohort_metrics": finalized,
            "applied_vs_holdout_deltas": {
                "applied_minus_holdout_error_rate": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
                "applied_minus_holdout_retry_rate": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
                "applied_minus_holdout_fallback_rate": round(_as_float(applied.get("fallback_rate")) - _as_float(holdout.get("fallback_rate")), 6),
                "applied_minus_holdout_latency_avg_ms": latency_delta,
                "applied_minus_holdout_observed_savings_usd": round(
                    _as_float(applied.get("observed_savings_usd")) - _as_float(holdout.get("observed_savings_usd")),
                    8,
                ),
                "applied_minus_holdout_or_bypass_blocker_count": blocker_delta,
                "applied_minus_holdout_or_bypass_observed_savings_usd": savings_delta,
            },
            "recommendation": recommendation,
            "recommended_local_next_step": recommendation,
            "top_blocker": _top_blocker(row["reason_counts"]),
            "reason_counts": _counter_rows([key for key, count in row["reason_counts"].items() for _ in range(count)]),
            "warning_counts": _counter_rows([key for key, count in row["warning_counts"].items() for _ in range(count)]),
            "verdict_counts": _counter_rows(row["verdicts"]),
            "recommendation_counts": _counter_rows(row["recommendations"]),
            "stale_evidence_action_count": row["stale_evidence_action_count"],
            "privacy": _privacy_summary(),
        }
        results.append(result)
    return results


def measure_optimization_promotion_impact(
    promotion_actions: Any,
    *,
    store_obj: Any,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_evidence_age_hours: float = 72.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "status": "invalid",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_policy_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "post_apply_since": since,
        "actions": [],
        "summary": {},
        "privacy": _privacy_summary(),
        "warnings": [],
    }
    if not isinstance(promotion_actions, dict):
        result["error"] = {"type": "invalid_promotion_actions", "message": "promotion action bundle must be a JSON object"}
        return result
    normalized, validation = normalize_promotion_canary_bundle(promotion_actions)
    if validation is not None:
        result["source_optimization_rollout_validation"] = validation
        if normalized is None:
            result.update({
                "status": "invalid",
                "error": {"type": "invalid_optimization_rollout_actions", "message": "optimization rollout action bundle did not pass local validation"},
            })
            result["warnings"].extend(validation.get("errors", []))
            return result
        promotion_actions = normalized
    if promotion_actions.get("schema") != PROMOTION_ACTIONS_SCHEMA:
        result["warnings"].append({"code": "unexpected-action-bundle-schema", "schema": promotion_actions.get("schema")})
    violations: list[dict[str, str]] = []
    _scan_raw_like(promotion_actions, "$", violations)
    if violations:
        result.update({
            "status": "privacy-blocked",
            "error": {"type": "privacy_blocked", "message": "promotion action bundle contains raw-like fields"},
            "privacy": {**_privacy_summary(), "input_privacy_violations": violations[:20]},
        })
        return result
    actions = promotion_actions.get("actions")
    if not isinstance(actions, list):
        result["error"] = {"type": "invalid_promotion_actions", "message": "promotion action bundle actions must be a list"}
        return result
    since_value = since or promotion_actions.get("generated_at")
    if since_value and _parse_utc_datetime(since_value) is None:
        result["error"] = {"type": "invalid_since", "message": "post-apply since timestamp must be ISO-8601"}
        return result

    now_dt = now or datetime.now(timezone.utc)
    rows = _call_rows(store_obj, limit=lookback_limit, since=since_value)
    adoption_by_call = provider_adoption_windows_by_call(store_obj, [row.get("id") for row in rows])
    observed = _observations(rows, adoption_by_call)
    action_results: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
        evidence_summary = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
        matched = [item for item in observed if _matches_action(item, action)]
        actual = _actual(matched)
        projection = _projected(action)
        thresholds = _thresholds(
            action,
            min_applied_samples=min_applied_samples,
            min_holdout_samples=min_holdout_samples,
            max_evidence_age_hours=max_evidence_age_hours,
        )
        provider_adoption_gate = build_provider_adoption_gate(
            matched,
            thresholds={
                key: value
                for key, value in thresholds.items()
                if key.startswith("min_provider_adoption") or key.startswith("max_applied")
            },
        )
        stale = _stale_evidence(actual, now=now_dt, max_age_hours=max_evidence_age_hours)
        next_step = _next_step(
            action=action,
            actual=actual,
            projection=projection,
            thresholds=thresholds,
            stale=stale,
            provider_adoption_gate=provider_adoption_gate,
        )
        action_results.append({
            "path": f"$.actions[{index}]",
            "action_id": action.get("action_id"),
            "action_type": action.get("action_type"),
            "action_family": _action_family(action),
            "policy_section": action.get("policy_section"),
            "policy_id": action.get("policy_id") or action.get("target_rule_id") or action.get("target_candidate_id") or action.get("action_id"),
            "rule_source": local_update.get("policy_source") or action.get("policy_source"),
            "source_evidence_schema": action.get("source_evidence_schema") or evidence_summary.get("schema") or promotion_actions.get("schema"),
            "target_candidate_id": action.get("target_candidate_id"),
            "target_rule_id": action.get("target_rule_id"),
            "projection": projection,
            "actual": actual,
            "stale_evidence": stale,
            "thresholds": thresholds,
            "provider_adoption_gate": provider_adoption_gate,
            "next_step": next_step,
            "status": "matched" if matched else "no-post-apply-matches",
            "read_only": True,
            "privacy": _privacy_summary(),
        })

    matched_count = sum(_as_int(action["actual"].get("matched_metadata_row_count")) for action in action_results)
    verdicts = [str(action["next_step"]["verdict"]) for action in action_results]
    family_impacts = _family_impacts(action_results)
    recommendations = [str(row.get("recommendation") or "needs-more-evidence") for row in family_impacts]
    result.update({
        "ok": True,
        "status": "matched" if matched_count else "no-post-apply-matches",
        "post_apply_since": since_value,
        "source_action_bundle": {
            "schema": promotion_actions.get("schema"),
            "generated_at": promotion_actions.get("generated_at"),
            "action_count": len(actions),
        },
        "actions": action_results,
        "family_impacts": family_impacts,
        "summary": {
            "sampled_call_count": len(rows),
            "observed_promotion_metadata_row_count": len(observed),
            "actual_matched_metadata_row_count": matched_count,
            "action_count": len(actions),
            "action_family_count": len(family_impacts),
            "actions_without_post_apply_matches": sum(1 for action in action_results if not action["actual"]["matched_metadata_row_count"]),
            "actual_canary_applied_count": sum(_as_int(action["actual"].get("actual_canary_applied_count")) for action in action_results),
            "actual_canary_holdout_count": sum(_as_int(action["actual"].get("actual_canary_holdout_count")) for action in action_results),
            "actual_skipped_count": sum(_as_int(action["actual"].get("actual_skipped_count")) for action in action_results),
            "actual_bypassed_or_disabled_count": sum(_as_int(action["actual"].get("actual_bypassed_or_disabled_count")) for action in action_results),
            "actual_safety_stopped_count": sum(_as_int(action["actual"].get("actual_safety_stopped_count")) for action in action_results),
            "observed_savings_usd": round(sum(_as_float(action["actual"].get("observed_savings_usd")) for action in action_results), 8),
            "next_step_counts": _counter_rows(verdicts),
            "recommendation_counts": _counter_rows(recommendations),
            "stale_evidence_action_count": sum(1 for action in action_results if action["stale_evidence"]["stale"]),
        },
    })
    return result
