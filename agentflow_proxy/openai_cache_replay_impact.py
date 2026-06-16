from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.openai_cache_replay_report import _as_float, _as_int, _json_obj
from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_source_surface
from agentflow_proxy.provider_adoption_gate import (
    build_provider_adoption_gate,
    provider_adoption_thresholds,
    provider_adoption_windows_by_call,
)
from agentflow_proxy.public_metadata import public_label
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_impact.v1"
QUALITY_GATE_SCHEMA = "agentflow.openai_cache_replay_quality_gate.v1"
LIFECYCLE_SCHEMA = "agentflow.openai_cache_replay_lifecycle_feedback.v1"
LOCAL_PROMOTION_EVIDENCE_SCHEMA = "agentflow.openai_cache_replay_local_promotion_evidence.v1"

DEFAULT_MIN_APPLIED_SAMPLES = 2
DEFAULT_MIN_HOLDOUT_SAMPLES = 1
DEFAULT_MAX_ERROR_RATE = 0.05
DEFAULT_MAX_ERROR_RATE_DELTA = 0.05
DEFAULT_MAX_RETRY_RATE_DELTA = 0.10
DEFAULT_MAX_LATENCY_REGRESSION_MS = 2_000
DEFAULT_MAX_INVALIDATION_RATE = 0.02
DEFAULT_MIN_CACHE_HIT_RATE = 0.01
DEFAULT_ROLLBACK_ERROR_RATE = 0.20
DEFAULT_MIN_SAVINGS_REALIZATION_RATIO = 0.50
DEFAULT_MAX_EVIDENCE_AGE_HOURS = 72.0

STALE_DEPENDENCY_REASONS = {
    "dependency-cap-exceeded",
    "dependency-missing",
    "file-dependency-changed",
    "file-dependency-evidence-absent",
    "file-dependency-invalidated",
    "file-dependency-missing",
    "file-watch-disabled",
    "stale-risk-blockers",
}

_PUBLIC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_RAW_ID_HINT_RE = re.compile(
    r"[/\\]|\s|raw|secret|token|cache[-_]?key|request[-_]?id|session[-_]?id|thread[-_]?id|sha256:[0-9a-f]{32,}",
    re.IGNORECASE,
)


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
    }


def _public_id(value: Any, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _PUBLIC_ID_RE.match(text) and not _RAW_ID_HINT_RE.search(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
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


def _reason_code(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text if re.match(r"^[a-z0-9][a-z0-9_.:-]{0,79}$", text) else fallback


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _top_counter_value(counter: Counter[str]) -> str | None:
    rows = _counter_rows(counter)
    return str(rows[0]["value"]) if rows else None


def _source_surface(row: dict[str, Any]) -> str:
    return str(row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any]) -> str:
    return str(row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _feature_unit(routing: dict[str, Any]) -> dict[str, Any]:
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        value = routing.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _cache_replay_rule(cache: dict[str, Any]) -> dict[str, Any] | None:
    rule = cache.get("pattern_rule") if isinstance(cache.get("pattern_rule"), dict) else None
    if isinstance(rule, dict):
        return rule
    pattern_rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), dict) else {}
    for candidate in pattern_rules.get("rules") or []:
        if isinstance(candidate, dict):
            return candidate
    for candidate in reversed(pattern_rules.get("skip_reasons") or []):
        if isinstance(candidate, dict) and (
            candidate.get("rule_id")
            or candidate.get("candidate_id")
            or candidate.get("canary")
            or candidate.get("safety_stop")
        ):
            return candidate
    return None


def _cache_replay_canary(cache: dict[str, Any], rule: dict[str, Any] | None) -> dict[str, Any]:
    value = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else None
    if value is not None:
        return value
    if isinstance(rule, dict) and isinstance(rule.get("canary"), dict):
        return rule["canary"]
    return {}


def _canary_public(canary: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key in ("enabled", "selected", "fraction", "threshold"):
        if canary.get(key) is not None:
            public[key] = canary.get(key)
    for key in ("cohort", "unit", "reason", "status"):
        if canary.get(key) is not None:
            public[key] = public_label(canary.get(key), "unknown")
    public["pattern_hashes_included"] = False
    return public


def _cohort(cache: dict[str, Any], rule: dict[str, Any], canary: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    reason = str(cache.get("reason") or canary.get("reason") or rule.get("reason") or "").strip().lower()
    canary_status = str(canary.get("status") or "").strip().lower()
    nested_canary = canary.get("canary") if isinstance(canary.get("canary"), dict) else {}
    canary_cohort = str(canary.get("canary_cohort") or nested_canary.get("cohort") or "").strip().lower()
    rule_canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    safety_stop = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else cache.get("safety_stop")
    if isinstance(safety_stop, dict) or "safety-stop" in reason or status == "safety_stopped":
        return "safety_stop"
    if status == "invalidated" or canary_status == "invalidated" or cache.get("invalidated"):
        return "invalidated"
    if canary_status == "holdout" or reason == "canary_holdout" or canary_cohort == "canary_holdout":
        return "holdout"
    if rule_canary.get("selected") is False or rule_canary.get("cohort") == "canary_holdout":
        return "holdout"
    if canary_status == "applied" or status == "hit":
        return "applied"
    return "blocked"


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_rows": 0,
        "retry_attempts": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "observed_savings_usd": 0.0,
        "projected_savings_usd": 0.0,
        "actual_saved_cost_usd": 0.0,
        "cache_hit_count": 0,
        "miss_count": 0,
        "bypass_skipped_count": 0,
        "invalidation_count": 0,
        "stale_dependency_count": 0,
        "remaining_blocker_buckets": Counter(),
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "count": count,
        "error_count": _as_int(raw.get("error_count")),
        "retry_rows": _as_int(raw.get("retry_rows")),
        "retry_attempts": _as_int(raw.get("retry_attempts")),
        "cache_hit_count": _as_int(raw.get("cache_hit_count")),
        "invalidation_count": _as_int(raw.get("invalidation_count")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_rows")) / count, 6) if count else 0.0,
        "cache_hit_rate": round(_as_int(raw.get("cache_hit_count")) / count, 6) if count else 0.0,
        "invalidation_rate": round(_as_int(raw.get("invalidation_count")) / count, 6) if count else 0.0,
        "stale_dependency_rate": round(_as_int(raw.get("stale_dependency_count")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd")), 8),
        "cost_baseline_usd": round(_as_float(raw.get("cost_baseline_usd")), 8),
        "observed_savings_usd": round(_as_float(raw.get("observed_savings_usd")), 8),
        "actual_saved_cost_usd": round(_as_float(raw.get("actual_saved_cost_usd")), 8),
        "projected_savings_usd": round(_as_float(raw.get("projected_savings_usd")), 8),
        "miss_count": _as_int(raw.get("miss_count")),
        "bypass_skipped_count": _as_int(raw.get("bypass_skipped_count")),
        "stale_dependency_count": _as_int(raw.get("stale_dependency_count")),
        "remaining_blocker_breakdown": _counter_rows(raw.get("remaining_blocker_buckets") or Counter()),
    }


def _candidate_projected_hits(rule: dict[str, Any]) -> int:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    for source in (graduation, rule):
        for key in ("projected_hits", "projected_hit_count", "expected_hits", "expected_hit_count"):
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None:
                return _as_int(value)
    return 0


def _candidate_projected_savings(rule: dict[str, Any]) -> float:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    for source in (graduation, rule):
        for key in (
            "projected_savings_usd",
            "projected_saved_cost_usd",
            "expected_savings_usd",
            "expected_saved_cost_usd",
        ):
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None:
                return _as_float(value)
    return 0.0


def _candidate_projected_cohort_rows(rule: dict[str, Any]) -> int:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    for key in ("sample_count", "row_count", "matched_count"):
        if graduation.get(key) is not None:
            return _as_int(graduation.get(key))
    return 0


def _replay_source_schema(rule: dict[str, Any]) -> str | None:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else {}
    value = graduation.get("source_schema") or graduation.get("schema")
    return public_label(value, "unknown") if value else None


def _new_candidate(candidate_id: str, rule_id: str | None, rule: dict[str, Any], row: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "rule_id": rule_id,
        "policy_source": public_label(rule.get("policy_source") or "unknown", "unknown"),
        "source_surface": public_label(feature.get("source_surface") or _source_surface(row), "unknown"),
        "endpoint": public_label(feature.get("endpoint") or _endpoint(row), "unknown"),
        "category": public_label(row.get("category") or feature.get("category") or "unknown", "unknown"),
        "workflow_phase": public_label(feature.get("workflow_phase") or row.get("category") or "unknown", "unknown"),
        "cohorts": {
            "applied": _empty_cohort(),
            "holdout": _empty_cohort(),
            "blocked": _empty_cohort(),
            "invalidated": _empty_cohort(),
            "safety_stop": _empty_cohort(),
        },
        "projected_hit_count": _candidate_projected_hits(rule),
        "dry_run_projected_savings_usd": _candidate_projected_savings(rule),
        "projected_cohort_row_count": _candidate_projected_cohort_rows(rule),
        "replay_source_schema": _replay_source_schema(rule),
        "status_buckets": Counter(),
        "endpoint_buckets": Counter(),
        "category_buckets": Counter(),
        "workflow_phase_buckets": Counter(),
        "cache_decision_buckets": Counter(),
        "invalidation_reason_buckets": Counter(),
        "dependency_health_buckets": Counter(),
        "stale_dependency_buckets": Counter(),
        "reason_buckets": Counter(),
        "remaining_blocker_buckets": Counter(),
        "provider_adoption_observations": [],
        "latest_observed_at": None,
        "oldest_observed_at": None,
        "canary": _canary_public(rule.get("canary") if isinstance(rule.get("canary"), dict) else {}),
    }


def _projected_savings(cache: dict[str, Any], canary: dict[str, Any], row: dict[str, Any]) -> float:
    for value in (
        cache.get("estimated_saved_cost_usd"),
        canary.get("projected_input_savings_usd"),
        canary.get("projected_savings_usd"),
    ):
        amount = _as_float(value)
        if amount > 0:
            return amount
    baseline = _as_float(row.get("cost_baseline_usd"))
    actual = _as_float(row.get("cost_est_usd"))
    return max(0.0, baseline - actual)


def _observed_savings(cache: dict[str, Any], row: dict[str, Any], cohort: str) -> float:
    if cohort != "applied":
        return 0.0
    baseline = _as_float(row.get("cost_baseline_usd"))
    actual = _as_float(row.get("cost_est_usd"))
    if baseline or actual:
        return baseline - actual
    return _as_float(cache.get("estimated_saved_cost_usd"))


def _cache_outcome_status(cache: dict[str, Any], row: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    if status:
        return status
    return "hit" if _as_int(row.get("cache_hit")) else "unknown"


def _remaining_blocker(cache: dict[str, Any], canary: dict[str, Any], cohort: str) -> str | None:
    reason = (
        cache.get("invalidation_reason")
        or cache.get("reason")
        or canary.get("reason")
        or "unknown"
    )
    status = str(cache.get("status") or canary.get("status") or "").strip().lower()
    if cohort == "safety_stop":
        return _reason_code(reason, "local-canary-safety-stop")
    if cohort == "invalidated":
        return _reason_code(reason, "cache-replay-invalidated")
    if cohort in {"blocked", "holdout"}:
        return _reason_code(reason, "cache-replay-bypassed")
    if status in {"miss", "bypassed", "skipped", "disabled", "blocked", "invalidated"}:
        return _reason_code(reason, "cache-replay-not-hit")
    return None


def _dependency_health(cache: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: set[str] = set()
    for value in cache.get("cache_replay_blocker_reasons") or []:
        blockers.add(_reason_code(value))
    for key in ("reason", "invalidation_reason"):
        value = cache.get(key)
        if value:
            blockers.add(_reason_code(value))
    audit = cache.get("file_dependency_audit") if isinstance(cache.get("file_dependency_audit"), dict) else {}
    if audit.get("invalidation_reason"):
        blockers.add(_reason_code(audit.get("invalidation_reason")))
    if audit and not audit.get("file_dependency_evidence_available") and not audit.get("safe_invalidation_evidence"):
        blockers.add("file-dependency-missing")

    stale = sorted(reason for reason in blockers if reason in STALE_DEPENDENCY_REASONS or "stale-risk" in reason)
    if stale:
        return "stale-risk", stale
    if cache.get("invalidated") or cache.get("invalidation_reason"):
        return "invalidated", sorted(blockers)
    if audit.get("safe_invalidation_evidence") or cache.get("safe_invalidation_evidence"):
        return "fresh", []
    return "unknown", []


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    params: tuple[Any, ...]
    where = "where cache_json is not null and coalesce(provider, 'anthropic') = 'openai'"
    if since:
        where += " and created_at >= ?"
        params = (since, capped)
    else:
        params = (capped,)
    return [
        dict(row)
        for row in store_obj.conn.execute(
            f"""
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model, stream,
                   cache_hit, status_code, latency_ms, retry_count, cost_est_usd,
                   cost_baseline_usd, category, routing_json, cache_json
            from calls
            {where}
            order by created_at desc
            limit ?
            """,
            params,
        ).fetchall()
    ]


def _add_row(
    candidate: dict[str, Any],
    row: dict[str, Any],
    cache: dict[str, Any],
    canary: dict[str, Any],
    cohort: str,
    provider_adoption_windows: list[dict[str, Any]] | None = None,
) -> None:
    raw = candidate["cohorts"][cohort]
    raw["count"] += 1
    status_code = _as_int(row.get("status_code"), -1)
    if status_code >= 400:
        raw["error_count"] += 1
    retry_count = _as_int(row.get("retry_count"))
    if retry_count > 0:
        raw["retry_rows"] += 1
        raw["retry_attempts"] += retry_count
    latency_ms = _as_int(row.get("latency_ms"), -1)
    if latency_ms >= 0:
        raw["latency_ms_total"] += latency_ms
        raw["latency_sample_count"] += 1
    raw["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
    raw["cost_baseline_usd"] += _as_float(row.get("cost_baseline_usd"))
    observed_savings = _observed_savings(cache, row, cohort)
    raw["observed_savings_usd"] += observed_savings
    raw["actual_saved_cost_usd"] += observed_savings
    raw["projected_savings_usd"] += _projected_savings(cache, canary, row)
    cache_status = _cache_outcome_status(cache, row)
    if _as_int(row.get("cache_hit")) or cache_status == "hit":
        raw["cache_hit_count"] += 1
    elif cache_status == "miss":
        raw["miss_count"] += 1
    elif cache_status in {"bypassed", "skipped", "disabled", "blocked"} or cohort in {"blocked", "holdout"}:
        raw["bypass_skipped_count"] += 1
    if cohort == "invalidated" or cache.get("invalidated") or cache.get("invalidation_reason"):
        raw["invalidation_count"] += 1
    dependency_status, stale_dependency_reasons = _dependency_health(cache)
    if stale_dependency_reasons:
        raw["stale_dependency_count"] += 1

    candidate["status_buckets"][_status_bucket(row.get("status_code"))] += 1
    candidate["endpoint_buckets"][_reason_code(row.get("endpoint") or row.get("source_surface"), "unknown")] += 1
    candidate["category_buckets"][_reason_code(row.get("category"), "unknown")] += 1
    candidate["workflow_phase_buckets"][_reason_code(candidate.get("workflow_phase"), "unknown")] += 1
    candidate["cache_decision_buckets"][_reason_code(f"{cache.get('status') or 'unknown'}:{cache.get('reason') or 'unknown'}")] += 1
    candidate["dependency_health_buckets"][dependency_status] += 1
    for reason in stale_dependency_reasons:
        candidate["stale_dependency_buckets"][reason] += 1
    candidate["reason_buckets"][_reason_code(cache.get("reason") or canary.get("reason"), "unknown")] += 1
    remaining_blocker = _remaining_blocker(cache, canary, cohort)
    if remaining_blocker:
        raw["remaining_blocker_buckets"][remaining_blocker] += 1
        candidate["remaining_blocker_buckets"][remaining_blocker] += 1
    if cache.get("invalidation_reason"):
        candidate["invalidation_reason_buckets"][_reason_code(cache.get("invalidation_reason"))] += 1
    created_at = row.get("created_at")
    if created_at:
        if candidate["latest_observed_at"] is None or str(created_at) > str(candidate["latest_observed_at"]):
            candidate["latest_observed_at"] = str(created_at)
        if candidate["oldest_observed_at"] is None or str(created_at) < str(candidate["oldest_observed_at"]):
            candidate["oldest_observed_at"] = str(created_at)
    candidate["provider_adoption_observations"].append({
        "cohort": cohort,
        "provider_adoption_windows": provider_adoption_windows or [],
    })


def _stale_evidence(latest_observed_at: str | None, *, now: datetime, max_age_hours: float) -> dict[str, Any]:
    latest = _parse_time(latest_observed_at)
    if latest is None:
        return {"stale": False, "age_hours": None, "max_age_hours": round(float(max_age_hours), 3)}
    age_hours = (now.astimezone(timezone.utc) - latest).total_seconds() / 3600.0
    return {"stale": age_hours > max_age_hours, "age_hours": round(age_hours, 3), "max_age_hours": round(float(max_age_hours), 3)}


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _first_real_hit_status(*, observed_hits: int, applied_count: int, holdout_count: int) -> str:
    if observed_hits > 0:
        return "observed-hit"
    if applied_count > 0:
        return "awaiting-hit"
    if holdout_count > 0:
        return "holdout-only"
    return "not-evaluated"


def _canary_hit_measurement(
    *,
    candidate: dict[str, Any],
    cohorts: dict[str, dict[str, Any]],
    observed_savings_usd: float,
    legacy_projected_savings_usd: float,
) -> dict[str, Any]:
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    projected_hits = _as_int(candidate.get("projected_hit_count"))
    dry_run_projected_savings = _as_float(candidate.get("dry_run_projected_savings_usd"))
    projected_saved_usd = dry_run_projected_savings if dry_run_projected_savings > 0 else legacy_projected_savings_usd
    observed_hits = _as_int(applied.get("cache_hit_count"))
    observed_misses = _as_int(applied.get("miss_count"))
    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    holdout_forwards = _as_int(holdout.get("bypass_skipped_count")) + _as_int(holdout.get("miss_count"))
    return {
        "schema": "agentflow.openai_cache_replay_canary_hit_measurement.v1",
        "replay_source_schema": candidate.get("replay_source_schema"),
        "first_real_hit_status": _first_real_hit_status(
            observed_hits=observed_hits,
            applied_count=applied_count,
            holdout_count=holdout_count,
        ),
        "first_real_hit_observed": observed_hits > 0,
        "projected_hits": projected_hits,
        "projected_saved_usd": round(projected_saved_usd, 8),
        "dry_run_projected_savings_usd": round(dry_run_projected_savings, 8),
        "observed_hits": observed_hits,
        "observed_saved_usd": round(observed_savings_usd, 8),
        "hit_realization_rate": _ratio(float(observed_hits), float(projected_hits)),
        "savings_realization_rate": _ratio(float(observed_savings_usd), float(projected_saved_usd)),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "applied_hit_count": observed_hits,
        "applied_miss_count": observed_misses,
        "applied_bypass_skipped_count": _as_int(applied.get("bypass_skipped_count")),
        "holdout_cache_hit_count": _as_int(holdout.get("cache_hit_count")),
        "holdout_forwarded_count": holdout_forwards,
        "safety_stop_count": _as_int(cohorts["safety_stop"].get("count")),
        "invalidated_count": _as_int(cohorts["invalidated"].get("count")),
        "privacy": _privacy_summary(),
    }


def _aggregate_canary_hit_measurements(quality_gates: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = [
        item.get("canary_hit_measurement")
        for item in quality_gates
        if isinstance(item.get("canary_hit_measurement"), dict)
    ]
    projected_hits = sum(_as_int(item.get("projected_hits")) for item in measurements)
    projected_saved_usd = sum(_as_float(item.get("projected_saved_usd")) for item in measurements)
    observed_hits = sum(_as_int(item.get("observed_hits")) for item in measurements)
    observed_saved_usd = sum(_as_float(item.get("observed_saved_usd")) for item in measurements)
    applied_count = sum(_as_int(item.get("applied_count")) for item in measurements)
    holdout_count = sum(_as_int(item.get("holdout_count")) for item in measurements)
    return {
        "schema": "agentflow.openai_cache_replay_canary_hit_measurement.v1",
        "candidate_count": len(measurements),
        "first_real_hit_status": _first_real_hit_status(
            observed_hits=observed_hits,
            applied_count=applied_count,
            holdout_count=holdout_count,
        ),
        "first_real_hit_observed": observed_hits > 0,
        "first_real_hit_candidate_count": sum(1 for item in measurements if item.get("first_real_hit_observed")),
        "projected_hits": projected_hits,
        "projected_saved_usd": round(projected_saved_usd, 8),
        "dry_run_projected_savings_usd": round(
            sum(_as_float(item.get("dry_run_projected_savings_usd")) for item in measurements),
            8,
        ),
        "observed_hits": observed_hits,
        "observed_saved_usd": round(observed_saved_usd, 8),
        "hit_realization_rate": _ratio(float(observed_hits), float(projected_hits)),
        "savings_realization_rate": _ratio(float(observed_saved_usd), float(projected_saved_usd)),
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "applied_hit_count": sum(_as_int(item.get("applied_hit_count")) for item in measurements),
        "applied_miss_count": sum(_as_int(item.get("applied_miss_count")) for item in measurements),
        "applied_bypass_skipped_count": sum(_as_int(item.get("applied_bypass_skipped_count")) for item in measurements),
        "holdout_cache_hit_count": sum(_as_int(item.get("holdout_cache_hit_count")) for item in measurements),
        "holdout_forwarded_count": sum(_as_int(item.get("holdout_forwarded_count")) for item in measurements),
        "safety_stop_count": sum(_as_int(item.get("safety_stop_count")) for item in measurements),
        "invalidated_count": sum(_as_int(item.get("invalidated_count")) for item in measurements),
        "privacy": _privacy_summary(),
    }


def _candidate_local_promotion_evidence(item: dict[str, Any]) -> dict[str, Any]:
    verdict = str(item.get("verdict") or "unknown")
    blockers = sorted(
        {
            _reason_code(value)
            for value in [
                *(item.get("reason_codes") or []),
                item.get("top_remaining_blocker"),
            ]
            if value
        }
    )
    if verdict in {"widen", "promote"}:
        readiness = "promotion-ready"
        action = "promote-openai-cache-replay-rule-draft"
    elif verdict == "rollback":
        readiness = "rollback-required"
        action = "rollback-or-disable-openai-cache-replay-rule"
    elif verdict == "more-samples":
        readiness = "collecting-evidence"
        action = "collect-more-applied-and-holdout-cache-replay-evidence"
    else:
        readiness = "canary-only"
        action = "keep-openai-cache-replay-canary-only"
    if _as_int(item.get("safety_stop_count")):
        readiness = "rollback-required"
        action = "review-openai-cache-replay-safety-stop"
    return {
        "schema": "agentflow.openai_cache_replay_candidate_promotion_evidence.v1",
        "candidate_id": item.get("candidate_id"),
        "rule_id": item.get("rule_id"),
        "readiness": readiness,
        "recommended_local_action": action,
        "verdict": verdict,
        "reason_codes": blockers,
        "coverage": {
            "sample_count": _as_int(item.get("sample_count")),
            "applied_count": _as_int(item.get("applied_count")),
            "holdout_count": _as_int(item.get("holdout_count")),
            "blocked_count": _as_int(item.get("blocked_count")),
            "invalidated_count": _as_int(item.get("invalidated_count")),
            "safety_stop_count": _as_int(item.get("safety_stop_count")),
        },
        "outcomes": {
            "observed_hits": _as_int(item.get("actual_hits")),
            "miss_count": _as_int(item.get("miss_count")),
            "bypass_skipped_count": _as_int(item.get("bypass_skipped_count")),
            "hit_realization_rate": (item.get("canary_hit_measurement") or {}).get("hit_realization_rate")
            if isinstance(item.get("canary_hit_measurement"), dict)
            else None,
        },
        "savings": {
            "projected_saved_usd": round(_as_float(item.get("projected_saved_usd")), 8),
            "observed_saved_usd": round(_as_float(item.get("actual_saved_cost_usd")), 8),
            "savings_realization_rate": (item.get("canary_hit_measurement") or {}).get("savings_realization_rate")
            if isinstance(item.get("canary_hit_measurement"), dict)
            else None,
        },
        "target_local_rule_file": "cache_rules.yaml",
        "target_local_policy_section": "cache.rules",
        "privacy": _privacy_summary(),
    }


def _local_promotion_evidence(
    *,
    status: str,
    summary: dict[str, Any],
    quality_gates: list[dict[str, Any]],
    reason_counts: Counter[str],
    remaining_blocker_counts: Counter[str],
) -> dict[str, Any]:
    observed_rows = _as_int(summary.get("observed_openai_cache_replay_metadata_row_count"))
    applied_count = _as_int(summary.get("applied_count"))
    holdout_count = _as_int(summary.get("holdout_count"))
    projected_saved_usd = _as_float(summary.get("projected_saved_usd"))
    observed_saved_usd = _as_float(summary.get("actual_saved_cost_usd"))
    projected_hits = _as_int(summary.get("projected_hits"))
    actual_hits = _as_int(summary.get("actual_hits"))
    candidates = [_candidate_local_promotion_evidence(item) for item in quality_gates]
    if not observed_rows:
        readiness = "missing-canary-evidence"
        action = "stage-cache-replay-canary"
        reason_codes = ["missing-cache-replay-canary-lifecycle-evidence"]
        top_blocker = "missing-cache-replay-canary-lifecycle-evidence"
    elif any(item["readiness"] == "rollback-required" for item in candidates):
        readiness = "rollback-required"
        action = "rollback-or-disable-openai-cache-replay-rule"
        reason_codes = sorted(reason_counts) or ["cache-replay-safety-or-quality-blocker"]
        top_blocker = _top_counter_value(reason_counts) or _top_counter_value(remaining_blocker_counts)
    elif any(item["readiness"] == "promotion-ready" for item in candidates):
        readiness = "promotion-ready"
        action = "promote-openai-cache-replay-rule-draft"
        reason_codes = ["target-savings-met"]
        top_blocker = _top_counter_value(remaining_blocker_counts)
    elif applied_count or holdout_count:
        readiness = "canary-only"
        action = "keep-openai-cache-replay-canary-only"
        reason_codes = sorted(reason_counts) or ["cache-replay-needs-more-evidence"]
        top_blocker = _top_counter_value(reason_counts) or _top_counter_value(remaining_blocker_counts)
    else:
        readiness = "collecting-evidence"
        action = "collect-more-applied-and-holdout-cache-replay-evidence"
        reason_codes = sorted(reason_counts) or ["missing-applied-or-holdout-evidence"]
        top_blocker = _top_counter_value(reason_counts) or _top_counter_value(remaining_blocker_counts)
    return {
        "schema": LOCAL_PROMOTION_EVIDENCE_SCHEMA,
        "status": readiness,
        "source_status": status,
        "recommended_local_action": {
            "action": action,
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.rules",
            "policy_source": "local-file-backed",
            "reason_codes": reason_codes,
        },
        "coverage": {
            "observed_replay_metadata_rows": observed_rows,
            "candidate_count": _as_int(summary.get("candidate_count")),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "blocked_count": _as_int(summary.get("blocked_count")),
            "invalidated_count": _as_int(summary.get("invalidated_count")),
            "safety_stop_count": _as_int(summary.get("safety_stop_count")),
            "applied_rate": _ratio(float(applied_count), float(observed_rows)),
            "holdout_rate": _ratio(float(holdout_count), float(observed_rows)),
        },
        "outcomes": {
            "observed_hits": actual_hits,
            "miss_count": _as_int(summary.get("miss_count")),
            "bypass_skipped_count": _as_int(summary.get("bypass_skipped_count")),
            "hit_realization_rate": _ratio(float(actual_hits), float(projected_hits)),
        },
        "savings": {
            "projected_hits": projected_hits,
            "projected_saved_usd": round(projected_saved_usd, 8),
            "observed_saved_usd": round(observed_saved_usd, 8),
            "savings_realization_rate": _ratio(observed_saved_usd, projected_saved_usd),
        },
        "top_blocker": top_blocker,
        "candidate_evidence": candidates,
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy_summary(),
    }


def _decide_gate(
    *,
    cohorts: dict[str, dict[str, Any]],
    deltas: dict[str, Any],
    stale: dict[str, Any],
    thresholds: dict[str, Any],
    provider_adoption_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    reason_codes: list[str] = []
    warning_codes: list[str] = []

    if _as_int(applied.get("count")) < _as_int(thresholds["min_applied_samples"]):
        reason_codes.append("insufficient-applied-samples")
    if _as_int(holdout.get("count")) < _as_int(thresholds["min_holdout_samples"]):
        reason_codes.append("insufficient-holdout-samples")
    if reason_codes:
        return {"verdict": "more-samples", "reason_codes": reason_codes, "warning_codes": warning_codes}

    if _as_int(cohorts["safety_stop"].get("count")):
        reason_codes.append("safety-stop-observed")
    if _as_int(cohorts["invalidated"].get("count")) and _as_float(cohorts["invalidated"].get("invalidation_rate")) > _as_float(thresholds["max_invalidation_rate"]):
        reason_codes.append("invalidation-rate-above-threshold")
    if _as_float(applied.get("observed_savings_usd")) < 0:
        reason_codes.append("negative-observed-savings")
    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        reason_codes.append("rollback-error-rate")
    if reason_codes:
        return {"verdict": "rollback", "reason_codes": reason_codes, "warning_codes": warning_codes}

    if stale.get("stale"):
        reason_codes.append("stale-evidence")
    if any(_as_int(cohort.get("stale_dependency_count")) for cohort in cohorts.values()):
        reason_codes.append("stale-dependency-blocker")
    if _as_float(applied.get("error_rate")) > _as_float(thresholds["max_error_rate"]):
        reason_codes.append("applied-error-rate-above-threshold")
    if _as_float(deltas.get("applied_minus_holdout_error_rate")) > _as_float(thresholds["max_error_rate_delta"]):
        reason_codes.append("error-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_retry_rate")) > _as_float(thresholds["max_retry_rate_delta"]):
        reason_codes.append("retry-rate-regression")
    latency_delta = deltas.get("applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and _as_float(latency_delta) > _as_float(thresholds["max_latency_regression_ms"]):
        reason_codes.append("latency-regression")
    gate = provider_adoption_gate or {}
    if gate.get("blocking"):
        reason_codes.extend(str(code) for code in gate.get("reason_codes") or [])
    else:
        warning_codes.extend(str(code) for code in gate.get("warning_codes") or [])
    if _as_float(applied.get("cache_hit_rate")) < _as_float(thresholds["min_cache_hit_rate"]):
        reason_codes.append("cache-hit-rate-below-threshold")
    projected = _as_float(applied.get("projected_savings_usd"))
    observed = _as_float(applied.get("observed_savings_usd"))
    if projected > 0 and observed / projected < _as_float(thresholds["min_savings_realization_ratio"]):
        reason_codes.append("savings-below-projection")
    elif observed > 0:
        warning_codes.append("target-savings-met")

    if reason_codes:
        return {"verdict": "hold", "reason_codes": sorted(set(reason_codes)), "warning_codes": sorted(set(warning_codes))}
    return {"verdict": "widen", "reason_codes": ["target-savings-met"], "warning_codes": sorted(set(warning_codes))}


def _finalize_candidate(candidate: dict[str, Any], *, now: datetime, thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {key: _finalize_cohort(value) for key, value in candidate["cohorts"].items()}
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    deltas = {
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_cache_hit_rate": round(_as_float(applied["cache_hit_rate"]) - _as_float(holdout["cache_hit_rate"]), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
    }
    stale = _stale_evidence(
        candidate.get("latest_observed_at"),
        now=now,
        max_age_hours=_as_float(thresholds["max_evidence_age_hours"]),
    )
    provider_adoption_gate = build_provider_adoption_gate(
        candidate.get("provider_adoption_observations") or [],
        thresholds={
            key: value
            for key, value in thresholds.items()
            if key.startswith("min_provider_adoption") or key.startswith("max_applied")
        },
        block_on_missing=True,
        block_on_insufficient=True,
    )
    gate = _decide_gate(
        cohorts=cohorts,
        deltas=deltas,
        stale=stale,
        thresholds=thresholds,
        provider_adoption_gate=provider_adoption_gate,
    )
    sample_count = sum(_as_int(value.get("count")) for value in cohorts.values())
    projected = sum(_as_float(value.get("projected_savings_usd")) for value in cohorts.values())
    observed = _as_float(applied.get("observed_savings_usd"))
    applied_count = _as_int(applied.get("count"))
    holdout_count = _as_int(holdout.get("count"))
    blocked_count = _as_int(cohorts["blocked"].get("count"))
    invalidated_count = _as_int(cohorts["invalidated"].get("count"))
    safety_stop_count = _as_int(cohorts["safety_stop"].get("count"))
    skipped_count = (
        sum(_as_int(value.get("bypass_skipped_count")) for value in cohorts.values())
        + invalidated_count
        + safety_stop_count
    )
    actual_hit_count = sum(_as_int(value.get("cache_hit_count")) for value in cohorts.values())
    miss_count = sum(_as_int(value.get("miss_count")) for value in cohorts.values())
    projected_hits = _as_int(candidate.get("projected_hit_count"))
    canary_hit_measurement = _canary_hit_measurement(
        candidate=candidate,
        cohorts=cohorts,
        observed_savings_usd=observed,
        legacy_projected_savings_usd=projected,
    )
    return {
        "schema": QUALITY_GATE_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "rule_id": candidate.get("rule_id"),
        "policy_source": candidate.get("policy_source"),
        "source_surface": candidate.get("source_surface"),
        "endpoint": candidate.get("endpoint"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "sample_count": sample_count,
        "replay_ready": bool(applied_count or holdout_count or actual_hit_count),
        "readiness": "replay-ready" if applied_count or holdout_count or actual_hit_count else "skipped",
        "replay_source_schema": candidate.get("replay_source_schema"),
        "projected_hit_count": projected_hits,
        "projected_hits": projected_hits,
        "projected_saved_usd": canary_hit_measurement["projected_saved_usd"],
        "dry_run_projected_savings_usd": canary_hit_measurement["dry_run_projected_savings_usd"],
        "projected_cohort_row_count": _as_int(candidate.get("projected_cohort_row_count")),
        "actual_hit_count": actual_hit_count,
        "actual_hits": actual_hit_count,
        "actual_saved_cost_usd": round(observed, 8),
        "miss_count": miss_count,
        "bypass_skipped_count": sum(_as_int(value.get("bypass_skipped_count")) for value in cohorts.values()),
        "skipped_count": skipped_count,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "blocked_count": blocked_count,
        "invalidated_count": invalidated_count,
        "safety_stop_count": safety_stop_count,
        "cohort_counts": {key: value["count"] for key, value in cohorts.items()},
        "cohort_metrics": cohorts,
        "applied_vs_holdout_deltas": deltas,
        "observed_savings_usd": round(observed, 8),
        "projected_savings_usd": round(projected, 8),
        "canary_hit_measurement": canary_hit_measurement,
        "first_real_hit_status": canary_hit_measurement["first_real_hit_status"],
        "first_real_hit_observed": canary_hit_measurement["first_real_hit_observed"],
        "status_code_breakdown": _counter_rows(candidate["status_buckets"]),
        "endpoint_breakdown": _counter_rows(candidate["endpoint_buckets"]),
        "category_breakdown": _counter_rows(candidate["category_buckets"]),
        "workflow_phase_breakdown": _counter_rows(candidate["workflow_phase_buckets"]),
        "cache_decision_breakdown": _counter_rows(candidate["cache_decision_buckets"]),
        "invalidation_reason_breakdown": _counter_rows(candidate["invalidation_reason_buckets"]),
        "dependency_health_breakdown": _counter_rows(candidate["dependency_health_buckets"]),
        "stale_dependency_breakdown": _counter_rows(candidate["stale_dependency_buckets"]),
        "reason_code_breakdown": _counter_rows(candidate["reason_buckets"]),
        "remaining_blocker_breakdown": _counter_rows(candidate["remaining_blocker_buckets"]),
        "top_remaining_blocker": _top_counter_value(candidate["remaining_blocker_buckets"]),
        "canary": candidate.get("canary") or {"pattern_hashes_included": False},
        "oldest_observed_at": candidate.get("oldest_observed_at"),
        "latest_observed_at": candidate.get("latest_observed_at"),
        "stale_evidence": stale,
        "thresholds": thresholds,
        "provider_adoption_gate": provider_adoption_gate,
        "verdict": gate["verdict"],
        "reason_codes": gate["reason_codes"],
        "warning_codes": gate["warning_codes"],
        "next_action": {
            "widen": "widen_local_openai_cache_replay_rule",
            "hold": "keep_current_openai_cache_replay_fraction",
            "rollback": "rollback_or_disable_openai_cache_replay_rule",
            "more-samples": "collect_more_applied_and_holdout_cache_replay_evidence",
        }[gate["verdict"]],
        "privacy": _privacy_summary(),
    }


def build_openai_cache_replay_impact_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = DEFAULT_MIN_APPLIED_SAMPLES,
    min_holdout_samples: int = DEFAULT_MIN_HOLDOUT_SAMPLES,
    max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
    max_error_rate_delta: float = DEFAULT_MAX_ERROR_RATE_DELTA,
    max_retry_rate_delta: float = DEFAULT_MAX_RETRY_RATE_DELTA,
    max_latency_regression_ms: int = DEFAULT_MAX_LATENCY_REGRESSION_MS,
    max_invalidation_rate: float = DEFAULT_MAX_INVALIDATION_RATE,
    min_cache_hit_rate: float = DEFAULT_MIN_CACHE_HIT_RATE,
    rollback_error_rate: float = DEFAULT_ROLLBACK_ERROR_RATE,
    min_savings_realization_ratio: float = DEFAULT_MIN_SAVINGS_REALIZATION_RATIO,
    max_evidence_age_hours: float = DEFAULT_MAX_EVIDENCE_AGE_HOURS,
    now: datetime | None = None,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    thresholds = {
        "min_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "max_invalidation_rate": round(float(max_invalidation_rate), 6),
        "min_cache_hit_rate": round(float(min_cache_hit_rate), 6),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
        "min_savings_realization_ratio": round(float(min_savings_realization_ratio), 6),
        "max_evidence_age_hours": round(float(max_evidence_age_hours), 3),
    }
    thresholds.update(provider_adoption_thresholds())
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    adoption_by_call = provider_adoption_windows_by_call(store_obj, [row.get("id") for row in rows])
    candidates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    cohort_counts: Counter[str] = Counter()
    for row in rows:
        cache = _json_obj(row.get("cache_json"))
        rule = _cache_replay_rule(cache)
        if rule is None:
            continue
        routing = _json_obj(row.get("routing_json"))
        feature = _feature_unit(routing)
        canary = _cache_replay_canary(cache, rule)
        cid = _public_id(rule.get("candidate_id") or canary.get("candidate_id") or rule.get("rule_id"), "candidate-id")
        rid = _public_id(rule.get("rule_id") or canary.get("rule_id"), "rule-id")
        if cid is None:
            cid = rid or "unknown-openai-cache-replay"
        aggregate = candidates.setdefault(cid, _new_candidate(cid, rid, rule, row, feature))
        cohort = _cohort(cache, rule, canary)
        _add_row(aggregate, row, cache, canary, cohort, adoption_by_call.get(str(row.get("id") or ""), []))
        observed_rows += 1
        cohort_counts[cohort] += 1

    now_dt = now or datetime.now(timezone.utc)
    quality_gates = [
        _finalize_candidate(candidate, now=now_dt, thresholds=thresholds)
        for candidate in candidates.values()
    ]
    quality_gates.sort(key=lambda item: (str(item.get("verdict")), str(item.get("candidate_id"))))
    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in quality_gates)
    reason_counts: Counter[str] = Counter()
    remaining_blocker_counts: Counter[str] = Counter()
    for item in quality_gates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        for row in item.get("remaining_blocker_breakdown") or []:
            remaining_blocker_counts[str(row.get("value") or "unknown")] += _as_int(row.get("count"))

    top_level_gate = {
        "schema": QUALITY_GATE_SCHEMA,
        "thresholds": thresholds,
        "verdict_counts": _counter_rows(verdict_counts),
        "reason_code_counts": _counter_rows(reason_counts),
        "candidate_results": [
            {
                "candidate_id": item.get("candidate_id"),
                "rule_id": item.get("rule_id"),
                "verdict": item.get("verdict"),
                "reason_codes": item.get("reason_codes") or [],
                "cohort_counts": item.get("cohort_counts") or {},
                "projected_hits": item.get("projected_hits") or 0,
                "projected_saved_usd": item.get("projected_saved_usd") or 0.0,
                "actual_hits": item.get("actual_hits") or 0,
                "actual_saved_cost_usd": item.get("actual_saved_cost_usd") or 0.0,
                "miss_count": item.get("miss_count") or 0,
                "bypass_skipped_count": item.get("bypass_skipped_count") or 0,
                "first_real_hit_status": item.get("first_real_hit_status"),
                "first_real_hit_observed": bool(item.get("first_real_hit_observed")),
                "top_remaining_blocker": item.get("top_remaining_blocker"),
                "observed_savings_usd": item.get("observed_savings_usd"),
                "projected_savings_usd": item.get("projected_savings_usd"),
                "canary_hit_measurement": item.get("canary_hit_measurement") or {},
            }
            for item in quality_gates
        ],
        "default_apply": False,
        "wrote_local_files": False,
        "provider_calls_made": False,
    }
    status = "matched" if observed_rows else "no-openai-cache-replay-metadata"
    summary = {
        "sampled_call_count": len(rows),
        "observed_openai_cache_replay_metadata_row_count": observed_rows,
        "candidate_count": len(quality_gates),
        "applied_count": cohort_counts.get("applied", 0),
        "holdout_count": cohort_counts.get("holdout", 0),
        "blocked_count": cohort_counts.get("blocked", 0),
        "invalidated_count": cohort_counts.get("invalidated", 0),
        "safety_stop_count": cohort_counts.get("safety_stop", 0),
        "replay_ready_cohort_count": sum(1 for item in quality_gates if item.get("readiness") == "replay-ready"),
        "skipped_cohort_count": sum(1 for item in quality_gates if item.get("readiness") == "skipped"),
        "projected_hits": sum(_as_int(item.get("projected_hits")) for item in quality_gates),
        "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in quality_gates), 8),
        "dry_run_projected_savings_usd": round(
            sum(_as_float(item.get("dry_run_projected_savings_usd")) for item in quality_gates),
            8,
        ),
        "actual_hits": sum(_as_int(item.get("actual_hits")) for item in quality_gates),
        "actual_saved_cost_usd": round(sum(_as_float(item.get("actual_saved_cost_usd")) for item in quality_gates), 8),
        "miss_count": sum(_as_int(item.get("miss_count")) for item in quality_gates),
        "bypass_skipped_count": sum(_as_int(item.get("bypass_skipped_count")) for item in quality_gates),
        "top_remaining_blocker": _top_counter_value(remaining_blocker_counts),
        "observed_savings_usd": round(sum(_as_float(item.get("observed_savings_usd")) for item in quality_gates), 8),
        "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in quality_gates), 8),
        "canary_hit_measurement": _aggregate_canary_hit_measurements(quality_gates),
    }
    summary["first_real_hit_status"] = summary["canary_hit_measurement"]["first_real_hit_status"]
    summary["first_real_hit_observed"] = summary["canary_hit_measurement"]["first_real_hit_observed"]
    summary["first_real_hit_candidate_count"] = summary["canary_hit_measurement"]["first_real_hit_candidate_count"]
    promotion_evidence = _local_promotion_evidence(
        status=status,
        summary=summary,
        quality_gates=quality_gates,
        reason_counts=reason_counts,
        remaining_blocker_counts=remaining_blocker_counts,
    )
    summary["recommended_local_action"] = promotion_evidence["recommended_local_action"]["action"]
    summary["local_promotion_status"] = promotion_evidence["status"]
    summary["top_blocker"] = promotion_evidence["top_blocker"]
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
        "status": status,
        "summary": summary,
        "local_promotion_evidence": promotion_evidence,
        "recommended_local_action": promotion_evidence["recommended_local_action"],
        "cohort_breakdown": _counter_rows(cohort_counts),
        "remaining_blocker_breakdown": _counter_rows(remaining_blocker_counts),
        "quality_gate": top_level_gate,
        "candidates": quality_gates,
        "privacy": {
            **_privacy_summary(),
            "basis": "local calls table metadata plus sanitized cache replay canary and decision summaries only",
        },
    }


def build_openai_cache_replay_lifecycle_feedback(result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        return None
    return {
        "schema": LIFECYCLE_SCHEMA,
        "event_type": "openai_cache_replay_impact",
        "lifecycle_kind": "openai_cache_replay",
        "lifecycle_phase": "impact",
        "generated_at": result.get("generated_at") or utc_now(),
        "summary": result.get("summary") or {},
        "quality_gate": result.get("quality_gate") or {},
        "candidate_results": [
            {
                "candidate_id": item.get("candidate_id"),
                "rule_id": item.get("rule_id"),
                "policy_source": item.get("policy_source"),
                "verdict": item.get("verdict"),
                "reason_codes": item.get("reason_codes") or [],
                "warning_codes": item.get("warning_codes") or [],
                "cohort_counts": item.get("cohort_counts") or {},
                "readiness": item.get("readiness"),
                "projected_hits": item.get("projected_hits") or 0,
                "projected_saved_usd": item.get("projected_saved_usd") or 0.0,
                "actual_hits": item.get("actual_hits") or 0,
                "actual_saved_cost_usd": item.get("actual_saved_cost_usd") or 0.0,
                "miss_count": item.get("miss_count") or 0,
                "bypass_skipped_count": item.get("bypass_skipped_count") or 0,
                "skipped_count": item.get("skipped_count") or 0,
                "safety_stop_count": item.get("safety_stop_count") or 0,
                "top_remaining_blocker": item.get("top_remaining_blocker"),
                "observed_savings_usd": item.get("observed_savings_usd"),
                "projected_savings_usd": item.get("projected_savings_usd"),
                "canary_hit_measurement": item.get("canary_hit_measurement") or {},
                "endpoint": item.get("endpoint"),
                "category": item.get("category"),
                "workflow_phase": item.get("workflow_phase"),
            }
            for item in result.get("candidates") or []
            if isinstance(item, dict)
        ],
        "privacy": result.get("privacy") or _privacy_summary(),
    }
