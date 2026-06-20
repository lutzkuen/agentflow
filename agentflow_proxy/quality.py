from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from agentflow_proxy.codex_turn_policy import CODEX_APP_SOURCE_SURFACE


QUALITY_SIGNAL_SCHEMA = "agentflow.quality_signals.v1"
ABANDONED_AFTER_SECONDS = 30 * 60

ADOPTION_SIGNAL_BY_STATUS = {
    "fulfilled": ("tool-use-fulfilled", "info", "provider tool-use was followed by a matching tool result"),
    "pending": ("tool-use-pending", "info", "provider tool-use has not yet reached a terminal adoption outcome"),
    "abandoned": ("tool-use-abandoned", "warning", "provider tool-use was not followed by a tool result before the local TTL"),
    "orphan_result": ("orphan-tool-result", "warning", "tool result arrived without a matching pending provider tool-use window"),
    "unknown": ("unsupported-tool-protocol-shape", "warning", "provider tool-use metadata had an unsupported protocol shape"),
}
ADOPTION_RISK_STATUSES = {"abandoned", "orphan_result", "unknown"}


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(created_at: Any, *, now: datetime | None = None) -> int | None:
    created = _parse_dt(created_at)
    if created is None:
        return None
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return max(0, int((reference.astimezone(timezone.utc) - created).total_seconds()))


def _signal(code: str, severity: str, reason: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "reason": reason}


def _risk_level(signals: Iterable[dict[str, str]]) -> str:
    severities = {str(signal.get("severity") or "") for signal in signals}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    return "info"


def _public_label(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if len(text) <= 128 and text[0].isalnum() and all(ch in allowed for ch in text):
        lowered = text.lower()
        blocked = (
            "raw",
            "secret",
            "api_key",
            "apikey",
            "request_id",
            "session_id",
            "tenant_id",
            "thread_id",
            "provider_body",
            "tool_payload",
            "cache_key",
        )
        if not any(token in lowered for token in blocked):
            return text
    return fallback


def _status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _provider_adoption_summary(rows: Iterable[dict[str, Any]] | None) -> dict[str, Any] | None:
    sanitized_rows: list[dict[str, Any]] = []
    age_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}
    total_tool_uses = 0
    total_tool_results = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "unknown")
        age_bucket = _public_label(row.get("age_bucket"), fallback="unknown")
        relationship = _public_label(row.get("relationship"), fallback="unknown")
        age_counts[age_bucket] = age_counts.get(age_bucket, 0) + 1
        relationship_counts[relationship] = relationship_counts.get(relationship, 0) + 1
        total_tool_uses += _as_int(row.get("tool_use_count"))
        total_tool_results += _as_int(row.get("tool_result_count"))
        sanitized_rows.append({
            "status": status,
            "reason": _public_label(row.get("reason"), fallback="unknown"),
            "age_bucket": age_bucket,
            "relationship": relationship,
            "tool_use_count": _as_int(row.get("tool_use_count")),
            "tool_result_count": _as_int(row.get("tool_result_count")),
        })
    if not sanitized_rows:
        return None
    counts = _status_counts(sanitized_rows)
    return {
        "schema": "agentflow.provider_adoption_quality.v1",
        "window_count": len(sanitized_rows),
        "status_counts": dict(sorted(counts.items())),
        "risk_window_count": sum(counts.get(status, 0) for status in ADOPTION_RISK_STATUSES),
        "age_bucket_counts": dict(sorted(age_counts.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "tool_use_count": total_tool_uses,
        "tool_result_count": total_tool_results,
        "windows": sanitized_rows[:20],
        "privacy": {
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "tool_ids_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "correlation_digests_included": False,
        },
    }


def _cohort_value(value: Any) -> str | None:
    label = _public_label(value, fallback="")
    return label or None


def _collect_optimization_cohorts(
    *,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    cohorts: list[dict[str, str]] = []

    def add(family: str, meta: Any, *, nested_key: str | None = None) -> None:
        if not isinstance(meta, dict):
            return
        candidate = meta.get(nested_key) if nested_key else meta
        if not isinstance(candidate, dict):
            return
        cohort = _cohort_value(candidate.get("cohort") or candidate.get("canary_cohort"))
        status = _cohort_value(candidate.get("status") or candidate.get("lifecycle_event"))
        reason = _cohort_value(candidate.get("reason"))
        if cohort is None and status not in {"applied", "holdout", "canary_applied", "canary_holdout", "safety_stopped"}:
            return
        item: dict[str, str] = {"family": family}
        if status:
            item["status"] = status
        if cohort:
            item["cohort"] = cohort
        if reason:
            item["reason"] = reason
        policy_id = _cohort_value(candidate.get("policy_id") or candidate.get("candidate_id") or candidate.get("rule_id"))
        if policy_id:
            item["policy_id"] = policy_id
        if item not in cohorts:
            cohorts.append(item)

    routing = routing_meta or {}
    crunch = crunch_meta or {}
    cache = cache_meta or {}
    add("routing_experiment", routing, nested_key="routing_experiment")
    add("phase_routing", routing, nested_key="phase_canary")
    add("openai_routing", routing, nested_key="openai_canary")
    managed = routing.get("managed_recommendation")
    if isinstance(managed, dict):
        add("managed_recommendation", managed)
        add("managed_recommendation", managed, nested_key="canary")
    add("old_context_summarization", crunch, nested_key="old_context_summarization")
    add("cache_replay", cache, nested_key="cache_replay_canary")
    if cache.get("status") in {"holdout", "applied"}:
        add("cache", cache)
    return cohorts[:20]


def _optimized_from_decisions(
    *,
    requested_model: Any = None,
    routed_model: Any = None,
    cache_hit: Any = None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
) -> bool:
    routing = routing_meta or {}
    crunch = crunch_meta or {}
    cache = cache_meta or {}
    return bool(
        cache_hit
        or cache.get("status") == "hit"
        or crunch.get("changed")
        or routing.get("applied")
        or (requested_model and routed_model and requested_model != routed_model)
    )


def _compact(
    *,
    source_surface: str,
    status: str,
    signals: list[dict[str, str]],
    optimized: bool,
    observed_age_seconds: int | None = None,
    latency_ms: Any = None,
    retry_count: Any = None,
    provider_adoption: dict[str, Any] | None = None,
    optimization_cohorts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": QUALITY_SIGNAL_SCHEMA,
        "source_surface": source_surface,
        "status": status,
        "optimized": bool(optimized),
        "signals": signals,
        "signal_codes": [signal["code"] for signal in signals],
        "signal_count": len(signals),
        "risk_level": _risk_level(signals),
    }
    if observed_age_seconds is not None:
        payload["observed_age_seconds"] = observed_age_seconds
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if retry_count is not None:
        payload["retry_count"] = _as_int(retry_count)
    if provider_adoption is not None:
        payload["provider_adoption"] = provider_adoption
    if optimization_cohorts:
        payload["optimization_cohorts"] = optimization_cohorts
    return payload


def derive_provider_quality_signals(
    *,
    source_surface: str,
    status_code: Any,
    retry_count: Any = 0,
    latency_ms: Any = None,
    error: Any = None,
    requested_model: Any = None,
    routed_model: Any = None,
    cache_hit: Any = None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
    provider_adoption_windows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status = _as_int(status_code)
    retries = _as_int(retry_count)
    error_text = str(error or "")
    local_throttled = error_text.startswith("temporarily limiting requests")
    optimized = _optimized_from_decisions(
        requested_model=requested_model,
        routed_model=routed_model,
        cache_hit=cache_hit,
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )

    signals: list[dict[str, str]] = []
    if local_throttled:
        outcome = "local_throttled"
        signals.append(_signal("local-throttled", "warning", "proxy deferred the request due to local tier cooldown"))
    elif 200 <= status < 400:
        outcome = "success"
        signals.append(_signal("success", "info", "provider request completed successfully"))
    elif status:
        outcome = "failure"
        signals.append(_signal("failure", "error", "provider request returned an error status"))
    elif error_text:
        outcome = "failure"
        signals.append(_signal("failure", "error", "proxy or upstream request failed"))
    else:
        outcome = "unknown"
        signals.append(_signal("unknown", "warning", "outcome metadata was incomplete"))

    if retries > 0:
        code = "retry-after-error" if outcome == "success" else "retry-exhausted"
        signals.append(_signal(code, "warning", "request needed retry handling before final outcome"))
    if status in {429, 529} and not local_throttled:
        signals.append(_signal("upstream-rate-limited", "warning", "upstream provider returned rate-limit or overload status"))
    if optimized and outcome == "failure":
        signals.append(_signal("optimized-failure", "error", "optimized request had an error outcome"))
    elif optimized and outcome == "success":
        signals.append(_signal("optimized-success", "info", "optimized request completed successfully"))

    adoption_summary = _provider_adoption_summary(provider_adoption_windows)
    if adoption_summary is not None:
        for status in sorted(adoption_summary.get("status_counts") or {}):
            mapped = ADOPTION_SIGNAL_BY_STATUS.get(status)
            if mapped is None:
                continue
            code, severity, reason = mapped
            signals.append(_signal(code, severity, reason))
        if optimized and adoption_summary.get("risk_window_count"):
            signals.append(_signal("optimized-adoption-risk", "warning", "optimized request had risky provider tool-use adoption metadata"))

    optimization_cohorts = _collect_optimization_cohorts(
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )

    return _compact(
        source_surface=source_surface,
        status=outcome,
        signals=signals,
        optimized=optimized,
        latency_ms=latency_ms,
        retry_count=retries,
        provider_adoption=adoption_summary,
        optimization_cohorts=optimization_cohorts,
    )


def derive_codex_turn_quality_signals(
    *,
    created_at: Any = None,
    response_event_id: Any = None,
    error_code: Any = None,
    error_message: Any = None,
    latency_ms: Any = None,
    routing_meta: dict[str, Any] | None = None,
    crunch_meta: dict[str, Any] | None = None,
    cache_meta: dict[str, Any] | None = None,
    now: datetime | None = None,
    abandoned_after_seconds: int = ABANDONED_AFTER_SECONDS,
) -> dict[str, Any]:
    optimized = _optimized_from_decisions(
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )
    signals: list[dict[str, str]] = []
    age = _age_seconds(created_at, now=now)
    if error_code is not None:
        outcome = "failure"
        signals.append(_signal("failure", "error", "Codex app turn returned a JSON-RPC error"))
    elif response_event_id:
        outcome = "success"
        signals.append(_signal("success", "info", "Codex app turn produced a response event"))
    elif age is not None and age >= abandoned_after_seconds:
        outcome = "abandoned"
        signals.append(_signal("abandoned", "warning", "Codex app turn has no response after the abandoned threshold"))
    else:
        outcome = "pending"
        signals.append(_signal("pending", "info", "Codex app turn has no response event yet"))

    if optimized and outcome == "failure":
        signals.append(_signal("optimized-failure", "error", "optimized Codex app turn had an error outcome"))
    elif optimized and outcome == "success":
        signals.append(_signal("optimized-success", "info", "optimized Codex app turn completed successfully"))
    if error_message and outcome == "failure":
        signals.append(_signal("jsonrpc-error", "error", "Codex app response included JSON-RPC error metadata"))

    return _compact(
        source_surface=CODEX_APP_SOURCE_SURFACE,
        status=outcome,
        signals=signals,
        optimized=optimized,
        observed_age_seconds=age,
        latency_ms=latency_ms,
    )


def summarize_quality_signals(units: Iterable[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    optimized_units = 0
    optimized_successes = 0
    optimized_failures = 0
    units_count = 0
    for unit in units:
        quality = unit.get("quality_signals")
        if not isinstance(quality, dict):
            quality = (unit.get("outcome_features") or {}).get("quality_signals")
        if not isinstance(quality, dict):
            continue
        units_count += 1
        status = str(quality.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if quality.get("optimized"):
            optimized_units += 1
            if status == "success":
                optimized_successes += 1
            elif status in {"failure", "local_throttled", "abandoned"}:
                optimized_failures += 1
        for code in quality.get("signal_codes") or []:
            key = str(code)
            signal_counts[key] = signal_counts.get(key, 0) + 1

    return {
        "schema": "agentflow.quality_signal_summary.v1",
        "units": units_count,
        "optimized_units": optimized_units,
        "optimized_successes": optimized_successes,
        "optimized_failures": optimized_failures,
        "by_status": [
            {"status": key, "count": value}
            for key, value in sorted(status_counts.items())
        ],
        "by_signal": [
            {"signal": key, "count": value}
            for key, value in sorted(signal_counts.items())
        ],
    }
