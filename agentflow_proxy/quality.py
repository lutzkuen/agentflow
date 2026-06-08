from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from agentflow_proxy.codex_app_policy import CODEX_APP_SOURCE_SURFACE


QUALITY_SIGNAL_SCHEMA = "agentflow.quality_signals.v1"
ABANDONED_AFTER_SECONDS = 30 * 60


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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": QUALITY_SIGNAL_SCHEMA,
        "source_surface": source_surface,
        "status": status,
        "optimized": bool(optimized),
        "signals": signals,
        "signal_codes": [signal["code"] for signal in signals],
        "signal_count": len(signals),
    }
    if observed_age_seconds is not None:
        payload["observed_age_seconds"] = observed_age_seconds
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    if retry_count is not None:
        payload["retry_count"] = _as_int(retry_count)
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

    return _compact(
        source_surface=source_surface,
        status=outcome,
        signals=signals,
        optimized=optimized,
        latency_ms=latency_ms,
        retry_count=retries,
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
