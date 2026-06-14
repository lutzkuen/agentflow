from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.router import OPENAI_LARGE_DEFAULT, OPENAI_SMALL_DEFAULT, OPENAI_TINY_DEFAULT
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_routing_opportunity.v1"
DEFAULT_MIN_SAMPLES = 5
DEFAULT_MAX_ERROR_RATE = 0.05
DEFAULT_MAX_RETRY_RATE = 0.20
DEFAULT_MAX_EVIDENCE_AGE_HOURS = 72.0
DEFAULT_SMALL_TEXT_CHARS_LT = 6000
DEFAULT_TINY_TEXT_CHARS_LT = 1500


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


OPENAI_SMALL_TEXT_CHARS_LT = _env_int("AGENTFLOW_OPENAI_SMALL_TEXT_CHARS_LT", DEFAULT_SMALL_TEXT_CHARS_LT)
OPENAI_TINY_TEXT_CHARS_LT = _env_int("AGENTFLOW_OPENAI_TINY_TEXT_CHARS_LT", DEFAULT_TINY_TEXT_CHARS_LT)


def _json_obj(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        import json

        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
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


def _text_bucket(text_chars: int) -> str:
    if text_chars < 1500:
        return "lt-1_5k"
    if text_chars < 6000:
        return "1_5k-6k"
    if text_chars < 32000:
        return "6k-32k"
    return "gte-32k"


def _token_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "unknown"
    if tokens < 1000:
        return "lt-1k"
    if tokens < 4000:
        return "1k-4k"
    if tokens < 16000:
        return "4k-16k"
    return "gte-16k"


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any]) -> bool:
    if "has_tools" in routing:
        return bool(routing.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    if category.startswith("tool-"):
        return True
    reason = str(cache.get("reason") or "").lower()
    return "tool" in reason


def _text_chars(row: dict[str, Any], routing: dict[str, Any]) -> int:
    chars = _as_int(routing.get("text_chars"))
    if chars > 0:
        return chars
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * 4)


def _input_tokens(row: dict[str, Any], text_chars: int) -> int:
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    if tokens > 0:
        return tokens
    return max(0, text_chars // 4)


def _output_tokens(row: dict[str, Any]) -> int:
    return _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))


def _source_surface(row: dict[str, Any]) -> str:
    return str(row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any]) -> str:
    return str(row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _simulate_openai_route(
    *,
    requested_model: str,
    category: str,
    has_tools: bool,
    text_chars: int,
) -> tuple[str | None, str, str]:
    requested_l = requested_model.lower()
    if requested_l == "gpt-5.4" and text_chars < OPENAI_SMALL_TEXT_CHARS_LT and not has_tools:
        return "gpt-5.4-mini", "proposed-canary-default-off", "gpt-5.4-large-to-mini-short-non-tool"
    if requested_l == OPENAI_LARGE_DEFAULT.lower() and text_chars < OPENAI_SMALL_TEXT_CHARS_LT and not has_tools:
        return OPENAI_SMALL_DEFAULT, "existing-threshold", "large-to-small-short-non-tool"
    if requested_l == OPENAI_SMALL_DEFAULT.lower() and text_chars < OPENAI_TINY_TEXT_CHARS_LT and not has_tools:
        return OPENAI_TINY_DEFAULT, "existing-threshold", "small-to-tiny-short-non-tool"

    if category == "short-completion" and text_chars < OPENAI_TINY_TEXT_CHARS_LT:
        return OPENAI_TINY_DEFAULT, "proposed-canary-default-off", "short-completion-to-tiny"
    if category in {"chat", "summary"} and text_chars < OPENAI_SMALL_TEXT_CHARS_LT:
        return OPENAI_SMALL_DEFAULT, "proposed-canary-default-off", "chat-summary-to-small"
    if category == "tool-light" and text_chars < OPENAI_SMALL_TEXT_CHARS_LT:
        return OPENAI_SMALL_DEFAULT, "proposed-canary-default-off", "tool-light-to-small-needs-tool-safety"
    return None, "none", "no-local-routing-shape-match"


def _target_supported(model: str | None) -> bool:
    if not model:
        return False
    return bool(pricing_basis(model, provider="openai").get("cost_known"))


def _projected_savings(row: dict[str, Any], requested_model: str, target_model: str, input_tokens: int, output_tokens: int) -> tuple[float, list[str]]:
    blockers: list[str] = []
    if input_tokens <= 0 and output_tokens <= 0:
        blockers.append("missing-baseline-cost")
        return 0.0, blockers

    target_cost = estimate_cost(target_model, input_tokens, output_tokens, provider="openai")
    requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider="openai")
    baseline_cost = _as_float(row.get("cost_baseline_usd"))
    if target_cost is None:
        blockers.append("unsupported-target-model")
        return 0.0, blockers
    if requested_cost is None and baseline_cost <= 0:
        blockers.append("missing-baseline-cost")
        return 0.0, blockers

    baseline = baseline_cost if baseline_cost > 0 else float(requested_cost or 0.0)
    return max(0.0, baseline - target_cost), blockers


def _canary_cohort(canary: dict[str, Any]) -> str:
    status = str(canary.get("status") or "").strip()
    cohort = str(canary.get("cohort") or "").strip()
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


def _empty_lifecycle() -> dict[str, Any]:
    return {
        "cohort_counts": {
            "canary_applied": 0,
            "canary_holdout": 0,
            "safety_stopped": 0,
            "skipped": 0,
            "bypassed_or_disabled": 0,
            "unknown": 0,
        },
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "reason_counts": {},
        "oldest_observed_at": None,
        "latest_observed_at": None,
    }


def _canary_matches_candidate(bucket: dict[str, Any], canary: dict[str, Any]) -> bool:
    requested = str(canary.get("requested_model") or canary.get("original_model") or "").strip().lower()
    target = str(canary.get("target_model") or "").strip().lower()
    if requested and requested != str(bucket.get("requested_model") or "").lower():
        return False
    if target and target != str(bucket.get("target_model") or "").lower():
        return False
    return True


def _add_canary_lifecycle(bucket: dict[str, Any], row: dict[str, Any], canary: dict[str, Any]) -> None:
    if not canary or not _canary_matches_candidate(bucket, canary):
        return
    lifecycle = bucket.setdefault("openai_canary_lifecycle", _empty_lifecycle())
    cohort = _canary_cohort(canary)
    lifecycle["cohort_counts"][cohort] = _as_int(lifecycle["cohort_counts"].get(cohort)) + 1
    status_code = _as_int(row.get("status_code"), -1)
    if status_code >= 400:
        lifecycle["error_count"] += 1
    if _as_int(row.get("retry_count")) > 0:
        lifecycle["retry_count"] += 1
    if canary.get("fallback_reason"):
        lifecycle["fallback_count"] += 1
    _increment(lifecycle["reason_counts"], canary.get("reason") or "unknown")

    created_at = row.get("created_at")
    if created_at:
        created = str(created_at)
        if lifecycle["latest_observed_at"] is None or created > str(lifecycle["latest_observed_at"]):
            lifecycle["latest_observed_at"] = created
        if lifecycle["oldest_observed_at"] is None or created < str(lifecycle["oldest_observed_at"]):
            lifecycle["oldest_observed_at"] = created


def _finalize_lifecycle(raw: dict[str, Any] | None, *, matched_count: int) -> dict[str, Any]:
    raw = raw or _empty_lifecycle()
    counts = {key: _as_int(value) for key, value in (raw.get("cohort_counts") or {}).items()}
    observed = sum(counts.values())
    applied = _as_int(counts.get("canary_applied"))
    holdout = _as_int(counts.get("canary_holdout"))
    safety_stopped = _as_int(counts.get("safety_stopped"))
    latest = _parse_time(raw.get("latest_observed_at"))
    age_hours = None
    stale = False
    if latest is not None:
        age_hours = round((datetime.now(timezone.utc) - latest).total_seconds() / 3600.0, 3)
        stale = age_hours > DEFAULT_MAX_EVIDENCE_AGE_HOURS

    blocker_counts: dict[str, int] = {}
    if observed == 0:
        blocker_counts["missing-canary-lifecycle-evidence"] = matched_count
    if applied == 0:
        blocker_counts["missing-applied-coverage"] = matched_count
    if holdout == 0:
        blocker_counts["missing-holdout-coverage"] = matched_count
    if _as_int(raw.get("error_count")):
        blocker_counts["error-observed"] = _as_int(raw.get("error_count"))
    if _as_int(raw.get("retry_count")):
        blocker_counts["retry-observed"] = _as_int(raw.get("retry_count"))
    if _as_int(raw.get("fallback_count")):
        blocker_counts["fallback-observed"] = _as_int(raw.get("fallback_count"))
    if safety_stopped:
        blocker_counts["safety-stop-observed"] = safety_stopped
    if stale:
        blocker_counts["stale-evidence"] = observed

    return {
        "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
        "status": "matched" if observed else "no-openai-canary-metadata",
        "observed_count": observed,
        "cohort_counts": {
            "canary_applied": applied,
            "canary_holdout": holdout,
            "safety_stopped": safety_stopped,
            "skipped": _as_int(counts.get("skipped")),
            "bypassed_or_disabled": _as_int(counts.get("bypassed_or_disabled")),
            "unknown": _as_int(counts.get("unknown")),
        },
        "coverage": {
            "matched_count": matched_count,
            "observed_rate": round(observed / matched_count, 6) if matched_count else 0.0,
            "applied_rate": round(applied / matched_count, 6) if matched_count else 0.0,
            "holdout_rate": round(holdout / matched_count, 6) if matched_count else 0.0,
        },
        "error_count": _as_int(raw.get("error_count")),
        "retry_count": _as_int(raw.get("retry_count")),
        "fallback_count": _as_int(raw.get("fallback_count")),
        "oldest_observed_at": raw.get("oldest_observed_at"),
        "latest_observed_at": raw.get("latest_observed_at"),
        "stale_evidence": {
            "stale": stale,
            "age_hours": age_hours,
            "max_age_hours": DEFAULT_MAX_EVIDENCE_AGE_HOURS,
        },
        "reason_breakdown": _breakdown(raw.get("reason_counts") or {}),
        "blocker_codes": sorted(blocker_counts),
        "blocker_reason_breakdown": _breakdown(blocker_counts),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
        },
    }


def _candidate_id(
    *,
    endpoint: str,
    requested_family: str,
    category: str,
    has_tools: bool,
    stream: bool,
    text_bucket: str,
    token_bucket: str,
    target_model: str,
) -> str:
    target = target_model.lower().replace(".", "-")
    family = requested_family.lower().replace(".", "-")
    tool_flag = "tools" if has_tools else "no-tools"
    stream_flag = "stream" if stream else "nonstream"
    return f"openai-route:{endpoint}:{family}:{category}:{tool_flag}:{stream_flag}:{text_bucket}:{token_bucket}:to-{target}"


def _new_bucket(row: dict[str, Any], *, candidate_id: str, target_model: str, target_policy: str, target_reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_surface": _source_surface(row),
        "endpoint": _endpoint(row),
        "requested_model_family": str(row.get("requested_model_family") or openai_model_family(row.get("requested_model")) or "unknown"),
        "requested_model": row.get("requested_model") or "unknown",
        "target_model": target_model,
        "simulated_policy": target_policy,
        "simulated_reason": target_reason,
        "category": row.get("category") or "unknown",
        "has_tools": False,
        "stream": False,
        "text_bucket": "unknown",
        "token_bucket": "unknown",
        "matched_count": 0,
        "blocked_count": 0,
        "current_routed_count": 0,
        "projected_savings_usd": 0.0,
        "estimated_baseline_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error_count": 0,
        "retry_count": 0,
        "latency_values": [],
        "status_counts": {},
        "cache_status_counts": {},
        "blocker_counts": {},
        "row_blocked_count": 0,
        "openai_canary_lifecycle": _empty_lifecycle(),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    matched = _as_int(bucket.get("matched_count"))
    latency_values = bucket.pop("latency_values", [])
    blocker_counts = bucket.pop("blocker_counts", {})
    row_blocked = _as_int(bucket.pop("row_blocked_count", 0))
    group_blockers: list[str] = []
    error_rate = (_as_int(bucket.get("error_count")) / matched) if matched else 0.0
    retry_rate = (_as_int(bucket.get("retry_count")) / matched) if matched else 0.0
    if matched < DEFAULT_MIN_SAMPLES:
        group_blockers.append("insufficient-samples")
        blocker_counts["insufficient-samples"] = matched
    if error_rate > DEFAULT_MAX_ERROR_RATE:
        group_blockers.append("high-recent-error-rate")
        blocker_counts["high-recent-error-rate"] = matched
    if retry_rate > DEFAULT_MAX_RETRY_RATE:
        group_blockers.append("high-recent-retry-rate")
        blocker_counts["high-recent-retry-rate"] = matched
    if _as_int(bucket.get("stream_count")) == matched and matched > 0:
        group_blockers.append("stream-only-evidence")
        blocker_counts["stream-only-evidence"] = matched

    blocked = matched if group_blockers else row_blocked
    avg_latency = round(sum(latency_values) / len(latency_values)) if latency_values else None
    suggested = 0.0
    if matched and blocked == 0:
        suggested = 0.05 if matched < 100 else 0.10

    bucket["projected_savings_usd"] = round(_as_float(bucket.get("projected_savings_usd")), 6)
    bucket["estimated_baseline_cost_usd"] = round(_as_float(bucket.get("estimated_baseline_cost_usd")), 6)
    bucket["estimated_savings_per_1000_calls_usd"] = round(
        (_as_float(bucket.get("projected_savings_usd")) / matched) * 1000.0,
        6,
    ) if matched else 0.0
    bucket["blocked_count"] = blocked
    bucket["error_rate"] = round(error_rate, 4) if matched else 0.0
    bucket["retry_rate"] = round(retry_rate, 4) if matched else 0.0
    bucket["avg_latency_ms"] = avg_latency
    bucket["blockers"] = sorted(set(group_blockers + list(blocker_counts.keys()))) if blocked else []
    bucket["blocker_reason_breakdown"] = _breakdown(blocker_counts)
    bucket["status_breakdown"] = _breakdown(bucket.pop("status_counts", {}))
    bucket["cache_status_breakdown"] = _breakdown(bucket.pop("cache_status_counts", {}))
    bucket["openai_canary_lifecycle_evidence"] = _finalize_lifecycle(
        bucket.pop("openai_canary_lifecycle", None),
        matched_count=matched,
    )
    bucket["suggested_canary_fraction"] = suggested
    bucket["privacy"] = {"metadata_only": True, "candidate_id_derived_from_raw_body": False}
    bucket.pop("stream_count", None)
    return bucket


def build_openai_routing_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    buckets: dict[str, dict[str, Any]] = {}
    blocker_totals: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    unmatched_reason_counts: dict[str, int] = {}
    openai_count = 0
    current_routed_total = 0

    for row in rows:
        if str(row.get("provider") or "").lower() != "openai":
            continue
        openai_count += 1
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
        routed_model = str(row.get("routed_model") or routing.get("routed_model") or requested_model)
        category = str(row.get("category") or routing.get("category") or "unknown")
        surface = _source_surface(row)
        endpoint = _endpoint(row)
        has_tools = _has_tools(row, routing, cache)
        stream = bool(_as_int(row.get("stream")))
        text_chars = _text_chars(row, routing)
        input_tokens = _input_tokens(row, text_chars)
        output_tokens = _output_tokens(row)
        text_bucket = _text_bucket(text_chars)
        token_bucket = _token_bucket(input_tokens)
        requested_family = str(row.get("requested_model_family") or openai_model_family(requested_model) or "unknown")
        cache_status = str(cache.get("status") or ("hit" if _as_int(row.get("cache_hit")) else "missing"))
        openai_canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
        current_routed = bool(requested_model and routed_model and requested_model != routed_model)
        if current_routed:
            current_routed_total += 1

        _increment(category_counts, category)
        _increment(surface_counts, surface)
        _increment(cache_status_counts, cache_status)

        target_model, policy, reason = _simulate_openai_route(
            requested_model=requested_model,
            category=category,
            has_tools=has_tools,
            text_chars=text_chars,
        )
        if not target_model:
            _increment(unmatched_reason_counts, reason)
            continue
        if target_model.lower() == requested_model.lower():
            _increment(unmatched_reason_counts, "target-same-as-requested")
            continue

        cid = _candidate_id(
            endpoint=endpoint,
            requested_family=requested_family,
            category=category,
            has_tools=has_tools,
            stream=stream,
            text_bucket=text_bucket,
            token_bucket=token_bucket,
            target_model=target_model,
        )
        bucket = buckets.setdefault(
            cid,
            _new_bucket(row, candidate_id=cid, target_model=target_model, target_policy=policy, target_reason=reason),
        )
        bucket["matched_count"] += 1
        bucket["has_tools"] = bool(bucket["has_tools"] or has_tools)
        bucket["stream"] = bool(bucket["stream"] or stream)
        bucket["text_bucket"] = text_bucket
        bucket["token_bucket"] = token_bucket
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["current_routed_count"] += int(current_routed)
        bucket["error_count"] += int(_as_int(row.get("status_code")) >= 400)
        bucket["retry_count"] += int(_as_int(row.get("retry_count")) > 0)
        _add_canary_lifecycle(bucket, row, openai_canary)
        if stream:
            bucket["stream_count"] = _as_int(bucket.get("stream_count")) + 1
        latency = _as_int(row.get("latency_ms"))
        if latency > 0:
            bucket["latency_values"].append(latency)
        _increment(bucket["status_counts"], row.get("status_code") or "unknown")
        _increment(bucket["cache_status_counts"], cache_status)

        row_blockers: list[str] = []
        if has_tools:
            row_blockers.append("tools-disabled")
        if requested_family in {"unknown", "other", "none"}:
            row_blockers.append("unknown-model-family")
        if not _target_supported(target_model):
            row_blockers.append("unsupported-target-model")

        savings, cost_blockers = _projected_savings(row, requested_model, target_model, input_tokens, output_tokens)
        requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider="openai")
        row_baseline = _as_float(row.get("cost_baseline_usd"))
        bucket["estimated_baseline_cost_usd"] += row_baseline if row_baseline > 0 else _as_float(requested_cost)
        row_blockers.extend(cost_blockers)
        if row_blockers:
            bucket["row_blocked_count"] += 1
            for blocker in sorted(set(row_blockers)):
                _increment(bucket["blocker_counts"], blocker)
        else:
            bucket["projected_savings_usd"] += savings

    candidates = [_finalize_bucket(bucket) for bucket in buckets.values()]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("matched_count")) - _as_int(item.get("blocked_count")),
            _as_int(item.get("matched_count")),
        ),
        reverse=True,
    )

    matched_count = sum(_as_int(item.get("matched_count")) for item in candidates)
    blocked_count = sum(_as_int(item.get("blocked_count")) for item in candidates)
    projected_savings = sum(_as_float(item.get("projected_savings_usd")) for item in candidates)
    estimated_baseline_cost = sum(_as_float(item.get("estimated_baseline_cost_usd")) for item in candidates)
    canary_applied_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("canary_applied")) for item in candidates)
    canary_holdout_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("canary_holdout")) for item in candidates)
    canary_safety_stopped_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("safety_stopped")) for item in candidates)
    canary_error_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("error_count")) for item in candidates)
    canary_retry_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("retry_count")) for item in candidates)
    canary_fallback_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("fallback_count")) for item in candidates)
    for item in candidates:
        for blocker in item.get("blocker_reason_breakdown") or []:
            _increment(blocker_totals, blocker["value"], _as_int(blocker.get("count")))

    eligible_candidates = [item for item in candidates if _as_int(item.get("blocked_count")) == 0]
    suggested_fraction = max((_as_float(item.get("suggested_canary_fraction")) for item in eligible_candidates), default=0.0)

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": openai_count,
            "candidate_count": len(candidates),
            "matched_count": matched_count,
            "current_routed_count": current_routed_total,
            "blocked_count": blocked_count,
            "eligible_count": max(0, matched_count - blocked_count),
            "projected_savings_usd": round(projected_savings, 6),
            "estimated_baseline_cost_usd": round(estimated_baseline_cost, 6),
            "suggested_canary_fraction": suggested_fraction,
            "openai_canary_applied_count": canary_applied_count,
            "openai_canary_holdout_count": canary_holdout_count,
            "openai_canary_safety_stopped_count": canary_safety_stopped_count,
            "openai_canary_error_count": canary_error_count,
            "openai_canary_retry_count": canary_retry_count,
            "openai_canary_fallback_count": canary_fallback_count,
        },
        "simulation_policy": {
            "schema": "agentflow.openai_routing_simulation_policy.v1",
            "provider_calls_made": False,
            "existing_route_openai_model_thresholds": {
                "large_model": OPENAI_LARGE_DEFAULT,
                "small_model": OPENAI_SMALL_DEFAULT,
                "tiny_model": OPENAI_TINY_DEFAULT,
                "small_text_chars_lt": OPENAI_SMALL_TEXT_CHARS_LT,
                "tiny_text_chars_lt": OPENAI_TINY_TEXT_CHARS_LT,
                "openai_routing_default_enabled": False,
            },
            "proposed_file_backed_canary_policy": {
                "policy_id": "local-openai-routing-canary-opportunity-v1",
                "status": "simulated-only",
                "default_enabled": False,
                "eligible_categories": ["chat", "short-completion"],
                "blocked_until_policy_support": ["tool-light"],
            },
            "quality_gate_policy": {
                "min_samples": DEFAULT_MIN_SAMPLES,
                "max_error_rate": DEFAULT_MAX_ERROR_RATE,
                "max_retry_rate": DEFAULT_MAX_RETRY_RATE,
            },
        },
        "category_breakdown": _breakdown(category_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "cache_status_breakdown": _breakdown(cache_status_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "unmatched_reason_breakdown": _breakdown(unmatched_reason_counts),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "basis": "local calls table metadata plus sanitized routing/cache decision summaries only",
        },
    }
