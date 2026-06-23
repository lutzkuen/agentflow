from __future__ import annotations

import json
from typing import Any

from tokenclaw.pricing import estimate_cost, provider_prompt_cache_accounting


SCHEMA = "tokenclaw.realized_savings_attribution.v1"


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _money(value: Any) -> float:
    return round(max(_as_float(value), 0.0), 8)


def json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _crunch_tokens_saved(crunch: dict[str, Any]) -> int:
    tokens = _as_int(crunch.get("tokens_saved_est") or crunch.get("saved_tokens"))
    summary = crunch.get("old_context_summarization")
    if isinstance(summary, dict) and summary.get("status") == "applied":
        tokens += _as_int(summary.get("tokens_saved_est") or summary.get("saved_tokens"))
    return max(tokens, 0)


def realized_savings_attribution(
    *,
    requested_model: Any,
    routed_model: Any,
    provider: Any = "anthropic",
    actual_input_tokens: Any = None,
    input_tokens_est: Any = None,
    actual_output_tokens: Any = None,
    output_tokens_est: Any = None,
    cache_creation_input_tokens: Any = 0,
    cache_read_input_tokens: Any = 0,
    cost_est_usd: Any = None,
    cost_baseline_usd: Any = None,
    crunch_json: Any = None,
    routing_json: Any = None,
    cache_json: Any = None,
    cache_hit: Any = 0,
) -> dict[str, Any]:
    provider_label = str(provider or "anthropic").lower()
    requested = str(requested_model or "")
    routed = str(routed_model or requested or "")
    target = routed or requested
    crunch = json_obj(crunch_json)
    routing = json_obj(routing_json)
    cache = json_obj(cache_json)

    input_tokens = _as_int(actual_input_tokens if actual_input_tokens is not None else input_tokens_est)
    output_tokens = _as_int(actual_output_tokens if actual_output_tokens is not None else output_tokens_est)
    cache_creation_tokens = max(_as_int(cache_creation_input_tokens), 0)
    cache_read_tokens = max(_as_int(cache_read_input_tokens), 0)
    actual_cost = _as_float(cost_est_usd)
    baseline_cost = _as_float(cost_baseline_usd)

    routing_savings = 0.0
    if requested and routed and requested != routed:
        requested_cost = estimate_cost(
            requested,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            provider=provider_label,
        )
        routed_cost = estimate_cost(
            routed,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            provider=provider_label,
        )
        if requested_cost is not None and routed_cost is not None:
            routing_savings = max(requested_cost - routed_cost, 0.0)

    prompt_cache = provider_prompt_cache_accounting(
        target,
        provider=provider_label,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    provider_prompt_cache_discount = _as_float(prompt_cache.get("read_discount_usd"))
    provider_prompt_cache_net_discount = _as_float(prompt_cache.get("net_provider_cache_discount_usd"))

    tokens_saved_est = _crunch_tokens_saved(crunch)
    provider_cache_overlap_tokens = min(tokens_saved_est, cache_read_tokens)
    realized_crunch_input_tokens_saved = max(tokens_saved_est - provider_cache_overlap_tokens, 0)
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    summary_cost = _as_float(summary.get("summary_cost_est_usd"))
    crunch_savings = 0.0
    if realized_crunch_input_tokens_saved > 0:
        before = estimate_cost(
            target,
            input_tokens + realized_crunch_input_tokens_saved,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            provider=provider_label,
        )
        after = estimate_cost(
            target,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            provider=provider_label,
        )
        if before is not None and after is not None:
            crunch_savings = max(before - after - summary_cost, 0.0)

    cache_savings = 0.0
    if _as_int(cache_hit):
        for key in ("realized_cache_savings_usd", "actual_saved_cost_usd", "estimated_saved_cost_usd"):
            if key in cache:
                cache_savings = max(_as_float(cache.get(key)), 0.0)
                break
        else:
            cache_savings = max(baseline_cost - actual_cost - routing_savings - crunch_savings, 0.0)

    tokenclaw_total = routing_savings + crunch_savings + cache_savings
    total_observed_delta = max(baseline_cost - actual_cost, 0.0)
    return {
        "schema": SCHEMA,
        "provider": provider_label,
        "requested_model": requested,
        "routed_model": routed,
        "token_basis": "provider-reported" if actual_input_tokens is not None or actual_output_tokens is not None else "estimated-from-request",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "crunch_tokens_saved_est": tokens_saved_est,
        "provider_cache_overlap_tokens": provider_cache_overlap_tokens,
        "realized_crunch_input_tokens_saved": realized_crunch_input_tokens_saved,
        "realized_routing_savings_usd": _money(routing_savings),
        "realized_crunch_savings_usd": _money(crunch_savings),
        "realized_cache_savings_usd": _money(cache_savings),
        "provider_prompt_cache_discount_usd": _money(provider_prompt_cache_discount),
        "provider_prompt_cache_net_discount_usd": round(provider_prompt_cache_net_discount, 8),
        "tokenclaw_realized_savings_usd": _money(tokenclaw_total),
        "total_observed_delta_usd": _money(total_observed_delta),
        "double_count_guard": {
            "provider_prompt_cache_discount_separate": True,
            "crunch_tokens_overlapping_provider_cache_read_not_attributed": provider_cache_overlap_tokens,
        },
        "policy_sources": sorted({
            str(source)
            for meta in (routing, crunch, cache)
            for source in (meta.get("policy_source"), meta.get("final_policy_source"))
            if source
        }),
    }


def attach_realized_savings_to_metadata(
    *,
    crunch_json: Any,
    routing_json: Any,
    cache_json: Any,
    attribution: dict[str, Any],
) -> tuple[Any, Any, Any]:
    crunch = json_obj(crunch_json)
    routing = json_obj(routing_json)
    cache = json_obj(cache_json)

    if crunch:
        crunch["realized_savings"] = {
            "schema": "tokenclaw.realized_crunch_savings.v1",
            "tokens_saved_est": attribution["crunch_tokens_saved_est"],
            "provider_cache_overlap_tokens": attribution["provider_cache_overlap_tokens"],
            "realized_input_tokens_saved": attribution["realized_crunch_input_tokens_saved"],
            "realized_crunch_savings_usd": attribution["realized_crunch_savings_usd"],
        }
        crunch["realized_crunch_savings_usd"] = attribution["realized_crunch_savings_usd"]

    if routing:
        routing["realized_routing_savings_usd"] = attribution["realized_routing_savings_usd"]

    if cache:
        cache["realized_cache_savings_usd"] = attribution["realized_cache_savings_usd"]

    summary = {
        "schema": SCHEMA,
        "realized_routing_savings_usd": attribution["realized_routing_savings_usd"],
        "realized_crunch_savings_usd": attribution["realized_crunch_savings_usd"],
        "realized_cache_savings_usd": attribution["realized_cache_savings_usd"],
        "provider_prompt_cache_discount_usd": attribution["provider_prompt_cache_discount_usd"],
        "tokenclaw_realized_savings_usd": attribution["tokenclaw_realized_savings_usd"],
        "provider_prompt_cache_discount_separate": True,
    }
    if crunch:
        crunch["realized_savings_attribution"] = summary
    if routing:
        routing["realized_savings_attribution"] = summary
    if cache:
        cache["realized_savings_attribution"] = summary

    return (
        json.dumps(crunch, sort_keys=True, separators=(",", ":"), ensure_ascii=False) if crunch else crunch_json,
        json.dumps(routing, sort_keys=True, separators=(",", ":"), ensure_ascii=False) if routing else routing_json,
        json.dumps(cache, sort_keys=True, separators=(",", ":"), ensure_ascii=False) if cache else cache_json,
    )
