from __future__ import annotations

import os
import json
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_old_context_summary_opportunity.v1"
DEFAULT_MIN_REQUEST_CHARS = 32_000
DEFAULT_MIN_OLDER_CONTEXT_CHARS = 8_000
DEFAULT_MAX_SUMMARY_CHARS = 4_000
DEFAULT_SUMMARY_COMPRESSION_RATIO = 0.125
DEFAULT_SUMMARY_MODEL = "gpt-5-mini"

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
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    provider_ready = _summary_provider_configured() if summary_provider_configured is None else bool(summary_provider_configured)
    model = summary_model or _summary_model()
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
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

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    blocker_totals: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    model_family_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    openai_count = 0
    eligible_count = 0
    blocked_count = 0
    feature_rows = 0
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
        if feature:
            feature_rows += 1
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

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": openai_count,
            "feature_row_count": feature_rows,
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
        },
        "endpoint_breakdown": _breakdown(endpoint_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "category_breakdown": _breakdown(category_counts),
        "requested_model_family_breakdown": _breakdown(model_family_counts),
        "cache_status_breakdown": _breakdown(cache_status_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "groups": output_groups,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "session_ids_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local calls table metadata plus sanitized OpenAI feature/cache/routing summaries only",
        },
    }
