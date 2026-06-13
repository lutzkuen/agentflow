from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from agentflow_proxy.public_metadata import public_label
from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.request_shape_rollups.v1"
ROLLUP_ROW_SCHEMA = "agentflow.request_shape_rollup_row.v1"


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
    label = public_label(key, "unknown")
    counter[label] = counter.get(label, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _provider_family(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").strip().lower()
    if provider:
        return public_label(provider, "unknown")
    path = str(row.get("path") or "")
    if "responses" in path or "chat/completions" in path:
        return "openai"
    if "messages" in path:
        return "anthropic"
    return "unknown"


def _endpoint(row: dict[str, Any]) -> str:
    endpoint = row.get("endpoint")
    if endpoint:
        return public_label(endpoint, "unknown")
    path = str(row.get("path") or "")
    if "chat/completions" in path:
        return "chat_completions"
    if "responses" in path:
        return "responses"
    if "messages" in path:
        return "messages"
    return "unknown"


def _source_surface(row: dict[str, Any], provider: str, endpoint: str) -> str:
    source = row.get("source_surface")
    if source:
        return public_label(source, "unknown")
    if provider == "openai":
        return f"openai_{endpoint}"
    if provider == "anthropic":
        return "anthropic_messages"
    return "unknown"


def _model_family(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    if "claude" in text:
        for family in ("haiku", "sonnet", "opus"):
            if family in text:
                return f"claude-{family}"
        return "claude"
    if text.startswith("gpt-5"):
        return "gpt-5"
    if text.startswith("gpt-4"):
        return "gpt-4"
    return public_label(text, fallback)


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
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
    if tokens < 500:
        return "lt_500_tokens"
    if tokens < 2_000:
        return "500_2k_tokens"
    if tokens < 8_000:
        return "2k_8k_tokens"
    if tokens < 32_000:
        return "8k_32k_tokens"
    return "gte_32k_tokens"


def _cost_bucket(cost: float) -> str:
    if cost <= 0:
        return "unknown"
    if cost < 0.001:
        return "lt_0_001_usd"
    if cost < 0.01:
        return "0_001_0_01_usd"
    if cost < 0.05:
        return "0_01_0_05_usd"
    if cost < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _savings_bucket(savings: float) -> str:
    if savings <= 0:
        return "none"
    if savings < 0.001:
        return "lt_0_001_usd"
    if savings < 0.01:
        return "0_001_0_01_usd"
    if savings < 0.05:
        return "0_01_0_05_usd"
    if savings < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


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


def _retry_bucket(retries: int) -> str:
    if retries <= 0:
        return "0"
    if retries == 1:
        return "1"
    if retries <= 3:
        return "2_3"
    return "4_plus"


def _cache_status(row: dict[str, Any], cache: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    if status:
        return public_label(status, "unknown")
    return "hit" if _as_int(row.get("cache_hit")) else "missing"


def _routing_status(row: dict[str, Any], routing: dict[str, Any]) -> str:
    requested = str(row.get("requested_model") or routing.get("requested_model") or "")
    routed = str(row.get("routed_model") or routing.get("routed_model") or requested)
    if requested and routed and requested != routed:
        return "routed"
    if routing.get("enabled") is False:
        return "disabled"
    return "passthrough"


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any]) -> bool:
    if routing.get("has_tools") is not None:
        return bool(routing.get("has_tools"))
    tool_features = routing.get("tool_features") if isinstance(routing.get("tool_features"), dict) else {}
    if tool_features.get("has_tools") is not None:
        return bool(tool_features.get("has_tools"))
    if cache.get("has_tools") is not None:
        return bool(cache.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    reason = str(cache.get("reason") or "").lower()
    return category.startswith("tool") or "tool" in reason


def _workflow_phase(row: dict[str, Any], routing: dict[str, Any]) -> str:
    for key in ("workflow_phase", "phase", "category"):
        value = routing.get(key)
        if value:
            return public_label(value, "unknown")
    return public_label(row.get("category"), "unknown")


def _blocker_codes(
    *,
    row: dict[str, Any],
    cache: dict[str, Any],
    routing: dict[str, Any],
    cache_status: str,
    routing_status: str,
    stream: bool,
    has_tools: bool,
) -> list[str]:
    blockers: set[str] = set()
    reason = str(cache.get("reason") or "").lower()
    routing_reason = str(routing.get("reason") or "").lower()
    status_bucket = _status_bucket(row.get("status_code"))
    if stream:
        blockers.add("unsupported-streaming-shape")
    if has_tools and ("tools-disabled" in reason or "tool" in reason and "disabled" in reason):
        blockers.add("tool-call-cache-disabled")
    if "semantic" in reason and "disabled" in reason:
        blockers.add("semantic-cache-disabled")
    if cache_status in {"miss", "missing"} or "exact-miss" in reason:
        blockers.add("exact-cache-miss")
    if cache_status == "skipped" and not blockers:
        blockers.add("cache-skipped")
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    if "thinking" in routing_reason:
        blockers.add("thinking-routing-guard")
    if "rate" in routing_reason or status_bucket in {"4xx", "5xx"} and _as_int(row.get("retry_count")) > 0:
        blockers.add("rate-or-error-pressure")
    if routing_status == "passthrough" and not blockers:
        blockers.add("routing-rule-required")
    return sorted(public_label(code, "unknown") for code in blockers if code)


def _candidate_families(
    *,
    cache_status: str,
    routing_status: str,
    blockers: list[str],
    observed_savings: float,
    cost: float,
) -> list[str]:
    families: set[str] = set()
    if cache_status != "hit":
        families.add("cache_replay")
    if any(
        code.startswith("exact-cache")
        or code.startswith("cache-")
        or code in {"unsupported-streaming-shape", "tool-call-cache-disabled", "semantic-cache-disabled"}
        for code in blockers
    ):
        families.add("cache_blocker")
    if routing_status == "passthrough" and cost > 0:
        families.add("routing_candidate")
    if routing_status == "routed" or observed_savings > 0:
        families.add("routing_evidence")
    return sorted(families or {"observability"})


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = stable_json(basis)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    provider = str(basis.get("provider_family") or "unknown").replace("_", "-")
    endpoint = str(basis.get("endpoint") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"request-shape:{provider}:{endpoint}:{category}:{digest}"


def _new_group(basis: dict[str, Any], *, candidate_id: str, rollup_key: str) -> dict[str, Any]:
    return {
        "schema": ROLLUP_ROW_SCHEMA,
        "rollup_key": rollup_key,
        "candidate_id": candidate_id,
        **basis,
        "row_count": 0,
        "error_count": 0,
        "retry_count": 0,
        "cache_hit_count": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "observed_savings_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "candidate_family_counts": {},
        "blocker_counts": {},
        "status_counts": {},
        "retry_bucket_counts": {},
        "cost_bucket_counts": {},
        "savings_bucket_counts": {},
        "cache_reason_counts": {},
    }


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    candidate_family_counts = group.pop("candidate_family_counts", {})
    blocker_counts = group.pop("blocker_counts", {})
    metadata = {
        "schema": "agentflow.request_shape_rollup_metadata.v1",
        "status_breakdown": _breakdown(group.pop("status_counts", {})),
        "retry_bucket_breakdown": _breakdown(group.pop("retry_bucket_counts", {})),
        "cost_bucket_breakdown": _breakdown(group.pop("cost_bucket_counts", {})),
        "savings_bucket_breakdown": _breakdown(group.pop("savings_bucket_counts", {})),
        "cache_reason_breakdown": _breakdown(group.pop("cache_reason_counts", {})),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_breakdown": _breakdown(blocker_counts),
        "raw_body_required": False,
        "aggregate_only": True,
    }
    candidate_families = sorted(candidate_family_counts)
    blocker_codes = sorted(blocker_counts)
    group["candidate_families"] = candidate_families
    group["blocker_codes"] = blocker_codes
    group["cost_est_usd"] = round(_as_float(group.get("cost_est_usd")), 6)
    group["baseline_cost_usd"] = round(_as_float(group.get("baseline_cost_usd")), 6)
    group["observed_savings_usd"] = round(_as_float(group.get("observed_savings_usd")), 6)
    group["metadata"] = metadata
    group["privacy"] = {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_responses_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "cache_keys_included": False,
        "request_fingerprints_included": False,
    }
    return group


def _persistable_row(
    *,
    run_id: str,
    generated_at: str,
    window_start: str | None,
    window_end: str | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{run_id}:{row['rollup_key']}",
        "run_id": run_id,
        "generated_at": generated_at,
        "window_start": window_start,
        "window_end": window_end,
        "rollup_key": row["rollup_key"],
        "candidate_id": row["candidate_id"],
        "source_surface": row["source_surface"],
        "endpoint": row["endpoint"],
        "provider_family": row["provider_family"],
        "requested_model_family": row["requested_model_family"],
        "routed_model_family": row["routed_model_family"],
        "category": row["category"],
        "workflow_phase": row["workflow_phase"],
        "stream": 1 if row["stream"] else 0,
        "has_tools": 1 if row["has_tools"] else 0,
        "text_bucket": row["text_bucket"],
        "token_bucket": row["token_bucket"],
        "cache_status": row["cache_status"],
        "routing_status": row["routing_status"],
        "candidate_families_json": stable_json(row["candidate_families"]),
        "blocker_codes_json": stable_json(row["blocker_codes"]),
        "row_count": row["row_count"],
        "error_count": row["error_count"],
        "retry_count": row["retry_count"],
        "cache_hit_count": row["cache_hit_count"],
        "cost_est_usd": row["cost_est_usd"],
        "baseline_cost_usd": row["baseline_cost_usd"],
        "observed_savings_usd": row["observed_savings_usd"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "metadata_json": stable_json(row["metadata"]),
    }


def build_request_shape_rollups_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    persist: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    generated_at = utc_now()
    run_id = run_id or f"shape-rollups-{uuid4().hex[:12]}"
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, crunch_json,
                   routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    candidate_family_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    body_rows_read = 0
    window_start: str | None = None
    window_end: str | None = None

    for row in rows:
        created_at = str(row.get("created_at") or "")
        if created_at:
            window_start = created_at if window_start is None else min(window_start, created_at)
            window_end = created_at if window_end is None else max(window_end, created_at)
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        provider = _provider_family(row)
        endpoint = _endpoint(row)
        source_surface = _source_surface(row, provider, endpoint)
        requested_family = public_label(row.get("requested_model_family"), "") or _model_family(row.get("requested_model"))
        routed_family = public_label(row.get("routed_model_family"), "") or _model_family(
            row.get("routed_model"),
            requested_family,
        )
        category = public_label(row.get("category") or routing.get("category"), "unknown")
        workflow_phase = _workflow_phase(row, routing)
        stream = bool(_as_int(row.get("stream")))
        has_tools = _has_tools(row, routing, cache)
        text_chars = _as_int(routing.get("text_chars"))
        input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        if text_chars <= 0 and input_tokens > 0:
            text_chars = input_tokens * 4
        output_tokens = _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))
        cost = _as_float(row.get("cost_est_usd"))
        baseline = _as_float(row.get("cost_baseline_usd"))
        observed_savings = max(0.0, baseline - cost)
        cache_status = _cache_status(row, cache)
        cache_reason = public_label(cache.get("reason"), "unknown")
        routing_status = _routing_status(row, routing)
        blockers = _blocker_codes(
            row=row,
            cache=cache,
            routing=routing,
            cache_status=cache_status,
            routing_status=routing_status,
            stream=stream,
            has_tools=has_tools,
        )
        candidate_families = _candidate_families(
            cache_status=cache_status,
            routing_status=routing_status,
            blockers=blockers,
            observed_savings=observed_savings,
            cost=cost,
        )
        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "provider_family": provider,
            "requested_model_family": requested_family,
            "routed_model_family": routed_family,
            "category": category,
            "workflow_phase": workflow_phase,
            "stream": stream,
            "has_tools": has_tools,
            "text_bucket": _text_bucket(text_chars),
            "token_bucket": _token_bucket(input_tokens),
            "cache_status": cache_status,
            "routing_status": routing_status,
        }
        rollup_key = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
        candidate_id = _candidate_id(basis)
        group = groups.setdefault(rollup_key, _new_group(basis, candidate_id=candidate_id, rollup_key=rollup_key))
        group["row_count"] += 1
        group["error_count"] += int(_status_bucket(row.get("status_code")) in {"4xx", "5xx"})
        group["retry_count"] += _as_int(row.get("retry_count"))
        group["cache_hit_count"] += int(_as_int(row.get("cache_hit")) > 0 or cache_status == "hit")
        group["cost_est_usd"] += cost
        group["baseline_cost_usd"] += baseline
        group["observed_savings_usd"] += observed_savings
        group["input_tokens"] += input_tokens
        group["output_tokens"] += output_tokens
        _increment(provider_counts, provider)
        _increment(group["status_counts"], _status_bucket(row.get("status_code")))
        _increment(group["retry_bucket_counts"], _retry_bucket(_as_int(row.get("retry_count"))))
        _increment(group["cost_bucket_counts"], _cost_bucket(cost))
        _increment(group["savings_bucket_counts"], _savings_bucket(observed_savings))
        _increment(group["cache_reason_counts"], cache_reason)
        for family in candidate_families:
            _increment(candidate_family_counts, family)
            _increment(group["candidate_family_counts"], family)
        for blocker in blockers:
            _increment(blocker_counts, blocker)
            _increment(group["blocker_counts"], blocker)

    rollups = [_finalize_group(group) for group in groups.values()]
    rollups.sort(
        key=lambda item: (
            _as_float(item.get("observed_savings_usd")),
            _as_float(item.get("cost_est_usd")),
            _as_int(item.get("row_count")),
            item.get("candidate_id") or "",
        ),
        reverse=True,
    )
    persistable = [
        _persistable_row(
            run_id=run_id,
            generated_at=generated_at,
            window_start=window_start,
            window_end=window_end,
            row=row,
        )
        for row in rollups
    ]
    persisted_count = 0
    if persist and hasattr(store_obj, "persist_request_shape_rollups"):
        persisted_count = store_obj.persist_request_shape_rollups(
            run_id=run_id,
            generated_at=generated_at,
            rows=persistable,
        )

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "run_id": run_id,
        "limit": capped_limit,
        "persisted": bool(persisted_count),
        "persisted_count": persisted_count,
        "window": {
            "start": window_start,
            "end": window_end,
            "source": "recent-local-call-metadata",
        },
        "summary": {
            "rows_considered": len(rows),
            "rollup_count": len(rollups),
            "collapsed_rows": max(0, len(rows) - len(rollups)),
            "total_cost_est_usd": round(sum(_as_float(row.get("cost_est_usd")) for row in rollups), 6),
            "total_baseline_cost_usd": round(sum(_as_float(row.get("baseline_cost_usd")) for row in rollups), 6),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in rollups), 6),
            "body_rows_read": body_rows_read,
        },
        "provider_breakdown": _breakdown(provider_counts),
        "candidate_family_breakdown": _breakdown(candidate_family_counts),
        "blocker_code_breakdown": _breakdown(blocker_counts),
        "rollups": rollups,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "tenant_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
