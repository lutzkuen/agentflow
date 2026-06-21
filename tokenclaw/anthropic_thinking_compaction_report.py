from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.crunch import TOKEN_CHARS
from tokenclaw.pricing import estimate_blended_input_savings
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.anthropic_thinking_compaction_opportunity.v1"
_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


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
    value = str(key or "unknown")
    counter[value] = counter.get(value, 0) + amount


def _breakdown(counter: dict[str, int], *, label: str = "value") -> list[dict[str, Any]]:
    return [
        {label: key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 8_000:
        return "lt_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    if chars < 512_000:
        return "128k_512k_chars"
    return "gte_512k_chars"


def _token_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "0"
    if tokens < 1_000:
        return "lt_1k"
    if tokens < 8_000:
        return "1k_8k"
    if tokens < 32_000:
        return "8k_32k"
    if tokens < 128_000:
        return "32k_128k"
    return "gte_128k"


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2_5"
    if count <= 20:
        return "6_20"
    return "gt_20"


def _model_family(model: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    return "unknown"


def _source_surface(provider: str, path: str) -> str:
    if provider == "anthropic":
        return "anthropic_messages"
    return provider or (path.strip("/") or "unknown")


def _endpoint(provider: str, path: str) -> str:
    if provider == "anthropic":
        return "messages" if "messages" in path else (path.strip("/") or "unknown")
    return path.strip("/") or "unknown"


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


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"anthropic-thinking-compaction:{digest}"


def _message_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _thinking_block_text(block: dict[str, Any]) -> str:
    for key in ("thinking", "text", "data"):
        value = block.get(key)
        if isinstance(value, str):
            return value
    return ""


def _message_thinking_block_count(message: Any) -> int:
    return sum(1 for block in _message_blocks(message) if block.get("type") in _THINKING_BLOCK_TYPES)


def _has_tool_result(message: Any) -> bool:
    return any(block.get("type") == "tool_result" for block in _message_blocks(message))


def _body_thinking_metadata(raw_request_json: Any, *, max_fingerprinted_blocks: int = 200) -> dict[str, Any]:
    body = _json_obj(raw_request_json)
    if not body:
        return {
            "body_available": False,
            "top_level_thinking": False,
            "thinking_block_count": 0,
            "redacted_thinking_block_count": 0,
            "thinking_history_chars": 0,
            "unique_thinking_fingerprint_count": 0,
            "tool_result_after_thinking_count": 0,
            "fingerprint_limit_reached": False,
        }

    thinking_param = body.get("thinking")
    top_level_thinking = bool(thinking_param)
    if isinstance(thinking_param, dict) and str(thinking_param.get("type") or "").lower() == "disabled":
        top_level_thinking = False

    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    block_count = 0
    redacted_count = 0
    chars = 0
    fingerprints: set[str] = set()
    fingerprint_limit_reached = False
    tool_result_after_thinking_count = 0
    previous_thinking_blocks = 0

    for message in messages:
        if not isinstance(message, dict):
            previous_thinking_blocks = 0
            continue
        if message.get("role") == "user" and previous_thinking_blocks and _has_tool_result(message):
            tool_result_after_thinking_count += 1
        if message.get("role") != "assistant":
            previous_thinking_blocks = 0
            continue
        message_thinking_blocks = 0
        for block in _message_blocks(message):
            if block.get("type") not in _THINKING_BLOCK_TYPES:
                continue
            block_count += 1
            message_thinking_blocks += 1
            if block.get("type") == "redacted_thinking":
                redacted_count += 1
            text = _thinking_block_text(block)
            chars += len(text)
            if text and len(fingerprints) < max_fingerprinted_blocks:
                fingerprints.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
            elif text:
                fingerprint_limit_reached = True
        previous_thinking_blocks = message_thinking_blocks

    return {
        "body_available": True,
        "top_level_thinking": top_level_thinking,
        "thinking_block_count": block_count,
        "redacted_thinking_block_count": redacted_count,
        "thinking_history_chars": chars,
        "unique_thinking_fingerprint_count": len(fingerprints),
        "tool_result_after_thinking_count": tool_result_after_thinking_count,
        "fingerprint_limit_reached": fingerprint_limit_reached,
    }


def _new_group(basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(basis),
        **basis,
        "matched_count": 0,
        "metadata_candidate_count": 0,
        "body_verified_candidate_count": 0,
        "successful_count": 0,
        "error_count": 0,
        "plateau_pair_count": 0,
        "body_rows": 0,
        "metadata_only_rows": 0,
        "top_level_thinking_rows": 0,
        "thinking_history_block_rows": 0,
        "tool_result_after_thinking_rows": 0,
        "thinking_history_block_count": 0,
        "redacted_thinking_block_count": 0,
        "unique_thinking_fingerprint_count": 0,
        "estimated_input_chars": 0,
        "estimated_input_tokens": 0,
        "actual_input_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_cache_read_tokens": 0,
        "thinking_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "projected_saved_usd": 0.0,
        "status_counts": {},
        "blocker_counts": {},
    }


def _row_blockers(
    *,
    provider: str,
    category: str,
    thinking_signal: bool,
    body_meta: dict[str, Any],
    text_chars: int,
    min_text_chars: int,
    status_code: Any,
) -> list[str]:
    blockers: set[str] = set()
    if provider != "anthropic":
        blockers.add("non-anthropic-provider")
    if category != "tool-result":
        blockers.add("non-tool-result-category")
    if not thinking_signal:
        blockers.add("no-thinking-session-signal")
    if text_chars < min_text_chars:
        blockers.add("below-min-text-size")
    if _as_int(status_code) >= 400:
        blockers.add("error-response")
    if not body_meta["body_available"]:
        blockers.add("request-body-unavailable")
    else:
        if body_meta["top_level_thinking"]:
            blockers.add("active-top-level-thinking-request")
        if _as_int(body_meta["thinking_block_count"]) <= 0:
            blockers.add("body-thinking-blocks-missing")
        if _as_int(body_meta["tool_result_after_thinking_count"]) <= 0 and category == "tool-result":
            blockers.add("tool-result-thinking-continuation-unverified")
    return sorted(blockers) or ["ready-for-thinking-compaction-review"]


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    group["estimated_cost_usd"] = round(_as_float(group["estimated_cost_usd"]), 6)
    group["projected_saved_usd"] = round(_as_float(group["projected_saved_usd"]), 6)
    group["status_breakdown"] = _breakdown(group.pop("status_counts", {}), label="status")
    group["blocker_reason_breakdown"] = _breakdown(group.pop("blocker_counts", {}))
    group["blockers"] = [item["value"] for item in group["blocker_reason_breakdown"]]
    group["candidate_status"] = "candidate" if _as_int(group["metadata_candidate_count"]) else "blocked"
    group["privacy"] = {
        "metadata_only_output": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_tool_payloads_included": False,
        "raw_thinking_text_included": False,
        "thinking_block_fingerprints_included": False,
        "raw_responses_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
    }
    return group


def build_anthropic_thinking_compaction_opportunity_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_text_chars: int = 8_000,
    max_plateau_delta_ratio: float = 0.03,
    metadata_compaction_ratio: float = 0.20,
    body_compaction_ratio: float = 0.65,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    text_floor = max(1, int(min_text_chars or 8_000))
    delta_ratio = max(0.0, min(float(max_plateau_delta_ratio), 1.0))
    metadata_ratio = max(0.0, min(float(metadata_compaction_ratio), 0.95))
    body_ratio = max(0.0, min(float(body_compaction_ratio), 0.95))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select * from (
                select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                       source_surface, endpoint, requested_model, routed_model,
                       requested_model_family, routed_model_family, stream, status_code,
                       input_tokens_est, actual_input_tokens, actual_output_tokens,
                       cache_creation_input_tokens, cache_read_input_tokens,
                       thinking_output_tokens, cost_est_usd, cost_baseline_usd,
                       category, crunch_json, routing_json, cache_json, request_json, session_id
                from calls
                order by created_at desc
                limit ?
            ) recent_calls
            order by created_at asc
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    plateau_previous_by_session: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    privacy_mode_counts: dict[str, int] = {}
    text_bucket_counts: dict[str, int] = {}
    thinking_token_bucket_counts: dict[str, int] = {}
    blocker_totals: dict[str, int] = {}
    scanned_rows = 0
    matched_rows = 0
    metadata_candidate_rows = 0
    body_verified_candidate_rows = 0

    for row in rows:
        scanned_rows += 1
        provider = str(row.get("provider") or "anthropic").lower()
        path = str(row.get("path") or "")
        routing = _json_obj(row.get("routing_json"))
        category = str(row.get("category") or routing.get("category") or "unknown")
        reason = str(routing.get("reason") or "unknown")
        requested_model = row.get("requested_model")
        routed_model = row.get("routed_model") or requested_model
        source_surface = str(row.get("source_surface") or routing.get("source_surface") or _source_surface(provider, path))
        endpoint = str(row.get("endpoint") or routing.get("endpoint") or _endpoint(provider, path))
        actual_input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        cache_creation_tokens = _as_int(row.get("cache_creation_input_tokens"))
        cache_read_tokens = _as_int(row.get("cache_read_input_tokens"))
        thinking_tokens = _as_int(row.get("thinking_output_tokens"))
        text_chars = _as_int(routing.get("text_chars")) or (actual_input_tokens + cache_creation_tokens + cache_read_tokens) * TOKEN_CHARS
        input_tokens = max(0, text_chars // TOKEN_CHARS)
        text_bucket = _text_bucket(text_chars)
        body_meta = _body_thinking_metadata(row.get("request_json"))
        privacy_mode = "local-body-derived-metadata" if body_meta["body_available"] else "metadata-only"
        thinking_signal = (
            provider == "anthropic"
            and (
                "thinking" in reason
                or _as_int(body_meta["thinking_block_count"]) > 0
                or thinking_tokens > 0
            )
        )
        session_id = row.get("session_id")
        plateau_pair = False
        if session_id:
            previous_chars = plateau_previous_by_session.get(str(session_id))
            plateau_pair = bool(
                previous_chars
                and previous_chars >= text_floor
                and text_chars >= text_floor
                and abs(text_chars - previous_chars) / max(previous_chars, 1) <= delta_ratio
            )
            plateau_previous_by_session[str(session_id)] = text_chars

        metadata_candidate = bool(
            provider == "anthropic"
            and category == "tool-result"
            and thinking_signal
            and text_chars >= text_floor
            and _as_int(row.get("status_code")) < 400
        )
        body_verified_candidate = bool(
            metadata_candidate
            and body_meta["body_available"]
            and _as_int(body_meta["thinking_block_count"]) > 0
            and _as_int(body_meta["tool_result_after_thinking_count"]) > 0
            and not body_meta["top_level_thinking"]
        )
        if metadata_candidate:
            metadata_candidate_rows += 1
        if body_verified_candidate:
            body_verified_candidate_rows += 1

        projected_saved_chars = 0
        if metadata_candidate:
            if _as_int(body_meta["thinking_history_chars"]) > 0:
                projected_saved_chars = int(_as_int(body_meta["thinking_history_chars"]) * body_ratio)
            else:
                projected_saved_chars = int(text_chars * metadata_ratio)
        projected_saved_tokens = max(0, projected_saved_chars // TOKEN_CHARS)
        projected_saved_usd = estimate_blended_input_savings(
            str(routed_model or requested_model or ""),
            tokens_saved=projected_saved_tokens,
            input_tokens=actual_input_tokens,
            cache_read_tokens=cache_read_tokens,
            provider=provider,
        ) or 0.0

        basis = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": str(routing.get("workflow_phase") or routing.get("phase") or category or "unknown"),
            "routing_reason": reason,
            "requested_model_family": str(row.get("requested_model_family") or _model_family(requested_model)),
            "routed_model_family": str(row.get("routed_model_family") or _model_family(routed_model)),
            "stream": bool(_as_int(row.get("stream"))),
            "text_bucket": text_bucket,
            "thinking_output_token_bucket": _token_bucket(thinking_tokens),
            "prompt_cache_read_token_bucket": _token_bucket(cache_read_tokens),
            "thinking_history_block_count_bucket": _count_bucket(_as_int(body_meta["thinking_block_count"])),
            "plateau_status": "adjacent-plateau" if plateau_pair else "not-adjacent-plateau",
            "privacy_mode": privacy_mode,
        }

        _increment(provider_counts, provider)
        _increment(category_counts, category)
        _increment(reason_counts, reason)
        _increment(privacy_mode_counts, privacy_mode)
        _increment(text_bucket_counts, text_bucket)
        _increment(thinking_token_bucket_counts, _token_bucket(thinking_tokens))

        include = provider == "anthropic" and (thinking_signal or category in {"tool-result", "tool-heavy"})
        if not include:
            continue
        matched_rows += 1
        group_key = _candidate_id(basis)
        group = groups.setdefault(group_key, _new_group(basis))
        group["matched_count"] += 1
        group["metadata_candidate_count"] += int(metadata_candidate)
        group["body_verified_candidate_count"] += int(body_verified_candidate)
        group["successful_count"] += int(_as_int(row.get("status_code")) < 400)
        group["error_count"] += int(_as_int(row.get("status_code")) >= 400)
        group["plateau_pair_count"] += int(plateau_pair)
        group["body_rows"] += int(body_meta["body_available"])
        group["metadata_only_rows"] += int(not body_meta["body_available"])
        group["top_level_thinking_rows"] += int(body_meta["top_level_thinking"])
        group["thinking_history_block_rows"] += int(_as_int(body_meta["thinking_block_count"]) > 0)
        group["tool_result_after_thinking_rows"] += int(_as_int(body_meta["tool_result_after_thinking_count"]) > 0)
        group["thinking_history_block_count"] += _as_int(body_meta["thinking_block_count"])
        group["redacted_thinking_block_count"] += _as_int(body_meta["redacted_thinking_block_count"])
        group["unique_thinking_fingerprint_count"] += _as_int(body_meta["unique_thinking_fingerprint_count"])
        group["estimated_input_chars"] += text_chars
        group["estimated_input_tokens"] += input_tokens
        group["actual_input_tokens"] += actual_input_tokens
        group["prompt_cache_creation_tokens"] += cache_creation_tokens
        group["prompt_cache_read_tokens"] += cache_read_tokens
        group["thinking_output_tokens"] += thinking_tokens
        group["estimated_cost_usd"] += _as_float(row.get("cost_est_usd")) or _as_float(row.get("cost_baseline_usd"))
        group["projected_saved_chars"] += projected_saved_chars
        group["projected_saved_tokens"] += projected_saved_tokens
        group["projected_saved_usd"] += projected_saved_usd
        _increment(group["status_counts"], _status_bucket(row.get("status_code")))
        blockers = _row_blockers(
            provider=provider,
            category=category,
            thinking_signal=thinking_signal,
            body_meta=body_meta,
            text_chars=text_chars,
            min_text_chars=text_floor,
            status_code=row.get("status_code"),
        )
        for blocker in blockers:
            _increment(group["blocker_counts"], blocker)
            _increment(blocker_totals, blocker)

    candidates = [_finalize_group(group) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            _as_int(item.get("metadata_candidate_count")),
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("projected_saved_tokens")),
            _as_int(item.get("matched_count")),
        ),
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "scanned_call_count": scanned_rows,
            "matched_count": matched_rows,
            "candidate_count": sum(1 for item in candidates if _as_int(item.get("metadata_candidate_count")) > 0),
            "metadata_candidate_count": metadata_candidate_rows,
            "body_verified_candidate_count": body_verified_candidate_rows,
            "plateau_pair_count": sum(_as_int(item.get("plateau_pair_count")) for item in candidates),
            "body_rows": sum(_as_int(item.get("body_rows")) for item in candidates),
            "metadata_only_rows": sum(_as_int(item.get("metadata_only_rows")) for item in candidates),
            "thinking_history_block_count": sum(_as_int(item.get("thinking_history_block_count")) for item in candidates),
            "thinking_output_tokens": sum(_as_int(item.get("thinking_output_tokens")) for item in candidates),
            "prompt_cache_creation_tokens": sum(_as_int(item.get("prompt_cache_creation_tokens")) for item in candidates),
            "prompt_cache_read_tokens": sum(_as_int(item.get("prompt_cache_read_tokens")) for item in candidates),
            "estimated_cost_usd": round(sum(_as_float(item.get("estimated_cost_usd")) for item in candidates), 6),
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in candidates),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in candidates), 6),
        },
        "projection_policy": {
            "schema": "tokenclaw.anthropic_thinking_compaction_projection_policy.v1",
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "default_apply": False,
            "raw_body_required": False,
            "min_text_chars": text_floor,
            "max_plateau_delta_ratio": delta_ratio,
            "metadata_compaction_ratio": metadata_ratio,
            "body_compaction_ratio": body_ratio,
            "savings_pricing": "cache-read blended input price for the routed/requested Anthropic model",
            "method": "rank local Anthropic thinking-session cohorts from calls metadata; optional local request bodies are scanned only for bounded thinking block counts and aggregate chars",
        },
        "provider_breakdown": _breakdown(provider_counts),
        "category_breakdown": _breakdown(category_counts),
        "routing_reason_breakdown": _breakdown(reason_counts, label="reason"),
        "privacy_mode_breakdown": _breakdown(privacy_mode_counts, label="mode"),
        "text_bucket_breakdown": _breakdown(text_bucket_counts, label="bucket"),
        "thinking_output_token_bucket_breakdown": _breakdown(thinking_token_bucket_counts, label="bucket"),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            "metadata_only_output": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_tool_payloads_included": False,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprints_included": False,
            "raw_responses_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local call metadata by default; local body logging, when present, contributes only aggregate thinking block metadata and never emitted text or identifiers",
        },
    }
