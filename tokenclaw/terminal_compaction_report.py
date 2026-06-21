from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.pricing import estimate_cost
from tokenclaw.store import utc_now
from tokenclaw.terminal_features import terminal_log_features_from_text


SCHEMA = "tokenclaw.terminal_output_compaction_opportunity.v1"
TOKEN_CHARS = 4


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


def _hash_public_basis(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 8_000:
        return "lt_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _source_surface(provider: str, path: str) -> str:
    if provider == "anthropic":
        return "anthropic_messages"
    if provider == "openai":
        if "chat/completions" in path:
            return "openai_chat_completions"
        if "responses" in path:
            return "openai_responses"
        return "openai"
    return "unknown"


def _endpoint(provider: str, path: str) -> str:
    if provider == "anthropic":
        return "messages" if "messages" in path else (path.strip("/") or "unknown")
    if provider == "openai":
        if "chat/completions" in path:
            return "chat_completions"
        if "responses" in path:
            return "responses"
    return path.strip("/") or "unknown"


def _model_family(model: Any, provider: str) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    if provider == "openai":
        if "mini" in text:
            return "mini"
        if text:
            return text.split("-", 2)[0]
    return "unknown"


def _extract_tool_result_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_tool_result_text(item))
        return parts
    if not isinstance(value, dict):
        return []
    block_type = str(value.get("type") or "")
    if block_type == "tool_result":
        return _extract_tool_result_text(value.get("content"))
    if block_type in {"text", "input_text"}:
        text = value.get("text")
        return [text] if isinstance(text, str) else []
    if not block_type:
        parts: list[str] = []
        for key in ("text", "content"):
            if key in value:
                parts.extend(_extract_tool_result_text(value.get(key)))
        return parts
    return []


def _tool_result_texts(raw_request_json: Any) -> list[str]:
    body = _json_obj(raw_request_json)
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    texts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            texts.extend(_extract_tool_result_text(message.get("content")))
    return [text for text in texts if isinstance(text, str) and text.strip()]


def _terminal_features_from_routing(routing: dict[str, Any], crunch: dict[str, Any]) -> dict[str, Any]:
    for root in (routing, crunch):
        for key in ("terminal_log_features", "terminal_features"):
            value = root.get(key)
            if isinstance(value, dict):
                return value
        for value in root.values():
            if isinstance(value, dict):
                nested = value.get("terminal_log_features") or value.get("terminal_features")
                if isinstance(nested, dict):
                    return nested
    return {}


def _fraction_midpoint(bucket: Any) -> float:
    mapping = {
        "none": 0.0,
        "lt_10pct": 0.05,
        "10_25pct": 0.175,
        "25_50pct": 0.375,
        "50_75pct": 0.625,
        "gte_75pct": 0.85,
    }
    return mapping.get(str(bucket or "none"), 0.0)


def _plateau_key(session_id: Any) -> str:
    if not session_id:
        return "missing-session"
    return "session:" + hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()


def _new_plateau_state() -> dict[str, Any]:
    return {"previous_chars": None, "pairs": 0}


def _candidate_id(basis: dict[str, Any]) -> str:
    provider = str(basis.get("provider") or "unknown")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"terminal-output-compaction:{provider}:{category}:{_hash_public_basis(basis)}"


def _new_group(basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(basis),
        **basis,
        "matched_count": 0,
        "successful_count": 0,
        "error_count": 0,
        "plateau_pair_count": 0,
        "body_rows": 0,
        "metadata_only_rows": 0,
        "terminal_signal_rows": 0,
        "estimated_input_chars": 0,
        "estimated_input_tokens": 0,
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
    plateau_pair: bool,
    terminal_fraction: float,
    body_available: bool,
    projected_saved_chars: int,
    status_code: Any,
) -> list[str]:
    blockers: set[str] = set()
    if provider != "anthropic":
        blockers.add("non-anthropic-provider")
    if category != "tool-result":
        blockers.add("non-tool-result-category")
    if not plateau_pair:
        blockers.add("not-adjacent-context-plateau")
    if terminal_fraction <= 0:
        blockers.add("terminal-output-signal-missing")
    if not body_available:
        blockers.add("request-body-unavailable")
    if projected_saved_chars <= 0:
        blockers.add("no-compaction-savings-projected")
    if _as_int(status_code) >= 400:
        blockers.add("error-response")
    return sorted(blockers) or ["ready-for-dry-run-review"]


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    group["estimated_cost_usd"] = round(_as_float(group["estimated_cost_usd"]), 6)
    group["projected_saved_usd"] = round(_as_float(group["projected_saved_usd"]), 6)
    group["status_breakdown"] = _breakdown(group.pop("status_counts", {}))
    group["blocker_reason_breakdown"] = _breakdown(group.pop("blocker_counts", {}))
    group["blockers"] = [item["value"] for item in group["blocker_reason_breakdown"]]
    group["privacy"] = {
        "metadata_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_tool_payloads_included": False,
        "raw_terminal_text_included": False,
        "raw_responses_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "terminal_feature_hashes_included": False,
    }
    return group


def build_terminal_output_compaction_opportunity_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_text_chars: int = 8_000,
    max_plateau_delta_ratio: float = 0.03,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    text_floor = max(1, int(min_text_chars or 8_000))
    delta_ratio = max(0.0, min(float(max_plateau_delta_ratio), 1.0))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select * from (
                select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                       source_surface, endpoint, requested_model, routed_model,
                       requested_model_family, routed_model_family, stream, status_code,
                       input_tokens_est, actual_input_tokens, cost_est_usd, cost_baseline_usd,
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
    plateau_states: dict[str, dict[str, Any]] = {}
    provider_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    session_bucket_counts: dict[str, int] = {}
    blocker_totals: dict[str, int] = {}
    scanned_rows = 0
    matched_rows = 0

    for row in rows:
        scanned_rows += 1
        provider = str(row.get("provider") or "anthropic").lower()
        path = str(row.get("path") or "")
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        source_surface = str(row.get("source_surface") or routing.get("source_surface") or _source_surface(provider, path))
        endpoint = str(row.get("endpoint") or routing.get("endpoint") or _endpoint(provider, path))
        category = str(row.get("category") or routing.get("category") or "unknown")
        requested_model = row.get("requested_model")
        routed_model = row.get("routed_model") or requested_model
        observed_input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        text_chars = _as_int(routing.get("text_chars")) or observed_input_tokens * TOKEN_CHARS
        input_tokens = max(observed_input_tokens, max(0, text_chars // TOKEN_CHARS))
        text_bucket = _text_bucket(text_chars)
        plateau_state = plateau_states.setdefault(_plateau_key(row.get("session_id")), _new_plateau_state())
        previous_chars = plateau_state.get("previous_chars")
        plateau_pair = bool(
            previous_chars
            and previous_chars >= text_floor
            and text_chars >= text_floor
            and abs(text_chars - previous_chars) / max(previous_chars, 1) <= delta_ratio
        )
        plateau_state["previous_chars"] = text_chars
        if plateau_pair:
            plateau_state["pairs"] = _as_int(plateau_state.get("pairs")) + 1

        body_texts = _tool_result_texts(row.get("request_json"))
        body_available = bool(body_texts)
        if body_texts:
            features = terminal_log_features_from_text("\n".join(body_texts))
        else:
            features = _terminal_features_from_routing(routing, crunch)
        terminal_fraction = _fraction_midpoint(features.get("terminal_output_char_fraction_bucket"))
        terminal_signal = terminal_fraction > 0
        projected_saved_chars = 0
        if provider == "anthropic" and category == "tool-result" and plateau_pair and terminal_signal:
            terminal_chars_est = int(text_chars * terminal_fraction)
            projected_saved_chars = max(0, terminal_chars_est // 2)
        projected_saved_tokens = projected_saved_chars // TOKEN_CHARS
        projected_saved_usd = estimate_cost(str(routed_model or requested_model or ""), projected_saved_tokens, 0, provider=provider) or 0.0

        session_bucket = "session-observed" if row.get("session_id") else "session-missing"
        basis = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": str(routing.get("workflow_phase") or routing.get("phase") or category or "unknown"),
            "requested_model_family": str(row.get("requested_model_family") or _model_family(requested_model, provider)),
            "routed_model_family": str(row.get("routed_model_family") or _model_family(routed_model, provider)),
            "stream": bool(_as_int(row.get("stream"))),
            "text_bucket": text_bucket,
            "session_bucket": session_bucket,
            "terminal_output_char_fraction_bucket": str(features.get("terminal_output_char_fraction_bucket") or "none"),
        }

        _increment(provider_counts, provider)
        _increment(category_counts, category)
        _increment(surface_counts, source_surface)
        _increment(session_bucket_counts, session_bucket)

        include = provider == "anthropic" or category == "tool-result" or plateau_pair or terminal_signal
        if not include:
            continue
        matched_rows += 1
        group_key = _candidate_id(basis)
        group = groups.setdefault(group_key, _new_group(basis))
        group["matched_count"] += 1
        group["successful_count"] += int(_as_int(row.get("status_code")) < 400)
        group["error_count"] += int(_as_int(row.get("status_code")) >= 400)
        group["plateau_pair_count"] += int(plateau_pair)
        group["body_rows"] += int(body_available)
        group["metadata_only_rows"] += int(not body_available)
        group["terminal_signal_rows"] += int(terminal_signal)
        group["estimated_input_chars"] += text_chars
        group["estimated_input_tokens"] += input_tokens or max(0, text_chars // TOKEN_CHARS)
        group["estimated_cost_usd"] += _as_float(row.get("cost_est_usd")) or _as_float(row.get("cost_baseline_usd"))
        group["projected_saved_chars"] += projected_saved_chars
        group["projected_saved_tokens"] += projected_saved_tokens
        group["projected_saved_usd"] += projected_saved_usd
        status_bucket = "2xx" if _as_int(row.get("status_code")) < 300 else ("4xx" if _as_int(row.get("status_code")) < 500 else "5xx")
        _increment(group["status_counts"], status_bucket)
        blockers = _row_blockers(
            provider=provider,
            category=category,
            plateau_pair=plateau_pair,
            terminal_fraction=terminal_fraction,
            body_available=body_available,
            projected_saved_chars=projected_saved_chars,
            status_code=row.get("status_code"),
        )
        for blocker in blockers:
            _increment(group["blocker_counts"], blocker)
            _increment(blocker_totals, blocker)

    candidates = [_finalize_group(group) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("projected_saved_chars")),
            _as_int(item.get("plateau_pair_count")),
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
            "candidate_count": len(candidates),
            "plateau_pair_count": sum(_as_int(item.get("plateau_pair_count")) for item in candidates),
            "terminal_signal_rows": sum(_as_int(item.get("terminal_signal_rows")) for item in candidates),
            "body_rows": sum(_as_int(item.get("body_rows")) for item in candidates),
            "metadata_only_rows": sum(_as_int(item.get("metadata_only_rows")) for item in candidates),
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in candidates),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in candidates), 6),
        },
        "projection_policy": {
            "schema": "tokenclaw.terminal_output_compaction_projection_policy.v1",
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "raw_body_required": False,
            "default_apply": False,
            "min_text_chars": text_floor,
            "max_plateau_delta_ratio": delta_ratio,
            "method": "rank local metadata cohorts with adjacent large-context plateau evidence and terminal-output feature buckets; optional local request bodies are used only to derive metadata features",
            "body_off_projection": "metadata-only terminal feature buckets can show blockers but do not invent raw-line precision",
        },
        "provider_breakdown": _breakdown(provider_counts),
        "category_breakdown": _breakdown(category_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "session_bucket_breakdown": _breakdown(session_bucket_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_tool_payloads_included": False,
            "raw_terminal_text_included": False,
            "raw_responses_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local call metadata, privacy-safe terminal feature buckets, and local-only body scanning when body logging is enabled",
        },
    }
