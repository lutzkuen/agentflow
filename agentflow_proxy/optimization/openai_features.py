from __future__ import annotations

from typing import Any

from agentflow_proxy.recommendations import build_optimization_unit, build_outcome_feedback
from agentflow_proxy.router import extract_text
from agentflow_proxy.terminal_features import terminal_log_features_from_text


OPENAI_FEATURE_SUMMARY_SCHEMA = "agentflow.openai_feature_summary.v1"
OPENAI_OUTCOME_SUMMARY_SCHEMA = "agentflow.openai_outcome_summary.v1"
OPENAI_PREFLIGHT_SCHEMA = "agentflow.openai_preflight_feature_unit.v1"

CHAR_BUCKETS = (
    (2_000, "lt_2k_chars"),
    (8_000, "2k_8k_chars"),
    (32_000, "8k_32k_chars"),
    (128_000, "32k_128k_chars"),
)
TOKEN_BUCKETS = (
    (1_000, "lt_1k_tokens"),
    (4_000, "1k_4k_tokens"),
    (16_000, "4k_16k_tokens"),
    (64_000, "16k_64k_tokens"),
)


def openai_source_surface(path: str) -> str:
    path_l = (path or "").lower()
    if "chat/completions" in path_l:
        return "openai_chat"
    return "openai_responses"


def openai_endpoint(path: str) -> str:
    path_l = (path or "").lower()
    if "chat/completions" in path_l:
        return "chat_completions"
    if "responses" in path_l:
        return "responses"
    return (path or "").strip("/") or "unknown"


def openai_app_family(model: str | None) -> str:
    model_l = (model or "").lower()
    if "codex" in model_l:
        return "codex"
    return "generic_openai"


def openai_model_family(model: str | None) -> str | None:
    if not model:
        return None
    model_l = model.lower()
    if "gpt-5" in model_l and "codex" in model_l:
        return "gpt-5-codex"
    for family in ("gpt-5", "gpt-4", "gpt-3", "codex"):
        if family in model_l:
            return family
    return "other"


def openai_call_store_fields(path: str, requested_model: str | None, routed_model: str | None) -> dict[str, Any]:
    return {
        "source_surface": openai_source_surface(path),
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": openai_model_family(routed_model),
    }


def _safe_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _bucket_number(value: Any, buckets: tuple[tuple[int, str], ...], fallback: str) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 0:
        return "unknown"
    for upper, label in buckets:
        if number < upper:
            return label
    return fallback


def _text_bucket(value: Any) -> str:
    return _bucket_number(value, CHAR_BUCKETS, "gte_128k_chars")


def _token_bucket(value: Any) -> str:
    return _bucket_number(value, TOKEN_BUCKETS, "gte_64k_tokens")


def _metadata_text_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_metadata_text_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(_metadata_text_chars(item) for item in value.values())
    return 0


def _openai_old_context_features(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    input_items = body.get("input") if isinstance(body.get("input"), list) else []

    if messages:
        old_items = messages[:-4] if len(messages) > 4 else []
        old_chars = _metadata_text_chars(old_items)
        return {
            "shape": "chat_messages",
            "conversation_item_count": len(messages),
            "older_context_item_count": len(old_items),
            "older_context_text_bucket": _text_bucket(old_chars),
            "older_context_token_bucket": _token_bucket(old_chars // 4),
            "system_or_developer_present": any(
                isinstance(item, dict) and item.get("role") in {"system", "developer"}
                for item in messages
            ),
            "raw_payload_included": False,
        }

    if input_items:
        old_items = input_items[:-4] if len(input_items) > 4 else []
        old_chars = _metadata_text_chars(old_items)
        return {
            "shape": "responses_input_items",
            "conversation_item_count": len(input_items),
            "older_context_item_count": len(old_items),
            "older_context_text_bucket": _text_bucket(old_chars),
            "older_context_token_bucket": _token_bucket(old_chars // 4),
            "system_or_developer_present": bool(body.get("instructions")),
            "raw_payload_included": False,
        }

    input_value = body.get("input")
    input_chars = _metadata_text_chars(input_value)
    return {
        "shape": "single_input" if input_value is not None else "unknown",
        "conversation_item_count": 1 if input_value is not None else 0,
        "older_context_item_count": 0,
        "older_context_text_bucket": "none",
        "older_context_token_bucket": "none",
        "system_or_developer_present": bool(body.get("instructions")),
        "input_text_bucket": _text_bucket(input_chars),
        "raw_payload_included": False,
    }


def _openai_cache_eligibility_hints(*, stream: bool, tool_features: dict[str, Any]) -> dict[str, Any]:
    has_tools = bool(tool_features.get("has_tools"))
    return {
        "exact_cache_candidate": not stream and not has_tools,
        "semantic_cache_candidate": not stream and not has_tools,
        "streaming_bypass_hint": bool(stream),
        "tool_call_bypass_hint": has_tools,
        "tool_cache_requires_invalidation": has_tools,
        "reason_hint": "streaming" if stream else ("tool-request" if has_tools else "cacheable-shape"),
        "raw_cache_key_included": False,
    }


def _openai_tool_features(body: dict[str, Any]) -> dict[str, Any]:
    input_items = body.get("input") if isinstance(body.get("input"), list) else []
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    response_tool_item_types = {
        str(item.get("type"))
        for item in input_items
        if isinstance(item, dict)
        and str(item.get("type") or "").lower()
        in {
            "function_call",
            "function_call_output",
            "tool_call",
            "tool_result",
            "computer_call",
            "file_search_call",
            "web_search_call",
        }
    }
    chat_tool_call_count = 0
    chat_tool_result_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            chat_tool_call_count += len(tool_calls)
        if message.get("role") == "tool" or message.get("tool_call_id"):
            chat_tool_result_count += 1
    declared_tool_count = _safe_len(body.get("tools")) + _safe_len(body.get("functions"))
    has_tools = any((
        declared_tool_count > 0,
        bool(body.get("tool_choice")),
        bool(body.get("function_call")),
        bool(response_tool_item_types),
        chat_tool_call_count > 0,
        chat_tool_result_count > 0,
    ))
    return {
        "has_tools": has_tools,
        "declared_tool_count": declared_tool_count,
        "tool_choice_present": bool(body.get("tool_choice") or body.get("function_call")),
        "parallel_tool_calls_present": "parallel_tool_calls" in body,
        "chat_tool_call_count": chat_tool_call_count,
        "chat_tool_result_count": chat_tool_result_count,
        "response_tool_item_types": sorted(response_tool_item_types),
    }


def build_openai_preflight_feature_unit(
    *,
    body: dict[str, Any],
    path: str,
    requested_model: str,
    routing_meta: dict[str, Any],
    category: str | None,
    stream: bool,
    input_tokens_est: int | None,
) -> dict[str, Any]:
    tool_features = _openai_tool_features(body)
    terminal_log_features = terminal_log_features_from_text(extract_text(body))
    normalized_routing = dict(routing_meta)
    normalized_routing.update({
        "provider": "openai",
        "source_surface": openai_source_surface(path),
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": None,
        "has_tools": bool(routing_meta.get("has_tools")) or bool(tool_features["has_tools"]),
        "policy_source": "preflight",
    })
    unit = build_optimization_unit(
        provider="openai",
        path=path,
        requested_model=requested_model,
        routed_model=None,
        routing_meta=normalized_routing,
        crunch_meta={"status": "not-run", "changed": False},
        cache_meta={"status": "not-evaluated", "reason": "preflight"},
        category=category,
        stream=stream,
        input_tokens_est=input_tokens_est,
        session_id=None,
    )
    input_features = unit.setdefault("input_features", {})
    for key in (
        "local_routed_model",
        "local_routing_reason",
        "local_routing_policy_source",
        "crunch_changed",
        "crunch_saved_chars",
    ):
        input_features.pop(key, None)
    input_features.update({
        "local_mutation_stage": "preflight",
        "source_surface": unit["source_surface"],
        "endpoint": openai_endpoint(path),
        "path_class": openai_endpoint(path),
        "provider": "openai",
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": None,
        "pre_crunch_text_bucket": _text_bucket(routing_meta.get("text_chars")),
        "pre_crunch_input_token_bucket": _token_bucket(input_tokens_est),
        "old_context": _openai_old_context_features(body),
        "cache_eligibility": _openai_cache_eligibility_hints(stream=stream, tool_features=tool_features),
        "terminal_log_features": terminal_log_features,
        "raw_payload_included": False,
    })
    unit.update({
        "schema": OPENAI_PREFLIGHT_SCHEMA,
        "provider": "openai",
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": None,
        "model_family_changed": False,
        "candidate_target_model": None,
        "grouping_identifiers": {},
    })
    unit.setdefault("tool_features", {}).update(tool_features)
    unit.setdefault("privacy_summary", {}).update({
        "preflight_metadata_only": True,
        "raw_payload_included": False,
    })
    return unit


def build_openai_request_feature_unit(
    *,
    body: dict[str, Any],
    path: str,
    requested_model: str,
    routed_model: str,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    category: str | None,
    stream: bool,
    input_tokens_est: int | None,
    session_id: str | None = None,
) -> dict[str, Any]:
    tool_features = _openai_tool_features(body)
    terminal_log_features = terminal_log_features_from_text(extract_text(body))
    normalized_routing = dict(routing_meta)
    normalized_routing.update({
        "provider": "openai",
        "source_surface": openai_source_surface(path),
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": openai_model_family(routed_model),
        "has_tools": bool(routing_meta.get("has_tools")) or bool(tool_features["has_tools"]),
    })
    unit = build_optimization_unit(
        provider="openai",
        path=path,
        requested_model=requested_model,
        routed_model=routed_model,
        routing_meta=normalized_routing,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
        category=category,
        stream=stream,
        input_tokens_est=input_tokens_est,
        session_id=session_id,
    )
    unit.update({
        "schema": "agentflow.openai_optimization_unit.v1",
        "provider": "openai",
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": openai_model_family(routed_model),
        "model_family_changed": openai_model_family(requested_model) != openai_model_family(routed_model),
    })
    unit.setdefault("input_features", {})["source_surface"] = unit["source_surface"]
    unit["input_features"]["endpoint"] = unit["endpoint"]
    unit["input_features"]["provider"] = "openai"
    unit["input_features"]["requested_model_family"] = unit["requested_model_family"]
    unit["input_features"]["routed_model_family"] = unit["routed_model_family"]
    unit["input_features"]["terminal_log_features"] = terminal_log_features
    unit.setdefault("tool_features", {}).update(tool_features)
    return unit


def summarize_openai_request_feature_unit(unit: dict[str, Any]) -> dict[str, Any]:
    input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
    tool_features = unit.get("tool_features") if isinstance(unit.get("tool_features"), dict) else {}
    pattern_features = unit.get("pattern_features") if isinstance(unit.get("pattern_features"), dict) else {}
    summary = {
        "schema": OPENAI_FEATURE_SUMMARY_SCHEMA,
        "provider": "openai",
        "source_surface": unit.get("source_surface"),
        "endpoint": unit.get("endpoint"),
        "app_family": unit.get("app_family"),
        "requested_model_family": unit.get("requested_model_family"),
        "routed_model_family": unit.get("routed_model_family"),
        "model_family_changed": bool(unit.get("model_family_changed")),
        "stream": bool(input_features.get("stream")),
        "category": input_features.get("category"),
        "workflow_phase": input_features.get("workflow_phase"),
        "text_bucket": input_features.get("text_bucket"),
        "input_token_bucket": input_features.get("input_token_bucket"),
        "has_tools": bool(tool_features.get("has_tools")),
        "declared_tool_count": tool_features.get("declared_tool_count"),
        "chat_tool_call_count": tool_features.get("chat_tool_call_count"),
        "chat_tool_result_count": tool_features.get("chat_tool_result_count"),
        "response_tool_item_types": tool_features.get("response_tool_item_types") or [],
        "pattern_hash": pattern_features.get("pattern_hash"),
        "crunch_pattern_hash": pattern_features.get("crunch_pattern_hash"),
        "cache_pattern_hash": pattern_features.get("cache_pattern_hash"),
        "replayability_level": unit.get("replayability_level"),
        "privacy_summary": unit.get("privacy_summary"),
        "raw_payload_included": False,
    }
    if input_features.get("local_mutation_stage") is not None:
        summary["local_mutation_stage"] = input_features.get("local_mutation_stage")
    if input_features.get("path_class") is not None:
        summary["path_class"] = input_features.get("path_class")
    if input_features.get("old_context") is not None:
        summary["old_context"] = input_features.get("old_context")
    if input_features.get("cache_eligibility") is not None:
        summary["cache_eligibility"] = input_features.get("cache_eligibility")
    if input_features.get("terminal_log_features") is not None:
        summary["terminal_log_features"] = input_features.get("terminal_log_features")
    return summary


def build_openai_outcome_feature_unit(
    *,
    path: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    input_tokens_est: int | None,
    output_tokens_est: int | None,
    actual_input_tokens: int | None,
    actual_output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    thinking_output_tokens: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    cache_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    category: str | None,
    session_id: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    unit = build_outcome_feedback(
        provider="openai",
        path=path,
        requested_model=requested_model,
        routed_model=routed_model,
        status_code=status_code,
        latency_ms=latency_ms,
        retry_count=retry_count,
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        actual_input_tokens=actual_input_tokens,
        actual_output_tokens=actual_output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        thinking_output_tokens=thinking_output_tokens,
        cost_est_usd=cost_est_usd,
        cost_baseline_usd=cost_baseline_usd,
        cache_meta=cache_meta,
        crunch_meta=crunch_meta,
        routing_meta=routing_meta,
        category=category,
        session_id=session_id,
        error=error,
    )
    unit.update({
        "schema": "agentflow.openai_outcome_unit.v1",
        "endpoint": openai_endpoint(path),
        "requested_model_family": openai_model_family(requested_model),
        "routed_model_family": openai_model_family(routed_model),
    })
    return unit


def summarize_openai_outcome_feature_unit(unit: dict[str, Any]) -> dict[str, Any]:
    quality = unit.get("quality_signals") if isinstance(unit.get("quality_signals"), dict) else {}
    cache_decision = unit.get("cache_decision") if isinstance(unit.get("cache_decision"), dict) else {}
    routing_decision = unit.get("routing_decision") if isinstance(unit.get("routing_decision"), dict) else {}
    baseline = unit.get("cost_baseline_usd")
    actual = unit.get("cost_est_usd")
    savings = None
    if isinstance(baseline, (int, float)) and isinstance(actual, (int, float)):
        savings = max(float(baseline) - float(actual), 0.0)
    summary = {
        "schema": OPENAI_OUTCOME_SUMMARY_SCHEMA,
        "provider": "openai",
        "source_surface": unit.get("source_surface"),
        "endpoint": unit.get("endpoint"),
        "requested_model_family": unit.get("requested_model_family"),
        "routed_model_family": unit.get("routed_model_family"),
        "status_code": unit.get("status_code"),
        "latency_ms": unit.get("latency_ms"),
        "retry_count": unit.get("retry_count"),
        "input_tokens": unit.get("input_tokens"),
        "output_tokens": unit.get("output_tokens"),
        "actual_input_tokens": unit.get("actual_input_tokens"),
        "actual_output_tokens": unit.get("actual_output_tokens"),
        "cache_read_input_tokens": unit.get("cache_read_input_tokens"),
        "reasoning_tokens": unit.get("thinking_output_tokens"),
        "cost_est_usd": actual,
        "cost_baseline_usd": baseline,
        "savings_est_usd": savings,
        "cache_status": cache_decision.get("status"),
        "cache_reason": cache_decision.get("reason"),
        "routing_reason": routing_decision.get("reason"),
        "fallback_reason": routing_decision.get("fallback_reason"),
        "quality_outcome": quality.get("outcome"),
        "quality_risk": quality.get("risk"),
        "raw_payload_included": False,
    }
    if unit.get("terminal_log_features") is not None:
        summary["terminal_log_features"] = unit.get("terminal_log_features")
    return summary
