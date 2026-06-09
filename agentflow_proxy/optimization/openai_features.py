from __future__ import annotations

from typing import Any

from agentflow_proxy.recommendations import build_optimization_unit, build_outcome_feedback


OPENAI_FEATURE_SUMMARY_SCHEMA = "agentflow.openai_feature_summary.v1"
OPENAI_OUTCOME_SUMMARY_SCHEMA = "agentflow.openai_outcome_summary.v1"


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
    unit.setdefault("tool_features", {}).update(tool_features)
    return unit


def summarize_openai_request_feature_unit(unit: dict[str, Any]) -> dict[str, Any]:
    input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
    tool_features = unit.get("tool_features") if isinstance(unit.get("tool_features"), dict) else {}
    pattern_features = unit.get("pattern_features") if isinstance(unit.get("pattern_features"), dict) else {}
    return {
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
    return {
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
