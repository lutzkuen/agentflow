from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agentflow_proxy.provider_context import ProviderContext
from agentflow_proxy.pricing import MODEL_ALIASES, estimate_cost
from agentflow_proxy.prompt_features import prompt_difficulty_features_from_text
from agentflow_proxy.headers import (
    ClientJsonRequestError,
    build_anthropic_forward_headers,
    build_anthropic_summary_headers,
    client_json_error_body,
    read_json_object_body,
)
from agentflow_proxy.limiter import TierBackoffActive, model_tier, tier_backoff_headers, tier_backoff_payload
from agentflow_proxy.router import (
    extract_text, has_tools, categorize_request, route_model,
    STRIP_THINKING_HISTORY, _has_top_level_thinking, strip_thinking_history_blocks, uses_thinking,
)
from agentflow_proxy.crunch import (
    TOKEN_CHARS, estimate_tokens_from_text, build_embedding,
    crunch_body, inject_prompt_cache, has_cache_control_blocks,
    maybe_summarize_old_context, OLD_CONTEXT_SUMMARY_MODEL,
    CRUNCH_POLICY, CRUNCH_POLICY_SOURCE, CRUNCH_RULES_PATH,
)
from agentflow_proxy.cache import (
    CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD, CACHE_POLICY, CACHE_POLICY_SOURCE, CACHE_RULES_PATH,
    build_cache_replay_lifecycle_feedback, cache_replay_lifecycle_feedback_public_meta,
    cache_decision_meta, cache_file_dependency_audit, cache_hit_decision_meta, cache_key_for, cache_lookup_meta,
    cache_replay_canary_decision, cache_replay_scope_for_meta, response_output_text,
    stream_cache_payload, validate_stream_cache_payload,
    streaming_cache_lookup_meta, cache_file_dependency_snapshots,
)
from agentflow_proxy.optimization_coordinator_enforcement import enforce_optimization_coordinator
from agentflow_proxy.optimization_coordinator_feedback import (
    optimization_coordinator_lifecycle_feedback_public_meta,
    queue_optimization_coordinator_lifecycle_feedback as queue_optimization_coordinator_lifecycle_event,
)
from agentflow_proxy.errors import (
    INTERNAL_PROXY_ERROR_MESSAGE,
    public_proxy_error_body,
    upstream_error_text,
)
from agentflow_proxy.provider_adoption import capture_provider_tool_adoption
from agentflow_proxy.routing_experiments import (
    ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES,
    compare_response_outputs,
    routing_experiment_outcome_event,
    routing_experiment_feedback_features,
    routing_experiment_decision,
)
from agentflow_proxy.session_memory_hints import build_session_memory_optimization_hints
from agentflow_proxy.recommendations import (
    attach_observed_savings_to_routing_meta,
    apply_recommendation_to_body,
    build_old_context_summary_outcome_event,
    build_old_context_summary_outcome_feedback,
    build_outcome_feedback,
    build_phase_routing_outcome_event,
    build_phase_routing_outcome_feedback,
    build_optimization_unit,
    fetch_policy_decision,
    fetch_recommendation,
    CACHE_REPLAY_LIFECYCLE_SOURCE_SURFACE,
    OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
    PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
    pattern_feature_diagnostics,
    policy_decisions_enabled,
    queue_policy_event_feedback,
    queue_outcome_feedback,
)
from agentflow_proxy.store import stable_json, utc_now


SESSION_COST_ALERT_USD = float(os.getenv("AGENTFLOW_SESSION_COST_ALERT_USD", "5.0"))
MAX_THINKING_BUDGET_TOKENS = int(os.getenv("AGENTFLOW_MAX_THINKING_BUDGET_TOKENS", "0"))


async def _queue_optimization_coordinator_lifecycle_feedback(
    context: ProviderContext,
    *,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    enforcement: dict[str, Any],
    status_code: int | None = None,
    retry_count: int | None = None,
    cost_est_usd: float | None = None,
    cost_baseline_usd: float | None = None,
) -> None:
    decision = routing_meta.get("optimization_coordinator")
    if not isinstance(decision, dict):
        return
    try:
        meta = await queue_optimization_coordinator_lifecycle_event(
            context.store,
            decision,
            enforcement=enforcement,
            status_code=status_code,
            retry_count=retry_count,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            flush_immediately=False,
        )
        public_meta = optimization_coordinator_lifecycle_feedback_public_meta(meta)
    except Exception as exc:
        public_meta = {
            "enabled": False,
            "status": "skipped",
            "reason": "coordinator-lifecycle-feedback-error",
            "error_type": type(exc).__name__,
            "payload_included": False,
        }
    enforcement["lifecycle_feedback"] = public_meta
    for meta_obj in (routing_meta, crunch_meta, cache_meta):
        meta_obj["optimization_coordinator_lifecycle_feedback"] = public_meta


def _anthropic_preflight_routing_meta(
    body: dict[str, Any],
    *,
    requested_model: str,
    category: str | None,
) -> dict[str, Any]:
    text = extract_text(body)
    return {
        "enabled": False,
        "requested_model": requested_model,
        "routed_model": str(body.get("model") or requested_model),
        "reason": "preflight feature extraction before local mutation",
        "text_chars": len(text),
        "has_tools": has_tools(body),
        "category": category,
        "policy_source": "preflight",
        "provider": "anthropic",
        "prompt_difficulty_features": prompt_difficulty_features_from_text(text),
    }


def _managed_crunch_profile_from_recommendation(
    recommendation_meta: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(recommendation_meta, dict) or recommendation_meta.get("status") != "received":
        return None

    crunch = recommendation_meta.get("crunch")
    if not isinstance(crunch, dict):
        return None

    if not any(
        key in crunch
        for key in (
            "repeated_provider_scaffolding",
            "old_context_summarization",
            "enhanced_crunch",
            "threshold_chars",
            "thresholds",
            "profile",
        )
    ):
        return None

    profile = copy.deepcopy(crunch)
    profile["policy_source"] = str(
        profile.get("policy_source")
        or recommendation_meta.get("policy_source")
        or "managed-recommended"
    )
    for key in ("policy_id", "candidate_id", "recommendation_id"):
        if profile.get(key) is None and recommendation_meta.get(key) is not None:
            profile[key] = recommendation_meta[key]
    return profile


def _record_routing_rate_limit_fallback(
    routing_meta: dict[str, Any],
    *,
    requested_model: str,
    from_model: Any,
) -> None:
    routing_meta["fallback_reason"] = "rate_limited"
    phase_canary = routing_meta.get("phase_canary")
    if isinstance(phase_canary, dict):
        phase_canary["fallback_reason"] = "rate_limited"
        phase_canary["fallback_from_model"] = str(from_model)
        phase_canary["actual_forwarded_model"] = str(requested_model)


def _anthropic_stream_primary_response_body(
    *,
    output_text: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    thinking_output_tokens: int | None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if cache_creation_input_tokens:
        usage["cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens:
        usage["cache_read_input_tokens"] = cache_read_input_tokens
    if thinking_output_tokens is not None:
        usage["thinking_output_tokens"] = thinking_output_tokens
    return {
        "type": "message",
        "content": [{"type": "text", "text": output_text}] if output_text else [],
        "usage": usage,
        "agentflow_streaming_capture": {
            "complete": True,
            "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "raw_stream_included": False,
        },
    }


def _mark_streaming_experiment_skip(
    routing_meta: dict[str, Any],
    *,
    reason: str,
    stream_complete: bool,
    status_code: int | None,
    error: str | None,
) -> None:
    experiment_meta = routing_meta.get("routing_experiment")
    if not isinstance(experiment_meta, dict):
        return
    if experiment_meta.get("mode") != "shadow_candidate_pass_through":
        return
    experiment_meta.update(
        {
            "status": "skipped",
            "sampled": False,
            "reason": reason,
            "streaming": {
                "complete": bool(stream_complete),
                "primary_status_code": status_code,
                "primary_error_present": bool(error),
                "raw_stream_included": False,
            },
        }
    )
    routing_meta["routing_experiment"] = experiment_meta


def _attach_session_memory_hints(
    *,
    context: ProviderContext,
    session_id: str | None,
    stream: bool,
    has_tool_blocks: bool,
    category: str | None,
    text_chars: int,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    current_thinking: bool,
) -> None:
    file_audit = cache_meta.get("file_dependency_audit") if isinstance(cache_meta.get("file_dependency_audit"), dict) else {}
    pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    hints = build_session_memory_optimization_hints(
        store_obj=context.store,
        session_id=session_id,
        stream=stream,
        has_tool_blocks=has_tool_blocks,
        category=category,
        text_chars=text_chars,
        routing_meta=routing_meta,
        crunch_policy=CRUNCH_POLICY,
        crunch_policy_source=CRUNCH_POLICY_SOURCE,
        crunch_rule_path=CRUNCH_RULES_PATH,
        cache_policy=CACHE_POLICY,
        cache_policy_source=CACHE_POLICY_SOURCE,
        cache_rule_path=CACHE_RULES_PATH,
        safe_invalidation_evidence=bool(file_audit.get("safe_invalidation_evidence")),
        reviewed_cache_pattern_rule=bool(pattern_rule),
        current_thinking=current_thinking,
    )
    crunch_meta["session_memory_hints"] = hints["crunch"]
    cache_meta["session_memory_hints"] = hints["cache"]
    cache_meta["session_memory_replayability"] = {
        "status": hints["cache"]["status"],
        "replayability_level": hints["cache"]["replayability_level"],
        "cacheability_hint": hints["cache"]["cacheability_hint"],
        "dry_run_projection": hints["cache"]["dry_run_projection"],
    }


async def _record_managed_outcome_feedback(
    *,
    context: ProviderContext,
    call_id: str,
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
    stream: bool,
    session_id: str | None,
    error: str | None = None,
) -> None:
    dirty_routing_meta = False
    dirty_cache_meta = False
    cache_replay_feedback = build_cache_replay_lifecycle_feedback(
        cache_meta=cache_meta,
        provider="anthropic",
        source_surface="anthropic_messages",
        requested_model=requested_model,
        routed_model=routed_model,
        status_code=status_code,
        latency_ms=latency_ms,
        retry_count=retry_count,
        cost_est_usd=cost_est_usd,
        cost_baseline_usd=cost_baseline_usd,
        category=category,
        stream=stream,
    )
    if cache_replay_feedback is not None:
        meta = await queue_policy_event_feedback(
            context.store,
            cache_replay_feedback,
            source_surface=CACHE_REPLAY_LIFECYCLE_SOURCE_SURFACE,
        )
        cache_meta["cache_replay_lifecycle_feedback"] = cache_replay_lifecycle_feedback_public_meta(meta)
        dirty_cache_meta = True
    summary_feedback = build_old_context_summary_outcome_feedback(
        provider="anthropic",
        path=path,
        requested_model=requested_model,
        routed_model=routed_model,
        status_code=status_code,
        latency_ms=latency_ms,
        retry_count=retry_count,
        cache_hit=cache_meta.get("status") == "hit",
        crunch_meta=crunch_meta,
        category=category,
        error=error,
    )
    if summary_feedback is not None:
        event = build_old_context_summary_outcome_event(summary_feedback)
        meta = await queue_policy_event_feedback(
            context.store,
            event,
            source_surface=OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
        )
        routing_meta["old_context_summary_feedback"] = {
            "enabled": bool(meta.get("enabled")),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "endpoint": meta.get("endpoint"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "status_code": meta.get("status_code"),
            "latency_ms": meta.get("latency_ms"),
            "source_surface": OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
            "payload_included": False,
        }
        dirty_routing_meta = True

    phase_feedback = build_phase_routing_outcome_feedback(
        provider="anthropic",
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
        thinking_output_tokens=thinking_output_tokens,
        cost_est_usd=cost_est_usd,
        cost_baseline_usd=cost_baseline_usd,
        cache_meta=cache_meta,
        crunch_meta=crunch_meta,
        routing_meta=routing_meta,
        category=category,
        error=error,
    )
    if phase_feedback is not None:
        try:
            event = build_phase_routing_outcome_event(phase_feedback)
            meta = await queue_policy_event_feedback(
                context.store,
                event,
                source_surface=PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
            )
        except Exception as exc:
            meta = {
                "enabled": True,
                "status": "error",
                "reason": "queue-failed",
                "endpoint": "/v1/policy-events",
                "error": repr(exc),
            }
        routing_meta["phase_routing_feedback"] = {
            "enabled": bool(meta.get("enabled")),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "endpoint": meta.get("endpoint"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "status_code": meta.get("status_code"),
            "latency_ms": meta.get("latency_ms"),
            "source_surface": PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
            "payload_included": False,
        }
        dirty_routing_meta = True

    managed = routing_meta.get("managed_recommendation")
    if not isinstance(managed, dict) or not managed.get("enabled"):
        if dirty_routing_meta:
            context.store.update_call_routing_json(call_id, stable_json(routing_meta))
        if dirty_cache_meta:
            context.store.update_call_cache_json(call_id, stable_json(cache_meta))
        return
    experiment_meta = routing_meta.get("routing_experiment")
    if isinstance(experiment_meta, dict) and experiment_meta.get("sampled"):
        experiment_meta["managed_feedback"] = {
            "enabled": True,
            "status": "pending",
            "reason": "outcome-feedback-pending",
            "optimization_unit_id": managed.get("optimization_unit_id"),
        }
    provider_adoption_windows = []
    if hasattr(context.store, "provider_tool_adoption_windows_for_call_ids"):
        provider_adoption_windows = context.store.provider_tool_adoption_windows_for_call_ids([call_id]).get(call_id, [])
    outcome = build_outcome_feedback(
        provider="anthropic",
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
        provider_adoption_windows=provider_adoption_windows,
    )
    managed["outcome_feedback"] = await queue_outcome_feedback(
        context.store,
        managed,
        outcome,
        source_surface="anthropic_messages",
    )
    if isinstance(experiment_meta, dict) and experiment_meta.get("sampled"):
        feedback_meta = managed["outcome_feedback"]
        experiment_meta["managed_feedback"] = {
            "enabled": bool(feedback_meta.get("enabled")),
            "status": feedback_meta.get("status"),
            "reason": feedback_meta.get("reason"),
            "optimization_unit_id": feedback_meta.get("optimization_unit_id") or managed.get("optimization_unit_id"),
            "status_code": feedback_meta.get("status_code"),
            "latency_ms": feedback_meta.get("latency_ms"),
        }
        experiment_id = experiment_meta.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id:
            context.store.update_routing_experiment_json(experiment_id, stable_json(experiment_meta))
    context.store.update_call_routing_json(call_id, stable_json(routing_meta))
    if dirty_cache_meta:
        context.store.update_call_cache_json(call_id, stable_json(cache_meta))


async def _check_session_cost_alert(context: ProviderContext, sid: str) -> None:
    row = context.store.conn.execute(
        "SELECT COALESCE(SUM(cost_est_usd), 0.0) as cost, COUNT(*) as calls "
        "FROM calls WHERE session_id = ? AND date(created_at) = date('now')",
        (sid,),
    ).fetchone()
    cost = float(row["cost"]) if row else 0.0
    calls = int(row["calls"]) if row else 0
    if cost >= SESSION_COST_ALERT_USD:
        logging.warning(
            "Session %s daily cost $%.2f (%d calls) exceeds alert threshold $%.2f",
            sid[:8], cost, calls, SESSION_COST_ALERT_USD,
        )


def provider_disabled_response(context: ProviderContext, expected: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": f"AgentFlow is running in {context.provider!r} provider mode, not {expected!r}.",
                "type": "provider_mismatch",
            }
        },
        status_code=404,
    )


def _count_thinking_chars(response_body: dict) -> int:
    total = 0
    for block in (response_body or {}).get("content") or []:
        if isinstance(block, dict) and block.get("type") == "thinking":
            total += len(block.get("thinking") or "")
    return total


def _strip_model_incompatible_params(body: dict[str, Any], routing_meta: dict[str, Any], requested_model: str) -> None:
    if body.get("model") == requested_model:
        return
    stripped = list(routing_meta.get("stripped_params") or [])
    thinking_block = body.get("thinking")
    if isinstance(thinking_block, dict) and "effort" in thinking_block:
        del thinking_block["effort"]
        if "thinking.effort" not in stripped:
            stripped.append("thinking.effort")
    for key in ("effort", "thinking", "budget_tokens", "interleaved_thinking"):
        if key in body:
            del body[key]
            if key not in stripped:
                stripped.append(key)
    if stripped:
        routing_meta["stripped_params"] = stripped


def _cache_key_variants_for_models(
    body: dict[str, Any],
    path: str,
    *,
    provider: str,
    upstream: str,
    replay_scope: str | None,
    replay_scope_id: str | None,
    requested_model: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []

    def add_variant(candidate: dict[str, Any]) -> None:
        key = cache_key_for(
            candidate,
            path,
            provider=provider,
            upstream=upstream,
            replay_scope=replay_scope,
            replay_scope_id=replay_scope_id,
        )
        if all(existing_key != key for existing_key, _ in variants):
            variants.append((key, candidate))

    add_variant(body)
    requested = str(requested_model or "").strip()
    current = str(body.get("model") or "").strip()
    if requested and current and requested != current:
        requested_body = copy.deepcopy(body)
        requested_body["model"] = requested
        add_variant(requested_body)
    return variants


_ANTHROPIC_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def _anthropic_message_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _anthropic_tool_use_ids(message: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ids
    for block in _anthropic_message_blocks(message):
        if block.get("type") != "tool_use":
            continue
        tool_id = str(block.get("id") or "")
        if tool_id:
            ids.add(tool_id)
    return ids


def _anthropic_thinking_block_count(message: Any) -> int:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return 0
    return sum(
        1
        for block in _anthropic_message_blocks(message)
        if block.get("type") in _ANTHROPIC_THINKING_BLOCK_TYPES
    )


def _previous_assistant_message(messages: list[Any], index: int) -> dict[str, Any] | None:
    if index <= 0:
        return None
    candidate = messages[index - 1]
    if isinstance(candidate, dict) and candidate.get("role") == "assistant":
        return candidate
    return None


def _anthropic_shadow_tool_result_audit(body: dict[str, Any]) -> dict[str, Any]:
    """Return metadata-only Anthropic tool-use protocol diagnostics for shadow calls."""
    tool_use_ids: set[str] = set()
    tool_result_count = 0
    orphan_count = 0
    non_adjacent_count = 0
    tool_result_from_thinking_turn_count = 0
    thinking_blocks_before_tool_results = 0
    messages = body.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            tool_use_ids.update(_anthropic_tool_use_ids(message))
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            previous_assistant = _previous_assistant_message(messages, index)
            previous_tool_use_ids = _anthropic_tool_use_ids(previous_assistant)
            previous_thinking_blocks = _anthropic_thinking_block_count(previous_assistant)
            for block in _anthropic_message_blocks(message):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_result_count += 1
                tool_use_id = str(block.get("tool_use_id") or "")
                if not tool_use_id or tool_use_id not in tool_use_ids:
                    orphan_count += 1
                    continue
                if tool_use_id not in previous_tool_use_ids:
                    non_adjacent_count += 1
                    continue
                if previous_thinking_blocks:
                    tool_result_from_thinking_turn_count += 1
                    thinking_blocks_before_tool_results += previous_thinking_blocks
    status = "ok"
    reason = None
    if orphan_count:
        status = "unsupported"
        reason = "orphan-tool-result"
    elif non_adjacent_count:
        status = "unsupported"
        reason = "non-adjacent-tool-result"
    return {
        "schema": "agentflow.anthropic_shadow_tool_result_audit.v1",
        "status": status,
        "reason": reason,
        "tool_result_count": tool_result_count,
        "assistant_tool_use_count": len(tool_use_ids),
        "orphan_tool_result_count": orphan_count,
        "non_adjacent_tool_result_count": non_adjacent_count,
        "tool_result_from_thinking_turn_count": tool_result_from_thinking_turn_count,
        "thinking_blocks_before_tool_results": thinking_blocks_before_tool_results,
        "raw_tool_ids_included": False,
        "tool_payloads_included": False,
    }


def _assistant_empty_content_count(body: dict[str, Any]) -> int:
    count = 0
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content") == []:
            count += 1
    return count


def _prepare_anthropic_shadow_request(
    request_body: dict[str, Any],
    *,
    shadow_model: str,
    primary_model: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    shadow_body = copy.deepcopy(request_body)
    shadow_body["model"] = shadow_model
    shadow_body["stream"] = False
    diagnostics: dict[str, Any] = {
        "schema": "agentflow.anthropic_shadow_request_preflight.v1",
        "status": "ok",
        "reason": None,
        "primary_model": primary_model,
        "shadow_model": shadow_model,
        "stream_forced_non_streaming": True,
        "raw_request_included": False,
        "raw_prompts_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
    }
    sanitization: dict[str, Any] = {}
    _strip_model_incompatible_params(shadow_body, sanitization, primary_model)
    if sanitization.get("stripped_params"):
        diagnostics["stripped_params"] = sanitization["stripped_params"]
    tool_audit = _anthropic_shadow_tool_result_audit(shadow_body)
    diagnostics["tool_result_audit"] = tool_audit
    if tool_audit["status"] == "unsupported":
        diagnostics.update({"status": "unsupported", "reason": tool_audit["reason"]})
        return None, diagnostics
    if model_tier(shadow_model) == "haiku":
        diagnostics["candidate_would_strip_thinking_history"] = True
        if int(tool_audit.get("tool_result_from_thinking_turn_count") or 0) > 0:
            diagnostics.update({
                "status": "unsupported",
                "reason": "tool-result-thinking-continuation",
                "thinking_history_blocks_detected": tool_audit.get("thinking_blocks_before_tool_results"),
            })
            return None, diagnostics
        shadow_body, stripped_thinking = strip_thinking_history_blocks(shadow_body)
        if stripped_thinking:
            diagnostics["thinking_history_blocks_stripped"] = stripped_thinking
    empty_assistant_count = _assistant_empty_content_count(shadow_body)
    if empty_assistant_count:
        diagnostics.update({
            "status": "unsupported",
            "reason": "thinking-history-only-assistant-message",
            "empty_assistant_message_count": empty_assistant_count,
        })
        return None, diagnostics
    return shadow_body, diagnostics


def _shadow_http_error_class(status_code: int | None, response_body: dict[str, Any] | None) -> str | None:
    if status_code is None or status_code < 400:
        return None
    if status_code == 400:
        default = "shadow-http-400"
    elif status_code < 500:
        default = "shadow-http-4xx"
    else:
        default = "shadow-http-5xx"
    error_obj = response_body.get("error") if isinstance(response_body, dict) else None
    if isinstance(error_obj, dict):
        error_type = str(error_obj.get("type") or "").strip()
        if error_type and len(error_type) <= 80:
            return f"{default}:{error_type}"
    return default



def build_forward_headers(request: Request) -> dict[str, str]:
    return build_anthropic_forward_headers(request.headers)


def build_summary_headers(request: Request) -> dict[str, str]:
    return build_anthropic_summary_headers(request.headers)


async def _fetch_old_context_summary(context: ProviderContext, summary_request: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    model = str(summary_request.get("model") or OLD_CONTEXT_SUMMARY_MODEL)
    try:
        async with context.limiter.semaphores[model_tier(model)]:
            async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                await context.limiter.await_backoff(model)
                await context.limiter.throttle_forward()
                r = await client.post(
                    context.anthropic_upstream.rstrip("/") + "/v1/messages",
                    headers=headers,
                    json=summary_request,
                )
    except Exception:
        logging.exception("agentflow old-context summary error")
        return {
            "summary": None,
            "summary_status_code": None,
            "summary_error": INTERNAL_PROXY_ERROR_MESSAGE,
        }

    try:
        body = r.json()
    except Exception:
        return {
            "summary": None,
            "summary_status_code": r.status_code,
            "summary_error": r.text[:500],
        }

    parts = [
        str(block.get("text") or "")
        for block in body.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    usage = body.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    meta = {
        "summary": "\n".join(parts).strip() if r.status_code < 400 else None,
        "usage": usage,
        "summary_status_code": r.status_code,
        "summary_input_tokens": input_tokens,
        "summary_output_tokens": output_tokens,
    }
    if input_tokens is not None or output_tokens is not None:
        meta["summary_cost_est_usd"] = estimate_cost(model, input_tokens or 0, output_tokens or 0) or 0.0
    if r.status_code >= 400:
        meta["summary_error"] = stable_json(body)[:500]
    return meta


async def _run_anthropic_routing_experiment(
    *,
    context: ProviderContext,
    call_id: str,
    path: str,
    headers: dict[str, str],
    request_body: dict[str, Any],
    routing_meta: dict[str, Any],
    experiment_meta: dict[str, Any],
    primary_response_body: dict[str, Any],
    primary_status_code: int,
    primary_latency_ms: int,
    primary_cost_est_usd: Optional[float],
    input_tokens_est: int,
) -> None:
    experiment_id = str(uuid.uuid4())
    shadow_model = str(experiment_meta.get("shadow_model") or routing_meta.get("requested_model") or "")
    primary_model = str(experiment_meta.get("primary_model") or request_body.get("model") or routing_meta.get("routed_model") or "")
    shadow_body, shadow_preflight = _prepare_anthropic_shadow_request(
        request_body,
        shadow_model=shadow_model,
        primary_model=primary_model,
    )
    experiment_meta["shadow_request_preflight"] = shadow_preflight
    shadow_status_code: Optional[int] = None
    shadow_response_body: Optional[dict[str, Any]] = None
    shadow_latency_ms: Optional[int] = None
    shadow_cost: Optional[float] = None
    error: Optional[str] = None

    shadow_started = time.time()
    if shadow_body is None:
        shadow_latency_ms = 0
        reason = str(shadow_preflight.get("reason") or "unsupported-shape")
        error = f"shadow-unsupported-shape:{reason}"
    else:
        try:
            async with context.limiter.semaphores[model_tier(shadow_model)]:
                async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                    await context.limiter.await_backoff(shadow_model)
                    await context.limiter.throttle_forward()
                    r = await client.post(context.anthropic_upstream.rstrip("/") + path, headers=headers, json=shadow_body)
            shadow_latency_ms = int((time.time() - shadow_started) * 1000)
            shadow_status_code = r.status_code
            try:
                shadow_response_body = r.json()
            except Exception:
                shadow_response_body = None
            http_error_class = _shadow_http_error_class(shadow_status_code, shadow_response_body)
            if http_error_class:
                error = http_error_class
        except Exception as exc:
            shadow_latency_ms = int((time.time() - shadow_started) * 1000)
            error = repr(exc)

    if shadow_response_body is not None:
        usage = shadow_response_body.get("usage") or {}
        shadow_in = usage.get("input_tokens")
        shadow_out = usage.get("output_tokens")
        cache_creation = usage.get("cache_creation_input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0
        shadow_out_est = estimate_tokens_from_text(response_output_text(shadow_response_body))
        shadow_cost = estimate_cost(
            shadow_model,
            shadow_in if shadow_in is not None else input_tokens_est,
            shadow_out if shadow_out is not None else shadow_out_est,
            cache_creation,
            cache_read,
        )

    comparison = compare_response_outputs(primary_response_body, shadow_response_body)
    feedback_features = routing_experiment_feedback_features(
        experiment_id=experiment_id,
        experiment_meta=experiment_meta,
        routing_meta=routing_meta,
        comparison=comparison,
        primary_model=primary_model,
        shadow_model=shadow_model,
        primary_status_code=primary_status_code,
        shadow_status_code=shadow_status_code,
        primary_latency_ms=primary_latency_ms,
        shadow_latency_ms=shadow_latency_ms,
        primary_cost_est_usd=primary_cost_est_usd,
        shadow_cost_est_usd=shadow_cost,
        error=error,
    )
    experiment_meta.update(
        {
            "experiment_id": experiment_id,
            "status": feedback_features["status"],
            "primary_model": primary_model,
            "shadow_model": shadow_model,
            "primary_status_code": primary_status_code,
            "shadow_status_code": shadow_status_code,
            "primary_output_chars": comparison["primary_output_chars"],
            "shadow_output_chars": comparison["shadow_output_chars"],
            "primary_output_sha256": comparison["primary_output_sha256"],
            "shadow_output_sha256": comparison["shadow_output_sha256"],
            "output_similarity": comparison["output_similarity"],
            "passed_threshold": comparison["passed_threshold"],
            "reason_codes": feedback_features.get("reason_codes", []),
            "cost_delta_usd": feedback_features.get("cost_delta_usd"),
            "latency_delta_ms": feedback_features.get("latency_delta_ms"),
            "optimization_feedback": feedback_features,
            "managed_feedback": {
                "enabled": False,
                "status": "not-exported",
                "reason": "managed-feedback-not-attempted",
            },
        }
    )
    try:
        event = routing_experiment_outcome_event(feedback_features)
        feedback_meta = await queue_policy_event_feedback(
            context.store,
            event,
            source_surface=ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
            queue_when_disabled=True,
            flush_immediately=False,
        )
    except Exception as exc:
        feedback_meta = {
            "enabled": True,
            "status": "error",
            "reason": "queue-failed",
            "endpoint": "/v1/policy-events",
            "error": repr(exc),
        }
    experiment_meta["managed_feedback"] = {
        "enabled": bool(feedback_meta.get("enabled")),
        "status": feedback_meta.get("status"),
        "reason": feedback_meta.get("reason"),
        "endpoint": feedback_meta.get("endpoint"),
        "queue_id": feedback_meta.get("queue_id"),
        "attempts": feedback_meta.get("attempts"),
        "status_code": feedback_meta.get("status_code"),
        "latency_ms": feedback_meta.get("latency_ms"),
        "source_surface": ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
        "payload_included": False,
    }
    store_bodies = context.log_bodies or ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES
    context.store.log_routing_experiment(
        id=experiment_id,
        call_id=call_id,
        created_at=utc_now(),
        provider=experiment_meta.get("provider") or "anthropic",
        source_surface=experiment_meta.get("source_surface") or "anthropic_messages",
        stream=1 if experiment_meta.get("stream") else 0,
        requested_model=experiment_meta.get("requested_model") or routing_meta.get("requested_model"),
        routed_model=experiment_meta.get("routed_model") or routing_meta.get("routed_model"),
        primary_model=primary_model,
        shadow_model=shadow_model,
        category=routing_meta.get("category"),
        routing_reason=routing_meta.get("reason"),
        input_tokens_est=input_tokens_est,
        primary_status_code=primary_status_code,
        shadow_status_code=shadow_status_code,
        primary_latency_ms=primary_latency_ms,
        shadow_latency_ms=shadow_latency_ms,
        primary_output_chars=comparison["primary_output_chars"],
        shadow_output_chars=comparison["shadow_output_chars"],
        primary_output_sha256=comparison["primary_output_sha256"],
        shadow_output_sha256=comparison["shadow_output_sha256"],
        output_similarity=comparison["output_similarity"],
        passed_threshold=1 if comparison["passed_threshold"] else 0,
        primary_cost_est_usd=primary_cost_est_usd,
        shadow_cost_est_usd=shadow_cost,
        budget_limit_usd=experiment_meta.get("daily_budget_usd"),
        budget_spent_before_usd=experiment_meta.get("budget_spent_usd"),
        budget_remaining_before_usd=experiment_meta.get("budget_remaining_usd"),
        budget_spent_after_usd=round(
            float(experiment_meta.get("budget_spent_usd") or 0.0) + float(shadow_cost or 0.0),
            6,
        ),
        error=error,
        routing_json=stable_json(routing_meta),
        experiment_json=stable_json(experiment_meta),
        primary_response_json=stable_json(primary_response_body) if store_bodies else None,
        shadow_response_json=(
            stable_json(shadow_response_body)
            if store_bodies
            and shadow_response_body is not None
            and shadow_status_code is not None
            and shadow_status_code < 400
            else None
        ),
    )



async def anthropic_messages(context: ProviderContext, request: Request) -> Response:
    if context.provider != "anthropic":
        return provider_disabled_response(context, "anthropic")
    started = time.time()
    call_id = str(uuid.uuid4())
    path = "/v1/messages"
    client_ip = (request.client.host if request.client else "unknown")
    session_id = request.headers.get("x-session-id") or hashlib.sha256(
        (client_ip + datetime.now(timezone.utc).strftime("%Y-%m-%d")).encode()
    ).hexdigest()[:16]
    try:
        raw_body = await read_json_object_body(request)
    except ClientJsonRequestError as exc:
        latency_ms = int((time.time() - started) * 1000)
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=None, routed_model=None, stream=0, cache_hit=0,
            status_code=400, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None,
            actual_input_tokens=None, actual_output_tokens=None,
            cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=None, routing_json=None,
            cache_json=stable_json(cache_decision_meta("skipped", "invalid-json")),
            error=exc.message, request_json=None, response_json=None,
            session_id=session_id, category=None, retry_count=0,
        )
        return JSONResponse(client_json_error_body("anthropic", exc.message), status_code=400)
    stream = bool(raw_body.get("stream"))
    requested_model = str(raw_body.get("model") or "")
    if requested_model in MODEL_ALIASES:
        raw_body["model"] = MODEL_ALIASES[requested_model]
    error: Optional[str] = None
    status_code = 200
    crunch_meta: dict[str, Any] = {}
    routing_meta: dict[str, Any] = {}
    cache_meta: dict[str, Any] = cache_decision_meta("skipped", "not-evaluated")
    cache_hit = False
    response_body: Optional[dict[str, Any]] = None
    retry_count = 0
    net_retries = 0
    summary_extra_cost = 0.0

    category = categorize_request(raw_body)

    try:
        headers = build_forward_headers(request)
        summary_headers = build_summary_headers(request)
        raw_body, summary_meta = await maybe_summarize_old_context(
            raw_body,
            exact_cache_enabled=CACHE_ENABLED,
            get_cached_summary=context.store.get_cache,
            set_cached_summary=lambda key, value: context.store.set_cache(
                key,
                OLD_CONTEXT_SUMMARY_MODEL,
                len(stable_json(value)),
                value,
            ),
            fetch_summary=lambda summary_request: _fetch_old_context_summary(context, summary_request, summary_headers),
            store_obj=context.store,
        )
        summary_extra_cost = float(summary_meta.get("summary_cost_est_usd") or 0.0)
        if summary_meta.get("status") == "applied":
            print(
                "agentflow_old_context_summary "
                f"reason={summary_meta.get('reason')} "
                f"cache_hit={int(bool(summary_meta.get('summary_cache_hit')))} "
                f"cost_est_usd={summary_extra_cost:.6f} "
                f"tokens_saved_est={int(summary_meta.get('tokens_saved_est') or 0)} "
                f"placement={summary_meta.get('placement')}",
                file=sys.stderr,
            )
        preflight_routing_meta = _anthropic_preflight_routing_meta(
            raw_body,
            requested_model=str(raw_body.get("model") or requested_model),
            category=category,
        )
        preflight_cache_meta = cache_decision_meta("skipped", "preflight")
        preflight_crunch_meta = {"old_context_summarization": summary_meta}
        preflight_input_tokens = estimate_tokens_from_text(extract_text(raw_body))
        preflight_recommendation_unit = build_optimization_unit(
            provider="anthropic",
            path=path,
            requested_model=str(raw_body.get("model") or requested_model),
            routed_model=str(raw_body.get("model") or requested_model),
            routing_meta=preflight_routing_meta,
            crunch_meta=preflight_crunch_meta,
            cache_meta=preflight_cache_meta,
            category=category,
            stream=stream,
            input_tokens_est=preflight_input_tokens,
            session_id=session_id,
        )
        preflight_pattern_features = pattern_feature_diagnostics(preflight_recommendation_unit)
        if policy_decisions_enabled():
            recommendation_meta = {
                "enabled": False,
                "status": "skipped",
                "reason": "policy-decision-routing-preflight",
                "fallback": "local-policy",
                "applied": False,
            }
        else:
            recommendation_meta = await fetch_recommendation(preflight_recommendation_unit)
        managed_crunch_profile = _managed_crunch_profile_from_recommendation(recommendation_meta)
        if managed_crunch_profile:
            crunched, crunch_meta = crunch_body(
                raw_body,
                store_obj=context.store,
                managed_profile=managed_crunch_profile,
                routing_meta=preflight_routing_meta,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )
        else:
            crunched, crunch_meta = crunch_body(
                raw_body,
                store_obj=context.store,
                routing_meta=preflight_routing_meta,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint="messages",
            )
        crunch_meta["old_context_summarization"] = summary_meta
        crunched, prompt_cached = inject_prompt_cache(crunched)
        if STRIP_THINKING_HISTORY and category == "tool-result" and not _has_top_level_thinking(crunched):
            tokens_before = estimate_tokens_from_text(extract_text(crunched))
            crunched, _n_stripped = strip_thinking_history_blocks(crunched)
            if _n_stripped > 0:
                tokens_after = estimate_tokens_from_text(extract_text(crunched))
                print(f"strip_thinking_history: blocks={_n_stripped} tokens_before={tokens_before} tokens_after={tokens_after}", flush=True)
        else:
            _n_stripped = 0
        routed_model, routing_meta = route_model(crunched, session_id=session_id)
        if _n_stripped > 0:
            routing_meta["thinking_history_stripped"] = _n_stripped
        resolved_requested_model = crunched.get("model", requested_model)
        experiment_meta = routing_experiment_decision(
            crunched,
            {
                **routing_meta,
                "requested_model": str(resolved_requested_model),
                "routed_model": str(resolved_requested_model),
            },
            stream=stream,
            provider="anthropic",
            source_surface="anthropic_messages",
            store_obj=context.store,
        )
        sampled_shadow_pass_through = (
            experiment_meta.get("mode") == "shadow_candidate_pass_through"
            and bool(experiment_meta.get("sampled"))
        )
        if experiment_meta.get("mode") == "shadow_candidate_pass_through":
            routing_meta["routing_experiment"] = experiment_meta
        if sampled_shadow_pass_through:
            experiment_meta["local_route_candidate_model"] = routed_model
            experiment_meta["local_route_candidate_reason"] = routing_meta.get("reason")
            experiment_meta["primary_route_override_reason"] = "shadow-candidate-pass-through-primary-requested"
            routing_meta["local_route_candidate_model"] = routed_model
            routing_meta["local_route_candidate_reason"] = routing_meta.get("reason")
            routing_meta["reason"] = "shadow experiment pass-through primary kept on requested model"
            routed_model = str(resolved_requested_model)
            routing_meta["routed_model"] = routed_model
        crunched["model"] = routed_model
        _strip_model_incompatible_params(crunched, routing_meta, str(resolved_requested_model))
        _thinking_param = crunched.get("thinking")
        if (
            MAX_THINKING_BUDGET_TOKENS > 0
            and isinstance(_thinking_param, dict)
            and isinstance(_thinking_param.get("budget_tokens"), int)
            and _thinking_param["budget_tokens"] > MAX_THINKING_BUDGET_TOKENS
        ):
            _original_budget = _thinking_param["budget_tokens"]
            crunched["thinking"]["budget_tokens"] = MAX_THINKING_BUDGET_TOKENS
            routing_meta["thinking_capped"] = True
            print(f"thinking_cap: original={_original_budget} cap={MAX_THINKING_BUDGET_TOKENS}", flush=True)
        crunched_text = extract_text(crunched)
        routing_meta["prompt_difficulty_features"] = prompt_difficulty_features_from_text(crunched_text)
        input_tokens = estimate_tokens_from_text(crunched_text)
        recommendation_unit = build_optimization_unit(
            provider="anthropic",
            path=path,
            requested_model=str(resolved_requested_model),
            routed_model=str(crunched.get("model") or routed_model),
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            category=category,
            stream=stream,
            input_tokens_est=input_tokens,
            session_id=session_id,
        )
        routing_meta["managed_pattern_features"] = pattern_feature_diagnostics(recommendation_unit)
        routing_meta["managed_preflight_pattern_features"] = preflight_pattern_features
        if policy_decisions_enabled():
            recommendation_meta = await fetch_policy_decision(recommendation_unit)
        recommendation_meta = apply_recommendation_to_body(
            provider="anthropic",
            body=crunched,
            routing_meta=routing_meta,
            recommendation_meta=recommendation_meta,
        )
        if crunched.get("model") in MODEL_ALIASES:
            normalized_model = MODEL_ALIASES[str(crunched.get("model"))]
            recommendation_meta["target_model_normalized"] = normalized_model
            crunched["model"] = normalized_model
            routing_meta["routed_model"] = normalized_model
        routing_meta["managed_recommendation"] = recommendation_meta
        if sampled_shadow_pass_through:
            managed_candidate_model = str(crunched.get("model") or "")
            if managed_candidate_model and managed_candidate_model != str(resolved_requested_model):
                experiment_meta["managed_route_candidate_model"] = managed_candidate_model
                experiment_meta["managed_route_candidate_reason"] = recommendation_meta.get("reason")
                experiment_meta["managed_route_override_reason"] = "shadow-candidate-pass-through-primary-requested"
                recommendation_meta["applied"] = False
                recommendation_meta["changed_model"] = False
                recommendation_meta["apply_reason"] = "shadow-experiment-primary-pass-through"
                recommendation_meta["fallback"] = "local-policy"
                crunched["model"] = str(resolved_requested_model)
            routing_meta["routed_model"] = str(resolved_requested_model)
        _strip_model_incompatible_params(crunched, routing_meta, str(resolved_requested_model))
        if prompt_cached or has_cache_control_blocks(crunched):
            existing = headers.get("anthropic-beta", "")
            if "prompt-caching" not in existing:
                headers["anthropic-beta"] = (existing + ",prompt-caching-2024-07-31" if existing else "prompt-caching-2024-07-31")

        if stream:
            has_tool_blocks = has_tools(crunched)
            has_thinking_blocks = uses_thinking(crunched)
            can_stream_cache, cache_meta = streaming_cache_lookup_meta(
                has_tool_blocks,
                has_thinking_blocks=has_thinking_blocks,
                pattern_features=routing_meta.get("managed_pattern_features"),
                store_obj=context.store,
            )
            file_deps = cache_file_dependency_snapshots(crunched) if (can_stream_cache or has_tool_blocks) else []
            if can_stream_cache or has_tool_blocks:
                cache_meta["file_dependency_audit"] = cache_file_dependency_audit(crunched)
                cache_meta["file_dependency_count"] = cache_meta["file_dependency_audit"]["snapshot_count"]
                cache_meta["file_dependency_evidence_available"] = bool(
                    cache_meta["file_dependency_audit"]["file_dependency_evidence_available"]
                )
                cache_meta["safe_invalidation_evidence"] = bool(
                    cache_meta["file_dependency_audit"]["safe_invalidation_evidence"]
                )
            _attach_session_memory_hints(
                context=context,
                session_id=session_id,
                stream=True,
                has_tool_blocks=has_tool_blocks,
                category=category,
                text_chars=input_tokens * TOKEN_CHARS,
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                current_thinking=_has_top_level_thinking(crunched),
            )
            coordinator_enforcement = enforce_optimization_coordinator(
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                provider="anthropic",
                source_surface="anthropic_messages",
                endpoint=path,
                requested_model=str(resolved_requested_model),
                routed_model=str(crunched.get("model") or routed_model),
                input_tokens_est=input_tokens,
                category=category,
                stream=True,
                session_id=session_id,
                provider_body=crunched,
                local_routed_model=str(routed_model),
            )
            if "cache_replay" in coordinator_enforcement.get("suppressed_managed_families", []):
                can_stream_cache = False
            await _queue_optimization_coordinator_lifecycle_feedback(
                context,
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                enforcement=coordinator_enforcement,
            )
            replay_scope, replay_scope_id, replay_pattern_rule = cache_replay_scope_for_meta(cache_meta, session_id)
            if replay_pattern_rule is not None:
                cache_meta["replay_scope"] = replay_scope
                cache_meta["replay_scope_id_available"] = bool(replay_scope_id)
            if can_stream_cache and replay_pattern_rule is not None:
                min_call_count = max(1, int(replay_pattern_rule.get("min_call_count") or 1))
                if min_call_count > 1:
                    prior_observed = context.store.cache_pattern_observed_call_count(
                        rule_id=str(replay_pattern_rule.get("rule_id") or ""),
                        matched_hashes=replay_pattern_rule.get("matched_hashes") or [],
                        requested_model=requested_model,
                        routed_model=str(crunched.get("model") or ""),
                        category=category,
                        stream=True,
                    )
                    observed_count = prior_observed + 1
                    cache_meta["pattern_rule_warmup"] = {
                        "schema": "agentflow.cache_pattern_warmup.v1",
                        "rule_id": replay_pattern_rule.get("rule_id"),
                        "candidate_id": replay_pattern_rule.get("candidate_id"),
                        "policy_source": replay_pattern_rule.get("policy_source"),
                        "min_call_count": min_call_count,
                        "prior_observed_call_count": prior_observed,
                        "observed_call_count": observed_count,
                        "met": observed_count >= min_call_count,
                        "metadata_only": True,
                    }
                    if observed_count < min_call_count:
                        can_stream_cache = False
                        cache_meta["status"] = "skipped"
                        cache_meta["reason"] = "streaming-min-call-count-not-met"
            stream_cache_key_variants = _cache_key_variants_for_models(
                crunched,
                path,
                provider="anthropic",
                upstream=context.anthropic_upstream,
                replay_scope=replay_scope,
                replay_scope_id=replay_scope_id,
                requested_model=requested_model,
            )
            key = stream_cache_key_variants[0][0]
            if len(stream_cache_key_variants) > 1:
                cache_meta["cache_key_variant_count"] = len(stream_cache_key_variants)
            if can_stream_cache:
                replay_allowed = True
                if replay_pattern_rule is not None:
                    dependency_audit = context.store.cache_file_dependency_audit(key)
                    replay_allowed, replay_canary = cache_replay_canary_decision(
                        cache_meta=cache_meta,
                        dependency_audit=dependency_audit,
                        session_id=session_id,
                    )
                    cache_meta["cache_replay_canary"] = replay_canary
                    if not replay_allowed:
                        can_stream_cache = False
                        cache_meta["status"] = replay_canary.get("status", "bypassed")
                        cache_meta["reason"] = str(replay_canary.get("reason") or "cache-replay-canary-bypassed")
                        if replay_canary.get("status") == "invalidated":
                            cache_meta["invalidated"] = True
                            cache_meta["invalidation_reason"] = str(replay_canary.get("reason") or "dependency-invalidated")
                            context.store.delete_cache(key)
                cached = None
            if can_stream_cache:
                cached = None
                invalidated_reason = None
                for lookup_key, _lookup_body in stream_cache_key_variants:
                    cached, invalidated_reason = context.store.get_cache_with_reason(lookup_key)
                    if cached is not None or invalidated_reason:
                        key = lookup_key
                        break
                if invalidated_reason:
                    cache_meta["reason"] = invalidated_reason
                    cache_meta["invalidated"] = True
                    cache_meta["invalidation_reason"] = invalidated_reason
                cached_frames, stream_validation = validate_stream_cache_payload(cached, provider="anthropic")
                if cached_frames:
                    cached_usage = cached.get("usage") or {}
                    cached_output_text = str(cached.get("output_text") or "")

                    async def replay_cached_stream() -> AsyncIterator[bytes]:
                        try:
                            for frame in cached_frames:
                                yield frame
                        finally:
                            latency_ms = int((time.time() - started) * 1000)
                            cached_out = cached_usage.get("output_tokens")
                            out_tokens = (
                                int(cached_out)
                                if isinstance(cached_out, int)
                                else estimate_tokens_from_text(cached_output_text)
                            )
                            cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens)
                            hit_cache_meta = cache_hit_decision_meta(
                                "streaming-exact-match",
                                hit_type="streaming-exact",
                                exact_enabled=can_stream_cache,
                                semantic_enabled=False,
                                lookup_meta=cache_meta,
                                estimated_saved_cost_usd=cost_baseline,
                            )
                            hit_cache_meta["stream_replay"] = {
                                key: value
                                for key, value in stream_validation.items()
                                if key not in {"event_names"}
                            }
                            context.store.log_call(
                                id=call_id, created_at=utc_now(), path=path,
                                requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                                cache_hit=1, status_code=200, latency_ms=latency_ms,
                                input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                                actual_input_tokens=None, actual_output_tokens=None,
                                cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                                cache_json=stable_json(hit_cache_meta),
                                error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                                response_json=stable_json(cached) if context.log_bodies else None,
                                session_id=session_id, category=category,
                                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                                retry_count=0,
                            )
                            await _record_managed_outcome_feedback(
                                context=context,
                                call_id=call_id,
                                path=path,
                                requested_model=requested_model,
                                routed_model=str(crunched.get("model")),
                                status_code=200,
                                latency_ms=latency_ms,
                                retry_count=0,
                                input_tokens_est=input_tokens,
                                output_tokens_est=out_tokens,
                                actual_input_tokens=None,
                                actual_output_tokens=None,
                                cache_creation_input_tokens=0,
                                cache_read_input_tokens=0,
                                thinking_output_tokens=None,
                                cost_est_usd=summary_extra_cost,
                                cost_baseline_usd=cost_baseline,
                                cache_meta=hit_cache_meta,
                                crunch_meta=crunch_meta,
                                routing_meta=routing_meta,
                                category=category,
                                stream=True,
                                session_id=session_id,
                            )
                            await _check_session_cost_alert(context, session_id)

                    return StreamingResponse(
                        replay_cached_stream(),
                        media_type="text/event-stream",
                        headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))},
                    )
                if cached is not None:
                    cache_meta["status"] = "bypassed"
                    cache_meta["reason"] = "malformed-stream-cache"
                    cache_meta["malformed_stream_cache"] = stream_validation
                    context.store.delete_cache(key)

            async def gen() -> AsyncIterator[bytes]:
                nonlocal status_code, error
                actual_in: Optional[int] = None
                actual_out: Optional[int] = None
                cache_creation_in: int = 0
                cache_read_in: int = 0
                thinking_chars: int = 0
                stream_frames: list[bytes] = []
                upstream_error_chunks: list[bytes] = []
                output_text_parts: list[str] = []
                sse_frame_buf = b""
                stream_retry_count = 0
                stream_net_retries = 0
                stream_complete = False
                stream_cancelled = False
                stream_tool_use_ids: list[str] = []
                stream_tool_use_missing_ids = 0

                def parse_sse_usage(frame: bytes) -> None:
                    nonlocal actual_in, actual_out, cache_creation_in, cache_read_in, thinking_chars, stream_complete, stream_tool_use_missing_ids
                    for line in frame.decode("utf-8", errors="replace").splitlines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            continue
                        try:
                            data = json.loads(payload)
                        except Exception:
                            continue
                        t = data.get("type")
                        if t == "message_start":
                            u = (data.get("message") or {}).get("usage", {})
                            actual_in = u.get("input_tokens")
                            cache_creation_in = u.get("cache_creation_input_tokens") or 0
                            cache_read_in = u.get("cache_read_input_tokens") or 0
                        elif t == "message_stop":
                            stream_complete = True
                        elif t == "message_delta":
                            out = (data.get("usage") or {}).get("output_tokens")
                            if out is not None:
                                actual_out = out
                        elif t == "content_block_start":
                            block = data.get("content_block") or {}
                            if isinstance(block, dict) and block.get("type") == "tool_use":
                                tool_id = block.get("id")
                                if tool_id:
                                    stream_tool_use_ids.append(str(tool_id))
                                else:
                                    stream_tool_use_missing_ids += 1
                        elif t == "content_block_delta":
                            delta = (data.get("delta") or {})
                            if delta.get("type") == "thinking_delta":
                                thinking_chars += len(delta.get("thinking") or "")
                            elif delta.get("type") == "text_delta":
                                output_text_parts.append(str(delta.get("text") or ""))

                try:
                    async with context.limiter.semaphores[model_tier(crunched["model"])]:
                        async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                            while True:
                                await context.limiter.await_backoff(crunched["model"])
                                await context.limiter.throttle_forward()
                                try:
                                    async with client.stream("POST", context.anthropic_upstream.rstrip("/") + path, headers=headers, json=crunched) as r:
                                        status_code = r.status_code
                                        if status_code in (429, 529) and stream_retry_count < 3:
                                            stream_retry_count += 1
                                            delay = (2 ** (stream_retry_count - 1)) * (1.0 + random.random() * 0.5)
                                            print(f"rate_limit: status={status_code} retry={stream_retry_count} delay={delay:.1f}s")
                                            await context.limiter.record_backoff(crunched["model"], r.headers)
                                            if stream_retry_count == 1 and crunched.get("model") != resolved_requested_model:
                                                _rate_limited_model = crunched.get("model")
                                                crunched["model"] = resolved_requested_model
                                                _record_routing_rate_limit_fallback(
                                                    routing_meta,
                                                    requested_model=str(resolved_requested_model),
                                                    from_model=_rate_limited_model,
                                                )
                                                print(f"rate_limit_fallback: routing {_rate_limited_model!r} -> {resolved_requested_model!r}")
                                            await asyncio.sleep(delay)
                                            continue
                                        async for chunk in r.aiter_bytes():
                                            sse_frame_buf += chunk
                                            while b"\n\n" in sse_frame_buf:
                                                frame, sse_frame_buf = sse_frame_buf.split(b"\n\n", 1)
                                                event_bytes = frame + b"\n\n"
                                                stream_frames.append(event_bytes)
                                                if status_code >= 400:
                                                    upstream_error_chunks.append(event_bytes)
                                                yield event_bytes
                                                parse_sse_usage(frame)
                                        if sse_frame_buf:
                                            stream_frames.append(sse_frame_buf)
                                            if status_code >= 400:
                                                upstream_error_chunks.append(sse_frame_buf)
                                            yield sse_frame_buf
                                            parse_sse_usage(sse_frame_buf)
                                            sse_frame_buf = b""
                                        break
                                except httpx.NetworkError as exc:
                                    if stream_net_retries < 2:
                                        stream_net_retries += 1
                                        print(f"network_error: {exc!r} retry={stream_net_retries}", flush=True)
                                        await asyncio.sleep(2.0)
                                        continue
                                    raise
                except TierBackoffActive as exc:
                    status_code = 429
                    error = exc.message
                    payload = tier_backoff_payload(exc)
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                except asyncio.CancelledError:
                    stream_cancelled = True
                    raise
                except Exception as exc:
                    logging.exception("agentflow anthropic streaming proxy error")
                    status_code = 500
                    error = repr(exc)
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(public_proxy_error_body('anthropic', exc))}\n\n"
                    ).encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    cost_in = actual_in if actual_in is not None else input_tokens
                    cost_out = actual_out if actual_out is not None else 0
                    cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_creation_in, cache_read_in)
                    if cost is not None:
                        cost += summary_extra_cost
                    cost_baseline = estimate_cost(requested_model, cost_in + cache_creation_in + cache_read_in, cost_out)
                    if cache_creation_in or cache_read_in:
                        print(f"prompt_cache: creation={cache_creation_in} read={cache_read_in}")
                    if status_code >= 400 and error is None:
                        error = upstream_error_text(b"".join(upstream_error_chunks), status_code)
                    if status_code < 400 and error is None and can_stream_cache and stream_frames:
                        stream_usage = {
                            "input_tokens": actual_in,
                            "output_tokens": actual_out,
                            "cache_creation_input_tokens": cache_creation_in,
                            "cache_read_input_tokens": cache_read_in,
                            "thinking_output_tokens": thinking_chars // TOKEN_CHARS if thinking_chars else None,
                        }
                        stream_payload = stream_cache_payload(
                            stream_frames,
                            provider="anthropic",
                            usage=stream_usage,
                            output_text="".join(output_text_parts),
                        )
                        stored_stream_cache_entries = 0
                        for store_key, store_body in stream_cache_key_variants:
                            context.store.set_cache(
                                store_key,
                                str(store_body.get("model")),
                                len(stable_json(store_body)),
                                stream_payload,
                                file_deps=file_deps,
                            )
                            stored_stream_cache_entries += 1
                        cache_meta["stream_cache_store"] = {
                            "status": "stored" if stored_stream_cache_entries else "not-stored",
                            "entry_count": stored_stream_cache_entries,
                            "file_dependency_count": len(file_deps),
                            "cache_keys_included": False,
                            "response_body_included": False,
                            "sse_frames_included": False,
                        }
                    thinking_tokens = thinking_chars // TOKEN_CHARS if thinking_chars else None
                    experiment_meta = routing_meta.get("routing_experiment")
                    if (
                        isinstance(experiment_meta, dict)
                        and experiment_meta.get("mode") == "shadow_candidate_pass_through"
                        and experiment_meta.get("sampled")
                    ):
                        if stream_cancelled:
                            _mark_streaming_experiment_skip(
                                routing_meta,
                                reason="streaming-cancelled",
                                stream_complete=stream_complete,
                                status_code=status_code,
                                error=error,
                            )
                        elif status_code >= 400 or error is not None:
                            _mark_streaming_experiment_skip(
                                routing_meta,
                                reason="streaming-primary-error",
                                stream_complete=stream_complete,
                                status_code=status_code,
                                error=error,
                            )
                        elif not stream_complete:
                            _mark_streaming_experiment_skip(
                                routing_meta,
                                reason="streaming-incomplete",
                                stream_complete=stream_complete,
                                status_code=status_code,
                                error=error,
                            )
                        else:
                            experiment_meta["streaming"] = {
                                "complete": True,
                                "primary_status_code": status_code,
                                "primary_error_present": False,
                                "raw_stream_included": False,
                            }
                            primary_response_body = _anthropic_stream_primary_response_body(
                                output_text="".join(output_text_parts),
                                input_tokens=actual_in,
                                output_tokens=actual_out,
                                cache_creation_input_tokens=cache_creation_in,
                                cache_read_input_tokens=cache_read_in,
                                thinking_output_tokens=thinking_tokens,
                            )
                            await _run_anthropic_routing_experiment(
                                context=context,
                                call_id=call_id,
                                path=path,
                                headers=headers,
                                request_body=crunched,
                                routing_meta=routing_meta,
                                experiment_meta=experiment_meta,
                                primary_response_body=primary_response_body,
                                primary_status_code=status_code,
                                primary_latency_ms=latency_ms,
                                primary_cost_est_usd=cost,
                                input_tokens_est=input_tokens,
                            )
                            if experiment_meta.get("status") in {"shadow-error", "shadow-unavailable"}:
                                experiment_meta["reason"] = "streaming-shadow-error"
                            elif experiment_meta.get("status") == "shadow-http-400":
                                experiment_meta["reason"] = "streaming-shadow-http-400"
                            elif experiment_meta.get("status") == "shadow-unsupported-shape":
                                experiment_meta["reason"] = "streaming-shadow-unsupported-shape"
                    attach_observed_savings_to_routing_meta(
                        routing_meta,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost_baseline,
                        status_code=status_code,
                    )
                    context.store.log_call(
                        id=call_id, created_at=utc_now(), path=path,
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_meta),
                        error=error, request_json=stable_json(crunched) if context.log_bodies else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=thinking_tokens,
                    )
                    capture_provider_tool_adoption(
                        context.store,
                        provider="anthropic",
                        path=path,
                        call_id=call_id,
                        session_id=session_id,
                        request_body=crunched,
                        response_tool_use_ids=stream_tool_use_ids,
                        response_tool_use_missing_ids=stream_tool_use_missing_ids,
                        requested_model=requested_model,
                        routed_model=str(crunched.get("model")),
                        status_code=status_code,
                        category=category,
                        routing_meta=routing_meta,
                        crunch_meta=crunch_meta,
                        cache_meta=cache_meta,
                    )
                    await _record_managed_outcome_feedback(
                        context=context,
                        call_id=call_id,
                        path=path,
                        requested_model=requested_model,
                        routed_model=str(crunched.get("model")),
                        status_code=status_code,
                        latency_ms=latency_ms,
                        retry_count=stream_retry_count,
                        input_tokens_est=input_tokens,
                        output_tokens_est=None,
                        actual_input_tokens=actual_in,
                        actual_output_tokens=actual_out,
                        cache_creation_input_tokens=cache_creation_in,
                        cache_read_input_tokens=cache_read_in,
                        thinking_output_tokens=thinking_tokens,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost_baseline,
                        cache_meta=cache_meta,
                        crunch_meta=crunch_meta,
                        routing_meta=routing_meta,
                        category=category,
                        stream=True,
                        session_id=session_id,
                        error=error,
                    )
                    await _check_session_cost_alert(context, session_id)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-agentflow-cache": "miss" if can_stream_cache else "skip-streaming", "x-agentflow-routed-model": str(crunched.get("model"))},
            )

        has_tool_blocks = has_tools(crunched)
        can_cache, can_semantic_cache, cache_meta = cache_lookup_meta(
            has_tool_blocks,
            pattern_features=routing_meta.get("managed_pattern_features"),
            store_obj=context.store,
        )
        file_deps = cache_file_dependency_snapshots(crunched) if (can_cache or has_tool_blocks) else []
        if can_cache or has_tool_blocks:
            cache_meta["file_dependency_audit"] = cache_file_dependency_audit(crunched)
            cache_meta["file_dependency_count"] = cache_meta["file_dependency_audit"]["snapshot_count"]
            cache_meta["file_dependency_evidence_available"] = bool(
                cache_meta["file_dependency_audit"]["file_dependency_evidence_available"]
            )
            cache_meta["safe_invalidation_evidence"] = bool(
                cache_meta["file_dependency_audit"]["safe_invalidation_evidence"]
            )
        _attach_session_memory_hints(
            context=context,
            session_id=session_id,
            stream=False,
            has_tool_blocks=has_tool_blocks,
            category=category,
            text_chars=input_tokens * TOKEN_CHARS,
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            current_thinking=_has_top_level_thinking(crunched),
        )
        coordinator_enforcement = enforce_optimization_coordinator(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint=path,
            requested_model=str(resolved_requested_model),
            routed_model=str(crunched.get("model") or routed_model),
            input_tokens_est=input_tokens,
            category=category,
            stream=False,
            session_id=session_id,
            provider_body=crunched,
            local_routed_model=str(routed_model),
        )
        if "cache_replay" in coordinator_enforcement.get("suppressed_managed_families", []):
            can_cache = False
            can_semantic_cache = False
        await _queue_optimization_coordinator_lifecycle_feedback(
            context,
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            enforcement=coordinator_enforcement,
        )
        replay_scope, replay_scope_id, replay_pattern_rule = cache_replay_scope_for_meta(cache_meta, session_id)
        if replay_pattern_rule is not None:
            cache_meta["replay_scope"] = replay_scope
            cache_meta["replay_scope_id_available"] = bool(replay_scope_id)
        key = cache_key_for(
            crunched,
            path,
            provider="anthropic",
            upstream=context.anthropic_upstream,
            replay_scope=replay_scope,
            replay_scope_id=replay_scope_id,
        )
        emb: Optional[list[float]] = None
        if can_cache:
            replay_allowed = True
            if replay_pattern_rule is not None:
                dependency_audit = context.store.cache_file_dependency_audit(key)
                replay_allowed, replay_canary = cache_replay_canary_decision(
                    cache_meta=cache_meta,
                    dependency_audit=dependency_audit,
                    session_id=session_id,
                )
                cache_meta["cache_replay_canary"] = replay_canary
                if not replay_allowed:
                    cache_meta["status"] = replay_canary.get("status", "bypassed")
                    cache_meta["reason"] = str(replay_canary.get("reason") or "cache-replay-canary-bypassed")
                    if replay_canary.get("status") == "invalidated":
                        cache_meta["invalidated"] = True
                        cache_meta["invalidation_reason"] = str(replay_canary.get("reason") or "dependency-invalidated")
                        context.store.delete_cache(key)
            cached = None
            if replay_allowed:
                cached, invalidated_reason = context.store.get_cache_with_reason(key)
                if invalidated_reason:
                    cache_meta["reason"] = invalidated_reason
                    cache_meta["invalidated"] = True
                    cache_meta["invalidation_reason"] = invalidated_reason
            if cached is not None:
                cache_hit = True
                response_body = cached
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(response_body))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens)
                hit_cache_meta = cache_hit_decision_meta(
                    "exact-match",
                    hit_type="exact",
                                exact_enabled=can_cache,
                                semantic_enabled=can_semantic_cache,
                                lookup_meta=cache_meta,
                                estimated_saved_cost_usd=(
                                    cost_baseline - summary_extra_cost
                                    if cost_baseline is not None
                                    else None
                                ),
                            )
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path,
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(hit_cache_meta),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(response_body) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                capture_provider_tool_adoption(
                    context.store,
                    provider="anthropic",
                    path=path,
                    call_id=call_id,
                    session_id=session_id,
                    request_body=crunched,
                    response_body=response_body,
                    requested_model=requested_model,
                    routed_model=str(crunched.get("model")),
                    status_code=200,
                    category=category,
                    routing_meta=routing_meta,
                    crunch_meta=crunch_meta,
                    cache_meta=hit_cache_meta,
                )
                await _record_managed_outcome_feedback(
                    context=context,
                    call_id=call_id,
                    path=path,
                    requested_model=requested_model,
                    routed_model=str(crunched.get("model")),
                    status_code=200,
                    latency_ms=latency_ms,
                    retry_count=0,
                    input_tokens_est=input_tokens,
                    output_tokens_est=out_tokens,
                    actual_input_tokens=None,
                    actual_output_tokens=None,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=None,
                    cost_est_usd=summary_extra_cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    stream=False,
                    session_id=session_id,
                )
                return JSONResponse(response_body, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        if can_semantic_cache:
            emb = build_embedding(extract_text(crunched))
            sem_resp = context.store.get_semantic_cache(emb, str(crunched.get("model")), SEMANTIC_CACHE_THRESHOLD)
            if sem_resp is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(sem_resp))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens)
                hit_cache_meta = cache_hit_decision_meta(
                    "semantic-match",
                    hit_type="semantic",
                                exact_enabled=can_cache,
                                semantic_enabled=can_semantic_cache,
                                lookup_meta=cache_meta,
                                estimated_saved_cost_usd=(
                                    cost_baseline - summary_extra_cost
                                    if cost_baseline is not None
                                    else None
                                ),
                            )
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path,
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(hit_cache_meta),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(sem_resp) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                capture_provider_tool_adoption(
                    context.store,
                    provider="anthropic",
                    path=path,
                    call_id=call_id,
                    session_id=session_id,
                    request_body=crunched,
                    response_body=sem_resp,
                    requested_model=requested_model,
                    routed_model=str(crunched.get("model")),
                    status_code=200,
                    category=category,
                    routing_meta=routing_meta,
                    crunch_meta=crunch_meta,
                    cache_meta=hit_cache_meta,
                )
                await _record_managed_outcome_feedback(
                    context=context,
                    call_id=call_id,
                    path=path,
                    requested_model=requested_model,
                    routed_model=str(crunched.get("model")),
                    status_code=200,
                    latency_ms=latency_ms,
                    retry_count=0,
                    input_tokens_est=input_tokens,
                    output_tokens_est=out_tokens,
                    actual_input_tokens=None,
                    actual_output_tokens=None,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=None,
                    cost_est_usd=summary_extra_cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    stream=False,
                    session_id=session_id,
                )
                return JSONResponse(sem_resp, headers={"x-agentflow-cache": "semantic-hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with context.limiter.semaphores[model_tier(crunched["model"])]:
            async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                while True:
                    await context.limiter.await_backoff(crunched["model"])
                    await context.limiter.throttle_forward()
                    try:
                        r = await client.post(context.anthropic_upstream.rstrip("/") + path, headers=headers, json=crunched)
                    except httpx.NetworkError as exc:
                        if net_retries < 2:
                            net_retries += 1
                            print(f"network_error: {exc!r} retry={net_retries}", flush=True)
                            await asyncio.sleep(2.0)
                            continue
                        raise
                    if r.status_code in (429, 529) and retry_count < 3:
                        retry_count += 1
                        delay = (2 ** (retry_count - 1)) * (1.0 + random.random() * 0.5)
                        print(f"rate_limit: status={r.status_code} retry={retry_count} delay={delay:.1f}s")
                        await context.limiter.record_backoff(crunched["model"], r.headers)
                        if retry_count == 1 and crunched.get("model") != resolved_requested_model:
                            _rate_limited_model = crunched.get("model")
                            crunched["model"] = resolved_requested_model
                            _record_routing_rate_limit_fallback(
                                routing_meta,
                                requested_model=str(resolved_requested_model),
                                from_model=_rate_limited_model,
                            )
                            print(f"rate_limit_fallback: routing {_rate_limited_model!r} -> {resolved_requested_model!r}")
                        await asyncio.sleep(delay)
                        continue
                    break
        status_code = r.status_code
        try:
            response_body = r.json()
        except Exception:
            latency_ms = int((time.time() - started) * 1000)
            cost = estimate_cost(str(crunched.get("model")), input_tokens, 0)
            if cost is not None:
                cost += summary_extra_cost
            cost_baseline = estimate_cost(requested_model, input_tokens, 0)
            error = upstream_error_text(r.text, status_code)
            context.store.log_call(
                id=call_id, created_at=utc_now(), path=path,
                requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                input_tokens_est=input_tokens, output_tokens_est=None,
                actual_input_tokens=None, actual_output_tokens=None,
                cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                cache_json=stable_json(cache_meta),
                error=error,
                request_json=stable_json(crunched) if context.log_bodies else None, response_json=None,
                session_id=session_id, category=category,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                retry_count=retry_count,
            )
            await _record_managed_outcome_feedback(
                context=context,
                call_id=call_id,
                path=path,
                requested_model=requested_model,
                routed_model=str(crunched.get("model")),
                status_code=status_code,
                latency_ms=latency_ms,
                retry_count=retry_count,
                input_tokens_est=input_tokens,
                output_tokens_est=None,
                actual_input_tokens=None,
                actual_output_tokens=None,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                thinking_output_tokens=None,
                cost_est_usd=cost,
                cost_baseline_usd=cost_baseline,
                cache_meta=cache_meta,
                crunch_meta=crunch_meta,
                routing_meta=routing_meta,
                category=category,
                stream=False,
                session_id=session_id,
                error=error,
            )
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/plain"))

        if r.status_code < 400 and can_cache and response_body is not None:
            context.store.set_cache(
                key,
                str(crunched.get("model")),
                len(stable_json(crunched)),
                response_body,
                file_deps=file_deps,
            )
        if can_semantic_cache and emb is not None and r.status_code < 400 and response_body is not None:
            context.store.set_semantic_cache(key, str(crunched.get("model")), emb, response_body, len(stable_json(crunched)))

        usage = (response_body or {}).get("usage") or {}
        actual_in = usage.get("input_tokens")
        actual_out = usage.get("output_tokens")
        cache_creation_in = usage.get("cache_creation_input_tokens") or 0
        cache_read_in = usage.get("cache_read_input_tokens") or 0
        if cache_creation_in or cache_read_in:
            print(f"prompt_cache: creation={cache_creation_in} read={cache_read_in}")
        thinking_chars = _count_thinking_chars(response_body) if response_body else 0
        out_tokens = estimate_tokens_from_text(response_output_text(response_body)) if response_body else 0
        cost_in = actual_in if actual_in is not None else input_tokens
        cost_out = actual_out if actual_out is not None else out_tokens
        cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_creation_in, cache_read_in)
        if cost is not None:
            cost += summary_extra_cost
        cost_baseline = estimate_cost(requested_model, cost_in + cache_creation_in + cache_read_in, cost_out)
        latency_ms = int((time.time() - started) * 1000)
        experiment_meta = routing_meta.get("routing_experiment")
        if not isinstance(experiment_meta, dict) or experiment_meta.get("mode") != "shadow_candidate_pass_through":
            experiment_meta = routing_experiment_decision(
                crunched,
                routing_meta,
                stream=False,
                provider="anthropic",
                source_surface="anthropic_messages",
                store_obj=context.store,
            )
        routing_meta["routing_experiment"] = experiment_meta
        if experiment_meta.get("sampled") and status_code < 400 and response_body is not None:
            await _run_anthropic_routing_experiment(
                context=context,
                call_id=call_id,
                path=path,
                headers=headers,
                request_body=crunched,
                routing_meta=routing_meta,
                experiment_meta=experiment_meta,
                primary_response_body=response_body,
                primary_status_code=status_code,
                primary_latency_ms=latency_ms,
                primary_cost_est_usd=cost,
                input_tokens_est=input_tokens,
            )
        error = None if status_code < 400 else upstream_error_text(response_body, status_code)
        thinking_tokens = thinking_chars // TOKEN_CHARS if thinking_chars else None
        attach_observed_savings_to_routing_meta(
            routing_meta,
            cost_est_usd=cost,
            cost_baseline_usd=cost_baseline,
            status_code=status_code,
        )
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error,
            request_json=stable_json(crunched) if context.log_bodies else None,
            response_json=stable_json(response_body) if context.log_bodies else None,
            session_id=session_id, category=category,
            cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
            thinking_output_tokens=thinking_tokens,
        )
        capture_provider_tool_adoption(
            context.store,
            provider="anthropic",
            path=path,
            call_id=call_id,
            session_id=session_id,
            request_body=crunched,
            response_body=response_body,
            requested_model=requested_model,
            routed_model=str(crunched.get("model")),
            status_code=status_code,
            category=category,
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
        )
        await _record_managed_outcome_feedback(
            context=context,
            call_id=call_id,
            path=path,
            requested_model=requested_model,
            routed_model=str(crunched.get("model")),
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
            input_tokens_est=input_tokens,
            output_tokens_est=out_tokens,
            actual_input_tokens=actual_in,
            actual_output_tokens=actual_out,
            cache_creation_input_tokens=cache_creation_in,
            cache_read_input_tokens=cache_read_in,
            thinking_output_tokens=thinking_tokens,
            cost_est_usd=cost,
            cost_baseline_usd=cost_baseline,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            stream=False,
            session_id=session_id,
            error=error,
        )
        await _check_session_cost_alert(context, session_id)
        return JSONResponse(response_body, status_code=status_code, headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))})

    except TierBackoffActive as exc:
        routed_model_for_log: Optional[str] = None
        try:
            routed_model_for_log = str(crunched.get("model"))
        except Exception:
            routed_model_for_log = None
        error = exc.message
        status_code = 429
        response_body = tier_backoff_payload(exc)
        latency_ms = int((time.time() - started) * 1000)
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=routed_model_for_log, stream=int(stream), cache_hit=0,
            status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if context.log_bodies else None,
            response_json=stable_json(response_body) if context.log_bodies else None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        await _record_managed_outcome_feedback(
            context=context,
            call_id=call_id,
            path=path,
            requested_model=requested_model,
            routed_model=routed_model_for_log,
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
            input_tokens_est=locals().get("input_tokens"),
            output_tokens_est=None,
            actual_input_tokens=None,
            actual_output_tokens=None,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=None,
            cost_baseline_usd=None,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            stream=stream,
            session_id=session_id,
            error=error,
        )
        return JSONResponse(
            response_body,
            status_code=status_code,
            headers=tier_backoff_headers(exc, routed_model_for_log or ""),
        )
    except Exception as exc:
        logging.exception("agentflow anthropic proxy error")
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if context.log_bodies else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        await _record_managed_outcome_feedback(
            context=context,
            call_id=call_id,
            path=path,
            requested_model=requested_model,
            routed_model=None,
            status_code=500,
            latency_ms=latency_ms,
            retry_count=retry_count,
            input_tokens_est=locals().get("input_tokens"),
            output_tokens_est=None,
            actual_input_tokens=None,
            actual_output_tokens=None,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=None,
            cost_baseline_usd=None,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            stream=stream,
            session_id=session_id,
            error=error,
        )
        return JSONResponse(public_proxy_error_body("anthropic", exc), status_code=500)
