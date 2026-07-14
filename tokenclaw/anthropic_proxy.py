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
from tokenclaw.http_client import async_client
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from tokenclaw.provider_context import ProviderContext
from tokenclaw.pricing import MODEL_ALIASES, estimate_cost
from tokenclaw.prompt_features import prompt_difficulty_features_from_text
from tokenclaw.headers import (
    ClientJsonRequestError,
    build_anthropic_forward_headers,
    build_anthropic_summary_headers,
    client_json_error_body,
    read_json_object_body,
)
from tokenclaw.limiter import TierBackoffActive, model_tier, tier_backoff_headers, tier_backoff_payload
from tokenclaw.router import (
    extract_text, has_tools, categorize_request, classify_workflow_phase, route_model,
    STRIP_THINKING_HISTORY, _has_top_level_thinking, strip_thinking_history_blocks, uses_thinking,
)
from tokenclaw.crunch import (
    TOKEN_CHARS, estimate_tokens_from_text, build_embedding,
    crunch_body, inject_prompt_cache, has_cache_control_blocks,
    maybe_summarize_old_context, OLD_CONTEXT_SUMMARY_MODEL,
    CRUNCH_POLICY, CRUNCH_POLICY_SOURCE, CRUNCH_RULES_PATH,
)
from tokenclaw.cache import (
    CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD, CACHE_POLICY, CACHE_POLICY_SOURCE, CACHE_RULES_PATH,
    build_cache_replay_lifecycle_feedback, cache_replay_lifecycle_feedback_public_meta,
    cache_decision_meta, cache_file_dependency_audit, cache_hit_decision_meta, cache_key_for, cache_lookup_meta,
    cache_replay_canary_decision, cache_replay_scope_for_meta, response_output_text,
    stream_cache_payload, validate_stream_cache_payload,
    streaming_cache_lookup_meta, cache_file_dependency_snapshots,
)
from tokenclaw.optimization_coordinator_enforcement import enforce_optimization_coordinator
from tokenclaw.optimization_coordinator_feedback import (
    optimization_coordinator_lifecycle_feedback_public_meta,
    queue_optimization_coordinator_lifecycle_feedback as queue_optimization_coordinator_lifecycle_event,
)
from tokenclaw.errors import (
    INTERNAL_PROXY_ERROR_MESSAGE,
    public_proxy_error_body,
    upstream_error_text,
)
from tokenclaw.provider_adoption import capture_provider_tool_adoption
from tokenclaw.routing_experiments import (
    ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES,
    _app_family as _experiment_app_family,
    compare_response_outputs,
    prefetch_server_experiment_policy,
    routing_experiment_outcome_event,
    routing_experiment_feedback_features,
    routing_experiment_decision,
    _public_label as _public_metadata_label,
    _today_shadow_spend_usd,
)
from tokenclaw.session_memory_hints import build_session_memory_optimization_hints
from tokenclaw.streaming_tool_cache_invalidation_drill import (
    build_streaming_tool_cache_invalidation_drill_cache_meta,
)
from tokenclaw.recommendations import (
    attach_observed_savings_to_routing_meta,
    apply_recommendation_to_body,
    build_old_context_summary_outcome_event,
    build_old_context_summary_outcome_feedback,
    build_outcome_feedback,
    build_phase_routing_outcome_event,
    build_phase_routing_outcome_feedback,
    build_optimization_unit,
    build_request_facts_envelope,
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
from tokenclaw.local_compaction_canary_ramp import build_thinking_tail_feedback_freshness
from tokenclaw.managed_session_tier import (
    apply_session_tier_to_body,
    count_tool_definitions,
    fetch_or_get_session_tier,
)
from tokenclaw.store import stable_json, utc_now


SESSION_COST_ALERT_USD = float(os.getenv("TOKENCLAW_SESSION_COST_ALERT_USD", "5.0"))
MAX_THINKING_BUDGET_TOKENS = int(os.getenv("TOKENCLAW_MAX_THINKING_BUDGET_TOKENS", "0"))

# Anthropic rejects non-streaming requests whose ``max_tokens`` is large enough
# that generation could exceed the 10 minute non-streaming limit, returning a
# ``400 invalid_request_error``. Shadow comparison calls force non-streaming so
# they can collect a full JSON body to diff, so the primary request's larger
# ``max_tokens`` (common for Opus 4.8 streaming traffic) must be clamped down to
# a value every model accepts without streaming. 8192 sits at or below the
# non-streaming ceiling for all current Claude models.
ANTHROPIC_SHADOW_NONSTREAMING_MAX_TOKENS = int(
    os.getenv("TOKENCLAW_ANTHROPIC_SHADOW_NONSTREAMING_MAX_TOKENS", "8192")
)
# Daily ceiling on TokenClaw-attributable Anthropic shadow spend. The managed
# shadow path is the only Anthropic shadow trigger and previously had no cap; as
# coverage widens toward "all incoming Claude traffic" (e.g. thinking traffic now
# eligible), an uncapped path could run up real candidate-model spend. The cap
# bounds it: once today's shadow spend reaches it, sampling stops for the day and
# the primary request is served unchanged. <=0 disables the cap.
MANAGED_SHADOW_DAILY_BUDGET_USD = float(
    os.getenv("TOKENCLAW_MANAGED_SHADOW_DAILY_BUDGET_USD", "10.0")
)
# Experiment flag: keep the shadow model's OWN extended thinking enabled instead of
# forcing it off. The forced-off shadow compares a thinking primary (opus) against a
# no-thinking shadow (sonnet), which understates the shadow on agent-loop turns where
# reasoning drives the next action. Keeping thinking on tests the fair counterfactual
# (what live routing to sonnet would actually produce). Default off; enable to A/B.
ANTHROPIC_SHADOW_KEEP_THINKING = os.getenv(
    "TOKENCLAW_ANTHROPIC_SHADOW_KEEP_THINKING", "0"
).strip().lower() in {"1", "true", "yes", "on"}
# Execute the shadow as a streaming call when the primary streamed. The forced
# non-streaming shadow had to clamp max_tokens to the non-streaming ceiling
# (64000 -> 8192 on agent traffic), handicapping the shadow's thinking + output
# budget relative to the primary it is compared against. Default on; the kill
# switch restores the old clamped non-streaming probes.
ANTHROPIC_SHADOW_STREAMING = os.getenv(
    "TOKENCLAW_ANTHROPIC_SHADOW_STREAMING", "1"
).strip().lower() not in {"", "0", "false", "no", "off"}
# Upstream invalid-request messages describe the request shape (parameter names
# and limits), not prompt content, so a short truncated copy is safe to retain
# for diagnosing future 4xx causes without leaking raw prompts. The cap stays at
# or below the shared metadata-label sanitizer's length threshold so a clean,
# request-shape message survives sanitization instead of being redacted for length.
SHADOW_HTTP_ERROR_DETAIL_MAX_CHARS = 160


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
    phase_meta = classify_workflow_phase(body, category)
    return {
        "enabled": False,
        "requested_model": requested_model,
        "routed_model": str(body.get("model") or requested_model),
        "reason": "preflight feature extraction before local mutation",
        "text_chars": len(text),
        "has_tools": has_tools(body),
        "category": category,
        "workflow_phase": phase_meta.get("workflow_phase"),
        "workflow_phase_reason": phase_meta.get("workflow_phase_reason"),
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
    routing_meta["local_result"] = "fallback"
    routing_meta["routing_outcome_label"] = "fallback"
    session_tier = routing_meta.get("managed_session_tier")
    if isinstance(session_tier, dict) and session_tier.get("applied"):
        session_tier["fallback_reason"] = "rate_limited"
        session_tier["fallback_from_model"] = str(from_model)
        session_tier["actual_forwarded_model"] = str(requested_model)
        session_tier["local_result"] = "fallback"
    phase_canary = routing_meta.get("phase_canary")
    if isinstance(phase_canary, dict):
        phase_canary["fallback_reason"] = "rate_limited"
        phase_canary["fallback_from_model"] = str(from_model)
        phase_canary["actual_forwarded_model"] = str(requested_model)


class _AnthropicSseContentAccumulator:
    """Rebuild assistant content (text + tool_use blocks), stop_reason, and usage
    from an Anthropic SSE stream.

    The routing-experiment comparison needs the streamed primary's FULL content:
    reconstructing only the text deltas made every streaming tool turn compare as
    "no tool calls" against the shadow's complete body, scoring near zero
    regardless of true equivalence. Thinking deltas are deliberately not
    accumulated — comparisons score visible output, and thinking never leaves the
    proxy.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []
        self.stop_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_creation_input_tokens: int = 0
        self.cache_read_input_tokens: int = 0

    def observe(self, data: dict[str, Any]) -> None:
        event_type = data.get("type")
        if event_type == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            self.input_tokens = usage.get("input_tokens")
            self.cache_creation_input_tokens = usage.get("cache_creation_input_tokens") or 0
            self.cache_read_input_tokens = usage.get("cache_read_input_tokens") or 0
        elif event_type == "message_delta":
            delta = data.get("delta") or {}
            stop = delta.get("stop_reason")
            if stop:
                self.stop_reason = str(stop)
            out = (data.get("usage") or {}).get("output_tokens")
            if out is not None:
                self.output_tokens = out
        elif event_type == "content_block_start":
            index = data.get("index")
            block = data.get("content_block") or {}
            if not isinstance(index, int) or not isinstance(block, dict) or index in self._blocks:
                return
            block_type = block.get("type")
            if block_type == "text":
                self._blocks[index] = {"type": "text", "text": str(block.get("text") or "")}
                self._order.append(index)
            elif block_type == "tool_use":
                self._blocks[index] = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                    "_partial_json": "",
                }
                self._order.append(index)
        elif event_type == "content_block_delta":
            index = data.get("index") if isinstance(data.get("index"), int) else 0
            delta = data.get("delta") or {}
            block = self._blocks.get(index)
            if block is None:
                # Providers always send content_block_start first, but tolerate a
                # missing one (minimal fixtures, dropped frames) for text deltas
                # rather than silently losing output.
                if delta.get("type") != "text_delta":
                    return
                block = {"type": "text", "text": ""}
                self._blocks[index] = block
                self._order.append(index)
            if delta.get("type") == "text_delta" and block.get("type") == "text":
                block["text"] += str(delta.get("text") or "")
            elif delta.get("type") == "input_json_delta" and block.get("type") == "tool_use":
                block["_partial_json"] += str(delta.get("partial_json") or "")
        elif event_type == "content_block_stop":
            index = data.get("index")
            block = self._blocks.get(index) if isinstance(index, int) else None
            if block is not None and block.get("type") == "tool_use" and block.get("_partial_json"):
                try:
                    parsed = json.loads(block["_partial_json"])
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    block["input"] = parsed

    def observe_sse_frame(self, frame: bytes) -> None:
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
            if isinstance(data, dict):
                self.observe(data)

    def content(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for index in self._order:
            block = dict(self._blocks[index])
            block.pop("_partial_json", None)
            if block.get("type") == "text" and not block.get("text"):
                continue
            blocks.append(block)
        return blocks


def _anthropic_stream_primary_response_body(
    *,
    output_text: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    thinking_output_tokens: int | None,
    content_blocks: list[dict[str, Any]] | None = None,
    stop_reason: str | None = None,
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
    if content_blocks:
        content = content_blocks
    else:
        content = [{"type": "text", "text": output_text}] if output_text else []
    body: dict[str, Any] = {
        "type": "message",
        "content": content,
        "usage": usage,
        "tokenclaw_streaming_capture": {
            "complete": True,
            "output_text_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            "raw_stream_included": False,
        },
    }
    if stop_reason:
        body["stop_reason"] = stop_reason
    return body


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


def _managed_shadow_experiment_decision(
    *,
    recommendation_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    requested_model: str,
    primary_model: str,
    stream: bool,
    input_tokens_est: int,
    random_value: float | None = None,
    store_obj: Any | None = None,
) -> dict[str, Any] | None:
    shadow = recommendation_meta.get("shadow")
    if not isinstance(shadow, dict) or shadow.get("status") != "recommended":
        return None
    shadow_model = str(shadow.get("target_model") or "").strip()
    if not shadow_model or shadow_model == primary_model:
        return None
    try:
        fraction = max(0.0, min(1.0, float(shadow.get("fraction") or 0.0)))
    except (TypeError, ValueError):
        fraction = 0.0
    source_surface = str(recommendation_meta.get("source_surface") or "anthropic_messages")
    budget_limit = MANAGED_SHADOW_DAILY_BUDGET_USD
    budget_spent = (
        _today_shadow_spend_usd(store_obj, provider="anthropic", source_surface=source_surface)
        if budget_limit > 0
        else 0.0
    )
    budget_remaining = max(0.0, budget_limit - budget_spent) if budget_limit > 0 else None
    budget_exhausted = budget_limit > 0 and budget_spent >= budget_limit
    selected = (
        not budget_exhausted
        and fraction > 0.0
        and (random.random() if random_value is None else random_value) < fraction
    )
    reason_codes = [
        str(item)
        for item in (shadow.get("reason_codes") or [])
        if isinstance(item, str) and item
    ]
    return {
        "schema": "tokenclaw.routing_experiment_decision.v1",
        "enabled": True,
        "mode": "shadow_candidate_pass_through",
        "kill_switch": False,
        "status": "selected" if selected else "skipped",
        "sampled": bool(selected),
        "reason": (
            "managed-shadow-sampled"
            if selected
            else "managed-shadow-budget-exhausted"
            if budget_exhausted
            else "managed-shadow-not-sampled"
        ),
        "counterfactual": True,
        "shadow_only": True,
        "policy_source": "managed-recommended",
        "rule_path": "managed://tokenclaw-server/policy-decision/shadow",
        "sample_rate": round(fraction, 6),
        "sample_rate_scope": "managed-policy-decision-shadow",
        "daily_budget_usd": round(budget_limit, 6) if budget_limit > 0 else None,
        "daily_budget_scope": "managed-policy-decision-shadow",
        "profile_id": str(shadow.get("policy_id") or "managed-shadow-recommendation"),
        "budget_spent_usd": round(budget_spent, 6) if budget_limit > 0 else None,
        "budget_remaining_usd": round(budget_remaining, 6) if budget_remaining is not None else None,
        "budget_exhausted": bool(budget_exhausted),
        "budget_cap_scope": "managed-policy-decision-shadow",
        "similarity_threshold": None,
        "min_samples_for_confidence": None,
        "provider": "anthropic",
        "source_surface": str(recommendation_meta.get("source_surface") or "anthropic_messages"),
        "stream": bool(stream),
        "streaming_shadow_supported": bool(stream),
        "requested_model": requested_model,
        "routed_model": shadow_model,
        "shadow_model": shadow_model,
        "primary_model": primary_model,
        "user_visible_model": primary_model,
        "category": str(routing_meta.get("category") or ""),
        "workflow_phase": str(routing_meta.get("workflow_phase") or ""),
        "text_chars": int(routing_meta.get("text_chars") or 0),
        "input_tokens_est": input_tokens_est,
        "min_text_chars": 0,
        "min_text_chars_scope": "managed-policy-decision-shadow",
        "max_text_chars": 0,
        "max_text_chars_scope": "managed-policy-decision-shadow",
        "eligibility_overrides_applied": [],
        "candidate_id": shadow.get("policy_id") or recommendation_meta.get("policy_id"),
        "selected_candidate": {
            "candidate_id": shadow.get("policy_id") or recommendation_meta.get("policy_id"),
            "requested_model": requested_model,
            "routed_model": shadow_model,
            "policy_source": "managed-recommended",
            "source_surface": recommendation_meta.get("source_surface") or "anthropic_messages",
        },
        "eligible_candidate_count": 1,
        "eligible_candidate_ids": [str(shadow.get("policy_id") or recommendation_meta.get("policy_id") or "managed-shadow-recommendation")],
        "candidate_selector": "managed-policy-decision-shadow",
        "candidate_policy_shape": "managed_policy_decision_shadow",
        "candidate_selector_basis": {
            "source": "managed-policy-decision-shadow",
            "decision_id": recommendation_meta.get("decision_id"),
            "policy_id": recommendation_meta.get("policy_id"),
            "metadata_only": True,
        },
        "managed_shadow": {
            "schema": "tokenclaw.managed_shadow_local_execution.v1",
            "decision_id": recommendation_meta.get("decision_id"),
            "policy_id": recommendation_meta.get("policy_id"),
            "shadow_policy_id": shadow.get("policy_id"),
            "mode": shadow.get("mode"),
            "fraction": fraction,
            "reason_codes": reason_codes,
            "required_local_gates": shadow.get("required_local_gates") or [],
            "metadata_only": True,
            "provider_forwarding_by_managed_server": False,
            "locally_executed": True,
        },
        "reason_codes": ["managed-shadow-recommendation", *reason_codes],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
        },
    }


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
                "message": f"TokenClaw is running in {context.provider!r} provider mode, not {expected!r}.",
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


def _cache_ttl_seconds(cache_meta: dict[str, Any]) -> int:
    pattern = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
    raw = pattern.get("ttl_seconds") or cache_meta.get("ttl_seconds") or 3600
    try:
        return max(60, int(raw))
    except (TypeError, ValueError):
        return 3600


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
        "schema": "tokenclaw.anthropic_shadow_tool_result_audit.v1",
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


def _system_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize a ``system`` value (string or block list) to a content-block list."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(block)
            elif isinstance(block, str) and block.strip():
                blocks.append({"type": "text", "text": block})
        return blocks
    return []


def _fold_system_role_messages(body: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Fold ``role: 'system'`` messages into the top-level ``system`` parameter.

    Some Claude clients place the system prompt as a ``system``-role entry inside
    ``messages`` instead of the top-level ``system`` field. Opus 4.8 tolerates
    this, but other models (e.g. Sonnet 4.6) reject it with a 400
    ``invalid_request_error: role 'system' is not supported on this model``. The
    shadow leg forwards the primary body to a different model, so any such message
    must be moved to the top-level ``system`` param for the request to validate —
    otherwise every opus->sonnet canary silently 400s and records no evidence.
    Content is moved, never logged.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    folded_blocks: list[dict[str, Any]] = []
    kept: list[Any] = []
    folded = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            folded_blocks.extend(_system_content_blocks(message.get("content")))
            folded += 1
            continue
        kept.append(message)
    if not folded:
        return
    combined = _system_content_blocks(body.get("system")) + folded_blocks
    if combined:
        body["system"] = combined
    body["messages"] = kept
    diagnostics["system_role_messages_folded"] = folded


def _strip_thinking_dependent_context_management(
    body: dict[str, Any], diagnostics: dict[str, Any]
) -> None:
    """Drop context-management edits that require thinking from a no-thinking shadow.

    Anthropic's context-management ``clear_thinking_*`` strategy requires ``thinking``
    to be enabled or adaptive. The shadow leg disables thinking (the ``thinking``
    param is stripped and streaming forced off), so leaving such an edit in place
    makes the request self-contradictory and Anthropic returns a 400
    ``invalid_request_error: clear_thinking_... strategy requires thinking to be
    enabled or adaptive``. Remove only the thinking-dependent edits; other context
    edits (e.g. ``clear_tool_uses_*``) are fine and stay. Drop the container if it
    empties. Operates on request shape only; no prompt content is read.
    """
    context_management = body.get("context_management")
    if not isinstance(context_management, dict):
        return
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return
    kept: list[Any] = []
    removed = 0
    for edit in edits:
        edit_type = str(edit.get("type") or "").lower() if isinstance(edit, dict) else ""
        if "clear_thinking" in edit_type:
            removed += 1
            continue
        kept.append(edit)
    if not removed:
        return
    if kept:
        context_management["edits"] = kept
    else:
        body.pop("context_management", None)
    diagnostics["thinking_dependent_context_edits_stripped"] = removed


def _normalize_shadow_thinking_for_keep(body: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Keep the shadow model's own thinking enabled, normalized to a valid shape.

    Drops opus-only sub-params the shadow model may reject and clamps the thinking
    budget below the (already clamped) non-streaming ``max_tokens`` so the request
    validates. The shadow then reasons with its OWN thinking rather than being
    forced to answer cold — the fair routing counterfactual.
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return
    thinking.pop("effort", None)
    body.pop("interleaved_thinking", None)
    try:
        max_tokens = int(body.get("max_tokens") or ANTHROPIC_SHADOW_NONSTREAMING_MAX_TOKENS)
    except (TypeError, ValueError):
        max_tokens = ANTHROPIC_SHADOW_NONSTREAMING_MAX_TOKENS
    ceiling = max(1024, max_tokens - 512)
    try:
        requested_budget = int(thinking.get("budget_tokens") or 0)
    except (TypeError, ValueError):
        requested_budget = 0
    budget = requested_budget if requested_budget > 0 else min(4096, ceiling)
    budget = max(1024, min(budget, ceiling))
    thinking["budget_tokens"] = budget
    thinking["type"] = "enabled"
    diagnostics["shadow_thinking_mode"] = "kept"
    diagnostics["shadow_thinking_budget"] = budget


def _normalize_shadow_non_streaming_max_tokens(
    body: dict[str, Any], diagnostics: dict[str, Any]
) -> None:
    """Clamp ``max_tokens`` so the forced non-streaming shadow call validates.

    Anthropic returns a ``400 invalid_request_error`` for non-streaming requests
    whose ``max_tokens`` could exceed the 10 minute non-streaming limit. The
    shadow leg forces ``stream=False``, so a primary request built for streaming
    with a large ``max_tokens`` (e.g. Opus 4.8 traffic) would otherwise fail
    before any similarity evidence can be recorded.
    """
    ceiling = ANTHROPIC_SHADOW_NONSTREAMING_MAX_TOKENS
    if ceiling <= 0:
        return
    try:
        current = int(body.get("max_tokens"))
    except (TypeError, ValueError):
        return
    if current > ceiling:
        body["max_tokens"] = ceiling
        diagnostics["max_tokens_clamped_for_non_streaming"] = True
        diagnostics["max_tokens_original"] = current
        diagnostics["max_tokens_effective"] = ceiling


# Substrings of models that accept the 1M-token long-context beta
# (`anthropic-beta: context-1m-*`). Shadows reuse the client's request headers, so
# this beta must be dropped for a shadow target that does not support it (e.g. Haiku)
# or Anthropic rejects the shadow 400 "authentication style is incompatible with the
# long context beta header". Keep the list to genuinely 1M-capable models so the
# working opus->sonnet-5 shadow (sonnet-5 supports 1M) is untouched.
_ANTHROPIC_1M_CONTEXT_MODEL_HINTS = (
    "sonnet-5",
    "sonnet-4-5",
    "sonnet-4.5",
    "opus-4-5",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
)


def _model_supports_1m_context(model: str) -> bool:
    lowered = (model or "").lower()
    return any(hint in lowered for hint in _ANTHROPIC_1M_CONTEXT_MODEL_HINTS)


def _shadow_headers_for_model(headers: dict[str, str], shadow_model: str) -> dict[str, str]:
    """Return headers safe to forward to ``shadow_model``.

    Drops the 1M long-context beta (``context-1m-*``) from ``anthropic-beta`` when the
    shadow target does not support it, so a Haiku/small-model shadow of a 1M-context
    request does not 400. Other beta tokens (e.g. ``prompt-caching``) are preserved.
    Returns the original dict unchanged (same identity) when nothing needs stripping.
    """
    if _model_supports_1m_context(shadow_model):
        return headers
    beta = headers.get("anthropic-beta")
    if not beta or "context-1m" not in beta.lower():
        return headers
    kept = [
        token.strip()
        for token in beta.split(",")
        if token.strip() and "context-1m" not in token.strip().lower()
    ]
    new_headers = dict(headers)
    if kept:
        new_headers["anthropic-beta"] = ",".join(kept)
    else:
        new_headers.pop("anthropic-beta", None)
    return new_headers


def _prepare_anthropic_shadow_request(
    request_body: dict[str, Any],
    *,
    shadow_model: str,
    primary_model: str,
    stream_shadow: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    shadow_body = copy.deepcopy(request_body)
    shadow_body["model"] = shadow_model
    shadow_body["stream"] = bool(stream_shadow)
    diagnostics: dict[str, Any] = {
        "schema": "tokenclaw.anthropic_shadow_request_preflight.v1",
        "status": "ok",
        "reason": None,
        "primary_model": primary_model,
        "shadow_model": shadow_model,
        "stream_forced_non_streaming": not stream_shadow,
        "shadow_streaming": bool(stream_shadow),
        "raw_request_included": False,
        "raw_prompts_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
    }
    if not stream_shadow:
        # Non-streaming requests above the 10-minute ceiling 400; a streaming
        # shadow keeps the primary's original max_tokens and thinking headroom.
        _normalize_shadow_non_streaming_max_tokens(shadow_body, diagnostics)
    _fold_system_role_messages(shadow_body, diagnostics)
    sanitization: dict[str, Any] = {}
    keep_thinking = ANTHROPIC_SHADOW_KEEP_THINKING and _has_top_level_thinking(shadow_body)
    if keep_thinking:
        # Keep the shadow model's own thinking on (fair counterfactual); just
        # normalize it. clear_thinking context edits stay valid because thinking
        # remains enabled.
        _normalize_shadow_thinking_for_keep(shadow_body, diagnostics)
    else:
        _strip_model_incompatible_params(shadow_body, sanitization, primary_model)
        if sanitization.get("stripped_params"):
            diagnostics["stripped_params"] = sanitization["stripped_params"]
        # Thinking is now disabled on the shadow, so any context-management strategy
        # that requires thinking (clear_thinking_*) must go or the request 400s.
        _strip_thinking_dependent_context_management(shadow_body, diagnostics)
        diagnostics["shadow_thinking_mode"] = "disabled"
    # The assistant thinking/redacted_thinking history carries the PRIMARY model's
    # signatures, which never validate on a different shadow model, so strip it
    # whether or not the shadow keeps its own thinking enabled. (Without thinking it
    # would also be the self-contradictory "thinking blocks without thinking
    # enabled" 400 that silently killed every opus->sonnet tool-result canary.)
    diagnostics["candidate_would_strip_thinking_history"] = True
    pre_sanitization_tool_audit = _anthropic_shadow_tool_result_audit(shadow_body)
    if int(pre_sanitization_tool_audit.get("thinking_blocks_before_tool_results") or 0) > 0:
        diagnostics["pre_sanitization_tool_result_audit"] = pre_sanitization_tool_audit
    shadow_body, stripped_thinking = strip_thinking_history_blocks(shadow_body)
    if stripped_thinking:
        diagnostics["thinking_history_blocks_stripped"] = stripped_thinking
    tool_audit = _anthropic_shadow_tool_result_audit(shadow_body)
    diagnostics["tool_result_audit"] = tool_audit
    if tool_audit["status"] == "unsupported":
        diagnostics.update({"status": "unsupported", "reason": tool_audit["reason"]})
        return None, diagnostics
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


def _shadow_http_error_detail(response_body: dict[str, Any] | None) -> str | None:
    """Return the truncated upstream error message for diagnosing 4xx shadow calls.

    Anthropic ``invalid_request_error`` messages describe the offending request
    shape (e.g. an out-of-range ``max_tokens`` or a non-streaming length limit),
    so a short truncated copy keeps the cause diagnosable instead of collapsing
    every 400 to its error type, without retaining prompt content.
    """
    if not isinstance(response_body, dict):
        return None
    error_obj = response_body.get("error")
    if not isinstance(error_obj, dict):
        return None
    message = str(error_obj.get("message") or "").strip()
    if not message:
        return None
    if len(message) > SHADOW_HTTP_ERROR_DETAIL_MAX_CHARS:
        message = message[: SHADOW_HTTP_ERROR_DETAIL_MAX_CHARS - 1] + "…"
    # Run the truncated message through the shared metadata-label sanitizer so any
    # value that looks like it carries raw prompt/identifier content is redacted
    # rather than persisted, while request-shape messages survive intact.
    label = _public_metadata_label(message, fallback="redacted-metadata-label")
    return label



def build_forward_headers(request: Request) -> dict[str, str]:
    return build_anthropic_forward_headers(request.headers)


def build_summary_headers(request: Request) -> dict[str, str]:
    return build_anthropic_summary_headers(request.headers)


async def _fetch_old_context_summary(context: ProviderContext, summary_request: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    model = str(summary_request.get("model") or OLD_CONTEXT_SUMMARY_MODEL)
    try:
        async with context.limiter.semaphores[model_tier(model)]:
            async with async_client(timeout=context.http_timeout) as client:
                await context.limiter.await_backoff(model)
                await context.limiter.throttle_forward()
                r = await client.post(
                    context.anthropic_upstream.rstrip("/") + "/v1/messages",
                    headers=headers,
                    json=summary_request,
                )
    except Exception:
        logging.exception("tokenclaw old-context summary error")
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


async def _collect_anthropic_streaming_shadow(
    client: Any,
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    """Run the shadow as a streaming call and reconstruct its message body.

    Streaming lets the shadow keep the primary's original max_tokens and
    thinking headroom (the non-streaming ceiling forced a 64000 -> 8192 clamp on
    agent traffic). On HTTP errors the JSON error body is returned so the
    shadow-error classification and detail capture keep working.
    """
    accumulator = _AnthropicSseContentAccumulator()
    async with client.stream("POST", url, headers=headers, json=body) as r:
        status_code = r.status_code
        if status_code >= 400:
            try:
                raw = await r.aread()
                return status_code, json.loads(raw)
            except Exception:
                return status_code, None
        buf = b""
        async for chunk in r.aiter_bytes():
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                accumulator.observe_sse_frame(frame)
        if buf:
            accumulator.observe_sse_frame(buf)
    usage: dict[str, Any] = {}
    if accumulator.input_tokens is not None:
        usage["input_tokens"] = accumulator.input_tokens
    if accumulator.output_tokens is not None:
        usage["output_tokens"] = accumulator.output_tokens
    if accumulator.cache_creation_input_tokens:
        usage["cache_creation_input_tokens"] = accumulator.cache_creation_input_tokens
    if accumulator.cache_read_input_tokens:
        usage["cache_read_input_tokens"] = accumulator.cache_read_input_tokens
    reconstructed: dict[str, Any] = {
        "type": "message",
        "content": accumulator.content(),
        "usage": usage,
        "tokenclaw_streaming_capture": {
            "complete": True,
            "raw_stream_included": False,
        },
    }
    if accumulator.stop_reason:
        reconstructed["stop_reason"] = accumulator.stop_reason
    return status_code, reconstructed


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
        stream_shadow=ANTHROPIC_SHADOW_STREAMING and bool(request_body.get("stream")),
    )
    experiment_meta["shadow_request_preflight"] = shadow_preflight
    shadow_headers = _shadow_headers_for_model(headers, shadow_model)
    if shadow_headers is not headers:
        experiment_meta["shadow_long_context_beta_stripped"] = True
    shadow_status_code: Optional[int] = None
    shadow_response_body: Optional[dict[str, Any]] = None
    shadow_latency_ms: Optional[int] = None
    shadow_cost: Optional[float] = None
    error: Optional[str] = None
    shadow_http_error_detail: Optional[str] = None

    shadow_started = time.time()
    if shadow_body is None:
        shadow_latency_ms = 0
        reason = str(shadow_preflight.get("reason") or "unsupported-shape")
        error = f"shadow-unsupported-shape:{reason}"
    else:
        try:
            async with context.limiter.semaphores[model_tier(shadow_model)]:
                async with async_client(timeout=context.http_timeout) as client:
                    await context.limiter.await_backoff(shadow_model)
                    await context.limiter.throttle_forward()
                    if shadow_body.get("stream"):
                        shadow_status_code, shadow_response_body = await _collect_anthropic_streaming_shadow(
                            client,
                            url=context.anthropic_upstream.rstrip("/") + path,
                            headers=shadow_headers,
                            body=shadow_body,
                        )
                    else:
                        r = await client.post(context.anthropic_upstream.rstrip("/") + path, headers=shadow_headers, json=shadow_body)
                        shadow_status_code = r.status_code
                        try:
                            shadow_response_body = r.json()
                        except Exception:
                            shadow_response_body = None
            shadow_latency_ms = int((time.time() - shadow_started) * 1000)
            http_error_class = _shadow_http_error_class(shadow_status_code, shadow_response_body)
            if http_error_class:
                error = http_error_class
                shadow_http_error_detail = _shadow_http_error_detail(shadow_response_body)
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

    # Counterfactual routed cost: what the shadow model WOULD cost serving the
    # primary's exact token profile (same cache_read/cache_creation split). The raw
    # shadow probe above is a fresh, uncached call, so on cached traffic it costs far
    # more than the cached primary — making routing look more expensive than it is.
    # Live routing to the cheaper model gets the same caching, so this counterfactual
    # is the fair economics for the promotion decision. Budget still uses the real
    # probe cost (shadow_cost); promotion uses this.
    shadow_routed_cost: Optional[float] = None
    primary_usage = primary_response_body.get("usage") if isinstance(primary_response_body, dict) else None
    if isinstance(primary_usage, dict):
        shadow_routed_cost = estimate_cost(
            shadow_model,
            int(primary_usage.get("input_tokens") or 0) or input_tokens_est,
            int(primary_usage.get("output_tokens") or 0),
            int(primary_usage.get("cache_creation_input_tokens") or 0),
            int(primary_usage.get("cache_read_input_tokens") or 0),
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
        shadow_routed_cost_est_usd=shadow_routed_cost,
        error=error,
        shadow_http_error_detail=shadow_http_error_detail,
    )
    experiment_meta.update(
        {
            "experiment_id": experiment_id,
            "status": feedback_features["status"],
            "primary_model": primary_model,
            "shadow_model": shadow_model,
            "primary_status_code": primary_status_code,
            "shadow_status_code": shadow_status_code,
            "shadow_http_error_detail": shadow_http_error_detail,
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
        shadow_routed_cost_est_usd=shadow_routed_cost,
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
                "tokenclaw_old_context_summary "
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
        preflight_request_facts = build_request_facts_envelope(
            provider="anthropic",
            path=path,
            body=raw_body,
            requested_model=str(raw_body.get("model") or requested_model),
            stream=stream,
            input_tokens_est=preflight_input_tokens,
            session_id=session_id,
            category=category,
            workflow_phase=str(preflight_routing_meta.get("workflow_phase") or "") or None,
            prompt_difficulty_features=preflight_routing_meta.get("prompt_difficulty_features"),
        )
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
        routing_meta["server_experiment_policy_fetch"] = await prefetch_server_experiment_policy(
            provider="anthropic",
            source_surface="anthropic_messages",
            app_family=_experiment_app_family(
                "anthropic", "anthropic_messages", str(resolved_requested_model)
            ),
        )
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
        session_tier_meta = await fetch_or_get_session_tier(
            preflight_recommendation_unit,
            session_id=session_id,
            tool_count=count_tool_definitions(raw_body),
        )
        session_tier_routed_model = apply_session_tier_to_body(
            crunched,
            routing_meta,
            session_tier_meta,
            session_id=session_id,
            stream=stream,
        )
        if session_tier_routed_model:
            routed_model = session_tier_routed_model
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
        routing_meta["managed_request_facts"] = preflight_request_facts
        if policy_decisions_enabled():
            recommendation_unit.setdefault("input_features", {})[
                "thinking_tail_feedback_freshness"
            ] = build_thinking_tail_feedback_freshness(context.store)
            recommendation_meta = await fetch_policy_decision(recommendation_unit, request_facts=preflight_request_facts)
        recommendation_meta = apply_recommendation_to_body(
            provider="anthropic",
            body=crunched,
            routing_meta=routing_meta,
            recommendation_meta=recommendation_meta,
            store_obj=context.store,
            session_id=session_id,
        )
        if crunched.get("model") in MODEL_ALIASES:
            normalized_model = MODEL_ALIASES[str(crunched.get("model"))]
            recommendation_meta["target_model_normalized"] = normalized_model
            crunched["model"] = normalized_model
            routing_meta["routed_model"] = normalized_model
        routing_meta["managed_recommendation"] = recommendation_meta
        managed_shadow_experiment = _managed_shadow_experiment_decision(
            recommendation_meta=recommendation_meta,
            routing_meta=routing_meta,
            requested_model=str(resolved_requested_model),
            primary_model=str(crunched.get("model") or routed_model),
            stream=stream,
            input_tokens_est=input_tokens,
            store_obj=context.store,
        )
        if managed_shadow_experiment is not None:
            routing_meta["routing_experiment"] = managed_shadow_experiment
            experiment_meta = managed_shadow_experiment
            sampled_shadow_pass_through = bool(managed_shadow_experiment.get("sampled"))
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
            cache_meta["streaming_tool_cache_invalidation_drill"] = (
                build_streaming_tool_cache_invalidation_drill_cache_meta(
                    cache_meta,
                    category=category,
                    stream=True,
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint=path,
                    requested_model=str(resolved_requested_model),
                    routed_model=str(crunched.get("model") or routed_model),
                )
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
                        "schema": "tokenclaw.cache_pattern_warmup.v1",
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
                        headers={"x-tokenclaw-cache": "hit", "x-tokenclaw-routed-model": str(crunched.get("model"))},
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
                # Full-content capture (text + tool_use blocks + stop_reason) so
                # the routing-experiment comparison sees the streamed primary's
                # real output, not just its text deltas.
                stream_content = _AnthropicSseContentAccumulator()

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
                        if isinstance(data, dict):
                            stream_content.observe(data)
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
                        async with async_client(timeout=context.http_timeout) as client:
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
                    logging.exception("tokenclaw anthropic streaming proxy error")
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
                    # Price the counterfactual on the SAME provider cache profile as
                    # the actual call. Folding cache reads into uncached input tokens
                    # inflated the baseline (and the observed_savings_usd fed to
                    # managed outcome feedback) ~22x on heavily-cached Claude Code
                    # traffic; the dashboard's realized_savings_attribution has
                    # always priced this counterfactual cache-aware.
                    cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_creation_in, cache_read_in)
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
                                ttl_seconds=_cache_ttl_seconds(cache_meta),
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
                                content_blocks=stream_content.content(),
                                stop_reason=stream_content.stop_reason,
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
                headers={"x-tokenclaw-cache": "miss" if can_stream_cache else "skip-streaming", "x-tokenclaw-routed-model": str(crunched.get("model"))},
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
                return JSONResponse(response_body, headers={"x-tokenclaw-cache": "hit", "x-tokenclaw-routed-model": str(crunched.get("model"))})

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
                return JSONResponse(sem_resp, headers={"x-tokenclaw-cache": "semantic-hit", "x-tokenclaw-routed-model": str(crunched.get("model"))})

        async with context.limiter.semaphores[model_tier(crunched["model"])]:
            async with async_client(timeout=context.http_timeout) as client:
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
                ttl_seconds=_cache_ttl_seconds(cache_meta),
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
        # Same cache-aware counterfactual as the streaming path above.
        cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_creation_in, cache_read_in)
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
        return JSONResponse(response_body, status_code=status_code, headers={"x-tokenclaw-cache": "miss", "x-tokenclaw-routed-model": str(crunched.get("model"))})

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
        logging.exception("tokenclaw anthropic proxy error")
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
