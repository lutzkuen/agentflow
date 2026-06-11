from __future__ import annotations

from typing import Any

from agentflow_proxy.optimization.openai_pipeline import serialize_openai_outcome_summary
from agentflow_proxy.openai_optimization_governor import (
    LIFECYCLE_SOURCE_SURFACE,
    build_openai_optimization_lifecycle_event,
    openai_optimization_lifecycle_public_meta,
)
from agentflow_proxy.recommendations import build_outcome_feedback, queue_outcome_feedback, queue_policy_event_feedback
from agentflow_proxy.store import stable_json


def attach_openai_outcome_summary(
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
) -> None:
    routing_meta["openai_outcome_unit"] = serialize_openai_outcome_summary(
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


async def record_managed_outcome_feedback(
    *,
    context: Any | None = None,
    store: Any | None = None,
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
    session_id: str | None,
    error: str | None = None,
) -> None:
    store_obj = store if store is not None else getattr(context, "store", None)
    if store_obj is None:
        raise ValueError("record_managed_outcome_feedback requires a store or context with a store")
    lifecycle = build_openai_optimization_lifecycle_event(
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
        path=path,
        requested_model=requested_model,
        routed_model=routed_model,
        status_code=status_code,
        latency_ms=latency_ms,
        retry_count=retry_count,
        cost_est_usd=cost_est_usd,
        cost_baseline_usd=cost_baseline_usd,
        category=category,
        stream=False,
        call_id=call_id,
    )
    if lifecycle is not None:
        lifecycle_meta = await queue_policy_event_feedback(
            store_obj,
            lifecycle,
            source_surface=LIFECYCLE_SOURCE_SURFACE,
            queue_when_disabled=True,
            flush_immediately=False,
        )
        routing_meta["openai_optimization_lifecycle_feedback"] = openai_optimization_lifecycle_public_meta(lifecycle_meta)
        if hasattr(store_obj, "update_call_routing_json"):
            store_obj.update_call_routing_json(call_id, stable_json(routing_meta))
    managed = routing_meta.get("managed_recommendation")
    if not isinstance(managed, dict) or not managed.get("enabled"):
        return
    outcome = build_outcome_feedback(
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
    managed["outcome_feedback"] = await queue_outcome_feedback(
        store_obj,
        managed,
        outcome,
        source_surface=str(outcome.get("source_surface") or "openai_responses"),
    )
    store_obj.update_call_routing_json(call_id, stable_json(routing_meta))
