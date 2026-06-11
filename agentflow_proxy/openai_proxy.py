from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import httpx
import websockets
from fastapi import Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from agentflow_proxy.cache import (
    SEMANTIC_CACHE_THRESHOLD,
    attach_file_dependency_cache_meta,
    cache_decision_meta,
    cache_file_dependency_audit,
    cache_file_dependency_snapshots,
    cache_hit_decision_meta,
    cache_key_for,
    cache_lookup_meta,
    cache_replay_canary_decision,
    cache_replay_scope_for_meta,
)
from agentflow_proxy.crunch import build_embedding, crunch_body, estimate_tokens_from_text
from agentflow_proxy.errors import (
    INTERNAL_PROXY_ERROR_MESSAGE,
    public_proxy_error_body,
    upstream_error_text,
)
from agentflow_proxy.headers import (
    ClientJsonRequestError,
    build_openai_forward_headers,
    build_openai_websocket_headers,
    client_json_error_body,
    read_json_object_body,
)
from agentflow_proxy.limiter import TierBackoffActive, model_tier, tier_backoff_headers, tier_backoff_payload
from agentflow_proxy.openai_old_context_summary import (
    add_summary_cost,
    input_tokens_after_summary,
    maybe_apply_openai_old_context_summary,
)
from agentflow_proxy.openai_optimization_governor import (
    attach_openai_optimization_governor,
    selected_openai_governor_family,
)
from agentflow_proxy.optimization.openai_features import (
    openai_call_store_fields,
)
from agentflow_proxy.optimization.openai_outcomes import (
    attach_openai_outcome_summary,
    record_managed_outcome_feedback,
)
from agentflow_proxy.optimization.openai_pipeline import (
    execute_openai_local_policy,
    extract_openai_preflight_features,
    fetch_openai_policy_decision,
    parse_openai_request_body,
)
from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.provider_context import ProviderContext
from agentflow_proxy.recommendations import queue_policy_event_feedback
from agentflow_proxy.routing_experiments import (
    ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES,
    compare_response_outputs,
    routing_experiment_decision,
    routing_experiment_feedback_features,
    routing_experiment_outcome_event,
)
from agentflow_proxy.router import extract_text, has_tools
from agentflow_proxy.store import stable_json, utc_now


SESSION_COST_ALERT_USD = float(os.getenv("AGENTFLOW_SESSION_COST_ALERT_USD", "5.0"))


def _openai_cache_replay_blockers(
    *,
    cache_meta: dict[str, Any],
    has_tool_blocks: bool,
    stream: bool,
) -> list[str]:
    blockers: list[str] = []
    if stream:
        blockers.append("unsupported-streaming-shape")
    if has_tool_blocks and not bool(cache_meta.get("tool_cache_enabled")):
        blockers.append("tool-call-cache-disabled")
    if cache_meta.get("status") != "hit":
        blockers.append("replay-rule-required")
    return blockers


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


def build_forward_headers(context: ProviderContext, request: Request, *, force_json: bool = True) -> dict[str, str]:
    return build_openai_forward_headers(
        request.headers,
        auth_mode=context.openai_auth_mode,
        api_key=os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        force_json=force_json,
    )


def build_websocket_headers(context: ProviderContext, websocket: WebSocket) -> dict[str, str]:
    return build_openai_websocket_headers(
        websocket.headers,
        auth_mode=context.openai_auth_mode,
        api_key=os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )


def websocket_url(context: ProviderContext, path: str) -> str:
    base = context.openai_upstream.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return base + path


async def read_json_body(request: Request) -> Optional[dict[str, Any]]:
    return await read_json_object_body(
        request,
        allow_compressed=True,
        passthrough_unsupported_encoding=True,
    )


def usage_tokens(body: dict[str, Any]) -> tuple[Optional[int], Optional[int], int, int]:
    usage = (body or {}).get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    return input_tokens, output_tokens, cached_tokens, reasoning_tokens


def response_output_text(resp: dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
    for choice in resp.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    return "\n".join(parts)


def _openai_endpoint_for_path(path: str) -> str:
    return "chat_completions" if "chat/completions" in (path or "").lower() else "responses"


def _openai_cache_replay_response_compatible(response_body: Any, path: str) -> tuple[bool, str]:
    if not isinstance(response_body, dict):
        return False, "non-json-response"
    if response_body.get("error") is not None:
        return False, "error-response"
    endpoint = _openai_endpoint_for_path(path)
    if endpoint == "chat_completions":
        choices = response_body.get("choices")
        if isinstance(choices, list) and all(isinstance(choice, dict) for choice in choices):
            return True, "chat-compatible"
        return False, "chat-choices-missing"
    if (
        response_body.get("object") == "response"
        or isinstance(response_body.get("output"), list)
        or isinstance(response_body.get("output_text"), str)
    ):
        return True, "responses-compatible"
    return False, "responses-output-missing"


def _record_openai_cache_replay_bypass(cache_meta: dict[str, Any], replay_canary: dict[str, Any]) -> None:
    cache_meta["cache_replay_canary"] = replay_canary
    status = str(replay_canary.get("status") or "bypassed")
    reason = str(replay_canary.get("reason") or "cache-replay-canary-bypassed")
    cache_meta["status"] = status
    cache_meta["reason"] = reason
    if status == "invalidated":
        cache_meta["invalidated"] = True
        cache_meta["invalidation_reason"] = reason


def _openai_cache_replay_store_allowed(cache_meta: dict[str, Any], *, has_tool_blocks: bool) -> tuple[bool, str]:
    if has_tool_blocks and not bool(cache_meta.get("safe_invalidation_evidence")):
        audit = cache_meta.get("file_dependency_audit") if isinstance(cache_meta.get("file_dependency_audit"), dict) else {}
        return False, str(audit.get("invalidation_reason") or "file-dependency-missing")
    return True, "store-compatible"


async def _fetch_openai_old_context_summary(
    context: ProviderContext,
    summary_request: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    model = str(summary_request.get("model") or "")
    try:
        async with context.limiter.semaphores[model_tier(model)]:
            async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                await context.limiter.await_backoff(model)
                await context.limiter.throttle_forward()
                r = await client.post(
                    context.openai_upstream.rstrip("/") + "/v1/responses",
                    headers=headers,
                    json=summary_request,
                )
    except Exception:
        logging.exception("agentflow openai old-context summary error")
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

    input_tokens, output_tokens, _cache_read, _reasoning_tokens = usage_tokens(body)
    meta = {
        "summary": response_output_text(body).strip() if r.status_code < 400 else None,
        "summary_status_code": r.status_code,
        "summary_input_tokens": input_tokens,
        "summary_output_tokens": output_tokens,
    }
    if input_tokens is not None or output_tokens is not None:
        meta["summary_cost_est_usd"] = estimate_cost(
            model,
            input_tokens or 0,
            output_tokens or 0,
            provider="openai",
        ) or 0.0
    if r.status_code >= 400:
        meta["summary_error"] = stable_json(body)[:500]
    return meta


def _media_type(headers: httpx.Headers) -> Optional[str]:
    content_type = headers.get("content-type")
    if content_type:
        return content_type.split(";", 1)[0]
    return None


def _headers_for_client(headers: httpx.Headers) -> dict[str, str]:
    excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


def _session_id(request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return request.headers.get("x-session-id") or hashlib.sha256(
        (client_ip + datetime.now(timezone.utc).strftime("%Y-%m-%d")).encode()
    ).hexdigest()[:16]


async def _run_openai_routing_experiment(
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
    shadow_body = copy.deepcopy(request_body)
    shadow_body["model"] = shadow_model
    shadow_status_code: Optional[int] = None
    shadow_response_body: Optional[dict[str, Any]] = None
    shadow_latency_ms: Optional[int] = None
    shadow_cost: Optional[float] = None
    error: Optional[str] = None

    shadow_started = time.time()
    try:
        async with context.limiter.semaphores[model_tier(shadow_model)]:
            async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                await context.limiter.await_backoff(shadow_model)
                await context.limiter.throttle_forward()
                r = await client.post(context.openai_upstream.rstrip("/") + path, headers=headers, json=shadow_body)
        shadow_latency_ms = int((time.time() - shadow_started) * 1000)
        shadow_status_code = r.status_code
        try:
            shadow_response_body = r.json()
        except Exception:
            error = upstream_error_text(r.text, r.status_code)
            shadow_response_body = None
    except Exception as exc:
        shadow_latency_ms = int((time.time() - shadow_started) * 1000)
        error = repr(exc)

    if shadow_response_body is not None:
        shadow_in, shadow_out, cache_read, _reasoning_tokens = usage_tokens(shadow_response_body)
        shadow_out_est = estimate_tokens_from_text(response_output_text(shadow_response_body))
        shadow_cost = estimate_cost(
            shadow_model,
            shadow_in if shadow_in is not None else input_tokens_est,
            shadow_out if shadow_out is not None else shadow_out_est,
            cache_read=cache_read,
            provider="openai",
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
        provider=experiment_meta.get("provider") or "openai",
        source_surface=experiment_meta.get("source_surface") or "openai_responses",
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
        shadow_response_json=stable_json(shadow_response_body) if store_bodies and shadow_response_body is not None else None,
    )


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
            sid[:8],
            cost,
            calls,
            SESSION_COST_ALERT_USD,
        )


async def openai_passthrough(context: ProviderContext, request: Request, path: str) -> Response:
    if context.provider != "openai":
        return provider_disabled_response(context, "openai")
    headers = build_forward_headers(context, request, force_json=False)
    content = await request.body()
    async with httpx.AsyncClient(timeout=context.http_timeout) as client:
        r = await client.request(
            request.method,
            context.openai_upstream.rstrip("/") + path,
            headers=headers,
            content=content if content else None,
            params=dict(request.query_params),
        )
    return Response(
        r.content,
        status_code=r.status_code,
        media_type=_media_type(r.headers),
        headers=_headers_for_client(r.headers),
    )


async def openai_responses_websocket(context: ProviderContext, websocket: WebSocket) -> None:
    await websocket.accept()
    if context.provider != "openai":
        await websocket.close(code=1008, reason="provider mismatch")
        return

    headers = build_websocket_headers(context, websocket)
    upstream_url = websocket_url(context, "/v1/responses")
    try:
        async with websockets.connect(upstream_url, additional_headers=headers) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    msg = await websocket.receive()
                    msg_type = msg.get("type")
                    if msg_type == "websocket.disconnect":
                        await upstream.close()
                        return
                    if "text" in msg and msg["text"] is not None:
                        await upstream.send(msg["text"])
                    elif "bytes" in msg and msg["bytes"] is not None:
                        await upstream.send(msg["bytes"])

            async def upstream_to_client() -> None:
                async for msg in upstream:
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_text(msg)

            tasks = {
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
    except WebSocketDisconnect:
        return
    except Exception:
        logging.exception("agentflow openai websocket proxy error")
        try:
            await websocket.close(code=1011, reason=INTERNAL_PROXY_ERROR_MESSAGE)
        except Exception:
            pass


async def openai_optimized(context: ProviderContext, request: Request, path: str) -> Response:
    if context.provider != "openai":
        return provider_disabled_response(context, "openai")

    started = time.time()
    call_id = str(uuid.uuid4())
    session_id = _session_id(request)
    try:
        raw_body = await read_json_body(request)
    except ClientJsonRequestError as exc:
        latency_ms = int((time.time() - started) * 1000)
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=None, routed_model=None, stream=0, cache_hit=0,
            status_code=400, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None,
            actual_input_tokens=None, actual_output_tokens=None,
            cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=None, routing_json=None,
            cache_json=stable_json(cache_decision_meta("skipped", "invalid-json")),
            error=exc.message, request_json=None, response_json=None,
            session_id=session_id, category=None, retry_count=0,
            **openai_call_store_fields(path, None, None),
        )
        return JSONResponse(client_json_error_body("openai", exc.message), status_code=400)
    if raw_body is None:
        return await context.openai_passthrough_handler(context, request, path)
    parsed = parse_openai_request_body(raw_body, context.openai_model_list)
    stream = parsed.stream
    requested_model = parsed.requested_model
    category = parsed.category
    preflight = extract_openai_preflight_features(parsed, path=path)
    preflight_decision = await fetch_openai_policy_decision(preflight)
    status_code = 200
    error: Optional[str] = None
    crunch_meta: dict[str, Any] = {}
    routing_meta: dict[str, Any] = {}
    summary_meta: dict[str, Any] = {}
    cache_meta: dict[str, Any] = cache_decision_meta("skipped", "not-evaluated")
    retry_count = 0
    net_retries = 0

    try:
        local_policy = execute_openai_local_policy(
            raw_body=raw_body,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=stream,
            session_id=session_id,
            preflight=preflight,
            policy_decision=preflight_decision,
            store_obj=context.store,
            cruncher=crunch_body,
        )
        crunched = local_policy.provider_body
        crunch_meta = local_policy.crunch_meta
        routing_meta = local_policy.routing_meta
        input_tokens = local_policy.input_tokens_est
        resolved_requested_model = local_policy.resolved_requested_model
        managed_cache_profile = local_policy.managed_cache_profile
        headers = build_forward_headers(context, request)
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=stream,
            session_id=session_id,
        )
        if selected_openai_governor_family(routing_meta) == "routing":
            summary_meta = {
                "schema": "agentflow.openai_old_context_summary.v1",
                "enabled": False,
                "status": "suppressed",
                "applied": False,
                "changed": False,
                "reason_codes": ["conflicts-with-selected-family"],
                "policy_source": "local-default",
                "privacy": {
                    "raw_source_included": False,
                    "raw_summary_included": False,
                    "raw_request_body_included": False,
                    "summary_text_included": False,
                    "file_paths_included": False,
                    "cache_key_included": False,
                    "session_id_included": False,
                },
            }
        else:
            crunched, summary_meta = await maybe_apply_openai_old_context_summary(
                body=crunched,
                path=path,
                requested_model=requested_model,
                category=category,
                stream=stream,
                fetch_summary=lambda summary_request: _fetch_openai_old_context_summary(context, summary_request, headers),
                get_cached_summary=context.store.get_cache,
                set_cached_summary=lambda key, value: context.store.set_cache(
                    key,
                    str(value.get("summary_model") or summary_meta.get("summary_model") or "openai-summary"),
                    0,
                    value,
                ),
            )
            if summary_meta.get("applied"):
                input_tokens = input_tokens_after_summary(crunched)
        crunch_meta["old_context_summarization"] = summary_meta
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=stream,
            session_id=session_id,
        )
        call_fields = openai_call_store_fields(path, requested_model, str(crunched.get("model")))
        experiment_meta = routing_experiment_decision(
            crunched,
            routing_meta,
            stream=stream,
            provider="openai",
            source_surface=str(call_fields.get("source_surface") or "openai_responses"),
            store_obj=context.store,
        )
        routing_meta["routing_experiment"] = experiment_meta

        if stream:
            async def gen() -> AsyncIterator[bytes]:
                nonlocal status_code, error
                actual_in: Optional[int] = None
                actual_out: Optional[int] = None
                cache_read_in = 0
                reasoning_tokens = 0
                upstream_error_chunks: list[bytes] = []
                sse_frame_buf = b""
                stream_retry_count = 0
                stream_net_retries = 0

                def parse_sse_usage(frame: bytes) -> None:
                    nonlocal actual_in, actual_out, cache_read_in, reasoning_tokens
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
                        if data.get("type") == "response.completed":
                            data = data.get("response") or data
                        in_tok, out_tok, cached_tok, reason_tok = usage_tokens(data)
                        if in_tok is not None:
                            actual_in = in_tok
                        if out_tok is not None:
                            actual_out = out_tok
                        cache_read_in = max(cache_read_in, cached_tok)
                        reasoning_tokens = max(reasoning_tokens, reason_tok)

                try:
                    async with context.limiter.semaphores[model_tier(crunched["model"])]:
                        async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                            while True:
                                await context.limiter.await_backoff(crunched["model"])
                                await context.limiter.throttle_forward()
                                try:
                                    async with client.stream("POST", context.openai_upstream.rstrip("/") + path, headers=headers, json=crunched) as r:
                                        status_code = r.status_code
                                        if status_code in (429, 529) and stream_retry_count < 3:
                                            stream_retry_count += 1
                                            delay = (2 ** (stream_retry_count - 1)) * (1.0 + random.random() * 0.5)
                                            print(f"openai_rate_limit: status={status_code} retry={stream_retry_count} delay={delay:.1f}s")
                                            await context.limiter.record_backoff(crunched["model"], r.headers)
                                            if stream_retry_count == 1 and crunched.get("model") != resolved_requested_model:
                                                _rate_limited_model = crunched.get("model")
                                                crunched["model"] = resolved_requested_model
                                                routing_meta["fallback_reason"] = "rate_limited"
                                                routing_meta["fallback_model"] = resolved_requested_model
                                                openai_canary = routing_meta.get("openai_canary")
                                                if isinstance(openai_canary, dict):
                                                    openai_canary["fallback_reason"] = "rate_limited"
                                                    openai_canary["fallback_model"] = resolved_requested_model
                                                    openai_canary["actual_forwarded_model"] = resolved_requested_model
                                                print(f"openai_rate_limit_fallback: routing {_rate_limited_model!r} -> {resolved_requested_model!r}")
                                            await asyncio.sleep(delay)
                                            continue
                                        async for chunk in r.aiter_bytes():
                                            sse_frame_buf += chunk
                                            while b"\n\n" in sse_frame_buf:
                                                frame, sse_frame_buf = sse_frame_buf.split(b"\n\n", 1)
                                                event_bytes = frame + b"\n\n"
                                                if status_code >= 400:
                                                    upstream_error_chunks.append(event_bytes)
                                                yield event_bytes
                                                parse_sse_usage(frame)
                                        if sse_frame_buf:
                                            if status_code >= 400:
                                                upstream_error_chunks.append(sse_frame_buf)
                                            yield sse_frame_buf
                                            parse_sse_usage(sse_frame_buf)
                                            sse_frame_buf = b""
                                        break
                                except httpx.NetworkError as exc:
                                    if stream_net_retries < 2:
                                        stream_net_retries += 1
                                        print(f"openai_network_error: {exc!r} retry={stream_net_retries}", flush=True)
                                        await asyncio.sleep(2.0)
                                        continue
                                    raise
                except TierBackoffActive as exc:
                    status_code = 429
                    error = exc.message
                    payload = tier_backoff_payload(exc)
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
                except Exception as exc:
                    logging.exception("agentflow openai streaming proxy error")
                    status_code = 500
                    error = repr(exc)
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(public_proxy_error_body('openai', exc))}\n\n"
                    ).encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    cost_in = actual_in if actual_in is not None else input_tokens
                    cost_out = actual_out if actual_out is not None else 0
                    cost = add_summary_cost(
                        estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai"),
                        summary_meta,
                    )
                    cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
                    if status_code >= 400 and error is None:
                        error = upstream_error_text(b"".join(upstream_error_chunks), status_code)
                    stream_cache_meta = cache_decision_meta("skipped", "streaming")
                    stream_has_tool_blocks = has_tools(crunched)
                    if stream_has_tool_blocks or category in {"tool-light", "tool-result", "tool-heavy"}:
                        stream_file_deps = cache_file_dependency_snapshots(crunched)
                        attach_file_dependency_cache_meta(
                            stream_cache_meta,
                            snapshots=stream_file_deps,
                            audit=cache_file_dependency_audit(crunched),
                            blocker_reasons=_openai_cache_replay_blockers(
                                cache_meta=stream_cache_meta,
                                has_tool_blocks=stream_has_tool_blocks,
                                stream=True,
                            ),
                        )
                    attach_openai_optimization_governor(
                        routing_meta=routing_meta,
                        crunch_meta=crunch_meta,
                        cache_meta=stream_cache_meta,
                        summary_meta=summary_meta,
                        path=path,
                        requested_model=requested_model,
                        category=category,
                        stream=True,
                        session_id=session_id,
                    )
                    attach_openai_outcome_summary(
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
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=cache_read_in,
                        thinking_output_tokens=reasoning_tokens or None,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost_baseline,
                        cache_meta=stream_cache_meta,
                        crunch_meta=crunch_meta,
                        routing_meta=routing_meta,
                        category=category,
                        session_id=session_id,
                        error=error,
                    )
                    context.store.log_call(
                        id=call_id, created_at=utc_now(), path=path, provider="openai",
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(stream_cache_meta),
                        error=error, request_json=stable_json(crunched) if context.log_bodies else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=reasoning_tokens or None,
                        **openai_call_store_fields(path, requested_model, str(crunched.get("model"))),
                    )
                    await record_managed_outcome_feedback(
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
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=cache_read_in,
                        thinking_output_tokens=reasoning_tokens or None,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost_baseline,
                        cache_meta=stream_cache_meta,
                        crunch_meta=crunch_meta,
                        routing_meta=routing_meta,
                        category=category,
                        session_id=session_id,
                        error=error,
                    )
                    await _check_session_cost_alert(context, session_id)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-agentflow-cache": "skip-streaming", "x-agentflow-routed-model": str(crunched.get("model"))},
            )

        has_tool_blocks = has_tools(crunched)
        selected_before_cache = selected_openai_governor_family(routing_meta)
        if selected_before_cache in {"routing", "old_context_summary"}:
            can_cache = False
            can_semantic_cache = False
            cache_meta = cache_decision_meta("skipped", "conflicts-with-selected-family")
            cache_meta["conflicting_selected_family"] = selected_before_cache
        else:
            can_cache, can_semantic_cache, cache_meta = cache_lookup_meta(
                has_tool_blocks,
                pattern_features=routing_meta.get("managed_pattern_features"),
                store_obj=context.store,
                managed_profile=managed_cache_profile,
            )
        inspect_cache_dependencies = can_cache or (
            has_tool_blocks and selected_before_cache not in {"routing", "old_context_summary"}
        )
        file_deps = cache_file_dependency_snapshots(crunched) if inspect_cache_dependencies else []
        if inspect_cache_dependencies:
            attach_file_dependency_cache_meta(
                cache_meta,
                snapshots=file_deps,
                audit=cache_file_dependency_audit(crunched),
                blocker_reasons=_openai_cache_replay_blockers(
                    cache_meta=cache_meta,
                    has_tool_blocks=has_tool_blocks,
                    stream=False,
                ),
            )
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=False,
            session_id=session_id,
        )
        replay_scope, replay_scope_id, replay_pattern_rule = cache_replay_scope_for_meta(cache_meta, session_id)
        if replay_pattern_rule is not None:
            cache_meta["replay_scope"] = replay_scope
            cache_meta["replay_scope_id_available"] = bool(replay_scope_id)
            if not can_cache:
                _replay_allowed, replay_canary = cache_replay_canary_decision(
                    cache_meta=cache_meta,
                    dependency_audit=cache_meta.get("file_dependency_audit"),
                    session_id=session_id,
                )
                if replay_canary:
                    _record_openai_cache_replay_bypass(cache_meta, replay_canary)
        key = cache_key_for(
            crunched,
            path,
            provider="openai",
            upstream=context.openai_upstream,
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
                    _record_openai_cache_replay_bypass(cache_meta, replay_canary)
                    if replay_canary.get("status") == "invalidated":
                        context.store.delete_cache(key)
            cached = None
            if replay_allowed:
                cached, invalidated_reason = context.store.get_cache_with_reason(key)
                if invalidated_reason:
                    cache_meta["reason"] = invalidated_reason
                    cache_meta["invalidated"] = True
                    cache_meta["invalidation_reason"] = invalidated_reason
            if cached is not None:
                compatible, compatibility_reason = _openai_cache_replay_response_compatible(cached, path)
                if not compatible:
                    context.store.delete_cache(key)
                    cache_meta["status"] = "bypassed"
                    cache_meta["reason"] = compatibility_reason
                    cache_meta["cache_replay_shape"] = {
                        "status": "incompatible",
                        "reason": compatibility_reason,
                        "endpoint": _openai_endpoint_for_path(path),
                    }
                    cached = None
            if cached is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(cached))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                hit_cache_meta = cache_hit_decision_meta(
                    "exact-match",
                    hit_type="exact",
                    exact_enabled=can_cache,
                    semantic_enabled=can_semantic_cache,
                    lookup_meta=cache_meta,
                    estimated_saved_cost_usd=cost_baseline,
                )
                cost = add_summary_cost(0.0, summary_meta)
                attach_openai_optimization_governor(
                    routing_meta=routing_meta,
                    crunch_meta=crunch_meta,
                    cache_meta=hit_cache_meta,
                    summary_meta=summary_meta,
                    path=path,
                    requested_model=requested_model,
                    category=category,
                    stream=False,
                    session_id=session_id,
                )
                attach_openai_outcome_summary(
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
                    cost_est_usd=cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    session_id=session_id,
                )
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path, provider="openai",
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(hit_cache_meta),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(cached) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
                    **openai_call_store_fields(path, requested_model, str(crunched.get("model"))),
                )
                await record_managed_outcome_feedback(
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
                    cost_est_usd=cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    session_id=session_id,
                )
                return JSONResponse(cached, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        if can_semantic_cache:
            emb = build_embedding(extract_text(crunched))
            semantic_threshold = float(cache_meta.get("semantic_threshold", SEMANTIC_CACHE_THRESHOLD))
            sem_resp = context.store.get_semantic_cache(emb, str(crunched.get("model")), semantic_threshold)
            if sem_resp is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(sem_resp))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                hit_cache_meta = cache_hit_decision_meta(
                    "semantic-match",
                    hit_type="semantic",
                    exact_enabled=can_cache,
                    semantic_enabled=can_semantic_cache,
                    lookup_meta=cache_meta,
                    estimated_saved_cost_usd=cost_baseline,
                )
                cost = add_summary_cost(0.0, summary_meta)
                attach_openai_optimization_governor(
                    routing_meta=routing_meta,
                    crunch_meta=crunch_meta,
                    cache_meta=hit_cache_meta,
                    summary_meta=summary_meta,
                    path=path,
                    requested_model=requested_model,
                    category=category,
                    stream=False,
                    session_id=session_id,
                )
                attach_openai_outcome_summary(
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
                    cost_est_usd=cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    session_id=session_id,
                )
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path, provider="openai",
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(hit_cache_meta),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(sem_resp) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
                    **openai_call_store_fields(path, requested_model, str(crunched.get("model"))),
                )
                await record_managed_outcome_feedback(
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
                    cost_est_usd=cost,
                    cost_baseline_usd=cost_baseline,
                    cache_meta=hit_cache_meta,
                    crunch_meta=crunch_meta,
                    routing_meta=routing_meta,
                    category=category,
                    session_id=session_id,
                )
                return JSONResponse(sem_resp, headers={"x-agentflow-cache": "semantic-hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with context.limiter.semaphores[model_tier(crunched["model"])]:
            async with httpx.AsyncClient(timeout=context.http_timeout) as client:
                while True:
                    await context.limiter.await_backoff(crunched["model"])
                    await context.limiter.throttle_forward()
                    try:
                        r = await client.post(context.openai_upstream.rstrip("/") + path, headers=headers, json=crunched)
                    except httpx.NetworkError as exc:
                        if net_retries < 2:
                            net_retries += 1
                            print(f"openai_network_error: {exc!r} retry={net_retries}", flush=True)
                            await asyncio.sleep(2.0)
                            continue
                        raise
                    if r.status_code in (429, 529) and retry_count < 3:
                        retry_count += 1
                        delay = (2 ** (retry_count - 1)) * (1.0 + random.random() * 0.5)
                        print(f"openai_rate_limit: status={r.status_code} retry={retry_count} delay={delay:.1f}s")
                        await context.limiter.record_backoff(crunched["model"], r.headers)
                        if retry_count == 1 and crunched.get("model") != resolved_requested_model:
                            _rate_limited_model = crunched.get("model")
                            crunched["model"] = resolved_requested_model
                            routing_meta["fallback_reason"] = "rate_limited"
                            routing_meta["fallback_model"] = resolved_requested_model
                            openai_canary = routing_meta.get("openai_canary")
                            if isinstance(openai_canary, dict):
                                openai_canary["fallback_reason"] = "rate_limited"
                                openai_canary["fallback_model"] = resolved_requested_model
                                openai_canary["actual_forwarded_model"] = resolved_requested_model
                            print(f"openai_rate_limit_fallback: routing {_rate_limited_model!r} -> {resolved_requested_model!r}")
                        await asyncio.sleep(delay)
                        continue
                    break

        status_code = r.status_code
        try:
            response_body = r.json()
        except Exception:
            latency_ms = int((time.time() - started) * 1000)
            cost = add_summary_cost(
                estimate_cost(str(crunched.get("model")), input_tokens, 0, provider="openai"),
                summary_meta,
            )
            cost_baseline = estimate_cost(requested_model, input_tokens, 0, provider="openai")
            error = upstream_error_text(r.text, status_code)
            attach_openai_optimization_governor(
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                summary_meta=summary_meta,
                path=path,
                requested_model=requested_model,
                category=category,
                stream=False,
                session_id=session_id,
            )
            attach_openai_outcome_summary(
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
                session_id=session_id,
                error=error,
            )
            context.store.log_call(
                id=call_id, created_at=utc_now(), path=path, provider="openai",
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
                **openai_call_store_fields(path, requested_model, str(crunched.get("model"))),
            )
            await record_managed_outcome_feedback(
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
                session_id=session_id,
                error=error,
            )
            return Response(
                r.content,
                status_code=status_code,
                media_type=_media_type(r.headers),
                headers=_headers_for_client(r.headers),
            )

        if r.status_code < 400 and can_cache and response_body is not None:
            compatible, compatibility_reason = _openai_cache_replay_response_compatible(response_body, path)
            store_allowed, store_reason = _openai_cache_replay_store_allowed(cache_meta, has_tool_blocks=has_tool_blocks)
            if compatible and store_allowed:
                context.store.set_cache(
                    key,
                    str(crunched.get("model")),
                    len(stable_json(crunched)),
                    response_body,
                    file_deps=file_deps,
                )
                if replay_pattern_rule is not None:
                    cache_meta["cache_replay_store"] = {
                        "status": "stored",
                        "reason": "compatible-success-response",
                        "endpoint": _openai_endpoint_for_path(path),
                        "response_shape": compatibility_reason,
                        "cache_key_included": False,
                    }
            elif replay_pattern_rule is not None:
                cache_meta["cache_replay_store"] = {
                    "status": "skipped",
                    "reason": compatibility_reason if not compatible else store_reason,
                    "endpoint": _openai_endpoint_for_path(path),
                    "cache_key_included": False,
                }
        if can_semantic_cache and emb is not None and r.status_code < 400 and response_body is not None:
            context.store.set_semantic_cache(key, str(crunched.get("model")), emb, response_body, len(stable_json(crunched)))

        actual_in, actual_out, cache_read_in, reasoning_tokens = usage_tokens(response_body)
        out_tokens = estimate_tokens_from_text(response_output_text(response_body)) if response_body else 0
        cost_in = actual_in if actual_in is not None else input_tokens
        cost_out = actual_out if actual_out is not None else out_tokens
        cost = add_summary_cost(
            estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai"),
            summary_meta,
        )
        cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
        latency_ms = int((time.time() - started) * 1000)
        error = None if status_code < 400 else upstream_error_text(response_body, status_code)
        if experiment_meta.get("sampled") and status_code < 400 and response_body is not None:
            await _run_openai_routing_experiment(
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
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=False,
            session_id=session_id,
        )
        attach_openai_outcome_summary(
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
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read_in,
            thinking_output_tokens=reasoning_tokens or None,
            cost_est_usd=cost,
            cost_baseline_usd=cost_baseline,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            session_id=session_id,
            error=error,
        )
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
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
            cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
            thinking_output_tokens=reasoning_tokens or None,
            **openai_call_store_fields(path, requested_model, str(crunched.get("model"))),
        )
        await record_managed_outcome_feedback(
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
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read_in,
            thinking_output_tokens=reasoning_tokens or None,
            cost_est_usd=cost,
            cost_baseline_usd=cost_baseline,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            session_id=session_id,
            error=error,
        )
        await _check_session_cost_alert(context, session_id)
        return JSONResponse(
            response_body,
            status_code=status_code,
            headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))},
        )
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
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=stream,
            session_id=session_id,
        )
        attach_openai_outcome_summary(
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
            session_id=session_id,
            error=error,
        )
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=routed_model_for_log, stream=int(stream), cache_hit=0,
            status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if context.log_bodies else None,
            response_json=stable_json(response_body) if context.log_bodies else None,
            session_id=session_id, category=category, retry_count=retry_count,
            **openai_call_store_fields(path, requested_model, routed_model_for_log),
        )
        await record_managed_outcome_feedback(
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
            session_id=session_id,
            error=error,
        )
        return JSONResponse(
            response_body,
            status_code=status_code,
            headers=tier_backoff_headers(exc, routed_model_for_log or ""),
        )
    except Exception as exc:
        logging.exception("agentflow openai proxy error")
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        attach_openai_optimization_governor(
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            summary_meta=summary_meta,
            path=path,
            requested_model=requested_model,
            category=category,
            stream=stream,
            session_id=session_id,
        )
        attach_openai_outcome_summary(
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
            session_id=session_id,
            error=error,
        )
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if context.log_bodies else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
            **openai_call_store_fields(path, requested_model, None),
        )
        await record_managed_outcome_feedback(
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
            session_id=session_id,
            error=error,
        )
        return JSONResponse(public_proxy_error_body("openai", exc), status_code=500)
