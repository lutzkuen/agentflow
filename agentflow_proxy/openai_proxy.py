from __future__ import annotations

import asyncio
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
    cache_decision_meta,
    cache_file_dependency_snapshots,
    cache_key_for,
    cache_lookup_meta,
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
from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.provider_context import ProviderContext
from agentflow_proxy.recommendations import (
    apply_recommendation_to_body,
    build_optimization_unit,
    fetch_recommendation,
)
from agentflow_proxy.router import categorize_request, extract_text, has_tools, route_openai_model
from agentflow_proxy.store import stable_json, utc_now


SESSION_COST_ALERT_USD = float(os.getenv("AGENTFLOW_SESSION_COST_ALERT_USD", "5.0"))


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
        )
        return JSONResponse(client_json_error_body("openai", exc.message), status_code=400)
    if raw_body is None:
        return await context.openai_passthrough_handler(context, request, path)
    stream = bool(raw_body.get("stream"))
    requested_model = str(raw_body.get("model") or context.openai_model_list[0])
    raw_body.setdefault("model", requested_model)
    category = categorize_request(raw_body)
    status_code = 200
    error: Optional[str] = None
    crunch_meta: dict[str, Any] = {}
    routing_meta: dict[str, Any] = {}
    cache_meta: dict[str, Any] = cache_decision_meta("skipped", "not-evaluated")
    retry_count = 0
    net_retries = 0

    try:
        crunched, crunch_meta = crunch_body(raw_body)
        routed_model, routing_meta = route_openai_model(crunched)
        resolved_requested_model = str(crunched.get("model") or requested_model)
        crunched["model"] = routed_model
        input_tokens = estimate_tokens_from_text(extract_text(crunched))
        recommendation_unit = build_optimization_unit(
            provider="openai",
            path=path,
            requested_model=resolved_requested_model,
            routed_model=str(crunched.get("model") or routed_model),
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
            category=category,
            stream=stream,
            input_tokens_est=input_tokens,
        )
        recommendation_meta = await fetch_recommendation(recommendation_unit)
        recommendation_meta = apply_recommendation_to_body(
            provider="openai",
            body=crunched,
            routing_meta=routing_meta,
            recommendation_meta=recommendation_meta,
        )
        routing_meta["managed_recommendation"] = recommendation_meta
        headers = build_forward_headers(context, request)

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
                    cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai")
                    cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
                    if status_code >= 400 and error is None:
                        error = upstream_error_text(b"".join(upstream_error_chunks), status_code)
                    context.store.log_call(
                        id=call_id, created_at=utc_now(), path=path, provider="openai",
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_decision_meta("skipped", "streaming")),
                        error=error, request_json=stable_json(crunched) if context.log_bodies else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=reasoning_tokens or None,
                    )
                    await _check_session_cost_alert(context, session_id)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-agentflow-cache": "skip-streaming", "x-agentflow-routed-model": str(crunched.get("model"))},
            )

        has_tool_blocks = has_tools(crunched)
        can_cache, can_semantic_cache, cache_meta = cache_lookup_meta(has_tool_blocks)
        key = cache_key_for(
            crunched,
            path,
            provider="openai",
            upstream=context.openai_upstream,
        )
        emb: Optional[list[float]] = None
        if can_cache:
            cached, invalidated_reason = context.store.get_cache_with_reason(key)
            if invalidated_reason:
                cache_meta["reason"] = invalidated_reason
                cache_meta["invalidated"] = True
            if cached is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(cached))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path, provider="openai",
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=0.0, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta(
                        "hit",
                        "exact-match",
                        hit_type="exact",
                        exact_enabled=can_cache,
                        semantic_enabled=can_semantic_cache,
                    )),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(cached) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(cached, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        if can_semantic_cache:
            emb = build_embedding(extract_text(crunched))
            sem_resp = context.store.get_semantic_cache(emb, str(crunched.get("model")), SEMANTIC_CACHE_THRESHOLD)
            if sem_resp is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(sem_resp))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                context.store.log_call(
                    id=call_id, created_at=utc_now(), path=path, provider="openai",
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=0.0, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta(
                        "hit",
                        "semantic-match",
                        hit_type="semantic",
                        exact_enabled=can_cache,
                        semantic_enabled=can_semantic_cache,
                    )),
                    error=None, request_json=stable_json(crunched) if context.log_bodies else None,
                    response_json=stable_json(sem_resp) if context.log_bodies else None,
                    session_id=session_id, category=category, retry_count=0,
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
                            print(f"openai_rate_limit_fallback: routing {_rate_limited_model!r} -> {resolved_requested_model!r}")
                        await asyncio.sleep(delay)
                        continue
                    break

        status_code = r.status_code
        try:
            response_body = r.json()
        except Exception:
            latency_ms = int((time.time() - started) * 1000)
            cost = estimate_cost(str(crunched.get("model")), input_tokens, 0, provider="openai")
            cost_baseline = estimate_cost(requested_model, input_tokens, 0, provider="openai")
            context.store.log_call(
                id=call_id, created_at=utc_now(), path=path, provider="openai",
                requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                input_tokens_est=input_tokens, output_tokens_est=None,
                actual_input_tokens=None, actual_output_tokens=None,
                cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                cache_json=stable_json(cache_meta),
                error=upstream_error_text(r.text, status_code),
                request_json=stable_json(crunched) if context.log_bodies else None, response_json=None,
                session_id=session_id, category=category,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                retry_count=retry_count,
            )
            return Response(
                r.content,
                status_code=status_code,
                media_type=_media_type(r.headers),
                headers=_headers_for_client(r.headers),
            )

        if r.status_code < 400 and can_cache and response_body is not None:
            context.store.set_cache(
                key,
                str(crunched.get("model")),
                len(stable_json(crunched)),
                response_body,
                file_deps=cache_file_dependency_snapshots(crunched),
            )
        if can_semantic_cache and emb is not None and r.status_code < 400 and response_body is not None:
            context.store.set_semantic_cache(key, str(crunched.get("model")), emb, response_body, len(stable_json(crunched)))

        actual_in, actual_out, cache_read_in, reasoning_tokens = usage_tokens(response_body)
        out_tokens = estimate_tokens_from_text(response_output_text(response_body)) if response_body else 0
        cost_in = actual_in if actual_in is not None else input_tokens
        cost_out = actual_out if actual_out is not None else out_tokens
        cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai")
        cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
        latency_ms = int((time.time() - started) * 1000)
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=None if status_code < 400 else upstream_error_text(response_body, status_code),
            request_json=stable_json(crunched) if context.log_bodies else None,
            response_json=stable_json(response_body) if context.log_bodies else None,
            session_id=session_id, category=category,
            cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
            thinking_output_tokens=reasoning_tokens or None,
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
        context.store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if context.log_bodies else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse(public_proxy_error_body("openai", exc), status_code=500)
