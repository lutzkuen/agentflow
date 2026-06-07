from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

PROVIDER = os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()
ANTHROPIC_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
OPENAI_UPSTREAM = os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com")
DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_PORT = int(os.getenv("AGENTFLOW_PORT", "4000"))
DEFAULT_HOST = os.getenv("AGENTFLOW_HOST", "0.0.0.0")

LOG_BODIES = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
HTTP_TIMEOUT = float(os.getenv("AGENTFLOW_HTTP_TIMEOUT", "600"))
MIN_REQUEST_INTERVAL_MS = int(os.getenv("AGENTFLOW_MIN_REQUEST_INTERVAL_MS", "0"))
MAX_TIER_BACKOFF_WAIT = float(os.getenv("AGENTFLOW_MAX_TIER_BACKOFF_WAIT", "30"))
# 0 = disabled (unlimited concurrency). Default 2 prevents burst collisions before global backoff coordinates.
MAX_CONCURRENT_PER_TIER = int(os.getenv("AGENTFLOW_MAX_CONCURRENT_PER_TIER", "2"))
SESSION_COST_ALERT_USD = float(os.getenv("AGENTFLOW_SESSION_COST_ALERT_USD", "5.0"))
# 0 = disabled (no cap). Set to a positive int to cap thinking budget_tokens per turn.
MAX_THINKING_BUDGET_TOKENS = int(os.getenv("AGENTFLOW_MAX_THINKING_BUDGET_TOKENS", "0"))
OPENAI_MODEL_LIST = list(dict.fromkeys([
    os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL", "gpt-5-codex"),
    os.getenv("AGENTFLOW_OPENAI_SMALL_MODEL", "gpt-5-mini"),
    os.getenv("AGENTFLOW_OPENAI_TINY_MODEL", "gpt-5-nano"),
    "gpt-5.5",
    "gpt-5.2-codex",
    "gpt-5-codex",
]))
OPENAI_AUTH_MODE = os.getenv("AGENTFLOW_OPENAI_AUTH_MODE", "client").lower()


def _model_tier(model: str) -> str:
    return model_tier(model)


def _tier_backoff_status(now: Optional[float] = None) -> list[dict[str, Any]]:
    return _limiter.status(now)



async def _await_tier_backoff(model: str) -> None:
    await _limiter.await_backoff(model)


def _tier_backoff_payload(exc: TierBackoffActive) -> dict[str, Any]:
    return tier_backoff_payload(exc)


def _tier_backoff_headers(exc: TierBackoffActive, model: str) -> dict[str, str]:
    return tier_backoff_headers(exc, model)


async def _record_tier_backoff(model: str, response_headers: Any, default_seconds: float = 60.0) -> None:
    await _limiter.record_backoff(model, response_headers, default_seconds)


async def _throttle_forward() -> None:
    await _limiter.throttle_forward()


async def _check_session_cost_alert(sid: str) -> None:
    row = store.conn.execute(
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


def _recent_session_spending_summary(hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
    rows = store.conn.execute("""
        SELECT session_id,
               coalesce(provider, 'anthropic') as provider,
               requested_model,
               routed_model,
               coalesce(routed_model, requested_model) as model,
               COUNT(*) as calls,
               SUM(coalesce(cost_est_usd, 0)) as cost_usd,
               SUM(coalesce(cost_baseline_usd, 0)) as baseline_usd,
               SUM(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tokens,
               SUM(coalesce(actual_output_tokens, output_tokens_est, 0)) as output_tokens,
               SUM(coalesce(thinking_output_tokens, 0)) as thinking_tokens,
               SUM(coalesce(cache_creation_input_tokens, 0)) as cache_creation_tokens,
               SUM(coalesce(cache_read_input_tokens, 0)) as cache_read_tokens
        FROM calls
        WHERE datetime(created_at) >= datetime('now', ?)
          AND session_id IS NOT NULL
        GROUP BY session_id,
                 coalesce(provider, 'anthropic'),
                 requested_model,
                 routed_model
    """, (f"-{int(hours)} hours",)).fetchall()

    by_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        session_id = row["session_id"]
        bucket = by_session.setdefault(
            session_id,
            {
                "session_id": session_id,
                "sid": session_id[:8],
                "calls": 0,
                "cost_usd": 0.0,
                "baseline_savings_usd": 0.0,
                "routing_savings_usd": 0.0,
                "prompt_cache_savings_usd": 0.0,
                "thinking_tokens": 0,
                "thinking_cost_usd": 0.0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            },
        )
        calls = int(row["calls"] or 0)
        cost = float(row["cost_usd"] or 0.0)
        baseline = float(row["baseline_usd"] or 0.0)
        provider = str(row["provider"] or "anthropic").lower()
        requested_model = row["requested_model"]
        routed_model = row["routed_model"]
        model = row["model"]
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        thinking_tokens = int(row["thinking_tokens"] or 0)
        cache_creation_tokens = int(row["cache_creation_tokens"] or 0)
        cache_read_tokens = int(row["cache_read_tokens"] or 0)

        bucket["calls"] += calls
        bucket["cost_usd"] += cost
        bucket["baseline_savings_usd"] += max(baseline - cost, 0.0)
        bucket["thinking_tokens"] += thinking_tokens
        bucket["cache_creation_tokens"] += cache_creation_tokens
        bucket["cache_read_tokens"] += cache_read_tokens

        if requested_model and routed_model and requested_model != routed_model:
            requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider=provider) or 0.0
            routed_cost = estimate_cost(routed_model, input_tokens, output_tokens, provider=provider) or 0.0
            bucket["routing_savings_usd"] += max(requested_cost - routed_cost, 0.0)
        if cache_read_tokens:
            full_read_cost = estimate_cost(model, cache_read_tokens, 0, provider=provider) or 0.0
            bucket["prompt_cache_savings_usd"] += 0.90 * full_read_cost
        if thinking_tokens:
            bucket["thinking_cost_usd"] += estimate_cost(model, 0, thinking_tokens, provider=provider) or 0.0

    summaries = sorted(by_session.values(), key=lambda r: r["cost_usd"], reverse=True)[:limit]
    for row in summaries:
        for key in (
            "cost_usd",
            "baseline_savings_usd",
            "routing_savings_usd",
            "prompt_cache_savings_usd",
            "thinking_cost_usd",
        ):
            row[key] = round(float(row[key]), 6)
    return summaries


def _log_recent_session_spending_summary(event: str, hours: int = 24, limit: int = 10) -> None:
    rows = _recent_session_spending_summary(hours=hours, limit=limit)
    if not rows:
        print(f"agentflow_session_summary event={event} window={hours}h sessions=0", file=sys.stderr)
        return
    for row in rows:
        print(
            "agentflow_session_summary "
            f"event={event} window={hours}h session={row['sid']} calls={row['calls']} "
            f"cost_usd={row['cost_usd']:.4f} baseline_savings_usd={row['baseline_savings_usd']:.4f} "
            f"routing_savings_usd={row['routing_savings_usd']:.4f} "
            f"prompt_cache_savings_usd={row['prompt_cache_savings_usd']:.4f} "
            f"thinking_tokens={row['thinking_tokens']} thinking_cost_usd={row['thinking_cost_usd']:.4f} "
            f"cache_creation_tokens={row['cache_creation_tokens']} cache_read_tokens={row['cache_read_tokens']}",
            file=sys.stderr,
        )


from agentflow_proxy.store import Store, utc_now, stable_json
from agentflow_proxy import provider_handlers
from agentflow_proxy import stats as stats_views
from agentflow_proxy.pricing import MODEL_PRICES, MODEL_ALIASES, estimate_blended_input_savings, estimate_cost
from agentflow_proxy.headers import (
    ClientJsonRequestError,
    build_anthropic_forward_headers,
    build_openai_forward_headers as build_openai_forward_headers_from_mapping,
    build_openai_websocket_headers as build_openai_websocket_headers_from_mapping,
    client_json_error_body,
    read_json_object_body,
)
from agentflow_proxy.limiter import (
    TierBackoffActive,
    TierLimiter,
    model_tier,
    tier_backoff_headers,
    tier_backoff_payload,
)
from agentflow_proxy.router import (
    extract_text, has_tools, categorize_request, route_model,
    HAIKU_DEFAULT, SONNET_DEFAULT, OPUS_DEFAULT,
    route_openai_model,
    STRIP_THINKING_HISTORY, _has_top_level_thinking, strip_thinking_history_blocks,
)
from agentflow_proxy.crunch import (
    TOKEN_CHARS, sha256_text, estimate_tokens_from_text, build_embedding,
    crunch_body, inject_prompt_cache, has_cache_control_blocks,
    maybe_summarize_old_context, OLD_CONTEXT_SUMMARY_MODEL,
)
from agentflow_proxy.cache import (
    CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD,
    cache_decision_meta, cache_key_for, cache_lookup_meta, response_output_text,
    is_stream_cache_payload, stream_cache_frames, stream_cache_payload,
    streaming_cache_lookup_meta,
    cache_file_dependency_snapshots,
)
from agentflow_proxy.errors import (
    INTERNAL_PROXY_ERROR_MESSAGE,
    public_proxy_error_body,
    upstream_error_text,
)
from agentflow_proxy.routing_experiments import (
    ROUTING_EXPERIMENT_MIN_SAMPLES,
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES,
    compare_response_outputs,
    routing_experiment_decision,
)
from agentflow_proxy.recommendations import (
    apply_recommendation_to_body,
    build_optimization_unit,
    fetch_recommendation,
)


_limiter = TierLimiter(
    min_request_interval_ms=MIN_REQUEST_INTERVAL_MS,
    max_tier_backoff_wait=MAX_TIER_BACKOFF_WAIT,
    max_concurrent_per_tier=MAX_CONCURRENT_PER_TIER,
)
_tier_backoff_until = _limiter.backoff_until
_tier_semaphores = _limiter.semaphores

store = Store(DEFAULT_DB)
app = FastAPI(title=f"AgentFlow {PROVIDER.title()} Proxy", version="0.1.0")
_SERVER_CONTEXT = sys.modules[__name__]


@app.on_event("startup")
async def _log_startup_session_spending_summary() -> None:
    _log_recent_session_spending_summary("startup")


@app.on_event("shutdown")
async def _log_shutdown_session_spending_summary() -> None:
    _log_recent_session_spending_summary("shutdown")


def configure_provider(
    provider: str,
    anthropic_upstream: str = ANTHROPIC_UPSTREAM,
    openai_upstream: str = OPENAI_UPSTREAM,
    openai_auth_mode: str = OPENAI_AUTH_MODE,
) -> None:
    global PROVIDER, ANTHROPIC_UPSTREAM, OPENAI_UPSTREAM, DEFAULT_UPSTREAM, OPENAI_AUTH_MODE
    provider = provider.lower()
    if provider not in {"anthropic", "openai"}:
        raise ValueError("provider must be 'anthropic' or 'openai'")
    openai_auth_mode = openai_auth_mode.lower()
    if openai_auth_mode not in {"client", "proxy"}:
        raise ValueError("openai auth mode must be 'client' or 'proxy'")
    PROVIDER = provider
    ANTHROPIC_UPSTREAM = anthropic_upstream
    OPENAI_UPSTREAM = openai_upstream
    OPENAI_AUTH_MODE = openai_auth_mode
    DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
    app.title = f"AgentFlow {PROVIDER.title()} Proxy"


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


def provider_disabled_response(expected: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": f"AgentFlow is running in {PROVIDER!r} provider mode, not {expected!r}.",
                "type": "provider_mismatch",
            }
        },
        status_code=404,
    )


def build_forward_headers(request: Request) -> dict[str, str]:
    return build_anthropic_forward_headers(request.headers)


async def _fetch_old_context_summary(summary_request: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    model = str(summary_request.get("model") or OLD_CONTEXT_SUMMARY_MODEL)
    try:
        async with _tier_semaphores[_model_tier(model)]:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                await _await_tier_backoff(model)
                await _throttle_forward()
                r = await client.post(
                    ANTHROPIC_UPSTREAM.rstrip("/") + "/v1/messages",
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
    primary_model = str(request_body.get("model") or routing_meta.get("routed_model") or "")
    shadow_body = copy.deepcopy(request_body)
    shadow_body["model"] = shadow_model
    shadow_status_code: Optional[int] = None
    shadow_response_body: Optional[dict[str, Any]] = None
    shadow_latency_ms: Optional[int] = None
    shadow_cost: Optional[float] = None
    error: Optional[str] = None

    shadow_started = time.time()
    try:
        async with _tier_semaphores[_model_tier(shadow_model)]:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                await _await_tier_backoff(shadow_model)
                await _throttle_forward()
                r = await client.post(ANTHROPIC_UPSTREAM.rstrip("/") + path, headers=headers, json=shadow_body)
        shadow_latency_ms = int((time.time() - shadow_started) * 1000)
        shadow_status_code = r.status_code
        try:
            shadow_response_body = r.json()
        except Exception:
            error = r.text[:1000]
            shadow_response_body = None
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
    store_bodies = LOG_BODIES or ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES
    store.log_routing_experiment(
        id=experiment_id,
        call_id=call_id,
        created_at=utc_now(),
        requested_model=routing_meta.get("requested_model"),
        routed_model=routing_meta.get("routed_model"),
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
        error=error,
        routing_json=stable_json(routing_meta),
        experiment_json=stable_json(experiment_meta),
        primary_response_json=stable_json(primary_response_body) if store_bodies else None,
        shadow_response_json=stable_json(shadow_response_body) if store_bodies and shadow_response_body is not None else None,
    )


def build_openai_forward_headers(request: Request, *, force_json: bool = True) -> dict[str, str]:
    return build_openai_forward_headers_from_mapping(
        request.headers,
        auth_mode=OPENAI_AUTH_MODE,
        api_key=os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        force_json=force_json,
    )


async def read_openai_json_body(request: Request) -> Optional[dict[str, Any]]:
    return await read_json_object_body(
        request,
        allow_compressed=True,
        passthrough_unsupported_encoding=True,
    )


def build_openai_websocket_headers(websocket: WebSocket) -> dict[str, str]:
    return build_openai_websocket_headers_from_mapping(
        websocket.headers,
        auth_mode=OPENAI_AUTH_MODE,
        api_key=os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )


def openai_websocket_url(path: str) -> str:
    base = OPENAI_UPSTREAM.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return base + path


def openai_usage_tokens(body: dict[str, Any]) -> tuple[Optional[int], Optional[int], int, int]:
    usage = (body or {}).get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    reasoning_tokens = int(output_details.get("reasoning_tokens") or 0)
    return input_tokens, output_tokens, cached_tokens, reasoning_tokens


def openai_response_output_text(resp: dict[str, Any]) -> str:
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


def _openai_passthrough_media_type(headers: httpx.Headers) -> Optional[str]:
    content_type = headers.get("content-type")
    if content_type:
        return content_type.split(";", 1)[0]
    return None


def _openai_headers_for_client(headers: httpx.Headers) -> dict[str, str]:
    excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


def _openai_session_id(request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return request.headers.get("x-session-id") or hashlib.sha256(
        (client_ip + datetime.now(timezone.utc).strftime("%Y-%m-%d")).encode()
    ).hexdigest()[:16]


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "provider": PROVIDER,
        "db": DEFAULT_DB,
        "upstream": DEFAULT_UPSTREAM,
        "openai_auth_mode": OPENAI_AUTH_MODE if PROVIDER == "openai" else None,
        "time": utc_now(),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    if PROVIDER == "openai":
        return {"object": "list", "data": [{"id": model, "object": "model"} for model in OPENAI_MODEL_LIST]}
    return {
        "data": [
            {"id": HAIKU_DEFAULT, "type": "model"},
            {"id": SONNET_DEFAULT, "type": "model"},
            {"id": OPUS_DEFAULT, "type": "model"},
        ]
    }


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    return await provider_handlers.anthropic_messages(_SERVER_CONTEXT, request)


async def _anthropic_messages_impl(request: Request) -> Response:
    if PROVIDER != "anthropic":
        return provider_disabled_response("anthropic")
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
        store.log_call(
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
        raw_body, summary_meta = await maybe_summarize_old_context(
            raw_body,
            exact_cache_enabled=CACHE_ENABLED,
            get_cached_summary=store.get_cache,
            set_cached_summary=lambda key, value: store.set_cache(
                key,
                OLD_CONTEXT_SUMMARY_MODEL,
                len(stable_json(value)),
                value,
            ),
            fetch_summary=lambda summary_request: _fetch_old_context_summary(summary_request, headers),
        )
        summary_extra_cost = float(summary_meta.get("summary_cost_est_usd") or 0.0)
        crunched, crunch_meta = crunch_body(raw_body)
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
        routed_model, routing_meta = route_model(crunched)
        if _n_stripped > 0:
            routing_meta["thinking_history_stripped"] = _n_stripped
        resolved_requested_model = crunched.get("model", requested_model)
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
        input_tokens = estimate_tokens_from_text(extract_text(crunched))
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
        )
        recommendation_meta = await fetch_recommendation(recommendation_unit)
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
        _strip_model_incompatible_params(crunched, routing_meta, str(resolved_requested_model))
        if prompt_cached or has_cache_control_blocks(crunched):
            existing = headers.get("anthropic-beta", "")
            if "prompt-caching" not in existing:
                headers["anthropic-beta"] = (existing + ",prompt-caching-2024-07-31" if existing else "prompt-caching-2024-07-31")

        if stream:
            has_tool_blocks = has_tools(crunched)
            can_stream_cache, cache_meta = streaming_cache_lookup_meta(has_tool_blocks)
            key = cache_key_for(
                crunched,
                path,
                provider="anthropic",
                upstream=ANTHROPIC_UPSTREAM,
            )
            if can_stream_cache:
                cached = store.get_cache(key)
                if is_stream_cache_payload(cached, provider="anthropic"):
                    cached_frames = stream_cache_frames(cached)
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
                            store.log_call(
                                id=call_id, created_at=utc_now(), path=path,
                                requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                                cache_hit=1, status_code=200, latency_ms=latency_ms,
                                input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                                actual_input_tokens=None, actual_output_tokens=None,
                                cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                                cache_json=stable_json(cache_decision_meta(
                                    "hit",
                                    "streaming-exact-match",
                                    hit_type="streaming-exact",
                                    exact_enabled=can_stream_cache,
                                    semantic_enabled=False,
                                )),
                                error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                                response_json=stable_json(cached) if LOG_BODIES else None,
                                session_id=session_id, category=category,
                                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                                retry_count=0,
                            )
                            await _check_session_cost_alert(session_id)

                    return StreamingResponse(
                        replay_cached_stream(),
                        media_type="text/event-stream",
                        headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))},
                    )

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

                def parse_sse_usage(frame: bytes) -> None:
                    nonlocal actual_in, actual_out, cache_creation_in, cache_read_in, thinking_chars
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
                        elif t == "message_delta":
                            out = (data.get("usage") or {}).get("output_tokens")
                            if out is not None:
                                actual_out = out
                        elif t == "content_block_delta":
                            delta = (data.get("delta") or {})
                            if delta.get("type") == "thinking_delta":
                                thinking_chars += len(delta.get("thinking") or "")
                            elif delta.get("type") == "text_delta":
                                output_text_parts.append(str(delta.get("text") or ""))

                try:
                    async with _tier_semaphores[_model_tier(crunched["model"])]:
                        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                            while True:
                                await _await_tier_backoff(crunched["model"])
                                await _throttle_forward()
                                try:
                                    async with client.stream("POST", ANTHROPIC_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched) as r:
                                        status_code = r.status_code
                                        if status_code in (429, 529) and stream_retry_count < 3:
                                            stream_retry_count += 1
                                            delay = (2 ** (stream_retry_count - 1)) * (1.0 + random.random() * 0.5)
                                            print(f"rate_limit: status={status_code} retry={stream_retry_count} delay={delay:.1f}s")
                                            await _record_tier_backoff(crunched["model"], r.headers)
                                            if stream_retry_count == 1 and crunched.get("model") != resolved_requested_model:
                                                _rate_limited_model = crunched.get("model")
                                                crunched["model"] = resolved_requested_model
                                                routing_meta["fallback_reason"] = "rate_limited"
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
                    payload = _tier_backoff_payload(exc)
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
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
                        store.set_cache(
                            key,
                            str(crunched.get("model")),
                            len(stable_json(crunched)),
                            stream_cache_payload(
                                stream_frames,
                                provider="anthropic",
                                usage=stream_usage,
                                output_text="".join(output_text_parts),
                            ),
                            file_deps=cache_file_dependency_snapshots(crunched),
                        )
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path,
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_meta),
                        error=error, request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=thinking_chars // TOKEN_CHARS if thinking_chars else None,
                    )
                    await _check_session_cost_alert(session_id)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"x-agentflow-cache": "miss" if can_stream_cache else "skip-streaming", "x-agentflow-routed-model": str(crunched.get("model"))},
            )

        has_tool_blocks = has_tools(crunched)
        can_cache, can_semantic_cache, cache_meta = cache_lookup_meta(has_tool_blocks)
        key = cache_key_for(
            crunched,
            path,
            provider="anthropic",
            upstream=ANTHROPIC_UPSTREAM,
        )
        emb: Optional[list[float]] = None
        if can_cache:
            cached = store.get_cache(key)
            if cached is not None:
                cache_hit = True
                response_body = cached
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(response_body))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens)
                store.log_call(
                    id=call_id, created_at=utc_now(), path=path,
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta(
                        "hit",
                        "exact-match",
                        hit_type="exact",
                        exact_enabled=can_cache,
                        semantic_enabled=can_semantic_cache,
                    )),
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(response_body) if LOG_BODIES else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(response_body, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        if can_semantic_cache:
            emb = build_embedding(extract_text(crunched))
            sem_resp = store.get_semantic_cache(emb, str(crunched.get("model")), SEMANTIC_CACHE_THRESHOLD)
            if sem_resp is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(sem_resp))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens)
                store.log_call(
                    id=call_id, created_at=utc_now(), path=path,
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=summary_extra_cost, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta(
                        "hit",
                        "semantic-match",
                        hit_type="semantic",
                        exact_enabled=can_cache,
                        semantic_enabled=can_semantic_cache,
                    )),
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(sem_resp) if LOG_BODIES else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(sem_resp, headers={"x-agentflow-cache": "semantic-hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with _tier_semaphores[_model_tier(crunched["model"])]:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                while True:
                    await _await_tier_backoff(crunched["model"])
                    await _throttle_forward()
                    try:
                        r = await client.post(ANTHROPIC_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched)
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
                        await _record_tier_backoff(crunched["model"], r.headers)
                        if retry_count == 1 and crunched.get("model") != resolved_requested_model:
                            _rate_limited_model = crunched.get("model")
                            crunched["model"] = resolved_requested_model
                            routing_meta["fallback_reason"] = "rate_limited"
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
            store.log_call(
                id=call_id, created_at=utc_now(), path=path,
                requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                input_tokens_est=input_tokens, output_tokens_est=None,
                actual_input_tokens=None, actual_output_tokens=None,
                cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                cache_json=stable_json(cache_meta),
                error=upstream_error_text(r.text, status_code),
                request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                session_id=session_id, category=category,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                retry_count=retry_count,
            )
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/plain"))

        if r.status_code < 400 and can_cache and response_body is not None:
            store.set_cache(
                key,
                str(crunched.get("model")),
                len(stable_json(crunched)),
                response_body,
                file_deps=cache_file_dependency_snapshots(crunched),
            )
        if can_semantic_cache and emb is not None and r.status_code < 400 and response_body is not None:
            store.set_semantic_cache(key, str(crunched.get("model")), emb, response_body, len(stable_json(crunched)))

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
        experiment_meta = routing_experiment_decision(crunched, routing_meta, stream=False)
        routing_meta["routing_experiment"] = experiment_meta
        if experiment_meta.get("sampled") and status_code < 400 and response_body is not None:
            await _run_anthropic_routing_experiment(
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
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=None if status_code < 400 else upstream_error_text(response_body, status_code),
            request_json=stable_json(crunched) if LOG_BODIES else None,
            response_json=stable_json(response_body) if LOG_BODIES else None,
            session_id=session_id, category=category,
            cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
            thinking_output_tokens=thinking_chars // TOKEN_CHARS if thinking_chars else None,
        )
        await _check_session_cost_alert(session_id)
        return JSONResponse(response_body, status_code=status_code, headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))})

    except TierBackoffActive as exc:
        routed_model_for_log: Optional[str] = None
        try:
            routed_model_for_log = str(crunched.get("model"))
        except Exception:
            routed_model_for_log = None
        error = exc.message
        status_code = 429
        response_body = _tier_backoff_payload(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=routed_model_for_log, stream=int(stream), cache_hit=0,
            status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None,
            response_json=stable_json(response_body) if LOG_BODIES else None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse(
            response_body,
            status_code=status_code,
            headers=_tier_backoff_headers(exc, routed_model_for_log or ""),
        )
    except Exception as exc:
        logging.exception("agentflow anthropic proxy error")
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse(public_proxy_error_body("anthropic", exc), status_code=500)


async def openai_optimized(request: Request, path: str) -> Response:
    return await provider_handlers.openai_optimized(_SERVER_CONTEXT, request, path)


async def _openai_optimized_impl(request: Request, path: str) -> Response:
    if PROVIDER != "openai":
        return provider_disabled_response("openai")

    started = time.time()
    call_id = str(uuid.uuid4())
    session_id = _openai_session_id(request)
    try:
        raw_body = await read_openai_json_body(request)
    except ClientJsonRequestError as exc:
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
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
        return await openai_passthrough(request, path)
    stream = bool(raw_body.get("stream"))
    requested_model = str(raw_body.get("model") or OPENAI_MODEL_LIST[0])
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
        headers = build_openai_forward_headers(request)

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
                        in_tok, out_tok, cached_tok, reason_tok = openai_usage_tokens(data)
                        if in_tok is not None:
                            actual_in = in_tok
                        if out_tok is not None:
                            actual_out = out_tok
                        cache_read_in = max(cache_read_in, cached_tok)
                        reasoning_tokens = max(reasoning_tokens, reason_tok)

                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        while True:
                            await _throttle_forward()
                            try:
                                async with client.stream("POST", OPENAI_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched) as r:
                                    status_code = r.status_code
                                    if status_code in (429, 529) and stream_retry_count < 3:
                                        stream_retry_count += 1
                                        delay = (2 ** (stream_retry_count - 1)) * (1.0 + random.random() * 0.5)
                                        print(f"openai_rate_limit: status={status_code} retry={stream_retry_count} delay={delay:.1f}s")
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
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path, provider="openai",
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_decision_meta("skipped", "streaming")),
                        error=error, request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=reasoning_tokens or None,
                    )
                    await _check_session_cost_alert(session_id)

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
            upstream=OPENAI_UPSTREAM,
        )
        emb: Optional[list[float]] = None
        if can_cache:
            cached = store.get_cache(key)
            if cached is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(openai_response_output_text(cached))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                store.log_call(
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
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(cached) if LOG_BODIES else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(cached, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        if can_semantic_cache:
            emb = build_embedding(extract_text(crunched))
            sem_resp = store.get_semantic_cache(emb, str(crunched.get("model")), SEMANTIC_CACHE_THRESHOLD)
            if sem_resp is not None:
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(openai_response_output_text(sem_resp))
                cost_baseline = estimate_cost(requested_model, input_tokens, out_tokens, provider="openai")
                store.log_call(
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
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(sem_resp) if LOG_BODIES else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(sem_resp, headers={"x-agentflow-cache": "semantic-hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            while True:
                await _throttle_forward()
                try:
                    r = await client.post(OPENAI_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched)
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
            store.log_call(
                id=call_id, created_at=utc_now(), path=path, provider="openai",
                requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                input_tokens_est=input_tokens, output_tokens_est=None,
                actual_input_tokens=None, actual_output_tokens=None,
                cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                cache_json=stable_json(cache_meta),
                error=upstream_error_text(r.text, status_code),
                request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                session_id=session_id, category=category,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                retry_count=retry_count,
            )
            return Response(
                r.content,
                status_code=status_code,
                media_type=_openai_passthrough_media_type(r.headers),
                headers=_openai_headers_for_client(r.headers),
            )

        if r.status_code < 400 and can_cache and response_body is not None:
            store.set_cache(
                key,
                str(crunched.get("model")),
                len(stable_json(crunched)),
                response_body,
                file_deps=cache_file_dependency_snapshots(crunched),
            )
        if can_semantic_cache and emb is not None and r.status_code < 400 and response_body is not None:
            store.set_semantic_cache(key, str(crunched.get("model")), emb, response_body, len(stable_json(crunched)))

        actual_in, actual_out, cache_read_in, reasoning_tokens = openai_usage_tokens(response_body)
        out_tokens = estimate_tokens_from_text(openai_response_output_text(response_body)) if response_body else 0
        cost_in = actual_in if actual_in is not None else input_tokens
        cost_out = actual_out if actual_out is not None else out_tokens
        cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai")
        cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=None if status_code < 400 else upstream_error_text(response_body, status_code),
            request_json=stable_json(crunched) if LOG_BODIES else None,
            response_json=stable_json(response_body) if LOG_BODIES else None,
            session_id=session_id, category=category,
            cache_creation_input_tokens=0, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
            thinking_output_tokens=reasoning_tokens or None,
        )
        await _check_session_cost_alert(session_id)
        return JSONResponse(
            response_body,
            status_code=status_code,
            headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))},
        )
    except Exception as exc:
        logging.exception("agentflow openai proxy error")
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_meta),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse(public_proxy_error_body("openai", exc), status_code=500)


async def openai_passthrough(request: Request, path: str) -> Response:
    return await provider_handlers.openai_passthrough(_SERVER_CONTEXT, request, path)


async def _openai_passthrough_impl(request: Request, path: str) -> Response:
    if PROVIDER != "openai":
        return provider_disabled_response("openai")
    headers = build_openai_forward_headers(request, force_json=False)
    content = await request.body()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.request(
            request.method,
            OPENAI_UPSTREAM.rstrip("/") + path,
            headers=headers,
            content=content if content else None,
            params=dict(request.query_params),
        )
    return Response(
        r.content,
        status_code=r.status_code,
        media_type=_openai_passthrough_media_type(r.headers),
        headers=_openai_headers_for_client(r.headers),
    )


@app.post("/v1/responses")
async def openai_responses(request: Request) -> Response:
    return await openai_optimized(request, "/v1/responses")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    return await openai_optimized(request, "/v1/chat/completions")


@app.websocket("/v1/responses")
async def openai_responses_websocket(websocket: WebSocket) -> None:
    await provider_handlers.openai_responses_websocket(_SERVER_CONTEXT, websocket)


async def _openai_responses_websocket_impl(websocket: WebSocket) -> None:
    await websocket.accept()
    if PROVIDER != "openai":
        await websocket.close(code=1008, reason="provider mismatch")
        return

    headers = build_openai_websocket_headers(websocket)
    upstream_url = openai_websocket_url("/v1/responses")
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
    except Exception as exc:
        logging.exception("agentflow openai websocket proxy error")
        try:
            await websocket.close(code=1011, reason=INTERNAL_PROXY_ERROR_MESSAGE)
        except Exception:
            pass


@app.api_route("/v1/responses/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_responses_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/responses/{rest}")


@app.api_route("/v1/files", methods=["GET", "POST", "DELETE"])
async def openai_files_root_passthrough(request: Request) -> Response:
    return await openai_passthrough(request, "/v1/files")


@app.api_route("/v1/files/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_files_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/files/{rest}")


@app.api_route("/v1/uploads", methods=["GET", "POST", "DELETE"])
async def openai_uploads_root_passthrough(request: Request) -> Response:
    return await openai_passthrough(request, "/v1/uploads")


@app.api_route("/v1/uploads/{rest:path}", methods=["GET", "POST", "DELETE"])
async def openai_uploads_passthrough(request: Request, rest: str) -> Response:
    return await openai_passthrough(request, f"/v1/uploads/{rest}")


@app.get("/agentflow/stats")
async def stats() -> dict[str, Any]:
    return await stats_views.stats(store, DEFAULT_DB)


@app.get("/agentflow/stats/limiter")
async def stats_limiter() -> dict[str, Any]:
    return await stats_views.stats_limiter(
        store,
        _tier_backoff_status,
        {
            "min_request_interval_ms": MIN_REQUEST_INTERVAL_MS,
            "max_tier_backoff_wait_s": MAX_TIER_BACKOFF_WAIT,
            "max_concurrent_per_tier": MAX_CONCURRENT_PER_TIER,
        },
    )


@app.get("/agentflow/stats/activity")
async def stats_activity(limit: int = 100) -> dict[str, Any]:
    return await stats_views.stats_activity(store, limit=limit)


@app.get("/agentflow/stats/full")
async def stats_full() -> dict[str, Any]:
    return await stats_views.stats_full(store)


@app.get("/agentflow/stats/weekly")
async def stats_weekly() -> dict[str, Any]:
    return await stats_views.stats_weekly(store)


@app.get("/agentflow/stats/sessions")
async def stats_sessions() -> dict[str, Any]:
    return await stats_views.stats_sessions(store)


@app.get("/agentflow/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return stats_views.dashboard_html()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow provider-specific local proxy")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default=PROVIDER)
    parser.add_argument("--anthropic-upstream", default=ANTHROPIC_UPSTREAM)
    parser.add_argument("--openai-upstream", default=OPENAI_UPSTREAM)
    parser.add_argument("--openai-auth-mode", choices=("client", "proxy"), default=OPENAI_AUTH_MODE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["AGENTFLOW_PROVIDER"] = args.provider
    os.environ["AGENTFLOW_ANTHROPIC_UPSTREAM"] = args.anthropic_upstream
    os.environ["AGENTFLOW_OPENAI_UPSTREAM"] = args.openai_upstream
    os.environ["AGENTFLOW_OPENAI_AUTH_MODE"] = args.openai_auth_mode
    configure_provider(args.provider, args.anthropic_upstream, args.openai_upstream, args.openai_auth_mode)
    import uvicorn
    if args.reload:
        uvicorn.run("agentflow_proxy.server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
