from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional runtime dependency
    zstd = None

load_dotenv()

PROVIDER = os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()
ANTHROPIC_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
OPENAI_UPSTREAM = os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com")
DEFAULT_UPSTREAM = ANTHROPIC_UPSTREAM if PROVIDER == "anthropic" else OPENAI_UPSTREAM
DEFAULT_DB = os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
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

_forward_lock = asyncio.Lock()
_last_forward_time: float = 0.0
_tier_backoff_until: dict[str, float] = {}
_tier_backoff_update_lock = asyncio.Lock()
# Per-tier semaphores cap concurrent in-flight forwarded requests. Requests queue rather than racing.
_sem_value = MAX_CONCURRENT_PER_TIER if MAX_CONCURRENT_PER_TIER > 0 else 9999
_tier_semaphores: dict[str, asyncio.Semaphore] = {
    "haiku": asyncio.Semaphore(_sem_value),
    "sonnet": asyncio.Semaphore(_sem_value),
    "opus": asyncio.Semaphore(_sem_value),
}


@dataclass(frozen=True)
class TierBackoffActive(Exception):
    tier: str
    remaining: float

    @property
    def retry_after(self) -> int:
        return max(1, int(self.remaining + 0.999))

    @property
    def message(self) -> str:
        return f"temporarily limiting requests for {self.tier} tier; retry after {self.retry_after}s"


def _model_tier(model: str) -> str:
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"


async def _await_tier_backoff(model: str) -> None:
    tier = _model_tier(model)
    remaining = _tier_backoff_until.get(tier, 0.0) - time.time()
    if remaining <= 0:
        return
    if remaining > MAX_TIER_BACKOFF_WAIT:
        print(
            f"tier_backoff: tier={tier} remaining={remaining:.1f}s "
            f"exceeds_max_wait={MAX_TIER_BACKOFF_WAIT:.1f}s"
        )
        raise TierBackoffActive(tier=tier, remaining=remaining)
    print(f"tier_backoff: tier={tier} waiting={remaining:.1f}s")
    await asyncio.sleep(remaining)


def _tier_backoff_payload(exc: TierBackoffActive) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": exc.message,
        },
    }


def _tier_backoff_headers(exc: TierBackoffActive, model: str) -> dict[str, str]:
    return {
        "retry-after": str(exc.retry_after),
        "x-agentflow-routed-model": model,
    }


async def _record_tier_backoff(model: str, response_headers: Any, default_seconds: float = 60.0) -> None:
    tier = _model_tier(model)
    raw = response_headers.get("retry-after")
    try:
        delay = float(raw) if raw else default_seconds
    except (ValueError, TypeError):
        delay = default_seconds
    new_until = time.time() + delay
    async with _tier_backoff_update_lock:
        if new_until > _tier_backoff_until.get(tier, 0.0):
            _tier_backoff_until[tier] = new_until


async def _throttle_forward() -> None:
    global _last_forward_time
    if MIN_REQUEST_INTERVAL_MS <= 0:
        return
    async with _forward_lock:
        now = time.time()
        elapsed_ms = (now - _last_forward_time) * 1000
        if elapsed_ms < MIN_REQUEST_INTERVAL_MS:
            await asyncio.sleep((MIN_REQUEST_INTERVAL_MS - elapsed_ms) / 1000)
        _last_forward_time = time.time()


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


from agentflow_proxy.store import Store, utc_now, stable_json
from agentflow_proxy.pricing import MODEL_PRICES, MODEL_ALIASES, estimate_cost
from agentflow_proxy.router import (
    extract_text, has_tools, categorize_request, route_model,
    HAIKU_DEFAULT, SONNET_DEFAULT, OPUS_DEFAULT,
    route_openai_model,
    STRIP_THINKING_HISTORY, _has_top_level_thinking, strip_thinking_history_blocks,
)
from agentflow_proxy.crunch import (
    TOKEN_CHARS, sha256_text, estimate_tokens_from_text, build_embedding,
    crunch_body, inject_prompt_cache, has_cache_control_blocks,
)
from agentflow_proxy.cache import (
    CACHE_ENABLED, CACHE_TOOL_CALLS, SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD,
    cache_decision_meta, cache_key_for, response_output_text,
)


store = Store(DEFAULT_DB)
app = FastAPI(title=f"AgentFlow {PROVIDER.title()} Proxy", version="0.1.0")


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
    # Pass auth through. This server does not require/store credentials.
    allowed = {
        "authorization", "x-api-key", "anthropic-version", "anthropic-beta",
        "content-type", "accept", "user-agent",
    }
    headers: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in allowed:
            headers[k] = v
    headers.setdefault("anthropic-version", os.getenv("ANTHROPIC_VERSION", "2023-06-01"))
    headers["content-type"] = "application/json"
    return headers


def build_openai_forward_headers(request: Request, *, force_json: bool = True) -> dict[str, str]:
    allowed = {
        "authorization", "openai-organization", "openai-project",
        "content-type", "content-encoding", "accept", "user-agent",
    }
    headers: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in allowed and not (lk == "authorization" and OPENAI_AUTH_MODE == "proxy"):
            headers[k] = v
    if OPENAI_AUTH_MODE == "proxy" or "authorization" not in {k.lower() for k in headers}:
        api_key = os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    if force_json:
        headers.pop("content-encoding", None)
        headers.pop("Content-Encoding", None)
        headers["content-type"] = "application/json"
    return headers


async def read_openai_json_body(request: Request) -> Optional[dict[str, Any]]:
    body = await request.body()
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    try:
        if encoding in {"zstd", "zstandard"}:
            if zstd is None:
                return None
            body = zstd.ZstdDecompressor().decompress(body)
        elif encoding == "gzip":
            body = gzip.decompress(body)
        elif encoding in {"deflate", "zlib"}:
            body = zlib.decompress(body)
        elif encoding:
            return None
        parsed = json.loads(body)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_openai_websocket_headers(websocket: WebSocket) -> dict[str, str]:
    allowed = {"authorization", "openai-organization", "openai-project", "accept", "user-agent"}
    headers: dict[str, str] = {}
    for k, v in websocket.headers.items():
        lk = k.lower()
        if lk in allowed and not (lk == "authorization" and OPENAI_AUTH_MODE == "proxy"):
            headers[k] = v
    if OPENAI_AUTH_MODE == "proxy" or "authorization" not in {k.lower() for k in headers}:
        api_key = os.getenv("AGENTFLOW_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    return headers


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
    if PROVIDER != "anthropic":
        return provider_disabled_response("anthropic")
    started = time.time()
    call_id = str(uuid.uuid4())
    path = "/v1/messages"
    client_ip = (request.client.host if request.client else "unknown")
    session_id = request.headers.get("x-session-id") or hashlib.sha256(
        (client_ip + datetime.now(timezone.utc).strftime("%Y-%m-%d")).encode()
    ).hexdigest()[:16]
    raw_body = await request.json()
    stream = bool(raw_body.get("stream"))
    requested_model = str(raw_body.get("model") or "")
    if requested_model in MODEL_ALIASES:
        raw_body["model"] = MODEL_ALIASES[requested_model]
    error: Optional[str] = None
    status_code = 200
    crunch_meta: dict[str, Any] = {}
    routing_meta: dict[str, Any] = {}
    cache_hit = False
    response_body: Optional[dict[str, Any]] = None
    retry_count = 0
    net_retries = 0

    category = categorize_request(raw_body)

    try:
        crunched, crunch_meta = crunch_body(raw_body)
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
        if routed_model != resolved_requested_model:
            _incompatible = [k for k in ("effort", "thinking", "budget_tokens", "interleaved_thinking") if k in crunched]
            for k in _incompatible:
                del crunched[k]
            # Also strip effort nested inside a thinking dict (e.g. {"type": "disabled", "effort": "high"})
            _thinking_block = crunched.get("thinking")
            if isinstance(_thinking_block, dict) and "effort" in _thinking_block:
                del crunched["thinking"]["effort"]
                if "thinking.effort" not in _incompatible:
                    _incompatible.append("thinking.effort")
            if _incompatible:
                routing_meta["stripped_params"] = _incompatible
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
        headers = build_forward_headers(request)
        if prompt_cached or has_cache_control_blocks(crunched):
            existing = headers.get("anthropic-beta", "")
            if "prompt-caching" not in existing:
                headers["anthropic-beta"] = (existing + ",prompt-caching-2024-07-31" if existing else "prompt-caching-2024-07-31")

        # Streaming is passed through and not cached, but still logged when the stream finishes.
        if stream:
            async def gen() -> AsyncIterator[bytes]:
                nonlocal status_code, error
                actual_in: Optional[int] = None
                actual_out: Optional[int] = None
                cache_creation_in: int = 0
                cache_read_in: int = 0
                thinking_chars: int = 0
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
                                            if stream_retry_count == 1 and routed_model != resolved_requested_model:
                                                crunched["model"] = resolved_requested_model
                                                routing_meta["fallback_reason"] = "rate_limited"
                                                print(f"rate_limit_fallback: routing {routed_model!r} -> {resolved_requested_model!r}")
                                            await asyncio.sleep(delay)
                                            continue
                                        async for chunk in r.aiter_bytes():
                                            sse_frame_buf += chunk
                                            while b"\n\n" in sse_frame_buf:
                                                frame, sse_frame_buf = sse_frame_buf.split(b"\n\n", 1)
                                                event_bytes = frame + b"\n\n"
                                                yield event_bytes
                                                parse_sse_usage(frame)
                                        if sse_frame_buf:
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
                    error = repr(exc)
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n".encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    cost_in = actual_in if actual_in is not None else input_tokens
                    cost_out = actual_out if actual_out is not None else 0
                    cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_creation_in, cache_read_in)
                    cost_baseline = estimate_cost(requested_model, cost_in + cache_creation_in + cache_read_in, cost_out)
                    if cache_creation_in or cache_read_in:
                        print(f"prompt_cache: creation={cache_creation_in} read={cache_read_in}")
                    if status_code >= 400 and error is None:
                        error = f"upstream_error: status={status_code}"
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path,
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_decision_meta("skip-streaming")),
                        error=error, request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                        thinking_output_tokens=thinking_chars // TOKEN_CHARS if thinking_chars else None,
                    )
                    await _check_session_cost_alert(session_id)

            return StreamingResponse(gen(), media_type="text/event-stream")

        can_cache = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tools(crunched))
        key = cache_key_for(crunched, path)
        can_semantic_cache = SEMANTIC_CACHE_ENABLED and not has_tools(crunched)
        _cache_miss_type = "miss" if can_cache or can_semantic_cache else ("skip-tools" if CACHE_ENABLED else "skip-disabled")
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
                    cost_est_usd=0.0, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta("exact")),
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
                    cost_est_usd=0.0, cost_baseline_usd=cost_baseline,
                    crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    cache_json=stable_json(cache_decision_meta("semantic")),
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
                        if retry_count == 1 and routed_model != resolved_requested_model:
                            crunched["model"] = resolved_requested_model
                            routing_meta["fallback_reason"] = "rate_limited"
                            print(f"rate_limit_fallback: routing {routed_model!r} -> {resolved_requested_model!r}")
                        await asyncio.sleep(delay)
                        continue
                    break
        status_code = r.status_code
        try:
            response_body = r.json()
        except Exception:
            latency_ms = int((time.time() - started) * 1000)
            cost = estimate_cost(str(crunched.get("model")), input_tokens, 0)
            cost_baseline = estimate_cost(requested_model, input_tokens, 0)
            store.log_call(
                id=call_id, created_at=utc_now(), path=path,
                requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                input_tokens_est=input_tokens, output_tokens_est=None,
                actual_input_tokens=None, actual_output_tokens=None,
                cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                cache_json=stable_json(cache_decision_meta(_cache_miss_type)),
                error=r.text[:1000],
                request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                session_id=session_id, category=category,
                cache_creation_input_tokens=0, cache_read_input_tokens=0,
                retry_count=retry_count,
            )
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/plain"))

        if r.status_code < 400 and can_cache and response_body is not None:
            store.set_cache(key, str(crunched.get("model")), len(stable_json(crunched)), response_body)
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
        cost_baseline = estimate_cost(requested_model, cost_in + cache_creation_in + cache_read_in, cost_out)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_decision_meta(_cache_miss_type)),
            error=None if status_code < 400 else stable_json(response_body)[:1000],
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
            cache_json=stable_json(cache_decision_meta("miss")),
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
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_decision_meta("miss")),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse({"type": "error", "error": {"type": "agentflow_proxy_error", "message": error}}, status_code=500)


async def openai_optimized(request: Request, path: str) -> Response:
    if PROVIDER != "openai":
        return provider_disabled_response("openai")

    started = time.time()
    call_id = str(uuid.uuid4())
    session_id = _openai_session_id(request)
    raw_body = await read_openai_json_body(request)
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
    retry_count = 0
    net_retries = 0

    try:
        crunched, crunch_meta = crunch_body(raw_body)
        routed_model, routing_meta = route_openai_model(crunched)
        resolved_requested_model = str(crunched.get("model") or requested_model)
        crunched["model"] = routed_model
        input_tokens = estimate_tokens_from_text(extract_text(crunched))
        headers = build_openai_forward_headers(request)

        if stream:
            async def gen() -> AsyncIterator[bytes]:
                nonlocal status_code, error
                actual_in: Optional[int] = None
                actual_out: Optional[int] = None
                cache_read_in = 0
                reasoning_tokens = 0
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
                                        if stream_retry_count == 1 and routed_model != resolved_requested_model:
                                            crunched["model"] = resolved_requested_model
                                            routing_meta["fallback_reason"] = "rate_limited"
                                            print(f"openai_rate_limit_fallback: routing {routed_model!r} -> {resolved_requested_model!r}")
                                        await asyncio.sleep(delay)
                                        continue
                                    async for chunk in r.aiter_bytes():
                                        sse_frame_buf += chunk
                                        while b"\n\n" in sse_frame_buf:
                                            frame, sse_frame_buf = sse_frame_buf.split(b"\n\n", 1)
                                            event_bytes = frame + b"\n\n"
                                            yield event_bytes
                                            parse_sse_usage(frame)
                                    if sse_frame_buf:
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
                    error = repr(exc)
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n".encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    cost_in = actual_in if actual_in is not None else input_tokens
                    cost_out = actual_out if actual_out is not None else 0
                    cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out, cache_read=cache_read_in, provider="openai")
                    cost_baseline = estimate_cost(requested_model, cost_in, cost_out, cache_read=cache_read_in, provider="openai")
                    if status_code >= 400 and error is None:
                        error = f"upstream_error: status={status_code}"
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path, provider="openai",
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        cache_json=stable_json(cache_decision_meta("skip-streaming")),
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

        can_cache = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tools(crunched))
        key = cache_key_for(crunched, path)
        can_semantic_cache = SEMANTIC_CACHE_ENABLED and not has_tools(crunched)
        _cache_miss_type = "miss" if can_cache or can_semantic_cache else ("skip-tools" if CACHE_ENABLED else "skip-disabled")
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
                    cache_json=stable_json(cache_decision_meta("exact")),
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
                    cache_json=stable_json(cache_decision_meta("semantic")),
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
                    if retry_count == 1 and routed_model != resolved_requested_model:
                        crunched["model"] = resolved_requested_model
                        routing_meta["fallback_reason"] = "rate_limited"
                        print(f"openai_rate_limit_fallback: routing {routed_model!r} -> {resolved_requested_model!r}")
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
                cache_json=stable_json(cache_decision_meta(_cache_miss_type)),
                error=r.text[:1000],
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
            store.set_cache(key, str(crunched.get("model")), len(stable_json(crunched)), response_body)
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
            cache_json=stable_json(cache_decision_meta(_cache_miss_type)),
            error=None if status_code < 400 else stable_json(response_body)[:1000],
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
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path, provider="openai",
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            cache_json=stable_json(cache_decision_meta("miss")),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse({"error": {"type": "agentflow_proxy_error", "message": error}}, status_code=500)


async def openai_passthrough(request: Request, path: str) -> Response:
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
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
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
    conn = store.conn
    calls = conn.execute("select count(*) c from calls").fetchone()["c"]
    cache_hits = conn.execute("select count(*) c from calls where cache_hit = 1").fetchone()["c"]
    routed = conn.execute("select coalesce(provider, 'anthropic') as provider, requested_model, routed_model, count(*) c from calls group by coalesce(provider, 'anthropic'), requested_model, routed_model order by c desc limit 20").fetchall()
    recent = conn.execute("select coalesce(provider, 'anthropic') as provider, created_at, requested_model, routed_model, cache_hit, status_code, latency_ms, cost_est_usd from calls order by created_at desc limit 20").fetchall()
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": (cache_hits / calls) if calls else 0,
        "db": DEFAULT_DB,
        "routing": [dict(r) for r in routed],
        "recent": [dict(r) for r in recent],
    }


@app.get("/agentflow/stats/full")
async def stats_full() -> dict[str, Any]:
    conn = store.conn

    def q(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def s(sql: str, params: tuple = ()) -> Any:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

    total_calls = s("select count(*) from calls") or 0
    today_calls = s("select count(*) from calls where date(created_at) = date('now')") or 0
    total_cost = s("select sum(cost_est_usd) from calls") or 0.0
    today_cost = s("select sum(cost_est_usd) from calls where date(created_at) = date('now')") or 0.0
    cache_hits = s("select count(*) from calls where cache_hit = 1") or 0
    cache_cost_saved = s("select count(*) * 0.003 from calls where cache_hit = 1") or 0.0  # rough avg
    avg_latency = s("select avg(latency_ms) from calls where latency_ms is not null") or 0
    routed_count = s("select count(*) from calls where requested_model != routed_model and routed_model is not null") or 0
    crunched_count = s("select count(*) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    errors = s("select count(*) from calls where status_code >= 400") or 0

    # Estimate routing savings: calls where model was downgraded, cost diff
    routing_savings = 0.0
    today_routing_savings = 0.0
    downgraded = q("""
        select coalesce(provider, 'anthropic') as provider, requested_model, routed_model,
               coalesce(actual_input_tokens, input_tokens_est, 0) as in_tok,
               coalesce(actual_output_tokens, output_tokens_est, 0) as out_tok,
               (date(created_at) = date('now')) as is_today
        from calls where requested_model != routed_model and routed_model is not null
    """)
    for row in downgraded:
        req_cost = estimate_cost(row["requested_model"], row["in_tok"], row["out_tok"], provider=row["provider"]) or 0
        act_cost = estimate_cost(row["routed_model"], row["in_tok"], row["out_tok"], provider=row["provider"]) or 0
        delta = max(0.0, req_cost - act_cost)
        routing_savings += delta
        if row["is_today"]:
            today_routing_savings += delta

    today_cache_savings = s("select count(*) * 0.003 from calls where cache_hit = 1 and date(created_at) = date('now')") or 0.0

    crunch_chars_saved = s("select sum(json_extract(crunch_json, '$.saved_chars')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    crunch_tokens_saved = s("select sum(json_extract(crunch_json, '$.tokens_saved_est')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    avg_crunch_ratio = s("select avg(json_extract(crunch_json, '$.crunch_ratio')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    prompt_cache_creation_tokens = s("select sum(cache_creation_input_tokens) from calls") or 0
    prompt_cache_read_tokens = s("select sum(cache_read_input_tokens) from calls") or 0
    prompt_cache_hits = s("select count(*) from calls where cache_read_input_tokens > 0") or 0
    prompt_cache_hit_rate = round(prompt_cache_hits / total_calls, 4) if total_calls else 0

    prompt_cache_savings = 0.0
    today_prompt_cache_savings = 0.0
    cache_read_by_model = q("""
        select coalesce(routed_model, requested_model) as model,
               sum(cache_read_input_tokens) as read_tok,
               sum(case when date(created_at) = date('now') then cache_read_input_tokens else 0 end) as today_read_tok,
               coalesce(provider, 'anthropic') as provider
        from calls where cache_read_input_tokens > 0
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in cache_read_by_model:
        full_cost = estimate_cost(row["model"], row["read_tok"], 0, provider=row["provider"]) or 0
        prompt_cache_savings += 0.90 * full_cost
        today_full_cost = estimate_cost(row["model"], row["today_read_tok"] or 0, 0, provider=row["provider"]) or 0
        today_prompt_cache_savings += 0.90 * today_full_cost

    thinking_output_tokens = int(s("select sum(thinking_output_tokens) from calls") or 0)
    today_thinking_output_tokens = int(s("select sum(thinking_output_tokens) from calls where date(created_at) = date('now')") or 0)
    thinking_cost = 0.0
    today_thinking_cost = 0.0
    thinking_by_model = q("""
        select coalesce(routed_model, requested_model) as model,
               sum(thinking_output_tokens) as think_tok,
               sum(case when date(created_at) = date('now') then coalesce(thinking_output_tokens, 0) else 0 end) as today_think_tok,
               coalesce(provider, 'anthropic') as provider
        from calls where thinking_output_tokens > 0
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in thinking_by_model:
        thinking_cost += estimate_cost(row["model"], 0, row["think_tok"] or 0, provider=row["provider"]) or 0
        today_thinking_cost += estimate_cost(row["model"], 0, row["today_think_tok"] or 0, provider=row["provider"]) or 0

    recent = q("""
        select id, coalesce(provider, 'anthropic') as provider, created_at, requested_model, routed_model, stream, cache_hit,
               status_code, latency_ms,
               coalesce(actual_input_tokens, input_tokens_est) as input_tokens,
               coalesce(actual_output_tokens, output_tokens_est) as output_tokens,
               cost_est_usd,
               json_extract(crunch_json, '$.changed') as crunched,
               json_extract(crunch_json, '$.saved_chars') as crunch_saved_chars,
               json_extract(routing_json, '$.reason') as routing_reason,
               error
        from calls order by created_at desc limit 50
    """)

    routing_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, requested_model, routed_model, count(*) as count
        from calls group by coalesce(provider, 'anthropic'), requested_model, routed_model order by count desc limit 15
    """)

    category_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, coalesce(category, 'unknown') as category, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(provider, 'anthropic'), coalesce(category, 'unknown') order by count desc
    """)

    provider_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(provider, 'anthropic') order by count desc
    """)

    return {
        "summary": {
            "total_calls": total_calls,
            "today_calls": today_calls,
            "total_cost_usd": round(total_cost, 6),
            "today_cost_usd": round(today_cost, 6),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_calls, 4) if total_calls else 0,
            "routing_savings_usd": round(routing_savings, 6),
            "today_routing_savings_usd": round(today_routing_savings, 6),
            "cache_savings_usd": round(cache_cost_saved, 6),
            "today_cache_savings_usd": round(today_cache_savings, 6),
            "total_savings_usd": round(routing_savings + cache_cost_saved, 6),
            "avg_latency_ms": round(avg_latency),
            "routed_count": routed_count,
            "crunched_count": crunched_count,
            "crunch_chars_saved": crunch_chars_saved,
            "crunch_tokens_saved": int(crunch_tokens_saved),
            "avg_crunch_ratio": round(avg_crunch_ratio, 4),
            "errors": errors,
            "prompt_cache_creation_tokens": int(prompt_cache_creation_tokens),
            "prompt_cache_read_tokens": int(prompt_cache_read_tokens),
            "prompt_cache_hit_rate": prompt_cache_hit_rate,
            "prompt_cache_savings_usd": round(prompt_cache_savings, 6),
            "today_prompt_cache_savings_usd": round(today_prompt_cache_savings, 6),
            "thinking_output_tokens": thinking_output_tokens,
            "today_thinking_output_tokens": today_thinking_output_tokens,
            "thinking_cost_usd": round(thinking_cost, 6),
            "today_thinking_cost_usd": round(today_thinking_cost, 6),
        },
        "recent": recent,
        "routing_breakdown": routing_breakdown,
        "category_breakdown": category_breakdown,
        "provider_breakdown": provider_breakdown,
    }


@app.get("/agentflow/stats/weekly")
async def stats_weekly() -> dict[str, Any]:
    conn = store.conn
    rows = conn.execute("""
        select
            date(created_at) as day,
            count(*) as total_calls,
            sum(case when status_code = 200 then 1 else 0 end) as successful_calls,
            sum(case when status_code >= 400 then 1 else 0 end) as errors,
            sum(cache_hit) as cache_hits,
            round(avg(latency_ms)) as avg_latency_ms,
            round(sum(coalesce(cost_est_usd, 0)), 6) as cost_est_usd,
            round(sum(coalesce(cost_baseline_usd, 0)), 6) as cost_baseline_usd
        from calls
        where date(created_at) >= date('now', '-6 days')
        group by date(created_at)
        order by day asc
    """).fetchall()
    days = []
    for r in rows:
        row = dict(r)
        row["savings_usd"] = round((row["cost_baseline_usd"] or 0) - (row["cost_est_usd"] or 0), 6)
        days.append(row)
    totals = {
        "day": "Total",
        "total_calls": sum(r["total_calls"] for r in days),
        "successful_calls": sum(r["successful_calls"] or 0 for r in days),
        "errors": sum(r["errors"] or 0 for r in days),
        "cache_hits": sum(r["cache_hits"] or 0 for r in days),
        "avg_latency_ms": round(sum(r["avg_latency_ms"] or 0 for r in days) / len(days)) if days else None,
        "cost_est_usd": round(sum(r["cost_est_usd"] or 0 for r in days), 6),
        "cost_baseline_usd": round(sum(r["cost_baseline_usd"] or 0 for r in days), 6),
        "savings_usd": round(sum(r["savings_usd"] for r in days), 6),
    }
    return {"days": days, "totals": totals}


@app.get("/agentflow/stats/sessions")
async def stats_sessions() -> dict[str, Any]:
    conn = store.conn
    rows = conn.execute("""
        SELECT SUBSTR(session_id,1,8) as sid, session_id, COUNT(*) as calls,
            ROUND(SUM(cost_est_usd),6) as cost_usd,
            SUM(CASE WHEN category='tool-result' THEN 1 ELSE 0 END) as tool_result,
            SUM(CASE WHEN category='tool-heavy' THEN 1 ELSE 0 END) as tool_heavy,
            SUM(CASE WHEN category='short-completion' THEN 1 ELSE 0 END) as short_completion,
            SUM(CASE WHEN category='code-gen' THEN 1 ELSE 0 END) as code_gen,
            SUM(CASE WHEN category='chat' THEN 1 ELSE 0 END) as chat,
            SUM(CASE WHEN category IS NULL OR category NOT IN ('tool-result','tool-heavy','short-completion','code-gen','chat') THEN 1 ELSE 0 END) as other
        FROM calls
        WHERE DATE(created_at) = DATE('now') AND session_id IS NOT NULL
        GROUP BY session_id ORDER BY cost_usd DESC LIMIT 20
    """).fetchall()
    return {"sessions": [dict(r) for r in rows]}


@app.get("/agentflow/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentFlow</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:ui-monospace,monospace;background:#0d1117;color:#c9d1d9;font-size:13px}
  a{color:#58a6ff;text-decoration:none}
  header{background:#161b22;border-bottom:1px solid #30363d;padding:14px 24px;display:flex;align-items:center;gap:16px}
  header h1{font-size:16px;font-weight:600;color:#f0f6fc}
  header .sub{color:#8b949e;font-size:12px}
  .dot{width:8px;height:8px;border-radius:50%;background:#3fb950;display:inline-block;margin-right:6px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .cards{display:flex;gap:12px;padding:16px 24px;flex-wrap:wrap}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;min-width:150px;flex:1}
  .card .label{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  .card .value{font-size:22px;font-weight:600;color:#f0f6fc}
  .card .sub{color:#8b949e;font-size:11px;margin-top:3px}
  .card.green .value{color:#3fb950}
  .card.yellow .value{color:#d29922}
  .card.blue .value{color:#58a6ff}
  .tabs{display:flex;padding:0 24px;border-bottom:1px solid #30363d}
  .tab-btn{background:none;border:none;border-bottom:2px solid transparent;color:#8b949e;cursor:pointer;font-family:inherit;font-size:13px;margin-bottom:-1px;padding:10px 16px}
  .tab-btn.active{border-bottom-color:#58a6ff;color:#f0f6fc}
  .tab-panel{display:none}
  .tab-panel.active{display:block}
  .section{padding:0 24px 24px}
  .section h2{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#8b949e;margin-bottom:10px;padding-top:4px}
  table{width:100%;border-collapse:collapse}
  th{text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:6px 10px;border-bottom:1px solid #21262d;font-weight:400}
  td{padding:6px 10px;border-bottom:1px solid #161b22;vertical-align:middle;white-space:nowrap}
  tr:hover td{background:#161b22}
  .badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:500}
  .badge.hit{background:#1a3a1f;color:#3fb950}
  .badge.miss{background:#1c1c1c;color:#8b949e}
  .badge.stream{background:#1a2a3a;color:#58a6ff}
  .badge.err{background:#3a1a1a;color:#f85149}
  .badge.routed{background:#2d2208;color:#d29922}
  .badge.crunched{background:#1a1a3a;color:#79c0ff}
  .badge.provider{background:#20242b;color:#c9d1d9}
  .model{max-width:160px;overflow:hidden;text-overflow:ellipsis;color:#c9d1d9}
  .model.downgraded{color:#d29922}
  .cost{color:#3fb950;font-variant-numeric:tabular-nums}
  .latency{color:#8b949e;font-variant-numeric:tabular-nums}
  .tokens{color:#8b949e;font-variant-numeric:tabular-nums}
  .ts{color:#8b949e;font-size:11px}
  .err-row td{background:#1a0a0a}
  .totals-row td{border-top:1px solid #30363d;font-weight:600}
  .savings{color:#3fb950;font-variant-numeric:tabular-nums}
  .baseline{color:#8b949e;font-variant-numeric:tabular-nums}
  #status{margin-left:auto;font-size:11px;color:#8b949e}
  .arrow{color:#8b949e;margin:0 3px}
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <h1>AgentFlow</h1>
  <span class="sub">provider-aware proxy · cost reduction dashboard</span>
  <span id="status">loading...</span>
</header>

<div class="cards" id="cards">
  <div class="card"><div class="label">Calls today</div><div class="value" id="c-today">—</div><div class="sub" id="c-total">— total</div></div>
  <div class="card"><div class="label">Cost today</div><div class="value" id="c-cost">—</div><div class="sub" id="c-cost-total">— total</div></div>
  <div class="card green"><div class="label">Saved by routing</div><div class="value" id="c-routing">—</div><div class="sub" id="c-routed-n">— calls routed</div></div>
  <div class="card green"><div class="label">Saved by cache</div><div class="value" id="c-cache-saved">—</div><div class="sub" id="c-cache-rate">— hit rate</div></div>
  <div class="card green"><div class="label">Provider cache discount</div><div class="value" id="c-prompt-cache-saved">—</div><div class="sub" id="c-prompt-cache-rate">— provider cache hit rate</div></div>
  <div class="card blue"><div class="label">Avg latency</div><div class="value" id="c-latency">—</div><div class="sub" id="c-crunched">— crunched</div></div>
  <div class="card yellow"><div class="label">Thinking cost today</div><div class="value" id="c-thinking-cost">—</div><div class="sub" id="c-thinking-tok">— thinking tokens</div></div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('recent')">Recent calls</button>
  <button class="tab-btn" onclick="showTab('weekly')">7-day stats</button>
  <button class="tab-btn" onclick="showTab('categories')">By category</button>
  <button class="tab-btn" onclick="showTab('sessions')">Sessions</button>
</div>

<div class="tab-panel active" id="tab-recent">
<div class="section">
  <h2>Recent calls</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Provider</th><th>Requested</th><th>Used</th><th>Tokens in/out</th><th>Cost</th><th>Latency</th><th>Flags</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-weekly">
<div class="section">
  <h2>7-day daily statistics</h2>
  <table>
    <thead><tr>
      <th>Date</th><th>Calls</th><th>Success</th><th>Errors</th><th>Cache hits</th><th>Avg latency</th><th>Cost (actual)</th><th>Cost (baseline)</th><th>Savings</th>
    </tr></thead>
    <tbody id="weekly-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-categories">
<div class="section">
  <h2>Calls by request category</h2>
  <table>
    <thead><tr>
      <th>Category</th><th>Calls</th><th>Cost</th><th>Routed</th>
    </tr></thead>
    <tbody id="cat-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-sessions">
<div class="section">
  <h2>Sessions today</h2>
  <table>
    <thead><tr>
      <th>Session</th><th>Calls</th><th>Cost</th><th>tool-result</th><th>tool-heavy</th><th>short-comp</th><th>code-gen</th><th>chat</th><th>other</th>
    </tr></thead>
    <tbody id="sess-tbody"></tbody>
  </table>
</div>
</div>

<script>
function fmt(n,d=4){if(n==null)return'—';return'$'+n.toFixed(d)}
function fmtMs(n){if(n==null)return'—';return n<1000?n+'ms':(n/1000).toFixed(1)+'s'}
function fmtTok(n){if(n==null)return'?';return n>=1000?(n/1000).toFixed(1)+'k':String(n)}
function ago(ts){
  if(!ts)return'—';
  const d=Math.floor((Date.now()-new Date(ts).getTime())/1000);
  if(isNaN(d))return'—';
  if(d<60)return d+'s';if(d<3600)return Math.floor(d/60)+'m';
  if(d<86400)return Math.floor(d/3600)+'h';return Math.floor(d/86400)+'d';
}
function shortModel(m){
  if(!m)return'—';
  return m.replace('claude-','').replace(/-20\\d{6}$/,'');
}
function shortProvider(p){
  if(!p)return'—';
  return p==='anthropic'?'Claude':p.charAt(0).toUpperCase()+p.slice(1);
}

function showTab(name){
  ['recent','weekly','categories','sessions'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===name);
  });
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active',['recent','weekly','categories','sessions'][i]===name);
  });
}

async function refreshWeekly(){
  try{
    const r=await fetch('/agentflow/stats/weekly');
    const d=await r.json();
    const tb=document.getElementById('weekly-tbody');
    const rows=[...d.days,{...d.totals,_total:true}];
    tb.innerHTML=rows.map(row=>{
      const cls=row._total?' class="totals-row"':'';
      const errColor=row.errors?'color:#f85149':'color:#8b949e';
      return `<tr${cls}>
        <td class="ts">${row.day}</td>
        <td>${(row.total_calls??0).toLocaleString()}</td>
        <td style="color:#3fb950">${(row.successful_calls??0).toLocaleString()}</td>
        <td style="${errColor}">${(row.errors??0).toLocaleString()}</td>
        <td>${(row.cache_hits??0).toLocaleString()}</td>
        <td class="latency">${fmtMs(row.avg_latency_ms)}</td>
        <td class="cost">${fmt(row.cost_est_usd,5)}</td>
        <td class="baseline">${fmt(row.cost_baseline_usd,5)}</td>
        <td class="savings">${fmt(row.savings_usd,5)}</td>
      </tr>`;
    }).join('');
  }catch(e){}
}

async function refresh(){
  try{
    const r=await fetch('/agentflow/stats/full');
    const d=await r.json();
    const s=d.summary;

    document.getElementById('c-today').textContent=s.today_calls.toLocaleString();
    document.getElementById('c-total').textContent=s.total_calls.toLocaleString()+' total';
    document.getElementById('c-cost').textContent=fmt(s.today_cost_usd,4);
    document.getElementById('c-cost-total').textContent=fmt(s.total_cost_usd,4)+' total';
    document.getElementById('c-routing').textContent=fmt(s.today_routing_savings_usd,4);
    document.getElementById('c-routed-n').textContent=s.routed_count+' calls routed';
    document.getElementById('c-cache-saved').textContent=fmt(s.today_cache_savings_usd,4);
    document.getElementById('c-cache-rate').textContent=Math.round(s.cache_hit_rate*100)+'% hit rate';
    document.getElementById('c-prompt-cache-saved').textContent=fmt(s.today_prompt_cache_savings_usd,4);
    document.getElementById('c-prompt-cache-rate').textContent=Math.round((s.prompt_cache_hit_rate||0)*100)+'% provider cache hit rate';
    document.getElementById('c-latency').textContent=fmtMs(s.avg_latency_ms);
    document.getElementById('c-crunched').textContent=s.crunched_count+' crunched · ~'+s.crunch_tokens_saved+' tokens saved · '+Math.round((s.avg_crunch_ratio||0)*100)+'% avg ratio';
    document.getElementById('c-thinking-cost').textContent=fmt(s.today_thinking_cost_usd,4);
    document.getElementById('c-thinking-tok').textContent=fmtTok(s.today_thinking_output_tokens||0)+' thinking tokens';

    const tb=document.getElementById('tbody');
    tb.innerHTML=d.recent.map(row=>{
      const routed=row.routed_model&&row.routed_model!==row.requested_model;
      const errClass=row.status_code>=400?'err-row':'';
      const flags=[
        row.cache_hit?'<span class="badge hit">cache</span>':'<span class="badge miss">miss</span>',
        row.stream?'<span class="badge stream">stream</span>':'',
        routed?'<span class="badge routed">routed</span>':'',
        row.crunched?'<span class="badge crunched">crunched</span>':'',
        row.status_code>=400?`<span class="badge err">${row.status_code}</span>`:'',
      ].filter(Boolean).join(' ');
      const usedModel=`<span class="model${routed?' downgraded':''}">${shortModel(row.routed_model||row.requested_model)}</span>`;
      return `<tr class="${errClass}">
        <td class="ts">${ago(row.created_at)}</td>
        <td><span class="badge provider">${shortProvider(row.provider)}</span></td>
        <td class="model">${shortModel(row.requested_model)}</td>
        <td>${usedModel}</td>
        <td class="tokens">${fmtTok(row.input_tokens)}<span class="arrow">/</span>${fmtTok(row.output_tokens)}</td>
        <td class="cost">${fmt(row.cost_est_usd,5)}</td>
        <td class="latency">${fmtMs(row.latency_ms)}</td>
        <td>${flags}</td>
      </tr>`;
    }).join('');

    document.getElementById('status').textContent='updated '+new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById('status').textContent='error: '+e.message;
  }
}

async function refreshCategories(){
  try{
    const r=await fetch('/agentflow/stats/full');
    const d=await r.json();
    const tb=document.getElementById('cat-tbody');
    const rows=d.category_breakdown||[];
    const total=rows.reduce((s,r)=>s+(r.count||0),0)||1;
    tb.innerHTML=rows.map(row=>{
      const pct=Math.round((row.count/total)*100);
      return `<tr>
        <td><span class="badge provider">${shortProvider(row.provider)}</span> <span class="badge miss">${row.category}</span></td>
        <td>${(row.count||0).toLocaleString()} <span style="color:#8b949e;font-size:11px">(${pct}%)</span></td>
        <td class="cost">${fmt(row.cost_usd,5)}</td>
        <td class="tokens">${(row.routed_count||0).toLocaleString()}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="4" style="color:#8b949e">No data yet</td></tr>';
  }catch(e){}
}

async function refreshSessions(){
  try{
    const r=await fetch('/agentflow/stats/sessions');
    const d=await r.json();
    const tb=document.getElementById('sess-tbody');
    const rows=d.sessions||[];
    tb.innerHTML=rows.map(row=>`<tr>
      <td class="ts">${row.sid}</td>
      <td>${(row.calls||0).toLocaleString()}</td>
      <td class="cost">${fmt(row.cost_usd,5)}</td>
      <td class="tokens">${row.tool_result||0}</td>
      <td class="tokens">${row.tool_heavy||0}</td>
      <td class="tokens">${row.short_completion||0}</td>
      <td class="tokens">${row.code_gen||0}</td>
      <td class="tokens">${row.chat||0}</td>
      <td class="tokens">${row.other||0}</td>
    </tr>`).join('')||'<tr><td colspan="9" style="color:#8b949e">No sessions today</td></tr>';
  }catch(e){}
}

refresh();
refreshWeekly();
refreshCategories();
refreshSessions();
setInterval(refresh,5000);
setInterval(refreshWeekly,30000);
setInterval(refreshCategories,30000);
setInterval(refreshSessions,30000);
</script>
</body>
</html>"""


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
