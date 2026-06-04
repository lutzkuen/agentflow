from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
DEFAULT_DB = os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_PORT = int(os.getenv("AGENTFLOW_PORT", "4000"))
DEFAULT_HOST = os.getenv("AGENTFLOW_HOST", "0.0.0.0")

CACHE_ENABLED = os.getenv("AGENTFLOW_CACHE", "1") != "0"
LOG_BODIES = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
# Avoid caching tool-using agent turns by default. Exact cache can be dangerous when tools reflect filesystem state.
CACHE_TOOL_CALLS = os.getenv("AGENTFLOW_CACHE_TOOL_CALLS", "0") == "1"
HTTP_TIMEOUT = float(os.getenv("AGENTFLOW_HTTP_TIMEOUT", "600"))
SEMANTIC_CACHE_ENABLED = os.getenv("AGENTFLOW_SEMANTIC_CACHE", "0") == "1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("AGENTFLOW_SEMANTIC_THRESHOLD", "0.95"))
MIN_REQUEST_INTERVAL_MS = int(os.getenv("AGENTFLOW_MIN_REQUEST_INTERVAL_MS", "0"))

_forward_lock = asyncio.Lock()
_last_forward_time: float = 0.0


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


from agentflow_proxy.store import Store, utc_now, stable_json
from agentflow_proxy.pricing import MODEL_PRICES, MODEL_ALIASES, estimate_cost
from agentflow_proxy.router import extract_text, has_tools, categorize_request, route_model, HAIKU_DEFAULT, SONNET_DEFAULT, OPUS_DEFAULT
from agentflow_proxy.crunch import (
    TOKEN_CHARS, sha256_text, estimate_tokens_from_text, build_embedding,
    crunch_body, inject_prompt_cache, has_cache_control_blocks,
)


store = Store(DEFAULT_DB)
app = FastAPI(title="AgentFlow Claude Proxy", version="0.1.0")


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


def cache_key_for(body: dict[str, Any], path: str) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    return sha256_text(path + "\n" + stable_json(body))


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "db": DEFAULT_DB, "upstream": DEFAULT_UPSTREAM, "time": utc_now()}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "data": [
            {"id": HAIKU_DEFAULT, "type": "model"},
            {"id": SONNET_DEFAULT, "type": "model"},
            {"id": OPUS_DEFAULT, "type": "model"},
        ]
    }


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
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

    category = categorize_request(raw_body)

    try:
        crunched, crunch_meta = crunch_body(raw_body)
        crunched, prompt_cached = inject_prompt_cache(crunched)
        routed_model, routing_meta = route_model(crunched)
        resolved_requested_model = crunched.get("model", requested_model)
        crunched["model"] = routed_model
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
                sse_frame_buf = b""
                stream_retry_count = 0

                def parse_sse_usage(frame: bytes) -> None:
                    nonlocal actual_in, actual_out, cache_creation_in, cache_read_in
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

                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        while True:
                            await _throttle_forward()
                            async with client.stream("POST", DEFAULT_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched) as r:
                                status_code = r.status_code
                                if status_code in (429, 529) and stream_retry_count < 3:
                                    stream_retry_count += 1
                                    delay = (2 ** (stream_retry_count - 1)) * (1.0 + random.random() * 0.5)
                                    print(f"rate_limit: status={status_code} retry={stream_retry_count} delay={delay:.1f}s")
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
                except Exception as exc:
                    error = repr(exc)
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n".encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    cost_in = actual_in if actual_in is not None else input_tokens
                    cost_out = actual_out if actual_out is not None else 0
                    cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out)
                    cost_baseline = estimate_cost(requested_model, cost_in, cost_out)
                    if cache_creation_in or cache_read_in:
                        print(f"prompt_cache: creation={cache_creation_in} read={cache_read_in}")
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path,
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        actual_input_tokens=actual_in, actual_output_tokens=actual_out,
                        cost_est_usd=cost, cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        error=error, request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                        session_id=session_id, category=category,
                        cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
                        retry_count=stream_retry_count,
                    )

            return StreamingResponse(gen(), media_type="text/event-stream")

        can_cache = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tools(crunched))
        key = cache_key_for(crunched, path)
        can_semantic_cache = SEMANTIC_CACHE_ENABLED and not has_tools(crunched)
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
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(sem_resp) if LOG_BODIES else None,
                    session_id=session_id, category=category, retry_count=0,
                )
                return JSONResponse(sem_resp, headers={"x-agentflow-cache": "semantic-hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            while True:
                await _throttle_forward()
                r = await client.post(DEFAULT_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched)
                if r.status_code in (429, 529) and retry_count < 3:
                    retry_count += 1
                    delay = (2 ** (retry_count - 1)) * (1.0 + random.random() * 0.5)
                    print(f"rate_limit: status={r.status_code} retry={retry_count} delay={delay:.1f}s")
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
        out_tokens = estimate_tokens_from_text(response_output_text(response_body)) if response_body else 0
        cost_in = actual_in if actual_in is not None else input_tokens
        cost_out = actual_out if actual_out is not None else out_tokens
        cost = estimate_cost(str(crunched.get("model")), cost_in, cost_out)
        cost_baseline = estimate_cost(requested_model, cost_in, cost_out)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            actual_input_tokens=actual_in, actual_output_tokens=actual_out,
            cost_est_usd=cost, cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            error=None if status_code < 400 else stable_json(response_body)[:1000],
            request_json=stable_json(crunched) if LOG_BODIES else None,
            response_json=stable_json(response_body) if LOG_BODIES else None,
            session_id=session_id, category=category,
            cache_creation_input_tokens=cache_creation_in, cache_read_input_tokens=cache_read_in,
            retry_count=retry_count,
        )
        return JSONResponse(response_body, status_code=status_code, headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))})

    except Exception as exc:
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None, cost_baseline_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
            session_id=session_id, category=category, retry_count=retry_count,
        )
        return JSONResponse({"type": "error", "error": {"type": "agentflow_proxy_error", "message": error}}, status_code=500)


@app.get("/agentflow/stats")
async def stats() -> dict[str, Any]:
    conn = store.conn
    calls = conn.execute("select count(*) c from calls").fetchone()["c"]
    cache_hits = conn.execute("select count(*) c from calls where cache_hit = 1").fetchone()["c"]
    routed = conn.execute("select requested_model, routed_model, count(*) c from calls group by requested_model, routed_model order by c desc limit 20").fetchall()
    recent = conn.execute("select created_at, requested_model, routed_model, cache_hit, status_code, latency_ms, cost_est_usd from calls order by created_at desc limit 20").fetchall()
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
    downgraded = q("""
        select requested_model, routed_model,
               coalesce(actual_input_tokens, input_tokens_est, 0) as in_tok,
               coalesce(actual_output_tokens, output_tokens_est, 0) as out_tok
        from calls where requested_model != routed_model and routed_model is not null
    """)
    for row in downgraded:
        req_cost = estimate_cost(row["requested_model"], row["in_tok"], row["out_tok"]) or 0
        act_cost = estimate_cost(row["routed_model"], row["in_tok"], row["out_tok"]) or 0
        routing_savings += max(0.0, req_cost - act_cost)

    crunch_chars_saved = s("select sum(json_extract(crunch_json, '$.saved_chars')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    crunch_tokens_saved = s("select sum(json_extract(crunch_json, '$.tokens_saved_est')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    avg_crunch_ratio = s("select avg(json_extract(crunch_json, '$.crunch_ratio')) from calls where json_extract(crunch_json, '$.changed') = 1") or 0
    prompt_cache_creation_tokens = s("select sum(cache_creation_input_tokens) from calls") or 0
    prompt_cache_read_tokens = s("select sum(cache_read_input_tokens) from calls") or 0

    recent = q("""
        select id, created_at, requested_model, routed_model, stream, cache_hit,
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
        select requested_model, routed_model, count(*) as count
        from calls group by requested_model, routed_model order by count desc limit 15
    """)

    category_breakdown = q("""
        select coalesce(category, 'unknown') as category, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(category, 'unknown') order by count desc
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
            "cache_savings_usd": round(cache_cost_saved, 6),
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
        },
        "recent": recent,
        "routing_breakdown": routing_breakdown,
        "category_breakdown": category_breakdown,
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
  <span class="sub">Claude proxy · cost reduction dashboard</span>
  <span id="status">loading...</span>
</header>

<div class="cards" id="cards">
  <div class="card"><div class="label">Calls today</div><div class="value" id="c-today">—</div><div class="sub" id="c-total">— total</div></div>
  <div class="card"><div class="label">Cost today</div><div class="value" id="c-cost">—</div><div class="sub" id="c-cost-total">— total</div></div>
  <div class="card green"><div class="label">Saved by routing</div><div class="value" id="c-routing">—</div><div class="sub" id="c-routed-n">— calls routed</div></div>
  <div class="card green"><div class="label">Saved by cache</div><div class="value" id="c-cache-saved">—</div><div class="sub" id="c-cache-rate">— hit rate</div></div>
  <div class="card blue"><div class="label">Avg latency</div><div class="value" id="c-latency">—</div><div class="sub" id="c-crunched">— crunched</div></div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('recent')">Recent calls</button>
  <button class="tab-btn" onclick="showTab('weekly')">7-day stats</button>
  <button class="tab-btn" onclick="showTab('categories')">By category</button>
</div>

<div class="tab-panel active" id="tab-recent">
<div class="section">
  <h2>Recent calls</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Requested</th><th>Used</th><th>Tokens in/out</th><th>Cost</th><th>Latency</th><th>Flags</th>
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

function showTab(name){
  ['recent','weekly','categories'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===name);
  });
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active',['recent','weekly','categories'][i]===name);
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
    document.getElementById('c-routing').textContent=fmt(s.routing_savings_usd,4);
    document.getElementById('c-routed-n').textContent=s.routed_count+' calls routed';
    document.getElementById('c-cache-saved').textContent=fmt(s.cache_savings_usd,4);
    document.getElementById('c-cache-rate').textContent=Math.round(s.cache_hit_rate*100)+'% hit rate';
    document.getElementById('c-latency').textContent=fmtMs(s.avg_latency_ms);
    document.getElementById('c-crunched').textContent=s.crunched_count+' crunched · ~'+s.crunch_tokens_saved+' tokens saved · '+Math.round((s.avg_crunch_ratio||0)*100)+'% avg ratio';

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
        <td><span class="badge miss">${row.category}</span></td>
        <td>${(row.count||0).toLocaleString()} <span style="color:#8b949e;font-size:11px">(${pct}%)</span></td>
        <td class="cost">${fmt(row.cost_usd,5)}</td>
        <td class="tokens">${(row.routed_count||0).toLocaleString()}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="4" style="color:#8b949e">No data yet</td></tr>';
  }catch(e){}
}

refresh();
refreshWeekly();
refreshCategories();
setInterval(refresh,5000);
setInterval(refreshWeekly,30000);
setInterval(refreshCategories,30000);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentFlow Claude-compatible local proxy")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("agentflow_proxy.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
