from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
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
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com")
DEFAULT_DB = os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_PORT = int(os.getenv("AGENTFLOW_PORT", "4000"))
DEFAULT_HOST = os.getenv("AGENTFLOW_HOST", "127.0.0.1")

# Approximate public list prices in USD per million tokens. Update in config/env as needed.
MODEL_PRICES = {
    "claude-opus-4.5": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4.5": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-haiku-4.5": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-haiku": (0.8, 4.0),
}

HAIKU_DEFAULT = os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4.5")
SONNET_DEFAULT = os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4.5")
OPUS_DEFAULT = os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4.5")

CACHE_ENABLED = os.getenv("AGENTFLOW_CACHE", "1") != "0"
CRUNCH_ENABLED = os.getenv("AGENTFLOW_CRUNCH", "1") != "0"
ROUTING_ENABLED = os.getenv("AGENTFLOW_ROUTING", "1") != "0"
LOG_BODIES = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
# Avoid caching tool-using agent turns by default. Exact cache can be dangerous when tools reflect filesystem state.
CACHE_TOOL_CALLS = os.getenv("AGENTFLOW_CACHE_TOOL_CALLS", "0") == "1"
CRUNCH_THRESHOLD_CHARS = int(os.getenv("AGENTFLOW_CRUNCH_THRESHOLD_CHARS", "24000"))
HTTP_TIMEOUT = float(os.getenv("AGENTFLOW_HTTP_TIMEOUT", "600"))

TOKEN_CHARS = 4  # rough estimator only


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens_from_text(text: str) -> int:
    return max(1, int(len(text) / TOKEN_CHARS))


def extract_text(obj: Any) -> str:
    parts: list[str] = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            parts.append(extract_text(x))
    elif isinstance(obj, dict):
        # Only text-ish values for token estimate. Avoid binary/source blobs where possible.
        for k, v in obj.items():
            if k in {"text", "content", "input", "system", "name", "type"}:
                parts.append(extract_text(v))
            elif isinstance(v, (list, dict)):
                parts.append(extract_text(v))
    return "\n".join(p for p in parts if p)


def normalize_text(s: str) -> str:
    # Conservative crunching: whitespace cleanup only; do not paraphrase.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip() if len(s) > 200 else s


def has_tools(body: dict[str, Any]) -> bool:
    if body.get("tools"):
        return True
    text = stable_json(body.get("messages", []))
    return "tool_use" in text or "tool_result" in text


def crunch_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative request cruncher.

    It deliberately does NOT summarize with another model. For agent use this is safer.
    Current tactics:
    - normalize whitespace in large text blocks
    - deduplicate exact repeated text blocks within the same request
    - if extremely large, compress older non-tool text blocks to bounded heads/tails
    """
    if not CRUNCH_ENABLED:
        return body, {"enabled": False, "changed": False}

    new_body = copy.deepcopy(body)
    before = len(stable_json(new_body))
    seen: dict[str, int] = {}
    replacements = 0
    shortened = 0

    def process_content(content: Any, allow_shorten: bool) -> Any:
        nonlocal replacements, shortened
        if isinstance(content, str):
            txt = normalize_text(content)
            h = sha256_text(txt)
            if len(txt) > 1000 and h in seen:
                replacements += 1
                return f"[AgentFlow: exact duplicate text block omitted; same as earlier block #{seen[h]} hash={h[:12]}]"
            seen[h] = len(seen) + 1
            if allow_shorten and len(txt) > 8000:
                shortened += 1
                return txt[:3500] + f"\n\n[AgentFlow: middle of long older text block omitted; hash={h[:12]}; original_chars={len(txt)}]\n\n" + txt[-2500:]
            return txt
        if isinstance(content, list):
            out = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    item = copy.deepcopy(item)
                    item["text"] = process_content(item["text"], allow_shorten)
                elif isinstance(item, dict) and item.get("type") in {"tool_use", "tool_result"}:
                    # Tool blocks are protocol/state sensitive; don't alter.
                    pass
                elif isinstance(item, dict):
                    item = process_content(item, allow_shorten)
                elif isinstance(item, str):
                    item = process_content(item, allow_shorten)
                out.append(item)
            return out
        if isinstance(content, dict):
            out = copy.deepcopy(content)
            for k, v in list(out.items()):
                if k in {"text", "content"}:
                    out[k] = process_content(v, allow_shorten)
            return out
        return content

    # System can be string or list of blocks.
    if "system" in new_body:
        new_body["system"] = process_content(new_body["system"], allow_shorten=False)

    messages = new_body.get("messages") or []
    huge = before > CRUNCH_THRESHOLD_CHARS
    for idx, msg in enumerate(messages):
        # only shorten older text, not the latest user/assistant context
        allow_shorten = huge and idx < max(0, len(messages) - 4)
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = process_content(msg["content"], allow_shorten=allow_shorten)

    after = len(stable_json(new_body))
    return new_body, {
        "enabled": True,
        "changed": after != before,
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "duplicate_blocks_replaced": replacements,
        "long_blocks_shortened": shortened,
    }


def route_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    if not ROUTING_ENABLED:
        return requested, {"enabled": False, "requested_model": requested, "routed_model": requested, "reason": "routing disabled"}

    requested_l = requested.lower()
    text_chars = len(extract_text(body))
    tools = has_tools(body)
    max_tokens = int(body.get("max_tokens") or 4096)

    routed = requested
    reason = "keep requested model"

    # Conservative routing: don't downgrade tool-heavy Claude Code calls too aggressively.
    if not tools:
        if "opus" in requested_l and text_chars < 24000:
            routed = SONNET_DEFAULT
            reason = "non-tool Opus request under threshold routed to Sonnet"
        elif "sonnet" in requested_l and text_chars < 6000 and max_tokens <= 2048:
            routed = HAIKU_DEFAULT
            reason = "small non-tool Sonnet request routed to Haiku"
    else:
        # Tool calls are likely agent-state-sensitive. Only route tiny tool-free-looking continuations.
        if "opus" in requested_l and text_chars < 8000:
            routed = SONNET_DEFAULT
            reason = "small tool request Opus routed to Sonnet; disable with AGENTFLOW_ROUTING=0 if unsafe"

    return routed, {
        "enabled": True,
        "requested_model": requested,
        "routed_model": routed,
        "reason": reason,
        "text_chars": text_chars,
        "has_tools": tools,
    }


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        create table if not exists cache (
          cache_key text primary key,
          created_at text not null,
          model text not null,
          response_json text not null,
          request_chars integer,
          response_chars integer
        )
        """)
        cur.execute("""
        create table if not exists calls (
          id text primary key,
          created_at text not null,
          path text not null,
          requested_model text,
          routed_model text,
          stream integer,
          cache_hit integer,
          status_code integer,
          latency_ms integer,
          input_tokens_est integer,
          output_tokens_est integer,
          cost_est_usd real,
          crunch_json text,
          routing_json text,
          error text,
          request_json text,
          response_json text
        )
        """)
        self.conn.commit()

    def get_cache(self, key: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("select response_json from cache where cache_key = ?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row["response_json"])

    def set_cache(self, key: str, model: str, request_chars: int, response: dict[str, Any]) -> None:
        response_json = stable_json(response)
        self.conn.execute(
            "insert or replace into cache(cache_key, created_at, model, response_json, request_chars, response_chars) values (?, ?, ?, ?, ?, ?)",
            (key, utc_now(), model, response_json, request_chars, len(response_json)),
        )
        self.conn.commit()

    def log_call(self, **kwargs: Any) -> None:
        cols = [
            "id", "created_at", "path", "requested_model", "routed_model", "stream", "cache_hit", "status_code",
            "latency_ms", "input_tokens_est", "output_tokens_est", "cost_est_usd", "crunch_json", "routing_json",
            "error", "request_json", "response_json",
        ]
        values = [kwargs.get(c) for c in cols]
        self.conn.execute(
            f"insert into calls({','.join(cols)}) values ({','.join(['?']*len(cols))})",
            values,
        )
        self.conn.commit()


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


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    prices = None
    ml = model.lower()
    for name, val in MODEL_PRICES.items():
        if name in ml:
            prices = val
            break
    if not prices:
        return None
    in_per_m, out_per_m = prices
    return (input_tokens / 1_000_000) * in_per_m + (output_tokens / 1_000_000) * out_per_m


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
    raw_body = await request.json()
    stream = bool(raw_body.get("stream"))
    requested_model = str(raw_body.get("model") or "")
    error: Optional[str] = None
    status_code = 200
    crunch_meta: dict[str, Any] = {}
    routing_meta: dict[str, Any] = {}
    cache_hit = False
    response_body: Optional[dict[str, Any]] = None

    try:
        crunched, crunch_meta = crunch_body(raw_body)
        routed_model, routing_meta = route_model(crunched)
        crunched["model"] = routed_model
        input_tokens = estimate_tokens_from_text(extract_text(crunched))
        headers = build_forward_headers(request)

        # Streaming is passed through and not cached, but still logged when the stream finishes.
        if stream:
            async def gen() -> AsyncIterator[bytes]:
                nonlocal status_code, error
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        async with client.stream("POST", DEFAULT_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched) as r:
                            status_code = r.status_code
                            async for chunk in r.aiter_bytes():
                                yield chunk
                except Exception as exc:
                    error = repr(exc)
                    yield f"event: error\ndata: {json.dumps({'error': error})}\n\n".encode("utf-8")
                finally:
                    latency_ms = int((time.time() - started) * 1000)
                    store.log_call(
                        id=call_id, created_at=utc_now(), path=path,
                        requested_model=requested_model, routed_model=crunched.get("model"), stream=1,
                        cache_hit=0, status_code=status_code, latency_ms=latency_ms,
                        input_tokens_est=input_tokens, output_tokens_est=None,
                        cost_est_usd=None, crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                        error=error, request_json=stable_json(crunched) if LOG_BODIES else None, response_json=None,
                    )

            return StreamingResponse(gen(), media_type="text/event-stream")

        can_cache = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tools(crunched))
        key = cache_key_for(crunched, path)
        if can_cache:
            cached = store.get_cache(key)
            if cached is not None:
                cache_hit = True
                response_body = cached
                latency_ms = int((time.time() - started) * 1000)
                out_tokens = estimate_tokens_from_text(response_output_text(response_body))
                store.log_call(
                    id=call_id, created_at=utc_now(), path=path,
                    requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
                    cache_hit=1, status_code=200, latency_ms=latency_ms,
                    input_tokens_est=input_tokens, output_tokens_est=out_tokens,
                    cost_est_usd=0.0, crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
                    error=None, request_json=stable_json(crunched) if LOG_BODIES else None,
                    response_json=stable_json(response_body) if LOG_BODIES else None,
                )
                return JSONResponse(response_body, headers={"x-agentflow-cache": "hit", "x-agentflow-routed-model": str(crunched.get("model"))})

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.post(DEFAULT_UPSTREAM.rstrip("/") + path, headers=headers, json=crunched)
        status_code = r.status_code
        try:
            response_body = r.json()
        except Exception:
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/plain"))

        if r.status_code < 400 and can_cache and response_body is not None:
            store.set_cache(key, str(crunched.get("model")), len(stable_json(crunched)), response_body)

        out_tokens = estimate_tokens_from_text(response_output_text(response_body)) if response_body else 0
        cost = estimate_cost(str(crunched.get("model")), input_tokens, out_tokens)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=crunched.get("model"), stream=0,
            cache_hit=0, status_code=status_code, latency_ms=latency_ms,
            input_tokens_est=input_tokens, output_tokens_est=out_tokens,
            cost_est_usd=cost, crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            error=None if status_code < 400 else stable_json(response_body)[:1000],
            request_json=stable_json(crunched) if LOG_BODIES else None,
            response_json=stable_json(response_body) if LOG_BODIES else None,
        )
        return JSONResponse(response_body, status_code=status_code, headers={"x-agentflow-cache": "miss", "x-agentflow-routed-model": str(crunched.get("model"))})

    except Exception as exc:
        error = repr(exc)
        latency_ms = int((time.time() - started) * 1000)
        store.log_call(
            id=call_id, created_at=utc_now(), path=path,
            requested_model=requested_model, routed_model=None, stream=int(stream), cache_hit=0,
            status_code=500, latency_ms=latency_ms,
            input_tokens_est=None, output_tokens_est=None, cost_est_usd=None,
            crunch_json=stable_json(crunch_meta), routing_json=stable_json(routing_meta),
            error=error, request_json=stable_json(raw_body) if LOG_BODIES else None, response_json=None,
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
