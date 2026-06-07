from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentflow_proxy.store import Store, stable_json, utc_now

load_dotenv()

DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_HOST = os.getenv("AGENTFLOW_CODEX_APP_PROXY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENTFLOW_CODEX_APP_PROXY_PORT", "4013"))
DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", "ws://127.0.0.1:4014")
LOG_EVENTS = os.getenv("AGENTFLOW_CODEX_APP_LOG_EVENTS", "1") != "0"

store = Store(DEFAULT_DB)
app = FastAPI(title="AgentFlow Codex App-Server Proxy", version="0.1.0")


def _input_text_chars(value: Any) -> int:
    total = 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_input_text_chars(item) for item in value)
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"text", "inputText", "input_text", "value", "prompt"}:
                total += _input_text_chars(nested)
            elif isinstance(nested, (dict, list)):
                total += _input_text_chars(nested)
    return total


def _thread_id(params: Any) -> Optional[str]:
    if isinstance(params, dict):
        value = params.get("threadId") or params.get("thread_id")
        return str(value) if value is not None else None
    return None


def _request_id(msg: dict[str, Any]) -> Optional[str]:
    value = msg.get("id")
    return str(value) if value is not None else None


def _record_message(
    raw: str | bytes,
    *,
    direction: str,
    session_id: str,
    request_started: dict[str, float],
) -> None:
    if not LOG_EVENTS:
        return
    message_chars = len(raw)
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction=direction,
                message_chars=message_chars,
                session_id=session_id,
            )
            return
    else:
        text = raw
    try:
        msg = json.loads(text)
    except Exception:
        store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction=direction,
            message_chars=message_chars,
            session_id=session_id,
        )
        return

    if not isinstance(msg, dict):
        return

    params = msg.get("params")
    method = msg.get("method")
    rid = _request_id(msg)
    params_chars = len(stable_json(params)) if params is not None else None
    input_value = params.get("input") if isinstance(params, dict) else None
    input_items = len(input_value) if isinstance(input_value, list) else None
    input_text_chars = _input_text_chars(input_value)
    result = msg.get("result")
    error = msg.get("error")
    latency_ms: Optional[int] = None
    if direction == "server_to_client" and rid is not None:
        started = request_started.pop(rid, None)
        if started is not None:
            latency_ms = int((time.time() - started) * 1000)
    if direction == "client_to_server" and rid is not None and method is not None:
        request_started[rid] = time.time()

    store.log_codex_app_event(
        id=str(uuid.uuid4()),
        created_at=utc_now(),
        direction=direction,
        method=str(method) if method is not None else None,
        request_id=rid,
        thread_id=_thread_id(params),
        message_chars=message_chars,
        params_chars=params_chars,
        input_items=input_items,
        input_text_chars=input_text_chars if input_text_chars else None,
        result_chars=len(stable_json(result)) if result is not None else None,
        error_code=error.get("code") if isinstance(error, dict) else None,
        error_message=(error.get("message")[:500] if isinstance(error, dict) and isinstance(error.get("message"), str) else None),
        latency_ms=latency_ms,
        session_id=session_id,
    )


def _upstream_url(path: str, query: str) -> str:
    base = DEFAULT_UPSTREAM.rstrip("/")
    if path and path != "/":
        base += path if path.startswith("/") else "/" + path
    if query:
        base += "?" + query
    return base


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "mode": "codex-app-proxy", "db": DEFAULT_DB, "upstream": DEFAULT_UPSTREAM, "time": utc_now()}


@app.websocket("/{path:path}")
async def relay(websocket: WebSocket, path: str = "") -> None:
    session_id = str(uuid.uuid4())
    await websocket.accept()
    upstream_url = _upstream_url("/" + path if path else "", str(websocket.query_params))
    request_started: dict[str, float] = {}

    try:
        async with websockets.connect(upstream_url) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        await upstream.close()
                        return
                    if msg.get("text") is not None:
                        _record_message(msg["text"], direction="client_to_server", session_id=session_id, request_started=request_started)
                        await upstream.send(msg["text"])
                    elif msg.get("bytes") is not None:
                        _record_message(msg["bytes"], direction="client_to_server", session_id=session_id, request_started=request_started)
                        await upstream.send(msg["bytes"])

            async def upstream_to_client() -> None:
                async for msg in upstream:
                    _record_message(msg, direction="server_to_client", session_id=session_id, request_started=request_started)
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_text(msg)

            tasks = {asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())}
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


def main() -> None:
    global DEFAULT_UPSTREAM
    parser = argparse.ArgumentParser(description="AgentFlow Codex app-server websocket proxy")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    args = parser.parse_args()

    DEFAULT_UPSTREAM = args.upstream

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
