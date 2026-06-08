from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentflow_proxy.crunch import crunch_body
from agentflow_proxy.router import route_model
from agentflow_proxy.store import Store, stable_json, utc_now

load_dotenv()

DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_HOST = os.getenv("AGENTFLOW_CODEX_APP_PROXY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENTFLOW_CODEX_APP_PROXY_PORT", "4013"))
DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", "ws://127.0.0.1:4014")
LOG_EVENTS = os.getenv("AGENTFLOW_CODEX_APP_LOG_EVENTS", "1") != "0"
DB_BUSY_TIMEOUT_MS = int(os.getenv("AGENTFLOW_CODEX_APP_DB_BUSY_TIMEOUT_MS", "100"))
CODEX_APP_OPTIMIZE = os.getenv("AGENTFLOW_CODEX_APP_OPTIMIZE", "1") != "0"

store = Store(DEFAULT_DB)
if getattr(store, "backend", None) == "sqlite":
    store.conn.execute(f"pragma busy_timeout = {DB_BUSY_TIMEOUT_MS}")
app = FastAPI(title="AgentFlow Codex App-Server Proxy", version="0.1.0")

_CODEX_ACTION_KEY_HINTS = {
    "approval",
    "approvalrequest",
    "approval_request",
    "apply_patch",
    "cmd",
    "command",
    "exec",
    "function_call",
    "patch",
    "shell",
    "tool_call",
    "tool_calls",
}
_CODEX_ACTION_VALUE_HINTS = {
    "approval_request",
    "apply_patch",
    "command",
    "exec",
    "function_call",
    "shell",
    "tool_call",
    "tool_result",
    "tool_use",
}
_CODEX_MODEL_FIELDS = ("model", "modelId", "model_id")


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


def _policy_decision(kind: str, status: str, reason: str, *, enabled: bool = CODEX_APP_OPTIMIZE) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "status": status,
        "reason": reason,
        "policy_source": "local-default",
        "surface": "codex_app_turn",
        "decision_type": kind,
        "applied": status == "applied",
    }


def _not_applied_metadata(reason: str, *, enabled: bool = CODEX_APP_OPTIMIZE) -> dict[str, dict[str, Any]]:
    return {
        "routing": _policy_decision("routing", "not-applied", reason, enabled=enabled),
        "crunch": _policy_decision("crunch", "not-applied", reason, enabled=enabled),
        "cache": _policy_decision("cache", "not-applied", reason, enabled=enabled),
    }


def _contains_action_hint(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_l = str(key).replace("-", "_").lower()
            if key_l in _CODEX_ACTION_KEY_HINTS:
                return True
            if key_l == "type" and isinstance(nested, str) and nested.strip().lower() in _CODEX_ACTION_VALUE_HINTS:
                return True
            if isinstance(nested, (dict, list)) and _contains_action_hint(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_action_hint(item) for item in value)
    return False


def _model_field(params: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in _CODEX_MODEL_FIELDS:
        value = params.get(field)
        if value is not None:
            return field, str(value)
    return None, None


def _codex_route_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_field, requested_model = _model_field(params)
    if not requested_model:
        return params, _policy_decision("routing", "not-applicable", "codex-turn-start-model-field-absent")

    route_body = copy.deepcopy(params)
    route_body["model"] = requested_model
    routed_model, routing_meta = route_model(route_body)
    routing_meta = dict(routing_meta)
    routing_meta.update({
        "surface": "codex_app_turn",
        "decision_type": "routing",
        "status": "applied" if routed_model != requested_model else "skipped",
        "applied": routed_model != requested_model,
        "model_field": model_field,
    })
    if routed_model != requested_model and model_field is not None:
        routed_params = copy.deepcopy(params)
        routed_params[model_field] = routed_model
        return routed_params, routing_meta
    return params, routing_meta


def _codex_crunch_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    crunched, crunch_meta = crunch_body(params)
    crunch_meta = dict(crunch_meta)
    changed = bool(crunch_meta.get("changed"))
    crunch_meta.update({
        "surface": "codex_app_turn",
        "decision_type": "crunch",
        "status": "applied" if changed else "skipped",
        "reason": "codex-turn-start-crunched" if changed else "no-change",
        "applied": changed,
    })
    return crunched, crunch_meta


def _optimize_client_message(raw: str | bytes) -> tuple[str | bytes, dict[str, dict[str, Any]] | None]:
    if isinstance(raw, bytes):
        return raw, _not_applied_metadata("binary-frame")

    try:
        msg = json.loads(raw)
    except Exception:
        return raw, _not_applied_metadata("non-json-frame")

    if not isinstance(msg, dict):
        return raw, _not_applied_metadata("json-message-not-object")

    method = msg.get("method")
    if method != "turn/start":
        return raw, _not_applied_metadata("method-not-eligible")

    if not CODEX_APP_OPTIMIZE:
        return raw, _not_applied_metadata("codex-app-optimization-disabled", enabled=False)

    params = msg.get("params")
    if not isinstance(params, dict):
        return raw, _not_applied_metadata("params-not-object")

    if _contains_action_hint(params):
        return raw, _not_applied_metadata("action-like-params")

    crunched_params, crunch_meta = _codex_crunch_params(params)
    routed_params, routing_meta = _codex_route_params(crunched_params)
    cache_meta = _policy_decision("cache", "not-applied", "codex-app-cache-not-implemented")

    optimized = copy.deepcopy(msg)
    optimized["params"] = routed_params
    if optimized == msg:
        return raw, {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
    return json.dumps(optimized, separators=(",", ":"), ensure_ascii=False), {
        "routing": routing_meta,
        "crunch": crunch_meta,
        "cache": cache_meta,
    }


def _log_codex_app_event(**kwargs: Any) -> None:
    try:
        store.log_codex_app_event(**kwargs)
    except Exception as exc:
        print(f"AgentFlow Codex app telemetry skipped: {exc}", file=sys.stderr)


def _record_message(
    raw: str | bytes,
    *,
    direction: str,
    session_id: str,
    request_started: dict[str, float],
    optimization_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not LOG_EVENTS:
        return
    message_chars = len(raw)
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            _log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction=direction,
                message_chars=message_chars,
                session_id=session_id,
                routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
                crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
                cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
            )
            return
    else:
        text = raw
    try:
        msg = json.loads(text)
    except Exception:
        _log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction=direction,
            message_chars=message_chars,
            session_id=session_id,
            routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
            crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
            cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
        )
        return

    if not isinstance(msg, dict):
        _log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction=direction,
            message_chars=message_chars,
            session_id=session_id,
            routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
            crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
            cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
        )
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

    _log_codex_app_event(
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
        routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
        crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
        cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
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
                        forwarded, optimization_metadata = _optimize_client_message(msg["text"])
                        _record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id=session_id,
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        await upstream.send(forwarded)
                    elif msg.get("bytes") is not None:
                        forwarded, optimization_metadata = _optimize_client_message(msg["bytes"])
                        _record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id=session_id,
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        await upstream.send(forwarded)

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
