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

from agentflow_proxy.cache import cache_file_dependency_snapshots, cache_key_for
from agentflow_proxy.crunch import TOKEN_CHARS, crunch_body, crunch_codex_turn_params
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
CODEX_APP_CACHE = os.getenv("AGENTFLOW_CODEX_APP_CACHE", "0") == "1"

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
_CODEX_SAFE_TURN_PARAM_KEYS = {
    "input",
    "instructions",
    "max_tokens",
    "maxTokens",
    "model",
    "modelId",
    "model_id",
    "temperature",
    "threadId",
    "thread_id",
    "top_p",
    "topP",
}
_CODEX_TEXT_INPUT_TYPES = {"text", "input_text"}
_INTERNAL_REPLAY_FRAME_KEY = "_agentflow_replay_frame"
_INTERNAL_CACHE_KEY = "_agentflow_cache_key"


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


def _codex_cache_decision(
    status: str,
    reason: str,
    *,
    enabled: bool = CODEX_APP_CACHE,
    eligible: bool = False,
    hit_type: str | None = None,
    cache_key: str | None = None,
    replayability_level: str = "features_only",
    file_dependencies: list[dict[str, Any]] | None = None,
    invalidation_reason: str | None = None,
) -> dict[str, Any]:
    meta = _policy_decision("cache", status, reason, enabled=enabled)
    meta.update({
        "eligible": bool(eligible),
        "hit_type": hit_type or "",
        "exact_enabled": bool(CODEX_APP_CACHE),
        "replayability_level": replayability_level,
    })
    if cache_key:
        meta["cache_key"] = cache_key
    if file_dependencies:
        meta["file_dependency_count"] = len(file_dependencies)
        meta["file_dependencies"] = [
            {"path": dep.get("path"), "exists": bool(dep.get("exists"))}
            for dep in file_dependencies
            if dep.get("path")
        ]
    if invalidation_reason:
        meta["invalidation_reason"] = invalidation_reason
    return meta


def _not_applied_metadata(reason: str, *, enabled: bool = CODEX_APP_OPTIMIZE) -> dict[str, dict[str, Any]]:
    return {
        "routing": _policy_decision("routing", "not-applied", reason, enabled=enabled),
        "crunch": _policy_decision("crunch", "not-applied", reason, enabled=enabled),
        "cache": _codex_cache_decision(
            "skipped",
            reason,
            enabled=CODEX_APP_CACHE and enabled,
            eligible=False,
        ),
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
    if not crunch_meta.get("enabled", True):
        crunch_meta.update({
            "surface": "codex_app_turn",
            "decision_type": "crunch",
            "status": "skipped",
            "reason": "disabled",
            "applied": False,
        })
        return crunched, crunch_meta

    codex_crunched, codex_meta = crunch_codex_turn_params(crunched)
    before = int(crunch_meta.get("before_chars") or len(stable_json(params)))
    after = len(stable_json(codex_crunched))
    changed = after != before
    if codex_meta.get("changed"):
        crunched = codex_crunched
        crunch_meta["codex_repeated_scaffolding"] = codex_meta
        crunch_meta["codex_patterns"] = codex_meta.get("patterns") or []
        crunch_meta["codex_pattern_types"] = codex_meta.get("pattern_types") or []
        crunch_meta["repeated_codex_sections_replaced"] = codex_meta.get("repeated_sections_replaced", 0)
        crunch_meta["older_codex_input_blocks_shortened"] = codex_meta.get("older_input_blocks_shortened", 0)
    else:
        crunch_meta["codex_repeated_scaffolding"] = codex_meta

    crunch_meta.update({
        "surface": "codex_app_turn",
        "decision_type": "crunch",
        "status": "applied" if changed else "skipped",
        "reason": "codex-repeated-scaffolding-crunched" if codex_meta.get("changed") else ("codex-turn-start-crunched" if changed else "no-change"),
        "applied": changed,
        "changed": changed,
        "after_chars": after,
        "saved_chars": before - after,
        "tokens_after_est": after // TOKEN_CHARS,
        "tokens_saved_est": (before - after) // TOKEN_CHARS,
        "crunch_ratio": round((before - after) / before, 4) if before > 0 else 0,
    })
    return crunched, crunch_meta


def _is_text_only_input(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        if not value:
            return False
        return all(_is_text_only_input(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "text").strip().lower()
        if block_type not in _CODEX_TEXT_INPUT_TYPES:
            return False
        text_value = value.get("text", value.get("input_text", value.get("value")))
        return isinstance(text_value, str)
    return False


def _deterministic_sampling(params: dict[str, Any]) -> tuple[bool, str | None]:
    if "temperature" in params:
        try:
            if float(params["temperature"]) != 0.0:
                return False, "non-deterministic-temperature"
        except (TypeError, ValueError):
            return False, "invalid-temperature"
    for key in ("top_p", "topP"):
        if key in params:
            try:
                if float(params[key]) != 1.0:
                    return False, "non-deterministic-top-p"
            except (TypeError, ValueError):
                return False, "invalid-top-p"
    return True, None


def _codex_cache_eligibility(params: dict[str, Any]) -> tuple[bool, str]:
    unknown_keys = sorted(str(key) for key in params if str(key) not in _CODEX_SAFE_TURN_PARAM_KEYS)
    if unknown_keys:
        return False, "unknown-param-shape"
    if _contains_action_hint(params):
        return False, "action-like-params"
    if not _is_text_only_input(params.get("input")):
        return False, "non-text-input"
    deterministic, reason = _deterministic_sampling(params)
    if not deterministic:
        return False, reason or "non-deterministic-sampling"
    return True, "safe-text-only-turn-start"


def _codex_cache_key_for_message(msg: dict[str, Any]) -> str:
    key_msg = copy.deepcopy(msg)
    key_msg.pop("id", None)
    return cache_key_for(
        key_msg,
        "codex-app://turn/start",
        provider="codex-app",
        upstream=DEFAULT_UPSTREAM,
        namespace=os.getenv("AGENTFLOW_CODEX_APP_CACHE_NAMESPACE", os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default")),
    )


def _is_safe_codex_response_obj(value: Any, request_id: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if request_id is not None and str(value.get("id")) != str(request_id):
        return False
    if "error" in value:
        return False
    if value.get("method") is not None:
        return False
    if "result" not in value:
        return False
    return not _contains_action_hint(value.get("result"))


def _codex_cache_payload(response_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "agentflow_cache_type": "codex-app-jsonrpc-response",
        "version": 1,
        "response": response_obj,
    }


def _codex_cached_response(payload: Any, request_id: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("agentflow_cache_type") != "codex-app-jsonrpc-response":
        return None
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    replay = copy.deepcopy(response)
    if request_id is not None:
        replay["id"] = request_id
    if not _is_safe_codex_response_obj(replay, request_id):
        return None
    return json.dumps(replay, separators=(",", ":"), ensure_ascii=False)


def _public_metadata(metadata: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]] | None:
    if metadata is None:
        return None
    public: dict[str, dict[str, Any]] = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            public[key] = {k: v for k, v in value.items() if not str(k).startswith("_agentflow_")}
        else:
            public[key] = value
    return public


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

    optimized = copy.deepcopy(msg)
    optimized["params"] = routed_params
    eligible, eligible_reason = _codex_cache_eligibility(routed_params)
    if not CODEX_APP_CACHE:
        cache_meta = _codex_cache_decision(
            "skipped",
            "codex-app-cache-disabled",
            enabled=False,
            eligible=eligible,
            replayability_level="local-exact-response" if eligible else "features_only",
        )
    elif not eligible:
        cache_meta = _codex_cache_decision("skipped", eligible_reason, enabled=True, eligible=False)
    else:
        cache_key = _codex_cache_key_for_message(optimized)
        file_deps = cache_file_dependency_snapshots(routed_params)
        cached, invalidation_reason = store.get_cache_with_reason(cache_key)
        if cached is not None:
            replay_frame = _codex_cached_response(cached, msg.get("id"))
            if replay_frame is not None:
                cache_meta = _codex_cache_decision(
                    "hit",
                    "exact-match",
                    enabled=True,
                    eligible=True,
                    hit_type="exact",
                    cache_key=cache_key,
                    replayability_level="local-exact-response",
                    file_dependencies=file_deps,
                )
                cache_meta[_INTERNAL_REPLAY_FRAME_KEY] = replay_frame
            else:
                store.delete_cache(cache_key)
                cache_meta = _codex_cache_decision(
                    "miss",
                    "unsafe-cached-envelope",
                    enabled=True,
                    eligible=True,
                    cache_key=cache_key,
                    replayability_level="local-exact-response",
                    file_dependencies=file_deps,
                )
                cache_meta[_INTERNAL_CACHE_KEY] = cache_key
        else:
            cache_meta = _codex_cache_decision(
                "miss",
                invalidation_reason or "exact-miss",
                enabled=True,
                eligible=True,
                cache_key=cache_key,
                replayability_level="local-exact-response",
                file_dependencies=file_deps,
                invalidation_reason=invalidation_reason,
            )
            cache_meta[_INTERNAL_CACHE_KEY] = cache_key
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
    optimization_metadata = _public_metadata(optimization_metadata)
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


def _maybe_store_codex_cache_response(
    raw: str | bytes,
    *,
    pending_cache: dict[str, dict[str, Any]],
) -> None:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return
    else:
        text = raw
    try:
        msg = json.loads(text)
    except Exception:
        return
    if not isinstance(msg, dict):
        return
    request_id = _request_id(msg)
    if request_id is None:
        return
    pending = pending_cache.pop(request_id, None)
    if not pending:
        return
    if not _is_safe_codex_response_obj(msg, request_id):
        return
    store.set_cache(
        pending["cache_key"],
        "codex-app",
        int(pending.get("request_chars") or 0),
        _codex_cache_payload(msg),
        file_deps=pending.get("file_deps") or [],
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
    pending_cache: dict[str, dict[str, Any]] = {}

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
                        cache_meta = (optimization_metadata or {}).get("cache") or {}
                        _record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id=session_id,
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                        )
                        replay_frame = cache_meta.get(_INTERNAL_REPLAY_FRAME_KEY)
                        if isinstance(replay_frame, str):
                            _record_message(
                                replay_frame,
                                direction="server_to_client",
                                session_id=session_id,
                                request_started=request_started,
                            )
                            await websocket.send_text(replay_frame)
                            continue
                        cache_key = cache_meta.get(_INTERNAL_CACHE_KEY)
                        if isinstance(cache_key, str) and cache_meta.get("status") == "miss":
                            try:
                                cached_msg = json.loads(forwarded) if isinstance(forwarded, str) else None
                            except Exception:
                                cached_msg = None
                            request_id = _request_id(cached_msg) if isinstance(cached_msg, dict) else None
                            params = cached_msg.get("params") if isinstance(cached_msg, dict) else None
                            if request_id is not None:
                                pending_cache[request_id] = {
                                    "cache_key": cache_key,
                                    "request_chars": len(forwarded),
                                    "file_deps": cache_file_dependency_snapshots(params if isinstance(params, dict) else cached_msg),
                                }
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
                    _maybe_store_codex_cache_response(msg, pending_cache=pending_cache)
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
