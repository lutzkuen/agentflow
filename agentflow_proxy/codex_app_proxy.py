from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentflow_proxy.cache import cache_file_dependency_snapshots, cache_key_for
from agentflow_proxy.codex_app_policy import (
    CODEX_ACTION_KEY_HINTS,
    CODEX_ACTION_VALUE_HINTS,
    CODEX_MODEL_FIELDS,
    CODEX_SAFE_TURN_PARAM_KEYS,
    CODEX_TEXT_INPUT_TYPES,
    CODEX_APP_SOURCE_SURFACE,
    DEFAULT_CODEX_APP_UPSTREAM,
    codex_model_state_signal,
    codex_app_cache_enabled,
    codex_app_optimize_enabled,
    codex_app_summary_model_hint_enabled,
    codex_app_summary_model_hint_target,
)
from agentflow_proxy.crunch import TOKEN_CHARS, crunch_body, crunch_codex_turn_params
from agentflow_proxy.recommendations import (
    build_codex_turn_optimization_unit,
    build_codex_turn_outcome_feedback,
    fetch_recommendation,
    queue_codex_outcome_feedback,
)
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.router import route_model
from agentflow_proxy.store import Store, stable_json, utc_now

load_dotenv()

DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_HOST = os.getenv("AGENTFLOW_CODEX_APP_PROXY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENTFLOW_CODEX_APP_PROXY_PORT", "4013"))
DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", DEFAULT_CODEX_APP_UPSTREAM)
LOG_EVENTS = os.getenv("AGENTFLOW_CODEX_APP_LOG_EVENTS", "1") != "0"
DB_BUSY_TIMEOUT_MS = int(os.getenv("AGENTFLOW_CODEX_APP_DB_BUSY_TIMEOUT_MS", "100"))
CODEX_APP_OPTIMIZE = codex_app_optimize_enabled()
CODEX_APP_CACHE = codex_app_cache_enabled()
CODEX_APP_SUMMARY_MODEL_HINT = codex_app_summary_model_hint_enabled()
CODEX_APP_SUMMARY_MODEL_HINT_TARGET = codex_app_summary_model_hint_target()
CODEX_APP_SESSION_COST_ALERT_USD = float(os.getenv(
    "AGENTFLOW_CODEX_APP_SESSION_COST_ALERT_USD",
    os.getenv("AGENTFLOW_SESSION_COST_ALERT_USD", "5.0"),
))

store = Store(DEFAULT_DB)
if getattr(store, "backend", None) == "sqlite":
    store.conn.execute(f"pragma busy_timeout = {DB_BUSY_TIMEOUT_MS}")
app = FastAPI(title="AgentFlow Codex App-Server Proxy", version="0.1.0")

_INTERNAL_REPLAY_FRAME_KEY = "_agentflow_replay_frame"
_INTERNAL_CACHE_KEY = "_agentflow_cache_key"
_CODEX_SUMMARY_PHASE_RE = re.compile(
    r"\b(summarize|summarise|summary|recap|wrap[- ]?up|final answer|final response|status update|handoff)\b",
    re.IGNORECASE,
)
_CODEX_TERMINAL_TEXT_RE = re.compile(
    r"(^|\s)(\$|bash|sh|zsh|terminal|shell|command|run\s+`|execute\s+`|python\s+-m|npm\s+|pnpm\s+|yarn\s+|pytest\b)",
    re.IGNORECASE,
)
_CODEX_FILE_AFFECTING_TEXT_RE = re.compile(
    r"\b(apply\s+patch|edit|modify|delete|remove|rename|move|save\s+to|write\s+to|write\s+.*\bfile|create\s+(?:a\s+)?file|touch\s+|mkdir\s+|rm\s+|mv\s+|cp\s+)\b",
    re.IGNORECASE,
)
_codex_app_session_alert_windows: dict[tuple[str, str, str], int] = {}


def _json_obj(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


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


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _thread_id(params: Any) -> Optional[str]:
    if isinstance(params, dict):
        value = params.get("threadId") or params.get("thread_id")
        return str(value) if value is not None else None
    return None


def _request_id(msg: dict[str, Any]) -> Optional[str]:
    value = msg.get("id")
    return str(value) if value is not None else None


def _codex_alert_key(*, session_id: str | None, thread_id: str | None, request_id: str | None) -> tuple[str, str]:
    if thread_id:
        return str(thread_id), "thread_id"
    if session_id:
        return str(session_id), "session_id"
    if request_id:
        return f"request:{request_id}", "request_id"
    return "codex:unknown", "unknown"


def _workflow_phase_from_metadata(routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    for meta in (cache, crunch, routing):
        phase = meta.get("workflow_phase") if isinstance(meta, dict) else None
        if phase:
            return str(phase)
    return "unknown"


def _estimate_codex_turn_spend(row: Any) -> dict[str, Any]:
    cache = _json_obj(row["cache_json"])
    crunch = _json_obj(row["crunch_json"])
    routing = _json_obj(row["routing_json"])
    input_tokens = max(0, int(_as_int(row["input_text_chars"]) / TOKEN_CHARS))
    output_tokens = max(0, int(_as_int(row["response_result_chars"]) / TOKEN_CHARS))
    model = codex_app_model()
    processing_mode = codex_app_processing_mode()
    turn_cost = estimate_cost(
        model,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=processing_mode,
    )
    if turn_cost is None:
        turn_cost_value = 0.0
        cost_known = False
    else:
        turn_cost_value = float(turn_cost)
        cost_known = True
    baseline_cost = turn_cost_value
    if crunch.get("tokens_before_est") is not None:
        crunch_baseline = estimate_cost(
            str(routing.get("requested_model") or model),
            _as_int(crunch.get("tokens_before_est")),
            output_tokens,
            provider="openai",
            processing_mode=processing_mode,
        )
        if crunch_baseline is not None:
            baseline_cost = float(crunch_baseline)
    if cache.get("status") == "hit":
        return {
            "cost_usd": 0.0,
            "baseline_cost_usd": baseline_cost,
            "cache_savings_usd": baseline_cost,
            "crunch_savings_usd": 0.0,
            "cost_known": cost_known,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": True,
            "crunched": bool(crunch.get("changed") or crunch.get("applied")),
            "workflow_phase": _workflow_phase_from_metadata(routing, crunch, cache),
        }
    return {
        "cost_usd": turn_cost_value,
        "baseline_cost_usd": baseline_cost,
        "cache_savings_usd": 0.0,
        "crunch_savings_usd": max(baseline_cost - turn_cost_value, 0.0),
        "cost_known": cost_known,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit": False,
        "crunched": bool(crunch.get("changed") or crunch.get("applied")),
        "workflow_phase": _workflow_phase_from_metadata(routing, crunch, cache),
    }


def _check_codex_app_session_cost_alert(window: dict[str, Any]) -> None:
    threshold = float(CODEX_APP_SESSION_COST_ALERT_USD)
    if threshold <= 0:
        return
    session_key, basis = _codex_alert_key(
        session_id=str(window.get("session_id") or "") or None,
        thread_id=str(window.get("thread_id") or "") or None,
        request_id=str(window.get("request_id") or "") or None,
    )
    if basis == "unknown":
        return
    if basis == "thread_id":
        where = "s.thread_id = ?"
    elif basis == "session_id":
        where = "s.thread_id is null and s.session_id = ?"
    else:
        where = "s.thread_id is null and s.session_id is null and s.request_id = ?"
    try:
        rows = store.conn.execute(f"""
            select s.request_id,
                   s.input_text_chars,
                   s.routing_json,
                   s.crunch_json,
                   s.cache_json,
                   (
                       select r.result_chars from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_result_chars
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
              and date(s.created_at) = date('now')
              and {where}
        """, (session_key if basis != "request_id" else session_key.removeprefix("request:"),)).fetchall()
    except Exception as exc:
        print(f"AgentFlow Codex app spend alert skipped: {exc}", file=sys.stderr)
        return
    turns = 0
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    cache_savings_usd = 0.0
    crunch_savings_usd = 0.0
    cache_hits = 0
    crunched_turns = 0
    phase_counts: dict[str, int] = {}
    cost_known = True
    for row in rows:
        turns += 1
        features = _estimate_codex_turn_spend(row)
        cost_usd += float(features["cost_usd"])
        input_tokens += int(features["input_tokens"])
        output_tokens += int(features["output_tokens"])
        cache_savings_usd += float(features["cache_savings_usd"])
        crunch_savings_usd += float(features["crunch_savings_usd"])
        cache_hits += 1 if features["cache_hit"] else 0
        crunched_turns += 1 if features["crunched"] else 0
        cost_known = cost_known and bool(features["cost_known"])
        phase = str(features["workflow_phase"] or "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    if not turns or cost_usd < threshold:
        return
    window_index = int(cost_usd / threshold)
    today = utc_now()[:10]
    state_key = (today, basis, session_key)
    previous_window = int(_codex_app_session_alert_windows.get(state_key) or 0)
    if window_index <= previous_window:
        return
    _codex_app_session_alert_windows[state_key] = window_index
    phase_mix = ",".join(f"{phase}:{count}" for phase, count in sorted(phase_counts.items())) or "unknown:0"
    logging.warning(
        "Codex app %s %s daily estimated cost $%.2f (%d turns, input_tokens_est=%d, "
        "output_tokens_est=%d, phases=%s, cache_hits=%d, crunched_turns=%d, "
        "cache_savings_usd=%.4f, crunch_savings_usd=%.4f, cost_known=%s) "
        "exceeds alert threshold $%.2f window=%d",
        basis,
        session_key[:8],
        cost_usd,
        turns,
        input_tokens,
        output_tokens,
        phase_mix,
        cache_hits,
        crunched_turns,
        cache_savings_usd,
        crunch_savings_usd,
        str(cost_known).lower(),
        threshold,
        window_index,
    )


def _policy_decision(kind: str, status: str, reason: str, *, enabled: bool = CODEX_APP_OPTIMIZE) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "status": status,
        "reason": reason,
        "policy_source": "local-default",
        "surface": CODEX_APP_SOURCE_SURFACE,
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
    workflow_phase: str | None = None,
    workflow_phase_reason: str | None = None,
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
    if workflow_phase:
        meta["workflow_phase"] = workflow_phase
    if workflow_phase_reason:
        meta["workflow_phase_reason"] = workflow_phase_reason
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
            if key_l in CODEX_ACTION_KEY_HINTS:
                return True
            if key_l == "type" and isinstance(nested, str) and nested.strip().lower() in CODEX_ACTION_VALUE_HINTS:
                return True
            if isinstance(nested, (dict, list)) and _contains_action_hint(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_action_hint(item) for item in value)
    return False


def _model_field(params: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in CODEX_MODEL_FIELDS:
        value = params.get(field)
        if value is not None:
            return field, str(value)
    return None, None


def _codex_route_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_field, requested_model = _model_field(params)
    if not requested_model:
        meta = _policy_decision(
            "routing",
            "skipped" if CODEX_APP_SUMMARY_MODEL_HINT else "not-applicable",
            "codex-turn-start-model-field-absent",
        )
        meta.update({
            "canary": "codex-app-summary-model-hint",
            "canary_enabled": bool(CODEX_APP_SUMMARY_MODEL_HINT),
        })
        return params, meta

    if CODEX_APP_SUMMARY_MODEL_HINT:
        target_model = str(CODEX_APP_SUMMARY_MODEL_HINT_TARGET or "").strip()
        base_meta = _policy_decision("routing", "skipped", "summary-model-hint-not-applied")
        base_meta.update({
            "canary": "codex-app-summary-model-hint",
            "canary_enabled": True,
            "model_field": model_field,
            "requested_model": requested_model,
            "routed_model": requested_model,
            "target_model": target_model,
        })
        if not target_model:
            base_meta["reason"] = "summary-model-hint-target-absent"
            return params, base_meta
        if _contains_action_hint(params):
            base_meta["reason"] = "action-like-params"
            return params, base_meta
        eligible, reason, eligibility_meta = _codex_cache_eligibility(params)
        base_meta.update({
            "workflow_phase": eligibility_meta.get("workflow_phase") or "unknown",
            "workflow_phase_reason": eligibility_meta.get("workflow_phase_reason"),
        })
        if eligibility_meta.get("unknown_keys"):
            base_meta["unknown_keys"] = eligibility_meta["unknown_keys"]
        if not eligible:
            base_meta["reason"] = reason
            return params, base_meta
        if target_model == requested_model:
            base_meta["reason"] = "summary-model-hint-target-matches-requested"
            return params, base_meta
        routed_params = copy.deepcopy(params)
        routed_params[model_field] = target_model
        base_meta.update({
            "status": "applied",
            "reason": "safe-summary-model-hint-canary",
            "routed_model": target_model,
            "applied": True,
            "policy_id": "local-codex-app-summary-model-hint-canary",
            "hint_type": "model",
            "safety_gates": {
                "known_model_field": True,
                "safe_param_shape": True,
                "text_only_input": True,
                "action_like_params": False,
                "workflow_phase": "summary",
            },
        })
        return routed_params, base_meta

    route_body = copy.deepcopy(params)
    route_body["model"] = requested_model
    routed_model, routing_meta = route_model(route_body)
    routing_meta = dict(routing_meta)
    routing_meta.update({
        "surface": CODEX_APP_SOURCE_SURFACE,
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
            "surface": CODEX_APP_SOURCE_SURFACE,
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
        "surface": CODEX_APP_SOURCE_SURFACE,
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
        if block_type not in CODEX_TEXT_INPUT_TYPES:
            return False
        text_value = value.get("text", value.get("input_text", value.get("value")))
        return isinstance(text_value, str)
    return False


def _codex_input_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_codex_input_texts(item))
        return texts
    if isinstance(value, dict):
        block_type = str(value.get("type") or "text").strip().lower()
        if block_type not in CODEX_TEXT_INPUT_TYPES:
            return []
        text_value = value.get("text", value.get("input_text", value.get("value")))
        return [text_value] if isinstance(text_value, str) else []
    return []


def _codex_summary_phase_reason(params: dict[str, Any]) -> str | None:
    text = "\n".join(_codex_input_texts(params.get("input"))).strip()
    if not text:
        return None
    if _CODEX_SUMMARY_PHASE_RE.search(text):
        return "summary-text-intent"
    return None


def _codex_text_safety_skip_reason(params: dict[str, Any]) -> str | None:
    text = "\n".join(_codex_input_texts(params.get("input")))
    if _CODEX_TERMINAL_TEXT_RE.search(text):
        return "terminal-interaction-text"
    if _CODEX_FILE_AFFECTING_TEXT_RE.search(text):
        return "file-affecting-text"
    return None


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


def _codex_cache_eligibility(params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    unknown_keys = sorted(str(key) for key in params if str(key) not in CODEX_SAFE_TURN_PARAM_KEYS)
    if unknown_keys:
        return False, "unknown-param-shape", {"unknown_keys": unknown_keys[:8]}
    model_field, requested_model = _model_field(params)
    if not model_field or not requested_model:
        return False, "model-field-unknown", {"workflow_phase": "unknown"}
    if _contains_action_hint(params):
        return False, "action-like-params", {"workflow_phase": "unknown"}
    if not _is_text_only_input(params.get("input")):
        return False, "non-text-input", {"workflow_phase": "unknown"}
    text_skip_reason = _codex_text_safety_skip_reason(params)
    if text_skip_reason:
        return False, text_skip_reason, {"workflow_phase": "unknown"}
    phase_reason = _codex_summary_phase_reason(params)
    if not phase_reason:
        return False, "workflow-phase-not-summary", {"workflow_phase": "unknown"}
    deterministic, reason = _deterministic_sampling(params)
    if not deterministic:
        return False, reason or "non-deterministic-sampling", {
            "workflow_phase": "summary",
            "workflow_phase_reason": phase_reason,
            "model_field": model_field,
        }
    return True, "safe-summary-text-only-turn-start", {
        "workflow_phase": "summary",
        "workflow_phase_reason": phase_reason,
        "model_field": model_field,
    }


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


def _jsonrpc_message(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _event_window_key(session_id: str, thread_id: str | None, request_id: str | None) -> str | None:
    if thread_id:
        return f"{session_id}\0thread\0{thread_id}"
    if request_id:
        return f"{session_id}\0request\0{request_id}"
    return None


def _model_field_state_from_metadata(optimization_metadata: dict[str, dict[str, Any]] | None) -> tuple[str, str | None]:
    routing = (optimization_metadata or {}).get("routing") or {}
    field = routing.get("model_field")
    if field:
        return "present", str(field)
    reason = str(routing.get("reason") or "")
    if reason == "codex-turn-start-model-field-absent":
        return "absent", None
    if routing.get("requested_model") or routing.get("routed_model"):
        return "present_unknown_field", None
    return "unknown", None


def _public_model_state(signal: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(signal, dict):
        return None
    state = str(signal.get("state") or "")
    if state not in {"derived_present", "derived_absent"}:
        return None
    summary = {
        "state": state,
        "field": signal.get("field"),
        "source_method": signal.get("source_method"),
        "confidence": signal.get("confidence") or "medium",
        "reason": signal.get("reason") or "metadata-model-state",
    }
    if state == "derived_present" and signal.get("normalized_model"):
        summary["normalized_model"] = signal.get("normalized_model")
    return summary


def _model_state_scope_keys(session_id: str, thread_id: str | None) -> list[str]:
    keys = []
    if thread_id:
        keys.append(f"{session_id}\0thread-model\0{thread_id}")
    keys.append(f"{session_id}\0session-model")
    return keys


def _remember_model_state(
    model_states: dict[str, dict[str, Any]] | None,
    *,
    session_id: str,
    thread_id: str | None,
    signal: dict[str, Any] | None,
) -> None:
    public = _public_model_state(signal)
    if model_states is None or not public:
        return
    keys = _model_state_scope_keys(session_id, thread_id)
    for key in keys[:1 if thread_id else 1]:
        model_states[key] = dict(public)


def _lookup_model_state(
    model_states: dict[str, dict[str, Any]] | None,
    *,
    session_id: str,
    thread_id: str | None,
) -> dict[str, Any] | None:
    if model_states is None:
        return None
    for key in _model_state_scope_keys(session_id, thread_id):
        signal = model_states.get(key)
        if signal:
            return dict(signal)
    return None


def _apply_derived_model_state(window: dict[str, Any], signal: dict[str, Any] | None) -> None:
    public = _public_model_state(signal)
    if not public:
        return
    current = str(window.get("model_field_state") or "unknown")
    if current not in {"unknown", "absent", "derived_absent"} and public["state"] != "derived_present":
        return
    if current in {"present", "present_unknown_field"}:
        return
    window["model_field_state"] = public["state"]
    window["model_field"] = public.get("field")
    window["model_state"] = public


def _update_method_count(target: dict[str, int], method: str | None) -> None:
    key = str(method or "response")
    target[key] = int(target.get(key) or 0) + 1


def _new_event_window(
    *,
    start_event_id: str,
    monotonic_started_at: float,
    created_at: str,
    session_id: str,
    request_id: str | None,
    thread_id: str | None,
    method: str | None,
    message_chars: int,
    params_chars: int | None,
    input_items: int | None,
    input_text_chars: int,
    optimization_metadata: dict[str, dict[str, Any]] | None,
    derived_model_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_state, model_field = _model_field_state_from_metadata(optimization_metadata)
    window = {
        "schema": "agentflow.codex_app_event_window.v1",
        "start_event_id": start_event_id,
        "created_at": created_at,
        "session_id": session_id,
        "request_id": request_id,
        "thread_id": thread_id,
        "event_count": 1,
        "method_counts": {},
        "direction_counts": {"client_to_server": 1},
        "first_event_delta_ms": 0,
        "last_event_delta_ms": 0,
        "input_items": input_items,
        "input_text_chars": input_text_chars,
        "start_message_chars": message_chars,
        "start_params_chars": params_chars,
        "result_chars": 0,
        "server_message_chars": 0,
        "error_count": 0,
        "model_field_state": model_state,
        "model_field": model_field,
        "_monotonic_started_at": monotonic_started_at,
    }
    _apply_derived_model_state(window, derived_model_state)
    _update_method_count(window["method_counts"], method)
    return window


def _persist_event_window(start_event_id: str, window: dict[str, Any]) -> None:
    public = {key: value for key, value in window.items() if not key.startswith("_")}
    try:
        store.update_codex_app_event_window_json(start_event_id, stable_json(public))
    except AttributeError:
        return
    except Exception as exc:
        print(f"AgentFlow Codex app event-window metadata update skipped: {exc}", file=sys.stderr)


def _record_event_window(
    active_turn_windows: dict[str, dict[str, Any]] | None,
    *,
    event_id: str,
    created_at: str,
    direction: str,
    session_id: str,
    request_id: str | None,
    thread_id: str | None,
    method: str | None,
    message_chars: int,
    params_chars: int | None,
    input_items: int | None,
    input_text_chars: int,
    result_chars: int | None,
    error_code: Any,
    optimization_metadata: dict[str, dict[str, Any]] | None,
    model_state_signal: dict[str, Any] | None = None,
    model_states: dict[str, dict[str, Any]] | None = None,
) -> None:
    if active_turn_windows is None:
        return
    key = _event_window_key(session_id, thread_id, request_id)
    if not key:
        return
    if direction == "client_to_server" and method == "turn/start":
        derived_model_state = _lookup_model_state(
            model_states,
            session_id=session_id,
            thread_id=thread_id,
        )
        window = _new_event_window(
            start_event_id=event_id,
            monotonic_started_at=time.time(),
            created_at=created_at,
            session_id=session_id,
            request_id=request_id,
            thread_id=thread_id,
            method=method,
            message_chars=message_chars,
            params_chars=params_chars,
            input_items=input_items,
            input_text_chars=input_text_chars,
            optimization_metadata=optimization_metadata,
            derived_model_state=derived_model_state,
        )
        active_turn_windows[key] = window
        _persist_event_window(event_id, window)
        return

    window = active_turn_windows.get(key)
    if not window and direction == "server_to_client" and request_id is not None:
        for active_key, candidate in active_turn_windows.items():
            if (
                str(candidate.get("session_id") or "") == str(session_id or "")
                and str(candidate.get("request_id") or "") == str(request_id)
            ):
                key = active_key
                window = candidate
                break
    if not window:
        return
    window["event_count"] = int(window.get("event_count") or 0) + 1
    direction_counts = window.setdefault("direction_counts", {})
    direction_counts[direction] = int(direction_counts.get(direction) or 0) + 1
    method_counts = window.setdefault("method_counts", {})
    _update_method_count(method_counts, method)
    if direction == "server_to_client":
        window["server_message_chars"] = int(window.get("server_message_chars") or 0) + int(message_chars or 0)
    if result_chars is not None:
        window["result_chars"] = int(window.get("result_chars") or 0) + int(result_chars or 0)
    if error_code is not None:
        window["error_count"] = int(window.get("error_count") or 0) + 1
    _apply_derived_model_state(window, model_state_signal)
    started_at = float(window.get("_monotonic_started_at") or time.time())
    window["last_event_delta_ms"] = max(0, int((time.time() - started_at) * 1000))
    start_event_id = str(window.get("start_event_id") or "")
    if start_event_id:
        _persist_event_window(start_event_id, window)
    if direction == "server_to_client" and (result_chars is not None or error_code is not None):
        _check_codex_app_session_cost_alert(window)
    if direction == "server_to_client" and method in {"turn/completed", "thread/closed"}:
        active_turn_windows.pop(key, None)


async def _attach_codex_managed_recommendation(
    raw: str | bytes,
    optimization_metadata: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict) or msg.get("method") != "turn/start":
        return None
    params = msg.get("params")
    if not isinstance(params, dict):
        return None
    metadata = optimization_metadata or _not_applied_metadata("missing-optimization-metadata")
    routing = metadata.setdefault("routing", _policy_decision("routing", "not-applied", "missing-routing-metadata"))
    crunch = metadata.setdefault("crunch", _policy_decision("crunch", "not-applied", "missing-crunch-metadata"))
    cache = metadata.setdefault("cache", _codex_cache_decision("skipped", "missing-cache-metadata", eligible=False))
    request_id = _request_id(msg)
    input_value = params.get("input")
    input_text_chars = _input_text_chars(input_value)
    input_items = len(input_value) if isinstance(input_value, list) else None
    unit = build_codex_turn_optimization_unit(
        method="turn/start",
        request_id_present=request_id is not None,
        thread_id_present=_thread_id(params) is not None,
        params_chars=len(stable_json(params)),
        input_items=input_items,
        input_text_chars=input_text_chars if input_text_chars else None,
        routing_meta=routing,
        crunch_meta=crunch,
        cache_meta=cache,
    )
    managed = await fetch_recommendation(unit)
    managed.setdefault("applied", False)
    if managed.get("status") == "received":
        managed["applied"] = False
        managed["apply_reason"] = "codex-app-managed-recommendation-observed-only"
    managed["source_surface"] = "codex_turn"
    routing["managed_recommendation"] = managed
    return {
        "request_id": request_id,
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
        "managed": managed,
        "input_text_chars": input_text_chars if input_text_chars else None,
    }


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
    eligible, eligible_reason, eligibility_meta = _codex_cache_eligibility(routed_params)
    workflow_phase = eligibility_meta.get("workflow_phase")
    workflow_phase_reason = eligibility_meta.get("workflow_phase_reason")
    if not CODEX_APP_CACHE:
        cache_meta = _codex_cache_decision(
            "skipped",
            "codex-app-cache-disabled",
            enabled=False,
            eligible=eligible,
            replayability_level="local-exact-response" if eligible else "features_only",
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
        )
    elif not eligible:
        cache_meta = _codex_cache_decision(
            "skipped",
            eligible_reason,
            enabled=True,
            eligible=False,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
        )
        if eligibility_meta.get("unknown_keys"):
            cache_meta["unknown_keys"] = eligibility_meta["unknown_keys"]
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
                    workflow_phase=workflow_phase,
                    workflow_phase_reason=workflow_phase_reason,
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
                    workflow_phase=workflow_phase,
                    workflow_phase_reason=workflow_phase_reason,
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
                workflow_phase=workflow_phase,
                workflow_phase_reason=workflow_phase_reason,
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
    active_turn_windows: dict[str, dict[str, Any]] | None = None,
    model_states: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    if not LOG_EVENTS:
        return None
    optimization_metadata = _public_metadata(optimization_metadata)
    message_chars = len(raw)
    event_id = str(uuid.uuid4())
    created_at = utc_now()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            _log_codex_app_event(
                id=event_id,
                created_at=created_at,
                direction=direction,
                message_chars=message_chars,
                session_id=session_id,
                routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
                crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
                cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
            )
            return event_id
    else:
        text = raw
    try:
        msg = json.loads(text)
    except Exception:
        _log_codex_app_event(
            id=event_id,
            created_at=created_at,
            direction=direction,
            message_chars=message_chars,
            session_id=session_id,
            routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
            crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
            cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
        )
        return event_id

    if not isinstance(msg, dict):
        _log_codex_app_event(
            id=event_id,
            created_at=created_at,
            direction=direction,
            message_chars=message_chars,
            session_id=session_id,
            routing_json=stable_json(optimization_metadata["routing"]) if optimization_metadata else None,
            crunch_json=stable_json(optimization_metadata["crunch"]) if optimization_metadata else None,
            cache_json=stable_json(optimization_metadata["cache"]) if optimization_metadata else None,
        )
        return event_id

    params = msg.get("params")
    method = msg.get("method")
    rid = _request_id(msg)
    params_chars = len(stable_json(params)) if params is not None else None
    input_value = params.get("input") if isinstance(params, dict) else None
    input_items = len(input_value) if isinstance(input_value, list) else None
    input_text_chars = _input_text_chars(input_value)
    thread_id = _thread_id(params)
    model_state_signal = codex_model_state_signal(method, params)
    _remember_model_state(
        model_states,
        session_id=session_id,
        thread_id=thread_id,
        signal=model_state_signal,
    )
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
        id=event_id,
        created_at=created_at,
        direction=direction,
        method=str(method) if method is not None else None,
        request_id=rid,
        thread_id=thread_id,
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
    _record_event_window(
        active_turn_windows,
        event_id=event_id,
        created_at=created_at,
        direction=direction,
        session_id=session_id,
        request_id=rid,
        thread_id=thread_id,
        method=str(method) if method is not None else None,
        message_chars=message_chars,
        params_chars=params_chars,
        input_items=input_items,
        input_text_chars=input_text_chars,
        result_chars=len(stable_json(result)) if result is not None else None,
        error_code=error.get("code") if isinstance(error, dict) else None,
        optimization_metadata=optimization_metadata,
        model_state_signal=model_state_signal,
        model_states=model_states,
    )
    return event_id


async def _record_codex_managed_outcome(
    raw: str | bytes,
    *,
    session_id: str,
    request_started: dict[str, float],
    pending_managed: dict[str, dict[str, Any]],
) -> None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict):
        return
    request_id = _request_id(msg)
    if request_id is None:
        return
    pending = pending_managed.pop(request_id, None)
    if not pending:
        return
    result = msg.get("result")
    error = msg.get("error")
    latency_ms: int | None = None
    started = request_started.get(request_id)
    if started is not None:
        latency_ms = int((time.time() - started) * 1000)
    result_chars = len(stable_json(result)) if result is not None else None
    error_code = error.get("code") if isinstance(error, dict) else None
    error_message = error.get("message") if isinstance(error, dict) and isinstance(error.get("message"), str) else None
    managed = pending["managed"]
    outcome = build_codex_turn_outcome_feedback(
        recommendation_meta=managed,
        routing_meta=pending["routing"],
        crunch_meta=pending["crunch"],
        cache_meta=pending["cache"],
        result_chars=result_chars,
        error_code=error_code,
        error_message=error_message,
        latency_ms=latency_ms,
        input_text_chars=pending.get("input_text_chars"),
        session_id=session_id,
    )
    managed["outcome_feedback"] = await queue_codex_outcome_feedback(store, managed, outcome)
    pending["routing"]["managed_recommendation"] = managed
    start_event_id = pending.get("start_event_id")
    if isinstance(start_event_id, str):
        try:
            store.update_codex_app_event_routing_json(start_event_id, stable_json(pending["routing"]))
        except Exception as exc:
            print(f"AgentFlow Codex app managed feedback metadata update skipped: {exc}", file=sys.stderr)


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
    pending_managed: dict[str, dict[str, Any]] = {}
    active_turn_windows: dict[str, dict[str, Any]] = {}
    model_states: dict[str, dict[str, Any]] = {}

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
                        managed_pending = await _attach_codex_managed_recommendation(forwarded, optimization_metadata)
                        cache_meta = (optimization_metadata or {}).get("cache") or {}
                        start_event_id = _record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id=session_id,
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                            active_turn_windows=active_turn_windows,
                            model_states=model_states,
                        )
                        if managed_pending and isinstance(managed_pending.get("request_id"), str):
                            managed_pending["start_event_id"] = start_event_id
                            pending_managed[str(managed_pending["request_id"])] = managed_pending
                        replay_frame = cache_meta.get(_INTERNAL_REPLAY_FRAME_KEY)
                        if isinstance(replay_frame, str):
                            await _record_codex_managed_outcome(
                                replay_frame,
                                session_id=session_id,
                                request_started=request_started,
                                pending_managed=pending_managed,
                            )
                            _record_message(
                                replay_frame,
                                direction="server_to_client",
                                session_id=session_id,
                                request_started=request_started,
                                active_turn_windows=active_turn_windows,
                                model_states=model_states,
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
                        managed_pending = await _attach_codex_managed_recommendation(forwarded, optimization_metadata)
                        start_event_id = _record_message(
                            forwarded,
                            direction="client_to_server",
                            session_id=session_id,
                            request_started=request_started,
                            optimization_metadata=optimization_metadata,
                            active_turn_windows=active_turn_windows,
                            model_states=model_states,
                        )
                        if managed_pending and isinstance(managed_pending.get("request_id"), str):
                            managed_pending["start_event_id"] = start_event_id
                            pending_managed[str(managed_pending["request_id"])] = managed_pending
                        await upstream.send(forwarded)

            async def upstream_to_client() -> None:
                async for msg in upstream:
                    _maybe_store_codex_cache_response(msg, pending_cache=pending_cache)
                    await _record_codex_managed_outcome(
                        msg,
                        session_id=session_id,
                        request_started=request_started,
                        pending_managed=pending_managed,
                    )
                    _record_message(
                        msg,
                        direction="server_to_client",
                        session_id=session_id,
                        request_started=request_started,
                        active_turn_windows=active_turn_windows,
                        model_states=model_states,
                    )
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
