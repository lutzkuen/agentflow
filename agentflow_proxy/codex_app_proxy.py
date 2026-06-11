from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentflow_proxy.cache import cache_file_dependency_audit, cache_file_dependency_snapshots, cache_key_for
from agentflow_proxy.codex_app_policy import (
    CODEX_ACTION_KEY_HINTS,
    CODEX_ACTION_VALUE_HINTS,
    CODEX_MODEL_FIELDS,
    CODEX_APP_POLICY,
    CODEX_APP_POLICY_ACTION_FAMILIES,
    CODEX_APP_POLICY_ACTION_KEYS,
    CODEX_APP_POLICY_CONDITION_KEYS,
    CODEX_SAFE_TURN_PARAM_KEYS,
    CODEX_TEXT_INPUT_TYPES,
    CODEX_APP_SOURCE_SURFACE,
    CODEX_APP_POLICY_SOURCE,
    DEFAULT_CODEX_APP_UPSTREAM,
    codex_model_state_signal,
    codex_app_cache_canary,
    codex_app_cache_enabled,
    codex_app_cache_namespace,
    codex_app_cache_ttl_seconds,
    codex_app_optimize_enabled,
    codex_app_summary_model_hint_enabled,
    codex_app_summary_model_hint_canary,
    codex_app_summary_model_hint_target,
)
from agentflow_proxy.crunch import TOKEN_CHARS, crunch_body, crunch_codex_turn_params
from agentflow_proxy.recommendations import (
    build_codex_app_canary_lifecycle_feedback,
    build_codex_turn_optimization_unit,
    build_codex_turn_outcome_feedback,
    fetch_recommendation,
    pattern_feature_diagnostics,
    queue_codex_app_canary_lifecycle_feedback,
    queue_codex_outcome_feedback,
    queue_policy_event_feedback,
)
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.routing_experiments import (
    ROUTING_EXPERIMENT_OUTCOME_SOURCE_SURFACE,
    ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES,
    compare_response_outputs,
    response_output_text,
    routing_experiment_decision,
    routing_experiment_feedback_features,
    routing_experiment_outcome_event,
)
from agentflow_proxy.prompt_features import prompt_difficulty_features_from_text
from agentflow_proxy.router import route_model
from agentflow_proxy.store import Store, stable_json, utc_now
from agentflow_proxy.terminal_features import terminal_log_features_from_text

load_dotenv()

DEFAULT_DB = os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3"))
DEFAULT_HOST = os.getenv("AGENTFLOW_CODEX_APP_PROXY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENTFLOW_CODEX_APP_PROXY_PORT", "4013"))
DEFAULT_UPSTREAM = os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", DEFAULT_CODEX_APP_UPSTREAM)
LOG_EVENTS = os.getenv("AGENTFLOW_CODEX_APP_LOG_EVENTS", "1") != "0"
DB_BUSY_TIMEOUT_MS = int(os.getenv("AGENTFLOW_CODEX_APP_DB_BUSY_TIMEOUT_MS", "100"))
CODEX_APP_OPTIMIZE = codex_app_optimize_enabled()
CODEX_APP_CACHE = codex_app_cache_enabled()
CODEX_APP_CACHE_TTL_SECONDS = codex_app_cache_ttl_seconds()
CODEX_APP_CACHE_CANARY = codex_app_cache_canary()
CODEX_APP_SUMMARY_MODEL_HINT = codex_app_summary_model_hint_enabled()
CODEX_APP_SUMMARY_MODEL_HINT_TARGET = codex_app_summary_model_hint_target()
CODEX_APP_SUMMARY_MODEL_HINT_CANARY = codex_app_summary_model_hint_canary()
CODEX_APP_WEBSOCKET_MAX_SIZE = int(os.getenv("AGENTFLOW_CODEX_APP_WEBSOCKET_MAX_SIZE", str(64 * 1024 * 1024)))
CODEX_APP_RULES = [
    rule for rule in (CODEX_APP_POLICY.get("rules") or [])
    if isinstance(rule, dict)
]
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
_CODEX_STALE_RISK_TEXT_RE = re.compile(
    r"\b(today|latest|current|recent|now|this\s+(?:run|session|turn|state)|up[- ]?to[- ]?date)\b",
    re.IGNORECASE,
)
_CODEX_PATH_LIKE_TEXT_RE = re.compile(r"(^|\s)(/|\.{1,2}/|~/|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
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


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _known_enum(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        return None
    return normalized if normalized in allowed else "other"


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _count_bucket(value: Any) -> str | None:
    number = _as_float(value)
    if number is None:
        return None
    if number <= 0:
        return "0"
    if number < 10:
        return "1_9"
    if number < 100:
        return "10_99"
    if number < 1000:
        return "100_999"
    if number < 10000:
        return "1k_10k"
    return "10k_plus"


def _percent_bucket(value: Any) -> str | None:
    number = _as_float(value)
    if number is None:
        return None
    if number < 50:
        return "lt_50"
    if number < 75:
        return "50_75"
    if number < 90:
        return "75_90"
    if number < 100:
        return "90_99"
    return "100_plus"


def _seconds_bucket(seconds: Any) -> str | None:
    number = _as_float(seconds)
    if number is None:
        return None
    if number <= 0:
        return "now"
    if number < 60:
        return "lt_1m"
    if number < 3600:
        return "1m_1h"
    if number < 21600:
        return "1h_6h"
    if number < 86400:
        return "6h_24h"
    return "1d_plus"


def _reset_seconds(value: Any) -> float | None:
    direct = _as_float(value)
    if direct is not None:
        return direct
    if not isinstance(value, str):
        return None
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _token_usage_metadata(method: str | None, params: Any) -> dict[str, Any] | None:
    if method != "thread/tokenUsage/updated" or not isinstance(params, dict):
        return None
    mapping = {
        "input_tokens": ("inputTokens", "input_tokens", "input"),
        "cached_input_tokens": (
            "cachedInputTokens",
            "cached_input_tokens",
            "cacheReadInputTokens",
            "cache_read_input_tokens",
            "cached",
        ),
        "output_tokens": ("outputTokens", "output_tokens", "output"),
        "reasoning_output_tokens": (
            "reasoningOutputTokens",
            "reasoning_output_tokens",
            "reasoningTokens",
            "reasoning_tokens",
            "reasoning",
        ),
        "total_tokens": ("totalTokens", "total_tokens", "total"),
    }
    raw_key_set = {raw_key for raw_keys in mapping.values() for raw_key in raw_keys}

    def parse_candidate(info: dict[str, Any]) -> dict[str, int]:
        usage: dict[str, int] = {}
        for public_key, raw_keys in mapping.items():
            for raw_key in raw_keys:
                if raw_key in info:
                    parsed = _as_float(info.get(raw_key))
                    if parsed is not None and parsed >= 0:
                        usage[public_key] = int(parsed)
                    break
        if usage and "total_tokens" not in usage:
            usage["total_tokens"] = sum(
                usage.get(key, 0)
                for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
            )
        return usage

    candidates: list[dict[str, int]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if raw_key_set.intersection(value.keys()):
                parsed = parse_candidate(value)
                if parsed:
                    candidates.append(parsed)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    walk(nested)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    walk(item)

    preferred = params.get("usage") if isinstance(params.get("usage"), dict) else None
    if preferred:
        parsed_preferred = parse_candidate(preferred)
        if parsed_preferred:
            candidates.append(parsed_preferred)
    walk(params)
    usage = max(candidates, key=lambda row: row.get("total_tokens", 0), default={})
    if not usage:
        return None
    return {
        "schema": "agentflow.codex_app_metadata.v1",
        "kind": "token_usage",
        "method": method,
        "thread_id_present": bool(_thread_id(params)),
        "token_usage": {
            **usage,
            "total_tokens_bucket": _count_bucket(usage.get("total_tokens")),
        },
        "privacy": {
            "metadata_only": True,
            "raw_params_included": False,
            "raw_prompts_included": False,
            "raw_commands_included": False,
            "raw_transcripts_included": False,
            "arbitrary_strings_included": False,
        },
    }


def _rate_limit_metadata(method: str | None, params: Any) -> dict[str, Any] | None:
    if method != "account/rateLimits/updated" or not isinstance(params, dict):
        return None
    limits = params.get("rateLimits") or params.get("rate_limits") or params.get("limits")
    if not isinstance(limits, dict):
        return None
    allowed_plans = {"free", "plus", "pro", "team", "business", "enterprise", "chatgpt_plus", "chatgpt_pro", "unknown"}
    plan = _known_enum(_first_present(limits, "planType", "plan_type", "plan"), allowed_plans)
    scopes: list[dict[str, Any]] = []
    for name in ("primary", "secondary", "daily", "weekly", "monthly", "hourly"):
        item = limits.get(name)
        if not isinstance(item, dict):
            continue
        used_percent = _as_float(_first_present(item, "usedPercent", "used_percent"))
        remaining = _as_float(_first_present(item, "remaining", "remainingRequests", "remaining_requests"))
        reset_seconds = _reset_seconds(_first_present(
            item,
            "resetAfterSeconds",
            "reset_after_seconds",
            "resetInSeconds",
            "reset_in_seconds",
        ))
        if reset_seconds is None:
            reset_seconds = _reset_seconds(_first_present(item, "resetAt", "reset_at"))
        scope: dict[str, Any] = {
            "name": name,
            "used_percent": round(used_percent, 3) if used_percent is not None else None,
            "used_percent_bucket": _percent_bucket(used_percent),
            "remaining_bucket": _count_bucket(remaining),
            "reset_bucket": _seconds_bucket(reset_seconds),
        }
        if remaining is not None:
            scope["remaining"] = int(max(0, remaining))
        scopes.append({key: value for key, value in scope.items() if value is not None})
    if not scopes and plan is None:
        return None
    max_used = max((_as_float(scope.get("used_percent")) or 0 for scope in scopes), default=0)
    pressure = "critical" if max_used >= 100 else "high" if max_used >= 90 else "elevated" if max_used >= 75 else "normal"
    return {
        "schema": "agentflow.codex_app_metadata.v1",
        "kind": "rate_limits",
        "method": method,
        "rate_limits": {
            "plan_type": plan,
            "pressure": pressure,
            "scopes": scopes,
        },
        "privacy": {
            "metadata_only": True,
            "raw_params_included": False,
            "raw_prompts_included": False,
            "raw_commands_included": False,
            "raw_transcripts_included": False,
            "arbitrary_strings_included": False,
        },
    }


def _codex_app_signal_metadata(method: str | None, params: Any) -> dict[str, Any] | None:
    return _token_usage_metadata(method, params) or _rate_limit_metadata(method, params)


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
    else:
        basis = "workflow_window"
        session_key = f"codex-workflow:{utc_now()[:10]}"
        where = "s.thread_id is null"
    try:
        params = (session_key,) if basis == "thread_id" else ()
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
        """, params).fetchall()
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
        "policy_source": CODEX_APP_POLICY_SOURCE,
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
    file_dependency_audit_meta: dict[str, Any] | None = None,
    invalidation_reason: str | None = None,
    workflow_phase: str | None = None,
    workflow_phase_reason: str | None = None,
    outcome_bucket: str | None = None,
    canary_sample: dict[str, Any] | None = None,
    ttl_seconds: int | None = None,
) -> dict[str, Any]:
    meta = _policy_decision("cache", status, reason, enabled=enabled)
    meta.update({
        "eligible": bool(eligible),
        "hit_type": hit_type or "",
        "exact_enabled": bool(CODEX_APP_CACHE),
        "replayability_level": replayability_level,
        "cache_key_present": bool(cache_key),
        "cache_key_included": False,
        "outcome_bucket": outcome_bucket or _codex_cache_outcome_bucket(status, reason),
    })
    if ttl_seconds is not None:
        meta["ttl_seconds"] = max(0, int(ttl_seconds))
    if file_dependency_audit_meta is None and file_dependencies:
        file_dependency_audit_meta = cache_file_dependency_audit(snapshots=file_dependencies)
    if file_dependency_audit_meta:
        meta["file_dependency_audit"] = file_dependency_audit_meta
        meta["file_dependency_count"] = file_dependency_audit_meta.get("snapshot_count", 0)
        meta["file_dependency_evidence_available"] = bool(
            file_dependency_audit_meta.get("file_dependency_evidence_available")
        )
        meta["safe_invalidation_evidence"] = bool(file_dependency_audit_meta.get("safe_invalidation_evidence"))
    if invalidation_reason:
        meta["invalidation_reason"] = invalidation_reason
    if workflow_phase:
        meta["workflow_phase"] = workflow_phase
    if workflow_phase_reason:
        meta["workflow_phase_reason"] = workflow_phase_reason
    if canary_sample:
        meta["canary"] = "codex-app-exact-cache"
        meta["canary_cohort"] = canary_sample.get("cohort")
        meta["canary_sample"] = canary_sample
    return meta


def _codex_cache_outcome_bucket(status: str, reason: str) -> str:
    if reason == "codex-app-cache-disabled":
        return "disabled"
    if status == "hit":
        return "hit"
    if status == "holdout" or reason in {"codex-app-cache-canary-holdout", "canary_holdout"}:
        return "holdout"
    if status == "unsafe-skip" or reason in {
        "action-like-params",
        "non-text-input",
        "unknown-param-shape",
        "terminal-interaction-text",
        "file-affecting-text",
        "unsafe-cached-envelope",
    }:
        return "unsafe-skip"
    if reason in {
        "dependency-changed",
        "dependency-deleted",
        "codex-cache-ttl-expired",
    }:
        return "invalidated"
    if reason in {
        "stale-risk-blockers",
        "file-dependency-missing",
        "dependency-missing",
        "dependency-cap-exceeded",
        "file-watch-disabled",
    }:
        return "stale-risk"
    if status == "miss":
        return "miss"
    return status or "unknown"


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


def _attach_codex_local_pattern_features(
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
    thread_id = _thread_id(params)
    input_value = params.get("input")
    input_text_chars = _input_text_chars(input_value)
    input_items = len(input_value) if isinstance(input_value, list) else None
    unit = build_codex_turn_optimization_unit(
        method="turn/start",
        request_id_present=request_id is not None,
        thread_id_present=thread_id is not None,
        params_chars=len(stable_json(params)),
        input_items=input_items,
        input_text_chars=input_text_chars if input_text_chars else None,
        routing_meta=routing,
        crunch_meta=crunch,
        cache_meta=cache,
        request_id=str(request_id) if request_id is not None else None,
        thread_id=thread_id,
        terminal_log_features=routing.get("terminal_log_features") or _codex_terminal_log_features(params),
        prompt_difficulty_features=routing.get("prompt_difficulty_features") or _codex_prompt_difficulty_features(params),
    )
    routing["terminal_log_features"] = unit["input_features"].get("terminal_log_features")
    routing["prompt_difficulty_features"] = unit["input_features"].get("prompt_difficulty_features")
    routing["managed_pattern_features"] = pattern_feature_diagnostics(unit)
    return {
        "request_id": request_id,
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
        "unit": unit,
        "input_text_chars": input_text_chars if input_text_chars else None,
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


def _summary_model_hint_cost_delta(
    params: dict[str, Any],
    *,
    requested_model: str,
    target_model: str,
) -> dict[str, Any]:
    input_tokens = max(0, _input_text_chars(params.get("input")) // TOKEN_CHARS)
    requested_cost = estimate_cost(
        requested_model,
        input_tokens,
        0,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    target_cost = estimate_cost(
        target_model,
        input_tokens,
        0,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    return {
        "basis": "input-text-chars-estimated",
        "input_tokens_est": input_tokens,
        "requested_input_cost_est_usd": round(float(requested_cost), 8) if requested_cost is not None else None,
        "target_input_cost_est_usd": round(float(target_cost), 8) if target_cost is not None else None,
        "delta_usd": round(float(requested_cost) - float(target_cost), 8)
        if requested_cost is not None and target_cost is not None
        else None,
        "cost_known": requested_cost is not None and target_cost is not None,
    }


def _summary_model_hint_canary_sample(
    params: dict[str, Any],
    *,
    requested_model: str,
    target_model: str,
) -> dict[str, Any]:
    canary = dict(CODEX_APP_SUMMARY_MODEL_HINT_CANARY or {})
    fraction = min(max(float(canary.get("fraction", 1.0) or 0.0), 0.0), 1.0)
    holdout_fraction = min(max(float(canary.get("holdout_fraction", 0.0) or 0.0), 0.0), 1.0)
    salt = str(canary.get("salt") or "codex-app-summary-model-hint")
    unit = str(canary.get("unit") or "source_hash").strip().lower().replace("-", "_")
    if unit == "thread_id":
        material = str(_thread_id(params) or "")
        if not material:
            unit = "source_hash"
    if unit == "model_and_size":
        material = f"{requested_model}:{target_model}:{_input_text_chars(params.get('input'))}"
    elif unit == "thread_id":
        material = str(_thread_id(params) or "")
    else:
        unit = "source_hash"
        material = stable_json(params)
    digest = hashlib.sha256(f"{salt}\0{unit}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    applied_cutoff = min(1.0, holdout_fraction + fraction)
    if sample < holdout_fraction:
        cohort = "canary_holdout"
        status = "holdout"
        reason = "summary-model-hint-canary-holdout"
    elif sample < applied_cutoff:
        cohort = "canary_applied"
        status = "applied"
        reason = "safe-summary-model-hint-canary"
    else:
        cohort = "not_selected"
        status = "eligible-skipped"
        reason = "summary-model-hint-canary-not-selected"
    return {
        "enabled": True,
        "cohort": cohort,
        "status": status,
        "reason": reason,
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "sample_unit": unit,
        "sample_bucket": round(sample, 6),
        "hash_basis": "local-only-salted-policy-sample",
        "raw_basis_included": False,
    }


def _codex_exact_cache_canary_sample(params: dict[str, Any], *, requested_model: str) -> dict[str, Any]:
    canary = dict(CODEX_APP_CACHE_CANARY or {})
    fraction = min(max(float(canary.get("fraction", 1.0) or 0.0), 0.0), 1.0)
    holdout_fraction = min(max(float(canary.get("holdout_fraction", 0.0) or 0.0), 0.0), 1.0)
    salt = str(canary.get("salt") or "codex-app-exact-cache")
    unit = str(canary.get("unit") or "source_hash").strip().lower().replace("-", "_")
    if unit == "thread_id":
        material = str(_thread_id(params) or "")
        if not material:
            unit = "source_hash"
    if unit == "model_and_size":
        material = f"{requested_model}:{_input_text_chars(params.get('input'))}"
    elif unit == "thread_id":
        material = str(_thread_id(params) or "")
    else:
        unit = "source_hash"
        material = stable_json(params)
    digest = hashlib.sha256(f"{salt}\0{unit}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    applied_cutoff = min(1.0, holdout_fraction + fraction)
    if sample < holdout_fraction:
        cohort = "canary_holdout"
        status = "holdout"
        reason = "codex-app-cache-canary-holdout"
    elif sample < applied_cutoff:
        cohort = "canary_applied"
        status = "applied"
        reason = "safe-summary-exact-cache-canary"
    else:
        cohort = "not_selected"
        status = "eligible-skipped"
        reason = "codex-app-cache-canary-not-selected"
    return {
        "enabled": True,
        "cohort": cohort,
        "status": status,
        "reason": reason,
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "sample_unit": unit,
        "sample_bucket": round(sample, 6),
        "hash_basis": "local-only-salted-policy-sample",
        "raw_basis_included": False,
    }


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
            "summary_model_hint": {
                "status": "unsafe-skipped",
                "target_model": CODEX_APP_SUMMARY_MODEL_HINT_TARGET if CODEX_APP_SUMMARY_MODEL_HINT else "",
                "requested_model": None,
                "model_field_state": "absent",
                "workflow_phase": "unknown",
                "eligible": False,
                "skip_reason": "codex-turn-start-model-field-absent",
            },
        })
        return params, meta

    if CODEX_APP_SUMMARY_MODEL_HINT:
        target_model = str(CODEX_APP_SUMMARY_MODEL_HINT_TARGET or "").strip()
        base_meta = _policy_decision("routing", "skipped", "summary-model-hint-not-applied")
        base_meta.update({
            "canary": "codex-app-summary-model-hint",
            "canary_enabled": True,
            "canary_policy": {
                "fraction": CODEX_APP_SUMMARY_MODEL_HINT_CANARY.get("fraction"),
                "holdout_fraction": CODEX_APP_SUMMARY_MODEL_HINT_CANARY.get("holdout_fraction"),
                "sample_unit": CODEX_APP_SUMMARY_MODEL_HINT_CANARY.get("unit"),
            },
            "summary_model_hint": {
                "status": "unsafe-skipped",
                "target_model": target_model,
                "requested_model": requested_model,
                "model_field_state": "present",
                "workflow_phase": "unknown",
                "eligible": False,
                "skip_reason": "summary-model-hint-not-applied",
            },
            "model_field": model_field,
            "requested_model": requested_model,
            "routed_model": requested_model,
            "target_model": target_model,
        })
        if not target_model:
            base_meta["reason"] = "summary-model-hint-target-absent"
            base_meta["summary_model_hint"].update({
                "status": "eligible-skipped",
                "skip_reason": "summary-model-hint-target-absent",
            })
            return params, base_meta
        if _contains_action_hint(params):
            base_meta["reason"] = "action-like-params"
            base_meta["summary_model_hint"].update({
                "status": "unsafe-skipped",
                "skip_reason": "action-like-params",
            })
            return params, base_meta
        eligible, reason, eligibility_meta = _codex_cache_eligibility(params)
        base_meta.update({
            "workflow_phase": eligibility_meta.get("workflow_phase") or "unknown",
            "workflow_phase_reason": eligibility_meta.get("workflow_phase_reason"),
        })
        base_meta["summary_model_hint"].update({
            "workflow_phase": eligibility_meta.get("workflow_phase") or "unknown",
            "workflow_phase_reason": eligibility_meta.get("workflow_phase_reason"),
            "eligible": bool(eligible),
            "skip_reason": reason,
            "estimated_cost_delta": _summary_model_hint_cost_delta(
                params,
                requested_model=requested_model,
                target_model=target_model,
            ),
        })
        if eligibility_meta.get("unknown_keys"):
            base_meta["unknown_keys"] = eligibility_meta["unknown_keys"]
        if not eligible:
            base_meta["reason"] = reason
            base_meta["summary_model_hint"]["status"] = "unsafe-skipped"
            return params, base_meta
        if target_model == requested_model:
            base_meta["reason"] = "summary-model-hint-target-matches-requested"
            base_meta["summary_model_hint"].update({
                "status": "eligible-skipped",
                "skip_reason": "summary-model-hint-target-matches-requested",
            })
            return params, base_meta
        sample = _summary_model_hint_canary_sample(
            params,
            requested_model=requested_model,
            target_model=target_model,
        )
        base_meta["canary_cohort"] = sample["cohort"]
        base_meta["canary_sample"] = sample
        base_meta["summary_model_hint"]["canary_cohort"] = sample["cohort"]
        base_meta["summary_model_hint"]["canary_sample"] = sample
        if sample["status"] == "holdout":
            base_meta.update({
                "status": "skipped",
                "reason": sample["reason"],
                "routed_model": requested_model,
                "applied": False,
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
            base_meta["summary_model_hint"].update({
                "status": "holdout",
                "eligible": True,
                "skip_reason": sample["reason"],
            })
            return params, base_meta
        if sample["status"] == "eligible-skipped":
            base_meta.update({
                "status": "skipped",
                "reason": sample["reason"],
                "routed_model": requested_model,
                "applied": False,
            })
            base_meta["summary_model_hint"].update({
                "status": "eligible-skipped",
                "eligible": True,
                "skip_reason": sample["reason"],
            })
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
        base_meta["summary_model_hint"].update({
            "status": "applied",
            "eligible": True,
            "skip_reason": None,
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


def _codex_terminal_log_features(params: dict[str, Any]) -> dict[str, Any]:
    return terminal_log_features_from_text("\n".join(_codex_input_texts(params.get("input"))))


def _codex_prompt_difficulty_features(params: dict[str, Any]) -> dict[str, Any]:
    return prompt_difficulty_features_from_text("\n".join(_codex_input_texts(params.get("input"))))


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


def _codex_stale_risk_skip_reason(params: dict[str, Any], file_dependency_audit_meta: dict[str, Any]) -> str | None:
    text = "\n".join(_codex_input_texts(params.get("input")))
    invalidation_reason = file_dependency_audit_meta.get("invalidation_reason")
    if invalidation_reason in {"dependency-missing", "dependency-cap-exceeded", "file-watch-disabled"}:
        return str(invalidation_reason)
    if _CODEX_PATH_LIKE_TEXT_RE.search(text) and not file_dependency_audit_meta.get("safe_invalidation_evidence"):
        return "file-dependency-missing"
    if _CODEX_STALE_RISK_TEXT_RE.search(text) and not file_dependency_audit_meta.get("safe_invalidation_evidence"):
        return "stale-risk-blockers"
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


def _codex_input_size_bucket(chars: int) -> str:
    if chars < 2_000:
        return "small"
    if chars < 8_000:
        return "medium"
    return "large"


def _policy_value_as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _normalized_policy_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _condition_value_matches(expected: Any, actual: Any) -> bool:
    expected_bool = _policy_value_as_bool(expected)
    if expected_bool is not None:
        actual_bool = _policy_value_as_bool(actual)
        return actual_bool is not None and actual_bool == expected_bool
    if isinstance(actual, (list, tuple, set)):
        return _normalized_policy_value(expected) in {_normalized_policy_value(item) for item in actual}
    if isinstance(expected, (list, tuple, set)):
        return _normalized_policy_value(actual) in {_normalized_policy_value(item) for item in expected}
    expected_s = _normalized_policy_value(expected)
    actual_s = _normalized_policy_value(actual)
    if expected_s in {"present", "derived_present"} and actual_s in {"present", "derived_present"}:
        return True
    return expected_s == actual_s


def _codex_rule_candidate_id(rule: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "recommendation_id", "policy_id", "id", "rule_id"):
        value = rule.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    managed = rule.get("managed_recommendation")
    if isinstance(managed, dict):
        for key in ("candidate_id", "recommendation_id", "policy_id"):
            value = managed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"codex-app-rule-{index + 1}"


def _codex_rule_public_meta(
    rule: dict[str, Any],
    *,
    index: int,
    matched: bool,
    blockers: list[str] | None = None,
    cohort: str | None = None,
    cohort_reason: str | None = None,
) -> dict[str, Any]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    candidate_id = _codex_rule_candidate_id(rule, index)
    return {
        "schema": "agentflow.codex_app_rule_execution.v1",
        "rule_id": str(rule.get("id") or rule.get("rule_id") or candidate_id),
        "candidate_id": candidate_id,
        "policy_source": rule.get("policy_source") or CODEX_APP_POLICY_SOURCE,
        "matched": bool(matched),
        "condition_keys": sorted(str(key) for key in conditions),
        "action_keys": sorted(str(key) for key in action),
        "blockers": list(blockers or []),
        "canary_cohort": cohort,
        "cohort_reason": cohort_reason,
        "raw_conditions_included": False,
        "raw_actions_included": False,
        "raw_params_included": False,
    }


def _codex_rule_features(
    params: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    eligible, eligible_reason, eligibility_meta = _codex_cache_eligibility(params)
    file_dependency_audit_meta: dict[str, Any] = {}
    stale_signal: str | None = None
    if eligible:
        file_dependency_audit_meta = cache_file_dependency_audit(params)
        stale_signal = _codex_stale_risk_skip_reason(params, file_dependency_audit_meta)
    model_field, requested_model = _model_field(params)
    has_action_like = _contains_action_hint(params)
    workflow_phase = str(eligibility_meta.get("workflow_phase") or "unknown")
    replayability_level = "local-exact-response" if eligible and not stale_signal else "features_only"
    features = {
        "app_family": "codex",
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "workflow_phase": workflow_phase,
        "model_field_state": "present" if model_field and requested_model else "absent",
        "input_size_bucket": _codex_input_size_bucket(_input_text_chars(params.get("input"))),
        "cache_eligible": bool(eligible and not stale_signal),
        "cache_status": "eligible" if eligible and not stale_signal else "skipped",
        "replayability_level": replayability_level,
        "has_action_like_params": bool(has_action_like),
        "stale_risk": bool(stale_signal),
        "stale_risk_signal": stale_signal or "none",
        "supported_action_family": list(CODEX_APP_POLICY_ACTION_FAMILIES),
        "requested_model": requested_model,
        "model_field": model_field,
    }
    if eligibility_meta.get("workflow_phase_reason"):
        features["workflow_phase_reason"] = eligibility_meta.get("workflow_phase_reason")
    return features, stale_signal or eligible_reason, eligibility_meta, file_dependency_audit_meta


def _codex_rule_condition_blockers(rule: dict[str, Any], features: dict[str, Any]) -> list[str]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    blockers: list[str] = []
    supported_conditions = set(CODEX_APP_POLICY_CONDITION_KEYS)
    for key in sorted(set(conditions) - supported_conditions):
        blockers.append(f"unsupported-condition:{key}")
    for key, expected in conditions.items():
        if key not in supported_conditions:
            continue
        actual = features.get(key)
        if key not in features or actual is None or actual == "":
            blockers.append(f"insufficient-metadata:{key}")
            continue
        if not _condition_value_matches(expected, actual):
            if key == "stale_risk" and features.get("stale_risk"):
                blockers.append(str(features.get("stale_risk_signal") or "stale-risk-blockers"))
            elif key == "cache_eligible" and features.get("stale_risk"):
                blockers.append(str(features.get("stale_risk_signal") or "stale-risk-blockers"))
            elif key == "workflow_phase" and _normalized_policy_value(expected) == "summary":
                blockers.append("workflow-phase-not-summary")
            elif key == "model_field_state" and _normalized_policy_value(actual) == "absent":
                blockers.append("codex-turn-start-model-field-absent")
            elif key == "has_action_like_params" and bool(actual):
                blockers.append("action-like-params")
            else:
                blockers.append(f"condition-mismatch:{key}")
    return blockers


def _codex_rule_action_blockers(rule: dict[str, Any]) -> list[str]:
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    unsupported = sorted(str(key) for key in action if str(key) not in CODEX_APP_POLICY_ACTION_KEYS)
    return [f"unsupported-action:{key}" for key in unsupported]


def _codex_rule_safety_blocker(action: dict[str, Any], features: dict[str, Any], fallback_reason: str) -> str | None:
    if action.get("pass_through_reason"):
        return str(action.get("pass_through_reason"))
    if features.get("model_field_state") != "present":
        return "codex-turn-start-model-field-absent"
    if features.get("has_action_like_params"):
        return "action-like-params"
    if features.get("workflow_phase") != "summary":
        return "workflow-phase-not-summary"
    if not features.get("cache_eligible"):
        return str(features.get("stale_risk_signal") or fallback_reason or "not-safe-summary-turn")
    return None


def _codex_rule_canary_sample(
    rule: dict[str, Any],
    *,
    candidate_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    canary = rule.get("canary")
    if not isinstance(canary, dict):
        managed = rule.get("managed_recommendation")
        canary = managed.get("canary") if isinstance(managed, dict) and isinstance(managed.get("canary"), dict) else {}
    enabled = _policy_value_as_bool(canary.get("enabled")) if isinstance(canary, dict) else None
    if enabled is False:
        return {
            "enabled": False,
            "cohort": "applied",
            "status": "applied",
            "reason": "no-canary",
            "raw_basis_included": False,
        }
    fraction = float(canary.get("fraction", canary.get("canary_fraction", 1.0)) or 1.0) if isinstance(canary, dict) else 1.0
    holdout_fraction = float(canary.get("holdout_fraction", 0.0) or 0.0) if isinstance(canary, dict) else 0.0
    fraction = min(max(fraction, 0.0), 1.0)
    holdout_fraction = min(max(holdout_fraction, 0.0), 1.0)
    salt = str(canary.get("salt") or "codex-app-rule-canary") if isinstance(canary, dict) else "codex-app-rule-canary"
    unit = str(canary.get("unit") or "source_hash").strip().lower().replace("-", "_") if isinstance(canary, dict) else "source_hash"
    if unit == "thread_id":
        material = str(_thread_id(params) or "")
        if not material:
            unit = "source_hash"
    if unit == "model_and_size":
        material = f"{candidate_id}:{_model_field(params)[1] or ''}:{_input_text_chars(params.get('input'))}"
    elif unit == "thread_id":
        material = str(_thread_id(params) or "")
    else:
        unit = "source_hash"
        material = stable_json(params)
    digest = hashlib.sha256(f"{salt}\0{candidate_id}\0{unit}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    applied_cutoff = min(1.0, holdout_fraction + fraction)
    if sample < holdout_fraction:
        cohort = "canary_holdout"
        status = "holdout"
        reason = "codex-app-rule-canary-holdout"
    elif sample < applied_cutoff:
        cohort = "canary_applied"
        status = "applied"
        reason = "codex-app-rule-canary-applied"
    else:
        cohort = "not_selected"
        status = "eligible-skipped"
        reason = "codex-app-rule-canary-not-selected"
    return {
        "enabled": True,
        "cohort": cohort,
        "status": status,
        "reason": reason,
        "fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "sample_unit": unit,
        "sample_bucket": round(sample, 6),
        "hash_basis": "local-only-salted-policy-sample",
        "raw_basis_included": False,
    }


def _safety_number(value: Any, default: float, *, minimum: float | None = None) -> float:
    parsed = _as_float(value)
    if parsed is None:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _safety_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _codex_rule_safety_policy(rule: dict[str, Any]) -> dict[str, Any]:
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
    thresholds = safety.get("thresholds") if isinstance(safety.get("thresholds"), dict) else safety
    return {
        "enabled": _policy_value_as_bool(safety.get("enabled")) is not False,
        "window": _safety_int(thresholds.get("window"), 200, minimum=1),
        "min_outcome_samples": _safety_int(thresholds.get("min_outcome_samples"), 5, minimum=1),
        "min_holdout_samples": _safety_int(thresholds.get("min_holdout_samples"), 1, minimum=0),
        "max_error_rate": _safety_number(thresholds.get("max_error_rate"), 0.20, minimum=0.0),
        "max_error_rate_delta": _safety_number(thresholds.get("max_error_rate_delta"), 0.05, minimum=0.0),
        "max_retry_rate": _safety_number(thresholds.get("max_retry_rate"), 0.20, minimum=0.0),
        "max_retry_rate_delta": _safety_number(thresholds.get("max_retry_rate_delta"), 0.05, minimum=0.0),
        "max_latency_ms": _safety_number(thresholds.get("max_latency_ms"), 120_000.0, minimum=0.0),
        "max_latency_delta_ms": _safety_number(thresholds.get("max_latency_delta_ms"), 30_000.0, minimum=0.0),
        "max_stale_risk_rate": _safety_number(thresholds.get("max_stale_risk_rate"), 0.25, minimum=0.0),
        "unsafe_cache_envelope_limit": _safety_int(thresholds.get("unsafe_cache_envelope_limit"), 1, minimum=1),
        "cache_invalidation_failure_limit": _safety_int(thresholds.get("cache_invalidation_failure_limit"), 3, minimum=1),
    }


def _codex_rule_safety_stop_meta(
    *,
    rule_meta: dict[str, Any],
    policy: dict[str, Any],
    reason_codes: list[str],
    trigger_metrics: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    clean_reasons = sorted({str(reason) for reason in reason_codes if str(reason or "").strip()})
    return {
        "schema": "agentflow.codex_app_rule_safety_stop.v1",
        "enabled": bool(policy.get("enabled", True)),
        "tripped": bool(clean_reasons),
        "status": "stopped" if clean_reasons else "clear",
        "reason": "local-canary-safety-stop" if clean_reasons else "clear",
        "reason_codes": clean_reasons,
        "source": source,
        "policy_source": rule_meta.get("policy_source") or CODEX_APP_POLICY_SOURCE,
        "rule_id": rule_meta.get("rule_id"),
        "candidate_id": rule_meta.get("candidate_id"),
        "sample_count": _as_int(trigger_metrics.get("sample_count")),
        "applied_sample_count": _as_int(trigger_metrics.get("applied_sample_count")),
        "holdout_sample_count": _as_int(trigger_metrics.get("holdout_sample_count")),
        "trigger_metrics": trigger_metrics,
        "thresholds": {key: value for key, value in policy.items() if key != "enabled"},
        "clearance": "reviewed policy update, rollback, or a clean recent outcome window clears this runtime stop",
        "privacy": {
            "metadata_only": True,
            "raw_params_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
        },
    }


def _codex_rule_explicit_safety_stop(
    rule: dict[str, Any],
    *,
    rule_meta: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    safety = rule.get("safety_stop") if isinstance(rule.get("safety_stop"), dict) else {}
    if not safety or not policy.get("enabled", True):
        return None
    status = str(safety.get("status") or "").strip().lower().replace("-", "_")
    tripped = (
        bool(safety.get("tripped"))
        or bool(safety.get("stopped"))
        or bool(safety.get("active"))
        or status in {"stopped", "safety_stopped", "safety_stop"}
    )
    raw_reasons = safety.get("reason_codes") or safety.get("trigger_metrics") or safety.get("reasons") or []
    reason_codes = [str(reason) for reason in raw_reasons] if isinstance(raw_reasons, list) else []
    if safety.get("reason") and not reason_codes:
        reason_codes = [str(safety.get("reason"))]
    if not tripped and not reason_codes:
        return None
    trigger_metrics = {
        "sample_count": _as_int(safety.get("sample_count")),
        "applied_sample_count": _as_int(safety.get("applied_sample_count")),
        "holdout_sample_count": _as_int(safety.get("holdout_sample_count")),
    }
    return _codex_rule_safety_stop_meta(
        rule_meta=rule_meta,
        policy=policy,
        reason_codes=reason_codes or ["explicit-safety-stop"],
        trigger_metrics=trigger_metrics,
        source="policy-file",
    )


def _codex_rule_observed_safety_stop(
    rule: dict[str, Any],
    *,
    rule_meta: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    if not policy.get("enabled", True) or not hasattr(store, "conn"):
        return None
    window = _as_int(policy.get("window")) or 200
    try:
        rows = store.conn.execute("""
            select s.routing_json,
                   s.cache_json,
                   (
                       select r.error_code from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_error_code,
                   (
                       select r.latency_ms from codex_app_events r
                       where r.direction = 'server_to_client'
                         and r.request_id = s.request_id
                         and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                       order by r.created_at desc
                       limit 1
                   ) as response_latency_ms
            from codex_app_events s
            where s.direction = 'client_to_server'
              and s.method = 'turn/start'
            order by s.created_at desc
            limit ?
        """, (window,)).fetchall()
    except Exception:
        return None

    target_rule_id = str(rule_meta.get("rule_id") or "")
    target_candidate_id = str(rule_meta.get("candidate_id") or "")
    applied = {"count": 0, "errors": 0, "retries": 0, "latency_total": 0, "latency_count": 0}
    holdout = {"count": 0, "errors": 0, "retries": 0, "latency_total": 0, "latency_count": 0}
    unsafe_cache_envelopes = 0
    stale_risk = 0
    invalidated = 0
    sample_count = 0
    for row in rows:
        routing = _json_obj(row["routing_json"])
        cache = _json_obj(row["cache_json"])
        rule_bits = []
        for decision in (routing, cache):
            meta = decision.get("codex_app_rule") if isinstance(decision.get("codex_app_rule"), dict) else {}
            if meta:
                rule_bits.append(meta)
        if not any(
            str(meta.get("rule_id") or "") == target_rule_id
            or str(meta.get("candidate_id") or "") == target_candidate_id
            for meta in rule_bits
        ):
            continue
        sample_count += 1
        cache_reason = str(cache.get("reason") or "")
        cache_bucket = str(cache.get("outcome_bucket") or "")
        if cache_reason == "unsafe-cached-envelope" or cache.get("status") == "unsafe-skip":
            unsafe_cache_envelopes += 1
        if cache_bucket == "stale-risk" or cache_reason in {
            "stale-risk-blockers",
            "file-dependency-missing",
            "dependency-missing",
            "dependency-cap-exceeded",
            "file-watch-disabled",
        }:
            stale_risk += 1
        if cache_bucket == "invalidated" or cache_reason in {
            "dependency-changed",
            "dependency-deleted",
            "codex-cache-ttl-expired",
        }:
            invalidated += 1

        cohort = str(routing.get("canary_cohort") or cache.get("canary_cohort") or "")
        bucket = applied if cohort == "canary_applied" else holdout if cohort == "canary_holdout" else None
        if bucket is None:
            continue
        bucket["count"] += 1
        if row["response_error_code"] is not None:
            bucket["errors"] += 1
        retry_count = _as_int(routing.get("retry_count") or cache.get("retry_count"))
        if retry_count:
            bucket["retries"] += 1
        latency = _as_int(row["response_latency_ms"])
        if latency:
            bucket["latency_total"] += latency
            bucket["latency_count"] += 1

    applied_count = _as_int(applied["count"])
    holdout_count = _as_int(holdout["count"])
    applied_error_rate = (applied["errors"] / applied_count) if applied_count else 0.0
    holdout_error_rate = (holdout["errors"] / holdout_count) if holdout_count else 0.0
    applied_retry_rate = (applied["retries"] / applied_count) if applied_count else 0.0
    holdout_retry_rate = (holdout["retries"] / holdout_count) if holdout_count else 0.0
    applied_latency = (applied["latency_total"] / applied["latency_count"]) if applied["latency_count"] else 0.0
    holdout_latency = (holdout["latency_total"] / holdout["latency_count"]) if holdout["latency_count"] else 0.0
    stale_rate = stale_risk / sample_count if sample_count else 0.0
    reason_codes: list[str] = []
    min_samples = _as_int(policy.get("min_outcome_samples")) or 1
    min_holdout = _as_int(policy.get("min_holdout_samples"))
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    holdout_fraction = _as_float(canary.get("holdout_fraction")) if isinstance(canary, dict) else 0.0
    if holdout_fraction and applied_count >= min_samples and holdout_count < min_holdout:
        reason_codes.append("missing-holdout-evidence")
    if applied_count >= min_samples and applied_error_rate > _as_float(policy.get("max_error_rate")):
        reason_codes.append("applied-error-rate-above-threshold")
    if holdout_count >= min_holdout and applied_error_rate - holdout_error_rate > _as_float(policy.get("max_error_rate_delta")):
        reason_codes.append("applied-error-rate-regression")
    if applied_count >= min_samples and applied_retry_rate > _as_float(policy.get("max_retry_rate")):
        reason_codes.append("applied-retry-rate-above-threshold")
    if holdout_count >= min_holdout and applied_retry_rate - holdout_retry_rate > _as_float(policy.get("max_retry_rate_delta")):
        reason_codes.append("applied-retry-rate-regression")
    if applied["latency_count"] >= min_samples and applied_latency > _as_float(policy.get("max_latency_ms")):
        reason_codes.append("applied-latency-above-threshold")
    if holdout["latency_count"] >= min_holdout and applied_latency - holdout_latency > _as_float(policy.get("max_latency_delta_ms")):
        reason_codes.append("applied-latency-regression")
    if unsafe_cache_envelopes >= _as_int(policy.get("unsafe_cache_envelope_limit")):
        reason_codes.append("unsafe-cache-envelope")
    if sample_count >= min_samples and stale_rate > _as_float(policy.get("max_stale_risk_rate")):
        reason_codes.append("stale-risk-spike")
    if invalidated >= _as_int(policy.get("cache_invalidation_failure_limit")):
        reason_codes.append("cache-invalidation-failures")
    if not reason_codes:
        return None
    trigger_metrics = {
        "sample_count": sample_count,
        "applied_sample_count": applied_count,
        "holdout_sample_count": holdout_count,
        "applied_error_rate": round(applied_error_rate, 6),
        "holdout_error_rate": round(holdout_error_rate, 6),
        "applied_retry_rate": round(applied_retry_rate, 6),
        "holdout_retry_rate": round(holdout_retry_rate, 6),
        "applied_avg_latency_ms": round(applied_latency, 3) if applied_latency else None,
        "holdout_avg_latency_ms": round(holdout_latency, 3) if holdout_latency else None,
        "unsafe_cache_envelope_count": unsafe_cache_envelopes,
        "stale_risk_count": stale_risk,
        "stale_risk_rate": round(stale_rate, 6),
        "cache_invalidation_failure_count": invalidated,
    }
    return _codex_rule_safety_stop_meta(
        rule_meta=rule_meta,
        policy=policy,
        reason_codes=reason_codes,
        trigger_metrics={key: value for key, value in trigger_metrics.items() if value is not None},
        source="recent-local-outcomes",
    )


def _codex_rule_runtime_safety_stop(
    rule: dict[str, Any],
    *,
    rule_meta: dict[str, Any],
) -> dict[str, Any] | None:
    policy = _codex_rule_safety_policy(rule)
    explicit = _codex_rule_explicit_safety_stop(rule, rule_meta=rule_meta, policy=policy)
    if explicit is not None:
        return explicit
    return _codex_rule_observed_safety_stop(rule, rule_meta=rule_meta, policy=policy)


def _codex_rule_plan(params: dict[str, Any]) -> dict[str, Any]:
    features, safety_reason, eligibility_meta, file_dependency_audit_meta = _codex_rule_features(params)
    mismatch_counts: dict[str, int] = {}
    first_rule_meta: dict[str, Any] | None = None
    for index, rule in enumerate(CODEX_APP_RULES):
        conditions_blockers = _codex_rule_condition_blockers(rule, features)
        action_blockers = _codex_rule_action_blockers(rule)
        rule_meta = _codex_rule_public_meta(
            rule,
            index=index,
            matched=not conditions_blockers,
            blockers=conditions_blockers + action_blockers,
        )
        if first_rule_meta is None:
            first_rule_meta = rule_meta
        if conditions_blockers:
            for blocker in conditions_blockers:
                mismatch_counts[blocker] = int(mismatch_counts.get(blocker) or 0) + 1
            continue
        if action_blockers:
            return {
                "status": "blocked",
                "reason": action_blockers[0],
                "rule": rule,
                "rule_index": index,
                "rule_meta": rule_meta,
                "features": features,
                "eligibility_meta": eligibility_meta,
                "file_dependency_audit": file_dependency_audit_meta,
            }
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        safety_blocker = _codex_rule_safety_blocker(action, features, safety_reason)
        if safety_blocker:
            rule_meta["blockers"] = list(rule_meta.get("blockers") or []) + [safety_blocker]
            return {
                "status": "blocked",
                "reason": safety_blocker,
                "rule": rule,
                "rule_index": index,
                "rule_meta": rule_meta,
                "features": features,
                "eligibility_meta": eligibility_meta,
                "file_dependency_audit": file_dependency_audit_meta,
            }
        safety_stop = _codex_rule_runtime_safety_stop(rule, rule_meta=rule_meta)
        if safety_stop is not None:
            rule_meta["blockers"] = list(rule_meta.get("blockers") or []) + ["local-canary-safety-stop"]
            rule_meta["safety_stop"] = {
                "tripped": True,
                "reason_codes": safety_stop.get("reason_codes") or [],
                "source": safety_stop.get("source"),
            }
            return {
                "status": "safety_stopped",
                "reason": "local-canary-safety-stop",
                "rule": rule,
                "rule_index": index,
                "rule_meta": rule_meta,
                "features": features,
                "eligibility_meta": eligibility_meta,
                "file_dependency_audit": file_dependency_audit_meta,
                "safety_stop": safety_stop,
            }
        candidate_id = str(rule_meta["candidate_id"])
        sample = _codex_rule_canary_sample(rule, candidate_id=candidate_id, params=params)
        rule_meta = _codex_rule_public_meta(
            rule,
            index=index,
            matched=True,
            blockers=[],
            cohort=sample.get("cohort"),
            cohort_reason=sample.get("reason"),
        )
        return {
            "status": str(sample.get("status") or "applied"),
            "reason": str(sample.get("reason") or "codex-app-rule-applied"),
            "rule": rule,
            "rule_index": index,
            "rule_meta": rule_meta,
            "features": features,
            "eligibility_meta": eligibility_meta,
            "file_dependency_audit": file_dependency_audit_meta,
            "canary_sample": sample,
        }
    blockers = [
        {"reason": reason, "count": count}
        for reason, count in sorted(mismatch_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    priority = [
        "action-like-params",
        "codex-turn-start-model-field-absent",
        "workflow-phase-not-summary",
        "stale-risk-blockers",
        "file-dependency-missing",
        "dependency-missing",
        "dependency-cap-exceeded",
        "file-watch-disabled",
    ]
    reason = next((item for item in priority if item in mismatch_counts), None)
    if reason is None:
        reason = blockers[0]["reason"] if blockers else "no-matching-codex-app-rule"
    return {
        "status": "blocked",
        "reason": reason,
        "rule": None,
        "rule_index": -1,
        "rule_meta": first_rule_meta or {
            "schema": "agentflow.codex_app_rule_execution.v1",
            "matched": False,
            "blockers": [reason],
            "raw_params_included": False,
        },
        "features": features,
        "eligibility_meta": eligibility_meta,
        "file_dependency_audit": file_dependency_audit_meta,
        "blocker_breakdown": blockers,
    }


def _codex_cache_key_for_message(msg: dict[str, Any]) -> str:
    key_msg = copy.deepcopy(msg)
    key_msg.pop("id", None)
    return cache_key_for(
        key_msg,
        "codex-app://turn/start",
        provider="codex-app",
        upstream=DEFAULT_UPSTREAM,
        namespace=codex_app_cache_namespace(),
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


def _codex_cache_payload(
    response_obj: dict[str, Any],
    *,
    ttl_seconds: int,
    file_deps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "agentflow_cache_type": "codex-app-jsonrpc-response",
        "version": 1,
        "created_at": utc_now(),
        "ttl_seconds": max(0, int(ttl_seconds)),
        "metadata": {
            "schema": "agentflow.codex_app_exact_cache_entry.v1",
            "response_body_storage": "local-cache-table",
            "file_dependency_count": len(file_deps or []),
            "file_dependency_evidence_available": bool(file_deps),
            "raw_request_included": False,
            "cache_key_included": False,
        },
        "response": response_obj,
    }


def _codex_payload_expired(payload: dict[str, Any]) -> bool:
    ttl_seconds = _as_int(payload.get("ttl_seconds"))
    if ttl_seconds <= 0:
        return False
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        return False
    try:
        text = created_at[:-1] + "+00:00" if created_at.endswith("Z") else created_at
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return age_seconds > ttl_seconds


def _codex_cached_response(payload: Any, request_id: Any) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, "invalid-cache-payload"
    if payload.get("agentflow_cache_type") != "codex-app-jsonrpc-response":
        return None, "invalid-cache-payload"
    if _codex_payload_expired(payload):
        return None, "codex-cache-ttl-expired"
    response = payload.get("response")
    if not isinstance(response, dict):
        return None, "unsafe-cached-envelope"
    replay = copy.deepcopy(response)
    if request_id is not None:
        replay["id"] = request_id
    if not _is_safe_codex_response_obj(replay, request_id):
        return None, "unsafe-cached-envelope"
    return json.dumps(replay, separators=(",", ":"), ensure_ascii=False), None


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


def _codex_phase_signal(method: Any) -> str | None:
    method_l = str(method or "").replace("_", "").replace("-", "").lower()
    if not method_l:
        return None
    if method_l in {"initialize", "threadstart", "threadconfigure"}:
        return "idle_control"
    if "commandexecution" in method_l or "toolcall" in method_l or "toolresult" in method_l:
        return "tool_execution"
    if "diff" in method_l or "patch" in method_l:
        return "verification"
    if "plan" in method_l:
        return "planning"
    if "agentmessage" in method_l or "message/delta" in str(method or "").lower():
        return "summary"
    return None


def _codex_phase_from_method_counts(method_counts: Any) -> dict[str, Any] | None:
    if not isinstance(method_counts, dict):
        return None
    signal_counts: dict[str, int] = {}
    signal_methods: dict[str, list[str]] = {}
    for method, count_raw in method_counts.items():
        signal = _codex_phase_signal(method)
        if not signal:
            continue
        count = _as_int(count_raw)
        if count <= 0:
            count = 1
        signal_counts[signal] = int(signal_counts.get(signal) or 0) + count
        methods = signal_methods.setdefault(signal, [])
        method_s = str(method or "")
        if method_s and method_s not in methods:
            methods.append(method_s)
    for phase in ("tool_execution", "verification", "planning", "summary", "idle_control"):
        if signal_counts.get(phase):
            return {
                "workflow_phase": phase,
                "workflow_phase_reason": f"event-window-signal:{phase}",
                "workflow_phase_source": "event_window",
                "workflow_phase_confidence": "high",
                "workflow_phase_signals": signal_methods.get(phase, [])[:5],
            }
    return None


def _codex_phase_from_decision_metadata(
    optimization_metadata: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    for decision in (optimization_metadata or {}).values():
        if not isinstance(decision, dict):
            continue
        phase = str(decision.get("workflow_phase") or "").strip()
        if phase and phase != "unknown":
            return {
                "workflow_phase": phase,
                "workflow_phase_reason": str(decision.get("workflow_phase_reason") or "decision-metadata"),
                "workflow_phase_source": "decision_metadata",
                "workflow_phase_confidence": str(decision.get("workflow_phase_confidence") or "medium"),
                "workflow_phase_signals": list(decision.get("workflow_phase_signals") or []),
            }
    reasons = {
        str(decision.get("reason") or "")
        for decision in (optimization_metadata or {}).values()
        if isinstance(decision, dict)
    }
    if "action-like-params" in reasons:
        return {
            "workflow_phase": "tool_execution",
            "workflow_phase_reason": "decision-reason:action-like-params",
            "workflow_phase_source": "decision_metadata",
            "workflow_phase_confidence": "medium",
            "workflow_phase_signals": ["action-like-params"],
        }
    return None


def _update_event_window_workflow_phase(
    window: dict[str, Any],
    optimization_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    phase = _codex_phase_from_method_counts(window.get("method_counts"))
    if phase is None:
        phase = _codex_phase_from_decision_metadata(optimization_metadata)
    if phase is None:
        return
    window.update(phase)


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
    _update_event_window_workflow_phase(window, optimization_metadata)
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
    _update_event_window_workflow_phase(window, optimization_metadata)
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
    local = _attach_codex_local_pattern_features(raw, optimization_metadata)
    if local is None:
        return None
    request_id = local["request_id"]
    routing = local["routing"]
    crunch = local["crunch"]
    cache = local["cache"]
    unit = local["unit"]
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
        "input_text_chars": local["input_text_chars"],
    }


def _codex_rule_cache_lookup_or_record(
    optimized: dict[str, Any],
    routed_params: dict[str, Any],
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    features = plan.get("features") if isinstance(plan.get("features"), dict) else {}
    eligibility_meta = plan.get("eligibility_meta") if isinstance(plan.get("eligibility_meta"), dict) else {}
    file_dependency_audit_meta = plan.get("file_dependency_audit") if isinstance(plan.get("file_dependency_audit"), dict) else {}
    workflow_phase = str(features.get("workflow_phase") or eligibility_meta.get("workflow_phase") or "unknown")
    workflow_phase_reason = eligibility_meta.get("workflow_phase_reason")
    replayability_level = str(features.get("replayability_level") or "features_only")
    rule_meta = plan.get("rule_meta") if isinstance(plan.get("rule_meta"), dict) else {}
    canary_sample = plan.get("canary_sample") if isinstance(plan.get("canary_sample"), dict) else None
    file_deps = cache_file_dependency_snapshots(routed_params)
    stale_reason = _codex_stale_risk_skip_reason(routed_params, file_dependency_audit_meta)
    if stale_reason:
        cache_meta = _codex_cache_decision(
            "skipped",
            stale_reason,
            enabled=True,
            eligible=True,
            replayability_level=replayability_level,
            file_dependencies=file_deps,
            file_dependency_audit_meta=file_dependency_audit_meta,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="stale-risk",
            canary_sample=canary_sample,
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["codex_app_rule"] = rule_meta
        return cache_meta

    cache_key = _codex_cache_key_for_message(optimized)
    cached, invalidation_reason = store.get_cache_with_reason(cache_key)
    if cached is None:
        cache_meta = _codex_cache_decision(
            "miss",
            invalidation_reason or "exact-miss",
            enabled=True,
            eligible=True,
            cache_key=cache_key,
            replayability_level="local-exact-response",
            file_dependencies=file_deps,
            file_dependency_audit_meta=file_dependency_audit_meta,
            invalidation_reason=invalidation_reason,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="invalidated" if invalidation_reason else "miss",
            canary_sample=canary_sample,
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["codex_app_rule"] = rule_meta
        cache_meta[_INTERNAL_CACHE_KEY] = cache_key
        return cache_meta

    replay_frame, cached_skip_reason = _codex_cached_response(cached, optimized.get("id"))
    if cached_skip_reason in {"unsafe-cached-envelope", "codex-cache-ttl-expired"}:
        store.delete_cache(cache_key)
    elif cached_skip_reason:
        replay_frame = None
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
            file_dependency_audit_meta=file_dependency_audit_meta,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="hit",
            canary_sample=canary_sample,
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["codex_app_rule"] = rule_meta
        cache_meta[_INTERNAL_REPLAY_FRAME_KEY] = replay_frame
        return cache_meta

    reason = cached_skip_reason or "unsafe-cached-envelope"
    status = "unsafe-skip" if reason == "unsafe-cached-envelope" else "miss"
    cache_meta = _codex_cache_decision(
        status,
        reason,
        enabled=True,
        eligible=True,
        cache_key=cache_key,
        replayability_level="local-exact-response",
        file_dependencies=file_deps,
        file_dependency_audit_meta=file_dependency_audit_meta,
        workflow_phase=workflow_phase,
        workflow_phase_reason=workflow_phase_reason,
        outcome_bucket=_codex_cache_outcome_bucket(status, reason),
        canary_sample=canary_sample,
        ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
    )
    cache_meta["codex_app_rule"] = rule_meta
    cache_meta[_INTERNAL_CACHE_KEY] = cache_key
    return cache_meta


def _optimize_client_message_with_rules(
    raw: str,
    msg: dict[str, Any],
    params: dict[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    plan = _codex_rule_plan(params)
    features = plan.get("features") if isinstance(plan.get("features"), dict) else {}
    rule = plan.get("rule") if isinstance(plan.get("rule"), dict) else None
    action = rule.get("action") if isinstance(rule, dict) and isinstance(rule.get("action"), dict) else {}
    rule_meta = plan.get("rule_meta") if isinstance(plan.get("rule_meta"), dict) else {}
    workflow_phase = str(features.get("workflow_phase") or "unknown")
    workflow_phase_reason = (plan.get("eligibility_meta") or {}).get("workflow_phase_reason") if isinstance(plan.get("eligibility_meta"), dict) else None
    policy_source = str(rule_meta.get("policy_source") or CODEX_APP_POLICY_SOURCE)
    requested_model = str(features.get("requested_model") or "")
    model_field = features.get("model_field")
    target_model = str(action.get("recommended_model") or action.get("model_hint") or "").strip()

    if plan.get("status") == "safety_stopped":
        reason = "local-canary-safety-stop"
        safety_stop = plan.get("safety_stop") if isinstance(plan.get("safety_stop"), dict) else {}
        routing_meta = _policy_decision("routing", "safety_stopped", reason, enabled=True)
        routing_meta.update({
            "policy_source": policy_source,
            "requested_model": requested_model or None,
            "routed_model": requested_model or None,
            "target_model": target_model or None,
            "model_field": model_field,
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "canary": "codex-app-rule",
            "canary_cohort": "safety_stopped",
            "applied": False,
            "codex_app_rule": rule_meta,
            "safety_stop": safety_stop,
        })
        crunch_meta = _policy_decision("crunch", "skipped", reason, enabled=True)
        crunch_meta.update({
            "policy_source": policy_source,
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "applied": False,
            "codex_app_rule": rule_meta,
            "safety_stop": safety_stop,
        })
        cache_meta = _codex_cache_decision(
            "skipped",
            reason,
            enabled=True,
            eligible=bool(features.get("cache_eligible")),
            replayability_level=str(features.get("replayability_level") or "features_only"),
            file_dependency_audit_meta=plan.get("file_dependency_audit") if isinstance(plan.get("file_dependency_audit"), dict) else None,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="safety_stop",
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta.update({
            "policy_source": policy_source,
            "canary": "codex-app-rule",
            "canary_cohort": "safety_stopped",
            "codex_app_rule": rule_meta,
            "safety_stop": safety_stop,
        })
        metadata = {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata

    if plan.get("status") in {"blocked", "eligible-skipped"} or not rule:
        reason = str(plan.get("reason") or "no-matching-codex-app-rule")
        routing_meta = _policy_decision("routing", "skipped", reason, enabled=True)
        routing_meta.update({
            "policy_source": policy_source,
            "requested_model": requested_model or None,
            "routed_model": requested_model or None,
            "model_field": model_field,
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "codex_app_rule": rule_meta,
        })
        crunch_meta = _policy_decision("crunch", "skipped", reason, enabled=True)
        crunch_meta.update({
            "policy_source": policy_source,
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "codex_app_rule": rule_meta,
        })
        cache_meta = _codex_cache_decision(
            "skipped",
            reason,
            enabled=True,
            eligible=bool(features.get("cache_eligible")),
            replayability_level=str(features.get("replayability_level") or "features_only"),
            file_dependency_audit_meta=plan.get("file_dependency_audit") if isinstance(plan.get("file_dependency_audit"), dict) else None,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket=_codex_cache_outcome_bucket("skipped", reason),
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["policy_source"] = policy_source
        cache_meta["codex_app_rule"] = rule_meta
        metadata = {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata

    if plan.get("status") == "holdout":
        reason = str(plan.get("reason") or "codex-app-rule-canary-holdout")
        sample = plan.get("canary_sample") if isinstance(plan.get("canary_sample"), dict) else None
        routing_meta = _policy_decision("routing", "skipped", reason, enabled=True)
        routing_meta.update({
            "policy_source": policy_source,
            "requested_model": requested_model,
            "routed_model": requested_model,
            "target_model": target_model or None,
            "model_field": model_field,
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "canary": "codex-app-rule",
            "canary_cohort": sample.get("cohort") if sample else None,
            "canary_sample": sample,
            "codex_app_rule": rule_meta,
        })
        crunch_meta = _policy_decision("crunch", "skipped", reason, enabled=True)
        crunch_meta.update({"policy_source": policy_source, "workflow_phase": workflow_phase, "codex_app_rule": rule_meta})
        cache_meta = _codex_cache_decision(
            "holdout" if _policy_value_as_bool(action.get("cache_eligible")) is True else "skipped",
            reason,
            enabled=True,
            eligible=bool(features.get("cache_eligible")),
            replayability_level=str(features.get("replayability_level") or "features_only"),
            file_dependency_audit_meta=plan.get("file_dependency_audit") if isinstance(plan.get("file_dependency_audit"), dict) else None,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="holdout",
            canary_sample=sample,
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["policy_source"] = policy_source
        cache_meta["codex_app_rule"] = rule_meta
        metadata = {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata

    routed_params = copy.deepcopy(params)
    routed_model = requested_model
    routing_status = "skipped"
    routing_reason = str(action.get("reason") or "codex-app-rule-no-routing-action")
    if target_model and model_field and target_model != requested_model:
        routed_params[str(model_field)] = target_model
        routed_model = target_model
        routing_status = "applied"
        routing_reason = str(action.get("reason") or "codex-app-rule-summary-model-hint")
    elif target_model:
        routing_reason = "codex-app-rule-target-matches-requested"

    routing_meta = _policy_decision("routing", routing_status, routing_reason, enabled=True)
    routing_meta.update({
        "policy_source": policy_source,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "target_model": target_model or None,
        "model_field": model_field,
        "workflow_phase": workflow_phase,
        "workflow_phase_reason": workflow_phase_reason,
        "canary": "codex-app-rule",
        "canary_cohort": (plan.get("canary_sample") or {}).get("cohort") if isinstance(plan.get("canary_sample"), dict) else None,
        "canary_sample": plan.get("canary_sample"),
        "codex_app_rule": rule_meta,
        "safety_gates": {
            "known_model_field": True,
            "safe_param_shape": True,
            "text_only_input": True,
            "action_like_params": False,
            "workflow_phase": "summary",
            "stale_risk": False,
        },
    })

    crunch_profile = action.get("crunch_profile")
    crunch_meta = _policy_decision(
        "crunch",
        "hinted" if crunch_profile else "skipped",
        "codex-app-rule-crunch-profile-hint" if crunch_profile else "codex-app-rule-no-crunch-action",
        enabled=True,
    )
    crunch_meta.update({
        "policy_source": policy_source,
        "workflow_phase": workflow_phase,
        "workflow_phase_reason": workflow_phase_reason,
        "profile_hint": str(crunch_profile).strip() if crunch_profile else None,
        "applied": False,
        "codex_app_rule": rule_meta,
    })

    optimized = copy.deepcopy(msg)
    optimized["params"] = routed_params
    if _policy_value_as_bool(action.get("cache_eligible")) is True:
        cache_meta = _codex_rule_cache_lookup_or_record(optimized, routed_params, plan=plan)
    else:
        cache_meta = _codex_cache_decision(
            "skipped",
            str(action.get("cache_eligibility_reason") or "codex-app-rule-no-cache-action"),
            enabled=True,
            eligible=bool(features.get("cache_eligible")),
            replayability_level=str(features.get("replayability_level") or "features_only"),
            file_dependency_audit_meta=plan.get("file_dependency_audit") if isinstance(plan.get("file_dependency_audit"), dict) else None,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="disabled",
            canary_sample=plan.get("canary_sample") if isinstance(plan.get("canary_sample"), dict) else None,
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        cache_meta["codex_app_rule"] = rule_meta
    cache_meta["policy_source"] = policy_source

    metadata = {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
    if optimized == msg:
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata
    forwarded = json.dumps(optimized, separators=(",", ":"), ensure_ascii=False)
    _attach_codex_local_pattern_features(forwarded, metadata)
    return forwarded, metadata


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

    if CODEX_APP_RULES:
        return _optimize_client_message_with_rules(raw, msg, params)

    if _contains_action_hint(params):
        metadata = _not_applied_metadata("action-like-params")
        if CODEX_APP_SUMMARY_MODEL_HINT:
            _params, metadata["routing"] = _codex_route_params(params)
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata

    crunched_params, crunch_meta = _codex_crunch_params(params)
    routed_params, routing_meta = _codex_route_params(crunched_params)
    routing_meta["terminal_log_features"] = _codex_terminal_log_features(params)
    routing_meta["prompt_difficulty_features"] = _codex_prompt_difficulty_features(params)

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
            outcome_bucket="disabled",
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
    elif not eligible:
        cache_meta = _codex_cache_decision(
            "skipped",
            eligible_reason,
            enabled=True,
            eligible=False,
            workflow_phase=workflow_phase,
            workflow_phase_reason=workflow_phase_reason,
            outcome_bucket="unsafe-skip",
            ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
        )
        if eligibility_meta.get("unknown_keys"):
            cache_meta["unknown_keys"] = eligibility_meta["unknown_keys"]
    else:
        cache_key = _codex_cache_key_for_message(optimized)
        file_deps = cache_file_dependency_snapshots(routed_params)
        file_dependency_audit_meta = cache_file_dependency_audit(routed_params)
        requested_model = str(routed_params.get(eligibility_meta.get("model_field") or "model") or "")
        canary_sample = _codex_exact_cache_canary_sample(routed_params, requested_model=requested_model)
        if canary_sample["status"] == "holdout":
            cache_meta = _codex_cache_decision(
                "holdout",
                canary_sample["reason"],
                enabled=True,
                eligible=True,
                replayability_level="local-exact-response",
                file_dependencies=file_deps,
                file_dependency_audit_meta=file_dependency_audit_meta,
                workflow_phase=workflow_phase,
                workflow_phase_reason=workflow_phase_reason,
                outcome_bucket="holdout",
                canary_sample=canary_sample,
                ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
            )
        elif canary_sample["status"] == "eligible-skipped":
            cache_meta = _codex_cache_decision(
                "skipped",
                canary_sample["reason"],
                enabled=True,
                eligible=True,
                replayability_level="local-exact-response",
                file_dependencies=file_deps,
                file_dependency_audit_meta=file_dependency_audit_meta,
                workflow_phase=workflow_phase,
                workflow_phase_reason=workflow_phase_reason,
                outcome_bucket="disabled",
                canary_sample=canary_sample,
                ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
            )
        elif stale_reason := _codex_stale_risk_skip_reason(routed_params, file_dependency_audit_meta):
            cache_meta = _codex_cache_decision(
                "skipped",
                stale_reason,
                enabled=True,
                eligible=True,
                replayability_level="local-exact-response",
                file_dependencies=file_deps,
                file_dependency_audit_meta=file_dependency_audit_meta,
                workflow_phase=workflow_phase,
                workflow_phase_reason=workflow_phase_reason,
                outcome_bucket="stale-risk",
                canary_sample=canary_sample,
                ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
            )
        else:
            cached, invalidation_reason = store.get_cache_with_reason(cache_key)
            if cached is None:
                cache_meta = _codex_cache_decision(
                    "miss",
                    invalidation_reason or "exact-miss",
                    enabled=True,
                    eligible=True,
                    cache_key=cache_key,
                    replayability_level="local-exact-response",
                    file_dependencies=file_deps,
                    file_dependency_audit_meta=file_dependency_audit_meta,
                    invalidation_reason=invalidation_reason,
                    workflow_phase=workflow_phase,
                    workflow_phase_reason=workflow_phase_reason,
                    outcome_bucket="invalidated" if invalidation_reason else "miss",
                    canary_sample=canary_sample,
                    ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
                )
                cache_meta[_INTERNAL_CACHE_KEY] = cache_key
            else:
                replay_frame, cached_skip_reason = _codex_cached_response(cached, msg.get("id"))
                if cached_skip_reason in {"unsafe-cached-envelope", "codex-cache-ttl-expired"}:
                    store.delete_cache(cache_key)
                elif cached_skip_reason:
                    replay_frame = None
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
                        file_dependency_audit_meta=file_dependency_audit_meta,
                        workflow_phase=workflow_phase,
                        workflow_phase_reason=workflow_phase_reason,
                        outcome_bucket="hit",
                        canary_sample=canary_sample,
                        ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
                    )
                    cache_meta[_INTERNAL_REPLAY_FRAME_KEY] = replay_frame
                else:
                    reason = cached_skip_reason or "unsafe-cached-envelope"
                    status = "unsafe-skip" if reason == "unsafe-cached-envelope" else "miss"
                    cache_meta = _codex_cache_decision(
                        status,
                        reason,
                        enabled=True,
                        eligible=True,
                        cache_key=cache_key,
                        replayability_level="local-exact-response",
                        file_dependencies=file_deps,
                        file_dependency_audit_meta=file_dependency_audit_meta,
                        workflow_phase=workflow_phase,
                        workflow_phase_reason=workflow_phase_reason,
                        outcome_bucket=_codex_cache_outcome_bucket(status, reason),
                        canary_sample=canary_sample,
                        ttl_seconds=CODEX_APP_CACHE_TTL_SECONDS,
                    )
                    cache_meta[_INTERNAL_CACHE_KEY] = cache_key
    if optimized == msg:
        metadata = {"routing": routing_meta, "crunch": crunch_meta, "cache": cache_meta}
        _attach_codex_local_pattern_features(raw, metadata)
        return raw, metadata
    forwarded = json.dumps(optimized, separators=(",", ":"), ensure_ascii=False)
    metadata = {
        "routing": routing_meta,
        "crunch": crunch_meta,
        "cache": cache_meta,
    }
    _attach_codex_local_pattern_features(forwarded, metadata)
    return forwarded, metadata


def _attach_codex_routing_experiment_pending(
    raw: str | bytes,
    *,
    optimization_metadata: dict[str, dict[str, Any]] | None,
    start_event_id: str | None,
    pending_routing_experiments: dict[str, dict[str, Any]],
) -> None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict):
        return
    method = msg.get("method")
    if method != "turn/start":
        return
    request_id = _request_id(msg)
    if request_id is None:
        return
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        return
    result = _decide_codex_routing_experiment(params, optimization_metadata)
    if result is None:
        return
    experiment_meta, routing_meta = result
    start_routing_meta = dict((optimization_metadata or {}).get("routing") or {})
    if start_event_id:
        routing_for_event = dict(start_routing_meta or routing_meta)
        routing_for_event["routing_experiment"] = experiment_meta
        try:
            store.update_codex_app_event_routing_json(start_event_id, stable_json(routing_for_event))
        except Exception as exc:
            print(f"AgentFlow Codex routing experiment metadata update skipped: {exc}", file=sys.stderr)
    if not experiment_meta.get("sampled"):
        return
    pending_routing_experiments[request_id] = {
        "experiment_meta": experiment_meta,
        "routing_meta": routing_meta,
        "start_routing_meta": start_routing_meta,
        "input_text_chars": _input_text_chars(params.get("input")),
        "start_event_id": start_event_id or "",
    }


def _decide_codex_routing_experiment(
    params: dict[str, Any],
    optimization_metadata: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return metadata for sampled and skipped Codex routing experiment decisions."""
    _, requested_model = _model_field(params)
    routing = (optimization_metadata or {}).get("routing") or {}
    requested_model = str(requested_model or routing.get("requested_model") or "")
    routed_model = str(routing.get("routed_model") or requested_model)
    workflow_phase = str(routing.get("workflow_phase") or "unknown")
    text_chars = _input_text_chars(params.get("input"))
    experiment_routing_meta: dict[str, Any] = {
        "requested_model": requested_model,
        "routed_model": routed_model,
        "category": "codex-turn",
        "workflow_phase": workflow_phase,
        "text_chars": text_chars,
    }
    try:
        experiment_meta = routing_experiment_decision(
            params,
            experiment_routing_meta,
            stream=False,
            provider="openai",
            source_surface="codex_turn",
            store_obj=store,
        )
    except Exception as exc:
        print(f"AgentFlow Codex routing experiment decision skipped: {exc}", file=sys.stderr)
        return None
    return experiment_meta, experiment_routing_meta


def _codex_result_output_text(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            if item:
                parts.append(item)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)
        elif isinstance(item, dict):
            for nested in item.values():
                if isinstance(nested, (str, list, dict)):
                    walk(nested)

    walk(value)
    return "\n".join(parts)


def _codex_result_comparison_body(result: Any) -> dict[str, Any]:
    text = _codex_result_output_text(result)
    if text:
        return {"output_text": text}
    return result if isinstance(result, dict) else {}


async def _maybe_run_codex_routing_experiment(
    raw: str | bytes,
    *,
    request_started: dict[str, float],
    pending_routing_experiments: dict[str, dict[str, Any]],
) -> None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict):
        return
    request_id = _request_id(msg)
    if request_id is None:
        return
    pending = pending_routing_experiments.pop(request_id, None)
    if not pending:
        return

    experiment_meta = pending["experiment_meta"]
    routing_meta = pending["routing_meta"]
    input_text_chars = int(pending.get("input_text_chars") or 0)
    start_event_id = pending.get("start_event_id") or ""

    started = request_started.get(request_id)
    primary_latency_ms = int((time.time() - started) * 1000) if started is not None else None

    error_obj = msg.get("error")
    result = msg.get("result")
    if isinstance(error_obj, dict):
        primary_status_code = 500
    elif result is not None:
        primary_status_code = 200
    else:
        primary_status_code = None

    primary_response_body = _codex_result_comparison_body(result) if primary_status_code == 200 else {}
    primary_output_text = response_output_text(primary_response_body)
    primary_output_chars = len(primary_output_text)

    # Shadow WebSocket turn would mutate upstream session state; record limitation only
    shadow_limitation = "websocket-stateful-turn-shadow-unsafe"
    comparison = compare_response_outputs(primary_response_body if primary_status_code == 200 else {}, None)

    experiment_id = str(uuid.uuid4())
    shadow_model = str(experiment_meta.get("shadow_model") or "")
    primary_model = str(experiment_meta.get("primary_model") or routing_meta.get("requested_model") or "")

    input_tokens_est = max(0, input_text_chars // TOKEN_CHARS)
    output_tokens_est = max(0, primary_output_chars // TOKEN_CHARS)
    requested_model_for_cost = str(routing_meta.get("requested_model") or "")
    primary_cost_est_usd: float | None = None
    if requested_model_for_cost and (input_tokens_est or output_tokens_est):
        raw_cost = estimate_cost(
            requested_model_for_cost,
            input_tokens_est,
            output_tokens_est,
            provider="openai",
            processing_mode=codex_app_processing_mode(),
        )
        if raw_cost is not None:
            primary_cost_est_usd = float(raw_cost)

    feedback_features = routing_experiment_feedback_features(
        experiment_id=experiment_id,
        experiment_meta=experiment_meta,
        routing_meta=routing_meta,
        comparison=comparison,
        primary_model=primary_model,
        shadow_model=shadow_model,
        primary_status_code=primary_status_code,
        shadow_status_code=None,
        primary_latency_ms=primary_latency_ms,
        shadow_latency_ms=None,
        primary_cost_est_usd=primary_cost_est_usd,
        shadow_cost_est_usd=None,
        error=shadow_limitation,
    )
    experiment_meta.update({
        "experiment_id": experiment_id,
        "status": feedback_features["status"],
        "shadow_limitation": shadow_limitation,
        "primary_model": primary_model,
        "shadow_model": shadow_model,
        "primary_status_code": primary_status_code,
        "shadow_status_code": None,
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
    })

    try:
        event = routing_experiment_outcome_event(feedback_features)
        feedback_meta = await queue_policy_event_feedback(
            store,
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
    if start_event_id:
        routing_for_event = dict(pending.get("start_routing_meta") or {})
        if not routing_for_event:
            routing_for_event = dict(routing_meta)
        routing_for_event["routing_experiment"] = experiment_meta
        try:
            store.update_codex_app_event_routing_json(start_event_id, stable_json(routing_for_event))
        except Exception as exc:
            print(f"AgentFlow Codex routing experiment metadata update skipped: {exc}", file=sys.stderr)

    try:
        store.log_routing_experiment(
            id=experiment_id,
            call_id=start_event_id or None,
            created_at=utc_now(),
            provider="openai",
            source_surface="codex_turn",
            requested_model=experiment_meta.get("requested_model"),
            routed_model=experiment_meta.get("routed_model"),
            primary_model=primary_model,
            shadow_model=shadow_model,
            category=routing_meta.get("category"),
            routing_reason=routing_meta.get("reason"),
            input_tokens_est=input_tokens_est,
            primary_status_code=primary_status_code,
            shadow_status_code=None,
            primary_latency_ms=primary_latency_ms,
            shadow_latency_ms=None,
            primary_output_chars=comparison["primary_output_chars"],
            shadow_output_chars=comparison["shadow_output_chars"],
            primary_output_sha256=comparison["primary_output_sha256"],
            shadow_output_sha256=comparison["shadow_output_sha256"],
            output_similarity=comparison["output_similarity"],
            passed_threshold=0,
            primary_cost_est_usd=primary_cost_est_usd,
            shadow_cost_est_usd=None,
            budget_limit_usd=experiment_meta.get("daily_budget_usd"),
            budget_spent_before_usd=experiment_meta.get("budget_spent_usd"),
            budget_remaining_before_usd=experiment_meta.get("budget_remaining_usd"),
            budget_spent_after_usd=float(experiment_meta.get("budget_spent_usd") or 0.0),
            error=shadow_limitation,
            routing_json=stable_json(routing_meta),
            experiment_json=stable_json(experiment_meta),
            primary_response_json=stable_json(primary_response_body) if ROUTING_EXPERIMENT_STORE_RESPONSE_BODIES else None,
        )
    except Exception as exc:
        print(f"AgentFlow Codex routing experiment log skipped: {exc}", file=sys.stderr)


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
    metadata = _codex_app_signal_metadata(str(method) if method is not None else None, params)
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
        metadata_json=stable_json(metadata) if metadata else None,
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


def _attach_codex_canary_lifecycle_pending(
    raw: str | bytes,
    optimization_metadata: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict) or msg.get("method") != "turn/start":
        return None
    request_id = _request_id(msg)
    if request_id is None:
        return None
    metadata = optimization_metadata or {}
    routing = metadata.get("routing") if isinstance(metadata.get("routing"), dict) else {}
    crunch = metadata.get("crunch") if isinstance(metadata.get("crunch"), dict) else {}
    cache = metadata.get("cache") if isinstance(metadata.get("cache"), dict) else {}
    actions: list[str] = []
    if routing.get("canary") in {"codex-app-summary-model-hint", "codex-app-rule"}:
        actions.append("routing")
    if cache.get("canary") in {"codex-app-exact-cache", "codex-app-rule"} or isinstance(cache.get("canary_sample"), dict):
        actions.append("cache")
    if not actions:
        return None
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    return {
        "request_id": str(request_id),
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
        "actions": actions,
        "input_text_chars": _input_text_chars(params.get("input")),
    }


async def _record_codex_canary_lifecycle_outcome(
    raw: str | bytes,
    *,
    request_started: dict[str, float],
    pending_lifecycle: dict[str, dict[str, Any]],
) -> None:
    msg = _jsonrpc_message(raw)
    if not isinstance(msg, dict):
        return
    request_id = _request_id(msg)
    if request_id is None:
        return
    pending = pending_lifecycle.pop(request_id, None)
    if not pending:
        return
    result = msg.get("result")
    error = msg.get("error")
    started = request_started.get(request_id)
    latency_ms = int((time.time() - started) * 1000) if started is not None else None
    result_chars = len(stable_json(result)) if result is not None else None
    error_code = error.get("code") if isinstance(error, dict) else None
    error_message = error.get("message") if isinstance(error, dict) and isinstance(error.get("message"), str) else None

    feedback_results: dict[str, Any] = {}
    for action in pending.get("actions") or []:
        event = build_codex_app_canary_lifecycle_feedback(
            action_family=str(action),
            routing_meta=pending["routing"],
            crunch_meta=pending["crunch"],
            cache_meta=pending["cache"],
            result_chars=result_chars,
            error_code=error_code,
            error_message=error_message,
            latency_ms=latency_ms,
            input_text_chars=pending.get("input_text_chars"),
        )
        if event is None:
            continue
        meta = await queue_codex_app_canary_lifecycle_feedback(
            store,
            event,
            flush_immediately=False,
        )
        feedback_results[str(action)] = {
            "enabled": bool(meta.get("enabled")),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "endpoint": meta.get("endpoint"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "payload_included": False,
        }
    if not feedback_results:
        return
    pending["routing"]["managed_lifecycle_feedback"] = {
        "schema": "agentflow.codex_app_canary_lifecycle_queue_meta.v1",
        "source_surface": "codex_app_canary_lifecycle",
        "results": feedback_results,
        "payload_included": False,
    }
    start_event_id = pending.get("start_event_id")
    if isinstance(start_event_id, str):
        try:
            store.update_codex_app_event_routing_json(start_event_id, stable_json(pending["routing"]))
        except Exception as exc:
            print(f"AgentFlow Codex app canary lifecycle feedback metadata update skipped: {exc}", file=sys.stderr)


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
        _codex_cache_payload(
            msg,
            ttl_seconds=_as_int(pending.get("ttl_seconds")),
            file_deps=pending.get("file_deps") or [],
        ),
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
    pending_lifecycle: dict[str, dict[str, Any]] = {}
    pending_routing_experiments: dict[str, dict[str, Any]] = {}
    active_turn_windows: dict[str, dict[str, Any]] = {}
    model_states: dict[str, dict[str, Any]] = {}

    try:
        async with websockets.connect(upstream_url, max_size=CODEX_APP_WEBSOCKET_MAX_SIZE) as upstream:
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
                        lifecycle_pending = _attach_codex_canary_lifecycle_pending(forwarded, optimization_metadata)
                        if lifecycle_pending and isinstance(lifecycle_pending.get("request_id"), str):
                            lifecycle_pending["start_event_id"] = start_event_id
                            pending_lifecycle[str(lifecycle_pending["request_id"])] = lifecycle_pending
                        _attach_codex_routing_experiment_pending(
                            forwarded,
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending_routing_experiments,
                        )
                        replay_frame = cache_meta.get(_INTERNAL_REPLAY_FRAME_KEY)
                        if isinstance(replay_frame, str):
                            await _record_codex_managed_outcome(
                                replay_frame,
                                session_id=session_id,
                                request_started=request_started,
                                pending_managed=pending_managed,
                            )
                            await _record_codex_canary_lifecycle_outcome(
                                replay_frame,
                                request_started=request_started,
                                pending_lifecycle=pending_lifecycle,
                            )
                            await _maybe_run_codex_routing_experiment(
                                replay_frame,
                                request_started=request_started,
                                pending_routing_experiments=pending_routing_experiments,
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
                        if isinstance(cache_key, str) and cache_meta.get("outcome_bucket") in {"miss", "invalidated"}:
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
                                    "ttl_seconds": cache_meta.get("ttl_seconds"),
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
                        lifecycle_pending = _attach_codex_canary_lifecycle_pending(forwarded, optimization_metadata)
                        if lifecycle_pending and isinstance(lifecycle_pending.get("request_id"), str):
                            lifecycle_pending["start_event_id"] = start_event_id
                            pending_lifecycle[str(lifecycle_pending["request_id"])] = lifecycle_pending
                        _attach_codex_routing_experiment_pending(
                            forwarded,
                            optimization_metadata=optimization_metadata,
                            start_event_id=start_event_id,
                            pending_routing_experiments=pending_routing_experiments,
                        )
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
                    await _record_codex_canary_lifecycle_outcome(
                        msg,
                        request_started=request_started,
                        pending_lifecycle=pending_lifecycle,
                    )
                    await _maybe_run_codex_routing_experiment(
                        msg,
                        request_started=request_started,
                        pending_routing_experiments=pending_routing_experiments,
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
