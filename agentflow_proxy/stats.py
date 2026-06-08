from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Any, Optional

from agentflow_proxy.limiter import model_tier
from agentflow_proxy.policy_files import policy_file_status
from agentflow_proxy.pricing import codex_app_pricing_basis, estimate_blended_input_savings, estimate_cost
from agentflow_proxy.routing_experiments import ROUTING_EXPERIMENT_MIN_SAMPLES
from agentflow_proxy.store import utc_now

CODEX_APP_PRICING_BASIS = codex_app_pricing_basis()
CODEX_APP_MODEL = str(CODEX_APP_PRICING_BASIS["model"])
CODEX_APP_COST_BASIS = str(CODEX_APP_PRICING_BASIS["cost_basis"])
CODEX_APP_PROCESSING_MODE = str(CODEX_APP_PRICING_BASIS["processing_mode"])
CODEX_APP_COST_KNOWN = bool(CODEX_APP_PRICING_BASIS["cost_known"])
CODEX_APP_TELEMETRY_ONLY_REASON = "codex-app-telemetry-only"
TOKEN_CHARS = 4


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


def _copy_policy(value: Any) -> Any:
    return json.loads(json.dumps(value))


async def stats_policies() -> dict[str, Any]:
    from agentflow_proxy import cache, crunch, router, routing_experiments

    state = {
        "schema": "agentflow.policy_state.v1",
        "routing": {
            "enabled": bool(router.ROUTING_ENABLED),
            "policy_source": router.ROUTING_RULES_SOURCE,
            "rule_path": router.ROUTING_RULES_PATH,
            "file": policy_file_status(
                router.ROUTING_RULES_PATH,
                loaded_at=router.ROUTING_RULES_LOADED_AT,
                loaded_snapshot=router.ROUTING_RULES_LOADED_FILE,
            ),
            "rules": _copy_policy(router.ROUTING_RULES),
            "defaults": {
                "haiku": router.HAIKU_DEFAULT,
                "sonnet": router.SONNET_DEFAULT,
                "opus": router.OPUS_DEFAULT,
            },
            "openai": {
                "enabled": bool(router.OPENAI_ROUTING_ENABLED),
                "large": router.OPENAI_LARGE_DEFAULT,
                "small": router.OPENAI_SMALL_DEFAULT,
                "tiny": router.OPENAI_TINY_DEFAULT,
            },
            "strip_thinking_history": bool(router.STRIP_THINKING_HISTORY),
        },
        "crunch": {
            "enabled": bool(crunch.CRUNCH_ENABLED),
            "policy_source": crunch.CRUNCH_POLICY_SOURCE,
            "rule_path": crunch.CRUNCH_RULES_PATH,
            "file": policy_file_status(
                crunch.CRUNCH_RULES_PATH,
                loaded_at=crunch.CRUNCH_RULES_LOADED_AT,
                loaded_snapshot=crunch.CRUNCH_RULES_LOADED_FILE,
            ),
            "threshold_chars": crunch.CRUNCH_THRESHOLD_CHARS,
            "prompt_cache": {
                "enabled": bool(crunch.PROMPT_CACHE_ENABLED),
                "min_chars": crunch.PROMPT_CACHE_MIN_CHARS,
            },
            "old_context_summarization": _copy_policy(crunch.OLD_CONTEXT_SUMMARY_POLICY),
            "thinking_deduplication": _copy_policy(crunch.THINKING_DEDUP_POLICY),
        },
        "cache": {
            "enabled": bool(cache.CACHE_ENABLED or cache.SEMANTIC_CACHE_ENABLED),
            "policy_source": cache.CACHE_POLICY_SOURCE,
            "rule_path": cache.CACHE_RULES_PATH,
            "file": policy_file_status(
                cache.CACHE_RULES_PATH,
                loaded_at=cache.CACHE_RULES_LOADED_AT,
                loaded_snapshot=cache.CACHE_RULES_LOADED_FILE,
            ),
            "exact_cache": {
                "enabled": bool(cache.CACHE_ENABLED),
                "cache_tool_calls": bool(cache.CACHE_TOOL_CALLS),
            },
            "semantic_cache": {
                "enabled": bool(cache.SEMANTIC_CACHE_ENABLED),
                "threshold": cache.SEMANTIC_CACHE_THRESHOLD,
            },
            "file_watch": {
                "enabled": bool(cache.CACHE_FILE_WATCH_ENABLED),
                "root": cache.CACHE_FILE_WATCH_ROOT,
                "max_paths": cache.CACHE_FILE_WATCH_MAX_PATHS,
            },
        },
        "routing_experiments": {
            "enabled": bool(routing_experiments.ROUTING_EXPERIMENT_ENABLED),
            "policy_source": routing_experiments.ROUTING_EXPERIMENT_POLICY_SOURCE,
            "rule_path": routing_experiments.ROUTING_EXPERIMENT_RULES_PATH,
            "file": policy_file_status(
                routing_experiments.ROUTING_EXPERIMENT_RULES_PATH,
                loaded_at=routing_experiments.ROUTING_EXPERIMENT_RULES_LOADED_AT,
                loaded_snapshot=routing_experiments.ROUTING_EXPERIMENT_RULES_LOADED_FILE,
            ),
            "policy": _copy_policy(routing_experiments.ROUTING_EXPERIMENT_POLICY),
        },
    }
    sections = ("routing", "crunch", "cache", "routing_experiments")
    reload_required_sections = [
        section
        for section in sections
        if bool((state.get(section, {}).get("file") or {}).get("reload_required"))
    ]
    state["summary"] = {
        "policy_count": len(sections),
        "loaded_file_count": sum(
            1
            for section in sections
            if bool(
                (((state.get(section, {}).get("file") or {}).get("loaded") or {}).get("exists"))
            )
        ),
        "manual_policy_count": sum(
            1
            for section in sections
            if state.get(section, {}).get("policy_source") == "local-manual"
        ),
        "local_default_policy_count": sum(
            1
            for section in sections
            if state.get(section, {}).get("policy_source") == "local-default"
        ),
        "reload_required": bool(reload_required_sections),
        "reload_required_sections": reload_required_sections,
    }
    return state


async def stats_policy_events(limit: int = 50) -> dict[str, Any]:
    from agentflow_proxy.policy_events import recent_policy_events

    return recent_policy_events(limit=limit)


def _source_surface(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        if "chat/completions" in path_l:
            return "openai_chat"
        return "openai_responses"
    return "unknown"


def _app_family_for_call(provider: str, requested_model: Any, path: str) -> str:
    provider_l = (provider or "").lower()
    model_l = str(requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in (path or "").lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def estimate_tokens_from_text_chars(chars: Any) -> int:
    char_count = max(_as_int(chars), 0)
    if char_count <= 0:
        return 0
    return max(1, int(char_count / TOKEN_CHARS))


def _codex_turn_estimates(input_text_chars: Any, result_chars: Any) -> dict[str, Any]:
    input_tokens = estimate_tokens_from_text_chars(input_text_chars)
    output_tokens = estimate_tokens_from_text_chars(result_chars)
    cost = estimate_cost(
        CODEX_APP_MODEL,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    cost_known = cost is not None
    cost_value = float(cost) if cost_known else None
    return {
        "model": CODEX_APP_MODEL,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
        "cost_est_usd": cost_value,
        "baseline_cost_est_usd": cost_value,
        "hard_floor_usd": cost_value,
        "cost_basis": CODEX_APP_COST_BASIS,
        "pricing_basis": CODEX_APP_PRICING_BASIS,
        "cost_known": cost_known,
        "cost_estimated": cost_known,
    }


def _codex_estimates_with_cache(input_text_chars: Any, result_chars: Any, cache: dict[str, Any]) -> dict[str, Any]:
    estimates = _codex_turn_estimates(input_text_chars, result_chars)
    if cache.get("status") == "hit":
        baseline = float(estimates["baseline_cost_est_usd"] or estimates["cost_est_usd"] or 0.0)
        estimates["cost_est_usd"] = 0.0
        estimates["hard_floor_usd"] = 0.0
        estimates["baseline_cost_est_usd"] = baseline
        estimates["cache_savings_usd"] = baseline
        estimates["cost_known"] = True
        estimates["cost_estimated"] = True
    else:
        estimates["cache_savings_usd"] = 0.0
    return estimates


def _codex_not_applied_decision(kind: str) -> dict[str, Any]:
    return {
        "status": "not-applied",
        "reason": CODEX_APP_TELEMETRY_ONLY_REASON,
        "policy_source": "local-default",
        "surface": "codex_app_turn",
        "decision_type": kind,
        "applied": False,
    }


def _codex_turn_risk_features(row: dict[str, Any]) -> dict[str, Any]:
    input_items = _as_int(row.get("input_items"))
    input_text_chars = _as_int(row.get("input_text_chars"))
    params_chars = _as_int(row.get("params_chars"))
    method = str(row.get("method") or "turn/start")
    raw_prompt_logging_enabled = os.getenv("AGENTFLOW_LOG_BODIES", "0") == "1"
    return {
        "mutation_safe": False,
        "mutation_safe_reason": CODEX_APP_TELEMETRY_ONLY_REASON,
        "method": method,
        "params_shape": {
            "has_params": params_chars > 0,
            "params_chars": params_chars,
            "has_input": input_items > 0 or input_text_chars > 0,
            "input_items": input_items,
            "input_text_chars": input_text_chars,
        },
        "tool_or_approval_hints": {
            "captured": False,
            "tool_use_present": None,
            "approval_required": None,
            "reason": "raw-params-not-stored",
        },
        "raw_prompt_logging_enabled": raw_prompt_logging_enabled,
        "raw_prompt_stored": False,
        "raw_response_stored": False,
    }


def _sanitize_error_sample(error: Any, limit: int = 180) -> str | None:
    if not error:
        return None
    text = str(error)
    try:
        body = json.loads(text)
    except Exception:
        body = None
    if isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict):
            text = str(error_body.get("message") or error_body.get("code") or error_body.get("type") or text)
        elif error_body:
            text = str(error_body)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return text or None


def _error_type(status_code: Any, error: Any) -> str:
    status = _as_int(status_code)
    sample = (_sanitize_error_sample(error, limit=500) or "").lower()
    if sample.startswith("temporarily limiting requests"):
        return "local_rate_limit"
    if status in (429, 529) or "rate_limit" in sample or "rate limit" in sample:
        return "upstream_rate_limit"
    if "does not support the effort parameter" in sample:
        return "model_incompatible_param"
    if "adaptive thinking is not supported" in sample:
        return "model_incompatible_thinking"
    if "connecterror" in sample or "temporary failure in name resolution" in sample:
        return "network_connect_error"
    if "readtimeout" in sample or "timeout" in sample:
        return "network_timeout"
    if status in (401, 403) or "invalid_api_key" in sample or "incorrect api key" in sample:
        return "auth_error"
    if status:
        return f"http_{status}"
    return "unknown_error"


def _error_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False, limit: int = 30) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        status_code = _as_int(row.get("status_code"))
        error_type = _error_type(status_code, row.get("error"))
        error_sample = _sanitize_error_sample(row.get("error")) or f"HTTP {status_code}"
        model = str(row.get("model") or "")
        tier = model_tier(model)
        key = (
            row.get("provider") or "anthropic",
            status_code,
            tier,
            row.get("requested_model"),
            row.get("routed_model"),
            error_type,
            error_sample,
        )
        bucket = grouped.setdefault(
            key,
            {
                "provider": key[0],
                "status_code": status_code,
                "tier": tier,
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "model": row.get("model"),
                "error_type": error_type,
                "error_sample": error_sample,
                "count": 0,
                "last_seen_at": row.get("created_at"),
            },
        )
        bucket["count"] += 1
        if str(row.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = row.get("created_at")

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda r: (r["count"], str(r.get("last_seen_at") or "")), reverse=True)
    return breakdown[:limit]


def _legacy_cache_decision(row: dict[str, Any]) -> dict[str, str]:
    status_code = _as_int(row.get("status_code"))
    if _as_int(row.get("cache_hit")):
        return {
            "status": "hit",
            "reason": "legacy-cache-hit",
            "hit_type": "exact",
            "policy_source": "legacy-inferred",
            "source_surface": str(row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))),
        }
    if _as_int(row.get("stream")):
        return {
            "status": "skipped",
            "reason": "legacy-streaming",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": str(row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))),
        }
    if status_code >= 400:
        return {
            "status": "skipped",
            "reason": "legacy-upstream-error",
            "hit_type": "",
            "policy_source": "legacy-inferred",
            "source_surface": str(row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))),
        }
    return {
        "status": "missing",
        "reason": "legacy-unknown",
        "hit_type": "",
        "policy_source": "legacy-inferred",
        "source_surface": str(row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or ""))),
    }


def _cache_decision_for_breakdown(row: dict[str, Any]) -> dict[str, str]:
    cache = _json_obj(row.get("cache_json"))
    if cache:
        policy_source = str(cache.get("policy_source") or "unknown")
        source_surface = str(cache.get("surface") or row.get("source_surface") or _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or "")))
        if not cache.get("status") and not cache.get("reason"):
            legacy_hit_type = str(cache.get("hit_type") or "")
            if legacy_hit_type == "skip-streaming":
                return {
                    "status": "skipped",
                    "reason": "legacy-streaming",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            if legacy_hit_type == "miss":
                return {
                    "status": "miss",
                    "reason": "legacy-exact-miss",
                    "hit_type": "",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            if legacy_hit_type == "hit":
                return {
                    "status": "hit",
                    "reason": "legacy-cache-hit",
                    "hit_type": "exact",
                    "policy_source": policy_source,
                    "source_surface": source_surface,
                }
            return {
                "status": "missing",
                "reason": "legacy-partial-cache-json",
                "hit_type": legacy_hit_type,
                "policy_source": policy_source,
                "source_surface": source_surface,
            }
        return {
            "status": str(cache.get("status") or "missing"),
            "reason": str(cache.get("reason") or "unknown"),
            "hit_type": str(cache.get("hit_type") or ""),
            "policy_source": policy_source,
            "source_surface": source_surface,
        }
    return _legacy_cache_decision(row)


def _cache_decision_breakdown(rows: list[dict[str, Any]], *, today_only: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if today_only and not row.get("is_today"):
            continue
        decision = _cache_decision_for_breakdown(row)
        key = (
            decision["source_surface"],
            decision["status"],
            decision["reason"],
            decision["hit_type"],
            decision["policy_source"],
        )
        bucket = grouped.setdefault(
            key,
            {
                "source_surface": key[0],
                "status": key[1],
                "reason": key[2],
                "hit_type": key[3],
                "policy_source": key[4],
                "count": 0,
            },
        )
        bucket["count"] += 1

    breakdown = list(grouped.values())
    breakdown.sort(key=lambda r: r["count"], reverse=True)
    return breakdown


def _usage_bucket_identity(app_family: str, session_id: Any) -> dict[str, Any]:
    engineer = os.getenv("AGENTFLOW_ENGINEER") or None
    app = os.getenv("AGENTFLOW_APP") or app_family or "unknown"
    session = str(session_id or "")
    sid = session[:8] if session else None
    if engineer:
        bucket_id = f"engineer:{engineer}|app:{app}"
        label = f"{engineer} / {app}"
        bucket_kind = "engineer_app"
    elif session:
        bucket_id = f"app:{app}|session:{session}"
        label = f"{app} / session {sid}"
        bucket_kind = "app_session"
    else:
        bucket_id = f"app:{app}|session:unknown"
        label = f"{app} / unknown session"
        bucket_kind = "app_unknown_session"
    return {
        "bucket_id": bucket_id,
        "bucket_label": label,
        "bucket_kind": bucket_kind,
        "engineer": engineer,
        "app": app,
        "app_family": app_family or "unknown",
        "session_id": session or None,
        "sid": sid,
        "label_sources": {
            "engineer": "env:AGENTFLOW_ENGINEER" if engineer else None,
            "app": "env:AGENTFLOW_APP" if os.getenv("AGENTFLOW_APP") else "inferred_app_family",
            "session": "stored_session_id" if session else None,
        },
    }


def _new_usage_bucket(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        "provider_calls": 0,
        "codex_turns": 0,
        "turns": 0,
        "provider_input_tokens": 0,
        "provider_output_tokens": 0,
        "provider_total_tokens": 0,
        "codex_input_text_chars": 0,
        "codex_result_chars": 0,
        "codex_input_tokens_est": 0,
        "codex_output_tokens_est": 0,
        "codex_total_tokens_est": 0,
        "codex_cost_est_usd": 0.0,
        "codex_baseline_cost_est_usd": 0.0,
        "codex_hard_floor_usd": 0.0,
        "codex_exact_cache_savings_usd": 0.0,
        "codex_cost_estimated": False,
        "spend_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "baseline_cost_usd": 0.0,
        "routing_savings_usd": 0.0,
        "crunch_savings_usd": 0.0,
        "cache_savings_usd": 0.0,
        "token_basis": "unknown",
        "cost_basis": "unknown",
        "source_surfaces": [],
        "baseline_provider_cost_usd": 0.0,
        "captured_savings_usd": 0.0,
        "hard_floor_usd": None,
        "provider_cost_known": False,
        "codex_cost_known": False,
        "excludes_unknown_codex_app_cost": False,
        "codex_mutation_safe_turns": 0,
        "codex_telemetry_only_turns": 0,
        "optimized_calls": 0,
        "routed_calls": 0,
        "crunched_calls": 0,
        "local_cache_hits": 0,
        "prompt_cache_read_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_cache_read_savings_usd": 0.0,
        "prompt_cache_creation_cost_usd": 0.0,
        "thinking_tokens": 0,
        "thinking_cost_usd": 0.0,
        "errors": 0,
        "rate_limited": 0,
        "unrouted_high_cost_calls": 0,
        "large_tool_result_calls": 0,
        "context_plateau_pairs": 0,
        "_prev_text_chars_by_session": {},
        "_hint_codes": set(),
        "_token_bases": set(),
        "_cost_bases": set(),
        "_source_surface_counts": {},
        "remaining_saving_potential_hints": [],
    }


def _add_accounting_to_usage_bucket(bucket: dict[str, Any], unit: dict[str, Any]) -> None:
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        bucket[field] += _as_int(unit.get(field))
    for field in (
        "baseline_cost_usd",
        "routing_savings_usd",
        "crunch_savings_usd",
        "cache_savings_usd",
    ):
        bucket[field] += _as_float(unit.get(field))
    bucket["_token_bases"].add(str(unit.get("token_basis") or "unknown"))
    bucket["_cost_bases"].add(str(unit.get("cost_basis") or "unknown"))
    source_surface = str(unit.get("source_surface") or "unknown")
    surface_counts = bucket["_source_surface_counts"]
    surface_counts[source_surface] = surface_counts.get(source_surface, 0) + 1


def _add_usage_hint(bucket: dict[str, Any], code: str, label: str, detail: str) -> None:
    if code in bucket["_hint_codes"]:
        return
    bucket["_hint_codes"].add(code)
    bucket["remaining_saving_potential_hints"].append({
        "code": code,
        "label": label,
        "detail": detail,
    })


def _provider_activity_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    routing = _json_obj(r.get("routing_json"))
    crunch = _json_obj(r.get("crunch_json"))
    cache = _json_obj(r.get("cache_json"))
    provider = str(r.get("provider") or "anthropic")
    requested_model = r.get("requested_model")
    routed_model = r.get("routed_model")
    target_model = routed_model or requested_model
    input_tokens = r.get("actual_input_tokens")
    if input_tokens is None:
        input_tokens = r.get("input_tokens_est")
    output_tokens = r.get("actual_output_tokens")
    if output_tokens is None:
        output_tokens = r.get("output_tokens_est")
    return {
        "unit_id": f"provider_call:{r.get('id')}",
        "created_at": r.get("created_at"),
        "source_surface": _source_surface(provider, str(r.get("path") or "")),
        "granularity": "provider_request",
        "app_family": _app_family_for_call(provider, requested_model, str(r.get("path") or "")),
        "requested_model": requested_model,
        "target_model": target_model,
        "routed_model": routed_model,
        "input_features": {
            "path": r.get("path"),
            "stream": bool(r.get("stream")),
            "category": r.get("category") or routing.get("category"),
            "text_chars": routing.get("text_chars"),
            "input_tokens": input_tokens,
            "input_tokens_est": r.get("input_tokens_est"),
            "actual_input_tokens": r.get("actual_input_tokens"),
            "cache_creation_input_tokens": r.get("cache_creation_input_tokens") or 0,
            "cache_read_input_tokens": r.get("cache_read_input_tokens") or 0,
        },
        "tool_features": {
            "has_tools": routing.get("has_tools"),
            "category": r.get("category") or routing.get("category"),
            "thinking_history_stripped": routing.get("thinking_history_stripped"),
            "stripped_params": routing.get("stripped_params") or [],
        },
        "optimization_features": {
            "routing": routing,
            "crunch": crunch,
            "cache": cache,
            "policy_sources": sorted({
                str(source)
                for source in (
                    routing.get("policy_source"),
                    routing.get("final_policy_source"),
                    crunch.get("policy_source"),
                    cache.get("policy_source"),
                )
                if source
            }),
        },
        "outcome_features": {
            "status_code": r.get("status_code"),
            "latency_ms": r.get("latency_ms"),
            "cache_hit": bool(r.get("cache_hit")),
            "retry_count": r.get("retry_count") or 0,
            "output_tokens": output_tokens,
            "thinking_output_tokens": r.get("thinking_output_tokens") or 0,
            "cost_est_usd": r.get("cost_est_usd"),
            "cost_baseline_usd": r.get("cost_baseline_usd"),
            "error": r.get("error"),
        },
        "replayability_level": "raw_body_opt_in" if r.get("request_json") else "features_only",
        "local_ids": {
            "calls_id": r.get("id"),
            "session_id": r.get("session_id"),
        },
    }


def _codex_turn_activity_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    error_code = r.get("response_error_code")
    response_event_id = r.get("response_event_id")
    status = "error" if error_code is not None else ("success" if response_event_id else "pending")
    routing = _json_obj(r.get("routing_json")) or _codex_not_applied_decision("routing")
    crunch = _json_obj(r.get("crunch_json")) or _codex_not_applied_decision("crunch")
    cache = _json_obj(r.get("cache_json")) or _codex_not_applied_decision("cache")
    estimates = _codex_estimates_with_cache(r.get("input_text_chars"), r.get("response_result_chars"), cache)
    requested_model = routing.get("requested_model") or estimates["model"]
    target_model = routing.get("routed_model") or requested_model
    if crunch.get("tokens_before_est") is not None:
        baseline_input_tokens = _as_int(crunch.get("tokens_before_est"))
        baseline_output_tokens = estimates["output_tokens_est"]
        baseline_cost = estimate_cost(
            requested_model,
            baseline_input_tokens,
            baseline_output_tokens,
            provider="openai",
        )
        if baseline_cost is not None:
            estimates["baseline_cost_est_usd"] = float(baseline_cost)
            if cache.get("status") == "hit":
                estimates["cache_savings_usd"] = float(baseline_cost)
    risk = _codex_turn_risk_features(r)
    policy_sources = sorted({
        str(source)
        for source in (
            routing.get("policy_source"),
            routing.get("final_policy_source"),
            crunch.get("policy_source"),
            cache.get("policy_source"),
        )
        if source
    }) or ["local-default"]
    return {
        "schema": "agentflow.optimization_unit.v1",
        "unit_id": f"codex_turn:{r.get('start_event_id')}",
        "created_at": r.get("created_at"),
        "source_surface": "codex_app_turn",
        "granularity": "agent_turn",
        "app_family": "codex",
        "requested_model": requested_model,
        "target_model": target_model,
        "routed_model": routing.get("routed_model") if routing.get("applied") else None,
        "model_basis": "estimated",
        "input_features": {
            "category": "codex-app-turn",
            "input_text_chars": r.get("input_text_chars") or 0,
            "input_tokens_est": estimates["input_tokens_est"],
            "total_tokens_est": estimates["total_tokens_est"],
            "input_items": r.get("input_items") or 0,
            "params_chars": r.get("params_chars"),
            "message_chars": r.get("message_chars"),
            "cost_basis": estimates["cost_basis"],
        },
        "tool_features": {
            "method": "turn/start",
            "thread_id": r.get("thread_id"),
            "category": "codex-app-turn",
            "tool_or_approval_hints": risk["tool_or_approval_hints"],
            "mutation_safe": risk["mutation_safe"],
            "mutation_safe_reason": risk["mutation_safe_reason"],
        },
        "optimization_features": {
            "routing": routing,
            "crunch": crunch,
            "cache": cache,
            "policy_sources": policy_sources,
            "mutation_safe": risk["mutation_safe"],
            "mutation_safe_reason": risk["mutation_safe_reason"],
        },
        "risk_features": risk,
        "mutation_safe": risk["mutation_safe"],
        "outcome_features": {
            "status": status,
            "latency_ms": r.get("response_latency_ms"),
            "result_chars": r.get("response_result_chars"),
            "output_tokens_est": estimates["output_tokens_est"],
            "total_tokens_est": estimates["total_tokens_est"],
            "cost_est_usd": estimates["cost_est_usd"],
            "cost_baseline_usd": estimates["baseline_cost_est_usd"],
            "hard_floor_usd": estimates["hard_floor_usd"],
            "cache_savings_usd": estimates["cache_savings_usd"],
            "cost_basis": estimates["cost_basis"],
            "pricing_basis": estimates["pricing_basis"],
            "cost_known": estimates["cost_known"],
            "cost_estimated": estimates["cost_estimated"],
            "error_code": error_code,
            "error_message": r.get("response_error_message"),
        },
        "replayability_level": str(cache.get("replayability_level") or "features_only"),
        "local_ids": {
            "codex_app_start_event_id": r.get("start_event_id"),
            "codex_app_response_event_id": response_event_id,
            "request_id": r.get("request_id"),
            "thread_id": r.get("thread_id"),
            "session_id": r.get("session_id"),
        },
    }


def _policy_sources_from(*decisions: dict[str, Any]) -> list[str]:
    return sorted({
        str(source)
        for decision in decisions
        for source in (
            decision.get("policy_source"),
            decision.get("final_policy_source"),
        )
        if source
    })


def _provider_accounting_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    r = dict(row)
    routing = _json_obj(r.get("routing_json"))
    crunch = _json_obj(r.get("crunch_json"))
    cache = _json_obj(r.get("cache_json"))
    provider = str(r.get("provider") or "anthropic").lower()
    path = str(r.get("path") or "")
    requested_model = r.get("requested_model")
    routed_model = r.get("routed_model")
    target_model = routed_model or requested_model
    base_input_tokens = _as_int(
        r.get("actual_input_tokens")
        if r.get("actual_input_tokens") is not None
        else r.get("input_tokens_est")
    )
    output_tokens = _as_int(
        r.get("actual_output_tokens")
        if r.get("actual_output_tokens") is not None
        else r.get("output_tokens_est")
    )
    cache_creation_tokens = _as_int(r.get("cache_creation_input_tokens"))
    cache_read_tokens = _as_int(r.get("cache_read_input_tokens"))
    input_tokens = base_input_tokens + cache_creation_tokens + cache_read_tokens
    cost = _as_float(r.get("cost_est_usd"))
    baseline = _as_float(r.get("cost_baseline_usd")) or cost
    routing_savings = 0.0
    if routed_model and requested_model != routed_model:
        requested_cost = estimate_cost(
            str(requested_model or ""),
            base_input_tokens,
            output_tokens,
            provider=provider,
        ) or 0.0
        routed_cost = estimate_cost(
            str(routed_model or ""),
            base_input_tokens,
            output_tokens,
            provider=provider,
        ) or 0.0
        routing_savings = max(requested_cost - routed_cost, 0.0)

    crunch_tokens_saved = _as_int(crunch.get("tokens_saved_est"))
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else {}
    if summary:
        crunch_tokens_saved += _as_int(summary.get("tokens_saved_est"))
    crunch_gross = estimate_blended_input_savings(
        str(target_model or ""),
        tokens_saved=crunch_tokens_saved,
        input_tokens=base_input_tokens,
        cache_read_tokens=cache_read_tokens,
        provider=provider,
    ) or 0.0
    crunch_savings = max(crunch_gross - _as_float(summary.get("summary_cost_est_usd")), 0.0)

    cache_savings = 0.003 if _as_int(r.get("cache_hit")) else 0.0
    if cache_read_tokens:
        full_read_cost = estimate_cost(str(target_model or ""), cache_read_tokens, 0, provider=provider) or 0.0
        cached_read_input_tokens = cache_read_tokens if provider == "openai" else 0
        cached_read_cost = estimate_cost(
            str(target_model or ""),
            cached_read_input_tokens,
            0,
            cache_read=cache_read_tokens,
            provider=provider,
        ) or 0.0
        cache_savings += max(full_read_cost - cached_read_cost, 0.0)

    token_basis = "provider-reported"
    if r.get("actual_input_tokens") is None and r.get("actual_output_tokens") is None:
        token_basis = "estimated-from-request"

    return {
        "source_surface": _source_surface(provider, path),
        "granularity": "provider_request",
        "app_family": _app_family_for_call(provider, requested_model, path),
        "session_id": r.get("session_id"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "token_basis": token_basis,
        "cost_est_usd": cost,
        "cost_basis": "provider-reported",
        "baseline_cost_usd": baseline,
        "routing_savings_usd": routing_savings,
        "crunch_savings_usd": crunch_savings,
        "cache_savings_usd": cache_savings,
        "hard_floor_usd": cost,
        "policy_sources": _policy_sources_from(routing, crunch, cache),
        "is_today": bool(r.get("is_today")),
    }


def _codex_accounting_unit(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    unit = _codex_turn_activity_unit(row)
    input_features = unit["input_features"]
    outcome_features = unit["outcome_features"]
    optimization_features = unit["optimization_features"]
    cost = _as_float(outcome_features.get("cost_est_usd"))
    baseline = _as_float(outcome_features.get("cost_baseline_usd")) or cost
    cache_savings = _as_float(outcome_features.get("cache_savings_usd"))
    remaining_savings = max(baseline - cost - cache_savings, 0.0)
    routing_savings = remaining_savings if optimization_features["routing"].get("applied") else 0.0
    crunch_savings = 0.0
    if not routing_savings and optimization_features["crunch"].get("changed"):
        crunch_savings = remaining_savings
    return {
        "source_surface": unit["source_surface"],
        "granularity": unit["granularity"],
        "app_family": unit["app_family"],
        "session_id": unit["local_ids"].get("session_id"),
        "input_tokens": _as_int(input_features.get("input_tokens_est")),
        "output_tokens": _as_int(outcome_features.get("output_tokens_est")),
        "total_tokens": _as_int(outcome_features.get("total_tokens_est")),
        "token_basis": "estimated-from-chars",
        "cost_est_usd": cost,
        "cost_basis": str(outcome_features.get("cost_basis") or CODEX_APP_COST_BASIS),
        "baseline_cost_usd": baseline,
        "routing_savings_usd": routing_savings,
        "crunch_savings_usd": crunch_savings,
        "cache_savings_usd": cache_savings,
        "hard_floor_usd": _as_float(outcome_features.get("hard_floor_usd")),
        "policy_sources": list(optimization_features.get("policy_sources") or []),
        "is_today": bool(dict(row).get("is_today")),
    }


def _mixed_label(values: set[str], default: str = "unknown") -> str:
    clean = sorted(value for value in values if value)
    if not clean:
        return default
    if len(clean) == 1:
        return clean[0]
    return "mixed"


def _accounting_rollup(units: list[dict[str, Any]]) -> dict[str, Any]:
    total = {
        "units": 0,
        "provider_calls": 0,
        "codex_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "routing_savings_usd": 0.0,
        "crunch_savings_usd": 0.0,
        "cache_savings_usd": 0.0,
        "hard_floor_usd": 0.0,
        "_token_bases": set(),
        "_cost_bases": set(),
        "_policy_sources": set(),
    }
    by_surface: dict[str, dict[str, Any]] = {}
    savings_by_surface: dict[tuple[str, str], dict[str, Any]] = {}

    def add_common(bucket: dict[str, Any], unit: dict[str, Any]) -> None:
        bucket["units"] += 1
        if unit["granularity"] == "provider_request":
            bucket["provider_calls"] += 1
        if unit["source_surface"] == "codex_app_turn":
            bucket["codex_turns"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            bucket[field] += _as_int(unit.get(field))
        for field in (
            "cost_est_usd",
            "baseline_cost_usd",
            "routing_savings_usd",
            "crunch_savings_usd",
            "cache_savings_usd",
            "hard_floor_usd",
        ):
            bucket[field] += _as_float(unit.get(field))
        bucket["_token_bases"].add(str(unit.get("token_basis") or "unknown"))
        bucket["_cost_bases"].add(str(unit.get("cost_basis") or "unknown"))
        for source in unit.get("policy_sources") or []:
            bucket["_policy_sources"].add(str(source))

    for unit in units:
        add_common(total, unit)
        source_surface = str(unit.get("source_surface") or "unknown")
        bucket = by_surface.setdefault(
            source_surface,
            {
                "source_surface": source_surface,
                "granularities": set(),
                "app_families": set(),
                **{
                    key: 0 for key in (
                        "units",
                        "provider_calls",
                        "codex_turns",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                },
                "cost_est_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "routing_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "cache_savings_usd": 0.0,
                "hard_floor_usd": 0.0,
                "_token_bases": set(),
                "_cost_bases": set(),
                "_policy_sources": set(),
            },
        )
        bucket["granularities"].add(str(unit.get("granularity") or "unknown"))
        bucket["app_families"].add(str(unit.get("app_family") or "unknown"))
        add_common(bucket, unit)
        for optimization_type, field in (
            ("routing", "routing_savings_usd"),
            ("crunching", "crunch_savings_usd"),
            ("cache", "cache_savings_usd"),
        ):
            savings = _as_float(unit.get(field))
            if savings <= 0:
                continue
            key = (source_surface, optimization_type)
            row = savings_by_surface.setdefault(
                key,
                {
                    "source_surface": source_surface,
                    "optimization_type": optimization_type,
                    "savings_usd": 0.0,
                },
            )
            row["savings_usd"] += savings

    def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
        finalized = dict(bucket)
        finalized["token_basis"] = _mixed_label(finalized.pop("_token_bases"))
        finalized["cost_basis"] = _mixed_label(finalized.pop("_cost_bases"))
        finalized["policy_sources"] = sorted(finalized.pop("_policy_sources"))
        if isinstance(finalized.get("granularities"), set):
            finalized["granularities"] = sorted(finalized["granularities"])
        if isinstance(finalized.get("app_families"), set):
            finalized["app_families"] = sorted(finalized["app_families"])
        for field in (
            "cost_est_usd",
            "baseline_cost_usd",
            "routing_savings_usd",
            "crunch_savings_usd",
            "cache_savings_usd",
            "hard_floor_usd",
        ):
            finalized[field] = round(float(finalized[field]), 6)
        return finalized

    savings_rows = []
    for row in savings_by_surface.values():
        savings_rows.append({
            **row,
            "savings_usd": round(float(row["savings_usd"]), 6),
        })
    savings_rows.sort(key=lambda row: (row["source_surface"], row["optimization_type"]))

    source_rows = [finalize(bucket) for bucket in by_surface.values()]
    source_rows.sort(key=lambda row: row["source_surface"])
    return {
        **finalize(total),
        "source_surfaces": source_rows,
        "savings_by_source_surface": savings_rows,
    }



async def stats(store_obj: Any, default_db: str) -> dict[str, Any]:
    conn = store_obj.conn
    calls = conn.execute("select count(*) c from calls").fetchone()["c"]
    cache_hits = conn.execute("select count(*) c from calls where cache_hit = 1").fetchone()["c"]
    routed = conn.execute("select coalesce(provider, 'anthropic') as provider, requested_model, routed_model, count(*) c from calls group by coalesce(provider, 'anthropic'), requested_model, routed_model order by c desc limit 20").fetchall()
    recent = conn.execute("select coalesce(provider, 'anthropic') as provider, created_at, requested_model, routed_model, cache_hit, status_code, latency_ms, cost_est_usd from calls order by created_at desc limit 20").fetchall()
    return {
        "calls": calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": (cache_hits / calls) if calls else 0,
        "db": default_db,
        "routing": [dict(r) for r in routed],
        "recent": [dict(r) for r in recent],
    }


async def stats_limiter(store_obj: Any, tier_status: Any, limiter_config: dict[str, Any]) -> dict[str, Any]:
    conn = store_obj.conn
    recent_rows = conn.execute("""
        select created_at,
               status_code,
               coalesce(routed_model, requested_model) as model,
               coalesce(provider, 'anthropic') as provider,
               retry_count,
               latency_ms,
               error
        from calls
        where status_code in (429, 529)
           or error like 'temporarily limiting requests%'
        order by created_at desc
        limit 50
    """).fetchall()
    recent = []
    last_upstream_by_tier: dict[str, Optional[str]] = {
        "haiku": None,
        "sonnet": None,
        "opus": None,
    }
    local_throttled_recent = 0
    upstream_limited_recent = 0
    for row in recent_rows:
        error = row["error"] or ""
        tier = model_tier(str(row["model"] or ""))
        local_throttled = error.startswith("temporarily limiting requests")
        if local_throttled:
            local_throttled_recent += 1
        else:
            upstream_limited_recent += 1
            if last_upstream_by_tier.get(tier) is None:
                last_upstream_by_tier[tier] = row["created_at"]
        recent.append({
            "created_at": row["created_at"],
            "tier": tier,
            "provider": row["provider"],
            "model": row["model"],
            "status_code": row["status_code"],
            "retry_count": row["retry_count"] or 0,
            "latency_ms": row["latency_ms"],
            "local_throttled": local_throttled,
            "error": error[:240] if error else None,
        })

    tiers = tier_status()
    for tier in tiers:
        tier["last_upstream_429_at"] = last_upstream_by_tier.get(tier["tier"])

    return {
        "generated_at": utc_now(),
        "config": {
            "min_request_interval_ms": limiter_config["min_request_interval_ms"],
            "max_tier_backoff_wait_s": limiter_config["max_tier_backoff_wait_s"],
            "max_concurrent_per_tier": limiter_config["max_concurrent_per_tier"],
        },
        "tiers": tiers,
        "recent_rate_limits": recent,
        "summary": {
            "active_cooldowns": sum(1 for tier in tiers if tier["active"]),
            "local_throttled_recent": local_throttled_recent,
            "upstream_limited_recent": upstream_limited_recent,
        },
    }


async def stats_activity(store_obj: Any, limit: int = 100) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 100), 500))

    provider_rows = conn.execute("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, status_code,
               latency_ms, input_tokens_est, output_tokens_est,
               actual_input_tokens, actual_output_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, error,
               request_json, response_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens, retry_count,
               thinking_output_tokens
        from calls
        order by created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    provider_units = [_provider_activity_unit(row) for row in provider_rows]

    codex_rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms
        from codex_app_events s
        where s.direction = 'client_to_server' and s.method = 'turn/start'
        order by s.created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    codex_units = [_codex_turn_activity_unit(row) for row in codex_rows]

    units = sorted(
        provider_units + codex_units,
        key=lambda unit: str(unit.get("created_at") or ""),
        reverse=True,
    )[:capped_limit]

    def counts_by(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in units:
            value = str(unit.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    return {
        "generated_at": utc_now(),
        "schema": "agentflow.optimization_activity.v1",
        "summary": {
            "units": len(units),
            "provider_request_units": sum(1 for unit in units if unit["granularity"] == "provider_request"),
            "codex_turn_units": sum(1 for unit in units if unit["source_surface"] == "codex_app_turn"),
            "codex_app_turn_units": sum(1 for unit in units if unit["source_surface"] == "codex_app_turn"),
            "by_source_surface": counts_by("source_surface"),
            "by_granularity": counts_by("granularity"),
            "by_app_family": counts_by("app_family"),
            "by_replayability_level": counts_by("replayability_level"),
        },
        "units": units,
    }


async def stats_usage_by_owner(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
    buckets: dict[str, dict[str, Any]] = {}

    def bucket_for(app_family: str, session_id: Any) -> dict[str, Any]:
        identity = _usage_bucket_identity(app_family, session_id)
        bucket = buckets.get(identity["bucket_id"])
        if bucket is None:
            bucket = _new_usage_bucket(identity)
            buckets[identity["bucket_id"]] = bucket
        return bucket

    provider_rows = conn.execute("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, status_code, input_tokens_est,
               output_tokens_est, actual_input_tokens, actual_output_tokens,
               cost_est_usd, cost_baseline_usd, cache_hit, crunch_json,
               routing_json, cache_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens,
               thinking_output_tokens
        from calls
        where date(created_at) = date('now')
        order by coalesce(session_id, ''), created_at
    """).fetchall()

    min_plateau_chars = 8_000
    max_plateau_delta_ratio = 0.03
    high_cost_unrouted_usd = 0.01

    for row in provider_rows:
        r = dict(row)
        provider = str(r.get("provider") or "anthropic").lower()
        requested_model = r.get("requested_model")
        routed_model = r.get("routed_model")
        target_model = routed_model or requested_model
        app_family = _app_family_for_call(provider, requested_model, str(r.get("path") or ""))
        bucket = bucket_for(app_family, r.get("session_id"))
        routing = _json_obj(r.get("routing_json"))
        crunch = _json_obj(r.get("crunch_json"))
        cache = _json_obj(r.get("cache_json"))

        input_tokens = _as_int(r.get("actual_input_tokens") if r.get("actual_input_tokens") is not None else r.get("input_tokens_est"))
        output_tokens = _as_int(r.get("actual_output_tokens") if r.get("actual_output_tokens") is not None else r.get("output_tokens_est"))
        cache_creation_tokens = _as_int(r.get("cache_creation_input_tokens"))
        cache_read_tokens = _as_int(r.get("cache_read_input_tokens"))
        provider_input_tokens = input_tokens + cache_creation_tokens + cache_read_tokens
        cost = _as_float(r.get("cost_est_usd"))
        baseline = _as_float(r.get("cost_baseline_usd")) or cost
        status_code = _as_int(r.get("status_code"))
        category = r.get("category") or routing.get("category") or "unknown"
        text_chars = _as_int(routing.get("text_chars")) or input_tokens * 4
        thinking_tokens = _as_int(r.get("thinking_output_tokens"))
        _add_accounting_to_usage_bucket(bucket, _provider_accounting_unit({**r, "is_today": True}))

        bucket["provider_calls"] += 1
        bucket["turns"] += 1
        bucket["provider_input_tokens"] += provider_input_tokens
        bucket["provider_output_tokens"] += output_tokens
        bucket["provider_total_tokens"] += provider_input_tokens + output_tokens
        bucket["spend_usd"] += cost
        bucket["baseline_provider_cost_usd"] += baseline
        bucket["captured_savings_usd"] += max(baseline - cost, 0.0)
        bucket["provider_cost_known"] = True
        bucket["hard_floor_usd"] = _as_float(bucket["hard_floor_usd"]) + cost
        bucket["prompt_cache_creation_tokens"] += cache_creation_tokens
        bucket["prompt_cache_read_tokens"] += cache_read_tokens
        bucket["thinking_tokens"] += thinking_tokens

        if status_code >= 400:
            bucket["errors"] += 1
        if status_code in (429, 529):
            bucket["rate_limited"] += 1
        if routed_model and requested_model != routed_model:
            bucket["routed_calls"] += 1
        if crunch.get("changed"):
            bucket["crunched_calls"] += 1
        if r.get("cache_hit"):
            bucket["local_cache_hits"] += 1
        if routed_model and requested_model != routed_model or crunch.get("changed") or r.get("cache_hit") or cache_read_tokens:
            bucket["optimized_calls"] += 1
        if (not routed_model or requested_model == routed_model) and cost >= high_cost_unrouted_usd:
            bucket["unrouted_high_cost_calls"] += 1
        if category == "tool-result" and text_chars >= min_plateau_chars:
            bucket["large_tool_result_calls"] += 1

        session_key = str(r.get("session_id") or f"call:{r.get('id')}")
        prev_text = bucket["_prev_text_chars_by_session"].get(session_key)
        if (
            prev_text is not None
            and prev_text >= min_plateau_chars
            and text_chars >= min_plateau_chars
            and abs(text_chars - prev_text) / max(prev_text, 1) <= max_plateau_delta_ratio
        ):
            bucket["context_plateau_pairs"] += 1
        bucket["_prev_text_chars_by_session"][session_key] = text_chars

        if cache_creation_tokens:
            bucket["prompt_cache_creation_cost_usd"] += estimate_cost(
                target_model,
                0,
                0,
                cache_creation=cache_creation_tokens,
                provider=provider,
            ) or 0.0
        if cache_read_tokens:
            full_read_cost = estimate_cost(target_model, cache_read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = cache_read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                target_model,
                cached_read_input_tokens,
                0,
                cache_read=cache_read_tokens,
                provider=provider,
            ) or 0.0
            bucket["prompt_cache_read_savings_usd"] += max(full_read_cost - cached_read_cost, 0.0)
        if thinking_tokens:
            bucket["thinking_cost_usd"] += estimate_cost(target_model, 0, thinking_tokens, provider=provider) or 0.0

    codex_rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.method,
               s.message_chars,
               s.params_chars,
               s.input_items,
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
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code
               ,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
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
          and date(s.created_at) = date('now')
        order by coalesce(s.session_id, ''), s.created_at
    """).fetchall()

    for row in codex_rows:
        r = dict(row)
        unit = _codex_turn_activity_unit(r)
        input_features = unit["input_features"]
        outcome_features = unit["outcome_features"]
        bucket = bucket_for("codex", r.get("session_id"))
        _add_accounting_to_usage_bucket(bucket, _codex_accounting_unit({**r, "is_today": True}))
        bucket["codex_turns"] += 1
        bucket["turns"] += 1
        bucket["codex_input_text_chars"] += _as_int(r.get("input_text_chars"))
        bucket["codex_result_chars"] += _as_int(r.get("response_result_chars"))
        bucket["codex_input_tokens_est"] += _as_int(input_features.get("input_tokens_est"))
        bucket["codex_output_tokens_est"] += _as_int(outcome_features.get("output_tokens_est"))
        bucket["codex_total_tokens_est"] += _as_int(outcome_features.get("total_tokens_est"))
        bucket["codex_cost_est_usd"] += _as_float(outcome_features.get("cost_est_usd"))
        bucket["codex_baseline_cost_est_usd"] += _as_float(outcome_features.get("cost_baseline_usd"))
        bucket["codex_hard_floor_usd"] += _as_float(outcome_features.get("hard_floor_usd"))
        turn_cost_known = bool(outcome_features.get("cost_known"))
        if bucket["codex_turns"] == 1:
            bucket["codex_cost_known"] = turn_cost_known
            bucket["codex_cost_estimated"] = turn_cost_known
        else:
            bucket["codex_cost_known"] = bool(bucket["codex_cost_known"]) and turn_cost_known
            bucket["codex_cost_estimated"] = bool(bucket["codex_cost_estimated"]) and turn_cost_known
        bucket["excludes_unknown_codex_app_cost"] = not bool(bucket["codex_cost_known"])
        bucket["spend_usd"] += _as_float(outcome_features.get("cost_est_usd"))
        codex_saved = max(
            _as_float(outcome_features.get("cost_baseline_usd")) - _as_float(outcome_features.get("cost_est_usd")),
            0.0,
        )
        bucket["captured_savings_usd"] += codex_saved
        cache_decision = unit["optimization_features"]["cache"]
        if cache_decision.get("status") == "hit":
            bucket["local_cache_hits"] += 1
            bucket["codex_exact_cache_savings_usd"] += codex_saved
        if (
            unit["optimization_features"]["routing"].get("applied")
            or unit["optimization_features"]["crunch"].get("changed")
            or cache_decision.get("status") == "hit"
        ):
            bucket["optimized_calls"] += 1
        bucket["hard_floor_usd"] = _as_float(bucket["hard_floor_usd"]) + _as_float(outcome_features.get("hard_floor_usd"))
        if unit.get("mutation_safe"):
            bucket["codex_mutation_safe_turns"] += 1
        if unit["optimization_features"]["routing"].get("reason") == CODEX_APP_TELEMETRY_ONLY_REASON:
            bucket["codex_telemetry_only_turns"] += 1
        if r.get("response_error_code") is not None:
            bucket["errors"] += 1

    rows = []
    for bucket in buckets.values():
        if bucket["context_plateau_pairs"]:
            _add_usage_hint(
                bucket,
                "context_plateau",
                "Repeated context plateau",
                f"{bucket['context_plateau_pairs']} adjacent large-context turns stayed within 3% size.",
            )
        if bucket["thinking_tokens"]:
            _add_usage_hint(
                bucket,
                "thinking_output",
                "High thinking output",
                f"{bucket['thinking_tokens']:,} thinking tokens cost about ${bucket['thinking_cost_usd']:.4f}.",
            )
        if bucket["prompt_cache_creation_cost_usd"] > bucket["prompt_cache_read_savings_usd"] and bucket["prompt_cache_creation_tokens"]:
            _add_usage_hint(
                bucket,
                "cache_warmup",
                "Cache warmup not recouped",
                "Provider prompt-cache writes cost more than reads saved in this bucket today.",
            )
        if bucket["unrouted_high_cost_calls"]:
            _add_usage_hint(
                bucket,
                "unrouted_high_cost",
                "Unrouted high-cost calls",
                f"{bucket['unrouted_high_cost_calls']} provider calls cost at least ${high_cost_unrouted_usd:.2f} and stayed on the requested model.",
            )
        if bucket["large_tool_result_calls"]:
            _add_usage_hint(
                bucket,
                "large_tool_result_context",
                "Large tool-result context",
                f"{bucket['large_tool_result_calls']} tool-result turns carried at least {min_plateau_chars:,} chars.",
            )
        if bucket["rate_limited"]:
            _add_usage_hint(
                bucket,
                "rate_limited",
                "Rate-limit pressure",
                f"{bucket['rate_limited']} turns hit 429/529 responses.",
            )
        elif bucket["errors"]:
            _add_usage_hint(
                bucket,
                "errors",
                "Error signal",
                f"{bucket['errors']} turns returned errors.",
            )
        if bucket["provider_calls"] and not bucket["prompt_cache_read_tokens"] and bucket["provider_input_tokens"] >= 50_000:
            _add_usage_hint(
                bucket,
                "low_prompt_cache_reads",
                "Low prompt-cache reuse",
                "High provider input tokens had no prompt-cache reads today.",
            )

        bucket["spend_usd"] = round(float(bucket["spend_usd"]), 6)
        bucket["baseline_cost_usd"] = round(float(bucket["baseline_cost_usd"]), 6)
        bucket["routing_savings_usd"] = round(float(bucket["routing_savings_usd"]), 6)
        bucket["crunch_savings_usd"] = round(float(bucket["crunch_savings_usd"]), 6)
        bucket["cache_savings_usd"] = round(float(bucket["cache_savings_usd"]), 6)
        bucket["token_basis"] = _mixed_label(bucket["_token_bases"])
        if bucket["provider_calls"] and bucket["codex_turns"]:
            bucket["cost_basis"] = CODEX_APP_COST_BASIS
        elif bucket["provider_calls"]:
            bucket["cost_basis"] = "provider-reported"
        else:
            bucket["cost_basis"] = "codex-estimated-from-chars"
        bucket["source_surfaces"] = [
            {"source_surface": source_surface, "units": count}
            for source_surface, count in sorted(bucket["_source_surface_counts"].items())
        ]
        bucket["baseline_provider_cost_usd"] = round(float(bucket["baseline_provider_cost_usd"]), 6)
        bucket["captured_savings_usd"] = round(float(bucket["captured_savings_usd"]), 6)
        bucket["hard_floor_usd"] = round(float(bucket["hard_floor_usd"]), 6) if bucket["provider_cost_known"] or bucket["codex_cost_known"] else None
        bucket["codex_cost_est_usd"] = round(float(bucket["codex_cost_est_usd"]), 6)
        bucket["codex_baseline_cost_est_usd"] = round(float(bucket["codex_baseline_cost_est_usd"]), 6)
        bucket["codex_hard_floor_usd"] = round(float(bucket["codex_hard_floor_usd"]), 6)
        bucket["codex_exact_cache_savings_usd"] = round(float(bucket["codex_exact_cache_savings_usd"]), 6)
        bucket["prompt_cache_read_savings_usd"] = round(float(bucket["prompt_cache_read_savings_usd"]), 6)
        bucket["prompt_cache_creation_cost_usd"] = round(float(bucket["prompt_cache_creation_cost_usd"]), 6)
        bucket["thinking_cost_usd"] = round(float(bucket["thinking_cost_usd"]), 6)
        bucket["optimization_rate"] = round(bucket["optimized_calls"] / bucket["provider_calls"], 4) if bucket["provider_calls"] else None
        bucket["error_rate"] = round(bucket["errors"] / bucket["turns"], 4) if bucket["turns"] else 0.0
        bucket["potential_hint_count"] = len(bucket["remaining_saving_potential_hints"])
        bucket.pop("_prev_text_chars_by_session", None)
        bucket.pop("_hint_codes", None)
        bucket.pop("_token_bases", None)
        bucket.pop("_cost_bases", None)
        bucket.pop("_source_surface_counts", None)
        rows.append(bucket)

    rows.sort(
        key=lambda row: (
            row["spend_usd"] if row["provider_cost_known"] else -1.0,
            row["provider_total_tokens"],
            row["codex_turns"],
        ),
        reverse=True,
    )

    return {
        "generated_at": utc_now(),
        "schema": "agentflow.usage_by_owner.v1",
        "scope": "today",
        "grouping": {
            "priority": ["AGENTFLOW_ENGINEER", "AGENTFLOW_APP", "app_family", "session_id"],
            "cost_unknown_for": [],
            "raw_prompt_logging": False,
            "codex_cost_basis": CODEX_APP_COST_BASIS,
            "codex_app_model": CODEX_APP_MODEL,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
        },
        "summary": {
            "buckets": len(rows),
            "provider_calls": sum(row["provider_calls"] for row in rows),
            "codex_turns": sum(row["codex_turns"] for row in rows),
            "known_provider_spend_usd": round(
                sum(row["spend_usd"] - row["codex_cost_est_usd"] for row in rows),
                6,
            ),
            "provider_reported_spend_usd": round(
                sum(row["spend_usd"] - row["codex_cost_est_usd"] for row in rows),
                6,
            ),
            "codex_estimated_spend_usd": round(sum(row["codex_cost_est_usd"] for row in rows), 6),
            "codex_exact_cache_savings_usd": round(sum(row["codex_exact_cache_savings_usd"] for row in rows), 6),
            "calculated_spend_usd": round(sum(row["spend_usd"] for row in rows), 6),
            "captured_savings_usd": round(sum(row["captured_savings_usd"] for row in rows), 6),
            "hard_floor_usd": round(sum(row["hard_floor_usd"] or 0.0 for row in rows), 6),
            "codex_cost_unknown": False,
            "cost_basis": CODEX_APP_COST_BASIS if any(row["codex_turns"] for row in rows) else "provider-reported",
        },
        "buckets": rows,
    }


async def stats_full(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn

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
    crunch_savings = 0.0
    today_crunch_savings = 0.0
    crunch_by_model = q("""
        select coalesce(provider, 'anthropic') as provider,
               coalesce(routed_model, requested_model) as model,
               sum(coalesce(json_extract(crunch_json, '$.tokens_saved_est'), 0)) as saved_tok,
               sum(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tok,
               sum(coalesce(cache_read_input_tokens, 0)) as cache_read_tok,
               sum(case when date(created_at) = date('now') then coalesce(json_extract(crunch_json, '$.tokens_saved_est'), 0) else 0 end) as today_saved_tok,
               sum(case when date(created_at) = date('now') then coalesce(actual_input_tokens, input_tokens_est, 0) else 0 end) as today_input_tok,
               sum(case when date(created_at) = date('now') then coalesce(cache_read_input_tokens, 0) else 0 end) as today_cache_read_tok
        from calls
        where json_extract(crunch_json, '$.changed') = 1
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in crunch_by_model:
        crunch_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=int(row["saved_tok"] or 0),
            input_tokens=int(row["input_tok"] or 0),
            cache_read_tokens=int(row["cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
        today_crunch_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=int(row["today_saved_tok"] or 0),
            input_tokens=int(row["today_input_tok"] or 0),
            cache_read_tokens=int(row["today_cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0

    summary_applied_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    today_summary_applied_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
          and date(created_at) = date('now')
    """) or 0)
    summary_created_count = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.reason') = 'summary-created'
    """) or 0)
    summary_cache_hits = int(s("""
        select count(*) from calls
        where json_extract(crunch_json, '$.old_context_summarization.summary_cache_hit') = 1
    """) or 0)
    summary_extra_cost = float(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.summary_cost_est_usd'), 0))
        from calls
    """) or 0.0)
    today_summary_extra_cost = float(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.summary_cost_est_usd'), 0))
        from calls
        where date(created_at) = date('now')
    """) or 0.0)
    summary_chars_saved = int(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.saved_chars'), 0))
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    summary_tokens_saved = int(s("""
        select sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0))
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
    """) or 0)
    summary_savings = 0.0
    today_summary_savings = 0.0
    summary_by_model = q("""
        select coalesce(provider, 'anthropic') as provider,
               coalesce(routed_model, requested_model) as model,
               sum(coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0)) as saved_tok,
               sum(coalesce(actual_input_tokens, input_tokens_est, 0)) as input_tok,
               sum(coalesce(cache_read_input_tokens, 0)) as cache_read_tok,
               sum(case when date(created_at) = date('now') then coalesce(json_extract(crunch_json, '$.old_context_summarization.tokens_saved_est'), 0) else 0 end) as today_saved_tok,
               sum(case when date(created_at) = date('now') then coalesce(actual_input_tokens, input_tokens_est, 0) else 0 end) as today_input_tok,
               sum(case when date(created_at) = date('now') then coalesce(cache_read_input_tokens, 0) else 0 end) as today_cache_read_tok
        from calls
        where json_extract(crunch_json, '$.old_context_summarization.status') = 'applied'
        group by coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
    """)
    for row in summary_by_model:
        summary_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=max(0, int(row["saved_tok"] or 0)),
            input_tokens=int(row["input_tok"] or 0),
            cache_read_tokens=int(row["cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
        today_summary_savings += estimate_blended_input_savings(
            row["model"],
            tokens_saved=max(0, int(row["today_saved_tok"] or 0)),
            input_tokens=int(row["today_input_tok"] or 0),
            cache_read_tokens=int(row["today_cache_read_tok"] or 0),
            provider=row["provider"],
        ) or 0
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

    codex_app_total_events = int(s("select count(*) from codex_app_events") or 0)
    codex_app_today_events = int(s("select count(*) from codex_app_events where date(created_at) = date('now')") or 0)
    codex_app_sessions = int(s("select count(distinct session_id) from codex_app_events where session_id is not null") or 0)
    codex_app_turns = int(s("select count(*) from codex_app_events where direction = 'server_to_client' and method = 'turn/completed'") or 0)
    codex_app_today_turns = int(s("select count(*) from codex_app_events where direction = 'server_to_client' and method = 'turn/completed' and date(created_at) = date('now')") or 0)
    codex_app_last_event_at = s("select max(created_at) from codex_app_events")
    codex_app_input_text_chars = int(s("select sum(input_text_chars) from codex_app_events where direction = 'client_to_server' and method = 'turn/start'") or 0)
    codex_app_today_input_text_chars = int(s("select sum(input_text_chars) from codex_app_events where direction = 'client_to_server' and method = 'turn/start' and date(created_at) = date('now')") or 0)
    codex_app_avg_latency = s("select avg(latency_ms) from codex_app_events where latency_ms is not null") or 0
    codex_turn_rows = q("""
        select s.id as start_event_id,
               s.created_at,
               s.request_id,
               s.thread_id,
               s.session_id,
               s.message_chars,
               s.params_chars,
               s.input_items,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
               s.cache_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.error_message from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_message,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms,
               (date(s.created_at) = date('now')) as is_today
        from codex_app_events s
        where s.direction = 'client_to_server'
          and s.method = 'turn/start'
    """)
    codex_input_tokens_est = 0
    codex_output_tokens_est = 0
    codex_cost_est = 0.0
    codex_cache_savings = 0.0
    codex_cost_known = True
    today_codex_input_tokens_est = 0
    today_codex_output_tokens_est = 0
    today_codex_cost_est = 0.0
    today_codex_cache_savings = 0.0
    today_codex_cost_known = True
    for row in codex_turn_rows:
        cache = _json_obj(row.get("cache_json"))
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), row.get("response_result_chars"), cache)
        codex_input_tokens_est += estimates["input_tokens_est"]
        codex_output_tokens_est += estimates["output_tokens_est"]
        codex_cost_est += _as_float(estimates["cost_est_usd"])
        codex_cache_savings += estimates["cache_savings_usd"]
        codex_cost_known = codex_cost_known and bool(estimates["cost_known"])
        if row.get("is_today"):
            today_codex_input_tokens_est += estimates["input_tokens_est"]
            today_codex_output_tokens_est += estimates["output_tokens_est"]
            today_codex_cost_est += _as_float(estimates["cost_est_usd"])
            today_codex_cache_savings += estimates["cache_savings_usd"]
            today_codex_cost_known = today_codex_cost_known and bool(estimates["cost_known"])
    if not codex_turn_rows:
        codex_cost_known = CODEX_APP_COST_KNOWN
        today_codex_cost_known = CODEX_APP_COST_KNOWN

    provider_input_tokens = int(s("""
        select sum(
            coalesce(actual_input_tokens, input_tokens_est, 0)
            + coalesce(cache_creation_input_tokens, 0)
            + coalesce(cache_read_input_tokens, 0)
        ) from calls
    """) or 0)
    provider_output_tokens = int(s("""
        select sum(coalesce(actual_output_tokens, output_tokens_est, 0))
        from calls
    """) or 0)
    today_provider_input_tokens = int(s("""
        select sum(
            coalesce(actual_input_tokens, input_tokens_est, 0)
            + coalesce(cache_creation_input_tokens, 0)
            + coalesce(cache_read_input_tokens, 0)
        ) from calls
        where date(created_at) = date('now')
    """) or 0)
    today_provider_output_tokens = int(s("""
        select sum(coalesce(actual_output_tokens, output_tokens_est, 0))
        from calls
        where date(created_at) = date('now')
    """) or 0)
    provider_accounting_rows = q("""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, status_code,
               latency_ms, input_tokens_est, output_tokens_est,
               actual_input_tokens, actual_output_tokens, cost_est_usd,
               cost_baseline_usd, crunch_json, routing_json, cache_json, error,
               request_json, response_json, session_id, category,
               cache_creation_input_tokens, cache_read_input_tokens, retry_count,
               thinking_output_tokens,
               (date(created_at) = date('now')) as is_today
        from calls
    """)
    accounting_units = (
        [_provider_accounting_unit(row) for row in provider_accounting_rows]
        + [_codex_accounting_unit(row) for row in codex_turn_rows]
    )
    accounting_total = _accounting_rollup(accounting_units)
    accounting_today = _accounting_rollup([unit for unit in accounting_units if unit.get("is_today")])
    today_crunching_net_savings = today_crunch_savings + (today_summary_savings - today_summary_extra_cost)
    crunching_net_savings = crunch_savings + (summary_savings - summary_extra_cost)
    today_savings_buckets = {
        "routing_usd": round(today_routing_savings, 6),
        "crunching_usd": round(max(0.0, today_crunching_net_savings), 6),
        "exact_local_cache_usd": round(today_cache_savings + today_codex_cache_savings, 6),
        "provider_exact_local_cache_usd": round(today_cache_savings, 6),
        "codex_app_exact_local_cache_usd": round(today_codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(today_prompt_cache_savings, 6),
    }
    savings_buckets = {
        "routing_usd": round(routing_savings, 6),
        "crunching_usd": round(max(0.0, crunching_net_savings), 6),
        "exact_local_cache_usd": round(cache_cost_saved + codex_cache_savings, 6),
        "provider_exact_local_cache_usd": round(cache_cost_saved, 6),
        "codex_app_exact_local_cache_usd": round(codex_cache_savings, 6),
        "provider_prompt_cache_discount_usd": round(prompt_cache_savings, 6),
    }
    today_total_savings = sum(float(value or 0.0) for value in today_savings_buckets.values())
    total_savings = sum(float(value or 0.0) for value in savings_buckets.values())
    today_observed_baseline = today_cost + today_total_savings
    observed_baseline = total_cost + total_savings
    today_calculated_spend = today_cost + today_codex_cost_est
    calculated_spend = total_cost + codex_cost_est
    today_observed_baseline_with_codex = today_observed_baseline + today_codex_cost_est
    observed_baseline_with_codex = observed_baseline + codex_cost_est
    today_hard_floor = today_calculated_spend
    hard_floor = calculated_spend
    executive_summary = {
        "schema": "agentflow.executive_summary.v1",
        "accounting_today": accounting_today,
        "accounting_total": accounting_total,
        "tokens_today": {
            "total_tokens": today_provider_input_tokens + today_provider_output_tokens + today_codex_input_tokens_est + today_codex_output_tokens_est,
            "provider_total_tokens": today_provider_input_tokens + today_provider_output_tokens,
            "provider_input_tokens": today_provider_input_tokens,
            "provider_output_tokens": today_provider_output_tokens,
            "codex_app_turns": codex_app_today_turns,
            "codex_app_input_text_chars": codex_app_today_input_text_chars,
            "codex_app_input_tokens_est": today_codex_input_tokens_est,
            "codex_app_output_tokens_est": today_codex_output_tokens_est,
            "codex_app_total_tokens_est": today_codex_input_tokens_est + today_codex_output_tokens_est,
            "codex_app_cost_known": today_codex_cost_known,
            "codex_app_cost_estimated": today_codex_cost_known,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "tokens_total": {
            "total_tokens": provider_input_tokens + provider_output_tokens + codex_input_tokens_est + codex_output_tokens_est,
            "provider_total_tokens": provider_input_tokens + provider_output_tokens,
            "provider_input_tokens": provider_input_tokens,
            "provider_output_tokens": provider_output_tokens,
            "codex_app_turns": codex_app_turns,
            "codex_app_input_text_chars": codex_app_input_text_chars,
            "codex_app_input_tokens_est": codex_input_tokens_est,
            "codex_app_output_tokens_est": codex_output_tokens_est,
            "codex_app_total_tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "codex_app_cost_known": codex_cost_known,
            "codex_app_cost_estimated": codex_cost_known,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "spend": {
            "today_calculated_spend_usd": round(today_calculated_spend, 6),
            "calculated_spend_usd": round(calculated_spend, 6),
            "today_provider_spend_usd": round(today_cost, 6),
            "total_provider_spend_usd": round(total_cost, 6),
            "today_codex_app_estimated_spend_usd": round(today_codex_cost_est, 6),
            "codex_app_estimated_spend_usd": round(codex_cost_est, 6),
            "today_baseline_provider_cost_usd": round(today_observed_baseline, 6),
            "baseline_provider_cost_usd": round(observed_baseline, 6),
            "today_baseline_calculated_cost_usd": round(today_observed_baseline_with_codex, 6),
            "baseline_calculated_cost_usd": round(observed_baseline_with_codex, 6),
            "thinking_cost_today_usd": round(today_thinking_cost, 6),
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "savings": {
            "today_total_savings_usd": round(today_total_savings, 6),
            "total_savings_usd": round(total_savings, 6),
            "today_buckets": today_savings_buckets,
            "buckets": savings_buckets,
        },
        "hard_floor": {
            "today_unavoidable_provider_spend_usd": round(today_hard_floor, 6),
            "unavoidable_provider_spend_usd": round(hard_floor, 6),
            "today_unavoidable_calculated_spend_usd": round(today_hard_floor, 6),
            "unavoidable_calculated_spend_usd": round(hard_floor, 6),
            "today_baseline_minus_feasible_savings_usd": round(today_observed_baseline_with_codex - today_total_savings, 6),
            "excludes_unknown_codex_app_cost": not today_codex_cost_known,
            "codex_app_cost_estimated": today_codex_cost_known,
            "cost_basis": CODEX_APP_COST_BASIS,
        },
        "health": {
            "errors": errors,
            "avg_latency_ms": round(avg_latency),
            "rate_limit_cooldowns": None,
        },
    }

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

    cache_rows = q("""
        select created_at, stream, cache_hit, status_code, cache_json,
               path, coalesce(provider, 'anthropic') as provider,
               null as source_surface,
               (date(created_at) = date('now')) as is_today
        from calls
        union all
        select created_at, 0 as stream,
               case when json_extract(cache_json, '$.status') = 'hit' then 1 else 0 end as cache_hit,
               null as status_code,
               cache_json,
               'codex-app://turn/start' as path,
               'codex-app' as provider,
               'codex_app_turn' as source_surface,
               (date(created_at) = date('now')) as is_today
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
    """)
    cache_decision_breakdown = _cache_decision_breakdown(cache_rows)
    today_cache_decision_breakdown = _cache_decision_breakdown(cache_rows, today_only=True)

    error_rows = q("""
        select created_at,
               coalesce(provider, 'anthropic') as provider,
               status_code,
               requested_model,
               routed_model,
               coalesce(routed_model, requested_model) as model,
               error,
               (date(created_at) = date('now')) as is_today
        from calls
        where status_code >= 400
        order by created_at desc
    """)
    error_breakdown = _error_breakdown(error_rows)
    today_error_breakdown = _error_breakdown(error_rows, today_only=True)

    routing_experiment_rows = q("""
        select requested_model,
               routed_model,
               coalesce(category, 'unknown') as category,
               coalesce(routing_reason, 'unknown') as routing_reason,
               count(*) as samples,
               sum(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then 1 else 0 end) as compared_samples,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then output_similarity else null end) as avg_similarity,
               avg(case when primary_status_code < 400
                         and shadow_status_code < 400
                         and output_similarity is not null
                        then passed_threshold else null end) as pass_rate,
               round(sum(coalesce(primary_cost_est_usd, 0)), 6) as primary_cost_usd,
               round(sum(coalesce(shadow_cost_est_usd, 0)), 6) as shadow_cost_usd,
               max(created_at) as last_sample_at
        from routing_experiments
        group by requested_model, routed_model, coalesce(category, 'unknown'), coalesce(routing_reason, 'unknown')
        order by samples desc, last_sample_at desc
        limit 20
    """)
    routing_experiment_summary = []
    for row in routing_experiment_rows:
        compared_samples = int(row["compared_samples"] or 0)
        avg_similarity = row["avg_similarity"]
        pass_rate = row["pass_rate"]
        confidence_score = 0.0
        if avg_similarity is not None and compared_samples > 0:
            confidence_score = float(avg_similarity) * min(1.0, compared_samples / ROUTING_EXPERIMENT_MIN_SAMPLES)
        row["compared_samples"] = compared_samples
        row["avg_similarity"] = round(float(avg_similarity), 6) if avg_similarity is not None else None
        row["pass_rate"] = round(float(pass_rate), 4) if pass_rate is not None else None
        row["confidence_score"] = round(confidence_score, 6)
        row["min_samples_for_confidence"] = ROUTING_EXPERIMENT_MIN_SAMPLES
        routing_experiment_summary.append(row)
    routing_experiment_samples = int(s("select count(*) from routing_experiments") or 0)
    routing_experiment_compared = int(s("""
        select count(*) from routing_experiments
        where primary_status_code < 400
          and shadow_status_code < 400
          and output_similarity is not null
    """) or 0)
    routing_experiment_avg_similarity = s("""
        select avg(output_similarity) from routing_experiments
        where primary_status_code < 400
          and shadow_status_code < 400
          and output_similarity is not null
    """)

    provider_breakdown = q("""
        select coalesce(provider, 'anthropic') as provider, count(*) as count,
               round(sum(coalesce(cost_est_usd, 0)), 6) as cost_usd,
               sum(case when requested_model != routed_model and routed_model is not null then 1 else 0 end) as routed_count
        from calls group by coalesce(provider, 'anthropic') order by count desc
    """)
    if codex_app_total_events:
        provider_breakdown.append({
            "provider": "codex-app",
            "count": codex_app_turns,
            "cost_usd": round(codex_cost_est, 6),
            "routed_count": 0,
            "events": codex_app_total_events,
            "tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "cost_basis": "codex-estimated-from-chars",
        })

    codex_app_methods = q("""
        select direction, coalesce(method, '(response)') as method, count(*) as count,
               round(avg(latency_ms)) as avg_latency_ms,
               sum(coalesce(input_text_chars, 0)) as input_text_chars
        from codex_app_events
        group by direction, coalesce(method, '(response)')
        order by count desc
        limit 20
    """)
    codex_app_recent = q("""
        select created_at, direction, coalesce(method, '(response)') as method,
               request_id, thread_id, message_chars, input_items, input_text_chars,
               result_chars, error_code, error_message, latency_ms, session_id
        from codex_app_events
        order by created_at desc
        limit 50
    """)

    return {
        "executive_summary": executive_summary,
        "source_surface_accounting": accounting_total["source_surfaces"],
        "today_source_surface_accounting": accounting_today["source_surfaces"],
        "savings_by_source_surface": accounting_total["savings_by_source_surface"],
        "today_savings_by_source_surface": accounting_today["savings_by_source_surface"],
        "summary": {
            "total_calls": total_calls,
            "today_calls": today_calls,
            "total_cost_usd": round(total_cost, 6),
            "today_cost_usd": round(today_cost, 6),
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_calls, 4) if total_calls else 0,
            "routing_savings_usd": round(routing_savings, 6),
            "today_routing_savings_usd": round(today_routing_savings, 6),
            "cache_savings_usd": round(cache_cost_saved + codex_cache_savings, 6),
            "today_cache_savings_usd": round(today_cache_savings + today_codex_cache_savings, 6),
            "provider_cache_savings_usd": round(cache_cost_saved, 6),
            "today_provider_cache_savings_usd": round(today_cache_savings, 6),
            "codex_app_cache_savings_usd": round(codex_cache_savings, 6),
            "today_codex_app_cache_savings_usd": round(today_codex_cache_savings, 6),
            "total_savings_usd": round(routing_savings + cache_cost_saved + codex_cache_savings, 6),
            "avg_latency_ms": round(avg_latency),
            "routed_count": routed_count,
            "crunched_count": crunched_count,
            "crunch_chars_saved": crunch_chars_saved,
            "crunch_tokens_saved": int(crunch_tokens_saved),
            "crunch_savings_usd": round(crunch_savings, 6),
            "today_crunch_savings_usd": round(today_crunch_savings, 6),
            "avg_crunch_ratio": round(avg_crunch_ratio, 4),
            "old_context_summary_applied_count": summary_applied_count,
            "today_old_context_summary_applied_count": today_summary_applied_count,
            "old_context_summary_created_count": summary_created_count,
            "old_context_summary_cache_hits": summary_cache_hits,
            "old_context_summary_cache_hit_rate": round(summary_cache_hits / summary_applied_count, 4) if summary_applied_count else 0,
            "old_context_summary_chars_saved": summary_chars_saved,
            "old_context_summary_tokens_saved": summary_tokens_saved,
            "old_context_summary_cost_usd": round(summary_extra_cost, 6),
            "today_old_context_summary_cost_usd": round(today_summary_extra_cost, 6),
            "old_context_summary_savings_usd": round(summary_savings, 6),
            "today_old_context_summary_savings_usd": round(today_summary_savings, 6),
            "today_old_context_summary_net_usd": round(today_summary_savings - today_summary_extra_cost, 6),
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
            "codex_app_total_events": codex_app_total_events,
            "codex_app_today_events": codex_app_today_events,
            "codex_app_sessions": codex_app_sessions,
            "codex_app_turns": codex_app_turns,
            "codex_app_today_turns": codex_app_today_turns,
            "codex_app_last_event_at": codex_app_last_event_at,
            "codex_app_input_text_chars": codex_app_input_text_chars,
            "codex_app_input_tokens_est": codex_input_tokens_est,
            "codex_app_output_tokens_est": codex_output_tokens_est,
            "codex_app_total_tokens_est": codex_input_tokens_est + codex_output_tokens_est,
            "codex_app_cost_est_usd": round(codex_cost_est, 6),
            "codex_app_cache_savings_usd": round(codex_cache_savings, 6),
            "today_codex_app_input_tokens_est": today_codex_input_tokens_est,
            "today_codex_app_output_tokens_est": today_codex_output_tokens_est,
            "today_codex_app_total_tokens_est": today_codex_input_tokens_est + today_codex_output_tokens_est,
            "today_codex_app_cost_est_usd": round(today_codex_cost_est, 6),
            "today_codex_app_cache_savings_usd": round(today_codex_cache_savings, 6),
            "codex_app_cost_basis": CODEX_APP_COST_BASIS,
            "codex_app_model": CODEX_APP_MODEL,
            "codex_app_pricing_basis": CODEX_APP_PRICING_BASIS,
            "codex_app_avg_latency_ms": round(codex_app_avg_latency),
            "routing_experiment_samples": routing_experiment_samples,
            "routing_experiment_compared_samples": routing_experiment_compared,
            "routing_experiment_avg_similarity": (
                round(float(routing_experiment_avg_similarity), 6)
                if routing_experiment_avg_similarity is not None else None
            ),
        },
        "recent": recent,
        "routing_breakdown": routing_breakdown,
        "category_breakdown": category_breakdown,
        "cache_decision_breakdown": cache_decision_breakdown,
        "today_cache_decision_breakdown": today_cache_decision_breakdown,
        "error_breakdown": error_breakdown,
        "today_error_breakdown": today_error_breakdown,
        "routing_experiment_summary": routing_experiment_summary,
        "provider_breakdown": provider_breakdown,
        "codex_app_methods": codex_app_methods,
        "codex_app_recent": codex_app_recent,
    }


async def stats_weekly(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
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


async def stats_sessions(store_obj: Any) -> dict[str, Any]:
    conn = store_obj.conn
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
    sessions = [dict(r) for r in rows]
    session_ids = [row["session_id"] for row in sessions]
    plateau_rows = conn.execute("""
        SELECT session_id,
               created_at,
               CAST(coalesce(
                   json_extract(routing_json, '$.text_chars'),
                   coalesce(actual_input_tokens, input_tokens_est, 0) * 4,
                   0
               ) AS INTEGER) as text_chars,
               coalesce(provider, 'anthropic') as provider,
               coalesce(routed_model, requested_model) as model,
               coalesce(cost_est_usd, 0) as cost_usd,
               coalesce(cache_read_input_tokens, 0) as cache_read_tokens,
               coalesce(json_extract(crunch_json, '$.saved_chars'), 0) as crunch_saved_chars
        FROM calls
        WHERE DATE(created_at) = DATE('now') AND session_id IS NOT NULL
        ORDER BY session_id, created_at
    """).fetchall()
    plateau_by_session: dict[str, dict[str, Any]] = {}
    prev_by_session: dict[str, int] = {}
    min_plateau_chars = 8_000
    max_plateau_delta_ratio = 0.03
    flagged_plateau_pairs = 50

    def median_int(values: list[int]) -> int:
        if not values:
            return 0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[mid]
        return int(round((sorted_values[mid - 1] + sorted_values[mid]) / 2))

    def percentile_int(values: list[int], percentile: float) -> int:
        if not values:
            return 0
        sorted_values = sorted(values)
        idx = min(len(sorted_values) - 1, math.ceil((len(sorted_values) - 1) * percentile))
        return sorted_values[idx]

    for row in plateau_rows:
        sid = row["session_id"]
        text_chars = int(row["text_chars"] or 0)
        bucket = plateau_by_session.setdefault(
            sid,
            {
                "session_id": sid,
                "sid": sid[:8],
                "calls": 0,
                "cost_usd": 0.0,
                "plateau_pairs": 0,
                "large_text_values": [],
                "cache_read_savings_usd": 0.0,
                "crunch_saved_chars": 0,
            },
        )
        bucket["calls"] += 1
        bucket["cost_usd"] += float(row["cost_usd"] or 0.0)
        if text_chars >= min_plateau_chars:
            bucket["large_text_values"].append(text_chars)
        prev_text = prev_by_session.get(sid)
        if (
            prev_text is not None
            and prev_text >= min_plateau_chars
            and text_chars >= min_plateau_chars
            and abs(text_chars - prev_text) / max(prev_text, 1) <= max_plateau_delta_ratio
        ):
            bucket["plateau_pairs"] += 1
        prev_by_session[sid] = text_chars

        read_tokens = int(row["cache_read_tokens"] or 0)
        if read_tokens:
            provider = str(row["provider"] or "anthropic").lower()
            full_read_cost = estimate_cost(row["model"], read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                row["model"],
                cached_read_input_tokens,
                0,
                cache_read=read_tokens,
                provider=provider,
            ) or 0.0
            bucket["cache_read_savings_usd"] += max(full_read_cost - cached_read_cost, 0.0)
        bucket["crunch_saved_chars"] += int(row["crunch_saved_chars"] or 0)

    all_plateau_metrics = []
    for bucket in plateau_by_session.values():
        large_text_values = bucket.pop("large_text_values")
        bucket["median_text_chars"] = median_int(large_text_values)
        bucket["p90_text_chars"] = percentile_int(large_text_values, 0.9)
        bucket["cost_usd"] = round(float(bucket["cost_usd"]), 6)
        bucket["cache_read_savings_usd"] = round(float(bucket["cache_read_savings_usd"]), 6)
        bucket["flagged"] = int(bucket["plateau_pairs"]) > flagged_plateau_pairs
        all_plateau_metrics.append(bucket)
    context_plateaus = [
        bucket for bucket in all_plateau_metrics
        if bucket["plateau_pairs"] > 0
    ]
    context_plateaus.sort(key=lambda r: (r["flagged"], r["plateau_pairs"], r["cost_usd"]), reverse=True)
    context_plateaus = context_plateaus[:20]
    plateau_metrics_by_session = {
        row["session_id"]: row
        for row in all_plateau_metrics
    }
    thinking_by_session: dict[str, dict[str, float | int]] = {
        sid: {"thinking_tokens": 0, "thinking_cost_usd": 0.0}
        for sid in session_ids
    }
    prompt_cache_by_session: dict[str, dict[str, float | int]] = {
        sid: {
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_cost_usd": 0.0,
            "cache_read_savings_usd": 0.0,
        }
        for sid in session_ids
    }
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        thinking_rows = conn.execute(f"""
            SELECT session_id,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model, requested_model) as model,
                   SUM(coalesce(thinking_output_tokens, 0)) as thinking_tokens
            FROM calls
            WHERE DATE(created_at) = DATE('now')
              AND session_id IN ({placeholders})
              AND coalesce(thinking_output_tokens, 0) > 0
            GROUP BY session_id, coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
        """, tuple(session_ids)).fetchall()
        for row in thinking_rows:
            sid = row["session_id"]
            tokens = int(row["thinking_tokens"] or 0)
            thinking_by_session[sid]["thinking_tokens"] = int(thinking_by_session[sid]["thinking_tokens"]) + tokens
            thinking_by_session[sid]["thinking_cost_usd"] = float(thinking_by_session[sid]["thinking_cost_usd"]) + (
                estimate_cost(row["model"], 0, tokens, provider=row["provider"]) or 0.0
            )
        prompt_cache_rows = conn.execute(f"""
            SELECT session_id,
                   coalesce(provider, 'anthropic') as provider,
                   coalesce(routed_model, requested_model) as model,
                   SUM(coalesce(cache_creation_input_tokens, 0)) as cache_creation_tokens,
                   SUM(coalesce(cache_read_input_tokens, 0)) as cache_read_tokens
            FROM calls
            WHERE DATE(created_at) = DATE('now')
              AND session_id IN ({placeholders})
              AND (
                  coalesce(cache_creation_input_tokens, 0) > 0
                  OR coalesce(cache_read_input_tokens, 0) > 0
              )
            GROUP BY session_id, coalesce(provider, 'anthropic'), coalesce(routed_model, requested_model)
        """, tuple(session_ids)).fetchall()
        for row in prompt_cache_rows:
            sid = row["session_id"]
            creation_tokens = int(row["cache_creation_tokens"] or 0)
            read_tokens = int(row["cache_read_tokens"] or 0)
            bucket = prompt_cache_by_session[sid]
            bucket["cache_creation_tokens"] = int(bucket["cache_creation_tokens"]) + creation_tokens
            bucket["cache_read_tokens"] = int(bucket["cache_read_tokens"]) + read_tokens

            creation_cost = estimate_cost(
                row["model"],
                0,
                0,
                cache_creation=creation_tokens,
                provider=row["provider"],
            ) or 0.0
            provider = str(row["provider"]).lower()
            full_read_cost = estimate_cost(row["model"], read_tokens, 0, provider=provider) or 0.0
            cached_read_input_tokens = read_tokens if provider == "openai" else 0
            cached_read_cost = estimate_cost(
                row["model"],
                cached_read_input_tokens,
                0,
                cache_read=read_tokens,
                provider=provider,
            ) or 0.0
            bucket["cache_creation_cost_usd"] = float(bucket["cache_creation_cost_usd"]) + creation_cost
            bucket["cache_read_savings_usd"] = float(bucket["cache_read_savings_usd"]) + max(
                full_read_cost - cached_read_cost,
                0.0,
            )
    for row in sessions:
        thinking = thinking_by_session.get(row["session_id"], {})
        row["thinking_tokens"] = int(thinking.get("thinking_tokens", 0) or 0)
        row["thinking_cost_usd"] = round(float(thinking.get("thinking_cost_usd", 0.0) or 0.0), 6)
        prompt_cache = prompt_cache_by_session.get(row["session_id"], {})
        creation_tokens = int(prompt_cache.get("cache_creation_tokens", 0) or 0)
        read_tokens = int(prompt_cache.get("cache_read_tokens", 0) or 0)
        creation_cost = float(prompt_cache.get("cache_creation_cost_usd", 0.0) or 0.0)
        read_savings = float(prompt_cache.get("cache_read_savings_usd", 0.0) or 0.0)
        row["cache_creation_tokens"] = creation_tokens
        row["cache_read_tokens"] = read_tokens
        row["cache_write_read_token_ratio"] = round(creation_tokens / read_tokens, 3) if read_tokens else None
        row["cache_creation_cost_usd"] = round(creation_cost, 6)
        row["cache_read_savings_usd"] = round(read_savings, 6)
        row["cache_warmup_payback_ratio"] = round(creation_cost / read_savings, 3) if read_savings else None
        plateau = plateau_metrics_by_session.get(row["session_id"], {})
        row["plateau_pairs"] = int(plateau.get("plateau_pairs", 0) or 0)
        row["median_text_chars"] = int(plateau.get("median_text_chars", 0) or 0)
        row["p90_text_chars"] = int(plateau.get("p90_text_chars", 0) or 0)
    return {
        "sessions": sessions,
        "context_plateaus": context_plateaus,
        "context_plateau_policy": {
            "min_text_chars": min_plateau_chars,
            "max_delta_ratio": max_plateau_delta_ratio,
            "flagged_plateau_pairs": flagged_plateau_pairs,
        },
    }


def dashboard_html() -> str:
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
  .card .sub{color:#8b949e;font-size:11px;line-height:1.35;margin-top:3px}
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
  .table-wrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse}
  .activity-table{min-width:1080px}
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
  .flags{white-space:normal;min-width:170px}
  .err-row td{background:#1a0a0a}
  .totals-row td{border-top:1px solid #30363d;font-weight:600}
  .savings{color:#3fb950;font-variant-numeric:tabular-nums}
  .baseline{color:#8b949e;font-variant-numeric:tabular-nums}
  #status{margin-left:auto;font-size:11px;color:#8b949e}
  .arrow{color:#8b949e;margin:0 3px}
  @media (max-width:700px){
    header{padding:12px;gap:8px;flex-wrap:wrap}
    .cards{padding:12px;gap:8px}
    .card{min-width:130px;padding:12px}
    .tabs{padding:0 12px;overflow-x:auto}
    .tab-btn{padding:10px 12px;white-space:nowrap}
    .section{padding:0 12px 18px}
  }
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
  <div class="card"><div class="label">Tokens today</div><div class="value" id="c-tokens-today">—</div><div class="sub" id="c-tokens-sub">— provider split</div><div class="sub" id="c-tokens-codex">— Codex telemetry</div></div>
  <div class="card"><div class="label">Calculated spend</div><div class="value" id="c-spend">—</div><div class="sub" id="c-spend-sub">— total</div></div>
  <div class="card green"><div class="label">Savings</div><div class="value" id="c-savings">—</div><div class="sub" id="c-savings-sub">— buckets</div></div>
  <div class="card yellow"><div class="label">Hard floor</div><div class="value" id="c-floor">—</div><div class="sub" id="c-floor-sub">— baseline minus feasible savings</div></div>
  <div class="card blue"><div class="label">Ops health</div><div class="value" id="c-health">—</div><div class="sub" id="c-health-sub">— latency</div><div class="sub" id="c-health-cooldown">— cooldowns</div></div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('activity')">Recent calls</button>
  <button class="tab-btn" onclick="showTab('usage')">Usage by app / engineer</button>
  <button class="tab-btn" onclick="showTab('weekly')">7-day stats</button>
  <button class="tab-btn" onclick="showTab('categories')">By category</button>
  <button class="tab-btn" onclick="showTab('cache')">Cache</button>
  <button class="tab-btn" onclick="showTab('errors')">Errors</button>
  <button class="tab-btn" onclick="showTab('limiter')">Limiter</button>
  <button class="tab-btn" onclick="showTab('policies')">Policies</button>
  <button class="tab-btn" onclick="showTab('sessions')">Sessions</button>
</div>

<div class="tab-panel active" id="tab-activity">
<div class="section">
  <h2>Recent calls</h2>
  <div class="table-wrap">
  <table class="activity-table">
    <thead><tr>
      <th>Time</th><th>Surface</th><th>Granularity</th><th>App family</th><th>Requested</th><th>Target</th><th>Input</th><th>Output / status</th><th>Latency</th><th>Flags</th>
    </tr></thead>
    <tbody id="activity-tbody"></tbody>
  </table>
  </div>
</div>
</div>

<div class="tab-panel" id="tab-usage">
<div class="section">
  <h2>Usage by app / engineer</h2>
  <div class="table-wrap">
  <table class="activity-table">
    <thead><tr>
      <th>Bucket</th><th>Turns</th><th>Provider calls</th><th>Codex turns</th><th>Tokens</th><th>Spend</th><th>Captured savings</th><th>Hard floor</th><th>Optimized</th><th>Errors</th><th>Remaining saving potential</th><th>Cost basis</th>
    </tr></thead>
    <tbody id="usage-tbody"></tbody>
  </table>
  </div>
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

<div class="tab-panel" id="tab-cache">
<div class="section">
  <h2>Cache decisions today</h2>
  <table>
    <thead><tr>
      <th>Surface</th><th>Status</th><th>Reason</th><th>Hit type</th><th>Policy source</th><th>Calls</th>
    </tr></thead>
    <tbody id="cache-today-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Cache decisions all time</h2>
  <table>
    <thead><tr>
      <th>Surface</th><th>Status</th><th>Reason</th><th>Hit type</th><th>Policy source</th><th>Calls</th>
    </tr></thead>
    <tbody id="cache-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-errors">
<div class="section">
  <h2>Errors today</h2>
  <table>
    <thead><tr>
      <th>Type</th><th>Status</th><th>Provider</th><th>Tier</th><th>Requested</th><th>Routed</th><th>Calls</th><th>Last seen</th><th>Sample</th>
    </tr></thead>
    <tbody id="errors-today-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Errors all time</h2>
  <table>
    <thead><tr>
      <th>Type</th><th>Status</th><th>Provider</th><th>Tier</th><th>Requested</th><th>Routed</th><th>Calls</th><th>Last seen</th><th>Sample</th>
    </tr></thead>
    <tbody id="errors-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-limiter">
<div class="section">
  <h2>Tier limiter state</h2>
  <table>
    <thead><tr>
      <th>Tier</th><th>Status</th><th>Remaining</th><th>Cooldown until</th><th>Slots</th><th>Queued</th><th>Last upstream 429</th>
    </tr></thead>
    <tbody id="limiter-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Recent rate limits</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Tier</th><th>Provider</th><th>Status</th><th>Retries</th><th>Latency</th><th>Source</th><th>Error</th>
    </tr></thead>
    <tbody id="limiter-recent-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-policies">
<div class="section">
  <h2>Policy reload summary</h2>
  <table>
    <thead><tr>
      <th>Status</th><th>Policies</th><th>Loaded files</th><th>Manual</th><th>Local default</th><th>Reload needed</th>
    </tr></thead>
    <tbody id="policy-summary-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Effective policy files</h2>
  <table>
    <thead><tr>
      <th>Policy</th><th>Status</th><th>Source</th><th>Rule path</th><th>Effective settings</th>
    </tr></thead>
    <tbody id="policies-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Routing rules</h2>
  <table>
    <thead><tr>
      <th>#</th><th>Conditions</th><th>Action</th>
    </tr></thead>
    <tbody id="routing-rules-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Recent policy events</h2>
  <table>
    <thead><tr>
      <th>Time</th><th>Action</th><th>Status</th><th>Source</th><th>Details</th>
    </tr></thead>
    <tbody id="policy-events-tbody"></tbody>
  </table>
</div>
</div>

<div class="tab-panel" id="tab-sessions">
<div class="section">
  <h2>Sessions today</h2>
  <table>
    <thead><tr>
      <th>Session</th><th>Calls</th><th>Cost</th><th>Thinking</th><th>Thinking cost</th><th>Cache write</th><th>Cache read</th><th>Write/read</th><th>Write cost</th><th>Read saved</th><th>Payback</th><th>tool-result</th><th>tool-heavy</th><th>short-comp</th><th>code-gen</th><th>chat</th><th>other</th>
    </tr></thead>
    <tbody id="sess-tbody"></tbody>
  </table>
</div>
<div class="section">
  <h2>Context plateaus today</h2>
  <table>
    <thead><tr>
      <th>Session</th><th>Calls</th><th>Cost</th><th>Plateau pairs</th><th>Median chars</th><th>P90 chars</th><th>Cache read saved</th><th>Crunch saved chars</th><th>Flag</th>
    </tr></thead>
    <tbody id="plateau-tbody"></tbody>
  </table>
</div>
</div>

<script>
function fmt(n,d=4){if(n==null)return'—';return'$'+n.toFixed(d)}
function fmtMs(n){if(n==null)return'—';return n<1000?n+'ms':(n/1000).toFixed(1)+'s'}
function fmtSec(n){if(n==null)return'—';return n<60?n.toFixed(1)+'s':(n/60).toFixed(1)+'m'}
function fmtTok(n){if(n==null)return'?';if(n>=1000000)return(n/1000000).toFixed(1)+'M';return n>=1000?(n/1000).toFixed(1)+'k':String(n)}
function fmtRatio(n){if(n==null)return'—';return n.toFixed(2)+'x'}
function until(ts){
  if(!ts)return'—';
  const d=Math.ceil((new Date(ts).getTime()-Date.now())/1000);
  if(isNaN(d))return'—';
  if(d<=0)return'now';
  return fmtSec(d);
}
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
function esc(v){
  return String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function shortSurface(s){
  const labels={anthropic_messages:'Claude',openai_chat:'OpenAI chat',openai_responses:'OpenAI',codex_app_turn:'Codex turn',codex_turn:'Codex turn'};
  return labels[s]||s||'unknown';
}
function activityInput(unit){
  const f=unit.input_features||{};
  if(unit.granularity==='provider_request'){
    const tok=f.input_tokens??f.input_tokens_est;
    const cache=f.cache_read_input_tokens||0;
    const text=f.text_chars;
    const parts=[];
    if(tok!=null)parts.push(fmtTok(tok)+' tok');
    if(text!=null)parts.push(fmtTok(text)+' chars');
    if(cache)parts.push(fmtTok(cache)+' cached');
    return parts.join(' · ')||'—';
  }
  const parts=[];
  parts.push(fmtTok(f.input_text_chars||0)+' text chars');
  if(f.input_items!=null)parts.push((f.input_items||0)+' items');
  return parts.join(' · ');
}
function activityOutcome(unit){
  const o=unit.outcome_features||{};
  if(unit.granularity==='provider_request'){
    const status=o.status_code??'—';
    const cls=Number(status)>=400?'err':'hit';
    const out=o.output_tokens!=null?fmtTok(o.output_tokens)+' out':'— out';
    const cost=o.cost_est_usd==null?'cost unknown':fmt(o.cost_est_usd,5);
    return `<span class="badge ${cls}">${status}</span> <span class="tokens">${out}</span> <span class="cost">${cost}</span>`;
  }
  const cls=o.status==='error'?'err':o.status==='pending'?'miss':'hit';
  const chars=o.result_chars!=null?fmtTok(o.result_chars)+' result chars':'turn-level';
  const cost=o.cost_est_usd==null?'cost pending':fmt(o.cost_est_usd,5)+' est';
  return `<span class="badge ${cls}">${esc(o.status||'pending')}</span> <span class="tokens">${chars}</span> <span class="cost">${cost}</span> <span class="badge miss">Codex estimated from chars</span>`;
}
function activityFlags(unit){
  const flags=[];
  const opt=unit.optimization_features||{};
  const routing=opt.routing||{};
  const crunch=opt.crunch||{};
  const cache=opt.cache||{};
  if(unit.granularity==='agent_turn')flags.push('<span class="badge miss">not provider-replayable</span>');
  if(unit.replayability_level)flags.push(`<span class="badge provider">${esc(unit.replayability_level)}</span>`);
  if(routing.routed_model&&routing.routed_model!==routing.requested_model)flags.push('<span class="badge routed">routed</span>');
  if(cache.status)flags.push(`<span class="badge ${cache.status==='hit'?'hit':cache.status==='miss'?'miss':'stream'}">cache ${esc(cache.status)}</span>`);
  if(crunch.changed)flags.push('<span class="badge crunched">crunched</span>');
  const category=(unit.input_features&&unit.input_features.category)||(unit.tool_features&&unit.tool_features.category);
  if(category)flags.push(`<span class="badge provider">${esc(category)}</span>`);
  return flags.join(' ')||'<span class="badge miss">observed</span>';
}
function usageHints(row){
  const hints=row.remaining_saving_potential_hints||[];
  if(!hints.length)return'<span class="badge hit">no obvious signal</span>';
  return hints.slice(0,3).map(h=>`<span class="badge routed" title="${esc(h.detail)}">${esc(h.label)}</span>`).join(' ');
}

function showTab(name){
  const tabs=['activity','usage','weekly','categories','cache','errors','limiter','policies','sessions'];
  tabs.forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===name);
  });
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active',tabs[i]===name);
  });
}

async function refreshUsage(){
  try{
    const r=await fetch('/agentflow/stats/usage');
    const d=await r.json();
    const tb=document.getElementById('usage-tbody');
    const rows=d.buckets||[];
    tb.innerHTML=rows.map(row=>{
      const optimized=row.optimization_rate==null?'—':Math.round(row.optimization_rate*100)+'%';
      const errors=(row.errors||0)>0
        ? `<span class="badge err">${row.errors} (${Math.round((row.error_rate||0)*100)}%)</span>`
        : '<span class="badge hit">0</span>';
      const codexCost=row.codex_cost_estimated?'<span class="badge miss">Codex estimated</span>':'';
      const totalTokens=(row.provider_total_tokens||0)+(row.codex_total_tokens_est||0);
      return `<tr>
        <td><span class="badge provider">${esc(row.bucket_label)}</span></td>
        <td>${(row.turns||0).toLocaleString()}</td>
        <td>${(row.provider_calls||0).toLocaleString()}</td>
        <td>${(row.codex_turns||0).toLocaleString()}</td>
        <td class="tokens">${fmtTok(totalTokens)} total · ${fmtTok(row.provider_total_tokens||0)} provider · ${fmtTok(row.codex_total_tokens_est||0)} Codex est</td>
        <td class="cost">${(row.provider_cost_known||row.codex_cost_known)?fmt(row.spend_usd||0,5):'—'}</td>
        <td class="savings">${fmt(row.captured_savings_usd||0,5)}</td>
        <td class="cost">${row.hard_floor_usd==null?'—':fmt(row.hard_floor_usd,5)}</td>
        <td class="tokens">${optimized}</td>
        <td>${errors}</td>
        <td class="flags">${usageHints(row)}</td>
        <td class="flags"><span class="badge provider">${esc(row.cost_basis)}</span> ${codexCost}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="12" style="color:#8b949e">No app or engineer usage today</td></tr>';
  }catch(e){}
}

async function refreshActivity(){
  try{
    const r=await fetch('/agentflow/stats/activity?limit=100');
    const d=await r.json();
    const tb=document.getElementById('activity-tbody');
    const rows=d.units||[];
    tb.innerHTML=rows.map(unit=>{
      const o=unit.outcome_features||{};
      const err=(o.status_code&&o.status_code>=400)||o.status==='error';
      const requested=unit.requested_model?shortModel(unit.requested_model):(unit.granularity==='agent_turn'?'turn-level':'—');
      const target=unit.target_model?shortModel(unit.target_model):(unit.granularity==='agent_turn'?'not provider-replayable':'—');
      return `<tr class="${err?'err-row':''}">
        <td class="ts">${ago(unit.created_at)}</td>
        <td><span class="badge provider">${esc(shortSurface(unit.source_surface))}</span></td>
        <td><span class="badge stream">${esc(unit.granularity||'unknown')}</span></td>
        <td><span class="badge provider">${esc(unit.app_family||'unknown')}</span></td>
        <td class="model">${esc(requested)}</td>
        <td class="model">${esc(target)}</td>
        <td class="tokens">${esc(activityInput(unit))}</td>
        <td>${activityOutcome(unit)}</td>
        <td class="latency">${fmtMs(o.latency_ms)}</td>
        <td class="flags">${activityFlags(unit)}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="10" style="color:#8b949e">No recent activity yet</td></tr>';
  }catch(e){}
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
    const e=d.executive_summary||{};
    const acct=e.accounting_today||{};
    const acctTotal=e.accounting_total||{};
    const surfaces=acct.source_surfaces||[];
    const toks=e.tokens_today||{};
    const spend=e.spend||{};
    const savings=e.savings||{};
    const buckets=savings.today_buckets||{};
    const floor=e.hard_floor||{};
    const health=e.health||{};
    const sourceText=surfaces.length
      ? surfaces.map(row=>shortSurface(row.source_surface)+': '+fmtTok(row.total_tokens||0)+' '+(row.token_basis||'tokens')).join(' · ')
      : fmtTok(toks.provider_input_tokens||0)+' input · '+fmtTok(toks.provider_output_tokens||0)+' output provider tokens';
    const basisText=surfaces.length
      ? surfaces.map(row=>shortSurface(row.source_surface)+' '+(row.cost_basis||'unknown')).join(' · ')
      : (toks.codex_app_turns||0).toLocaleString()+' Codex turns · '+fmtTok(toks.codex_app_total_tokens_est||0)+' estimated tokens from '+fmtTok(toks.codex_app_input_text_chars||0)+' chars';

    document.getElementById('c-tokens-today').textContent=fmtTok(acct.total_tokens??toks.total_tokens??toks.provider_total_tokens??0);
    document.getElementById('c-tokens-sub').textContent=sourceText;
    document.getElementById('c-tokens-codex').textContent=basisText;
    document.getElementById('c-spend').textContent=fmt(acct.cost_est_usd??spend.today_calculated_spend_usd??spend.today_provider_spend_usd??0,4);
    document.getElementById('c-spend-sub').textContent=fmt(acctTotal.cost_est_usd??spend.calculated_spend_usd??spend.total_provider_spend_usd??0,4)+' total · '+fmt(spend.today_provider_spend_usd||0,4)+' provider reported · '+fmt(spend.today_codex_app_estimated_spend_usd||0,4)+' Codex est';
    document.getElementById('c-savings').textContent=fmt((acct.routing_savings_usd||0)+(acct.crunch_savings_usd||0)+(acct.cache_savings_usd||0)||savings.today_total_savings_usd||0,4);
    document.getElementById('c-savings-sub').textContent='routing '+fmt(acct.routing_savings_usd??buckets.routing_usd??0,4)+' · crunch '+fmt(acct.crunch_savings_usd??buckets.crunching_usd??0,4)+' · cache '+fmt(acct.cache_savings_usd??buckets.exact_local_cache_usd??0,4);
    document.getElementById('c-floor').textContent=fmt(acct.hard_floor_usd??floor.today_unavoidable_provider_spend_usd??0,4);
    document.getElementById('c-floor-sub').textContent='baseline '+fmt(spend.today_baseline_calculated_cost_usd??spend.today_baseline_provider_cost_usd??0,4)+' - feasible savings '+fmt(savings.today_total_savings_usd||0,4)+'; Codex estimated';
    document.getElementById('c-health').textContent=(health.errors||0).toLocaleString()+' errors';
    document.getElementById('c-health-sub').textContent='avg latency '+fmtMs(health.avg_latency_ms||0)+' · '+(s.today_calls||0).toLocaleString()+' provider calls today';

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

async function refreshCache(){
  try{
    const r=await fetch('/agentflow/stats/full');
    const d=await r.json();
    const renderRows=(rows)=>rows.map(row=>`<tr>
      <td><span class="badge provider">${esc(shortSurface(row.source_surface||'unknown'))}</span></td>
      <td><span class="badge ${row.status==='hit'?'hit':row.status==='miss'?'miss':'stream'}">${row.status}</span></td>
      <td class="model">${row.reason||'unknown'}</td>
      <td class="tokens">${row.hit_type||'—'}</td>
      <td><span class="badge provider">${row.policy_source||'unknown'}</span></td>
      <td>${(row.count||0).toLocaleString()}</td>
    </tr>`).join('')||'<tr><td colspan="6" style="color:#8b949e">No cache decision data yet</td></tr>';
    document.getElementById('cache-today-tbody').innerHTML=renderRows(d.today_cache_decision_breakdown||[]);
    document.getElementById('cache-tbody').innerHTML=renderRows(d.cache_decision_breakdown||[]);
  }catch(e){}
}

async function refreshErrors(){
  try{
    const r=await fetch('/agentflow/stats/full');
    const d=await r.json();
    const renderRows=(rows)=>rows.map(row=>`<tr>
      <td><span class="badge err">${esc(row.error_type||'unknown_error')}</span></td>
      <td>${row.status_code>=500?`<span class="badge err">${row.status_code}</span>`:`<span class="badge routed">${row.status_code}</span>`}</td>
      <td><span class="badge provider">${shortProvider(row.provider)}</span></td>
      <td><span class="badge provider">${esc(row.tier||'unknown')}</span></td>
      <td class="model">${esc(shortModel(row.requested_model))}</td>
      <td class="model">${esc(shortModel(row.routed_model))}</td>
      <td>${(row.count||0).toLocaleString()}</td>
      <td class="ts">${row.last_seen_at?ago(row.last_seen_at):'—'}</td>
      <td class="model" title="${esc(row.error_sample||'')}">${esc(row.error_sample||'—')}</td>
    </tr>`).join('')||'<tr><td colspan="9" style="color:#8b949e">No errors recorded</td></tr>';
    document.getElementById('errors-today-tbody').innerHTML=renderRows(d.today_error_breakdown||[]);
    document.getElementById('errors-tbody').innerHTML=renderRows(d.error_breakdown||[]);
  }catch(e){}
}

async function refreshLimiter(){
  try{
    const r=await fetch('/agentflow/stats/limiter');
    const d=await r.json();
    const tiers=d.tiers||[];
    const active=tiers.filter(t=>t.active);
    const queued=tiers.reduce((sum,t)=>sum+(t.queued_count||0),0);
    const longest=active.reduce((max,t)=>Math.max(max,t.seconds_remaining||0),0);
    document.getElementById('c-health-cooldown').textContent=active.length
      ? active.length+' cooldowns · '+queued+' queued · longest '+fmtSec(longest)
      : 'cooldowns clear · '+queued+' queued';

    const tb=document.getElementById('limiter-tbody');
    tb.innerHTML=tiers.map(row=>{
      const badge=row.active
        ? `<span class="badge err">cooldown</span>`
        : `<span class="badge hit">clear</span>`;
      const slots=row.available_slots==null?'—':`${row.available_slots}/${row.max_concurrent}`;
      return `<tr>
        <td><span class="badge provider">${row.tier}</span></td>
        <td>${badge}</td>
        <td class="latency">${fmtSec(row.seconds_remaining||0)}</td>
        <td class="ts">${until(row.cooldown_until)}</td>
        <td class="tokens">${slots}</td>
        <td class="tokens">${row.queued_count||0}</td>
        <td class="ts">${row.last_upstream_429_at?ago(row.last_upstream_429_at):'—'}</td>
      </tr>`;
    }).join('');

    const rb=document.getElementById('limiter-recent-tbody');
    const recent=d.recent_rate_limits||[];
    rb.innerHTML=recent.map(row=>`<tr>
      <td class="ts">${ago(row.created_at)}</td>
      <td><span class="badge provider">${row.tier}</span></td>
      <td><span class="badge provider">${shortProvider(row.provider)}</span></td>
      <td>${row.status_code>=500?`<span class="badge err">${row.status_code}</span>`:`<span class="badge routed">${row.status_code}</span>`}</td>
      <td class="tokens">${row.retry_count||0}</td>
      <td class="latency">${fmtMs(row.latency_ms)}</td>
      <td>${row.local_throttled?'<span class="badge err">local cooldown</span>':'<span class="badge routed">upstream</span>'}</td>
      <td class="model">${row.error||'—'}</td>
    </tr>`).join('')||'<tr><td colspan="8" style="color:#8b949e">No recent rate-limit responses</td></tr>';
  }catch(e){}
}

function policyStatus(enabled){
  return enabled?'<span class="badge hit">enabled</span>':'<span class="badge miss">disabled</span>';
}
function policySource(source){
  const cls=source==='local-manual'?'routed':'provider';
  return `<span class="badge ${cls}">${esc(source||'unknown')}</span>`;
}
function compactSettings(items){
  return items.filter(Boolean).map(item=>`<span class="badge stream">${esc(item)}</span>`).join(' ');
}
function policyReloadSetting(file){
  if(!file) return '';
  return file.reload_required?'reload required':'loaded';
}
function policyReloadBadge(summary){
  if(summary&&summary.reload_required){
    return '<span class="badge err">reload required</span>';
  }
  return '<span class="badge hit">loaded</span>';
}
async function refreshPolicies(){
  try{
    const r=await fetch('/agentflow/stats/policies');
    const d=await r.json();
    const summary=d.summary||{};
    const stale=(summary.reload_required_sections||[]).join(', ');
    document.getElementById('policy-summary-tbody').innerHTML=`<tr>
      <td>${policyReloadBadge(summary)}</td>
      <td class="tokens">${summary.policy_count??'—'}</td>
      <td class="tokens">${summary.loaded_file_count??'—'}</td>
      <td class="tokens">${summary.manual_policy_count??'—'}</td>
      <td class="tokens">${summary.local_default_policy_count??'—'}</td>
      <td class="flags">${stale?`<span class="badge err">${esc(stale)}</span>`:'<span class="badge hit">none</span>'}</td>
    </tr>`;
    const rows=[
      {
        name:'Routing',
        enabled:d.routing&&d.routing.enabled,
        source:d.routing&&d.routing.policy_source,
        path:d.routing&&d.routing.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.routing&&d.routing.file),
          'rules '+((d.routing&&d.routing.rules)||[]).length,
          d.routing&&d.routing.strip_thinking_history?'strip thinking history':'keep thinking history',
          d.routing&&d.routing.openai&&d.routing.openai.enabled?'OpenAI routing on':'OpenAI routing off'
        ])
      },
      {
        name:'Crunch',
        enabled:d.crunch&&d.crunch.enabled,
        source:d.crunch&&d.crunch.policy_source,
        path:d.crunch&&d.crunch.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.crunch&&d.crunch.file),
          'threshold '+fmtTok(d.crunch&&d.crunch.threshold_chars),
          d.crunch&&d.crunch.prompt_cache&&d.crunch.prompt_cache.enabled?'prompt cache on':'prompt cache off',
          d.crunch&&d.crunch.old_context_summarization&&d.crunch.old_context_summarization.enabled?'old-context summary on':'old-context summary off',
          d.crunch&&d.crunch.thinking_deduplication&&d.crunch.thinking_deduplication.enabled?'thinking dedupe on':'thinking dedupe off'
        ])
      },
      {
        name:'Cache',
        enabled:d.cache&&d.cache.enabled,
        source:d.cache&&d.cache.policy_source,
        path:d.cache&&d.cache.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.cache&&d.cache.file),
          d.cache&&d.cache.exact_cache&&d.cache.exact_cache.enabled?'exact on':'exact off',
          d.cache&&d.cache.exact_cache&&d.cache.exact_cache.cache_tool_calls?'tool cache on':'tool cache off',
          d.cache&&d.cache.semantic_cache&&d.cache.semantic_cache.enabled?'semantic on':'semantic off',
          d.cache&&d.cache.file_watch&&d.cache.file_watch.enabled?'file watch on':'file watch off'
        ])
      },
      {
        name:'Routing experiments',
        enabled:d.routing_experiments&&d.routing_experiments.enabled,
        source:d.routing_experiments&&d.routing_experiments.policy_source,
        path:d.routing_experiments&&d.routing_experiments.rule_path,
        settings:compactSettings([
          policyReloadSetting(d.routing_experiments&&d.routing_experiments.file),
          'sample '+(((d.routing_experiments&&d.routing_experiments.policy&&d.routing_experiments.policy.sample_rate)||0)*100).toFixed(1)+'%',
          'similarity '+((d.routing_experiments&&d.routing_experiments.policy&&d.routing_experiments.policy.similarity_threshold)||0)
        ])
      }
    ];
    document.getElementById('policies-tbody').innerHTML=rows.map(row=>`<tr>
      <td><span class="badge provider">${esc(row.name)}</span></td>
      <td>${policyStatus(row.enabled)}</td>
      <td>${policySource(row.source)}</td>
      <td class="model" title="${esc(row.path)}">${esc(row.path)}</td>
      <td class="flags">${row.settings}</td>
    </tr>`).join('');

    const ruleRows=(d.routing&&d.routing.rules)||[];
    document.getElementById('routing-rules-tbody').innerHTML=ruleRows.map((rule,i)=>`<tr>
      <td class="tokens">${i+1}</td>
      <td class="flags">${esc(JSON.stringify(rule.conditions||{}))}</td>
      <td class="flags">${esc(JSON.stringify(rule.action||{}))}</td>
    </tr>`).join('')||'<tr><td colspan="3" style="color:#8b949e">No routing rules loaded</td></tr>';

    const er=await fetch('/agentflow/stats/policy-events?limit=20');
    const ed=await er.json();
    const events=ed.events||[];
    document.getElementById('policy-events-tbody').innerHTML=events.map(event=>{
      const details=event.details||{};
      const parts=[];
      if(details.status_code!=null)parts.push('HTTP '+details.status_code);
      if(details.exit_code!=null)parts.push('exit '+details.exit_code);
      if(details.changed_sections&&details.changed_sections.length)parts.push('changed '+details.changed_sections.join(', '));
      if(details.change_count!=null)parts.push(details.change_count+' changes');
      if(details.error_count!=null)parts.push(details.error_count+' validation errors');
      if(details.reloaded_modules)parts.push(details.reloaded_modules.length+' modules');
      return `<tr>
        <td class="ts">${ago(event.created_at)}</td>
        <td><span class="badge provider">${esc(event.action)}</span></td>
        <td>${event.ok?'<span class="badge hit">ok</span>':'<span class="badge err">failed</span>'}</td>
        <td><span class="badge stream">${esc(details.source||'unknown')}</span></td>
        <td class="flags">${parts.map(p=>`<span class="badge miss">${esc(p)}</span>`).join(' ')||'<span class="badge miss">recorded</span>'}</td>
      </tr>`;
    }).join('')||'<tr><td colspan="5" style="color:#8b949e">No policy operator events recorded</td></tr>';
  }catch(e){}
}

async function refreshSessions(){
  try{
    const r=await fetch('/agentflow/stats/sessions');
    const d=await r.json();
    const tb=document.getElementById('sess-tbody');
    const pb=document.getElementById('plateau-tbody');
    const rows=d.sessions||[];
    tb.innerHTML=rows.map(row=>`<tr>
      <td class="ts">${row.sid}</td>
      <td>${(row.calls||0).toLocaleString()}</td>
      <td class="cost">${fmt(row.cost_usd,5)}</td>
      <td class="tokens">${fmtTok(row.thinking_tokens||0)}</td>
      <td class="cost">${fmt(row.thinking_cost_usd||0,5)}</td>
      <td class="tokens">${fmtTok(row.cache_creation_tokens||0)}</td>
      <td class="tokens">${fmtTok(row.cache_read_tokens||0)}</td>
      <td class="tokens">${fmtRatio(row.cache_write_read_token_ratio)}</td>
      <td class="cost">${fmt(row.cache_creation_cost_usd||0,5)}</td>
      <td class="savings">${fmt(row.cache_read_savings_usd||0,5)}</td>
      <td class="tokens">${fmtRatio(row.cache_warmup_payback_ratio)}</td>
      <td class="tokens">${row.tool_result||0}</td>
      <td class="tokens">${row.tool_heavy||0}</td>
      <td class="tokens">${row.short_completion||0}</td>
      <td class="tokens">${row.code_gen||0}</td>
      <td class="tokens">${row.chat||0}</td>
      <td class="tokens">${row.other||0}</td>
    </tr>`).join('')||'<tr><td colspan="17" style="color:#8b949e">No sessions today</td></tr>';
    const plateaus=d.context_plateaus||[];
    pb.innerHTML=plateaus.map(row=>`<tr>
      <td class="ts">${row.sid}</td>
      <td>${(row.calls||0).toLocaleString()}</td>
      <td class="cost">${fmt(row.cost_usd,5)}</td>
      <td class="tokens">${(row.plateau_pairs||0).toLocaleString()}</td>
      <td class="tokens">${fmtTok(row.median_text_chars||0)}</td>
      <td class="tokens">${fmtTok(row.p90_text_chars||0)}</td>
      <td class="savings">${fmt(row.cache_read_savings_usd||0,5)}</td>
      <td class="tokens">${fmtTok(row.crunch_saved_chars||0)}</td>
      <td>${row.flagged?'<span class="badge err">flagged</span>':'<span class="badge miss">watch</span>'}</td>
    </tr>`).join('')||'<tr><td colspan="9" style="color:#8b949e">No repeated large-context plateaus today</td></tr>';
  }catch(e){}
}

refreshActivity();
refreshUsage();
refresh();
refreshWeekly();
refreshCategories();
refreshCache();
refreshErrors();
refreshLimiter();
refreshPolicies();
refreshSessions();
setInterval(refreshActivity,5000);
setInterval(refreshUsage,30000);
setInterval(refresh,5000);
setInterval(refreshWeekly,30000);
setInterval(refreshCategories,30000);
setInterval(refreshCache,30000);
setInterval(refreshErrors,30000);
setInterval(refreshLimiter,5000);
setInterval(refreshPolicies,30000);
setInterval(refreshSessions,30000);
</script>
</body>
</html>"""
