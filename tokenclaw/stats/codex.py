from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Any

from tokenclaw.codex_turn_policy import CODEX_APP_SOURCE_SURFACE, codex_app_bundle_policy_state
from tokenclaw.golden_path import build_golden_path_summary
from tokenclaw.pricing import estimate_cost
from tokenclaw.quality import derive_codex_turn_quality_signals
from tokenclaw.store import utc_now
from tokenclaw.stats import (
    CODEX_APP_COST_BASIS,
    CODEX_APP_MODEL,
    CODEX_APP_PRICING_BASIS,
    CODEX_APP_PROCESSING_MODE,
    CODEX_APP_TELEMETRY_ONLY_REASON,
    TOKEN_CHARS,
    _as_float,
    _as_int,
    _avg_or_none,
    _count_breakdown,
    _decision_breakdown,
    _increment_count,
    _json_obj,
    _json_obj_has_value,
    _managed_feedback_queue_health,
    _median_int,
    _openai_provider_prompt_cache_discount,
    _parse_utc_datetime,
    _percentile_int,
    estimate_tokens_from_text_chars,
    stats_policies,
)

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
        "surface": CODEX_APP_SOURCE_SURFACE,
        "decision_type": kind,
        "applied": False,
    }

def _codex_turn_risk_features(row: dict[str, Any]) -> dict[str, Any]:
    input_items = _as_int(row.get("input_items"))
    input_text_chars = _as_int(row.get("input_text_chars"))
    params_chars = _as_int(row.get("params_chars"))
    method = str(row.get("method") or "turn/start")
    raw_prompt_logging_enabled = os.getenv("TOKENCLAW_LOG_BODIES", "0") == "1"
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

def _codex_summary_hint_status(routing: dict[str, Any], cache: dict[str, Any]) -> str | None:
    hint = routing.get("summary_model_hint") if isinstance(routing, dict) else None
    canary = routing.get("canary") if isinstance(routing, dict) else None
    if not isinstance(hint, dict) and canary != "codex-app-summary-model-hint":
        return None
    if isinstance(hint, dict):
        status = str(hint.get("status") or "").replace("_", "-").strip().lower()
        if status in {"applied", "holdout", "eligible-skipped", "unsafe-skipped"}:
            return status
    if routing.get("applied") or routing.get("status") == "applied":
        return "applied"
    reason = str(routing.get("reason") or "")
    if bool(cache.get("eligible")) or reason in {
        "summary-model-hint-target-matches-requested",
        "summary-model-hint-target-absent",
    }:
        return "eligible-skipped"
    return "unsafe-skipped"

def _codex_summary_hint_estimated_savings(
    routing: dict[str, Any],
    *,
    input_text_chars: Any,
    result_chars: Any,
    status: str,
) -> float:
    if status != "applied":
        return 0.0
    hint = routing.get("summary_model_hint") if isinstance(routing.get("summary_model_hint"), dict) else {}
    requested_model = str(routing.get("requested_model") or hint.get("requested_model") or "").strip()
    target_model = str(routing.get("routed_model") or routing.get("target_model") or hint.get("target_model") or "").strip()
    if not requested_model or not target_model or requested_model == target_model:
        delta = hint.get("estimated_cost_delta") if isinstance(hint.get("estimated_cost_delta"), dict) else {}
        return max(_as_float(delta.get("delta_usd")), 0.0)
    input_tokens = estimate_tokens_from_text_chars(input_text_chars)
    output_tokens = estimate_tokens_from_text_chars(result_chars)
    requested_cost = estimate_cost(
        requested_model,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    target_cost = estimate_cost(
        target_model,
        input_tokens,
        output_tokens,
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    if requested_cost is None or target_cost is None:
        delta = hint.get("estimated_cost_delta") if isinstance(hint.get("estimated_cost_delta"), dict) else {}
        return max(_as_float(delta.get("delta_usd")), 0.0)
    return max(float(requested_cost) - float(target_cost), 0.0)

def _new_codex_summary_hint_bucket(status: str, phase: str) -> dict[str, Any]:
    return {
        "bucket": status,
        "status": status,
        "workflow_phase": phase,
        "turns": 0,
        "completed": 0,
        "errors": 0,
        "pending": 0,
        "latency_values": [],
        "estimated_savings_usd": 0.0,
        "estimated_input_cost_delta_usd": 0.0,
        "projected_savings_usd": 0.0,
        "cache_hits": 0,
        "cache_eligible": 0,
        "cache_overlap_turns": 0,
        "crunch_applied": 0,
        "crunch_overlap_turns": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "requested_model_counts": {},
        "target_model_counts": {},
        "skip_reason_counts": {},
        "cache_status_counts": {},
        "crunch_status_counts": {},
    }

def _add_codex_summary_hint_bucket(
    buckets: dict[tuple[str, str], dict[str, Any]],
    *,
    phase: str,
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
    input_text_chars: Any,
    result_chars: Any,
    saved_chars: int,
    saved_tokens: int,
    has_response: bool,
    has_error: bool,
    latency: int,
) -> None:
    status = _codex_summary_hint_status(routing, cache)
    if status is None:
        return
    hint = routing.get("summary_model_hint") if isinstance(routing.get("summary_model_hint"), dict) else {}
    hint_phase = str(hint.get("workflow_phase") or routing.get("workflow_phase") or phase or "unknown")
    key = (status, hint_phase)
    bucket = buckets.setdefault(key, _new_codex_summary_hint_bucket(status, hint_phase))
    bucket["turns"] += 1
    if has_error:
        bucket["errors"] += 1
    elif has_response:
        bucket["completed"] += 1
    else:
        bucket["pending"] += 1
    if latency:
        bucket["latency_values"].append(latency)
    savings = _codex_summary_hint_estimated_savings(
        routing,
        input_text_chars=input_text_chars,
        result_chars=result_chars,
        status=status,
    )
    bucket["estimated_savings_usd"] += savings
    delta = hint.get("estimated_cost_delta") if isinstance(hint.get("estimated_cost_delta"), dict) else {}
    input_delta = max(_as_float(delta.get("delta_usd")), 0.0)
    bucket["estimated_input_cost_delta_usd"] += input_delta
    if status in {"applied", "holdout"}:
        bucket["projected_savings_usd"] += input_delta
    if cache.get("status") == "hit":
        bucket["cache_hits"] += 1
    if cache.get("eligible"):
        bucket["cache_eligible"] += 1
    if cache.get("status") in {"hit", "miss"} or cache.get("eligible"):
        bucket["cache_overlap_turns"] += 1
    if crunch.get("applied"):
        bucket["crunch_applied"] += 1
        bucket["crunch_overlap_turns"] += 1
    bucket["saved_chars"] += saved_chars
    bucket["tokens_saved_est"] += saved_tokens
    requested_model = str(routing.get("requested_model") or hint.get("requested_model") or "unknown")
    target_model = str(routing.get("target_model") or routing.get("routed_model") or hint.get("target_model") or "unknown")
    _increment_count(bucket["requested_model_counts"], requested_model)
    _increment_count(bucket["target_model_counts"], target_model)
    _increment_count(bucket["skip_reason_counts"], hint.get("skip_reason") or routing.get("reason") or "none")
    _increment_count(bucket["cache_status_counts"], cache.get("status") or "missing")
    _increment_count(bucket["crunch_status_counts"], crunch.get("status") or "missing")

def _finalize_codex_summary_hint_buckets(
    grouped: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for bucket in grouped.values():
        latency_values = list(bucket.pop("latency_values", []))
        turns = _as_int(bucket.get("turns"))
        errors = _as_int(bucket.get("errors"))
        bucket["error_rate"] = round(errors / turns, 4) if turns else 0
        bucket["avg_latency_ms"] = _avg_or_none(latency_values)
        bucket["estimated_savings_usd"] = round(_as_float(bucket.get("estimated_savings_usd")), 8)
        bucket["estimated_input_cost_delta_usd"] = round(_as_float(bucket.get("estimated_input_cost_delta_usd")), 8)
        bucket["projected_savings_usd"] = round(_as_float(bucket.get("projected_savings_usd")), 8)
        bucket["requested_models"] = _count_breakdown(dict(bucket.pop("requested_model_counts", {})))
        bucket["target_models"] = _count_breakdown(dict(bucket.pop("target_model_counts", {})))
        bucket["skip_reasons"] = _count_breakdown(dict(bucket.pop("skip_reason_counts", {})))
        bucket["cache_statuses"] = _count_breakdown(dict(bucket.pop("cache_status_counts", {})))
        bucket["crunch_statuses"] = _count_breakdown(dict(bucket.pop("crunch_status_counts", {})))
        result.append(bucket)
    order = {"applied": 0, "holdout": 1, "eligible-skipped": 2, "unsafe-skipped": 3}
    result.sort(key=lambda item: (str(item.get("workflow_phase") or ""), order.get(str(item.get("status")), 99)))
    return result

def _codex_summary_hint_status_totals(buckets: list[dict[str, Any]], status: str) -> dict[str, Any]:
    rows = [row for row in buckets if row.get("status") == status]
    turns = sum(_as_int(row.get("turns")) for row in rows)
    errors = sum(_as_int(row.get("errors")) for row in rows)
    completed = sum(_as_int(row.get("completed")) for row in rows)
    pending = sum(_as_int(row.get("pending")) for row in rows)
    lat_weight = 0
    lat_total = 0.0
    for row in rows:
        avg = row.get("avg_latency_ms")
        completed_or_error = _as_int(row.get("completed")) + _as_int(row.get("errors"))
        if avg is not None and completed_or_error > 0:
            lat_weight += completed_or_error
            lat_total += _as_float(avg) * completed_or_error
    return {
        "turns": turns,
        "completed": completed,
        "pending": pending,
        "errors": errors,
        "error_rate": round(errors / turns, 4) if turns else 0,
        "avg_latency_ms": round(lat_total / lat_weight, 2) if lat_weight else None,
        "estimated_savings_usd": round(
            sum(_as_float(row.get("estimated_savings_usd")) for row in rows),
            8,
        ),
        "projected_savings_usd": round(
            sum(_as_float(row.get("projected_savings_usd")) for row in rows),
            8,
        ),
        "estimated_input_cost_delta_usd": round(
            sum(_as_float(row.get("estimated_input_cost_delta_usd")) for row in rows),
            8,
        ),
    }

def _codex_summary_hint_canary_summary(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    applied = _codex_summary_hint_status_totals(buckets, "applied")
    holdout = _codex_summary_hint_status_totals(buckets, "holdout")
    eligible = _codex_summary_hint_status_totals(buckets, "eligible-skipped")
    unsafe = _codex_summary_hint_status_totals(buckets, "unsafe-skipped")
    latency_delta = None
    if applied["avg_latency_ms"] is not None and holdout["avg_latency_ms"] is not None:
        latency_delta = round(_as_float(applied["avg_latency_ms"]) - _as_float(holdout["avg_latency_ms"]), 2)
    return {
        "candidate_count": applied["turns"] + holdout["turns"] + eligible["turns"],
        "applied_count": applied["turns"],
        "holdout_count": holdout["turns"],
        "eligible_skipped_count": eligible["turns"],
        "unsafe_skip_count": unsafe["turns"],
        "error_count": applied["errors"] + holdout["errors"] + eligible["errors"] + unsafe["errors"],
        "applied_error_rate": applied["error_rate"],
        "holdout_error_rate": holdout["error_rate"],
        "applied_minus_holdout_error_rate": round(applied["error_rate"] - holdout["error_rate"], 4),
        "applied_avg_latency_ms": applied["avg_latency_ms"],
        "holdout_avg_latency_ms": holdout["avg_latency_ms"],
        "applied_minus_holdout_latency_avg_ms": latency_delta,
        "applied_estimated_savings_usd": applied["estimated_savings_usd"],
        "holdout_projected_savings_usd": holdout["projected_savings_usd"],
        "candidate_projected_savings_usd": round(
            applied["projected_savings_usd"] + holdout["projected_savings_usd"] + eligible["projected_savings_usd"],
            8,
        ),
        "cost_delta_basis": "metadata-only estimated input/output token costs; holdout savings are projected",
    }

def _codex_crunch_pattern_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        crunch = _json_obj(row.get("crunch_json"))
        patterns = crunch.get("codex_patterns")
        if not isinstance(patterns, list):
            codex_meta = crunch.get("codex_repeated_scaffolding")
            patterns = codex_meta.get("patterns") if isinstance(codex_meta, dict) else []
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            pattern_type = str(pattern.get("type") or "unknown")
            bucket = grouped.setdefault(
                pattern_type,
                {"type": pattern_type, "turns": 0, "count": 0, "saved_chars_est": 0},
            )
            bucket["turns"] += 1
            bucket["count"] += _as_int(pattern.get("count"))
            bucket["saved_chars_est"] += _as_int(pattern.get("saved_chars_est"))
    result = list(grouped.values())
    result.sort(key=lambda r: (r["saved_chars_est"], r["count"]), reverse=True)
    return result

_CODEX_DECISION_KEYS = ("routing_json", "crunch_json", "cache_json")

def _codex_decision_metadata_state(row: dict[str, Any]) -> str:
    present = sum(1 for key in _CODEX_DECISION_KEYS if _json_obj_has_value(row.get(key)))
    if present == len(_CODEX_DECISION_KEYS):
        return "complete"
    if present:
        return "not-instrumented"
    if _json_obj_has_value(row.get("event_window_json")):
        return "current-missing"
    return "historical-unavailable"

def _codex_missing_decision(decision_key: str, metadata_state: str) -> dict[str, Any]:
    decision_name = decision_key.replace("_json", "")
    if metadata_state == "historical-unavailable":
        return {
            "status": "historical-unavailable",
            "reason": f"{decision_name}-decision-metadata-historical-unavailable",
            "applied": False,
            "eligible": False,
            "policy_source": "unknown",
        }
    if metadata_state == "not-instrumented":
        return {
            "status": "not-instrumented",
            "reason": f"{decision_name}-decision-metadata-not-instrumented",
            "applied": False,
            "eligible": False,
            "policy_source": "unknown",
        }
    return {
        "status": "missing",
        "reason": f"{decision_name}-decision-metadata-current-missing",
        "applied": False,
        "eligible": False,
        "policy_source": "unknown",
    }

def _codex_normalized_decision(row: dict[str, Any], decision_key: str, metadata_state: str) -> dict[str, Any]:
    decision = _json_obj(row.get(decision_key))
    if decision:
        return decision
    return _codex_missing_decision(decision_key, metadata_state)

def _codex_model_field_state(routing: dict[str, Any], event_window_raw: Any = None) -> tuple[str, str | None]:
    field = routing.get("model_field")
    if field:
        return "present", str(field)
    reason = str(routing.get("reason") or "")
    if routing.get("requested_model") or routing.get("routed_model"):
        return "present_unknown_field", None
    event_window = _json_obj(event_window_raw)
    window_state = str(event_window.get("model_field_state") or "")
    if window_state in {"derived_present", "derived_absent"}:
        window_field = event_window.get("model_field")
        return window_state, str(window_field) if window_field else None
    model_state = event_window.get("model_state")
    if isinstance(model_state, dict):
        state = str(model_state.get("state") or "")
        if state in {"derived_present", "derived_absent"}:
            field = model_state.get("field")
            return state, str(field) if field else None
    if reason == "codex-turn-start-model-field-absent":
        return "absent", None
    return "unknown", None

def _codex_param_shape_category(row: dict[str, Any], routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    reasons = {
        str(decision.get("reason") or "")
        for decision in (routing, crunch, cache)
        if isinstance(decision, dict)
    }
    if "action-like-params" in reasons:
        return "action-like-params"
    if "unknown-param-shape" in reasons:
        return "unknown-param-shape"
    if "non-text-input" in reasons:
        return "non-text-input"
    if "params-not-object" in reasons:
        return "params-not-object"
    if "codex-app-cache-disabled" in reasons:
        if _as_int(row.get("input_text_chars")) > 0:
            return "text-input-cache-disabled"
        return "cache-disabled-unknown-shape"
    if _as_int(row.get("input_text_chars")) > 0:
        return "text-input"
    if _as_int(row.get("params_chars")) > 0:
        return "params-without-text"
    return "unknown"

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

def _codex_phase_from_signal_counts(
    signal_counts: dict[str, int],
    signal_methods: dict[str, list[str]],
    *,
    reason_prefix: str,
    source: str,
) -> dict[str, Any] | None:
    priority = ("tool_execution", "verification", "planning", "summary", "idle_control")
    for phase in priority:
        if signal_counts.get(phase):
            return {
                "phase": phase,
                "reason": f"{reason_prefix}:{phase}",
                "source": source,
                "signals": signal_methods.get(phase, [])[:5],
            }
    return None

def _codex_signal_counts_from_method_counts(method_counts: Any) -> tuple[dict[str, int], dict[str, list[str]]]:
    signal_counts: dict[str, int] = {}
    signal_methods: dict[str, list[str]] = {}
    if not isinstance(method_counts, dict):
        return signal_counts, signal_methods
    for method, count_raw in method_counts.items():
        signal = _codex_phase_signal(method)
        if not signal:
            continue
        count = _as_int(count_raw)
        if count <= 0:
            count = 1
        signal_counts[signal] = signal_counts.get(signal, 0) + count
        methods = signal_methods.setdefault(signal, [])
        method_s = str(method or "")
        if method_s and method_s not in methods:
            methods.append(method_s)
    return signal_counts, signal_methods

def _codex_public_event_window(raw: Any) -> dict[str, Any]:
    window = _json_obj(raw)
    if not window:
        return {}
    public: dict[str, Any] = {
        "schema": window.get("schema"),
        "event_count": _as_int(window.get("event_count")),
        "method_counts": dict(window.get("method_counts") or {}) if isinstance(window.get("method_counts"), dict) else {},
        "direction_counts": dict(window.get("direction_counts") or {}) if isinstance(window.get("direction_counts"), dict) else {},
        "first_event_delta_ms": _as_int(window.get("first_event_delta_ms")),
        "last_event_delta_ms": _as_int(window.get("last_event_delta_ms")),
        "input_items": window.get("input_items"),
        "input_text_chars": _as_int(window.get("input_text_chars")),
        "start_message_chars": _as_int(window.get("start_message_chars")),
        "start_params_chars": _as_int(window.get("start_params_chars")),
        "result_chars": _as_int(window.get("result_chars")),
        "server_message_chars": _as_int(window.get("server_message_chars")),
        "error_count": _as_int(window.get("error_count")),
        "model_field_state": window.get("model_field_state") or "unknown",
        "model_field": window.get("model_field"),
        "model_state": dict(window.get("model_state") or {}) if isinstance(window.get("model_state"), dict) else {},
        "workflow_phase": window.get("workflow_phase") or "unknown",
        "workflow_phase_reason": window.get("workflow_phase_reason") or "unknown",
        "workflow_phase_source": window.get("workflow_phase_source") or "unknown",
        "workflow_phase_confidence": window.get("workflow_phase_confidence") or "unknown",
        "workflow_phase_signals": list(window.get("workflow_phase_signals") or []),
        "request_id_present": bool(window.get("request_id")),
        "thread_id_present": bool(window.get("thread_id")),
        "session_id_present": bool(window.get("session_id")),
    }
    signal_counts, signal_methods = _codex_signal_counts_from_method_counts(public["method_counts"])
    public["phase_signal_counts"] = dict(signal_counts)
    public["phase_signal_methods"] = {
        phase: methods[:5]
        for phase, methods in signal_methods.items()
    }
    return public

def _codex_same_scope(event: dict[str, Any], row: dict[str, Any]) -> bool:
    row_session = str(row.get("session_id") or "")
    event_session = str(event.get("session_id") or "")
    row_thread = str(row.get("thread_id") or "")
    event_thread = str(event.get("thread_id") or "")
    if row_thread and event_thread:
        return row_thread == event_thread
    if row_session and event_session:
        return row_session == event_session
    return False

def _codex_turn_bounds(turn_rows: list[dict[str, Any]]) -> dict[str, str | None]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        key = (str(row.get("session_id") or ""), str(row.get("thread_id") or ""))
        grouped.setdefault(key, []).append(row)
    bounds: dict[str, str | None] = {}
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda item: str(item.get("created_at") or ""))
        for index, row in enumerate(ordered):
            next_row = ordered[index + 1] if index + 1 < len(ordered) else None
            bounds[str(row.get("start_event_id"))] = str(next_row.get("created_at")) if next_row else None
    return bounds

def _token_drift_bucket(value: int) -> str:
    absolute = abs(int(value or 0))
    if absolute == 0:
        return "zero"
    if absolute < 100:
        return "lt_100"
    if absolute < 1000:
        return "100_999"
    if absolute < 10000:
        return "1k_10k"
    return "10k_plus"

_CODEX_TOKEN_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)

def _codex_count_bucket(value: Any) -> str:
    number = _as_int(value)
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

def _empty_codex_token_totals() -> dict[str, int]:
    return {key: 0 for key in _CODEX_TOKEN_USAGE_KEYS}

def _codex_public_scope_hash(scope_type: str, scope_value: Any) -> str | None:
    value = str(scope_value or "")
    if not value:
        return None
    digest = hashlib.sha256(f"{scope_type}:{value}".encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"

def _codex_public_token_usage(raw_usage: Any) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    result = _empty_codex_token_totals()
    for key in _CODEX_TOKEN_USAGE_KEYS:
        result[key] = max(0, _as_int(usage.get(key)))
    if not result["total_tokens"]:
        result["total_tokens"] = sum(
            result[key]
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
        )
    result["total_tokens_bucket"] = _codex_count_bucket(result["total_tokens"])
    return result

def _codex_token_usage_delta(
    current: dict[str, int],
    previous: dict[str, int] | None,
) -> tuple[dict[str, int], bool]:
    if previous is None:
        return {key: current.get(key, 0) for key in _CODEX_TOKEN_USAGE_KEYS}, False
    reset = any(current.get(key, 0) < previous.get(key, 0) for key in _CODEX_TOKEN_USAGE_KEYS)
    if reset:
        return {key: current.get(key, 0) for key in _CODEX_TOKEN_USAGE_KEYS}, True
    return {
        key: max(0, current.get(key, 0) - previous.get(key, 0))
        for key in _CODEX_TOKEN_USAGE_KEYS
    }, False

def _codex_token_usage_cost(usage: dict[str, int]) -> tuple[float, bool]:
    input_total = _as_int(usage.get("input_tokens")) + _as_int(usage.get("cached_input_tokens"))
    output_total = _as_int(usage.get("output_tokens")) + _as_int(usage.get("reasoning_output_tokens"))
    cost = estimate_cost(
        CODEX_APP_MODEL,
        input_total,
        output_total,
        cache_read=_as_int(usage.get("cached_input_tokens")),
        provider="openai",
        processing_mode=CODEX_APP_PROCESSING_MODE,
    )
    if cost is None:
        return 0.0, False
    return float(cost), True

def _codex_turn_token_estimates(row: dict[str, Any]) -> dict[str, int]:
    input_tokens = max(0, int(_as_int(row.get("input_text_chars")) / TOKEN_CHARS))
    output_tokens = max(0, int(_as_int(row.get("response_result_chars")) / TOKEN_CHARS))
    return {
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
    }

def _codex_turn_model(row: dict[str, Any]) -> str:
    routing = row.get("routing_json_normalized")
    if not isinstance(routing, dict):
        routing = _json_obj(row.get("routing_json"))
    return str(
        routing.get("routed_model")
        or routing.get("target_model")
        or routing.get("requested_model")
        or CODEX_APP_MODEL
    )

def _codex_usage_matching_turn(
    event: dict[str, Any],
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    event_at = str(event.get("created_at") or "")
    event_thread = str(event.get("thread_id") or "")
    event_session = str(event.get("session_id") or "")
    if event_thread:
        candidates = [
            row for row in turn_rows
            if str(row.get("thread_id") or "") == event_thread
        ]
    elif event_session:
        candidates = [
            row for row in turn_rows
            if str(row.get("session_id") or "") == event_session
        ]
    else:
        return None
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: str(row.get("created_at") or ""))
    if event_at and event_at < str(ordered[0].get("created_at") or ""):
        return None
    for index, row in enumerate(ordered):
        next_row = ordered[index + 1] if index + 1 < len(ordered) else None
        start_at = str(row.get("created_at") or "")
        next_at = str(next_row.get("created_at") or "") if next_row else None
        if event_at >= start_at and (next_at is None or event_at < next_at):
            return row
    return ordered[-1]

def _codex_usage_reconciliation_status(
    event: dict[str, Any],
    *,
    matched_turn: dict[str, Any] | None,
    reset: bool,
) -> str:
    if reset:
        return "reset"
    if not str(event.get("thread_id") or ""):
        if str(event.get("session_id") or ""):
            return "aggregate-only"
        return "missing-thread"
    if matched_turn is None:
        return "stale"
    return "reconciled"

def _codex_token_usage_reconciliation_state(status_counts: dict[str, int]) -> str:
    for status in ("reset", "missing-thread", "stale", "aggregate-only"):
        if status_counts.get(status):
            return status
    if status_counts.get("reconciled"):
        return "reconciled"
    return "no-token-usage"

def _add_codex_token_totals(target: dict[str, int], usage: dict[str, int]) -> None:
    for key in _CODEX_TOKEN_USAGE_KEYS:
        target[key] = target.get(key, 0) + _as_int(usage.get(key))

def _codex_quota_token_usage_report(
    event_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rate_limit_updates: list[dict[str, Any]] = []
    token_usage_updates: list[dict[str, Any]] = []
    token_totals = _empty_codex_token_totals()
    raw_counter_totals = _empty_codex_token_totals()
    status_counts: dict[str, int] = {}
    status_token_totals: dict[str, dict[str, int]] = {}
    phase_buckets: dict[str, dict[str, Any]] = {}
    model_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    thread_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    previous_by_scope: dict[tuple[str, str], dict[str, int]] = {}
    reconciled_estimates = {"input_tokens_est": 0, "output_tokens_est": 0, "total_tokens_est": 0}
    reconciled_cost_usd = 0.0
    reconciled_cost_known = True
    for event in event_rows:
        metadata = _json_obj(event.get("metadata_json"))
        kind = metadata.get("kind")
        if kind == "rate_limits":
            rate_limit_updates.append({
                "created_at": event.get("created_at"),
                **metadata,
            })
        elif kind == "token_usage":
            usage = _codex_public_token_usage(metadata.get("token_usage"))
            scope_type = "thread" if event.get("thread_id") else ("session" if event.get("session_id") else "aggregate")
            scope_value = str(event.get("thread_id") or event.get("session_id") or "codex-app")
            scope_key = (scope_type, scope_value)
            delta, reset = _codex_token_usage_delta(usage, previous_by_scope.get(scope_key))
            previous_by_scope[scope_key] = usage
            matched_turn = _codex_usage_matching_turn(event, turn_rows)
            status = _codex_usage_reconciliation_status(event, matched_turn=matched_turn, reset=reset)
            _increment_count(status_counts, status)
            _add_codex_token_totals(status_token_totals.setdefault(status, _empty_codex_token_totals()), delta)
            _add_codex_token_totals(token_totals, delta)
            _add_codex_token_totals(raw_counter_totals, usage)
            turn_estimates = (
                _codex_turn_token_estimates(matched_turn)
                if matched_turn is not None and status == "reconciled"
                else {"input_tokens_est": 0, "output_tokens_est": 0, "total_tokens_est": 0}
            )
            for key in reconciled_estimates:
                reconciled_estimates[key] += _as_int(turn_estimates.get(key))
            cost, cost_known = _codex_token_usage_cost(delta)
            reconciled_cost_usd += cost
            reconciled_cost_known = reconciled_cost_known and cost_known
            phase = (
                str((matched_turn or {}).get("workflow_phase") or "unknown")
                if matched_turn is not None and status == "reconciled"
                else status
            )
            model = _codex_turn_model(matched_turn or {}) if matched_turn is not None and status == "reconciled" else CODEX_APP_MODEL
            phase_bucket = phase_buckets.setdefault(
                phase,
                {
                    "workflow_phase": phase,
                    "updates": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 0,
                    "tokenclaw_total_tokens_est": 0,
                    "cost_usd": 0.0,
                    "status_counts": {},
                },
            )
            phase_bucket["updates"] += 1
            for key in _CODEX_TOKEN_USAGE_KEYS:
                phase_bucket[key] += _as_int(delta.get(key))
            phase_bucket["tokenclaw_total_tokens_est"] += _as_int(turn_estimates.get("total_tokens_est"))
            phase_bucket["cost_usd"] += cost
            _increment_count(phase_bucket["status_counts"], status)
            model_key = (model, CODEX_APP_PROCESSING_MODE)
            model_bucket = model_buckets.setdefault(
                model_key,
                {
                    "model": model,
                    "processing_mode": CODEX_APP_PROCESSING_MODE,
                    "updates": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "status_counts": {},
                },
            )
            model_bucket["updates"] += 1
            for key in _CODEX_TOKEN_USAGE_KEYS:
                model_bucket[key] += _as_int(delta.get(key))
            model_bucket["cost_usd"] += cost
            _increment_count(model_bucket["status_counts"], status)
            public_hash = _codex_public_scope_hash(scope_type, scope_value)
            if public_hash:
                thread_key = (scope_type, public_hash)
                thread_bucket = thread_buckets.setdefault(
                    thread_key,
                    {
                        "scope_type": scope_type,
                        "scope_hash": public_hash,
                        "thread_id_included": False,
                        "session_id_included": False,
                        "updates": 0,
                        "input_tokens": 0,
                        "cached_input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning_output_tokens": 0,
                        "total_tokens": 0,
                        "tokenclaw_total_tokens_est": 0,
                        "cost_usd": 0.0,
                        "status_counts": {},
                    },
                )
                thread_bucket["updates"] += 1
                for key in _CODEX_TOKEN_USAGE_KEYS:
                    thread_bucket[key] += _as_int(delta.get(key))
                thread_bucket["tokenclaw_total_tokens_est"] += _as_int(turn_estimates.get("total_tokens_est"))
                thread_bucket["cost_usd"] += cost
                _increment_count(thread_bucket["status_counts"], status)
            token_usage_updates.append({
                "created_at": event.get("created_at"),
                "method": metadata.get("method"),
                "kind": metadata.get("kind"),
                "thread_id_present": bool(event.get("thread_id") or metadata.get("thread_id_present")),
                "session_id_present": bool(event.get("session_id")),
                "scope_type": scope_type,
                "scope_hash": _codex_public_scope_hash(scope_type, scope_value),
                "status": status,
                "reset_detected": reset,
                "token_usage": usage,
                "token_usage_delta": {
                    **delta,
                    "total_tokens_bucket": _codex_count_bucket(delta.get("total_tokens")),
                },
            })
    latest_rate_limit = rate_limit_updates[-1] if rate_limit_updates else None
    latest_token_usage = token_usage_updates[-1] if token_usage_updates else None
    estimated_input = sum(max(0, int(_as_int(row.get("input_text_chars")) / TOKEN_CHARS)) for row in turn_rows)
    estimated_output = sum(max(0, int(_as_int(row.get("response_result_chars")) / TOKEN_CHARS)) for row in turn_rows)
    estimated_total = estimated_input + estimated_output
    usage_total = token_totals["total_tokens"]
    drift = {
        "input_tokens": token_totals["input_tokens"] - estimated_input,
        "output_tokens": token_totals["output_tokens"] - estimated_output,
        "total_tokens": usage_total - estimated_total,
    }
    return {
        "schema": "tokenclaw.codex_app_quota_token_usage.v1",
        "rate_limit_update_count": len(rate_limit_updates),
        "token_usage_update_count": len(token_usage_updates),
        "latest_rate_limits": latest_rate_limit.get("rate_limits") if latest_rate_limit else None,
        "latest_rate_limit_at": latest_rate_limit.get("created_at") if latest_rate_limit else None,
        "latest_token_usage": latest_token_usage.get("token_usage") if latest_token_usage else None,
        "latest_token_usage_delta": latest_token_usage.get("token_usage_delta") if latest_token_usage else None,
        "latest_token_usage_at": latest_token_usage.get("created_at") if latest_token_usage else None,
        "token_usage_totals": token_totals,
        "raw_counter_totals": raw_counter_totals,
        "reconciled_cost_usd": round(reconciled_cost_usd, 8),
        "reconciled_cost_known": reconciled_cost_known,
        "tokenclaw_estimated_totals": {
            "input_tokens_est": estimated_input,
            "output_tokens_est": estimated_output,
            "total_tokens_est": estimated_total,
        },
        "matched_tokenclaw_estimated_totals": reconciled_estimates,
        "reconciliation": {
            "input_drift_tokens": drift["input_tokens"],
            "output_drift_tokens": drift["output_tokens"],
            "total_drift_tokens": drift["total_tokens"],
            "total_drift_bucket": _codex_token_usage_reconciliation_state(status_counts),
            "total_drift_size_bucket": _token_drift_bucket(drift["total_tokens"]),
            "status": _codex_token_usage_reconciliation_state(status_counts),
            "status_breakdown": _count_breakdown(status_counts),
            "status_token_totals": [
                {"status": status, **totals}
                for status, totals in sorted(status_token_totals.items())
            ],
            "drift_ratio": round(drift["total_tokens"] / usage_total, 4) if usage_total else None,
            "basis": "tokenUsage monotonic deltas reconciled to scoped Codex turn windows minus AgentFlow char-derived estimates",
        },
        "by_workflow_phase": [
            {
                **{key: value for key, value in bucket.items() if key != "status_counts"},
                "cost_usd": round(_as_float(bucket.get("cost_usd")), 8),
                "status_breakdown": _count_breakdown(dict(bucket.get("status_counts") or {})),
            }
            for bucket in sorted(phase_buckets.values(), key=lambda item: item["total_tokens"], reverse=True)
        ],
        "by_model": [
            {
                **{key: value for key, value in bucket.items() if key != "status_counts"},
                "cost_usd": round(_as_float(bucket.get("cost_usd")), 8),
                "status_breakdown": _count_breakdown(dict(bucket.get("status_counts") or {})),
            }
            for bucket in sorted(model_buckets.values(), key=lambda item: item["total_tokens"], reverse=True)
        ],
        "by_thread": [
            {
                **{key: value for key, value in bucket.items() if key != "status_counts"},
                "cost_usd": round(_as_float(bucket.get("cost_usd")), 8),
                "status_breakdown": _count_breakdown(dict(bucket.get("status_counts") or {})),
            }
            for bucket in sorted(thread_buckets.values(), key=lambda item: item["total_tokens"], reverse=True)[:20]
        ],
        "privacy": {
            "metadata_only": True,
            "raw_params_included": False,
            "raw_prompts_included": False,
            "raw_commands_included": False,
            "raw_transcripts_included": False,
            "raw_thread_ids_included": False,
            "raw_session_ids_included": False,
            "arbitrary_payload_strings_included": False,
        },
    }

def _codex_workflow_phase(
    row: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    next_start_at: str | None,
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    for decision in (routing, crunch, cache):
        phase_value = str(decision.get("workflow_phase") or "").strip() if isinstance(decision, dict) else ""
        if phase_value and phase_value != "unknown":
            return {
                "phase": phase_value,
                "reason": str(decision.get("workflow_phase_reason") or "decision-metadata"),
                "source": "decision_metadata",
                "signals": list(decision.get("workflow_phase_signals") or []),
            }

    event_window = _json_obj(row.get("event_window_json"))
    if event_window:
        window_phase = str(event_window.get("workflow_phase") or "").strip()
        if window_phase and window_phase != "unknown":
            return {
                "phase": window_phase,
                "reason": str(event_window.get("workflow_phase_reason") or "event-window-metadata"),
                "source": str(event_window.get("workflow_phase_source") or "event_window"),
                "signals": list(event_window.get("workflow_phase_signals") or []),
            }
        signal_counts, signal_methods = _codex_signal_counts_from_method_counts(event_window.get("method_counts"))
        phase = _codex_phase_from_signal_counts(
            signal_counts,
            signal_methods,
            reason_prefix="event-window-signal",
            source="event_window",
        )
        if phase:
            return phase

    start_at = str(row.get("created_at") or "")
    scoped: list[dict[str, Any]] = []
    for event in events:
        event_at = str(event.get("created_at") or "")
        if event_at < start_at:
            continue
        if next_start_at and event_at >= next_start_at:
            continue
        if _codex_same_scope(event, row):
            scoped.append(event)

    signal_counts: dict[str, int] = {}
    signal_methods: dict[str, list[str]] = {}
    for event in scoped:
        signal = _codex_phase_signal(event.get("method"))
        if not signal:
            continue
        _increment_count(signal_counts, signal)
        methods = signal_methods.setdefault(signal, [])
        method = str(event.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    phase = _codex_phase_from_signal_counts(
        signal_counts,
        signal_methods,
        reason_prefix="event-method-signal",
        source="event_sequence",
    )
    if phase:
        return phase

    reasons = {
        str(decision.get("reason") or "")
        for decision in (routing, crunch, cache)
        if isinstance(decision, dict)
    }
    if "action-like-params" in reasons:
        return {
            "phase": "tool_execution",
            "reason": "decision-reason:action-like-params",
            "source": "decision_metadata",
            "signals": ["action-like-params"],
        }
    if _as_int(row.get("input_text_chars")) <= 0 and _as_int(row.get("params_chars")) > 0:
        return {
            "phase": "idle_control",
            "reason": "params-without-text-input",
            "source": "size_metadata",
            "signals": [],
        }
    return {
        "phase": "unknown",
        "reason": "insufficient-metadata",
        "source": "metadata_only_classifier",
        "signals": [],
    }

def _new_codex_phase_bucket(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "turns": 0,
        "completed": 0,
        "errors": 0,
        "pending": 0,
        "input_text_chars": 0,
        "result_chars": 0,
        "input_tokens_est": 0,
        "output_tokens_est": 0,
        "total_tokens_est": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_est_usd": 0.0,
        "hard_floor_usd": 0.0,
        "cost_known_turns": 0,
        "routing_applied": 0,
        "crunch_applied": 0,
        "cache_hits": 0,
        "saved_chars": 0,
        "tokens_saved_est": 0,
        "latency_values": [],
        "reason_counts": {},
        "signal_methods": {},
    }

def _finalize_codex_phase_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    latency_values = list(bucket.pop("latency_values", []))
    reason_counts = dict(bucket.pop("reason_counts", {}))
    signal_methods = dict(bucket.pop("signal_methods", {}))
    turns = _as_int(bucket.get("turns"))
    errors = _as_int(bucket.get("errors"))
    bucket["error_rate"] = round(errors / turns, 4) if turns else 0
    bucket["avg_latency_ms"] = _avg_or_none(latency_values)
    bucket["phase_reasons"] = _count_breakdown(reason_counts)
    bucket["signal_methods"] = [
        {"method": method, "count": count}
        for method, count in sorted(signal_methods.items(), key=lambda item: item[1], reverse=True)
    ]
    return bucket

def _codex_plateau_scope(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("thread_id"):
        return str(row["thread_id"]), "thread_id"
    if row.get("session_id"):
        return str(row["session_id"]), "session_id"
    if row.get("request_id"):
        return f"request:{row['request_id']}", "request_id"
    return "unknown", "unknown"

def _codex_original_session_key(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("thread_id"):
        return str(row["thread_id"]), "thread_id"
    if row.get("session_id"):
        return str(row["session_id"]), "session_id"
    if row.get("request_id"):
        return f"request:{row['request_id']}", "request_id"
    return "codex:unknown", "unknown"

def _codex_metadata_workflow_groups(
    rows: list[dict[str, Any]],
    *,
    idle_gap_seconds: int = 30 * 60,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: str(item.get("created_at") or ""))
    groups_by_event: dict[str, dict[str, Any]] = {}
    group_index = 0
    current_group: dict[str, Any] | None = None
    previous_at: datetime | None = None

    def new_window(row: dict[str, Any], started_at: datetime | None) -> dict[str, Any]:
        nonlocal group_index
        group_index += 1
        started_text = started_at.isoformat() if started_at else str(row.get("created_at") or "unknown")
        model_state, _model_field = _codex_model_field_state(
            _json_obj(row.get("routing_json")),
            row.get("event_window_json"),
        )
        digest = hashlib.sha256(
            f"codex_turn|codex|{started_text}|{model_state}|{group_index}".encode("utf-8")
        ).hexdigest()[:16]
        return {
            "key": f"codex-workflow:{digest}",
            "basis": "workflow_window",
            "group_start_at": started_text,
            "group_index": group_index,
            "idle_gap_seconds": idle_gap_seconds,
            "model_state_counts": {},
            "original_key_basis_counts": {},
            "original_key_count": 0,
            "_original_keys": set(),
        }

    for row in ordered:
        created_at = _parse_utc_datetime(row.get("created_at"))
        thread_id = str(row.get("thread_id") or "").strip()
        if thread_id:
            digest = hashlib.sha256(f"codex_turn|codex|thread_id|{thread_id}".encode("utf-8")).hexdigest()[:16]
            current = {
                "key": f"codex-workflow:{digest}",
                "basis": "workflow_thread_id",
                "group_start_at": None,
                "group_index": None,
                "idle_gap_seconds": idle_gap_seconds,
                "model_state_counts": {},
                "original_key_basis_counts": {},
                "original_key_count": 0,
                "_original_keys": set(),
            }
        else:
            gap_seconds = (
                (created_at - previous_at).total_seconds()
                if created_at is not None and previous_at is not None
                else None
            )
            if current_group is None or (gap_seconds is not None and gap_seconds > idle_gap_seconds):
                current_group = new_window(row, created_at)
            current = current_group

        model_state, _model_field = _codex_model_field_state(
            _json_obj(row.get("routing_json")),
            row.get("event_window_json"),
        )
        original_key, original_basis = _codex_original_session_key(row)
        current["model_state_counts"][model_state] = current["model_state_counts"].get(model_state, 0) + 1
        current["original_key_basis_counts"][original_basis] = (
            current["original_key_basis_counts"].get(original_basis, 0) + 1
        )
        current["_original_keys"].add(f"{original_basis}:{original_key}")
        current["original_key_count"] = len(current["_original_keys"])
        event_id = str(row.get("start_event_id") or row.get("id") or row.get("request_id") or "")
        if event_id:
            groups_by_event[event_id] = current
        if not thread_id and created_at is not None:
            previous_at = created_at

    public: dict[str, dict[str, Any]] = {}
    for event_id, group in groups_by_event.items():
        public[event_id] = {
            "key": group["key"],
            "basis": group["basis"],
            "group_start_at": group.get("group_start_at"),
            "group_index": group.get("group_index"),
            "idle_gap_seconds": group["idle_gap_seconds"],
            "model_state_counts": dict(group.get("model_state_counts") or {}),
            "original_key_basis_counts": dict(group.get("original_key_basis_counts") or {}),
            "original_key_count": int(group.get("original_key_count") or 0),
        }
    return public

def _codex_phase_from_decision_metadata(
    routing: dict[str, Any],
    crunch: dict[str, Any],
    cache: dict[str, Any],
) -> str:
    for meta in (cache, crunch, routing):
        phase = meta.get("workflow_phase") if isinstance(meta, dict) else None
        if phase:
            return str(phase)
    return "unknown"

def _codex_meaningful_crunch(row: dict[str, Any], *, min_ratio: float) -> bool:
    input_chars = _as_int(row.get("input_text_chars"))
    if input_chars <= 0:
        return False
    return (_as_int(row.get("saved_chars")) / input_chars) >= min_ratio

def _codex_plateau_candidate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    min_input_chars = 8_000
    max_delta_ratio = 0.03
    min_candidate_pairs = 2
    meaningful_crunch_ratio = 0.05
    conservative_opportunity_ratio = 0.10

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        scope_id, scope_basis = _codex_plateau_scope(row)
        groups.setdefault((scope_basis, scope_id), []).append(row)

    candidates: list[dict[str, Any]] = []
    total_plateau_pairs = 0
    total_candidate_pairs = 0
    total_opportunity_chars = 0
    for (scope_basis, scope_id), scoped_rows in groups.items():
        ordered = sorted(scoped_rows, key=lambda item: str(item.get("created_at") or ""))
        input_values = [_as_int(row.get("input_text_chars")) for row in ordered]
        large_values = [value for value in input_values if value >= min_input_chars]
        if len(large_values) < min_candidate_pairs + 1:
            continue

        plateau_pairs = 0
        candidate_pairs = 0
        repeated_chars_est = 0
        candidate_repeated_chars_est = 0
        method_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        cache_status_counts: dict[str, int] = {}
        crunch_status_counts: dict[str, int] = {}
        for row in ordered:
            _increment_count(phase_counts, row.get("workflow_phase") or "unknown")
            _increment_count(cache_status_counts, row.get("cache_status") or "missing")
            _increment_count(crunch_status_counts, row.get("crunch_status") or "missing")
            event_window = _json_obj(row.get("event_window_json"))
            window_methods = event_window.get("method_counts") if isinstance(event_window, dict) else None
            if isinstance(window_methods, dict):
                for method, count in window_methods.items():
                    method_counts[str(method or "unknown")] = method_counts.get(str(method or "unknown"), 0) + max(1, _as_int(count))
            else:
                _increment_count(method_counts, row.get("method") or "turn/start")

        for previous, current in zip(ordered, ordered[1:]):
            previous_chars = _as_int(previous.get("input_text_chars"))
            current_chars = _as_int(current.get("input_text_chars"))
            if previous_chars < min_input_chars or current_chars < min_input_chars:
                continue
            delta_ratio = abs(current_chars - previous_chars) / max(previous_chars, 1)
            if delta_ratio > max_delta_ratio:
                continue
            plateau_pairs += 1
            repeated_chars = min(previous_chars, current_chars)
            repeated_chars_est += repeated_chars
            cache_hit = previous.get("cache_status") == "hit" or current.get("cache_status") == "hit"
            meaningful_crunch = (
                _codex_meaningful_crunch(previous, min_ratio=meaningful_crunch_ratio)
                or _codex_meaningful_crunch(current, min_ratio=meaningful_crunch_ratio)
            )
            if not cache_hit and not meaningful_crunch:
                candidate_pairs += 1
                candidate_repeated_chars_est += repeated_chars

        total_plateau_pairs += plateau_pairs
        total_candidate_pairs += candidate_pairs
        if candidate_pairs < min_candidate_pairs:
            continue

        current_saved_chars = sum(_as_int(row.get("saved_chars")) for row in ordered)
        current_saved_tokens = sum(_as_int(row.get("tokens_saved_est")) for row in ordered)
        opportunity_chars = max(
            int(candidate_repeated_chars_est * conservative_opportunity_ratio) - current_saved_chars,
            0,
        )
        opportunity_tokens = estimate_tokens_from_text_chars(opportunity_chars)
        opportunity_cost = estimate_cost(
            CODEX_APP_MODEL,
            opportunity_tokens,
            0,
            provider="openai",
            processing_mode=CODEX_APP_PROCESSING_MODE,
        )
        total_opportunity_chars += opportunity_chars
        candidates.append({
            "candidate_id": f"codex-context-plateau:{scope_basis}:{scope_id[:24]}",
            "scope_id": scope_id,
            "sid": scope_id[:8] if scope_id else None,
            "scope_basis": scope_basis,
            "turns": len(ordered),
            "large_turns": len(large_values),
            "plateau_count": plateau_pairs,
            "plateau_pairs": plateau_pairs,
            "candidate_pairs": candidate_pairs,
            "median_input_chars": _median_int(large_values),
            "p90_input_chars": _percentile_int(large_values, 0.9),
            "min_input_chars": min(large_values),
            "max_input_chars": max(large_values),
            "current_saved_chars": current_saved_chars,
            "current_saved_tokens_est": current_saved_tokens,
            "estimated_repeated_chars": repeated_chars_est,
            "estimated_candidate_repeated_chars": candidate_repeated_chars_est,
            "estimated_opportunity_saved_chars": opportunity_chars,
            "estimated_opportunity_tokens": opportunity_tokens,
            "estimated_opportunity_usd": round(float(opportunity_cost or 0.0), 6) if opportunity_cost is not None else None,
            "opportunity_basis": "10pct-of-unoptimized-adjacent-large-plateau-chars-minus-current-saved-chars",
            "cache_status_counts": _count_breakdown(cache_status_counts),
            "crunch_status_counts": _count_breakdown(crunch_status_counts),
            "workflow_phase_counts": _count_breakdown(phase_counts),
            "method_counts": [
                {"method": method, "count": count}
                for method, count in sorted(method_counts.items(), key=lambda item: item[1], reverse=True)[:10]
            ],
        })

    candidates.sort(
        key=lambda item: (
            item["estimated_opportunity_saved_chars"],
            item["candidate_pairs"],
            item["p90_input_chars"],
        ),
        reverse=True,
    )
    candidates = candidates[:20]
    return {
        "policy": {
            "min_input_chars": min_input_chars,
            "max_adjacent_delta_ratio": max_delta_ratio,
            "min_candidate_pairs": min_candidate_pairs,
            "meaningful_crunch_ratio": meaningful_crunch_ratio,
            "conservative_opportunity_ratio": conservative_opportunity_ratio,
            "privacy_basis": "metadata-only input sizes, decision status, event-window method counts, and scope IDs",
        },
        "summary": {
            "scopes_considered": len(groups),
            "plateau_pairs": total_plateau_pairs,
            "candidate_pairs": total_candidate_pairs,
            "candidate_count": len(candidates),
            "estimated_opportunity_saved_chars": total_opportunity_chars,
            "estimated_opportunity_tokens": estimate_tokens_from_text_chars(total_opportunity_chars),
        },
        "candidates": candidates,
    }

async def stats_codex_effectiveness(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 500), 5000))
    rows = conn.execute("""
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
               s.event_window_json,
               s.metadata_json,
               (
                   select r.id from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_event_id,
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
               ) as response_latency_ms,
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
        order by s.created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    turn_rows = [dict(row) for row in rows]
    min_start_at = min((str(row.get("created_at") or "") for row in turn_rows), default="")
    event_scan_limit = max(15000, min(200000, capped_limit * 1000))
    if turn_rows:
        event_sql = """
            select created_at, direction, method, request_id, thread_id, session_id, metadata_json
            from codex_app_events
            where created_at >= ?
            order by created_at asc
            limit ?
            """
        event_params = (min_start_at or "0000-00-00T00:00:00+00:00", event_scan_limit)
    else:
        event_sql = """
            select *
            from (
                select created_at, direction, method, request_id, thread_id, session_id, metadata_json
                from codex_app_events
                order by created_at desc
                limit ?
            )
            order by created_at asc
            """
        event_params = (event_scan_limit,)
    event_rows = [dict(row) for row in conn.execute(event_sql, event_params).fetchall()]
    events_by_thread: dict[str, list[dict[str, Any]]] = {}
    events_by_session_without_thread: dict[str, list[dict[str, Any]]] = {}
    events_by_session: dict[str, list[dict[str, Any]]] = {}
    for event in event_rows:
        session_key = str(event.get("session_id") or "")
        thread_key = str(event.get("thread_id") or "")
        if session_key:
            events_by_session.setdefault(session_key, []).append(event)
            if not thread_key:
                events_by_session_without_thread.setdefault(session_key, []).append(event)
        if thread_key:
            events_by_thread.setdefault(thread_key, []).append(event)
    turn_bounds = _codex_turn_bounds(turn_rows)

    model_field_counts: dict[str, int] = {}
    model_field_names: dict[str, int] = {}
    param_shape_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    phase_source_counts: dict[str, int] = {}
    phase_buckets: dict[str, dict[str, Any]] = {}
    optimized_latency: list[int] = []
    pass_through_latency: list[int] = []
    optimized_errors = 0
    pass_through_errors = 0
    optimized_count = 0
    pass_through_count = 0
    pending_count = 0
    error_count = 0
    success_count = 0
    total_saved_chars = 0
    total_saved_tokens = 0
    total_cache_savings_usd = 0.0
    total_codex_scaffolding_saved_chars = 0
    action_like_skips = 0
    unknown_param_skips = 0
    non_text_skips = 0
    decision_metadata_counts: dict[str, int] = {}
    current_missing_decision_counts: dict[str, int] = {}
    not_instrumented_decision_counts: dict[str, int] = {}
    historical_unavailable_decision_counts: dict[str, int] = {}
    managed_status_counts: dict[str, int] = {}
    managed_feedback_status_counts: dict[str, int] = {}
    managed_feedback_reason_counts: dict[str, int] = {}
    managed_feedback_queue_counts: dict[str, int] = {}
    managed_pattern_fingerprint_rows = 0
    managed_pattern_hash_count = 0
    safety_stop_rows = 0
    safety_stop_reason_counts: dict[str, int] = {}
    latest_safety_stop: dict[str, Any] | None = None
    summary_hint_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    if hasattr(store_obj, "managed_outcome_feedback_summary"):
        try:
            for row in store_obj.managed_outcome_feedback_summary(source_surface=CODEX_APP_SOURCE_SURFACE):
                managed_feedback_queue_counts[str(row.get("status") or "unknown")] = _as_int(row.get("count"))
        except Exception:
            managed_feedback_queue_counts = {}

    recent_samples: list[dict[str, Any]] = []
    plateau_candidate_rows: list[dict[str, Any]] = []
    for row in turn_rows:
        metadata_state = _codex_decision_metadata_state(row)
        row["decision_metadata_state"] = metadata_state
        _increment_count(decision_metadata_counts, metadata_state)
        routing = _codex_normalized_decision(row, "routing_json", metadata_state)
        crunch = _codex_normalized_decision(row, "crunch_json", metadata_state)
        cache = _codex_normalized_decision(row, "cache_json", metadata_state)
        row["routing_json_normalized"] = routing
        row["crunch_json_normalized"] = crunch
        row["cache_json_normalized"] = cache
        for decision_key, decision in (
            ("routing", routing),
            ("crunch", crunch),
            ("cache", cache),
        ):
            status = str(decision.get("status") or "")
            if status == "missing":
                _increment_count(current_missing_decision_counts, decision_key)
            elif status == "not-instrumented":
                _increment_count(not_instrumented_decision_counts, decision_key)
            elif status == "historical-unavailable":
                _increment_count(historical_unavailable_decision_counts, decision_key)
        managed = routing.get("managed_recommendation") if isinstance(routing, dict) else None
        feedback = managed.get("outcome_feedback") if isinstance(managed, dict) else None
        pattern_diagnostics = routing.get("managed_pattern_features") if isinstance(routing, dict) else None
        if isinstance(pattern_diagnostics, dict) and pattern_diagnostics.get("present"):
            managed_pattern_fingerprint_rows += 1
            managed_pattern_hash_count += _as_int(pattern_diagnostics.get("pattern_hash_count"))
        if isinstance(managed, dict):
            _increment_count(managed_status_counts, managed.get("status") or "unknown")
            if isinstance(feedback, dict):
                _increment_count(managed_feedback_status_counts, feedback.get("status") or "unknown")
                _increment_count(managed_feedback_reason_counts, feedback.get("reason") or "unknown")
            else:
                _increment_count(managed_feedback_status_counts, "pending")
        model_state, model_field = _codex_model_field_state(routing, row.get("event_window_json"))
        _increment_count(model_field_counts, model_state)
        if model_field:
            _increment_count(model_field_names, model_field)
        shape = _codex_param_shape_category(row, routing, crunch, cache)
        _increment_count(param_shape_counts, shape)
        _increment_count(method_counts, row.get("method") or "turn/start")
        session_key = str(row.get("session_id") or "")
        thread_key = str(row.get("thread_id") or "")
        if thread_key:
            phase_events = list(events_by_thread.get(thread_key, []))
            if session_key:
                phase_events.extend(events_by_session_without_thread.get(session_key, []))
        elif session_key:
            phase_events = events_by_session.get(session_key, [])
        else:
            phase_events = event_rows
        phase_meta = _codex_workflow_phase(
            row,
            events=phase_events,
            next_start_at=turn_bounds.get(str(row.get("start_event_id"))),
            routing=routing,
            crunch=crunch,
            cache=cache,
        )
        phase = str(phase_meta.get("phase") or "unknown")
        row["workflow_phase"] = phase
        _increment_count(phase_counts, phase)
        _increment_count(phase_source_counts, phase_meta.get("source") or "unknown")

        reasons = {
            str(decision.get("reason") or "")
            for decision in (routing, crunch, cache)
            if isinstance(decision, dict)
        }
        row_safety_stop: dict[str, Any] | None = None
        for decision_name, decision in (("routing", routing), ("crunch", crunch), ("cache", cache)):
            safety = decision.get("safety_stop") if isinstance(decision.get("safety_stop"), dict) else {}
            stopped = (
                bool(safety.get("tripped"))
                or str(decision.get("status") or "") == "safety_stopped"
            )
            if not stopped:
                continue
            row_safety_stop = safety or {
                "tripped": True,
                "reason_codes": [str(decision.get("reason") or "local-canary-safety-stop")],
            }
            reason_codes = row_safety_stop.get("reason_codes") if isinstance(row_safety_stop.get("reason_codes"), list) else []
            if not reason_codes:
                reason_codes = [str(decision.get("reason") or "local-canary-safety-stop")]
            for reason_code in reason_codes:
                _increment_count(safety_stop_reason_counts, reason_code)
            if latest_safety_stop is None:
                latest_safety_stop = {
                    "created_at": row.get("created_at"),
                    "decision_type": decision_name,
                    "status": row_safety_stop.get("status") or "stopped",
                    "reason": row_safety_stop.get("reason") or "local-canary-safety-stop",
                    "reason_codes": list(reason_codes),
                    "source": row_safety_stop.get("source"),
                    "policy_source": row_safety_stop.get("policy_source") or decision.get("policy_source"),
                    "rule_id": row_safety_stop.get("rule_id"),
                    "candidate_id": row_safety_stop.get("candidate_id"),
                    "sample_count": _as_int(row_safety_stop.get("sample_count")),
                    "applied_sample_count": _as_int(row_safety_stop.get("applied_sample_count")),
                    "holdout_sample_count": _as_int(row_safety_stop.get("holdout_sample_count")),
                    "privacy": {
                        "metadata_only": True,
                        "raw_params_included": False,
                        "raw_prompts_included": False,
                        "raw_responses_included": False,
                        "cache_keys_included": False,
                    },
                }
        if row_safety_stop is not None:
            safety_stop_rows += 1
        if "action-like-params" in reasons:
            action_like_skips += 1
        if "unknown-param-shape" in reasons:
            unknown_param_skips += 1
        if "non-text-input" in reasons:
            non_text_skips += 1

        saved_chars = _as_int(crunch.get("saved_chars"))
        saved_tokens = _as_int(crunch.get("tokens_saved_est"))
        total_saved_chars += saved_chars
        total_saved_tokens += saved_tokens
        codex_scaffolding = crunch.get("codex_repeated_scaffolding")
        if isinstance(codex_scaffolding, dict):
            total_codex_scaffolding_saved_chars += _as_int(codex_scaffolding.get("saved_chars"))

        optimized = bool(routing.get("applied") or crunch.get("applied") or cache.get("status") == "hit")
        has_response = bool(row.get("response_event_id"))
        has_error = row.get("response_error_code") is not None
        latency = _as_int(row.get("response_latency_ms"))
        result_chars = _as_int(row.get("response_result_chars"))
        if has_error:
            error_count += 1
        elif has_response:
            success_count += 1
        else:
            pending_count += 1

        if optimized:
            optimized_count += 1
            if has_error:
                optimized_errors += 1
            if latency:
                optimized_latency.append(latency)
        else:
            pass_through_count += 1
            if has_error:
                pass_through_errors += 1
            if latency:
                pass_through_latency.append(latency)

        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), result_chars, cache)
        total_cache_savings_usd += _as_float(estimates.get("cache_savings_usd"))
        phase_bucket = phase_buckets.setdefault(phase, _new_codex_phase_bucket(phase))
        phase_bucket["turns"] += 1
        phase_bucket["input_text_chars"] += _as_int(row.get("input_text_chars"))
        phase_bucket["result_chars"] += result_chars
        phase_bucket["input_tokens_est"] += _as_int(estimates.get("input_tokens_est"))
        phase_bucket["output_tokens_est"] += _as_int(estimates.get("output_tokens_est"))
        phase_bucket["total_tokens_est"] += _as_int(estimates.get("total_tokens_est"))
        phase_bucket["cost_est_usd"] += _as_float(estimates.get("cost_est_usd"))
        phase_bucket["baseline_cost_est_usd"] += _as_float(estimates.get("baseline_cost_est_usd"))
        phase_bucket["hard_floor_usd"] += _as_float(estimates.get("hard_floor_usd"))
        if estimates.get("cost_known"):
            phase_bucket["cost_known_turns"] += 1
        if routing.get("applied"):
            phase_bucket["routing_applied"] += 1
        if crunch.get("applied"):
            phase_bucket["crunch_applied"] += 1
        if cache.get("status") == "hit":
            phase_bucket["cache_hits"] += 1
        phase_bucket["saved_chars"] += saved_chars
        phase_bucket["tokens_saved_est"] += saved_tokens
        if has_error:
            phase_bucket["errors"] += 1
        elif has_response:
            phase_bucket["completed"] += 1
        else:
            phase_bucket["pending"] += 1
        if latency:
            phase_bucket["latency_values"].append(latency)
        _increment_count(phase_bucket["reason_counts"], phase_meta.get("reason") or "unknown")
        for signal_method in phase_meta.get("signals") or []:
            _increment_count(phase_bucket["signal_methods"], signal_method)
        _add_codex_summary_hint_bucket(
            summary_hint_buckets,
            phase=phase,
            routing=routing,
            crunch=crunch,
            cache=cache,
            input_text_chars=row.get("input_text_chars"),
            result_chars=result_chars,
            saved_chars=saved_chars,
            saved_tokens=saved_tokens,
            has_response=has_response,
            has_error=has_error,
            latency=latency,
        )

        if len(recent_samples) < 20:
            recent_samples.append({
                "created_at": row.get("created_at"),
                "method": row.get("method") or "turn/start",
                "workflow_phase": phase,
                "workflow_phase_reason": phase_meta.get("reason"),
                "workflow_phase_source": phase_meta.get("source"),
                "workflow_phase_signals": phase_meta.get("signals") or [],
                "event_window": _codex_public_event_window(row.get("event_window_json")),
                "decision_metadata_state": metadata_state,
                "model_field": model_state,
                "param_shape": shape,
                "routing_status": routing.get("status") or "missing",
                "routing_reason": routing.get("reason") or "unknown",
                "crunch_status": crunch.get("status") or "missing",
                "crunch_reason": crunch.get("reason") or "unknown",
                "codex_pattern_types": [
                    str(pattern.get("type"))
                    for pattern in (crunch.get("codex_patterns") or [])
                    if isinstance(pattern, dict) and pattern.get("type")
                ],
                "cache_status": cache.get("status") or "missing",
                "cache_reason": cache.get("reason") or "unknown",
                "managed_recommendation_status": (managed or {}).get("status") if isinstance(managed, dict) else "missing",
                "managed_feedback_status": (feedback or {}).get("status") if isinstance(feedback, dict) else ("pending" if isinstance(managed, dict) else "missing"),
                "managed_feedback_reason": (feedback or {}).get("reason") if isinstance(feedback, dict) else None,
                "safety_stop": {
                    "tripped": bool(row_safety_stop),
                    "reason_codes": row_safety_stop.get("reason_codes") if isinstance(row_safety_stop, dict) else [],
                    "source": row_safety_stop.get("source") if isinstance(row_safety_stop, dict) else None,
                },
                "managed_pattern_features": {
                    "present": bool((pattern_diagnostics or {}).get("present")) if isinstance(pattern_diagnostics, dict) else False,
                    "pattern_hash_count": _as_int((pattern_diagnostics or {}).get("pattern_hash_count")) if isinstance(pattern_diagnostics, dict) else 0,
                    "hash_basis": (pattern_diagnostics or {}).get("hash_basis") if isinstance(pattern_diagnostics, dict) else None,
                    "text_bucket": (pattern_diagnostics or {}).get("text_bucket") if isinstance(pattern_diagnostics, dict) else None,
                    "token_bucket": (pattern_diagnostics or {}).get("token_bucket") if isinstance(pattern_diagnostics, dict) else None,
                    "pattern_types": (pattern_diagnostics or {}).get("pattern_types") if isinstance(pattern_diagnostics, dict) else [],
                    "raw_pattern_strings_included": False,
                },
                "input_text_chars": _as_int(row.get("input_text_chars")),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "outcome": "error" if has_error else ("success" if has_response else "pending"),
                "latency_ms": latency or None,
                "error_code": row.get("response_error_code"),
            })
        plateau_candidate_rows.append({
            "created_at": row.get("created_at"),
            "method": row.get("method") or "turn/start",
            "request_id": row.get("request_id"),
            "thread_id": row.get("thread_id"),
            "session_id": row.get("session_id"),
            "input_text_chars": _as_int(row.get("input_text_chars")),
            "saved_chars": saved_chars,
            "tokens_saved_est": saved_tokens,
            "workflow_phase": phase,
            "routing_status": routing.get("status") or "missing",
            "crunch_status": crunch.get("status") or "missing",
            "cache_status": cache.get("status") or "missing",
            "event_window_json": row.get("event_window_json"),
        })

    total = len(turn_rows)
    plateau_candidate_report = _codex_plateau_candidate_report(plateau_candidate_rows)
    quota_token_usage = _codex_quota_token_usage_report(event_rows, turn_rows)
    summary_model_hint_buckets = _finalize_codex_summary_hint_buckets(summary_hint_buckets)
    summary_model_hint_turns = sum(_as_int(row.get("turns")) for row in summary_model_hint_buckets)
    summary_model_hint_errors = sum(_as_int(row.get("errors")) for row in summary_model_hint_buckets)
    summary_model_hint_pending = sum(_as_int(row.get("pending")) for row in summary_model_hint_buckets)
    summary_model_hint_savings = sum(_as_float(row.get("estimated_savings_usd")) for row in summary_model_hint_buckets)
    summary_model_hint_canary = _codex_summary_hint_canary_summary(summary_model_hint_buckets)
    return {
        "schema": "tokenclaw.codex_app_effectiveness.v1",
        "generated_at": utc_now(),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "limit": capped_limit,
        "privacy": {
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_responses_included": False,
            "basis": "stored metadata, sizes, hashes, and decision JSON only",
        },
        "event_scan": {
            "events_considered": len(event_rows),
            "event_scan_limit": event_scan_limit,
            "truncated": bool(turn_rows) and len(event_rows) >= event_scan_limit,
        },
        "summary": {
            "turn_start_rows": total,
            "completed_rows": success_count,
            "error_rows": error_count,
            "pending_rows": pending_count,
            "model_field_present": (
                model_field_counts.get("present", 0)
                + model_field_counts.get("present_unknown_field", 0)
                + model_field_counts.get("derived_present", 0)
            ),
            "model_field_derived": model_field_counts.get("derived_present", 0),
            "model_field_absent": model_field_counts.get("absent", 0) + model_field_counts.get("derived_absent", 0),
            "model_field_unknown": model_field_counts.get("unknown", 0),
            "decision_metadata_complete_rows": decision_metadata_counts.get("complete", 0),
            "decision_metadata_historical_unavailable_rows": decision_metadata_counts.get("historical-unavailable", 0),
            "decision_metadata_not_instrumented_rows": decision_metadata_counts.get("not-instrumented", 0),
            "decision_metadata_current_missing_rows": decision_metadata_counts.get("current-missing", 0),
            "current_missing_decisions": sum(current_missing_decision_counts.values()),
            "not_instrumented_decisions": sum(not_instrumented_decision_counts.values()),
            "historical_unavailable_decisions": sum(historical_unavailable_decision_counts.values()),
            "routing_applied": sum(1 for row in turn_rows if _json_obj(row.get("routing_json_normalized")).get("applied")),
            "crunch_applied": sum(1 for row in turn_rows if _json_obj(row.get("crunch_json_normalized")).get("applied")),
            "cache_hits": sum(1 for row in turn_rows if _json_obj(row.get("cache_json_normalized")).get("status") == "hit"),
            "cache_eligible": sum(1 for row in turn_rows if bool(_json_obj(row.get("cache_json_normalized")).get("eligible"))),
            "cache_estimated_savings_usd": round(total_cache_savings_usd, 8),
            "summary_model_hint_rows": summary_model_hint_turns,
            "summary_model_hint_applied": sum(
                _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "applied"
            ),
            "summary_model_hint_eligible_skipped": sum(
                _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "eligible-skipped"
            ),
            "summary_model_hint_holdout": sum(
                _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "holdout"
            ),
            "summary_model_hint_unsafe_skipped": sum(
                _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "unsafe-skipped"
            ),
            "summary_model_hint_pending": summary_model_hint_pending,
            "summary_model_hint_error_rate": round(summary_model_hint_errors / summary_model_hint_turns, 4)
            if summary_model_hint_turns
            else 0,
            "summary_model_hint_estimated_savings_usd": round(summary_model_hint_savings, 8),
            "action_like_skips": action_like_skips,
            "unknown_param_skips": unknown_param_skips,
            "non_text_input_skips": non_text_skips,
            "workflow_phase_known": total - phase_counts.get("unknown", 0),
            "workflow_phase_unknown": phase_counts.get("unknown", 0),
            "total_input_text_chars": sum(_as_int(row.get("input_text_chars")) for row in turn_rows),
            "total_saved_chars": total_saved_chars,
            "total_saved_tokens_est": total_saved_tokens,
            "codex_repeated_scaffolding_saved_chars": total_codex_scaffolding_saved_chars,
            "optimized_rows": optimized_count,
            "pass_through_rows": pass_through_count,
            "optimized_error_rate": round(optimized_errors / optimized_count, 4) if optimized_count else 0,
            "pass_through_error_rate": round(pass_through_errors / pass_through_count, 4) if pass_through_count else 0,
            "optimized_avg_latency_ms": _avg_or_none(optimized_latency),
            "pass_through_avg_latency_ms": _avg_or_none(pass_through_latency),
            "managed_recommendation_rows": sum(managed_status_counts.values()),
            "managed_pattern_fingerprint_rows": managed_pattern_fingerprint_rows,
            "managed_pattern_hash_count": managed_pattern_hash_count,
            "managed_recommendation_enabled": sum(
                1
                for row in turn_rows
                if bool((_json_obj(row.get("routing_json_normalized")).get("managed_recommendation") or {}).get("enabled"))
            ),
            "managed_recommendation_disabled": sum(
                1
                for row in turn_rows
                if isinstance(_json_obj(row.get("routing_json_normalized")).get("managed_recommendation"), dict)
                and not bool((_json_obj(row.get("routing_json_normalized")).get("managed_recommendation") or {}).get("enabled"))
            ),
            "managed_feedback_sent": managed_feedback_status_counts.get("sent", 0),
            "managed_feedback_skipped": (
                managed_feedback_status_counts.get("skipped", 0)
                + managed_feedback_status_counts.get("disabled", 0)
            ),
            "managed_feedback_queued": managed_feedback_status_counts.get("queued", 0),
            "managed_feedback_error": (
                managed_feedback_status_counts.get("error", 0)
                + managed_feedback_status_counts.get("retryable-error", 0)
                + managed_feedback_status_counts.get("dropped-after-limit", 0)
            ),
            "managed_feedback_retryable_error": managed_feedback_status_counts.get("retryable-error", 0),
            "managed_feedback_dropped_after_limit": managed_feedback_status_counts.get("dropped-after-limit", 0),
            "managed_feedback_pending": managed_feedback_status_counts.get("pending", 0),
            "managed_feedback_queue_sent": managed_feedback_queue_counts.get("sent", 0),
            "managed_feedback_queue_queued": managed_feedback_queue_counts.get("queued", 0),
            "managed_feedback_queue_error": (
                managed_feedback_queue_counts.get("retryable-error", 0)
                + managed_feedback_queue_counts.get("dropped-after-limit", 0)
            ),
            "safety_stop_rows": safety_stop_rows,
            "active_safety_stop_count": safety_stop_rows,
            "repeated_context_plateau_candidate_count": plateau_candidate_report["summary"]["candidate_count"],
            "repeated_context_plateau_pairs": plateau_candidate_report["summary"]["plateau_pairs"],
            "repeated_context_plateau_opportunity_chars": plateau_candidate_report["summary"]["estimated_opportunity_saved_chars"],
            "rate_limit_update_rows": quota_token_usage["rate_limit_update_count"],
            "token_usage_update_rows": quota_token_usage["token_usage_update_count"],
            "token_usage_total_tokens": quota_token_usage["token_usage_totals"]["total_tokens"],
            "token_usage_reconciliation_drift_bucket": quota_token_usage["reconciliation"]["total_drift_bucket"],
        },
        "decision_metadata_breakdown": _count_breakdown(decision_metadata_counts),
        "current_missing_decision_breakdown": _count_breakdown(current_missing_decision_counts),
        "not_instrumented_decision_breakdown": _count_breakdown(not_instrumented_decision_counts),
        "historical_unavailable_decision_breakdown": _count_breakdown(historical_unavailable_decision_counts),
        "model_field_breakdown": _count_breakdown(model_field_counts),
        "model_field_names": _count_breakdown(model_field_names),
        "method_breakdown": _count_breakdown(method_counts),
        "param_shape_breakdown": _count_breakdown(param_shape_counts),
        "workflow_phase_breakdown": [
            _finalize_codex_phase_bucket(bucket)
            for bucket in sorted(phase_buckets.values(), key=lambda item: item["turns"], reverse=True)
        ],
        "summary_model_hint": {
            "schema": "tokenclaw.codex_app_summary_model_hint.v1",
            "summary": {
                "turns": summary_model_hint_turns,
                "applied": sum(
                    _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "applied"
                ),
                "eligible_skipped": sum(
                    _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "eligible-skipped"
                ),
                "holdout": sum(
                    _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "holdout"
                ),
                "unsafe_skipped": sum(
                    _as_int(row.get("turns")) for row in summary_model_hint_buckets if row.get("status") == "unsafe-skipped"
                ),
                "candidate_count": summary_model_hint_canary["candidate_count"],
                "unsafe_skip_count": summary_model_hint_canary["unsafe_skip_count"],
                "pending": summary_model_hint_pending,
                "errors": summary_model_hint_errors,
                "error_rate": round(summary_model_hint_errors / summary_model_hint_turns, 4)
                if summary_model_hint_turns
                else 0,
                "estimated_savings_usd": round(summary_model_hint_savings, 8),
            },
            "canary": summary_model_hint_canary,
            "buckets": summary_model_hint_buckets,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_params_included": False,
                "raw_responses_included": False,
                "raw_transcripts_included": False,
                "raw_request_ids_included": False,
                "raw_commands_included": False,
                "basis": "routing, crunch, cache, size, latency, and JSON-RPC outcome metadata",
            },
        },
        "summary_model_hint_buckets": summary_model_hint_buckets,
        "workflow_phase_counts": _count_breakdown(phase_counts),
        "workflow_phase_source_breakdown": _count_breakdown(phase_source_counts),
        "routing_breakdown": _decision_breakdown(turn_rows, "routing_json"),
        "crunch_breakdown": _decision_breakdown(turn_rows, "crunch_json"),
        "crunch_pattern_breakdown": _codex_crunch_pattern_breakdown(turn_rows),
        "cache_breakdown": _decision_breakdown(turn_rows, "cache_json"),
        "managed_recommendation_breakdown": _count_breakdown(managed_status_counts),
        "managed_pattern_fingerprints": {
            "schema": "tokenclaw.managed_pattern_fingerprint_diagnostics.v1",
            "rows_with_fingerprints": managed_pattern_fingerprint_rows,
            "pattern_hash_count": managed_pattern_hash_count,
            "raw_pattern_strings_included": False,
            "basis": "stored routing metadata only",
        },
        "managed_feedback_breakdown": _count_breakdown(managed_feedback_status_counts),
        "managed_feedback_reason_breakdown": _count_breakdown(managed_feedback_reason_counts),
        "managed_feedback_queue_breakdown": _count_breakdown(managed_feedback_queue_counts),
        "safety_stop": {
            "schema": "tokenclaw.codex_app_safety_stop_state.v1",
            "active": bool(safety_stop_rows),
            "rows": safety_stop_rows,
            "latest": latest_safety_stop,
            "reason_code_breakdown": _count_breakdown(safety_stop_reason_counts),
            "privacy": {
                "metadata_only": True,
                "raw_params_included": False,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "cache_keys_included": False,
            },
        },
        "quota_and_token_usage": quota_token_usage,
        "repeated_context_plateau_candidates": plateau_candidate_report,
        "outcome_by_optimization": [
            {
                "bucket": "optimized",
                "count": optimized_count,
                "errors": optimized_errors,
                "error_rate": round(optimized_errors / optimized_count, 4) if optimized_count else 0,
                "avg_latency_ms": _avg_or_none(optimized_latency),
            },
            {
                "bucket": "pass_through",
                "count": pass_through_count,
                "errors": pass_through_errors,
                "error_rate": round(pass_through_errors / pass_through_count, 4) if pass_through_count else 0,
                "avg_latency_ms": _avg_or_none(pass_through_latency),
            },
        ],
        "recent_samples": recent_samples,
    }

def _codex_rule_report_meta(decision: dict[str, Any], *, action_family: str) -> dict[str, Any] | None:
    rule = decision.get("codex_app_rule") if isinstance(decision.get("codex_app_rule"), dict) else {}
    if rule:
        rule_id = str(rule.get("rule_id") or rule.get("candidate_id") or "unknown-codex-app-rule")
        candidate_id = str(rule.get("candidate_id") or rule_id)
        return {
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "policy_id": rule.get("policy_id") or candidate_id,
            "policy_source": rule.get("policy_source") or decision.get("policy_source") or "unknown",
            "condition_keys": list(rule.get("condition_keys") or []),
            "action_keys": list(rule.get("action_keys") or []),
        }
    if action_family == "routing" and decision.get("canary") == "codex-app-summary-model-hint":
        policy_id = str(decision.get("policy_id") or "local-codex-app-summary-model-hint-canary")
        return {
            "rule_id": policy_id,
            "candidate_id": policy_id,
            "policy_id": policy_id,
            "policy_source": decision.get("policy_source") or "local-default",
            "condition_keys": [],
            "action_keys": ["model_hint"],
        }
    if action_family == "cache" and (
        decision.get("canary") == "codex-app-exact-cache" or isinstance(decision.get("canary_sample"), dict)
    ):
        policy_id = str(decision.get("policy_id") or "local-codex-app-exact-cache-canary")
        return {
            "rule_id": policy_id,
            "candidate_id": policy_id,
            "policy_id": policy_id,
            "policy_source": decision.get("policy_source") or "local-default",
            "condition_keys": [],
            "action_keys": ["cache_eligible"],
        }
    return None

def _codex_rule_report_cohort(decision: dict[str, Any]) -> str:
    safety = decision.get("safety_stop") if isinstance(decision.get("safety_stop"), dict) else {}
    if bool(safety.get("tripped")) or str(decision.get("status") or "") == "safety_stopped":
        return "safety_stopped"
    cohort = str(decision.get("canary_cohort") or "")
    if cohort in {"canary_applied", "canary_holdout", "not_selected"}:
        return cohort
    status = str(decision.get("status") or "")
    if status == "applied":
        return "canary_applied"
    if status == "holdout":
        return "canary_holdout"
    if status in {"skipped", "not-applied", "eligible-skipped"}:
        return "skipped"
    return status or "unknown"

def _new_codex_rule_report_bucket(meta: dict[str, Any], *, action_family: str, rule_path: str | None) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.codex_app_canary_rule_impact.v1",
        "action_family": action_family,
        "rule_id": meta.get("rule_id"),
        "candidate_id": meta.get("candidate_id"),
        "policy_id": meta.get("policy_id"),
        "policy_source": meta.get("policy_source"),
        "rule_path": rule_path,
        "condition_keys": list(meta.get("condition_keys") or []),
        "action_keys": list(meta.get("action_keys") or []),
        "observed_rows": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "fallback_count": 0,
        "pass_through_count": 0,
        "safety_stopped_count": 0,
        "pending_count": 0,
        "completed_count": 0,
        "estimated_cost_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "estimated_savings_usd": 0.0,
        "latency_values": [],
        "cohorts": {
            "canary_applied": {"count": 0, "error_count": 0, "latency_values": [], "cost_usd": 0.0, "baseline_cost_usd": 0.0},
            "canary_holdout": {"count": 0, "error_count": 0, "latency_values": [], "cost_usd": 0.0, "baseline_cost_usd": 0.0},
            "skipped": {"count": 0, "error_count": 0, "latency_values": [], "cost_usd": 0.0, "baseline_cost_usd": 0.0},
            "safety_stopped": {"count": 0, "error_count": 0, "latency_values": [], "cost_usd": 0.0, "baseline_cost_usd": 0.0},
            "unknown": {"count": 0, "error_count": 0, "latency_values": [], "cost_usd": 0.0, "baseline_cost_usd": 0.0},
        },
        "cache": {
            "hit_count": 0,
            "miss_count": 0,
            "holdout_count": 0,
            "invalidation_count": 0,
            "stale_risk_count": 0,
            "unsafe_skip_count": 0,
        },
        "reason_counts": {},
        "cache_reason_counts": {},
        "safety_stop_reason_counts": {},
        "last_observed_at": None,
    }

def _finalize_codex_rule_report_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for name, cohort in bucket["cohorts"].items():
        count = _as_int(cohort.get("count"))
        latency_values = list(cohort.get("latency_values") or [])
        cost = _as_float(cohort.get("cost_usd"))
        baseline = _as_float(cohort.get("baseline_cost_usd"))
        cohorts[name] = {
            "count": count,
            "error_count": _as_int(cohort.get("error_count")),
            "error_rate": round(_as_int(cohort.get("error_count")) / count, 6) if count else 0.0,
            "avg_latency_ms": _avg_or_none(latency_values),
            "cost_usd": round(cost, 8),
            "baseline_cost_usd": round(baseline, 8),
            "estimated_savings_usd": round(max(baseline - cost, 0.0), 8),
        }
    applied = cohorts["canary_applied"]
    holdout = cohorts["canary_holdout"]
    latency_delta = None
    if applied["avg_latency_ms"] is not None and holdout["avg_latency_ms"] is not None:
        latency_delta = round(_as_float(applied["avg_latency_ms"]) - _as_float(holdout["avg_latency_ms"]), 2)
    bucket = dict(bucket)
    bucket["estimated_cost_usd"] = round(_as_float(bucket.get("estimated_cost_usd")), 8)
    bucket["baseline_cost_usd"] = round(_as_float(bucket.get("baseline_cost_usd")), 8)
    bucket["estimated_savings_usd"] = round(max(_as_float(bucket.get("baseline_cost_usd")) - _as_float(bucket.get("estimated_cost_usd")), 0.0), 8)
    bucket["avg_latency_ms"] = _avg_or_none(list(bucket.pop("latency_values", []) or []))
    bucket["applied_minus_holdout_latency_avg_ms"] = latency_delta
    bucket["applied_minus_holdout_error_rate"] = round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6)
    bucket["cohorts"] = cohorts
    bucket["reason_breakdown"] = _count_breakdown(bucket.pop("reason_counts", {}))
    bucket["cache_reason_breakdown"] = _count_breakdown(bucket.pop("cache_reason_counts", {}))
    bucket["safety_stop_reason_breakdown"] = _count_breakdown(bucket.pop("safety_stop_reason_counts", {}))
    bucket["privacy"] = {
        "metadata_only": True,
        "raw_prompts_included": False,
        "raw_params_included": False,
        "raw_responses_included": False,
        "raw_tool_payloads_included": False,
        "request_ids_included": False,
        "thread_ids_included": False,
        "local_session_ids_included": False,
        "cache_keys_included": False,
        "provider_bodies_included": False,
    }
    return bucket

async def stats_codex_canary_impact(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10000))
    conn = store_obj.conn
    rows = conn.execute("""
        select s.id as start_event_id,
               s.created_at,
               s.input_text_chars,
               s.routing_json,
               s.crunch_json,
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
               ) as response_latency_ms,
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
        order by s.created_at desc
        limit ?
    """, (capped_limit,)).fetchall()
    policy_state = codex_app_bundle_policy_state()
    rule_path = policy_state.get("rule_path")
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    total_rows = 0
    for raw_row in rows:
        row = dict(raw_row)
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        result_chars = _as_int(row.get("response_result_chars"))
        latency = _as_int(row.get("response_latency_ms"))
        has_error = row.get("response_error_code") is not None
        has_response = row.get("response_error_code") is not None or result_chars > 0 or latency > 0
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), result_chars, cache)
        for action_family, decision in (("routing", routing), ("cache", cache)):
            meta = _codex_rule_report_meta(decision, action_family=action_family)
            if meta is None:
                continue
            key = (action_family, str(meta["rule_id"]), str(meta["candidate_id"]))
            bucket = buckets.setdefault(
                key,
                _new_codex_rule_report_bucket(meta, action_family=action_family, rule_path=str(rule_path) if rule_path else None),
            )
            total_rows += 1
            bucket["observed_rows"] += 1
            bucket["last_observed_at"] = max(str(bucket.get("last_observed_at") or ""), str(row.get("created_at") or "")) or None
            status = str(decision.get("status") or "")
            reason = str(decision.get("reason") or "unknown")
            _increment_count(bucket["reason_counts"], reason)
            if has_error:
                bucket["error_count"] += 1
            elif has_response:
                bucket["completed_count"] += 1
            else:
                bucket["pending_count"] += 1
            if decision.get("fallback_reason"):
                bucket["fallback_count"] += 1
            if status in {"skipped", "not-applied"} and not decision.get("applied"):
                bucket["pass_through_count"] += 1
            if latency:
                bucket["latency_values"].append(latency)
            cost = _as_float(estimates.get("cost_est_usd"))
            baseline = _as_float(estimates.get("baseline_cost_est_usd"))
            bucket["estimated_cost_usd"] += cost
            bucket["baseline_cost_usd"] += baseline

            cohort_name = _codex_rule_report_cohort(decision)
            if cohort_name == "canary_applied":
                bucket["applied_count"] += 1
            elif cohort_name == "canary_holdout":
                bucket["holdout_count"] += 1
            elif cohort_name == "safety_stopped":
                bucket["safety_stopped_count"] += 1
            elif cohort_name in {"skipped", "not_selected", "eligible-skipped"}:
                bucket["skipped_count"] += 1
                cohort_name = "skipped"
            if cohort_name not in bucket["cohorts"]:
                cohort_name = "unknown"
            cohort = bucket["cohorts"][cohort_name]
            cohort["count"] += 1
            if has_error:
                cohort["error_count"] += 1
            if latency:
                cohort["latency_values"].append(latency)
            cohort["cost_usd"] += cost
            cohort["baseline_cost_usd"] += baseline

            safety = decision.get("safety_stop") if isinstance(decision.get("safety_stop"), dict) else {}
            for reason_code in safety.get("reason_codes") or []:
                _increment_count(bucket["safety_stop_reason_counts"], str(reason_code))

            if action_family == "cache":
                cache_status = str(cache.get("status") or "missing")
                cache_reason = str(cache.get("reason") or "unknown")
                cache_bucket = str(cache.get("outcome_bucket") or _codex_cache_readiness_cohort(cache))
                _increment_count(bucket["cache_reason_counts"], cache_reason)
                if cache_status == "hit":
                    bucket["cache"]["hit_count"] += 1
                elif cache_status == "miss":
                    bucket["cache"]["miss_count"] += 1
                if cache_status == "holdout" or cache_bucket == "holdout":
                    bucket["cache"]["holdout_count"] += 1
                if cache_bucket == "invalidated":
                    bucket["cache"]["invalidation_count"] += 1
                if cache_bucket == "stale-risk":
                    bucket["cache"]["stale_risk_count"] += 1
                if cache_bucket == "unsafe-skip":
                    bucket["cache"]["unsafe_skip_count"] += 1

    rules = [_finalize_codex_rule_report_bucket(bucket) for bucket in buckets.values()]
    rules.sort(key=lambda item: (_as_int(item.get("observed_rows")), str(item.get("rule_id") or "")), reverse=True)
    return {
        "schema": "tokenclaw.codex_app_canary_impact_by_rule.v1",
        "generated_at": utc_now(),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "limit": capped_limit,
        "summary": {
            "rule_candidate_count": len(rules),
            "observed_rule_action_rows": total_rows,
            "applied_count": sum(_as_int(row.get("applied_count")) for row in rules),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in rules),
            "skipped_count": sum(_as_int(row.get("skipped_count")) for row in rules),
            "error_count": sum(_as_int(row.get("error_count")) for row in rules),
            "fallback_count": sum(_as_int(row.get("fallback_count")) for row in rules),
            "pass_through_count": sum(_as_int(row.get("pass_through_count")) for row in rules),
            "safety_stopped_count": sum(_as_int(row.get("safety_stopped_count")) for row in rules),
            "cache_hit_count": sum(_as_int(row.get("cache", {}).get("hit_count")) for row in rules),
            "cache_miss_count": sum(_as_int(row.get("cache", {}).get("miss_count")) for row in rules),
            "cache_holdout_count": sum(_as_int(row.get("cache", {}).get("holdout_count")) for row in rules),
            "cache_invalidation_count": sum(_as_int(row.get("cache", {}).get("invalidation_count")) for row in rules),
            "estimated_savings_usd": round(sum(_as_float(row.get("estimated_savings_usd")) for row in rules), 8),
        },
        "policy": {
            "policy_source": policy_state.get("policy_source"),
            "rule_path": rule_path,
            "reload_required": bool(((policy_state.get("file") or {}) if isinstance(policy_state.get("file"), dict) else {}).get("reload_required")),
        },
        "rules": rules,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "request_ids_included": False,
            "thread_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "provider_bodies_included": False,
            "managed_server_calls_made": False,
        },
    }

def _breakdown_lookup(rows: list[dict[str, Any]], key: str = "value") -> dict[str, int]:
    return {str(row.get(key) or "unknown"): _as_int(row.get("count")) for row in rows if isinstance(row, dict)}

def _codex_cache_readiness_cohort(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "missing")
    reason = str(row.get("reason") or "unknown")
    if reason in {"codex-app-cache-disabled", "cache-disabled", "streaming-cache-disabled"}:
        return "disabled"
    if status == "hit":
        return "hit"
    if status == "holdout" or reason in {"codex-app-cache-canary-holdout", "canary_holdout"}:
        return "holdout"
    if reason in {"dependency-changed", "dependency-deleted", "dependency-created", "codex-cache-ttl-expired"}:
        return "invalidated"
    if reason in {"file-dependency-missing", "dependency-missing", "dependency-cap-exceeded", "file-watch-disabled"}:
        return "stale-risk"
    if status == "unsafe-skip" or reason in {"action-like-params", "non-text-input", "unknown-param-shape", "unsafe-cached-envelope"}:
        return "unsafe-skip"
    if status == "miss":
        return "miss"
    return status or "unknown"

def _codex_cache_readiness_cohorts(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in cache_rows:
        if not isinstance(row, dict):
            continue
        cohort = _codex_cache_readiness_cohort(row)
        bucket = grouped.setdefault(
            cohort,
            {
                "cohort": cohort,
                "count": 0,
                "status_breakdown": {},
                "reason_breakdown": {},
                "policy_source_breakdown": {},
            },
        )
        count = _as_int(row.get("count"))
        bucket["count"] += count
        for breakdown_key, value in (
            ("status_breakdown", row.get("status") or "missing"),
            ("reason_breakdown", row.get("reason") or "unknown"),
            ("policy_source_breakdown", row.get("policy_source") or "unknown"),
        ):
            values = bucket[breakdown_key]
            label = str(value)
            values[label] = _as_int(values.get(label)) + count

    result: list[dict[str, Any]] = []
    for bucket in grouped.values():
        bucket["status_breakdown"] = _count_breakdown(bucket["status_breakdown"])
        bucket["reason_breakdown"] = _count_breakdown(bucket["reason_breakdown"])
        bucket["policy_source_breakdown"] = _count_breakdown(bucket["policy_source_breakdown"])
        result.append(bucket)
    result.sort(key=lambda item: (_as_int(item.get("count")), str(item.get("cohort") or "")), reverse=True)
    return result

def _codex_readiness_check(name: str, status: str, detail: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "metrics": metrics,
    }

def _openai_codex_blocker(codex_readiness: dict[str, Any], *, state: str) -> str | None:
    if state == "demo_only":
        return "no-live-openai-or-codex-savings-evidence"
    for check in codex_readiness.get("readiness_checks") or []:
        if not isinstance(check, dict):
            continue
        if check.get("status") == "blocked":
            return str(check.get("name") or check.get("detail") or "codex-readiness-blocked")
    if state == "ready":
        return "waiting-for-applied-savings-evidence"
    return None

async def stats_openai_codex_readiness(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 5000))
    golden = build_golden_path_summary(store=store_obj, limit=capped_limit)
    codex_readiness = await stats_codex_readiness(store_obj, limit=min(capped_limit, 1000))
    codex_impact = await stats_codex_canary_impact(store_obj, limit=capped_limit)

    live = golden.get("live_evidence") if isinstance(golden.get("live_evidence"), dict) else {}
    fixture = golden.get("fixture") if isinstance(golden.get("fixture"), dict) else {}
    codex_summary = codex_readiness.get("summary") if isinstance(codex_readiness.get("summary"), dict) else {}
    codex_exact_cache = codex_readiness.get("exact_cache") if isinstance(codex_readiness.get("exact_cache"), dict) else {}
    codex_impact_summary = codex_impact.get("summary") if isinstance(codex_impact.get("summary"), dict) else {}

    openai_live_savings = _as_float(live.get("estimated_tokenclaw_savings_usd"))
    openai_live_family = str(live.get("local_action_family") or "none")
    openai_live_active = bool(
        live.get("status") == "active"
        and (
            openai_live_savings > 0.0
            or _as_int(live.get("routing_applied_count")) > 0
            or _as_int(live.get("crunch_changed_count")) > 0
        )
    )

    codex_hint_savings = _as_float(codex_summary.get("summary_model_hint_estimated_savings_usd"))
    codex_cache_savings = _as_float(codex_exact_cache.get("estimated_saved_cost_usd"))
    codex_impact_savings = _as_float(codex_impact_summary.get("estimated_savings_usd"))
    codex_live_savings = max(codex_hint_savings + codex_cache_savings, codex_impact_savings)
    codex_turn_count = _as_int(codex_summary.get("turn_start_rows"))
    codex_active = bool(
        codex_live_savings > 0.0
        or _as_int(codex_summary.get("summary_model_hint_applied")) > 0
        or _as_int(codex_summary.get("exact_cache_hits")) > 0
    )
    codex_ready = codex_readiness.get("status") == "ready"
    codex_blocked = any(
        isinstance(check, dict) and check.get("status") == "blocked"
        for check in codex_readiness.get("readiness_checks") or []
    )

    live_savings = openai_live_savings + codex_live_savings
    fixture_savings = _as_float(fixture.get("estimated_tokenclaw_savings_usd") or golden.get("estimated_tokenclaw_savings_usd"))
    if openai_live_active or codex_active:
        state = "active"
        tokenclaw_savings = live_savings
        evidence_basis = "live-local-metadata"
    elif codex_ready:
        state = "ready"
        tokenclaw_savings = 0.0
        evidence_basis = "ready-local-metadata"
    elif codex_turn_count > 0 and codex_blocked:
        state = "blocked"
        tokenclaw_savings = 0.0
        evidence_basis = "blocked-local-metadata"
    else:
        state = "demo_only"
        tokenclaw_savings = fixture_savings
        evidence_basis = "golden-path-fixture"

    codex_family = "cache" if codex_cache_savings >= codex_hint_savings and codex_cache_savings > 0 else "routing"
    if not codex_active:
        codex_family = "none"
    if openai_live_active and (openai_live_savings >= codex_live_savings or not codex_active):
        active_surface = str(live.get("surface") or "openai_responses")
        active_action_family = openai_live_family if openai_live_family in {"routing", "crunch", "cache"} else "routing"
    elif codex_active or codex_ready:
        active_surface = CODEX_APP_SOURCE_SURFACE
        active_action_family = codex_family if codex_active else "none"
    else:
        active_surface = "none"
        active_action_family = "none"

    top_blocker = _openai_codex_blocker(codex_readiness, state=state)
    provider_prompt_cache_discount = _openai_provider_prompt_cache_discount(store_obj, limit=capped_limit)

    return {
        "schema": "tokenclaw.openai_codex_savings_readiness.v1",
        "generated_at": utc_now(),
        "vertical": "openai_codex_savings",
        "state": state,
        "active_surface": active_surface,
        "active_action_family": active_action_family,
        "demonstrated_action_family": fixture.get("local_action_family") or "none",
        "tokenclaw_generated_savings_usd": round(tokenclaw_savings, 8),
        "live_tokenclaw_generated_savings_usd": round(live_savings, 8),
        "demonstrated_tokenclaw_savings_usd": round(fixture_savings, 8),
        "provider_prompt_cache_discount_usd": provider_prompt_cache_discount,
        "top_blocker_reason": top_blocker,
        "rollback_available": bool(state == "active" and active_action_family in {"routing", "crunch", "cache"}),
        "managed_server_required": False,
        "evidence_basis": evidence_basis,
        "surfaces": {
            "openai_responses": {
                "state": "active" if openai_live_active else "demo_available",
                "tokenclaw_generated_savings_usd": round(openai_live_savings, 8),
                "local_action_family": openai_live_family,
                "routing_applied_count": _as_int(live.get("routing_applied_count")),
                "crunch_changed_count": _as_int(live.get("crunch_changed_count")),
            },
            CODEX_APP_SOURCE_SURFACE: {
                "state": "active" if codex_active else str(codex_readiness.get("status") or "no-data"),
                "tokenclaw_generated_savings_usd": round(codex_live_savings, 8),
                "turn_start_rows": codex_turn_count,
                "summary_model_hint_applied": _as_int(codex_summary.get("summary_model_hint_applied")),
                "exact_cache_hits": _as_int(codex_summary.get("exact_cache_hits")),
            },
        },
        "savings_breakdown": {
            "tokenclaw_generated_savings_usd": round(tokenclaw_savings, 8),
            "live_tokenclaw_generated_savings_usd": round(live_savings, 8),
            "demonstrated_tokenclaw_savings_usd": round(fixture_savings, 8),
            "provider_prompt_cache_discount_usd": provider_prompt_cache_discount,
            "basis": "AgentFlow-generated routing/crunch/cache savings are kept separate from provider prompt-cache discounts.",
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_response_bodies_included": False,
            "raw_transcripts_included": False,
            "request_ids_included": False,
            "thread_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_read_only": True,
        },
    }

async def stats_codex_readiness(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 5000))
    effectiveness = await stats_codex_effectiveness(store_obj, limit=capped_limit)
    policies = await stats_policies()
    summary = effectiveness.get("summary") if isinstance(effectiveness.get("summary"), dict) else {}
    quota = effectiveness.get("quota_and_token_usage") if isinstance(effectiveness.get("quota_and_token_usage"), dict) else {}
    reconciliation = quota.get("reconciliation") if isinstance(quota.get("reconciliation"), dict) else {}
    hint = effectiveness.get("summary_model_hint") if isinstance(effectiveness.get("summary_model_hint"), dict) else {}
    hint_summary = hint.get("summary") if isinstance(hint.get("summary"), dict) else {}
    hint_canary = hint.get("canary") if isinstance(hint.get("canary"), dict) else {}
    source_surfaces = policies.get("source_surfaces") if isinstance(policies.get("source_surfaces"), dict) else {}
    surface_policy = source_surfaces.get(CODEX_APP_SOURCE_SURFACE) if isinstance(source_surfaces.get(CODEX_APP_SOURCE_SURFACE), dict) else {}
    codex_policy = policies.get("codex_app") if isinstance(policies.get("codex_app"), dict) else {}
    cache_rows = effectiveness.get("cache_breakdown") if isinstance(effectiveness.get("cache_breakdown"), list) else []
    cache_cohorts = _codex_cache_readiness_cohorts(cache_rows)
    cache_counts = _breakdown_lookup(cache_cohorts, key="cohort")
    phase_counts = _breakdown_lookup(effectiveness.get("workflow_phase_counts") or [])
    phase_unknown = _as_int(summary.get("workflow_phase_unknown") or phase_counts.get("unknown"))
    turn_count = _as_int(summary.get("turn_start_rows"))
    phase_known = max(_as_int(summary.get("workflow_phase_known")), 0)
    phase_known_rate = round(phase_known / turn_count, 4) if turn_count else 0.0
    unknown_phase_reasons: list[dict[str, Any]] = []
    for row in effectiveness.get("workflow_phase_breakdown") or []:
        if isinstance(row, dict) and str(row.get("phase") or "") == "unknown":
            unknown_phase_reasons = list(row.get("phase_reasons") or [])
            break
    feedback_queue = _managed_feedback_queue_health(store_obj, sample_limit=5, source_surface=CODEX_APP_SOURCE_SURFACE)
    feedback_summary = feedback_queue.get("summary") if isinstance(feedback_queue.get("summary"), dict) else {}

    token_reconciliation_status = str(reconciliation.get("status") or reconciliation.get("total_drift_bucket") or "unknown")
    checks = [
        _codex_readiness_check(
            "Recent Codex telemetry",
            "ready" if turn_count > 0 else "no-data",
            "Codex turn/start rows are available for readiness scoring." if turn_count > 0 else "No recent Codex turns are available in the selected sample.",
            {"turn_start_rows": turn_count, "completed_rows": _as_int(summary.get("completed_rows")), "pending_rows": _as_int(summary.get("pending_rows")), "error_rows": _as_int(summary.get("error_rows"))},
        ),
        _codex_readiness_check(
            "Workflow phase coverage",
            "ready" if turn_count > 0 and phase_unknown == 0 else ("partial" if phase_known > 0 else "blocked"),
            "All sampled turns have known workflow phases." if turn_count > 0 and phase_unknown == 0 else "Some sampled turns still have unknown workflow phases.",
            {"known": phase_known, "unknown": phase_unknown, "known_rate": phase_known_rate},
        ),
        _codex_readiness_check(
            "Token reconciliation",
            "ready" if token_reconciliation_status == "reconciled" else ("partial" if quota.get("token_usage_update_count") else "no-data"),
            "Codex tokenUsage updates reconcile to sampled AgentFlow turn windows." if token_reconciliation_status == "reconciled" else "Token usage is not fully reconciled for the sampled Codex turns.",
            {
                "status": token_reconciliation_status,
                "token_usage_updates": _as_int(quota.get("token_usage_update_count")),
                "total_drift_tokens": _as_int(reconciliation.get("total_drift_tokens")),
                "drift_size_bucket": reconciliation.get("total_drift_size_bucket"),
            },
        ),
        _codex_readiness_check(
            "Summary model hint canary",
            "ready" if _as_int(hint_summary.get("applied")) > 0 and _as_int(hint_summary.get("holdout")) > 0 else ("partial" if _as_int(hint_summary.get("turns")) > 0 else "no-data"),
            "Applied and holdout summary-model-hint cohorts are both present." if _as_int(hint_summary.get("applied")) > 0 and _as_int(hint_summary.get("holdout")) > 0 else "Summary-model-hint metadata is missing applied or holdout evidence.",
            {"turns": _as_int(hint_summary.get("turns")), "applied": _as_int(hint_summary.get("applied")), "holdout": _as_int(hint_summary.get("holdout")), "unsafe_skipped": _as_int(hint_summary.get("unsafe_skipped")), "error_rate": _as_float(hint_summary.get("error_rate"))},
        ),
        _codex_readiness_check(
            "Exact response cache canary",
            "ready" if cache_counts.get("hit", 0) > 0 or cache_counts.get("holdout", 0) > 0 else ("partial" if cache_counts.get("miss", 0) > 0 else "disabled"),
            "Exact-cache canary cohorts have replay or holdout evidence." if cache_counts.get("hit", 0) > 0 or cache_counts.get("holdout", 0) > 0 else "Exact-cache cohorts are disabled or have only miss/skip evidence.",
            {"disabled": cache_counts.get("disabled", 0), "miss": cache_counts.get("miss", 0), "hit": cache_counts.get("hit", 0), "holdout": cache_counts.get("holdout", 0), "invalidated": cache_counts.get("invalidated", 0)},
        ),
        _codex_readiness_check(
            "Managed feedback queue",
            "ready" if _as_int(feedback_summary.get("due")) == 0 and _as_int(feedback_summary.get("retryable_error")) == 0 and _as_int(feedback_summary.get("dropped_after_limit")) == 0 else "blocked",
            "No due, retryable, or dropped Codex feedback rows are present." if _as_int(feedback_summary.get("due")) == 0 and _as_int(feedback_summary.get("retryable_error")) == 0 and _as_int(feedback_summary.get("dropped_after_limit")) == 0 else "Managed feedback has due, retryable, or dropped rows.",
            {"queued": _as_int(feedback_summary.get("queued")), "due": _as_int(feedback_summary.get("due")), "retryable_error": _as_int(feedback_summary.get("retryable_error")), "dropped_after_limit": _as_int(feedback_summary.get("dropped_after_limit")), "sent": _as_int(feedback_summary.get("sent"))},
        ),
    ]
    if turn_count <= 0:
        readiness = "no-data"
    elif any(check["status"] == "blocked" for check in checks):
        readiness = "blocked"
    elif any(check["status"] in {"partial", "disabled", "no-data"} for check in checks):
        readiness = "partial"
    else:
        readiness = "ready"

    return {
        "schema": "tokenclaw.codex_optimization_readiness.v1",
        "generated_at": utc_now(),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "limit": capped_limit,
        "status": readiness,
        "summary": {
            "turn_start_rows": turn_count,
            "completed_rows": _as_int(summary.get("completed_rows")),
            "error_rows": _as_int(summary.get("error_rows")),
            "pending_rows": _as_int(summary.get("pending_rows")),
            "phase_known_rate": phase_known_rate,
            "token_reconciliation_status": token_reconciliation_status,
            "summary_model_hint_applied": _as_int(hint_summary.get("applied")),
            "summary_model_hint_holdout": _as_int(hint_summary.get("holdout")),
            "summary_model_hint_estimated_savings_usd": round(_as_float(hint_summary.get("estimated_savings_usd")), 8),
            "exact_cache_hits": cache_counts.get("hit", 0),
            "exact_cache_holdouts": cache_counts.get("holdout", 0),
            "exact_cache_misses": cache_counts.get("miss", 0),
            "exact_cache_invalidations": cache_counts.get("invalidated", 0),
            "managed_feedback_queued": _as_int(feedback_summary.get("queued")),
            "managed_feedback_due": _as_int(feedback_summary.get("due")),
        },
        "policy": {
            "optimization_enabled": bool(((surface_policy.get("optimization") or {}) if isinstance(surface_policy.get("optimization"), dict) else {}).get("enabled")),
            "policy_source": codex_policy.get("policy_source") or surface_policy.get("policy_source"),
            "rule_path": codex_policy.get("rule_path") or surface_policy.get("rule_path"),
            "reload_required": bool(((codex_policy.get("file") or {}) if isinstance(codex_policy.get("file"), dict) else {}).get("reload_required")),
            "review_only": bool(codex_policy.get("review_only")),
            "cache_enabled": bool(((surface_policy.get("cache") or {}) if isinstance(surface_policy.get("cache"), dict) else {}).get("enabled")),
            "summary_model_hint": ((surface_policy.get("routing") or {}) if isinstance(surface_policy.get("routing"), dict) else {}).get("summary_model_hint") or {},
        },
        "readiness_checks": checks,
        "workflow_phase": {
            "known": phase_known,
            "unknown": phase_unknown,
            "known_rate": phase_known_rate,
            "counts": effectiveness.get("workflow_phase_counts") or [],
            "source_breakdown": effectiveness.get("workflow_phase_source_breakdown") or [],
            "unknown_reasons": unknown_phase_reasons,
        },
        "token_reconciliation": {
            "status": token_reconciliation_status,
            "drift_tokens": _as_int(reconciliation.get("total_drift_tokens")),
            "drift_size_bucket": reconciliation.get("total_drift_size_bucket"),
            "status_breakdown": reconciliation.get("status_breakdown") or [],
            "reconciled_cost_usd": round(_as_float(quota.get("reconciled_cost_usd")), 8),
            "reconciled_cost_known": bool(quota.get("reconciled_cost_known")),
        },
        "summary_model_hint": {
            "summary": hint_summary,
            "canary": hint_canary,
            "buckets": hint.get("buckets") or [],
            "estimated_savings_usd": round(_as_float(hint_summary.get("estimated_savings_usd")), 8),
        },
        "exact_cache": {
            "cohorts": cache_cohorts,
            "decision_breakdown": cache_rows,
            "estimated_saved_cost_usd": round(_as_float(summary.get("cache_estimated_savings_usd")), 8),
            "cost_delta_basis": "Codex exact cache impact is estimated from metadata-only local replay decisions and char-derived Codex turn costs.",
        },
        "managed_feedback_queue": feedback_queue,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "raw_commands_included": False,
            "request_ids_included": False,
            "thread_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "queue_payload_json_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
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
    quality_signals = derive_codex_turn_quality_signals(
        created_at=r.get("created_at"),
        response_event_id=response_event_id,
        error_code=error_code,
        error_message=r.get("response_error_message"),
        latency_ms=r.get("response_latency_ms"),
        routing_meta=routing,
        crunch_meta=crunch,
        cache_meta=cache,
    )
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
        "feature_schema_version": "tokenclaw.optimization_unit_features.v1",
        "schema": "tokenclaw.optimization_unit.v1",
        "unit_id": f"codex_turn:{r.get('start_event_id')}",
        "created_at": r.get("created_at"),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "app_family": "codex",
        "requested_model": requested_model,
        "candidate_target_model": target_model,
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
            "quality_signals": quality_signals,
        },
        "quality_signals": quality_signals,
        "replayability_level": str(cache.get("replayability_level") or "features_only"),
        "privacy_summary": {
            "telemetry_profile": "metadata-only",
            "raw_body_storage": False,
            "metadata_only": True,
            "aggregate_only": False,
        },
        "local_ids": {
            "codex_app_start_event_id": r.get("start_event_id"),
            "codex_app_response_event_id": response_event_id,
            "request_id": r.get("request_id"),
            "thread_id": r.get("thread_id"),
            "session_id": r.get("session_id"),
        },
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
        "provider_prompt_cache_discount_usd": 0.0,
        "provider_prompt_cache_net_discount_usd": 0.0,
        "hard_floor_usd": _as_float(outcome_features.get("hard_floor_usd")),
        "policy_sources": list(optimization_features.get("policy_sources") or []),
        "is_today": bool(dict(row).get("is_today")),
    }
