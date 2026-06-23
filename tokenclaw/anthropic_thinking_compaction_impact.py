from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.crunch import TOKEN_CHARS
from tokenclaw.pricing import estimate_blended_input_savings
from tokenclaw.managed_egress import assert_managed_egress_safe
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.anthropic_thinking_compaction_impact.v1"
FEEDBACK_SCHEMA = "tokenclaw.anthropic_thinking_compaction_budget_feedback.v1"
MANAGED_FEEDBACK_SCHEMA = "tokenclaw.anthropic_thinking_compaction_outcome_feedback.v1"
MANAGED_FEEDBACK_SOURCE_SURFACE = "anthropic_thinking_compaction_outcome"
MANAGED_FEEDBACK_EVENT_TYPE = "crunch_outcome_rollup"
SERVER_OUTCOME_FIELDS_SCHEMA = "tokenclaw.anthropic_thinking_compaction_server_outcome_fields.v1"
CONTINUATION_QUALITY_SCHEMA = "tokenclaw.thinking_tail_continuation_quality.v1"
CONTINUATION_WINDOW_MINUTES = 30

# The metadata-only outcome fields the managed server asks local thinking-tail
# clients to supply (tokenclaw_server#330, requested through ``/v1/client-contract``
# for ``action_family=crunch`` + ``source_surface=anthropic_messages`` thinking-tail
# coverage). The rollup carries each of these names so managed scoring can move a
# cohort between hold/canary/widen/rollback from real applied-vs-holdout evidence.
# Every value is a privacy-preserving aggregate; raw status codes, latencies, and
# per-call identifiers are never emitted.
SERVER_REQUESTED_OUTCOME_FIELDS = (
    "action_state",
    "traffic_treatment",
    "cohort",
    "applied_action_families",
    "vetoed_action_families",
    "status_code",
    "error_class",
    "latency_ms",
    "retry_count",
    "fallback_count",
    "actual_input_tokens",
    "actual_output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "saved_chars",
    "saved_tokens",
    "cost_est_usd",
    "realized_savings_usd",
    "safety_stop_reason",
    "rollback_reason",
)

_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")

# Map each observed canary cohort to the action state the executor reached for it.
_COHORT_ACTION_STATE = {
    "applied": "applied",
    "holdout": "heldout",
    "safety_stop": "safety_stopped",
    "skipped": "skipped",
}


def _fallback_observed(value: Any) -> bool:
    """Detect a fallback/rate-limit fallback from bounded metadata flags only.

    Reads fixed boolean/count/enum flag names; never inspects raw bodies.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"fallback", "fallback_observed", "fallback_used", "rate_limit_fallback"} and bool(item):
                return True
            if lowered in {"fallback_count", "fallbacks"} and _as_int(item) > 0:
                return True
            if lowered in {"fallback_reason", "fallback_model", "fallback_from_model"} and item:
                return True
            if _fallback_observed(item):
                return True
    elif isinstance(value, list):
        return any(_fallback_observed(item) for item in value)
    return False


def _error_class(error_count: int, status_breakdown: list[dict[str, Any]]) -> str:
    """Bucketed error class for a cohort, derived from status-class counts only."""
    if error_count <= 0:
        return "none"
    has_client = any(item.get("value") == "4xx" and _as_int(item.get("count")) for item in status_breakdown)
    has_server = any(item.get("value") == "5xx" and _as_int(item.get("count")) for item in status_breakdown)
    if has_client and has_server:
        return "mixed"
    if has_client:
        return "client_error"
    if has_server:
        return "server_error"
    return "error"


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _reason(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    return text if _REASON_RE.match(text) else fallback


def _breakdown(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _status_bucket(value: Any) -> str:
    code = _as_int(value, -1)
    if code < 0:
        return "unknown"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _model_family(value: Any) -> str:
    text = str(value or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    return "unknown"


def _privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_thinking_text_included": False,
        "thinking_block_fingerprints_included": False,
        "raw_tool_payloads_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_files": False,
        "wrote_store": False,
    }


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_rows": 0,
        "retry_attempts": 0,
        "fallback_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "before_chars": 0,
        "saved_chars": 0,
        "planned_saved_chars": 0,
        "tokens_saved_est": 0,
        "planned_saved_tokens": 0,
        "gross_savings_usd": 0.0,
        "projected_savings_usd": 0.0,
        "compaction_cost_usd": 0.0,
        "net_savings_usd": 0.0,
        "thinking_output_tokens": 0,
        "actual_input_tokens": 0,
        "actual_output_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_cache_read_tokens": 0,
        "missing_usage_count": 0,
        "non_positive_savings_count": 0,
        "status_counts": Counter(),
        "reason_counts": Counter(),
    }


def _empty_continuation_cohort() -> dict[str, Any]:
    return {
        "source_count": 0,
        "evaluated_count": 0,
        "missing_next_call_count": 0,
        "success_continuation_count": 0,
        "downstream_issue_count": 0,
        "immediate_error_count": 0,
        "retry_rows": 0,
        "retry_attempts": 0,
        "fallback_count": 0,
        "tool_continuation_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "status_counts": Counter(),
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    error_count = _as_int(raw.get("error_count"))
    retry_rows = _as_int(raw.get("retry_rows"))
    before_chars = _as_int(raw.get("before_chars"))
    saved_chars = _as_int(raw.get("saved_chars"))
    planned_saved_chars = _as_int(raw.get("planned_saved_chars"))
    thinking_tokens = _as_int(raw.get("thinking_output_tokens"))
    cache_creation = _as_int(raw.get("prompt_cache_creation_tokens"))
    cache_read = _as_int(raw.get("prompt_cache_read_tokens"))
    return {
        "count": count,
        "error_count": error_count,
        "retry_rows": retry_rows,
        "retry_attempts": _as_int(raw.get("retry_attempts")),
        "fallback_count": _as_int(raw.get("fallback_count")),
        "error_rate": round(error_count / count, 6) if count else 0.0,
        "retry_rate": round(retry_rows / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "cost_avg_usd": round(_as_float(raw.get("cost_est_usd")) / count, 8) if count else 0.0,
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd")), 8),
        "cost_baseline_usd": round(_as_float(raw.get("cost_baseline_usd")), 8),
        "before_chars": before_chars,
        "saved_chars": saved_chars,
        "planned_saved_chars": planned_saved_chars,
        "avg_crunch_ratio": round(saved_chars / before_chars, 6) if before_chars and saved_chars else 0.0,
        "projected_crunch_ratio": round(planned_saved_chars / before_chars, 6) if before_chars and planned_saved_chars else 0.0,
        "tokens_saved_est": _as_int(raw.get("tokens_saved_est")),
        "planned_saved_tokens": _as_int(raw.get("planned_saved_tokens")),
        "gross_savings_usd": round(_as_float(raw.get("gross_savings_usd")), 8),
        "projected_savings_usd": round(_as_float(raw.get("projected_savings_usd")), 8),
        "compaction_cost_usd": round(_as_float(raw.get("compaction_cost_usd")), 8),
        "net_savings_usd": round(_as_float(raw.get("net_savings_usd")), 8),
        "net_savings_per_call_usd": round(_as_float(raw.get("net_savings_usd")) / count, 8) if count else 0.0,
        "thinking_output_tokens": thinking_tokens,
        "thinking_tokens_avg": round(thinking_tokens / count, 2) if count else 0.0,
        "actual_input_tokens": _as_int(raw.get("actual_input_tokens")),
        "actual_output_tokens": _as_int(raw.get("actual_output_tokens")),
        "prompt_cache_creation_tokens": cache_creation,
        "prompt_cache_read_tokens": cache_read,
        "prompt_cache_read_to_creation_ratio": round(cache_read / cache_creation, 6) if cache_creation else None,
        "missing_usage_count": _as_int(raw.get("missing_usage_count")),
        "missing_usage_rate": round(_as_int(raw.get("missing_usage_count")) / count, 6) if count else 0.0,
        "non_positive_savings_count": _as_int(raw.get("non_positive_savings_count")),
        "non_positive_savings_rate": round(_as_int(raw.get("non_positive_savings_count")) / count, 6) if count else 0.0,
        "status_breakdown": _breakdown(raw.get("status_counts") if isinstance(raw.get("status_counts"), Counter) else Counter()),
        "reason_breakdown": _breakdown(raw.get("reason_counts") if isinstance(raw.get("reason_counts"), Counter) else Counter()),
    }


def _finalize_continuation_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    source_count = _as_int(raw.get("source_count"))
    evaluated_count = _as_int(raw.get("evaluated_count"))
    success_count = _as_int(raw.get("success_continuation_count"))
    issue_count = _as_int(raw.get("downstream_issue_count"))
    immediate_errors = _as_int(raw.get("immediate_error_count"))
    retry_rows = _as_int(raw.get("retry_rows"))
    fallback_count = _as_int(raw.get("fallback_count"))
    tool_count = _as_int(raw.get("tool_continuation_count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    return {
        "source_count": source_count,
        "evaluated_count": evaluated_count,
        "missing_next_call_count": _as_int(raw.get("missing_next_call_count")),
        "evaluation_coverage_rate": round(evaluated_count / source_count, 6) if source_count else 0.0,
        "success_continuation_count": success_count,
        "success_continuation_rate": round(success_count / evaluated_count, 6) if evaluated_count else 0.0,
        "downstream_issue_count": issue_count,
        "downstream_issue_rate": round(issue_count / evaluated_count, 6) if evaluated_count else 0.0,
        "immediate_error_count": immediate_errors,
        "immediate_error_rate": round(immediate_errors / evaluated_count, 6) if evaluated_count else 0.0,
        "retry_rows": retry_rows,
        "retry_attempts": _as_int(raw.get("retry_attempts")),
        "retry_rate": round(retry_rows / evaluated_count, 6) if evaluated_count else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / evaluated_count, 6) if evaluated_count else 0.0,
        "tool_continuation_count": tool_count,
        "tool_continuation_rate": round(tool_count / evaluated_count, 6) if evaluated_count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "status_breakdown": _breakdown(raw.get("status_counts") if isinstance(raw.get("status_counts"), Counter) else Counter()),
    }


def _continuation_deltas(applied: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    applied_latency = applied.get("latency_avg_ms")
    holdout_latency = holdout.get("latency_avg_ms")
    latency_delta = None
    if applied_latency is not None and holdout_latency is not None:
        latency_delta = round(_as_float(applied_latency) - _as_float(holdout_latency), 2)
    return {
        "success_continuation_rate_delta": round(
            _as_float(applied.get("success_continuation_rate")) - _as_float(holdout.get("success_continuation_rate")),
            6,
        ),
        "downstream_issue_rate_delta": round(
            _as_float(applied.get("downstream_issue_rate")) - _as_float(holdout.get("downstream_issue_rate")),
            6,
        ),
        "immediate_error_rate_delta": round(
            _as_float(applied.get("immediate_error_rate")) - _as_float(holdout.get("immediate_error_rate")),
            6,
        ),
        "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
        "fallback_rate_delta": round(_as_float(applied.get("fallback_rate")) - _as_float(holdout.get("fallback_rate")), 6),
        "tool_continuation_rate_delta": round(
            _as_float(applied.get("tool_continuation_rate")) - _as_float(holdout.get("tool_continuation_rate")),
            6,
        ),
        "latency_avg_ms_delta": latency_delta,
    }


def _row_meta(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    meta = crunch.get("anthropic_thinking_history_compaction")
    return meta if isinstance(meta, dict) else {}


def _cohort(meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    lifecycle = meta.get("lifecycle_feedback") if isinstance(meta.get("lifecycle_feedback"), dict) else {}
    status = str(meta.get("status") or lifecycle.get("status") or "").strip().lower()
    cohort = str(canary.get("cohort") or lifecycle.get("cohort") or "").strip().lower()
    reason = str(meta.get("reason") or "").strip().lower()
    if cohort == "canary_applied" or status == "applied" or bool(meta.get("applied")):
        return "applied"
    if cohort == "canary_holdout" or status == "holdout" or reason == "canary_holdout":
        return "holdout"
    if status in {"safety_stop", "safety-stopped"} or meta.get("safety_stop_state") == "stopped" or "safety-stop" in reason:
        return "safety_stop"
    return "skipped"


def _rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit or 500), 10_000))
    where = "where created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since, capped) if since else (capped,)
    sql = f"""
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint, requested_model, routed_model,
               requested_model_family, routed_model_family, stream, status_code,
               latency_ms, cost_est_usd, cost_baseline_usd, category, retry_count,
               actual_input_tokens, actual_output_tokens, input_tokens_est,
               cache_creation_input_tokens, cache_read_input_tokens,
               thinking_output_tokens, crunch_json, routing_json, cache_json, session_id
        from calls
        {where}
        order by created_at desc
        limit ?
    """
    return [dict(row) for row in reversed(store_obj.conn.execute(sql, params).fetchall())]


def _group_key(parts: dict[str, Any]) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _new_group(parts: dict[str, Any]) -> dict[str, Any]:
    candidate = public_id(parts.get("candidate_id"), prefix="thinking-compaction-candidate", fallback="unknown")
    rule = public_id(parts.get("rule_id"), prefix="thinking-compaction-rule", fallback="unknown")
    return {
        **parts,
        "group_key": _group_key(parts),
        "candidate_id": candidate,
        "rule_id": rule,
        "cohorts": {
            "applied": _empty_cohort(),
            "holdout": _empty_cohort(),
            "skipped": _empty_cohort(),
            "safety_stop": _empty_cohort(),
        },
        "first_observed_at": None,
        "last_observed_at": None,
        "affected_sessions": set(),
        "session_costs": {},
        "session_net_savings": {},
        "continuation": {
            "applied": _empty_continuation_cohort(),
            "holdout": _empty_continuation_cohort(),
            "skipped": _empty_continuation_cohort(),
            "safety_stop": _empty_continuation_cohort(),
        },
    }


def _sort_key(row: dict[str, Any]) -> tuple[datetime, str]:
    parsed = _parse_time(row.get("created_at"))
    if parsed is None:
        parsed = datetime.min.replace(tzinfo=timezone.utc)
    return parsed, str(row.get("id") or "")


def _session_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session = row.get("session_id")
        if not session:
            continue
        by_session.setdefault(str(session), []).append(row)
    for session_rows in by_session.values():
        session_rows.sort(key=_sort_key)
    return by_session


def _next_session_row(
    row: dict[str, Any],
    by_session: dict[str, list[dict[str, Any]]],
    *,
    max_minutes: int = CONTINUATION_WINDOW_MINUTES,
) -> dict[str, Any] | None:
    session = row.get("session_id")
    if not session:
        return None
    current_at = _parse_time(row.get("created_at"))
    if current_at is None:
        return None
    max_age_seconds = max(60, int(max_minutes) * 60)
    current_id = str(row.get("id") or "")
    for candidate in by_session.get(str(session), []):
        if str(candidate.get("id") or "") == current_id:
            continue
        candidate_at = _parse_time(candidate.get("created_at"))
        if candidate_at is None or candidate_at <= current_at:
            continue
        if (candidate_at - current_at).total_seconds() > max_age_seconds:
            return None
        return candidate
    return None


def _row_category_and_phase(row: dict[str, Any]) -> tuple[str, str]:
    routing = _json_obj(row.get("routing_json"))
    category = public_label(row.get("category") or routing.get("category") or "unknown", "unknown")
    phase = public_label(routing.get("workflow_phase") or routing.get("phase") or category or "unknown", "unknown")
    return category, phase


def _is_tool_continuation(row: dict[str, Any]) -> bool:
    category, phase = _row_category_and_phase(row)
    normalized = {category.lower().replace("_", "-"), phase.lower().replace("_", "-")}
    return bool(normalized & {"tool-result", "tool-execution"})


def _row_fallback_observed(row: dict[str, Any]) -> bool:
    def runtime_fallback(value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered == "managed_recommendation":
                    continue
                if lowered in {"fallback", "fallback_observed", "fallback_used", "rate_limit_fallback"}:
                    if isinstance(item, bool) and item:
                        return True
                    if lowered != "fallback" and bool(item):
                        return True
                if lowered in {"fallback_count", "fallbacks"} and _as_int(item) > 0:
                    return True
                if lowered in {"fallback_reason", "fallback_model", "fallback_from_model"} and item:
                    return True
                if runtime_fallback(item):
                    return True
        elif isinstance(value, list):
            return any(runtime_fallback(item) for item in value)
        return False

    return any(runtime_fallback(_json_obj(row.get(column))) for column in ("routing_json", "crunch_json", "cache_json"))


def _add_continuation_row(raw: dict[str, Any], next_row: dict[str, Any] | None) -> None:
    raw["source_count"] += 1
    if next_row is None:
        raw["missing_next_call_count"] += 1
        return

    raw["evaluated_count"] += 1
    status_code = _as_int(next_row.get("status_code"), -1)
    retry_count = _as_int(next_row.get("retry_count"))
    fallback = _row_fallback_observed(next_row)
    errored = status_code >= 400 if status_code >= 0 else False
    issue = errored or retry_count > 0 or fallback
    latency_ms = _as_int(next_row.get("latency_ms"), -1)

    raw["status_counts"][_status_bucket(status_code)] += 1
    raw["immediate_error_count"] += int(errored)
    raw["retry_rows"] += int(retry_count > 0)
    raw["retry_attempts"] += retry_count
    raw["fallback_count"] += int(fallback)
    raw["downstream_issue_count"] += int(issue)
    raw["success_continuation_count"] += int(not issue)
    raw["tool_continuation_count"] += int(_is_tool_continuation(next_row))
    if latency_ms >= 0:
        raw["latency_ms_total"] += latency_ms
        raw["latency_sample_count"] += 1


def _finalize_continuation_quality(group: dict[str, Any]) -> dict[str, Any]:
    cohorts = {
        name: _finalize_continuation_cohort(raw)
        for name, raw in (group.get("continuation") or {}).items()
        if isinstance(raw, dict)
    }
    for name in ("applied", "holdout", "skipped", "safety_stop"):
        cohorts.setdefault(name, _finalize_continuation_cohort(_empty_continuation_cohort()))
    return {
        "schema": CONTINUATION_QUALITY_SCHEMA,
        "window": {
            "same_session_scoped": True,
            "next_call_only": True,
            "max_minutes": CONTINUATION_WINDOW_MINUTES,
        },
        "cohorts": cohorts,
        "applied_minus_holdout": _continuation_deltas(cohorts["applied"], cohorts["holdout"]),
        "privacy": _privacy(),
    }


def _parts(row: dict[str, Any], routing: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    provider = str(row.get("provider") or "anthropic").lower()
    source_surface = row.get("source_surface") or routing.get("source_surface") or "anthropic_messages"
    endpoint = row.get("endpoint") or ("messages" if str(row.get("path") or "").endswith("/messages") else "unknown")
    requested = row.get("requested_model")
    routed = row.get("routed_model") or requested
    return {
        "provider": public_label(provider, "unknown"),
        "source_surface": public_label(source_surface, "unknown"),
        "endpoint": public_label(endpoint, "unknown"),
        "category": public_label(meta.get("category") or row.get("category") or routing.get("category") or "unknown", "unknown"),
        "workflow_phase": public_label(routing.get("workflow_phase") or routing.get("phase") or row.get("category") or "unknown", "unknown"),
        "requested_model_family": public_label(row.get("requested_model_family") or _model_family(requested), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family") or _model_family(routed), "unknown"),
        "stream": bool(_as_int(row.get("stream"))),
        "policy_source": public_label(meta.get("policy_source") or "unknown", "unknown"),
        "rule_id": meta.get("rule_id") or "local-anthropic-thinking-history-compaction-canary",
        "candidate_id": meta.get("candidate_id") or meta.get("rule_id") or "local-anthropic-thinking-history-compaction-canary",
    }


def _tokens_from_meta(meta: dict[str, Any], key: str, chars_key: str) -> int:
    tokens = _as_int(meta.get(key))
    if tokens <= 0 and _as_int(meta.get(chars_key)) > 0:
        tokens = max(0, _as_int(meta.get(chars_key)) // TOKEN_CHARS)
    return tokens


def _chars_from_meta(meta: dict[str, Any], key: str, tokens_key: str) -> int:
    chars = _as_int(meta.get(key))
    if chars <= 0 and _as_int(meta.get(tokens_key)) > 0:
        chars = max(0, _as_int(meta.get(tokens_key)) * TOKEN_CHARS)
    return chars


def _gross_savings(row: dict[str, Any], tokens_saved: int) -> float:
    return estimate_blended_input_savings(
        str(row.get("routed_model") or row.get("requested_model") or "claude-sonnet-4-6"),
        tokens_saved=max(0, tokens_saved),
        input_tokens=_as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est")),
        cache_read_tokens=_as_int(row.get("cache_read_input_tokens")),
        provider=str(row.get("provider") or "anthropic"),
    ) or 0.0


def _add_row(group: dict[str, Any], row: dict[str, Any], meta: dict[str, Any], cohort_name: str) -> None:
    cohort = group["cohorts"].setdefault(cohort_name, _empty_cohort())
    count = _as_int(cohort.get("count")) + 1
    cohort["count"] = count
    status_code = _as_int(row.get("status_code"), -1)
    retry_count = _as_int(row.get("retry_count"))
    latency_ms = _as_int(row.get("latency_ms"), -1)
    errored = status_code >= 400 if status_code >= 0 else False
    cohort["error_count"] += int(errored)
    cohort["retry_rows"] += int(retry_count > 0)
    cohort["retry_attempts"] += retry_count
    cohort["fallback_count"] += int(_fallback_observed(meta))
    if latency_ms >= 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
    cohort["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
    cohort["cost_baseline_usd"] += _as_float(row.get("cost_baseline_usd"))
    cohort["thinking_output_tokens"] += _as_int(row.get("thinking_output_tokens"))
    cohort["actual_input_tokens"] += _as_int(row.get("actual_input_tokens"))
    cohort["actual_output_tokens"] += _as_int(row.get("actual_output_tokens"))
    cohort["prompt_cache_creation_tokens"] += _as_int(row.get("cache_creation_input_tokens"))
    cohort["prompt_cache_read_tokens"] += _as_int(row.get("cache_read_input_tokens"))
    cohort["missing_usage_count"] += int(_as_int(row.get("actual_input_tokens")) <= 0 or _as_int(row.get("actual_output_tokens")) <= 0)
    cohort["status_counts"][_status_bucket(status_code)] += 1
    cohort["reason_counts"][_reason(meta.get("reason"), "unknown")] += 1

    tokens_saved = _tokens_from_meta(meta, "tokens_saved_est", "saved_chars")
    planned_tokens = _tokens_from_meta(meta, "planned_saved_tokens", "planned_saved_chars")
    saved_chars = _chars_from_meta(meta, "saved_chars", "tokens_saved_est")
    planned_saved_chars = _chars_from_meta(meta, "planned_saved_chars", "planned_saved_tokens")
    if cohort_name == "applied" and planned_tokens <= 0:
        planned_tokens = tokens_saved
    if cohort_name == "applied" and planned_saved_chars <= 0:
        planned_saved_chars = saved_chars
    gross = _gross_savings(row, tokens_saved if cohort_name == "applied" else planned_tokens)
    projected_gross = _gross_savings(row, planned_tokens)
    compaction_cost = _as_float(meta.get("compaction_cost_usd"))
    net = gross - compaction_cost if cohort_name == "applied" else 0.0
    cohort["before_chars"] += _as_int(meta.get("before_chars"))
    cohort["saved_chars"] += saved_chars
    cohort["planned_saved_chars"] += planned_saved_chars
    cohort["tokens_saved_est"] += tokens_saved
    cohort["planned_saved_tokens"] += planned_tokens
    cohort["gross_savings_usd"] += gross
    cohort["projected_savings_usd"] += projected_gross
    cohort["compaction_cost_usd"] += compaction_cost
    cohort["net_savings_usd"] += net
    if cohort_name == "applied" and net <= 0:
        cohort["non_positive_savings_count"] += 1

    session = row.get("session_id")
    if session:
        session_key = public_id(session, prefix="thinking-session", fallback="session")
        group["affected_sessions"].add(session_key)
        group["session_costs"][session_key] = _as_float(group["session_costs"].get(session_key)) + _as_float(row.get("cost_est_usd"))
        group["session_net_savings"][session_key] = _as_float(group["session_net_savings"].get(session_key)) + net

    observed_at = _parse_time(row.get("created_at"))
    if observed_at is not None:
        first = group.get("first_observed_at")
        last = group.get("last_observed_at")
        group["first_observed_at"] = observed_at if first is None or observed_at < first else first
        group["last_observed_at"] = observed_at if last is None or observed_at > last else last


def _deltas(applied: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    applied_latency = applied.get("latency_avg_ms")
    holdout_latency = holdout.get("latency_avg_ms")
    latency_delta = None
    if applied_latency is not None and holdout_latency is not None:
        latency_delta = round(_as_float(applied_latency) - _as_float(holdout_latency), 2)
    return {
        "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
        "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
        "latency_avg_ms_delta": latency_delta,
        "cost_avg_usd_delta": round(_as_float(applied.get("cost_avg_usd")) - _as_float(holdout.get("cost_avg_usd")), 8),
        "thinking_tokens_avg_delta": round(_as_float(applied.get("thinking_tokens_avg")) - _as_float(holdout.get("thinking_tokens_avg")), 2),
        "net_savings_per_call_usd_delta": round(_as_float(applied.get("net_savings_per_call_usd")), 8),
        "applied_minus_holdout_error_rate": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
    }


def _budget_feedback(candidate: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    applied = candidate["cohorts"]["applied"]
    holdout = candidate["cohorts"]["holdout"]
    safety = candidate["cohorts"]["safety_stop"]
    deltas = candidate["deltas"]
    reasons: list[str] = []
    action = "hold"

    if _as_int(safety.get("count")) > 0:
        action = "suppress"
        reasons.append("safety-stop-observed")
    if _as_int(applied.get("count")) < thresholds["min_applied_samples"]:
        reasons.append("insufficient-applied-samples")
    if _as_int(holdout.get("count")) < thresholds["min_holdout_samples"]:
        reasons.append("insufficient-holdout-samples")
    if reasons and action != "suppress":
        action = "keep-holdout"
    elif _as_float(applied.get("error_rate")) >= thresholds["suppress_error_rate"]:
        action = "suppress"
        reasons.append("suppress-absolute-error-rate")
    elif _as_float(deltas.get("error_rate_delta")) >= thresholds["max_error_rate_delta"]:
        action = "suppress"
        reasons.append("suppress-error-rate-delta")
    elif _as_float(deltas.get("retry_rate_delta")) >= thresholds["max_retry_rate_delta"]:
        action = "hold"
        reasons.append("hold-retry-rate-delta")
    elif _as_float(applied.get("net_savings_usd")) < thresholds["min_net_savings_usd"]:
        action = "hold"
        reasons.append("hold-minimum-net-savings")
    elif _as_float(applied.get("non_positive_savings_rate")) > thresholds["max_non_positive_savings_rate"]:
        action = "hold"
        reasons.append("hold-non-positive-savings")
    elif not reasons:
        action = "widen"
        reasons.append("impact-positive")

    return {
        "schema": FEEDBACK_SCHEMA,
        "target_action_family": "anthropic_thinking_history_compaction",
        "recommended_budget_action": action,
        "reason_codes": reasons,
        "candidate_id": candidate.get("candidate_id"),
        "rule_id": candidate.get("rule_id"),
        "policy_source": candidate.get("policy_source"),
        "cohort_counts": {
            "applied": _as_int(applied.get("count")),
            "holdout": _as_int(holdout.get("count")),
            "skipped": _as_int(candidate["cohorts"]["skipped"].get("count")),
            "safety_stop": _as_int(safety.get("count")),
        },
        "applied_minus_holdout": deltas,
        "observed_net_savings_usd": round(_as_float(applied.get("net_savings_usd")), 8),
        "projected_holdout_savings_usd": round(_as_float(holdout.get("projected_savings_usd")), 8),
        "thinking_token_trend": {
            "applied_avg": applied.get("thinking_tokens_avg"),
            "holdout_avg": holdout.get("thinking_tokens_avg"),
            "delta": deltas.get("thinking_tokens_avg_delta"),
        },
        "prompt_cache_interaction": {
            "applied_cache_read_tokens": _as_int(applied.get("prompt_cache_read_tokens")),
            "holdout_cache_read_tokens": _as_int(holdout.get("prompt_cache_read_tokens")),
            "applied_cache_creation_tokens": _as_int(applied.get("prompt_cache_creation_tokens")),
            "holdout_cache_creation_tokens": _as_int(holdout.get("prompt_cache_creation_tokens")),
        },
        "local_only": True,
        "read_only": True,
        "wrote_store": False,
        "privacy": _privacy(),
    }


def _finalize(group: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {name: _finalize_cohort(raw) for name, raw in group["cohorts"].items()}
    session_costs = list(group["session_costs"].values())
    session_savings = list(group["session_net_savings"].values())
    candidate = {
        "candidate_id": group["candidate_id"],
        "rule_id": group["rule_id"],
        "policy_source": group["policy_source"],
        "provider": group["provider"],
        "source_surface": group["source_surface"],
        "endpoint": group["endpoint"],
        "category": group["category"],
        "workflow_phase": group["workflow_phase"],
        "requested_model_family": group["requested_model_family"],
        "routed_model_family": group["routed_model_family"],
        "stream": group["stream"],
        "first_observed_at": group["first_observed_at"].isoformat() if group.get("first_observed_at") else None,
        "last_observed_at": group["last_observed_at"].isoformat() if group.get("last_observed_at") else None,
        "cohorts": cohorts,
        "deltas": _deltas(cohorts["applied"], cohorts["holdout"]),
        "session_budget_impact": {
            "affected_session_count": len(group["affected_sessions"]),
            "max_session_cost_usd": round(max(session_costs), 8) if session_costs else 0.0,
            "max_session_net_savings_usd": round(max(session_savings), 8) if session_savings else 0.0,
            "session_ids_included": False,
            "raw_session_ids_included": False,
        },
        "continuation_quality": _finalize_continuation_quality(group),
        "privacy": _privacy(),
    }
    candidate["budget_governor_feedback"] = _budget_feedback(candidate, thresholds)
    candidate["verdict"] = candidate["budget_governor_feedback"]["recommended_budget_action"]
    candidate["reason_codes"] = candidate["budget_governor_feedback"]["reason_codes"]
    return candidate


def _summary_feedback(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter(str((item.get("budget_governor_feedback") or {}).get("recommended_budget_action") or "unknown") for item in candidates)
    priority = ("suppress", "hold", "keep-holdout", "widen")
    selected = next((item for item in priority if actions.get(item)), "no-data")
    reasons = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reasons[str(reason)] += 1
    return {
        "schema": FEEDBACK_SCHEMA,
        "target_action_family": "anthropic_thinking_history_compaction",
        "recommended_budget_action": selected,
        "action_breakdown": _breakdown(actions),
        "reason_code_counts": _breakdown(reasons),
        "local_only": True,
        "read_only": True,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }


def _impact_decision_from_action(action: Any) -> str:
    text = str(action or "").strip().lower()
    if text in {"suppress", "stop", "rollback"}:
        return "stop"
    if text in {"widen", "promote"}:
        return "widen"
    return "remain-staged"


def _impact_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(item.get("canary_impact_decision") or "remain-staged") for item in candidates)
    priority = ("stop", "remain-staged", "widen")
    selected = next((item for item in priority if decisions.get(item)), "remain-staged")
    reasons = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reasons[str(reason)] += 1
    return {
        "schema": "tokenclaw.anthropic_thinking_compaction_canary_impact_decision.v1",
        "decision": selected,
        "decision_breakdown": _breakdown(decisions),
        "reason_code_counts": _breakdown(reasons),
        "local_only": True,
        "read_only": True,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }


def _server_outcome_fields(
    cohort: dict[str, Any],
    *,
    cohort_name: str,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Map the tokenclaw_server#330 outcome field names to local cohort aggregates.

    Each requested field name is present so managed scoring stops asking for
    missing coverage. Values are privacy-preserving aggregates (status is reported
    as bucketed classes, latency/retry as cohort aggregates), never raw per-call
    codes or identifiers. ``safety_stop_reason``/``rollback_reason`` carry the
    candidate-level reason only when the cohort or treatment reached that state.
    """
    status_breakdown = cohort.get("status_breakdown") or []
    error_count = _as_int(cohort.get("error_count"))
    is_safety_stop = cohort_name == "safety_stop"
    safety_stop_reason = ctx.get("safety_stop_reason") if is_safety_stop else None
    return {
        "schema": SERVER_OUTCOME_FIELDS_SCHEMA,
        "action_state": _COHORT_ACTION_STATE.get(cohort_name, "skipped"),
        "traffic_treatment": ctx.get("traffic_treatment", "hold"),
        "cohort": cohort_name,
        "applied_action_families": list(ctx.get("applied_action_families") or []),
        "vetoed_action_families": list(ctx.get("vetoed_action_families") or []),
        # Privacy-preserving status reporting: bucketed status classes, not raw codes.
        "status_code": status_breakdown,
        "error_class": _error_class(error_count, status_breakdown),
        "latency_ms": cohort.get("latency_avg_ms"),
        "retry_count": _as_int(cohort.get("retry_attempts")),
        "fallback_count": _as_int(cohort.get("fallback_count")),
        "actual_input_tokens": _as_int(cohort.get("actual_input_tokens")),
        "actual_output_tokens": _as_int(cohort.get("actual_output_tokens")),
        "cache_creation_input_tokens": _as_int(cohort.get("prompt_cache_creation_tokens")),
        "cache_read_input_tokens": _as_int(cohort.get("prompt_cache_read_tokens")),
        "saved_chars": _as_int(cohort.get("saved_chars")),
        "saved_tokens": _as_int(cohort.get("tokens_saved_est")),
        "cost_est_usd": round(_as_float(cohort.get("cost_est_usd")), 8),
        "realized_savings_usd": round(_as_float(cohort.get("gross_savings_usd")), 8),
        "safety_stop_reason": safety_stop_reason,
        "rollback_reason": ctx.get("rollback_reason"),
    }


def _cohort_outcome(cohort: dict[str, Any], *, cohort_name: str = "unknown", ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "count": _as_int(cohort.get("count")),
        "error_count": _as_int(cohort.get("error_count")),
        "retry_rows": _as_int(cohort.get("retry_rows")),
        "retry_attempts": _as_int(cohort.get("retry_attempts")),
        "fallback_count": _as_int(cohort.get("fallback_count")),
        "error_rate": round(_as_float(cohort.get("error_rate")), 6),
        "retry_rate": round(_as_float(cohort.get("retry_rate")), 6),
        "latency_avg_ms": cohort.get("latency_avg_ms"),
        "status_breakdown": cohort.get("status_breakdown") or [],
        "reason_breakdown": cohort.get("reason_breakdown") or [],
        "before_chars": _as_int(cohort.get("before_chars")),
        "saved_chars": _as_int(cohort.get("saved_chars")),
        "planned_saved_chars": _as_int(cohort.get("planned_saved_chars")),
        "estimated_saved_tokens": _as_int(cohort.get("tokens_saved_est")),
        "planned_saved_tokens": _as_int(cohort.get("planned_saved_tokens")),
        "realized_crunch_savings_usd": round(_as_float(cohort.get("gross_savings_usd")), 8),
        "projected_crunch_savings_usd": round(_as_float(cohort.get("projected_savings_usd")), 8),
        "compaction_cost_usd": round(_as_float(cohort.get("compaction_cost_usd")), 8),
        "net_crunch_savings_usd": round(_as_float(cohort.get("net_savings_usd")), 8),
        "prompt_cache_blended_pricing_inputs": {
            "actual_input_tokens": _as_int(cohort.get("actual_input_tokens")),
            "actual_output_tokens": _as_int(cohort.get("actual_output_tokens")),
            "prompt_cache_creation_tokens": _as_int(cohort.get("prompt_cache_creation_tokens")),
            "prompt_cache_read_tokens": _as_int(cohort.get("prompt_cache_read_tokens")),
            "thinking_output_tokens": _as_int(cohort.get("thinking_output_tokens")),
            "cost_est_usd": round(_as_float(cohort.get("cost_est_usd")), 8),
            "cost_baseline_usd": round(_as_float(cohort.get("cost_baseline_usd")), 8),
        },
        "missing_usage_count": _as_int(cohort.get("missing_usage_count")),
        "non_positive_savings_count": _as_int(cohort.get("non_positive_savings_count")),
        "server_outcome_fields": _server_outcome_fields(cohort, cohort_name=cohort_name, ctx=ctx or {}),
    }


def _top_reason(breakdown: Any) -> str | None:
    if not isinstance(breakdown, list):
        return None
    for item in breakdown:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value and str(value) != "unknown":
            return str(value)
    return None


def _candidate_outcome_context(candidate: dict[str, Any]) -> dict[str, Any]:
    """Derive candidate-level treatment/action-state/family/reason metadata.

    Treatment is inferred from the observed cohorts and impact decision: a stop or
    safety-stop is a rollback, an applied cohort is a canary, a holdout-only cohort
    is holdout, and anything else is a hold. All values are bounded enum strings.
    """
    cohorts = candidate.get("cohorts") if isinstance(candidate.get("cohorts"), dict) else {}
    applied = _as_int(((cohorts.get("applied") or {})).get("count"))
    holdout = _as_int(((cohorts.get("holdout") or {})).get("count"))
    safety_stop = cohorts.get("safety_stop") if isinstance(cohorts.get("safety_stop"), dict) else {}
    safety_count = _as_int(safety_stop.get("count"))
    decision = str(candidate.get("canary_impact_decision") or candidate.get("verdict") or "").strip().lower()
    safety_stop_reason = _top_reason(safety_stop.get("reason_breakdown")) if safety_count else None

    if safety_count > 0 or decision in {"stop", "suppress", "rollback"}:
        treatment = "rollback"
        action_state = "safety_stopped"
    elif applied > 0:
        treatment = "canary"
        action_state = "applied"
    elif holdout > 0:
        treatment = "holdout"
        action_state = "heldout"
    else:
        treatment = "hold"
        action_state = "held"

    rollback_reason = safety_stop_reason if treatment == "rollback" else None
    return {
        "traffic_treatment": treatment,
        "action_state": action_state,
        "applied_action_families": ["crunch"] if applied > 0 else [],
        "vetoed_action_families": [],
        "safety_stop_reason": safety_stop_reason,
        "rollback_reason": rollback_reason,
    }


def _crunch_outcome(candidate: dict[str, Any]) -> dict[str, Any]:
    cohorts = candidate.get("cohorts") if isinstance(candidate.get("cohorts"), dict) else {}
    applied = cohorts.get("applied") if isinstance(cohorts.get("applied"), dict) else {}
    holdout = cohorts.get("holdout") if isinstance(cohorts.get("holdout"), dict) else {}
    skipped = cohorts.get("skipped") if isinstance(cohorts.get("skipped"), dict) else {}
    safety = cohorts.get("safety_stop") if isinstance(cohorts.get("safety_stop"), dict) else {}
    deltas = candidate.get("deltas") if isinstance(candidate.get("deltas"), dict) else {}
    ctx = _candidate_outcome_context(candidate)
    return {
        "target_action_family": "anthropic_thinking_history_compaction",
        "candidate_id": public_id(candidate.get("candidate_id"), prefix="thinking-compaction-candidate", fallback="unknown"),
        "rule_id": public_id(candidate.get("rule_id"), prefix="thinking-compaction-rule", fallback="unknown"),
        "policy_source": public_label(candidate.get("policy_source") or "unknown", "unknown"),
        "provider": public_label(candidate.get("provider") or "anthropic", "unknown"),
        "source_surface": public_label(candidate.get("source_surface") or "anthropic_messages", "unknown"),
        "endpoint": public_label(candidate.get("endpoint") or "messages", "unknown"),
        "category": public_label(candidate.get("category") or "unknown", "unknown"),
        "workflow_phase": public_label(candidate.get("workflow_phase") or "unknown", "unknown"),
        "requested_model_family": public_label(candidate.get("requested_model_family") or "unknown", "unknown"),
        "routed_model_family": public_label(candidate.get("routed_model_family") or "unknown", "unknown"),
        "stream": bool(candidate.get("stream")),
        "canary_impact_decision": public_label(candidate.get("canary_impact_decision") or candidate.get("verdict") or "unknown", "unknown"),
        "reason_codes": [public_label(value, "unknown") for value in candidate.get("reason_codes") or []],
        # Candidate-level outcome treatment requested by tokenclaw_server#330.
        "action_state": ctx["action_state"],
        "traffic_treatment": ctx["traffic_treatment"],
        "applied_action_families": ctx["applied_action_families"],
        "vetoed_action_families": ctx["vetoed_action_families"],
        "safety_stop_reason": ctx["safety_stop_reason"],
        "rollback_reason": ctx["rollback_reason"],
        "cohort_counts": {
            "applied": _as_int(applied.get("count")),
            "holdout": _as_int(holdout.get("count")),
            "skipped": _as_int(skipped.get("count")),
            "safety_stop": _as_int(safety.get("count")),
        },
        "outcomes": {
            "applied": _cohort_outcome(applied, cohort_name="applied", ctx=ctx),
            "holdout": _cohort_outcome(holdout, cohort_name="holdout", ctx=ctx),
            "skipped": _cohort_outcome(skipped, cohort_name="skipped", ctx=ctx),
            "safety_stop": _cohort_outcome(safety, cohort_name="safety_stop", ctx=ctx),
        },
        "applied_minus_holdout": {
            "error_rate_delta": round(_as_float(deltas.get("error_rate_delta")), 6),
            "retry_rate_delta": round(_as_float(deltas.get("retry_rate_delta")), 6),
            "latency_avg_ms_delta": deltas.get("latency_avg_ms_delta"),
            "cost_avg_usd_delta": round(_as_float(deltas.get("cost_avg_usd_delta")), 8),
            "thinking_tokens_avg_delta": round(_as_float(deltas.get("thinking_tokens_avg_delta")), 2),
        },
        "continuation_quality": candidate.get("continuation_quality") or {
            "schema": CONTINUATION_QUALITY_SCHEMA,
            "cohorts": {},
            "applied_minus_holdout": {},
            "privacy": _privacy(),
        },
    }


def _breakdown_dict(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        value = public_label(item.get("value") or "unknown", "unknown")
        count = _as_int(item.get("count"))
        if count:
            counts[value] = counts.get(value, 0) + count
    return dict(sorted(counts.items()))


def build_anthropic_thinking_compaction_managed_feedback(
    impact_report: dict[str, Any],
    *,
    now: str | None = None,
) -> dict[str, Any] | None:
    """Build a metadata-only policy event for managed crunch outcome scoring."""
    if not isinstance(impact_report, dict):
        return None
    candidates = [
        _crunch_outcome(candidate)
        for candidate in impact_report.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    if not candidates:
        return None

    from tokenclaw import __version__

    summary = impact_report.get("summary") if isinstance(impact_report.get("summary"), dict) else {}
    basis = {
        "schema": impact_report.get("schema"),
        "generated_at": impact_report.get("generated_at"),
        "candidate_ids": sorted(str(item.get("candidate_id")) for item in candidates if item.get("candidate_id")),
        "rule_ids": sorted(str(item.get("rule_id")) for item in candidates if item.get("rule_id")),
        "cohort_counts": {
            "applied": _as_int(summary.get("applied_count")),
            "holdout": _as_int(summary.get("holdout_count")),
            "skipped": _as_int(summary.get("skipped_count")),
            "safety_stop": _as_int(summary.get("safety_stop_count")),
        },
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    payload = {
        "schema": MANAGED_FEEDBACK_SCHEMA,
        "event_type": MANAGED_FEEDBACK_EVENT_TYPE,
        "occurred_at": now or utc_now(),
        "source_surface": MANAGED_FEEDBACK_SOURCE_SURFACE,
        "target_action_family": "anthropic_thinking_history_compaction",
        "recommendation_id": f"anthropic-thinking-compaction:{digest[:24]}",
        "bundle_hash": f"sha256:{digest}",
        "policy_sections": ["crunch"],
        # Declares which tokenclaw_server#330 outcome fields this rollup supplies for
        # each cohort under ``crunch_outcomes[].outcomes[].server_outcome_fields``.
        "measured_outcome_fields": list(SERVER_REQUESTED_OUTCOME_FIELDS),
        "local_tool_version": __version__,
        "generated_from_schema": impact_report.get("schema"),
        "impact_generated_at": impact_report.get("generated_at"),
        "lookback_limit": _as_int(impact_report.get("lookback_limit")),
        "summary": {
            "candidate_group_count": _as_int(summary.get("candidate_group_count")),
            "applied_count": _as_int(summary.get("applied_count")),
            "holdout_count": _as_int(summary.get("holdout_count")),
            "skipped_count": _as_int(summary.get("skipped_count")),
            "safety_stop_count": _as_int(summary.get("safety_stop_count")),
            "observed_saved_chars": _as_int(summary.get("observed_saved_chars")),
            "observed_saved_tokens": _as_int(summary.get("observed_saved_tokens")),
            "observed_saved_usd": round(_as_float(summary.get("observed_saved_usd")), 8),
            "projected_saved_chars": _as_int(summary.get("projected_saved_chars")),
            "projected_saved_tokens": _as_int(summary.get("projected_saved_tokens")),
            "projected_saved_usd": round(_as_float(summary.get("projected_saved_usd")), 8),
            "net_savings_usd": round(_as_float(summary.get("net_savings_usd")), 8),
            "projected_holdout_savings_usd": round(_as_float(summary.get("projected_holdout_savings_usd")), 8),
            "applied_minus_holdout_error_rate": round(_as_float(summary.get("applied_minus_holdout_error_rate")), 6),
            "applied_minus_holdout_retry_rate": round(_as_float(summary.get("applied_minus_holdout_retry_rate")), 6),
            "budget_governor_action": public_label(summary.get("budget_governor_action") or "unknown", "unknown"),
            "canary_impact_decision": public_label(summary.get("canary_impact_decision") or "unknown", "unknown"),
            "status_breakdown": _breakdown_dict(summary.get("status_breakdown")),
            "cohort_breakdown": _breakdown_dict(summary.get("cohort_breakdown")),
            "reason_code_counts": _breakdown_dict(summary.get("reason_code_counts")),
        },
        "crunch_outcomes": candidates,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprints_included": False,
            "raw_tool_payloads_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
        },
    }
    assert_managed_egress_safe(payload)
    return payload


async def queue_anthropic_thinking_compaction_managed_feedback(
    store_obj: Any,
    impact_report: dict[str, Any],
    *,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    from tokenclaw import recommendations

    payload = build_anthropic_thinking_compaction_managed_feedback(impact_report)
    if payload is None:
        return {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-anthropic-thinking-compaction-outcomes",
            "auth_configured": recommendations.managed_auth_configured(),
            "payload_included": False,
        }
    return await recommendations.queue_policy_event_feedback(
        store_obj,
        payload,
        source_surface=MANAGED_FEEDBACK_SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=flush_immediately,
    )


def build_anthropic_thinking_compaction_impact_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_error_rate_delta: float = 0.05,
    max_retry_rate_delta: float = 0.10,
    min_net_savings_usd: float = 0.0,
    max_non_positive_savings_rate: float = 0.0,
    suppress_error_rate: float = 0.20,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    thresholds = {
        "min_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "min_net_savings_usd": round(float(min_net_savings_usd), 8),
        "max_non_positive_savings_rate": round(float(max_non_positive_savings_rate), 6),
        "suppress_error_rate": round(float(suppress_error_rate), 6),
    }
    groups: dict[str, dict[str, Any]] = {}
    cohort_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    observed = 0
    skipped_blockers: Counter[str] = Counter()
    compaction_events: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    sampled_rows = _rows(store_obj, limit=lookback_limit, since=since)
    rows_by_session = _session_rows(sampled_rows)
    for row in sampled_rows:
        meta = _row_meta(row)
        if not meta:
            continue
        observed += 1
        routing = _json_obj(row.get("routing_json"))
        parts = _parts(row, routing, meta)
        key = _group_key(parts)
        group = groups.setdefault(key, _new_group(parts))
        cohort = _cohort(meta)
        _add_row(group, row, meta, cohort)
        compaction_events.append((group, row, cohort))
        cohort_counts[cohort] += 1
        status_counts[_status_bucket(row.get("status_code"))] += 1
        reason = _reason(meta.get("reason"), "unknown")
        reason_counts[reason] += 1
        if cohort in {"skipped", "safety_stop"}:
            skipped_blockers[reason] += 1

    for group, row, cohort in compaction_events:
        continuation = group.get("continuation") if isinstance(group.get("continuation"), dict) else {}
        raw = continuation.setdefault(cohort, _empty_continuation_cohort())
        _add_continuation_row(raw, _next_session_row(row, rows_by_session))

    candidates = [_finalize(group, thresholds) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            {"suppress": 0, "hold": 1, "keep-holdout": 2, "widen": 3}.get(str(item.get("verdict")), 4),
            -_as_float(((item.get("cohorts") or {}).get("applied") or {}).get("net_savings_usd")),
            str(item.get("candidate_id") or ""),
        )
    )
    feedback = _summary_feedback(candidates)
    applied_count = cohort_counts.get("applied", 0)
    holdout_count = cohort_counts.get("holdout", 0)
    skipped_count = cohort_counts.get("skipped", 0)
    safety_stop_count = cohort_counts.get("safety_stop", 0)
    coverage_total = applied_count + holdout_count + skipped_count + safety_stop_count
    applied_errors = sum(_as_int(((row.get("cohorts") or {}).get("applied") or {}).get("error_count")) for row in candidates)
    holdout_errors = sum(_as_int(((row.get("cohorts") or {}).get("holdout") or {}).get("error_count")) for row in candidates)
    applied_retries = sum(_as_int(((row.get("cohorts") or {}).get("applied") or {}).get("retry_rows")) for row in candidates)
    holdout_retries = sum(_as_int(((row.get("cohorts") or {}).get("holdout") or {}).get("retry_rows")) for row in candidates)
    applied_tokens_saved = sum(_as_int(((row.get("cohorts") or {}).get("applied") or {}).get("tokens_saved_est")) for row in candidates)
    planned_saved_tokens = sum(
        _as_int(((row.get("cohorts") or {}).get(name) or {}).get("planned_saved_tokens"))
        for row in candidates
        for name in ("applied", "holdout")
    )
    net_savings = round(sum(_as_float(((row.get("cohorts") or {}).get("applied") or {}).get("net_savings_usd")) for row in candidates), 8)
    observed_gross_savings = round(sum(_as_float(((row.get("cohorts") or {}).get("applied") or {}).get("gross_savings_usd")) for row in candidates), 8)
    projected_holdout_savings = round(sum(_as_float(((row.get("cohorts") or {}).get("holdout") or {}).get("projected_savings_usd")) for row in candidates), 8)
    applied_saved_chars = sum(_as_int(((row.get("cohorts") or {}).get("applied") or {}).get("saved_chars")) for row in candidates)
    planned_saved_chars = sum(
        _as_int(((row.get("cohorts") or {}).get(name) or {}).get("planned_saved_chars"))
        for row in candidates
        for name in ("applied", "holdout")
    )
    projected_saved_usd = round(
        sum(
            _as_float(((row.get("cohorts") or {}).get(name) or {}).get("projected_savings_usd"))
            for row in candidates
            for name in ("applied", "holdout")
        ),
        8,
    )
    applied_before_chars = sum(_as_int(((row.get("cohorts") or {}).get("applied") or {}).get("before_chars")) for row in candidates)
    lifecycle_coverage = {
        "schema": "tokenclaw.anthropic_thinking_compaction_lifecycle_coverage.v1",
        "observed_count": coverage_total,
        "applied_count": applied_count,
        "holdout_count": holdout_count,
        "skipped_count": skipped_count,
        "safety_stop_count": safety_stop_count,
        "applied_rate": round(applied_count / coverage_total, 6) if coverage_total else 0.0,
        "holdout_rate": round(holdout_count / coverage_total, 6) if coverage_total else 0.0,
        "safety_stop_rate": round(safety_stop_count / coverage_total, 6) if coverage_total else 0.0,
        "applied_error_count": applied_errors,
        "holdout_error_count": holdout_errors,
        "applied_error_rate": round(applied_errors / applied_count, 6) if applied_count else 0.0,
        "holdout_error_rate": round(holdout_errors / holdout_count, 6) if holdout_count else 0.0,
        "applied_retry_count": applied_retries,
        "holdout_retry_count": holdout_retries,
        "applied_retry_rate": round(applied_retries / applied_count, 6) if applied_count else 0.0,
        "holdout_retry_rate": round(holdout_retries / holdout_count, 6) if holdout_count else 0.0,
        "observed_saved_chars": applied_saved_chars,
        "tokens_saved_est": applied_tokens_saved,
        "observed_saved_tokens": applied_tokens_saved,
        "observed_saved_usd": observed_gross_savings,
        "planned_saved_tokens": planned_saved_tokens,
        "projected_saved_chars": planned_saved_chars,
        "projected_saved_tokens": planned_saved_tokens,
        "projected_saved_usd": projected_saved_usd,
        "net_savings_usd": net_savings,
        "projected_holdout_savings_usd": projected_holdout_savings,
        "avg_crunch_ratio": round(applied_saved_chars / applied_before_chars, 6) if applied_before_chars and applied_saved_chars else 0.0,
        "applied_minus_holdout_error_rate": round((applied_errors / applied_count if applied_count else 0.0) - (holdout_errors / holdout_count if holdout_count else 0.0), 6),
        "applied_minus_holdout_retry_rate": round((applied_retries / applied_count if applied_count else 0.0) - (holdout_retries / holdout_count if holdout_count else 0.0), 6),
        "metadata_only": True,
        "raw_payload_included": False,
    }
    for candidate in candidates:
        action = (candidate.get("budget_governor_feedback") or {}).get("recommended_budget_action")
        candidate["canary_impact_decision"] = _impact_decision_from_action(action)
        candidate["observed_saved_chars"] = _as_int(((candidate.get("cohorts") or {}).get("applied") or {}).get("saved_chars"))
        candidate["observed_saved_tokens"] = _as_int(((candidate.get("cohorts") or {}).get("applied") or {}).get("tokens_saved_est"))
        candidate["observed_saved_usd"] = _as_float(((candidate.get("cohorts") or {}).get("applied") or {}).get("gross_savings_usd"))
        candidate["projected_saved_chars"] = sum(
            _as_int(((candidate.get("cohorts") or {}).get(name) or {}).get("planned_saved_chars"))
            for name in ("applied", "holdout")
        )
        candidate["projected_saved_tokens"] = sum(
            _as_int(((candidate.get("cohorts") or {}).get(name) or {}).get("planned_saved_tokens"))
            for name in ("applied", "holdout")
        )
        candidate["projected_saved_usd"] = round(
            sum(
                _as_float(((candidate.get("cohorts") or {}).get(name) or {}).get("projected_savings_usd"))
                for name in ("applied", "holdout")
            ),
            8,
        )
        candidate["avg_crunch_ratio"] = ((candidate.get("cohorts") or {}).get("applied") or {}).get("avg_crunch_ratio", 0.0)
    canary_impact_decision = _impact_decision(candidates)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "since": since,
        "thresholds": thresholds,
        "status": "matched" if observed else "no-anthropic-thinking-compaction-metadata",
        "summary": {
            "sampled_call_count": len(sampled_rows),
            "observed_thinking_compaction_metadata_row_count": observed,
            "candidate_group_count": len(candidates),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "safety_stop_count": safety_stop_count,
            "observed_saved_chars": applied_saved_chars,
            "observed_saved_tokens": applied_tokens_saved,
            "observed_saved_usd": observed_gross_savings,
            "avg_crunch_ratio": lifecycle_coverage["avg_crunch_ratio"],
            "tokens_saved_est": applied_tokens_saved,
            "planned_saved_tokens": planned_saved_tokens,
            "projected_saved_chars": planned_saved_chars,
            "projected_saved_tokens": planned_saved_tokens,
            "projected_saved_usd": projected_saved_usd,
            "net_savings_usd": net_savings,
            "projected_holdout_savings_usd": projected_holdout_savings,
            "applied_minus_holdout_error_rate": lifecycle_coverage["applied_minus_holdout_error_rate"],
            "applied_minus_holdout_retry_rate": lifecycle_coverage["applied_minus_holdout_retry_rate"],
            "lifecycle_coverage": lifecycle_coverage,
            "thinking_output_tokens": sum(
                _as_int(((row.get("cohorts") or {}).get(name) or {}).get("thinking_output_tokens"))
                for row in candidates
                for name in ("applied", "holdout", "skipped", "safety_stop")
            ),
            "prompt_cache_read_tokens": sum(
                _as_int(((row.get("cohorts") or {}).get(name) or {}).get("prompt_cache_read_tokens"))
                for row in candidates
                for name in ("applied", "holdout", "skipped", "safety_stop")
            ),
            "affected_session_count": sum(_as_int((row.get("session_budget_impact") or {}).get("affected_session_count")) for row in candidates),
            "budget_governor_action": feedback["recommended_budget_action"],
            "canary_impact_decision": canary_impact_decision["decision"],
            "status_breakdown": _breakdown(status_counts),
            "cohort_breakdown": _breakdown(cohort_counts),
            "reason_code_counts": _breakdown(reason_counts),
            "blocker_reason_breakdown": _breakdown(skipped_blockers),
        },
        "budget_governor_feedback": feedback,
        "canary_impact_decision": canary_impact_decision,
        "candidates": candidates,
        "privacy": _privacy(),
    }
