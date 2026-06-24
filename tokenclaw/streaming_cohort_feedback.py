from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
from typing import Any

from tokenclaw.managed_egress import (
    ManagedEgressBlocked,
    assert_managed_egress_safe,
    managed_egress_blocked_meta,
)
from tokenclaw.store import utc_now


STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA = "tokenclaw.streaming_agentic_cohort_rollup.v1"
STREAMING_AGENTIC_COHORT_EVENT_TYPE = "streaming_agentic_cohort_rollup"
STREAMING_AGENTIC_COHORT_SOURCE_SURFACE = "streaming_agentic_cohort_rollup"
STREAMING_AGENTIC_COHORT_ENDPOINT = "/v1/policy-events"

AGENTIC_STREAMING_CATEGORIES = {"tool-result", "tool-heavy"}
LOCAL_ACTION_COUNT_KEYS = ("applied", "holdout", "vetoed", "held", "unsupported", "fallback", "noop")


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return 0.0


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    text = re.sub(r"[^a-z0-9_.:/-]+", "-", text)[:120].strip("-")
    raw_markers = (
        "raw",
        "provider-body",
        "cache-key",
        "file-path",
        "request-id",
        "session-id",
        "thread-id",
        "secret",
        "authorization",
        "api-key",
        "/home/",
        "/tmp/",
        "sk-",
    )
    if not text or any(marker in text for marker in raw_markers):
        return fallback
    return text


def _model_family(value: Any, fallback: str = "unknown") -> str:
    text = _safe_label(value, fallback="")
    if not text:
        return fallback
    if "claude" in text:
        for family in ("haiku", "sonnet", "opus"):
            if family in text:
                return f"claude-{family}"
        return "claude"
    if text.startswith("gpt-5"):
        return "gpt-5"
    if text.startswith("gpt-4"):
        return "gpt-4"
    return text[:80] or fallback


def _increment(counts: dict[str, int], value: Any, fallback: str = "unknown") -> None:
    label = _safe_label(value, fallback)
    counts[label] = counts.get(label, 0) + 1


def _status_bucket(status_code: Any) -> str:
    status = _as_int(status_code)
    if status <= 0:
        return "unknown"
    if 200 <= status < 300:
        return "2xx"
    if 300 <= status < 400:
        return "3xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return str(status)


def _breakdown(counts: dict[str, int], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": int(value)}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _cohort_id(key: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:24]
    return f"streaming-cohort:{digest}"


def _cohort_key(row: dict[str, Any], routing: dict[str, Any]) -> tuple[str, ...]:
    requested = _model_family(row.get("requested_model_family") or row.get("requested_model"))
    routed = _model_family(row.get("routed_model_family") or row.get("routed_model") or requested, requested)
    return (
        _safe_label(row.get("provider"), "anthropic"),
        _safe_label(row.get("source_surface"), "anthropic_messages"),
        _safe_label(row.get("endpoint") or row.get("path"), "unknown"),
        _safe_label(row.get("category"), "unknown"),
        requested,
        routed,
        _safe_label(row.get("routing_outcome_label") or routing.get("status"), "unknown"),
    )


def _empty_cohort(key: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.streaming_agentic_cohort.v1",
        "cohort_ref": _cohort_id(key),
        "provider": key[0],
        "source_surface": key[1],
        "endpoint": key[2],
        "category": key[3],
        "requested_model_family": key[4],
        "routed_model_family": key[5],
        "routing_outcome_label": key[6],
        "stream": True,
        "row_count": 0,
        "success_count": 0,
        "error_count": 0,
        "status_counts": {},
        "routing_outcome_counts": {},
        "routing_reason_counts": {},
        "cache_status_counts": {},
        "cache_reason_counts": {},
        "crunch_status_counts": {},
        "old_context_summary_status_counts": {},
        "local_action_counts": {key: 0 for key in LOCAL_ACTION_COUNT_KEYS},
        "cache_hit_count": 0,
        "retry_row_count": 0,
        "retry_attempts": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "actual_input_tokens": 0,
        "input_tokens_est": 0,
        "actual_output_tokens": 0,
        "output_tokens_est": 0,
        "thinking_output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "observed_savings_usd": 0.0,
        "crunch_saved_chars": 0,
        "crunch_tokens_saved_est": 0,
        "first_seen_at": None,
        "last_seen_at": None,
    }


def _local_action_bucket(routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    candidates: list[Any] = [
        routing.get("local_result"),
        routing.get("status"),
        routing.get("reason"),
        routing.get("routing_outcome_label"),
        routing.get("server_traffic_treatment"),
        cache.get("status"),
        crunch.get("status"),
    ]
    managed = routing.get("managed_recommendation")
    if isinstance(managed, dict):
        candidates.extend([
            managed.get("local_result"),
            managed.get("status"),
            managed.get("apply_reason"),
            managed.get("server_traffic_treatment"),
        ])
        if managed.get("applied") is True:
            return "applied"
    for section in (routing.get("routing"), routing.get("crunch"), routing.get("cache")):
        if isinstance(section, dict):
            candidates.extend([section.get("status"), section.get("reason")])
            if section.get("applied") is True:
                return "applied"
    text = " ".join(_safe_label(value, "") for value in candidates).replace("_", "-")
    if "veto" in text:
        return "vetoed"
    if "unsupported" in text:
        return "unsupported"
    if "holdout" in text or "heldout" in text:
        return "holdout"
    if "fallback" in text:
        return "fallback"
    if "held" in text:
        return "held"
    if "applied" in text or routing.get("applied") is True or cache.get("applied") is True or crunch.get("applied") is True:
        return "applied"
    return "noop"


def _add_row(cohort: dict[str, Any], row: dict[str, Any]) -> None:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    status_code = _as_int(row.get("status_code"))
    created_at = str(row.get("created_at") or "")

    cohort["row_count"] += 1
    cohort["success_count"] += int(200 <= status_code < 300)
    cohort["error_count"] += int(status_code >= 400 or bool(row.get("error_present")))
    _increment(cohort["status_counts"], _status_bucket(status_code))
    _increment(cohort["routing_outcome_counts"], row.get("routing_outcome_label") or routing.get("status"))
    _increment(cohort["routing_reason_counts"], routing.get("reason") or routing.get("fallback_reason"))
    _increment(cohort["cache_status_counts"], cache.get("status"))
    _increment(cohort["cache_reason_counts"], cache.get("reason"))
    _increment(cohort["crunch_status_counts"], crunch.get("status") or ("changed" if crunch.get("changed") else "unchanged"))

    old_context = routing.get("old_context_summary_feedback")
    if isinstance(old_context, dict):
        _increment(cohort["old_context_summary_status_counts"], old_context.get("status"))
    else:
        _increment(cohort["old_context_summary_status_counts"], "not-present")

    action_bucket = _local_action_bucket(routing, crunch, cache)
    cohort["local_action_counts"][action_bucket] += 1

    retry_count = _as_int(row.get("retry_count"))
    latency_ms = _as_int(row.get("latency_ms"))
    cost_est = _as_float(row.get("cost_est_usd"))
    baseline = _as_float(row.get("cost_baseline_usd"))
    cohort["cache_hit_count"] += int(bool(row.get("cache_hit")))
    cohort["retry_row_count"] += int(retry_count > 0)
    cohort["retry_attempts"] += retry_count
    if latency_ms > 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
    cohort["actual_input_tokens"] += _as_int(row.get("actual_input_tokens"))
    cohort["input_tokens_est"] += _as_int(row.get("input_tokens_est"))
    cohort["actual_output_tokens"] += _as_int(row.get("actual_output_tokens"))
    cohort["output_tokens_est"] += _as_int(row.get("output_tokens_est"))
    cohort["thinking_output_tokens"] += _as_int(row.get("thinking_output_tokens"))
    cohort["cache_creation_input_tokens"] += _as_int(row.get("cache_creation_input_tokens"))
    cohort["cache_read_input_tokens"] += _as_int(row.get("cache_read_input_tokens"))
    cohort["cost_est_usd"] += cost_est
    cohort["cost_baseline_usd"] += baseline
    cohort["observed_savings_usd"] += max(0.0, baseline - cost_est)
    cohort["crunch_saved_chars"] += _as_int(crunch.get("saved_chars"))
    cohort["crunch_tokens_saved_est"] += _as_int(crunch.get("tokens_saved_est"))
    if created_at:
        if cohort["first_seen_at"] is None or created_at < cohort["first_seen_at"]:
            cohort["first_seen_at"] = created_at
        if cohort["last_seen_at"] is None or created_at > cohort["last_seen_at"]:
            cohort["last_seen_at"] = created_at


def _finalize_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    result = dict(cohort)
    sample_count = int(result.pop("latency_sample_count") or 0)
    latency_total = int(result.pop("latency_ms_total") or 0)
    result["avg_latency_ms"] = round(latency_total / sample_count, 2) if sample_count else None
    for field in ("cost_est_usd", "cost_baseline_usd", "observed_savings_usd"):
        result[field] = round(float(result[field] or 0.0), 8)
    for field in (
        "status_counts",
        "routing_outcome_counts",
        "routing_reason_counts",
        "cache_status_counts",
        "cache_reason_counts",
        "crunch_status_counts",
        "old_context_summary_status_counts",
        "local_action_counts",
    ):
        result[field.replace("_counts", "_breakdown")] = _breakdown(result.pop(field))
    result["privacy"] = {
        "metadata_only": True,
        "raw_prompts_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "raw_identifiers_included": False,
    }
    return result


def build_streaming_agentic_cohort_rollup_feedback(
    rows: list[dict[str, Any]],
    *,
    window_hours: int = 24,
    max_cohorts: int = 50,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or utc_now()
    end_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    start_at = (end_at.astimezone(timezone.utc) - timedelta(hours=max(1, int(window_hours)))).isoformat()

    cohorts: dict[tuple[str, ...], dict[str, Any]] = {}
    rows_considered = 0
    for row in rows:
        if not bool(row.get("stream")):
            continue
        category = _safe_label(row.get("category"), "")
        if category not in AGENTIC_STREAMING_CATEGORIES:
            continue
        routing = _json_obj(row.get("routing_json"))
        key = _cohort_key(row, routing)
        cohort = cohorts.setdefault(key, _empty_cohort(key))
        _add_row(cohort, row)
        rows_considered += 1

    finalized = [_finalize_cohort(item) for item in cohorts.values()]
    finalized.sort(key=lambda item: (-float(item.get("cost_est_usd") or 0.0), -int(item.get("row_count") or 0), item["cohort_ref"]))
    capped = max(1, min(int(max_cohorts or 1), 100))
    payload = {
        "schema": STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA,
        "event_type": STREAMING_AGENTIC_COHORT_EVENT_TYPE,
        "generated_at": generated_at,
        "window": {
            "schema": "tokenclaw.streaming_agentic_cohort_window.v1",
            "hours": max(1, int(window_hours)),
            "start_at": start_at,
            "end_at": generated_at,
        },
        "selection": {
            "stream": True,
            "categories": sorted(AGENTIC_STREAMING_CATEGORIES),
            "bounded": True,
            "max_cohorts": capped,
        },
        "rows_considered": rows_considered,
        "cohort_count": len(finalized[:capped]),
        "rollups": finalized[:capped],
        "privacy_summary": {
            "schema": "tokenclaw.streaming_agentic_cohort_rollup_privacy.v1",
            "metadata_only": True,
            "raw_payload_included": False,
            "raw_prompt_included": False,
            "raw_response_included": False,
            "provider_body_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "raw_identifiers_included": False,
            "secrets_included": False,
            "api_key_value_included": False,
        },
    }
    assert_managed_egress_safe(payload)
    return payload


async def record_streaming_agentic_cohort_rollup_feedback(
    store_obj: Any,
    *,
    window_hours: int = 24,
    max_rows: int = 10000,
    max_cohorts: int = 50,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    if not hasattr(store_obj, "streaming_agentic_cohort_feedback_rows"):
        return {
            "enabled": False,
            "endpoint": STREAMING_AGENTIC_COHORT_ENDPOINT,
            "status": "error",
            "reason": "store-missing-streaming-cohort-rollup-query",
            "payload_included": False,
        }
    try:
        rows = store_obj.streaming_agentic_cohort_feedback_rows(
            window_hours=max(1, int(window_hours)),
            limit=max(1, int(max_rows)),
        )
        payload = build_streaming_agentic_cohort_rollup_feedback(
            rows,
            window_hours=window_hours,
            max_cohorts=max_cohorts,
        )
    except ManagedEgressBlocked as exc:
        return {
            "enabled": False,
            "payload_included": False,
            **managed_egress_blocked_meta(
                endpoint=STREAMING_AGENTIC_COHORT_ENDPOINT,
                violations=exc.violations,
            ),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "endpoint": STREAMING_AGENTIC_COHORT_ENDPOINT,
            "status": "error",
            "reason": "rollup-build-failed",
            "error": repr(exc),
            "payload_included": False,
        }

    try:
        from tokenclaw.recommendations import queue_policy_event_feedback

        meta = await queue_policy_event_feedback(
            store_obj,
            payload,
            source_surface=STREAMING_AGENTIC_COHORT_SOURCE_SURFACE,
            queue_when_disabled=True,
            flush_immediately=flush_immediately,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "endpoint": STREAMING_AGENTIC_COHORT_ENDPOINT,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "payload_included": False,
        }
    meta.update({
        "source_surface": STREAMING_AGENTIC_COHORT_SOURCE_SURFACE,
        "payload_included": False,
        "cohort_count": payload["cohort_count"],
        "rows_considered": payload["rows_considered"],
    })
    return meta


def streaming_agentic_cohort_feedback_config() -> dict[str, int]:
    return {
        "interval_seconds": int(os.getenv("TOKENCLAW_STREAMING_COHORT_FEEDBACK_INTERVAL_SECONDS", "3600") or "0"),
        "window_hours": int(os.getenv("TOKENCLAW_STREAMING_COHORT_FEEDBACK_WINDOW_HOURS", "24") or "24"),
        "max_rows": int(os.getenv("TOKENCLAW_STREAMING_COHORT_FEEDBACK_MAX_ROWS", "10000") or "10000"),
        "max_cohorts": int(os.getenv("TOKENCLAW_STREAMING_COHORT_FEEDBACK_MAX_COHORTS", "50") or "50"),
    }


__all__ = [
    "STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA",
    "STREAMING_AGENTIC_COHORT_SOURCE_SURFACE",
    "build_streaming_agentic_cohort_rollup_feedback",
    "record_streaming_agentic_cohort_rollup_feedback",
    "streaming_agentic_cohort_feedback_config",
]
