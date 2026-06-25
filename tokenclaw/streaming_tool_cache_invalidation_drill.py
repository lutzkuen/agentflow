"""Local streaming tool-cache invalidation drill.

This is *evidence collection only*. It reads the path-free dependency/invalidation
metadata that the proxy already records per call in ``cache_json`` and produces a
bounded, metadata-only drill rollup that classifies streaming tool-cache candidates
into invalidation blocker codes.

It exists to remove the "missing-invalidation-evidence" blocker that currently keeps
streaming tool-call cache replay switched off, without ever serving a cached response,
creating a cache entry eligible for replay, or mutating a provider request. It runs
entirely off the hot request path over already-logged call rows.

See tokenclaw#921. Sequenced before any replay-activation issue.
"""

from __future__ import annotations

from collections import Counter
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


DRILL_SCHEMA = "tokenclaw.streaming_tool_cache_invalidation_drill.v1"
DRILL_EVENT_TYPE = "streaming_tool_cache_invalidation_drill"
DRILL_SOURCE_SURFACE = "streaming_tool_cache_invalidation_drill"
DRILL_ENDPOINT = "/v1/policy-events"

# Categories that carry tool blocks in streaming agentic traffic.
TOOL_STREAMING_CATEGORIES = {"tool-result", "tool-heavy"}

# Blocker codes the drill emits. These intentionally line up with the five
# acceptance evidence classes in tokenclaw#921.
BLOCKER_STABLE = "stable-dependency-evidence"
BLOCKER_MISSING = "missing-invalidation-evidence"
BLOCKER_STALE = "stale-dependency-evidence"
BLOCKER_UNSAFE = "unsafe-tool-call-shape"
BLOCKER_NO_TOOL_REPEAT = "streaming-no-tool-repeat"

ALL_BLOCKER_CODES = (
    BLOCKER_STABLE,
    BLOCKER_MISSING,
    BLOCKER_STALE,
    BLOCKER_UNSAFE,
    BLOCKER_NO_TOOL_REPEAT,
)

# Existing reason vocabulary (mirrors openai_cache_replay_blocker_outcomes) used to
# read the per-call evidence the proxy already records.
STALE_DEPENDENCY_REASONS = {
    "dependency-cap-exceeded",
    "dependency-changed",
    "dependency-created",
    "dependency-deleted",
    "file-dependency-changed",
    "file-dependency-invalidated",
    "stale-dependency-blocker",
    "stale-risk-blockers",
}
UNSAFE_TOOL_REASONS = {
    "unsafe-dependency-evidence",
    "unsafe-tool-call-shape",
    "unsafe-tool-calls-without-invalidation",
}
MISSING_INVALIDATION_REASONS = {
    "dependency-audit-missing",
    "dependency-missing",
    "file-dependency-missing",
    "file-watch-disabled",
    "invalidation-evidence-missing",
    "safe-invalidation-required",
    "tool-call-cache-disabled",
}

# Cache decision reasons that indicate tool blocks were present even when the
# request category is not one of the explicit tool categories.
TOOL_CACHE_DECISION_REASONS = {"streaming-tools-disabled", "tools-disabled"}

NEXT_ACTION_BY_BLOCKER = {
    BLOCKER_STABLE: "stage-streaming-tool-cache-replay-canary",
    BLOCKER_STALE: "refresh-streaming-tool-cache-dependency-evidence",
    BLOCKER_UNSAFE: "collect-safe-streaming-tool-invalidation-evidence",
    BLOCKER_MISSING: "collect-safe-streaming-tool-invalidation-evidence",
    BLOCKER_NO_TOOL_REPEAT: "keep-streaming-no-tool-repeat-observing",
}


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


def _breakdown(counts: Counter[str] | dict[str, int], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": int(value)}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        if value
    ]


def _has_tool_blocks(row: dict[str, Any], cache: dict[str, Any]) -> bool:
    category = _safe_label(row.get("category"), "")
    if category in TOOL_STREAMING_CATEGORIES:
        return True
    for flag_key in ("has_tool_blocks", "has_tools", "tool_calls_present"):
        if bool(row.get(flag_key)) or bool(cache.get(flag_key)):
            return True
    reason = _safe_label(cache.get("reason"), "")
    if reason in TOOL_CACHE_DECISION_REASONS or "tool" in reason:
        return True
    return False


def _cache_reasons(cache: dict[str, Any], audit: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for key in ("reason", "invalidation_reason", "verdict"):
        value = cache.get(key)
        if value:
            reasons.add(_safe_label(value))
    for key in ("blockers", "reason_codes", "warning_codes"):
        values = cache.get(key)
        if isinstance(values, list):
            for value in values:
                if value:
                    reasons.add(_safe_label(value))
    invalidation = audit.get("invalidation_reason")
    if invalidation:
        reasons.add(_safe_label(invalidation))
    return reasons


def _ttl_invalidation_bucket(audit: dict[str, Any], cache: dict[str, Any]) -> str:
    if audit:
        if audit.get("file_watch_enabled"):
            if _as_int(audit.get("snapshot_count")) > 0:
                return "file-watch-snapshot"
            return "file-watch-no-snapshot"
        return "file-dependency-audit-only"
    ttl_meta = cache.get("ttl") if isinstance(cache.get("ttl"), dict) else None
    if ttl_meta or cache.get("ttl_seconds") or cache.get("expires_at_bucket"):
        return "ttl-window"
    return "no-invalidation-strategy"


def _replayability_level(cache: dict[str, Any]) -> str:
    replay = cache.get("session_memory_replayability")
    if isinstance(replay, dict) and replay.get("replayability_level"):
        return _safe_label(replay.get("replayability_level"))
    hints = cache.get("session_memory_hints")
    if isinstance(hints, dict) and hints.get("replayability_level"):
        return _safe_label(hints.get("replayability_level"))
    if cache.get("replayability_level"):
        return _safe_label(cache.get("replayability_level"))
    return "unknown"


def _classify_row(row: dict[str, Any], cache: dict[str, Any], audit: dict[str, Any]) -> str:
    """Map a single streaming call's recorded evidence to one blocker code.

    Ordering is deliberate: a stale or unsafe signal always wins over a "looks
    stable" reading, and missing evidence is the residual case for tool traffic
    that has no safe-invalidation proof yet.
    """
    if not _has_tool_blocks(row, cache):
        return BLOCKER_NO_TOOL_REPEAT
    reasons = _cache_reasons(cache, audit)
    if reasons & STALE_DEPENDENCY_REASONS:
        return BLOCKER_STALE
    if reasons & UNSAFE_TOOL_REASONS:
        return BLOCKER_UNSAFE
    if bool(audit.get("safe_invalidation_evidence")):
        return BLOCKER_STABLE
    return BLOCKER_MISSING


def _dependency_stability(blocker: str) -> str:
    return {
        BLOCKER_STABLE: "stable",
        BLOCKER_STALE: "stale",
        BLOCKER_UNSAFE: "unsafe",
        BLOCKER_MISSING: "missing",
        BLOCKER_NO_TOOL_REPEAT: "not-applicable",
    }.get(blocker, "unknown")


def _cohort_key(row: dict[str, Any], blocker: str, replayability: str, ttl_bucket: str) -> tuple[str, ...]:
    requested = _model_family(row.get("requested_model_family") or row.get("requested_model"))
    routed = _model_family(row.get("routed_model_family") or row.get("routed_model") or requested, requested)
    return (
        _safe_label(row.get("provider"), "anthropic"),
        _safe_label(row.get("source_surface"), "anthropic_messages"),
        _safe_label(row.get("endpoint") or row.get("path"), "unknown"),
        _safe_label(row.get("category"), "unknown"),
        requested,
        routed,
        blocker,
        replayability,
        ttl_bucket,
    )


def _cohort_ref(key: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:24]
    return f"streaming-tool-cache-drill:{digest}"


def _empty_cohort(key: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.streaming_tool_cache_invalidation_drill_cohort.v1",
        "cohort_ref": _cohort_ref(key),
        "provider": key[0],
        "source_surface": key[1],
        "endpoint": key[2],
        "category": key[3],
        "requested_model_family": key[4],
        "routed_model_family": key[5],
        "stale_risk_blocker": key[6],
        "dependency_stability": _dependency_stability(key[6]),
        "replayability_level": key[7],
        "ttl_invalidation_bucket": key[8],
        "tool_calls_present": key[6] != BLOCKER_NO_TOOL_REPEAT,
        "stream": True,
        "next_action": NEXT_ACTION_BY_BLOCKER.get(key[6], "keep-streaming-tool-cache-drill-observing"),
        "observed_repeat_count": 0,
        "row_count": 0,
        "safe_invalidation_evidence_count": 0,
        "cache_decision_reason_counts": Counter(),
        "invalidation_reason_counts": Counter(),
        "snapshot_count_bucket_counts": Counter(),
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "input_tokens_est": 0,
        "output_tokens_est": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "observed_cache_hit_count": 0,
        "first_seen_at": None,
        "last_seen_at": None,
    }


def _add_row(cohort: dict[str, Any], row: dict[str, Any], cache: dict[str, Any], audit: dict[str, Any]) -> None:
    cohort["row_count"] += 1
    cohort["observed_repeat_count"] += 1
    cohort["safe_invalidation_evidence_count"] += int(bool(audit.get("safe_invalidation_evidence")))
    cohort["observed_cache_hit_count"] += int(bool(row.get("cache_hit")))

    reason = _safe_label(cache.get("reason"), "not-evaluated")
    cohort["cache_decision_reason_counts"][reason] += 1
    invalidation = audit.get("invalidation_reason")
    cohort["invalidation_reason_counts"][_safe_label(invalidation, "none")] += 1
    cohort["snapshot_count_bucket_counts"][_safe_label(audit.get("snapshot_count_bucket"), "unknown")] += 1

    cohort["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
    cohort["cost_baseline_usd"] += _as_float(row.get("cost_baseline_usd"))
    cohort["input_tokens_est"] += _as_int(row.get("input_tokens_est"))
    cohort["output_tokens_est"] += _as_int(row.get("output_tokens_est"))
    cohort["cache_creation_input_tokens"] += _as_int(row.get("cache_creation_input_tokens"))
    cohort["cache_read_input_tokens"] += _as_int(row.get("cache_read_input_tokens"))

    created_at = str(row.get("created_at") or "")
    if created_at:
        if cohort["first_seen_at"] is None or created_at < cohort["first_seen_at"]:
            cohort["first_seen_at"] = created_at
        if cohort["last_seen_at"] is None or created_at > cohort["last_seen_at"]:
            cohort["last_seen_at"] = created_at


def _finalize_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    result = dict(cohort)
    for field in ("cost_est_usd", "cost_baseline_usd"):
        result[field] = round(float(result[field] or 0.0), 8)
    for field in (
        "cache_decision_reason_counts",
        "invalidation_reason_counts",
        "snapshot_count_bucket_counts",
    ):
        result[field.replace("_counts", "_breakdown")] = _breakdown(result.pop(field))
    result["privacy"] = {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_responses_included": False,
        "provider_bodies_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
        "raw_identifiers_included": False,
    }
    return result


def build_streaming_tool_cache_invalidation_drill(
    rows: list[dict[str, Any]],
    *,
    window_hours: int = 24,
    max_cohorts: int = 50,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, metadata-only drill payload from logged streaming calls.

    The drill only *reads* evidence. It never serves a cached response, creates a
    cache entry, or mutates a provider request.
    """
    generated_at = generated_at or utc_now()
    end_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    start_at = (end_at.astimezone(timezone.utc) - timedelta(hours=max(1, int(window_hours)))).isoformat()

    cohorts: dict[tuple[str, ...], dict[str, Any]] = {}
    blocker_counts: Counter[str] = Counter()
    stability_counts: Counter[str] = Counter()
    rows_considered = 0
    observed_cache_hits = 0

    for row in rows:
        if not bool(row.get("stream")):
            continue
        cache = _json_obj(row.get("cache_json"))
        audit = cache.get("file_dependency_audit")
        audit = dict(audit) if isinstance(audit, dict) else {}
        blocker = _classify_row(row, cache, audit)
        replayability = _replayability_level(cache)
        ttl_bucket = _ttl_invalidation_bucket(audit, cache)
        key = _cohort_key(row, blocker, replayability, ttl_bucket)
        cohort = cohorts.setdefault(key, _empty_cohort(key))
        _add_row(cohort, row, cache, audit)
        blocker_counts[blocker] += 1
        stability_counts[_dependency_stability(blocker)] += 1
        observed_cache_hits += int(bool(row.get("cache_hit")))
        rows_considered += 1

    finalized = [_finalize_cohort(item) for item in cohorts.values()]
    finalized.sort(
        key=lambda item: (
            -float(item.get("cost_est_usd") or 0.0),
            -int(item.get("observed_repeat_count") or 0),
            item["cohort_ref"],
        )
    )
    capped = max(1, min(int(max_cohorts or 1), 100))
    selected = finalized[:capped]

    blockers_present = {row.get("stale_risk_blocker") for row in selected}
    payload = {
        "schema": DRILL_SCHEMA,
        "event_type": DRILL_EVENT_TYPE,
        "generated_at": generated_at,
        "read_only": True,
        "serves_cached_responses": False,
        "creates_cache_entry": False,
        "replay_eligible_entries_created": 0,
        "mutates_provider_requests": False,
        "wrote_local_files": False,
        "wrote_store": False,
        "tool_cache_replay_enabled": False,
        "window": {
            "schema": "tokenclaw.streaming_tool_cache_invalidation_drill_window.v1",
            "hours": max(1, int(window_hours)),
            "start_at": start_at,
            "end_at": generated_at,
        },
        "selection": {
            "stream": True,
            "bounded": True,
            "max_cohorts": capped,
            "tool_categories": sorted(TOOL_STREAMING_CATEGORIES),
        },
        "rows_considered": rows_considered,
        "cohort_count": len(selected),
        "observed_cache_hit_count": observed_cache_hits,
        "blocker_breakdown": _breakdown(blocker_counts, limit=len(ALL_BLOCKER_CODES) + 2),
        "dependency_stability_breakdown": _breakdown(stability_counts),
        "drills": selected,
        "acceptance": {
            "classifies_stable_dependency_evidence": BLOCKER_STABLE in blockers_present,
            "classifies_missing_invalidation_evidence": BLOCKER_MISSING in blockers_present,
            "classifies_stale_dependency_evidence": BLOCKER_STALE in blockers_present,
            "classifies_unsafe_tool_call_shape": BLOCKER_UNSAFE in blockers_present,
            "classifies_streaming_no_tool_repeat": BLOCKER_NO_TOOL_REPEAT in blockers_present,
            "no_cache_hit_created": True,
            "no_replay_eligible_entry_created": True,
            "metadata_only": True,
            "evidence_collection_only": True,
        },
        "privacy_summary": {
            "schema": "tokenclaw.streaming_tool_cache_invalidation_drill_privacy.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "raw_payload_included": False,
            "raw_prompt_included": False,
            "raw_response_included": False,
            "provider_body_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "raw_identifiers_included": False,
            "secrets_included": False,
            "api_key_value_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
    assert_managed_egress_safe(payload)
    return payload


def build_streaming_tool_cache_invalidation_drill_from_store(
    store_obj: Any,
    *,
    window_hours: int = 24,
    max_rows: int = 10000,
    max_cohorts: int = 50,
) -> dict[str, Any]:
    """Read-only local-stats helper: build the drill report directly from the store."""
    if not hasattr(store_obj, "streaming_tool_cache_invalidation_drill_rows"):
        return {
            "schema": DRILL_SCHEMA,
            "status": "error",
            "reason": "store-missing-streaming-tool-cache-drill-query",
            "rows_considered": 0,
            "cohort_count": 0,
            "drills": [],
        }
    rows = store_obj.streaming_tool_cache_invalidation_drill_rows(
        window_hours=max(1, int(window_hours)),
        limit=max(1, int(max_rows)),
    )
    return build_streaming_tool_cache_invalidation_drill(
        rows,
        window_hours=window_hours,
        max_cohorts=max_cohorts,
    )


async def record_streaming_tool_cache_invalidation_drill_feedback(
    store_obj: Any,
    *,
    window_hours: int = 24,
    max_rows: int = 10000,
    max_cohorts: int = 50,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    if not hasattr(store_obj, "streaming_tool_cache_invalidation_drill_rows"):
        return {
            "enabled": False,
            "endpoint": DRILL_ENDPOINT,
            "status": "error",
            "reason": "store-missing-streaming-tool-cache-drill-query",
            "payload_included": False,
        }
    try:
        rows = store_obj.streaming_tool_cache_invalidation_drill_rows(
            window_hours=max(1, int(window_hours)),
            limit=max(1, int(max_rows)),
        )
        payload = build_streaming_tool_cache_invalidation_drill(
            rows,
            window_hours=window_hours,
            max_cohorts=max_cohorts,
        )
    except ManagedEgressBlocked as exc:
        return {
            "enabled": False,
            "payload_included": False,
            **managed_egress_blocked_meta(
                endpoint=DRILL_ENDPOINT,
                violations=exc.violations,
            ),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "endpoint": DRILL_ENDPOINT,
            "status": "error",
            "reason": "drill-build-failed",
            "error": repr(exc),
            "payload_included": False,
        }

    try:
        from tokenclaw.recommendations import queue_policy_event_feedback

        meta = await queue_policy_event_feedback(
            store_obj,
            payload,
            source_surface=DRILL_SOURCE_SURFACE,
            queue_when_disabled=True,
            flush_immediately=flush_immediately,
        )
    except Exception as exc:
        return {
            "enabled": False,
            "endpoint": DRILL_ENDPOINT,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "payload_included": False,
        }
    meta.update({
        "source_surface": DRILL_SOURCE_SURFACE,
        "payload_included": False,
        "cohort_count": payload["cohort_count"],
        "rows_considered": payload["rows_considered"],
    })
    return meta


def streaming_tool_cache_invalidation_drill_config() -> dict[str, int]:
    return {
        "interval_seconds": int(
            os.getenv("TOKENCLAW_STREAMING_TOOL_CACHE_DRILL_INTERVAL_SECONDS", "0") or "0"
        ),
        "window_hours": int(
            os.getenv("TOKENCLAW_STREAMING_TOOL_CACHE_DRILL_WINDOW_HOURS", "24") or "24"
        ),
        "max_rows": int(
            os.getenv("TOKENCLAW_STREAMING_TOOL_CACHE_DRILL_MAX_ROWS", "10000") or "10000"
        ),
        "max_cohorts": int(
            os.getenv("TOKENCLAW_STREAMING_TOOL_CACHE_DRILL_MAX_COHORTS", "50") or "50"
        ),
    }


__all__ = [
    "DRILL_SCHEMA",
    "DRILL_EVENT_TYPE",
    "DRILL_SOURCE_SURFACE",
    "ALL_BLOCKER_CODES",
    "build_streaming_tool_cache_invalidation_drill",
    "build_streaming_tool_cache_invalidation_drill_from_store",
    "record_streaming_tool_cache_invalidation_drill_feedback",
    "streaming_tool_cache_invalidation_drill_config",
]
