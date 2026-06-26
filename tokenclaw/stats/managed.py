from __future__ import annotations

import json
import hashlib
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tokenclaw.managed_mode import managed_mode_public_meta, managed_product_mode
from tokenclaw.public_metadata import public_label
from tokenclaw.recommendations import (
    OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE,
    PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE,
    PHASE_ROUTING_OUTCOME_SOURCE_SURFACE,
    pattern_decision_summaries,
)
from tokenclaw.routing_experiments import build_routing_experiment_report
from tokenclaw.store import utc_now
from tokenclaw.stats import (
    _as_float,
    _as_int,
    _avg_or_none,
    _breakdown_from_counts,
    _count_breakdown,
    _decision_breakdown,
    _endpoint_label,
    _increment_count,
    _json_obj,
    _json_obj_has_value,
    _latest_policy_event,
    _local_path_class,
    _metadata_only_privacy,
    _money,
    _parse_utc_datetime,
    _percentile_int,
    _read_workbench_draft_manifest,
    _safe_count_breakdown,
    _sanitize_error_sample,
    _seconds_since_iso,
    _source_surface,
    _status_code_bucket,
    _url_host_state,
    estimate_tokens_from_text_chars,
    stats_activity,
    stats_policies,
)

MANAGED_PATTERN_ADOPTION_STAGES = (
    "received",
    "reviewed",
    "dry_run",
    "applied",
    "canary_applied",
    "canary_holdout",
    "bypassed",
    "errored",
    "rolled_back",
    "rejected",
)
OPENAI_GOVERNOR_SCHEMA = "tokenclaw.openai_optimization_governor.v1"
OPENAI_GOVERNOR_FAMILIES = ("routing", "old_context_summary", "cache_replay")


async def _stats_policies_facade(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from tokenclaw import stats as stats_facade

    return await stats_facade.stats_policies(*args, **kwargs)


async def _stats_activity_facade(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from tokenclaw import stats as stats_facade

    return await stats_facade.stats_activity(*args, **kwargs)


def _managed_pattern_is_bypass(summary: dict[str, Any]) -> bool:
    outcome = str(summary.get("outcome") or "")
    status = str(summary.get("status") or "")
    reason = str(summary.get("reason") or "")
    return outcome == "bypassed" or status in {"bypass", "bypassed"} or "bypass" in reason or "disabled" in reason

def _safe_public_bool_env_configured(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(str(value).strip())

def _managed_feed_state() -> dict[str, Any]:
    from tokenclaw.recommendations import (
        managed_auth_configured,
        policy_decisions_enabled,
        recommendation_server_configured,
        recommendation_server_url,
    )

    mode = managed_mode_public_meta()
    server_configured = recommendation_server_configured()
    raw_url_state = _url_host_state(recommendation_server_url() if server_configured else None)
    url_host_state = {
        "configured": bool(raw_url_state.get("configured")),
        "scheme": raw_url_state.get("scheme"),
        "host_loopback": raw_url_state.get("host_loopback"),
        "host_included": False,
        "redacted_url_included": False,
        "raw_url_included": False,
    }
    policy_decision_configured = (
        _safe_public_bool_env_configured("TOKENCLAW_POLICY_DECISIONS_ENABLED")
        or _safe_public_bool_env_configured("TOKENCLAW_POLICY_DECISION_ENABLED")
        or bool(mode.get("configured"))
    )
    return {
        "schema": "tokenclaw.managed_feed_state.v1",
        "mode": mode.get("mode"),
        "configured": bool(mode.get("configured")),
        "managed_enabled": bool(mode.get("managed_enabled")),
        "local_rules_only": bool(mode.get("local_rules_only")),
        "server_calls_enabled": bool(mode.get("server_calls_enabled")),
        "local_application_enabled": bool(mode.get("local_application_enabled")),
        "families": dict(mode.get("families") or {}),
        "reason": mode.get("reason"),
        "server": {
            "configured": bool(server_configured),
            "auth_configured": bool(managed_auth_configured()),
            "url_host_state": url_host_state,
        },
        "policy_decisions": {
            "configured": bool(policy_decision_configured),
            "enabled": bool(policy_decisions_enabled()),
        },
        "privacy": {
            "metadata_only": True,
            "server_url_included": False,
            "api_key_value_included": False,
            "raw_values_included": False,
        },
    }

_MANAGED_SUCCESS_STATUSES = {"received", "applied", "dry-run", "accepted", "success"}

def _managed_decision_source(routing: dict[str, Any], managed: dict[str, Any]) -> str:
    managed_source = str(managed.get("policy_source") or "").strip()
    status = str(managed.get("status") or "").strip()
    if managed_source == "managed-enforced" or bool(managed.get("managed_enforced")):
        return "managed-enforced"
    if managed_source == "managed-recommended" or bool(managed.get("applied")) or status in _MANAGED_SUCCESS_STATUSES:
        return "managed-recommended"
    routing_source = str(routing.get("policy_source") or "").strip()
    if routing_source == "local-manual":
        return "local-manual"
    return "off/pass-through"

def _managed_attempt_status(managed: dict[str, Any]) -> str:
    status = str(managed.get("status") or "").strip()
    reason = str(managed.get("reason") or "").strip()
    if status in _MANAGED_SUCCESS_STATUSES or bool(managed.get("applied")):
        return "succeeded"
    if status.startswith("skipped") or status in {"out-of-scope", "disabled", ""} or reason in {
        "disabled",
        "policy-decision-metadata-missing",
        "source-surface-not-canonical",
        "server-url-not-configured",
    }:
        return "skipped"
    if status in {"error", "invalid"} or "error" in status:
        return "failed"
    if bool(managed.get("enabled")) or bool(managed.get("policy_decision_enabled")):
        return "attempted"
    return "skipped"

def _managed_feed_decision_summary(
    conn: Any,
    *,
    since: str | None,
    day_field: bool = False,
) -> dict[str, Any]:
    where = "where created_at >= ?" if since else ""
    params: tuple[Any, ...] = (since,) if since else ()
    select_day = "date(created_at) as day," if day_field else ""
    rows = conn.execute(
        f"""
        select {select_day}
               requested_model,
               routed_model,
               routing_json,
               managed_routing_json
        from calls
        {where}
        """,
        params,
    ).fetchall()

    backing_counts = {
        "local-manual": 0,
        "managed-recommended": 0,
        "managed-enforced": 0,
        "off/pass-through": 0,
    }
    attempts = {
        "attempted": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
    }
    by_day: dict[str, dict[str, Any]] = {}

    def empty_day() -> dict[str, Any]:
        return {
            "backing_counts": dict(backing_counts.fromkeys(backing_counts, 0)),
            "policy_decision_calls": dict(attempts.fromkeys(attempts, 0)),
        }

    for row in rows:
        raw_routing = row["routing_json"] if hasattr(row, "keys") else row[2 if day_field else 2]
        raw_managed = row["managed_routing_json"] if hasattr(row, "keys") else row[3 if day_field else 3]
        routing = _json_obj(raw_routing)
        managed = _json_obj(raw_managed) or _json_obj(routing.get("managed_recommendation"))
        source = _managed_decision_source(routing, managed)
        backing_counts[source] = backing_counts.get(source, 0) + 1
        attempt = _managed_attempt_status(managed)
        attempts[attempt] = attempts.get(attempt, 0) + 1
        if attempt == "succeeded":
            attempts["attempted"] += 1
        elif attempt == "failed":
            attempts["attempted"] += 1
        if day_field:
            day = str(row["day"] if hasattr(row, "keys") else row[0] or "")
            bucket = by_day.setdefault(day, empty_day())
            bucket["backing_counts"][source] = bucket["backing_counts"].get(source, 0) + 1
            bucket["policy_decision_calls"][attempt] = bucket["policy_decision_calls"].get(attempt, 0) + 1
            if attempt in {"succeeded", "failed", "attempted"}:
                if attempt != "attempted":
                    bucket["policy_decision_calls"]["attempted"] = bucket["policy_decision_calls"].get("attempted", 0) + 1

    return {
        "schema": "tokenclaw.managed_feed_decision_summary.v1",
        "window_start": since,
        "total_calls": len(rows),
        "policy_decision_calls": attempts,
        "backing_counts": backing_counts,
        "by_day": by_day,
        "privacy": _metadata_only_privacy(),
    }

def _managed_feedback_error_class(row: dict[str, Any]) -> str | None:
    error = _sanitize_error_sample(row.get("last_error"), limit=240)
    if not error:
        status_code = _as_int(row.get("last_status_code"))
        return f"http_{status_code}" if status_code else None
    head = error.split(":", 1)[0].strip()
    return head[:80] if head else "error"

def _managed_feedback_safe_error_class(row: dict[str, Any]) -> str | None:
    status_code = _as_int(row.get("last_status_code"))
    if status_code:
        return f"http_{status_code}"
    if row.get("last_error"):
        return "error-present"
    return None

def _public_managed_feedback_row(row: dict[str, Any] | None, *, now: datetime) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": row.get("optimization_unit_id"),
        "status": row.get("status"),
        "attempts": _as_int(row.get("attempts")),
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "last_error_class": _managed_feedback_error_class(row),
        "sent_at": row.get("sent_at"),
        "age_seconds": _seconds_since_iso(row.get("created_at"), now),
        "due_age_seconds": _seconds_since_iso(row.get("next_attempt_at"), now)
        if row.get("status") in {"queued", "retryable-error"}
        else None,
        "payload_included": False,
    }

def _managed_feedback_queue_health(
    store_obj: Any | None,
    *,
    sample_limit: int = 5,
    source_surface: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    try:
        from tokenclaw import recommendations as recommendations_module

        drain_recommendations_enabled = recommendations_module.recommendations_enabled()
        drain_server_configured = recommendations_module.recommendation_server_configured()
        drain_auth_configured = recommendations_module.managed_auth_configured()
        drain_max_attempts = recommendations_module.outcome_feedback_queue_max_attempts()
        drain_retry_delay_seconds = recommendations_module.outcome_feedback_queue_retry_delay_seconds()
        drain_max_retry_delay_seconds = recommendations_module.outcome_feedback_queue_max_retry_delay_seconds()
    except Exception:
        drain_recommendations_enabled = False
        drain_server_configured = False
        drain_auth_configured = False
        drain_max_attempts = None
        drain_retry_delay_seconds = None
        drain_max_retry_delay_seconds = None

    product_mode = managed_product_mode()
    drain_blocked_reason = None
    if not drain_recommendations_enabled:
        if product_mode.local_rules_only or product_mode.mode == "local_only":
            drain_blocked_reason = "local-only-managed-mode"
        else:
            drain_blocked_reason = "recommendations-disabled"
    elif not drain_server_configured:
        drain_blocked_reason = "server-url-not-configured"
    drain_state = {
        "schema": "tokenclaw.managed_feedback_drain_state.v1",
        "enabled": bool(drain_recommendations_enabled and drain_server_configured),
        "blocked_reason": drain_blocked_reason,
        "recommendations_enabled": bool(drain_recommendations_enabled),
        "server_configured": bool(drain_server_configured),
        "auth_configured": bool(drain_auth_configured),
        "max_attempts": drain_max_attempts,
        "retry_delay_seconds": drain_retry_delay_seconds,
        "max_retry_delay_seconds": drain_max_retry_delay_seconds,
    }
    if store_obj is None or not hasattr(store_obj, "managed_outcome_feedback_rows"):
        return {
            "available": False,
            "drain": drain_state,
            "summary": {
                "total": 0,
                "queued": 0,
                "due": 0,
                "retryable_error": 0,
                "dropped_after_limit": 0,
                "sent": 0,
                "oldest_due_age_seconds": None,
            },
            "status_breakdown": [],
            "source_surface_breakdown": [],
            "oldest_due": None,
            "last_successful_flush": None,
            "due_samples": [],
            "privacy": {
                "metadata_only": True,
                "payload_json_included": False,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "provider_bodies_included": False,
            },
        }

    try:
        rows = store_obj.managed_outcome_feedback_rows(source_surface=source_surface, limit=10000)
    except Exception:
        rows = []

    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    due_rows: list[dict[str, Any]] = []
    stale_sending_rows: list[dict[str, Any]] = []
    sent_rows: list[dict[str, Any]] = []
    queued_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    recent_window = now - timedelta(hours=1)
    sent_last_window = 0
    failed_last_window = 0
    stale_sending_cutoff = now - timedelta(minutes=10)
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        source = str(row.get("source_surface") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if status == "queued":
            queued_rows.append(row)
        if status in {"queued", "retryable-error"}:
            pending_rows.append(row)
            due_at = _parse_utc_datetime(row.get("next_attempt_at"))
            if due_at is not None and due_at <= now:
                due_rows.append(row)
        if status == "sending":
            updated_at = _parse_utc_datetime(row.get("updated_at"))
            if updated_at is not None and updated_at <= stale_sending_cutoff:
                stale_sending_rows.append(row)
        if status == "sent":
            sent_rows.append(row)
            sent_at = _parse_utc_datetime(row.get("sent_at") or row.get("updated_at"))
            if sent_at is not None and sent_at >= recent_window:
                sent_last_window += 1
        if status in {"retryable-error", "dropped-after-limit", "error"}:
            failed_at = _parse_utc_datetime(row.get("updated_at"))
            if failed_at is not None and failed_at >= recent_window:
                failed_last_window += 1

    due_rows.sort(key=lambda row: _parse_utc_datetime(row.get("next_attempt_at")) or now)
    stale_sending_rows.sort(key=lambda row: _parse_utc_datetime(row.get("updated_at")) or now)
    oldest_due = due_rows[0] if due_rows else None
    oldest_stale_sending = stale_sending_rows[0] if stale_sending_rows else None
    oldest_pending = min(
        pending_rows,
        key=lambda row: _parse_utc_datetime(row.get("created_at")) or now,
        default={},
    )
    oldest_queued = min(
        queued_rows,
        key=lambda row: _parse_utc_datetime(row.get("created_at")) or now,
        default={},
    )
    sent_rows.sort(
        key=lambda row: _parse_utc_datetime(row.get("sent_at") or row.get("updated_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    last_successful = sent_rows[0] if sent_rows else None
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "due": len(due_rows),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "sent": status_counts.get("sent", 0),
        "dropped_after_limit": status_counts.get("dropped-after-limit", 0),
        "error": status_counts.get("error", 0),
        "stale_sending": len(stale_sending_rows),
        "oldest_due_age_seconds": _seconds_since_iso(oldest_due.get("next_attempt_at"), now) if oldest_due else None,
        "oldest_queued_age_seconds": _seconds_since_iso(oldest_queued.get("created_at"), now),
        "oldest_stale_sending_age_seconds": _seconds_since_iso(
            oldest_stale_sending.get("updated_at"), now
        ) if oldest_stale_sending else None,
        "oldest_pending_age_seconds": _seconds_since_iso(oldest_pending.get("created_at"), now),
        "recent_window_seconds": 3600,
        "sent_last_window": sent_last_window,
        "failed_last_window": failed_last_window,
    }
    return {
        "available": True,
        "drain": drain_state,
        "summary": summary,
        "status_breakdown": _breakdown_from_counts(status_counts),
        "source_surface_breakdown": _breakdown_from_counts(source_counts),
        "oldest_due": _public_managed_feedback_row(oldest_due, now=now),
        "oldest_stale_sending": _public_managed_feedback_row(oldest_stale_sending, now=now),
        "last_successful_flush": _public_managed_feedback_row(last_successful, now=now),
        "due_samples": [
            item
            for item in (
                _public_managed_feedback_row(row, now=now)
                for row in due_rows[: max(0, int(sample_limit or 0))]
            )
            if item is not None
        ],
        "stale_sending_samples": [
            item
            for item in (
                _public_managed_feedback_row(row, now=now)
                for row in stale_sending_rows[: max(0, int(sample_limit or 0))]
            )
            if item is not None
        ],
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "secrets_included": False,
        },
    }

def _normalize_feedback_dimension(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text or default

def _collect_action_family(value: Any, families: set[str]) -> None:
    family = _normalize_feedback_dimension(value, "")
    if family:
        families.add(family)

def _managed_feedback_inferred_action_family(payload: dict[str, Any], source_surface: Any) -> str | None:
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            source_surface,
            payload.get("schema"),
            payload.get("event_type"),
            (payload.get("metadata") or {}).get("schema") if isinstance(payload.get("metadata"), dict) else None,
        )
    )
    if "routing" in haystack:
        return "routing"
    if "cache" in haystack:
        return "cache"
    if any(token in haystack for token in ("crunch", "compaction", "summary", "dedup", "scaffold", "thinking")):
        return "crunch"
    return None

def _managed_feedback_payload_action_families(payload: dict[str, Any], *, source_surface: Any = None) -> list[str]:
    families: set[str] = set()
    for key in ("action_family", "local_action_family", "policy_section", "optimization_family"):
        _collect_action_family(payload.get(key), families)
    for key in (
        "applied_families",
        "vetoed_families",
        "held_families",
        "heldout_families",
        "unsupported_families",
        "supported_local_action_families",
        "enabled_local_action_families",
    ):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                _collect_action_family(value, families)
    actions = payload.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if isinstance(action, dict):
                _collect_action_family(action.get("family") or action.get("action_family") or action.get("type"), families)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in ("action_family", "local_action_family", "policy_section", "optimization_family"):
        _collect_action_family(metadata.get(key), families)
    for key in ("action_snapshots", "actions", "pattern_policy_evidence"):
        items = metadata.get(key) if key != "pattern_policy_evidence" else payload.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    _collect_action_family(
                        item.get("action_family")
                        or item.get("local_action_family")
                        or item.get("policy_section")
                        or item.get("optimization_family")
                        or item.get("family"),
                        families,
                    )
    if not families:
        inferred = _managed_feedback_inferred_action_family(payload, source_surface)
        if inferred:
            families.add(inferred)
    return sorted(families) or ["unknown"]

def _managed_feedback_queue_expectation(action_family: str, status: str, drain_state: dict[str, Any]) -> tuple[str, str]:
    product_mode = managed_product_mode()
    if status in {"dropped-after-limit", "error"}:
        return "actionable", "feedback delivery has failed permanently or with an unclassified error"
    if status == "retryable-error":
        return "actionable", "retryable feedback delivery errors are waiting for another drain attempt"
    if status == "sending":
        return "actionable", "feedback row is in-flight; stale sending rows require recovery if age keeps growing"
    if not product_mode.server_calls_enabled:
        return "expected", product_mode.reason or "managed server calls are disabled"
    if action_family in product_mode.family_enabled and not product_mode.family_enabled.get(action_family):
        return "expected", f"managed {action_family} family is disabled locally"
    if drain_state.get("blocked_reason"):
        return "expected", str(drain_state.get("blocked_reason"))
    if not drain_state.get("enabled"):
        return "expected", "managed feedback drain is not enabled"
    if status == "queued":
        return "watch", "queued feedback should drain when due; stale age growth is actionable"
    return "ok", "feedback row does not currently require operator action"

def stats_managed_feedback_queue_freshness(store_obj: Any | None, *, limit: int = 10000) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    base_health = _managed_feedback_queue_health(store_obj, sample_limit=5)
    drain_state = base_health.get("drain") if isinstance(base_health.get("drain"), dict) else {}
    product_mode = managed_product_mode()
    rows: list[dict[str, Any]] = []
    if store_obj is not None:
        try:
            if hasattr(store_obj, "managed_outcome_feedback_freshness_rows"):
                rows = store_obj.managed_outcome_feedback_freshness_rows(limit=limit)
            elif hasattr(store_obj, "managed_outcome_feedback_rows"):
                rows = store_obj.managed_outcome_feedback_rows(limit=limit)
        except Exception:
            rows = []

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    expectation_counts: dict[str, int] = {}
    scanned_rows = 0
    emitted_memberships = 0
    for row in rows[: max(1, min(int(limit or 1), 10000))]:
        scanned_rows += 1
        status = _normalize_feedback_dimension(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        payload = _json_obj(row.get("payload_json"))
        source_surface = _normalize_feedback_dimension(row.get("source_surface"))
        families = _managed_feedback_payload_action_families(payload, source_surface=source_surface)
        endpoint = str(row.get("endpoint") or "unknown")
        created_at = _parse_utc_datetime(row.get("created_at"))
        updated_at = _parse_utc_datetime(row.get("updated_at"))
        sent_at = _parse_utc_datetime(row.get("sent_at"))
        next_attempt_at = _parse_utc_datetime(row.get("next_attempt_at"))
        for family in families:
            emitted_memberships += 1
            family_counts[family] = family_counts.get(family, 0) + 1
            expectation_state, expectation_reason = _managed_feedback_queue_expectation(family, status, drain_state)
            expectation_counts[expectation_state] = expectation_counts.get(expectation_state, 0) + 1
            key = (family, source_surface, endpoint, status)
            bucket = grouped.setdefault(
                key,
                {
                    "action_family": family,
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "status": status,
                    "row_count": 0,
                    "attempt_count": 0,
                    "max_attempts": 0,
                    "last_status_code": None,
                    "last_error_class": None,
                    "oldest_queued_at": None,
                    "newest_queued_at": None,
                    "oldest_queued_age_seconds": None,
                    "oldest_pending_age_seconds": None,
                    "newest_updated_at": None,
                    "latest_sent_at": None,
                    "next_attempt_at": None,
                    "due_count": 0,
                    "expectation_state": expectation_state,
                    "expectation_reason": expectation_reason,
                    "payload_json_included": False,
                },
            )
            bucket["row_count"] += 1
            attempts = _as_int(row.get("attempts"))
            bucket["attempt_count"] += attempts
            bucket["max_attempts"] = max(_as_int(bucket.get("max_attempts")), attempts)
            if row.get("last_status_code") is not None:
                bucket["last_status_code"] = row.get("last_status_code")
            error_class = _managed_feedback_safe_error_class(row)
            if error_class:
                bucket["last_error_class"] = error_class
            if status in {"queued", "retryable-error"}:
                created_text = row.get("created_at")
                if bucket["oldest_queued_at"] is None or str(created_text or "") < str(bucket["oldest_queued_at"]):
                    bucket["oldest_queued_at"] = created_text
                    bucket["oldest_queued_age_seconds"] = _seconds_since_iso(created_text, now)
                if bucket["newest_queued_at"] is None or str(created_text or "") > str(bucket["newest_queued_at"]):
                    bucket["newest_queued_at"] = created_text
                if next_attempt_at is not None and next_attempt_at <= now:
                    bucket["due_count"] += 1
                    pending_age = _seconds_since_iso(row.get("next_attempt_at"), now)
                    if bucket["oldest_pending_age_seconds"] is None or (
                        pending_age is not None and pending_age > bucket["oldest_pending_age_seconds"]
                    ):
                        bucket["oldest_pending_age_seconds"] = pending_age
                if bucket["next_attempt_at"] is None or str(row.get("next_attempt_at") or "") < str(bucket["next_attempt_at"]):
                    bucket["next_attempt_at"] = row.get("next_attempt_at")
            if updated_at is not None and (
                bucket["newest_updated_at"] is None or str(row.get("updated_at") or "") > str(bucket["newest_updated_at"])
            ):
                bucket["newest_updated_at"] = row.get("updated_at")
            if sent_at is not None and (
                bucket["latest_sent_at"] is None or str(row.get("sent_at") or "") > str(bucket["latest_sent_at"])
            ):
                bucket["latest_sent_at"] = row.get("sent_at")
            if created_at is not None and status == "sent" and bucket["latest_sent_at"] is None:
                bucket["latest_sent_at"] = row.get("updated_at")

    groups = sorted(
        grouped.values(),
        key=lambda item: (
            {"actionable": 0, "watch": 1, "expected": 2, "ok": 3}.get(str(item.get("expectation_state")), 9),
            -_as_int(item.get("row_count")),
            str(item.get("action_family")),
            str(item.get("source_surface")),
            str(item.get("endpoint")),
            str(item.get("status")),
        ),
    )
    due_count = sum(_as_int(group.get("due_count")) for group in groups)
    return {
        "schema": "tokenclaw.managed_feedback_queue_freshness.v1",
        "generated_at": utc_now(),
        "available": bool(store_obj is not None and rows is not None),
        "read_only": True,
        "summary": {
            "queue_rows_scanned": scanned_rows,
            "group_memberships": emitted_memberships,
            "group_count": len(groups),
            "queued": status_counts.get("queued", 0),
            "due": due_count,
            "sent": status_counts.get("sent", 0),
            "retryable_error": status_counts.get("retryable-error", 0),
            "dropped_after_limit": status_counts.get("dropped-after-limit", 0),
            "actionable_groups": expectation_counts.get("actionable", 0),
            "watch_groups": expectation_counts.get("watch", 0),
            "expected_groups": expectation_counts.get("expected", 0),
        },
        "managed_mode": product_mode.public_meta(),
        "drain": drain_state,
        "status_breakdown": _breakdown_from_counts(status_counts),
        "action_family_breakdown": _breakdown_from_counts(family_counts),
        "expectation_breakdown": _breakdown_from_counts(expectation_counts),
        "groups": groups,
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "raw_error_text_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }

def _find_nested_dict(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        found = value.get(key)
        if isinstance(found, dict):
            return found
        for item in value.values():
            nested = _find_nested_dict(item, key)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_nested_dict(item, key)
            if nested is not None:
                return nested
    return None

def _safe_thinking_tail_readiness(readiness: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(readiness, dict):
        return {
            "available": False,
            "status": "missing",
            "reason_codes": ["thinking-tail-readiness-missing"],
            "metadata_only": True,
            "raw_payload_included": False,
        }
    schedule = readiness.get("widening_schedule") if isinstance(readiness.get("widening_schedule"), dict) else {}
    reason_codes = readiness.get("reason_codes")
    if not isinstance(reason_codes, list):
        reason_codes = []
    return {
        "available": True,
        "schema": readiness.get("schema"),
        "status": readiness.get("status") or ("ready" if readiness.get("ready") is True else "blocked"),
        "ready": bool(readiness.get("ready")),
        "candidate_id": readiness.get("candidate_id") or schedule.get("candidate_id"),
        "traffic_treatment": readiness.get("traffic_treatment") or schedule.get("traffic_treatment"),
        "next_fraction_cap": schedule.get("next_fraction_cap"),
        "holdout_fraction": schedule.get("holdout_fraction"),
        "expires_at": schedule.get("expires_at"),
        "reason_codes": [str(code) for code in reason_codes if str(code or "").strip()],
        "metadata_only": True,
        "raw_payload_included": False,
    }

def _safe_latest_managed_activation_proof(conn: Any, *, limit: int = 1000) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 10000))
    try:
        rows = conn.execute(
            """
            select created_at, managed_routing_json, routing_json
            from calls
            where managed_routing_json is not null
               or routing_json like '%managed_recommendation%'
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        raw_managed = row["managed_routing_json"] if hasattr(row, "keys") else row[1]
        raw_routing = row["routing_json"] if hasattr(row, "keys") else row[2]
        managed = _json_obj(raw_managed)
        if not managed:
            routing = _json_obj(raw_routing)
            managed = _json_obj(routing.get("managed_recommendation"))
        if not managed:
            continue
        crunch = managed.get("crunch") if isinstance(managed.get("crunch"), dict) else {}
        contract = managed.get("client_contract") if isinstance(managed.get("client_contract"), dict) else {}
        readiness = _safe_thinking_tail_readiness(_find_nested_dict(managed, "thinking_tail_readiness"))
        status = str(managed.get("status") or "observed")
        return {
            "status": "observed",
            "latest_observed_at": row["created_at"] if hasattr(row, "keys") else row[0],
            "decision_status": status,
            "decision_reason": managed.get("reason"),
            "policy_id": managed.get("policy_id"),
            "decision_id": managed.get("decision_id"),
            "local_action_family": "crunch" if readiness.get("available") else managed.get("local_action_family"),
            "candidate_id": readiness.get("candidate_id") or crunch.get("candidate_id") or managed.get("candidate_id"),
            "contract": {
                "status": contract.get("status"),
                "active": bool(contract.get("active")),
                "contract_id": contract.get("contract_id"),
                "cache_status": contract.get("cache_status"),
                "metadata_only": True,
            },
            "thinking_tail_readiness": readiness,
            "metadata_only": True,
            "raw_payload_included": False,
        }
    return {
        "status": "missing",
        "reason": "managed-activation-proof-not-observed",
        "latest_observed_at": None,
        "decision_status": None,
        "candidate_id": None,
        "thinking_tail_readiness": _safe_thinking_tail_readiness(None),
        "metadata_only": True,
        "raw_payload_included": False,
    }

def _safe_thinking_tail_loop_status(
    store_obj: Any | None,
    *,
    activation_proof: dict[str, Any],
    feedback_freshness: dict[str, Any],
    managed_state: dict[str, Any],
    limit: int = 500,
) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 500))
    try:
        from tokenclaw.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report

        impact = build_anthropic_thinking_compaction_impact_report(store_obj, limit=capped) if store_obj is not None else {}
    except Exception as exc:
        impact = {
            "schema": "tokenclaw.anthropic_thinking_compaction_impact.v1",
            "status": "unavailable",
            "summary": {},
            "candidates": [],
            "error_type": type(exc).__name__,
            "privacy": _metadata_only_privacy(),
        }
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    lifecycle = summary.get("lifecycle_coverage") if isinstance(summary.get("lifecycle_coverage"), dict) else {}
    readiness = (
        activation_proof.get("thinking_tail_readiness")
        if isinstance(activation_proof.get("thinking_tail_readiness"), dict)
        else {}
    )
    backlog = (
        feedback_freshness.get("backlog_proof")
        if isinstance(feedback_freshness.get("backlog_proof"), dict)
        else {}
    )
    candidates = [row for row in impact.get("candidates") or [] if isinstance(row, dict)]
    first_observed_at = None
    latest_outcome_at = None
    for row in candidates:
        first = row.get("first_observed_at")
        latest = row.get("last_observed_at")
        if first and (first_observed_at is None or str(first) < first_observed_at):
            first_observed_at = str(first)
        if latest and (latest_outcome_at is None or str(latest) > latest_outcome_at):
            latest_outcome_at = str(latest)
    if latest_outcome_at is None:
        latest_outcome_at = feedback_freshness.get("newest_local_outcome_at")

    applied = _as_int(summary.get("applied_count"))
    holdout = _as_int(summary.get("holdout_count"))
    skipped = _as_int(summary.get("skipped_count"))
    safety_stop = _as_int(summary.get("safety_stop_count"))
    observed = _as_int(summary.get("observed_thinking_compaction_metadata_row_count"))
    feedback_pending = _as_int(backlog.get("pending"))
    feedback_due = _as_int(backlog.get("due"))
    feedback_sent = _as_int(backlog.get("sent"))
    feedback_dropped = _as_int(backlog.get("dropped"))
    reason_codes = [str(code) for code in feedback_freshness.get("reason_codes") or [] if str(code or "").strip()]

    if safety_stop:
        state = "safety-stopped"
    elif applied and _as_float(summary.get("net_savings_usd")) > 0:
        state = "saving"
    elif applied:
        state = "applied-observed"
    elif holdout:
        state = "held-for-evidence"
    elif readiness.get("ready"):
        state = "ready-to-widen"
    elif feedback_pending or feedback_due or feedback_dropped:
        state = "feedback-blocked"
    elif observed:
        state = "observed"
    else:
        state = "no-local-evidence"

    if feedback_dropped:
        feedback_status = "blocked"
        feedback_blocker = "thinking-tail-feedback-dropped"
    elif feedback_due:
        feedback_status = "blocked"
        feedback_blocker = "thinking-tail-feedback-due"
    elif feedback_pending:
        feedback_status = "queued"
        feedback_blocker = "thinking-tail-feedback-pending"
    elif feedback_sent:
        feedback_status = "sent"
        feedback_blocker = None
    else:
        feedback_status = "missing"
        feedback_blocker = "thinking-tail-feedback-missing"

    decision_status = activation_proof.get("decision_status")
    policy_source = "managed-recommended" if activation_proof.get("status") == "observed" else "local-default"
    if not bool((managed_state.get("managed_enabled") if isinstance(managed_state, dict) else False)):
        policy_source = "local-default"

    return {
        "schema": "tokenclaw.thinking_tail_compaction_loop_status.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "action_family": "crunch",
        "candidate_id": activation_proof.get("candidate_id") or readiness.get("candidate_id") or backlog.get("candidate_id"),
        "current_rule_state": {
            "state": state,
            "impact_status": impact.get("status") or "unknown",
            "policy_source": policy_source,
            "decision_status": decision_status,
            "traffic_treatment": readiness.get("traffic_treatment"),
            "next_fraction_cap": readiness.get("next_fraction_cap"),
            "holdout_fraction": readiness.get("holdout_fraction"),
            "ready": bool(readiness.get("ready")),
            "top_blocker_reason": reason_codes[0] if reason_codes and reason_codes[0] != "thinking-tail-feedback-fresh" else None,
        },
        "canary": {
            "observed_count": observed,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": skipped,
            "safety_stop_count": safety_stop,
            "applied_rate": _as_float(lifecycle.get("applied_rate")),
            "holdout_rate": _as_float(lifecycle.get("holdout_rate")),
            "safety_stop_rate": _as_float(lifecycle.get("safety_stop_rate")),
            "target_fraction": readiness.get("next_fraction_cap"),
            "holdout_fraction": readiness.get("holdout_fraction"),
        },
        "savings": {
            "realized_savings_usd": round(_as_float(summary.get("net_savings_usd")), 8),
            "observed_gross_savings_usd": round(_as_float(summary.get("observed_saved_usd")), 8),
            "projected_savings_usd": round(_as_float(summary.get("projected_saved_usd")), 8),
            "projected_holdout_savings_usd": round(_as_float(summary.get("projected_holdout_savings_usd")), 8),
            "observed_saved_tokens": _as_int(summary.get("observed_saved_tokens")),
            "projected_saved_tokens": _as_int(summary.get("projected_saved_tokens")),
        },
        "quality": {
            "applied_minus_holdout_error_rate": round(_as_float(summary.get("applied_minus_holdout_error_rate")), 6),
            "applied_minus_holdout_retry_rate": round(_as_float(summary.get("applied_minus_holdout_retry_rate")), 6),
            "budget_governor_action": summary.get("budget_governor_action"),
            "canary_impact_decision": summary.get("canary_impact_decision"),
        },
        "outcome_window": {
            "first_observed_at": first_observed_at,
            "latest_outcome_at": latest_outcome_at,
            "newest_local_outcome_age_seconds": feedback_freshness.get("newest_local_outcome_age_seconds"),
            "last_successful_drain_at": feedback_freshness.get("last_successful_drain_at"),
            "last_successful_drain_age_seconds": feedback_freshness.get("last_successful_drain_age_seconds"),
        },
        "managed_feedback": {
            "status": feedback_status,
            "blocked_reason": feedback_blocker,
            "queued": _as_int(backlog.get("queued")),
            "pending": feedback_pending,
            "due": feedback_due,
            "sent": feedback_sent,
            "dropped": feedback_dropped,
            "retryable_error": _as_int(backlog.get("retryable_error")),
            "queue_fraction": _as_float(backlog.get("queue_fraction")),
            "status_counts": dict(backlog.get("status_counts") or {}),
            "payload_json_included": False,
        },
        "reason_codes": reason_codes,
        "lookback_limit": capped,
        "privacy": {
            **_metadata_only_privacy(),
            "payload_json_included": False,
            "raw_messages_included": False,
            "raw_thinking_text_included": False,
            "raw_provider_bodies_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
        },
    }

def stats_managed_activation_status(store_obj: Any | None, *, limit: int = 10000) -> dict[str, Any]:
    capped = max(1, min(int(limit or 1), 10000))
    managed_state = _managed_feed_state()
    freshness = stats_managed_feedback_queue_freshness(store_obj, limit=capped)
    freshness_summary = freshness.get("summary") if isinstance(freshness.get("summary"), dict) else {}
    try:
        from tokenclaw.local_compaction_canary_ramp import build_thinking_tail_feedback_freshness

        thinking_tail_feedback = build_thinking_tail_feedback_freshness(store_obj, limit=capped) if store_obj is not None else {}
    except Exception as exc:
        thinking_tail_feedback = {
            "schema": "tokenclaw.thinking_tail_feedback_freshness.v1",
            "status": "unavailable",
            "reason_codes": ["thinking-tail-feedback-freshness-unavailable"],
            "error_type": type(exc).__name__,
            "payload_json_included": False,
        }
    proof = _safe_latest_managed_activation_proof(store_obj.conn, limit=capped) if store_obj is not None else _safe_latest_managed_activation_proof(None, limit=capped)
    backlog_proof = thinking_tail_feedback.get("backlog_proof") if isinstance(thinking_tail_feedback.get("backlog_proof"), dict) else {}
    reason_codes: list[str] = []
    if proof.get("status") != "observed":
        reason_codes.append(str(proof.get("reason") or "managed-activation-proof-not-observed"))
    for code in thinking_tail_feedback.get("reason_codes") or []:
        text = str(code or "")
        if text and text != "thinking-tail-feedback-fresh":
            reason_codes.append(text)
    if _as_int(freshness_summary.get("actionable_groups")):
        reason_codes.append("managed-feedback-actionable-groups")
    elif _as_int(freshness_summary.get("watch_groups")):
        reason_codes.append("managed-feedback-watch-groups")
    clean_reasons = sorted({code for code in reason_codes if code})
    status = "ready" if not clean_reasons else "blocked"
    if proof.get("status") != "observed":
        status = "no-proof"
    loop_status = _safe_thinking_tail_loop_status(
        store_obj,
        activation_proof=proof,
        feedback_freshness=thinking_tail_feedback,
        managed_state=managed_state,
        limit=min(capped, 500),
    )
    return {
        "schema": "tokenclaw.managed_activation_dashboard_status.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "status": status,
        "top_blocker_reason": clean_reasons[0] if clean_reasons else None,
        "reason_codes": clean_reasons,
        "managed_mode": managed_state,
        "activation_proof": proof,
        "thinking_tail_feedback_freshness": {
            "schema": thinking_tail_feedback.get("schema"),
            "status": thinking_tail_feedback.get("status"),
            "stale": bool(thinking_tail_feedback.get("stale")),
            "reason_codes": [str(code) for code in thinking_tail_feedback.get("reason_codes") or []],
            "newest_local_outcome_at": thinking_tail_feedback.get("newest_local_outcome_at"),
            "newest_local_outcome_age_seconds": thinking_tail_feedback.get("newest_local_outcome_age_seconds"),
            "last_successful_drain_at": thinking_tail_feedback.get("last_successful_drain_at"),
            "last_successful_drain_age_seconds": thinking_tail_feedback.get("last_successful_drain_age_seconds"),
            "backlog_proof": {
                "action_family": backlog_proof.get("action_family"),
                "candidate_id": backlog_proof.get("candidate_id"),
                "row_count": _as_int(backlog_proof.get("row_count")),
                "queued": _as_int(backlog_proof.get("queued")),
                "retryable_error": _as_int(backlog_proof.get("retryable_error")),
                "sent": _as_int(backlog_proof.get("sent")),
                "dropped": _as_int(backlog_proof.get("dropped")),
                "due": _as_int(backlog_proof.get("due")),
                "pending": _as_int(backlog_proof.get("pending")),
                "queue_fraction": _as_float(backlog_proof.get("queue_fraction")),
                "status_counts": dict(backlog_proof.get("status_counts") or {}),
                "payload_json_included": False,
            },
            "payload_json_included": False,
        },
        "thinking_tail_compaction_loop_status": loop_status,
        "feedback_burndown": {
            "schema": "tokenclaw.managed_activation_feedback_burndown.v1",
            "queue_rows_scanned": _as_int(freshness_summary.get("queue_rows_scanned")),
            "queued": _as_int(freshness_summary.get("queued")),
            "due": _as_int(freshness_summary.get("due")),
            "sent": _as_int(freshness_summary.get("sent")),
            "retryable_error": _as_int(freshness_summary.get("retryable_error")),
            "dropped_after_limit": _as_int(freshness_summary.get("dropped_after_limit")),
            "actionable_groups": _as_int(freshness_summary.get("actionable_groups")),
            "watch_groups": _as_int(freshness_summary.get("watch_groups")),
            "expected_groups": _as_int(freshness_summary.get("expected_groups")),
            "status_breakdown": freshness.get("status_breakdown") or [],
            "action_family_breakdown": freshness.get("action_family_breakdown") or [],
            "expectation_breakdown": freshness.get("expectation_breakdown") or [],
            "drain": freshness.get("drain") if isinstance(freshness.get("drain"), dict) else {},
            "payload_json_included": False,
        },
        "privacy": {
            **_metadata_only_privacy(),
            "payload_json_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_server_responses_included": False,
            "api_key_value_included": False,
            "managed_server_calls_made": False,
            "provider_calls_made": False,
        },
    }

MANAGED_OPENAI_ACTIVATION_ACTIONS = {
    "fetch-review",
    "draft-stage",
    "openai-optimization-draft-dry-run",
    "draft-apply",
    "rollback",
}

MANAGED_OPENAI_SUPPORTED_ACTION_FAMILIES = ("routing", "old_context_summarization", "cache")

def _openai_activation_counts_by_family(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, Any]] = []
    for family, counts in raw.items():
        if not isinstance(counts, dict):
            continue
        rows.append({
            "family": str(family or "unknown"),
            "selected": _as_int(counts.get("selected")),
            "suppressed": _as_int(counts.get("suppressed")),
            "omitted": _as_int(counts.get("omitted")),
        })
    rows.sort(key=lambda row: (-row["selected"], row["family"]))
    return rows

def _openai_activation_conflict_summary(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        key: _as_int(raw.get(key))
        for key in (
            "conflict_count",
            "selected_action_count",
            "suppressed_action_count",
            "omitted_action_count",
            "local_capability_gap_count",
        )
        if _as_int(raw.get(key)) > 0
    }

def _openai_activation_review_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    review = metadata.get("openai_optimization_review") if isinstance(metadata.get("openai_optimization_review"), dict) else {}
    selected_actions = review.get("selected_actions") if isinstance(review.get("selected_actions"), list) else []
    suppressed_actions = review.get("suppressed_actions") if isinstance(review.get("suppressed_actions"), list) else []
    omitted_actions = review.get("omitted_actions") if isinstance(review.get("omitted_actions"), list) else []
    selected_families = sorted({
        str(action.get("action_family") or "unknown")
        for action in selected_actions
        if isinstance(action, dict)
    })
    return {
        "schema": review.get("schema"),
        "source": review.get("source"),
        "review_bundle_schema": review.get("review_bundle_schema"),
        "selected_action_count": _as_int(review.get("selected_action_count")) or len(selected_actions),
        "suppressed_action_count": _as_int(review.get("suppressed_action_count")) or len(suppressed_actions),
        "omitted_action_count": _as_int(review.get("omitted_action_count")) or len(omitted_actions),
        "staged_action_count": _as_int(review.get("staged_action_count")) or len(selected_actions),
        "staged_policy_sections": review.get("staged_policy_sections") if isinstance(review.get("staged_policy_sections"), list) else [],
        "selected_action_families": selected_families,
        "counts_by_family": _openai_activation_counts_by_family(review.get("counts_by_family")),
        "conflict_summary": _openai_activation_conflict_summary(review.get("conflict_summary")),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "payload_json_included": False,
        },
    }

def _managed_openai_activation_staged_drafts(limit: int = 10) -> tuple[list[dict[str, Any]], int]:
    from tokenclaw.policy_files import _draft_workspace_root

    raw_workspace = os.getenv("TOKENCLAW_POLICY_DRAFT_DIR")
    workspace = _draft_workspace_root(raw_workspace)
    if not workspace.exists() or not workspace.is_dir():
        return [], 0

    rows: list[tuple[str, float, dict[str, Any]]] = []
    unreadable = 0
    try:
        children = list(workspace.iterdir())
    except OSError:
        return [], 1
    for child in children:
        manifest_path = child / "draft.json" if child.is_dir() else child
        if not manifest_path.exists() or manifest_path.name != "draft.json":
            continue
        manifest = _read_workbench_draft_manifest(manifest_path)
        if manifest is None:
            unreadable += 1
            continue
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        if not isinstance(metadata.get("openai_optimization_review"), dict):
            continue
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        review = _openai_activation_review_metadata(manifest)
        rows.append((
            str(manifest.get("created_at") or ""),
            mtime,
            {
                "draft_id": manifest.get("draft_id"),
                "created_at": manifest.get("created_at"),
                "changed": bool(manifest.get("changed")),
                "changed_sections": manifest.get("changed_sections") if isinstance(manifest.get("changed_sections"), list) else [],
                "change_count": _as_int(manifest.get("change_count")),
                "openai_optimization_review": review,
                "workspace_path_class": _local_path_class(raw_workspace or "~/.tokenclaw/policy_drafts"),
                "workspace_path_included": False,
                "manifest_path_included": False,
                "bundle_path_included": False,
                "raw_payload_included": False,
            },
        ))
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _created, _mtime, row in rows[: max(1, min(int(limit), 50))]], unreadable

def _public_managed_openai_activation_event(event: dict[str, Any] | None, *, now: datetime) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    review = details.get("openai_optimization_review") if isinstance(details.get("openai_optimization_review"), dict) else {}
    counts = {
        "selected_action_count": _as_int(review.get("selected_action_count")),
        "suppressed_action_count": _as_int(review.get("suppressed_action_count")),
        "omitted_action_count": _as_int(review.get("omitted_action_count")),
        "local_capability_gap_count": _as_int(review.get("local_capability_gap_count")),
        "candidate_count": _as_int(details.get("candidate_count")),
        "change_count": _as_int(details.get("change_count")),
        "safety_warning_count": _as_int(details.get("safety_warning_count")),
        "openai_rows_considered": _as_int(details.get("openai_rows_considered")),
        "applied_if_enabled_total": _as_int(details.get("applied_if_enabled_total")),
        "suppressed_total": _as_int(details.get("suppressed_total")),
        "planned_action_count": _as_int(details.get("planned_action_count")),
    }
    return {
        "created_at": event.get("created_at"),
        "age_seconds": _seconds_since_iso(event.get("created_at"), now),
        "action": event.get("action"),
        "ok": bool(event.get("ok")),
        "source": details.get("source"),
        "status": details.get("status") or review.get("status"),
        "draft_id": details.get("draft_id") or details.get("draft"),
        "dry_run": bool(details.get("dry_run")),
        "provenance_status": details.get("provenance_status"),
        "provenance_managed_bundle": details.get("provenance_managed_bundle"),
        "status_code": details.get("status_code"),
        "error_type": details.get("error_type"),
        "exit_code": details.get("exit_code"),
        "changed_sections": details.get("changed_sections") if isinstance(details.get("changed_sections"), list) else [],
        "applied_sections": details.get("applied_sections") if isinstance(details.get("applied_sections"), list) else [],
        "counts": {key: value for key, value in counts.items() if value not in (None, 0)},
        "payload_included": False,
        "raw_payload_included": False,
        "file_paths_included": False,
        "provider_calls_made": bool(details.get("provider_calls_made")),
        "managed_server_calls_made": bool(details.get("managed_server_calls_made")),
        "active_policy_files_written": bool(details.get("active_policy_files_written")),
    }

def _managed_openai_activation_status(
    *,
    latest_fetch: dict[str, Any] | None,
    latest_dry_run: dict[str, Any] | None,
    latest_apply: dict[str, Any] | None,
    staged_draft_count: int,
    feedback_summary: dict[str, Any],
) -> tuple[str, str]:
    if latest_fetch and latest_fetch.get("ok") is False:
        return "fetch-blocked", "latest managed fetch/review failed"
    if latest_dry_run and latest_dry_run.get("ok") is False:
        return "dry-run-blocked", "latest OpenAI draft dry-run failed"
    if latest_apply and latest_apply.get("ok") is False:
        return "apply-blocked", "latest OpenAI draft apply was blocked or failed"
    if _as_int(feedback_summary.get("dropped_after_limit")) > 0:
        return "feedback-blocked", "OpenAI optimization lifecycle feedback has dropped rows"
    if _as_int(feedback_summary.get("due")) > 0 or _as_int(feedback_summary.get("retryable_error")) > 0:
        return "feedback-due", "OpenAI optimization lifecycle feedback has due or retryable queue rows"
    if latest_apply and latest_apply.get("status") == "applied":
        return "canary-active", "latest apply activated local canaries"
    if latest_dry_run and latest_dry_run.get("ok"):
        return "draft-ready", "latest staged draft has a passing local dry-run"
    if staged_draft_count:
        return "draft-staged", "managed OpenAI draft is staged and waiting for dry-run"
    if latest_fetch and latest_fetch.get("ok"):
        return "bundle-reviewed", "managed OpenAI bundle was fetched and reviewed"
    return "local-only", "no managed OpenAI activation metadata found"

async def stats_managed_openai_activation(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.openai_optimization_governor import LIFECYCLE_SOURCE_SURFACE
    from tokenclaw.policy_events import recent_policy_events

    capped_limit = max(1, min(int(limit or 500), 5000))
    now = datetime.now(timezone.utc)
    events = [
        event
        for event in recent_policy_events(limit=500).get("events", [])
        if isinstance(event, dict) and str(event.get("action") or "") in MANAGED_OPENAI_ACTIVATION_ACTIONS
    ]
    latest_fetch = _public_managed_openai_activation_event(_latest_policy_event(events, "fetch-review"), now=now)
    latest_stage = _public_managed_openai_activation_event(_latest_policy_event(events, "draft-stage"), now=now)
    latest_dry_run = _public_managed_openai_activation_event(_latest_policy_event(events, "openai-optimization-draft-dry-run"), now=now)
    latest_apply = _public_managed_openai_activation_event(_latest_policy_event(events, "draft-apply"), now=now)
    latest_rollback = _public_managed_openai_activation_event(_latest_policy_event(events, "rollback"), now=now)
    public_events = [
        item
        for item in (_public_managed_openai_activation_event(event, now=now) for event in events)
        if item is not None
    ]
    staged_drafts, unreadable_draft_count = _managed_openai_activation_staged_drafts(limit=10)
    latest_draft = staged_drafts[0] if staged_drafts else None
    latest_review = (latest_draft or {}).get("openai_optimization_review") if isinstance(latest_draft, dict) else {}
    if not isinstance(latest_review, dict):
        latest_review = {}

    feedback_queue = _managed_feedback_queue_health(
        store_obj,
        sample_limit=5,
        source_surface=LIFECYCLE_SOURCE_SURFACE,
    )
    feedback_summary = feedback_queue.get("summary") if isinstance(feedback_queue.get("summary"), dict) else {}
    readiness: dict[str, Any] = {}
    try:
        readiness = await stats_openai_optimization_readiness(store_obj, limit=min(capped_limit, 1000))
    except Exception:
        readiness = {}
    readiness_summary = readiness.get("summary") if isinstance(readiness.get("summary"), dict) else {}

    status, status_reason = _managed_openai_activation_status(
        latest_fetch=latest_fetch,
        latest_dry_run=latest_dry_run,
        latest_apply=latest_apply,
        staged_draft_count=len(staged_drafts),
        feedback_summary=feedback_summary,
    )
    selected_count = _as_int(latest_review.get("selected_action_count")) or _as_int(((latest_fetch or {}).get("counts") or {}).get("selected_action_count"))
    suppressed_count = _as_int(latest_review.get("suppressed_action_count")) or _as_int(((latest_fetch or {}).get("counts") or {}).get("suppressed_action_count"))
    omitted_count = _as_int(latest_review.get("omitted_action_count")) or _as_int(((latest_fetch or {}).get("counts") or {}).get("omitted_action_count"))
    family_counts = latest_review.get("counts_by_family") if isinstance(latest_review.get("counts_by_family"), list) else []

    return {
        "schema": "tokenclaw.managed_openai_activation.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "read_only": True,
        "limit": capped_limit,
        "summary": {
            "policy_event_count": len(public_events),
            "staged_draft_count": len(staged_drafts),
            "unreadable_staged_draft_count": unreadable_draft_count,
            "selected_action_count": selected_count,
            "suppressed_action_count": suppressed_count,
            "omitted_action_count": omitted_count,
            "staged_action_count": _as_int(latest_review.get("staged_action_count")),
            "conflict_count": _as_int((latest_review.get("conflict_summary") or {}).get("conflict_count")) if isinstance(latest_review.get("conflict_summary"), dict) else 0,
            "openai_lifecycle_feedback_total": _as_int(feedback_summary.get("total")),
            "openai_lifecycle_feedback_queued": _as_int(feedback_summary.get("queued")),
            "openai_lifecycle_feedback_due": _as_int(feedback_summary.get("due")),
            "openai_lifecycle_feedback_retryable_error": _as_int(feedback_summary.get("retryable_error")),
            "openai_lifecycle_feedback_dropped_after_limit": _as_int(feedback_summary.get("dropped_after_limit")),
            "oldest_pending_age_seconds": feedback_summary.get("oldest_pending_age_seconds"),
            "selected_call_count": _as_int(readiness_summary.get("selected_call_count")),
            "conflicting_call_count": _as_int(readiness_summary.get("conflicting_call_count")),
            "holdout_family_count": _as_int(readiness_summary.get("holdout_family_count")),
            "safety_stop_family_count": _as_int(readiness_summary.get("safety_stop_family_count")),
        },
        "bundle_health": {
            "latest_fetch_review": latest_fetch,
            "provenance_status": (latest_fetch or {}).get("provenance_status"),
            "provenance_managed_bundle": (latest_fetch or {}).get("provenance_managed_bundle"),
            "selected_action_count": selected_count,
            "suppressed_action_count": suppressed_count,
            "omitted_action_count": omitted_count,
            "supported_local_action_families": list(MANAGED_OPENAI_SUPPORTED_ACTION_FAMILIES),
            "selected_action_families": latest_review.get("selected_action_families") if isinstance(latest_review.get("selected_action_families"), list) else [],
            "counts_by_family": family_counts,
            "conflict_summary": _openai_activation_conflict_summary(latest_review.get("conflict_summary")),
        },
        "drafts": {
            "count": len(staged_drafts),
            "unreadable_count": unreadable_draft_count,
            "latest": latest_draft,
            "recent": staged_drafts,
        },
        "dry_run": latest_dry_run,
        "apply_or_rollback": latest_apply or latest_rollback,
        "active_canary_health": {
            "state": readiness.get("state"),
            "state_reason": readiness.get("state_reason"),
            "selected_call_count": _as_int(readiness_summary.get("selected_call_count")),
            "conflicting_call_count": _as_int(readiness_summary.get("conflicting_call_count")),
            "holdout_family_count": _as_int(readiness_summary.get("holdout_family_count")),
            "safety_stop_family_count": _as_int(readiness_summary.get("safety_stop_family_count")),
            "selected_family_breakdown": readiness.get("selected_family_breakdown") if isinstance(readiness.get("selected_family_breakdown"), list) else [],
            "suppression_reason_breakdown": readiness.get("suppression_reason_breakdown") if isinstance(readiness.get("suppression_reason_breakdown"), list) else [],
        },
        "feedback_queue": feedback_queue,
        "recent_events": public_events[:25],
        "next_read_only_command": (
            "tokenclaw-openai-optimization-draft-dry-run <draft-id> --pretty"
            if staged_drafts and not (latest_dry_run and latest_dry_run.get("ok"))
            else "tokenclaw-openai-optimization-draft-apply <draft-id> --dry-run --pretty"
            if latest_dry_run and latest_dry_run.get("ok") and not (latest_apply and latest_apply.get("status") == "applied")
            else None
        ),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "dashboard_read_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "local_session_ids_included": False,
            "payload_json_included": False,
            "policy_file_contents_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local policy-event metadata, staged draft manifests, OpenAI governor aggregates, and queue row status/age counts only",
        },
    }

def _phase_routing_feedback_rows(store_obj: Any, *, source_surface: str, limit: int) -> list[dict[str, Any]]:
    if store_obj is None:
        return []
    capped = max(1, min(int(limit or 500), 5000))
    if hasattr(store_obj, "conn"):
        try:
            rows = store_obj.conn.execute(
                """
                select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                       payload_json, status, attempts, next_attempt_at, last_error,
                       last_status_code, sent_at
                from managed_outcome_feedback_queue
                where source_surface = ?
                order by created_at desc
                limit ?
                """,
                (source_surface, capped),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            pass
    if not hasattr(store_obj, "managed_outcome_feedback_rows"):
        return []
    try:
        return store_obj.managed_outcome_feedback_rows(source_surface=source_surface, limit=capped)
    except Exception:
        return []

def _phase_routing_public_lifecycle_rows(rows: list[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for row in rows:
        payload = _json_obj(row.get("payload_json"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        public.append({
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "age_seconds": _seconds_since_iso(row.get("created_at"), now),
            "source_surface": row.get("source_surface"),
            "endpoint": row.get("endpoint"),
            "status": row.get("status"),
            "attempts": _as_int(row.get("attempts")),
            "last_status_code": row.get("last_status_code"),
            "sent_at": row.get("sent_at"),
            "event_type": payload.get("event_type"),
            "recommendation_id_present": bool(payload.get("recommendation_id")),
            "command": metadata.get("command"),
            "local_result_status": metadata.get("local_result_status"),
            "dry_run": bool(metadata.get("dry_run")),
            "read_only": bool(metadata.get("read_only", True)),
            "policy_source": metadata.get("policy_source"),
            "sampled_call_count": _as_int(metadata.get("sampled_call_count")),
            "rule_count": _as_int(metadata.get("rule_count")),
            "matched_count": _as_int(metadata.get("matched_count")),
            "projected_candidate_count": _as_int(metadata.get("projected_candidate_count")),
            "excluded_count": _as_int(metadata.get("excluded_count")),
            "projected_savings_usd": round(_as_float(metadata.get("projected_savings_usd")), 6),
            "risk_warning_count": _as_int(metadata.get("risk_warning_count")),
            "candidate_rule_count": len(metadata.get("candidate_rule_ids") or []) if isinstance(metadata.get("candidate_rule_ids"), list) else 0,
            "excluded_count_by_reason": _count_breakdown(metadata.get("excluded_count_by_reason") if isinstance(metadata.get("excluded_count_by_reason"), dict) else {}),
            "payload_included": False,
            "raw_payload_included": False,
        })
    return public

def _phase_routing_public_health_rows(limit: int = 25) -> list[dict[str, Any]]:
    from tokenclaw.policy_events import recent_policy_events

    rows: list[dict[str, Any]] = []
    for event in recent_policy_events(limit=100).get("events", []):
        if not isinstance(event, dict):
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        health = details.get("recommendation_health") if isinstance(details.get("recommendation_health"), dict) else {}
        for item in health.get("rows") or []:
            if not isinstance(item, dict):
                continue
            haystack = " ".join(
                str(value or "")
                for value in (
                    item.get("candidate_id"),
                    item.get("rule_id"),
                    item.get("policy_id"),
                    item.get("kind"),
                    item.get("code"),
                    details.get("policy_sections"),
                )
            ).lower()
            if "phase" not in haystack and "routing" not in haystack:
                continue
            rows.append({
                "event_created_at": event.get("created_at"),
                "action": event.get("action"),
                "ok": bool(event.get("ok")),
                "kind": item.get("kind"),
                "code": item.get("code"),
                "candidate_id": item.get("candidate_id"),
                "rule_id": item.get("rule_id") or item.get("policy_id"),
                "status": health.get("status"),
                "warning_count": _as_int(health.get("warning_count")),
                "details_included": False,
                "raw_payload_included": False,
            })
            if len(rows) >= max(1, int(limit)):
                return rows
    return rows

def _phase_canary_cohort(meta: dict[str, Any]) -> str:
    status = str(meta.get("status") or "unknown")
    cohort = str(meta.get("cohort") or "")
    reason = str(meta.get("reason") or "")
    if status == "applied" or cohort in {"applied", "canary_applied"}:
        return "applied"
    if status == "holdout" or cohort in {"holdout", "canary_holdout"}:
        return "holdout"
    if status == "safety_stopped" or cohort == "bypassed_or_disabled" or "safety-stop" in reason:
        return "safety_stopped"
    if status in {"not_selected", "ineligible", "disabled"}:
        return status
    if cohort == "skipped":
        return "not_selected"
    return "unknown"

def _new_phase_canary_bucket(policy: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": meta.get("policy_id") or policy.get("policy_id"),
        "policy_source": meta.get("policy_source") or policy.get("policy_source"),
        "target_model": meta.get("target_model") or policy.get("target_model"),
        "last_decision_at": None,
        "observed_rows": 0,
        "enabled_rows": 0,
        "applied_rows": 0,
        "holdout_rows": 0,
        "not_selected_rows": 0,
        "ineligible_rows": 0,
        "safety_stop_rows": 0,
        "error_rows": 0,
        "retry_rows": 0,
        "fallback_rows": 0,
        "current_cost_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "observed_savings_usd": 0.0,
        "cohorts": {
            "applied": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "holdout": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "not_selected": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "ineligible": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "safety_stopped": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
            "unknown": {"count": 0, "error_count": 0, "retry_count": 0, "fallback_count": 0, "latency_ms_total": 0, "latency_sample_count": 0},
        },
    }

def _finalize_phase_canary_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for name, raw in bucket["cohorts"].items():
        count = _as_int(raw.get("count"))
        latency_samples = _as_int(raw.get("latency_sample_count"))
        cohorts[name] = {
            "count": count,
            "error_count": _as_int(raw.get("error_count")),
            "retry_count": _as_int(raw.get("retry_count")),
            "fallback_count": _as_int(raw.get("fallback_count")),
            "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
            "retry_rate": round(_as_int(raw.get("retry_count")) / count, 6) if count else 0.0,
            "fallback_rate": round(_as_int(raw.get("fallback_count")) / count, 6) if count else 0.0,
            "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        }
    applied = cohorts["applied"]
    holdout = cohorts["holdout"]
    latency_delta = None
    if applied["latency_avg_ms"] is not None and holdout["latency_avg_ms"] is not None:
        latency_delta = round(_as_float(applied["latency_avg_ms"]) - _as_float(holdout["latency_avg_ms"]), 2)
    return {
        "policy_id": bucket.get("policy_id"),
        "policy_source": bucket.get("policy_source"),
        "target_model": bucket.get("target_model"),
        "last_decision_at": bucket.get("last_decision_at"),
        "observed_rows": _as_int(bucket.get("observed_rows")),
        "enabled_rows": _as_int(bucket.get("enabled_rows")),
        "applied_rows": _as_int(bucket.get("applied_rows")),
        "holdout_rows": _as_int(bucket.get("holdout_rows")),
        "not_selected_rows": _as_int(bucket.get("not_selected_rows")),
        "ineligible_rows": _as_int(bucket.get("ineligible_rows")),
        "safety_stop_rows": _as_int(bucket.get("safety_stop_rows")),
        "error_rows": _as_int(bucket.get("error_rows")),
        "retry_rows": _as_int(bucket.get("retry_rows")),
        "fallback_rows": _as_int(bucket.get("fallback_rows")),
        "error_rate": round(_as_int(bucket.get("error_rows")) / _as_int(bucket.get("observed_rows")), 6) if _as_int(bucket.get("observed_rows")) else 0.0,
        "retry_rate": round(_as_int(bucket.get("retry_rows")) / _as_int(bucket.get("observed_rows")), 6) if _as_int(bucket.get("observed_rows")) else 0.0,
        "fallback_rate": round(_as_int(bucket.get("fallback_rows")) / _as_int(bucket.get("observed_rows")), 6) if _as_int(bucket.get("observed_rows")) else 0.0,
        "current_cost_usd": round(_as_float(bucket.get("current_cost_usd")), 6),
        "baseline_cost_usd": round(_as_float(bucket.get("baseline_cost_usd")), 6),
        "observed_savings_usd": round(_as_float(bucket.get("observed_savings_usd")), 6),
        "applied_minus_holdout_error_rate": round(_as_float(applied["error_rate"]) - _as_float(holdout["error_rate"]), 6),
        "applied_minus_holdout_retry_rate": round(_as_float(applied["retry_rate"]) - _as_float(holdout["retry_rate"]), 6),
        "applied_minus_holdout_latency_avg_ms": latency_delta,
        "cohorts": cohorts,
    }

async def stats_phase_routing(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.phase_routing_report import build_phase_routing_report, _load_recent_rows

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    now = datetime.now(timezone.utc)
    policy_state = await _stats_policies_facade()
    routing_policy = policy_state.get("routing") if isinstance(policy_state.get("routing"), dict) else {}
    phase_policy = routing_policy.get("phase_canary") if isinstance(routing_policy.get("phase_canary"), dict) else {}
    opportunity = build_phase_routing_report(store_obj, limit=capped_limit)
    rows = _load_recent_rows(store_obj, limit=capped_limit)

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    feedback_status_counts: dict[str, int] = {}
    feedback_reason_counts: dict[str, int] = {}
    safety_stop_reason_counts: dict[str, int] = {}
    latest_safety_stop: dict[str, Any] | None = None
    buckets: dict[str, dict[str, Any]] = {}

    for row in rows:
        routing = _json_obj(row.get("routing_json"))
        canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
        feedback = routing.get("phase_routing_feedback") if isinstance(routing.get("phase_routing_feedback"), dict) else {}
        if feedback:
            status = str(feedback.get("status") or "unknown")
            reason = str(feedback.get("reason") or "unknown")
            feedback_status_counts[status] = feedback_status_counts.get(status, 0) + 1
            feedback_reason_counts[reason] = feedback_reason_counts.get(reason, 0) + 1
        if not canary:
            continue
        policy_id = str(canary.get("policy_id") or phase_policy.get("policy_id") or "local-phase-sonnet-haiku-canary-v1")
        bucket = buckets.setdefault(policy_id, _new_phase_canary_bucket(phase_policy, canary))
        created_at = row.get("created_at")
        if created_at and str(created_at) > str(bucket.get("last_decision_at") or ""):
            bucket["last_decision_at"] = created_at
        bucket["observed_rows"] += 1
        bucket["enabled_rows"] += int(bool(canary.get("enabled")))
        cohort = _phase_canary_cohort(canary)
        status = str(canary.get("status") or "unknown")
        reason = str(canary.get("reason") or "unknown")
        phase = str(canary.get("workflow_phase") or routing.get("workflow_phase") or "unknown")
        confidence = str(canary.get("workflow_phase_confidence") or routing.get("workflow_phase_confidence") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        if cohort == "applied":
            bucket["applied_rows"] += 1
        elif cohort == "holdout":
            bucket["holdout_rows"] += 1
        elif cohort == "not_selected":
            bucket["not_selected_rows"] += 1
        elif cohort == "ineligible":
            bucket["ineligible_rows"] += 1
        elif cohort == "safety_stopped":
            bucket["safety_stop_rows"] += 1
        errored = _as_int(row.get("status_code")) >= 400
        retried = _as_int(row.get("retry_count")) > 0
        fallback = bool(routing.get("fallback_reason") or canary.get("fallback_reason"))
        bucket["error_rows"] += int(errored)
        bucket["retry_rows"] += int(retried)
        bucket["fallback_rows"] += int(fallback)
        current = _as_float(row.get("cost_est_usd"))
        baseline = _as_float(row.get("cost_baseline_usd"))
        bucket["current_cost_usd"] += current
        bucket["baseline_cost_usd"] += baseline
        if cohort == "applied":
            bucket["observed_savings_usd"] += max(0.0, baseline - current)
        cohort_bucket = bucket["cohorts"].get(cohort, bucket["cohorts"]["unknown"])
        cohort_bucket["count"] += 1
        cohort_bucket["error_count"] += int(errored)
        cohort_bucket["retry_count"] += int(retried)
        cohort_bucket["fallback_count"] += int(fallback)
        latency = _as_int(row.get("latency_ms"))
        if latency > 0:
            cohort_bucket["latency_ms_total"] += latency
            cohort_bucket["latency_sample_count"] += 1
        safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
        if safety:
            if safety.get("tripped") or status == "safety_stopped":
                latest_safety_stop = {
                    "policy_id": policy_id,
                    "created_at": created_at,
                    "status": safety.get("status") or status,
                    "tripped": bool(safety.get("tripped") or status == "safety_stopped"),
                    "sample_count": _as_int(safety.get("sample_count")),
                    "holdout_sample_count": _as_int(safety.get("holdout_sample_count")),
                    "reason_codes": safety.get("reason_codes") or [],
                }
            for reason_code in safety.get("reason_codes") or []:
                safety_stop_reason_counts[str(reason_code)] = safety_stop_reason_counts.get(str(reason_code), 0) + 1

    canary_health = [_finalize_phase_canary_bucket(bucket) for bucket in buckets.values()]
    canary_health.sort(key=lambda item: (str(item.get("last_decision_at") or ""), item.get("observed_rows") or 0), reverse=True)
    latest_canary = canary_health[0] if canary_health else {}
    outcome_queue = _managed_feedback_queue_health(store_obj, sample_limit=5, source_surface=PHASE_ROUTING_OUTCOME_SOURCE_SURFACE)
    lifecycle_rows = _phase_routing_feedback_rows(
        store_obj,
        source_surface=PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE,
        limit=500,
    )
    lifecycle_public_rows = _phase_routing_public_lifecycle_rows(lifecycle_rows, now=now)
    lifecycle_queue = _managed_feedback_queue_health(store_obj, sample_limit=5, source_surface=PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE)
    latest_lifecycle = lifecycle_public_rows[0] if lifecycle_public_rows else None
    lifecycle_summary = {
        "feedback_count": len(lifecycle_public_rows),
        "latest_dry_run_matched_count": _as_int((latest_lifecycle or {}).get("matched_count")),
        "latest_dry_run_projected_candidate_count": _as_int((latest_lifecycle or {}).get("projected_candidate_count")),
        "latest_dry_run_projected_savings_usd": round(_as_float((latest_lifecycle or {}).get("projected_savings_usd")), 6),
        "latest_dry_run_risk_warning_count": _as_int((latest_lifecycle or {}).get("risk_warning_count")),
        "pending_lifecycle_feedback_count": _as_int((lifecycle_queue.get("summary") or {}).get("queued")) + _as_int((lifecycle_queue.get("summary") or {}).get("retryable_error")),
        "due_lifecycle_feedback_count": _as_int((lifecycle_queue.get("summary") or {}).get("due")),
    }

    observed_rows = sum(_as_int(row.get("observed_rows")) for row in canary_health)
    rollout_status = "disabled"
    if phase_policy.get("enabled"):
        rollout_status = "not-deployed-yet" if observed_rows <= 0 else "observed"
        if sum(_as_int(row.get("applied_rows")) for row in canary_health) > 0:
            rollout_status = "canary-observed"
        if sum(_as_int(row.get("holdout_rows")) for row in canary_health) > 0:
            rollout_status = "canary-and-holdout-observed"
        if latest_safety_stop:
            rollout_status = "safety-stopped"

    return {
        "schema": "tokenclaw.phase_routing_dashboard.v1",
        "generated_at": utc_now(),
        "limit": capped_limit,
        "status": rollout_status,
        "summary": {
            "sampled_call_count": _as_int(opportunity.get("sampled_call_count")),
            "opportunity_candidate_count": _as_int((opportunity.get("summary") or {}).get("candidate_count")),
            "current_routed_count": _as_int((opportunity.get("summary") or {}).get("current_routed_count")),
            "projected_savings_usd": round(_as_float((opportunity.get("summary") or {}).get("projected_savings_usd")), 6),
            "canary_observed_rows": observed_rows,
            "canary_applied_rows": sum(_as_int(row.get("applied_rows")) for row in canary_health),
            "canary_holdout_rows": sum(_as_int(row.get("holdout_rows")) for row in canary_health),
            "safety_stop_rows": sum(_as_int(row.get("safety_stop_rows")) for row in canary_health),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in canary_health), 6),
            "outcome_feedback_queued": _as_int((outcome_queue.get("summary") or {}).get("queued")),
            "outcome_feedback_due": _as_int((outcome_queue.get("summary") or {}).get("due")),
            "lifecycle_feedback_count": len(lifecycle_public_rows),
            "managed_health_warning_count": len(_phase_routing_public_health_rows(limit=25)),
        },
        "policy": {
            "enabled": bool(phase_policy.get("enabled")),
            "policy_id": phase_policy.get("policy_id"),
            "policy_source": routing_policy.get("policy_source"),
            "rule_path": routing_policy.get("rule_path"),
            "reload_required": bool(((routing_policy.get("file") or {}).get("reload_required"))),
            "model_pattern": phase_policy.get("model_pattern"),
            "target_model": phase_policy.get("target_model"),
            "eligible_workflow_phases": phase_policy.get("eligible_workflow_phases") or [],
            "excluded_workflow_phases": phase_policy.get("excluded_workflow_phases") or [],
            "min_workflow_phase_confidence": phase_policy.get("min_workflow_phase_confidence"),
            "canary_fraction": _as_float(phase_policy.get("canary_fraction")),
            "holdout_fraction": _as_float(phase_policy.get("holdout_fraction")),
            "safety_stop": phase_policy.get("safety_stop") if isinstance(phase_policy.get("safety_stop"), dict) else {},
        },
        "state_flags": {
            "disabled": not bool(phase_policy.get("enabled")),
            "not_deployed_yet": bool(phase_policy.get("enabled")) and observed_rows <= 0,
            "no_observed_rows": observed_rows <= 0,
            "no_applied_canary_rows": observed_rows > 0 and sum(_as_int(row.get("applied_rows")) for row in canary_health) <= 0,
            "no_holdout_rows": observed_rows > 0 and sum(_as_int(row.get("holdout_rows")) for row in canary_health) <= 0,
            "safety_stopped": bool(latest_safety_stop),
            "read_only": True,
        },
        "opportunity": opportunity,
        "canary_health": canary_health,
        "latest_canary": latest_canary,
        "status_breakdown": _count_breakdown(status_counts),
        "reason_breakdown": _count_breakdown(reason_counts),
        "phase_breakdown": _count_breakdown(phase_counts),
        "confidence_breakdown": _count_breakdown(confidence_counts),
        "feedback_status_breakdown": _count_breakdown(feedback_status_counts),
        "feedback_reason_breakdown": _count_breakdown(feedback_reason_counts),
        "safety_stop": {
            "active": bool(latest_safety_stop),
            "latest": latest_safety_stop,
            "reason_code_breakdown": _count_breakdown(safety_stop_reason_counts),
        },
        "lifecycle": {
            "summary": lifecycle_summary,
            "latest": latest_lifecycle,
            "recent": lifecycle_public_rows[:25],
            "feedback_queue": lifecycle_queue,
        },
        "managed_feedback_queue": outcome_queue,
        "managed_recommendation_health": {
            "rows": _phase_routing_public_health_rows(limit=25),
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "raw_responses_included": False,
                "provider_bodies_included": False,
                "raw_health_details_included": False,
            },
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "error_samples_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "queue_payload_json_included": False,
            "policy_file_contents_included": False,
        },
    }

def _managed_error_class(meta: dict[str, Any]) -> str | None:
    reason = str(meta.get("reason") or "")
    if reason:
        return reason
    status = str(meta.get("status") or "")
    if status in {"error", "invalid"}:
        return status
    error = _sanitize_error_sample(meta.get("error"), limit=240)
    if not error:
        return None
    head = error.split(":", 1)[0].strip()
    return head[:80] if head else "error"

def _managed_breakdown(grouped: dict[str, int]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": value} for key, value in grouped.items()]
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows

def _day_key(raw: Any) -> str:
    text = str(raw or "")
    return text[:10] if len(text) >= 10 else "unknown"

def _managed_policy_event_sections(details: dict[str, Any], action: str) -> list[str]:
    values: list[str] = []
    for key in ("applied_sections", "restored_sections", "changed_sections", "policy_sections"):
        raw = details.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
        elif raw:
            values.append(str(raw))
    if not values and action.startswith("rollout-actions"):
        values.extend(["crunch", "cache"])
    if not values:
        values.append("unknown")
    return sorted(set(values))

def _managed_policy_event_stage(event: dict[str, Any]) -> str | None:
    action = str(event.get("action") or "")
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    ok = bool(event.get("ok"))
    if action in {"fetch-review", "review", "rollout-actions-review"}:
        if ok:
            return "reviewed"
        return "errored" if details.get("status_code") else "rejected"
    if action == "rollout-actions-dry-run":
        return "dry_run" if ok else "rejected"
    if action in {"apply", "rollout-actions-apply"}:
        if bool(details.get("dry_run")):
            return "dry_run" if ok else "rejected"
        return "applied" if ok else "rejected"
    if action == "rollback":
        return "rolled_back" if ok else "errored"
    if action == "pattern-canary-safety-stop":
        return "rolled_back"
    return None

def _managed_policy_lifecycle_rows(policy_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in policy_events:
        if not isinstance(event, dict):
            continue
        stage = _managed_policy_event_stage(event)
        if stage is None:
            continue
        action = str(event.get("action") or "unknown")
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        candidate_ids = details.get("candidate_ids") if isinstance(details.get("candidate_ids"), list) else []
        for section in _managed_policy_event_sections(details, action):
            rows.append({
                "schema": "tokenclaw.managed_pattern_lifecycle_event.v1",
                "day": _day_key(event.get("created_at")),
                "created_at": event.get("created_at"),
                "lifecycle_stage": stage,
                "action": action,
                "ok": bool(event.get("ok")),
                "policy_section": section,
                "candidate_count": _as_int(details.get("candidate_count")) or len(candidate_ids),
                "candidate_ids": [str(item) for item in candidate_ids[:5]],
                "dry_run": bool(details.get("dry_run")),
                "changed_action_count": _as_int(details.get("changed_action_count")),
                "changed_file_count": len(details.get("changed_files") or []) if isinstance(details.get("changed_files"), list) else 0,
                "error_type": details.get("error_type"),
                "status_code": details.get("status_code"),
                "raw_payload_included": False,
            })
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows

def _managed_pattern_summary_stage(summary: dict[str, Any], status_code: Any) -> str:
    status = str(summary.get("status") or "")
    outcome = str(summary.get("outcome") or "")
    reason = str(summary.get("reason") or "")
    cohort = str(summary.get("cohort") or "")
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    if _as_int(status_code) >= 400 or outcome == "errored" or status == "error":
        return "errored"
    if outcome == "holdout" or status == "holdout" or cohort == "canary_holdout" or canary.get("status") == "holdout":
        return "canary_holdout"
    if _managed_pattern_is_bypass(summary):
        return "bypassed"
    if _as_int(summary.get("applied_count")) > 0 and (canary.get("enabled") or cohort == "canary_applied"):
        return "canary_applied"
    if _as_int(summary.get("applied_count")) > 0 or outcome == "applied" or status == "applied":
        return "applied"
    if "reject" in reason:
        return "rejected"
    if "rollback" in reason:
        return "rolled_back"
    return "received"

def _managed_pattern_blocker_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reason = summary.get("reason")
    if reason and str(reason) not in {"unknown", "ok", "applied"}:
        reasons.append(str(reason))
    for item in summary.get("skip_reasons") or []:
        if isinstance(item, dict) and item.get("reason"):
            reasons.append(str(item.get("reason")))
    safety_stop = summary.get("safety_stop")
    if isinstance(safety_stop, dict) and safety_stop.get("reason"):
        reasons.append(str(safety_stop.get("reason")))
    return sorted(set(reasons))

def _managed_pattern_add_adoption_row(
    grouped: dict[tuple[str, ...], dict[str, Any]],
    *,
    summary: dict[str, Any],
    created_at: Any,
    status_code: Any,
    cost_est_usd: Any,
) -> None:
    pattern_hash = str(summary.get("pattern_hash") or "")
    if not pattern_hash.startswith("sha256:"):
        return
    policy_source = str(summary.get("policy_source") or "")
    if (
        not policy_source.startswith("managed-")
        and summary.get("candidate_id") is None
        and not isinstance(summary.get("canary"), dict)
    ):
        return
    stage = _managed_pattern_summary_stage(summary, status_code)
    key = (
        _day_key(created_at),
        stage,
        str(summary.get("decision_type") or "unknown"),
        str(summary.get("source_surface") or "unknown"),
        str(summary.get("app_family") or "unknown"),
        str(summary.get("workflow_phase") or summary.get("category") or "unknown"),
        str(summary.get("category") or "unknown"),
        policy_source or "unknown",
        str(summary.get("candidate_id") or "unknown"),
        str(summary.get("rule_id") or "unknown"),
        pattern_hash,
    )
    bucket = grouped.setdefault(
        key,
        {
            "schema": "tokenclaw.managed_pattern_adoption_bucket.v1",
            "day": key[0],
            "lifecycle_stage": key[1],
            "policy_section": key[2],
            "source_surface": key[3],
            "app_family": key[4],
            "workflow_phase": key[5],
            "category": key[6],
            "policy_source": key[7],
            "candidate_id": None if key[8] == "unknown" else key[8],
            "rule_id": None if key[9] == "unknown" else key[9],
            "pattern_hash": key[10],
            "affected_calls": 0,
            "success_count": 0,
            "error_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "bypassed_count": 0,
            "saved_chars": 0,
            "tokens_saved_est": 0,
            "estimated_cost_savings_usd": 0.0,
            "cost_est_usd": 0.0,
            "status_code_counts": {},
            "safety_blocker_reasons": {},
            "raw_payload_included": False,
        },
    )
    bucket["affected_calls"] += 1
    status = _as_int(status_code)
    if status >= 400:
        bucket["error_count"] += 1
    elif status:
        bucket["success_count"] += 1
    bucket["applied_count"] += _as_int(summary.get("applied_count"))
    if stage == "canary_holdout":
        bucket["holdout_count"] += 1
    if stage == "bypassed":
        bucket["bypassed_count"] += 1
    bucket["saved_chars"] += _as_int(summary.get("saved_chars"))
    bucket["tokens_saved_est"] += _as_int(summary.get("tokens_saved_est"))
    bucket["estimated_cost_savings_usd"] += _as_float(summary.get("estimated_cost_savings_usd"))
    bucket["cost_est_usd"] += _as_float(cost_est_usd)
    status_bucket = _status_code_bucket(status_code)
    bucket["status_code_counts"][status_bucket] = bucket["status_code_counts"].get(status_bucket, 0) + 1
    for reason in _managed_pattern_blocker_reasons(summary):
        bucket["safety_blocker_reasons"][reason] = bucket["safety_blocker_reasons"].get(reason, 0) + 1

def _managed_pattern_finalize_adoption_rows(grouped: dict[tuple[str, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        affected = _as_int(bucket.get("affected_calls"))
        errors = _as_int(bucket.get("error_count"))
        bucket["error_rate"] = round(errors / affected, 4) if affected else 0.0
        bucket["estimated_cost_savings_usd"] = round(_as_float(bucket.get("estimated_cost_savings_usd")), 8)
        bucket["cost_est_usd"] = round(_as_float(bucket.get("cost_est_usd")), 8)
        bucket["status_code_counts"] = _count_breakdown(bucket.get("status_code_counts") or {})
        bucket["safety_blocker_reasons"] = _count_breakdown(bucket.get("safety_blocker_reasons") or {})
        rows.append(bucket)
    rows.sort(
        key=lambda row: (
            row.get("day") or "",
            _as_int(row.get("affected_calls")),
            _as_float(row.get("estimated_cost_savings_usd")),
        ),
        reverse=True,
    )
    return rows

def _managed_pattern_holdout_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        stage = str(row.get("lifecycle_stage") or "")
        if stage not in {"canary_applied", "canary_holdout", "bypassed", "errored"}:
            continue
        key = (
            str(row.get("policy_section") or "unknown"),
            str(row.get("candidate_id") or "unknown"),
            str(row.get("rule_id") or "unknown"),
            str(row.get("pattern_hash") or ""),
            str(row.get("source_surface") or "unknown"),
            str(row.get("app_family") or "unknown"),
            str(row.get("workflow_phase") or row.get("category") or "unknown"),
            str(row.get("category") or "unknown"),
        )
        bucket = grouped.setdefault(
            key,
            {
                "schema": "tokenclaw.managed_pattern_holdout_comparison.v1",
                "policy_section": key[0],
                "candidate_id": None if key[1] == "unknown" else key[1],
                "rule_id": None if key[2] == "unknown" else key[2],
                "pattern_hash": key[3],
                "source_surface": key[4],
                "app_family": key[5],
                "workflow_phase": key[6],
                "category": key[7],
                "canary_applied_count": 0,
                "canary_holdout_count": 0,
                "bypassed_count": 0,
                "errored_count": 0,
                "applied_error_count": 0,
                "holdout_error_count": 0,
                "estimated_cost_savings_usd": 0.0,
                "raw_payload_included": False,
            },
        )
        affected = _as_int(row.get("affected_calls"))
        errors = _as_int(row.get("error_count"))
        if stage == "canary_applied":
            bucket["canary_applied_count"] += affected
            bucket["applied_error_count"] += errors
            bucket["estimated_cost_savings_usd"] += _as_float(row.get("estimated_cost_savings_usd"))
        elif stage == "canary_holdout":
            bucket["canary_holdout_count"] += affected
            bucket["holdout_error_count"] += errors
        elif stage == "bypassed":
            bucket["bypassed_count"] += affected
        elif stage == "errored":
            bucket["errored_count"] += affected
    comparisons: list[dict[str, Any]] = []
    for bucket in grouped.values():
        applied = _as_int(bucket.get("canary_applied_count"))
        holdout = _as_int(bucket.get("canary_holdout_count"))
        bucket["applied_error_rate"] = round(_as_int(bucket.get("applied_error_count")) / applied, 4) if applied else None
        bucket["holdout_error_rate"] = round(_as_int(bucket.get("holdout_error_count")) / holdout, 4) if holdout else None
        bucket["estimated_cost_savings_usd"] = round(_as_float(bucket.get("estimated_cost_savings_usd")), 8)
        if applied or holdout:
            comparisons.append(bucket)
    comparisons.sort(
        key=lambda row: (
            _as_int(row.get("canary_applied_count")) + _as_int(row.get("canary_holdout_count")),
            _as_float(row.get("estimated_cost_savings_usd")),
        ),
        reverse=True,
    )
    return comparisons

def _managed_pattern_adoption_from_store(
    store_obj: Any,
    *,
    limit: int,
    policy_events: list[dict[str, Any]],
    managed_summary: dict[str, Any],
) -> dict[str, Any]:
    conn = store_obj.conn
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    capped_limit = max(1, min(int(limit or 500), 5000))

    provider_rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   cost_est_usd, cost_baseline_usd, crunch_json, routing_json, cache_json, category
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]
    for row in provider_rows:
        routing = _json_obj(row.get("routing_json"))
        for summary in pattern_decision_summaries(
            provider=str(row.get("provider") or "anthropic"),
            path=str(row.get("path") or ""),
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            status_code=_as_int(row.get("status_code")) if row.get("status_code") is not None else None,
            cost_est_usd=_as_float(row.get("cost_est_usd")) if row.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(row.get("cost_baseline_usd")) if row.get("cost_baseline_usd") is not None else None,
            cache_meta=_json_obj(row.get("cache_json")),
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=row.get("category") or routing.get("category"),
        ):
            if isinstance(summary, dict):
                _managed_pattern_add_adoption_row(
                    grouped,
                    summary=summary,
                    created_at=row.get("created_at"),
                    status_code=row.get("status_code"),
                    cost_est_usd=row.get("cost_est_usd"),
                )

    codex_rows = [
        dict(row)
        for row in conn.execute(
            """
            select s.id as start_event_id,
                   s.created_at,
                   s.request_id,
                   s.session_id,
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
            """,
            (capped_limit,),
        ).fetchall()
    ]
    for row in codex_rows:
        routing = _json_obj(row.get("routing_json"))
        cache_meta = _json_obj(row.get("cache_json"))
        estimates = _codex_estimates_with_cache(row.get("input_text_chars"), row.get("response_result_chars"), cache_meta)
        status_code = 500 if row.get("response_error_code") is not None else (200 if row.get("response_event_id") else None)
        for summary in pattern_decision_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing.get("requested_model") or CODEX_APP_MODEL,
            routed_model=routing.get("routed_model") or routing.get("requested_model") or CODEX_APP_MODEL,
            status_code=status_code,
            cost_est_usd=_as_float(estimates.get("cost_est_usd")) if estimates.get("cost_est_usd") is not None else None,
            cost_baseline_usd=_as_float(estimates.get("baseline_cost_est_usd")) if estimates.get("baseline_cost_est_usd") is not None else None,
            cache_meta=cache_meta,
            crunch_meta=_json_obj(row.get("crunch_json")),
            routing_meta=routing,
            category=routing.get("category") or "codex_turn",
        ):
            if not isinstance(summary, dict):
                continue
            summary = dict(summary)
            summary["source_surface"] = CODEX_APP_SOURCE_SURFACE
            summary["app_family"] = "codex"
            summary["category"] = routing.get("category") or summary.get("category") or "codex_turn"
            summary["workflow_phase"] = routing.get("workflow_phase") or summary.get("workflow_phase") or summary["category"]
            _managed_pattern_add_adoption_row(
                grouped,
                summary=summary,
                created_at=row.get("created_at"),
                status_code=status_code,
                cost_est_usd=estimates.get("cost_est_usd"),
            )

    outcome_rows = _managed_pattern_finalize_adoption_rows(grouped)
    lifecycle_rows = _managed_policy_lifecycle_rows(policy_events)
    funnel_counts = {stage: 0 for stage in MANAGED_PATTERN_ADOPTION_STAGES}
    funnel_counts["received"] += _as_int(managed_summary.get("received_count"))
    funnel_counts["applied"] += _as_int(managed_summary.get("applied_count"))
    funnel_counts["errored"] += _as_int(managed_summary.get("server_error_count")) + _as_int(managed_summary.get("invalid_count"))
    for row in outcome_rows:
        stage = str(row.get("lifecycle_stage") or "")
        if stage in funnel_counts:
            funnel_counts[stage] += _as_int(row.get("affected_calls"))
    for row in lifecycle_rows:
        stage = str(row.get("lifecycle_stage") or "")
        if stage in funnel_counts:
            funnel_counts[stage] += 1

    blocker_counts: dict[str, int] = {}
    for row in outcome_rows:
        for item in row.get("safety_blocker_reasons") or []:
            blocker_counts[str(item.get("value") or "unknown")] = blocker_counts.get(str(item.get("value") or "unknown"), 0) + _as_int(item.get("count"))

    return {
        "schema": "tokenclaw.managed_pattern_adoption.v1",
        "summary": {
            "lifecycle_event_count": len(lifecycle_rows),
            "pattern_outcome_bucket_count": len(outcome_rows),
            "holdout_comparison_count": len(_managed_pattern_holdout_comparisons(outcome_rows)),
            "affected_calls": sum(_as_int(row.get("affected_calls")) for row in outcome_rows),
            "estimated_cost_savings_usd": round(sum(_as_float(row.get("estimated_cost_savings_usd")) for row in outcome_rows), 8),
            "error_count": sum(_as_int(row.get("error_count")) for row in outcome_rows),
            "raw_payload_included": False,
        },
        "funnel": [
            {"stage": stage, "count": funnel_counts.get(stage, 0)}
            for stage in MANAGED_PATTERN_ADOPTION_STAGES
        ],
        "pattern_outcomes_by_day": outcome_rows[:100],
        "holdout_comparisons": _managed_pattern_holdout_comparisons(outcome_rows)[:50],
        "lifecycle_events": lifecycle_rows[:50],
        "top_safety_blockers": _managed_breakdown(blocker_counts)[:20],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_contents_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "request_ids_included": False,
            "basis": "local policy events plus stored pattern decision metadata, hashes, status codes, latency, cost, and size-derived savings only",
        },
    }

async def stats_managed_recommendations(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.recommendations import (
        managed_auth_configured,
        policy_decision_canary_fraction,
        policy_decision_min_confidence,
        policy_decisions_enabled,
        recommendation_failure_mode,
        recommendation_server_configured,
        recommendation_server_url,
        recommendation_timeout_seconds,
        recommendations_enabled,
    )
    from tokenclaw.policy_events import recent_policy_events

    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 500), 5000))
    rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   requested_model, routed_model, status_code, latency_ms,
                   cost_est_usd, cost_baseline_usd, routing_json, managed_routing_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    feedback_status_counts: dict[str, int] = {}
    feedback_reason_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    latency_values: list[int] = []
    feedback_latency_values: list[int] = []
    confidence_values_24h: list[float] = []
    recent: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    metadata_rows = 0
    historical_null_rows = 0
    enabled_count = 0
    disabled_count = 0
    received_count = 0
    applied_count = 0
    changed_model_count = 0
    server_error_count = 0
    invalid_count = 0
    feedback_present_count = 0
    feedback_sent_count = 0
    feedback_skipped_count = 0
    feedback_failed_count = 0
    feedback_sanitized_count = 0
    observed_savings_usd = 0.0
    applied_observed_savings_usd = 0.0
    changed_model_observed_savings_usd = 0.0
    positive_savings_count = 0
    requests_24h = 0
    applied_24h = 0
    last_recommendation_error_class = None
    last_feedback_error_class = None

    for row in rows:
        row_created_at = _parse_utc_datetime(row.get("created_at"))
        is_recent_24h = row_created_at is not None and (now - row_created_at) <= timedelta(hours=24)
        routing = _json_obj(row.get("routing_json"))
        managed = _json_obj(row.get("managed_routing_json"))
        if not isinstance(managed, dict) or not managed:
            managed = routing.get("managed_recommendation")
        if not isinstance(managed, dict) or not managed:
            historical_null_rows += 1
            if len(recent) < 50:
                recent.append({
                    "created_at": row.get("created_at"),
                    "provider": row.get("provider"),
                    "source_surface": _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or "")),
                    "requested_model": row.get("requested_model"),
                    "routed_model": row.get("routed_model"),
                    "status_code": row.get("status_code"),
                    "recommendation_status": "missing",
                    "recommendation_reason": "historical-null",
                    "recommendation_enabled": None,
                    "applied": False,
                    "changed_model": False,
                    "policy_id": None,
                    "target_model": None,
                    "fallback": None,
                    "observed_savings_usd": 0.0,
                    "latency_ms": None,
                    "feedback_status": "missing",
                    "feedback_reason": "historical-null",
                    "feedback_latency_ms": None,
                    "feedback_error_class": None,
                })
            continue

        metadata_rows += 1
        if is_recent_24h:
            requests_24h += 1
        status = str(managed.get("status") or "missing")
        reason = str(managed.get("reason") or "unknown")
        _increment_count(status_counts, status)
        _increment_count(reason_counts, reason)
        if managed.get("enabled") is False:
            disabled_count += 1
        elif managed.get("enabled") is True:
            enabled_count += 1
        if status == "received":
            received_count += 1
        managed_applied = bool(managed.get("applied"))
        if managed_applied:
            applied_count += 1
            if is_recent_24h:
                applied_24h += 1
        if bool(managed.get("changed_model")):
            changed_model_count += 1
        row_savings = max(_as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd")), 0.0)
        if _as_int(row.get("status_code")) >= 400:
            row_savings = 0.0
        managed_changed_model = bool(managed.get("changed_model"))
        attributed_to_managed = bool(managed_applied and managed_changed_model)
        if attributed_to_managed and row_savings > 0:
            positive_savings_count += 1
        if attributed_to_managed:
            applied_observed_savings_usd += row_savings
            observed_savings_usd += row_savings
        if managed_changed_model:
            changed_model_observed_savings_usd += row_savings
        if status == "error" or reason == "server-error":
            server_error_count += 1
            if last_recommendation_error_class is None:
                last_recommendation_error_class = _managed_error_class(managed)
        if status == "invalid":
            invalid_count += 1
            if last_recommendation_error_class is None:
                last_recommendation_error_class = _managed_error_class(managed)
        fallback = managed.get("fallback")
        if fallback:
            _increment_count(fallback_counts, fallback)
        policy_id = managed.get("policy_id")
        if policy_id:
            _increment_count(policy_counts, policy_id)
        latency_ms = managed.get("latency_ms")
        if latency_ms is not None:
            latency_values.append(_as_int(latency_ms))
        if is_recent_24h:
            try:
                confidence_values_24h.append(float(managed.get("confidence")))
            except (TypeError, ValueError):
                pass

        feedback = managed.get("outcome_feedback")
        feedback_status = "missing"
        feedback_reason = "missing"
        feedback_error_class = None
        feedback_latency_ms = None
        if isinstance(feedback, dict) and feedback:
            feedback_present_count += 1
            feedback_sanitized_count += 1
            feedback_status = str(feedback.get("status") or "missing")
            feedback_reason = str(feedback.get("reason") or "unknown")
            _increment_count(feedback_status_counts, feedback_status)
            _increment_count(feedback_reason_counts, feedback_reason)
            if feedback_status == "sent":
                feedback_sent_count += 1
            elif feedback_status == "skipped":
                feedback_skipped_count += 1
            elif feedback_status in {"error", "invalid"}:
                feedback_failed_count += 1
                feedback_error_class = _managed_error_class(feedback)
                if last_feedback_error_class is None:
                    last_feedback_error_class = feedback_error_class
            feedback_latency_ms = feedback.get("latency_ms")
            if feedback_latency_ms is not None:
                feedback_latency_values.append(_as_int(feedback_latency_ms))
        else:
            _increment_count(feedback_status_counts, "missing")

        if len(recent) < 50:
            recent.append({
                "created_at": row.get("created_at"),
                "provider": row.get("provider"),
                "source_surface": _source_surface(str(row.get("provider") or "anthropic"), str(row.get("path") or "")),
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "status_code": row.get("status_code"),
                "recommendation_status": status,
                "recommendation_reason": reason,
                "recommendation_enabled": managed.get("enabled"),
                "applied": bool(managed.get("applied")),
                "changed_model": bool(managed.get("changed_model")),
                "policy_id": managed.get("policy_id"),
                "target_model": managed.get("target_model"),
                "fallback": fallback,
                "observed_savings_usd": round(row_savings, 8) if attributed_to_managed else 0.0,
                "observed_savings_attributed_to_managed": attributed_to_managed,
                "latency_ms": _as_int(latency_ms) if latency_ms is not None else None,
                "feedback_status": feedback_status,
                "feedback_reason": feedback_reason,
                "feedback_latency_ms": _as_int(feedback_latency_ms) if feedback_latency_ms is not None else None,
                "feedback_error_class": feedback_error_class,
            })

    policy_events = recent_policy_events(limit=500).get("events", [])
    latest_fetch_review_health: dict[str, Any] | None = None
    for event in policy_events:
        if not isinstance(event, dict) or event.get("action") != "fetch-review":
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        health = details.get("recommendation_health")
        if isinstance(health, dict) and health:
            latest_fetch_review_health = {
                **health,
                "event_created_at": event.get("created_at"),
                "event_ok": bool(event.get("ok")),
            }
            break

    summary_payload = {
        "window_calls": len(rows),
        "metadata_rows": metadata_rows,
        "historical_null_rows": historical_null_rows,
        "enabled_count": enabled_count,
        "disabled_count": disabled_count,
        "received_count": received_count,
        "applied_count": applied_count,
        "changed_model_count": changed_model_count,
        "server_error_count": server_error_count,
        "invalid_count": invalid_count,
        "fallback_count": sum(fallback_counts.values()),
        "avg_recommendation_latency_ms": _avg_or_none(latency_values),
        "feedback_present_count": feedback_present_count,
        "feedback_sent_count": feedback_sent_count,
        "feedback_skipped_count": feedback_skipped_count,
        "feedback_failed_count": feedback_failed_count,
        "feedback_sanitized_count": feedback_sanitized_count,
        "observed_savings_usd": round(observed_savings_usd, 8),
        "applied_observed_savings_usd": round(applied_observed_savings_usd, 8),
        "changed_model_observed_savings_usd": round(changed_model_observed_savings_usd, 8),
        "positive_savings_count": positive_savings_count,
        "requests_24h": requests_24h,
        "applied_24h": applied_24h,
        "avg_confidence_24h": (
            round(sum(confidence_values_24h) / len(confidence_values_24h), 6)
            if confidence_values_24h
            else None
        ),
        "observed_savings_basis": "calls.cost_baseline_usd-minus-cost_est_usd",
        "observed_savings_attribution": "managed-recommendation-model-change",
        "avg_feedback_latency_ms": _avg_or_none(feedback_latency_values),
        "last_recommendation_error_class": last_recommendation_error_class,
        "last_feedback_error_class": last_feedback_error_class,
        "policy_id_count": len(policy_counts),
    }

    return {
        "schema": "tokenclaw.managed_recommendations.v1",
        "generated_at": utc_now(),
        "limit": capped_limit,
        "current_config": {
            "enabled": recommendations_enabled(),
            "policy_decisions_enabled": policy_decisions_enabled(),
            "mode": "managed-recommendation-bridge" if recommendations_enabled() else "local-only",
            "server_url": recommendation_server_url(),
            "server_configured": recommendation_server_configured(),
            "timeout_seconds": recommendation_timeout_seconds(),
            "min_confidence": policy_decision_min_confidence(),
            "canary_fraction": policy_decision_canary_fraction(),
            "failure_mode": recommendation_failure_mode(),
            "auth_configured": managed_auth_configured(),
            "api_key_value_included": False,
            "offline_state": (
                "managed recommendations disabled; local policy remains authoritative"
                if not recommendations_enabled()
                else "managed recommendation bridge enabled"
            ),
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "basis": "stored recommendation and outcome-feedback metadata only",
        },
        "summary": summary_payload,
        "adoption": _managed_pattern_adoption_from_store(
            store_obj,
            limit=capped_limit,
            policy_events=policy_events,
            managed_summary=summary_payload,
        ),
        "status_breakdown": _managed_breakdown(status_counts),
        "reason_breakdown": _managed_breakdown(reason_counts),
        "fallback_breakdown": _managed_breakdown(fallback_counts),
        "policy_ids": _managed_breakdown(policy_counts),
        "feedback_status_breakdown": _managed_breakdown(feedback_status_counts),
        "feedback_reason_breakdown": _managed_breakdown(feedback_reason_counts),
        "recommendation_health": {
            "latest_fetch_review": latest_fetch_review_health,
            "rows": (
                latest_fetch_review_health.get("rows", [])
                if isinstance(latest_fetch_review_health, dict)
                else []
            ),
        },
        "recent": recent,
    }

def _openai_recommendation_state(managed: dict[str, Any]) -> tuple[str, str]:
    if not managed:
        return "skipped", "no-recommendation-metadata"
    mode = str(managed.get("mode") or "").strip().lower()
    status = str(managed.get("status") or "").strip().lower()
    lifecycle = str(managed.get("lifecycle_event") or "").strip().lower()
    apply_reason = str(managed.get("apply_reason") or managed.get("reason") or "").strip()
    if mode == "observe-only":
        return "observed-only", apply_reason or "observe-only-local-policy"
    if lifecycle == "dry_run" or status == "dry-run":
        return "dry-run", apply_reason or "dry-run-local-fallback"
    if lifecycle == "canary_applied" or status == "applied" or managed.get("applied") is True:
        return "canary-applied", apply_reason or "canary-selected"
    if lifecycle == "holdout" or status == "holdout":
        return "holdout", apply_reason or "canary-holdout"
    if lifecycle == "fallback" or managed.get("fallback"):
        return "fallback", apply_reason or str(managed.get("fallback") or "local-policy")
    if status in {"skipped", "noop", "missing"}:
        return "skipped", apply_reason or status
    return status or "skipped", apply_reason or "unknown"

def _openai_candidate_id(managed: dict[str, Any], requested_model: Any, routed_model: Any) -> str:
    for key in ("policy_id", "recommendation_id", "candidate_id"):
        value = managed.get(key)
        if value:
            return str(value)
    target = managed.get("target_model") or managed.get("would_route_model")
    if target:
        return f"target:{target}"
    if requested_model or routed_model:
        return f"local:{requested_model or 'unknown'}->{routed_model or requested_model or 'unknown'}"
    return "uncategorized"

def _openai_quality_gate_outcome(managed: dict[str, Any], state: str) -> str:
    if not managed:
        return "not-evaluated"
    if state == "observed-only":
        return "observe-only"
    reason = str(managed.get("apply_reason") or managed.get("reason") or "")
    if reason in {"provider-mismatch", "unsupported-openai-target-model", "missing-target-model"}:
        return "failed-provider-target"
    if reason == "prompt-shaping-not-locally-representable":
        return "failed-local-representability"
    projection = managed.get("projection") if isinstance(managed.get("projection"), dict) else {}
    risk = projection.get("risk") if isinstance(projection.get("risk"), dict) else {}
    if risk.get("missing_fields"):
        return "missing-risk-metadata"
    if _as_float(risk.get("error_rate")) > 0.05:
        return "failed-error-rate"
    if _as_float(risk.get("retry_rate")) > 0.10:
        return "failed-retry-rate"
    if _as_float(risk.get("fallback_rate")) > 0.10:
        return "failed-fallback-rate"
    latency_regression = _as_float(risk.get("latency_regression_ratio"))
    if latency_regression and latency_regression > 1.25:
        return "failed-latency-regression"
    if state in {"dry-run", "holdout", "canary-applied"}:
        return "passed-local-gates"
    if state == "fallback":
        return "fallback-local-policy"
    return "not-evaluated"

def _new_openai_scoreboard_bucket(candidate_id: str, source_surface: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_surface": source_surface,
        "calls": 0,
        "state_counts": {},
        "quality_gate_counts": {},
        "requested_model_counts": {},
        "target_model_counts": {},
        "actual_cost_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "observed_cost_savings_usd": 0.0,
        "projected_cost_savings_usd": 0.0,
        "tokens_saved_est": 0,
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "cache_hit_count": 0,
        "latency_values": [],
        "canary_applied_latency_values": [],
        "holdout_latency_values": [],
    }

def _finalize_openai_scoreboard_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    calls = _as_int(bucket.get("calls"))
    applied_latency = bucket.pop("canary_applied_latency_values", [])
    holdout_latency = bucket.pop("holdout_latency_values", [])
    latency_values = bucket.pop("latency_values", [])
    applied_avg = _avg_or_none(applied_latency)
    holdout_avg = _avg_or_none(holdout_latency)
    bucket["actual_cost_usd"] = round(_as_float(bucket.get("actual_cost_usd")), 6)
    bucket["baseline_cost_usd"] = round(_as_float(bucket.get("baseline_cost_usd")), 6)
    bucket["observed_cost_savings_usd"] = round(_as_float(bucket.get("observed_cost_savings_usd")), 6)
    bucket["projected_cost_savings_usd"] = round(_as_float(bucket.get("projected_cost_savings_usd")), 6)
    bucket["error_rate"] = round(_as_int(bucket.get("error_count")) / calls, 4) if calls else 0
    bucket["retry_rate"] = round(_as_int(bucket.get("retry_count")) / calls, 4) if calls else 0
    bucket["fallback_rate"] = round(_as_int(bucket.get("fallback_count")) / calls, 4) if calls else 0
    bucket["cache_hit_rate"] = round(_as_int(bucket.get("cache_hit_count")) / calls, 4) if calls else 0
    bucket["avg_latency_ms"] = _avg_or_none(latency_values)
    bucket["canary_applied_avg_latency_ms"] = applied_avg
    bucket["holdout_avg_latency_ms"] = holdout_avg
    bucket["observed_latency_delta_ms"] = (
        applied_avg - holdout_avg if applied_avg is not None and holdout_avg is not None else None
    )
    bucket["state_breakdown"] = _managed_breakdown(bucket.pop("state_counts", {}))
    bucket["quality_gate_breakdown"] = _managed_breakdown(bucket.pop("quality_gate_counts", {}))
    bucket["requested_models"] = _managed_breakdown(bucket.pop("requested_model_counts", {}))
    bucket["target_models"] = _managed_breakdown(bucket.pop("target_model_counts", {}))
    return bucket

def _openai_governor_from_metadata(*metas: dict[str, Any]) -> dict[str, Any]:
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        governor = meta.get("openai_optimization_governor")
        if isinstance(governor, dict) and governor.get("schema") == OPENAI_GOVERNOR_SCHEMA:
            return governor
    return {}

def _openai_dashboard_dimension(value: Any, *, default: str = "unknown") -> str:
    if value in (None, ""):
        return default
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    unsafe_terms = {
        "api_key",
        "authorization",
        "bearer",
        "body",
        "cache_key",
        "content",
        "file_path",
        "payload",
        "prompt",
        "raw",
        "request_id",
        "secret",
        "session_id",
        "tenant",
    }
    if (
        len(text) > 96
        or any(char.isspace() for char in text)
        or any(char in text for char in ("/", "\\", "{", "}", "[", "]", "\"", "'"))
        or any(term in lowered for term in unsafe_terms)
    ):
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text

def _openai_dashboard_reason_codes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: set[str] = set()
    for value in values:
        code = _openai_dashboard_dimension(value, default="")
        if code:
            result.add(code)
    return sorted(result)

def _openai_governor_endpoint(row: dict[str, Any], governor: dict[str, Any]) -> str:
    endpoint = row.get("endpoint") or governor.get("endpoint")
    if endpoint:
        return _openai_dashboard_dimension(endpoint)
    path = str(row.get("path") or "")
    if "chat/completions" in path:
        return "chat"
    return "responses"

async def stats_openai_optimization_readiness(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, category,
                   routing_json, crunch_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    openai_rows = 0
    governor_rows = 0
    conflict_count = 0
    selected_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    source_surface_counts: dict[str, int] = {}
    requested_model_family_counts: dict[str, int] = {}
    routed_model_family_counts: dict[str, int] = {}
    suppression_reason_counts: dict[str, int] = {}
    safety_reason_counts: dict[str, int] = {}
    family_counts: dict[str, dict[str, Any]] = {
        family: {
            "family": family,
            "eligible_count": 0,
            "selected_count": 0,
            "suppressed_count": 0,
            "holdout_count": 0,
            "safety_stop_count": 0,
            "not_eligible_count": 0,
            "status_counts": {},
            "policy_source_counts": {},
            "suppression_reason_counts": {},
        }
        for family in OPENAI_GOVERNOR_FAMILIES
    }
    recent_conflicts: list[dict[str, Any]] = []

    for row in rows:
        provider = str(row.get("provider") or "anthropic").lower()
        if provider != "openai":
            continue
        openai_rows += 1
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        governor = _openai_governor_from_metadata(routing, crunch, cache)
        if not governor:
            continue

        governor_rows += 1
        selected = _openai_dashboard_dimension(governor.get("selected_action_family") or "none")
        canary = governor.get("canary") if isinstance(governor.get("canary"), dict) else {}
        cohort = _openai_dashboard_dimension(canary.get("cohort") or "unknown")
        endpoint = _openai_governor_endpoint(row, governor)
        category = _openai_dashboard_dimension(row.get("category") or routing.get("category"))
        source_surface = _openai_dashboard_dimension(
            row.get("source_surface") or _source_surface("openai", str(row.get("path") or ""))
        )
        requested_family = _openai_dashboard_dimension(
            row.get("requested_model_family") or row.get("requested_model")
        )
        routed_family = _openai_dashboard_dimension(row.get("routed_model_family") or row.get("routed_model"))
        eligible = [
            _openai_dashboard_dimension(family)
            for family in governor.get("eligible_action_families") or []
            if family
        ]
        suppressed_items = [
            item
            for item in (governor.get("suppressed_families") or [])
            if isinstance(item, dict)
        ]
        conflict_items = [
            item
            for item in suppressed_items
            if "conflicts-with-selected-family" in _openai_dashboard_reason_codes(item.get("reason_codes"))
        ]

        _increment_count(selected_counts, selected)
        _increment_count(cohort_counts, cohort)
        _increment_count(endpoint_counts, endpoint)
        _increment_count(category_counts, category)
        _increment_count(source_surface_counts, source_surface)
        _increment_count(requested_model_family_counts, requested_family)
        _increment_count(routed_model_family_counts, routed_family)

        family_status = governor.get("family_status") if isinstance(governor.get("family_status"), dict) else {}
        suppressed_by_family: dict[str, list[str]] = {}
        for item in suppressed_items:
            family = _openai_dashboard_dimension(item.get("family"))
            reasons = _openai_dashboard_reason_codes(item.get("reason_codes"))
            if reasons:
                suppressed_by_family.setdefault(family, []).extend(reasons)
            for reason in reasons:
                _increment_count(suppression_reason_counts, reason)

        for family in OPENAI_GOVERNOR_FAMILIES:
            status = family_status.get(family) if isinstance(family_status.get(family), dict) else {}
            row_counts = family_counts[family]
            family_selected = bool(status.get("selected")) or selected == family
            family_eligible = bool(status.get("eligible"))
            family_suppressed_reasons = sorted(set(suppressed_by_family.get(family, [])))
            status_value = _openai_dashboard_dimension(status.get("status") or "unknown")
            policy_source = _openai_dashboard_dimension(status.get("policy_source") or governor.get("policy_source"))
            if family_eligible:
                row_counts["eligible_count"] += 1
            else:
                row_counts["not_eligible_count"] += 1
            if family_selected:
                row_counts["selected_count"] += 1
            if family_suppressed_reasons:
                row_counts["suppressed_count"] += 1
            if cohort in {"governor_holdout", "canary_holdout", "holdout"} or "missing-holdout" in family_suppressed_reasons:
                row_counts["holdout_count"] += 1
            if (
                "safety" in status_value
                or "stale-evidence" in family_suppressed_reasons
                or any(reason.startswith("safety") for reason in family_suppressed_reasons)
            ):
                row_counts["safety_stop_count"] += 1
                for reason in family_suppressed_reasons or [status_value]:
                    _increment_count(safety_reason_counts, reason)
            _increment_count(row_counts["status_counts"], status_value)
            _increment_count(row_counts["policy_source_counts"], policy_source)
            for reason in family_suppressed_reasons:
                _increment_count(row_counts["suppression_reason_counts"], reason)

        if len(eligible) > 1 and selected != "none" and conflict_items:
            conflict_count += 1
            if len(recent_conflicts) < 25:
                recent_conflicts.append({
                    "created_at": row.get("created_at"),
                    "endpoint": endpoint,
                    "source_surface": source_surface,
                    "category": category,
                    "requested_model_family": requested_family,
                    "routed_model_family": routed_family,
                    "governor_cohort": cohort,
                    "selected_action_family": selected,
                    "eligible_action_families": eligible,
                    "suppressed_families": [
                        {
                            "family": _openai_dashboard_dimension(item.get("family")),
                            "status": _openai_dashboard_dimension(item.get("status")),
                            "reason_codes": _openai_dashboard_reason_codes(item.get("reason_codes")),
                        }
                        for item in conflict_items
                    ],
                })

    family_rows: list[dict[str, Any]] = []
    for family in OPENAI_GOVERNOR_FAMILIES:
        row = dict(family_counts[family])
        row["status_breakdown"] = _managed_breakdown(row.pop("status_counts", {}))
        row["policy_source_breakdown"] = _managed_breakdown(row.pop("policy_source_counts", {}))
        row["suppression_reason_breakdown"] = _managed_breakdown(row.pop("suppression_reason_counts", {}))
        family_rows.append(row)

    state = "no-openai-traffic"
    reason = "no OpenAI calls were found in this local report window"
    if openai_rows and not governor_rows:
        state = "missing-governor-metadata"
        reason = "OpenAI calls exist but have no unified optimization governor metadata yet"
    elif conflict_count:
        state = "conflicts-observed"
        reason = "multiple OpenAI optimization families were eligible and the governor suppressed conflicts"
    elif selected_counts.get("none", 0) < governor_rows:
        state = "single-family-active"
        reason = "the governor selected one OpenAI optimization family for at least one call"
    elif cohort_counts.get("governor_holdout", 0):
        state = "holdout"
        reason = "governor holdout evidence is present but no family was selected"
    elif governor_rows:
        state = "collecting-evidence"
        reason = "governor metadata exists but no family has been selected in this window"

    return {
        "schema": "tokenclaw.openai_optimization_readiness.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "state": state,
        "state_reason": reason,
        "summary": {
            "openai_call_count": openai_rows,
            "governor_metadata_row_count": governor_rows,
            "missing_governor_metadata_count": max(0, openai_rows - governor_rows),
            "conflicting_call_count": conflict_count,
            "selected_call_count": max(0, governor_rows - selected_counts.get("none", 0)),
            "suppressed_family_count": sum(_as_int(row.get("suppressed_count")) for row in family_rows),
            "holdout_family_count": sum(_as_int(row.get("holdout_count")) for row in family_rows),
            "safety_stop_family_count": sum(_as_int(row.get("safety_stop_count")) for row in family_rows),
        },
        "selected_family_breakdown": _managed_breakdown(selected_counts),
        "governor_cohort_breakdown": _managed_breakdown(cohort_counts),
        "suppression_reason_breakdown": _managed_breakdown(suppression_reason_counts),
        "safety_stop_reason_breakdown": _managed_breakdown(safety_reason_counts),
        "endpoint_breakdown": _managed_breakdown(endpoint_counts),
        "category_breakdown": _managed_breakdown(category_counts),
        "source_surface_breakdown": _managed_breakdown(source_surface_counts),
        "requested_model_family_breakdown": _managed_breakdown(requested_model_family_counts),
        "routed_model_family_breakdown": _managed_breakdown(routed_model_family_counts),
        "families": family_rows,
        "recent_conflicts": recent_conflicts,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": False,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "tenant_ids_included": False,
            "api_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "sanitized openai_optimization_governor metadata plus coarse local call dimensions only",
        },
    }

async def stats_openai_scoreboard(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    conn = store_obj.conn
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, requested_model, routed_model, status_code, latency_ms,
                   input_tokens_est, output_tokens_est, actual_input_tokens, actual_output_tokens,
                   cost_est_usd, cost_baseline_usd, cache_hit, retry_count,
                   routing_json, crunch_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    state_counts: dict[str, int] = {}
    quality_gate_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    source_surface_counts: dict[str, int] = {}
    candidate_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    recent: list[dict[str, Any]] = []
    latency_values: list[int] = []
    applied_latency_values: list[int] = []
    holdout_latency_values: list[int] = []

    openai_rows = 0
    anthropic_rows = 0
    error_count = 0
    retry_count = 0
    fallback_count = 0
    cache_hit_count = 0
    actual_cost = 0.0
    baseline_cost = 0.0
    observed_savings = 0.0
    projected_savings = 0.0
    tokens_saved = 0
    input_tokens = 0
    output_tokens = 0

    for row in rows:
        provider = str(row.get("provider") or "anthropic").lower()
        if provider == "anthropic":
            anthropic_rows += 1
            continue
        if provider != "openai":
            continue

        openai_rows += 1
        source_surface = str(row.get("source_surface") or _source_surface(provider, str(row.get("path") or "")))
        _increment_count(source_surface_counts, source_surface)
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        managed = routing.get("managed_recommendation") if isinstance(routing.get("managed_recommendation"), dict) else {}
        state, state_reason = _openai_recommendation_state(managed)
        quality_gate = _openai_quality_gate_outcome(managed, state)
        candidate_id = _openai_candidate_id(managed, row.get("requested_model"), row.get("routed_model"))
        bucket = candidate_buckets.setdefault(
            (candidate_id, source_surface),
            _new_openai_scoreboard_bucket(candidate_id, source_surface),
        )

        row_actual_cost = _as_float(row.get("cost_est_usd"))
        row_baseline_cost = _as_float(row.get("cost_baseline_usd"))
        row_observed_savings = max(0.0, row_baseline_cost - row_actual_cost)
        projection = managed.get("projection") if isinstance(managed.get("projection"), dict) else {}
        row_projected_savings = max(0.0, _as_float(projection.get("projected_input_savings_usd")))
        row_tokens_saved = _as_int(crunch.get("tokens_saved_est"))
        status_code = _as_int(row.get("status_code"))
        errored = status_code >= 400
        retried = _as_int(row.get("retry_count")) > 0
        fallback = state == "fallback"
        cache_status = str(cache.get("status") or ("hit" if _as_int(row.get("cache_hit")) else "missing"))
        cache_hit = bool(_as_int(row.get("cache_hit")) or cache_status == "hit")
        latency = _as_int(row.get("latency_ms"))

        _increment_count(state_counts, state)
        _increment_count(quality_gate_counts, quality_gate)
        _increment_count(cache_status_counts, cache_status)
        if fallback:
            fallback_count += 1
            _increment_count(fallback_reason_counts, state_reason)
        if errored:
            error_count += 1
        if retried:
            retry_count += 1
        if cache_hit:
            cache_hit_count += 1
        if latency > 0:
            latency_values.append(latency)
            if state == "canary-applied":
                applied_latency_values.append(latency)
            elif state == "holdout":
                holdout_latency_values.append(latency)

        actual_cost += row_actual_cost
        baseline_cost += row_baseline_cost
        observed_savings += row_observed_savings
        projected_savings += row_projected_savings
        tokens_saved += row_tokens_saved
        input_tokens += _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
        output_tokens += _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))

        bucket["calls"] += 1
        _increment_count(bucket["state_counts"], state)
        _increment_count(bucket["quality_gate_counts"], quality_gate)
        _increment_count(bucket["requested_model_counts"], row.get("requested_model") or "unknown")
        _increment_count(
            bucket["target_model_counts"],
            managed.get("target_model") or managed.get("would_route_model") or row.get("routed_model") or "unknown",
        )
        bucket["actual_cost_usd"] += row_actual_cost
        bucket["baseline_cost_usd"] += row_baseline_cost
        bucket["observed_cost_savings_usd"] += row_observed_savings
        bucket["projected_cost_savings_usd"] += row_projected_savings
        bucket["tokens_saved_est"] += row_tokens_saved
        bucket["error_count"] += int(errored)
        bucket["retry_count"] += int(retried)
        bucket["fallback_count"] += int(fallback)
        bucket["cache_hit_count"] += int(cache_hit)
        if latency > 0:
            bucket["latency_values"].append(latency)
            if state == "canary-applied":
                bucket["canary_applied_latency_values"].append(latency)
            elif state == "holdout":
                bucket["holdout_latency_values"].append(latency)

        if len(recent) < 50:
            recent.append({
                "created_at": row.get("created_at"),
                "source_surface": source_surface,
                "requested_model": row.get("requested_model"),
                "routed_model": row.get("routed_model"),
                "status_code": status_code,
                "latency_ms": latency or None,
                "state": state,
                "state_reason": state_reason,
                "candidate_id": candidate_id,
                "quality_gate_outcome": quality_gate,
                "observed_cost_savings_usd": round(row_observed_savings, 6),
                "projected_cost_savings_usd": round(row_projected_savings, 6),
                "tokens_saved_est": row_tokens_saved,
                "cache_status": cache_status,
                "retry_count": _as_int(row.get("retry_count")),
                "raw_payload_included": False,
            })

    applied_avg = _avg_or_none(applied_latency_values)
    holdout_avg = _avg_or_none(holdout_latency_values)
    verdict = "no-openai-traffic"
    if openai_rows:
        if error_count:
            verdict = "watch-errors"
        elif fallback_count and fallback_count == openai_rows:
            verdict = "not-helping-all-fallback"
        elif observed_savings > 0 or projected_savings > 0:
            verdict = "helping"
        else:
            verdict = "insufficient-evidence"

    candidate_rows = [_finalize_openai_scoreboard_bucket(bucket) for bucket in candidate_buckets.values()]
    candidate_rows.sort(
        key=lambda item: (
            _as_float(item.get("observed_cost_savings_usd")) + _as_float(item.get("projected_cost_savings_usd")),
            _as_int(item.get("calls")),
        ),
        reverse=True,
    )

    return {
        "schema": "tokenclaw.openai_optimization_scoreboard.v1",
        "generated_at": utc_now(),
        "limit": capped_limit,
        "question": "Are OpenAI optimizations helping?",
        "answer": verdict,
        "summary": {
            "openai_call_count": openai_rows,
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "observed_cost_savings_usd": round(observed_savings, 6),
            "projected_cost_savings_usd": round(projected_savings, 6),
            "tokens_saved_est": tokens_saved,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error_count": error_count,
            "error_rate": round(error_count / openai_rows, 4) if openai_rows else 0,
            "retry_count": retry_count,
            "retry_rate": round(retry_count / openai_rows, 4) if openai_rows else 0,
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_count / openai_rows, 4) if openai_rows else 0,
            "cache_hit_count": cache_hit_count,
            "cache_hit_rate": round(cache_hit_count / openai_rows, 4) if openai_rows else 0,
            "avg_latency_ms": _avg_or_none(latency_values),
            "canary_applied_avg_latency_ms": applied_avg,
            "holdout_avg_latency_ms": holdout_avg,
            "observed_latency_delta_ms": (
                applied_avg - holdout_avg if applied_avg is not None and holdout_avg is not None else None
            ),
        },
        "state_breakdown": _managed_breakdown(state_counts),
        "source_surface_breakdown": _managed_breakdown(source_surface_counts),
        "quality_gate_breakdown": _managed_breakdown(quality_gate_counts),
        "cache_status_breakdown": _managed_breakdown(cache_status_counts),
        "fallback_reason_breakdown": _managed_breakdown(fallback_reason_counts),
        "candidates": candidate_rows,
        "recent": recent,
        "companion_sections": {
            "anthropic_recommendations": {
                "status": "no-traffic" if anthropic_rows == 0 else "has-traffic",
                "sample_count": anthropic_rows,
                "display": "suppressed" if anthropic_rows == 0 else "available",
                "reason": (
                    "No Claude / Anthropic samples were found in this local report window."
                    if anthropic_rows == 0
                    else "Claude / Anthropic samples exist; use phase-routing reports for Claude-specific evidence."
                ),
            }
        },
        "quality_gate_policy": {
            "max_error_rate": 0.05,
            "max_retry_rate": 0.10,
            "max_fallback_rate": 0.10,
            "max_latency_regression_ratio": 1.25,
            "basis": "metadata fields stored in managed recommendation projection risk blocks",
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "tool_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "secrets_included": False,
            "cache_keys_included": False,
            "local_session_ids_included": False,
            "provider_calls_made": False,
            "basis": "local calls table metrics plus sanitized routing/crunch/cache decision metadata only",
        },
    }

def _openai_canary_policy_state() -> dict[str, Any]:
    from tokenclaw import router as routing

    policy = dict(getattr(routing, "ROUTING_OPENAI_CANARY", {}) or {})
    safety = policy.get("safety_stop") if isinstance(policy.get("safety_stop"), dict) else {}
    return {
        "enabled": bool(policy.get("enabled")),
        "policy_id": policy.get("policy_id"),
        "promotion_action_id": policy.get("promotion_action_id"),
        "target_candidate_id": policy.get("target_candidate_id"),
        "policy_source": policy.get("policy_source") or getattr(routing, "ROUTING_RULES_SOURCE", None),
        "rule_path_included": False,
        "target_model": policy.get("target_model"),
        "model_pattern": policy.get("model_pattern"),
        "canary_fraction": _as_float(policy.get("canary_fraction")),
        "holdout_fraction": _as_float(policy.get("holdout_fraction")),
        "eligible_categories": list(policy.get("eligible_categories") or []),
        "excluded_categories": list(policy.get("excluded_categories") or []),
        "allow_tools": bool(policy.get("allow_tools")),
        "allow_stream": bool(policy.get("allow_stream")),
        "min_text_chars": _as_int(policy.get("min_text_chars")),
        "max_text_chars": _as_int(policy.get("max_text_chars")),
        "min_input_tokens_est": _as_int(policy.get("min_input_tokens_est")),
        "max_input_tokens_est": _as_int(policy.get("max_input_tokens_est")),
        "safety_stop": {
            "enabled": bool(safety.get("enabled", True)),
            "window_hours": _as_int(safety.get("window_hours")),
            "min_samples": _as_int(safety.get("min_samples")),
            "min_holdout_samples": _as_int(safety.get("min_holdout_samples")),
            "max_error_rate": _as_float(safety.get("max_error_rate")),
            "max_retry_rate": _as_float(safety.get("max_retry_rate")),
            "max_fallback_rate": _as_float(safety.get("max_fallback_rate")),
            "max_latency_regression_ratio": _as_float(safety.get("max_latency_regression_ratio")),
            "limit": _as_int(safety.get("limit")),
        },
    }

def _openai_canary_readiness_state(policy: dict[str, Any], impact: dict[str, Any]) -> tuple[str, str]:
    if not policy.get("enabled"):
        return "disabled", "local OpenAI canary policy is disabled"
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    if _as_int(summary.get("candidate_count")) <= 0:
        return "collecting_evidence", "enabled policy has no matched local canary metadata yet"
    verdict_counts = {
        str(row.get("value")): _as_int(row.get("count"))
        for row in summary.get("verdict_counts", [])
        if isinstance(row, dict)
    }
    if verdict_counts.get("rollback", 0) > 0:
        return "rollback", "at least one candidate has a rollback verdict"
    if verdict_counts.get("hold", 0) > 0:
        return "hold", "at least one candidate is holding on risk or stale evidence"
    if verdict_counts.get("widen", 0) > 0:
        return "ready_to_widen", "candidate evidence meets widening thresholds"
    return "collecting_evidence", "canary metadata exists but needs more applied and holdout evidence"

def _openai_canary_top_blockers(impact: dict[str, Any]) -> list[dict[str, Any]]:
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    blockers: dict[str, int] = {}
    for row in summary.get("reason_code_counts", []):
        if not isinstance(row, dict):
            continue
        value = str(row.get("value") or "unknown")
        if value in {"target-savings-met"}:
            continue
        blockers[value] = blockers.get(value, 0) + _as_int(row.get("count"))
    for candidate in impact.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        for row in candidate.get("reason_buckets", []):
            if not isinstance(row, dict):
                continue
            value = str(row.get("value") or "unknown")
            if value in {"selected-canary", "selected-holdout"}:
                continue
            blockers[value] = blockers.get(value, 0) + _as_int(row.get("count"))
    return _managed_breakdown(blockers)[:10]

def _openai_canary_candidate_rows(impact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in impact.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        counts = candidate.get("cohort_counts") if isinstance(candidate.get("cohort_counts"), dict) else {}
        deltas = (
            candidate.get("applied_vs_holdout_deltas")
            if isinstance(candidate.get("applied_vs_holdout_deltas"), dict)
            else {}
        )
        rows.append({
            "candidate_id": candidate.get("candidate_id"),
            "policy_id": candidate.get("policy_id"),
            "policy_source": candidate.get("policy_source"),
            "source_surface": candidate.get("source_surface"),
            "original_model": candidate.get("original_model"),
            "candidate_target_model": candidate.get("candidate_target_model"),
            "verdict": candidate.get("verdict"),
            "next_action": candidate.get("next_action"),
            "reason_codes": candidate.get("reason_codes") or [],
            "warning_codes": candidate.get("warning_codes") or [],
            "sample_count": _as_int(candidate.get("sample_count")),
            "applied_count": _as_int(counts.get("canary_applied")),
            "holdout_count": _as_int(counts.get("canary_holdout")),
            "not_selected_count": _as_int(counts.get("skipped")),
            "disabled_or_bypassed_count": _as_int(counts.get("bypassed_or_disabled")),
            "safety_stopped_count": _as_int(counts.get("safety_stopped")),
            "observed_savings_usd": _as_float(candidate.get("observed_savings_usd")),
            "projected_savings_usd": _as_float(candidate.get("projected_savings_usd")),
            "error_rate_delta": _as_float(deltas.get("applied_minus_holdout_error_rate")),
            "retry_rate_delta": _as_float(deltas.get("applied_minus_holdout_retry_rate")),
            "fallback_rate_delta": _as_float(deltas.get("applied_minus_holdout_fallback_rate")),
            "latency_delta_ms": deltas.get("applied_minus_holdout_latency_avg_ms"),
            "latest_observed_at": candidate.get("latest_observed_at"),
            "stale_evidence": candidate.get("stale_evidence"),
            "top_reasons": candidate.get("reason_buckets") or [],
            "privacy": candidate.get("privacy") or {},
        })
    return rows

async def stats_openai_canary_readiness(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.openai_canary_impact import build_openai_canary_impact_report

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    impact = build_openai_canary_impact_report(store_obj, limit=capped_limit)
    policy = _openai_canary_policy_state()
    state, reason = _openai_canary_readiness_state(policy, impact)
    summary = impact.get("summary") if isinstance(impact.get("summary"), dict) else {}
    top_blockers = _openai_canary_top_blockers(impact)
    candidates = _openai_canary_candidate_rows(impact)
    latest_verdicts = [
        {
            "candidate_id": row.get("candidate_id"),
            "verdict": row.get("verdict"),
            "next_action": row.get("next_action"),
            "reason_codes": row.get("reason_codes"),
            "latest_observed_at": row.get("latest_observed_at"),
        }
        for row in candidates[:10]
    ]
    return {
        "schema": "tokenclaw.openai_canary_readiness.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "state": state,
        "state_reason": reason,
        "policy": policy,
        "summary": {
            "eligible_candidate_count": _as_int(summary.get("candidate_count")),
            "sampled_call_count": _as_int(summary.get("sampled_call_count")),
            "observed_openai_canary_metadata_row_count": _as_int(
                summary.get("observed_openai_canary_metadata_row_count")
            ),
            "canary_applied_count": _as_int(summary.get("canary_applied_count")),
            "canary_holdout_count": _as_int(summary.get("canary_holdout_count")),
            "not_selected_count": sum(_as_int(row.get("not_selected_count")) for row in candidates),
            "disabled_or_bypassed_count": sum(_as_int(row.get("disabled_or_bypassed_count")) for row in candidates),
            "safety_stopped_count": _as_int(summary.get("safety_stopped_count")),
            "active_safety_stop_count": _as_int(summary.get("safety_stopped_count")),
            "observed_savings_usd": _as_float(summary.get("observed_savings_usd")),
            "projected_savings_usd": _as_float(summary.get("projected_savings_usd")),
            "verdict_counts": summary.get("verdict_counts") or [],
            "top_blockers": top_blockers,
            "latest_verdicts": latest_verdicts,
        },
        "candidates": candidates,
        "impact_report": impact,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "tenant_ids_included": False,
            "api_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_file_paths_included": False,
            "basis": "local routing policy metadata plus sanitized OpenAI canary impact aggregates",
        },
    }

async def stats_claude_canary_impact(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.claude_canary_impact import build_claude_canary_impact_report

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    return build_claude_canary_impact_report(store_obj, limit=capped_limit)

def _claude_routing_funnel_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_transcripts_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "raw_request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "api_keys_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "dashboard_mutations_available": False,
        "basis": "local Anthropic routing decision, shadow comparison, and Claude canary aggregate metadata only",
    }

def _claude_route_public_label(*parts: Any) -> str:
    payload = json.dumps([str(part or "unknown") for part in parts], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"claude-route-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:14]}"

def _claude_public_metadata_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    lowered = text.lower()
    unsafe_markers = (
        "api_key",
        "authorization",
        "bearer ",
        "cache_key",
        "content",
        "file_path",
        "message",
        "password",
        "prompt",
        "provider_body",
        "raw_",
        "request_id",
        "response",
        "secret",
        "session_id",
        "sk-",
        "tool_payload",
        "tool_result",
        "transcript",
    )
    if any(marker in lowered for marker in unsafe_markers) or "/" in text or "\\" in text:
        return fallback
    cleaned = text.replace(" ", "-").replace("_", "-").lower()
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in ".:-")
    return cleaned[:80] if cleaned else fallback

def _claude_route_group_key(
    *,
    requested_model: Any,
    routed_model: Any,
    category: Any,
    workflow_phase: Any,
    stream: Any,
    source_surface: Any = "anthropic_messages",
) -> tuple[str, str, str, str, str]:
    return (
        _claude_public_metadata_label(source_surface, "anthropic_messages"),
        _claude_public_metadata_label(requested_model, "unknown-requested"),
        _claude_public_metadata_label(routed_model, "unknown-target"),
        _claude_public_metadata_label(category, "unknown-category"),
        _claude_public_metadata_label(workflow_phase, "unknown-phase"),
        "stream" if bool(stream) else "nonstream",
    )

def _new_claude_route_funnel_candidate(key: tuple[str, str, str, str, str]) -> dict[str, Any]:
    source_surface, requested_model, routed_model, category, workflow_phase, stream_label = key
    return {
        "candidate_id": _claude_route_public_label(*key),
        "provider": "anthropic",
        "source_surface": source_surface,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "model_pair": f"{requested_model} -> {routed_model}",
        "category": category,
        "workflow_phase": workflow_phase,
        "stream": stream_label == "stream",
        "observed_count": 0,
        "eligible_count": 0,
        "eligible_unsampled_count": 0,
        "sampled_count": 0,
        "compared_count": 0,
        "passed_comparison_count": 0,
        "promoted_count": 0,
        "canary_applied_count": 0,
        "canary_holdout_count": 0,
        "canary_held_count": 0,
        "canary_widened_count": 0,
        "canary_rollback_count": 0,
        "safety_stop_count": 0,
        "fallback_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "observed_savings_usd": 0.0,
        "comparison_cost_delta_usd": 0.0,
        "avg_similarity_total": 0.0,
        "avg_similarity_count": 0,
        "latency_delta_ms_total": 0.0,
        "latency_delta_count": 0,
        "budget_exhausted": False,
        "latest_evidence_at": None,
        "oldest_evidence_at": None,
        "blocker_counts": {},
        "stage_counts": {},
        "verdict_counts": {},
        "privacy": _claude_routing_funnel_privacy(),
    }

def _claude_route_touch(candidate: dict[str, Any], created_at: Any) -> None:
    if not created_at:
        return
    text = str(created_at)
    if candidate.get("latest_evidence_at") is None or text > str(candidate.get("latest_evidence_at")):
        candidate["latest_evidence_at"] = text
    if candidate.get("oldest_evidence_at") is None or text < str(candidate.get("oldest_evidence_at")):
        candidate["oldest_evidence_at"] = text

def _claude_route_count(candidate: dict[str, Any], field: str, amount: int = 1) -> None:
    candidate[field] = _as_int(candidate.get(field)) + amount

def _claude_route_increment_breakdown(candidate: dict[str, Any], field: str, key: Any, amount: int = 1) -> None:
    value = _claude_public_metadata_label(key, "redacted-reason") if field == "blocker_counts" else str(key or "unknown")
    bucket = candidate.setdefault(field, {})
    bucket[value] = _as_int(bucket.get(value)) + amount

def _claude_route_is_eligible(row: dict[str, Any], routing: dict[str, Any], canary: dict[str, Any]) -> bool:
    if canary:
        return True
    explicit = routing.get("claude_routing_promotion")
    if isinstance(explicit, dict) and explicit.get("eligible") is not None:
        return bool(explicit.get("eligible"))
    status = str((routing.get("routing_experiment") or {}).get("status") if isinstance(routing.get("routing_experiment"), dict) else "")
    if status in {"eligible", "sampled", "compared"}:
        return True
    requested = str(row.get("requested_model") or "").lower()
    routed = str(row.get("routed_model") or routing.get("candidate_target_model") or "").lower()
    category = str(row.get("category") or routing.get("category") or "").lower()
    phase = str(routing.get("workflow_phase") or "").lower()
    if "claude" in requested and "sonnet" in requested and ("haiku" in routed or category in {"tool-result", "tool-light"}):
        return True
    return phase in {"tool-execution", "summary"} and "sonnet" in requested

def _claude_route_blocker_reason(routing: dict[str, Any], canary: dict[str, Any]) -> str | None:
    if canary:
        reason = canary.get("reason")
        return _claude_public_metadata_label(reason, "redacted-reason") if reason else None
    explicit = routing.get("claude_routing_promotion")
    if isinstance(explicit, dict):
        for key in ("blocker", "reason", "status"):
            if explicit.get(key):
                return _claude_public_metadata_label(explicit.get(key), "redacted-reason")
    reason = routing.get("reason") or routing.get("routing_reason")
    if reason:
        text = str(reason)
        if "eligible" not in text.lower() and "routed" not in text.lower():
            return _claude_public_metadata_label(text, "redacted-reason")
    return None

def _claude_route_stage_from_canary(canary: dict[str, Any]) -> str:
    status = str(canary.get("status") or "")
    cohort = str(canary.get("cohort") or "")
    if status == "applied" or cohort == "canary_applied":
        return "canary-applied"
    if status == "holdout" or cohort == "canary_holdout":
        return "holdout"
    if status == "safety_stopped" or (isinstance(canary.get("safety_stop"), dict) and canary["safety_stop"].get("tripped")):
        return "safety-stopped"
    if status in {"disabled", "ineligible", "noop"}:
        return "held"
    return status or cohort or "unknown"

def _claude_route_finalize_candidate(candidate: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    sampled = _as_int(candidate.get("sampled_count"))
    compared = _as_int(candidate.get("compared_count"))
    eligible = _as_int(candidate.get("eligible_count"))
    candidate["eligible_unsampled_count"] = max(0, eligible - sampled - _as_int(candidate.get("canary_applied_count")) - _as_int(candidate.get("canary_holdout_count")))
    candidate["observed_savings_usd"] = round(_as_float(candidate.get("cost_baseline_usd")) - _as_float(candidate.get("cost_est_usd")), 8)
    candidate["cost_est_usd"] = round(_as_float(candidate.get("cost_est_usd")), 8)
    candidate["cost_baseline_usd"] = round(_as_float(candidate.get("cost_baseline_usd")), 8)
    candidate["comparison_cost_delta_usd"] = round(_as_float(candidate.get("comparison_cost_delta_usd")), 8)
    candidate["avg_similarity"] = (
        round(_as_float(candidate.get("avg_similarity_total")) / _as_int(candidate.get("avg_similarity_count")), 6)
        if _as_int(candidate.get("avg_similarity_count"))
        else None
    )
    candidate["pass_rate"] = round(_as_int(candidate.get("passed_comparison_count")) / compared, 6) if compared else None
    candidate["avg_latency_delta_ms"] = (
        round(_as_float(candidate.get("latency_delta_ms_total")) / _as_int(candidate.get("latency_delta_count")), 2)
        if _as_int(candidate.get("latency_delta_count"))
        else None
    )
    candidate["error_rate"] = round(_as_int(candidate.get("error_count")) / max(1, _as_int(candidate.get("observed_count"))), 6)
    candidate["fallback_rate"] = round(_as_int(candidate.get("fallback_count")) / max(1, _as_int(candidate.get("observed_count"))), 6)
    candidate["retry_rate"] = round(_as_int(candidate.get("retry_count")) / max(1, _as_int(candidate.get("observed_count"))), 6)
    age = _seconds_since_iso(candidate.get("latest_evidence_at"), now)
    candidate["stale_evidence"] = {
        "stale": age is not None and age > 72 * 3600,
        "age_seconds": age,
        "max_age_seconds": 72 * 3600,
    }
    candidate["blocker_counts"] = _count_breakdown(candidate.get("blocker_counts") or {})
    candidate["stage_counts"] = _count_breakdown(candidate.get("stage_counts") or {})
    candidate["verdict_counts"] = _count_breakdown(candidate.get("verdict_counts") or {})
    candidate.pop("avg_similarity_total", None)
    candidate.pop("avg_similarity_count", None)
    candidate.pop("latency_delta_ms_total", None)
    candidate.pop("latency_delta_count", None)
    return candidate

async def stats_claude_routing_promotion_funnel(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.claude_canary_impact import build_claude_canary_impact_report
    from tokenclaw.policy_events import recent_policy_events
    from tokenclaw.router import ROUTING_RULES

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    candidates: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    observed_rows = 0
    permanent_rule_count = sum(1 for rule in ROUTING_RULES if isinstance(rule, dict))
    promoted_permanent_rule_count = 0
    for rule in ROUTING_RULES:
        if not isinstance(rule, dict):
            continue
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        if metadata.get("promoted_from_canary") or metadata.get("source") == "claude-canary-promote":
            promoted_permanent_rule_count += 1
    promotion_events = [
        _public_rollout_policy_event(event, now=now)
        for event in recent_policy_events(limit=100).get("events", [])
        if isinstance(event, dict) and event.get("action") == "routing-canary-promote"
    ][:10]

    call_rows = store_obj.conn.execute(
        """
        select created_at, requested_model, routed_model, stream, status_code, latency_ms,
               retry_count, cost_est_usd, cost_baseline_usd, routing_json, category,
               provider, source_surface
        from calls
        where coalesce(provider, 'anthropic') = 'anthropic'
          and coalesce(source_surface, 'anthropic_messages') = 'anthropic_messages'
        order by created_at desc
        limit ?
        """,
        (capped_limit,),
    ).fetchall()
    for raw_row in call_rows:
        row = dict(raw_row)
        observed_rows += 1
        routing = _json_obj(row.get("routing_json"))
        canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
        target_model = (
            canary.get("target_model")
            or canary.get("candidate_target_model")
            or row.get("routed_model")
            or routing.get("candidate_target_model")
            or "unknown-target"
        )
        key = _claude_route_group_key(
            requested_model=canary.get("original_model") or canary.get("requested_model") or row.get("requested_model"),
            routed_model=target_model,
            category=canary.get("category") or row.get("category") or routing.get("category"),
            workflow_phase=canary.get("workflow_phase") or routing.get("workflow_phase"),
            stream=row.get("stream"),
            source_surface=row.get("source_surface") or "anthropic_messages",
        )
        candidate = candidates.setdefault(key, _new_claude_route_funnel_candidate(key))
        _claude_route_count(candidate, "observed_count")
        _claude_route_touch(candidate, row.get("created_at"))
        candidate["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
        candidate["cost_baseline_usd"] += _as_float(row.get("cost_baseline_usd"))
        if _as_int(row.get("status_code")) >= 400:
            _claude_route_count(candidate, "error_count")
        if _as_int(row.get("retry_count")) > 0:
            _claude_route_count(candidate, "retry_count")
        if routing.get("fallback_reason") or canary.get("fallback_reason"):
            _claude_route_count(candidate, "fallback_count")

        if _claude_route_is_eligible(row, routing, canary):
            _claude_route_count(candidate, "eligible_count")
            _claude_route_increment_breakdown(candidate, "stage_counts", "eligible")
        else:
            _claude_route_increment_breakdown(candidate, "stage_counts", "observed-not-eligible")
        blocker = _claude_route_blocker_reason(routing, canary)
        if blocker:
            _claude_route_increment_breakdown(candidate, "blocker_counts", blocker)

        if canary:
            stage = _claude_route_stage_from_canary(canary)
            _claude_route_increment_breakdown(candidate, "stage_counts", stage)
            if stage == "canary-applied":
                _claude_route_count(candidate, "canary_applied_count")
            elif stage == "holdout":
                _claude_route_count(candidate, "canary_holdout_count")
            elif stage == "safety-stopped":
                _claude_route_count(candidate, "safety_stop_count")
            elif stage == "held":
                _claude_route_count(candidate, "canary_held_count")
            safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
            for reason in safety.get("reason_codes") or []:
                _claude_route_increment_breakdown(candidate, "blocker_counts", reason)

    experiment_rows = store_obj.conn.execute(
        """
        select created_at, requested_model, routed_model, stream, category, source_surface,
               primary_status_code, shadow_status_code, primary_latency_ms, shadow_latency_ms,
               output_similarity, passed_threshold, primary_cost_est_usd, shadow_cost_est_usd,
               budget_limit_usd, budget_remaining_before_usd, budget_spent_after_usd,
               routing_json, experiment_json, error
        from routing_experiments
        where coalesce(provider, 'anthropic') = 'anthropic'
          and coalesce(source_surface, 'anthropic_messages') = 'anthropic_messages'
        order by created_at desc
        limit ?
        """,
        (capped_limit,),
    ).fetchall()
    for raw_row in experiment_rows:
        row = dict(raw_row)
        routing = _json_obj(row.get("routing_json"))
        experiment = _json_obj(row.get("experiment_json"))
        key = _claude_route_group_key(
            requested_model=row.get("requested_model"),
            routed_model=row.get("routed_model"),
            category=row.get("category"),
            workflow_phase=routing.get("workflow_phase") or experiment.get("workflow_phase"),
            stream=row.get("stream"),
            source_surface=row.get("source_surface") or "anthropic_messages",
        )
        candidate = candidates.setdefault(key, _new_claude_route_funnel_candidate(key))
        _claude_route_touch(candidate, row.get("created_at"))
        _claude_route_count(candidate, "sampled_count")
        _claude_route_increment_breakdown(candidate, "stage_counts", "sampled")
        if row.get("output_similarity") is not None or row.get("passed_threshold") is not None:
            _claude_route_count(candidate, "compared_count")
            _claude_route_increment_breakdown(candidate, "stage_counts", "compared")
        if _as_int(row.get("passed_threshold")):
            _claude_route_count(candidate, "passed_comparison_count")
        if row.get("output_similarity") is not None:
            candidate["avg_similarity_total"] += _as_float(row.get("output_similarity"))
            candidate["avg_similarity_count"] += 1
        if row.get("primary_latency_ms") is not None and row.get("shadow_latency_ms") is not None:
            candidate["latency_delta_ms_total"] += _as_float(row.get("shadow_latency_ms")) - _as_float(row.get("primary_latency_ms"))
            candidate["latency_delta_count"] += 1
        candidate["comparison_cost_delta_usd"] += _as_float(row.get("shadow_cost_est_usd")) - _as_float(row.get("primary_cost_est_usd"))
        if _as_int(row.get("shadow_status_code")) >= 400:
            _claude_route_increment_breakdown(candidate, "blocker_counts", "shadow-error")
        if _as_int(row.get("primary_status_code")) >= 400:
            _claude_route_increment_breakdown(candidate, "blocker_counts", "primary-error")
        if row.get("error"):
            _claude_route_increment_breakdown(candidate, "blocker_counts", "comparison-error")
        if row.get("budget_remaining_before_usd") is not None and _as_float(row.get("budget_remaining_before_usd")) <= 0:
            candidate["budget_exhausted"] = True
            _claude_route_increment_breakdown(candidate, "blocker_counts", "shadow-budget-exhausted")

    try:
        canary_impact = build_claude_canary_impact_report(store_obj, limit=capped_limit)
    except Exception:
        canary_impact = {"candidates": [], "summary": {}}
    for verdict_row in canary_impact.get("candidates") or []:
        if not isinstance(verdict_row, dict):
            continue
        key = _claude_route_group_key(
            requested_model=verdict_row.get("original_model"),
            routed_model=verdict_row.get("candidate_target_model"),
            category=verdict_row.get("category"),
            workflow_phase=verdict_row.get("workflow_phase"),
            stream=verdict_row.get("stream"),
            source_surface=verdict_row.get("source_surface") or "anthropic_messages",
        )
        candidate = candidates.setdefault(key, _new_claude_route_funnel_candidate(key))
        verdict = str(verdict_row.get("verdict") or "unknown")
        _claude_route_increment_breakdown(candidate, "verdict_counts", verdict)
        if verdict == "widen":
            _claude_route_count(candidate, "canary_widened_count")
            _claude_route_increment_breakdown(candidate, "stage_counts", "widened")
        elif verdict == "hold":
            _claude_route_count(candidate, "canary_held_count")
            _claude_route_increment_breakdown(candidate, "stage_counts", "held")
        elif verdict == "rollback":
            _claude_route_count(candidate, "canary_rollback_count")
            _claude_route_increment_breakdown(candidate, "stage_counts", "rolled-back")
        elif verdict == "needs_more_samples":
            _claude_route_increment_breakdown(candidate, "stage_counts", "needs-more-samples")
        for reason in verdict_row.get("reason_codes") or []:
            _claude_route_increment_breakdown(candidate, "blocker_counts", reason)
        _claude_route_touch(candidate, verdict_row.get("latest_observed_at"))

    for candidate in candidates.values():
        compared = _as_int(candidate.get("compared_count"))
        if compared and (_as_int(candidate.get("passed_comparison_count")) / compared) >= 0.9:
            candidate["promoted_count"] = max(_as_int(candidate.get("promoted_count")), 1)
            _claude_route_increment_breakdown(candidate, "stage_counts", "promoted")

    rows = [_claude_route_finalize_candidate(candidate, now=now) for candidate in candidates.values()]
    rows.sort(
        key=lambda row: (
            -_as_int(row.get("canary_widened_count")),
            -_as_int(row.get("canary_applied_count")),
            -_as_int(row.get("compared_count")),
            -_as_int(row.get("eligible_unsampled_count")),
            str(row.get("candidate_id")),
        )
    )

    summary_counts = {
        "observed_count": observed_rows,
        "candidate_count": len(rows),
        "eligible_count": sum(_as_int(row.get("eligible_count")) for row in rows),
        "eligible_unsampled_count": sum(_as_int(row.get("eligible_unsampled_count")) for row in rows),
        "sampled_count": sum(_as_int(row.get("sampled_count")) for row in rows),
        "compared_count": sum(_as_int(row.get("compared_count")) for row in rows),
        "promoted_count": sum(_as_int(row.get("promoted_count")) for row in rows),
        "canary_applied_count": sum(_as_int(row.get("canary_applied_count")) for row in rows),
        "holdout_count": sum(_as_int(row.get("canary_holdout_count")) for row in rows),
        "widened_count": sum(_as_int(row.get("canary_widened_count")) for row in rows),
        "held_count": sum(_as_int(row.get("canary_held_count")) for row in rows),
        "rolled_back_count": sum(_as_int(row.get("canary_rollback_count")) for row in rows),
        "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in rows),
        "fallback_count": sum(_as_int(row.get("fallback_count")) for row in rows),
        "error_count": sum(_as_int(row.get("error_count")) for row in rows),
        "budget_exhausted_count": sum(1 for row in rows if row.get("budget_exhausted")),
    }
    blocker_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for row in rows:
        for item in row.get("blocker_counts") or []:
            blocker_counts[str(item.get("value") or "unknown")] = blocker_counts.get(str(item.get("value") or "unknown"), 0) + _as_int(item.get("count"))
        for item in row.get("stage_counts") or []:
            stage_counts[str(item.get("value") or "unknown")] = stage_counts.get(str(item.get("value") or "unknown"), 0) + _as_int(item.get("count"))
        for item in row.get("verdict_counts") or []:
            verdict_counts[str(item.get("value") or "unknown")] = verdict_counts.get(str(item.get("value") or "unknown"), 0) + _as_int(item.get("count"))

    return {
        "schema": "tokenclaw.claude_routing_promotion_funnel.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            **summary_counts,
            "cost_est_usd": round(sum(_as_float(row.get("cost_est_usd")) for row in rows), 8),
            "cost_baseline_usd": round(sum(_as_float(row.get("cost_baseline_usd")) for row in rows), 8),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in rows), 8),
            "comparison_cost_delta_usd": round(sum(_as_float(row.get("comparison_cost_delta_usd")) for row in rows), 8),
            "last_evidence_at": max((str(row.get("latest_evidence_at")) for row in rows if row.get("latest_evidence_at")), default=None),
            "permanent_rule_count": permanent_rule_count,
            "promoted_permanent_rule_count": promoted_permanent_rule_count,
            "promotion_event_count": len(promotion_events),
        },
        "stage_counts": _count_breakdown(stage_counts),
        "verdict_counts": _count_breakdown(verdict_counts),
        "blocker_counts": _count_breakdown(blocker_counts),
        "candidates": rows,
        "promotion_events": promotion_events,
        "source_reports": {
            "claude_canary_impact_schema": canary_impact.get("schema") if isinstance(canary_impact, dict) else None,
        },
        "privacy": _claude_routing_funnel_privacy(),
    }

def _shadow_routing_candidate_id(row: dict[str, Any]) -> str:
    explicit = row.get("candidate_id") or row.get("configured_candidate_id")
    if explicit:
        return str(explicit)
    parts = [
        row.get("provider"),
        row.get("source_surface"),
        row.get("requested_model"),
        row.get("routed_model"),
        row.get("category"),
        row.get("workflow_phase"),
    ]
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"shadow-route-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

def _shadow_routing_row_key(row: dict[str, Any]) -> tuple[str, str, bool, str, str, str, str]:
    return (
        str(row.get("provider") or "unknown"),
        str(row.get("source_surface") or "unknown"),
        bool(row.get("stream")),
        str(row.get("requested_model") or ""),
        str(row.get("routed_model") or ""),
        str(row.get("category") or "unknown"),
        str(row.get("workflow_phase") or "unknown"),
    )

def _shadow_routing_readiness_state(verdict: str, mode: str) -> str:
    if verdict == "promote":
        return "ready-to-stage" if mode == "shadow_candidate_pass_through" else "ready-to-widen"
    if verdict == "needs_more_samples":
        return "needs-more-samples"
    if verdict == "hold":
        return "held"
    if verdict == "reject":
        return "rejected"
    return "unknown"

def _shadow_routing_scoreboard_to_candidate_row(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("readiness_status") or "insufficient-evidence")
    verdict = {
        "ready": "promote",
        "regressing": "reject",
        "insufficient-evidence": "needs_more_samples",
    }.get(status, "unknown")
    sample_rate = _as_float(row.get("sample_rate") or policy.get("sample_rate"))
    applied_count = _as_int(row.get("applied_count"))
    holdout_count = _as_int(row.get("holdout_count"))
    return {
        "candidate_id": str(row.get("candidate_id") or "unknown"),
        "label": row.get("label"),
        "provider": str(row.get("provider") or "unknown"),
        "source_surface": str(row.get("source_surface") or "unknown"),
        "stream": bool(row.get("stream")),
        "requested_model": row.get("requested_model"),
        "routed_model": row.get("routed_model"),
        "model_pair": f"{row.get('requested_model') or 'unknown'} -> {row.get('routed_model') or 'unknown'}",
        "category": str(row.get("category") or "unknown"),
        "workflow_phase": str(row.get("workflow_phase") or "unknown"),
        "sample_mode": "configured-shadow-candidate",
        "sample_rate": sample_rate,
        "sample_count": _as_int(row.get("sample_count")),
        "compared_count": _as_int(row.get("compared_count")),
        "compared_coverage": round(_as_int(row.get("compared_count")) / max(1, _as_int(row.get("sample_count"))), 6),
        "pass_rate": row.get("pass_rate"),
        "avg_similarity": row.get("avg_similarity"),
        "primary_error_rate": 0.0,
        "shadow_error_rate": _as_float(row.get("candidate_error_rate")),
        "fallback_or_retry_count": _as_int(row.get("fallback_or_retry_count")),
        "fallback_or_retry_rate": _as_float(row.get("fallback_or_retry_rate")),
        "cost_delta_usd": _as_float(row.get("realized_cost_delta_vs_baseline_usd")),
        "realized_cost_delta_vs_baseline_usd": _as_float(row.get("realized_cost_delta_vs_baseline_usd")),
        "realized_cost_delta_vs_holdout_usd": _as_float(row.get("realized_cost_delta_vs_holdout_usd")),
        "baseline_cost_usd": _as_float(row.get("baseline_cost_usd")),
        "candidate_cost_usd": _as_float(row.get("candidate_cost_usd")),
        "avg_latency_delta_ms": None,
        "last_sample_at": row.get("latest_evidence_at"),
        "last_sample_age_hours": None,
        "promotion_verdict": verdict,
        "readiness_state": status,
        "readiness_status": status,
        "promotion_ready": status == "ready",
        "promotion_scope": "stage_local_canary_from_shadow",
        "evidence_kind": "shadow_candidate_scoreboard",
        "reason_codes": [str(reason) for reason in (row.get("reason_codes") or []) if reason is not None][:10],
        "thresholds": row.get("min_sample_gate") or {},
        "freshness": {
            "age_hours": None,
            "fresh": bool(row.get("latest_evidence_at")),
        },
        "canary": {
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "applied_fraction": sample_rate if applied_count else 0.0,
            "holdout_fraction": sample_rate if holdout_count else 0.0,
            "shadow_only": holdout_count > 0,
            "canary_evidence": applied_count > 0,
            "safety_stop_state": "clear",
            "safety_stop_count": 0,
        },
        "budget": {
            "daily_budget_usd": _as_float(policy.get("daily_budget_usd")),
            "today_shadow_spend_usd": _as_float(policy.get("today_shadow_spend_usd")),
            "daily_budget_exhausted": bool(policy.get("daily_budget_exhausted")),
        },
        "routing_reasons": [],
        "privacy": row.get("privacy") or {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "cache_keys_included": False,
        },
    }

def _shadow_routing_candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    rows: list[dict[str, Any]] = []
    for candidate in report.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        promotion = candidate.get("promotion") if isinstance(candidate.get("promotion"), dict) else {}
        coverage = promotion.get("coverage") if isinstance(promotion.get("coverage"), dict) else {}
        thresholds = promotion.get("thresholds") if isinstance(promotion.get("thresholds"), dict) else {}
        budget = promotion.get("budget") if isinstance(promotion.get("budget"), dict) else {}
        mode = str(candidate.get("mode") or "unknown")
        verdict = str(candidate.get("promotion_verdict") or promotion.get("verdict") or "unknown")
        samples = _as_int(candidate.get("samples"))
        compared = _as_int(candidate.get("compared_samples"))
        shadow_only = bool(promotion.get("shadow_only") or mode == "shadow_candidate_pass_through")
        canary_evidence = bool(promotion.get("canary_evidence") or mode == "applied_routed_down")
        applied_count = samples if canary_evidence else 0
        holdout_count = samples if shadow_only else 0
        fallback_retry_count = _as_int(candidate.get("fallback_or_retry_count"))
        sample_rate = _as_float(policy.get("sample_rate"))
        row = {
            "candidate_id": _shadow_routing_candidate_id(candidate),
            "provider": str(candidate.get("provider") or "unknown"),
            "source_surface": str(candidate.get("source_surface") or "unknown"),
            "requested_model": candidate.get("requested_model"),
            "routed_model": candidate.get("routed_model"),
            "model_pair": f"{candidate.get('requested_model') or 'unknown'} -> {candidate.get('routed_model') or 'unknown'}",
            "category": str(candidate.get("category") or "unknown"),
            "workflow_phase": str(candidate.get("workflow_phase") or "unknown"),
            "sample_mode": mode,
            "sample_rate": sample_rate,
            "sample_count": samples,
            "compared_count": compared,
            "compared_coverage": _as_float(candidate.get("compared_coverage")),
            "pass_rate": candidate.get("pass_rate"),
            "avg_similarity": candidate.get("avg_similarity"),
            "primary_error_rate": _as_float(candidate.get("primary_error_rate")),
            "shadow_error_rate": _as_float(candidate.get("shadow_error_rate")),
            "fallback_or_retry_count": fallback_retry_count,
            "fallback_or_retry_rate": round(fallback_retry_count / samples, 6) if samples else 0.0,
            "cost_delta_usd": _as_float(candidate.get("cost_delta_usd")),
            "avg_latency_delta_ms": candidate.get("avg_latency_delta_ms"),
            "last_sample_at": candidate.get("last_sample_at"),
            "last_sample_age_hours": candidate.get("last_sample_age_hours"),
            "promotion_verdict": verdict,
            "readiness_state": _shadow_routing_readiness_state(verdict, mode),
            "promotion_ready": bool(promotion.get("promotion_ready")),
            "promotion_scope": str(promotion.get("promotion_scope") or "unknown"),
            "evidence_kind": str(promotion.get("evidence_kind") or ("shadow_pass_through" if shadow_only else "applied_canary")),
            "reason_codes": [
                str(reason)
                for reason in (candidate.get("promotion_reason_codes") or promotion.get("reason_codes") or [])
                if reason is not None
            ][:10],
            "thresholds": {
                "min_samples": _as_int(thresholds.get("min_samples")),
                "min_compared_coverage": _as_float(thresholds.get("min_compared_coverage")),
                "min_similarity_pass_rate": _as_float(thresholds.get("min_similarity_pass_rate")),
                "max_shadow_error_rate": _as_float(thresholds.get("max_shadow_error_rate")),
                "max_primary_error_rate": _as_float(thresholds.get("max_primary_error_rate")),
                "freshness_max_age_hours": _as_int(thresholds.get("freshness_max_age_hours")),
            },
            "freshness": {
                "age_hours": candidate.get("last_sample_age_hours"),
                "fresh": "stale-evidence" not in (candidate.get("promotion_reason_codes") or []),
            },
            "canary": {
                "applied_count": applied_count,
                "holdout_count": holdout_count,
                "applied_fraction": sample_rate if canary_evidence else 0.0,
                "holdout_fraction": sample_rate if shadow_only else 0.0,
                "shadow_only": shadow_only,
                "canary_evidence": canary_evidence,
                "safety_stop_state": "clear",
                "safety_stop_count": 0,
            },
            "budget": {
                "daily_budget_usd": _as_float(budget.get("daily_budget_usd")),
                "today_shadow_spend_usd": _as_float(budget.get("today_shadow_spend_usd")),
                "daily_budget_exhausted": bool(budget.get("daily_budget_exhausted")),
            },
            "routing_reasons": candidate.get("routing_reasons") or [],
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "content_free": True,
                "raw_prompts_included": False,
                "raw_provider_bodies_included": False,
                "raw_responses_included": False,
                "tool_payloads_included": False,
                "request_ids_included": False,
                "raw_session_ids_included": False,
                "filesystem_paths_included": False,
                "cache_keys_included": False,
                "api_keys_included": False,
            },
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("readiness_state") or ""),
            -_as_int(item.get("sample_count")),
            str(item.get("candidate_id") or ""),
        )
    )
    return rows

async def stats_shadow_routing_promotion_readiness(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 1000))
    report = build_routing_experiment_report(store_obj, limit=capped_limit)
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    legacy_candidates = _shadow_routing_candidate_rows(report)
    scoreboard = report.get("readiness_scoreboard") if isinstance(report.get("readiness_scoreboard"), dict) else {}
    scoreboard_rows = [
        _shadow_routing_scoreboard_to_candidate_row(row, policy)
        for row in scoreboard.get("candidates", [])
        if isinstance(row, dict)
    ]
    legacy_by_key = {_shadow_routing_row_key(row): row for row in legacy_candidates}
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, bool, str, str, str, str]] = set()
    for row in scoreboard_rows:
        key = _shadow_routing_row_key(row)
        merged = {**row, **legacy_by_key.get(key, {})}
        for field in (
            "readiness_status",
            "realized_cost_delta_vs_baseline_usd",
            "realized_cost_delta_vs_holdout_usd",
            "baseline_cost_usd",
            "candidate_cost_usd",
        ):
            if field in row:
                merged[field] = row[field]
        merged["readiness_state"] = row.get("readiness_status") or merged.get("readiness_state")
        merged["reason_codes"] = sorted(set((merged.get("reason_codes") or []) + (row.get("reason_codes") or [])))
        candidates.append(merged)
        seen_keys.add(key)
    for row in legacy_candidates:
        key = _shadow_routing_row_key(row)
        if key not in seen_keys:
            candidates.append(row)
    candidates.sort(
        key=lambda row: (
            {"ready": 0, "regressing": 1, "insufficient-evidence": 2}.get(str(row.get("readiness_status") or row.get("readiness_state")), 3),
            -_as_int(row.get("sample_count")),
            str(row.get("candidate_id") or ""),
        )
    )
    verdict_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in candidates:
        _increment_count(verdict_counts, row.get("promotion_verdict"))
        _increment_count(readiness_counts, row.get("readiness_state"))
        for reason in row.get("reason_codes") or []:
            _increment_count(reason_counts, reason)
    return {
        "schema": "tokenclaw.shadow_routing_promotion_readiness.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "policy": {
            "profile_id": policy.get("profile_id"),
            "enabled": bool(policy.get("enabled")),
            "mode": policy.get("mode"),
            "policy_source": policy.get("policy_source"),
            "sample_rate": _as_float(policy.get("sample_rate")),
            "daily_budget_usd": _as_float(policy.get("daily_budget_usd")),
            "today_shadow_spend_usd": _as_float(policy.get("today_shadow_spend_usd")),
            "daily_budget_exhausted": bool(policy.get("daily_budget_exhausted")),
            "provider_count": len(policy.get("providers") or []),
            "source_surface_count": len(policy.get("source_surfaces") or []),
            "model_pair_count": len(policy.get("model_pairs") or []),
        },
        "summary": {
            "candidate_count": len(candidates),
            "sample_count": _as_int(summary.get("sample_count")),
            "compared_count": _as_int(summary.get("comparison_count")),
            "readiness_ready_count": (scoreboard.get("summary") or {}).get("ready_count", 0),
            "readiness_insufficient_evidence_count": (scoreboard.get("summary") or {}).get("insufficient_evidence_count", 0),
            "readiness_regressing_count": (scoreboard.get("summary") or {}).get("regressing_count", 0),
            "readiness_total_realized_cost_delta_vs_baseline_usd": (scoreboard.get("summary") or {}).get("total_realized_cost_delta_vs_baseline_usd", 0.0),
            "readiness_total_realized_cost_delta_vs_holdout_usd": (scoreboard.get("summary") or {}).get("total_realized_cost_delta_vs_holdout_usd", 0.0),
            "min_samples": (scoreboard.get("summary") or {}).get("min_samples", 0),
            "shadow_only_count": sum(_as_int((row.get("canary") or {}).get("holdout_count")) for row in candidates),
            "applied_canary_count": sum(_as_int((row.get("canary") or {}).get("applied_count")) for row in candidates),
            "promotion_ready_count": sum(1 for row in candidates if row.get("promotion_verdict") == "promote"),
            "hold_count": sum(1 for row in candidates if row.get("promotion_verdict") == "hold"),
            "needs_more_samples_count": sum(1 for row in candidates if row.get("promotion_verdict") == "needs_more_samples"),
            "reject_count": sum(1 for row in candidates if row.get("promotion_verdict") == "reject"),
            "avg_similarity": summary.get("avg_similarity"),
            "pass_rate": summary.get("pass_rate"),
            "cost_delta_usd": _as_float(summary.get("cost_delta_usd")),
            "avg_latency_delta_ms": summary.get("avg_latency_delta_ms"),
            "sample_mode_counts": summary.get("sample_mode_counts") or {},
            "managed_feedback_status_counts": summary.get("feedback_status_counts") or {},
        },
        "promotion_verdict_counts": _count_breakdown(verdict_counts),
        "readiness_counts": _count_breakdown(readiness_counts),
        "reason_counts": _count_breakdown(reason_counts),
        "candidates": candidates,
        "source_reports": {
            "routing_experiment_report_schema": report.get("schema"),
            "routing_candidate_readiness_scoreboard_schema": scoreboard.get("schema"),
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "api_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "sanitized local routing experiment promotion aggregates only",
        },
    }

ROUTING_CANDIDATE_LIFECYCLE_STATES = (
    "uncovered",
    "candidate",
    "collecting",
    "scored",
    "staged",
    "promoted",
    "rolled-back",
    "blocked",
)

def _routing_candidate_lifecycle_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "read_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "individual_candidate_ids_included": False,
        "filesystem_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }

def _routing_candidate_lifecycle_source(candidate_id: Any) -> str:
    text = str(candidate_id or "")
    if text.startswith("dashboard-"):
        return "dashboard-added"
    if text.startswith("routing-pathway-"):
        return "managed-pathway"
    if text:
        return "configured-policy"
    return "observed-only"

def _routing_candidate_stage_row(stage: str, count: int, total: int, source_counts: Counter[str], blocker_counts: Counter[str]) -> dict[str, Any]:
    blockers = _count_breakdown(dict(blocker_counts))
    sources = _count_breakdown(dict(source_counts))
    return {
        "stage": stage,
        "count": int(count),
        "share": round(count / total, 6) if total else 0.0,
        "source_breakdown": sources,
        "top_blocker_reason": blockers[0]["value"] if blockers else None,
        "top_blocker_count": blockers[0]["count"] if blockers else 0,
        "blocker_breakdown": blockers[:8],
        "privacy": _routing_candidate_lifecycle_privacy(),
    }

async def stats_routing_candidate_lifecycle_burndown(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 1000))
    activity = await _stats_activity_facade(store_obj, limit=min(capped_limit, 500))
    shadow = await stats_shadow_routing_promotion_readiness(store_obj, limit=capped_limit)

    state_counts: Counter[str] = Counter({state: 0 for state in ROUTING_CANDIDATE_LIFECYCLE_STATES})
    source_counts: Counter[str] = Counter()
    source_by_state: dict[str, Counter[str]] = {state: Counter() for state in ROUTING_CANDIDATE_LIFECYCLE_STATES}
    blocker_counts: Counter[str] = Counter()
    blocker_by_state: dict[str, Counter[str]] = {state: Counter() for state in ROUTING_CANDIDATE_LIFECYCLE_STATES}
    row_examples: list[dict[str, Any]] = []

    for unit in activity.get("units", []):
        if not isinstance(unit, dict):
            continue
        features = unit.get("optimization_features") if isinstance(unit.get("optimization_features"), dict) else {}
        coverage = features.get("routing_candidate") if isinstance(features.get("routing_candidate"), dict) else {}
        status = str(coverage.get("status") or "")
        if status not in {"uncovered", "routing-off"}:
            continue
        state_counts["uncovered"] += 1
        source = "recent-activity"
        reason = public_label(coverage.get("reason") or "no-routing-candidate", "no-routing-candidate")
        source_counts[source] += 1
        source_by_state["uncovered"][source] += 1
        blocker_counts[reason] += 1
        blocker_by_state["uncovered"][reason] += 1

    for row in shadow.get("candidates", []):
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        source = _routing_candidate_lifecycle_source(candidate_id)
        source_counts[source] += 1

        state_counts["candidate"] += 1
        source_by_state["candidate"][source] += 1

        sample_count = _as_int(row.get("sample_count"))
        compared_count = _as_int(row.get("compared_count"))
        canary = row.get("canary") if isinstance(row.get("canary"), dict) else {}
        applied = _as_int(canary.get("applied_count"))
        holdout = _as_int(canary.get("holdout_count"))
        verdict = public_label(row.get("promotion_verdict"), "unknown")
        readiness = public_label(row.get("readiness_status") or row.get("readiness_state"), "unknown")
        reasons = [public_label(reason, "unknown") for reason in (row.get("reason_codes") or []) if reason]

        if sample_count > 0:
            state_counts["collecting"] += 1
            source_by_state["collecting"][source] += 1
        if compared_count > 0:
            state_counts["scored"] += 1
            source_by_state["scored"][source] += 1
        if applied > 0 or holdout > 0:
            state_counts["staged"] += 1
            source_by_state["staged"][source] += 1
        if verdict == "promote" or readiness == "ready":
            state_counts["promoted"] += 1
            source_by_state["promoted"][source] += 1
        if verdict == "reject" or readiness == "regressing":
            state_counts["rolled-back"] += 1
            source_by_state["rolled-back"][source] += 1
        if verdict in {"hold", "needs-more-samples", "needs_more_samples", "reject"} or readiness in {"insufficient-evidence", "regressing"}:
            state_counts["blocked"] += 1
            source_by_state["blocked"][source] += 1

        if reasons:
            for reason in reasons:
                blocker_counts[reason] += 1
                target_state = "blocked" if verdict != "promote" and readiness != "ready" else "promoted"
                blocker_by_state[target_state][reason] += 1
        elif sample_count <= 0:
            blocker_counts["not-collecting-shadow-evidence"] += 1
            blocker_by_state["blocked"]["not-collecting-shadow-evidence"] += 1

        if len(row_examples) < 8:
            row_examples.append({
                "source": source,
                "readiness_state": readiness,
                "promotion_verdict": verdict,
                "sample_count": sample_count,
                "compared_count": compared_count,
                "applied_count": applied,
                "holdout_count": holdout,
                "reason_codes": reasons[:5],
                "candidate_id_included": False,
            })

    total = sum(state_counts.values())
    stage_rows = [
        _routing_candidate_stage_row(
            state,
            state_counts[state],
            total,
            source_by_state[state],
            blocker_by_state[state],
        )
        for state in ROUTING_CANDIDATE_LIFECYCLE_STATES
    ]
    top_blockers = _count_breakdown(dict(blocker_counts))
    return {
        "schema": "tokenclaw.routing_candidate_lifecycle_burndown.v1",
        "generated_at": utc_now(),
        "status": "tracked" if total else "no-routing-candidate-lifecycle-data",
        "limit": capped_limit,
        "read_only": True,
        "default_polling_bounded": True,
        "stage_counts": {state: state_counts[state] for state in ROUTING_CANDIDATE_LIFECYCLE_STATES},
        "lifecycle_counts": stage_rows,
        "source_breakdown": _count_breakdown(dict(source_counts)),
        "top_blocker_reason": top_blockers[0]["value"] if top_blockers else None,
        "top_blocker_count": top_blockers[0]["count"] if top_blockers else 0,
        "top_blockers": top_blockers[:10],
        "examples": row_examples,
        "summary": {
            "uncovered_count": state_counts["uncovered"],
            "candidate_count": state_counts["candidate"],
            "collecting_count": state_counts["collecting"],
            "scored_count": state_counts["scored"],
            "staged_count": state_counts["staged"],
            "promoted_count": state_counts["promoted"],
            "rolled_back_count": state_counts["rolled-back"],
            "blocked_count": state_counts["blocked"],
            "dashboard_added_count": source_counts.get("dashboard-added", 0),
            "managed_pathway_count": source_counts.get("managed-pathway", 0),
            "configured_policy_count": source_counts.get("configured-policy", 0),
            "recent_uncovered_activity_count": source_counts.get("recent-activity", 0),
            "top_blocker_reason": top_blockers[0]["value"] if top_blockers else None,
            "top_blocker_count": top_blockers[0]["count"] if top_blockers else 0,
        },
        "source_reports": {
            "activity_schema": activity.get("schema"),
            "shadow_routing_promotion_readiness_schema": shadow.get("schema"),
            "routing_candidate_readiness_scoreboard_schema": (shadow.get("source_reports") or {}).get("routing_candidate_readiness_scoreboard_schema"),
        },
        "privacy": _routing_candidate_lifecycle_privacy(),
    }

async def stats_openai_routing_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.openai_routing_report import build_openai_routing_report

    return build_openai_routing_report(store_obj, limit=limit)

async def stats_routing_coverage_report(store_obj: Any, limit: int = 5000) -> dict[str, Any]:
    from tokenclaw.routing_coverage import build_routing_coverage_report

    return build_routing_coverage_report(store_obj, limit=limit)
