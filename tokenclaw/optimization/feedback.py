from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from typing import Any, Sequence

from tokenclaw.managed_mode import managed_product_mode
from tokenclaw.optimization.cli_support import (
    default_db_path,
    open_store_for_db,
    redact_secret,
    redact_url,
    write_json,
)


MANAGED_POLICY_API_KEY_ENV = "TOKENCLAW_MANAGED_API_KEY"


def parse_utc_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def seconds_since(raw: Any, now: datetime) -> int | None:
    parsed = parse_utc_iso(raw)
    if parsed is None:
        return None
    return max(0, int((now - parsed).total_seconds()))


def breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _safe_payload_json(row: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(row.get("payload_json") or "{}"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pattern_evidence_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="pattern_policy_evidence"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    item_count = 0
    status_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    endpoint_status: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        payload = _safe_payload_json(row)
        evidence = payload.get("pattern_policy_evidence")
        if not isinstance(evidence, list) or not evidence:
            continue
        row_count += 1
        row_items = [item for item in evidence if isinstance(item, dict)]
        item_count += len(row_items)
        status = str(row.get("status") or "unknown")
        endpoint = str(row.get("endpoint") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        key = (endpoint, status)
        bucket = endpoint_status.setdefault(
            key,
            {"endpoint": endpoint, "status": status, "queue_rows": 0, "evidence_items": 0},
        )
        bucket["queue_rows"] += 1
        bucket["evidence_items"] += len(row_items)
        for item in row_items:
            action = str(item.get("action_family") or "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "schema": "tokenclaw.managed_pattern_evidence_queue_status.v1",
        "queue_rows": row_count,
        "evidence_items": item_count,
        "status_breakdown": breakdown_from_counts(status_counts),
        "endpoint_breakdown": breakdown_from_counts(endpoint_counts),
        "action_family_breakdown": breakdown_from_counts(action_counts),
        "endpoint_status_breakdown": sorted(
            endpoint_status.values(),
            key=lambda item: (-int(item["queue_rows"]), str(item["endpoint"]), str(item["status"])),
        ),
        "payload_json_included": False,
    }


def _routing_experiment_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.routing_experiment_outcome_event.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    status_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        payload = _safe_payload_json(row)
        if payload.get("schema") != "tokenclaw.routing_experiment_outcome_event.v1":
            continue
        row_count += 1
        status = str(row.get("status") or "unknown")
        endpoint = str(row.get("endpoint") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        outcome = payload.get("outcome") if isinstance(payload.get("outcome"), dict) else {}
        outcome_status = str(outcome.get("status") or "unknown")
        outcome_counts[outcome_status] = outcome_counts.get(outcome_status, 0) + 1
        for reason in payload.get("reason_codes") or []:
            reason_text = str(reason or "unknown")
            reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
    return {
        "schema": "tokenclaw.routing_experiment_feedback_queue_status.v1",
        "queue_rows": row_count,
        "status_breakdown": breakdown_from_counts(status_counts),
        "endpoint_breakdown": breakdown_from_counts(endpoint_counts),
        "outcome_status_breakdown": breakdown_from_counts(outcome_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "payload_json_included": False,
    }


def _codex_app_canary_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.codex_app_canary_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    queue_state_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    rule_candidate_counts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        payload = _safe_payload_json(row)
        if payload.get("schema") != "tokenclaw.codex_app_canary_lifecycle_feedback.v1":
            continue
        row_count += 1
        raw_status = str(row.get("status") or "unknown")
        if raw_status == "sent":
            queue_state = "sent"
        elif raw_status in {"queued", "sending", "retryable-error"}:
            queue_state = "pending"
        elif raw_status in {"error", "dropped-after-limit"}:
            queue_state = "error"
        else:
            queue_state = raw_status
        queue_state_counts[queue_state] = queue_state_counts.get(queue_state, 0) + 1
        action = str(payload.get("action_family") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        outcome = payload.get("outcome") if isinstance(payload.get("outcome"), dict) else {}
        outcome_status = str(outcome.get("status") or payload.get("status") or "unknown")
        outcome_counts[outcome_status] = outcome_counts.get(outcome_status, 0) + 1
        cohort = str(payload.get("canary_cohort") or "unknown")
        cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
        rule_id = str(payload.get("rule_id") or payload.get("policy_id") or "unknown")
        candidate_id = str(payload.get("candidate_id") or payload.get("policy_id") or "unknown")
        rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
        candidate_counts[candidate_id] = candidate_counts.get(candidate_id, 0) + 1
        key = (rule_id, candidate_id)
        bucket = rule_candidate_counts.setdefault(
            key,
            {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "queue_rows": 0,
                "action_family_breakdown": {},
                "queue_state_breakdown": {},
                "outcome_status_breakdown": {},
                "canary_cohort_breakdown": {},
            },
        )
        bucket["queue_rows"] += 1
        for target, value in (
            ("action_family_breakdown", action),
            ("queue_state_breakdown", queue_state),
            ("outcome_status_breakdown", outcome_status),
            ("canary_cohort_breakdown", cohort),
        ):
            counts = bucket[target]
            counts[value] = counts.get(value, 0) + 1
    rule_candidate_breakdown: list[dict[str, Any]] = []
    for bucket in rule_candidate_counts.values():
        item = dict(bucket)
        for key in (
            "action_family_breakdown",
            "queue_state_breakdown",
            "outcome_status_breakdown",
            "canary_cohort_breakdown",
        ):
            item[key] = breakdown_from_counts(item[key])
        rule_candidate_breakdown.append(item)
    rule_candidate_breakdown.sort(
        key=lambda item: (-int(item["queue_rows"]), str(item["rule_id"]), str(item["candidate_id"]))
    )
    return {
        "schema": "tokenclaw.codex_app_canary_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "action_family_breakdown": breakdown_from_counts(action_counts),
        "outcome_status_breakdown": breakdown_from_counts(outcome_counts),
        "canary_cohort_breakdown": breakdown_from_counts(cohort_counts),
        "rule_id_breakdown": breakdown_from_counts(rule_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "rule_candidate_breakdown": rule_candidate_breakdown,
        "payload_json_included": False,
    }


def _queue_state(status: Any) -> str:
    raw_status = str(status or "unknown")
    if raw_status == "sent":
        return "sent"
    if raw_status in {"queued", "sending", "retryable-error"}:
        return "pending"
    if raw_status in {"error", "dropped-after-limit", "expired"}:
        return "error"
    return raw_status


def _normalize_feedback_dimension(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text or default


def _collect_action_family(value: Any, families: set[str]) -> None:
    family = _normalize_feedback_dimension(value, "")
    if family:
        families.add(family)


def _inferred_action_family(payload: dict[str, Any], source_surface: Any) -> str | None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            source_surface,
            payload.get("schema"),
            payload.get("event_type"),
            metadata.get("schema"),
        )
    )
    if "routing" in haystack:
        return "routing"
    if "cache" in haystack:
        return "cache"
    if any(token in haystack for token in ("crunch", "compaction", "summary", "dedup", "scaffold", "thinking")):
        return "crunch"
    return None


def _payload_action_families(payload: dict[str, Any], *, source_surface: Any = None) -> list[str]:
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
        inferred = _inferred_action_family(payload, source_surface)
        if inferred:
            families.add(inferred)
    return sorted(families) or ["unknown"]


def _add_count(counts: dict[str, int], value: Any, increment: int = 1) -> None:
    key = str(value or "unknown")
    counts[key] = counts.get(key, 0) + int(increment or 0)


def _breakdown_items(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        count = int(item.get("count") or 0)
        if count:
            _add_count(counts, value, count)
    return counts


def _numeric_sum(source: dict[str, Any], keys: tuple[str, ...]) -> int:
    total = 0
    for key in keys:
        try:
            total += int(source.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _routing_promotion_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.optimization_promotion_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    action_count = 0
    queue_state_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    action_type_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    model_pair_counts: dict[str, int] = {}
    policy_source_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    error_bucket_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    candidate_status: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = _safe_payload_json(row)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if metadata.get("schema") != "tokenclaw.optimization_promotion_lifecycle_feedback.v1":
            continue
        snapshots = [
            item
            for item in metadata.get("action_snapshots") or []
            if isinstance(item, dict) and str(item.get("policy_section") or "") == "routing"
        ]
        if not snapshots:
            continue
        row_count += 1
        queue_state = _queue_state(row.get("status"))
        event_type = str(payload.get("event_type") or metadata.get("command") or "unknown")
        _add_count(queue_state_counts, queue_state)
        _add_count(event_counts, event_type)
        for snapshot in snapshots:
            action_count += 1
            candidate_id = str(snapshot.get("target_candidate_id") or "unknown")
            rule_id = str(snapshot.get("target_rule_id") or "unknown")
            action_type = str(snapshot.get("action_type") or "unknown")
            policy_source = str(snapshot.get("policy_source") or "unknown")
            model_pair = str(snapshot.get("model_family_pair") or "unknown")
            source = str(snapshot.get("source_surface") or "unknown")
            verdict = str(snapshot.get("next_step_verdict") or snapshot.get("status") or metadata.get("local_result_status") or "unknown")

            _add_count(action_type_counts, action_type)
            _add_count(outcome_counts, verdict)
            _add_count(candidate_counts, candidate_id)
            _add_count(rule_counts, rule_id)
            _add_count(source_counts, source)
            _add_count(model_pair_counts, model_pair)
            _add_count(policy_source_counts, policy_source)
            actual_counts = snapshot.get("actual_cohort_counts") if isinstance(snapshot.get("actual_cohort_counts"), dict) else {}
            projected_counts = snapshot.get("projected_cohort_counts") if isinstance(snapshot.get("projected_cohort_counts"), dict) else {}
            for cohort_key in ("canary_applied", "canary_holdout", "skipped", "bypassed_or_disabled", "safety_stopped"):
                count = int(actual_counts.get(cohort_key) or 0)
                if count:
                    _add_count(cohort_counts, cohort_key, count)
            for bucket, count in _breakdown_items(snapshot.get("error_buckets")).items():
                _add_count(error_bucket_counts, bucket, count)
            for reason in snapshot.get("next_step_reason_codes") or []:
                _add_count(reason_counts, reason)
            for bucket, count in _breakdown_items(snapshot.get("reason_buckets")).items():
                _add_count(reason_counts, bucket, count)

            candidate = candidate_status.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "queue_rows": 0,
                    "action_count": 0,
                    "rule_id_breakdown": {},
                    "queue_state_breakdown": {},
                    "event_type_breakdown": {},
                    "action_type_breakdown": {},
                    "outcome_status_breakdown": {},
                    "source_surface_breakdown": {},
                    "model_family_pair_breakdown": {},
                    "policy_source_breakdown": {},
                    "cohort_count_breakdown": {},
                    "error_bucket_breakdown": {},
                    "reason_code_breakdown": {},
                    "observed_savings_usd": 0.0,
                    "safety_stopped_count": 0,
                    "applied_minus_holdout_error_rate": None,
                    "applied_minus_holdout_retry_rate": None,
                    "applied_minus_holdout_latency_avg_ms": None,
                    "payload_json_included": False,
                },
            )
            candidate["queue_rows"] += 1
            candidate["action_count"] += 1
            for key, value in (
                ("rule_id_breakdown", rule_id),
                ("queue_state_breakdown", queue_state),
                ("event_type_breakdown", event_type),
                ("action_type_breakdown", action_type),
                ("outcome_status_breakdown", verdict),
                ("source_surface_breakdown", source),
                ("model_family_pair_breakdown", model_pair),
                ("policy_source_breakdown", policy_source),
            ):
                _add_count(candidate[key], value)
            for cohort_key in ("canary_applied", "canary_holdout", "skipped", "bypassed_or_disabled", "safety_stopped"):
                count = int(actual_counts.get(cohort_key) or projected_counts.get(cohort_key) or 0)
                if count:
                    _add_count(candidate["cohort_count_breakdown"], cohort_key, count)
            for bucket, count in _breakdown_items(snapshot.get("error_buckets")).items():
                _add_count(candidate["error_bucket_breakdown"], bucket, count)
            for reason in snapshot.get("next_step_reason_codes") or []:
                _add_count(candidate["reason_code_breakdown"], reason)
            for bucket, count in _breakdown_items(snapshot.get("reason_buckets")).items():
                _add_count(candidate["reason_code_breakdown"], bucket, count)
            try:
                candidate["observed_savings_usd"] += float(snapshot.get("observed_savings_usd") or 0.0)
            except (TypeError, ValueError):
                pass
            candidate["safety_stopped_count"] += _numeric_sum(actual_counts, ("safety_stopped",))
            for source_key, target_key in (
                ("error_rate_delta", "applied_minus_holdout_error_rate"),
                ("retry_rate_delta", "applied_minus_holdout_retry_rate"),
                ("latency_avg_delta_ms", "applied_minus_holdout_latency_avg_ms"),
            ):
                if snapshot.get(source_key) is not None:
                    candidate[target_key] = snapshot.get(source_key)

    candidate_breakdown: list[dict[str, Any]] = []
    for bucket in candidate_status.values():
        item = dict(bucket)
        for key in (
            "rule_id_breakdown",
            "queue_state_breakdown",
            "event_type_breakdown",
            "action_type_breakdown",
            "outcome_status_breakdown",
            "source_surface_breakdown",
            "model_family_pair_breakdown",
            "policy_source_breakdown",
            "cohort_count_breakdown",
            "error_bucket_breakdown",
            "reason_code_breakdown",
        ):
            item[key] = breakdown_from_counts(item[key])
        item["observed_savings_usd"] = round(float(item["observed_savings_usd"]), 8)
        candidate_breakdown.append(item)
    candidate_breakdown.sort(key=lambda item: (-int(item["action_count"]), str(item["candidate_id"])))

    return {
        "schema": "tokenclaw.routing_promotion_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "action_count": action_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "event_type_breakdown": breakdown_from_counts(event_counts),
        "action_type_breakdown": breakdown_from_counts(action_type_counts),
        "outcome_status_breakdown": breakdown_from_counts(outcome_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "rule_id_breakdown": breakdown_from_counts(rule_counts),
        "source_surface_breakdown": breakdown_from_counts(source_counts),
        "model_family_pair_breakdown": breakdown_from_counts(model_pair_counts),
        "policy_source_breakdown": breakdown_from_counts(policy_source_counts),
        "cohort_count_breakdown": breakdown_from_counts(cohort_counts),
        "error_bucket_breakdown": breakdown_from_counts(error_bucket_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "candidate_breakdown": candidate_breakdown,
        "payload_json_included": False,
    }


def _terminal_output_compaction_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.terminal_output_compaction_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    action_count = 0
    queue_state_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    lifecycle_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    candidate_status: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = _safe_payload_json(row)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if metadata.get("schema") != "tokenclaw.terminal_output_compaction_lifecycle_feedback.v1":
            continue
        snapshots = [item for item in metadata.get("action_snapshots") or [] if isinstance(item, dict)]
        if not snapshots:
            continue
        row_count += 1
        queue_state = str(row.get("status") or "unknown")
        event_type = str(payload.get("event_type") or metadata.get("event_type") or metadata.get("command") or "unknown")
        _add_count(queue_state_counts, queue_state)
        _add_count(event_counts, event_type)
        for snapshot in snapshots:
            action_count += 1
            candidate_id = str(snapshot.get("candidate_id") or "unknown")
            rule_id = str(snapshot.get("rule_id") or "unknown")
            lifecycle_status = str(snapshot.get("lifecycle_status") or snapshot.get("decision_status") or event_type or "unknown")
            _add_count(lifecycle_counts, lifecycle_status)
            _add_count(candidate_counts, candidate_id)
            _add_count(rule_counts, rule_id)
            for key, count in (snapshot.get("actual_cohort_counts") or snapshot.get("cohort_counts") or {}).items():
                _add_count(cohort_counts, key, int(count or 0))
            for reason in snapshot.get("reason_codes") or snapshot.get("blockers") or []:
                _add_count(reason_counts, reason)

            candidate = candidate_status.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "queue_rows": 0,
                    "action_count": 0,
                    "rule_id_breakdown": {},
                    "queue_state_breakdown": {},
                    "event_type_breakdown": {},
                    "lifecycle_status_breakdown": {},
                    "cohort_count_breakdown": {},
                    "reason_code_breakdown": {},
                    "net_savings_usd": 0.0,
                    "projected_saved_tokens": 0,
                    "payload_json_included": False,
                },
            )
            candidate["queue_rows"] += 1
            candidate["action_count"] += 1
            for key, value in (
                ("rule_id_breakdown", rule_id),
                ("queue_state_breakdown", queue_state),
                ("event_type_breakdown", event_type),
                ("lifecycle_status_breakdown", lifecycle_status),
            ):
                _add_count(candidate[key], value)
            for cohort_key, count in (snapshot.get("actual_cohort_counts") or snapshot.get("cohort_counts") or {}).items():
                _add_count(candidate["cohort_count_breakdown"], cohort_key, int(count or 0))
            for reason in snapshot.get("reason_codes") or snapshot.get("blockers") or []:
                _add_count(candidate["reason_code_breakdown"], reason)
            try:
                candidate["net_savings_usd"] += float(snapshot.get("net_savings_usd") or 0.0)
            except (TypeError, ValueError):
                pass
            candidate["projected_saved_tokens"] += int(snapshot.get("projected_saved_tokens") or 0)

    candidate_breakdown: list[dict[str, Any]] = []
    for bucket in candidate_status.values():
        item = dict(bucket)
        for key in (
            "rule_id_breakdown",
            "queue_state_breakdown",
            "event_type_breakdown",
            "lifecycle_status_breakdown",
            "cohort_count_breakdown",
            "reason_code_breakdown",
        ):
            item[key] = breakdown_from_counts(item[key])
        item["net_savings_usd"] = round(float(item["net_savings_usd"]), 8)
        candidate_breakdown.append(item)
    candidate_breakdown.sort(key=lambda item: (-int(item["action_count"]), str(item["candidate_id"])))

    return {
        "schema": "tokenclaw.terminal_output_compaction_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "action_count": action_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "event_type_breakdown": breakdown_from_counts(event_counts),
        "lifecycle_status_breakdown": breakdown_from_counts(lifecycle_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "rule_id_breakdown": breakdown_from_counts(rule_counts),
        "cohort_count_breakdown": breakdown_from_counts(cohort_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "candidate_breakdown": candidate_breakdown,
        "payload_json_included": False,
    }


def _codex_terminal_transcript_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.codex_terminal_transcript_compaction_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    action_count = 0
    queue_state_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    lifecycle_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    candidate_status: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = _safe_payload_json(row)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if metadata.get("schema") != "tokenclaw.codex_terminal_transcript_compaction_lifecycle_feedback.v1":
            continue
        snapshots = [item for item in metadata.get("action_snapshots") or [] if isinstance(item, dict)]
        if not snapshots:
            continue
        row_count += 1
        queue_state = str(row.get("status") or "unknown")
        event_type = str(payload.get("event_type") or metadata.get("event_type") or metadata.get("command") or "unknown")
        _add_count(queue_state_counts, queue_state)
        _add_count(event_counts, event_type)
        for snapshot in snapshots:
            action_count += 1
            candidate_id = str(snapshot.get("candidate_id") or "unknown")
            rule_id = str(snapshot.get("rule_id") or "unknown")
            lifecycle_status = str(snapshot.get("lifecycle_status") or snapshot.get("decision_status") or event_type or "unknown")
            _add_count(lifecycle_counts, lifecycle_status)
            _add_count(candidate_counts, candidate_id)
            _add_count(rule_counts, rule_id)
            for key, count in (snapshot.get("actual_cohort_counts") or snapshot.get("cohort_counts") or {}).items():
                _add_count(cohort_counts, key, int(count or 0))
            for reason in snapshot.get("reason_codes") or snapshot.get("blockers") or []:
                _add_count(reason_counts, reason)

            candidate = candidate_status.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "queue_rows": 0,
                    "action_count": 0,
                    "rule_id_breakdown": {},
                    "queue_state_breakdown": {},
                    "event_type_breakdown": {},
                    "lifecycle_status_breakdown": {},
                    "cohort_count_breakdown": {},
                    "reason_code_breakdown": {},
                    "net_savings_usd": 0.0,
                    "payload_json_included": False,
                },
            )
            candidate["queue_rows"] += 1
            candidate["action_count"] += 1
            for key, value in (
                ("rule_id_breakdown", rule_id),
                ("queue_state_breakdown", queue_state),
                ("event_type_breakdown", event_type),
                ("lifecycle_status_breakdown", lifecycle_status),
            ):
                _add_count(candidate[key], value)
            for cohort_key, count in (snapshot.get("actual_cohort_counts") or snapshot.get("cohort_counts") or {}).items():
                _add_count(candidate["cohort_count_breakdown"], cohort_key, int(count or 0))
            for reason in snapshot.get("reason_codes") or snapshot.get("blockers") or []:
                _add_count(candidate["reason_code_breakdown"], reason)
            try:
                candidate["net_savings_usd"] += float(snapshot.get("net_savings_usd") or 0.0)
            except (TypeError, ValueError):
                pass

    candidate_breakdown: list[dict[str, Any]] = []
    for bucket in candidate_status.values():
        item = dict(bucket)
        for key in (
            "rule_id_breakdown",
            "queue_state_breakdown",
            "event_type_breakdown",
            "lifecycle_status_breakdown",
            "cohort_count_breakdown",
            "reason_code_breakdown",
        ):
            item[key] = breakdown_from_counts(item[key])
        item["net_savings_usd"] = round(float(item["net_savings_usd"]), 8)
        candidate_breakdown.append(item)
    candidate_breakdown.sort(key=lambda item: (-int(item["action_count"]), str(item["candidate_id"])))

    return {
        "schema": "tokenclaw.codex_terminal_transcript_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "action_count": action_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "event_type_breakdown": breakdown_from_counts(event_counts),
        "lifecycle_status_breakdown": breakdown_from_counts(lifecycle_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "rule_id_breakdown": breakdown_from_counts(rule_counts),
        "cohort_count_breakdown": breakdown_from_counts(cohort_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "candidate_breakdown": candidate_breakdown,
        "payload_json_included": False,
    }


def _openai_optimization_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.openai_optimization_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    event_count = 0
    queue_state_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    family_cohort_counts: dict[str, int] = {}
    status_bucket_counts: dict[str, int] = {}
    retry_bucket_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    rule_counts: dict[str, int] = {}
    family_breakdown: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = _safe_payload_json(row)
        if payload.get("schema") != "tokenclaw.openai_optimization_lifecycle_feedback.v1":
            continue
        events = [item for item in payload.get("family_events") or [] if isinstance(item, dict)]
        if not events:
            continue
        row_count += 1
        queue_state = _queue_state(row.get("status"))
        _add_count(queue_state_counts, queue_state)
        _add_count(status_bucket_counts, payload.get("status_bucket"))
        _add_count(retry_bucket_counts, payload.get("retry_bucket"))
        for event in events:
            event_count += 1
            family = str(event.get("action_family") or "unknown")
            cohort = str(event.get("cohort") or "unknown")
            _add_count(family_counts, family)
            _add_count(cohort_counts, cohort)
            _add_count(family_cohort_counts, f"{family}:{cohort}")
            if event.get("candidate_id"):
                _add_count(candidate_counts, event.get("candidate_id"))
            if event.get("rule_id"):
                _add_count(rule_counts, event.get("rule_id"))
            for reason in event.get("reason_codes") or []:
                _add_count(reason_counts, reason)
            family_item = family_breakdown.setdefault(
                family,
                {
                    "action_family": family,
                    "event_count": 0,
                    "cohort_breakdown": {},
                    "queue_state_breakdown": {},
                    "status_bucket_breakdown": {},
                    "retry_bucket_breakdown": {},
                    "reason_code_breakdown": {},
                    "candidate_id_breakdown": {},
                    "rule_id_breakdown": {},
                    "payload_json_included": False,
                },
            )
            family_item["event_count"] += 1
            for key, value in (
                ("cohort_breakdown", cohort),
                ("queue_state_breakdown", queue_state),
                ("status_bucket_breakdown", payload.get("status_bucket")),
                ("retry_bucket_breakdown", payload.get("retry_bucket")),
            ):
                _add_count(family_item[key], value)
            if event.get("candidate_id"):
                _add_count(family_item["candidate_id_breakdown"], event.get("candidate_id"))
            if event.get("rule_id"):
                _add_count(family_item["rule_id_breakdown"], event.get("rule_id"))
            for reason in event.get("reason_codes") or []:
                _add_count(family_item["reason_code_breakdown"], reason)

    family_items: list[dict[str, Any]] = []
    for item in family_breakdown.values():
        converted = dict(item)
        for key in (
            "cohort_breakdown",
            "queue_state_breakdown",
            "status_bucket_breakdown",
            "retry_bucket_breakdown",
            "reason_code_breakdown",
            "candidate_id_breakdown",
            "rule_id_breakdown",
        ):
            converted[key] = breakdown_from_counts(converted[key])
        family_items.append(converted)
    family_items.sort(key=lambda item: (-int(item["event_count"]), str(item["action_family"])))

    return {
        "schema": "tokenclaw.openai_optimization_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "family_event_count": event_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "action_family_breakdown": breakdown_from_counts(family_counts),
        "cohort_breakdown": breakdown_from_counts(cohort_counts),
        "family_cohort_breakdown": breakdown_from_counts(family_cohort_counts),
        "status_bucket_breakdown": breakdown_from_counts(status_bucket_counts),
        "retry_bucket_breakdown": breakdown_from_counts(retry_bucket_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "rule_id_breakdown": breakdown_from_counts(rule_counts),
        "family_breakdown": family_items,
        "payload_json_included": False,
    }


def _optimization_coordinator_lifecycle_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.optimization_coordinator_lifecycle_feedback.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    public_rows = (
        store.managed_outcome_feedback_rows(source_surface=source_surface, limit=10000)
        if hasattr(store, "managed_outcome_feedback_rows")
        else []
    )
    row_count = 0
    family_event_count = 0
    queue_state_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    suppressed_counts: dict[str, int] = {}
    lifecycle_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    status_bucket_counts: dict[str, int] = {}
    retry_bucket_counts: dict[str, int] = {}
    error_bucket_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    family_breakdown: dict[str, dict[str, Any]] = {}

    for row in rows:
        payload = _safe_payload_json(row)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if metadata.get("schema") != "tokenclaw.optimization_coordinator_lifecycle_feedback.v1":
            continue
        events = [item for item in metadata.get("family_events") or [] if isinstance(item, dict)]
        if not events:
            continue
        row_count += 1
        queue_state = _queue_state(row.get("status"))
        event_type = str(payload.get("event_type") or metadata.get("event_type") or "unknown")
        event_phase = str(metadata.get("event_phase") or "unknown")
        selected_family = str(metadata.get("selected_family") or "none")
        _add_count(queue_state_counts, queue_state)
        _add_count(event_counts, event_type)
        _add_count(phase_counts, event_phase)
        _add_count(selected_counts, selected_family)
        _add_count(status_bucket_counts, metadata.get("status_bucket"))
        _add_count(retry_bucket_counts, metadata.get("retry_bucket"))
        _add_count(error_bucket_counts, metadata.get("error_bucket"))
        for family in metadata.get("suppressed_families") or []:
            _add_count(suppressed_counts, family)
        for reason in metadata.get("reason_codes") or []:
            _add_count(reason_counts, reason)
        for event in events:
            family_event_count += 1
            family = str(event.get("action_family") or "unknown")
            lifecycle = str(event.get("lifecycle_status") or event.get("cohort") or "unknown")
            _add_count(lifecycle_counts, lifecycle)
            if event.get("candidate_id"):
                _add_count(candidate_counts, event.get("candidate_id"))
            for reason in event.get("reason_codes") or []:
                _add_count(reason_counts, reason)
            family_item = family_breakdown.setdefault(
                family,
                {
                    "action_family": family,
                    "event_count": 0,
                    "queue_state_breakdown": {},
                    "event_type_breakdown": {},
                    "event_phase_breakdown": {},
                    "lifecycle_status_breakdown": {},
                    "reason_code_breakdown": {},
                    "candidate_id_breakdown": {},
                    "payload_json_included": False,
                },
            )
            family_item["event_count"] += 1
            for key, value in (
                ("queue_state_breakdown", queue_state),
                ("event_type_breakdown", event_type),
                ("event_phase_breakdown", event_phase),
                ("lifecycle_status_breakdown", lifecycle),
            ):
                _add_count(family_item[key], value)
            if event.get("candidate_id"):
                _add_count(family_item["candidate_id_breakdown"], event.get("candidate_id"))
            for reason in event.get("reason_codes") or []:
                _add_count(family_item["reason_code_breakdown"], reason)

    retryable_failures = sum(
        1
        for row in public_rows
        if row.get("source_surface") == "optimization_coordinator_lifecycle" and row.get("status") == "retryable-error"
    )
    dropped_privacy_violations = sum(
        1
        for row in public_rows
        if row.get("source_surface") == "optimization_coordinator_lifecycle"
        and row.get("status") == "dropped-after-limit"
        and "unsafe" in str(row.get("last_error") or "").lower()
    )

    family_items: list[dict[str, Any]] = []
    for item in family_breakdown.values():
        converted = dict(item)
        for key in (
            "queue_state_breakdown",
            "event_type_breakdown",
            "event_phase_breakdown",
            "lifecycle_status_breakdown",
            "reason_code_breakdown",
            "candidate_id_breakdown",
        ):
            converted[key] = breakdown_from_counts(converted[key])
        family_items.append(converted)
    family_items.sort(key=lambda item: (-int(item["event_count"]), str(item["action_family"])))

    return {
        "schema": "tokenclaw.optimization_coordinator_lifecycle_queue_status.v1",
        "queue_rows": row_count,
        "family_event_count": family_event_count,
        "retryable_failures": retryable_failures,
        "dropped_privacy_violations": dropped_privacy_violations,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "event_type_breakdown": breakdown_from_counts(event_counts),
        "event_phase_breakdown": breakdown_from_counts(phase_counts),
        "selected_family_breakdown": breakdown_from_counts(selected_counts),
        "suppressed_family_breakdown": breakdown_from_counts(suppressed_counts),
        "lifecycle_status_breakdown": breakdown_from_counts(lifecycle_counts),
        "reason_code_breakdown": breakdown_from_counts(reason_counts),
        "status_bucket_breakdown": breakdown_from_counts(status_bucket_counts),
        "retry_bucket_breakdown": breakdown_from_counts(retry_bucket_counts),
        "error_bucket_breakdown": breakdown_from_counts(error_bucket_counts),
        "candidate_id_breakdown": breakdown_from_counts(candidate_counts),
        "family_breakdown": family_items,
        "payload_json_included": False,
    }


def _post_promotion_action_outcome_status(
    store: Any,
    *,
    source_surface: str | None,
) -> dict[str, Any]:
    rows = (
        store.managed_outcome_feedback_payload_rows(
            source_surface=source_surface, limit=10000, payload_contains="tokenclaw.promotion_blocker_action_outcome_rollup_ingest.v1"
        )
        if hasattr(store, "managed_outcome_feedback_payload_rows")
        else []
    )
    row_count = 0
    rollup_count = 0
    outcome_count = 0
    queue_state_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    policy_source_counts: dict[str, int] = {}
    for row in rows:
        payload = _safe_payload_json(row)
        if payload.get("schema") != "tokenclaw.promotion_blocker_action_outcome_rollup_ingest.v1":
            continue
        row_count += 1
        queue_state = _queue_state(row.get("status"))
        queue_state_counts[queue_state] = queue_state_counts.get(queue_state, 0) + 1
        rollups = payload.get("rollups") if isinstance(payload.get("rollups"), list) else []
        rollup_count += len([item for item in rollups if isinstance(item, dict)])
        for item in rollups:
            if not isinstance(item, dict):
                continue
            count = int(item.get("outcome_count") or item.get("row_count") or 1)
            outcome_count += count
            for target, value in (
                (action_counts, item.get("local_action_family") or item.get("action_family")),
                (outcome_counts, item.get("outcome_status")),
                (next_action_counts, item.get("next_action")),
                (policy_source_counts, item.get("policy_source")),
            ):
                key = str(value or "unknown")
                target[key] = target.get(key, 0) + count
    return {
        "schema": "tokenclaw.post_promotion_action_outcome_queue_status.v1",
        "queue_rows": row_count,
        "rollup_count": rollup_count,
        "outcome_count": outcome_count,
        "queue_state_breakdown": breakdown_from_counts(queue_state_counts),
        "local_action_family_breakdown": breakdown_from_counts(action_counts),
        "outcome_status_breakdown": breakdown_from_counts(outcome_counts),
        "next_action_breakdown": breakdown_from_counts(next_action_counts),
        "policy_source_breakdown": breakdown_from_counts(policy_source_counts),
        "payload_json_included": False,
    }


def public_feedback_row(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    optimization_unit_id = row.get("optimization_unit_id")
    if optimization_unit_id in (0, "0"):
        optimization_unit_id = None
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": optimization_unit_id,
        "status": row.get("status"),
        "attempts": row.get("attempts") or 0,
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "sent_at": row.get("sent_at"),
        "age_seconds": seconds_since(row.get("created_at"), now),
        "payload_included": False,
    }


def managed_feedback_config() -> dict[str, Any]:
    from tokenclaw import recommendations

    return {
        "enabled": recommendations.recommendations_enabled(),
        "server_url": redact_url(recommendations.recommendation_server_url()),
        "server_configured": recommendations.recommendation_server_configured(),
        "timeout_seconds": recommendations.recommendation_timeout_seconds(),
        "failure_mode": recommendations.recommendation_failure_mode(),
        "queue_max_attempts": recommendations.outcome_feedback_queue_max_attempts(),
        "queue_retry_delay_seconds": recommendations.outcome_feedback_queue_retry_delay_seconds(),
        "auth_configured": recommendations.managed_auth_configured(),
        "api_key_value_included": False,
    }


def _all_feedback_rows(store: Any, *, source_surface: str | None, limit: int = 10000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not hasattr(store, "managed_outcome_feedback_freshness_rows"):
        if hasattr(store, "managed_outcome_feedback_rows"):
            rows = store.managed_outcome_feedback_rows(source_surface=source_surface, limit=limit)
        return rows
    rows = store.managed_outcome_feedback_freshness_rows(limit=limit)
    if source_surface:
        rows = [row for row in rows if row.get("source_surface") == source_surface]
    return rows


def _feedback_row_families(row: dict[str, Any]) -> list[str]:
    return _payload_action_families(
        _safe_payload_json(row),
        source_surface=row.get("source_surface"),
    )


def _family_freshness_summary(rows: list[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        due = False
        if status in {"queued", "retryable-error"}:
            next_attempt = parse_utc_iso(row.get("next_attempt_at"))
            due = next_attempt is not None and next_attempt <= now
        for family in _feedback_row_families(row):
            bucket = buckets.setdefault(
                family,
                {
                    "action_family": family,
                    "queued": 0,
                    "retryable_error": 0,
                    "sent": 0,
                    "dropped": 0,
                    "expired": 0,
                    "due": 0,
                    "oldest_pending_age_seconds": None,
                    "newest_sent_at": None,
                    "payload_json_included": False,
                },
            )
            if status == "queued":
                bucket["queued"] += 1
            elif status == "retryable-error":
                bucket["retryable_error"] += 1
            elif status == "sent":
                bucket["sent"] += 1
            elif status == "expired":
                bucket["expired"] += 1
                bucket["dropped"] += 1
            elif status in {"dropped-after-limit", "error"}:
                bucket["dropped"] += 1
            if due:
                bucket["due"] += 1
            if status in {"queued", "retryable-error"}:
                age = seconds_since(row.get("created_at"), now)
                if age is not None and (
                    bucket["oldest_pending_age_seconds"] is None
                    or age > bucket["oldest_pending_age_seconds"]
                ):
                    bucket["oldest_pending_age_seconds"] = age
            if status == "sent" and row.get("sent_at"):
                if bucket["newest_sent_at"] is None or str(row.get("sent_at")) > str(bucket["newest_sent_at"]):
                    bucket["newest_sent_at"] = row.get("sent_at")
    return sorted(
        buckets.values(),
        key=lambda item: (-int(item["due"]), -int(item["queued"] + item["retryable_error"]), str(item["action_family"])),
    )


def _row_due(row: dict[str, Any], *, now: datetime) -> bool:
    if row.get("status") not in {"queued", "retryable-error"}:
        return False
    next_attempt = parse_utc_iso(row.get("next_attempt_at"))
    return bool(next_attempt is not None and next_attempt <= now)


def _row_enabled_for_mode(row: dict[str, Any], enabled_families: dict[str, bool]) -> bool:
    families = _feedback_row_families(row)
    known = [family for family in families if family in enabled_families]
    if not known:
        return True
    return any(enabled_families.get(family) for family in known)


def _select_activation_feedback_queue_ids(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    limit: int,
    per_family_limit: int,
    enabled_families: dict[str, bool],
) -> list[str]:
    due_rows = [
        row
        for row in rows
        if row.get("id") and _row_due(row, now=now) and _row_enabled_for_mode(row, enabled_families)
    ]
    due_rows.sort(
        key=lambda row: (
            parse_utc_iso(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    selected: list[str] = []
    selected_set: set[str] = set()
    per_family_counts: dict[str, int] = {}
    capped = max(1, min(int(limit or 1), 100))
    family_cap = max(1, min(int(per_family_limit or 1), capped))
    preferred_families = [family for family in ("routing", "crunch", "cache") if enabled_families.get(family)]
    preferred_families.append("unknown")
    for family in preferred_families:
        for row in due_rows:
            if len(selected) >= capped:
                return selected
            queue_id = str(row.get("id"))
            if queue_id in selected_set:
                continue
            families = _feedback_row_families(row)
            if family not in families:
                continue
            if per_family_counts.get(family, 0) >= family_cap:
                continue
            selected.append(queue_id)
            selected_set.add(queue_id)
            per_family_counts[family] = per_family_counts.get(family, 0) + 1
    for row in due_rows:
        if len(selected) >= capped:
            break
        queue_id = str(row.get("id"))
        if queue_id in selected_set:
            continue
        selected.append(queue_id)
        selected_set.add(queue_id)
    return selected


async def managed_feedback_activation_drain_result(
    store: Any,
    *,
    limit: int = 25,
    per_family_limit: int = 5,
    max_age_seconds: int = 7 * 24 * 60 * 60,
    source_surface: str | None = None,
    stale_sending_after_seconds: int = 600,
) -> dict[str, Any]:
    from tokenclaw import recommendations

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    capped = max(1, min(int(limit or 1), 100))
    family_cap = max(1, min(int(per_family_limit or 1), capped))
    product_mode = managed_product_mode()
    before_rows = _all_feedback_rows(store, source_surface=source_surface, limit=10000)
    before_summary = managed_feedback_status_result(store, source_surface=source_surface, sample_limit=capped)["summary"]
    if product_mode.local_rules_only or not product_mode.server_calls_enabled:
        return {
            "schema": "tokenclaw.managed_feedback_activation_drain.v1",
            "status": "disabled",
            "reason": product_mode.reason or "managed-server-calls-disabled",
            "generated_at": now_iso,
            "limit": capped,
            "per_family_limit": family_cap,
            "source_surface": source_surface,
            "product_mode": product_mode.public_meta(),
            "before": before_summary,
            "after": before_summary,
            "family_freshness_before": _family_freshness_summary(before_rows, now=now),
            "family_freshness_after": _family_freshness_summary(before_rows, now=now),
            "expired": 0,
            "exhausted_dropped": 0,
            "recovered_stale_sending": 0,
            "selected_queue_ids": [],
            "results": [],
            "result_breakdown": [],
            "managed_server_calls_made": False,
            "privacy": {
                "metadata_only": True,
                "payload_json_included": False,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "secrets_included": False,
            },
        }
    if not recommendations.recommendations_enabled():
        return {
            "schema": "tokenclaw.managed_feedback_activation_drain.v1",
            "status": "disabled",
            "reason": "managed-feedback-disabled",
            "generated_at": now_iso,
            "limit": capped,
            "per_family_limit": family_cap,
            "source_surface": source_surface,
            "product_mode": product_mode.public_meta(),
            "before": before_summary,
            "after": before_summary,
            "family_freshness_before": _family_freshness_summary(before_rows, now=now),
            "family_freshness_after": _family_freshness_summary(before_rows, now=now),
            "expired": 0,
            "exhausted_dropped": 0,
            "recovered_stale_sending": 0,
            "selected_queue_ids": [],
            "results": [],
            "result_breakdown": [],
            "managed_server_calls_made": False,
            "privacy": {
                "metadata_only": True,
                "payload_json_included": False,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "secrets_included": False,
            },
        }

    recovered = 0
    if hasattr(store, "recover_stale_managed_outcome_feedback_sends"):
        stale_before = (now - timedelta(seconds=max(1, int(stale_sending_after_seconds or 1)))).isoformat()
        recovered = store.recover_stale_managed_outcome_feedback_sends(
            stale_before=stale_before,
            now=now_iso,
        )
    expired = 0
    if max_age_seconds and int(max_age_seconds) > 0 and hasattr(store, "expire_stale_managed_outcome_feedback"):
        cutoff_at = (now - timedelta(seconds=max(1, int(max_age_seconds)))).isoformat()
        expired = store.expire_stale_managed_outcome_feedback(
            cutoff_at=cutoff_at,
            now=now_iso,
            source_surface=source_surface,
        )
    exhausted = 0
    if hasattr(store, "drop_exhausted_managed_outcome_feedback"):
        exhausted = store.drop_exhausted_managed_outcome_feedback(
            max_attempts=recommendations.outcome_feedback_queue_max_attempts(),
            now=now_iso,
            source_surface=source_surface,
        )

    selectable_rows = _all_feedback_rows(store, source_surface=source_surface, limit=10000)
    selected = _select_activation_feedback_queue_ids(
        selectable_rows,
        now=now,
        limit=capped,
        per_family_limit=family_cap,
        enabled_families=product_mode.family_enabled,
    )
    results = await recommendations.flush_selected_outcome_feedback(store, queue_ids=selected)
    result_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        result_counts[status] = result_counts.get(status, 0) + 1
    after_rows = _all_feedback_rows(store, source_surface=source_surface, limit=10000)
    after_summary = managed_feedback_status_result(store, source_surface=source_surface, sample_limit=capped)["summary"]
    return {
        "schema": "tokenclaw.managed_feedback_activation_drain.v1",
        "status": "completed",
        "reason": "ok",
        "generated_at": now_iso,
        "limit": capped,
        "per_family_limit": family_cap,
        "source_surface": source_surface,
        "product_mode": product_mode.public_meta(),
        "before": before_summary,
        "after": after_summary,
        "family_freshness_before": _family_freshness_summary(before_rows, now=now),
        "family_freshness_after": _family_freshness_summary(after_rows, now=now),
        "expired": expired,
        "exhausted_dropped": exhausted,
        "recovered_stale_sending": recovered,
        "selected_queue_ids": selected,
        "results": results,
        "result_breakdown": breakdown_from_counts(result_counts),
        "managed_server_calls_made": bool(results),
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }


def managed_feedback_status_result(
    store: Any,
    *,
    source_surface: str | None,
    sample_limit: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    generated_at = now.isoformat()
    status_counts = {
        str(row.get("status") or "unknown"): int(row.get("count") or 0)
        for row in store.managed_outcome_feedback_summary(source_surface=source_surface)
    } if hasattr(store, "managed_outcome_feedback_summary") else {}
    rows = (
        store.managed_outcome_feedback_rows(source_surface=source_surface, limit=10000)
        if hasattr(store, "managed_outcome_feedback_rows")
        else []
    )
    due_rows = (
        store.due_managed_outcome_feedback(limit=max(1, sample_limit), source_surface=source_surface)
        if hasattr(store, "due_managed_outcome_feedback")
        else []
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_surface") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    pending_rows = [
        row
        for row in rows
        if row.get("status") in {"queued", "retryable-error"}
    ]
    oldest_pending = min(
        pending_rows,
        key=lambda row: parse_utc_iso(row.get("created_at")) or now,
        default=None,
    )
    dropped = status_counts.get("dropped-after-limit", 0)
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "sent": status_counts.get("sent", 0),
        "dropped_after_limit": dropped,
        "error": status_counts.get("error", 0),
        "due": len(due_rows),
        "oldest_pending_age_seconds": seconds_since(oldest_pending.get("created_at"), now) if oldest_pending else None,
        "retry_limit_drops": dropped,
    }
    pattern_evidence = _pattern_evidence_status(store, source_surface=source_surface)
    routing_experiments = _routing_experiment_status(store, source_surface=source_surface)
    codex_app_canaries = _codex_app_canary_lifecycle_status(store, source_surface=source_surface)
    routing_promotion_lifecycle = _routing_promotion_lifecycle_status(store, source_surface=source_surface)
    terminal_output_compaction_lifecycle = _terminal_output_compaction_lifecycle_status(store, source_surface=source_surface)
    codex_terminal_transcript_lifecycle = _codex_terminal_transcript_lifecycle_status(store, source_surface=source_surface)
    openai_optimization_lifecycle = _openai_optimization_lifecycle_status(store, source_surface=source_surface)
    optimization_coordinator_lifecycle = _optimization_coordinator_lifecycle_status(store, source_surface=source_surface)
    post_promotion_action_outcomes = _post_promotion_action_outcome_status(store, source_surface=source_surface)
    return {
        "schema": "tokenclaw.managed_feedback_status.v1",
        "ok": True,
        "generated_at": generated_at,
        "source_surface": source_surface,
        "managed_feedback": managed_feedback_config(),
        "summary": summary,
        "pattern_evidence": pattern_evidence,
        "routing_experiments": routing_experiments,
        "codex_app_canaries": codex_app_canaries,
        "routing_promotion_lifecycle": routing_promotion_lifecycle,
        "terminal_output_compaction_lifecycle": terminal_output_compaction_lifecycle,
        "codex_terminal_transcript_lifecycle": codex_terminal_transcript_lifecycle,
        "openai_optimization_lifecycle": openai_optimization_lifecycle,
        "optimization_coordinator_lifecycle": optimization_coordinator_lifecycle,
        "post_promotion_action_outcomes": post_promotion_action_outcomes,
        "status_breakdown": breakdown_from_counts(status_counts),
        "source_surface_breakdown": breakdown_from_counts(source_counts),
        "oldest_pending": public_feedback_row(oldest_pending, now=now) if oldest_pending else None,
        "due_samples": [
            public_feedback_row(row, now=now)
            for row in due_rows[:max(0, sample_limit)]
        ],
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }


def safe_managed_feedback_flush_result(result: dict[str, Any]) -> dict[str, Any]:
    secret = os.getenv(MANAGED_POLICY_API_KEY_ENV)
    safe = redact_secret(result, secret)
    if isinstance(safe.get("managed_feedback"), dict):
        safe["managed_feedback"]["server_url"] = redact_url(safe["managed_feedback"].get("server_url"))
    for item in safe.get("results", []) if isinstance(safe.get("results"), list) else []:
        if isinstance(item, dict) and "server_url" in item:
            item["server_url"] = redact_url(item.get("server_url"))
    return safe


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local managed outcome feedback queue status")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help="AgentFlow database URL or SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum due queue samples to include, default: 20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    store = open_store_for_db(str(args.db))
    try:
        result = managed_feedback_status_result(
            store,
            source_surface=args.source_surface,
            sample_limit=max(0, min(args.limit, 100)),
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Flush due local managed outcome feedback queue rows in bounded batches")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help="AgentFlow database URL or SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum due rows to flush, default: 5, max: 100.")
    parser.add_argument(
        "--post-promotion-action-outcomes",
        action="store_true",
        help="Build and queue metadata-only post-promotion action outcome rollups before flushing.",
    )
    parser.add_argument(
        "--outcome-limit",
        type=int,
        default=1000,
        help="Maximum local promotion outcome feedback rows to roll up when --post-promotion-action-outcomes is set.",
    )
    parser.add_argument(
        "--activation",
        action="store_true",
        help="Run the managed-mode activation drain: expire stale rows, preserve backoff, and flush due rows by action family.",
    )
    parser.add_argument("--per-family-limit", type=int, default=5, help="Maximum activation-drain rows per action family, default: 5.")
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=7 * 24 * 60 * 60,
        help="Expire pending activation-drain feedback older than this many seconds, default: 7 days.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report due rows without claiming or sending them.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    limit = max(1, min(args.limit, 100))
    store = open_store_for_db(str(args.db))
    try:
        post_promotion_rollups: dict[str, Any] | None = None
        flush_source_surface = args.source_surface
        if args.post_promotion_action_outcomes:
            from tokenclaw.promotion_outcome_feedback import (
                POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE,
                build_post_promotion_action_outcome_rollups,
                queue_post_promotion_action_outcome_rollups,
            )

            flush_source_surface = flush_source_surface or POST_PROMOTION_ACTION_OUTCOME_ROLLUP_SOURCE_SURFACE
            if args.dry_run:
                built = build_post_promotion_action_outcome_rollups(
                    store,
                    limit=max(1, min(args.outcome_limit, 10000)),
                )
                post_promotion_rollups = {
                    "status": "would-queue" if built.get("status") == "ready" else built.get("status"),
                    "reason": "dry-run",
                    "rollup_count": len(built.get("rollups") or []),
                    "payload_included": False,
                    "privacy": built.get("privacy"),
                }
            else:
                post_promotion_rollups = asyncio.run(
                    queue_post_promotion_action_outcome_rollups(
                        store,
                        limit=max(1, min(args.outcome_limit, 10000)),
                        flush_immediately=False,
                    )
                )
        before = managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
        activation_drain: dict[str, Any] | None = None
        if args.activation and not args.dry_run:
            activation_drain = asyncio.run(
                managed_feedback_activation_drain_result(
                    store,
                    limit=limit,
                    per_family_limit=max(1, min(args.per_family_limit, 100)),
                    max_age_seconds=max(0, int(args.max_age_seconds or 0)),
                    source_surface=flush_source_surface,
                )
            )
            raw_results = activation_drain.get("results") if isinstance(activation_drain.get("results"), list) else []
            results = raw_results
            flush_status = str(activation_drain.get("status") or "completed")
            reason = str(activation_drain.get("reason") or "ok")
        elif args.dry_run:
            results = [
                {**row, "status": "would-send"}
                for row in before.get("due_samples", [])
            ]
            flush_status = "dry-run"
            reason = "dry-run"
        else:
            from tokenclaw import recommendations

            if recommendations.recommendations_enabled():
                results = asyncio.run(
                    recommendations.flush_queued_outcome_feedback(
                        store,
                        limit=limit,
                        source_surface=flush_source_surface,
                    )
                )
                flush_status = "completed"
                reason = "ok"
            else:
                results = []
                flush_status = "skipped"
                reason = "managed-feedback-disabled"
        after = managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
    finally:
        store.conn.close()

    result_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        result_counts[status] = result_counts.get(status, 0) + 1
        if (
            post_promotion_rollups
            and item.get("queue_id") == post_promotion_rollups.get("queue_id")
        ):
            post_promotion_rollups["status"] = "flushed" if status == "sent" else status
            post_promotion_rollups["reason"] = item.get("reason") or post_promotion_rollups.get("reason")
            post_promotion_rollups["attempts"] = item.get("attempts", post_promotion_rollups.get("attempts"))
    result = {
        "schema": "tokenclaw.managed_feedback_flush.v1",
        "ok": True,
        "dry_run": bool(args.dry_run),
        "source_surface": args.source_surface,
        "limit": limit,
        "flush": {
            "status": flush_status,
            "reason": reason,
            "attempted": len(results) if not args.dry_run else 0,
            "would_attempt": len(results) if args.dry_run else 0,
            "sent": result_counts.get("sent", 0),
            "retryable_error": result_counts.get("retryable-error", 0),
            "dropped_after_limit": result_counts.get("dropped-after-limit", 0),
        },
        "managed_feedback": managed_feedback_config(),
        "activation_drain": activation_drain,
        "post_promotion_action_outcome_rollups": post_promotion_rollups,
        "before": before["summary"],
        "after": after["summary"],
        "result_breakdown": breakdown_from_counts(result_counts),
        "results": results,
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }
    result = safe_managed_feedback_flush_result(result)
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0
