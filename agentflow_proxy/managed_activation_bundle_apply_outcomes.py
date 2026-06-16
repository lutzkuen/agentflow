from __future__ import annotations

from collections import Counter
from typing import Any

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.policy_events import recent_policy_events
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.managed_activation_bundle_apply_outcomes.v1"
ROW_SCHEMA = "agentflow.managed_activation_bundle_apply_outcome_row.v1"
PRIVACY_SCHEMA = "agentflow.managed_activation_bundle_apply_outcome_privacy.v1"

APPLY_EVENT_SOURCE = "managed-activation-bundle-apply"
SUPPORTED_FAMILIES = ("cache", "crunch")
TARGET_RULE_FILES = {"cache": "cache_rules.yaml", "crunch": "crunch_rules.yaml"}

STATUS_COUNT_FIELD = {
    "applied": "applied_count",
    "skipped": "skipped_count",
    "rolled-back": "rolled_back_count",
    "failed": "failed_count",
    "dry-run": "dry_run_count",
}


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "feature_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "absolute_paths_included": False,
        "policy_file_contents_included": False,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
    }


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback)


def _safe_ref(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return public_id(text, prefix="ref", fallback=None)


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"applied", "apply", "wrote", "written"}:
        return "applied"
    if text in {"planned", "dry-run", "preview"}:
        return "dry-run"
    if text in {"skipped", "skip", "not-selected"}:
        return "skipped"
    if text in {"failed", "fail", "rejected"}:
        return "failed"
    if text in {"rolled-back", "rollback"}:
        return "rolled-back"
    return "unknown"


def _draft_apply_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("action") == "draft-apply"
        and isinstance(event.get("details"), dict)
        and event["details"].get("source") == APPLY_EVENT_SOURCE
    ]


def _rolled_back_sections(events: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_apply_id: dict[str, set[str]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("action") != "rollback" or not event.get("ok"):
            continue
        details = event.get("details")
        if not isinstance(details, dict):
            continue
        apply_id = str(details.get("apply_id") or "").strip()
        if not apply_id:
            continue
        sections = {str(item) for item in details.get("restored_sections") or [] if isinstance(item, str)}
        if sections:
            by_apply_id.setdefault(apply_id, set()).update(sections)
    return by_apply_id


def _empty_bucket(family: str, status: str) -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "local_action_family": family,
        "policy_section": family,
        "rule_file_family": f"{family}_rules",
        "target_local_rule_file": TARGET_RULE_FILES.get(family),
        "policy_source": "managed-recommended",
        "apply_status": status,
        "rollback_status": "rolled-back" if status == "rolled-back" else "not-required",
        "apply_mode": "local-review",
        "bundle_id": None,
        "draft_id": None,
        "action_id": None,
        "recommendation_id": None,
        "row_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
        "rolled_back_count": 0,
        "failed_count": 0,
        "dry_run_count": 0,
        "blocker_codes": [],
        "privacy": _privacy(),
    }


def build_managed_activation_bundle_apply_outcomes(
    events: list[dict[str, Any]] | None = None,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    if events is None:
        events = recent_policy_events(limit=limit).get("events", [])
    events = [event for event in events if isinstance(event, dict)]

    apply_events = _draft_apply_events(events)
    rolled_back_by_apply_id = _rolled_back_sections(events)

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    blockers: dict[tuple[str, str], Counter[str]] = {}

    for event in apply_events:
        details = event.get("details") or {}
        apply_id = str(details.get("apply_id") or "").strip()
        rolled_back_sections = rolled_back_by_apply_id.get(apply_id, set())
        for raw_entry in details.get("entries") or []:
            if not isinstance(raw_entry, dict):
                continue
            family = _safe_label(raw_entry.get("local_action_family"), "unknown")
            if family not in SUPPORTED_FAMILIES:
                continue
            status = _normalize_status(raw_entry.get("status"))
            if status == "applied" and family in rolled_back_sections:
                status = "rolled-back"
            key = (family, status)
            bucket = buckets.setdefault(key, _empty_bucket(family, status))
            bucket["row_count"] += 1
            field = STATUS_COUNT_FIELD.get(status)
            if field:
                bucket[field] += 1
            if not bucket.get("bundle_id"):
                bucket["bundle_id"] = _safe_ref(details.get("draft_id"))
            if not bucket.get("draft_id"):
                bucket["draft_id"] = _safe_ref(raw_entry.get("draft_id"))
            if not bucket.get("action_id"):
                bucket["action_id"] = _safe_ref(raw_entry.get("action_id"))
            if not bucket.get("recommendation_id"):
                bucket["recommendation_id"] = _safe_ref(raw_entry.get("recommendation_id"))
            reason = raw_entry.get("reason")
            if reason:
                blockers.setdefault(key, Counter())[_safe_label(reason, "unknown")] += 1

    outcome_rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        bucket["blocker_codes"] = [
            {"code": code, "count": count}
            for code, count in blockers.get(key, Counter()).most_common(8)
        ]
        outcome_rows.append(bucket)

    summary = {
        "metadata_only": True,
        "aggregate_only": True,
        "draft_apply_event_count": len(apply_events),
        "rollback_event_count": sum(1 for sections in rolled_back_by_apply_id.values() if sections),
        "outcome_row_count": len(outcome_rows),
        "applied_count": sum(row["applied_count"] for row in outcome_rows),
        "skipped_count": sum(row["skipped_count"] for row in outcome_rows),
        "rolled_back_count": sum(row["rolled_back_count"] for row in outcome_rows),
        "failed_count": sum(row["failed_count"] for row in outcome_rows),
        "dry_run_count": sum(row["dry_run_count"] for row in outcome_rows),
    }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "tracked" if outcome_rows else "no-managed-activation-bundle-apply-evidence",
        "read_only": True,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "summary": summary,
        "outcomes": outcome_rows,
        "privacy": _privacy(),
    }
    violations = managed_egress_violations(result)
    result["egress_guard"] = {
        "schema": "agentflow.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        result["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return result
