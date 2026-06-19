from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.orchestrator_research import sanitize_value
from agentflow_proxy.public_metadata import public_id
from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.managed_activation_preview_outcomes.v1"
OUTCOME_SCHEMA = "agentflow.managed_activation_preview_outcome.v1"
PRIVACY_SCHEMA = "agentflow.managed_activation_preview_outcomes_privacy.v1"
DEFAULT_STALE_AFTER_HOURS = 72.0


def _parse_utc(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _hours_between(left: datetime | None, right: datetime) -> float | None:
    if left is None:
        return None
    return max(0.0, round((right - left).total_seconds() / 3600.0, 3))


def _privacy(*, managed_server_calls_made: bool = False) -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "feature_only": True,
        "metadata_only": True,
        "aggregate_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": bool(managed_server_calls_made),
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _decision_rows(preview_result: dict[str, Any]) -> list[dict[str, Any]]:
    preview = preview_result.get("preview")
    if isinstance(preview, dict):
        rows = preview.get("decisions")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _request_rows(preview_result: dict[str, Any]) -> list[dict[str, Any]]:
    request = preview_result.get("preview_request")
    if isinstance(request, dict):
        rows = request.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _preview_generated_at(preview_result: dict[str, Any]) -> str | None:
    request = preview_result.get("preview_request")
    if isinstance(request, dict) and request.get("generated_at"):
        return sanitize_value(request.get("generated_at"))
    return sanitize_value(preview_result.get("generated_at"))


def _row_fingerprint(handoff_ref: str, request_row: dict[str, Any], decision: dict[str, Any]) -> str:
    material = {
        "schema": OUTCOME_SCHEMA,
        "handoff_ref": handoff_ref,
        "local_action_family": request_row.get("local_action_family") or decision.get("local_action_family"),
        "evidence_schema": request_row.get("evidence_schema"),
    }
    return public_id(stable_json(material), prefix="managed-preview-outcome") or "managed-preview-outcome:unknown"


def _matching_decisions(preview_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _decision_rows(preview_result):
        ref = str(row.get("handoff_ref") or "").strip()
        if ref:
            result[ref] = row
    return result


def _decision_text(decision: dict[str, Any]) -> str:
    return str(decision.get("decision") or decision.get("status") or "").strip().lower()


def _disagrees_with_local(request_row: dict[str, Any], decision: dict[str, Any]) -> bool:
    if not decision:
        return False
    local_class = str(request_row.get("executor_action_class") or "").strip().lower()
    local_status = str(request_row.get("current_status") or request_row.get("successor_status") or "").strip().lower()
    decision_value = _decision_text(decision)
    if not decision_value:
        return False
    keep_active = {"keep-active", "keep-current-rule", "no-op", "noop"}
    keep_blocked = {"keep-blocked", "omitted", "no-op", "noop"}
    keep_retired = {"keep-retired", "retire", "retired", "omitted", "no-op", "noop"}
    activating = {"activate", "apply", "draft-local-policy", "promote", "widen", "review-only-recommendation"}
    if local_class == "keep-current-rule" or local_status == "full-rollout":
        return decision_value not in keep_active
    if local_class == "keep-blocked":
        return decision_value in activating
    if local_class == "retire":
        return decision_value not in keep_retired
    return False


def _classification(
    *,
    fetch_status: str,
    managed_server_calls_made: bool,
    stale: bool,
    missing: bool,
    failed_closed: bool,
    disagreement: bool,
) -> str:
    if failed_closed:
        return "failed-closed"
    if fetch_status in {"skipped", "no-data"} or (missing and not managed_server_calls_made):
        return "no-data-preview-health"
    if missing:
        return "missing-preview-decision"
    if stale:
        return "stale-preview"
    if disagreement:
        return "managed-local-disagreement"
    if fetch_status == "skipped":
        return "not-previewed"
    return "review-only"


def _next_action(
    *,
    classification: str,
    request_row: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    if classification in {"no-data-preview-health", "missing-preview-decision", "stale-preview", "not-previewed"}:
        return "refresh-managed-activation-preview"
    if classification == "failed-closed":
        return "keep-local-policy-authoritative-review-managed-preview"
    if classification == "managed-local-disagreement":
        return "review-managed-preview-disagreement"
    return str(
        decision.get("recommended_next_action")
        or decision.get("next_action")
        or request_row.get("executor_next_action")
        or "review-managed-activation-preview"
    ).strip()


def _outcome_from_rows(
    *,
    request_row: dict[str, Any],
    decision: dict[str, Any],
    preview_result: dict[str, Any],
    now: datetime,
    stale_after_hours: float,
) -> dict[str, Any]:
    handoff_ref = str(request_row.get("handoff_ref") or decision.get("handoff_ref") or "").strip()
    preview_time_raw = _preview_generated_at(preview_result)
    preview_time = _parse_utc(preview_time_raw)
    age_hours = _hours_between(preview_time, now)
    fetch = preview_result.get("fetch") if isinstance(preview_result.get("fetch"), dict) else {}
    fetch_status = str(fetch.get("status") or preview_result.get("status") or "").strip() or "unknown"
    fetch_reason = str(fetch.get("reason") or preview_result.get("reason") or "").strip()
    managed_server_calls_made = bool(fetch.get("managed_server_calls_made"))
    missing = not bool(decision)
    stale = bool(age_hours is not None and age_hours > max(0.0, stale_after_hours))
    preview_policy_write = bool(decision.get("policy_files_written"))
    preview_provider_call = bool(decision.get("provider_calls_made"))
    preview_managed_enforced = bool(decision.get("managed_enforced"))
    failed_closed = bool(
        fetch_status in {"blocked", "error"}
        or preview_policy_write
        or preview_provider_call
        or preview_managed_enforced
    )
    disagreement = _disagrees_with_local(request_row, decision)
    classification = _classification(
        fetch_status=fetch_status,
        managed_server_calls_made=managed_server_calls_made,
        stale=stale,
        missing=missing,
        failed_closed=failed_closed,
        disagreement=disagreement,
    )
    reason_codes = [
        str(item)
        for item in (
            _as_list(decision.get("reason_codes"))
            + _as_list(request_row.get("reason_codes"))
            + _as_list(request_row.get("blocker_codes"))
        )
        if str(item or "").strip()
    ]
    outcome = {
        "schema": OUTCOME_SCHEMA,
        "outcome_fingerprint": _row_fingerprint(handoff_ref, request_row, decision),
        "handoff_ref": sanitize_value(handoff_ref),
        "preview_ref": sanitize_value(decision.get("preview_ref")),
        "source_executor_ref": sanitize_value(request_row.get("source_executor_ref")),
        "source_activation_ref": sanitize_value(request_row.get("source_activation_ref")),
        "source_successor_ref": sanitize_value(request_row.get("source_successor_ref")),
        "preview_generated_at": sanitize_value(preview_time_raw),
        "preview_age_hours": age_hours,
        "stale_after_hours": float(stale_after_hours),
        "local_action_family": sanitize_value(
            request_row.get("local_action_family") or decision.get("local_action_family") or "unknown"
        ),
        "evidence_schema": sanitize_value(request_row.get("evidence_schema")),
        "executor_action_class": sanitize_value(request_row.get("executor_action_class")),
        "current_status": sanitize_value(request_row.get("current_status")),
        "decision": sanitize_value(decision.get("decision") or "missing"),
        "decision_status": sanitize_value(decision.get("status") or decision.get("decision") or "missing"),
        "classification": classification,
        "preview_status": classification,
        "preview_reason": sanitize_value(fetch_reason or classification),
        "fetch_status": sanitize_value(fetch_status),
        "fetch_reason": sanitize_value(fetch_reason),
        "next_action": _next_action(classification=classification, request_row=request_row, decision=decision),
        "omitted_reason": sanitize_value(decision.get("omitted_reason")),
        "no_op_reason": sanitize_value(decision.get("no_op_reason")),
        "reason_codes": sanitize_value(reason_codes),
        "review_only": True,
        "authoritative_for_active_policy": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": managed_server_calls_made,
        "managed_preview_policy_files_written": preview_policy_write,
        "managed_preview_provider_calls_made": preview_provider_call,
        "managed_preview_enforced": preview_managed_enforced,
        "stale": stale,
        "missing_preview_decision": classification == "missing-preview-decision",
        "preview_decision_missing": missing,
        "no_data_preview_health": classification == "no-data-preview-health",
        "failed_closed": failed_closed,
        "disagrees_with_local_evidence": disagreement,
        "privacy": _privacy(managed_server_calls_made=managed_server_calls_made),
    }
    return {
        key: value
        for key, value in sanitize_value(outcome).items()
        if value not in (None, "", []) or key in {"review_only", "policy_files_written", "provider_calls_made"}
    }


def ensure_managed_activation_preview_outcomes_table(store_obj: Any) -> None:
    store_obj.conn.execute(
        """
        create table if not exists managed_activation_preview_outcomes (
          fingerprint text primary key,
          created_at text not null,
          updated_at text not null,
          preview_generated_at text,
          handoff_ref text not null,
          preview_ref text,
          local_action_family text,
          evidence_schema text,
          decision text,
          classification text,
          next_action text,
          omitted_reason text,
          no_op_reason text,
          reason_codes_json text not null,
          stale integer not null default 0,
          missing_preview_decision integer not null default 0,
          failed_closed integer not null default 0,
          disagrees_with_local_evidence integer not null default 0,
          preview_age_hours real,
          outcome_json text not null
        )
        """
    )
    store_obj.conn.execute(
        """
        create index if not exists idx_managed_activation_preview_outcomes_updated
        on managed_activation_preview_outcomes(updated_at)
        """
    )
    store_obj.conn.execute(
        """
        create index if not exists idx_managed_activation_preview_outcomes_family
        on managed_activation_preview_outcomes(local_action_family, classification, updated_at)
        """
    )


def _persist_one(store_obj: Any, outcome: dict[str, Any], *, now_text: str) -> bool:
    existing = store_obj.conn.execute(
        "select fingerprint from managed_activation_preview_outcomes where fingerprint = ?",
        (outcome["outcome_fingerprint"],),
    ).fetchone()
    store_obj.conn.execute(
        """
        insert into managed_activation_preview_outcomes(
          fingerprint, created_at, updated_at, preview_generated_at, handoff_ref,
          preview_ref, local_action_family, evidence_schema, decision, classification,
          next_action, omitted_reason, no_op_reason, reason_codes_json, stale,
          missing_preview_decision, failed_closed, disagrees_with_local_evidence,
          preview_age_hours, outcome_json
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(fingerprint) do update set
          updated_at = excluded.updated_at,
          preview_generated_at = excluded.preview_generated_at,
          handoff_ref = excluded.handoff_ref,
          preview_ref = excluded.preview_ref,
          local_action_family = excluded.local_action_family,
          evidence_schema = excluded.evidence_schema,
          decision = excluded.decision,
          classification = excluded.classification,
          next_action = excluded.next_action,
          omitted_reason = excluded.omitted_reason,
          no_op_reason = excluded.no_op_reason,
          reason_codes_json = excluded.reason_codes_json,
          stale = excluded.stale,
          missing_preview_decision = excluded.missing_preview_decision,
          failed_closed = excluded.failed_closed,
          disagrees_with_local_evidence = excluded.disagrees_with_local_evidence,
          preview_age_hours = excluded.preview_age_hours,
          outcome_json = excluded.outcome_json
        """,
        (
            outcome["outcome_fingerprint"],
            now_text,
            now_text,
            outcome.get("preview_generated_at"),
            outcome.get("handoff_ref"),
            outcome.get("preview_ref"),
            outcome.get("local_action_family"),
            outcome.get("evidence_schema"),
            outcome.get("decision"),
            outcome.get("classification"),
            outcome.get("next_action"),
            outcome.get("omitted_reason"),
            outcome.get("no_op_reason"),
            stable_json(outcome.get("reason_codes") or []),
            int(bool(outcome.get("stale"))),
            int(bool(outcome.get("missing_preview_decision"))),
            int(bool(outcome.get("failed_closed"))),
            int(bool(outcome.get("disagrees_with_local_evidence"))),
            outcome.get("preview_age_hours"),
            stable_json(outcome),
        ),
    )
    return existing is None


def persist_managed_activation_preview_outcomes(
    store_obj: Any,
    preview_result: dict[str, Any],
    *,
    now: datetime | None = None,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
) -> dict[str, Any]:
    ensure_managed_activation_preview_outcomes_table(store_obj)
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_text = now_dt.isoformat()
    decisions = _matching_decisions(preview_result)
    outcomes = [
        _outcome_from_rows(
            request_row=row,
            decision=decisions.get(str(row.get("handoff_ref") or "").strip(), {}),
            preview_result=preview_result,
            now=now_dt,
            stale_after_hours=stale_after_hours,
        )
        for row in _request_rows(preview_result)
    ]
    created_count = 0
    for outcome in outcomes:
        if _persist_one(store_obj, outcome, now_text=now_text):
            created_count += 1
    report = build_managed_activation_preview_outcomes_report(
        store_obj,
        limit=max(len(outcomes), 1),
        now=now_dt,
        stale_after_hours=stale_after_hours,
    )
    report["import"] = {
        "schema": "agentflow.managed_activation_preview_outcome_import.v1",
        "imported_count": len(outcomes),
        "created_count": created_count,
        "updated_count": max(0, len(outcomes) - created_count),
        "policy_files_written": False,
        "provider_calls_made": False,
    }
    return report


def _row_to_outcome(row: dict[str, Any], *, now: datetime, stale_after_hours: float) -> dict[str, Any]:
    outcome = json.loads(row["outcome_json"])
    preview_time = _parse_utc(outcome.get("preview_generated_at"))
    age_hours = _hours_between(preview_time, now)
    if age_hours is not None:
        outcome["preview_age_hours"] = age_hours
        outcome["stale"] = bool(age_hours > max(0.0, stale_after_hours))
        if outcome["stale"] and outcome.get("classification") == "review-only":
            outcome["classification"] = "stale-preview"
            outcome["next_action"] = "refresh-managed-activation-preview"
    outcome["stale_after_hours"] = float(stale_after_hours)
    return sanitize_value(outcome)


def build_managed_activation_preview_outcomes_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    now: datetime | None = None,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
) -> dict[str, Any]:
    ensure_managed_activation_preview_outcomes_table(store_obj)
    capped = max(1, min(int(limit or 1), 20_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select *
            from managed_activation_preview_outcomes
            order by updated_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    outcomes = [_row_to_outcome(row, now=now_dt, stale_after_hours=stale_after_hours) for row in rows]
    classification_counts = Counter(str(row.get("classification") or "unknown") for row in outcomes)
    family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in outcomes)
    managed_calls_made = any(bool(row.get("managed_server_calls_made")) for row in outcomes)
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "tracked" if outcomes else "empty",
        "read_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "managed_dependency": "optional",
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": managed_calls_made,
        "summary": {
            "stored_preview_outcome_count": len(outcomes),
            "lookback_limit": capped,
            "stale_after_hours": float(stale_after_hours),
            "stale_count": sum(1 for row in outcomes if row.get("stale")),
            "missing_preview_decision_count": sum(1 for row in outcomes if row.get("missing_preview_decision")),
            "no_data_preview_health_count": sum(
                1 for row in outcomes if row.get("classification") == "no-data-preview-health"
            ),
            "failed_closed_count": sum(1 for row in outcomes if row.get("failed_closed")),
            "disagreement_count": sum(1 for row in outcomes if row.get("disagrees_with_local_evidence")),
            "policy_files_written": False,
            "provider_calls_made": False,
            "classification_counts": [
                {"value": key, "count": count} for key, count in sorted(classification_counts.items())
            ],
            "local_action_family_counts": [
                {"value": key, "count": count} for key, count in sorted(family_counts.items())
            ],
        },
        "outcomes": outcomes,
        "privacy": _privacy(managed_server_calls_made=managed_calls_made),
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
