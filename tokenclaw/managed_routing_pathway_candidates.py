from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.orchestrator_research import sanitize_value
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import stable_json, utc_now


SCHEMA = "agentflow.managed_routing_pathway_shadow_candidates.v1"
CANDIDATE_SCHEMA = "agentflow.managed_routing_pathway_shadow_candidate.v1"
ROW_SCHEMA = "agentflow.managed_routing_pathway_matrix_row_review.v1"
GROUP_SCHEMA = "agentflow.managed_routing_pathway_shadow_candidate_group.v1"
PRIVACY_SCHEMA = "agentflow.managed_routing_pathway_shadow_candidates_privacy.v1"
MATRIX_SCHEMA = "agentflow.routing_pathway_matrix.v1"
MATRIX_ENTRY_SCHEMA = "agentflow.routing_pathway_matrix_entry.v1"
DEFAULT_STALE_AFTER_HOURS = 72.0

SUPPORTED_EXECUTORS = {
    ("openai_responses", "generic_openai"): "openai-routing-shadow-candidate",
    ("openai_chat", "generic_openai"): "openai-routing-shadow-candidate",
    ("codex_turn", "codex"): "codex-routing-shadow-candidate",
}
ACCEPTED_ACTIONS = {"shadow", "canary", "promote", "review", "review-only"}
BLOCKED_ACTIONS = {"hold", "observe", "keep-blocked"}


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "feature_only": True,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_response_bodies_included": False,
        "provider_bodies_included": False,
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
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_hours(value: Any, now: datetime) -> float | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return max(0.0, round((now - parsed).total_seconds() / 3600.0, 3))


def _matrix_from_source(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("schema") == MATRIX_SCHEMA:
        return source
    matrix = source.get("routing_pathway_matrix")
    if isinstance(matrix, dict):
        return matrix
    decision = source.get("decision")
    if isinstance(decision, dict) and isinstance(decision.get("routing_pathway_matrix"), dict):
        return decision["routing_pathway_matrix"]
    preview = source.get("preview")
    if isinstance(preview, dict) and isinstance(preview.get("routing_pathway_matrix"), dict):
        return preview["routing_pathway_matrix"]
    return {}


def _source_generated_at(source: dict[str, Any], matrix: dict[str, Any]) -> str | None:
    for item in (matrix, source):
        value = item.get("generated_at") if isinstance(item, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pathways(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("pathways")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    row = matrix.get("pathway")
    if isinstance(row, dict):
        return [row]
    return []


def _label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback=fallback)


def _reason_codes(row: dict[str, Any], *extra: str) -> list[str]:
    reasons: list[str] = []
    for value in row.get("reason_codes") if isinstance(row.get("reason_codes"), list) else []:
        text = str(value or "").strip()
        if text and text not in reasons:
            reasons.append(_label(text, "unknown"))
    for value in extra:
        text = str(value or "").strip()
        if text and text not in reasons:
            reasons.append(_label(text, "unknown"))
    return reasons


def _group_key(row: dict[str, Any]) -> dict[str, str]:
    return {
        "source_surface": _label(row.get("source_surface")),
        "app_family": _label(row.get("app_family")),
        "category": _label(row.get("category")),
        "workflow_phase": _label(row.get("workflow_phase")),
        "requested_model_family": _label(row.get("requested_model_family")),
        "target_model_family": _label(row.get("target_model_family")),
        "text_bucket": _label(row.get("text_bucket")),
        "token_bucket": _label(row.get("token_bucket")),
    }


def _group_ref(group_key: dict[str, str]) -> str:
    return public_id(stable_json(group_key), prefix="routing-pathway-group") or "routing-pathway-group:unknown"


def _candidate_ref(row: dict[str, Any], group_key: dict[str, str]) -> str:
    material = {
        "schema": CANDIDATE_SCHEMA,
        "pathway_id": row.get("pathway_id"),
        "group_key": group_key,
        "requested_model": row.get("requested_model"),
        "target_model": row.get("target_model"),
    }
    return public_id(stable_json(material), prefix="routing-pathway-candidate") or "routing-pathway-candidate:unknown"


def _executor_compatibility(row: dict[str, Any]) -> dict[str, Any]:
    source_surface = _label(row.get("source_surface"))
    app_family = _label(row.get("app_family"))
    executor = SUPPORTED_EXECUTORS.get((source_surface, app_family))
    reason_codes: list[str] = []
    if executor is None:
        reason_codes.append("unsupported-local-routing-executor")
    return {
        "schema": "agentflow.managed_routing_pathway_local_executor_compatibility.v1",
        "compatible": executor is not None,
        "local_executor": executor,
        "supported_local_action_families": ["routing"],
        "reason_codes": reason_codes,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }


def _review_status(
    row: dict[str, Any],
    *,
    stale: bool,
    executor: dict[str, Any],
) -> tuple[str, str, list[str]]:
    if row.get("schema") not in (None, MATRIX_ENTRY_SCHEMA):
        return "omitted", "unsupported-routing-pathway-row-schema", _reason_codes(row, "unsupported-routing-pathway-row-schema")
    if stale:
        return "stale", "stale-routing-pathway-matrix", _reason_codes(row, "stale-routing-pathway-matrix")
    if not executor.get("compatible"):
        return "omitted", "unsupported-local-routing-executor", _reason_codes(row, "unsupported-local-routing-executor")
    if not str(row.get("requested_model") or "").strip():
        return "omitted", "missing-requested-model", _reason_codes(row, "missing-requested-model")
    if not str(row.get("target_model") or row.get("candidate_target_model") or "").strip():
        return "omitted", "missing-target-model", _reason_codes(row, "missing-target-model")

    action = str(row.get("suggested_next_action") or "").strip().lower()
    reason_texts = [str(value).lower() for value in _reason_codes(row)]
    if action in BLOCKED_ACTIONS:
        return "blocked", f"routing-pathway-{action}", _reason_codes(row, f"routing-pathway-{action}")
    if any("confidence-too-low" in value or "evidence-missing" in value for value in reason_texts):
        return "blocked", "routing-pathway-evidence-blocked", _reason_codes(row, "routing-pathway-evidence-blocked")
    if bool(row.get("activation_recommendation")) or action in ACCEPTED_ACTIONS:
        return "accepted", "review-only-shadow-routing-candidate", _reason_codes(row, "review-only-shadow-routing-candidate")
    return "blocked", "routing-pathway-observe-only", _reason_codes(row, "routing-pathway-observe-only")


def _candidate_from_row(
    row: dict[str, Any],
    *,
    source_generated_at: str | None,
    now: datetime,
    stale_after_hours: float,
) -> dict[str, Any]:
    age_hours = _age_hours(source_generated_at, now)
    stale = bool(age_hours is not None and age_hours > max(0.0, stale_after_hours))
    group_key = _group_key(row)
    group_ref = _group_ref(group_key)
    executor = _executor_compatibility(row)
    status, reason, reasons = _review_status(row, stale=stale, executor=executor)
    target_model = row.get("target_model") or row.get("candidate_target_model")
    candidate = {
        "schema": CANDIDATE_SCHEMA if status == "accepted" else ROW_SCHEMA,
        "status": status,
        "decision": "review-only-shadow-candidate" if status == "accepted" else status,
        "reason": reason,
        "reason_codes": reasons,
        "candidate_fingerprint": _candidate_ref(row, group_key),
        "group_ref": group_ref,
        "group_key": group_key,
        "pathway_id": public_id(row.get("pathway_id"), prefix="routing-pathway"),
        "rank": row.get("rank"),
        "pathway_type": _label(row.get("pathway_type")),
        "source_surface": group_key["source_surface"],
        "app_family": group_key["app_family"],
        "category": group_key["category"],
        "workflow_phase": group_key["workflow_phase"],
        "requested_model": _label(row.get("requested_model")),
        "requested_model_family": group_key["requested_model_family"],
        "target_model": _label(target_model),
        "target_model_family": group_key["target_model_family"],
        "text_bucket": group_key["text_bucket"],
        "token_bucket": group_key["token_bucket"],
        "sample_count": row.get("sample_count"),
        "compared_count": row.get("compared_count"),
        "pass_rate": row.get("pass_rate"),
        "shadow_error_rate": row.get("shadow_error_rate"),
        "fallback_rate": row.get("fallback_rate"),
        "retry_rate": row.get("retry_rate"),
        "error_rate": row.get("error_rate"),
        "estimated_savings_usd": row.get("estimated_savings_usd"),
        "cost_basis": _label(row.get("cost_basis"), "feature-metadata-estimate"),
        "latency_delta_ms": row.get("latency_delta_ms"),
        "route_down_probability": row.get("route_down_probability"),
        "suggested_next_action": _label(row.get("suggested_next_action")),
        "activation_recommendation": bool(row.get("activation_recommendation")),
        "canary": sanitize_value(row.get("canary") if isinstance(row.get("canary"), dict) else {}),
        "local_action_family": "routing",
        "local_executor_compatibility": executor,
        "review_only": True,
        "authoritative_for_active_policy": False,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "matrix_generated_at": source_generated_at,
        "matrix_age_hours": age_hours,
        "stale_after_hours": float(stale_after_hours),
        "stale": stale,
        "privacy": _privacy(),
    }
    return {
        key: value
        for key, value in candidate.items()
        if value not in (None, "", []) or key in {"policy_files_written", "provider_calls_made", "managed_server_calls_made"}
    }


def _groups(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("group_ref") or "routing-pathway-group:unknown"), []).append(candidate)
    result: list[dict[str, Any]] = []
    for group_ref, rows in grouped.items():
        first = rows[0]
        statuses = Counter(str(row.get("status") or "unknown") for row in rows)
        result.append(
            {
                "schema": GROUP_SCHEMA,
                "group_ref": group_ref,
                "group_key": first.get("group_key") or {},
                "candidate_count": len(rows),
                "accepted_count": statuses.get("accepted", 0),
                "blocked_count": statuses.get("blocked", 0),
                "omitted_count": statuses.get("omitted", 0),
                "stale_count": statuses.get("stale", 0),
                "candidate_fingerprints": [
                    row.get("candidate_fingerprint") for row in rows if row.get("candidate_fingerprint")
                ],
                "codex_specific": first.get("app_family") == "codex" or first.get("source_surface") == "codex_turn",
                "review_only": True,
                "policy_files_written": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            }
        )
    return sorted(result, key=lambda item: stable_json(item.get("group_key") or {}))


def build_managed_routing_pathway_shadow_candidates(
    source: dict[str, Any],
    *,
    now: datetime | None = None,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
) -> dict[str, Any]:
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    matrix = _matrix_from_source(source)
    generated_at = _source_generated_at(source, matrix)
    rows = _pathways(matrix)
    candidates = [
        _candidate_from_row(
            row,
            source_generated_at=generated_at,
            now=now_dt,
            stale_after_hours=stale_after_hours,
        )
        for row in rows
    ]
    accepted = [row for row in candidates if row.get("status") == "accepted"]
    blocked = [row for row in candidates if row.get("status") == "blocked"]
    omitted = [row for row in candidates if row.get("status") == "omitted"]
    stale = [row for row in candidates if row.get("status") == "stale"]
    status_counts = Counter(str(row.get("status") or "unknown") for row in candidates)
    app_counts = Counter(str(row.get("app_family") or "unknown") for row in candidates)
    source_counts = Counter(str(row.get("source_surface") or "unknown") for row in candidates)
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "review-only" if candidates else "empty",
        "source_schema": _label(source.get("schema")),
        "source_matrix_schema": _label(matrix.get("schema")),
        "matrix_generated_at": generated_at,
        "stale_after_hours": float(stale_after_hours),
        "review_only": True,
        "authoritative_for_active_policy": False,
        "managed_dependency": "optional",
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": False,
        "summary": {
            "matrix_row_count": len(rows),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "blocked_count": len(blocked),
            "omitted_count": len(omitted),
            "stale_count": len(stale),
            "group_count": len({row.get("group_ref") for row in candidates}),
            "codex_candidate_count": sum(1 for row in candidates if row.get("app_family") == "codex"),
            "generic_openai_candidate_count": sum(1 for row in candidates if row.get("app_family") == "generic_openai"),
            "status_counts": [{"value": key, "count": value} for key, value in sorted(status_counts.items())],
            "app_family_counts": [{"value": key, "count": value} for key, value in sorted(app_counts.items())],
            "source_surface_counts": [{"value": key, "count": value} for key, value in sorted(source_counts.items())],
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
        "groups": _groups(candidates),
        "accepted": accepted,
        "blocked": blocked,
        "omitted": omitted,
        "stale": stale,
        "candidates": accepted,
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
