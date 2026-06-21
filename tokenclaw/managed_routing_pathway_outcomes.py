from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.managed_routing_pathway_candidates import (
    CANDIDATE_SCHEMA,
    build_managed_routing_pathway_shadow_candidates,
)
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import stable_json, utc_now


SCHEMA = "tokenclaw.local_routing_pathway_outcome_feedback.v1"
ROW_SCHEMA = "tokenclaw.local_routing_pathway_outcome_feedback_row.v1"
PRIVACY_SCHEMA = "tokenclaw.local_routing_pathway_outcome_feedback_privacy.v1"
SEMANTIC_QUALITY_SCHEMA = "tokenclaw.local_routing_pathway_semantic_quality_outcome.v1"
LIFECYCLE_SCHEMA = "tokenclaw.local_routing_pathway_lifecycle_counts.v1"
DEFAULT_STALE_AFTER_HOURS = 72.0
DEFAULT_MIN_SEMANTIC_COMPARISONS = 20
DEFAULT_MIN_SEMANTIC_PASS_RATE = 0.90


def _privacy() -> dict[str, Any]:
    return {
        "schema": PRIVACY_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "feature_only": True,
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


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any, fallback: str = "unknown") -> str:
    return public_label(value, fallback=fallback)


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


def _candidate_rows(source: dict[str, Any], *, stale_after_hours: float) -> list[dict[str, Any]]:
    if source.get("schema") == "tokenclaw.managed_routing_pathway_shadow_candidates.v1":
        rows: list[dict[str, Any]] = []
        for key in ("accepted", "blocked", "stale", "omitted"):
            values = source.get(key)
            if isinstance(values, list):
                rows.extend([row for row in values if isinstance(row, dict)])
        if rows:
            return rows
    report = build_managed_routing_pathway_shadow_candidates(source, stale_after_hours=stale_after_hours)
    rows = []
    for key in ("accepted", "blocked", "stale", "omitted"):
        values = report.get(key)
        if isinstance(values, list):
            rows.extend([row for row in values if isinstance(row, dict)])
    return rows


def _candidate_ref(candidate: dict[str, Any]) -> str:
    value = candidate.get("candidate_fingerprint")
    if isinstance(value, str) and value.strip():
        return value.strip()
    material = {
        "source_surface": candidate.get("source_surface"),
        "app_family": candidate.get("app_family"),
        "category": candidate.get("category"),
        "workflow_phase": candidate.get("workflow_phase"),
        "requested_model": candidate.get("requested_model"),
        "target_model": candidate.get("target_model"),
    }
    return public_id(stable_json(material), prefix="routing-pathway-candidate") or "routing-pathway-candidate:unknown"


def _cohort(canary: dict[str, Any]) -> str:
    status = str(canary.get("status") or "").strip()
    cohort = str(canary.get("cohort") or "").strip()
    reason = str(canary.get("reason") or "").strip()
    safety = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
    if status == "applied" or cohort == "canary_applied":
        return "canary_applied"
    if status == "holdout" or cohort == "canary_holdout":
        return "canary_holdout"
    if status == "safety_stopped" or safety.get("tripped") or "safety-stop" in reason:
        return "safety_stopped"
    if status in {"disabled", "noop"} or cohort == "bypassed_or_disabled":
        return "bypassed_or_disabled"
    if status in {"ineligible", "not_selected", "skipped"} or cohort == "skipped":
        return "skipped"
    return "unknown"


def _model_matches(expected: Any, actual: Any) -> bool:
    expected_text = str(expected or "").strip().lower()
    actual_text = str(actual or "").strip().lower()
    return not expected_text or expected_text == "unknown" or expected_text == actual_text


def _metadata_matches(candidate: dict[str, Any], row: dict[str, Any], routing: dict[str, Any]) -> bool:
    source = row.get("source_surface") or routing.get("source_surface")
    if not _model_matches(candidate.get("source_surface"), source):
        return False
    category = row.get("category") or routing.get("category")
    if not _model_matches(candidate.get("category"), category):
        return False
    requested = row.get("requested_model") or routing.get("requested_model")
    if not _model_matches(candidate.get("requested_model"), requested):
        return False
    target = candidate.get("target_model")
    canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
    routed = row.get("routed_model") or routing.get("routed_model")
    canary_target = canary.get("target_model") or canary.get("candidate_target_model")
    return _model_matches(target, routed) or _model_matches(target, canary_target)


def _empty_lifecycle() -> dict[str, Any]:
    return {
        "schema": LIFECYCLE_SCHEMA,
        "matched_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "bypassed_count": 0,
        "unknown_count": 0,
        "safety_stop_count": 0,
        "error_count": 0,
        "fallback_count": 0,
        "retry_count": 0,
        "oldest_observed_at": None,
        "latest_observed_at": None,
        "reason_breakdown": [],
        "privacy": _privacy(),
    }


def _finalize_lifecycle(raw: dict[str, Any], reasons: Counter[str]) -> dict[str, Any]:
    result = dict(raw)
    result["reason_breakdown"] = [
        {"value": key, "count": value}
        for key, value in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    ]
    result["coverage_present"] = bool(_as_int(result.get("applied_count")) or _as_int(result.get("holdout_count")))
    return result


def _openai_lifecycle(store: Any, candidate: dict[str, Any], *, limit: int) -> dict[str, Any]:
    raw = _empty_lifecycle()
    reasons: Counter[str] = Counter()
    rows = store.conn.execute(
        """
        select created_at, requested_model, routed_model, source_surface, endpoint, category,
               status_code, retry_count, routing_json
        from calls
        where coalesce(provider, 'anthropic') = 'openai'
        order by created_at desc
        limit ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    for row_obj in rows:
        row = dict(row_obj)
        routing = _json_obj(row.get("routing_json"))
        if not _metadata_matches(candidate, row, routing):
            continue
        canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
        cohort = _cohort(canary)
        raw["matched_count"] = _as_int(raw.get("matched_count")) + 1
        if cohort == "canary_applied":
            raw["applied_count"] = _as_int(raw.get("applied_count")) + 1
        elif cohort == "canary_holdout":
            raw["holdout_count"] = _as_int(raw.get("holdout_count")) + 1
        elif cohort == "safety_stopped":
            raw["safety_stop_count"] = _as_int(raw.get("safety_stop_count")) + 1
        elif cohort == "bypassed_or_disabled":
            raw["bypassed_count"] = _as_int(raw.get("bypassed_count")) + 1
        elif cohort == "skipped":
            raw["skipped_count"] = _as_int(raw.get("skipped_count")) + 1
        else:
            raw["unknown_count"] = _as_int(raw.get("unknown_count")) + 1
        if _as_int(row.get("status_code")) >= 400:
            raw["error_count"] = _as_int(raw.get("error_count")) + 1
        if _as_int(row.get("retry_count")) > 0:
            raw["retry_count"] = _as_int(raw.get("retry_count")) + 1
        if canary.get("fallback_reason") or routing.get("fallback_reason"):
            raw["fallback_count"] = _as_int(raw.get("fallback_count")) + 1
        reason = str(canary.get("reason") or cohort or "unknown")
        reasons[_label(reason)] += 1
        created = row.get("created_at")
        if isinstance(created, str):
            if raw["latest_observed_at"] is None or created > str(raw["latest_observed_at"]):
                raw["latest_observed_at"] = created
            if raw["oldest_observed_at"] is None or created < str(raw["oldest_observed_at"]):
                raw["oldest_observed_at"] = created
    return _finalize_lifecycle(raw, reasons)


def _codex_lifecycle(store: Any, candidate: dict[str, Any], *, limit: int) -> dict[str, Any]:
    raw = _empty_lifecycle()
    reasons: Counter[str] = Counter()
    rows = store.conn.execute(
        """
        select created_at, input_text_chars, result_chars, error_code, routing_json, event_window_json
        from codex_app_events
        where direction = 'client_to_server'
          and method = 'turn/start'
        order by created_at desc
        limit ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    for row_obj in rows:
        row = dict(row_obj)
        routing = _json_obj(row.get("routing_json"))
        window = _json_obj(row.get("event_window_json"))
        requested = routing.get("requested_model") or (window.get("model_state") or {}).get("normalized_model")
        if not _model_matches(candidate.get("requested_model"), requested):
            continue
        workflow_phase = window.get("workflow_phase") or routing.get("workflow_phase")
        if not _model_matches(candidate.get("workflow_phase"), workflow_phase):
            continue
        raw["matched_count"] = _as_int(raw.get("matched_count")) + 1
        canary = routing.get("codex_routing_canary") if isinstance(routing.get("codex_routing_canary"), dict) else {}
        cohort = _cohort(canary)
        if cohort == "canary_applied":
            raw["applied_count"] = _as_int(raw.get("applied_count")) + 1
        elif cohort == "canary_holdout":
            raw["holdout_count"] = _as_int(raw.get("holdout_count")) + 1
        elif cohort == "safety_stopped":
            raw["safety_stop_count"] = _as_int(raw.get("safety_stop_count")) + 1
        elif canary:
            raw["skipped_count"] = _as_int(raw.get("skipped_count")) + 1
        else:
            raw["unknown_count"] = _as_int(raw.get("unknown_count")) + 1
        if _as_int(row.get("error_code")):
            raw["error_count"] = _as_int(raw.get("error_count")) + 1
        reasons[_label(canary.get("reason") if canary else "missing-codex-routing-canary-lifecycle")] += 1
        created = row.get("created_at")
        if isinstance(created, str):
            if raw["latest_observed_at"] is None or created > str(raw["latest_observed_at"]):
                raw["latest_observed_at"] = created
            if raw["oldest_observed_at"] is None or created < str(raw["oldest_observed_at"]):
                raw["oldest_observed_at"] = created
    return _finalize_lifecycle(raw, reasons)


def _semantic_quality(store: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("source_surface") not in {"openai_responses", "openai_chat"}:
        return {
            "schema": SEMANTIC_QUALITY_SCHEMA,
            "status": "not-applicable",
            "reason": "semantic-quality-evidence-not-required-for-surface",
            "clean_comparison_count": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": None,
            "privacy": _privacy(),
        }
    rows = store.conn.execute(
        """
        select primary_status_code, shadow_status_code, output_similarity, passed_threshold, error
        from routing_experiments
        where coalesce(provider, 'openai') = 'openai'
          and coalesce(source_surface, '') = ?
          and requested_model = ?
          and shadow_model = ?
          and coalesce(category, 'unknown') = ?
        """,
        (
            str(candidate.get("source_surface") or ""),
            str(candidate.get("requested_model") or ""),
            str(candidate.get("target_model") or ""),
            str(candidate.get("category") or "unknown"),
        ),
    ).fetchall()
    clean = 0
    passed = 0
    failed = 0
    for row_obj in rows:
        row = dict(row_obj)
        if _as_int(row.get("primary_status_code"), 999) >= 400 or _as_int(row.get("shadow_status_code"), 999) >= 400:
            continue
        if row.get("output_similarity") is None or row.get("error"):
            continue
        clean += 1
        if _as_int(row.get("passed_threshold")):
            passed += 1
        else:
            failed += 1
    pass_rate = round(passed / clean, 6) if clean else 0.0
    if clean <= 0:
        status = "missing"
        reason = "missing-semantic-quality-evidence"
    elif clean < DEFAULT_MIN_SEMANTIC_COMPARISONS:
        status = "missing"
        reason = "insufficient-semantic-quality-comparisons"
    elif pass_rate < DEFAULT_MIN_SEMANTIC_PASS_RATE:
        status = "regressed"
        reason = "semantic-quality-regression-observed"
    else:
        status = "passed"
        reason = "semantic-quality-gate-passed"
    return {
        "schema": SEMANTIC_QUALITY_SCHEMA,
        "status": status,
        "reason": reason,
        "clean_comparison_count": clean,
        "pass_count": passed,
        "fail_count": failed,
        "pass_rate": pass_rate,
        "min_clean_comparison_count": DEFAULT_MIN_SEMANTIC_COMPARISONS,
        "min_pass_rate": DEFAULT_MIN_SEMANTIC_PASS_RATE,
        "privacy": _privacy(),
    }


def _classify(candidate: dict[str, Any], lifecycle: dict[str, Any], semantic: dict[str, Any]) -> tuple[str, str, str]:
    candidate_status = str(candidate.get("status") or "unknown")
    if candidate_status == "stale" or bool(candidate.get("stale")):
        return "stale", "stale-routing-pathway-matrix", "refresh-routing-pathway-matrix"
    if candidate_status in {"blocked", "omitted"}:
        return "blocked", str(candidate.get("reason") or "routing-pathway-candidate-blocked"), str(candidate.get("suggested_next_action") or "review-routing-pathway-blocker")
    if _as_int(lifecycle.get("safety_stop_count")):
        return "blocked", "safety-stop-observed", "keep-routing-pathway-blocked"
    if _as_int(lifecycle.get("error_count")) or _as_int(lifecycle.get("fallback_count")):
        return "blocked", "local-routing-lifecycle-errors-observed", "review-routing-pathway-blocker"
    if semantic.get("status") == "regressed":
        return "regressed", "semantic-quality-regression-observed", "review-openai-routing-canary-blockers"
    applied = _as_int(lifecycle.get("applied_count"))
    holdout = _as_int(lifecycle.get("holdout_count"))
    matched = _as_int(lifecycle.get("matched_count"))
    if applied > 0 and holdout > 0 and semantic.get("status") in {"passed", "not-applicable"}:
        return "ready", "applied-and-holdout-coverage-present", "stage-narrow-routing-canary"
    if matched > 0:
        return "observed", "local-routing-pathway-observed-missing-coverage", "collect-routing-pathway-applied-holdout-coverage"
    return "missing-coverage", "missing-local-routing-pathway-coverage", "collect-routing-pathway-lifecycle-evidence"


def _outcome_row(store: Any, candidate: dict[str, Any], *, limit: int) -> dict[str, Any]:
    source_surface = _label(candidate.get("source_surface"))
    app_family = _label(candidate.get("app_family"))
    if source_surface in {"openai_responses", "openai_chat"}:
        lifecycle = _openai_lifecycle(store, candidate, limit=limit)
    elif source_surface == "codex_turn" or app_family == "codex":
        lifecycle = _codex_lifecycle(store, candidate, limit=limit)
    else:
        lifecycle = _empty_lifecycle()
    semantic = _semantic_quality(store, candidate)
    status, blocker_status, next_action = _classify(candidate, lifecycle, semantic)
    executor = candidate.get("local_executor_compatibility") if isinstance(candidate.get("local_executor_compatibility"), dict) else {}
    return {
        "schema": ROW_SCHEMA,
        "status": status,
        "candidate_status": _label(candidate.get("status")),
        "candidate_fingerprint": _candidate_ref(candidate),
        "source_surface": source_surface,
        "app_family": app_family,
        "category": _label(candidate.get("category")),
        "workflow_phase": _label(candidate.get("workflow_phase")),
        "requested_model": _label(candidate.get("requested_model")),
        "target_model": _label(candidate.get("target_model")),
        "requested_model_family": _label(candidate.get("requested_model_family")),
        "target_model_family": _label(candidate.get("target_model_family")),
        "local_action_family": "routing",
        "local_executor": _label(executor.get("local_executor")) if executor else None,
        "matched_count": _as_int(lifecycle.get("matched_count")),
        "applied_count": _as_int(lifecycle.get("applied_count")),
        "holdout_count": _as_int(lifecycle.get("holdout_count")),
        "skipped_count": _as_int(lifecycle.get("skipped_count")),
        "bypassed_count": _as_int(lifecycle.get("bypassed_count")),
        "unknown_count": _as_int(lifecycle.get("unknown_count")),
        "error_count": _as_int(lifecycle.get("error_count")),
        "fallback_count": _as_int(lifecycle.get("fallback_count")),
        "retry_count": _as_int(lifecycle.get("retry_count")),
        "safety_stop_count": _as_int(lifecycle.get("safety_stop_count")),
        "coverage": lifecycle,
        "semantic_regression_status": semantic.get("status"),
        "semantic_quality": semantic,
        "blocker_status": blocker_status,
        "recommended_next_action": next_action,
        "candidate_suggested_next_action": _label(candidate.get("suggested_next_action")),
        "matrix_age_hours": candidate.get("matrix_age_hours"),
        "stale": bool(candidate.get("stale")),
        "review_only": True,
        "authoritative_for_active_policy": False,
        "feature_only": True,
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy(),
    }


def build_local_routing_pathway_outcome_feedback(
    store: Any,
    source: dict[str, Any],
    *,
    limit: int = 1000,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
) -> dict[str, Any]:
    candidates = _candidate_rows(source, stale_after_hours=stale_after_hours)
    rows = [_outcome_row(store, candidate, limit=limit) for candidate in candidates]
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    source_counts = Counter(str(row.get("source_surface") or "unknown") for row in rows)
    app_counts = Counter(str(row.get("app_family") or "unknown") for row in rows)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": "tracked" if rows else "empty",
        "source_schema": _label(source.get("schema")),
        "source_candidate_schema": CANDIDATE_SCHEMA,
        "candidate_count": len(candidates),
        "review_only": True,
        "authoritative_for_active_policy": False,
        "managed_dependency": "optional",
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "summary": {
            "candidate_count": len(candidates),
            "outcome_count": len(rows),
            "observed_count": sum(1 for row in rows if row.get("status") == "observed"),
            "ready_count": sum(1 for row in rows if row.get("status") == "ready"),
            "blocked_count": sum(1 for row in rows if row.get("status") == "blocked"),
            "regressed_count": sum(1 for row in rows if row.get("status") == "regressed"),
            "stale_count": sum(1 for row in rows if row.get("status") == "stale"),
            "missing_coverage_count": sum(1 for row in rows if row.get("status") == "missing-coverage"),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in rows),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in rows),
            "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in rows),
            "error_count": sum(_as_int(row.get("error_count")) for row in rows),
            "fallback_count": sum(_as_int(row.get("fallback_count")) for row in rows),
            "retry_count": sum(_as_int(row.get("retry_count")) for row in rows),
            "status_counts": [{"value": key, "count": value} for key, value in sorted(status_counts.items())],
            "source_surface_counts": [{"value": key, "count": value} for key, value in sorted(source_counts.items())],
            "app_family_counts": [{"value": key, "count": value} for key, value in sorted(app_counts.items())],
        },
        "outcomes": rows,
        "privacy": _privacy(),
    }
    violations = managed_egress_violations(report)
    report["egress_guard"] = {
        "schema": "tokenclaw.managed_egress_guard.v1",
        "status": "passed" if not violations else "blocked",
        "blocked": bool(violations),
        "violation_count": len(violations),
        "raw_values_logged": False,
    }
    if violations:
        report["egress_guard"]["blocked_keys"] = sorted({item.get("key", "unknown") for item in violations})
    return report
