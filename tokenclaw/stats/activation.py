from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenclaw.paths import tokenclaw_config_path
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.recommendations import OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE
from tokenclaw.store import utc_now
from tokenclaw.stats import (
    MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
    MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT,
    _as_float,
    _as_int,
    _breakdown_from_counts,
    _copy_policy,
    _count_breakdown,
    _increment_count,
    _json_obj,
    _local_path_class,
    _money,
    _optimization_eval_reason_codes,
    _parse_utc_datetime,
    _safe_count_breakdown,
)

def _promotion_blocker_review_path() -> Path:
    raw = os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("promotion_blocker_recommendation_review.json")


def _post_promotion_priority_review_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_PRIORITY_REVIEW_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_priority_delta_review.json")


def _post_promotion_policy_draft_dry_run_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_POLICY_DRAFT_DRY_RUN_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_policy_draft_dry_run.json")


def _evidence_to_activation_plan_candidate_paths(package_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON"):
        raw = os.getenv(name)
        if raw:
            candidates.append(Path(raw).expanduser())
            return candidates
    ops_root = os.getenv("TOKENCLAW_OPS_ROOT")
    if ops_root:
        candidates.append(Path(ops_root).expanduser() / "runs" / "research" / "latest.plan.json")
        return candidates
    root = package_root or Path(__file__).resolve().parents[1]
    candidates.append(root.parent / "runs" / "research" / "latest.plan.json")
    for parent in (root, *root.parents):
        candidates.append(parent / "tokenclaw_ops" / "runs" / "research" / "latest.plan.json")
    candidates.append(tokenclaw_config_path("research/latest.plan.json"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _evidence_to_activation_plan_path() -> Path:
    candidates = _evidence_to_activation_plan_candidate_paths()
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _post_promotion_outcome_flush_status_path() -> Path:
    raw = os.getenv("TOKENCLAW_POST_PROMOTION_OUTCOME_FLUSH_STATUS_PATH")
    if raw:
        return Path(raw).expanduser()
    return tokenclaw_config_path("post_promotion_outcome_flush_status.json")


def _promotion_blocker_dashboard_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
    }


def _post_promotion_priority_handoff_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "file_paths_included": False,
        "individual_candidate_ids_included": False,
        "individual_action_ids_included": False,
        "individual_rule_ids_included": False,
        "artifact_payloads_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
    }


def _post_promotion_artifact_source(
    *,
    kind: str,
    path: Path,
    env_name: str,
    payload: dict[str, Any] | None,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "configured": bool(os.getenv(env_name)),
        "available": payload is not None,
        "status": status,
        "reason": reason,
        "schema": payload.get("schema") if isinstance(payload, dict) else None,
        "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
        "path_class": _local_path_class(path),
        "path_included": False,
        "payload_included": False,
    }


def _read_post_promotion_artifact(path: Path) -> tuple[dict[str, Any] | None, str, str]:
    if not path.exists():
        return None, "missing", "artifact-not-found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, "unavailable", f"artifact-unreadable:{exc.__class__.__name__}"
    if not isinstance(payload, dict):
        return None, "unavailable", "artifact-not-json-object"
    return payload, "available", "loaded-local-artifact"


def _post_promotion_action_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "widen-local-policy": 0,
        "collect-holdout-evidence": 0,
        "rollback-local-policy": 0,
        "keep-blocked": 0,
    }
    for candidate in candidates:
        action = public_label(candidate.get("next_action"), "keep-blocked")
        if action not in counts:
            action = "keep-blocked"
        counts[action] += 1
    return counts


def _post_promotion_status_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        status = public_label(candidate.get("status"), "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _post_promotion_noop_reasons(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reasons = candidate.get("no_op_reasons") if isinstance(candidate.get("no_op_reasons"), list) else []
        for reason in reasons:
            label = public_label(reason, "unknown")
            counts[label] = counts.get(label, 0) + 1
    return counts


def _post_promotion_handoff_freshness(generated_values: list[Any], *, now: datetime) -> dict[str, Any]:
    parsed_values = [_parse_utc_datetime(value) for value in generated_values if value]
    parsed = [value for value in parsed_values if value is not None]
    latest = max(parsed, default=None)
    if latest is None:
        return {
            "latest_artifact_at": None,
            "age_seconds": None,
            "state": "no-artifacts",
        }
    age_seconds = max(0, int((now - latest).total_seconds()))
    if age_seconds <= 6 * 60 * 60:
        state = "fresh"
    elif age_seconds <= 24 * 60 * 60:
        state = "aging"
    else:
        state = "stale"
    return {
        "latest_artifact_at": latest.isoformat(),
        "age_seconds": age_seconds,
        "state": state,
    }


def _post_promotion_next_safe_command(
    *,
    review_payload: dict[str, Any] | None,
    draft_payload: dict[str, Any] | None,
    flush_payload: dict[str, Any] | None,
    top_next_action: str | None,
) -> dict[str, Any]:
    if review_payload is None:
        return {
            "label": "fetch managed priority deltas",
            "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
            "read_only": True,
            "reason": "priority-review-missing",
        }
    if top_next_action == "collect-holdout-evidence":
        return {
            "label": "inspect holdout evidence successor",
            "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
            "read_only": True,
            "reason": "holdout-evidence-required",
        }
    if draft_payload is None and top_next_action in {"widen-local-policy", "rollback-local-policy", "keep-blocked"}:
        return {
            "label": "dry-run local policy handoff",
            "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
            "read_only": True,
            "reason": "policy-draft-dry-run-missing",
        }
    draft_status = public_label(draft_payload.get("status"), "unknown") if isinstance(draft_payload, dict) else "missing"
    if draft_status == "blocked":
        return {
            "label": "inspect dry-run impact gate blockers",
            "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
            "read_only": True,
            "reason": "impact-gate-blocked",
        }
    if flush_payload is None:
        return {
            "label": "dry-run post-promotion outcome flush",
            "command": "tokenclaw-managed-feedback-status --post-promotion-action-outcomes --dry-run --pretty",
            "read_only": True,
            "reason": "outcome-flush-status-missing",
        }
    return {
        "label": "review handoff status",
        "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
        "read_only": True,
        "reason": "handoff-artifacts-present",
    }


async def stats_post_promotion_priority_handoff() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    review_path = _post_promotion_priority_review_path()
    draft_path = _post_promotion_policy_draft_dry_run_path()
    flush_path = _post_promotion_outcome_flush_status_path()
    review_payload, review_status, review_reason = _read_post_promotion_artifact(review_path)
    draft_payload, draft_status, draft_reason = _read_post_promotion_artifact(draft_path)
    flush_payload, flush_status, flush_reason = _read_post_promotion_artifact(flush_path)

    if review_payload is not None and review_payload.get("schema") != "tokenclaw.post_promotion_priority_delta_review.v1":
        review_payload = None
        review_status = "unavailable"
        review_reason = "unexpected-priority-review-schema"
    if draft_payload is not None and draft_payload.get("schema") != "tokenclaw.post_promotion_policy_draft_dry_run.v1":
        draft_payload = None
        draft_status = "unavailable"
        draft_reason = "unexpected-policy-draft-schema"
    if flush_payload is not None and flush_payload.get("schema") not in {
        "tokenclaw.managed_feedback_flush.v1",
        "tokenclaw.post_promotion_action_outcome_rollup_flush_status.v1",
    }:
        flush_payload = None
        flush_status = "unavailable"
        flush_reason = "unexpected-outcome-flush-schema"

    review_summary = review_payload.get("summary") if isinstance(review_payload, dict) and isinstance(review_payload.get("summary"), dict) else {}
    review_candidates = review_payload.get("candidates") if isinstance(review_payload, dict) and isinstance(review_payload.get("candidates"), list) else []
    review_candidates = [item for item in review_candidates if isinstance(item, dict)]
    action_counts = _post_promotion_action_counts(review_candidates)
    status_counts = _post_promotion_status_counts(review_candidates)
    noop_reasons = _post_promotion_noop_reasons(review_candidates)
    top_next_action = public_label(review_summary.get("top_next_action"), "none") if review_payload else None
    if top_next_action in {None, "none"} and review_candidates:
        top_next_action = public_label(review_candidates[0].get("next_action"), "keep-blocked")

    draft_summary = draft_payload.get("summary") if isinstance(draft_payload, dict) and isinstance(draft_payload.get("summary"), dict) else {}
    impact_gate_status = "missing"
    if draft_payload is not None:
        blocked = _as_int(draft_summary.get("impact_gate_blocked_count"))
        passed = _as_int(draft_summary.get("impact_gate_pass_count"))
        if blocked:
            impact_gate_status = "blocked"
        elif passed or _as_int(draft_summary.get("impact_gate_count")):
            impact_gate_status = "passed"
        else:
            impact_gate_status = public_label(draft_payload.get("status"), "unknown")

    flush_nested = (
        flush_payload.get("post_promotion_action_outcome_rollups")
        if isinstance(flush_payload, dict) and isinstance(flush_payload.get("post_promotion_action_outcome_rollups"), dict)
        else flush_payload
        if isinstance(flush_payload, dict) and flush_payload.get("schema") == "tokenclaw.post_promotion_action_outcome_rollup_flush_status.v1"
        else {}
    )
    flush_summary = flush_payload.get("flush") if isinstance(flush_payload, dict) and isinstance(flush_payload.get("flush"), dict) else {}
    flush_status_label = (
        public_label(flush_nested.get("status"), "unknown")
        if isinstance(flush_nested, dict) and flush_nested
        else public_label(flush_summary.get("status"), "missing")
        if isinstance(flush_summary, dict) and flush_summary
        else "missing"
    )
    freshness = _post_promotion_handoff_freshness(
        [
            review_payload.get("generated_at") if isinstance(review_payload, dict) else None,
            draft_payload.get("generated_at") if isinstance(draft_payload, dict) else None,
            flush_payload.get("generated_at") if isinstance(flush_payload, dict) else None,
        ],
        now=now,
    )
    command = _post_promotion_next_safe_command(
        review_payload=review_payload,
        draft_payload=draft_payload,
        flush_payload=flush_payload,
        top_next_action=top_next_action,
    )
    available_count = sum(1 for payload in (review_payload, draft_payload, flush_payload) if payload is not None)
    overall_status = "available" if review_payload is not None else "no-data" if available_count == 0 else "partial"
    return {
        "schema": "tokenclaw.post_promotion_priority_handoff_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": overall_status,
        "status_reason": "priority handoff artifacts loaded" if review_payload is not None else "post-promotion priority review artifact not found",
        "summary": {
            "artifact_count": available_count,
            "priority_review_status": review_status,
            "priority_review_candidate_count": _as_int(review_summary.get("review_candidate_count")) or len(review_candidates),
            "recommended_count": _as_int(review_summary.get("recommended_count")),
            "noop_count": _as_int(review_summary.get("noop_count")),
            "top_next_action": top_next_action,
            "widen_count": action_counts["widen-local-policy"],
            "collect_holdout_evidence_count": action_counts["collect-holdout-evidence"],
            "rollback_count": action_counts["rollback-local-policy"],
            "keep_blocked_count": action_counts["keep-blocked"],
            "policy_draft_status": public_label(draft_payload.get("status"), draft_status) if isinstance(draft_payload, dict) else draft_status,
            "draft_count": _as_int(draft_summary.get("draft_count")),
            "widen_draft_count": _as_int(draft_summary.get("widen_draft_count")),
            "rollback_draft_count": _as_int(draft_summary.get("rollback_draft_count")),
            "omitted_count": _as_int(draft_summary.get("omitted_count")),
            "impact_gate_status": impact_gate_status,
            "impact_gate_blocked_count": _as_int(draft_summary.get("impact_gate_blocked_count")),
            "outcome_flush_status": flush_status_label,
            "outcome_rollup_count": _as_int(flush_nested.get("rollup_count")) if isinstance(flush_nested, dict) else 0,
            "outcome_flush_reason": public_label(flush_nested.get("reason"), "none") if isinstance(flush_nested, dict) and flush_nested else public_label(flush_reason, "none"),
            "freshness_state": freshness["state"],
            "latest_artifact_at": freshness["latest_artifact_at"],
            "latest_artifact_age_seconds": freshness["age_seconds"],
            "next_safe_command": command["command"],
            "next_command_reason": command["reason"],
        },
        "status_counts": _breakdown_from_counts(status_counts),
        "next_action_counts": _breakdown_from_counts(action_counts),
        "no_op_reason_counts": _breakdown_from_counts(noop_reasons)[:10],
        "impact_gate_blocker_reason_counts": _safe_count_breakdown(draft_summary.get("impact_gate_blocker_reason_counts")),
        "sources": {
            "priority_review": _post_promotion_artifact_source(
                kind="priority-review-report",
                path=review_path,
                env_name="TOKENCLAW_POST_PROMOTION_PRIORITY_REVIEW_PATH",
                payload=review_payload,
                status=review_status,
                reason=review_reason,
            ),
            "policy_draft_dry_run": _post_promotion_artifact_source(
                kind="policy-draft-dry-run-report",
                path=draft_path,
                env_name="TOKENCLAW_POST_PROMOTION_POLICY_DRAFT_DRY_RUN_PATH",
                payload=draft_payload,
                status=draft_status,
                reason=draft_reason,
            ),
            "outcome_flush_status": _post_promotion_artifact_source(
                kind="outcome-flush-status-report",
                path=flush_path,
                env_name="TOKENCLAW_POST_PROMOTION_OUTCOME_FLUSH_STATUS_PATH",
                payload=flush_payload,
                status=flush_status,
                reason=flush_reason,
            ),
        },
        "commands": [
            command,
            {
                "label": "fetch managed priority deltas",
                "command": "tokenclaw-post-promotion-priority-delta-review --pretty",
                "read_only": True,
            },
            {
                "label": "dry-run local policy handoff",
                "command": "tokenclaw-post-promotion-policy-draft-dry-run post_promotion_priority_delta_review.json --pretty",
                "read_only": True,
            },
            {
                "label": "dry-run post-promotion outcome flush",
                "command": "tokenclaw-managed-feedback-status --post-promotion-action-outcomes --dry-run --pretty",
                "read_only": True,
            },
        ],
        "privacy": _post_promotion_priority_handoff_privacy(),
    }


def _promotion_blocker_no_data(reason: str, *, source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or _promotion_blocker_review_path()
    return {
        "schema": "tokenclaw.promotion_blocker_next_actions_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": "no-data",
        "status_reason": reason,
        "source": {
            "kind": "local-review-report",
            "configured": bool(os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")),
            "available": False,
            "path_class": _local_path_class(path),
            "path_included": False,
        },
        "summary": {
            "source_recommendation_count": 0,
            "review_candidate_count": 0,
            "group_count": 0,
            "recommended_count": 0,
            "noop_count": 0,
            "stale_evidence_count": 0,
            "projected_savings_usd": 0.0,
            "top_local_action_family": None,
            "top_blocker_reason": None,
            "top_next_action": None,
            "top_expected_local_executor": None,
        },
        "family_counts": [],
        "top_blocker_reasons": [],
        "expected_local_executors": [],
        "next_actions": [],
        "groups": [],
        "commands": [
            {
                "label": "review local promotion blocker recommendations",
                "command": "tokenclaw-optimization-promotion-blocker-review recommendations.json --pretty",
                "read_only": True,
            }
        ],
        "privacy": _promotion_blocker_dashboard_privacy(),
    }


def _promotion_blocker_reason_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reasons = candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else []
        for reason in reasons:
            key = public_label(reason, "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts


def _promotion_blocker_stale_count(candidates: list[dict[str, Any]]) -> int:
    total = 0
    for candidate in candidates:
        reasons = candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else []
        noops = candidate.get("no_op_reasons") if isinstance(candidate.get("no_op_reasons"), list) else []
        values = [str(item).lower() for item in [*reasons, *noops]]
        if any("stale" in value for value in values):
            total += 1
    return total


def _promotion_blocker_public_group(group: dict[str, Any], *, candidate_limit: int = 3) -> dict[str, Any]:
    recommendations = group.get("recommendations") if isinstance(group.get("recommendations"), list) else []
    public_candidates: list[dict[str, Any]] = []
    for candidate in recommendations[: max(0, candidate_limit)]:
        if not isinstance(candidate, dict):
            continue
        file_backed = candidate.get("file_backed_policy_representation")
        public_candidates.append(
            {
                "rank": _as_int(candidate.get("rank")),
                "status": public_label(candidate.get("status"), "unknown"),
                "recommendation_type": public_label(candidate.get("recommendation_type"), "unknown"),
                "candidate_family": public_label(candidate.get("candidate_family"), "unknown"),
                "blocker_family": public_label(candidate.get("blocker_family"), "unknown"),
                "blocker_reason_codes": [
                    public_label(reason, "unknown")
                    for reason in (candidate.get("blocker_reason_codes") if isinstance(candidate.get("blocker_reason_codes"), list) else [])[:5]
                ],
                "next_action": public_label(candidate.get("next_action"), "unknown"),
                "safety_stop_reason_code": public_label(candidate.get("safety_stop_reason_code"), "none") if candidate.get("safety_stop_reason_code") else None,
                "recommended_blocker_state": public_label(candidate.get("recommended_blocker_state"), "unknown") if candidate.get("recommended_blocker_state") else None,
                "recommended_unblock_action": public_label(candidate.get("recommended_unblock_action"), "unknown") if candidate.get("recommended_unblock_action") else None,
                "expected_local_executor": public_label(candidate.get("expected_local_executor"), "none"),
                "projected_savings_usd": _money(candidate.get("projected_savings_usd")),
                "file_backed_policy_exists": bool(file_backed.get("exists")) if isinstance(file_backed, dict) else False,
                "required_local_review": True,
            }
        )
    reason_counts = group.get("blocker_reason_code_counts") if isinstance(group.get("blocker_reason_code_counts"), list) else []
    safety_reason_counts = group.get("safety_stop_reason_counts") if isinstance(group.get("safety_stop_reason_counts"), list) else []
    top_reason = reason_counts[0].get("value") if reason_counts and isinstance(reason_counts[0], dict) else None
    top_safety_reason = safety_reason_counts[0].get("value") if safety_reason_counts and isinstance(safety_reason_counts[0], dict) else None
    return {
        "rank": _as_int(group.get("rank")),
        "local_action_family": public_label(group.get("local_action_family"), "unknown"),
        "candidate_count": _as_int(group.get("candidate_count")),
        "recommended_count": _as_int(group.get("recommended_count")),
        "noop_count": _as_int(group.get("noop_count")),
        "projected_savings_usd": _money(group.get("projected_savings_usd")),
        "top_next_action": public_label(group.get("top_next_action"), "unknown"),
        "top_blocker_reason": public_label(top_reason, "none") if top_reason else None,
        "top_safety_stop_reason": public_label(top_safety_reason or group.get("top_safety_stop_reason"), "none") if (top_safety_reason or group.get("top_safety_stop_reason")) else None,
        "blocker_reason_code_counts": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in reason_counts[:5]
            if isinstance(row, dict)
        ],
        "safety_stop_reason_counts": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in safety_reason_counts[:5]
            if isinstance(row, dict)
        ],
        "sample_recommendations": public_candidates,
    }


async def stats_promotion_blocker_next_actions(limit: int = 20) -> dict[str, Any]:
    path = _promotion_blocker_review_path()
    if not path.exists():
        return _promotion_blocker_no_data("local promotion blocker review report not found", source_path=path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result = _promotion_blocker_no_data(f"local promotion blocker review report unreadable: {exc.__class__.__name__}", source_path=path)
        result["status"] = "unavailable"
        return result
    if not isinstance(payload, dict) or payload.get("schema") != "tokenclaw.promotion_blocker_recommendation_review.v1":
        result = _promotion_blocker_no_data("local report is not a promotion blocker recommendation review", source_path=path)
        result["status"] = "unavailable"
        return result

    bounded_limit = max(0, min(_as_int(limit), 50))
    source_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    candidates = [item for item in candidates if isinstance(item, dict)]
    groups = payload.get("groups") if isinstance(payload.get("groups"), list) else []
    public_groups = [
        _promotion_blocker_public_group(group)
        for group in groups[:bounded_limit]
        if isinstance(group, dict)
    ]
    reason_counts = _promotion_blocker_reason_counts(candidates)
    family_counts: dict[str, int] = {}
    executor_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    for candidate in candidates:
        family = public_label(candidate.get("local_action_family"), "unknown")
        executor = public_label(candidate.get("expected_local_executor"), "none")
        action = public_label(candidate.get("next_action"), "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        executor_counts[executor] = executor_counts.get(executor, 0) + 1
        next_action_counts[action] = next_action_counts.get(action, 0) + 1
    top_reasons = _breakdown_from_counts(reason_counts)[:10]
    safety_counts = source_summary.get("safety_stop_reason_counts") if isinstance(source_summary.get("safety_stop_reason_counts"), list) else []
    top_reason = top_reasons[0]["value"] if top_reasons else None
    top_executors = _breakdown_from_counts(executor_counts)[:10]
    top_actions = _breakdown_from_counts(next_action_counts)[:10]
    top_candidate = candidates[0] if candidates else {}
    generated_at = payload.get("generated_at") if isinstance(payload.get("generated_at"), str) else None
    return {
        "schema": "tokenclaw.promotion_blocker_next_actions_dashboard.v1",
        "ok": True,
        "read_only": True,
        "generated_at": utc_now(),
        "status": "available" if candidates else "no-data",
        "status_reason": "loaded local promotion blocker review report" if candidates else "local review report has no candidates",
        "source": {
            "kind": "local-review-report",
            "configured": bool(os.getenv("TOKENCLAW_PROMOTION_BLOCKER_REVIEW_PATH")),
            "available": True,
            "generated_at": generated_at,
            "source_schema": payload.get("source_schema"),
            "path_class": _local_path_class(path),
            "path_included": False,
        },
        "summary": {
            "source_recommendation_count": _as_int(source_summary.get("source_recommendation_count")),
            "review_candidate_count": _as_int(source_summary.get("review_candidate_count")),
            "group_count": _as_int(source_summary.get("group_count")),
            "recommended_count": _as_int(source_summary.get("recommended_count")),
            "noop_count": _as_int(source_summary.get("noop_count")),
            "stale_evidence_count": _promotion_blocker_stale_count(candidates),
            "projected_savings_usd": _money(source_summary.get("projected_savings_usd")),
            "top_local_action_family": public_label(source_summary.get("top_local_action_family"), "none"),
            "top_blocker_reason": top_reason,
            "top_safety_stop_reason": public_label(source_summary.get("top_safety_stop_reason"), "none") if source_summary.get("top_safety_stop_reason") else None,
            "safety_stop_reason_count": _as_int(source_summary.get("safety_stop_reason_count")),
            "top_next_action": public_label(source_summary.get("top_next_action"), "none"),
            "top_expected_local_executor": public_label(top_candidate.get("expected_local_executor"), "none") if top_candidate else None,
        },
        "family_counts": _breakdown_from_counts(family_counts)[:10],
        "top_blocker_reasons": top_reasons,
        "top_safety_stop_reasons": [
            {"value": public_label(row.get("value"), "unknown"), "count": _as_int(row.get("count"))}
            for row in safety_counts[:10]
            if isinstance(row, dict)
        ],
        "expected_local_executors": top_executors,
        "next_actions": top_actions,
        "groups": public_groups,
        "commands": [
            {
                "label": "review local promotion blocker recommendations",
                "command": "tokenclaw-optimization-promotion-blocker-review recommendations.json --pretty",
                "read_only": True,
            },
            {
                "label": "queue local shadow eval tasks",
                "command": "tokenclaw-optimization-eval-next --promotion-blocker-review review.json --dry-run --pretty",
                "read_only": True,
            },
            {
                "label": "inspect promotion funnel",
                "command": "tokenclaw-optimization-promotion-report --pretty",
                "read_only": True,
            },
        ],
        "privacy": _promotion_blocker_dashboard_privacy(),
    }


def _promotion_privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
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
        "basis": "optimization eval plan metadata, sanitized promotion verdicts, policy-event summaries, and stored canary decision metadata only",
    }


def _promotion_canary_meta_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("phase_canary", "promotion_canary", "optimization_promotion_canary"):
        value = decision.get(key)
        if isinstance(value, dict) and (value.get("target_candidate_id") or value.get("promotion_action_id") or value.get("action_id")):
            return value
    canary = decision.get("canary")
    if isinstance(canary, dict) and (canary.get("target_candidate_id") or canary.get("promotion_action_id") or canary.get("action_id")):
        return canary
    return None


def _promotion_new_observed_bucket(candidate_id: str, action_id: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "action_id": action_id,
        "policy_section": None,
        "policy_source": None,
        "canary_fraction": None,
        "holdout_fraction": None,
        "observed_count": 0,
        "applied_count": 0,
        "holdout_count": 0,
        "skipped_count": 0,
        "bypassed_count": 0,
        "safety_stop_count": 0,
        "applied_error_count": 0,
        "holdout_error_count": 0,
        "applied_retry_count": 0,
        "holdout_retry_count": 0,
        "applied_latency_ms_total": 0,
        "holdout_latency_ms_total": 0,
        "applied_latency_count": 0,
        "holdout_latency_count": 0,
        "observed_savings_usd": 0.0,
        "last_observed_at": None,
        "reason_counts": {},
        "source_surface_counts": {},
    }


def _promotion_observed_canary_rows(store_obj: Any, limit: int) -> dict[str, dict[str, Any]]:
    rows = store_obj.conn.execute(
        """
        select created_at, source_surface, status_code, latency_ms, retry_count,
               cost_est_usd, cost_baseline_usd, routing_json, crunch_json, cache_json
        from calls
        where routing_json is not null or crunch_json is not null or cache_json is not null
        order by created_at desc
        limit ?
        """,
        (max(1, min(int(limit or 500), 10_000)),),
    ).fetchall()
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        decisions = (
            _json_obj(row.get("routing_json")),
            _json_obj(row.get("crunch_json")),
            _json_obj(row.get("cache_json")),
        )
        meta = next((item for item in (_promotion_canary_meta_from_decision(decision) for decision in decisions) if item), None)
        if not meta:
            continue
        candidate_id = str(meta.get("target_candidate_id") or meta.get("candidate_id") or "unknown")
        if candidate_id == "unknown":
            continue
        action_id = meta.get("promotion_action_id") or meta.get("action_id")
        bucket = buckets.setdefault(candidate_id, _promotion_new_observed_bucket(candidate_id, str(action_id) if action_id else None))
        if action_id and not bucket.get("action_id"):
            bucket["action_id"] = str(action_id)
        for key in ("policy_section", "policy_source", "canary_fraction", "holdout_fraction"):
            if bucket.get(key) is None and meta.get(key) is not None:
                bucket[key] = meta.get(key)
        source_surface = str(row.get("source_surface") or meta.get("source_surface") or "unknown")
        _increment_count(bucket["source_surface_counts"], source_surface)

        status = str(meta.get("status") or "")
        cohort = str(meta.get("cohort") or "")
        reason = str(meta.get("reason") or "")
        if reason:
            _increment_count(bucket["reason_counts"], reason)
        safety = meta.get("safety_stop") if isinstance(meta.get("safety_stop"), dict) else {}
        for code in safety.get("reason_codes") or []:
            _increment_count(bucket["reason_counts"], code)

        bucket["observed_count"] += 1
        if status == "applied" or cohort == "canary_applied":
            bucket["applied_count"] += 1
            if _as_int(row.get("status_code")) >= 400:
                bucket["applied_error_count"] += 1
            if _as_int(row.get("retry_count")) > 0:
                bucket["applied_retry_count"] += 1
            latency = _as_int(row.get("latency_ms"))
            if latency > 0:
                bucket["applied_latency_ms_total"] += latency
                bucket["applied_latency_count"] += 1
            savings = _as_float(row.get("cost_baseline_usd")) - _as_float(row.get("cost_est_usd"))
            if savings > 0:
                bucket["observed_savings_usd"] += savings
        elif status == "holdout" or cohort == "canary_holdout":
            bucket["holdout_count"] += 1
            if _as_int(row.get("status_code")) >= 400:
                bucket["holdout_error_count"] += 1
            if _as_int(row.get("retry_count")) > 0:
                bucket["holdout_retry_count"] += 1
            latency = _as_int(row.get("latency_ms"))
            if latency > 0:
                bucket["holdout_latency_ms_total"] += latency
                bucket["holdout_latency_count"] += 1
        elif status == "safety_stopped" or reason in {"local-canary-safety-stop", "safety-stop-tripped"} or safety.get("tripped"):
            bucket["safety_stop_count"] += 1
            bucket["bypassed_count"] += 1
        elif status in {"skipped", "not_selected"} or cohort == "skipped":
            bucket["skipped_count"] += 1
        else:
            bucket["bypassed_count"] += 1
        created_at = row.get("created_at")
        if created_at and (not bucket.get("last_observed_at") or str(created_at) > str(bucket.get("last_observed_at"))):
            bucket["last_observed_at"] = str(created_at)
    return buckets


def _promotion_lifecycle_rows(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for event in events:
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if not details:
            continue
        text = " ".join(str(value or "") for value in (event.get("action"), details.get("command"), details.get("lifecycle_kind"), details.get("schema")))
        if "optimization" not in text and "promotion" not in text:
            continue
        candidate_ids = details.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate = details.get("target_candidate_id") or details.get("candidate_id")
            candidate_ids = [candidate] if candidate else []
        action_ids = details.get("action_ids")
        if not isinstance(action_ids, list):
            action = details.get("promotion_action_id") or details.get("action_id")
            action_ids = [action] if action else []
        for candidate in candidate_ids:
            candidate_id = str(candidate or "")
            if not candidate_id:
                continue
            bucket = buckets.setdefault(candidate_id, {
                "candidate_id": candidate_id,
                "latest_event_at": None,
                "event_count": 0,
                "action_ids": set(),
                "applied_count": 0,
                "holdout_count": 0,
                "skipped_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "reason_counts": {},
            })
            bucket["event_count"] += 1
            if event.get("created_at") and (not bucket.get("latest_event_at") or str(event.get("created_at")) > str(bucket.get("latest_event_at"))):
                bucket["latest_event_at"] = str(event.get("created_at"))
            for action_id in action_ids:
                if action_id:
                    bucket["action_ids"].add(str(action_id))
            bucket["applied_count"] += _as_int(details.get("actual_canary_applied_count") or details.get("canary_applied_count") or details.get("applied_count"))
            bucket["holdout_count"] += _as_int(details.get("actual_canary_holdout_count") or details.get("canary_holdout_count") or details.get("holdout_count"))
            bucket["skipped_count"] += _as_int(details.get("skipped_count"))
            bucket["safety_stop_count"] += _as_int(details.get("safety_stop_count"))
            if str(event.get("action") or "").endswith("rollback") or str(details.get("event_type") or "") == "rollback":
                bucket["rollback_count"] += 1
            for key in ("reason", "status", "event_type", "local_result_status"):
                if details.get(key):
                    _increment_count(bucket["reason_counts"], details.get(key))
            reason_counts = details.get("reason_counts") or details.get("reason_code_counts")
            if isinstance(reason_counts, dict):
                for reason, count in reason_counts.items():
                    bucket["reason_counts"][str(reason)] = bucket["reason_counts"].get(str(reason), 0) + _as_int(count)
    result: dict[str, dict[str, Any]] = {}
    for candidate_id, bucket in buckets.items():
        public = dict(bucket)
        public["action_ids"] = sorted(public["action_ids"])
        public["reason_counts"] = _count_breakdown(public["reason_counts"])
        result[candidate_id] = public
    return result


def _promotion_finalize_observed(bucket: dict[str, Any] | None) -> dict[str, Any]:
    if not bucket:
        return {
            "observed_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "skipped_count": 0,
            "bypassed_count": 0,
            "safety_stop_count": 0,
            "observed_savings_usd": 0.0,
            "applied_error_rate": 0.0,
            "holdout_error_rate": 0.0,
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "latency_delta_ms": None,
            "last_observed_at": None,
            "reason_counts": [],
            "source_surface_counts": [],
        }
    applied = _as_int(bucket.get("applied_count"))
    holdout = _as_int(bucket.get("holdout_count"))
    applied_latency = None
    holdout_latency = None
    if _as_int(bucket.get("applied_latency_count")):
        applied_latency = _as_float(bucket.get("applied_latency_ms_total")) / _as_int(bucket.get("applied_latency_count"))
    if _as_int(bucket.get("holdout_latency_count")):
        holdout_latency = _as_float(bucket.get("holdout_latency_ms_total")) / _as_int(bucket.get("holdout_latency_count"))
    applied_error_rate = (_as_int(bucket.get("applied_error_count")) / applied) if applied else 0.0
    holdout_error_rate = (_as_int(bucket.get("holdout_error_count")) / holdout) if holdout else 0.0
    applied_retry_rate = (_as_int(bucket.get("applied_retry_count")) / applied) if applied else 0.0
    holdout_retry_rate = (_as_int(bucket.get("holdout_retry_count")) / holdout) if holdout else 0.0
    return {
        "action_id": bucket.get("action_id"),
        "policy_section": bucket.get("policy_section"),
        "policy_source": bucket.get("policy_source"),
        "canary_fraction": bucket.get("canary_fraction"),
        "holdout_fraction": bucket.get("holdout_fraction"),
        "observed_count": _as_int(bucket.get("observed_count")),
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": _as_int(bucket.get("skipped_count")),
        "bypassed_count": _as_int(bucket.get("bypassed_count")),
        "safety_stop_count": _as_int(bucket.get("safety_stop_count")),
        "observed_savings_usd": round(_as_float(bucket.get("observed_savings_usd")), 8),
        "applied_error_rate": round(applied_error_rate, 6),
        "holdout_error_rate": round(holdout_error_rate, 6),
        "error_rate_delta": round(applied_error_rate - holdout_error_rate, 6),
        "applied_retry_rate": round(applied_retry_rate, 6),
        "holdout_retry_rate": round(holdout_retry_rate, 6),
        "retry_rate_delta": round(applied_retry_rate - holdout_retry_rate, 6),
        "latency_delta_ms": round(applied_latency - holdout_latency, 2) if applied_latency is not None and holdout_latency is not None else None,
        "last_observed_at": bucket.get("last_observed_at"),
        "reason_counts": _count_breakdown(bucket.get("reason_counts") or {}),
        "source_surface_counts": _count_breakdown(bucket.get("source_surface_counts") or {}),
    }


def _promotion_primary_state(verdict: str, eval_evidence: dict[str, Any], observed: dict[str, Any], lifecycle: dict[str, Any] | None) -> str:
    if _as_int(observed.get("safety_stop_count")) or _as_int((lifecycle or {}).get("safety_stop_count")):
        return "safety-stopped"
    if verdict == "rollback" or _as_int((lifecycle or {}).get("rollback_count")):
        return "rollback-recommended"
    if verdict == "widen":
        return "widening-eligible"
    if _as_int(observed.get("applied_count")) or _as_int(observed.get("holdout_count")):
        return "canary-active"
    if _as_int(eval_evidence.get("pass_count")):
        return "eval-passed"
    return "needs-eval"


def _promotion_policy_section(row: dict[str, Any]) -> str:
    values = [
        str(row.get("action_family") or ""),
        str(row.get("optimization_family") or ""),
    ]
    normalized = [value.strip().lower().replace("_", "-") for value in values]
    joined = " ".join(normalized)
    if "old-context" in joined or "summarization" in joined or "summary" in joined:
        return "old_context_summarization"
    if "routing" in joined:
        return "routing"
    if "cache" in joined:
        return "cache"
    if "crunch" in joined or "pattern" in joined:
        return "crunch"
    return "unsupported"


def _promotion_target_local_policy_section(policy_section: str) -> str | None:
    if policy_section == "routing":
        return "routing.rules"
    if policy_section == "cache":
        return "cache.rules"
    if policy_section == "crunch":
        return "crunch.rules"
    if policy_section == "old_context_summarization":
        return "crunch.old_context_summarization"
    return None


def _promotion_next_command(status: str) -> tuple[str, str]:
    if status in {"pending-lifecycle-feedback", "impact-stale", "needs-more-samples"}:
        return "promotion-impact", "tokenclaw-optimization-promotion-impact promotion-actions.json --pretty"
    if status == "supported":
        return "promotion-canaries-apply --dry-run", "tokenclaw-optimization-promotion-canaries-apply promotion-actions.json --dry-run --pretty"
    return "promotion-actions", "tokenclaw-optimization-promotion-actions --pretty"


def _promotion_impact_stale(last_evidence_at: Any, *, max_age_hours: int = 168) -> bool:
    parsed = _parse_utc_datetime(last_evidence_at)
    if parsed is None:
        return False
    age = datetime.now(timezone.utc) - parsed
    return age.total_seconds() > max(1, max_age_hours) * 3600


def _promotion_reason_has_any(reasons: list[str], *needles: str) -> bool:
    haystack = " ".join(str(reason or "").lower() for reason in reasons)
    return any(needle in haystack for needle in needles)


def _promotion_pending_lifecycle_rows(store_obj: Any) -> dict[str, dict[str, Any]]:
    if not hasattr(store_obj, "managed_outcome_feedback_payload_rows"):
        return {}
    try:
        rows = store_obj.managed_outcome_feedback_payload_rows(
            source_surface=OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE,
            limit=1000,
        )
    except Exception:
        return {}

    pending_statuses = {"queued", "retryable-error", "claimed"}
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        status = str(row.get("status") or "")
        if status not in pending_statuses:
            continue
        payload = _json_obj(row.get("payload_json"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        candidate_ids = metadata.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        action_ids = metadata.get("action_ids")
        if not isinstance(action_ids, list):
            action_ids = []
        for candidate in candidate_ids:
            candidate_id = str(candidate or "")
            if not candidate_id:
                continue
            bucket = buckets.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "pending_count": 0,
                    "statuses": {},
                    "commands": {},
                    "action_ids": set(),
                    "latest_queued_at": None,
                },
            )
            bucket["pending_count"] += 1
            _increment_count(bucket["statuses"], status)
            _increment_count(bucket["commands"], metadata.get("command") or payload.get("event_type") or "unknown")
            for action_id in action_ids:
                if action_id:
                    bucket["action_ids"].add(str(action_id))
            created_at = row.get("created_at") or row.get("updated_at") or payload.get("occurred_at")
            if created_at and (not bucket.get("latest_queued_at") or str(created_at) > str(bucket.get("latest_queued_at"))):
                bucket["latest_queued_at"] = str(created_at)
    result: dict[str, dict[str, Any]] = {}
    for candidate_id, bucket in buckets.items():
        public = dict(bucket)
        public["statuses"] = _count_breakdown(public["statuses"])
        public["commands"] = _count_breakdown(public["commands"])
        public["action_ids"] = sorted(public["action_ids"])
        result[candidate_id] = public
    return result


def _promotion_executor_readiness(
    *,
    verdict_row: dict[str, Any],
    primary_state: str,
    observed_row: dict[str, Any],
    lifecycle_row: dict[str, Any] | None,
    pending_lifecycle_row: dict[str, Any] | None,
    last_evidence_at: Any,
) -> dict[str, Any]:
    policy_section = _promotion_policy_section(verdict_row)
    supported = policy_section in {"routing", "crunch", "cache", "old_context_summarization"}
    reasons = _optimization_eval_reason_codes(verdict_row.get("reason_codes"))
    status = "supported"
    detail_reasons: list[str] = []
    if not supported:
        status = "unsupported"
        detail_reasons.append("unsupported-local-policy-section")
    elif policy_section == "cache" and _promotion_reason_has_any(reasons, "invalidation", "stale-risk"):
        status = "missing-invalidation-evidence"
        detail_reasons.append("cache-invalidation-evidence-required")
    elif primary_state in {"rollback-recommended", "safety-stopped"} or str(verdict_row.get("verdict") or "") == "rollback":
        status = "rollback-recommended"
    elif pending_lifecycle_row:
        status = "pending-lifecycle-feedback"
    elif primary_state == "canary-active" and _promotion_impact_stale(last_evidence_at):
        status = "impact-stale"
    elif primary_state == "widening-eligible" or str(verdict_row.get("verdict") or "") == "widen":
        status = "widening-eligible"
    elif primary_state in {"needs-eval", "canary-active"} and (
        _promotion_reason_has_any(reasons, "insufficient-")
        or _as_int((observed_row or {}).get("applied_count"))
        or _as_int((observed_row or {}).get("holdout_count"))
    ):
        status = "needs-more-samples"
    elif primary_state == "needs-eval":
        status = "missing-local-evidence"
    elif primary_state == "eval-passed":
        status = "supported"

    command_kind, command = _promotion_next_command(status)
    pending = pending_lifecycle_row or {}
    return {
        "status": status,
        "supported": supported,
        "policy_section": policy_section,
        "target_local_policy_section": _promotion_target_local_policy_section(policy_section),
        "next_command_kind": command_kind,
        "next_command": command,
        "reason_codes": sorted(set(detail_reasons + reasons)),
        "pending_lifecycle_feedback_count": _as_int(pending.get("pending_count")),
        "pending_lifecycle_feedback_statuses": pending.get("statuses") or [],
        "pending_lifecycle_feedback_commands": pending.get("commands") or [],
        "impact_stale": status == "impact-stale",
        "privacy": _promotion_privacy_summary(),
    }


def _promotion_action_dashboard_row(action: dict[str, Any], *, rank: int) -> dict[str, Any]:
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    cohorts = evidence.get("cohort_counts") if isinstance(evidence.get("cohort_counts"), dict) else {}
    local_review = action.get("local_review") if isinstance(action.get("local_review"), dict) else {}
    return {
        "rank": rank,
        "status": str(action.get("status") or "planned"),
        "action_type": str(action.get("action_type") or "unknown"),
        "verdict": str(action.get("verdict") or "unknown"),
        "action_family": str(action.get("action_family") or "unknown"),
        "optimization_family": str(action.get("optimization_family") or "unknown"),
        "source_surface": str(action.get("source_surface") or "unknown"),
        "app_family": str(action.get("app_family") or "unknown"),
        "policy_section": str(action.get("policy_section") or "unknown"),
        "target_local_policy_section": action.get("target_local_policy_section"),
        "projected_savings_usd": round(_as_float(evidence.get("projected_savings_usd")), 8),
        "sample_count": _as_int(evidence.get("sample_count")),
        "canary_applied_count": _as_int(cohorts.get("canary_applied")),
        "canary_holdout_count": _as_int(cohorts.get("canary_holdout")),
        "bypassed_or_disabled_count": _as_int(cohorts.get("bypassed_or_disabled")),
        "eval_result_count": _as_int(evidence.get("eval_result_count")),
        "eval_pass_count": _as_int(evidence.get("eval_pass_count")),
        "eval_fail_count": _as_int(evidence.get("eval_fail_count")),
        "eval_blocked_count": _as_int(evidence.get("eval_blocked_count")),
        "latest_eval_result_at": evidence.get("latest_eval_result_at"),
        "eval_evidence_stale": bool(evidence.get("eval_evidence_stale")),
        "current_canary_fraction": round(_as_float(action.get("current_canary_fraction")), 6),
        "canary_fraction": round(_as_float(action.get("canary_fraction")), 6),
        "holdout_fraction": round(_as_float(action.get("holdout_fraction")), 6),
        "review_command": str(local_review.get("review_command") or ""),
        "apply_preview_command": str(local_review.get("apply_preview_command") or ""),
        "privacy": _promotion_privacy_summary(),
    }


def _promotion_omission_dashboard_bucket(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": _as_int(row.get("rank")),
        "reason": str(row.get("reason") or "unknown"),
        "action_family": str(row.get("action_family") or "unknown"),
        "candidate_count": _as_int(row.get("candidate_count")),
        "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
        "next_action": str(row.get("next_action") or "unknown"),
        "reason_codes": _optimization_eval_reason_codes(row.get("reason_codes")),
        "privacy": _promotion_privacy_summary(),
    }


def _post_promotion_privacy_summary() -> dict[str, Any]:
    privacy = {
        **_promotion_privacy_summary(),
        "content_free": True,
        "local_only": True,
        "individual_candidate_ids_included": False,
        "individual_action_ids_included": False,
        "individual_rule_ids_included": False,
        "session_ids_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "provider_bodies_included": False,
        "policy_file_contents_included": False,
    }
    return privacy


def _post_promotion_family(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("_", "-")
    if text in {"cache", "cache-replay"}:
        return "cache"
    if text in {"routing", "provider-routing", "phase-routing"}:
        return "routing"
    if text in {"crunch", "old-context-summary", "old-context-summarization"}:
        return "crunch"
    return public_label(text or "unknown", "unknown")


def _post_promotion_state(entry: dict[str, Any]) -> str:
    status = str(entry.get("status") or "").strip().lower().replace("_", "-")
    recommendation = str(entry.get("recommendation") or "").strip().lower().replace("_", "-")
    if entry.get("rollback_needed") or "rollback" in status or recommendation == "rollback":
        return "blocked"
    if "safety" in status or status in {"regression-flagged", "keep-blocked"}:
        return "blocked"
    if status in {"needs-more-samples", "needs-more-evidence", "needs-review"}:
        return "needs-evidence"
    if status in {"positive", "promoted", "widened"} or recommendation in {"promote", "widen"}:
        return "improving"
    if _as_float(entry.get("observed_savings_usd")) > 0:
        return "measured"
    return "observed"


def _post_promotion_next_action(state: str, latest: dict[str, Any], blocker: str | None) -> str:
    status = str(latest.get("status") or "").strip().lower().replace("_", "-")
    recommendation = str(latest.get("recommendation") or "").strip().lower().replace("_", "-")
    if state == "blocked":
        if "safety" in status or (blocker and "safety" in blocker):
            return "review-post-promotion-safety-blocker"
        if "rollback" in status or recommendation == "rollback":
            return "rollback-or-keep-promotion-blocked"
        return "review-post-promotion-regression"
    if state == "needs-evidence":
        return "collect-post-promotion-holdout-evidence"
    if recommendation in {"promote", "widen"}:
        return "widen-local-promotion"
    if state in {"improving", "measured"}:
        return "continue-measuring-post-promotion-impact"
    return "inspect-post-promotion-feedback"


def _post_promotion_status(states: list[str], latest: dict[str, Any] | None) -> str:
    if "blocked" in states:
        return "blocked"
    if "needs-evidence" in states:
        return "needs-evidence"
    if "improving" in states:
        return "improving"
    if "measured" in states:
        return "measured"
    return "observed" if latest else "no-feedback"


def _post_promotion_latest_blocker(entries: list[dict[str, Any]], safety_groups: list[dict[str, Any]]) -> str | None:
    for group in sorted(safety_groups, key=lambda row: str(row.get("rank") or ""), reverse=False):
        for key in ("keep_blocked_reason", "blocker_code", "safety_stop_reason"):
            value = str(group.get(key) or "").strip()
            if value:
                return public_label(value, "post-promotion-blocker")
    for entry in sorted(entries, key=lambda row: str(row.get("created_at") or row.get("impact_generated_at") or ""), reverse=True):
        codes = []
        for key in ("reason_codes", "warning_codes"):
            value = entry.get(key)
            if isinstance(value, list):
                codes.extend(_optimization_eval_reason_codes(value))
        if codes:
            return codes[0]
        if entry.get("rollback_needed"):
            return "rollback-needed"
        status = str(entry.get("status") or "").strip()
        if status and status not in {"positive", "observed"}:
            return public_label(status, "post-promotion-blocker")
    return None


def _post_promotion_delta_row(
    family: str,
    entries: list[dict[str, Any]],
    safety_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = max(entries, key=lambda row: str(row.get("created_at") or row.get("impact_generated_at") or ""), default=None)
    states = [_post_promotion_state(entry) for entry in entries]
    if safety_groups and "blocked" not in states:
        states.append("blocked")
    status = _post_promotion_status(states, latest)
    applied = sum(_as_int(entry.get("applied_count")) for entry in entries) + sum(_as_int(group.get("applied_count")) for group in safety_groups)
    holdout = sum(_as_int(entry.get("holdout_count")) for entry in entries) + sum(_as_int(group.get("holdout_count")) for group in safety_groups)
    safety_stops = sum(_as_int(entry.get("safety_stop_count")) for entry in entries) + sum(_as_int(group.get("safety_stop_count")) for group in safety_groups)
    observed = sum(_as_float(entry.get("observed_savings_usd")) for entry in entries)
    projected = sum(_as_float(entry.get("projected_savings_usd")) for entry in entries) + sum(_as_float(group.get("savings_estimate_usd")) for group in safety_groups)
    latest_blocker = _post_promotion_latest_blocker(entries, safety_groups)
    recommendation_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for entry in entries:
        recommendation = public_label(entry.get("recommendation"), "none")
        status_label = public_label(entry.get("status"), "unknown")
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1
        status_counts[status_label] = status_counts.get(status_label, 0) + 1
    return {
        "schema": "tokenclaw.post_promotion_blocker_delta.v1",
        "local_action_family": family,
        "status": status,
        "entry_count": len(entries),
        "safety_stop_group_count": len(safety_groups),
        "applied_count": applied,
        "holdout_count": holdout,
        "safety_stop_count": safety_stops,
        "latest_blocker_reason": latest_blocker,
        "observed_savings_usd": round(observed, 8),
        "projected_savings_usd": round(projected, 8),
        "savings_delta_usd": round(observed - projected, 8),
        "next_action": _post_promotion_next_action(status, latest or {}, latest_blocker),
        "latest_feedback_at": (latest or {}).get("created_at") or (latest or {}).get("impact_generated_at"),
        "status_counts": _breakdown_from_counts(status_counts),
        "recommendation_counts": _breakdown_from_counts(recommendation_counts),
        "privacy": _post_promotion_privacy_summary(),
    }


async def stats_post_promotion_deltas(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    from tokenclaw.activation_lifecycle_feedback import activation_safety_stop_burndown_report
    from tokenclaw.promotion_outcome_feedback import promotion_outcome_feedback_summary

    capped_limit = max(1, min(int(limit or 1000), 10_000))
    feedback = promotion_outcome_feedback_summary(store_obj, limit=capped_limit)
    safety = activation_safety_stop_burndown_report(store_obj, limit=capped_limit)
    entries_by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in feedback.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        family = _post_promotion_family(entry.get("action_family") or entry.get("policy_section"))
        entries_by_family.setdefault(family, []).append(entry)

    safety_by_family: dict[str, list[dict[str, Any]]] = {}
    for group in safety.get("groups") or []:
        if not isinstance(group, dict):
            continue
        family = _post_promotion_family(group.get("action_family"))
        safety_by_family.setdefault(family, []).append(group)

    families = sorted(set(entries_by_family) | set(safety_by_family))
    deltas = [
        _post_promotion_delta_row(
            family,
            entries_by_family.get(family, []),
            safety_by_family.get(family, []),
        )
        for family in families
    ]
    deltas.sort(
        key=lambda row: (
            row.get("status") != "blocked",
            -_as_int(row.get("safety_stop_count")),
            -abs(_as_float(row.get("savings_delta_usd"))),
            str(row.get("local_action_family") or ""),
        )
    )
    top = deltas[0] if deltas else {}
    privacy = _post_promotion_privacy_summary()
    return {
        "schema": "tokenclaw.post_promotion_blocker_deltas_dashboard.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "status": "available" if deltas else "no-feedback",
        "summary": {
            "family_count": len(deltas),
            "entry_count": sum(_as_int(row.get("entry_count")) for row in deltas),
            "blocked_family_count": sum(1 for row in deltas if row.get("status") == "blocked"),
            "needs_evidence_family_count": sum(1 for row in deltas if row.get("status") == "needs-evidence"),
            "applied_count": sum(_as_int(row.get("applied_count")) for row in deltas),
            "holdout_count": sum(_as_int(row.get("holdout_count")) for row in deltas),
            "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in deltas),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in deltas), 8),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in deltas), 8),
            "savings_delta_usd": round(sum(_as_float(row.get("savings_delta_usd")) for row in deltas), 8),
            "top_local_action_family": top.get("local_action_family"),
            "top_status": top.get("status"),
            "top_blocker_reason": top.get("latest_blocker_reason"),
            "top_next_action": top.get("next_action"),
        },
        "deltas": deltas,
        "source_reports": {
            "promotion_outcome_feedback_schema": feedback.get("schema"),
            "activation_safety_stop_burndown_schema": safety.get("schema"),
            "feedback_entry_count": feedback.get("entry_count"),
            "safety_stop_group_count": (safety.get("summary") or {}).get("ranked_group_count") if isinstance(safety.get("summary"), dict) else None,
        },
        "privacy": privacy,
    }


async def stats_optimization_promotion_actions(store_obj: Any, limit: int = 50) -> dict[str, Any]:
    from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
    from tokenclaw.optimization_promotion_actions import build_optimization_promotion_actions
    from tokenclaw.optimization_promotion_report import build_optimization_promotion_report

    capped_limit = max(1, min(int(limit or 50), 100))
    plan = await build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=1)
    promotion = build_optimization_promotion_report(store_obj, plan=plan, limit=capped_limit)
    promotion_actions = build_optimization_promotion_actions(promotion)
    raw_actions = [
        row
        for row in promotion_actions.get("actions", [])
        if isinstance(row, dict)
    ][:capped_limit]
    actions = [
        _promotion_action_dashboard_row(row, rank=index)
        for index, row in enumerate(raw_actions, start=1)
    ]
    omitted = [
        row
        for row in promotion_actions.get("omission_buckets", [])
        if isinstance(row, dict)
    ][:capped_limit]
    omission_buckets = [_promotion_omission_dashboard_bucket(row) for row in omitted]
    summary = promotion_actions.get("summary") if isinstance(promotion_actions.get("summary"), dict) else {}
    return {
        "schema": "tokenclaw.optimization_promotion_actions_dashboard.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": _as_int(summary.get("candidate_count")),
            "action_count": _as_int(summary.get("action_count")),
            "displayed_action_count": len(actions),
            "omitted_count": _as_int(summary.get("omitted_count")),
            "displayed_omission_bucket_count": len(omission_buckets),
            "policy_section_counts": summary.get("policy_section_counts") if isinstance(summary.get("policy_section_counts"), list) else [],
            "action_family_counts": summary.get("action_family_counts") if isinstance(summary.get("action_family_counts"), list) else [],
            "omission_reason_counts": summary.get("omission_reason_counts") if isinstance(summary.get("omission_reason_counts"), list) else [],
            "top_omission_next_action": summary.get("top_omission_next_action"),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in actions), 8),
            "canary_applied_count": sum(_as_int(row.get("canary_applied_count")) for row in actions),
            "canary_holdout_count": sum(_as_int(row.get("canary_holdout_count")) for row in actions),
            "latest_eval_result_at": max((str(row.get("latest_eval_result_at")) for row in actions if row.get("latest_eval_result_at")), default=None),
        },
        "actions": actions,
        "omission_buckets": omission_buckets,
        "source_reports": {
            "eval_plan_schema": plan.get("schema") if isinstance(plan, dict) else None,
            "promotion_report_schema": promotion.get("schema") if isinstance(promotion, dict) else None,
            "promotion_actions_schema": promotion_actions.get("schema") if isinstance(promotion_actions, dict) else None,
        },
        "privacy": {
            **_promotion_privacy_summary(),
            "individual_candidate_ids_included": False,
            "individual_action_ids_included": False,
        },
    }


async def stats_optimization_promotion_funnel(store_obj: Any, limit: int = 500) -> dict[str, Any]:
    from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
    from tokenclaw.optimization_promotion_actions import build_optimization_promotion_actions
    from tokenclaw.optimization_promotion_report import build_optimization_promotion_report
    from tokenclaw.policy_events import recent_policy_events

    capped_limit = max(1, min(int(limit or 500), 10_000))
    plan = await build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=1)
    promotion = build_optimization_promotion_report(store_obj, plan=plan, limit=capped_limit)
    promotion_actions = build_optimization_promotion_actions(promotion)
    observed = _promotion_observed_canary_rows(store_obj, capped_limit * 5)
    events = recent_policy_events(limit=500).get("events", [])
    lifecycle = _promotion_lifecycle_rows(events if isinstance(events, list) else [])
    pending_lifecycle = _promotion_pending_lifecycle_rows(store_obj)

    candidates: list[dict[str, Any]] = []
    state_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    policy_section_counts: dict[str, int] = {}
    readiness_policy_counts: dict[str, int] = {}
    action_family_policy_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    all_candidate_ids = {
        str(row.get("candidate_id"))
        for row in promotion.get("candidates", [])
        if isinstance(row, dict) and row.get("candidate_id")
    } | set(observed) | set(lifecycle) | set(pending_lifecycle)
    verdicts = {
        str(row.get("candidate_id")): row
        for row in promotion.get("candidates", [])
        if isinstance(row, dict) and row.get("candidate_id")
    }
    for candidate_id in sorted(all_candidate_ids):
        verdict_row = verdicts.get(candidate_id, {"candidate_id": candidate_id, "verdict": "needs_eval", "eval_evidence": {}})
        observed_row = _promotion_finalize_observed(observed.get(candidate_id))
        lifecycle_row = lifecycle.get(candidate_id)
        pending_lifecycle_row = pending_lifecycle.get(candidate_id)
        verdict = str(verdict_row.get("verdict") or "needs_eval")
        eval_evidence = verdict_row.get("eval_evidence") if isinstance(verdict_row.get("eval_evidence"), dict) else {}
        primary_state = _promotion_primary_state(verdict, eval_evidence, observed_row, lifecycle_row)
        last_evidence_at = max(
            (
                str(value)
                for value in (
                    eval_evidence.get("latest_result_at"),
                    observed_row.get("last_observed_at"),
                    (lifecycle_row or {}).get("latest_event_at"),
                    (pending_lifecycle_row or {}).get("latest_queued_at"),
                )
                if value
            ),
            default=None,
        )
        executor_readiness = _promotion_executor_readiness(
            verdict_row=verdict_row,
            primary_state=primary_state,
            observed_row=observed_row,
            lifecycle_row=lifecycle_row,
            pending_lifecycle_row=pending_lifecycle_row,
            last_evidence_at=last_evidence_at,
        )
        _increment_count(state_counts, primary_state)
        _increment_count(readiness_counts, executor_readiness["status"])
        _increment_count(policy_section_counts, executor_readiness["policy_section"])
        _increment_count(readiness_policy_counts, f"{executor_readiness['policy_section']}:{executor_readiness['status']}")
        _increment_count(action_family_policy_counts, f"{verdict_row.get('action_family') or 'unknown'}:{executor_readiness['policy_section']}")
        row_reasons = _optimization_eval_reason_codes(verdict_row.get("reason_codes"))
        for reason in row_reasons:
            _increment_count(reason_counts, reason)
        for reason in (observed_row.get("reason_counts") or [])[:5]:
            _increment_count(reason_counts, reason.get("value"))
        candidates.append({
            "candidate_id": candidate_id,
            "action_id": observed_row.get("action_id") or ((lifecycle_row or {}).get("action_ids") or [None])[0],
            "action_family": str(verdict_row.get("action_family") or "unknown"),
            "optimization_family": str(verdict_row.get("optimization_family") or "unknown"),
            "source_surface": str(verdict_row.get("source_surface") or "unknown"),
            "app_family": str(verdict_row.get("app_family") or "unknown"),
            "candidate_target_model": verdict_row.get("candidate_target_model"),
            "candidate_profile": verdict_row.get("candidate_profile"),
            "projected_savings_usd": round(_as_float(verdict_row.get("projected_savings_usd")), 8),
            "observed_savings_usd": observed_row.get("observed_savings_usd", 0.0),
            "verdict": verdict,
            "primary_state": primary_state,
            "policy_section": executor_readiness["policy_section"],
            "target_local_policy_section": executor_readiness["target_local_policy_section"],
            "executor_readiness": executor_readiness,
            "next_command_kind": executor_readiness["next_command_kind"],
            "next_command": executor_readiness["next_command"],
            "eval_pass_count": _as_int(eval_evidence.get("pass_count")),
            "eval_fail_count": _as_int(eval_evidence.get("fail_count")),
            "eval_result_count": _as_int(eval_evidence.get("result_count")),
            "eval_evidence_stale": bool(eval_evidence.get("stale")),
            "canary": observed_row,
            "lifecycle": lifecycle_row or {},
            "pending_lifecycle_feedback": pending_lifecycle_row or {},
            "reason_codes": row_reasons,
            "top_reason_counts": observed_row.get("reason_counts") or [],
            "last_evidence_at": last_evidence_at,
            "privacy": _promotion_privacy_summary(),
        })

    candidates.sort(key=lambda row: (str(row.get("primary_state")), str(row.get("action_family")), str(row.get("candidate_id"))))
    return {
        "schema": "tokenclaw.optimization_promotion_funnel.v1",
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": len(candidates),
            "needs_eval_count": state_counts.get("needs-eval", 0),
            "eval_passed_count": state_counts.get("eval-passed", 0),
            "canary_active_count": state_counts.get("canary-active", 0),
            "widening_eligible_count": state_counts.get("widening-eligible", 0),
            "rollback_recommended_count": state_counts.get("rollback-recommended", 0),
            "safety_stopped_count": state_counts.get("safety-stopped", 0),
            "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in candidates), 8),
            "observed_savings_usd": round(sum(_as_float(row.get("observed_savings_usd")) for row in candidates), 8),
            "canary_applied_count": sum(_as_int((row.get("canary") or {}).get("applied_count")) for row in candidates),
            "canary_holdout_count": sum(_as_int((row.get("canary") or {}).get("holdout_count")) for row in candidates),
            "pending_lifecycle_feedback_count": sum(_as_int((row.get("executor_readiness") or {}).get("pending_lifecycle_feedback_count")) for row in candidates),
            "promotion_action_count": _as_int((promotion_actions.get("summary") or {}).get("action_count")),
            "promotion_omitted_count": _as_int((promotion_actions.get("summary") or {}).get("omitted_count")),
            "promotion_omission_bucket_count": _as_int((promotion_actions.get("summary") or {}).get("omission_bucket_count")),
            "top_promotion_omission_next_action": (promotion_actions.get("summary") or {}).get("top_omission_next_action"),
            "last_evidence_at": max((str(row.get("last_evidence_at")) for row in candidates if row.get("last_evidence_at")), default=None),
        },
        "state_counts": _count_breakdown(state_counts),
        "executor_readiness_counts": _count_breakdown(readiness_counts),
        "policy_section_counts": _count_breakdown(policy_section_counts),
        "executor_readiness_by_policy_section": _count_breakdown(readiness_policy_counts),
        "action_family_policy_section_counts": _count_breakdown(action_family_policy_counts),
        "reason_counts": _count_breakdown(reason_counts),
        "omission_buckets": promotion_actions.get("omission_buckets") if isinstance(promotion_actions.get("omission_buckets"), list) else [],
        "candidates": candidates,
        "source_reports": {
            "eval_plan_schema": plan.get("schema") if isinstance(plan, dict) else None,
            "promotion_report_schema": promotion.get("schema") if isinstance(promotion, dict) else None,
            "promotion_actions_schema": promotion_actions.get("schema") if isinstance(promotion_actions, dict) else None,
            "policy_event_count": len(events) if isinstance(events, list) else 0,
        },
        "privacy": _promotion_privacy_summary(),
    }


_EVIDENCE_NEXT_ACTION_ENTRY_FIELDS = {
    "rank",
    "lever",
    "local_action_family",
    "current_status",
    "state",
    "next_action",
    "blocker_codes",
    "sample_count",
    "applied_count",
    "holdout_count",
    "projected_hits",
    "actual_hits",
    "actual_saved_cost_usd",
    "projected_saved_usd",
    "savings_per_1000_calls_usd",
    "evidence_schema",
    "cohort_bucket",
    "issue_worthy_status",
    "expected_savings_path",
    "legacy_issue_title",
    "requested_model",
    "candidate_target_model",
    "omitted_reason",
    "follow_up_owner",
    "managed_dependency",
    "local_handoff_reason",
    "local_file_backed_representation",
}


def _public_evidence_next_action_entry(entry: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: _copy_policy(value)
        for key, value in entry.items()
        if key in _EVIDENCE_NEXT_ACTION_ENTRY_FIELDS and value not in (None, "", [])
    }
    if not isinstance(public.get("blocker_codes"), list):
        public["blocker_codes"] = []
    public["rank"] = _as_int(public.get("rank"))
    public["sample_count"] = _as_int(public.get("sample_count"))
    public["applied_count"] = _as_int(public.get("applied_count"))
    public["holdout_count"] = _as_int(public.get("holdout_count"))
    public["projected_hits"] = _as_int(public.get("projected_hits"))
    public["actual_hits"] = _as_int(public.get("actual_hits"))
    public["actual_saved_cost_usd"] = round(_as_float(public.get("actual_saved_cost_usd")), 8)
    public["projected_saved_usd"] = round(_as_float(public.get("projected_saved_usd")), 8)
    public["savings_per_1000_calls_usd"] = round(_as_float(public.get("savings_per_1000_calls_usd")), 8)
    issue = entry.get("prior_issue") if isinstance(entry.get("prior_issue"), dict) else None
    if issue:
        public["prior_issue"] = {
            key: issue.get(key)
            for key in ("number", "state", "title", "url")
            if issue.get(key) not in (None, "", [])
        }
    return public


def _empty_evidence_next_actions_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_evidence_to_activation_next_actions.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "summary": {
            "tracked_entry_count": 0,
            "top_lever": None,
            "top_current_status": None,
            "top_next_action": None,
            "top_blocker_codes": [],
            "top_expected_savings_path": None,
            "status_counts": [],
        },
        "source": source,
        "entries": [],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_read_only": True,
            "artifact_path_included": False,
        },
    }


async def stats_evidence_to_activation_next_actions(limit: int = 20) -> dict[str, Any]:
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_evidence_next_actions_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_evidence_next_actions_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _empty_evidence_next_actions_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )

    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    ledger = evidence.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        ledger = payload.get("evidence_to_activation_next_action_ledger")
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    if not isinstance(ledger, dict):
        ledger = stats_summary.get("evidence_to_activation_next_action_ledger")
    if not isinstance(ledger, dict):
        return _empty_evidence_next_actions_payload(
            status="no-ledger",
            status_reason="latest research plan does not contain an evidence-to-activation next-action ledger",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )

    capped = max(1, min(int(limit or 20), 100))
    entries = [
        _public_evidence_next_action_entry(entry)
        for entry in ledger.get("entries") or []
        if isinstance(entry, dict)
    ][:capped]
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
    public_summary = {
        key: _copy_policy(summary.get(key))
        for key in (
            "tracked_entry_count",
            "closed_issue_seen_count",
            "top_lever",
            "top_current_status",
            "top_next_action",
            "top_blocker_codes",
            "top_expected_savings_path",
            "status_counts",
            "issue_status_counts",
        )
        if summary.get(key) not in (None, "", [])
    }
    public_summary["tracked_entry_count"] = _as_int(public_summary.get("tracked_entry_count")) or len(entries)
    if not isinstance(public_summary.get("top_blocker_codes"), list):
        public_summary["top_blocker_codes"] = []
    if not isinstance(public_summary.get("status_counts"), list):
        public_summary["status_counts"] = []
    if not isinstance(public_summary.get("issue_status_counts"), list):
        public_summary["issue_status_counts"] = []

    ledger_privacy = ledger.get("privacy") if isinstance(ledger.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_evidence_to_activation_next_actions.v1",
        "generated_at": utc_now(),
        "status": "tracked" if entries else "empty",
        "status_reason": "latest research plan ledger loaded" if entries else "latest research plan ledger has no entries",
        "ledger_schema": ledger.get("schema"),
        "ledger_status": ledger.get("status"),
        "summary": public_summary,
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "entries": entries,
        "privacy": {
            "metadata_only": ledger_privacy.get("metadata_only", True) is True,
            "aggregate_only": ledger_privacy.get("aggregate_only", True) is True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "absolute_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "individual_candidate_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "dashboard_read_only": True,
            "artifact_path_included": False,
        },
    }


def _local_activation_queue_privacy(source_privacy: dict[str, Any] | None = None) -> dict[str, bool]:
    source_privacy = source_privacy if isinstance(source_privacy, dict) else {}
    return {
        "metadata_only": source_privacy.get("metadata_only", True) is True,
        "aggregate_only": source_privacy.get("aggregate_only", True) is True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tenant_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
        "dashboard_read_only": True,
        "artifact_path_included": False,
    }


def _managed_preview_coverage_privacy(*, managed_server_calls_made: bool = False) -> dict[str, bool]:
    return {
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
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "policy_file_contents_included": False,
        "provider_calls_made": False,
        "policy_files_written": False,
        "managed_server_calls_made": bool(managed_server_calls_made),
        "dashboard_read_only": True,
    }


def _empty_managed_preview_coverage(*, status: str, status_reason: str) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_coverage.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "preview_data_status": status,
        "lookback_limit": MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
        "sample_limit": MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT,
        "summary": {
            "stored_preview_outcome_count": 0,
            "sample_outcome_count": 0,
            "fresh_count": 0,
            "stale_count": 0,
            "missing_preview_decision_count": 0,
            "omission_count": 0,
            "failed_closed_count": 0,
            "agreement_count": 0,
            "disagreement_count": 0,
            "latest_preview_age_hours": None,
            "classification_counts": [],
            "local_action_family_counts": [],
        },
        "family_coverage": [],
        "sample_outcomes": [],
        "privacy": _managed_preview_coverage_privacy(),
    }


def _managed_preview_reason(outcome: dict[str, Any]) -> str:
    for key in ("omitted_reason", "no_op_reason"):
        value = str(outcome.get(key) or "").strip()
        if value:
            return value
    reason_codes = outcome.get("reason_codes")
    if isinstance(reason_codes, list):
        for value in reason_codes:
            text = str(value or "").strip()
            if text:
                return text
    return str(outcome.get("classification") or "unknown").strip() or "unknown"


def _managed_preview_public_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    public = {
        "local_action_family": outcome.get("local_action_family") or "unknown",
        "evidence_schema": outcome.get("evidence_schema"),
        "classification": outcome.get("classification") or "unknown",
        "decision": outcome.get("decision"),
        "decision_status": outcome.get("decision_status"),
        "next_action": outcome.get("next_action"),
        "omitted_reason": outcome.get("omitted_reason"),
        "no_op_reason": outcome.get("no_op_reason"),
        "reason_codes": outcome.get("reason_codes") if isinstance(outcome.get("reason_codes"), list) else [],
        "preview_age_hours": outcome.get("preview_age_hours"),
        "stale": bool(outcome.get("stale")),
        "missing_preview_decision": bool(outcome.get("missing_preview_decision")),
        "failed_closed": bool(outcome.get("failed_closed")),
        "disagrees_with_local_evidence": bool(outcome.get("disagrees_with_local_evidence")),
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_preview_policy_files_written": bool(outcome.get("managed_preview_policy_files_written")),
        "managed_preview_provider_calls_made": bool(outcome.get("managed_preview_provider_calls_made")),
        "privacy": _managed_preview_coverage_privacy(
            managed_server_calls_made=bool(outcome.get("managed_server_calls_made"))
        ),
    }
    return {key: _copy_policy(value) for key, value in public.items() if value not in (None, "", [])}


def _managed_preview_family_row(
    *,
    family: str,
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    ages = []
    for outcome in outcomes:
        reason = _managed_preview_reason(outcome)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        classification = str(outcome.get("classification") or "unknown")
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        if outcome.get("preview_age_hours") is not None:
            ages.append(_as_float(outcome.get("preview_age_hours")))
    agreement_count = sum(
        1
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
        and not bool(outcome.get("disagrees_with_local_evidence"))
    )
    top_reason = _breakdown_from_counts(reason_counts)[0]["value"] if reason_counts else None
    return {
        "local_action_family": family,
        "stored_preview_outcome_count": len(outcomes),
        "fresh_count": agreement_count,
        "stale_count": sum(1 for outcome in outcomes if outcome.get("stale")),
        "missing_preview_decision_count": sum(1 for outcome in outcomes if outcome.get("missing_preview_decision")),
        "omission_count": sum(1 for outcome in outcomes if outcome.get("omitted_reason")),
        "failed_closed_count": sum(1 for outcome in outcomes if outcome.get("failed_closed")),
        "agreement_count": agreement_count,
        "disagreement_count": sum(1 for outcome in outcomes if outcome.get("disagrees_with_local_evidence")),
        "latest_preview_age_hours": min(ages) if ages else None,
        "top_omitted_or_blocker_reason": top_reason,
        "reason_counts": _breakdown_from_counts(reason_counts)[:10],
        "classification_counts": _breakdown_from_counts(classification_counts),
    }


def _managed_preview_data_status(outcomes: list[dict[str, Any]]) -> str:
    if not outcomes:
        return "missing"
    fresh = [
        outcome
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
    ]
    if fresh:
        return "fresh"
    if any(bool(outcome.get("stale")) for outcome in outcomes):
        return "stale"
    return "missing"


def _managed_preview_coverage_for_family(coverage: dict[str, Any] | None, family: str | None) -> dict[str, Any] | None:
    if not isinstance(coverage, dict) or not family:
        return None
    for row in coverage.get("family_coverage") or []:
        if isinstance(row, dict) and str(row.get("local_action_family") or "") == str(family):
            return {
                "schema": "tokenclaw.dashboard_managed_activation_preview_family_coverage.v1",
                "status": coverage.get("status"),
                "preview_data_status": coverage.get("preview_data_status"),
                "stored_preview_outcome_count": row.get("stored_preview_outcome_count", 0),
                "fresh_count": row.get("fresh_count", 0),
                "stale_count": row.get("stale_count", 0),
                "missing_preview_decision_count": row.get("missing_preview_decision_count", 0),
                "omission_count": row.get("omission_count", 0),
                "failed_closed_count": row.get("failed_closed_count", 0),
                "agreement_count": row.get("agreement_count", 0),
                "disagreement_count": row.get("disagreement_count", 0),
                "latest_preview_age_hours": row.get("latest_preview_age_hours"),
                "top_omitted_or_blocker_reason": row.get("top_omitted_or_blocker_reason"),
            }
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_family_coverage.v1",
        "status": coverage.get("status"),
        "preview_data_status": coverage.get("preview_data_status"),
        "stored_preview_outcome_count": 0,
        "fresh_count": 0,
        "stale_count": 0,
        "missing_preview_decision_count": 0,
        "omission_count": 0,
        "failed_closed_count": 0,
        "agreement_count": 0,
        "disagreement_count": 0,
        "latest_preview_age_hours": None,
        "top_omitted_or_blocker_reason": None,
    }


def _managed_activation_preview_coverage(store_obj: Any | None) -> dict[str, Any]:
    if store_obj is None:
        return _empty_managed_preview_coverage(
            status="disabled",
            status_reason="local store was not provided for managed preview coverage",
        )
    try:
        from tokenclaw.managed_activation_preview_outcomes import (
            DEFAULT_STALE_AFTER_HOURS,
            build_managed_activation_preview_outcomes_report,
        )

        report = build_managed_activation_preview_outcomes_report(
            store_obj,
            limit=MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
            stale_after_hours=DEFAULT_STALE_AFTER_HOURS,
        )
    except sqlite3.Error:
        return _empty_managed_preview_coverage(
            status="missing",
            status_reason="managed preview outcome table is unavailable",
        )
    outcomes = [row for row in report.get("outcomes") or [] if isinstance(row, dict)]
    if not outcomes:
        return _empty_managed_preview_coverage(
            status="missing",
            status_reason="no managed preview outcome rows have been recorded",
        )
    classification_counts: dict[str, int] = {}
    family_groups: dict[str, list[dict[str, Any]]] = {}
    ages = []
    for outcome in outcomes:
        classification = str(outcome.get("classification") or "unknown")
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        family = str(outcome.get("local_action_family") or "unknown")
        family_groups.setdefault(family, []).append(outcome)
        if outcome.get("preview_age_hours") is not None:
            ages.append(_as_float(outcome.get("preview_age_hours")))
    agreement_count = sum(
        1
        for outcome in outcomes
        if not bool(outcome.get("stale"))
        and not bool(outcome.get("missing_preview_decision"))
        and not bool(outcome.get("failed_closed"))
        and not bool(outcome.get("disagrees_with_local_evidence"))
    )
    managed_calls_made = bool(report.get("managed_server_calls_made"))
    preview_data_status = _managed_preview_data_status(outcomes)
    return {
        "schema": "tokenclaw.dashboard_managed_activation_preview_coverage.v1",
        "generated_at": utc_now(),
        "status": "tracked",
        "status_reason": "bounded local managed preview outcomes loaded",
        "preview_data_status": preview_data_status,
        "lookback_limit": MANAGED_PREVIEW_COVERAGE_LOOKBACK_LIMIT,
        "sample_limit": MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT,
        "summary": {
            "stored_preview_outcome_count": len(outcomes),
            "sample_outcome_count": min(len(outcomes), MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT),
            "fresh_count": agreement_count,
            "stale_count": sum(1 for outcome in outcomes if outcome.get("stale")),
            "missing_preview_decision_count": sum(1 for outcome in outcomes if outcome.get("missing_preview_decision")),
            "omission_count": sum(1 for outcome in outcomes if outcome.get("omitted_reason")),
            "failed_closed_count": sum(1 for outcome in outcomes if outcome.get("failed_closed")),
            "agreement_count": agreement_count,
            "disagreement_count": sum(1 for outcome in outcomes if outcome.get("disagrees_with_local_evidence")),
            "latest_preview_age_hours": min(ages) if ages else None,
            "classification_counts": _breakdown_from_counts(classification_counts),
            "local_action_family_counts": _breakdown_from_counts({
                family: len(rows) for family, rows in family_groups.items()
            }),
        },
        "family_coverage": [
            _managed_preview_family_row(family=family, outcomes=rows)
            for family, rows in sorted(family_groups.items())
        ],
        "sample_outcomes": [
            _managed_preview_public_outcome(outcome)
            for outcome in outcomes[:MANAGED_PREVIEW_COVERAGE_SAMPLE_LIMIT]
        ],
        "source_report_schema": report.get("schema"),
        "source_report_status": report.get("status"),
        "privacy": _managed_preview_coverage_privacy(managed_server_calls_made=managed_calls_made),
    }


def _empty_local_activation_queue_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
    managed_preview_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_local_activation_next_action_queue.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "queue_schema": None,
        "queue_status": None,
        "source_schema": None,
        "source": source,
        "summary": {
            "queued_action_count": 0,
            "top_lever": None,
            "top_state": None,
            "top_current_status": None,
            "top_next_action": None,
            "top_unblock_reason": None,
            "top_realized_savings_usd": 0.0,
            "top_projected_savings_usd": 0.0,
            "total_realized_savings_usd": 0.0,
            "total_projected_savings_usd": 0.0,
            "lever_counts": [],
            "status_counts": [],
            "unblock_reason_counts": [],
        },
        "entries": [],
        "successor_burndown": _activation_successor_burndown([]),
        "activation_preview_agreement_burndown": _activation_preview_agreement_burndown({}),
        "managed_preview_coverage": managed_preview_coverage or _empty_managed_preview_coverage(
            status="disabled",
            status_reason="managed preview coverage was not requested",
        ),
        "privacy": _local_activation_queue_privacy(),
    }


def _empty_preview_gated_activation_issue_queue_payload(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.dashboard_preview_gated_activation_issue_queue.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "queue_schema": None,
        "queue_status": None,
        "source_schema": None,
        "source": source,
        "summary": {
            "successor_decision_count": 0,
            "issue_proposal_count": 0,
            "ready_count": 0,
            "blocked_count": 0,
            "stale_or_no_data_count": 0,
            "suppressed_count": 0,
            "decision_counts": [],
            "issue_status_counts": [],
            "preview_agreement_status_counts": [],
            "issue_queue_status_counts": [],
            "local_action_family_counts": [],
            "top_reason_counts": [],
            "top_ready_issue": None,
            "top_blocked_issue": None,
            "top_stale_or_no_data_issue": None,
        },
        "successor_decisions": [],
        "issue_proposals": [],
        "privacy": _local_activation_queue_privacy(),
    }


def _public_local_activation_queue_summary(summary: dict[str, Any], entry_count: int) -> dict[str, Any]:
    public = {
        key: _copy_policy(summary.get(key))
        for key in (
            "queued_action_count",
            "top_lever",
            "top_state",
            "top_current_status",
            "top_next_action",
            "top_unblock_reason",
            "top_blocking_reason",
            "top_freshness_state",
            "top_savings_per_1000_calls_usd",
            "top_freshness_adjusted_savings_per_1000_calls_usd",
            "top_rank_basis",
            "top_realized_savings_usd",
            "top_projected_savings_usd",
            "total_realized_savings_usd",
            "total_projected_savings_usd",
            "lever_counts",
            "status_counts",
            "unblock_reason_counts",
        )
        if summary.get(key) not in (None, "", [])
    }
    public["queued_action_count"] = _as_int(public.get("queued_action_count")) or entry_count
    public["top_realized_savings_usd"] = round(_as_float(public.get("top_realized_savings_usd")), 8)
    public["top_projected_savings_usd"] = round(_as_float(public.get("top_projected_savings_usd")), 8)
    public["top_savings_per_1000_calls_usd"] = round(_as_float(public.get("top_savings_per_1000_calls_usd")), 8)
    public["top_freshness_adjusted_savings_per_1000_calls_usd"] = round(
        _as_float(public.get("top_freshness_adjusted_savings_per_1000_calls_usd")),
        8,
    )
    public["total_realized_savings_usd"] = round(_as_float(public.get("total_realized_savings_usd")), 8)
    public["total_projected_savings_usd"] = round(_as_float(public.get("total_projected_savings_usd")), 8)
    for key in ("lever_counts", "status_counts", "unblock_reason_counts"):
        if not isinstance(public.get(key), list):
            public[key] = []
    return public


ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES = (
    "source-traffic-acquisition",
    "cache-reobserve",
    "crunch-canary",
)


def _activation_successor_family(row: dict[str, Any]) -> str | None:
    family = str(row.get("local_action_family") or row.get("lever") or "").strip()
    text_parts = [
        family,
        str(row.get("lever") or ""),
        str(row.get("next_action") or ""),
        str(row.get("unblock_reason") or ""),
        str(row.get("blocking_reason") or ""),
        str(row.get("current_status") or ""),
        str(row.get("state") or ""),
        str(row.get("evidence_schema") or ""),
    ]
    text_parts.extend(str(code or "") for code in row.get("blocker_codes") or [])
    text = " ".join(text_parts).lower()
    if family == "source-traffic-acquisition" or "source-traffic" in text or "no-source-traffic-for-request-shape-rollups" in text:
        return "source-traffic-acquisition"
    if family == "cache-reobserve" or "reobserve" in text or "rollback-cache-replay" in text or "cache-replay" in text:
        return "cache-reobserve"
    if family == "crunch-canary" or family == "crunch" or "crunch" in text:
        return "crunch-canary"
    return None


def _activation_status_bucket(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "freshness_state",
            "current_status",
            "state",
            "issue_worthy_status",
            "next_action",
            "unblock_reason",
            "blocking_reason",
            "duplicate_suppression_status",
        )
    ).lower()
    text += " " + " ".join(str(code or "").lower() for code in row.get("blocker_codes") or [])
    if "suppressed" in text or "duplicate" in text:
        return "suppressed"
    if "retired" in text or "superseded" in text:
        return "retired"
    if "stale" in text or "evidence-older-than-max-age" in text:
        return "stale"
    if "rollback" in text:
        return "rollback"
    if _as_int(row.get("applied_count")) > 0 or "applied" in text or "active" in text or "full-rollout" in text:
        return "applied"
    if _as_int(row.get("holdout_count")) > 0 or "holdout" in text:
        return "held-out"
    if "ready" in text or "review" in text:
        return "ready"
    if "no-data" in text or "missing" in text or "no-source-traffic" in text:
        return "missing"
    if "blocked" in text or "keep-blocked" in text:
        return "blocked"
    return "unknown"


def _activation_top_blocker(row: dict[str, Any]) -> str:
    blockers = row.get("blocker_codes") if isinstance(row.get("blocker_codes"), list) else []
    for value in blockers:
        text = str(value or "").strip()
        if text:
            return public_label(text, "unknown")
    for key in ("blocking_reason", "unblock_reason"):
        text = str(row.get(key) or "").strip()
        if text:
            return public_label(text, "unknown")
    return "none"


def _status_count_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    ordered = [
        "missing",
        "stale",
        "ready",
        "applied",
        "held-out",
        "blocked",
        "rollback",
        "retired",
        "suppressed",
        "unknown",
    ]
    rows = [{"value": status, "count": int(counter.get(status, 0))} for status in ordered if counter.get(status, 0)]
    rows.extend(
        {"value": status, "count": int(count)}
        for status, count in sorted(counter.items())
        if status not in ordered
    )
    return rows


def _activation_successor_burndown(entries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {family: [] for family in ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES}
    for row in entries:
        family = _activation_successor_family(row)
        if family is not None:
            grouped[family].append(row)

    rows: list[dict[str, Any]] = []
    for family in ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES:
        family_rows = grouped[family]
        status_counter: Counter[str] = Counter(_activation_status_bucket(row) for row in family_rows)
        next_counter: Counter[str] = Counter(
            public_label(row.get("next_action") or "inspect-local-evidence", "inspect-local-evidence")
            for row in family_rows
        )
        blocker_counter: Counter[str] = Counter(_activation_top_blocker(row) for row in family_rows)
        top = sorted(
            family_rows,
            key=lambda row: (
                _as_int(row.get("rank")) or 9999,
                -_as_float(row.get("realized_savings_usd")),
                -_as_float(row.get("projected_savings_usd")),
                -_as_int(row.get("sample_count")),
            ),
        )
        top_row = top[0] if top else {}
        rows.append(
            {
                "family": family,
                "status": "tracked" if family_rows else "missing",
                "row_count": len(family_rows),
                "status_counts": _status_count_rows(status_counter),
                "top_next_action": public_label(
                    (next_counter.most_common(1)[0][0] if next_counter else None)
                    or top_row.get("next_action")
                    or "none",
                    "none",
                ),
                "top_blocker": public_label(
                    (blocker_counter.most_common(1)[0][0] if blocker_counter else None)
                    or _activation_top_blocker(top_row)
                    or "none",
                    "none",
                ),
                "sample_count": sum(_as_int(row.get("sample_count")) for row in family_rows),
                "applied_count": sum(_as_int(row.get("applied_count")) for row in family_rows),
                "holdout_count": sum(_as_int(row.get("holdout_count")) for row in family_rows),
                "safety_stop_count": sum(_as_int(row.get("safety_stop_count")) for row in family_rows),
                "rollback_count": sum(_as_int(row.get("rollback_count")) for row in family_rows),
                "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in family_rows), 8),
                "realized_savings_usd": round(sum(_as_float(row.get("realized_savings_usd")) for row in family_rows), 8),
                "top_entry_rank": _as_int(top_row.get("rank")) if top_row else 0,
                "privacy": _local_activation_queue_privacy(),
            }
        )

    tracked_rows = [row for row in rows if row["row_count"] > 0]
    status_counter = Counter(row["status"] for row in rows)
    return {
        "schema": "tokenclaw.dashboard_activation_successor_burndown.v1",
        "status": "tracked" if tracked_rows else "missing",
        "summary": {
            "tracked_family_count": len(tracked_rows),
            "expected_family_count": len(ACTIVATION_SUCCESSOR_BURNDOWN_FAMILIES),
            "tracked_row_count": sum(row["row_count"] for row in rows),
            "status_counts": _status_count_rows(status_counter),
            "total_projected_savings_usd": round(sum(row["projected_savings_usd"] for row in rows), 8),
            "total_realized_savings_usd": round(sum(row["realized_savings_usd"] for row in rows), 8),
            "top_family": tracked_rows[0]["family"] if tracked_rows else None,
            "top_next_action": tracked_rows[0]["top_next_action"] if tracked_rows else None,
            "top_blocker": tracked_rows[0]["top_blocker"] if tracked_rows else None,
        },
        "families": rows,
        "privacy": _local_activation_queue_privacy(),
    }


CLOSED_LOOP_ACTIVATION_FAMILIES = ("cache", "crunch", "routing")

CLOSED_LOOP_ACTIVATION_STATES = (
    "preview_missing",
    "preview_agreed",
    "draft_ready",
    "applied_waiting_observation",
    "realized_savings",
    "retired_no_repeat",
    "rollback_required",
    "safety_stopped",
    "keep_blocked",
)


def _closed_loop_activation_family(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "cache" in text:
        return "cache"
    if "crunch" in text:
        return "crunch"
    if "routing" in text or "route" in text:
        return "routing"
    return text if text in CLOSED_LOOP_ACTIVATION_FAMILIES else None


def _closed_loop_activation_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "state",
        "current_status",
        "issue_worthy_status",
        "next_action",
        "unblock_reason",
        "blocking_reason",
        "freshness_state",
        "duplicate_suppression_status",
        "evidence_schema",
        "promotion_readiness",
        "promotion_decision",
        "decision",
    ):
        parts.append(str(row.get(key) or ""))
    parts.extend(str(code or "") for code in row.get("blocker_codes") or [])
    coverage = row.get("managed_preview_coverage") if isinstance(row.get("managed_preview_coverage"), dict) else {}
    for key in ("status", "preview_data_status", "top_omitted_or_blocker_reason"):
        parts.append(str(coverage.get(key) or ""))
    return " ".join(parts).lower()


def _closed_loop_activation_entry_states(row: dict[str, Any]) -> set[str]:
    text = _closed_loop_activation_text(row)
    states: set[str] = set()
    realized = _as_float(row.get("realized_savings_usd") or row.get("actual_saved_cost_usd") or row.get("observed_saved_usd"))
    applied = _as_int(row.get("applied_count"))
    safety_stops = _as_int(row.get("safety_stop_count"))
    rollbacks = _as_int(row.get("rollback_count"))
    preview_status = str(
        row.get("preview_verification_status")
        or (row.get("managed_preview_coverage") or {}).get("preview_data_status")
        or (row.get("managed_preview_coverage") or {}).get("status")
        or ""
    ).lower()

    if "no-data" in preview_status or "missing" in preview_status or "not-previewed" in preview_status:
        states.add("preview_missing")
    if row.get("preview_verified") or "preview-verified" in preview_status or "agreed" in text:
        states.add("preview_agreed")
    if (
        row.get("policy_write_candidate")
        or row.get("required_local_executor")
        or row.get("local_policy_patch")
        or "draft" in text
        or "stage" in text
    ):
        states.add("draft_ready")
    if safety_stops > 0 or "safety" in text and "stop" in text:
        states.add("safety_stopped")
    if rollbacks > 0 or row.get("rollback_required") or "rollback" in text:
        states.add("rollback_required")
    if "retire" in text or "retired" in text or "no-repeat" in text or "superseded" in text:
        states.add("retired_no_repeat")
    if realized > 0:
        states.add("realized_savings")
    if applied > 0 and realized <= 0 and not states.intersection({"rollback_required", "safety_stopped", "retired_no_repeat"}):
        states.add("applied_waiting_observation")
    if "keep-blocked" in text or "blocked" in text:
        states.add("keep_blocked")
    if not states:
        states.add("preview_missing")
    return states


def _closed_loop_entry_stale_age_hours(row: dict[str, Any]) -> float | None:
    rank_basis = row.get("rank_basis") if isinstance(row.get("rank_basis"), dict) else {}
    for value in (
        rank_basis.get("evidence_age_hours"),
        row.get("evidence_age_hours"),
        (row.get("managed_preview_coverage") or {}).get("latest_preview_age_hours")
        if isinstance(row.get("managed_preview_coverage"), dict)
        else None,
    ):
        if value is not None:
            return round(_as_float(value), 3)
    return None


def _closed_loop_activation_readiness(activation_burndown: dict[str, Any]) -> dict[str, Any]:
    family_rows: dict[str, dict[str, Any]] = {}
    for family in CLOSED_LOOP_ACTIVATION_FAMILIES:
        family_rows[family] = {
            "family": family,
            "row_count": 0,
            "state_counts": {state: 0 for state in CLOSED_LOOP_ACTIVATION_STATES},
            "top_next_action": None,
            "top_blocker": None,
            "top_state": None,
            "top_stale_evidence_age_hours": None,
            "projected_savings_usd": 0.0,
            "realized_savings_usd": 0.0,
            "sample_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "safety_stop_count": 0,
            "rollback_count": 0,
            "privacy": _local_activation_queue_privacy(),
        }

    entries = [row for row in activation_burndown.get("entries") or [] if isinstance(row, dict)]
    top_candidates: dict[str, dict[str, Any]] = {}
    for entry in entries:
        family = _closed_loop_activation_family(entry.get("local_action_family") or entry.get("lever"))
        if family not in family_rows:
            continue
        row = family_rows[family]
        states = _closed_loop_activation_entry_states(entry)
        row["row_count"] += 1
        for state in states:
            if state in row["state_counts"]:
                row["state_counts"][state] += 1
        row["projected_savings_usd"] += _as_float(entry.get("projected_savings_usd"))
        row["realized_savings_usd"] += _as_float(entry.get("realized_savings_usd"))
        row["sample_count"] += _as_int(entry.get("sample_count"))
        row["applied_count"] += _as_int(entry.get("applied_count"))
        row["holdout_count"] += _as_int(entry.get("holdout_count"))
        row["safety_stop_count"] += _as_int(entry.get("safety_stop_count"))
        row["rollback_count"] += _as_int(entry.get("rollback_count"))
        current_top = top_candidates.get(family)
        sort_key = (_as_int(entry.get("rank")) or 9999, -_as_float(entry.get("projected_savings_usd")))
        current_sort = (
            (_as_int(current_top.get("rank")) or 9999, -_as_float(current_top.get("projected_savings_usd")))
            if current_top
            else (999999, 0.0)
        )
        if sort_key < current_sort:
            top_candidates[family] = entry

    preview_burndown = activation_burndown.get("activation_preview_agreement_burndown")
    preview_families = preview_burndown.get("families") if isinstance(preview_burndown, dict) else []
    for preview in preview_families or []:
        if not isinstance(preview, dict):
            continue
        family = _closed_loop_activation_family(preview.get("local_action_family"))
        if family not in family_rows:
            continue
        row = family_rows[family]
        row["state_counts"]["preview_agreed"] += _as_int(preview.get("agreed_count"))
        row["state_counts"]["preview_missing"] += _as_int(preview.get("missing_count")) + _as_int(preview.get("not_previewed_count"))
        row["state_counts"]["draft_ready"] += _as_int(preview.get("dry_run_drafted_count"))
        if row["top_next_action"] is None and preview.get("top_next_action"):
            row["top_next_action"] = public_label(preview.get("top_next_action"), "none")
        if row["top_blocker"] is None and preview.get("top_reason_code"):
            row["top_blocker"] = public_label(preview.get("top_reason_code"), "none")

    for family, entry in top_candidates.items():
        row = family_rows[family]
        row["top_next_action"] = public_label(entry.get("next_action") or "inspect-local-evidence", "inspect-local-evidence")
        row["top_blocker"] = _activation_top_blocker(entry)
        ranked_states = [
            state
            for state in CLOSED_LOOP_ACTIVATION_STATES
            if row["state_counts"].get(state)
        ]
        row["top_state"] = ranked_states[0] if ranked_states else "preview_missing"
        row["top_stale_evidence_age_hours"] = _closed_loop_entry_stale_age_hours(entry)

    public_families: list[dict[str, Any]] = []
    total_state_counts: Counter[str] = Counter()
    for family in CLOSED_LOOP_ACTIVATION_FAMILIES:
        row = family_rows[family]
        state_counts = {state: _as_int(row["state_counts"].get(state)) for state in CLOSED_LOOP_ACTIVATION_STATES}
        total_state_counts.update(state_counts)
        row_count = _as_int(row.get("row_count"))
        status = "tracked" if row_count or any(state_counts.values()) else "missing"
        public_families.append(
            {
                "family": family,
                "status": status,
                "row_count": row_count,
                "state_counts": [{"state": state, "count": count} for state, count in state_counts.items()],
                "top_state": row.get("top_state") or ("preview_missing" if status == "tracked" else "missing"),
                "top_next_action": row.get("top_next_action") or "none",
                "top_blocker": row.get("top_blocker") or "none",
                "top_stale_evidence_age_hours": row.get("top_stale_evidence_age_hours"),
                "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
                "realized_savings_usd": round(_as_float(row.get("realized_savings_usd")), 8),
                "sample_count": _as_int(row.get("sample_count")),
                "applied_count": _as_int(row.get("applied_count")),
                "holdout_count": _as_int(row.get("holdout_count")),
                "safety_stop_count": _as_int(row.get("safety_stop_count")),
                "rollback_count": _as_int(row.get("rollback_count")),
                "privacy": row["privacy"],
            }
        )

    tracked = [row for row in public_families if row["status"] == "tracked"]
    top = sorted(
        tracked,
        key=lambda row: (
            -_as_float(row.get("projected_savings_usd")),
            -_as_float(row.get("realized_savings_usd")),
            -_as_int(row.get("sample_count")),
            str(row.get("family") or ""),
        ),
    )
    return {
        "schema": "tokenclaw.closed_loop_activation_readiness.v1",
        "generated_at": utc_now(),
        "status": "tracked" if tracked else "missing",
        "summary": {
            "family_count": len(public_families),
            "tracked_family_count": len(tracked),
            "row_count": sum(_as_int(row.get("row_count")) for row in public_families),
            "state_counts": [{"state": state, "count": int(total_state_counts.get(state, 0))} for state in CLOSED_LOOP_ACTIVATION_STATES],
            "top_family": top[0]["family"] if top else None,
            "top_state": top[0]["top_state"] if top else None,
            "top_next_action": top[0]["top_next_action"] if top else None,
            "top_blocker": top[0]["top_blocker"] if top else None,
            "top_projected_savings_usd": round(_as_float(top[0].get("projected_savings_usd")) if top else 0.0, 8),
            "top_realized_savings_usd": round(_as_float(top[0].get("realized_savings_usd")) if top else 0.0, 8),
            "top_stale_evidence_age_hours": top[0].get("top_stale_evidence_age_hours") if top else None,
            "total_projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in public_families), 8),
            "total_realized_savings_usd": round(sum(_as_float(row.get("realized_savings_usd")) for row in public_families), 8),
        },
        "families": public_families,
        "privacy": _local_activation_queue_privacy(
            activation_burndown.get("privacy") if isinstance(activation_burndown.get("privacy"), dict) else {}
        ),
    }


def _activation_successor_health_empty(
    *,
    status: str,
    status_reason: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tokenclaw.activation_successor_queue_health.v1",
        "generated_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "source": source,
        "summary": {
            "queued_action_count": 0,
            "successor_action_count": 0,
            "successor_decision_count": 0,
            "top_local_action_family": None,
            "top_state": None,
            "top_status": None,
            "top_next_action": None,
            "top_blocker": None,
            "top_blocker_codes": [],
            "top_preview_verification_status": None,
            "top_preview_verification_decision": None,
            "top_projected_savings_usd": 0.0,
            "top_realized_savings_usd": 0.0,
            "total_projected_savings_usd": 0.0,
            "total_realized_savings_usd": 0.0,
            "latest_preview_age_hours": None,
            "local_action_family_counts": [],
            "status_counts": [],
            "preview_gate_status_counts": [],
            "preview_gate_decision_counts": [],
            "blocker_counts": [],
            "next_action_counts": [],
        },
        "top_entries": [],
        "privacy": _local_activation_queue_privacy(),
    }


def _count_activation_values(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                counted = False
                for item in value:
                    text = str(item or "").strip()
                    if text:
                        counts[text] = counts.get(text, 0) + 1
                        counted = True
                if counted:
                    break
                continue
            text = str(value or "").strip()
            if text:
                counts[text] = counts.get(text, 0) + 1
                break
    return _breakdown_from_counts(counts)


def _activation_successor_preview_gate(row: dict[str, Any]) -> dict[str, Any]:
    gate = row.get("managed_preview_gate") if isinstance(row.get("managed_preview_gate"), dict) else {}
    health_gate = gate.get("health_gate") if isinstance(gate.get("health_gate"), dict) else {}
    return {
        "status": row.get("preview_verification_status")
        or gate.get("status")
        or health_gate.get("status"),
        "decision": row.get("preview_verification_decision")
        or gate.get("decision"),
        "latest_preview_age_hours": row.get("latest_preview_age_hours")
        if row.get("latest_preview_age_hours") is not None
        else health_gate.get("latest_preview_age_hours"),
    }


def _activation_successor_top_entry(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: (_as_int(row.get("rank")) or 9999, _as_int(row.get("ledger_rank")) or 9999))[0]


def _public_activation_successor_health_entry(entry: dict[str, Any]) -> dict[str, Any]:
    gate = _activation_successor_preview_gate(entry)
    blocker_codes = entry.get("blocker_codes") if isinstance(entry.get("blocker_codes"), list) else []
    public = {
        "rank": _as_int(entry.get("rank")),
        "ledger_rank": _as_int(entry.get("ledger_rank")),
        "local_action_family": public_label(entry.get("local_action_family") or entry.get("lever") or "unknown", "unknown"),
        "state": public_label(entry.get("state") or "", ""),
        "current_status": public_label(entry.get("current_status") or entry.get("successor_status") or "", ""),
        "next_action": public_label(
            entry.get("next_action") or entry.get("recommended_next_action") or "",
            "",
        ),
        "top_blocker": public_label(
            blocker_codes[0] if blocker_codes else entry.get("unblock_reason") or "",
            "",
        ),
        "blocker_codes": [public_label(code, "unknown") for code in blocker_codes],
        "preview_verification_status": public_label(gate.get("status") or "", ""),
        "preview_verification_decision": public_label(gate.get("decision") or "", ""),
        "latest_preview_age_hours": gate.get("latest_preview_age_hours"),
        "projected_savings_usd": round(_as_float(entry.get("projected_savings_usd") or entry.get("projected_saved_usd")), 8),
        "realized_savings_usd": round(_as_float(entry.get("realized_savings_usd") or entry.get("actual_saved_cost_usd")), 8),
        "sample_count": _as_int(entry.get("sample_count")),
    }
    return {key: value for key, value in public.items() if value not in (None, "", [])}


def _activation_successor_health_from_queue(
    queue: dict[str, Any],
    *,
    source: dict[str, Any],
    plan_generated_at: Any = None,
    limit: int = 5,
) -> dict[str, Any]:
    entries = [row for row in queue.get("entries") or [] if isinstance(row, dict)]
    successor_actions = [row for row in queue.get("successor_actions") or [] if isinstance(row, dict)]
    successor_decisions = [row for row in queue.get("successor_decisions") or [] if isinstance(row, dict)]
    rows = entries or successor_actions
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    top = _activation_successor_top_entry(rows)
    top_gate = _activation_successor_preview_gate(top or {})
    top_blocker_codes = top.get("blocker_codes") if isinstance(top, dict) and isinstance(top.get("blocker_codes"), list) else []
    preview_ages = [
        _as_float(_activation_successor_preview_gate(row).get("latest_preview_age_hours"))
        for row in rows
        if _activation_successor_preview_gate(row).get("latest_preview_age_hours") is not None
    ]
    gate_status_counts = summary.get("preview_gate_status_counts")
    if not isinstance(gate_status_counts, list):
        gate_status_counts = _count_activation_values(
            [{**row, **_activation_successor_preview_gate(row)} for row in rows],
            "status",
        )
    gate_decision_counts = summary.get("preview_gate_decision_counts")
    if not isinstance(gate_decision_counts, list):
        gate_decision_counts = _count_activation_values(
            [{**row, **_activation_successor_preview_gate(row)} for row in rows],
            "decision",
        )
    capped = max(1, min(int(limit or 5), 20))
    return {
        "schema": "tokenclaw.activation_successor_queue_health.v1",
        "generated_at": utc_now(),
        "status": "ranked" if rows else "empty",
        "status_reason": "latest local activation successor queue health loaded" if rows else "latest queue has no entries",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": plan_generated_at},
        "summary": {
            "queued_action_count": _as_int(summary.get("queued_action_count")) or len(entries),
            "successor_action_count": _as_int(summary.get("successor_action_count")) or len(successor_actions),
            "successor_decision_count": _as_int(summary.get("successor_decision_count")) or len(successor_decisions),
            "top_local_action_family": public_label(
                (top or {}).get("local_action_family") or (top or {}).get("lever") or summary.get("top_lever") or "",
                "",
            ),
            "top_state": public_label((top or {}).get("state") or summary.get("top_state") or "", ""),
            "top_status": public_label(
                (top or {}).get("current_status") or (top or {}).get("successor_status") or summary.get("top_current_status") or "",
                "",
            ),
            "top_next_action": public_label(
                (top or {}).get("next_action") or (top or {}).get("recommended_next_action") or summary.get("top_next_action") or "",
                "",
            ),
            "top_blocker": public_label(
                top_blocker_codes[0] if top_blocker_codes else (top or {}).get("unblock_reason") or summary.get("top_unblock_reason") or "",
                "",
            ),
            "top_blocker_codes": [public_label(code, "unknown") for code in top_blocker_codes],
            "top_preview_verification_status": public_label(top_gate.get("status") or "", ""),
            "top_preview_verification_decision": public_label(top_gate.get("decision") or "", ""),
            "top_projected_savings_usd": round(
                _as_float(summary.get("top_projected_savings_usd") or (top or {}).get("projected_savings_usd") or (top or {}).get("projected_saved_usd")),
                8,
            ),
            "top_realized_savings_usd": round(
                _as_float(summary.get("top_realized_savings_usd") or (top or {}).get("realized_savings_usd") or (top or {}).get("actual_saved_cost_usd")),
                8,
            ),
            "total_projected_savings_usd": round(_as_float(summary.get("total_projected_savings_usd")), 8),
            "total_realized_savings_usd": round(_as_float(summary.get("total_realized_savings_usd")), 8),
            "latest_preview_age_hours": min(preview_ages) if preview_ages else None,
            "local_action_family_counts": _count_activation_values(rows, "local_action_family", "lever"),
            "status_counts": summary.get("status_counts") if isinstance(summary.get("status_counts"), list) else _count_activation_values(rows, "current_status", "successor_status", "state"),
            "preview_gate_status_counts": gate_status_counts,
            "preview_gate_decision_counts": gate_decision_counts,
            "blocker_counts": _count_activation_values(rows, "blocker_codes", "unblock_reason"),
            "next_action_counts": _count_activation_values(rows, "next_action", "recommended_next_action"),
        },
        "top_entries": [_public_activation_successor_health_entry(row) for row in rows[:capped]],
        "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
    }


def build_activation_successor_queue_health(limit: int = 5) -> dict[str, Any]:
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _activation_successor_health_empty(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _activation_successor_health_empty(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _activation_successor_health_empty(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )
    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _activation_successor_health_empty(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )
    return _activation_successor_health_from_queue(
        queue,
        source=source,
        plan_generated_at=payload.get("generated_at"),
        limit=limit,
    )


def _activation_public_ref(value: Any, *, prefix: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return public_id(text, prefix=prefix, fallback=f"{prefix}:unknown")


def _activation_issue_queue_status(decision: dict[str, Any]) -> str:
    decision_text = str(decision.get("decision") or "").strip()
    issue_status = str(decision.get("issue_worthy_status") or "").strip()
    preview_status = str(decision.get("preview_agreement_status") or "").strip()
    preview_verified = bool(decision.get("preview_verified"))
    if issue_status == "suppressed" or decision_text in {"keep-current-rule", "suppress-duplicate"}:
        return "suppressed"
    if not preview_verified and preview_status in {
        "",
        "not-previewed",
        "missing-preview-decision",
        "no-data-preview-health",
        "stale-preview",
        "stale-preview-health",
        "incomplete-preview-health",
    }:
        return "stale/no-data"
    if issue_status == "blocked" or decision_text in {"keep-blocked", "review-stale-preview"}:
        return "blocked"
    if preview_verified and issue_status in {"ready", "review"} and decision_text in {"ready", "review", "review-only"}:
        return "ready"
    return issue_status or decision_text or "unknown"


def _activation_issue_queue_reason(decision: dict[str, Any]) -> str:
    for key in (
        "preview_omitted_reason",
        "preview_no_op_reason",
        "top_preview_omission_reason",
        "preview_agreement_status",
        "preview_verification_status",
        "decision",
    ):
        text = str(decision.get(key) or "").strip()
        if text:
            return text
    return "unknown"


def _public_preview_gated_successor_decision(decision: dict[str, Any]) -> dict[str, Any]:
    preview_requirement = str(decision.get("preview_requirement") or "").strip()
    managed_preview_required = bool(
        decision.get("managed_preview_required")
        or preview_requirement in {"required", "preview-required", "managed-preview-required"}
    )
    public = {
        "source_ref": _activation_public_ref(decision.get("source_fingerprint"), prefix="activation-ref"),
        "successor_action_ref": _activation_public_ref(
            decision.get("successor_action_fingerprint"),
            prefix="successor-ref",
        ),
        "local_action_family": public_label(decision.get("local_action_family") or "unknown", "unknown"),
        "decision": public_label(decision.get("decision") or "unknown", "unknown"),
        "recommended_next_action": public_label(
            decision.get("recommended_next_action") or "inspect-local-evidence",
            "inspect-local-evidence",
        ),
        "issue_worthy_status": public_label(decision.get("issue_worthy_status") or "unknown", "unknown"),
        "issue_queue_status": _activation_issue_queue_status(decision),
        "preview_agreement_status": public_label(
            decision.get("preview_agreement_status") or "not-previewed",
            "not-previewed",
        ),
        "preview_verified": bool(decision.get("preview_verified")),
        "preview_verification_status": public_label(decision.get("preview_verification_status") or "", ""),
        "preview_verification_decision": public_label(decision.get("preview_verification_decision") or "", ""),
        "preview_requirement": public_label(preview_requirement, ""),
        "managed_preview_required": managed_preview_required,
        "policy_write_candidate": bool(
            decision.get("policy_write_candidate")
            or decision.get("cache_apply_action_count")
            or decision.get("draft_action_count")
            or decision.get("dry_run_drafted")
        ),
        "preview_omitted_reason": public_label(decision.get("preview_omitted_reason") or "", ""),
        "preview_no_op_reason": public_label(decision.get("preview_no_op_reason") or "", ""),
        "top_preview_omission_reason": public_label(decision.get("top_preview_omission_reason") or "", ""),
        "privacy": _local_activation_queue_privacy(
            decision.get("privacy") if isinstance(decision.get("privacy"), dict) else {}
        ),
    }
    bool_keys = {"preview_verified", "managed_preview_required", "policy_write_candidate"}
    return {key: value for key, value in public.items() if value not in (None, "", []) or key in bool_keys}


PREVIEW_AGREEMENT_BUCKETS = (
    "agreed",
    "missing",
    "stale",
    "unsafe",
    "omitted",
    "blocked",
    "disagreed",
    "not_previewed",
)

PREVIEW_AGREEMENT_EXTRA_COUNTS = (
    "preview_optional",
    "preview_required",
    "dry_run_drafted",
)


def _preview_agreement_bucket(row: dict[str, Any]) -> str:
    status = str(row.get("preview_agreement_status") or "").strip().lower()
    verification_status = str(row.get("preview_verification_status") or "").strip().lower()
    decision = str(row.get("decision") or "").strip().lower()
    if row.get("preview_verified") or status == "agreed":
        return "agreed"
    if row.get("preview_omitted_reason") or row.get("top_preview_omission_reason") or "omitted" in status:
        return "omitted"
    if "unsafe" in status or "unsafe" in verification_status:
        return "unsafe"
    if "disagree" in status or "failed-closed" in status or "disagree" in verification_status:
        return "disagreed"
    if "stale" in status or "stale" in verification_status:
        return "stale"
    if "missing" in status or "no-data" in status or "missing" in verification_status or "no-data" in verification_status:
        return "missing"
    if status in {"", "not-previewed"}:
        return "not_previewed"
    if "blocked" in status or "blocked" in decision or "keep-blocked" in decision:
        return "blocked"
    return "blocked"


def _preview_agreement_extra_counts(row: dict[str, Any]) -> dict[str, int]:
    decision = str(row.get("preview_verification_decision") or row.get("decision") or "").strip().lower()
    requirement = str(row.get("preview_requirement") or "").strip().lower()
    managed_required = bool(row.get("managed_preview_required"))
    policy_write_candidate = bool(row.get("policy_write_candidate"))
    optional = (
        decision == "preview-optional"
        or requirement in {"optional", "preview-optional", "managed-preview-optional"}
    )
    required = managed_required or requirement in {"required", "preview-required", "managed-preview-required"}
    return {
        "preview_optional_count": int(bool(optional)),
        "preview_required_count": int(bool(required)),
        "dry_run_drafted_count": int(bool(policy_write_candidate or decision in {"draft", "dry-run-drafted"})),
    }


def _preview_agreement_reason(row: dict[str, Any]) -> str:
    for key in (
        "preview_omitted_reason",
        "preview_no_op_reason",
        "top_preview_omission_reason",
        "preview_verification_status",
        "preview_agreement_status",
        "decision",
    ):
        text = str(row.get(key) or "").strip()
        if text:
            return public_label(text, "unknown")
    return "unknown"


def _summary_preview_agreement_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("preview_agreement_by_local_action_family")
    if not isinstance(rows, list):
        return []
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("local_action_family") or "").strip()
        if not family:
            continue
        public: dict[str, Any] = {
            "local_action_family": public_label(family, "unknown"),
            "top_reason_code": public_label(row.get("top_reason_code") or row.get("top_reason") or "none", "none"),
            "top_next_action": public_label(row.get("top_next_action") or "none", "none"),
            "top_source_ref": _activation_public_ref(row.get("source_fingerprint") or row.get("top_source_fingerprint"), prefix="activation-ref"),
            "top_successor_action_ref": _activation_public_ref(
                row.get("successor_action_fingerprint") or row.get("top_successor_action_fingerprint"),
                prefix="successor-ref",
            ),
            "privacy": _local_activation_queue_privacy(row.get("privacy") if isinstance(row.get("privacy"), dict) else {}),
        }
        total = 0
        for bucket in PREVIEW_AGREEMENT_BUCKETS:
            count = _as_int(row.get(f"{bucket}_count"))
            if bucket == "unsafe" and not count:
                count = _as_int(row.get("request_shape_unsafe_count")) + _as_int(row.get("crunch_preview_quality_risk_count"))
            public[f"{bucket}_count"] = count
            total += count
        for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS:
            public[f"{bucket}_count"] = _as_int(row.get(f"{bucket}_count"))
        if not total:
            total = sum(_as_int(row.get(key)) for key in ("stored_preview_outcome_count", "row_count", "count"))
        public["total_count"] = total
        public_rows.append(public)
    return public_rows


def _activation_preview_agreement_burndown(queue: dict[str, Any]) -> dict[str, Any]:
    summary = queue.get("summary") if isinstance(queue.get("summary"), dict) else {}
    summary_rows = _summary_preview_agreement_rows(summary)
    raw_decisions: list[dict[str, Any]] = []
    if isinstance(queue, dict) and queue:
        try:
            from tokenclaw.orchestrator_research import (
                build_local_activation_successor_actions,
                build_local_activation_successor_decisions,
            )

            actions = queue.get("successor_actions")
            if not isinstance(actions, list):
                actions = build_local_activation_successor_actions(queue)
            decisions = queue.get("successor_decisions")
            if not isinstance(decisions, list):
                decisions = build_local_activation_successor_decisions({"successor_actions": actions})
            raw_decisions = [row for row in decisions if isinstance(row, dict)]
        except Exception:
            raw_decisions = []
    public_decisions = [_public_preview_gated_successor_decision(row) for row in raw_decisions]
    if not public_decisions:
        total_by_bucket = {
            f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in summary_rows)
            for bucket in PREVIEW_AGREEMENT_BUCKETS
        }
        extra_totals = {
            f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in summary_rows)
            for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS
        }
        return {
            "schema": "tokenclaw.dashboard_activation_preview_agreement_burndown.v1",
            "status": "tracked" if summary_rows else "missing",
            "summary": {
                "local_action_family_count": len(summary_rows),
                "successor_decision_count": 0,
                "total_count": sum(_as_int(row.get("total_count")) for row in summary_rows),
                **total_by_bucket,
                **extra_totals,
            },
            "families": summary_rows,
            "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in public_decisions:
        family = public_label(row.get("local_action_family") or "unknown", "unknown")
        grouped.setdefault(family, []).append(row)
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        bucket_counts = Counter(_preview_agreement_bucket(row) for row in rows)
        reason_counts = Counter(_preview_agreement_reason(row) for row in rows)
        top = rows[0]
        family_rows.append(
            {
                "local_action_family": family,
                "total_count": len(rows),
                **{f"{bucket}_count": int(bucket_counts.get(bucket, 0)) for bucket in PREVIEW_AGREEMENT_BUCKETS},
                **{
                    key: sum(_preview_agreement_extra_counts(row)[key] for row in rows)
                    for key in (f"{bucket}_count" for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS)
                },
                "top_preview_agreement_status": public_label(top.get("preview_agreement_status") or "not-previewed", "not-previewed"),
                "top_reason_code": reason_counts.most_common(1)[0][0] if reason_counts else "unknown",
                "top_next_action": public_label(top.get("recommended_next_action") or "inspect-local-evidence", "inspect-local-evidence"),
                "top_source_ref": top.get("source_ref"),
                "top_successor_action_ref": top.get("successor_action_ref"),
                "privacy": _local_activation_queue_privacy(top.get("privacy") if isinstance(top.get("privacy"), dict) else {}),
            }
        )
    total_counts = {
        f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in family_rows)
        for bucket in PREVIEW_AGREEMENT_BUCKETS
    }
    extra_totals = {
        f"{bucket}_count": sum(_as_int(row.get(f"{bucket}_count")) for row in family_rows)
        for bucket in PREVIEW_AGREEMENT_EXTRA_COUNTS
    }
    return {
        "schema": "tokenclaw.dashboard_activation_preview_agreement_burndown.v1",
        "status": "tracked" if family_rows else "missing",
        "summary": {
            "local_action_family_count": len(family_rows),
            "successor_decision_count": len(public_decisions),
            "total_count": len(public_decisions),
            **total_counts,
            **extra_totals,
        },
        "families": family_rows,
        "privacy": _local_activation_queue_privacy(queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}),
    }


def _proposal_status_label(labels: Any) -> str:
    if not isinstance(labels, list):
        return "status:unknown"
    for label in labels:
        text = str(label or "").strip()
        if text.startswith("status:"):
            return text
    return "status:unknown"


def _public_preview_gated_issue_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    labels = [public_label(label, "unknown") for label in proposal.get("labels") or [] if str(label or "").strip()]
    public = {
        "repo": public_label(proposal.get("repo") or "lutzkuen/tokenclaw", "lutzkuen/tokenclaw"),
        "title": public_label(proposal.get("title") or "Untitled activation successor issue", "Untitled activation successor issue"),
        "labels": labels,
        "status_label": _proposal_status_label(labels),
        "proposal_source": public_label(proposal.get("proposal_source") or "activation-successor", "activation-successor"),
        "source_ref": _activation_public_ref(proposal.get("fingerprint"), prefix="activation-ref"),
        "successor_action_ref": _activation_public_ref(
            proposal.get("successor_action_fingerprint"),
            prefix="successor-ref",
        ),
        "expected_savings_path": public_label(proposal.get("expected_savings_path") or "", ""),
        "privacy": _local_activation_queue_privacy(
            proposal.get("privacy") if isinstance(proposal.get("privacy"), dict) else {}
        ),
    }
    return {key: value for key, value in public.items() if value not in (None, "", [])}


def _top_issue(rows: list[dict[str, Any]], status: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("issue_queue_status") == status:
            return {
                "source_ref": row.get("source_ref"),
                "local_action_family": row.get("local_action_family"),
                "decision": row.get("decision"),
                "recommended_next_action": row.get("recommended_next_action"),
                "preview_agreement_status": row.get("preview_agreement_status"),
                "reason": _activation_issue_queue_reason(row),
            }
    return None


def _preview_gated_activation_issue_queue_payload(
    *,
    payload: dict[str, Any],
    queue: dict[str, Any],
    source: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    try:
        from tokenclaw.orchestrator_research import (
            _proposals_from_activation_successor_decisions,
            build_local_activation_successor_actions,
            build_local_activation_successor_decisions,
        )
    except Exception:
        return _empty_preview_gated_activation_issue_queue_payload(
            status="builder-unavailable",
            status_reason="activation successor builders could not be imported",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )

    actions = queue.get("successor_actions")
    if not isinstance(actions, list):
        actions = build_local_activation_successor_actions(queue)
    decisions = queue.get("successor_decisions")
    if not isinstance(decisions, list):
        decisions = build_local_activation_successor_decisions({"successor_actions": actions})
    raw_decisions = [row for row in decisions if isinstance(row, dict)]
    public_decisions = [_public_preview_gated_successor_decision(row) for row in raw_decisions]

    stats_summary = {"local_activation_next_action_queue": {**queue, "successor_actions": actions, "successor_decisions": raw_decisions}}
    derived_proposals = _proposals_from_activation_successor_decisions(stats_summary)
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    backlog_changes = payload.get("backlog_changes") if isinstance(payload.get("backlog_changes"), dict) else {}
    extra_proposals = backlog_changes.get("create_issues") if isinstance(backlog_changes.get("create_issues"), list) else []
    if not extra_proposals and isinstance(evidence.get("backlog_changes"), dict):
        extra_proposals = evidence["backlog_changes"].get("create_issues") if isinstance(evidence["backlog_changes"].get("create_issues"), list) else []
    seen_proposals: set[tuple[str, str]] = set()
    public_proposals: list[dict[str, Any]] = []
    for proposal in [*derived_proposals, *[row for row in extra_proposals if isinstance(row, dict)]]:
        public = _public_preview_gated_issue_proposal(proposal)
        key = (str(public.get("repo") or ""), str(public.get("title") or ""))
        if key in seen_proposals:
            continue
        seen_proposals.add(key)
        public_proposals.append(public)

    status_counts = Counter(str(row.get("issue_queue_status") or "unknown") for row in public_decisions)
    decision_counts = Counter(str(row.get("decision") or "unknown") for row in public_decisions)
    issue_status_counts = Counter(str(row.get("issue_worthy_status") or "unknown") for row in public_decisions)
    preview_counts = Counter(str(row.get("preview_agreement_status") or "not-previewed") for row in public_decisions)
    family_counts = Counter(str(row.get("local_action_family") or "unknown") for row in public_decisions)
    reason_counts = Counter(_activation_issue_queue_reason(row) for row in public_decisions)
    capped = max(1, min(int(limit or 20), 50))
    queue_privacy = queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_preview_gated_activation_issue_queue.v1",
        "generated_at": utc_now(),
        "status": "ranked" if public_decisions else "empty",
        "status_reason": "latest preview-gated activation issue queue loaded" if public_decisions else "latest queue has no successor decisions",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "summary": {
            "successor_decision_count": len(public_decisions),
            "issue_proposal_count": len(public_proposals),
            "ready_count": status_counts.get("ready", 0),
            "blocked_count": status_counts.get("blocked", 0),
            "stale_or_no_data_count": status_counts.get("stale/no-data", 0),
            "suppressed_count": status_counts.get("suppressed", 0),
            "decision_counts": _breakdown_from_counts(dict(decision_counts)),
            "issue_status_counts": _breakdown_from_counts(dict(issue_status_counts)),
            "preview_agreement_status_counts": _breakdown_from_counts(dict(preview_counts)),
            "issue_queue_status_counts": _breakdown_from_counts(dict(status_counts)),
            "local_action_family_counts": _breakdown_from_counts(dict(family_counts)),
            "top_reason_counts": _breakdown_from_counts(dict(reason_counts))[:10],
            "top_ready_issue": _top_issue(public_decisions, "ready"),
            "top_blocked_issue": _top_issue(public_decisions, "blocked"),
            "top_stale_or_no_data_issue": _top_issue(public_decisions, "stale/no-data"),
        },
        "successor_decisions": public_decisions[:capped],
        "issue_proposals": public_proposals[:capped],
        "privacy": _local_activation_queue_privacy(queue_privacy),
    }


def _public_local_activation_queue_entry(
    entry: dict[str, Any],
    *,
    managed_preview_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {
        "rank",
        "ledger_rank",
        "lever",
        "local_action_family",
        "state",
        "current_status",
        "issue_worthy_status",
        "next_action",
        "unblock_reason",
        "blocking_reason",
        "freshness_state",
        "rank_bucket",
        "rank_basis",
        "blocker_codes",
        "sample_count",
        "applied_count",
        "holdout_count",
        "fallback_count",
        "safety_stop_count",
        "rollback_count",
        "realized_savings_usd",
        "projected_savings_usd",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
        "target_local_rule_file",
        "target_local_policy_section",
        "duplicate_suppression_status",
        "duplicate_suppression_reason",
        "evidence_schema",
        "expected_savings_path",
        "requested_model",
        "candidate_target_model",
        "required_local_executor",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
    }
    public = {
        key: _copy_policy(value)
        for key, value in entry.items()
        if key in allowed and value not in (None, "", [])
    }
    for key in (
        "rank",
        "ledger_rank",
        "sample_count",
        "applied_count",
        "holdout_count",
        "fallback_count",
        "safety_stop_count",
        "rollback_count",
        "rank_bucket",
    ):
        public[key] = _as_int(public.get(key))
    for key in (
        "realized_savings_usd",
        "projected_savings_usd",
        "savings_per_1000_calls_usd",
        "freshness_adjusted_savings_per_1000_calls_usd",
    ):
        public[key] = round(_as_float(public.get(key)), 8)
    if not isinstance(public.get("blocker_codes"), list):
        public["blocker_codes"] = []
    public["privacy"] = _local_activation_queue_privacy(entry.get("privacy") if isinstance(entry.get("privacy"), dict) else {})
    family_coverage = _managed_preview_coverage_for_family(
        managed_preview_coverage,
        str(public.get("local_action_family") or entry.get("local_action_family") or ""),
    )
    if family_coverage is not None:
        public["managed_preview_coverage"] = family_coverage
    return public


def _extract_local_activation_queue_from_plan(payload: dict[str, Any]) -> dict[str, Any] | None:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    stats_summary = evidence.get("stats_summary") if isinstance(evidence.get("stats_summary"), dict) else {}
    for container in (evidence, payload, stats_summary):
        queue = container.get("local_activation_next_action_queue") if isinstance(container, dict) else None
        if isinstance(queue, dict):
            return queue

    ledger = None
    for container in (evidence, payload, stats_summary):
        candidate = container.get("evidence_to_activation_next_action_ledger") if isinstance(container, dict) else None
        if isinstance(candidate, dict):
            ledger = candidate
            break
    if not isinstance(ledger, dict):
        return None
    try:
        from tokenclaw.orchestrator_research import build_local_activation_next_action_queue

        return build_local_activation_next_action_queue({"evidence_to_activation_next_action_ledger": ledger})
    except Exception:
        return None


async def stats_local_activation_next_action_queue(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    managed_preview_coverage = _managed_activation_preview_coverage(store_obj)
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_local_activation_queue_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_local_activation_queue_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )
    if not isinstance(payload, dict):
        return _empty_local_activation_queue_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
            managed_preview_coverage=managed_preview_coverage,
        )

    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _empty_local_activation_queue_payload(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
            managed_preview_coverage=managed_preview_coverage,
        )

    capped = max(1, min(int(limit or 20), 50))
    all_entries = [
        _public_local_activation_queue_entry(entry, managed_preview_coverage=managed_preview_coverage)
        for entry in queue.get("entries") or []
        if isinstance(entry, dict)
    ]
    entries = all_entries[:capped]
    summary = _public_local_activation_queue_summary(
        queue.get("summary") if isinstance(queue.get("summary"), dict) else {},
        len(entries),
    )
    queue_privacy = queue.get("privacy") if isinstance(queue.get("privacy"), dict) else {}
    return {
        "schema": "tokenclaw.dashboard_local_activation_next_action_queue.v1",
        "generated_at": utc_now(),
        "status": "ranked" if entries else "empty",
        "status_reason": "latest local activation next-action queue loaded" if entries else "latest queue has no entries",
        "queue_schema": queue.get("schema"),
        "queue_status": queue.get("status"),
        "source_schema": queue.get("source_schema"),
        "source": {**source, "plan_generated_at": payload.get("generated_at")},
        "summary": summary,
        "entries": entries,
        "successor_burndown": _activation_successor_burndown(all_entries),
        "activation_preview_agreement_burndown": _activation_preview_agreement_burndown(queue),
        "managed_preview_coverage": managed_preview_coverage,
        "privacy": _local_activation_queue_privacy(queue_privacy),
    }


async def stats_activation_preview_burndown(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    queue_payload = await stats_local_activation_next_action_queue(limit=limit, store_obj=store_obj)
    burndown = queue_payload.get("activation_preview_agreement_burndown")
    if not isinstance(burndown, dict):
        burndown = _activation_preview_agreement_burndown({})
    return {
        **burndown,
        "generated_at": utc_now(),
        "queue_status": queue_payload.get("status"),
        "queue_status_reason": queue_payload.get("status_reason"),
        "queue_schema": queue_payload.get("queue_schema"),
        "queue_source_schema": queue_payload.get("source_schema"),
        "source": queue_payload.get("source") if isinstance(queue_payload.get("source"), dict) else {},
        "managed_preview_coverage": queue_payload.get("managed_preview_coverage"),
    }


async def stats_preview_gated_activation_issue_queue(limit: int = 20, store_obj: Any | None = None) -> dict[str, Any]:
    del store_obj
    path = _evidence_to_activation_plan_path()
    source: dict[str, Any] = {
        "kind": "orchestrator-research-plan",
        "configured": any(os.getenv(name) for name in ("TOKENCLAW_EVIDENCE_TO_ACTIVATION_PLAN_JSON", "TOKENCLAW_RESEARCH_PLAN_JSON")),
        "path_class": _local_path_class(path),
        "path_included": False,
        "available": False,
    }
    try:
        stat = path.stat()
        source.update(
            {
                "available": True,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_preview_gated_activation_issue_queue_payload(
            status="unavailable",
            status_reason="latest research plan artifact was not found",
            source=source,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source["available"] = bool(path.exists())
        return _empty_preview_gated_activation_issue_queue_payload(
            status="invalid-artifact",
            status_reason=f"latest research plan artifact could not be read: {type(exc).__name__}",
            source=source,
        )
    if not isinstance(payload, dict):
        return _empty_preview_gated_activation_issue_queue_payload(
            status="invalid-artifact",
            status_reason="latest research plan artifact is not a JSON object",
            source=source,
        )

    queue = _extract_local_activation_queue_from_plan(payload)
    if not isinstance(queue, dict):
        return _empty_preview_gated_activation_issue_queue_payload(
            status="no-queue",
            status_reason="latest research plan does not contain a local activation next-action queue",
            source={**source, "plan_generated_at": payload.get("generated_at")},
        )
    return _preview_gated_activation_issue_queue_payload(
        payload=payload,
        queue=queue,
        source=source,
        limit=limit,
    )
