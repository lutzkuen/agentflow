from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from tokenclaw.store import utc_now


SCHEMA = "agentflow.post_promotion_priority_delta_review.v1"
DELTA_SCHEMA = "agentflow.post_promotion_priority_delta_candidate.v1"
GROUP_SCHEMA = "agentflow.post_promotion_priority_delta_group.v1"
EXPECTED_SOURCE_SCHEMA = "agentflow.post_promotion_policy_priority_deltas.v1"

_VALID_NEXT_ACTIONS = {"widen-local-policy", "collect-holdout-evidence", "rollback-local-policy", "keep-blocked"}
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}$")
_ABSOLUTE_PATH_RE = re.compile(r"(^/[^/])|([A-Za-z]:\\\\)|(^\\\\\\\\)")
_SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9_.@+-]{1,160}$")

_ACTION_RANK = {"widen-local-policy": 0, "collect-holdout-evidence": 1, "rollback-local-policy": 2, "keep-blocked": 3}


def _privacy_summary() -> dict[str, Any]:
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
        "wrote_local_policy_files": False,
        "managed_enforced": False,
    }


def _safe_label(value: Any, *, default: str = "unknown", max_length: int = 200) -> str:
    text = str(value or "").strip().replace("_", "-")
    if not text:
        return default
    if _ABSOLUTE_PATH_RE.search(text):
        return "redacted-local-path"
    text = text[:max_length]
    return text if _LABEL_RE.match(text) else "unsanitized-label"


def _safe_optional_label(value: Any, *, max_length: int = 200) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return _safe_label(text, max_length=max_length)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_confidence(value: Any) -> float:
    return round(min(1.0, max(0.0, _as_float(value))), 6)


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _safe_list(value: Any, *, max_items: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = {_safe_label(item) for item in value[:max_items] if str(item or "").strip()}
    return sorted(labels)


def _safe_count_rows(values: list[str]) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count} for value, count in Counter(values).items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _safe_compatibility(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("status", "local_action_family", "minimum_local_client_version", "reason"):
        label = _safe_optional_label(value.get(key))
        if label:
            result[key] = label
    if isinstance(value.get("supported_local_action_families"), list):
        result["supported_local_action_families"] = _safe_list(value["supported_local_action_families"])
    return result


def _safe_evidence_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "record_count",
        "candidate_count",
        "rollup_count",
        "sample_count",
        "affected_call_count",
        "affected_row_count",
        "current_canary_count",
        "current_holdout_count",
        "current_applied_count",
        "canary_applied_count",
        "canary_holdout_count",
        "applied_count",
        "holdout_count",
        "required_applied_count",
        "required_holdout_count",
        "required_min_applied_count",
        "required_min_holdout_count",
        "min_applied_samples",
        "min_holdout_samples",
        "minimum_applied_samples",
        "minimum_holdout_samples",
        "min_canary_applied_samples",
        "min_canary_holdout_samples",
        "safety_stop_count",
        "safety_stopped_count",
    ):
        if key in value:
            result[key] = _as_int(value.get(key))
    for key in (
        "savings_delta_usd",
        "rank_score",
        "current_canary_fraction",
        "canary_fraction",
        "current_holdout_fraction",
        "holdout_fraction",
        "required_holdout_fraction",
        "minimum_holdout_fraction",
        "min_holdout_fraction",
        "projected_savings_usd",
        "observed_savings_usd",
    ):
        if key in value:
            result[key] = round(_as_float(value.get(key)), 8)
    for key in (
        "promotion_status",
        "policy_section",
        "source_surface",
        "action_family",
        "local_action_family",
        "feedback_window",
        "stability_score_label",
        "projected_savings_bucket",
        "candidate_family",
        "requested_model",
        "routed_model",
        "target_model",
        "candidate_target_model",
        "cache_profile",
        "crunch_profile",
        "profile",
        "policy_profile",
    ):
        label = _safe_optional_label(value.get(key))
        if label:
            result[key] = label
    for key in (
        "safety_stop_active",
        "safety_stop_tripped",
        "stale",
        "stale_evidence",
        "preserved_previous_rule",
        "previous_rule_preserved",
        "previous_rule_available",
    ):
        raw = value.get(key)
        if key == "stale_evidence" and isinstance(raw, dict):
            raw = raw.get("stale")
        safe = _safe_bool(raw)
        if safe is not None:
            result[key] = safe
    return result


def _candidate_count(candidate: dict[str, Any], *keys: str) -> int:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    for key in keys:
        number = _as_int(candidate.get(key) if key in candidate else evidence.get(key))
        if number > 0:
            return number
    return 0


def _candidate_float(candidate: dict[str, Any], *keys: str) -> float | None:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    for key in keys:
        value = candidate.get(key) if key in candidate else evidence.get(key)
        if value is None or isinstance(value, bool):
            continue
        number = _as_float(value, -1.0)
        if number >= 0.0:
            return round(number, 8)
    return None


def _candidate_bool(candidate: dict[str, Any], *keys: str) -> bool | None:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    for key in keys:
        value = candidate.get(key) if key in candidate else evidence.get(key)
        safe = _safe_bool(value)
        if safe is not None:
            return safe
    return None


def _candidate_label(candidate: dict[str, Any], *keys: str, default: str = "unknown") -> str:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    for key in keys:
        value = candidate.get(key) if key in candidate else evidence.get(key)
        label = _safe_optional_label(value)
        if label:
            return label
    return default


def _has_holdout_coverage(candidate: dict[str, Any]) -> bool:
    if _candidate_count(candidate, "current_holdout_count", "holdout_count", "canary_holdout_count") > 0:
        return True
    fraction = _candidate_float(candidate, "current_holdout_fraction", "holdout_fraction")
    return bool(fraction and fraction > 0.0)


def _safety_stop_active(candidate: dict[str, Any]) -> bool:
    if _candidate_count(candidate, "safety_stop_count", "safety_stopped_count") > 0:
        return True
    if _candidate_bool(candidate, "safety_stop_active", "safety_stop_tripped") is True:
        return True
    return any("safety-stop" in reason for reason in _safe_list(candidate.get("no_op_reasons")))


def _stale_evidence(candidate: dict[str, Any]) -> bool:
    return _candidate_bool(candidate, "stale_evidence", "stale") is True or any(
        "stale" in reason for reason in _safe_list(candidate.get("no_op_reasons"))
    )


def _unsupported_policy(candidate: dict[str, Any]) -> bool:
    return any("unsupported" in reason for reason in _safe_list(candidate.get("no_op_reasons")))


def _holdout_evidence_successor(candidate: dict[str, Any]) -> dict[str, Any] | None:
    if candidate.get("status") != "recommended":
        return None
    if candidate.get("next_action") != "widen-local-policy":
        return None
    if _has_holdout_coverage(candidate) or _safety_stop_active(candidate) or _stale_evidence(candidate) or _unsupported_policy(candidate):
        return None

    current_applied = _candidate_count(candidate, "current_applied_count", "applied_count", "canary_applied_count", "current_canary_count")
    current_holdout = _candidate_count(candidate, "current_holdout_count", "holdout_count", "canary_holdout_count")
    required_applied = max(
        1,
        _candidate_count(
            candidate,
            "required_applied_count",
            "required_min_applied_count",
            "min_applied_samples",
            "minimum_applied_samples",
            "min_canary_applied_samples",
        ),
    )
    required_holdout = max(
        1,
        _candidate_count(
            candidate,
            "required_holdout_count",
            "required_min_holdout_count",
            "min_holdout_samples",
            "minimum_holdout_samples",
            "min_canary_holdout_samples",
        ),
    )
    current_holdout_fraction = _candidate_float(candidate, "current_holdout_fraction", "holdout_fraction") or 0.0
    required_holdout_fraction = _candidate_float(candidate, "required_holdout_fraction", "minimum_holdout_fraction", "min_holdout_fraction") or 0.10
    return {
        "schema": "agentflow.post_promotion_holdout_evidence_successor.v1",
        "status": "needs-holdout-evidence",
        "reason": "missing-holdout-coverage",
        "original_next_action": "widen-local-policy",
        "next_action": "collect-holdout-evidence",
        "candidate_family": _candidate_label(candidate, "candidate_family", "recommendation_type"),
        "local_action_family": _candidate_label(candidate, "local_action_family", "action_family"),
        "action_family": _candidate_label(candidate, "action_family"),
        "source_surface": _candidate_label(candidate, "source_surface"),
        "policy_section": _candidate_label(candidate, "policy_section"),
        "requested_model": _candidate_label(candidate, "requested_model", default="unknown"),
        "routed_model": _candidate_label(candidate, "routed_model", "target_model", "candidate_target_model", default="unknown"),
        "cache_profile": _candidate_label(candidate, "cache_profile", default="none"),
        "crunch_profile": _candidate_label(candidate, "crunch_profile", "profile", "policy_profile", default="none"),
        "current_applied_count": current_applied,
        "current_holdout_count": current_holdout,
        "required_min_applied_count": required_applied,
        "required_min_holdout_count": required_holdout,
        "current_holdout_fraction": current_holdout_fraction,
        "required_holdout_fraction": required_holdout_fraction,
        "remaining_applied_count": max(0, required_applied - current_applied),
        "remaining_holdout_count": max(0, required_holdout - current_holdout),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": _privacy_summary(),
    }


def _delta_candidate(delta: dict[str, Any]) -> dict[str, Any]:
    next_action = _safe_label(delta.get("next_action"), default="keep-blocked")
    if next_action not in _VALID_NEXT_ACTIONS:
        next_action = "keep-blocked"

    status = _safe_label(delta.get("status"), default="noop")
    no_op_reasons = _safe_list(delta.get("no_op_reasons"))
    if status not in {"recommended", "noop"}:
        status = "noop"
        no_op_reasons = sorted(set(no_op_reasons + ["unsupported-delta-status"]))

    if next_action == "keep-blocked" and status == "recommended":
        no_op_reasons = sorted(set(no_op_reasons + ["next-action-is-keep-blocked"]))
        status = "noop"

    evidence_summary = _safe_evidence_summary(delta.get("evidence_summary"))
    compatibility = _safe_compatibility(delta.get("local_executor_compatibility"))

    candidate: dict[str, Any] = {
        "schema": DELTA_SCHEMA,
        "delta_id": _safe_label(delta.get("delta_id"), default="unknown-delta", max_length=240),
        "rank": _as_int(delta.get("rank")),
        "status": status,
        "next_action": next_action,
        "action_family": _safe_label(delta.get("action_family"), default="unknown"),
        "source_surface": _safe_label(delta.get("source_surface"), default="unknown"),
        "recommendation_type": _safe_label(delta.get("recommendation_type"), default="noop"),
        "no_op_reasons": no_op_reasons,
        "savings_delta_usd": round(_as_float(delta.get("savings_delta_usd")), 8),
        "confidence": _bounded_confidence(delta.get("confidence")),
        "local_executor_compatibility": compatibility,
        "evidence_summary": evidence_summary,
        "required_local_review": True,
        "read_only": True,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "privacy": _privacy_summary(),
    }
    for key in ("policy_section", "feedback_window", "stability_score_label"):
        value = _safe_optional_label(delta.get(key)) or evidence_summary.get(key)
        if value:
            candidate[key] = value
    for key in (
        "current_canary_fraction",
        "holdout_fraction",
        "current_holdout_fraction",
        "preserved_previous_rule",
        "previous_rule_preserved",
        "previous_rule_available",
        "safety_stop_active",
        "safety_stop_tripped",
        "stale_evidence",
        "stale",
    ):
        if key in delta:
            raw = delta.get(key)
            if key == "stale_evidence" and isinstance(raw, dict):
                raw = raw.get("stale")
            safe_bool = _safe_bool(raw)
            if safe_bool is not None:
                candidate[key] = safe_bool
            elif key in {"current_canary_fraction", "holdout_fraction", "current_holdout_fraction"}:
                candidate[key] = round(_as_float(delta.get(key)), 8)
    successor = _holdout_evidence_successor(candidate)
    if successor is not None:
        candidate["original_next_action"] = candidate["next_action"]
        candidate["next_action"] = "collect-holdout-evidence"
        candidate["no_op_reasons"] = sorted(set(no_op_reasons + ["missing-holdout-coverage"]))
        candidate["holdout_evidence_successor"] = successor
    return candidate


def _group_deltas(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        key = str(candidate.get("action_family") or "unknown")
        grouped.setdefault(key, []).append(candidate)

    rows: list[dict[str, Any]] = []
    for family, items in grouped.items():
        next_actions: list[str] = [str(item.get("next_action") or "keep-blocked") for item in items]
        items.sort(
            key=lambda row: (
                _ACTION_RANK.get(str(row.get("next_action") or "keep-blocked"), 99),
                -_as_float(row.get("savings_delta_usd")),
                _as_int(row.get("rank")),
                str(row.get("delta_id")),
            )
        )
        rows.append(
            {
                "schema": GROUP_SCHEMA,
                "action_family": family,
                "delta_count": len(items),
                "recommended_count": sum(1 for item in items if item.get("status") == "recommended"),
                "noop_count": sum(1 for item in items if item.get("status") == "noop"),
                "savings_delta_usd": round(sum(_as_float(item.get("savings_delta_usd")) for item in items), 8),
                "top_next_action": next_actions[0] if next_actions else None,
                "next_action_counts": _safe_count_rows(next_actions),
                "deltas": items,
                "privacy": _privacy_summary(),
            }
        )
    rows.sort(
        key=lambda row: (
            _ACTION_RANK.get(str(row.get("top_next_action") or "keep-blocked"), 99),
            -_as_float(row.get("savings_delta_usd")),
            str(row.get("action_family")),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_post_promotion_priority_delta_review(
    payload: Any,
    *,
    limit: int = 20,
    fetch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    source_deltas = payload.get("deltas") if isinstance(payload.get("deltas"), list) else []
    bounded_limit = max(0, min(int(limit or 0), 200))
    candidates = [
        _delta_candidate(item)
        for item in source_deltas[:bounded_limit]
        if isinstance(item, dict)
    ]
    groups = _group_deltas(candidates)
    all_next_actions = [str(item.get("next_action") or "keep-blocked") for item in candidates]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "read_only": True,
        "wrote_local_policy_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": bool(fetch and fetch.get("status") not in {"skipped", None}),
        "source_schema": payload.get("schema") if isinstance(payload.get("schema"), str) else None,
        "validation": {
            "ok": payload.get("schema") in {EXPECTED_SOURCE_SCHEMA, None},
            "expected_schema": EXPECTED_SOURCE_SCHEMA,
            "source_schema": payload.get("schema") if isinstance(payload.get("schema"), str) else None,
            "warnings": [] if payload.get("schema") in {EXPECTED_SOURCE_SCHEMA, None} else [
                {"path": "$.schema", "message": "unexpected post-promotion priority delta schema"}
            ],
        },
        "summary": {
            "source_delta_count": len(source_deltas),
            "review_candidate_count": len(candidates),
            "group_count": len(groups),
            "recommended_count": sum(1 for item in candidates if item.get("status") == "recommended"),
            "noop_count": sum(1 for item in candidates if item.get("status") == "noop"),
            "savings_delta_usd": round(sum(_as_float(item.get("savings_delta_usd")) for item in candidates), 8),
            "top_action_family": groups[0].get("action_family") if groups else None,
            "top_next_action": groups[0].get("top_next_action") if groups else None,
            "next_action_counts": _safe_count_rows(all_next_actions),
        },
        "groups": groups,
        "candidates": candidates,
        "omitted_actions": [
            {
                "delta_id": item["delta_id"],
                "action_family": item["action_family"],
                "next_action": "keep-blocked",
                "no_op_reasons": item["no_op_reasons"],
                "feature_only": True,
                "locally_executed": True,
                "provider_forwarding": False,
                "server_content_processing": False,
                "managed_enforced": False,
            }
            for item in candidates
            if item.get("status") == "noop"
        ],
        "privacy": _privacy_summary(),
    }
    if fetch:
        result["fetch"] = fetch
    json.dumps(result, sort_keys=True)
    return result
