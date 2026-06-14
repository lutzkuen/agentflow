from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.promotion_blocker_recommendation_review.v1"
CANDIDATE_SCHEMA = "agentflow.promotion_blocker_review_candidate.v1"
GROUP_SCHEMA = "agentflow.promotion_blocker_review_group.v1"
EXPECTED_SOURCE_SCHEMA = "agentflow.promotion_blocker_next_action_recommendations.v1"

_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}$")
_ABSOLUTE_PATH_RE = re.compile(r"(^/[^/])|([A-Za-z]:\\\\)|(^\\\\\\\\)")
_SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9_.@+-]{1,160}$")


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


def _safe_list(value: Any, *, max_items: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = {_safe_label(item) for item in value[:max_items] if str(item or "").strip()}
    return sorted(labels)


def _safe_count_rows(values: list[str]) -> list[dict[str, Any]]:
    rows = [{"value": value, "count": count} for value, count in Counter(values).items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _safe_file_backed_representation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "exists": False,
            "reason": "missing-file-backed-local-policy",
        }
    result: dict[str, Any] = {
        "exists": bool(value.get("exists")),
    }
    for key in ("policy_section", "policy_source", "reason"):
        label = _safe_optional_label(value.get(key))
        if label:
            result[key] = label
    rule_file = str(value.get("rule_file") or "").strip()
    if rule_file:
        result["rule_file"] = rule_file if _SAFE_FILE_RE.match(rule_file) else "redacted-local-path"
    return result


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
        "blocker_count",
    ):
        if key in value:
            result[key] = _as_int(value.get(key))
    for key in ("projected_savings_usd", "rank_score"):
        if key in value:
            result[key] = round(_as_float(value.get(key)), 8)
    for key in (
        "promotion_status",
        "rollup_created_at",
        "oldest_record_at",
        "newest_record_at",
        "projected_savings_bucket",
        "capability_checked",
        "capability_status",
    ):
        label = _safe_optional_label(value.get(key))
        if label:
            result[key] = label
    return result


def _candidate(recommendation: dict[str, Any]) -> dict[str, Any]:
    file_representation = _safe_file_backed_representation(recommendation.get("file_backed_policy_representation"))
    no_op_reasons = _safe_list(recommendation.get("no_op_reasons"))
    status = _safe_label(recommendation.get("status"), default="noop")
    if status not in {"recommended", "noop"}:
        status = "noop"
        no_op_reasons = sorted(set(no_op_reasons + ["unsupported-recommendation-status"]))
    return {
        "schema": CANDIDATE_SCHEMA,
        "recommendation_id": _safe_label(recommendation.get("recommendation_id"), default="unknown-recommendation", max_length=240),
        "rank": _as_int(recommendation.get("rank")),
        "status": status,
        "recommendation_type": _safe_label(recommendation.get("recommendation_type"), default="noop"),
        "local_action_family": _safe_label(recommendation.get("local_action_family")),
        "candidate_family": _safe_label(recommendation.get("candidate_family")),
        "source_surface": _safe_label(recommendation.get("source_surface")),
        "provider_family": _safe_optional_label(recommendation.get("provider_family")),
        "provider_endpoint": _safe_optional_label(recommendation.get("provider_endpoint")),
        "blocker_family": _safe_label(recommendation.get("blocker_family")),
        "blocker_reason_codes": _safe_list(recommendation.get("blocker_reason_codes")),
        "blocker_count": _as_int(recommendation.get("blocker_count")),
        "next_action": _safe_label(recommendation.get("next_action"), default="keep-blocked"),
        "expected_local_executor": _safe_optional_label(recommendation.get("expected_local_executor")),
        "file_backed_policy_representation": file_representation,
        "local_executor_compatibility": _safe_compatibility(recommendation.get("local_executor_compatibility")),
        "confidence": _bounded_confidence(recommendation.get("confidence")),
        "projected_savings_usd": round(_as_float(recommendation.get("projected_savings_usd")), 8),
        "no_op_reasons": no_op_reasons,
        "evidence_summary": _safe_evidence_summary(recommendation.get("evidence_summary")),
        "required_local_review": True,
        "read_only": True,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "privacy": _privacy_summary(),
    }


def _group_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("local_action_family") or "unknown"), []).append(candidate)
    rows: list[dict[str, Any]] = []
    for family, items in grouped.items():
        reason_codes: list[str] = []
        next_actions: list[str] = []
        for item in items:
            reason_codes.extend(item.get("blocker_reason_codes") if isinstance(item.get("blocker_reason_codes"), list) else [])
            next_actions.append(str(item.get("next_action") or "keep-blocked"))
        items.sort(key=lambda row: (-_as_float(row.get("projected_savings_usd")), _as_int(row.get("rank")), str(row.get("recommendation_id"))))
        rows.append(
            {
                "schema": GROUP_SCHEMA,
                "local_action_family": family,
                "candidate_count": len(items),
                "recommended_count": sum(1 for item in items if item.get("status") == "recommended"),
                "noop_count": sum(1 for item in items if item.get("status") == "noop"),
                "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in items), 8),
                "top_next_action": next_actions[0] if next_actions else None,
                "blocker_reason_code_counts": _safe_count_rows(reason_codes),
                "recommendations": items,
                "privacy": _privacy_summary(),
            }
        )
    rows.sort(key=lambda row: (-_as_float(row.get("projected_savings_usd")), str(row.get("local_action_family"))))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def build_promotion_blocker_recommendation_review(
    payload: Any,
    *,
    limit: int = 20,
    fetch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    source_recommendations = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
    bounded_limit = max(0, min(int(limit or 0), 200))
    candidates = [
        _candidate(item)
        for item in source_recommendations[:bounded_limit]
        if isinstance(item, dict)
    ]
    groups = _group_candidates(candidates)
    all_reasons = [reason for item in candidates for reason in item.get("blocker_reason_codes", [])]
    result = {
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
                {"path": "$.schema", "message": "unexpected promotion blocker recommendation schema"}
            ],
        },
        "summary": {
            "source_recommendation_count": len(source_recommendations),
            "review_candidate_count": len(candidates),
            "group_count": len(groups),
            "recommended_count": sum(1 for item in candidates if item.get("status") == "recommended"),
            "noop_count": sum(1 for item in candidates if item.get("status") == "noop"),
            "projected_savings_usd": round(sum(_as_float(item.get("projected_savings_usd")) for item in candidates), 8),
            "top_local_action_family": groups[0].get("local_action_family") if groups else None,
            "top_next_action": groups[0].get("top_next_action") if groups else None,
            "blocker_reason_code_counts": _safe_count_rows(all_reasons),
        },
        "groups": groups,
        "candidates": candidates,
        "omitted_actions": [
            {
                "recommendation_id": item["recommendation_id"],
                "local_action_family": item["local_action_family"],
                "candidate_family": item["candidate_family"],
                "blocker_family": item["blocker_family"],
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
    # The report is intentionally constructed from a whitelist. Keep this assertion local so
    # future additions fail closed during tests instead of leaking raw local data.
    json.dumps(result, sort_keys=True)
    return result
