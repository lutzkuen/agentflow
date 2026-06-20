from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.optimization_eval_plan import build_optimization_eval_plan
from tokenclaw.optimization_shadow_eval import run_optimization_shadow_eval
from tokenclaw.store import stable_json, utc_now

SCHEMA = "agentflow.optimization_eval_queue_run.v1"
PROMOTION_BACKFILL_SCHEMA = "agentflow.optimization_promotion_eval_backfill.v1"
PROMOTION_RECOMMENDATION_QUEUE_SCHEMA = "agentflow.promotion_recommendation_eval_queue.v1"

_LOCAL_REPLAY_LEVELS = {
    "local-exact-response",
    "local_exact_response",
    "local-provider-request",
    "local_provider_request",
    "raw_body_opt_in",
    "static_information",
}
_FEATURE_REPLAY_LEVELS = {
    "features_only",
    "metadata_only",
    "turn-metadata-only",
    "turn_metadata_only",
}
_RAW_PRIVACY_MARKERS = (
    "api-key",
    "body",
    "egress",
    "file-path",
    "filesystem",
    "identifier",
    "payload",
    "privacy",
    "prompt",
    "raw",
    "secret",
)


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _count_rows(counts: Counter[str]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": count} for key, count in counts.items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_provider_bodies_included": False,
        "raw_outputs_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "raw_session_ids_included": False,
        "filesystem_paths_included": False,
        "cache_keys_included": False,
        "api_keys_included": False,
        "local_only": True,
    }


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_time(row: dict[str, Any], *, plan_generated_at: str | None) -> datetime | None:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    for source in (row, evidence):
        for key in (
            "candidate_created_at",
            "candidate_seen_at",
            "last_seen_at",
            "latest_seen_at",
            "updated_at",
            "created_at",
            "generated_at",
        ):
            parsed = _parse_time(source.get(key))
            if parsed is not None:
                return parsed
    return _parse_time(plan_generated_at)


def _age_hours(row: dict[str, Any], *, now: datetime, plan_generated_at: str | None) -> float | None:
    row_time = _row_time(row, plan_generated_at=plan_generated_at)
    if row_time is None:
        return None
    return max(0.0, (now - row_time).total_seconds() / 3600.0)


def _replayability_rank(row: dict[str, Any]) -> int:
    level = str(row.get("replayability_level") or "").strip().lower()
    if level in _LOCAL_REPLAY_LEVELS:
        return 3
    if level == "static_information":
        return 2
    if level in _FEATURE_REPLAY_LEVELS:
        return 1
    return 0


def _privacy_safe(row: dict[str, Any]) -> bool:
    privacy = row.get("privacy") if isinstance(row.get("privacy"), dict) else {}
    if privacy:
        for key, value in privacy.items():
            key_text = str(key).lower()
            if key_text.endswith("_included") and bool(value):
                return False
            if key_text in {"content_free", "identifier_free", "metadata_only"} and value is False:
                return False
    text = " ".join(
        str(value or "").lower()
        for value in (
            row.get("replayability_level"),
            *(row.get("blocker_reason_codes") or []),
        )
    )
    return not any(marker in text for marker in _RAW_PRIVACY_MARKERS)


def _safe_reason_codes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if value})


def _eval_backfill_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons = _safe_reason_codes(candidate.get("reason_codes"))
    return [
        reason
        for reason in reasons
        if reason in {"eval-results-missing", "insufficient-eval-pass-results", "eval-queued"}
    ]


def _action_family_rank(value: Any) -> int:
    text = str(value or "").strip().lower().replace("-", "_")
    order = {"routing": 0, "cache": 1, "crunch": 2, "old_context_summarization": 2}
    return order.get(text, 9)


def _candidate_is_safety_blocked(candidate: dict[str, Any]) -> bool:
    reasons = set(_safe_reason_codes(candidate.get("reason_codes")))
    return bool(reasons & {"safety-stop-observed", "rollback-error-rate", "eval-failed"})


def _existing_queued_candidate_ids(store: Any) -> set[str]:
    if store is None or not hasattr(store, "conn"):
        return set()
    try:
        rows = store.conn.execute(
            """
            select candidate_id
            from optimization_eval_results
            where status_class = 'queued'
            """
        ).fetchall()
    except Exception:
        return set()
    return {str(row["candidate_id"] if hasattr(row, "keys") else row[0]) for row in rows if row}


def _existing_eval_result_ids(store: Any) -> set[str]:
    if store is None or not hasattr(store, "conn"):
        return set()
    try:
        rows = store.conn.execute("select id from optimization_eval_results").fetchall()
    except Exception:
        return set()
    return {str(row["id"] if hasattr(row, "keys") else row[0]) for row in rows if row}


def _promotion_eval_task(candidate: dict[str, Any], *, backfill_reason_codes: list[str]) -> dict[str, Any]:
    eval_evidence = candidate.get("eval_evidence") if isinstance(candidate.get("eval_evidence"), dict) else {}
    thresholds = candidate.get("thresholds") if isinstance(candidate.get("thresholds"), dict) else {}
    return {
        "schema": "agentflow.optimization_promotion_eval_task.v1",
        "candidate_id": str(candidate.get("candidate_id") or "unknown"),
        "optimization_family": str(candidate.get("optimization_family") or "unknown"),
        "action_family": str(candidate.get("action_family") or "unknown"),
        "source_surface": str(candidate.get("source_surface") or "unknown"),
        "app_family": str(candidate.get("app_family") or "unknown"),
        "candidate_target_model": candidate.get("candidate_target_model"),
        "candidate_profile": candidate.get("candidate_profile"),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "sample_count": _as_int(candidate.get("sample_count")),
        "backfill_reason_codes": backfill_reason_codes,
        "promotion_reason_codes": _safe_reason_codes(candidate.get("reason_codes")),
        "eval_status": {
            "result_count": _as_int(eval_evidence.get("result_count")),
            "pass_count": _as_int(eval_evidence.get("pass_count")),
            "queued_count": _as_int(eval_evidence.get("queued_count")),
            "min_eval_pass_count": _as_int(thresholds.get("min_eval_pass_count")) or 1,
        },
        "privacy": _privacy_summary(),
    }


def _safe_text(value: Any, *, default: str = "unknown", max_length: int = 240) -> str:
    text = str(value or "").strip().replace("_", "-")
    if not text:
        return default
    if text.startswith("/") or ":/" in text or ":\\" in text or text.startswith("\\\\"):
        return "redacted-local-path"
    return text[:max_length]


def _safe_metadata_dict(value: Any, *, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in sorted(allowed):
        if key not in value:
            continue
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
        elif isinstance(item, list):
            result[key] = _safe_reason_codes(item)
        else:
            result[key] = _safe_text(item, default="")
    return {
        key: item
        for key, item in result.items()
        if item != "" and item != [] and item != {}
    }


def _candidate_list_from_review(review: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = review.get("candidates") if isinstance(review.get("candidates"), list) else []
    if candidates:
        return [item for item in candidates if isinstance(item, dict)]
    groups = review.get("groups") if isinstance(review.get("groups"), list) else []
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        recommendations = group.get("recommendations") if isinstance(group.get("recommendations"), list) else []
        rows.extend(item for item in recommendations if isinstance(item, dict))
    return rows


def _recommendation_candidate_id(candidate: dict[str, Any]) -> str:
    recommendation_id = _safe_text(candidate.get("recommendation_id"), default="unknown-recommendation")
    return _stable_id("promotion-eval-recommendation", recommendation_id, candidate.get("local_action_family"), candidate.get("candidate_family"))


def _recommendation_result_id(candidate: dict[str, Any]) -> str:
    recommendation_id = _safe_text(candidate.get("recommendation_id"), default="unknown-recommendation")
    if recommendation_id != "unknown-recommendation":
        return f"promotion-eval-recommendation:{recommendation_id}"
    return _stable_id("promotion-eval-recommendation", candidate)


def _recommendation_noop_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons = _safe_reason_codes(candidate.get("no_op_reasons"))
    status = _safe_text(candidate.get("status"), default="noop")
    action_family = _safe_text(candidate.get("local_action_family"))
    blocker_codes = set(_safe_reason_codes(candidate.get("blocker_reason_codes")))
    next_action = _safe_text(candidate.get("next_action"), default="keep-blocked")
    recommendation_type = _safe_text(candidate.get("recommendation_type"), default="noop")
    executor = _safe_text(candidate.get("expected_local_executor"), default="")
    compatibility = candidate.get("local_executor_compatibility") if isinstance(candidate.get("local_executor_compatibility"), dict) else {}
    file_representation = candidate.get("file_backed_policy_representation") if isinstance(candidate.get("file_backed_policy_representation"), dict) else {}

    if status != "recommended":
        reasons.append("recommendation-not-recommended")
    if recommendation_type not in {"collect-eval-evidence", "shadow-eval", "run-shadow-eval"}:
        reasons.append("unsupported-recommendation-type")
    if next_action not in {"backfill-local-eval-evidence", "run-local-shadow-eval", "collect-local-eval-evidence"}:
        reasons.append("unsupported-next-action")
    if executor not in {"optimization-shadow-eval", "optimization-eval-queue", ""}:
        reasons.append("unsupported-local-executor")
    if action_family not in {"routing", "cache", "crunch", "old-context-summarization"}:
        reasons.append("unsupported-local-action-family")
    if not blocker_codes & {"eval-results-missing", "missing-eval-evidence", "insufficient-eval-pass-results"}:
        reasons.append("not-eval-blocker")
    if compatibility and _safe_text(compatibility.get("status"), default="unknown") not in {"compatible", "unknown"}:
        reasons.append("local-executor-incompatible")
    if file_representation and not bool(file_representation.get("exists")):
        reasons.append("missing-file-backed-local-policy")
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    promotion_status = _safe_text(evidence.get("promotion_status"), default="")
    if promotion_status and promotion_status not in {"needs-eval", "unknown"}:
        reasons.append("not-needs-eval")
    return sorted(set(reasons))


def _promotion_recommendation_eval_task(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence_summary") if isinstance(candidate.get("evidence_summary"), dict) else {}
    action_family = _safe_text(candidate.get("local_action_family"))
    return {
        "schema": "agentflow.promotion_recommendation_eval_task.v1",
        "candidate_id": _recommendation_candidate_id(candidate),
        "recommendation_id": _safe_text(candidate.get("recommendation_id"), default="unknown-recommendation"),
        "source": "promotion-blocker-recommendation",
        "optimization_family": _safe_text(candidate.get("candidate_family"), default=f"promotion-{action_family}"),
        "action_family": action_family,
        "source_surface": _safe_text(candidate.get("source_surface")),
        "provider_family": candidate.get("provider_family"),
        "provider_endpoint": candidate.get("provider_endpoint"),
        "blocker_family": _safe_text(candidate.get("blocker_family")),
        "blocker_reason_codes": _safe_reason_codes(candidate.get("blocker_reason_codes")),
        "recommended_eval_mode": "local-shadow-eval",
        "expected_local_executor": _safe_text(candidate.get("expected_local_executor"), default="optimization-shadow-eval"),
        "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        "sample_count": _as_int(evidence.get("record_count")) or _as_int(candidate.get("blocker_count")),
        "confidence": round(_as_float(candidate.get("confidence")), 6),
        "file_backed_policy_representation": _safe_metadata_dict(
            candidate.get("file_backed_policy_representation"),
            allowed={"exists", "policy_section", "policy_source", "reason", "rule_file"},
        ),
        "local_executor_compatibility": _safe_metadata_dict(
            candidate.get("local_executor_compatibility"),
            allowed={"status", "local_action_family", "minimum_local_client_version", "reason", "supported_local_action_families"},
        ),
        "privacy": _privacy_summary(),
    }


def queue_promotion_recommendation_eval_tasks(
    store: Any,
    review: dict[str, Any],
    *,
    family: str | None = None,
    limit: int = 25,
    apply: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    generated_at = now or utc_now()
    family_filter = _safe_text(family, default="") if family else None
    existing_ids = _existing_eval_result_ids(store)
    candidates = _candidate_list_from_review(review if isinstance(review, dict) else {})
    queued: list[dict[str, Any]] = []
    noops: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    noop_counts: Counter[str] = Counter()

    ranked = sorted(
        candidates,
        key=lambda row: (
            -_as_float(row.get("projected_savings_usd")),
            _action_family_rank(row.get("local_action_family")),
            _as_int(row.get("rank")),
            str(row.get("recommendation_id") or ""),
        ),
    )
    capped = max(1, min(int(limit or 25), 1000))
    for candidate in ranked:
        action_family = _safe_text(candidate.get("local_action_family"))
        action_counts[action_family] += 1
        recommendation_id = _safe_text(candidate.get("recommendation_id"), default="unknown-recommendation")
        if family_filter and action_family != family_filter:
            skipped.append({"recommendation_id": recommendation_id, "reason": "family-filtered", "local_action_family": action_family})
            continue
        task = _promotion_recommendation_eval_task(candidate)
        result_id = _recommendation_result_id(candidate)
        noop_reasons = _recommendation_noop_reasons(candidate)
        if result_id in existing_ids:
            skipped.append({"recommendation_id": recommendation_id, "reason": "already-recorded", "local_action_family": action_family})
            continue
        if noop_reasons:
            for reason in noop_reasons:
                noop_counts[reason] += 1
            noops.append({
                "schema": "agentflow.promotion_recommendation_eval_noop.v1",
                "id": result_id,
                "recommendation_id": recommendation_id,
                "candidate_id": task["candidate_id"],
                "local_action_family": action_family,
                "candidate_family": task["optimization_family"],
                "source_surface": task["source_surface"],
                "status_class": "noop",
                "reason_codes": noop_reasons,
                "projected_savings_usd": task["projected_savings_usd"],
                "privacy": _privacy_summary(),
            })
            continue
        if len(queued) >= capped:
            skipped.append({"recommendation_id": recommendation_id, "reason": "limit-exceeded", "local_action_family": action_family})
            continue
        queued.append({
            "schema": "agentflow.promotion_recommendation_eval_queue_row.v1",
            "id": result_id,
            "created_at": generated_at,
            "status_class": "queued",
            "reason_codes": sorted(set(["eval-queued", *task["blocker_reason_codes"]])),
            "task": task,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "privacy": _privacy_summary(),
        })

    written_queue_count = 0
    written_noop_count = 0
    run_id = _stable_id("promotion-recommendation-eval-queue", generated_at, family, [row["id"] for row in queued], [row["id"] for row in noops])
    if apply and store is not None:
        for row in queued:
            task = row["task"]
            result = {**row, "run_id": run_id}
            store.log_optimization_eval_result(
                id=row["id"],
                run_id=run_id,
                created_at=generated_at,
                candidate_id=task["candidate_id"],
                source_surface=task["source_surface"],
                optimization_family=task["optimization_family"],
                action_family=task["action_family"],
                status_class="queued",
                reason_codes_json=stable_json(row["reason_codes"]),
                score_json=stable_json({"queued": True, "source": "promotion-blocker-recommendation"}),
                cost_json=stable_json({"projected_savings_usd": task["projected_savings_usd"]}),
                result_json=stable_json(result),
            )
            written_queue_count += 1
            existing_ids.add(row["id"])
        for row in noops:
            if row["id"] in existing_ids:
                continue
            result = {**row, "run_id": run_id, "created_at": generated_at}
            store.log_optimization_eval_result(
                id=row["id"],
                run_id=run_id,
                created_at=generated_at,
                candidate_id=row["candidate_id"],
                source_surface=row["source_surface"],
                optimization_family=row["candidate_family"],
                action_family=row["local_action_family"],
                status_class="noop",
                reason_codes_json=stable_json(row["reason_codes"]),
                score_json=stable_json({"queued": False, "noop": True}),
                cost_json=stable_json({"projected_savings_usd": row["projected_savings_usd"]}),
                result_json=stable_json(result),
            )
            written_noop_count += 1
            existing_ids.add(row["id"])

    return {
        "schema": PROMOTION_RECOMMENDATION_QUEUE_SCHEMA,
        "generated_at": generated_at,
        "mode": "apply" if apply else "dry-run",
        "dry_run": not apply,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
        "wrote_eval_queue_rows": bool(written_queue_count),
        "wrote_noop_records": bool(written_noop_count),
        "run_id": run_id,
        "selection": {
            "family": family,
            "limit": capped,
            "sort": ["projected_savings_usd", "action_family", "rank", "recommendation_id"],
        },
        "summary": {
            "input_recommendation_count": len(candidates),
            "selected_task_count": len(queued),
            "noop_count": len(noops),
            "skipped_count": len(skipped),
            "written_task_count": written_queue_count,
            "written_noop_count": written_noop_count,
            "already_recorded_count": sum(1 for row in skipped if row.get("reason") == "already-recorded"),
            "action_family_counts": _count_rows(action_counts),
            "noop_reason_counts": _count_rows(noop_counts),
        },
        "tasks": [row["task"] for row in queued],
        "noop_records": noops,
        "skipped": skipped,
        "source_report_schema": review.get("schema") if isinstance(review, dict) else None,
        "privacy": _privacy_summary(),
    }


def _select_promotion_backfill_tasks(
    promotion_report: dict[str, Any],
    *,
    family: str | None,
    limit: int,
    already_queued: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    family_filter = str(family).strip() if family else None
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    considered = 0
    verdict_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    candidates = promotion_report.get("candidates") if isinstance(promotion_report, dict) else []
    if not isinstance(candidates, list):
        candidates = []

    ranked: list[tuple[dict[str, Any], list[str]]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        considered += 1
        candidate_id = str(item.get("candidate_id") or "")
        verdict = str(item.get("verdict") or "unknown")
        action_family = str(item.get("action_family") or "unknown")
        verdict_counts[verdict] += 1
        action_counts[action_family] += 1
        for reason in _safe_reason_codes(item.get("reason_codes")):
            reason_counts[reason] += 1
        skip_reason: str | None = None
        if not candidate_id:
            skip_reason = "missing-candidate-id"
        elif family_filter and str(item.get("optimization_family") or "") != family_filter:
            skip_reason = "family-filtered"
        elif verdict != "needs_eval":
            skip_reason = "not-needs-eval"
        elif candidate_id in already_queued or _as_int((item.get("eval_evidence") or {}).get("queued_count") if isinstance(item.get("eval_evidence"), dict) else 0) > 0:
            skip_reason = "already-queued"
        elif _candidate_is_safety_blocked(item):
            skip_reason = "safety-blocked"
        elif not _eval_backfill_reasons(item):
            skip_reason = "no-eval-blocker"
        elif not _privacy_safe(item):
            skip_reason = "privacy-risk"

        if skip_reason:
            skipped.append({
                "candidate_id": candidate_id or "unknown",
                "reason": skip_reason,
                "verdict": verdict,
                "optimization_family": str(item.get("optimization_family") or "unknown"),
                "action_family": action_family,
                "projected_savings_usd": round(_as_float(item.get("projected_savings_usd")), 8),
            })
            continue
        ranked.append((item, _eval_backfill_reasons(item)))

    ranked.sort(
        key=lambda entry: (
            -_as_float(entry[0].get("projected_savings_usd")),
            _action_family_rank(entry[0].get("action_family")),
            -len(_safe_reason_codes(entry[0].get("reason_codes"))),
            str(entry[0].get("candidate_id") or ""),
        )
    )
    capped = max(1, min(int(limit or 25), 1000))
    for candidate, backfill_reasons in ranked[:capped]:
        selected.append(_promotion_eval_task(candidate, backfill_reason_codes=backfill_reasons))
    for candidate, _backfill_reasons in ranked[capped:]:
        skipped.append({
            "candidate_id": str(candidate.get("candidate_id") or "unknown"),
            "reason": "limit-exceeded",
            "verdict": str(candidate.get("verdict") or "unknown"),
            "optimization_family": str(candidate.get("optimization_family") or "unknown"),
            "action_family": str(candidate.get("action_family") or "unknown"),
            "projected_savings_usd": round(_as_float(candidate.get("projected_savings_usd")), 8),
        })

    return selected, skipped, {
        "input_candidate_count": considered,
        "eligible_candidate_count": len(ranked),
        "verdict_counts": _count_rows(verdict_counts),
        "action_family_counts": _count_rows(action_counts),
        "reason_code_counts": _count_rows(reason_counts),
    }


def backfill_promotion_eval_tasks(
    store: Any,
    promotion_report: dict[str, Any],
    *,
    family: str | None = None,
    limit: int = 25,
    apply: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    generated_at = now or utc_now()
    queued_before = _existing_queued_candidate_ids(store)
    tasks, skipped, summary = _select_promotion_backfill_tasks(
        promotion_report,
        family=family,
        limit=limit,
        already_queued=queued_before,
    )
    run_id = _stable_id("promotion-eval-backfill", generated_at, family, [task["candidate_id"] for task in tasks])
    wrote_count = 0
    if apply and store is not None:
        for task in tasks:
            reason_codes = sorted(set(["eval-queued", *task["backfill_reason_codes"]]))
            result = {
                "schema": "agentflow.optimization_promotion_eval_queue_row.v1",
                "run_id": run_id,
                "created_at": generated_at,
                "status_class": "queued",
                "reason_codes": reason_codes,
                "task": task,
                "provider_call_made": False,
                "managed_server_call_made": False,
                "privacy": _privacy_summary(),
            }
            store.log_optimization_eval_result(
                id=f"promotion-eval-queued:{run_id}:{task['candidate_id']}",
                run_id=run_id,
                created_at=generated_at,
                candidate_id=task["candidate_id"],
                source_surface=task["source_surface"],
                optimization_family=task["optimization_family"],
                action_family=task["action_family"],
                status_class="queued",
                reason_codes_json=stable_json(reason_codes),
                score_json=stable_json({"queued": True}),
                cost_json=stable_json({"projected_savings_usd": task["projected_savings_usd"]}),
                result_json=stable_json(result),
            )
            wrote_count += 1

    return {
        "schema": PROMOTION_BACKFILL_SCHEMA,
        "generated_at": generated_at,
        "mode": "apply" if apply else "dry-run",
        "dry_run": not apply,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
        "wrote_eval_queue_rows": bool(wrote_count),
        "run_id": run_id,
        "selection": {
            "family": family,
            "limit": max(1, min(int(limit or 25), 1000)),
            "sort": ["projected_savings_usd", "action_family", "blocker_count", "candidate_id"],
        },
        "summary": {
            **summary,
            "selected_task_count": len(tasks),
            "skipped_candidate_count": len(skipped),
            "written_task_count": wrote_count,
            "already_queued_count": len(queued_before),
        },
        "tasks": tasks,
        "skipped": skipped,
        "source_report_schema": promotion_report.get("schema") if isinstance(promotion_report, dict) else None,
        "privacy": _privacy_summary(),
    }


def _annotate_stale(row: dict[str, Any], *, age_hours: float | None, max_age_hours: int | None) -> dict[str, Any]:
    copied = dict(row)
    evidence = dict(copied.get("evidence") if isinstance(copied.get("evidence"), dict) else {})
    blockers = [str(value) for value in (copied.get("blocker_reason_codes") or []) if value]
    if max_age_hours is not None and age_hours is not None and age_hours > max_age_hours:
        blockers.append("candidate-stale")
        evidence["candidate_age_hours"] = round(age_hours, 3)
        evidence["max_candidate_age_hours"] = max_age_hours
    copied["blocker_reason_codes"] = sorted(set(blockers))
    copied["evidence"] = evidence
    return copied


def _select_rows(
    rows: list[Any],
    *,
    family: str | None,
    limit: int,
    max_candidate_age_hours: int | None,
    plan_generated_at: str | None,
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_filter = str(family).strip() if family else None
    candidates: list[tuple[dict[str, Any], float | None, bool]] = []
    family_filtered = 0
    stale_count = 0
    privacy_risky = 0
    action_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    replay_counts: Counter[str] = Counter()

    for item in rows:
        if not isinstance(item, dict):
            continue
        if family_filter and str(item.get("optimization_family") or "") != family_filter:
            family_filtered += 1
            continue
        age = _age_hours(item, now=now, plan_generated_at=plan_generated_at)
        stale = max_candidate_age_hours is not None and age is not None and age > max_candidate_age_hours
        if stale:
            stale_count += 1
        privacy_ok = _privacy_safe(item)
        if not privacy_ok:
            privacy_risky += 1
        action_counts[str(item.get("action_family") or "unknown")] += 1
        family_counts[str(item.get("optimization_family") or "unknown")] += 1
        replay_counts[str(item.get("replayability_level") or "unknown")] += 1
        candidates.append((_annotate_stale(item, age_hours=age, max_age_hours=max_candidate_age_hours), age, privacy_ok))

    candidates.sort(
        key=lambda entry: (
            0 if entry[2] else 1,
            0 if "candidate-stale" not in (entry[0].get("blocker_reason_codes") or []) else 1,
            -_as_float(entry[0].get("projected_savings_usd")),
            -_replayability_rank(entry[0]),
            str(entry[0].get("candidate_id") or ""),
        )
    )
    capped = max(1, min(int(limit or 25), 1000))
    selected = [row for row, _age, _privacy_ok in candidates[:capped]]
    return selected, {
        "input_candidate_count": sum(1 for item in rows if isinstance(item, dict)),
        "candidate_count_after_family_filter": len(candidates),
        "family_filtered_count": family_filtered,
        "stale_candidate_count": stale_count,
        "privacy_risky_candidate_count": privacy_risky,
        "action_family_counts": _count_rows(action_counts),
        "optimization_family_counts": _count_rows(family_counts),
        "replayability_counts": _count_rows(replay_counts),
    }


async def run_optimization_eval_queue(
    store: Any,
    *,
    family: str | None = None,
    limit: int = 25,
    max_candidate_age_hours: int | None = None,
    execute: bool = False,
    budget_usd: float = 0.0,
    min_output_similarity: float = 0.9,
    plan_limit: int = 500,
    min_samples: int = 1,
    results_jsonl_path: str | None = None,
    plan: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    source_plan = plan
    if source_plan is None:
        source_plan = await build_optimization_eval_plan(store, limit=plan_limit, min_samples=min_samples)
    rows = source_plan.get("plans") if isinstance(source_plan, dict) else []
    if not isinstance(rows, list):
        rows = []

    selected, selection_summary = _select_rows(
        rows,
        family=family,
        limit=limit,
        max_candidate_age_hours=max_candidate_age_hours,
        plan_generated_at=source_plan.get("generated_at") if isinstance(source_plan, dict) else None,
        now=current_time,
    )
    bounded_plan = dict(source_plan)
    bounded_plan["plans"] = selected
    bounded_plan["queue_selection"] = {
        "family": family,
        "limit": max(1, min(int(limit or 25), 1000)),
        "max_candidate_age_hours": max_candidate_age_hours,
        "selected_candidate_ids": [str(row.get("candidate_id") or "unknown") for row in selected],
        "sort": ["privacy_eligible", "freshness", "projected_savings_usd", "replayability", "candidate_id"],
    }

    shadow = run_optimization_shadow_eval(
        bounded_plan,
        store=store,
        execute=execute,
        budget_usd=budget_usd,
        min_output_similarity=min_output_similarity,
        max_candidates=len(selected) or 1,
        results_jsonl_path=results_jsonl_path,
    )
    status_counts = {
        str(item.get("value") or "unknown"): _as_int(item.get("count"))
        for item in (shadow.get("summary") or {}).get("status_counts", [])
        if isinstance(item, dict)
    }
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "mode": shadow.get("mode"),
        "provider_calls_made": bool(shadow.get("provider_calls_made")),
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
        "wrote_result_records": bool(shadow.get("wrote_result_records")),
        "result_record_store": shadow.get("result_record_store"),
        "results_jsonl_path": shadow.get("results_jsonl_path"),
        "run_id": shadow.get("run_id"),
        "selection": bounded_plan["queue_selection"],
        "summary": {
            **selection_summary,
            "selected_candidate_count": len(selected),
            "result_count": len(shadow.get("results") or []),
            "status_counts": _count_rows(Counter(status_counts)),
            "provider_call_count": _as_int((shadow.get("summary") or {}).get("provider_call_count")),
            "budget_usd": (shadow.get("summary") or {}).get("budget_usd", round(max(0.0, float(budget_usd or 0.0)), 8)),
            "budget_remaining_usd": (shadow.get("summary") or {}).get("budget_remaining_usd"),
            "budget_exhausted_count": (shadow.get("summary") or {}).get("budget_exhausted_count", 0),
        },
        "selected_candidates": [
            {
                "candidate_id": str(row.get("candidate_id") or "unknown"),
                "optimization_family": str(row.get("optimization_family") or "unknown"),
                "action_family": str(row.get("action_family") or "unknown"),
                "projected_savings_usd": round(_as_float(row.get("projected_savings_usd")), 8),
                "replayability_level": str(row.get("replayability_level") or "unknown"),
                "blocker_reason_codes": [str(value) for value in (row.get("blocker_reason_codes") or []) if value],
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_provider_bodies_included": False,
                    "raw_outputs_included": False,
                    "tool_payloads_included": False,
                    "request_ids_included": False,
                    "raw_session_ids_included": False,
                    "filesystem_paths_included": False,
                    "api_keys_included": False,
                    "local_only": True,
                },
            }
            for row in selected
        ],
        "results": shadow.get("results") or [],
        "source_plan_schema": source_plan.get("schema") if isinstance(source_plan, dict) else None,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_outputs_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
            "api_keys_included": False,
            "local_only": True,
        },
    }


def run_optimization_eval_queue_sync(store: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_optimization_eval_queue(store, **kwargs))
