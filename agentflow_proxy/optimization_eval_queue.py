from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization_eval_plan import build_optimization_eval_plan
from agentflow_proxy.optimization_shadow_eval import run_optimization_shadow_eval
from agentflow_proxy.store import utc_now

SCHEMA = "agentflow.optimization_eval_queue_run.v1"

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
