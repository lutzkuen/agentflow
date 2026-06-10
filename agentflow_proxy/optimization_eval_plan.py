from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from typing import Any

from agentflow_proxy.phase_routing_report import build_phase_routing_report
from agentflow_proxy.stats import (
    stats_cache_replay_confidence,
    stats_cache_replayability,
    stats_managed_pattern_rollups,
    stats_old_context_summary,
)
from agentflow_proxy.store import utc_now

SCHEMA = "agentflow.optimization_eval_plan.v1"
ROW_SCHEMA = "agentflow.optimization_eval_plan_row.v1"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _breakdown_values(rows: Any, key: str = "value") -> list[str]:
    if not isinstance(rows, list):
        return []
    values: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            value = row.get(key) or row.get("reason") or row.get("blocker")
            if value:
                values.append(str(value))
    return sorted(set(values))


def _count_breakdown(counts: Counter[str]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": value} for key, value in counts.items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "identifier_free": True,
        "filesystem_path_free": True,
        "local_only": True,
    }


def _eval_mode(*, projected: int, applied: int = 0, holdout: int = 0, blockers: list[str] | None = None) -> str:
    blocker_set = set(blockers or [])
    if applied > 0 or holdout > 0:
        return "score-canary-holdout"
    if projected > 0:
        return "run-local-shadow-eval"
    if blocker_set & {"file-dependency-missing", "safe-invalidation-required", "tool-call-disabled"}:
        return "collect-invalidation-evidence"
    if blocker_set:
        return "collect-metadata-evidence"
    return "collect-baseline-evidence"


def _add_common(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
    optimization_family: str,
    action_family: str,
    source_surface: str,
    app_family: str,
    workflow_phase: str,
    category: str,
    candidate_target_model: str | None = None,
    candidate_profile: str | None = None,
    projected_savings_usd: float = 0.0,
    sample_count: int = 0,
    current_canary_count: int = 0,
    holdout_count: int = 0,
    blocker_reason_codes: list[str] | None = None,
    recommended_eval_mode: str | None = None,
    replayability_level: str = "metadata_only",
    evidence: dict[str, Any] | None = None,
) -> None:
    blockers = sorted(set(str(item) for item in (blocker_reason_codes or []) if item))
    rows.append({
        "schema": ROW_SCHEMA,
        "candidate_id": candidate_id,
        "optimization_family": optimization_family,
        "action_family": action_family,
        "source_surface": source_surface or "unknown",
        "app_family": app_family or "unknown",
        "granularity": "provider_request" if source_surface != "codex_app_turn" else "agent_turn",
        "workflow_phase": workflow_phase or "unknown",
        "category": category or "unknown",
        "candidate_target_model": candidate_target_model,
        "candidate_profile": candidate_profile,
        "projected_savings_usd": round(max(0.0, float(projected_savings_usd or 0.0)), 8),
        "sample_count": max(0, int(sample_count or 0)),
        "current_canary_count": max(0, int(current_canary_count or 0)),
        "holdout_count": max(0, int(holdout_count or 0)),
        "blocker_reason_codes": blockers,
        "recommended_eval_mode": recommended_eval_mode
        or _eval_mode(projected=max(0, int(sample_count or 0)), applied=current_canary_count, holdout=holdout_count, blockers=blockers),
        "replayability_level": replayability_level or "metadata_only",
        "evidence": evidence or {},
        "privacy": _privacy_summary(),
    })


def _phase_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("opportunities") or []:
        if not isinstance(item, dict) or not item.get("target_model"):
            continue
        blockers = _breakdown_values(item.get("blocked_count_by_reason")) + _breakdown_values(item.get("risk_exclusions"))
        candidate_id = _stable_id(
            "eval-routing",
            item.get("phase"),
            item.get("model_pair"),
            item.get("target_model"),
        )
        projected = _as_int(item.get("projected_candidate_count"))
        _add_common(
            rows,
            candidate_id=candidate_id,
            optimization_family="phase_routing",
            action_family="routing",
            source_surface="anthropic_messages",
            app_family="claude_code",
            workflow_phase=str(item.get("phase") or "unknown"),
            category=str(item.get("phase") or "unknown"),
            candidate_target_model=str(item.get("target_model") or ""),
            projected_savings_usd=_as_float(item.get("projected_savings_usd")),
            sample_count=_as_int(item.get("sample_count")),
            current_canary_count=_as_int(item.get("current_routed_count")),
            blocker_reason_codes=blockers,
            recommended_eval_mode=_eval_mode(projected=projected, applied=_as_int(item.get("current_routed_count")), blockers=blockers),
            replayability_level="features_only",
            evidence={
                "projected_candidate_count": projected,
                "current_routed_count": _as_int(item.get("current_routed_count")),
                "model_pair": item.get("model_pair"),
            },
        )
    return rows


def _cache_replayability_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_classes = {
        "replay-safe-exact-candidate",
        "streaming-non-tool-exact-candidate",
        "blocked-tool-result-invalidation",
        "blocked-structural",
    }
    for item in report.get("groups") or []:
        if not isinstance(item, dict):
            continue
        candidate_class = str(item.get("replay_candidate_class") or "unknown")
        if candidate_class not in candidate_classes and not _as_float(item.get("projected_repeated_call_cost_usd")):
            continue
        candidate_id = _stable_id(
            "eval-cache-replay",
            item.get("source_surface"),
            item.get("granularity"),
            item.get("category"),
            item.get("workflow_phase"),
            item.get("text_size_bucket"),
            candidate_class,
        )
        blockers = [str(value) for value in item.get("replayability_blockers") or [] if value]
        projected = 1 if candidate_class in {"replay-safe-exact-candidate", "streaming-non-tool-exact-candidate"} else 0
        _add_common(
            rows,
            candidate_id=candidate_id,
            optimization_family="cache_replayability",
            action_family="cache",
            source_surface=str(item.get("source_surface") or "unknown"),
            app_family="codex" if str(item.get("source_surface") or "") == "codex_app_turn" else "claude_code",
            workflow_phase=str(item.get("workflow_phase") or "unknown"),
            category=str(item.get("category") or "unknown"),
            candidate_profile=candidate_class,
            projected_savings_usd=_as_float(item.get("projected_repeated_call_cost_usd")),
            sample_count=_as_int(item.get("count")),
            blocker_reason_codes=blockers,
            recommended_eval_mode=_eval_mode(projected=projected, blockers=blockers),
            replayability_level=str(item.get("replayability_level") or "metadata_only"),
            evidence={
                "repeated": bool(item.get("repeated")),
                "sessions": _as_int(item.get("sessions")),
                "cacheability_bucket": item.get("cacheability_bucket"),
            },
        )
    return rows


def _cache_confidence_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("rules") or []:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id") or "")
        candidate_id = str(item.get("candidate_id") or "") or _stable_id(
            "eval-cache-confidence",
            rule_id,
            item.get("source_surface"),
            item.get("category"),
            item.get("stream"),
            item.get("has_tools"),
        )
        blockers = (
            _breakdown_values(item.get("invalidation_reasons"))
            + _breakdown_values(item.get("stale_risk_reasons"))
            + _breakdown_values(item.get("safety_stop_reasons"))
        )
        applied = _as_int(item.get("hit_count"))
        holdout = _as_int(item.get("holdout_count"))
        _add_common(
            rows,
            candidate_id=candidate_id,
            optimization_family="cache_replay_confidence",
            action_family="cache",
            source_surface=str(item.get("source_surface") or "unknown"),
            app_family="claude_code",
            workflow_phase=str(item.get("category") or "unknown"),
            category=str(item.get("category") or "unknown"),
            candidate_profile="exact_cache_replay",
            projected_savings_usd=_as_float(item.get("estimated_saved_cost_usd")),
            sample_count=_as_int(item.get("sample_count")),
            current_canary_count=applied,
            holdout_count=holdout,
            blocker_reason_codes=blockers,
            recommended_eval_mode=_eval_mode(projected=_as_int(item.get("miss_count")), applied=applied, holdout=holdout, blockers=blockers),
            replayability_level="local_exact_response",
            evidence={
                "rule_id": rule_id or None,
                "hit_count": applied,
                "miss_count": _as_int(item.get("miss_count")),
                "holdout_count": holdout,
                "error_rate": _as_float(item.get("error_rate")),
            },
        )
    return rows


def _old_context_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("quality_gates") or []:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        candidate_id = str(item.get("candidate_id") or "") or _stable_id(
            "eval-old-context",
            item.get("rule_id"),
            item.get("summary_model"),
            item.get("policy_source"),
        )
        blockers = [str(value) for value in item.get("reason_codes") or [] if value and value != "quality-gate-passed"]
        _add_common(
            rows,
            candidate_id=candidate_id,
            optimization_family="old_context_summarization",
            action_family="crunch",
            source_surface="anthropic_messages",
            app_family="claude_code",
            workflow_phase="old_context",
            category="old_context",
            candidate_target_model=str(item.get("summary_model") or ""),
            candidate_profile="old_context_summary",
            projected_savings_usd=_as_float(metrics.get("actual_net_savings_usd")),
            sample_count=_as_int(metrics.get("matched_metadata_row_count")),
            current_canary_count=_as_int(metrics.get("canary_applied_count")),
            holdout_count=_as_int(metrics.get("canary_holdout_count")),
            blocker_reason_codes=blockers,
            recommended_eval_mode=_eval_mode(
                projected=_as_int(metrics.get("matched_metadata_row_count")),
                applied=_as_int(metrics.get("canary_applied_count")),
                holdout=_as_int(metrics.get("canary_holdout_count")),
                blockers=blockers,
            ),
            replayability_level="features_only",
            evidence={
                "rule_id": item.get("rule_id"),
                "verdict": item.get("verdict"),
                "actual_tokens_saved_est": _as_int(metrics.get("actual_tokens_saved_est")),
                "summary_failure_count": _as_int(metrics.get("summary_failure_count")),
            },
        )
    return rows


def _pattern_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("cohorts") or []:
        if not isinstance(item, dict):
            continue
        section = str(item.get("policy_section") or "pattern")
        action_family = section if section in {"routing", "crunch", "cache"} else "pattern"
        candidate_id = str(item.get("candidate_id") or "") or _stable_id(
            "eval-pattern",
            section,
            item.get("source_surface"),
            item.get("app_family"),
            item.get("workflow_phase"),
            item.get("category"),
            item.get("pattern_family"),
        )
        blockers = _breakdown_values(item.get("local_bypass_reasons"))
        applied = _as_int(item.get("applied_count"))
        holdout = _as_int(item.get("holdout_count"))
        _add_common(
            rows,
            candidate_id=candidate_id,
            optimization_family="managed_pattern_candidate",
            action_family=action_family,
            source_surface=str(item.get("source_surface") or "unknown"),
            app_family=str(item.get("app_family") or "unknown"),
            workflow_phase=str(item.get("workflow_phase") or "unknown"),
            category=str(item.get("category") or "unknown"),
            candidate_profile=section,
            projected_savings_usd=_as_float(item.get("estimated_cost_savings_usd")),
            sample_count=_as_int(item.get("sample_count")),
            current_canary_count=applied,
            holdout_count=holdout,
            blocker_reason_codes=blockers,
            recommended_eval_mode=_eval_mode(projected=_as_int(item.get("sample_count")), applied=applied, holdout=holdout, blockers=blockers),
            replayability_level="features_only",
            evidence={
                "policy_section": section,
                "rule_id": item.get("rule_id"),
                "evidence_only": bool(item.get("evidence_only")),
                "minimum_sample_ready": bool((item.get("minimum_sample_readiness") or {}).get("ready")),
                "error_rate": _as_float(item.get("error_rate")),
            },
        )
    return rows


def _summarize(rows: list[dict[str, Any]], reports: dict[str, Any]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for row in rows:
        family_counts[str(row.get("optimization_family") or "unknown")] += 1
        action_counts[str(row.get("action_family") or "unknown")] += 1
        mode_counts[str(row.get("recommended_eval_mode") or "unknown")] += 1
        for blocker in row.get("blocker_reason_codes") or []:
            blocker_counts[str(blocker)] += 1
    return {
        "candidate_count": len(rows),
        "family_counts": _count_breakdown(family_counts),
        "action_family_counts": _count_breakdown(action_counts),
        "recommended_eval_mode_counts": _count_breakdown(mode_counts),
        "blocker_counts": _count_breakdown(blocker_counts),
        "projected_savings_usd": round(sum(_as_float(row.get("projected_savings_usd")) for row in rows), 8),
        "source_reports": [
            {"name": name, "schema": (report or {}).get("schema")}
            for name, report in sorted(reports.items())
        ],
    }


async def build_optimization_eval_plan(store: Any, *, limit: int = 500, min_samples: int = 1) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 10_000))
    sample_floor = max(1, int(min_samples or 1))
    phase = build_phase_routing_report(store, limit=capped_limit)
    cache_replayability, cache_confidence, old_context, pattern = await asyncio.gather(
        stats_cache_replayability(store, limit=capped_limit),
        stats_cache_replay_confidence(store, limit=capped_limit),
        stats_old_context_summary(store),
        stats_managed_pattern_rollups(store, limit=min(capped_limit, 5000), min_samples=sample_floor),
    )
    reports = {
        "cache_replay_confidence": cache_confidence,
        "cache_replayability": cache_replayability,
        "managed_pattern_rollups": pattern,
        "old_context_summarization": old_context,
        "phase_routing": phase,
    }
    rows = (
        _phase_rows(phase)
        + _cache_replayability_rows(cache_replayability)
        + _cache_confidence_rows(cache_confidence)
        + _old_context_rows(old_context)
        + _pattern_rows(pattern)
    )
    rows.sort(key=lambda row: (str(row.get("action_family")), str(row.get("optimization_family")), str(row.get("candidate_id"))))
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "min_samples": sample_floor,
        "summary": _summarize(rows, reports),
        "plans": rows,
        "privacy": _privacy_summary(),
    }
