from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization_eval_plan import build_optimization_eval_plan
from agentflow_proxy.public_metadata import public_id
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.optimization_promotion_report.v1"
VERDICT_SCHEMA = "agentflow.optimization_promotion_verdict.v1"

_REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


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


def _round(value: Any, places: int = 6) -> float:
    return round(_as_float(value), places)


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _reason_code(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if _REASON_CODE_RE.match(text):
        return text
    return "unsanitized-reason-code"


def _reason_codes(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    codes = {_reason_code(value) for value in values}
    return sorted(code for code in codes if code)


def _privacy_summary() -> dict[str, Any]:
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


def _count_rows(counts: Counter[str]) -> list[dict[str, Any]]:
    rows = [{"value": key, "count": count} for key, count in counts.items()]
    rows.sort(key=lambda row: (-_as_int(row["count"]), str(row["value"])))
    return rows


def _cohort_from_mapping(value: Any, *, fallback_count: int = 0) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    count = _as_int(value.get("count"), fallback_count)
    errors = _as_int(value.get("error_count"))
    retries = _as_int(value.get("retry_count"))
    safety_stops = _as_int(value.get("safety_stop_count"))
    fallbacks = _as_int(value.get("fallback_count"))
    latency = value.get("latency_avg_ms")
    savings = value.get("net_savings_usd", value.get("savings_usd"))
    return {
        "count": count,
        "error_count": errors,
        "retry_count": retries,
        "fallback_count": fallbacks,
        "safety_stop_count": safety_stops,
        "error_rate": _round(value.get("error_rate"), 6) if "error_rate" in value else (round(errors / count, 6) if count else 0.0),
        "retry_rate": _round(value.get("retry_rate"), 6) if "retry_rate" in value else (round(retries / count, 6) if count else 0.0),
        "safety_stop_rate": round(safety_stops / count, 6) if count else 0.0,
        "latency_avg_ms": None if latency is None else _round(latency, 2),
        "net_savings_usd": _round(savings, 8),
    }


def _merge_cohort_evidence(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(existing)
    existing_count = _as_int(result.get("count"))
    incoming_count = _as_int(incoming.get("count"))
    if incoming_count > existing_count:
        result.update(incoming)
    else:
        for key in ("error_count", "retry_count", "safety_stop_count"):
            result[key] = max(_as_int(result.get(key)), _as_int(incoming.get(key)))
        result["fallback_count"] = max(_as_int(result.get("fallback_count")), _as_int(incoming.get("fallback_count")))
        for key in ("error_rate", "retry_rate", "latency_avg_ms", "net_savings_usd", "savings_usd"):
            if result.get(key) in (None, "", 0, 0.0) and incoming.get(key) not in (None, ""):
                result[key] = incoming.get(key)
    return result


def _merge_extra_evidence(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if not existing:
        return dict(incoming)
    result = dict(existing)
    current = result.get("canary_evidence") if isinstance(result.get("canary_evidence"), dict) else {}
    new = incoming.get("canary_evidence") if isinstance(incoming.get("canary_evidence"), dict) else {}
    if new:
        merged = dict(current)
        for key in ("applied", "canary_applied", "holdout", "canary_holdout", "bypassed", "bypassed_or_disabled"):
            value = new.get(key)
            if not isinstance(value, dict):
                continue
            old = merged.get(key) if isinstance(merged.get(key), dict) else {}
            merged[key] = _merge_cohort_evidence(old, value)
        result["canary_evidence"] = merged
    for key, value in incoming.items():
        if key == "canary_evidence":
            continue
        if key == "__evidence_sources":
            sources = list(result.get("__evidence_sources") or [])
            sources.extend(item for item in value if isinstance(item, dict))
            result[key] = sources
        else:
            result.setdefault(key, value)
    return result


def _cohort_evidence(plan_row: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = plan_row.get("evidence") if isinstance(plan_row.get("evidence"), dict) else {}
    source = evidence.get("canary_evidence") if isinstance(evidence.get("canary_evidence"), dict) else {}
    if extra:
        source = {**source, **extra}

    applied = source.get("canary_applied", source.get("applied"))
    holdout = source.get("canary_holdout", source.get("holdout"))
    bypassed = source.get("bypassed_or_disabled", source.get("bypassed"))
    applied_count = _as_int(plan_row.get("current_canary_count"))
    holdout_count = _as_int(plan_row.get("holdout_count"))

    applied_row = _cohort_from_mapping(applied, fallback_count=applied_count)
    holdout_row = _cohort_from_mapping(holdout, fallback_count=holdout_count)
    bypassed_row = _cohort_from_mapping(bypassed)
    if not applied and "error_rate" in evidence:
        applied_row["error_rate"] = _round(evidence.get("error_rate"), 6)
    if "applied_error_rate" in source:
        applied_row["error_rate"] = _round(source.get("applied_error_rate"), 6)
    if "holdout_error_rate" in source:
        holdout_row["error_rate"] = _round(source.get("holdout_error_rate"), 6)
    if "applied_retry_rate" in source:
        applied_row["retry_rate"] = _round(source.get("applied_retry_rate"), 6)
    if "holdout_retry_rate" in source:
        holdout_row["retry_rate"] = _round(source.get("holdout_retry_rate"), 6)
    if "applied_latency_avg_ms" in source:
        applied_row["latency_avg_ms"] = _round(source.get("applied_latency_avg_ms"), 2)
    if "holdout_latency_avg_ms" in source:
        holdout_row["latency_avg_ms"] = _round(source.get("holdout_latency_avg_ms"), 2)
    if "safety_stop_count" in source:
        applied_row["safety_stop_count"] = _as_int(source.get("safety_stop_count"))
    if not applied_row["net_savings_usd"]:
        applied_row["net_savings_usd"] = _round(plan_row.get("projected_savings_usd"), 8)

    latency_delta = None
    if applied_row.get("latency_avg_ms") is not None and holdout_row.get("latency_avg_ms") is not None:
        latency_delta = round(_as_float(applied_row.get("latency_avg_ms")) - _as_float(holdout_row.get("latency_avg_ms")), 2)
    return {
        "cohorts": {
            "canary_applied": applied_row,
            "canary_holdout": holdout_row,
            "bypassed_or_disabled": bypassed_row,
        },
        "deltas": {
            "applied_minus_holdout_error_rate": round(_as_float(applied_row.get("error_rate")) - _as_float(holdout_row.get("error_rate")), 6),
            "applied_minus_holdout_retry_rate": round(_as_float(applied_row.get("retry_rate")) - _as_float(holdout_row.get("retry_rate")), 6),
            "applied_minus_holdout_latency_avg_ms": latency_delta,
        },
    }


def _thresholds(plan_row: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    evidence = plan_row.get("evidence") if isinstance(plan_row.get("evidence"), dict) else {}
    source = evidence.get("promotion_thresholds") if isinstance(evidence.get("promotion_thresholds"), dict) else {}
    result = {
        "min_eval_pass_count": _as_int(defaults.get("min_eval_pass_count"), 1),
        "min_canary_applied_samples": _as_int(defaults.get("min_canary_applied_samples"), 2),
        "min_canary_holdout_samples": _as_int(defaults.get("min_canary_holdout_samples"), 1),
        "max_error_rate": _round(defaults.get("max_error_rate", 0.05), 6),
        "max_error_rate_delta": _round(defaults.get("max_error_rate_delta", 0.05), 6),
        "max_retry_rate_delta": _round(defaults.get("max_retry_rate_delta", 0.10), 6),
        "max_latency_regression_ms": _as_int(defaults.get("max_latency_regression_ms"), 2_000),
        "rollback_error_rate": _round(defaults.get("rollback_error_rate", 0.40), 6),
        "max_evidence_age_hours": _as_int(defaults.get("max_evidence_age_hours"), 168),
    }
    for key in list(result):
        if key in source:
            result[key] = _as_int(source[key]) if key.startswith("min_") or key.endswith("_hours") or key.endswith("_ms") else _round(source[key], 6)
    return result


def _read_eval_results(store_obj: Any, *, limit: int) -> dict[str, list[dict[str, Any]]]:
    rows = store_obj.conn.execute(
        """
        select candidate_id, created_at, source_surface, optimization_family,
               action_family, status_class, reason_codes_json, score_json,
               cost_json, result_json
        from optimization_eval_results
        order by created_at desc
        limit ?
        """,
        (max(1, int(limit or 1000)),),
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        item["reason_codes"] = _reason_codes(_json_list(item.get("reason_codes_json")))
        item["score"] = _json_obj(item.get("score_json"))
        item["cost"] = _json_obj(item.get("cost_json"))
        item["result"] = _json_obj(item.get("result_json"))
        grouped.setdefault(str(item.get("candidate_id") or "unknown"), []).append(item)
    return grouped


def _eval_evidence(results: list[dict[str, Any]], *, now: datetime, thresholds: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    similarities: list[float] = []
    qualities: list[float] = []
    latest: datetime | None = None
    for result in results:
        status = str(result.get("status_class") or "unknown")
        counts[status] += 1
        for reason in result.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        score = result.get("score") if isinstance(result.get("score"), dict) else {}
        if score.get("output_similarity") is not None:
            similarities.append(_as_float(score.get("output_similarity")))
        if score.get("quality_score") is not None:
            qualities.append(_as_float(score.get("quality_score")))
        created_at = _parse_time(result.get("created_at"))
        if created_at and (latest is None or created_at > latest):
            latest = created_at

    stale = False
    if latest is not None and _as_int(thresholds.get("max_evidence_age_hours")) > 0:
        max_age_seconds = _as_int(thresholds.get("max_evidence_age_hours")) * 3600
        stale = (now - latest).total_seconds() > max_age_seconds

    return {
        "result_count": sum(counts.values()),
        "status_counts": _count_rows(counts),
        "pass_count": counts.get("pass", 0),
        "fail_count": counts.get("fail", 0),
        "blocked_count": counts.get("blocked", 0),
        "queued_count": counts.get("queued", 0),
        "unknown_count": counts.get("unknown", 0),
        "top_reason_codes": _count_rows(reason_counts),
        "latest_result_at": latest.isoformat() if latest else None,
        "stale": stale,
        "score_summary": {
            "avg_output_similarity": round(sum(similarities) / len(similarities), 6) if similarities else None,
            "avg_quality_score": round(sum(qualities) / len(qualities), 6) if qualities else None,
        },
    }


def _old_context_quality_gate(plan_row: dict[str, Any]) -> tuple[str | None, list[str]]:
    evidence = plan_row.get("evidence") if isinstance(plan_row.get("evidence"), dict) else {}
    verdict = str(evidence.get("verdict") or "")
    if verdict == "widen":
        return "widen", _reason_codes(evidence.get("quality_gate_reason_codes")) or ["external-quality-gate-passed"]
    if verdict == "promote":
        return "widen", ["old-context-quality-gate-passed"]
    if verdict == "rollback":
        return "rollback", _reason_codes(evidence.get("quality_gate_reason_codes")) or ["old-context-quality-gate-rollback"]
    if verdict == "hold":
        return "hold", _reason_codes(evidence.get("quality_gate_reason_codes")) or ["old-context-quality-gate-hold"]
    if verdict in {"insufficient-evidence", "needs_eval"}:
        return "needs_eval", _reason_codes(evidence.get("quality_gate_reason_codes")) or ["old-context-quality-gate-insufficient-evidence"]
    return None, []


def _active_plan_blockers(
    plan_blockers: list[str],
    *,
    evals: dict[str, Any],
    applied: dict[str, Any],
    holdout: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    active: list[str] = []
    for blocker in plan_blockers:
        if blocker == "insufficient-canary-applied-samples" and _as_int(applied.get("count")) >= _as_int(thresholds["min_canary_applied_samples"]):
            continue
        if blocker == "insufficient-canary-holdout-samples" and _as_int(holdout.get("count")) >= _as_int(thresholds["min_canary_holdout_samples"]):
            continue
        if blocker == "insufficient-eval-pass-results" and _as_int(evals.get("pass_count")) >= _as_int(thresholds["min_eval_pass_count"]):
            continue
        if blocker == "eval-results-missing" and (_as_int(evals.get("result_count")) > 0 or _as_int(evals.get("queued_count")) > 0):
            continue
        active.append(blocker)
    return active


def _decide_verdict(
    plan_row: dict[str, Any],
    *,
    evals: dict[str, Any],
    cohorts: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    plan_blockers = _reason_codes(plan_row.get("blocker_reason_codes"))
    applied = cohorts["cohorts"]["canary_applied"]
    holdout = cohorts["cohorts"]["canary_holdout"]
    bypassed = cohorts["cohorts"]["bypassed_or_disabled"]
    deltas = cohorts["deltas"]
    active_plan_blockers = _active_plan_blockers(
        plan_blockers,
        evals=evals,
        applied=applied,
        holdout=holdout,
        thresholds=thresholds,
    )
    reasons.extend(active_plan_blockers)

    gate_verdict, gate_reasons = _old_context_quality_gate(plan_row)
    reasons.extend(gate_reasons)

    if evals["fail_count"]:
        reasons.append("eval-failed")
        return "rollback", sorted(set(reasons)), "rollback_or_disable_candidate"
    if evals["stale"]:
        reasons.append("stale-eval-evidence")

    safety_stops = _as_int(applied.get("safety_stop_count")) + _as_int(bypassed.get("safety_stop_count"))
    if safety_stops:
        reasons.append("safety-stop-observed")
        return "rollback", sorted(set(reasons)), "rollback_or_disable_candidate"
    if _as_float(applied.get("error_rate")) >= _as_float(thresholds["rollback_error_rate"]):
        reasons.append("rollback-error-rate")
        return "rollback", sorted(set(reasons)), "rollback_or_disable_candidate"
    if gate_verdict == "rollback":
        return "rollback", sorted(set(reasons)), "rollback_or_disable_candidate"

    if _as_int(applied.get("count")) < _as_int(thresholds["min_canary_applied_samples"]):
        reasons.append("insufficient-canary-applied-samples")
    if _as_int(holdout.get("count")) < _as_int(thresholds["min_canary_holdout_samples"]):
        reasons.append("insufficient-canary-holdout-samples")
    if evals["pass_count"] < _as_int(thresholds["min_eval_pass_count"]):
        reasons.append("insufficient-eval-pass-results")
        if _as_int(evals.get("queued_count")) > 0:
            reasons.append("eval-queued")
    if evals["result_count"] == 0 and _as_int(evals.get("queued_count")) == 0:
        reasons.append("eval-results-missing")
    if active_plan_blockers:
        reasons.append("plan-blockers-present")

    insufficient = any(code.startswith("insufficient-") for code in reasons) or "eval-results-missing" in reasons
    if gate_verdict == "needs_eval" or insufficient:
        return "needs_eval", sorted(set(reasons)), "run_local_shadow_eval_or_collect_canary_holdout_evidence"

    hold_reasons: list[str] = []
    if _as_float(applied.get("error_rate")) > _as_float(thresholds["max_error_rate"]):
        hold_reasons.append("applied-error-rate-above-threshold")
    if _as_float(deltas.get("applied_minus_holdout_error_rate")) > _as_float(thresholds["max_error_rate_delta"]):
        hold_reasons.append("applied-error-rate-regression")
    if _as_float(deltas.get("applied_minus_holdout_retry_rate")) > _as_float(thresholds["max_retry_rate_delta"]):
        hold_reasons.append("applied-retry-rate-regression")
    latency_delta = deltas.get("applied_minus_holdout_latency_avg_ms")
    if latency_delta is not None and _as_float(latency_delta) > _as_float(thresholds["max_latency_regression_ms"]):
        hold_reasons.append("latency-regression-above-threshold")
    if _as_float(applied.get("net_savings_usd")) <= 0.0:
        hold_reasons.append("non-positive-net-savings")
    if gate_verdict == "hold":
        hold_reasons.append("old-context-quality-gate-hold")
    if hold_reasons:
        reasons.extend(hold_reasons)
        return "hold", sorted(set(reasons)), "keep_current_canary_fraction"

    reasons.extend(["eval-pass-threshold-met", "canary-holdout-thresholds-met", "promotion-thresholds-met"])
    return "widen", sorted(set(reasons)), "widen_local_canary"


def _extra_report_evidence(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}

    def add(candidate_id: Any, extra: dict[str, Any]) -> None:
        if not candidate_id:
            return
        key = str(candidate_id)
        by_candidate[key] = _merge_extra_evidence(by_candidate.get(key), extra)

    for report in reports:
        if not isinstance(report, dict):
            continue
        for candidate_id, extra in _activation_lifecycle_report_evidence(report).items():
            add(candidate_id, extra)
        quality_gate = report.get("quality_gate") if isinstance(report.get("quality_gate"), dict) else {}
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
        candidate_id = policy.get("candidate_id") or quality_gate.get("candidate_id") or summary.get("candidate_id")
        if candidate_id and quality_gate:
            add(candidate_id, {
                "quality_gate_verdict": quality_gate.get("verdict"),
                "quality_gate_reason_codes": _reason_codes(quality_gate.get("reason_codes")),
            })
        for action in report.get("actions") or []:
            if not isinstance(action, dict):
                continue
            candidate = action.get("candidate") if isinstance(action.get("candidate"), dict) else {}
            candidate_id = action.get("candidate_id") or candidate.get("candidate_id")
            if not candidate_id:
                continue
            actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
            add(candidate_id, {
                "canary_evidence": {
                    "applied": {"count": actual.get("actual_canary_applied_count")},
                    "holdout": {"count": actual.get("actual_canary_holdout_count")},
                    "bypassed": {"count": actual.get("actual_bypassed_or_disabled_count")},
                }
            })
        for candidate in report.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("candidate_id") or candidate.get("target_candidate_id")
            if not candidate_id:
                continue
            cohorts = candidate.get("cohort_metrics") if isinstance(candidate.get("cohort_metrics"), dict) else {}
            applied = cohorts.get("canary_applied") if isinstance(cohorts.get("canary_applied"), dict) else {}
            holdout = cohorts.get("canary_holdout") if isinstance(cohorts.get("canary_holdout"), dict) else {}
            bypassed = cohorts.get("bypassed_or_disabled") if isinstance(cohorts.get("bypassed_or_disabled"), dict) else {}
            safety = cohorts.get("safety_stopped") if isinstance(cohorts.get("safety_stopped"), dict) else {}
            extra = {
                "quality_gate_verdict": candidate.get("verdict"),
                "quality_gate_reason_codes": _reason_codes(candidate.get("reason_codes")),
                "canary_evidence": {
                    "applied": {
                        "count": applied.get("count"),
                        "error_count": applied.get("error_count"),
                        "retry_count": applied.get("retry_count"),
                        "error_rate": applied.get("error_rate"),
                        "retry_rate": applied.get("retry_rate"),
                        "latency_avg_ms": applied.get("latency_avg_ms"),
                        "net_savings_usd": applied.get("observed_savings_usd", candidate.get("observed_savings_usd")),
                    },
                    "holdout": {
                        "count": holdout.get("count"),
                        "error_count": holdout.get("error_count"),
                        "retry_count": holdout.get("retry_count"),
                        "error_rate": holdout.get("error_rate"),
                        "retry_rate": holdout.get("retry_rate"),
                        "latency_avg_ms": holdout.get("latency_avg_ms"),
                        "net_savings_usd": holdout.get("observed_savings_usd"),
                    },
                    "bypassed": {
                        "count": _as_int(bypassed.get("count")) + _as_int(safety.get("count")),
                        "error_count": _as_int(bypassed.get("error_count")) + _as_int(safety.get("error_count")),
                        "retry_count": _as_int(bypassed.get("retry_count")) + _as_int(safety.get("retry_count")),
                        "safety_stop_count": safety.get("count"),
                    },
                },
            }
            add(candidate_id, extra)
    return by_candidate


def _public_ref(value: Any, *, prefix: str = "policy") -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _promotion_action_family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"cache", "cache_replay"}:
        return "cache"
    if text in {"crunch", "old_context_summary", "old_context_summarization", "repeated_scaffold_crunch", "terminal_output_compaction", "anthropic_thinking_history_compaction"}:
        return "crunch"
    if text == "routing":
        return "routing"
    return text or "unknown"


def _activation_lifecycle_report_evidence(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lifecycle = report
    if report.get("schema") != "agentflow.activation_staged_lifecycle_feedback_summary.v1":
        nested = report.get("activation_lifecycle_feedback")
        lifecycle = nested if isinstance(nested, dict) else {}
    if lifecycle.get("schema") != "agentflow.activation_staged_lifecycle_feedback_summary.v1":
        return {}

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    rows = lifecycle.get("cohort_lifecycle_metadata") if isinstance(lifecycle.get("cohort_lifecycle_metadata"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = _promotion_action_family(row.get("action_family"))
        cohort = str(row.get("cohort_label") or "").strip().lower().replace("-", "_")
        keys = {
            str(value)
            for value in (row.get("candidate_id"), row.get("policy_ref"))
            if value not in (None, "")
        }
        if not keys:
            continue
        cohort_key = "applied" if cohort in {"applied", "canary_applied"} else "holdout" if cohort in {"holdout", "canary_holdout"} else "bypassed"
        evidence = {
            "count": _as_int(row.get("applied_count") if cohort_key == "applied" else row.get("holdout_count") if cohort_key == "holdout" else row.get("event_count")),
            "error_count": _as_int(row.get("error_count")),
            "retry_count": _as_int(row.get("retry_count")),
            "safety_stop_count": _as_int(row.get("safety_stop_count")),
            "fallback_count": _as_int(row.get("fallback_count")),
            "error_rate": _round(row.get("error_rate"), 6),
            "net_savings_usd": _round(row.get("savings_estimate_usd"), 8),
        }
        if evidence["count"] <= 0:
            evidence["count"] = _as_int(row.get("event_count"))
        for key in keys:
            group = grouped.setdefault((key, family), {
                "canary_evidence": {},
                "__evidence_sources": [{
                    "source": "activation_lifecycle_feedback",
                    "schema": lifecycle.get("schema"),
                    "action_family": family,
                    "queue_rows": _as_int(lifecycle.get("queue_rows")),
                    "family_event_count": _as_int(lifecycle.get("family_event_count")),
                }],
            })
            canary = group["canary_evidence"]
            current = canary.get(cohort_key) if isinstance(canary.get(cohort_key), dict) else {}
            canary[cohort_key] = _merge_cohort_evidence(current, evidence)

    result: dict[str, dict[str, Any]] = {}
    for (key, _family), extra in grouped.items():
        result[key] = _merge_extra_evidence(result.get(key), extra)
    return result


def _candidate_extra_keys(plan_row: dict[str, Any], candidate_id: str) -> list[str]:
    evidence = plan_row.get("evidence") if isinstance(plan_row.get("evidence"), dict) else {}
    values = [
        candidate_id,
        plan_row.get("candidate_id"),
        plan_row.get("target_candidate_id"),
        plan_row.get("rule_id"),
        plan_row.get("policy_id"),
        evidence.get("candidate_id"),
        evidence.get("target_candidate_id"),
        evidence.get("rule_id"),
        evidence.get("policy_id"),
    ]
    keys: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value)
        keys.append(text)
        safe_candidate = public_id(text, prefix="candidate")
        if safe_candidate:
            keys.append(safe_candidate)
            ref = _public_ref(safe_candidate)
            if ref:
                keys.append(ref)
        ref = _public_ref(text)
        if ref:
            keys.append(ref)
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _candidate_extra(extras: dict[str, dict[str, Any]], plan_row: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    merged: dict[str, Any] | None = None
    for key in _candidate_extra_keys(plan_row, candidate_id):
        if key in extras:
            merged = _merge_extra_evidence(merged, extras[key])
    return merged


def _candidate_row(
    plan_row: dict[str, Any],
    *,
    eval_results: list[dict[str, Any]],
    now: datetime,
    defaults: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_id = str(plan_row.get("candidate_id") or _stable_id("promotion-candidate", plan_row))
    if extra and extra.get("quality_gate_verdict"):
        plan_row = dict(plan_row)
        evidence = dict(plan_row.get("evidence") if isinstance(plan_row.get("evidence"), dict) else {})
        evidence.setdefault("verdict", extra.get("quality_gate_verdict"))
        if extra.get("quality_gate_reason_codes"):
            evidence.setdefault("quality_gate_reason_codes", extra.get("quality_gate_reason_codes"))
        plan_row["evidence"] = evidence
    thresholds = _thresholds(plan_row, defaults)
    evals = _eval_evidence(eval_results, now=now, thresholds=thresholds)
    extra_canary = extra.get("canary_evidence") if isinstance(extra, dict) and isinstance(extra.get("canary_evidence"), dict) else None
    cohorts = _cohort_evidence(plan_row, extra=extra_canary)
    verdict, reasons, next_action = _decide_verdict(plan_row, evals=evals, cohorts=cohorts, thresholds=thresholds)
    result = {
        "schema": VERDICT_SCHEMA,
        "candidate_id": candidate_id,
        "optimization_family": str(plan_row.get("optimization_family") or "unknown"),
        "action_family": str(plan_row.get("action_family") or "unknown"),
        "source_surface": str(plan_row.get("source_surface") or "unknown"),
        "app_family": str(plan_row.get("app_family") or "unknown"),
        "candidate_target_model": plan_row.get("candidate_target_model"),
        "candidate_profile": plan_row.get("candidate_profile"),
        "projected_savings_usd": _round(plan_row.get("projected_savings_usd"), 8),
        "sample_count": _as_int(plan_row.get("sample_count")),
        "cohort_counts": {
            "canary_applied": cohorts["cohorts"]["canary_applied"]["count"],
            "canary_holdout": cohorts["cohorts"]["canary_holdout"]["count"],
            "bypassed_or_disabled": cohorts["cohorts"]["bypassed_or_disabled"]["count"],
        },
        "cohort_metrics": cohorts["cohorts"],
        "applied_vs_holdout_deltas": cohorts["deltas"],
        "eval_evidence": evals,
        "thresholds": thresholds,
        "verdict": verdict,
        "reason_codes": reasons,
        "next_action": next_action,
        "privacy": _privacy_summary(),
    }
    sources = extra.get("__evidence_sources") if isinstance(extra, dict) and isinstance(extra.get("__evidence_sources"), list) else []
    if sources:
        result["evidence_sources"] = sources
    return result


def build_optimization_promotion_report(
    store_obj: Any,
    *,
    plan: dict[str, Any] | None = None,
    evidence_reports: list[dict[str, Any]] | None = None,
    limit: int = 500,
    min_samples: int = 1,
    min_eval_pass_count: int = 1,
    min_canary_applied_samples: int = 2,
    min_canary_holdout_samples: int = 1,
    max_evidence_age_hours: int = 168,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 10_000))
    if plan is None:
        plan = asyncio.run(build_optimization_eval_plan(store_obj, limit=capped_limit, min_samples=min_samples))
    rows = plan.get("plans") if isinstance(plan, dict) else []
    if not isinstance(rows, list):
        rows = []
    now = datetime.now(timezone.utc)
    evals_by_candidate = _read_eval_results(store_obj, limit=capped_limit * 5)
    reports = list(evidence_reports or [])
    try:
        from agentflow_proxy.activation_lifecycle_feedback import activation_lifecycle_feedback_summary

        reports.append(activation_lifecycle_feedback_summary(store_obj, limit=capped_limit * 20))
    except Exception:
        pass
    extras = _extra_report_evidence(reports)
    defaults = {
        "min_eval_pass_count": min_eval_pass_count,
        "min_canary_applied_samples": min_canary_applied_samples,
        "min_canary_holdout_samples": min_canary_holdout_samples,
        "max_evidence_age_hours": max_evidence_age_hours,
    }
    candidates: list[dict[str, Any]] = []
    for row in rows[:capped_limit]:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or _stable_id("promotion-candidate", row))
        candidates.append(
            _candidate_row(
                row,
                eval_results=evals_by_candidate.get(candidate_id, []),
                now=now,
                defaults=defaults,
                extra=_candidate_extra(extras, row, candidate_id),
            )
        )
    candidates.sort(key=lambda item: (str(item.get("action_family")), str(item.get("optimization_family")), str(item.get("candidate_id"))))
    verdict_counts: Counter[str] = Counter(str(item.get("verdict") or "unknown") for item in candidates)
    action_counts: Counter[str] = Counter(str(item.get("action_family") or "unknown") for item in candidates)
    reason_counts: Counter[str] = Counter()
    evidence_source_counts: Counter[str] = Counter()
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        for source in item.get("evidence_sources") or []:
            if isinstance(source, dict):
                evidence_source_counts[str(source.get("source") or "unknown")] += 1
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "limit": capped_limit,
        "summary": {
            "candidate_count": len(candidates),
            "verdict_counts": _count_rows(verdict_counts),
            "action_family_counts": _count_rows(action_counts),
            "reason_code_counts": _count_rows(reason_counts),
            "evidence_source_counts": _count_rows(evidence_source_counts),
            "projected_savings_usd": _round(sum(_as_float(item.get("projected_savings_usd")) for item in candidates), 8),
        },
        "candidates": candidates,
        "privacy": _privacy_summary(),
    }
