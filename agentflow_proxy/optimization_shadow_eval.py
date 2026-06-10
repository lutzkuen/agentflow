from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentflow_proxy.store import stable_json, utc_now

SCHEMA = "agentflow.optimization_shadow_eval.v1"
RESULT_SCHEMA = "agentflow.optimization_shadow_eval_result.v1"

_FEATURE_ONLY_REPLAY_LEVELS = {
    "features_only",
    "metadata_only",
    "turn-metadata-only",
    "turn_metadata_only",
}
_LOCAL_REPLAY_LEVELS = {
    "raw_body_opt_in",
    "local-exact-response",
    "local_exact_response",
    "local-provider-request",
    "local_provider_request",
    "static_information",
}
_RAW_FIELD_NAMES = {
    "api_key",
    "cache_key",
    "content",
    "file_path",
    "messages",
    "output",
    "params",
    "path",
    "prompt",
    "provider_body",
    "raw_body",
    "request",
    "request_id",
    "request_json",
    "response",
    "response_json",
    "session_id",
    "thread_id",
    "tool_payload",
    "transcript",
}


def _privacy_summary() -> dict[str, Any]:
    return {
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
    }


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_class(status_code: int | None) -> str | None:
    if status_code is None:
        return None
    if 200 <= status_code < 300:
        return "2xx"
    if 300 <= status_code < 400:
        return "3xx"
    if 400 <= status_code < 500:
        return "4xx"
    if status_code >= 500:
        return "5xx"
    return "unknown"


def _sanitize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if value})


def _clean_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _RAW_FIELD_NAMES:
                continue
            cleaned_item = _clean_metadata(item, depth=depth + 1)
            if cleaned_item is not None:
                cleaned[key_text] = cleaned_item
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value[:50]:
            cleaned_item = _clean_metadata(item, depth=depth + 1)
            if cleaned_item is not None:
                cleaned_list.append(cleaned_item)
        return cleaned_list
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fixture(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    for key in ("offline_eval", "shadow_eval_fixture", "fixture_result"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        value = evidence.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _estimated_eval_cost(row: dict[str, Any], fixture: dict[str, Any]) -> float:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    for source in (fixture, row, evidence):
        for key in ("estimated_eval_cost_usd", "candidate_cost_usd", "shadow_cost_est_usd"):
            if key in source:
                return max(0.0, _as_float(source.get(key)))
    baseline = _as_float(fixture.get("baseline_cost_usd"))
    candidate = _as_float(fixture.get("candidate_cost_usd"))
    if baseline or candidate:
        return max(0.0, baseline + candidate)
    return 0.0


def _score_fixture(
    row: dict[str, Any],
    fixture: dict[str, Any],
    *,
    min_output_similarity: float,
) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
    baseline_status = _as_int(fixture.get("baseline_status_code"))
    candidate_status = _as_int(fixture.get("candidate_status_code"))
    output_similarity = fixture.get("output_similarity")
    output_similarity_value = None if output_similarity is None else _as_float(output_similarity)
    quality_score = fixture.get("quality_score")
    quality_score_value = None if quality_score is None else _as_float(quality_score)
    baseline_cost = _as_float(fixture.get("baseline_cost_usd"))
    candidate_cost = _as_float(fixture.get("candidate_cost_usd"))
    baseline_latency = _as_int(fixture.get("baseline_latency_ms"))
    candidate_latency = _as_int(fixture.get("candidate_latency_ms"))

    reasons: list[str] = []
    if baseline_status is not None and not (200 <= baseline_status < 300):
        reasons.append("baseline-non-success")
    if candidate_status is not None and not (200 <= candidate_status < 300):
        reasons.append("candidate-non-success")
    if output_similarity_value is not None and output_similarity_value < min_output_similarity:
        reasons.append("output-similarity-below-threshold")
    if quality_score_value is not None and quality_score_value < min_output_similarity:
        reasons.append("quality-score-below-threshold")

    decisive_signal = (
        baseline_status is not None
        or candidate_status is not None
        or output_similarity_value is not None
        or quality_score_value is not None
    )
    if reasons:
        status = "fail"
    elif decisive_signal:
        status = "pass"
        reasons.append("offline-fixture-passed")
    else:
        status = "unknown"
        reasons.append("offline-fixture-missing-decisive-signals")

    score = {
        "baseline_status_class": _status_class(baseline_status),
        "candidate_status_class": _status_class(candidate_status),
        "output_similarity": output_similarity_value,
        "quality_score": quality_score_value,
        "min_output_similarity": min_output_similarity,
        "latency_delta_ms": (candidate_latency - baseline_latency) if candidate_latency is not None and baseline_latency is not None else None,
    }
    score = {key: value for key, value in score.items() if value is not None}

    cost = {
        "baseline_cost_usd": round(max(0.0, baseline_cost), 8),
        "candidate_cost_usd": round(max(0.0, candidate_cost), 8),
        "estimated_eval_cost_usd": round(_estimated_eval_cost(row, fixture), 8),
        "projected_savings_usd": round(max(0.0, _as_float(row.get("projected_savings_usd"))), 8),
    }
    if baseline_cost or candidate_cost:
        cost["observed_savings_usd"] = round(max(0.0, baseline_cost - candidate_cost), 8)
    return status, reasons, score, cost


def _evaluate_row(
    row: dict[str, Any],
    *,
    execute: bool,
    budget_remaining_usd: float,
    min_output_similarity: float,
) -> tuple[dict[str, Any], float]:
    candidate_id = str(row.get("candidate_id") or _stable_id("shadow-eval-candidate", row))
    fixture = _fixture(row)
    estimated_cost = _estimated_eval_cost(row, fixture)
    plan_blockers = _sanitize_list(row.get("blocker_reason_codes"))
    replayability_level = str(row.get("replayability_level") or "metadata_only")
    granularity = str(row.get("granularity") or "provider_request")
    reasons: list[str] = []
    status = "unknown"
    score: dict[str, Any] = {}
    cost: dict[str, Any] = {
        "estimated_eval_cost_usd": round(estimated_cost, 8),
        "budget_remaining_before_usd": round(max(0.0, budget_remaining_usd), 8),
    }
    provider_call_made = False
    consumed = 0.0

    if granularity != "provider_request":
        status = "blocked"
        reasons.append("not-provider-request")
    elif plan_blockers:
        status = "blocked"
        reasons.extend(plan_blockers)
    elif replayability_level in _FEATURE_ONLY_REPLAY_LEVELS:
        status = "blocked"
        reasons.append("local-replay-input-unavailable")
    elif execute and estimated_cost > budget_remaining_usd:
        status = "blocked"
        reasons.append("budget-cap-exceeded")
    elif fixture:
        status, reasons, score, fixture_cost = _score_fixture(
            row,
            fixture,
            min_output_similarity=min_output_similarity,
        )
        cost.update(fixture_cost)
    elif not execute:
        status = "unknown"
        reasons.append("execute-not-requested")
    elif replayability_level not in _LOCAL_REPLAY_LEVELS:
        status = "blocked"
        reasons.append("unsupported-replayability-level")
    else:
        status = "blocked"
        reasons.append("raw-local-replay-input-not-supplied")

    if provider_call_made:
        consumed = estimated_cost

    result = {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate_id,
        "optimization_family": str(row.get("optimization_family") or "unknown"),
        "action_family": str(row.get("action_family") or "unknown"),
        "source_surface": str(row.get("source_surface") or "unknown"),
        "app_family": str(row.get("app_family") or "unknown"),
        "granularity": granularity,
        "status_class": status,
        "reason_codes": sorted(set(reasons)),
        "replayability_level": replayability_level,
        "recommended_eval_mode": row.get("recommended_eval_mode"),
        "provider_call_made": provider_call_made,
        "score_summary": _clean_metadata(score),
        "cost_summary": _clean_metadata(cost),
        "privacy": _privacy_summary(),
    }
    return result, consumed


def run_optimization_shadow_eval(
    plan: dict[str, Any],
    *,
    store: Any | None = None,
    execute: bool = False,
    budget_usd: float = 0.0,
    min_output_similarity: float = 0.9,
    max_candidates: int = 100,
    results_jsonl_path: str | None = None,
) -> dict[str, Any]:
    rows = plan.get("plans") if isinstance(plan, dict) else []
    if not isinstance(rows, list):
        rows = []
    capped = max(1, min(int(max_candidates or 100), 1000))
    run_id = _stable_id("shadow-eval-run", utc_now(), len(rows), execute, budget_usd)
    budget_remaining = max(0.0, float(budget_usd or 0.0))
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    provider_calls = 0

    for row in rows[:capped]:
        if not isinstance(row, dict):
            continue
        result, consumed = _evaluate_row(
            row,
            execute=execute,
            budget_remaining_usd=budget_remaining,
            min_output_similarity=min_output_similarity,
        )
        result["run_id"] = run_id
        result["created_at"] = utc_now()
        budget_remaining = max(0.0, budget_remaining - consumed)
        provider_calls += 1 if result.get("provider_call_made") else 0
        status_counts[str(result.get("status_class") or "unknown")] += 1
        results.append(result)
        if store is not None:
            store.log_optimization_eval_result(
                id=_stable_id("shadow-eval-result", run_id, result.get("candidate_id")),
                run_id=run_id,
                created_at=result["created_at"],
                candidate_id=result["candidate_id"],
                source_surface=result["source_surface"],
                optimization_family=result["optimization_family"],
                action_family=result["action_family"],
                status_class=result["status_class"],
                reason_codes_json=stable_json(result["reason_codes"]),
                score_json=stable_json(result["score_summary"]),
                cost_json=stable_json(result["cost_summary"]),
                result_json=stable_json(result),
            )

    if results_jsonl_path:
        path = Path(results_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {
        "candidate_count": len(results),
        "status_counts": [{"value": key, "count": count} for key, count in sorted(status_counts.items())],
        "provider_call_count": provider_calls,
        "budget_usd": round(max(0.0, float(budget_usd or 0.0)), 8),
        "budget_remaining_usd": round(budget_remaining, 8),
        "budget_exhausted_count": sum(1 for result in results if "budget-cap-exceeded" in (result.get("reason_codes") or [])),
    }
    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "generated_at": utc_now(),
        "mode": "execute" if execute else "plan-only",
        "provider_calls_made": provider_calls > 0,
        "managed_server_calls_made": False,
        "wrote_local_policy_files": False,
        "wrote_result_records": bool(results),
        "result_record_store": "sqlite" if store is not None and getattr(store, "backend", "") == "sqlite" else ("postgres" if store is not None else "none"),
        "results_jsonl_path": str(results_jsonl_path) if results_jsonl_path else None,
        "summary": summary,
        "results": results,
        "privacy": _privacy_summary(),
    }
