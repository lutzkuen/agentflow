from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.router import OPENAI_LARGE_DEFAULT, OPENAI_SMALL_DEFAULT, OPENAI_TINY_DEFAULT
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_routing_opportunity.v1"
PROMOTION_DECISION_REPORT_SCHEMA = "agentflow.openai_routing_promotion_decision_report.v1"
PROMOTION_DECISION_SCHEMA = "agentflow.openai_routing_promotion_decision.v1"
ACTIVE_LOCAL_POLICY_OUTCOME_SCHEMA = "agentflow.openai_routing_active_local_policy_outcome.v1"
ACTIVE_LOCAL_POLICY_OUTCOME_GATE_SCHEMA = "agentflow.openai_routing_active_local_policy_outcome_gate.v1"
ACTIVE_LOCAL_POLICY_SAVINGS_DELTAS_SCHEMA = "agentflow.openai_routing_active_local_policy_savings_deltas.v1"
PROMOTION_VERDICT_OPTIONS = ["promotion-ready", "active-local-policy", "keep-staged", "keep-blocked", "rollback-required"]
DEFAULT_MIN_SAMPLES = 5
DEFAULT_MAX_ERROR_RATE = 0.05
DEFAULT_MAX_RETRY_RATE = 0.20
DEFAULT_MAX_EVIDENCE_AGE_HOURS = 72.0
DEFAULT_SMALL_TEXT_CHARS_LT = 6000
DEFAULT_TINY_TEXT_CHARS_LT = 1500
DEFAULT_OPENAI_GPT54_CANARY_TEXT_CHARS_LT = 16000
DEFAULT_MIN_HOLDOUT_VOLUME = 10


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


OPENAI_SMALL_TEXT_CHARS_LT = _env_int("AGENTFLOW_OPENAI_SMALL_TEXT_CHARS_LT", DEFAULT_SMALL_TEXT_CHARS_LT)
OPENAI_TINY_TEXT_CHARS_LT = _env_int("AGENTFLOW_OPENAI_TINY_TEXT_CHARS_LT", DEFAULT_TINY_TEXT_CHARS_LT)
OPENAI_GPT54_CANARY_TEXT_CHARS_LT = _env_int(
    "AGENTFLOW_OPENAI_GPT54_CANARY_TEXT_CHARS_LT",
    DEFAULT_OPENAI_GPT54_CANARY_TEXT_CHARS_LT,
)


def _json_obj(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        import json

        parsed = json.loads(value)
    except (TypeError, ValueError):
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


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = str(key or "unknown")
    counter[text] = counter.get(text, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _text_bucket(text_chars: int) -> str:
    if text_chars < 1500:
        return "lt-1_5k"
    if text_chars < 6000:
        return "1_5k-6k"
    if text_chars < 32000:
        return "6k-32k"
    return "gte-32k"


def _token_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "unknown"
    if tokens < 1000:
        return "lt-1k"
    if tokens < 4000:
        return "1k-4k"
    if tokens < 16000:
        return "4k-16k"
    return "gte-16k"


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any]) -> bool:
    if "has_tools" in routing:
        return bool(routing.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    if category.startswith("tool-"):
        return True
    reason = str(cache.get("reason") or "").lower()
    return "tool" in reason


def _text_chars(row: dict[str, Any], routing: dict[str, Any]) -> int:
    chars = _as_int(routing.get("text_chars"))
    if chars > 0:
        return chars
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * 4)


def _input_tokens(row: dict[str, Any], text_chars: int) -> int:
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    if tokens > 0:
        return tokens
    return max(0, text_chars // 4)


def _output_tokens(row: dict[str, Any]) -> int:
    return _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))


def _source_surface(row: dict[str, Any]) -> str:
    return str(row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any]) -> str:
    return str(row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _simulate_openai_route(
    *,
    requested_model: str,
    category: str,
    has_tools: bool,
    text_chars: int,
) -> tuple[str | None, str, str]:
    requested_l = requested_model.lower()
    if requested_l == "gpt-5.4" and text_chars < OPENAI_SMALL_TEXT_CHARS_LT and not has_tools:
        return "gpt-5.4-mini", "proposed-canary-default-off", "gpt-5.4-large-to-mini-short-non-tool"
    if requested_l == "gpt-5.4" and category == "tool-light" and text_chars < OPENAI_GPT54_CANARY_TEXT_CHARS_LT:
        return "gpt-5.4-mini", "local-openai-routing-canary", "gpt-5.4-large-to-mini-tool-light-canary"
    if requested_l == OPENAI_LARGE_DEFAULT.lower() and text_chars < OPENAI_SMALL_TEXT_CHARS_LT and not has_tools:
        return OPENAI_SMALL_DEFAULT, "existing-threshold", "large-to-small-short-non-tool"
    if requested_l == OPENAI_SMALL_DEFAULT.lower() and text_chars < OPENAI_TINY_TEXT_CHARS_LT and not has_tools:
        return OPENAI_TINY_DEFAULT, "existing-threshold", "small-to-tiny-short-non-tool"

    if category == "short-completion" and text_chars < OPENAI_TINY_TEXT_CHARS_LT:
        return OPENAI_TINY_DEFAULT, "proposed-canary-default-off", "short-completion-to-tiny"
    if category in {"chat", "summary"} and text_chars < OPENAI_SMALL_TEXT_CHARS_LT:
        return OPENAI_SMALL_DEFAULT, "proposed-canary-default-off", "chat-summary-to-small"
    if category == "tool-light" and text_chars < OPENAI_SMALL_TEXT_CHARS_LT:
        return OPENAI_SMALL_DEFAULT, "proposed-canary-default-off", "tool-light-to-small-needs-tool-safety"
    return None, "none", "no-local-routing-shape-match"


def _target_supported(model: str | None) -> bool:
    if not model:
        return False
    return bool(pricing_basis(model, provider="openai").get("cost_known"))


def _candidate_allows_tools(bucket: dict[str, Any]) -> bool:
    return (
        str(bucket.get("requested_model") or "").lower() == "gpt-5.4"
        and str(bucket.get("target_model") or "").lower() == "gpt-5.4-mini"
        and str(bucket.get("category") or "") == "tool-light"
        and str(bucket.get("simulated_policy") or "") == "local-openai-routing-canary"
    )


def _canary_lifecycle_target(
    *,
    openai_canary: dict[str, Any],
    requested_model: str,
) -> tuple[str | None, str, str] | None:
    if not openai_canary:
        return None
    canary_requested = str(
        openai_canary.get("requested_model")
        or openai_canary.get("original_model")
        or ""
    ).strip()
    canary_target = str(openai_canary.get("target_model") or "").strip()
    if not canary_target:
        return None
    if canary_requested and canary_requested.lower() != requested_model.lower():
        return None
    if canary_target.lower() == requested_model.lower():
        return None
    return (
        canary_target,
        "local-openai-routing-canary",
        "openai-canary-lifecycle-metadata-outside-simulated-route",
    )


def _projected_savings(row: dict[str, Any], requested_model: str, target_model: str, input_tokens: int, output_tokens: int) -> tuple[float, list[str]]:
    blockers: list[str] = []
    if input_tokens <= 0 and output_tokens <= 0:
        blockers.append("missing-baseline-cost")
        return 0.0, blockers

    target_cost = estimate_cost(target_model, input_tokens, output_tokens, provider="openai")
    requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider="openai")
    baseline_cost = _as_float(row.get("cost_baseline_usd"))
    if target_cost is None:
        blockers.append("unsupported-target-model")
        return 0.0, blockers
    if requested_cost is None and baseline_cost <= 0:
        blockers.append("missing-baseline-cost")
        return 0.0, blockers

    baseline = baseline_cost if baseline_cost > 0 else float(requested_cost or 0.0)
    return max(0.0, baseline - target_cost), blockers


def _canary_cohort(canary: dict[str, Any]) -> str:
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
    if status == "ineligible":
        return "skipped"
    if status in {"disabled", "noop"} or cohort == "bypassed_or_disabled":
        return "bypassed_or_disabled"
    if status in {"not_selected", "skipped"} or cohort == "skipped":
        return "skipped"
    return "unknown"


def _empty_lifecycle() -> dict[str, Any]:
    return {
        "cohort_counts": {
            "canary_applied": 0,
            "canary_holdout": 0,
            "safety_stopped": 0,
            "skipped": 0,
            "bypassed_or_disabled": 0,
            "unknown": 0,
        },
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "reason_counts": {},
        "skipped_reason_counts": {},
        "unknown_reason_counts": {},
        "safe_bypass_reason_counts": {},
        "unsupported_shape_reason_counts": {},
        "promotion_blocker_reason_counts": {},
        "unclassified_reason_counts": {},
        "cohort_costs": {
            "canary_applied": _empty_cohort_costs(),
            "canary_holdout": _empty_cohort_costs(),
        },
        "oldest_observed_at": None,
        "latest_observed_at": None,
    }


def _empty_cohort_costs() -> dict[str, Any]:
    return {
        "count": 0,
        "baseline_cost_usd": 0.0,
        "actual_cost_usd": 0.0,
        "target_cost_usd": 0.0,
        "realized_savings_usd": 0.0,
        "projected_savings_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error_count": 0,
        "fallback_count": 0,
        "retry_count": 0,
    }


def _add_costs(dest: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "count",
        "input_tokens",
        "output_tokens",
        "error_count",
        "fallback_count",
        "retry_count",
    ):
        dest[key] = _as_int(dest.get(key)) + _as_int(source.get(key))
    for key in (
        "baseline_cost_usd",
        "actual_cost_usd",
        "target_cost_usd",
        "realized_savings_usd",
        "projected_savings_usd",
    ):
        dest[key] = _as_float(dest.get(key)) + _as_float(source.get(key))


def _finalize_cohort_costs(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    baseline = _as_float(raw.get("baseline_cost_usd"))
    actual = _as_float(raw.get("actual_cost_usd"))
    target = _as_float(raw.get("target_cost_usd"))
    realized = _as_float(raw.get("realized_savings_usd"))
    projected = _as_float(raw.get("projected_savings_usd"))
    errors = _as_int(raw.get("error_count"))
    fallbacks = _as_int(raw.get("fallback_count"))
    retries = _as_int(raw.get("retry_count"))
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "count": count,
        "baseline_cost_usd": round(baseline, 8),
        "actual_cost_usd": round(actual, 8),
        "target_cost_usd": round(target, 8),
        "realized_savings_usd": round(realized, 8),
        "projected_savings_usd": round(projected, 8),
        "input_tokens": _as_int(raw.get("input_tokens")),
        "output_tokens": _as_int(raw.get("output_tokens")),
        "avg_baseline_cost_usd": round(baseline / count, 8) if count else 0.0,
        "avg_actual_cost_usd": round(actual / count, 8) if count else 0.0,
        "avg_target_cost_usd": round(target / count, 8) if count else 0.0,
        "avg_realized_savings_usd": round(realized / count, 8) if count else 0.0,
        "avg_projected_savings_usd": round(projected / count, 8) if count else 0.0,
        "savings_per_1000_calls_usd": round((realized / count) * 1000.0, 6) if count else 0.0,
        "projected_savings_per_1000_calls_usd": round((projected / count) * 1000.0, 6) if count else 0.0,
        "error_count": errors,
        "fallback_count": fallbacks,
        "retry_count": retries,
        "error_rate": round(errors / count, 6) if count else 0.0,
        "fallback_rate": round(fallbacks / count, 6) if count else 0.0,
        "retry_rate": round(retries / count, 6) if count else 0.0,
    }


def _cohort_cost_deltas(cohort_costs: dict[str, Any]) -> dict[str, Any]:
    applied = cohort_costs.get("canary_applied") if isinstance(cohort_costs.get("canary_applied"), dict) else {}
    holdout = cohort_costs.get("canary_holdout") if isinstance(cohort_costs.get("canary_holdout"), dict) else {}

    def f(row: dict[str, Any], key: str) -> float:
        return _as_float(row.get(key))

    return {
        "schema": ACTIVE_LOCAL_POLICY_SAVINGS_DELTAS_SCHEMA,
        "metadata_only": True,
        "aggregate_only": True,
        "applied_count": _as_int(applied.get("count")),
        "holdout_count": _as_int(holdout.get("count")),
        "applied_baseline_cost_usd": f(applied, "baseline_cost_usd"),
        "applied_actual_cost_usd": f(applied, "actual_cost_usd"),
        "applied_target_cost_usd": f(applied, "target_cost_usd"),
        "applied_realized_savings_usd": f(applied, "realized_savings_usd"),
        "applied_projected_savings_usd": f(applied, "projected_savings_usd"),
        "holdout_baseline_cost_usd": f(holdout, "baseline_cost_usd"),
        "holdout_actual_cost_usd": f(holdout, "actual_cost_usd"),
        "holdout_target_cost_usd": f(holdout, "target_cost_usd"),
        "holdout_realized_savings_usd": f(holdout, "realized_savings_usd"),
        "holdout_projected_savings_usd": f(holdout, "projected_savings_usd"),
        "realized_savings_delta_usd": round(f(applied, "realized_savings_usd") - f(holdout, "realized_savings_usd"), 8),
        "projected_savings_delta_usd": round(f(applied, "projected_savings_usd") - f(holdout, "projected_savings_usd"), 8),
        "applied_minus_holdout_actual_cost_avg_usd": round(f(applied, "avg_actual_cost_usd") - f(holdout, "avg_actual_cost_usd"), 8),
        "applied_minus_holdout_realized_savings_avg_usd": round(
            f(applied, "avg_realized_savings_usd") - f(holdout, "avg_realized_savings_usd"),
            8,
        ),
        "applied_minus_holdout_projected_savings_avg_usd": round(
            f(applied, "avg_projected_savings_usd") - f(holdout, "avg_projected_savings_usd"),
            8,
        ),
        "applied_minus_holdout_error_rate_delta": round(f(applied, "error_rate") - f(holdout, "error_rate"), 6),
        "applied_minus_holdout_fallback_rate_delta": round(f(applied, "fallback_rate") - f(holdout, "fallback_rate"), 6),
        "applied_minus_holdout_retry_rate_delta": round(f(applied, "retry_rate") - f(holdout, "retry_rate"), 6),
        "privacy": _openai_promotion_decision_privacy(),
    }


def _canary_matches_candidate(bucket: dict[str, Any], canary: dict[str, Any]) -> bool:
    requested = str(canary.get("requested_model") or canary.get("original_model") or "").strip().lower()
    target = str(canary.get("target_model") or "").strip().lower()
    if requested and requested != str(bucket.get("requested_model") or "").lower():
        return False
    if target and target != str(bucket.get("target_model") or "").lower():
        return False
    canary_category = str(canary.get("category") or "").strip().lower()
    bucket_category = str(bucket.get("category") or "").strip().lower()
    if canary_category and bucket_category and canary_category != bucket_category:
        return False
    surface_aliases = {
        "openai_provider_request": "openai",
        "openai-provider-request": "openai",
        "openai_responses": "openai",
        "openai-responses": "openai",
    }
    canary_surface = surface_aliases.get(str(canary.get("source_surface") or "").strip().lower(), str(canary.get("source_surface") or "").strip().lower())
    bucket_surface = surface_aliases.get(str(bucket.get("source_surface") or "").strip().lower(), str(bucket.get("source_surface") or "").strip().lower())
    if canary_surface and bucket_surface and canary_surface != bucket_surface:
        return False
    canary_endpoint = str(canary.get("endpoint") or "").strip().lower()
    bucket_endpoint = str(bucket.get("endpoint") or "").strip().lower()
    if canary_endpoint and bucket_endpoint and canary_endpoint != bucket_endpoint:
        return False
    return True


def _canary_reason(canary: dict[str, Any]) -> str:
    return str(canary.get("reason") or canary.get("status") or canary.get("cohort") or "unknown").strip() or "unknown"


def _durable_unknown_lifecycle_reason(canary: dict[str, Any]) -> tuple[str, str]:
    reason = _canary_reason(canary).lower()
    status = str(canary.get("status") or "").strip().lower()
    cohort = str(canary.get("cohort") or "").strip().lower()

    safe_bypass_markers = {
        "disabled",
        "noop",
        "canary-disabled",
        "openai-routing-disabled",
        "target-model-already-selected",
        "outside-canary-fraction",
        "outside-canary-and-holdout",
        "not-selected",
        "not_selected",
    }
    if status in {"disabled", "noop"} or reason in safe_bypass_markers:
        return "safe_bypass", reason

    unsupported_markers = {
        "category-not-enabled",
        "request-too-small",
        "request-too-large",
        "requested-model-not-enabled",
        "streaming-not-enabled",
        "tool-request-not-enabled",
        "token-estimate-too-small",
        "token-estimate-too-large",
        "workflow-phase-not-enabled",
    }
    if status == "ineligible" or reason in unsupported_markers:
        return "unsupported_shape", reason

    if "stale" in reason or "stale" in status or "stale" in cohort:
        return "promotion_blocker", "stale-canary-metadata"
    if "shape" in reason and ("mismatch" in reason or "miss" in reason):
        return "promotion_blocker", "shape-mismatch"
    if "marker" in reason or (not status and not cohort):
        return "promotion_blocker", "missing-canary-marker"
    if reason in {"missing-status", "unknown", "mystery"} or status in {"", "unknown", "mystery"} or cohort in {"", "unknown", "mystery", "none"}:
        return "promotion_blocker", "canary-lifecycle-logging-gap"
    return "promotion_blocker", "unrecognized-canary-lifecycle-state"


def _classify_canary_lifecycle_reason(cohort: str, canary: dict[str, Any]) -> tuple[str, str]:
    reason = _canary_reason(canary)
    status = str(canary.get("status") or "").strip()
    reason_l = reason.lower()
    if cohort in {"canary_applied", "canary_holdout"}:
        return "covered", reason
    if cohort == "safety_stopped" or "safety-stop" in reason_l:
        return "promotion_blocker", reason
    if cohort == "unknown":
        return _durable_unknown_lifecycle_reason(canary)
    if cohort == "skipped" and reason_l in {
        "outside-canary-fraction",
        "outside-canary-and-holdout",
        "not-selected",
        "not_selected",
    }:
        return "safe_bypass", reason
    if status in {"disabled", "noop"} or reason_l in {"disabled", "noop", "canary-disabled", "openai-routing-disabled"}:
        return "safe_bypass", reason
    unsupported_markers = {
        "category-not-enabled",
        "request-too-small",
        "request-too-large",
        "requested-model-not-enabled",
        "streaming-not-enabled",
        "tool-request-not-enabled",
        "token-estimate-too-small",
        "token-estimate-too-large",
        "workflow-phase-not-enabled",
    }
    if status == "ineligible" or reason_l in unsupported_markers:
        return "unsupported_shape", reason
    return "promotion_blocker", reason


def _add_canary_cohort_costs(bucket: dict[str, Any], row: dict[str, Any], canary: dict[str, Any], cohort: str) -> None:
    if cohort not in {"canary_applied", "canary_holdout"}:
        return
    lifecycle = bucket.setdefault("openai_canary_lifecycle", _empty_lifecycle())
    cohort_costs = lifecycle.setdefault("cohort_costs", {})
    costs = cohort_costs.setdefault(cohort, _empty_cohort_costs())

    requested_model = str(canary.get("requested_model") or canary.get("original_model") or row.get("requested_model") or "")
    routed_model = str(row.get("routed_model") or canary.get("actual_forwarded_model") or requested_model)
    target_model = str(canary.get("target_model") or bucket.get("target_model") or routed_model)
    routing = _json_obj(row.get("routing_json"))
    text_chars = _text_chars(row, routing)
    input_tokens = _input_tokens(row, text_chars)
    output_tokens = _output_tokens(row)

    baseline_cost = _as_float(row.get("cost_baseline_usd"))
    if baseline_cost <= 0:
        baseline_cost = _as_float(estimate_cost(requested_model, input_tokens, output_tokens, provider="openai"))
    actual_cost = _as_float(row.get("cost_est_usd"))
    if actual_cost <= 0:
        actual_cost = _as_float(estimate_cost(routed_model, input_tokens, output_tokens, provider="openai"))
    target_cost = _as_float(estimate_cost(target_model, input_tokens, output_tokens, provider="openai"))

    _add_costs(
        costs,
        {
            "count": 1,
            "baseline_cost_usd": baseline_cost,
            "actual_cost_usd": actual_cost,
            "target_cost_usd": target_cost,
            "realized_savings_usd": baseline_cost - actual_cost,
            "projected_savings_usd": baseline_cost - target_cost if target_cost > 0 else 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error_count": int(_as_int(row.get("status_code"), -1) >= 400),
            "fallback_count": int(bool(canary.get("fallback_reason"))),
            "retry_count": int(_as_int(row.get("retry_count")) > 0),
        },
    )


def _add_canary_lifecycle(bucket: dict[str, Any], row: dict[str, Any], canary: dict[str, Any]) -> None:
    if not canary or not _canary_matches_candidate(bucket, canary):
        return
    lifecycle = bucket.setdefault("openai_canary_lifecycle", _empty_lifecycle())
    cohort = _canary_cohort(canary)
    lifecycle["cohort_counts"][cohort] = _as_int(lifecycle["cohort_counts"].get(cohort)) + 1
    _add_canary_cohort_costs(bucket, row, canary, cohort)
    classification, classified_reason = _classify_canary_lifecycle_reason(cohort, canary)
    if cohort == "skipped":
        _increment(lifecycle["skipped_reason_counts"], classified_reason)
    if cohort == "unknown":
        _increment(lifecycle["unknown_reason_counts"], classified_reason)
    if cohort in {"skipped", "unknown", "bypassed_or_disabled"}:
        if classification == "safe_bypass":
            _increment(lifecycle["safe_bypass_reason_counts"], classified_reason)
        elif classification == "unsupported_shape":
            _increment(lifecycle["unsupported_shape_reason_counts"], classified_reason)
        elif classification == "promotion_blocker":
            _increment(lifecycle["promotion_blocker_reason_counts"], classified_reason)
        elif classification == "unclassified":
            _increment(lifecycle["unclassified_reason_counts"], classified_reason)
    status_code = _as_int(row.get("status_code"), -1)
    if status_code >= 400:
        lifecycle["error_count"] += 1
    if _as_int(row.get("retry_count")) > 0:
        lifecycle["retry_count"] += 1
    if canary.get("fallback_reason"):
        lifecycle["fallback_count"] += 1
    _increment(lifecycle["reason_counts"], canary.get("reason") or "unknown")

    created_at = row.get("created_at")
    if created_at:
        created = str(created_at)
        if lifecycle["latest_observed_at"] is None or created > str(lifecycle["latest_observed_at"]):
            lifecycle["latest_observed_at"] = created
        if lifecycle["oldest_observed_at"] is None or created < str(lifecycle["oldest_observed_at"]):
            lifecycle["oldest_observed_at"] = created


def _finalize_lifecycle(raw: dict[str, Any] | None, *, matched_count: int) -> dict[str, Any]:
    raw = raw or _empty_lifecycle()
    counts = {key: _as_int(value) for key, value in (raw.get("cohort_counts") or {}).items()}
    observed = sum(counts.values())
    applied = _as_int(counts.get("canary_applied"))
    holdout = _as_int(counts.get("canary_holdout"))
    safety_stopped = _as_int(counts.get("safety_stopped"))
    skipped = _as_int(counts.get("skipped"))
    bypassed = _as_int(counts.get("bypassed_or_disabled"))
    unknown = _as_int(counts.get("unknown"))
    safe_bypass_count = sum(_as_int(value) for value in (raw.get("safe_bypass_reason_counts") or {}).values())
    unsupported_shape_count = sum(_as_int(value) for value in (raw.get("unsupported_shape_reason_counts") or {}).values())
    promotion_blocker_count = sum(_as_int(value) for value in (raw.get("promotion_blocker_reason_counts") or {}).values())
    unclassified_count = sum(_as_int(value) for value in (raw.get("unclassified_reason_counts") or {}).values())
    latest = _parse_time(raw.get("latest_observed_at"))
    age_hours = None
    stale = False
    if latest is not None:
        age_hours = round((datetime.now(timezone.utc) - latest).total_seconds() / 3600.0, 3)
        stale = age_hours > DEFAULT_MAX_EVIDENCE_AGE_HOURS

    blocker_counts: dict[str, int] = {}
    integrity_warnings: dict[str, int] = {}
    if observed == 0:
        blocker_counts["missing-canary-lifecycle-evidence"] = matched_count
    if applied == 0:
        blocker_counts["missing-applied-coverage"] = matched_count
    if holdout == 0:
        blocker_counts["missing-holdout-coverage"] = matched_count
        if 0 < observed < DEFAULT_MIN_HOLDOUT_VOLUME:
            blocker_counts["insufficient-volume-for-holdout"] = max(observed, matched_count)
    if matched_count > 0 and observed > matched_count:
        integrity_warnings["lifecycle-observed-count-exceeds-matched-count"] = observed
    if _as_int(raw.get("error_count")):
        blocker_counts["error-observed"] = _as_int(raw.get("error_count"))
    if _as_int(raw.get("retry_count")):
        blocker_counts["retry-observed"] = _as_int(raw.get("retry_count"))
    if _as_int(raw.get("fallback_count")):
        blocker_counts["fallback-observed"] = _as_int(raw.get("fallback_count"))
    if safety_stopped:
        blocker_counts["safety-stop-observed"] = safety_stopped
    if unsupported_shape_count:
        blocker_counts["skipped-canary-unsupported-shape"] = unsupported_shape_count
    if promotion_blocker_count:
        blocker_counts["skipped-canary-promotion-blocker"] = promotion_blocker_count
    if unclassified_count:
        blocker_counts["unclassified-canary-lifecycle-rows"] = unclassified_count
    if stale:
        blocker_counts["stale-evidence"] = observed

    raw_costs = raw.get("cohort_costs") if isinstance(raw.get("cohort_costs"), dict) else {}
    cohort_costs = {
        "canary_applied": _finalize_cohort_costs(
            raw_costs.get("canary_applied") if isinstance(raw_costs.get("canary_applied"), dict) else _empty_cohort_costs()
        ),
        "canary_holdout": _finalize_cohort_costs(
            raw_costs.get("canary_holdout") if isinstance(raw_costs.get("canary_holdout"), dict) else _empty_cohort_costs()
        ),
    }

    return {
        "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
        "status": "matched" if observed else "no-openai-canary-metadata",
        "observed_count": observed,
        "cohort_counts": {
            "canary_applied": applied,
            "canary_holdout": holdout,
            "safety_stopped": safety_stopped,
            "skipped": skipped,
            "bypassed_or_disabled": bypassed,
            "unknown": unknown,
        },
        "coverage": {
            "matched_count": matched_count,
            "observed_rate": round(min(observed, matched_count) / matched_count, 6) if matched_count else 0.0,
            "applied_rate": round(min(applied, matched_count) / matched_count, 6) if matched_count else 0.0,
            "holdout_rate": round(min(holdout, matched_count) / matched_count, 6) if matched_count else 0.0,
        },
        "error_count": _as_int(raw.get("error_count")),
        "retry_count": _as_int(raw.get("retry_count")),
        "fallback_count": _as_int(raw.get("fallback_count")),
        "oldest_observed_at": raw.get("oldest_observed_at"),
        "latest_observed_at": raw.get("latest_observed_at"),
        "stale_evidence": {
            "stale": stale,
            "age_hours": age_hours,
            "max_age_hours": DEFAULT_MAX_EVIDENCE_AGE_HOURS,
        },
        "reason_breakdown": _breakdown(raw.get("reason_counts") or {}),
        "skipped_reason_breakdown": _breakdown(raw.get("skipped_reason_counts") or {}),
        "unknown_reason_breakdown": _breakdown(raw.get("unknown_reason_counts") or {}),
        "cohort_costs": cohort_costs,
        "savings_deltas": _cohort_cost_deltas(cohort_costs),
        "skipped_unknown_classification": {
            "schema": "agentflow.openai_routing_canary_skipped_unknown_classification.v1",
            "safe_bypass_count": safe_bypass_count,
            "unsupported_shape_count": unsupported_shape_count,
            "promotion_blocker_count": promotion_blocker_count,
            "unclassified_count": unclassified_count,
            "classified_count": safe_bypass_count + unsupported_shape_count + promotion_blocker_count,
            "requires_operator_review": bool(unknown or unsupported_shape_count or promotion_blocker_count or unclassified_count),
            "safe_bypass_reason_breakdown": _breakdown(raw.get("safe_bypass_reason_counts") or {}),
            "unsupported_shape_reason_breakdown": _breakdown(raw.get("unsupported_shape_reason_counts") or {}),
            "promotion_blocker_reason_breakdown": _breakdown(raw.get("promotion_blocker_reason_counts") or {}),
            "unclassified_reason_breakdown": _breakdown(raw.get("unclassified_reason_counts") or {}),
        },
        "blocker_codes": sorted(blocker_counts),
        "blocker_reason_breakdown": _breakdown(blocker_counts),
        "integrity_warning_codes": sorted(integrity_warnings),
        "integrity_warning_breakdown": _breakdown(integrity_warnings),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
        },
    }


def _routing_rule_metadata(bucket: dict[str, Any]) -> dict[str, Any]:
    category = str(bucket.get("category") or "unknown")
    conditions: dict[str, Any] = {
        "model_pattern": bucket.get("requested_model") or bucket.get("requested_model_family") or "gpt-5.4",
        "category": category,
        "source_surface": bucket.get("source_surface") or "openai_responses",
        "endpoint": bucket.get("endpoint") or "responses",
        "stream": bool(bucket.get("stream")),
    }
    if category == "tool-light":
        conditions["has_tools"] = True
        conditions["max_text_chars"] = OPENAI_GPT54_CANARY_TEXT_CHARS_LT
        conditions["max_input_tokens_est"] = 4000
    else:
        conditions["has_tools"] = bool(bucket.get("has_tools"))
        if category in {"chat", "short-completion"}:
            conditions["max_text_chars"] = OPENAI_SMALL_TEXT_CHARS_LT

    return {
        "schema": "agentflow.openai_routing_rule_metadata.v1",
        "policy_source": "local-manual-review",
        "target_local_rule_file": "routing_rules.yaml",
        "target_local_policy_section": "routing.rules",
        "required_local_executor": "openai-routing-canary",
        "rule_preview": {
            "id": (
                "promote-openai-routing:"
                f"{bucket.get('source_surface') or 'openai_responses'}:"
                f"{bucket.get('endpoint') or 'responses'}:"
                f"{category}:to-{str(bucket.get('target_model') or 'target').lower().replace('.', '-')}"
            ),
            "conditions": conditions,
            "action": {
                "route_to": bucket.get("target_model"),
                "reason": (
                    "promote OpenAI routing canary "
                    f"{bucket.get('source_surface') or 'openai_responses'}/"
                    f"{bucket.get('endpoint') or 'responses'}/{category}"
                ),
            },
        },
    }


def _promotion_readiness(bucket: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
    classification = (
        lifecycle.get("skipped_unknown_classification")
        if isinstance(lifecycle.get("skipped_unknown_classification"), dict)
        else {}
    )
    applied = _as_int(counts.get("canary_applied"))
    holdout = _as_int(counts.get("canary_holdout"))
    safety = _as_int(counts.get("safety_stopped"))
    skipped = _as_int(counts.get("skipped"))
    bypassed = _as_int(counts.get("bypassed_or_disabled"))
    unknown = _as_int(counts.get("unknown"))
    unsupported_shape_count = _as_int(classification.get("unsupported_shape_count"))
    promotion_blocker_count = _as_int(classification.get("promotion_blocker_count"))
    unclassified_count = _as_int(classification.get("unclassified_count"))
    observed = _as_int(lifecycle.get("observed_count"))
    error_count = _as_int(lifecycle.get("error_count"))
    fallback_count = _as_int(lifecycle.get("fallback_count"))
    retry_count = _as_int(lifecycle.get("retry_count"))
    stale = bool((lifecycle.get("stale_evidence") or {}).get("stale")) if isinstance(lifecycle.get("stale_evidence"), dict) else False
    estimated_savings = _as_float(bucket.get("estimated_savings_per_1000_calls_usd"))
    candidate_blockers = [str(code) for code in bucket.get("blockers") or []]
    lifecycle_blockers = [str(code) for code in lifecycle.get("blocker_codes") or []]
    missing_evidence = {
        "missing-canary-lifecycle-evidence",
        "missing-applied-coverage",
        "missing-holdout-coverage",
    }

    reason_codes: list[str] = []
    if observed <= 0:
        reason_codes.append("missing-canary-lifecycle-evidence")
    if applied <= 0:
        reason_codes.append("missing-applied-coverage")
    if holdout <= 0:
        reason_codes.append("missing-holdout-coverage")
    if safety:
        reason_codes.append("safety-stop-observed")
    if error_count:
        reason_codes.append("error-observed")
    if fallback_count:
        reason_codes.append("fallback-observed")
    if retry_count:
        reason_codes.append("retry-observed")
    if stale:
        reason_codes.append("stale-evidence")
    if unclassified_count:
        reason_codes.append("unclassified-canary-lifecycle-rows")
    if unsupported_shape_count:
        reason_codes.append("skipped-canary-unsupported-shape")
    if promotion_blocker_count:
        reason_codes.append("skipped-canary-promotion-blocker")
    if estimated_savings <= 0:
        reason_codes.append("non-positive-estimated-savings")
    for blocker in candidate_blockers:
        if blocker not in reason_codes:
            reason_codes.append(blocker)
    for blocker in lifecycle_blockers:
        if blocker not in reason_codes:
            reason_codes.append(blocker)

    unique_reasons = sorted(set(reason_codes))
    staged_review_reasons = missing_evidence | {
        "unclassified-canary-lifecycle-rows",
        "skipped-canary-unsupported-shape",
    }
    hard_blocking_reasons = [
        reason
        for reason in unique_reasons
        if reason not in staged_review_reasons
    ]
    if applied > 0 and holdout > 0 and not hard_blocking_reasons and not unclassified_count and not unsupported_shape_count and estimated_savings > 0:
        decision = "promote"
        next_action = "promote-openai-routing-rule-draft"
        reason = "promotion-ready"
    elif hard_blocking_reasons:
        decision = "blocked"
        next_action = "review-openai-routing-canary-blockers"
        reason = hard_blocking_reasons[0]
    elif unsupported_shape_count:
        decision = "narrow"
        next_action = "narrow-openai-routing-canary-shape"
        reason = "skipped-canary-unsupported-shape"
    else:
        decision = "keep-staged"
        next_action = (
            "classify-openai-routing-canary-skipped-unknown"
            if unclassified_count
            else "collect-openai-routing-canary-evidence"
        )
        reason = unique_reasons[0] if unique_reasons else "insufficient-promotion-evidence"

    return {
        "schema": "agentflow.openai_routing_canary_promotion_readiness.v1",
        "decision": decision,
        "promotion_ready": decision == "promote",
        "next_action": next_action,
        "reason": reason,
        "reason_codes": unique_reasons,
        "decision_options": ["promote", "keep-staged", "narrow", "blocked"],
        "evidence": {
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": skipped,
            "bypassed_or_disabled_count": bypassed,
            "unknown_count": unknown,
            "observed_count": observed,
            "matched_count": _as_int(bucket.get("matched_count")),
            "safety_stop_count": safety,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "estimated_savings_per_1000_calls_usd": estimated_savings,
            "latest_observed_at": lifecycle.get("latest_observed_at"),
            "oldest_observed_at": lifecycle.get("oldest_observed_at"),
            "stale_evidence": lifecycle.get("stale_evidence"),
            "skipped_reason_breakdown": lifecycle.get("skipped_reason_breakdown") or [],
            "unknown_reason_breakdown": lifecycle.get("unknown_reason_breakdown") or [],
            "skipped_unknown_classification": classification,
        },
        "quality_gates": {
            "requires_fresh_evidence": True,
            "requires_applied_coverage": True,
            "requires_holdout_coverage": True,
            "requires_classified_skipped_unknown_rows": True,
            "requires_zero_safety_stops": True,
            "requires_zero_errors": True,
            "requires_zero_fallbacks": True,
            "requires_zero_retries": True,
            "requires_positive_estimated_savings": True,
        },
        "routing_rule_metadata": _routing_rule_metadata(bucket),
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_requests_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "tool_payloads_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "policy_files_written": False,
        },
    }


def _candidate_id(
    *,
    endpoint: str,
    requested_family: str,
    category: str,
    has_tools: bool,
    stream: bool,
    text_bucket: str,
    token_bucket: str,
    target_model: str,
) -> str:
    target = target_model.lower().replace(".", "-")
    family = requested_family.lower().replace(".", "-")
    tool_flag = "tools" if has_tools else "no-tools"
    stream_flag = "stream" if stream else "nonstream"
    return f"openai-route:{endpoint}:{family}:{category}:{tool_flag}:{stream_flag}:{text_bucket}:{token_bucket}:to-{target}"


def _new_bucket(row: dict[str, Any], *, candidate_id: str, target_model: str, target_policy: str, target_reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "provider": "openai",
        "source_surface": _source_surface(row),
        "endpoint": _endpoint(row),
        "requested_model_family": str(row.get("requested_model_family") or openai_model_family(row.get("requested_model")) or "unknown"),
        "requested_model": row.get("requested_model") or "unknown",
        "target_model": target_model,
        "simulated_policy": target_policy,
        "simulated_reason": target_reason,
        "category": row.get("category") or "unknown",
        "has_tools": False,
        "stream": False,
        "text_bucket": "unknown",
        "token_bucket": "unknown",
        "matched_count": 0,
        "blocked_count": 0,
        "current_routed_count": 0,
        "projected_savings_usd": 0.0,
        "estimated_baseline_cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error_count": 0,
        "retry_count": 0,
        "latency_values": [],
        "status_counts": {},
        "cache_status_counts": {},
        "blocker_counts": {},
        "row_blocked_count": 0,
        "openai_canary_lifecycle": _empty_lifecycle(),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    matched = _as_int(bucket.get("matched_count"))
    latency_values = bucket.pop("latency_values", [])
    blocker_counts = bucket.pop("blocker_counts", {})
    row_blocked = _as_int(bucket.pop("row_blocked_count", 0))
    group_blockers: list[str] = []
    error_rate = (_as_int(bucket.get("error_count")) / matched) if matched else 0.0
    retry_rate = (_as_int(bucket.get("retry_count")) / matched) if matched else 0.0
    if matched < DEFAULT_MIN_SAMPLES:
        group_blockers.append("insufficient-samples")
        blocker_counts["insufficient-samples"] = matched
    if error_rate > DEFAULT_MAX_ERROR_RATE:
        group_blockers.append("high-recent-error-rate")
        blocker_counts["high-recent-error-rate"] = matched
    if retry_rate > DEFAULT_MAX_RETRY_RATE:
        group_blockers.append("high-recent-retry-rate")
        blocker_counts["high-recent-retry-rate"] = matched
    if _as_int(bucket.get("stream_count")) == matched and matched > 0:
        group_blockers.append("stream-only-evidence")
        blocker_counts["stream-only-evidence"] = matched

    blocked = matched if group_blockers else row_blocked
    avg_latency = round(sum(latency_values) / len(latency_values)) if latency_values else None
    suggested = 0.0
    if matched and blocked == 0:
        suggested = 0.05 if matched < 100 else 0.10

    bucket["projected_savings_usd"] = round(_as_float(bucket.get("projected_savings_usd")), 6)
    bucket["estimated_baseline_cost_usd"] = round(_as_float(bucket.get("estimated_baseline_cost_usd")), 6)
    bucket["estimated_savings_per_1000_calls_usd"] = round(
        (_as_float(bucket.get("projected_savings_usd")) / matched) * 1000.0,
        6,
    ) if matched else 0.0
    bucket["blocked_count"] = blocked
    bucket["error_rate"] = round(error_rate, 4) if matched else 0.0
    bucket["retry_rate"] = round(retry_rate, 4) if matched else 0.0
    bucket["avg_latency_ms"] = avg_latency
    bucket["blockers"] = sorted(set(group_blockers + list(blocker_counts.keys()))) if blocked else []
    bucket["blocker_reason_breakdown"] = _breakdown(blocker_counts)
    bucket["status_breakdown"] = _breakdown(bucket.pop("status_counts", {}))
    bucket["cache_status_breakdown"] = _breakdown(bucket.pop("cache_status_counts", {}))
    lifecycle = _finalize_lifecycle(
        bucket.pop("openai_canary_lifecycle", None),
        matched_count=matched,
    )
    bucket["openai_canary_lifecycle_evidence"] = lifecycle
    bucket["promotion_readiness"] = _promotion_readiness(bucket, lifecycle)
    bucket["suggested_canary_fraction"] = suggested
    bucket["privacy"] = {
        "metadata_only": True,
        "aggregate_only": True,
        "candidate_id_derived_from_raw_body": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_requests_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
    }
    bucket.pop("stream_count", None)
    return bucket


def build_openai_routing_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    buckets: dict[str, dict[str, Any]] = {}
    blocker_totals: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    unmatched_reason_counts: dict[str, int] = {}
    openai_count = 0
    current_routed_total = 0

    for row in rows:
        if str(row.get("provider") or "").lower() != "openai":
            continue
        openai_count += 1
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
        routed_model = str(row.get("routed_model") or routing.get("routed_model") or requested_model)
        category = str(row.get("category") or routing.get("category") or "unknown")
        surface = _source_surface(row)
        endpoint = _endpoint(row)
        has_tools = _has_tools(row, routing, cache)
        stream = bool(_as_int(row.get("stream")))
        text_chars = _text_chars(row, routing)
        input_tokens = _input_tokens(row, text_chars)
        output_tokens = _output_tokens(row)
        text_bucket = _text_bucket(text_chars)
        token_bucket = _token_bucket(input_tokens)
        requested_family = str(row.get("requested_model_family") or openai_model_family(requested_model) or "unknown")
        cache_status = str(cache.get("status") or ("hit" if _as_int(row.get("cache_hit")) else "missing"))
        openai_canary = routing.get("openai_canary") if isinstance(routing.get("openai_canary"), dict) else {}
        current_routed = bool(requested_model and routed_model and requested_model != routed_model)
        if current_routed:
            current_routed_total += 1

        _increment(category_counts, category)
        _increment(surface_counts, surface)
        _increment(cache_status_counts, cache_status)

        target_model, policy, reason = _simulate_openai_route(
            requested_model=requested_model,
            category=category,
            has_tools=has_tools,
            text_chars=text_chars,
        )
        if not target_model:
            canary_target = _canary_lifecycle_target(
                openai_canary=openai_canary,
                requested_model=requested_model,
            )
            if canary_target is None:
                _increment(unmatched_reason_counts, reason)
                continue
            target_model, policy, reason = canary_target
        if target_model.lower() == requested_model.lower():
            _increment(unmatched_reason_counts, "target-same-as-requested")
            continue

        cid = _candidate_id(
            endpoint=endpoint,
            requested_family=requested_family,
            category=category,
            has_tools=has_tools,
            stream=stream,
            text_bucket=text_bucket,
            token_bucket=token_bucket,
            target_model=target_model,
        )
        bucket = buckets.setdefault(
            cid,
            _new_bucket(row, candidate_id=cid, target_model=target_model, target_policy=policy, target_reason=reason),
        )
        bucket["matched_count"] += 1
        bucket["has_tools"] = bool(bucket["has_tools"] or has_tools)
        bucket["stream"] = bool(bucket["stream"] or stream)
        bucket["text_bucket"] = text_bucket
        bucket["token_bucket"] = token_bucket
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["current_routed_count"] += int(current_routed)
        bucket["error_count"] += int(_as_int(row.get("status_code")) >= 400)
        bucket["retry_count"] += int(_as_int(row.get("retry_count")) > 0)
        _add_canary_lifecycle(bucket, row, openai_canary)
        if stream:
            bucket["stream_count"] = _as_int(bucket.get("stream_count")) + 1
        latency = _as_int(row.get("latency_ms"))
        if latency > 0:
            bucket["latency_values"].append(latency)
        _increment(bucket["status_counts"], row.get("status_code") or "unknown")
        _increment(bucket["cache_status_counts"], cache_status)

        row_blockers: list[str] = []
        if has_tools and not _candidate_allows_tools(bucket):
            row_blockers.append("tools-disabled")
        if requested_family in {"unknown", "other", "none"}:
            row_blockers.append("unknown-model-family")
        if not _target_supported(target_model):
            row_blockers.append("unsupported-target-model")

        savings, cost_blockers = _projected_savings(row, requested_model, target_model, input_tokens, output_tokens)
        requested_cost = estimate_cost(requested_model, input_tokens, output_tokens, provider="openai")
        row_baseline = _as_float(row.get("cost_baseline_usd"))
        bucket["estimated_baseline_cost_usd"] += row_baseline if row_baseline > 0 else _as_float(requested_cost)
        row_blockers.extend(cost_blockers)
        if row_blockers:
            bucket["row_blocked_count"] += 1
            for blocker in sorted(set(row_blockers)):
                _increment(bucket["blocker_counts"], blocker)
        else:
            bucket["projected_savings_usd"] += savings

    candidates = [_finalize_bucket(bucket) for bucket in buckets.values()]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("matched_count")) - _as_int(item.get("blocked_count")),
            _as_int(item.get("matched_count")),
        ),
        reverse=True,
    )

    matched_count = sum(_as_int(item.get("matched_count")) for item in candidates)
    blocked_count = sum(_as_int(item.get("blocked_count")) for item in candidates)
    projected_savings = sum(_as_float(item.get("projected_savings_usd")) for item in candidates)
    estimated_baseline_cost = sum(_as_float(item.get("estimated_baseline_cost_usd")) for item in candidates)
    estimated_savings_per_1000_calls = round((projected_savings / matched_count) * 1000.0, 6) if matched_count else 0.0
    canary_applied_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("canary_applied")) for item in candidates)
    canary_holdout_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("canary_holdout")) for item in candidates)
    canary_safety_stopped_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("safety_stopped")) for item in candidates)
    canary_skipped_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("skipped")) for item in candidates)
    canary_bypassed_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("bypassed_or_disabled")) for item in candidates)
    canary_unknown_count = sum(_as_int(((item.get("openai_canary_lifecycle_evidence") or {}).get("cohort_counts") or {}).get("unknown")) for item in candidates)
    canary_error_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("error_count")) for item in candidates)
    canary_retry_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("retry_count")) for item in candidates)
    canary_fallback_count = sum(_as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("fallback_count")) for item in candidates)
    canary_stale_evidence_count = sum(
        _as_int((item.get("openai_canary_lifecycle_evidence") or {}).get("observed_count"))
        for item in candidates
        if ((item.get("openai_canary_lifecycle_evidence") or {}).get("stale_evidence") or {}).get("stale")
    )
    promotion_ready_count = sum(1 for item in candidates if (item.get("promotion_readiness") or {}).get("decision") == "promote")
    keep_staged_count = sum(1 for item in candidates if (item.get("promotion_readiness") or {}).get("decision") == "keep-staged")
    keep_blocked_count = sum(1 for item in candidates if (item.get("promotion_readiness") or {}).get("decision") in {"blocked", "keep-blocked"})
    for item in candidates:
        for blocker in item.get("blocker_reason_breakdown") or []:
            _increment(blocker_totals, blocker["value"], _as_int(blocker.get("count")))

    eligible_candidates = [item for item in candidates if _as_int(item.get("blocked_count")) == 0]
    suggested_fraction = max((_as_float(item.get("suggested_canary_fraction")) for item in eligible_candidates), default=0.0)

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": openai_count,
            "candidate_count": len(candidates),
            "matched_count": matched_count,
            "current_routed_count": current_routed_total,
            "blocked_count": blocked_count,
            "eligible_count": max(0, matched_count - blocked_count),
            "projected_savings_usd": round(projected_savings, 6),
            "estimated_baseline_cost_usd": round(estimated_baseline_cost, 6),
            "estimated_savings_per_1000_calls_usd": estimated_savings_per_1000_calls,
            "suggested_canary_fraction": suggested_fraction,
            "openai_canary_applied_count": canary_applied_count,
            "openai_canary_holdout_count": canary_holdout_count,
            "openai_canary_safety_stopped_count": canary_safety_stopped_count,
            "openai_canary_skipped_count": canary_skipped_count,
            "openai_canary_bypassed_or_disabled_count": canary_bypassed_count,
            "openai_canary_unknown_count": canary_unknown_count,
            "openai_canary_error_count": canary_error_count,
            "openai_canary_retry_count": canary_retry_count,
            "openai_canary_fallback_count": canary_fallback_count,
            "openai_canary_stale_evidence_count": canary_stale_evidence_count,
            "promotion_ready_count": promotion_ready_count,
            "keep_staged_count": keep_staged_count,
            "keep_blocked_count": keep_blocked_count,
        },
        "simulation_policy": {
            "schema": "agentflow.openai_routing_simulation_policy.v1",
            "provider_calls_made": False,
            "existing_route_openai_model_thresholds": {
                "large_model": OPENAI_LARGE_DEFAULT,
                "small_model": OPENAI_SMALL_DEFAULT,
                "tiny_model": OPENAI_TINY_DEFAULT,
                "small_text_chars_lt": OPENAI_SMALL_TEXT_CHARS_LT,
                "tiny_text_chars_lt": OPENAI_TINY_TEXT_CHARS_LT,
                "openai_routing_default_enabled": False,
            },
            "proposed_file_backed_canary_policy": {
                "policy_id": "local-openai-routing-canary-opportunity-v1",
                "status": "simulated-only",
                "default_enabled": False,
                "eligible_categories": ["chat", "short-completion", "tool-light"],
                "blocked_until_policy_support": [],
            },
            "quality_gate_policy": {
                "min_samples": DEFAULT_MIN_SAMPLES,
                "max_error_rate": DEFAULT_MAX_ERROR_RATE,
                "max_retry_rate": DEFAULT_MAX_RETRY_RATE,
            },
        },
        "category_breakdown": _breakdown(category_counts),
        "source_surface_breakdown": _breakdown(surface_counts),
        "cache_status_breakdown": _breakdown(cache_status_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "unmatched_reason_breakdown": _breakdown(unmatched_reason_counts),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "absolute_paths_included": False,
            "individual_candidate_ids_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "basis": "local calls table metadata plus sanitized routing/cache decision summaries only",
        },
    }


def _matches_promotion_target(
    candidate: dict[str, Any],
    *,
    requested_model: str,
    target_model: str,
    source_surface: str,
    endpoint: str,
    category: str,
) -> bool:
    return (
        str(candidate.get("requested_model") or "").lower() == requested_model.lower()
        and str(candidate.get("target_model") or "").lower() == target_model.lower()
        and str(candidate.get("source_surface") or "").lower() == source_surface.lower()
        and str(candidate.get("endpoint") or "").lower() == endpoint.lower()
        and str(candidate.get("category") or "").lower() == category.lower()
    )


def _merge_breakdown(counter: dict[str, int], breakdown: Any) -> None:
    if not isinstance(breakdown, list):
        return
    for item in breakdown:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None:
            continue
        _increment(counter, value, _as_int(item.get("count"), 1))


def _openai_promotion_decision_privacy() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_requests_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "tool_payloads_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "individual_candidate_ids_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "policy_files_written": False,
    }


def _stable_fingerprint(*parts: Any) -> str:
    normalized = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _candidate_set_metadata(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_routing_candidate_set_metadata.v1",
        "candidate_count": len(candidate_ids),
        "candidate_fingerprint": f"openai-routing-candidates:{_stable_fingerprint(*sorted(candidate_ids))}"
        if candidate_ids
        else None,
        "candidate_ids_included": False,
        "individual_candidate_ids_included": False,
        "metadata_only": True,
        "aggregate_only": True,
    }


def _promotion_verdict(
    *,
    decision: str,
    blocker_counts: dict[str, int],
    applied_count: int,
    current_routed_count: int,
) -> str:
    if decision == "active-local-policy":
        return "active-local-policy"
    if decision == "promote":
        return "promotion-ready"
    rollback_reasons = {
        "safety-stop-observed",
        "error-observed",
        "fallback-observed",
        "retry-observed",
    }
    if rollback_reasons.intersection(blocker_counts) and (applied_count > 0 or current_routed_count > 0):
        return "rollback-required"
    if decision == "keep-blocked":
        return "keep-blocked"
    return "keep-staged"


def _openai_promotion_rollback_metadata(
    *,
    routing_rule_metadata: dict[str, Any],
    reason_codes: list[str],
    promotion_verdict: str,
) -> dict[str, Any]:
    rule_preview = routing_rule_metadata.get("rule_preview") if isinstance(routing_rule_metadata.get("rule_preview"), dict) else {}
    rule_id = str(rule_preview.get("id") or "promote-openai-routing-canary")
    disabled_reason = reason_codes[0] if reason_codes else "operator-requested"
    return {
        "schema": "agentflow.openai_routing_promotion_rollback_metadata.v1",
        "required_for_promotion": True,
        "promotion_verdict": promotion_verdict,
        "rollback_action_type": "disable_openai_routing_rule",
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "target_rule_id": "[REDACTED_ID]",
        "rule_id_included": False,
        "policy_files_written": False,
        "disable_patch": {
            "rules": [
                {
                    "id": rule_id,
                    "enabled": False,
                    "disabled_reason": disabled_reason,
                }
            ]
        },
        "rollback_reason_codes": [
            "operator-requested",
            "safety-stop-observed",
            "error-observed",
            "fallback-observed",
            "retry-observed",
            "stale-evidence",
        ],
        "preserve_previous_rule_required": True,
        "privacy": _openai_promotion_decision_privacy(),
    }


def _openai_promotion_local_policy_patch(
    *,
    decision: str,
    routing_rule_metadata: dict[str, Any],
    lifecycle: dict[str, Any],
    savings_per_1000: float,
) -> dict[str, Any] | None:
    if decision != "promote":
        return None
    rule_preview = routing_rule_metadata.get("rule_preview") if isinstance(routing_rule_metadata.get("rule_preview"), dict) else {}
    rule = {
        "id": rule_preview.get("id") or "promote-openai-routing-canary",
        "enabled": True,
        "policy_source": "local-promoted-review",
        "conditions": rule_preview.get("conditions") if isinstance(rule_preview.get("conditions"), dict) else {},
        "action": rule_preview.get("action") if isinstance(rule_preview.get("action"), dict) else {},
        "metadata": {
            "schema": "agentflow.openai_routing_promotion_rule_metadata.v1",
            "source": "openai_routing_promotion_decision",
            "operator_apply_required": True,
            "policy_files_written": False,
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
            "promotion_evidence": {
                "schema": "agentflow.openai_routing_promotion_patch_evidence.v1",
                "applied_count": _as_int(lifecycle.get("applied_count")),
                "holdout_count": _as_int(lifecycle.get("holdout_count")),
                "safety_stop_count": _as_int(lifecycle.get("safety_stop_count")),
                "error_count": _as_int(lifecycle.get("error_count")),
                "fallback_count": _as_int(lifecycle.get("fallback_count")),
                "retry_count": _as_int(lifecycle.get("retry_count")),
                "savings_per_1000_calls_usd": savings_per_1000,
            },
            "privacy": _openai_promotion_decision_privacy(),
        },
    }
    return {
        "schema": "agentflow.openai_routing_local_policy_patch.v1",
        "patch_type": "promote_openai_routing_canary",
        "status": "drafted",
        "operator_apply_required": True,
        "policy_files_written": False,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "rules": [rule],
        "privacy": _openai_promotion_decision_privacy(),
    }


def _active_openai_local_policy_rule(target: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from agentflow_proxy import router as router_module
    except Exception:
        return None

    requested_model = str(target.get("requested_model") or "").lower()
    target_model = str(target.get("target_model") or "").lower()
    source_surface = str(target.get("source_surface") or "").lower()
    endpoint = str(target.get("endpoint") or "").lower()
    category = str(target.get("category") or "").lower()

    for rule in getattr(router_module, "ROUTING_RULES", []):
        if not isinstance(rule, dict) or rule.get("enabled") is False:
            continue
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        if str(action.get("route_to") or "").lower() != target_model:
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        provider_cond = str(conditions.get("provider") or "openai").lower()
        if provider_cond != "openai":
            continue
        model_pattern = str(conditions.get("model_pattern") or "").lower()
        if model_pattern and model_pattern not in requested_model:
            continue
        source_cond = str(conditions.get("source_surface") or source_surface).lower()
        if source_cond != source_surface:
            continue
        endpoint_cond = str(conditions.get("endpoint") or endpoint).lower()
        if endpoint_cond != endpoint:
            continue
        category_cond = str(conditions.get("category") or category).lower()
        if category_cond != category:
            continue
        if "has_tools" in conditions and bool(conditions.get("has_tools")) != category.startswith("tool-"):
            continue
        if "stream" in conditions and bool(conditions.get("stream")):
            continue
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        rule_id = str(rule.get("id") or "promoted-openai-routing-rule")
        return {
            "schema": "agentflow.openai_routing_active_local_policy_rule.v1",
            "status": "active-local-policy",
            "reason": "matching-openai-routing-rule-active-in-local-policy",
            "policy_source": str(rule.get("policy_source") or metadata.get("policy_source") or "local-promoted"),
            "promoted_from_canary": bool(metadata.get("promoted_from_canary")),
            "source": metadata.get("source") or "routing_rules.yaml",
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
            "rule_id_included": True,
            "target_rule_id": rule_id,
            "conditions": {
                "provider": "openai",
                "source_surface": source_surface,
                "endpoint": endpoint,
                "requested_model": requested_model,
                "target_model": target_model,
                "category": category,
            },
            "rollback_metadata": {
                "schema": "agentflow.openai_routing_promotion_rollback_metadata.v1",
                "required_for_promotion": True,
                "rollback_action_type": "disable_openai_routing_rule",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "target_rule_id": rule_id,
                "rule_id_included": True,
                "disable_patch": {
                    "rules": [
                        {
                            "id": rule_id,
                            "enabled": False,
                            "disabled_reason": "operator-requested",
                        }
                    ]
                },
                "rollback_reason_codes": [
                    "operator-requested",
                    "safety-stop-observed",
                    "error-observed",
                    "fallback-observed",
                    "retry-observed",
                    "stale-evidence",
                ],
                "preserve_previous_rule_required": True,
                "privacy": _openai_promotion_decision_privacy(),
            },
            "privacy": _openai_promotion_decision_privacy(),
        }
    return None


def _openai_promotion_duplicate_suppression(
    *,
    target: dict[str, Any],
    promotion_verdict: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    fingerprint = _stable_fingerprint(
        "openai-routing-promotion",
        target.get("source_surface"),
        target.get("endpoint"),
        target.get("category"),
        target.get("requested_model"),
        target.get("target_model"),
        promotion_verdict,
        ",".join(sorted(reason_codes)),
    )
    return {
        "schema": "agentflow.openai_routing_promotion_duplicate_suppression.v1",
        "fingerprint": f"routing-promotion:{fingerprint}",
        "reason": reason_codes[0] if reason_codes else promotion_verdict,
        "promotion_verdict": promotion_verdict,
        "metadata_only": True,
        "aggregate_only": True,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "suppresses_generic_routing_activation_issue": True,
        "suppresses_new_openai_routing_promotion_issue": True,
        "suppresses_repeated_canary_decision_issue": True,
    }


def _openai_active_policy_outcome_gate(
    *,
    applied_count: int,
    holdout_count: int,
    skipped_count: int,
    unknown_count: int,
    safety_stop_count: int,
    error_count: int,
    fallback_count: int,
    retry_count: int,
    stale_evidence_count: int,
    savings_per_1000: float,
    savings_deltas: dict[str, Any],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if applied_count <= 0:
        reason_codes.append("missing-applied-coverage")
    if holdout_count <= 0:
        reason_codes.append("missing-holdout-coverage")
    if safety_stop_count > 0:
        reason_codes.append("safety-stop-observed")
    if error_count > 0:
        reason_codes.append("error-observed")
    if fallback_count > 0:
        reason_codes.append("fallback-observed")
    if retry_count > 0:
        reason_codes.append("retry-observed")
    if stale_evidence_count > 0:
        reason_codes.append("stale-evidence")
    if unknown_count > 0:
        reason_codes.append("unknown-coverage-observed")
    if savings_per_1000 <= 0:
        reason_codes.append("non-positive-estimated-savings")
    if _as_float(savings_deltas.get("applied_minus_holdout_realized_savings_avg_usd")) < 0:
        reason_codes.append("negative-applied-holdout-savings-delta")

    rollback_reasons = {
        "safety-stop-observed",
        "error-observed",
        "fallback-observed",
        "retry-observed",
    }
    coverage_reasons = {"missing-applied-coverage", "missing-holdout-coverage"}
    if any(reason in rollback_reasons for reason in reason_codes):
        state = "rollback-required"
        gate_passed = False
    elif "stale-evidence" in reason_codes:
        state = "review-stale-evidence"
        gate_passed = False
    elif any(reason in coverage_reasons for reason in reason_codes) or "unknown-coverage-observed" in reason_codes:
        state = "keep-blocked"
        gate_passed = False
    elif (
        "non-positive-estimated-savings" in reason_codes
        or "negative-applied-holdout-savings-delta" in reason_codes
    ):
        state = "keep-blocked"
        gate_passed = False
    else:
        state = "keep-active"
        gate_passed = True
    next_action = state

    return {
        "schema": ACTIVE_LOCAL_POLICY_OUTCOME_GATE_SCHEMA,
        "state": state,
        "gate_passed": gate_passed,
        "deterministic_next_action": next_action,
        "next_action": next_action,
        "reason_codes": reason_codes,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "savings_per_1000_calls_usd": savings_per_1000,
        "decision_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
        "regression_counters": {
            "schema": "agentflow.openai_routing_active_local_policy_outcome_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": skipped_count,
            "unknown_count": unknown_count,
            "safety_stop_count": safety_stop_count,
            "error_count": error_count,
            "fallback_count": fallback_count,
            "retry_count": retry_count,
            "stale_evidence_count": stale_evidence_count,
            "stale_evidence": {
                "metadata_only": True,
                "aggregate_only": True,
                "stale": stale_evidence_count > 0,
                "status": "stale" if stale_evidence_count > 0 else "fresh",
            },
            "savings_per_1000_calls_usd": savings_per_1000,
            "applied_minus_holdout_error_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_error_rate_delta")),
            "applied_minus_holdout_fallback_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_fallback_rate_delta")),
            "applied_minus_holdout_retry_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_retry_rate_delta")),
        },
        "privacy": _openai_promotion_decision_privacy(),
    }


def _openai_active_policy_rollback_metadata(
    *,
    active_local_policy_rule: dict[str, Any],
    outcome_gate: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = outcome_gate.get("reason_codes") if isinstance(outcome_gate.get("reason_codes"), list) else []
    disabled_reason = str(reason_codes[0]) if reason_codes else "operator-requested"
    return {
        "schema": "agentflow.openai_routing_active_local_policy_rollback_metadata.v1",
        "required_for_active_policy_measurement": True,
        "rollback_action_type": "disable_openai_routing_rule",
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "target_rule_id": "[REDACTED_ID]",
        "rule_id_included": False,
        "policy_files_written": False,
        "disable_patch": {
            "rules": [
                {
                    "id": "[REDACTED_ID]",
                    "enabled": False,
                    "disabled_reason": disabled_reason,
                }
            ]
        },
        "rollback_reason_codes": [
            "operator-requested",
            "safety-stop-observed",
            "error-observed",
            "fallback-observed",
            "retry-observed",
            "stale-evidence",
        ],
        "preserve_previous_rule_required": bool(
            (active_local_policy_rule.get("rollback_metadata") or {}).get("preserve_previous_rule_required", True)
        )
        if isinstance(active_local_policy_rule.get("rollback_metadata"), dict)
        else True,
        "privacy": _openai_promotion_decision_privacy(),
    }


def _openai_active_local_policy_outcome(
    *,
    decision: str,
    next_action: str,
    reason: str,
    target: dict[str, Any],
    lifecycle: dict[str, Any],
    active_local_policy_rule: dict[str, Any] | None,
    matched_count: int,
    current_routed_count: int,
    candidate_count: int,
    candidate_set: dict[str, Any],
    projected_savings_usd: float,
    savings_per_1000: float,
) -> dict[str, Any] | None:
    if active_local_policy_rule is None:
        return None

    applied = _as_int(lifecycle.get("applied_count"))
    holdout = _as_int(lifecycle.get("holdout_count"))
    skipped = _as_int(lifecycle.get("skipped_count"))
    unknown = _as_int(lifecycle.get("unknown_count"))
    safety = _as_int(lifecycle.get("safety_stop_count"))
    errors = _as_int(lifecycle.get("error_count"))
    fallbacks = _as_int(lifecycle.get("fallback_count"))
    retries = _as_int(lifecycle.get("retry_count"))
    stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
    stale_evidence_count = _as_int(lifecycle.get("observed_count")) if stale.get("stale") else 0
    cohort_costs = lifecycle.get("cohort_costs") if isinstance(lifecycle.get("cohort_costs"), dict) else {}
    savings_deltas = lifecycle.get("savings_deltas") if isinstance(lifecycle.get("savings_deltas"), dict) else _cohort_cost_deltas({})
    outcome_gate = _openai_active_policy_outcome_gate(
        applied_count=applied,
        holdout_count=holdout,
        skipped_count=skipped,
        unknown_count=unknown,
        safety_stop_count=safety,
        error_count=errors,
        fallback_count=fallbacks,
        retry_count=retries,
        stale_evidence_count=stale_evidence_count,
        savings_per_1000=savings_per_1000,
        savings_deltas=savings_deltas,
    )
    rollback_metadata = _openai_active_policy_rollback_metadata(
        active_local_policy_rule=active_local_policy_rule,
        outcome_gate=outcome_gate,
    )
    coverage = {
        "schema": "agentflow.openai_routing_active_local_policy_coverage.v1",
        "matched_count": matched_count,
        "observed_count": _as_int(lifecycle.get("observed_count")),
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": skipped,
        "unknown_count": unknown,
        "current_routed_count": current_routed_count,
        "applied_rate": round(applied / matched_count, 6) if matched_count else 0.0,
        "holdout_rate": round(holdout / matched_count, 6) if matched_count else 0.0,
        "current_routed_rate": round(current_routed_count / matched_count, 6) if matched_count else 0.0,
        "metadata_only": True,
        "aggregate_only": True,
    }
    return {
        "schema": ACTIVE_LOCAL_POLICY_OUTCOME_SCHEMA,
        "status": "active-local-policy",
        "state": "active-local-policy",
        "current_status": "applied",
        "outcome": outcome_gate["state"],
        "outcome_decision": outcome_gate["state"],
        "decision": decision,
        "measurement_next_action": next_action,
        "next_action": outcome_gate["next_action"],
        "deterministic_next_action": outcome_gate["deterministic_next_action"],
        "reason_codes": outcome_gate["reason_codes"],
        "gate_passed": outcome_gate["gate_passed"],
        "reason": reason,
        "local_action_family": "routing",
        "target": target,
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
        "local_file_backed_representation": {
            "schema": "agentflow.openai_routing_local_file_backed_representation.v1",
            "exists": True,
            "policy_section": "routing",
            "policy_source": "local-file-backed",
            "rule_file": "routing_rules.yaml",
            "path_included": False,
            "policy_file_contents_included": False,
        },
        "active_local_policy_rule": {
            "schema": "agentflow.openai_routing_active_local_policy_rule_reference.v1",
            "status": active_local_policy_rule.get("status"),
            "reason": active_local_policy_rule.get("reason"),
            "policy_source": active_local_policy_rule.get("policy_source"),
            "promoted_from_canary": bool(active_local_policy_rule.get("promoted_from_canary")),
            "rule_id_included": False,
            "target_rule_id": "[REDACTED_ID]",
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
        },
        "candidate_set": candidate_set,
        "candidate_count": candidate_count,
        "candidate_ids_included": False,
        "matched_count": matched_count,
        "current_routed_count": current_routed_count,
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": skipped,
        "unknown_count": unknown,
        "stale_evidence_count": stale_evidence_count,
        "safety_stop_count": safety,
        "error_count": errors,
        "fallback_count": fallbacks,
        "retry_count": retries,
        "regression_counters": {
            "schema": "agentflow.openai_routing_active_local_policy_regression_counters.v1",
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": skipped,
            "unknown_count": unknown,
            "error_count": errors,
            "fallback_count": fallbacks,
            "retry_count": retries,
            "safety_stop_count": safety,
            "stale_evidence_count": stale_evidence_count,
            "rollback_count": 0,
            "applied_minus_holdout_error_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_error_rate_delta")),
            "applied_minus_holdout_fallback_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_fallback_rate_delta")),
            "applied_minus_holdout_retry_rate_delta": _as_float(savings_deltas.get("applied_minus_holdout_retry_rate_delta")),
            "metadata_only": True,
            "aggregate_only": True,
        },
        "outcome_gate": outcome_gate,
        "rollback_metadata": rollback_metadata,
        "coverage": coverage,
        "latest_observed_at": lifecycle.get("latest_observed_at"),
        "oldest_observed_at": lifecycle.get("oldest_observed_at"),
        "evidence_age_hours": lifecycle.get("evidence_age_hours"),
        "stale_evidence": stale,
        "cohort_costs": cohort_costs,
        "savings_deltas": savings_deltas,
        "realized_savings_usd": _as_float(savings_deltas.get("applied_realized_savings_usd")),
        "applied_realized_savings_usd": _as_float(savings_deltas.get("applied_realized_savings_usd")),
        "holdout_realized_savings_usd": _as_float(savings_deltas.get("holdout_realized_savings_usd")),
        "applied_minus_holdout_realized_savings_avg_usd": _as_float(
            savings_deltas.get("applied_minus_holdout_realized_savings_avg_usd")
        ),
        "savings_per_1000_calls_usd": savings_per_1000,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "expected_savings_path": "Measure post-apply outcomes for the active local OpenAI routing rule.",
        "privacy": _openai_promotion_decision_privacy(),
    }


def _build_openai_promotion_decision(
    *,
    report: dict[str, Any],
    requested_model: str,
    target_model: str,
    source_surface: str,
    endpoint: str,
    category: str,
) -> dict[str, Any]:
    candidates = [
        candidate
        for candidate in report.get("candidates", [])
        if isinstance(candidate, dict)
        and _matches_promotion_target(
            candidate,
            requested_model=requested_model,
            target_model=target_model,
            source_surface=source_surface,
            endpoint=endpoint,
            category=category,
        )
    ]

    matched_count = sum(_as_int(candidate.get("matched_count")) for candidate in candidates)
    current_routed_count = sum(_as_int(candidate.get("current_routed_count")) for candidate in candidates)
    candidate_blocked_count = sum(_as_int(candidate.get("blocked_count")) for candidate in candidates)
    projected_savings_usd = sum(_as_float(candidate.get("projected_savings_usd")) for candidate in candidates)
    baseline_cost_usd = sum(_as_float(candidate.get("estimated_baseline_cost_usd")) for candidate in candidates)
    savings_per_1000 = round((projected_savings_usd / matched_count) * 1000.0, 6) if matched_count else 0.0

    cohort_counts = {
        "canary_applied": 0,
        "canary_holdout": 0,
        "safety_stopped": 0,
        "skipped": 0,
        "bypassed_or_disabled": 0,
        "unknown": 0,
    }
    error_count = 0
    fallback_count = 0
    retry_count = 0
    stale_evidence_count = 0
    latest_observed_at: str | None = None
    oldest_observed_at: str | None = None
    skipped_reason_counts: dict[str, int] = {}
    unknown_reason_counts: dict[str, int] = {}
    safe_bypass_reason_counts: dict[str, int] = {}
    unsupported_shape_reason_counts: dict[str, int] = {}
    promotion_blocker_reason_counts: dict[str, int] = {}
    unclassified_reason_counts: dict[str, int] = {}
    cohort_cost_totals = {
        "canary_applied": _empty_cohort_costs(),
        "canary_holdout": _empty_cohort_costs(),
    }

    routing_rule_metadata: dict[str, Any] | None = None
    candidate_ids: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            candidate_ids.append(candidate_id)
        readiness = candidate.get("promotion_readiness") if isinstance(candidate.get("promotion_readiness"), dict) else {}
        if routing_rule_metadata is None and isinstance(readiness.get("routing_rule_metadata"), dict):
            routing_rule_metadata = readiness["routing_rule_metadata"]
        lifecycle = candidate.get("openai_canary_lifecycle_evidence") if isinstance(candidate.get("openai_canary_lifecycle_evidence"), dict) else {}
        counts = lifecycle.get("cohort_counts") if isinstance(lifecycle.get("cohort_counts"), dict) else {}
        for key in cohort_counts:
            cohort_counts[key] += _as_int(counts.get(key))
        error_count += _as_int(lifecycle.get("error_count"))
        fallback_count += _as_int(lifecycle.get("fallback_count"))
        retry_count += _as_int(lifecycle.get("retry_count"))
        latest = lifecycle.get("latest_observed_at")
        oldest = lifecycle.get("oldest_observed_at")
        if isinstance(latest, str) and (latest_observed_at is None or latest > latest_observed_at):
            latest_observed_at = latest
        if isinstance(oldest, str) and (oldest_observed_at is None or oldest < oldest_observed_at):
            oldest_observed_at = oldest
        stale = lifecycle.get("stale_evidence") if isinstance(lifecycle.get("stale_evidence"), dict) else {}
        if stale.get("stale"):
            stale_evidence_count += _as_int(lifecycle.get("observed_count"))
        _merge_breakdown(skipped_reason_counts, lifecycle.get("skipped_reason_breakdown"))
        _merge_breakdown(unknown_reason_counts, lifecycle.get("unknown_reason_breakdown"))
        classification = lifecycle.get("skipped_unknown_classification") if isinstance(lifecycle.get("skipped_unknown_classification"), dict) else {}
        _merge_breakdown(safe_bypass_reason_counts, classification.get("safe_bypass_reason_breakdown"))
        _merge_breakdown(unsupported_shape_reason_counts, classification.get("unsupported_shape_reason_breakdown"))
        _merge_breakdown(promotion_blocker_reason_counts, classification.get("promotion_blocker_reason_breakdown"))
        _merge_breakdown(unclassified_reason_counts, classification.get("unclassified_reason_breakdown"))
        cohort_costs = lifecycle.get("cohort_costs") if isinstance(lifecycle.get("cohort_costs"), dict) else {}
        for key in cohort_cost_totals:
            source_costs = cohort_costs.get(key)
            if isinstance(source_costs, dict):
                _add_costs(cohort_cost_totals[key], source_costs)

    applied = cohort_counts["canary_applied"]
    holdout = cohort_counts["canary_holdout"]
    safety = cohort_counts["safety_stopped"]
    skipped = cohort_counts["skipped"]
    unknown = cohort_counts["unknown"]
    observed_count = sum(cohort_counts.values())
    unsupported_shape_count = sum(unsupported_shape_reason_counts.values())
    promotion_blocker_count = sum(promotion_blocker_reason_counts.values())
    unclassified_count = sum(unclassified_reason_counts.values())
    finalized_cohort_costs = {
        "canary_applied": _finalize_cohort_costs(cohort_cost_totals["canary_applied"]),
        "canary_holdout": _finalize_cohort_costs(cohort_cost_totals["canary_holdout"]),
    }
    savings_deltas = _cohort_cost_deltas(finalized_cohort_costs)
    evidence_age_hours = None
    latest_observed = _parse_time(latest_observed_at)
    if latest_observed is not None:
        evidence_age_hours = round((datetime.now(timezone.utc) - latest_observed).total_seconds() / 3600.0, 3)

    blocker_counts: dict[str, int] = {}
    if not candidates:
        blocker_counts["no-matching-openai-routing-candidate"] = 1
    if matched_count < DEFAULT_MIN_SAMPLES:
        blocker_counts["insufficient-samples"] = matched_count
    if observed_count <= 0:
        blocker_counts["missing-canary-lifecycle-evidence"] = matched_count
    if applied <= 0:
        blocker_counts["missing-applied-coverage"] = matched_count
    if holdout <= 0:
        blocker_counts["missing-holdout-coverage"] = matched_count
    if safety:
        blocker_counts["safety-stop-observed"] = safety
    if error_count:
        blocker_counts["error-observed"] = error_count
    if fallback_count:
        blocker_counts["fallback-observed"] = fallback_count
    if retry_count:
        blocker_counts["retry-observed"] = retry_count
    if stale_evidence_count:
        blocker_counts["stale-evidence"] = stale_evidence_count
    if unclassified_count:
        blocker_counts["unclassified-canary-lifecycle-rows"] = unclassified_count
    if unsupported_shape_count:
        blocker_counts["skipped-canary-unsupported-shape"] = unsupported_shape_count
    if promotion_blocker_count:
        blocker_counts["skipped-canary-promotion-blocker"] = promotion_blocker_count
    if savings_per_1000 <= 0:
        blocker_counts["non-positive-estimated-savings"] = matched_count

    staged_review_reasons = {
        "missing-canary-lifecycle-evidence",
        "missing-applied-coverage",
        "missing-holdout-coverage",
        "unclassified-canary-lifecycle-rows",
        "skipped-canary-unsupported-shape",
    }
    hard_blockers = [reason for reason in sorted(blocker_counts) if reason not in staged_review_reasons]
    if applied > 0 and holdout > 0 and not blocker_counts and savings_per_1000 > 0:
        decision = "promote"
        next_action = "draft-openai-routing-rule"
        reason = "promotion-ready"
    elif hard_blockers:
        decision = "keep-blocked"
        next_action = "review-openai-routing-canary-blockers"
        reason = hard_blockers[0]
    elif unsupported_shape_count and not unknown and not unclassified_count:
        decision = "narrow"
        next_action = "narrow-openai-routing-canary-shape"
        reason = "skipped-canary-unsupported-shape"
    else:
        decision = "keep-staged"
        next_action = (
            "classify-openai-routing-canary-skipped-unknown"
            if unclassified_count or unsupported_shape_count
            else "collect-openai-routing-canary-evidence"
        )
        reason = sorted(blocker_counts)[0] if blocker_counts else "insufficient-promotion-evidence"

    if routing_rule_metadata is None:
        routing_rule_metadata = {
            "schema": "agentflow.openai_routing_rule_metadata.v1",
            "policy_source": "local-manual-review",
            "target_local_rule_file": "routing_rules.yaml",
            "target_local_policy_section": "routing.rules",
            "required_local_executor": "openai-routing-canary",
            "rule_preview": {
                "id": f"promote-openai-route:{endpoint}:gpt-5:{category}:to-{target_model.lower().replace('.', '-')}",
                "conditions": {
                    "model_pattern": requested_model,
                    "category": category,
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "has_tools": category.startswith("tool-"),
                },
                "action": {
                    "route_to": target_model,
                    "reason": f"promote OpenAI routing canary {source_surface}/{endpoint}/{category}",
                },
            },
        }

    blocked_count = matched_count if blocker_counts else 0
    privacy = _openai_promotion_decision_privacy()
    reason_codes = sorted(blocker_counts)
    target = {
        "provider": "openai",
        "source_surface": source_surface,
        "endpoint": endpoint,
        "category": category,
        "requested_model": requested_model,
        "target_model": target_model,
        "required_local_executor": "openai-routing-canary",
        "target_local_policy_section": "routing.rules",
        "target_local_rule_file": "routing_rules.yaml",
    }
    lifecycle = {
        "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
        "status": "matched" if observed_count else "no-openai-canary-metadata",
        "observed_count": observed_count,
        "cohort_counts": cohort_counts,
        "applied_count": applied,
        "holdout_count": holdout,
        "skipped_count": skipped,
        "unknown_count": unknown,
        "safety_stop_count": safety,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "retry_count": retry_count,
        "latest_observed_at": latest_observed_at,
        "oldest_observed_at": oldest_observed_at,
        "stale_evidence": {
            "schema": "agentflow.openai_routing_active_local_policy_stale_evidence.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "stale": stale_evidence_count > 0,
            "stale_evidence_count": stale_evidence_count,
            "age_hours": evidence_age_hours,
            "max_age_hours": DEFAULT_MAX_EVIDENCE_AGE_HOURS,
        },
        "evidence_age_hours": evidence_age_hours,
        "cohort_costs": finalized_cohort_costs,
        "savings_deltas": savings_deltas,
        "skipped_reason_breakdown": _breakdown(skipped_reason_counts),
        "unknown_reason_breakdown": _breakdown(unknown_reason_counts),
        "skipped_unknown_classification": {
            "schema": "agentflow.openai_routing_canary_skipped_unknown_classification.v1",
            "safe_bypass_count": sum(safe_bypass_reason_counts.values()),
            "unsupported_shape_count": unsupported_shape_count,
            "promotion_blocker_count": promotion_blocker_count,
            "unclassified_count": unclassified_count,
            "requires_operator_review": bool(unknown or unsupported_shape_count or promotion_blocker_count or unclassified_count),
            "safe_bypass_reason_breakdown": _breakdown(safe_bypass_reason_counts),
            "unsupported_shape_reason_breakdown": _breakdown(unsupported_shape_reason_counts),
            "promotion_blocker_reason_breakdown": _breakdown(promotion_blocker_reason_counts),
            "unclassified_reason_breakdown": _breakdown(unclassified_reason_counts),
        },
    }
    active_local_policy_rule = _active_openai_local_policy_rule(target)
    if decision == "promote" and active_local_policy_rule is not None:
        decision = "active-local-policy"
        next_action = "measure-openai-routing-rule-outcomes"
        reason = "matching-openai-routing-rule-active-in-local-policy"
        blocker_counts = {}
        reason_codes = []

    promotion_verdict = _promotion_verdict(
        decision=decision,
        blocker_counts=blocker_counts,
        applied_count=applied,
        current_routed_count=current_routed_count,
    )
    local_policy_patch = _openai_promotion_local_policy_patch(
        decision=decision,
        routing_rule_metadata=routing_rule_metadata,
        lifecycle=lifecycle,
        savings_per_1000=savings_per_1000,
    )
    rollback_metadata = _openai_promotion_rollback_metadata(
        routing_rule_metadata=routing_rule_metadata,
        reason_codes=reason_codes,
        promotion_verdict=promotion_verdict,
    )
    duplicate_suppression = _openai_promotion_duplicate_suppression(
        target=target,
        promotion_verdict=promotion_verdict,
        reason_codes=reason_codes,
    )
    candidate_set = _candidate_set_metadata(candidate_ids)
    active_local_policy_outcome = _openai_active_local_policy_outcome(
        decision=decision,
        next_action=next_action,
        reason=reason,
        target=target,
        lifecycle=lifecycle,
        active_local_policy_rule=active_local_policy_rule,
        matched_count=matched_count,
        current_routed_count=current_routed_count,
        candidate_count=len(candidates),
        candidate_set=candidate_set,
        projected_savings_usd=projected_savings_usd,
        savings_per_1000=savings_per_1000,
    )
    return {
        "schema": PROMOTION_DECISION_SCHEMA,
        "decision": decision,
        "promotion_verdict": promotion_verdict,
        "promotion_verdict_options": PROMOTION_VERDICT_OPTIONS,
        "promotion_ready": decision == "promote",
        "next_action": next_action,
        "reason": reason,
        "reason_codes": reason_codes,
        "blocker_reason_breakdown": _breakdown(blocker_counts),
        "target": target,
        "candidate_count": len(candidates),
        "candidate_set": candidate_set,
        "candidate_ids_included": False,
        "matched_count": matched_count,
        "current_routed_count": current_routed_count,
        "blocked_count": blocked_count,
        "candidate_blocked_count": candidate_blocked_count,
        "projected_savings_usd": round(projected_savings_usd, 6),
        "estimated_baseline_cost_usd": round(baseline_cost_usd, 6),
        "savings_per_1000_calls_usd": savings_per_1000,
        "estimated_savings_per_1000_calls_usd": savings_per_1000,
        "lifecycle": lifecycle,
        "quality_gates": {
            "requires_fresh_evidence": True,
            "requires_applied_coverage": True,
            "requires_holdout_coverage": True,
            "requires_classified_skipped_unknown_rows": True,
            "requires_zero_safety_stops": True,
            "requires_zero_errors": True,
            "requires_zero_fallbacks": True,
            "requires_zero_retries": True,
            "requires_positive_estimated_savings": True,
            "requires_file_backed_local_policy": True,
        },
        "routing_rule_metadata": routing_rule_metadata,
        "local_policy_patch": local_policy_patch,
        "active_local_policy_rule": active_local_policy_rule,
        "active_local_policy_outcome": active_local_policy_outcome,
        "rollback_metadata": rollback_metadata,
        "duplicate_suppression": duplicate_suppression,
        "privacy": privacy,
    }


def build_openai_routing_promotion_decision_report(
    store_obj: Any,
    limit: int = 1000,
    *,
    requested_model: str = "gpt-5.4",
    target_model: str = "gpt-5.4-mini",
    source_surface: str = "openai_responses",
    endpoint: str = "responses",
    category: str = "tool-light",
) -> dict[str, Any]:
    report = build_openai_routing_report(store_obj, limit=limit)
    decision = _build_openai_promotion_decision(
        report=report,
        requested_model=requested_model,
        target_model=target_model,
        source_surface=source_surface,
        endpoint=endpoint,
        category=category,
    )
    active_outcome = decision.get("active_local_policy_outcome") if isinstance(decision.get("active_local_policy_outcome"), dict) else {}
    return {
        "schema": PROMOTION_DECISION_REPORT_SCHEMA,
        "generated_at": utc_now(),
        "limit": report.get("limit"),
        "target": decision["target"],
        "decision": decision["decision"],
        "promotion_verdict": decision["promotion_verdict"],
        "promotion_verdict_options": PROMOTION_VERDICT_OPTIONS,
        "promotion_ready": decision["promotion_ready"],
        "summary": {
            "decision_count": 1,
            "promotion_verdict": decision["promotion_verdict"],
            "promote_count": 1 if decision["decision"] == "promote" else 0,
            "active_local_policy_count": 1 if decision["decision"] == "active-local-policy" else 0,
            "keep_staged_count": 1 if decision["decision"] == "keep-staged" else 0,
            "keep_blocked_count": 1 if decision["decision"] == "keep-blocked" else 0,
            "narrow_count": 1 if decision["decision"] == "narrow" else 0,
            "matched_count": decision["matched_count"],
            "blocked_count": decision["blocked_count"],
            "candidate_count": decision["candidate_count"],
            "active_local_policy_outcome_count": 1 if decision.get("active_local_policy_outcome") else 0,
            "active_local_policy_outcome_decision": active_outcome.get("outcome_decision"),
            "active_local_policy_next_action": active_outcome.get("deterministic_next_action"),
            "active_local_policy_gate_passed": active_outcome.get("gate_passed"),
            "active_local_policy_reason_codes": active_outcome.get("reason_codes") or [],
            "active_local_policy_rollback_action_type": (
                active_outcome.get("rollback_metadata", {}).get("rollback_action_type")
                if isinstance(active_outcome.get("rollback_metadata"), dict)
                else None
            ),
            "active_local_policy_evidence_age_hours": active_outcome.get("evidence_age_hours"),
            "active_local_policy_realized_savings_usd": active_outcome.get("realized_savings_usd"),
            "active_local_policy_applied_realized_savings_usd": active_outcome.get("applied_realized_savings_usd"),
            "active_local_policy_holdout_realized_savings_usd": active_outcome.get("holdout_realized_savings_usd"),
            "active_local_policy_applied_minus_holdout_realized_savings_avg_usd": active_outcome.get(
                "applied_minus_holdout_realized_savings_avg_usd"
            ),
            "active_local_policy_applied_minus_holdout_error_rate_delta": (
                active_outcome.get("savings_deltas", {}).get("applied_minus_holdout_error_rate_delta")
                if isinstance(active_outcome.get("savings_deltas"), dict)
                else None
            ),
            "active_local_policy_applied_minus_holdout_fallback_rate_delta": (
                active_outcome.get("savings_deltas", {}).get("applied_minus_holdout_fallback_rate_delta")
                if isinstance(active_outcome.get("savings_deltas"), dict)
                else None
            ),
            "active_local_policy_applied_minus_holdout_retry_rate_delta": (
                active_outcome.get("savings_deltas", {}).get("applied_minus_holdout_retry_rate_delta")
                if isinstance(active_outcome.get("savings_deltas"), dict)
                else None
            ),
            "candidate_ids_included": False,
            "current_routed_count": decision["current_routed_count"],
            "applied_count": decision["lifecycle"]["applied_count"],
            "holdout_count": decision["lifecycle"]["holdout_count"],
            "skipped_count": decision["lifecycle"]["skipped_count"],
            "bypassed_or_disabled_count": decision["lifecycle"]["cohort_counts"].get("bypassed_or_disabled", 0),
            "unknown_count": decision["lifecycle"]["unknown_count"],
            "safety_stop_count": decision["lifecycle"]["safety_stop_count"],
            "error_count": decision["lifecycle"]["error_count"],
            "fallback_count": decision["lifecycle"]["fallback_count"],
            "retry_count": decision["lifecycle"]["retry_count"],
            "next_action": decision["next_action"],
            "reason": decision["reason"],
            "reason_codes": decision["reason_codes"],
            "blocker_reason_breakdown": decision["blocker_reason_breakdown"],
            "skipped_reason_breakdown": decision["lifecycle"]["skipped_reason_breakdown"],
            "unknown_reason_breakdown": decision["lifecycle"]["unknown_reason_breakdown"],
            "savings_per_1000_calls_usd": decision["savings_per_1000_calls_usd"],
            "projected_savings_usd": decision["projected_savings_usd"],
            "target_local_policy_section": "routing.rules",
            "target_local_rule_file": "routing_rules.yaml",
        },
        "promotion_decision": decision,
        "decisions": [decision],
        "active_local_policy_outcomes": [decision["active_local_policy_outcome"]]
        if decision.get("active_local_policy_outcome")
        else [],
        "source_report_summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
        "privacy": _openai_promotion_decision_privacy()
        | {
            "basis": "local calls table metadata plus sanitized OpenAI routing canary lifecycle summaries only",
            "source_report_schema": report.get("schema"),
        },
    }
