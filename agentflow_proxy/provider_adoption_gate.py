from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


SCHEMA = "agentflow.provider_adoption_regression_gate.v1"

RISK_STATUSES = {"abandoned", "orphan-result", "unknown"}


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


def _public_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text and text[0].isalnum() and len(text) <= 80 and all(ch.isalnum() or ch in ".:-" for ch in text):
        return text
    return fallback


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _empty() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "window_count": 0,
        "fulfilled_count": 0,
        "abandoned_count": 0,
        "orphan_result_count": 0,
        "unknown_count": 0,
        "risk_window_count": 0,
        "tool_use_count": 0,
        "tool_result_count": 0,
        "_status_counts": Counter(),
        "_relationship_counts": Counter(),
    }


def _finalize(raw: dict[str, Any]) -> dict[str, Any]:
    total = _as_int(raw.get("window_count"))
    status_counts = raw.get("_status_counts") if isinstance(raw.get("_status_counts"), Counter) else Counter()
    relationship_counts = raw.get("_relationship_counts") if isinstance(raw.get("_relationship_counts"), Counter) else Counter()
    fulfilled = _as_int(raw.get("fulfilled_count"))
    abandoned = _as_int(raw.get("abandoned_count"))
    orphan = _as_int(raw.get("orphan_result_count"))
    unknown = _as_int(raw.get("unknown_count"))
    risk = _as_int(raw.get("risk_window_count"))
    return {
        "sample_count": _as_int(raw.get("sample_count")),
        "window_count": total,
        "fulfilled_count": fulfilled,
        "abandoned_count": abandoned,
        "orphan_result_count": orphan,
        "unknown_count": unknown,
        "risk_window_count": risk,
        "tool_use_count": _as_int(raw.get("tool_use_count")),
        "tool_result_count": _as_int(raw.get("tool_result_count")),
        "fulfilled_rate": _rate(fulfilled, total),
        "abandonment_rate": _rate(abandoned, total),
        "orphan_result_rate": _rate(orphan, total),
        "unknown_rate": _rate(unknown, total),
        "risk_rate": _rate(risk, total),
        "status_counts": dict(sorted(status_counts.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
    }


def provider_adoption_thresholds(
    *,
    min_fulfilled_samples: int = 1,
    max_applied_abandonment_rate: float = 0.02,
    max_applied_orphan_result_rate: float = 0.02,
    max_applied_risk_rate: float = 0.05,
    max_applied_vs_holdout_risk_rate_delta: float = 0.02,
) -> dict[str, Any]:
    return {
        "min_provider_adoption_fulfilled_samples": max(0, _as_int(min_fulfilled_samples)),
        "max_applied_abandonment_rate": round(float(max_applied_abandonment_rate), 6),
        "max_applied_orphan_result_rate": round(float(max_applied_orphan_result_rate), 6),
        "max_applied_provider_adoption_risk_rate": round(float(max_applied_risk_rate), 6),
        "max_applied_vs_holdout_provider_adoption_risk_rate_delta": round(float(max_applied_vs_holdout_risk_rate_delta), 6),
    }


def _cohort_name(value: Any) -> str:
    text = str(value or "unknown").strip()
    if text in {"canary_applied", "applied"}:
        return "applied"
    if text in {"canary_holdout", "holdout"}:
        return "holdout"
    return _public_label(text, "unknown")


def build_provider_adoption_gate(
    observations: Iterable[dict[str, Any]],
    *,
    thresholds: dict[str, Any] | None = None,
    block_on_missing: bool = False,
    block_on_insufficient: bool = False,
) -> dict[str, Any]:
    threshold_values = thresholds or provider_adoption_thresholds()
    cohorts = {"applied": _empty(), "holdout": _empty(), "other": _empty()}
    evidence_samples = 0
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        cohort = _cohort_name(observation.get("cohort"))
        bucket = cohorts.get(cohort, cohorts["other"])
        bucket["sample_count"] += 1
        windows = observation.get("provider_adoption_windows")
        if not isinstance(windows, list):
            windows = []
        if windows:
            evidence_samples += 1
        for window in windows:
            if not isinstance(window, dict):
                continue
            status = _public_label(window.get("status"), "unknown")
            relationship = _public_label(window.get("relationship"), "unknown")
            bucket["window_count"] += 1
            bucket["tool_use_count"] += _as_int(window.get("tool_use_count"))
            bucket["tool_result_count"] += _as_int(window.get("tool_result_count"))
            bucket["_status_counts"][status] += 1
            bucket["_relationship_counts"][relationship] += 1
            if status == "fulfilled":
                bucket["fulfilled_count"] += 1
            elif status == "abandoned":
                bucket["abandoned_count"] += 1
            elif status == "orphan-result":
                bucket["orphan_result_count"] += 1
            elif status == "unknown":
                bucket["unknown_count"] += 1
            if status in RISK_STATUSES or status in {"orphan-result"}:
                bucket["risk_window_count"] += 1

    finalized = {key: _finalize(value) for key, value in cohorts.items()}
    applied = finalized["applied"]
    holdout = finalized["holdout"]
    deltas = {
        "applied_minus_holdout_abandonment_rate": round(_as_float(applied.get("abandonment_rate")) - _as_float(holdout.get("abandonment_rate")), 6),
        "applied_minus_holdout_orphan_result_rate": round(_as_float(applied.get("orphan_result_rate")) - _as_float(holdout.get("orphan_result_rate")), 6),
        "applied_minus_holdout_risk_rate": round(_as_float(applied.get("risk_rate")) - _as_float(holdout.get("risk_rate")), 6),
    }

    reason_codes: list[str] = []
    warning_codes: list[str] = []
    if not evidence_samples:
        code = "provider-adoption-evidence-missing"
        if block_on_missing:
            reason_codes.append(code)
        else:
            warning_codes.append(code)
    elif _as_int(applied.get("fulfilled_count")) < _as_int(threshold_values.get("min_provider_adoption_fulfilled_samples")):
        code = "provider-adoption-samples-insufficient"
        if block_on_insufficient:
            reason_codes.append(code)
        else:
            warning_codes.append(code)

    if _as_float(applied.get("abandonment_rate")) > _as_float(threshold_values.get("max_applied_abandonment_rate")):
        reason_codes.append("provider-adoption-regression")
    if _as_float(applied.get("orphan_result_rate")) > _as_float(threshold_values.get("max_applied_orphan_result_rate")):
        reason_codes.append("provider-adoption-regression")
    if _as_float(applied.get("risk_rate")) > _as_float(threshold_values.get("max_applied_provider_adoption_risk_rate")):
        reason_codes.append("provider-adoption-regression")
    if _as_float(deltas.get("applied_minus_holdout_risk_rate")) > _as_float(threshold_values.get("max_applied_vs_holdout_provider_adoption_risk_rate_delta")):
        reason_codes.append("provider-adoption-regression")

    reason_codes = sorted(set(reason_codes))
    warning_codes = sorted(set(warning_codes))
    return {
        "schema": SCHEMA,
        "status": "blocked" if reason_codes else "warning" if warning_codes else "passed",
        "blocking": bool(reason_codes),
        "reason_codes": reason_codes,
        "warning_codes": warning_codes,
        "thresholds": threshold_values,
        "cohorts": finalized,
        "applied_vs_holdout_deltas": deltas,
        "evidence_sample_count": evidence_samples,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "tool_ids_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }


def provider_adoption_windows_by_call(store_obj: Any, call_ids: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    ids = [str(call_id) for call_id in call_ids if call_id]
    if not ids or not hasattr(store_obj, "provider_tool_adoption_windows_for_call_ids"):
        return {}
    return store_obj.provider_tool_adoption_windows_for_call_ids(ids)
