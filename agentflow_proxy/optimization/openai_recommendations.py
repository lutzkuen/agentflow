from __future__ import annotations

import hashlib
import os
from typing import Any

from agentflow_proxy.optimization.managed_actions import evaluate_managed_local_actions
from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.recommendations import (
    fetch_recommendation,
    fetch_policy_decision,
    policy_decision_canary_sample,
    policy_decision_min_confidence,
    policy_decisions_enabled,
)
from agentflow_proxy.store import stable_json


OPENAI_RECOMMENDATION_DECISION_SCHEMA = "agentflow.openai_managed_recommendation_decision.v1"
OPENAI_RECOMMENDATION_MODE_ENV = "AGENTFLOW_OPENAI_RECOMMENDATION_MODE"
OPENAI_RECOMMENDATION_CANARY_FRACTION_ENV = "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_FRACTION"
OPENAI_RECOMMENDATION_CANARY_SALT_ENV = "AGENTFLOW_OPENAI_RECOMMENDATION_CANARY_SALT"

VALID_OPENAI_RECOMMENDATION_MODES = {"observe-only", "dry-run", "canary"}


def openai_recommendation_mode() -> str:
    raw = os.getenv(OPENAI_RECOMMENDATION_MODE_ENV, "observe-only").strip().lower()
    aliases = {
        "observe": "observe-only",
        "observe_only": "observe-only",
        "off": "observe-only",
        "disabled": "observe-only",
        "dryrun": "dry-run",
        "dry_run": "dry-run",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in VALID_OPENAI_RECOMMENDATION_MODES else "observe-only"


def openai_canary_fraction() -> float:
    raw = os.getenv(OPENAI_RECOMMENDATION_CANARY_FRACTION_ENV, "0.05")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.05


def openai_canary_salt() -> str:
    return os.getenv(
        OPENAI_RECOMMENDATION_CANARY_SALT_ENV,
        "agentflow-openai-managed-canary-v1",
    )


def _safe_openai_target(target_model: Any) -> tuple[str | None, str | None]:
    if not isinstance(target_model, str) or not target_model.strip():
        return None, "missing-target-model"
    target = target_model.strip()
    target_l = target.lower()
    if target_l.startswith("claude-") or any(part in target_l for part in ("haiku", "sonnet", "opus")):
        return None, "provider-mismatch"
    if target_l.startswith(("gpt-", "o", "text-", "computer-use-", "codex")) or "gpt" in target_l:
        return target, None
    return None, "unsupported-openai-target-model"


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_num(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _projection(
    *,
    current_model: str,
    target_model: str | None,
    input_tokens_est: int | None,
    recommendation_meta: dict[str, Any],
) -> dict[str, Any]:
    current_cost = None
    target_cost = None
    savings = None
    if target_model and input_tokens_est is not None:
        current_cost = estimate_cost(current_model, input_tokens_est, 0, provider="openai")
        target_cost = estimate_cost(target_model, input_tokens_est, 0, provider="openai")
        if current_cost is not None and target_cost is not None:
            savings = current_cost - target_cost
    latency_ms = (
        _int_num(recommendation_meta.get("projected_latency_ms"))
        or _int_num(recommendation_meta.get("latency_ms_p50"))
        or _int_num(recommendation_meta.get("candidate_latency_ms_p50"))
    )
    sample_count = (
        _int_num(recommendation_meta.get("matched_sample_count"))
        or _int_num(recommendation_meta.get("sample_count"))
        or _int_num(recommendation_meta.get("candidate_sample_count"))
    )
    risk = {
        "error_rate": _num(recommendation_meta.get("error_rate")),
        "retry_rate": _num(recommendation_meta.get("retry_rate")),
        "fallback_rate": _num(recommendation_meta.get("fallback_rate")),
        "latency_regression_ratio": _num(recommendation_meta.get("latency_regression_ratio")),
        "missing_fields": [],
    }
    for key in ("error_rate", "retry_rate", "fallback_rate", "latency_regression_ratio"):
        if risk[key] is None:
            risk["missing_fields"].append(key)
    return {
        "schema": "agentflow.openai_recommendation_projection.v1",
        "input_tokens_est": input_tokens_est,
        "current_model": current_model,
        "target_model": target_model,
        "current_input_cost_est_usd": current_cost,
        "target_input_cost_est_usd": target_cost,
        "projected_input_savings_usd": savings,
        "latency_ms": latency_ms,
        "latency_basis": "managed-recommendation" if latency_ms is not None else "not-provided",
        "matched_sample_count": sample_count,
        "sample_count_basis": "managed-recommendation" if sample_count is not None else "not-provided",
        "risk": risk,
    }


def _managed_target_model(meta: dict[str, Any]) -> Any:
    routing = meta.get("routing")
    if isinstance(routing, dict) and routing.get("target_model") is not None:
        return routing.get("target_model")
    return meta.get("target_model")


def _cohort(recommendation_unit: dict[str, Any], recommendation_meta: dict[str, Any], fraction: float) -> dict[str, Any]:
    policy_id = str(recommendation_meta.get("policy_id") or "unknown-policy")
    pattern_features = recommendation_unit.get("pattern_features")
    if not isinstance(pattern_features, dict):
        pattern_features = {}
    basis = {
        "policy_id": policy_id,
        "recommendation_id": recommendation_meta.get("recommendation_id"),
        "optimization_unit_id": recommendation_meta.get("optimization_unit_id"),
        "source_surface": recommendation_unit.get("source_surface"),
        "app_family": recommendation_unit.get("app_family"),
        "pattern_hash": pattern_features.get("pattern_hash"),
        "grouping_identifiers": recommendation_unit.get("grouping_identifiers") or {},
    }
    digest = hashlib.sha256(f"{openai_canary_salt()}:{stable_json(basis)}".encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    selected = score < fraction
    return {
        "schema": "agentflow.openai_recommendation_canary.v1",
        "enabled": True,
        "fraction": fraction,
        "salt": openai_canary_salt(),
        "unit": "optimization_unit",
        "bucket": round(score, 8),
        "threshold": fraction,
        "selected": selected,
        "cohort": "canary_applied" if selected else "canary_holdout",
        "cohort_key_hash": f"sha256:{digest}",
    }


def _base_decision(
    *,
    mode: str,
    current_model: str,
    input_tokens_est: int | None,
    recommendation_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(recommendation_meta or {})
    return {
        **meta,
        "schema": OPENAI_RECOMMENDATION_DECISION_SCHEMA,
        "enabled": mode != "observe-only",
        "mode": mode,
        "provider": "openai",
        "status": meta.get("status") or "skipped",
        "applied": False,
        "changed_model": False,
        "fallback": "local-policy",
        "local_model_before_recommendation": current_model,
        "projection": _projection(
            current_model=current_model,
            target_model=_managed_target_model(meta) if isinstance(_managed_target_model(meta), str) else None,
            input_tokens_est=input_tokens_est,
            recommendation_meta=meta,
        ),
        "raw_payload_included": False,
    }


async def evaluate_openai_recommendation(
    *,
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    recommendation_unit: dict[str, Any],
    input_tokens_est: int | None,
) -> dict[str, Any]:
    """Fetch and apply an OpenAI managed recommendation with local fallback by default."""
    decision = await fetch_openai_recommendation_decision(
        recommendation_unit=recommendation_unit,
        current_model=str(body.get("model") or routing_meta.get("routed_model") or ""),
        input_tokens_est=input_tokens_est,
    )
    return apply_openai_recommendation_decision(
        body=body,
        routing_meta=routing_meta,
        decision=decision,
    )


async def fetch_openai_recommendation_decision(
    *,
    recommendation_unit: dict[str, Any],
    current_model: str,
    input_tokens_est: int | None,
) -> dict[str, Any]:
    """Fetch and evaluate an OpenAI managed recommendation without mutating the provider request."""
    mode = openai_recommendation_mode()
    if mode == "observe-only" and not policy_decisions_enabled():
        return _base_decision(mode=mode, current_model=current_model, input_tokens_est=input_tokens_est)

    fetched = await (fetch_policy_decision(recommendation_unit) if policy_decisions_enabled() else fetch_recommendation(recommendation_unit))
    decision = _base_decision(
        mode=mode,
        current_model=current_model,
        input_tokens_est=input_tokens_est,
        recommendation_meta=fetched,
    )
    if fetched.get("status") != "received":
        decision.update({
            "status": fetched.get("status") or "skipped",
            "apply_reason": fetched.get("reason") or "recommendation-not-received",
            "lifecycle_event": "fallback",
        })
        return decision

    target_model, target_error = _safe_openai_target(_managed_target_model(fetched))
    decision["projection"] = _projection(
        current_model=current_model,
        target_model=target_model,
        input_tokens_est=input_tokens_est,
        recommendation_meta=fetched,
    )
    has_local_action_sections = any(isinstance(fetched.get(section), dict) for section in ("crunch", "cache"))
    if target_error and not has_local_action_sections:
        decision.update({
            "status": "skipped",
            "apply_reason": target_error,
            "lifecycle_event": "fallback",
        })
        return decision
    if fetched.get("replacement_prompt_present"):
        decision.update({
            "status": "skipped",
            "apply_reason": "prompt-shaping-not-locally-representable",
            "replacement_prompt_applied": False,
            "lifecycle_event": "fallback",
        })
        return decision
    if fetched.get("policy_decision_schema") and fetched.get("routing_status") != "recommended":
        decision.update({
            "status": "skipped",
            "apply_reason": "routing-not-recommended",
            "lifecycle_event": "fallback",
        })
        return decision
    if fetched.get("policy_decision_schema"):
        confidence = float(fetched.get("confidence") or 0.0)
        probability = fetched.get("route_down_probability")
        if isinstance(probability, (int, float)):
            confidence = min(confidence, float(probability))
        if confidence < policy_decision_min_confidence():
            decision.update({
                "status": "skipped",
                "apply_reason": "routing-predictor-confidence-too-low",
                "lifecycle_event": "fallback",
            })
            return decision
        if not isinstance(fetched.get("model_artifact_version"), str) or not fetched.get("model_artifact_version"):
            decision.update({
                "status": "skipped",
                "apply_reason": "routing-predictor-model-version-missing",
                "lifecycle_event": "fallback",
            })
            return decision
        recommended_mode = str(fetched.get("recommended_mode") or "observe").lower()
        decision["local_policy_decision_mode"] = recommended_mode
        decision["mode"] = f"policy-decision-{recommended_mode}"
        if recommended_mode in {"observe", "shadow"}:
            decision.update({
                "enabled": False,
                "status": recommended_mode,
                "apply_reason": f"{recommended_mode}-only",
                "would_change_model": target_model is not None and target_model != current_model,
                "would_route_model": target_model,
                "local_action_taken": recommended_mode,
                "lifecycle_event": recommended_mode,
            })
            return decision
        if recommended_mode == "hold":
            decision.update({
                "status": "hold",
                "apply_reason": "routing-predictor-hold",
                "would_change_model": target_model is not None and target_model != current_model,
                "would_route_model": target_model,
                "local_action_taken": "noop",
                "lifecycle_event": "holdout",
            })
            return decision
        if recommended_mode != "canary":
            decision.update({
                "status": "skipped",
                "apply_reason": f"recommended-mode-{recommended_mode or 'missing'}",
                "lifecycle_event": "fallback",
            })
            return decision
    if target_model == current_model:
        decision.update({
            "status": "noop",
            "apply_reason": "target-model-already-selected",
            "lifecycle_event": "dry_run" if mode == "dry-run" else "noop",
        })
        return decision

    if mode == "dry-run":
        decision["local_actions"] = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "dry-run",
            "apply_reason": "dry-run-local-fallback",
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "lifecycle_event": "dry_run",
        })
        return decision

    fraction = openai_canary_fraction()
    canary = (
        policy_decision_canary_sample(fetched, current_model=current_model, target_model=target_model)
        if fetched.get("policy_decision_schema") and target_model
        else _cohort(recommendation_unit, fetched, fraction)
    )
    decision["canary"] = canary
    if fetched.get("policy_decision_schema"):
        decision["enabled"] = bool(canary.get("enabled"))
    if not canary["selected"]:
        decision["local_actions"] = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "holdout",
            "apply_reason": "canary-holdout",
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "lifecycle_event": "holdout",
        })
        return decision

    local_actions = evaluate_managed_local_actions(
        fetched,
        provider="openai",
        current_model=current_model,
        source_surface=recommendation_unit.get("source_surface"),
        application_enabled=True,
    )
    if local_actions.get("status") == "skipped":
        decision.update({
            "status": "skipped",
            "apply_reason": local_actions.get("apply_reason") or "local-action-validation-failed",
            "local_actions": local_actions,
            "lifecycle_event": "fallback",
        })
        return decision
    if target_error:
        decision["routing"] = {
            "status": "skipped",
            "applied": False,
            "target_model": _managed_target_model(fetched) if isinstance(_managed_target_model(fetched), str) else None,
            "apply_reason": target_error,
        }

    decision.update({
        "status": "selected",
        "selected_for_local_application": True,
        "fallback": None,
        "apply_reason": "canary-selected",
        "target_model_normalized": target_model,
        "local_actions": local_actions,
        "lifecycle_event": "canary_selected",
    })
    return decision


def apply_openai_recommendation_decision(
    *,
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Apply a preflight managed decision to the locally mutated OpenAI request when safe."""
    applied = dict(decision)
    target_model = applied.get("target_model_normalized")
    if applied.get("apply_reason") != "canary-selected" or not isinstance(target_model, str):
        local_actions = applied.get("local_actions")
        if isinstance(local_actions, dict):
            routing_meta["managed_local_actions"] = local_actions
        return applied

    current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
    applied["local_model_at_application"] = current_model
    if target_model == current_model:
        applied.update({
            "status": "noop",
            "applied": False,
            "changed_model": False,
            "apply_reason": "target-model-already-selected-locally",
            "lifecycle_event": "noop",
        })
        return applied

    body["model"] = target_model
    routing_meta["routed_model"] = target_model
    routing_meta["final_policy_source"] = "managed-recommended"
    routing_meta["managed_policy_id"] = applied.get("policy_id")
    routing_meta["managed_reason"] = applied.get("reason")
    applied.update({
        "status": "applied",
        "applied": True,
        "changed_model": True,
        "fallback": None,
        "apply_reason": "canary-selected",
        "lifecycle_event": "canary_applied",
    })
    local_actions = applied.get("local_actions")
    if isinstance(local_actions, dict):
        local_actions.setdefault("routing", {})
        if isinstance(local_actions["routing"], dict):
            local_actions["routing"].update({
                "status": "applied",
                "applied": True,
                "target_model": target_model,
                "apply_reason": "provider-compatible-local-route",
            })
        routing_meta["managed_local_actions"] = local_actions
    return applied
