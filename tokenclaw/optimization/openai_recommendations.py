from __future__ import annotations

import os
from typing import Any

from tokenclaw.action_executor import ActionExecutor
from tokenclaw.optimization.managed_actions import evaluate_managed_local_actions
from tokenclaw.pricing import estimate_cost
from tokenclaw.recommendations import (
    fetch_recommendation,
    fetch_policy_decision,
    policy_decisions_enabled,
)


OPENAI_RECOMMENDATION_DECISION_SCHEMA = "tokenclaw.openai_managed_recommendation_decision.v1"
OPENAI_RECOMMENDATION_MODE_ENV = "TOKENCLAW_OPENAI_RECOMMENDATION_MODE"

VALID_OPENAI_RECOMMENDATION_MODES = {"observe-only", "dry-run", "canary", "apply"}
LIVE_POLICY_DECISION_MODES = {"live", "enforced", "promoted", "promotion", "route_to"}


def openai_recommendation_mode() -> str:
    raw = os.getenv(OPENAI_RECOMMENDATION_MODE_ENV, "observe-only").strip().lower()
    aliases = {
        "observe": "observe-only",
        "observe_only": "observe-only",
        "off": "observe-only",
        "disabled": "observe-only",
        "dryrun": "dry-run",
        "dry_run": "dry-run",
        "live": "apply",
        "active": "apply",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in VALID_OPENAI_RECOMMENDATION_MODES else "observe-only"


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
        "schema": "tokenclaw.openai_recommendation_projection.v1",
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
    proposal = routing.get("route_proposal") if isinstance(routing, dict) else None
    if isinstance(proposal, dict) and proposal.get("target_model") is not None:
        return proposal.get("target_model")
    if meta.get("route_to") is not None:
        return meta.get("route_to")
    return meta.get("target_model")


def _routing_section(meta: dict[str, Any]) -> dict[str, Any]:
    routing = meta.get("routing")
    return routing if isinstance(routing, dict) else {}


def _route_proposal(meta: dict[str, Any]) -> dict[str, Any]:
    routing = _routing_section(meta)
    proposal = routing.get("route_proposal")
    return proposal if isinstance(proposal, dict) else {}


def _route_selected(meta: dict[str, Any]) -> bool | None:
    proposal = _route_proposal(meta)
    routing = _routing_section(meta)
    for source in (proposal, routing, meta):
        value = source.get("route_selected") if isinstance(source, dict) else None
        if isinstance(value, bool):
            return value
    membership = proposal.get("server_selected_canary_membership")
    if isinstance(membership, bool) and _traffic_treatment(meta) == "canary":
        return membership
    return None


def _traffic_treatment(meta: dict[str, Any]) -> str:
    proposal = _route_proposal(meta)
    routing = _routing_section(meta)
    for source in (proposal, routing, meta):
        value = source.get("traffic_treatment") if isinstance(source, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    mode = str(meta.get("recommended_mode") or "").strip().lower()
    return "live" if mode in LIVE_POLICY_DECISION_MODES else mode


def _local_application_enabled(mode: str) -> bool:
    return mode not in {"observe-only", "dry-run"}


def _server_selected_profile_action(meta: dict[str, Any]) -> bool:
    for name in ("crunch", "cache"):
        section = meta.get(name)
        if not isinstance(section, dict):
            continue
        status = str(section.get("status") or "").strip().lower()
        if status in {"recommended", "selected", "applied", "configured"}:
            return True
        if section.get("profile") not in (None, "", "off"):
            return True
    return False


def _has_unsupported_action_request(local_actions: dict[str, Any]) -> bool:
    for item in local_actions.get("unsupported_actions") or []:
        if not isinstance(item, dict):
            continue
        if item.get("reason") == "unsupported-action-type":
            return True
        if item.get("section") == "actions":
            return True
    return False


def _server_execution_selection(meta: dict[str, Any]) -> dict[str, Any]:
    """Interpret the server-selected execution state without computing policy strategy locally."""
    treatment = _traffic_treatment(meta)
    selected = _route_selected(meta)
    mode = str(meta.get("recommended_mode") or "").strip().lower()
    if meta.get("selected_for_local_application") is True:
        selected = True
        treatment = treatment or "live"
    if meta.get("selected_for_shadow_evaluation") is True:
        return {"state": "shadow", "reason": "server-selected-shadow", "traffic_treatment": treatment or "shadow"}

    if treatment in {"observe", "none"} or mode == "observe":
        return {"state": "held", "reason": "observe-only", "traffic_treatment": treatment or "observe"}
    if treatment == "shadow" or mode == "shadow":
        return {"state": "shadow", "reason": "shadow-only", "traffic_treatment": "shadow"}
    if treatment in {"hold", "held", "holdout"} or mode == "hold":
        reason = "server-canary-holdout" if treatment == "holdout" else "server-held"
        return {"state": "held", "reason": reason, "traffic_treatment": treatment or "hold"}
    if treatment in {"live", "canary", "route_to"} or mode in LIVE_POLICY_DECISION_MODES:
        if selected is False:
            reason = "server-canary-holdout" if treatment == "canary" else "server-route-not-selected"
            return {"state": "held", "reason": reason, "traffic_treatment": treatment}
        if selected is True or treatment in {"live", "route_to"} or mode in LIVE_POLICY_DECISION_MODES:
            return {"state": "apply", "reason": f"server-selected-{treatment or mode}", "traffic_treatment": treatment or mode}
        return {"state": "skipped", "reason": "server-route-selection-missing", "traffic_treatment": treatment}
    if _server_selected_profile_action(meta):
        return {"state": "apply", "reason": "server-selected-local-profile", "traffic_treatment": treatment or "profile"}
    if meta.get("policy_decision_schema"):
        return {"state": "skipped", "reason": f"server-action-state-{treatment or mode or 'missing'}", "traffic_treatment": treatment}
    return {"state": "skipped", "reason": "server-action-state-missing", "traffic_treatment": treatment}


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
    request_facts: dict[str, Any] | None = None,
    current_model: str,
    input_tokens_est: int | None,
) -> dict[str, Any]:
    """Fetch and evaluate an OpenAI managed recommendation without mutating the provider request."""
    mode = openai_recommendation_mode()
    if mode == "observe-only" and not policy_decisions_enabled():
        return _base_decision(mode=mode, current_model=current_model, input_tokens_est=input_tokens_est)

    fetched = await (
        fetch_policy_decision(recommendation_unit, request_facts=request_facts)
        if policy_decisions_enabled()
        else fetch_recommendation(recommendation_unit)
    )
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

    original_target = _managed_target_model(fetched)
    target_model, target_error = _safe_openai_target(original_target)
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
    validation_actions = evaluate_managed_local_actions(
        fetched,
        provider="openai",
        current_model=current_model,
        source_surface=recommendation_unit.get("source_surface"),
        application_enabled=False,
    )
    if validation_actions.get("status") == "skipped":
        decision.update({
            "status": "vetoed",
            "apply_reason": validation_actions.get("apply_reason") or "local-action-validation-failed",
            "local_actions": validation_actions,
            "lifecycle_event": "vetoed",
        })
        return decision
    if _has_unsupported_action_request(validation_actions):
        decision.update({
            "status": "vetoed",
            "apply_reason": "unsupported-action-type",
            "local_actions": validation_actions,
            "lifecycle_event": "vetoed",
        })
        return decision
    if fetched.get("policy_decision_schema") and fetched.get("routing_status") != "recommended" and not has_local_action_sections:
        decision.update({
            "status": "held",
            "apply_reason": "routing-not-recommended",
            "lifecycle_event": "held",
        })
        return decision
    server_selection = _server_execution_selection(fetched)
    decision["server_action_selection"] = server_selection
    decision["server_traffic_treatment"] = server_selection.get("traffic_treatment")
    recommended_mode = str(fetched.get("recommended_mode") or "").strip().lower()
    if recommended_mode:
        decision["server_recommended_mode"] = recommended_mode
        decision["local_policy_decision_mode"] = recommended_mode
        decision["mode"] = f"policy-decision-{recommended_mode}" if fetched.get("policy_decision_schema") else mode
    if target_model == current_model:
        decision.update({
            "status": "held",
            "apply_reason": "target-model-already-selected",
            "lifecycle_event": "held",
        })
        return decision

    if not _local_application_enabled(mode):
        decision["local_actions"] = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "dry-run" if mode == "dry-run" else "held",
            "apply_reason": "dry-run-local-gate" if mode == "dry-run" else "observe-only-local-gate",
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "local_action_taken": "dry-run" if mode == "dry-run" else "held",
            "lifecycle_event": "dry_run" if mode == "dry-run" else "held",
        })
        return decision

    if server_selection["state"] == "skipped":
        decision["local_actions"] = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "skipped",
            "apply_reason": server_selection["reason"],
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "local_action_taken": "skipped",
            "lifecycle_event": "skipped",
        })
        return decision

    if server_selection["state"] == "held":
        decision["local_actions"] = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "held",
            "apply_reason": server_selection["reason"],
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "local_action_taken": "held",
            "lifecycle_event": "held",
        })
        return decision

    if server_selection["state"] == "shadow":
        local_actions = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=False,
        )
        decision.update({
            "status": "held",
            "selected_for_local_application": False,
            "selected_for_shadow_evaluation": target_model is not None,
            "fallback": None if target_model is not None else "local-policy",
            "apply_reason": server_selection["reason"],
            "target_model_normalized": target_model,
            "would_change_model": target_model is not None and target_model != current_model,
            "would_route_model": target_model,
            "local_actions": local_actions,
            "local_action_taken": "shadow",
            "lifecycle_event": "shadow",
        })
        return decision

    if server_selection["state"] == "apply":
        local_actions = evaluate_managed_local_actions(
            fetched,
            provider="openai",
            current_model=current_model,
            source_surface=recommendation_unit.get("source_surface"),
            application_enabled=True,
        )
        if local_actions.get("status") == "skipped":
            decision.update({
                "status": "vetoed",
                "apply_reason": local_actions.get("apply_reason") or "local-action-validation-failed",
                "local_actions": local_actions,
                "lifecycle_event": "vetoed",
            })
            return decision
        decision.update({
            "status": "selected",
            "selected_for_local_application": True,
            "selected_for_shadow_evaluation": False,
            "fallback": None,
            "apply_reason": server_selection["reason"],
            "target_model_normalized": target_model,
            "local_actions": local_actions,
            "local_action_taken": "selected",
            "lifecycle_event": "selected",
        })
        return decision
    decision.update({
        "status": "skipped",
        "apply_reason": "server-action-state-missing",
        "lifecycle_event": "skipped",
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
    executor = ActionExecutor(provider="openai")
    if applied.get("selected_for_shadow_evaluation") is True and isinstance(target_model, str):
        current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
        execution = executor.execute(
            body=body,
            routing_meta=routing_meta,
            decision=applied,
            application_enabled=True,
            shadow_only=True,
            source_surface=str(routing_meta.get("source_surface") or ""),
        )
        if execution.get("status") == "vetoed":
            applied.update({
                "status": "vetoed",
                "applied": False,
                "changed_model": False,
                "fallback": "local-policy",
                "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
                "local_action_taken": "vetoed",
                "local_actions": execution,
                "action_executor": execution,
                "lifecycle_event": "vetoed",
            })
            routing_meta["managed_local_actions"] = execution
            return applied
        applied["local_model_at_application"] = current_model
        applied.update({
            "status": "held",
            "applied": False,
            "changed_model": False,
            "fallback": "local-policy",
            "apply_reason": applied.get("apply_reason") or "shadow-only",
            "local_action_taken": "shadow",
            "would_route_model": target_model,
            "shadow_model": target_model,
            "shadow_only": True,
            "live_promotion_required": True,
            "lifecycle_event": "held",
            "action_executor": execution,
        })
        routing_meta["managed_route_candidate_model"] = target_model
        routing_meta["managed_route_candidate_reason"] = applied.get("reason")
        routing_meta["managed_route_recommended_mode"] = applied.get("server_recommended_mode") or "shadow"
        routing_meta["managed_route_shadow_only"] = True
        routing_meta["managed_policy_id"] = applied.get("policy_id")
        routing_meta["managed_reason"] = applied.get("reason")
        local_actions = applied.get("local_actions")
        if isinstance(local_actions, dict):
            local_actions.setdefault("routing", {})
            if isinstance(local_actions["routing"], dict):
                local_actions["routing"].update({
                    "status": "held",
                    "applied": False,
                    "target_model": target_model,
                    "apply_reason": applied.get("apply_reason") or "shadow-only",
                })
            routing_meta["managed_local_actions"] = local_actions
        else:
            routing_meta["managed_local_actions"] = execution
        return applied

    if not applied.get("selected_for_local_application"):
        local_actions = applied.get("local_actions")
        if isinstance(local_actions, dict):
            routing_meta["managed_local_actions"] = local_actions
        return applied

    current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
    execution = executor.execute(
        body=body,
        routing_meta=routing_meta,
        decision=applied,
        application_enabled=True,
        shadow_only=False,
        source_surface=str(routing_meta.get("source_surface") or ""),
    )
    applied["local_model_at_application"] = current_model
    if execution.get("status") == "vetoed":
        applied.update({
            "status": "vetoed",
            "applied": False,
            "changed_model": False,
            "fallback": "local-policy",
            "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
            "local_action_taken": "vetoed",
            "local_actions": execution,
            "action_executor": execution,
            "lifecycle_event": "vetoed",
        })
        routing_meta["managed_local_actions"] = execution
        return applied
    if isinstance(target_model, str) and (target_model == current_model or execution.get("status") == "noop"):
        applied.update({
            "status": "held",
            "applied": False,
            "changed_model": False,
            "apply_reason": "target-model-already-selected-locally",
            "local_action_taken": "held",
            "lifecycle_event": "held",
            "action_executor": execution,
        })
        return applied

    applied.update({
        "status": "applied" if execution.get("status") == "applied" else "held",
        "applied": execution.get("status") == "applied",
        "changed_model": bool(execution.get("changed_model")),
        "fallback": execution.get("fallback"),
        "apply_reason": execution.get("apply_reason"),
        "local_action_taken": "applied" if execution.get("status") == "applied" else "held",
        "lifecycle_event": "applied" if execution.get("status") == "applied" else "held",
        "action_executor": execution,
    })
    local_actions = applied.get("local_actions")
    if isinstance(local_actions, dict):
        local_actions.setdefault("routing", {})
        if isinstance(local_actions["routing"], dict):
                local_actions["routing"].update({
                    "status": execution.get("routing", {}).get("status", "applied") if isinstance(execution.get("routing"), dict) else "applied",
                    "applied": bool(execution.get("changed_model")),
                    "target_model": target_model,
                    "apply_reason": "provider-compatible-local-route" if execution.get("changed_model") else execution.get("apply_reason"),
            })
        routing_meta["managed_local_actions"] = local_actions
    else:
        routing_meta["managed_local_actions"] = execution
    return applied
