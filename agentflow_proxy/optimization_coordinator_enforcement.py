from __future__ import annotations

import os
from typing import Any

from agentflow_proxy.optimization_coordinator import build_optimization_coordinator


ENFORCEMENT_ENV = "AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT"
SCHEMA = "agentflow.optimization_coordinator_enforcement.v1"
SUPPRESSION_REASON = "conflicts-with-coordinator-selection"
MANAGED_SOURCES = {"managed-recommended", "managed-enforced"}
CRUNCH_FAMILY_KEYS = {
    "old_context_summary": "old_context_summarization",
    "repeated_scaffold_crunch": "codex_repeated_scaffolding",
    "terminal_output_compaction": "terminal_output_compaction",
    "instruction_section_deduplication": "instruction_section_deduplication",
    "prompt_role": "instruction_section_deduplication",
}


def optimization_coordinator_enforcement_enabled() -> bool:
    return os.getenv(ENFORCEMENT_ENV, "0").strip().lower() not in {"", "0", "false", "no", "off"}


def _managed_source(meta: Any) -> bool:
    if not isinstance(meta, dict):
        return False
    if meta.get("policy_source") in MANAGED_SOURCES:
        return True
    for key in ("managed_profile", "managed_recommendation", "cache_replay_canary", "pattern_rule"):
        value = meta.get(key)
        if isinstance(value, dict) and value.get("policy_source") in MANAGED_SOURCES:
            return True
    return False


def _family_managed_meta(
    family: str,
    *,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
) -> dict[str, Any] | None:
    if family == "routing":
        managed = routing_meta.get("managed_recommendation")
        if isinstance(managed, dict) and (managed.get("applied") or _managed_source(managed) or routing_meta.get("final_policy_source") in MANAGED_SOURCES):
            return managed
        if routing_meta.get("final_policy_source") in MANAGED_SOURCES:
            return routing_meta
        return None
    if family == "cache_replay":
        if _managed_source(cache_meta):
            return cache_meta
        return None
    key = CRUNCH_FAMILY_KEYS.get(family)
    if key:
        meta = crunch_meta.get(key)
        if isinstance(meta, dict) and _managed_source(meta):
            return meta
    if family in {"old_context_summary", "repeated_scaffold_crunch", "terminal_output_compaction"} and _managed_source(crunch_meta):
        return crunch_meta
    return None


def _append_reason(meta: dict[str, Any], reason: str) -> None:
    raw = meta.get("reason_codes")
    reasons = [str(item) for item in raw] if isinstance(raw, list) else []
    if reason not in reasons:
        reasons.append(reason)
    meta["reason_codes"] = reasons


def _suppression_meta(family: str, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "suppressed",
        "family": family,
        "selected_family": decision.get("selected_action_family") or decision.get("selected_family") or "none",
        "reason_codes": [SUPPRESSION_REASON],
        "decision_hash": decision.get("decision_hash"),
        "metadata_only": True,
        "provider_body_included": False,
    }


def _suppress_family(
    family: str,
    *,
    decision: dict[str, Any],
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    provider_body: dict[str, Any] | None,
    local_routed_model: str | None,
) -> bool:
    meta = _family_managed_meta(family, routing_meta=routing_meta, crunch_meta=crunch_meta, cache_meta=cache_meta)
    if meta is None:
        return False
    suppression = _suppression_meta(family, decision)
    meta["optimization_coordinator_suppression"] = suppression
    _append_reason(meta, SUPPRESSION_REASON)
    if family == "routing":
        managed = routing_meta.get("managed_recommendation")
        if isinstance(managed, dict):
            managed.update({
                "applied": False,
                "changed_model": False,
                "apply_reason": SUPPRESSION_REASON,
                "fallback": "local-policy",
                "local_action_taken": "coordinator_suppressed",
            })
        if provider_body is not None and local_routed_model:
            provider_body["model"] = local_routed_model
            routing_meta["routed_model"] = local_routed_model
            routing_meta["coordinator_restored_model"] = local_routed_model
    elif family == "cache_replay":
        cache_meta.update({
            "status": "skipped",
            "reason": SUPPRESSION_REASON,
            "exact_enabled": False,
            "semantic_enabled": False,
        })
    else:
        meta.update({
            "status": "suppressed",
            "applied": False,
            "changed": False,
            "reason": SUPPRESSION_REASON,
        })
        crunch_meta.setdefault("optimization_coordinator_suppressed_crunch", [])
        if isinstance(crunch_meta["optimization_coordinator_suppressed_crunch"], list):
            crunch_meta["optimization_coordinator_suppressed_crunch"].append(family)
    return True


def _attach(meta: dict[str, Any], decision: dict[str, Any], enforcement: dict[str, Any]) -> None:
    meta["optimization_coordinator"] = decision
    meta["optimization_coordinator_enforcement"] = enforcement


def enforce_optimization_coordinator(
    *,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    provider: str,
    source_surface: str,
    endpoint: str,
    requested_model: str | None = None,
    routed_model: str | None = None,
    input_tokens_est: int | None = None,
    category: str | None = None,
    stream: bool = False,
    session_id: str | None = None,
    provider_body: dict[str, Any] | None = None,
    local_routed_model: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    active = optimization_coordinator_enforcement_enabled() if enabled is None else bool(enabled)
    if not active:
        return {"schema": SCHEMA, "enabled": False, "status": "skipped", "reason": "disabled"}

    try:
        row = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "requested_model": requested_model,
            "routed_model": routed_model,
            "input_tokens_est": input_tokens_est,
            "category": category,
            "stream": int(bool(stream)),
        }
        decision = build_optimization_coordinator(
            row=row,
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
        )
        selected = str(decision.get("selected_action_family") or decision.get("selected_family") or "none")
        suppressed_families: list[str] = []
        suppressed_managed_families: list[str] = []
        for item in decision.get("suppressed_families") or []:
            if not isinstance(item, dict):
                continue
            family = str(item.get("family") or "")
            if not family:
                continue
            suppressed_families.append(family)
            if _suppress_family(
                family,
                decision=decision,
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                provider_body=provider_body,
                local_routed_model=local_routed_model,
            ):
                suppressed_managed_families.append(family)
        enforcement = {
            "schema": SCHEMA,
            "enabled": True,
            "status": "applied",
            "selected_family": selected,
            "suppressed_families": sorted(set(suppressed_families)),
            "suppressed_managed_families": sorted(set(suppressed_managed_families)),
            "decision_hash": decision.get("decision_hash"),
            "metadata_only": True,
            "provider_body_included": False,
        }
        for meta in (routing_meta, crunch_meta, cache_meta):
            _attach(meta, decision, enforcement)
        return enforcement
    except Exception as exc:
        enforcement = {
            "schema": SCHEMA,
            "enabled": True,
            "status": "error",
            "reason": "coordinator-error",
            "error_type": type(exc).__name__,
            "metadata_only": True,
            "provider_body_included": False,
        }
        for meta in (routing_meta, crunch_meta, cache_meta):
            meta["optimization_coordinator_enforcement"] = enforcement
        return enforcement
