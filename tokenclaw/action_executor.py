from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tokenclaw.env import env
from tokenclaw.managed_mode import managed_product_mode
from tokenclaw.optimization.managed_actions import (
    cache_profile_from_decision,
    crunch_profile_from_decision,
    evaluate_managed_local_actions,
)


ACTION_EXECUTOR_RESULT_SCHEMA = "tokenclaw.action_executor_result.v1"
ACTION_EXECUTOR_OUTCOME_SCHEMA = "tokenclaw.action_executor_outcome_feedback.v1"
DEFAULT_ACTION_FAMILIES = ("routing", "crunch", "cache")
ACTION_EXECUTOR_ENABLED_ENV = "TOKENCLAW_ACTION_EXECUTOR_ENABLED"
ACTION_EXECUTOR_FAMILIES_ENV = "TOKENCLAW_ACTION_EXECUTOR_FAMILIES"
ACTION_EXECUTOR_REQUIRE_SIGNATURE_ENV = "TOKENCLAW_ACTION_EXECUTOR_REQUIRE_SIGNATURE"
HOLD_TRAFFIC_TREATMENTS = {"hold", "held", "observe", "none"}
HOLDOUT_TRAFFIC_TREATMENTS = {"holdout"}
SHADOW_TRAFFIC_TREATMENTS = {"shadow"}
VETO_TRAFFIC_TREATMENTS = {"veto", "vetoed", "safety_blocked", "blocked", "unsupported"}
LOCAL_RUNTIME_DECISION_KEYS = {
    "action_executor",
    "api_key_value_included",
    "applied",
    "applied_families",
    "apply_reason",
    "auth_configured",
    "auth_source",
    "canary",
    "canary_fraction",
    "changed_model",
    "client_contract",
    "client_routing_policy",
    "endpoint",
    "enabled",
    "fallback",
    "failure_mode",
    "latency_ms",
    "lifecycle_event",
    "live_promotion_mode",
    "local_action_taken",
    "local_actions",
    "local_canary",
    "local_model_at_application",
    "local_model_before_recommendation",
    "loopback_unauthenticated_allowed",
    "managed_measurement",
    "min_confidence",
    "mode",
    "policy_decision_enabled",
    "product_mode",
    "product_mode_application_enabled",
    "product_mode_enforced",
    "projection",
    "selected_for_local_application",
    "selected_for_shadow_evaluation",
    "server_action_selection",
    "server_recommended_mode",
    "server_traffic_treatment",
    "server_url",
    "shadow_model",
    "shadow_only",
    "status",
    "status_code",
    "timeout_seconds",
    "would_change_model",
    "would_route_model",
}


def _env_enabled(name: str, default: str = "1") -> bool:
    return str(env(name, default) or default).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_families() -> tuple[str, ...]:
    raw = str(env(ACTION_EXECUTOR_FAMILIES_ENV, ",".join(DEFAULT_ACTION_FAMILIES)) or "")
    families = tuple(
        item.strip().lower()
        for item in raw.replace(";", ",").split(",")
        if item.strip()
    )
    return families or DEFAULT_ACTION_FAMILIES


def _provider_compatible(provider: str, target_model: str) -> bool:
    target_l = target_model.lower()
    if provider == "anthropic":
        return target_l.startswith("claude-")
    if provider == "openai":
        return not target_l.startswith("claude-")
    return False


def _supported_target_model(provider: str, target_model: str) -> bool:
    if not _provider_compatible(provider, target_model):
        return False
    target_l = target_model.lower()
    if provider == "anthropic":
        return any(tier in target_l for tier in ("haiku", "sonnet", "opus"))
    return bool(target_l)


def _target_model(decision: dict[str, Any], local_actions: dict[str, Any]) -> str | None:
    for value in (
        decision.get("target_model_normalized"),
        decision.get("target_model_after_client_policy"),
        decision.get("target_model"),
        decision.get("route_to"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    routing = local_actions.get("routing") if isinstance(local_actions.get("routing"), dict) else {}
    value = routing.get("target_model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    decision_routing = decision.get("routing") if isinstance(decision.get("routing"), dict) else {}
    proposal = decision_routing.get("route_proposal") if isinstance(decision_routing.get("route_proposal"), dict) else {}
    value = proposal.get("target_model")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalized_treatment(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    treatment = value.strip().lower().replace("-", "_")
    aliases = {
        "apply": "live",
        "applied": "live",
        "enforced": "live",
        "selected": "canary",
        "canary_applied": "canary",
        "canary_holdout": "holdout",
        "shadow_only": "shadow",
        "observe_only": "observe",
        "dry_run": "observe",
        "dry-run": "observe",
        "noop": "none",
        "no_op": "none",
    }
    return aliases.get(treatment, treatment)


def _route_proposal(decision: dict[str, Any]) -> dict[str, Any]:
    routing = decision.get("routing")
    if not isinstance(routing, dict):
        return {}
    proposal = routing.get("route_proposal")
    return proposal if isinstance(proposal, dict) else {}


def _traffic_treatment(decision: dict[str, Any]) -> str | None:
    routing = decision.get("routing") if isinstance(decision.get("routing"), dict) else {}
    for source in (
        decision.get("server_action_selection"),
        _route_proposal(decision),
        routing,
        decision,
    ):
        if not isinstance(source, dict):
            continue
        treatment = _normalized_treatment(source.get("traffic_treatment") or source.get("server_traffic_treatment"))
        if treatment:
            return treatment
    mode = _normalized_treatment(decision.get("local_policy_decision_mode") or decision.get("recommended_mode") or decision.get("mode"))
    if mode in {"live", "canary", "shadow", "observe", "hold"}:
        return mode
    return None


def _server_route_selected(decision: dict[str, Any], treatment: str | None) -> bool | None:
    if decision.get("selected_for_local_application") is True:
        return True
    if decision.get("selected_for_local_application") is False:
        return False
    routing = decision.get("routing") if isinstance(decision.get("routing"), dict) else {}
    proposal = _route_proposal(decision)
    for source in (proposal, routing, decision):
        if not isinstance(source, dict):
            continue
        value = source.get("route_selected")
        if isinstance(value, bool):
            return value
    membership = proposal.get("server_selected_canary_membership")
    if isinstance(membership, bool) and treatment == "canary":
        return membership
    return None


def _bounded_fraction(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return None
    return None


def _treatment_fraction(decision: dict[str, Any], name: str) -> float | None:
    routing = decision.get("routing") if isinstance(decision.get("routing"), dict) else {}
    for source in (_route_proposal(decision), routing, decision):
        if not isinstance(source, dict):
            continue
        value = _bounded_fraction(source.get(name))
        if value is not None:
            return value
    return None


def _treatment_reason(treatment: str | None, selected: bool | None) -> str:
    if treatment == "holdout" or (treatment == "canary" and selected is False):
        return "server-canary-holdout"
    if treatment == "shadow":
        return "shadow-only"
    if treatment == "observe":
        return "observe-only"
    if treatment in {"hold", "held", "none"}:
        return "server-held"
    if treatment in VETO_TRAFFIC_TREATMENTS:
        return f"server-{treatment}"
    if selected is False:
        return "server-route-not-selected"
    return "server-traffic-treatment-held"


def _disabled_family(name: str) -> dict[str, Any]:
    return {
        "status": "vetoed",
        "applied": False,
        "apply_reason": "local-action-family-disabled",
        "veto_reason": "local-action-family-disabled",
        "outcome_status": "vetoed",
    }


def _hold_family(name: str, *, status: str, reason: str, target_model: str | None = None) -> dict[str, Any]:
    result = {
        "status": status,
        "applied": False,
        "apply_reason": reason,
        "outcome_status": status,
    }
    if status == "vetoed":
        result["veto_reason"] = reason
    if target_model and name == "routing":
        result["target_model"] = target_model
    return result


def _copy_family(local_actions: dict[str, Any], name: str) -> dict[str, Any]:
    value = local_actions.get(name)
    return dict(value) if isinstance(value, dict) else {"status": "not-present", "applied": False}


@dataclass(frozen=True)
class ActionExecutor:
    provider: str
    supported_action_families: tuple[str, ...] = field(default_factory=_env_families)
    enabled: bool = field(default_factory=lambda: _env_enabled(ACTION_EXECUTOR_ENABLED_ENV, "1"))
    require_signature: bool = field(default_factory=lambda: _env_enabled(ACTION_EXECUTOR_REQUIRE_SIGNATURE_ENV, "0"))
    now: datetime | None = None

    def execute(
        self,
        *,
        body: dict[str, Any],
        routing_meta: dict[str, Any],
        decision: dict[str, Any],
        application_enabled: bool,
        shadow_only: bool = False,
        source_surface: str | None = None,
    ) -> dict[str, Any]:
        current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
        product_mode = managed_product_mode()
        enforce_product_mode = product_mode.configured or product_mode.local_rules_only
        configured_supported = {family.strip().lower() for family in self.supported_action_families}
        supported = (
            {
                family
                for family in configured_supported
                if product_mode.family_enabled.get(family, False)
            }
            if enforce_product_mode
            else set(configured_supported)
        )
        application_allowed = bool(
            application_enabled
            and self.enabled
            and (product_mode.local_application_enabled if enforce_product_mode else True)
        )
        action_decision = {
            key: value
            for key, value in decision.items()
            if key not in LOCAL_RUNTIME_DECISION_KEYS
        }
        traffic_treatment = _traffic_treatment(decision)
        route_selected = _server_route_selected(decision, traffic_treatment)
        treatment_reason = _treatment_reason(traffic_treatment, route_selected)
        if "enabled" in decision:
            action_decision["enabled"] = decision["enabled"]
        if "status" in decision:
            action_decision["status"] = decision["status"]
        if action_decision.get("status") in {
            "selected",
            "shadow_selected",
            "dry-run",
            "applied",
            "noop",
            "holdout",
            "hold",
            "held",
        }:
            action_decision["status"] = "received"
        if "enabled" not in action_decision and action_decision.get("status") == "received":
            action_decision["enabled"] = True
        local_actions = evaluate_managed_local_actions(
            action_decision,
            provider=self.provider,
            current_model=current_model,
            source_surface=source_surface,
            application_enabled=application_allowed,
            now=self.now,
        )
        result = {
            "schema": ACTION_EXECUTOR_RESULT_SCHEMA,
            "enabled": bool(self.enabled),
            "provider": self.provider,
            "policy_id": decision.get("policy_id"),
            "decision_id": decision.get("decision_id"),
            "policy_source": "managed-recommended",
            "application_enabled": bool(application_enabled),
            "product_mode_application_enabled": product_mode.local_application_enabled if enforce_product_mode else True,
            "product_mode": product_mode.public_meta(),
            "product_mode_enforced": enforce_product_mode,
            "shadow_only": bool(shadow_only),
            "server_traffic_treatment": traffic_treatment,
            "server_route_selected": route_selected,
            "canary_fraction": _treatment_fraction(decision, "canary_fraction"),
            "holdout_fraction": _treatment_fraction(decision, "holdout_fraction"),
            "server_canary_membership_source": "server" if traffic_treatment in {"canary", "holdout"} else None,
            "supported_local_action_families": sorted(configured_supported),
            "enabled_local_action_families": sorted(supported),
            "local_model_before": current_model,
            "local_model_after": current_model,
            "status": "held",
            "applied": False,
            "changed_model": False,
            "fallback": "local-policy",
            "routing": _copy_family(local_actions, "routing"),
            "crunch": _copy_family(local_actions, "crunch"),
            "cache": _copy_family(local_actions, "cache"),
            "unsupported_actions": list(local_actions.get("unsupported_actions") or []),
            "effective_profiles": dict(local_actions.get("effective_profiles") or {}),
            "raw_payload_included": False,
        }

        if self.require_signature and not (
            decision.get("signed")
            or (isinstance(decision.get("provenance"), dict) and decision["provenance"].get("signature"))
        ):
            return self._finish(result, "vetoed", "unsigned-policy")
        if local_actions.get("status") == "skipped":
            return self._finish(result, "vetoed", str(local_actions.get("apply_reason") or "local-action-validation-failed"))
        if result["unsupported_actions"]:
            return self._finish(result, "vetoed", "unsupported-action-type")

        if not self.enabled:
            return self._finish(result, "held", "action-executor-disabled")
        if enforce_product_mode and not product_mode.server_calls_enabled:
            return self._finish(result, "held", product_mode.reason)
        if not application_enabled:
            return self._finish(result, "held", "local-application-disabled")
        if enforce_product_mode and not product_mode.local_application_enabled:
            return self._finish(result, "held", f"managed-mode-{product_mode.mode}")
        if traffic_treatment in VETO_TRAFFIC_TREATMENTS:
            return self._hold_server_treatment(
                result,
                decision,
                local_actions,
                status="vetoed",
                reason=treatment_reason,
            )
        if traffic_treatment in HOLDOUT_TRAFFIC_TREATMENTS or (
            traffic_treatment == "canary" and route_selected is False
        ):
            return self._hold_server_treatment(
                result,
                decision,
                local_actions,
                status="heldout",
                reason="server-canary-holdout",
            )
        if shadow_only or traffic_treatment in SHADOW_TRAFFIC_TREATMENTS or decision.get("selected_for_shadow_evaluation") is True:
            target = _target_model(decision, local_actions)
            if target:
                result["would_route_model"] = target
                result["routing"].update({
                    "status": "held",
                    "applied": False,
                    "target_model": target,
                    "apply_reason": "shadow-only",
                })
            return self._finish(result, "held", "shadow-only")
        if traffic_treatment in HOLD_TRAFFIC_TREATMENTS or route_selected is False:
            return self._hold_server_treatment(
                result,
                decision,
                local_actions,
                status="held",
                reason=treatment_reason,
            )
        if traffic_treatment == "canary" and route_selected is not True:
            return self._hold_server_treatment(
                result,
                decision,
                local_actions,
                status="held",
                reason="server-route-selection-missing",
            )

        for family in DEFAULT_ACTION_FAMILIES:
            if family not in supported and self._family_present(decision, local_actions, family):
                result[family] = _disabled_family(family)
                profiles = result.get("effective_profiles")
                if isinstance(profiles, dict):
                    profiles.pop(family, None)

        applied_families: list[str] = []
        target_model = _target_model(decision, local_actions)
        if target_model:
            if "routing" not in supported:
                result["routing"] = _disabled_family("routing")
            elif not _provider_compatible(self.provider, target_model):
                result["routing"].update({
                    "status": "vetoed",
                    "applied": False,
                    "target_model": target_model,
                    "apply_reason": "provider-mismatch",
                    "veto_reason": "provider-mismatch",
                })
            elif not _supported_target_model(self.provider, target_model):
                result["routing"].update({
                    "status": "vetoed",
                    "applied": False,
                    "target_model": target_model,
                    "apply_reason": "unsupported-target-model",
                    "veto_reason": "unsupported-target-model",
                })
            elif target_model == current_model:
                result["routing"].update({
                    "status": "noop",
                    "applied": False,
                    "target_model": target_model,
                    "apply_reason": "target-model-already-selected-locally",
                })
            else:
                body["model"] = target_model
                routing_meta["routed_model"] = target_model
                routing_meta["managed_routing_applied"] = True
                routing_meta["final_policy_source"] = "managed-recommended"
                routing_meta["managed_policy_id"] = decision.get("policy_id")
                routing_meta["managed_reason"] = decision.get("reason")
                routing_meta["managed_route_recommended_mode"] = decision.get("local_policy_decision_mode") or decision.get("recommended_mode")
                result["routing"].update({
                    "status": "applied",
                    "applied": True,
                    "target_model": target_model,
                    "apply_reason": "provider-body-model-rewrite",
                })
                result["local_model_after"] = target_model
                result["changed_model"] = True
                applied_families.append("routing")

        for family in ("crunch", "cache"):
            profile = result.get("effective_profiles", {}).get(family) if isinstance(result.get("effective_profiles"), dict) else None
            if isinstance(profile, dict):
                if family in supported:
                    result[family].update({
                        "status": result[family].get("status") if result[family].get("status") == "configured" else "applied",
                        "applied": result[family].get("applied") is not False,
                        "apply_reason": result[family].get("apply_reason") or "local-profile-selected",
                    })
                    applied_families.append(family)
                else:
                    result[family] = _disabled_family(family)

        if any(result[family].get("status") == "vetoed" for family in DEFAULT_ACTION_FAMILIES):
            return self._finish(result, "vetoed", "local-action-vetoed")
        if applied_families:
            result["applied_families"] = sorted(set(applied_families))
            return self._finish(result, "applied", "local-actions-applied")
        return self._finish(result, "noop", "no-local-actions-applied")

    def _family_present(self, decision: dict[str, Any], local_actions: dict[str, Any], family: str) -> bool:
        if isinstance(decision.get(family), dict):
            return True
        section = local_actions.get(family)
        return isinstance(section, dict) and section.get("status") not in {None, "not-present"}

    def _hold_server_treatment(
        self,
        result: dict[str, Any],
        decision: dict[str, Any],
        local_actions: dict[str, Any],
        *,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        target = _target_model(decision, local_actions)
        if target:
            result["would_route_model"] = target
        if isinstance(result.get("effective_profiles"), dict):
            result["effective_profiles"] = {}
        for family in DEFAULT_ACTION_FAMILIES:
            if self._family_present(decision, local_actions, family):
                result[family] = _hold_family(
                    family,
                    status=status,
                    reason=reason,
                    target_model=target,
                )
        return self._finish(result, status, reason)

    def _finish(self, result: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
        result["status"] = status
        result["apply_reason"] = reason
        result["fallback"] = None if status == "applied" else "local-policy"
        result["applied"] = status == "applied"
        result["outcome_feedback"] = {
            "schema": ACTION_EXECUTOR_OUTCOME_SCHEMA,
            "policy_id": result.get("policy_id"),
            "decision_id": result.get("decision_id"),
            "provider": self.provider,
            "status": status,
            "reason": reason,
            "applied_families": list(result.get("applied_families") or []),
            "vetoed_families": [
                family
                for family in DEFAULT_ACTION_FAMILIES
                if isinstance(result.get(family), dict) and result[family].get("status") == "vetoed"
            ],
            "held_families": [
                family
                for family in DEFAULT_ACTION_FAMILIES
                if isinstance(result.get(family), dict) and result[family].get("status") == "held"
            ],
            "heldout_families": [
                family
                for family in DEFAULT_ACTION_FAMILIES
                if isinstance(result.get(family), dict) and result[family].get("status") == "heldout"
            ],
            "server_traffic_treatment": result.get("server_traffic_treatment"),
            "server_route_selected": result.get("server_route_selected"),
            "canary_fraction": result.get("canary_fraction"),
            "holdout_fraction": result.get("holdout_fraction"),
            "product_mode": result.get("product_mode"),
            "raw_payload_included": False,
        }
        return result


def crunch_profile_from_executor_result(result: dict[str, Any]) -> dict[str, Any] | None:
    return crunch_profile_from_decision({"local_actions": result})


def cache_profile_from_executor_result(result: dict[str, Any]) -> dict[str, Any] | None:
    return cache_profile_from_decision({"local_actions": result})
