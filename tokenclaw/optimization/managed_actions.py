from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from tokenclaw.store import stable_json


MANAGED_LOCAL_ACTIONS_SCHEMA = "tokenclaw.managed_local_actions.v1"

SUPPORTED_CRUNCH_PROFILES = {
    "default",
    "conservative",
    "aggressive",
    "managed",
    "old_context_summarization",
    "summarization",
    "compression",
}
SUPPORTED_CACHE_PROFILES = {"default", "exact", "semantic", "disabled", "managed"}
THINKING_TAIL_WIDENING_SCHEDULE_SCHEMA = "agentflow.thinking_tail_widening_schedule.v1"
RAW_ACTION_KEYS = {
    "prompt",
    "replacement_prompt",
    "messages",
    "content",
    "raw_request",
    "raw_response",
    "provider_body",
    "cache_key",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return default


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _iso_expired(value: Any, *, now: datetime | None = None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        expires_at = datetime.fromisoformat(text)
    except ValueError:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return expires_at <= now


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.replace("-", ".").split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


def _minimum_local_version_compatible(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    from tokenclaw import __version__

    current = _version_tuple(__version__)
    required = _version_tuple(value)
    if not current or not required:
        return True
    return current >= required


def _contains_raw_like_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in RAW_ACTION_KEYS:
                return True
            if _contains_raw_like_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_like_key(item) for item in value)
    return False


def _decision_fingerprint(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _policy_payload(recommendation: dict[str, Any]) -> dict[str, Any]:
    payload = recommendation.get("policy_decision")
    if isinstance(payload, dict):
        return payload
    return recommendation


def _privacy_block_reason(payload: dict[str, Any]) -> str | None:
    privacy = payload.get("privacy_summary")
    if isinstance(privacy, dict):
        if privacy.get("metadata_only") is False:
            return "privacy-not-metadata-only"
        if privacy.get("raw_payload_included") is True or privacy.get("raw_body_storage") is True:
            return "raw-payload-flagged"
    if payload.get("raw_payload_included") is True:
        return "raw-payload-flagged"
    if _contains_raw_like_key(payload):
        return "raw-like-action-field"
    return None


def _routing_section(payload: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    section = payload.get("routing")
    if not isinstance(section, dict):
        section = {}
    target_model = section.get("target_model") or recommendation.get("target_model")
    result = {
        "status": "not-present",
        "applied": False,
        "target_model": target_model if isinstance(target_model, str) else None,
        "policy_id": section.get("policy_id") or recommendation.get("policy_id"),
        "reason": section.get("reason") or recommendation.get("reason"),
    }
    if isinstance(target_model, str) and target_model.strip():
        result["status"] = "candidate"
    return result


def _thinking_tail_schedule(section: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    readiness = section.get("thinking_tail_readiness")
    if not isinstance(readiness, dict):
        return {}, None
    schedule = readiness.get("widening_schedule")
    return readiness, schedule if isinstance(schedule, dict) else None


def _crunch_section(payload: dict[str, Any], *, now: datetime | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    section = payload.get("crunch")
    if not isinstance(section, dict):
        return {"status": "not-present", "applied": False}, None
    profile = str(section.get("profile") or "managed").strip().lower()
    readiness, schedule = _thinking_tail_schedule(section)
    result = {
        "status": "candidate",
        "applied": False,
        "profile": profile,
        "policy_id": section.get("policy_id"),
        "candidate_id": section.get("candidate_id"),
    }
    if readiness:
        result["thinking_tail_readiness"] = readiness
    if schedule:
        result["widening_schedule"] = schedule
        if schedule.get("schema") != THINKING_TAIL_WIDENING_SCHEDULE_SCHEMA:
            result.update({
                "status": "vetoed",
                "applied": False,
                "apply_reason": "unsupported-widening-schedule",
                "veto_reason": "unsupported-widening-schedule",
            })
            return result, None
        if _iso_expired(schedule.get("expires_at"), now=now):
            result.update({
                "status": "vetoed",
                "applied": False,
                "apply_reason": "expired-widening-schedule",
                "veto_reason": "expired-widening-schedule",
            })
            return result, None
    min_version = (
        readiness.get("minimum_local_client_version")
        if readiness
        else section.get("minimum_local_client_version")
    )
    if not _minimum_local_version_compatible(min_version):
        result.update({
            "status": "vetoed",
            "applied": False,
            "apply_reason": "minimum-local-client-version-not-met",
            "veto_reason": "minimum-local-client-version-not-met",
            "minimum_local_client_version": min_version,
        })
        return result, None
    for key in (
        "traffic_treatment",
        "server_traffic_treatment",
        "canary_fraction",
        "holdout_fraction",
        "canary_unit",
        "canary_salt",
    ):
        if section.get(key) is not None:
            result[key] = section[key]
    if profile not in SUPPORTED_CRUNCH_PROFILES:
        result.update({"status": "unsupported", "apply_reason": "unsupported-crunch-profile"})
        return result, None
    effective: dict[str, Any] = {
        "policy_source": "managed-recommended",
        "profile": profile,
        "policy_id": section.get("policy_id"),
        "candidate_id": section.get("candidate_id"),
    }
    if readiness:
        effective["thinking_tail_readiness"] = readiness
    if schedule:
        effective["widening_schedule"] = schedule
    for key in (
        "traffic_treatment",
        "server_traffic_treatment",
        "canary_fraction",
        "holdout_fraction",
        "canary_unit",
        "canary_salt",
    ):
        if section.get(key) is not None:
            effective[key] = section[key]
    threshold = _as_int(section.get("threshold_chars"))
    if threshold is None:
        thresholds = section.get("thresholds")
        if isinstance(thresholds, dict):
            threshold = _as_int(thresholds.get("threshold_chars") or thresholds.get("min_request_chars"))
    if threshold is not None and threshold > 0:
        effective["threshold_chars"] = threshold
        result["threshold_chars"] = threshold
    summary = section.get("old_context_summarization")
    if isinstance(summary, dict):
        enhanced = _enhanced_crunch_hint(section, summary)
        result["old_context_summarization"] = {
            "enabled": _as_bool(summary.get("enabled"), False),
            "model_hint": summary.get("model_hint"),
            "model_family": summary.get("model_family") or section.get("model_family"),
            "thresholds": summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {},
            "status": enhanced["state"],
            "state": enhanced["state"],
            "mode": enhanced["mode"],
            "profile": enhanced["profile"],
            "max_summary_cost_usd": enhanced.get("max_summary_cost_usd"),
            "canary_fraction": enhanced.get("canary_fraction"),
            "safety_stop_thresholds": enhanced.get("safety_stop_thresholds"),
        }
        result["enhanced_crunch"] = enhanced
        if enhanced["state"] == "fallback-not-configured":
            result.update({
                "status": "fallback-not-configured",
                "applied": False,
                "apply_reason": "local-enhanced-crunch-provider-not-configured",
            })
            return result, None
        result.update({
            "status": "configured",
            "applied": False,
            "apply_reason": "local-enhanced-crunch-provider-configured",
        })
        effective["old_context_summarization"] = result["old_context_summarization"]
        effective["enhanced_crunch"] = enhanced
    return result, effective


def _enhanced_crunch_hint(section: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    provider_hint = section.get("enhanced_crunch_provider")
    if not isinstance(provider_hint, dict):
        provider_hint = section.get("enhanced_provider") if isinstance(section.get("enhanced_provider"), dict) else {}
    mode = str(
        provider_hint.get("mode")
        or summary.get("mode")
        or section.get("mode")
        or "local_provider_account"
    ).strip().lower().replace("-", "_")
    profile = str(
        provider_hint.get("profile")
        or summary.get("profile")
        or section.get("profile")
        or "old_context_summarization"
    )
    try:
        from tokenclaw.crunch import enhanced_crunch_provider_public_meta

        provider_meta = enhanced_crunch_provider_public_meta({
            "policy_source": "managed-recommended",
            "profile": profile,
            "old_context_summarization": summary,
        })
    except Exception:
        provider_meta = {"configured": False, "state": "fallback-not-configured"}
    canary = summary.get("canary") if isinstance(summary.get("canary"), dict) else {}
    safety = summary.get("safety_stop") if isinstance(summary.get("safety_stop"), dict) else {}
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    max_cost = _as_float(
        summary.get("max_summary_cost_usd")
        or section.get("max_summary_cost_usd")
        or thresholds.get("max_summary_cost_usd")
    )
    state = "configured" if provider_meta.get("configured") else "fallback-not-configured"
    return {
        "schema": "tokenclaw.managed_enhanced_crunch_hint.v1",
        "state": state,
        "recommended": True,
        "configured": bool(provider_meta.get("configured")),
        "mode": mode,
        "profile": profile,
        "model_hint": summary.get("model_hint") or summary.get("model"),
        "model_family": summary.get("model_family") or section.get("model_family"),
        "thresholds": thresholds,
        "max_summary_cost_usd": max_cost,
        "canary_fraction": _as_float(
            summary.get("canary_fraction")
            or section.get("canary_fraction")
            or canary.get("fraction")
            or canary.get("canary_fraction")
        ),
        "safety_stop_thresholds": safety,
        "raw_source_included": False,
        "summary_request_content_included": False,
        "raw_summary_included": False,
        "provider_response_included": False,
        "cache_key_included": False,
        "endpoint_url_included": False,
    }


def _cache_section(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    section = payload.get("cache")
    if not isinstance(section, dict):
        return {"status": "not-present", "applied": False}, None
    profile = str(section.get("profile") or "managed").strip().lower()
    result = {
        "status": "candidate",
        "applied": False,
        "profile": profile,
        "policy_id": section.get("policy_id"),
        "candidate_id": section.get("candidate_id"),
    }
    if profile not in SUPPORTED_CACHE_PROFILES:
        result.update({"status": "unsupported", "apply_reason": "unsupported-cache-profile"})
        return result, None
    effective: dict[str, Any] = {
        "policy_source": "managed-recommended",
        "profile": profile,
        "policy_id": section.get("policy_id"),
        "candidate_id": section.get("candidate_id"),
    }
    if profile == "disabled":
        effective["exact_enabled"] = False
        effective["semantic_enabled"] = False
    elif profile == "exact":
        effective["exact_enabled"] = True
        effective["semantic_enabled"] = False
    elif profile == "semantic":
        effective["semantic_enabled"] = True
    if section.get("exact_enabled") is not None:
        effective["exact_enabled"] = _as_bool(section.get("exact_enabled"), False)
    if section.get("semantic_enabled") is not None:
        effective["semantic_enabled"] = _as_bool(section.get("semantic_enabled"), False)
    threshold = _as_float(section.get("semantic_threshold"))
    if threshold is not None:
        threshold = max(0.0, min(1.0, threshold))
        effective["semantic_threshold"] = threshold
        result["semantic_threshold"] = threshold
    return result, effective


def _unsupported_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    actions = payload.get("actions")
    if isinstance(actions, list):
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                unsupported.append({"index": index, "status": "unsupported", "reason": "non-object-action"})
                continue
            action_type = str(action.get("type") or action.get("action_type") or "unknown")
            unsupported.append({
                "index": index,
                "type": action_type,
                "status": "unsupported",
                "reason": "unsupported-action-type",
            })
    for key in payload:
        if key not in {
            "schema",
            "policy_decision",
            "provider",
            "source_surface",
            "enabled",
            "status",
            "expires_at",
            "expiry",
            "privacy_summary",
            "routing",
            "routing_action",
            "routing_status",
            "crunch",
            "cache",
            "actions",
            "decision_id",
            "target_model",
            "target_model_normalized",
            "target_model_after_client_policy",
            "route_to",
            "route_to_present",
            "confidence",
            "route_down_probability",
            "route_proposal",
            "route_selected",
            "policy_id",
            "policy_decision_schema",
            "reason",
            "reason_codes",
            "traffic_treatment",
            "server_traffic_treatment",
            "canary_fraction",
            "holdout_fraction",
            "canary_unit",
            "canary_salt",
            "server_selected_canary_membership",
            "optimization_unit_id",
            "recommendation_id",
            "recommendation_family",
            "candidate_id",
            "policy_source",
            "product_mode",
            "recommended_mode",
            "local_policy_decision_mode",
            "model_artifact_version",
            "model_evidence_hash",
            "predictor_rule_id",
            "explicit_routing_action",
            "required_local_gates",
            "replacement_prompt_present",
            "signed",
            "signature_required",
            "requires_signature",
            "provenance",
            "local_executor_compatibility",
            "local_action_requirements",
            "provider_capability_matrix_schema",
            "provider_capabilities",
            "capability_audit",
            "provider_forwarding",
            "server_content_processing",
            "omitted_actions",
            "raw_payload_included",
            "thinking_tail_readiness",
        }:
            unsupported.append({"section": key, "status": "unsupported", "reason": "unknown-section"})
    return unsupported


def evaluate_managed_local_actions(
    recommendation: dict[str, Any],
    *,
    provider: str,
    current_model: str,
    source_surface: str | None = None,
    application_enabled: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = _policy_payload(recommendation)
    meta = {
        "schema": MANAGED_LOCAL_ACTIONS_SCHEMA,
        "enabled": bool(recommendation.get("enabled")),
        "application_enabled": bool(application_enabled),
        "provider": provider,
        "source_surface": source_surface,
        "policy_source": "managed-recommended",
        "status": "not-evaluated",
        "applied": False,
        "raw_payload_included": False,
        "fingerprint": _decision_fingerprint(payload),
        "routing": _routing_section(payload, recommendation),
        "crunch": {"status": "not-present", "applied": False},
        "cache": {"status": "not-present", "applied": False},
        "unsupported_actions": _unsupported_actions(payload),
        "effective_profiles": {},
    }
    if not recommendation.get("enabled"):
        meta.update({"status": "skipped", "apply_reason": "managed-disabled"})
        return meta
    if recommendation.get("status") != "received":
        meta.update({"status": "skipped", "apply_reason": recommendation.get("reason") or "recommendation-not-received"})
        return meta
    payload_provider = payload.get("provider")
    if isinstance(payload_provider, str) and payload_provider and payload_provider.lower() != provider.lower():
        meta.update({"status": "skipped", "apply_reason": "provider-mismatch"})
        return meta
    payload_surface = payload.get("source_surface")
    if source_surface and isinstance(payload_surface, str) and payload_surface and payload_surface != source_surface:
        meta.update({"status": "skipped", "apply_reason": "source-surface-mismatch"})
        return meta
    expires_at = payload.get("expires_at") or payload.get("expiry")
    if _iso_expired(expires_at, now=now):
        meta.update({"status": "skipped", "apply_reason": "expired-policy", "expires_at": expires_at})
        return meta
    if (payload.get("signature_required") or payload.get("requires_signature")) and not payload.get("signed"):
        meta.update({"status": "skipped", "apply_reason": "unsigned-policy"})
        return meta
    privacy_reason = _privacy_block_reason(payload)
    if privacy_reason:
        meta.update({"status": "skipped", "apply_reason": privacy_reason})
        return meta

    crunch_meta, crunch_profile = _crunch_section(payload, now=now)
    cache_meta, cache_profile = _cache_section(payload)
    meta["crunch"] = crunch_meta
    meta["cache"] = cache_meta
    if crunch_meta.get("status") == "vetoed":
        meta.update({"status": "skipped", "apply_reason": crunch_meta.get("apply_reason") or "local-crunch-vetoed"})
        return meta

    if not application_enabled:
        meta.update({"status": "dry-run", "apply_reason": "local-action-dry-run"})
        return meta

    effective_profiles: dict[str, Any] = {}
    if crunch_profile is not None:
        if crunch_meta.get("status") == "configured":
            crunch_meta.update({"applied": False, "apply_reason": "local-enhanced-crunch-provider-configured"})
        else:
            crunch_meta.update({"status": "applied", "applied": True, "apply_reason": "local-profile-selected"})
        effective_profiles["crunch"] = crunch_profile
    if cache_profile is not None:
        cache_meta.update({"status": "applied", "applied": True, "apply_reason": "local-profile-selected"})
        effective_profiles["cache"] = cache_profile
    if meta["routing"].get("target_model"):
        meta["routing"].update({"status": "candidate", "local_model_before_recommendation": current_model})
    meta["effective_profiles"] = effective_profiles
    meta.update({
        "status": "applied" if effective_profiles or meta["routing"].get("target_model") else "noop",
        "applied": bool(effective_profiles),
        "apply_reason": "local-actions-selected" if effective_profiles else "no-local-profile-actions",
    })
    return meta


def crunch_profile_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    actions = decision.get("local_actions")
    if not isinstance(actions, dict):
        return None
    profiles = actions.get("effective_profiles")
    if not isinstance(profiles, dict):
        return None
    profile = profiles.get("crunch")
    return profile if isinstance(profile, dict) else None


def cache_profile_from_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    actions = decision.get("local_actions")
    if not isinstance(actions, dict):
        return None
    profiles = actions.get("effective_profiles")
    if not isinstance(profiles, dict):
        return None
    profile = profiles.get("cache")
    return profile if isinstance(profile, dict) else None
