from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from agentflow_proxy.store import stable_json


MANAGED_LOCAL_ACTIONS_SCHEMA = "agentflow.managed_local_actions.v1"

SUPPORTED_CRUNCH_PROFILES = {"default", "conservative", "aggressive", "managed"}
SUPPORTED_CACHE_PROFILES = {"default", "exact", "semantic", "disabled", "managed"}
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


def _crunch_section(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    section = payload.get("crunch")
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
    if profile not in SUPPORTED_CRUNCH_PROFILES:
        result.update({"status": "unsupported", "apply_reason": "unsupported-crunch-profile"})
        return result, None
    effective: dict[str, Any] = {
        "policy_source": "managed-recommended",
        "profile": profile,
        "policy_id": section.get("policy_id"),
        "candidate_id": section.get("candidate_id"),
    }
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
        result["old_context_summarization"] = {
            "enabled": _as_bool(summary.get("enabled"), False),
            "model_hint": summary.get("model_hint"),
            "thresholds": summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {},
            "status": "metadata-only-local-hint",
        }
        effective["old_context_summarization"] = result["old_context_summarization"]
    return result, effective


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
            "expires_at",
            "expiry",
            "privacy_summary",
            "routing",
            "crunch",
            "cache",
            "actions",
            "target_model",
            "confidence",
            "policy_id",
            "reason",
            "optimization_unit_id",
            "recommendation_id",
            "recommendation_family",
            "candidate_id",
            "policy_source",
            "signed",
            "signature_required",
            "requires_signature",
            "provenance",
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

    crunch_meta, crunch_profile = _crunch_section(payload)
    cache_meta, cache_profile = _cache_section(payload)
    meta["crunch"] = crunch_meta
    meta["cache"] = cache_meta

    if not application_enabled:
        meta.update({"status": "dry-run", "apply_reason": "local-action-dry-run"})
        return meta

    effective_profiles: dict[str, Any] = {}
    if crunch_profile is not None:
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
