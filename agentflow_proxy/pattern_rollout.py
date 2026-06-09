from __future__ import annotations

import hashlib
import json
from typing import Any


PATTERN_ROLLOUT_SCHEMA = "agentflow.pattern_policy_rollout.v1"
PATTERN_CANARY_DECISION_SCHEMA = "agentflow.pattern_canary_decision.v1"

_SAFE_FEATURE_KEYS = (
    "source_surface",
    "app_family",
    "category",
    "workflow_phase",
    "text_bucket",
    "token_bucket",
    "requested_model",
    "candidate_target_model",
    "replayability_level",
    "has_tools",
    "stream",
    "session_id_hash",
    "workflow_id_hash",
    "request_fingerprint",
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_pattern_rollout(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    rollout: dict[str, Any] = {
        "schema": str(value.get("schema") or PATTERN_ROLLOUT_SCHEMA),
        "recommendation_mode": str(value.get("recommendation_mode") or "full-review"),
        "canary_enabled": _as_bool(value.get("canary_enabled"), False),
        "canary_fraction": _as_float(value.get("canary_fraction"), 1.0),
        "canary_salt": str(value.get("canary_salt") or ""),
        "canary_unit": str(value.get("canary_unit") or "request_fingerprint"),
    }
    for key in (
        "widening_threshold",
        "rollback_threshold",
        "min_outcome_samples",
        "local_feedback_fields",
    ):
        if key in value:
            rollout[key] = value[key]
    return rollout


def pattern_rollout_public_meta(rollout: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = normalize_pattern_rollout(rollout)
    if not normalized:
        return None
    return {
        key: normalized.get(key)
        for key in (
            "schema",
            "recommendation_mode",
            "canary_enabled",
            "canary_fraction",
            "canary_salt",
            "canary_unit",
            "widening_threshold",
            "rollback_threshold",
            "min_outcome_samples",
            "local_feedback_fields",
        )
        if normalized.get(key) is not None
    }


def _safe_features(features: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in _SAFE_FEATURE_KEYS:
        value = features.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return safe


def pattern_canary_decision(
    *,
    rollout: dict[str, Any] | None,
    rule_id: str,
    candidate_id: Any = None,
    pattern_hashes: list[str] | tuple[str, ...] | None = None,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_pattern_rollout(rollout)
    hashes = sorted({str(item) for item in (pattern_hashes or []) if str(item).startswith("sha256:")})
    base: dict[str, Any] = {
        "schema": PATTERN_CANARY_DECISION_SCHEMA,
        "enabled": False,
        "selected": True,
        "status": "full",
        "cohort": "full",
        "rule_id": rule_id,
        "candidate_id": _as_str(candidate_id),
        "pattern_hashes": hashes,
        "raw_pattern_strings_included": False,
    }
    if not normalized or not normalized.get("canary_enabled"):
        return base

    fraction = _as_float(normalized.get("canary_fraction"), 0.0)
    basis = {
        "unit": normalized.get("canary_unit") or "request_fingerprint",
        "rule_id": rule_id,
        "candidate_id": _as_str(candidate_id),
        "pattern_hashes": hashes,
        "features": _safe_features(features),
    }
    material = {
        "salt": normalized.get("canary_salt") or "",
        "basis": basis,
    }
    digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    selected = bucket < fraction
    base.update({
        "enabled": True,
        "selected": selected,
        "status": "applied" if selected else "holdout",
        "cohort": "canary_applied" if selected else "canary_holdout",
        "reason": None if selected else "canary_holdout",
        "fraction": fraction,
        "salt": normalized.get("canary_salt") or "",
        "unit": normalized.get("canary_unit") or "request_fingerprint",
        "bucket": round(bucket, 8),
        "threshold": fraction,
        "cohort_key_hash": _hash(basis),
    })
    return base
