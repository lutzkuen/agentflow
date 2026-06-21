from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tokenclaw.optimization_promotion_actions import ACTION_SCHEMA, SCHEMA as PROMOTION_ACTIONS_SCHEMA
from tokenclaw.optimization_rollout_review import (
    OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA,
    validate_optimization_rollout_bundle,
)
from tokenclaw.store import utc_now


PROMOTION_CANARY_APPLY_SCHEMA = "tokenclaw.optimization_promotion_canary_apply.v1"
PROMOTION_CANARY_DECISION_SCHEMA = "tokenclaw.optimization_promotion_canary_decision.v1"
PROMOTION_CANARY_SAFETY_SCHEMA = "tokenclaw.optimization_promotion_canary_safety_stop.v1"

_POLICY_SECTION_FILES = {
    "routing": "routing_rules.yaml",
    "crunch": "crunch_rules.yaml",
    "cache": "cache_rules.yaml",
}
_OPENAI_SOURCE_SURFACES = {
    "openai",
    "openai_provider_request",
    "openai_responses",
    "openai_chat_completions",
}
_COHORT_SAFE_KEYS = (
    "request_fingerprint",
    "session_id_hash",
    "workflow_id_hash",
    "traffic_fingerprint",
    "optimization_unit_id",
    "source_surface",
    "app_family",
    "category",
    "workflow_phase",
    "text_bucket",
    "token_bucket",
    "requested_model",
    "candidate_target_model",
    "has_tools",
    "stream",
)
_RAW_LIKE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cache_key",
    "command",
    "content",
    "credential",
    "endpoint",
    "file_path",
    "generated_summary",
    "message",
    "param",
    "prompt",
    "provider_body",
    "raw_payload",
    "raw_request",
    "raw_response",
    "raw_context",
    "request_id",
    "secret",
    "session_id",
    "summary_text",
    "tool_payload",
    "transcript",
)
_ALLOWED_RAW_LIKE_KEYS = {
    "raw_prompts_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
    "raw_session_ids_included",
    "raw_content_included",
    "request_ids_included",
    "session_id_hash",
    "request_fingerprint",
    "traffic_fingerprint",
    "apply_preview_command",
    "review_command",
    "content_free",
    "api_endpoint",
    "provider_endpoint",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _nested_get(source: Any, *path: str) -> Any:
    value = source
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _as_float_default(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    return _as_float(value, default)


def _is_openai_routing_action(action: dict[str, Any]) -> bool:
    values = [
        action.get("provider"),
        action.get("provider_family"),
        action.get("source_surface"),
        action.get("provider_endpoint"),
        _nested_get(action, "local_policy_update", "provider"),
        _nested_get(action, "local_policy_update", "provider_family"),
        _nested_get(action, "local_policy_update", "source_surface"),
        _nested_get(action, "action", "schema"),
    ]
    normalized = {str(value or "").strip().lower() for value in values if value not in (None, "")}
    return (
        bool(normalized & _OPENAI_SOURCE_SURFACES)
        or "responses" in normalized
        or "chat_completions" in normalized
        or any("openai" in value for value in normalized)
    )


def _nested_rollout_action(action: dict[str, Any]) -> dict[str, Any]:
    nested = action.get("action") if isinstance(action.get("action"), dict) else {}
    proposed = nested.get("proposed_edit") if isinstance(nested.get("proposed_edit"), dict) else {}
    proposed_action = proposed.get("action") if isinstance(proposed.get("action"), dict) else {}
    return {
        "nested": nested,
        "proposed": proposed,
        "proposed_action": proposed_action,
    }


def _rollout_target_rule_id(action: dict[str, Any]) -> str:
    nested = _nested_rollout_action(action)
    return str(
        _first_present(
            action.get("target_rule_id"),
            nested["nested"].get("target_rule_id"),
            nested["proposed"].get("rule_id"),
            action.get("target_candidate_id"),
        )
        or "optimization-rollout-routing-canary"
    )


def _rollout_target_model(action: dict[str, Any]) -> Any:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    nested = _nested_rollout_action(action)
    return _first_present(
        local_update.get("candidate_target_model"),
        local_update.get("target_model"),
        action.get("candidate_target_model"),
        action.get("target_model"),
        nested["proposed_action"].get("route_to"),
        nested["nested"].get("target_model"),
        nested["nested"].get("route_to"),
    )


def _optimization_rollout_to_promotion_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for action in bundle.get("actions") or []:
        if not isinstance(action, dict):
            continue
        local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
        nested = _nested_rollout_action(action)
        target_rule_id = _rollout_target_rule_id(action)
        target_model = _rollout_target_model(action)
        canary = local_update.get("canary") if isinstance(local_update.get("canary"), dict) else {}
        canary_fraction = _as_float_default(
            _first_present(
                action.get("canary_fraction"),
                action.get("rollout_fraction"),
                local_update.get("canary_fraction"),
                canary.get("fraction"),
                canary.get("canary_fraction"),
                _nested_get(action, "evidence_summary", "suggested_canary_fraction"),
            ),
            0.10,
        )
        holdout_fraction = _as_float_default(
            _first_present(
                action.get("holdout_fraction"),
                local_update.get("holdout_fraction"),
                canary.get("holdout_fraction"),
            ),
            0.10,
        )
        if str(action.get("action_type") or "") in {"rollback", "retire", "disable"}:
            canary_fraction = 0.0
            holdout_fraction = 0.0

        update = {
            **local_update,
            "kind": "yaml-rule-canary",
            "policy_source": "managed-recommended",
            "managed_enforced": False,
            "candidate_target_model": target_model,
            "target_model": target_model,
            "source_surface": _first_present(local_update.get("source_surface"), action.get("source_surface")),
            "provider_endpoint": _first_present(local_update.get("provider_endpoint"), action.get("provider_endpoint")),
            "category": _first_present(local_update.get("category"), action.get("category")),
            "model_pattern": _first_present(
                local_update.get("model_pattern"),
                action.get("requested_model"),
                action.get("requested_model_family"),
                nested["proposed"].get("model_pattern"),
            ),
            "canary": {
                **canary,
                "enabled": canary_fraction > 0.0,
                "fraction": canary_fraction,
                "holdout_fraction": holdout_fraction,
                "salt": str(_first_present(canary.get("salt"), action.get("action_id"), target_rule_id)),
                "unit": str(_first_present(canary.get("unit"), canary.get("canary_unit"), "request_fingerprint")),
            },
        }
        if isinstance(action.get("evidence_summary"), dict):
            update["evidence_summary"] = action["evidence_summary"]
        if isinstance(action.get("privacy_summary"), dict):
            update["privacy_summary"] = action["privacy_summary"]

        actions.append({
            "schema": ACTION_SCHEMA,
            "action_id": str(action.get("action_id") or f"optimization-rollout:{target_rule_id}"),
            "action_type": str(action.get("action_type") or "widen"),
            "target_candidate_id": str(action.get("target_candidate_id") or target_rule_id),
            "target_rule_id": target_rule_id,
            "action_family": action.get("action_family"),
            "candidate_family": action.get("candidate_family"),
            "policy_section": action.get("policy_section"),
            "source_surface": action.get("source_surface"),
            "app_family": action.get("app_family"),
            "candidate_target_model": target_model,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "policy_source": "managed-recommended",
            "managed_enforced": False,
            "local_policy_update": update,
            "evidence_summary": action.get("evidence_summary"),
            "privacy_summary": action.get("privacy_summary"),
        })
    return {
        "schema": PROMOTION_ACTIONS_SCHEMA,
        "generated_at": bundle.get("generated_at") or utc_now(),
        "source_schema": bundle.get("schema"),
        "source_generated_at": bundle.get("generated_at"),
        "source_expires_at": bundle.get("expires_at"),
        "actions": actions,
        "omitted": bundle.get("omitted_actions", []),
        "privacy": {
            "metadata_only": True,
            "feature_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }


def normalize_promotion_canary_bundle(bundle: Any) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(bundle, dict):
        return bundle, None
    if bundle.get("schema") != OPTIMIZATION_ROLLOUT_ACTIONS_SCHEMA:
        return bundle, None
    validation = validate_optimization_rollout_bundle(bundle)
    if not validation.get("ok"):
        return None, validation
    converted = _optimization_rollout_to_promotion_bundle(bundle)
    converted["source_optimization_rollout_validation"] = validation
    return converted, validation


def _safe_cohort_features(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in _COHORT_SAFE_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return safe


def _truthy(value: Any) -> bool:
    return value not in (None, False, 0, "", [], {})


def _scan_raw_like(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else f"$.{key_text}"
            if lowered not in _ALLOWED_RAW_LIKE_KEYS and any(part in lowered for part in _RAW_LIKE_KEY_PARTS):
                if _truthy(item):
                    errors.append({"path": child_path, "message": "raw or local-identifier promotion rollout payloads are not accepted"})
                    continue
            _scan_raw_like(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value[:300]):
            _scan_raw_like(item, f"{path}[{index}]", errors)


def promotion_canary_decision(
    action: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    safety_stop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign a promotion rollout action to applied, holdout, or skipped using metadata only."""
    canary_fraction = _as_float(action.get("canary_fraction"))
    holdout_fraction = _as_float(action.get("holdout_fraction"))
    safe_features = _safe_cohort_features(metadata)
    action_id = str(action.get("action_id") or "")
    candidate_id = str(action.get("target_candidate_id") or "")
    target_rule_id = str(action.get("target_rule_id") or "")
    base = {
        "schema": PROMOTION_CANARY_DECISION_SCHEMA,
        "enabled": canary_fraction > 0 or holdout_fraction > 0,
        "selected": False,
        "status": "disabled",
        "cohort": "disabled",
        "reason": "disabled",
        "action_id": action_id,
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "policy_section": action.get("policy_section"),
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
        "raw_content_included": False,
        "safe_feature_keys": sorted(safe_features),
    }
    if safety_stop and safety_stop.get("tripped"):
        base.update({
            "enabled": True,
            "status": "safety_stopped",
            "cohort": "bypassed_or_disabled",
            "reason": "local-canary-safety-stop",
            "safety_stop": safety_stop,
        })
        return base
    if not base["enabled"]:
        return base

    basis = {
        "action_id": action_id,
        "target_candidate_id": candidate_id,
        "target_rule_id": target_rule_id,
        "policy_section": action.get("policy_section"),
        "features": safe_features,
    }
    digest = _stable_hash({"salt": action_id or candidate_id or target_rule_id, "basis": basis})
    bucket = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    base.update({
        "status": "skipped",
        "cohort": "skipped",
        "reason": "outside-canary-and-holdout",
        "bucket": round(bucket, 12),
        "cohort_key_hash": f"sha256:{_stable_hash(basis)}",
    })
    if bucket < holdout_fraction:
        base.update({
            "status": "holdout",
            "cohort": "canary_holdout",
            "reason": "selected-holdout",
        })
    elif bucket < holdout_fraction + canary_fraction:
        base.update({
            "selected": True,
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "selected-canary",
        })
    return base


def evaluate_promotion_canary_safety_stop(
    action: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_source = thresholds if isinstance(thresholds, dict) else {}
    min_samples = _as_int(threshold_source.get("min_samples"), 1)
    max_error_rate = float(threshold_source.get("max_error_rate", 0.05))
    max_5xx_rate = float(threshold_source.get("max_5xx_rate", max_error_rate))
    max_unsupported_model_errors = _as_int(threshold_source.get("max_unsupported_model_errors"), 0)
    max_cache_stale_risk_blockers = _as_int(threshold_source.get("max_cache_stale_risk_blockers"), 0)

    applied = [
        row for row in records
        if str(row.get("cohort") or row.get("status") or "") in {"canary_applied", "applied"}
    ]
    sample_count = len(applied)
    status_codes = [_as_int(row.get("status_code")) for row in applied]
    error_count = sum(1 for code in status_codes if code >= 400)
    five_xx_count = sum(1 for code in status_codes if code >= 500)
    unsupported_model_errors = sum(
        1
        for row in applied
        if "unsupported" in str(row.get("error_bucket") or row.get("error") or "").lower()
        and "model" in str(row.get("error_bucket") or row.get("error") or "").lower()
    )
    stale_risk_blockers = sum(
        1
        for row in applied
        if "stale-risk" in str(row.get("reason") or row.get("blocker") or row.get("error_bucket") or "").lower()
    )
    error_rate = error_count / sample_count if sample_count else 0.0
    five_xx_rate = five_xx_count / sample_count if sample_count else 0.0
    reason_codes: list[str] = []
    if sample_count >= min_samples and error_rate > max_error_rate:
        reason_codes.append("error-rate")
    if sample_count >= min_samples and five_xx_rate > max_5xx_rate:
        reason_codes.append("provider-5xx-rate")
    if unsupported_model_errors > max_unsupported_model_errors:
        reason_codes.append("unsupported-model-errors")
    if stale_risk_blockers > max_cache_stale_risk_blockers:
        reason_codes.append("cache-stale-risk-blockers")
    return {
        "schema": PROMOTION_CANARY_SAFETY_SCHEMA,
        "enabled": True,
        "status": "tripped" if reason_codes else ("insufficient-samples" if sample_count < min_samples else "ok"),
        "tripped": bool(reason_codes),
        "action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "sample_count": sample_count,
        "min_samples": min_samples,
        "error_count": error_count,
        "error_rate": round(error_rate, 6),
        "five_xx_count": five_xx_count,
        "five_xx_rate": round(five_xx_rate, 6),
        "unsupported_model_errors": unsupported_model_errors,
        "cache_stale_risk_blockers": stale_risk_blockers,
        "reason_codes": reason_codes,
    }


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {"rules": []}, None
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if isinstance(data, list):
        data = {"rules": data}
    if not isinstance(data, dict):
        data = {"rules": []}
    if not isinstance(data.get("rules"), list):
        data["rules"] = []
    return data, text


def _backup_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_policy_file(path: Path, text: str, *, backup_id: str | None = None) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: str | None = None
    if path.exists():
        suffix = backup_id if backup_id else _backup_suffix()
        backup = path.with_name(f"{path.name}.bak-{suffix}")
        backup.write_bytes(path.read_bytes())
        backup_path = str(backup)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return backup_path


def _normalize_pattern_hash(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        digest = text.split(":", 1)[1]
    else:
        digest = text
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return f"sha256:{digest}"


def _collect_pattern_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if (
                key_l in {"pattern_hash", "normalized_pattern_hash", "crunch_pattern_hash", "pattern_hashes", "hashes"}
                or key_l.endswith(("_pattern_hash", "_pattern_sha256"))
                or ("pattern" in key_l and key_l.endswith(("_hash", "_hashes", "_sha256")))
            ):
                if isinstance(item, list):
                    hashes.extend(hash_value for nested in item if (hash_value := _normalize_pattern_hash(nested)))
                elif (hash_value := _normalize_pattern_hash(item)) is not None:
                    hashes.append(hash_value)
            hashes.extend(_collect_pattern_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.extend(_collect_pattern_hashes(item))
    return sorted(set(hashes))


def _safe_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _routing_phase_canary(action: dict[str, Any]) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    canary = local_update.get("canary") if isinstance(local_update.get("canary"), dict) else {}
    target_model = _first_present(local_update.get("candidate_target_model"), local_update.get("target_model"), action.get("candidate_target_model"), "haiku")
    disabled = str(action.get("action_type") or "") == "rollback" or _as_float(action.get("canary_fraction")) <= 0
    category = _first_present(local_update.get("category"), action.get("category"))
    eligible_categories = _safe_string_list(local_update.get("eligible_categories"))
    if not eligible_categories and category:
        eligible_categories = [str(category)]
    excluded_categories = _safe_string_list(local_update.get("excluded_categories"))
    if not excluded_categories:
        excluded_categories = ["code-gen"]
    return {
        "enabled": not disabled,
        "policy_id": str(action.get("target_rule_id") or action.get("target_candidate_id") or "promotion-routing-canary"),
        "promotion_action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "policy_source": "managed-recommended",
        "provider": str(_first_present(local_update.get("provider"), local_update.get("provider_family"), action.get("provider"), "anthropic")),
        "source_surface": str(_first_present(local_update.get("source_surface"), action.get("source_surface"), "anthropic_messages")),
        "app_family": str(_first_present(local_update.get("app_family"), action.get("app_family"), "anthropic")),
        "model_pattern": str(_first_present(local_update.get("model_pattern"), action.get("requested_model_family"), "sonnet")),
        "target_model": target_model,
        "requested_model": str(_first_present(local_update.get("requested_model"), action.get("requested_model"), action.get("requested_model_family"), "sonnet")),
        "routed_model": str(_first_present(local_update.get("routed_model"), action.get("routed_model"), target_model)),
        "stream": _as_bool(_first_present(local_update.get("stream"), action.get("stream")), False),
        "eligible_workflow_phases": _safe_string_list(local_update.get("eligible_workflow_phases")) or ["tool-execution", "summary"],
        "excluded_workflow_phases": _safe_string_list(local_update.get("excluded_workflow_phases")) or ["planning", "thinking", "unknown"],
        "eligible_categories": eligible_categories,
        "excluded_categories": excluded_categories,
        "min_workflow_phase_confidence": str(local_update.get("min_workflow_phase_confidence") or "medium"),
        "min_text_chars": _as_int(local_update.get("min_text_chars"), 0),
        "max_text_chars": _as_int(local_update.get("max_text_chars"), 30000),
        "canary_fraction": 0.0 if disabled else _as_float(action.get("canary_fraction")),
        "holdout_fraction": 0.0 if disabled else _as_float(action.get("holdout_fraction")),
        "salt": str(_first_present(canary.get("salt"), action.get("action_id"), action.get("target_candidate_id"), "promotion-routing-canary")),
        "cohort_unit": str(_first_present(local_update.get("cohort_unit"), canary.get("unit"), canary.get("canary_unit"), "session")),
        "safety_gates": {
            "block_thinking_history": True,
            "block_top_level_thinking": True,
            "strip_model_incompatible_params": True,
            "fallback_to_requested_on_rate_limit": True,
            "content_free": True,
            "provider_calls_made_by_apply": False,
        },
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 10,
            "min_holdout_samples": 5,
            "max_error_rate": 0.05,
            "max_retry_rate": 0.20,
            "max_fallback_rate": 0.20,
            "max_latency_regression_ratio": 1.50,
            "limit": 500,
        },
    }


def _routing_openai_canary(action: dict[str, Any]) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    canary = local_update.get("canary") if isinstance(local_update.get("canary"), dict) else {}
    target_model = _first_present(local_update.get("candidate_target_model"), local_update.get("target_model"), action.get("candidate_target_model"), "gpt-5-mini")
    disabled = str(action.get("action_type") or "") in {"rollback", "retire", "disable"} or _as_float(action.get("canary_fraction")) <= 0
    category = _first_present(local_update.get("category"), action.get("category"))
    eligible_categories = _safe_string_list(local_update.get("eligible_categories"))
    if not eligible_categories and category:
        eligible_categories = [str(category)]
    if not eligible_categories:
        eligible_categories = ["chat", "short-completion"]
    excluded_categories = _safe_string_list(local_update.get("excluded_categories")) or [
        "tool-result",
        "tool-heavy",
        "tool-light",
        "code-gen",
        "long-context",
    ]
    return {
        "enabled": not disabled,
        "policy_id": str(action.get("target_rule_id") or action.get("target_candidate_id") or "promotion-openai-routing-canary"),
        "promotion_action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "policy_source": "managed-recommended",
        "model_pattern": str(_first_present(local_update.get("model_pattern"), action.get("requested_model_family"), "gpt-5")),
        "target_model": target_model,
        "eligible_categories": eligible_categories,
        "excluded_categories": excluded_categories,
        "allow_tools": _as_bool(local_update.get("allow_tools"), False),
        "allow_stream": _as_bool(local_update.get("allow_stream"), False),
        "min_text_chars": _as_int(local_update.get("min_text_chars"), 0),
        "max_text_chars": _as_int(local_update.get("max_text_chars"), 8000),
        "min_input_tokens_est": _as_int(local_update.get("min_input_tokens_est"), 0),
        "max_input_tokens_est": _as_int(local_update.get("max_input_tokens_est"), 2000),
        "canary_fraction": 0.0 if disabled else _as_float(action.get("canary_fraction")),
        "holdout_fraction": 0.0 if disabled else _as_float(action.get("holdout_fraction")),
        "salt": str(_first_present(canary.get("salt"), action.get("action_id"), action.get("target_candidate_id"), "promotion-openai-routing-canary")),
        "safety_stop": {
            "enabled": True,
            "window_hours": 24,
            "min_samples": 20,
            "min_holdout_samples": 10,
            "max_error_rate": 0.03,
            "max_retry_rate": 0.10,
            "max_fallback_rate": 0.10,
            "max_latency_regression_ratio": 1.50,
            "limit": 1000,
        },
    }


def _managed_enforced(action: dict[str, Any]) -> bool:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    return (
        bool(action.get("managed_enforced"))
        or bool(local_update.get("managed_enforced"))
        or str(local_update.get("policy_source") or "").strip() == "managed-enforced"
        or str(action.get("policy_source") or "").strip() == "managed-enforced"
    )


def _crunch_pattern_rule(action: dict[str, Any], *, existing: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    conditions_update = local_update.get("conditions") if isinstance(local_update.get("conditions"), dict) else {}
    action_update = local_update.get("action") if isinstance(local_update.get("action"), dict) else {}
    if not action_update and isinstance(local_update.get("crunch_action"), dict):
        action_update = local_update["crunch_action"]

    disabled = str(action.get("action_type") or "") == "rollback" or _as_float(action.get("canary_fraction")) <= 0
    pattern_hashes = (
        _collect_pattern_hashes(conditions_update)
        or _collect_pattern_hashes(local_update)
        or _collect_pattern_hashes(action)
    )
    if not pattern_hashes and isinstance(existing, dict):
        pattern_hashes = _collect_pattern_hashes(existing.get("conditions")) or _collect_pattern_hashes(existing)
    if not pattern_hashes:
        return None, "missing-pattern-hashes"

    existing_conditions = existing.get("conditions") if isinstance(existing, dict) and isinstance(existing.get("conditions"), dict) else {}
    conditions: dict[str, Any] = {
        "pattern_hashes": pattern_hashes,
        "min_repeated_count": _as_int(
            conditions_update.get("min_repeated_count", existing_conditions.get("min_repeated_count", 2)),
            2,
        ),
        "keep_recent_matches": _as_int(
            conditions_update.get("keep_recent_matches", existing_conditions.get("keep_recent_matches", 1)),
            1,
        ),
    }
    for key in ("model_pattern", "category", "workflow_phase"):
        value = conditions_update.get(key, existing_conditions.get(key))
        if value is not None:
            conditions[key] = str(value)
    category_not_in = conditions_update.get("category_not_in", existing_conditions.get("category_not_in"))
    if category_not_in is not None:
        conditions["category_not_in"] = _safe_string_list(category_not_in)
    for key in ("min_text_chars", "max_text_chars", "max_applications"):
        value = conditions_update.get(key, existing_conditions.get(key))
        if value is not None:
            conditions[key] = _as_int(value)

    existing_action = existing.get("action") if isinstance(existing, dict) and isinstance(existing.get("action"), dict) else {}
    action_type = str(action_update.get("type") or action_update.get("kind") or existing_action.get("type") or "shorten").strip().lower()
    if action_type not in {"shorten", "omit"}:
        action_type = "shorten"
    pattern_action: dict[str, Any] = {
        "type": action_type,
        "head_chars": _as_int(action_update.get("head_chars", existing_action.get("head_chars", 1200)), 1200),
        "tail_chars": _as_int(action_update.get("tail_chars", existing_action.get("tail_chars", 800)), 800),
        "max_replacement_chars": _as_int(
            action_update.get("max_replacement_chars", existing_action.get("max_replacement_chars", 2400)),
            2400,
        ),
    }
    marker = action_update.get("marker", existing_action.get("marker"))
    if marker is not None:
        pattern_action["marker"] = str(marker)

    safety_update = local_update.get("safety_stop") if isinstance(local_update.get("safety_stop"), dict) else {}
    existing_rollout = existing.get("rollout") if isinstance(existing, dict) and isinstance(existing.get("rollout"), dict) else {}
    rollout: dict[str, Any] = {
        "schema": "tokenclaw.pattern_policy_rollout.v1",
        "recommendation_mode": "canary",
        "canary_enabled": not disabled,
        "canary_fraction": 0.0 if disabled else _as_float(action.get("canary_fraction")),
        "holdout_fraction": 0.0 if disabled else _as_float(action.get("holdout_fraction")),
        "canary_salt": str(action.get("action_id") or action.get("target_candidate_id") or action.get("target_rule_id") or ""),
        "canary_unit": str(local_update.get("canary_unit") or existing_rollout.get("canary_unit") or "request_fingerprint"),
        "rollback_threshold": safety_update.get(
            "rollback_threshold",
            safety_update.get("max_regression_rate", existing_rollout.get("rollback_threshold", 0.2)),
        ),
        "min_outcome_samples": _as_int(
            safety_update.get("min_outcome_samples", safety_update.get("min_samples", existing_rollout.get("min_outcome_samples", 5))),
            5,
        ),
    }
    if safety_update.get("window") is not None:
        rollout["window"] = _as_int(safety_update.get("window"))

    rule: dict[str, Any] = {
        "id": str(action.get("target_rule_id") or action.get("target_candidate_id") or (existing or {}).get("id") or "promotion-crunch-canary"),
        "enabled": not disabled,
        "policy_source": "managed-recommended",
        "candidate_id": action.get("target_candidate_id") or (existing or {}).get("candidate_id"),
        "promotion_action_id": action.get("action_id"),
        "conditions": conditions,
        "action": pattern_action,
        "rollout": rollout,
        "rollout_action": {
            "schema": ACTION_SCHEMA,
            "action_type": action.get("action_type"),
            "target_candidate_id": action.get("target_candidate_id"),
            "target_rule_id": action.get("target_rule_id"),
            "canary_fraction": rollout["canary_fraction"],
            "holdout_fraction": rollout["holdout_fraction"],
            "managed_enforced": False,
        },
    }
    module_family = local_update.get("module_family") or local_update.get("candidate_profile") or action.get("candidate_profile")
    if module_family is not None:
        rule["module_family"] = str(module_family)
    description = local_update.get("description") or (existing or {}).get("description")
    if description is not None:
        rule["description"] = str(description)
    return rule, None


def _nested_get_bool(*sources: Any, default: bool = False) -> bool:
    for value in sources:
        if value is not None:
            return _as_bool(value, default)
    return default


def _cache_pattern_rule(action: dict[str, Any], *, existing: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    conditions_update = local_update.get("conditions") if isinstance(local_update.get("conditions"), dict) else {}
    action_update = local_update.get("action") if isinstance(local_update.get("action"), dict) else {}
    if not action_update and isinstance(local_update.get("cache_action"), dict):
        action_update = local_update["cache_action"]

    disabled = str(action.get("action_type") or "") == "rollback"
    pattern_hashes = (
        _collect_pattern_hashes(conditions_update)
        or _collect_pattern_hashes(local_update)
        or _collect_pattern_hashes(action)
    )
    if not pattern_hashes and isinstance(existing, dict):
        pattern_hashes = _collect_pattern_hashes(existing.get("conditions")) or _collect_pattern_hashes(existing)
    if not pattern_hashes:
        return None, "missing-pattern-hashes"

    existing_conditions = existing.get("conditions") if isinstance(existing, dict) and isinstance(existing.get("conditions"), dict) else {}
    conditions: dict[str, Any] = {"pattern_hashes": pattern_hashes}
    for key in (
        "model_pattern",
        "category",
        "workflow_phase",
        "source_surface",
        "app_family",
        "text_bucket",
        "token_bucket",
        "cacheability_bucket",
    ):
        value = conditions_update.get(key, local_update.get(key, existing_conditions.get(key)))
        if value is not None:
            conditions[key] = str(value)
    for key in ("replayability_levels", "replayability_level", "category_not_in"):
        value = conditions_update.get(key, local_update.get(key, existing_conditions.get(key)))
        if value is not None:
            conditions[key] = _safe_string_list(value)
    if "replayability_levels" not in conditions and "replayability_level" not in conditions:
        conditions["replayability_levels"] = _safe_string_list(
            local_update.get("replayability_level")
            or action.get("replayability_level")
            or existing_conditions.get("replayability_levels")
            or "local-exact-response"
        )
    for key in (
        "has_tools",
        "stream",
        "static_information_hint",
        "time_sensitive_hint",
        "user_specific_hint",
        "exact_cache_candidate_hint",
    ):
        if key in conditions_update or key in local_update or key in existing_conditions:
            conditions[key] = _nested_get_bool(
                conditions_update.get(key),
                local_update.get(key),
                existing_conditions.get(key),
            )

    existing_action = existing.get("action") if isinstance(existing, dict) and isinstance(existing.get("action"), dict) else {}
    has_tools = _nested_get_bool(
        conditions.get("has_tools"),
        action_update.get("allow_tool_calls"),
        local_update.get("has_tools"),
        action.get("has_tools"),
        existing_conditions.get("has_tools"),
        default=False,
    )
    stream = _nested_get_bool(
        conditions.get("stream"),
        action_update.get("streaming"),
        local_update.get("stream"),
        action.get("stream"),
        existing_conditions.get("stream"),
        default=False,
    )
    category = str(conditions.get("category") or action.get("category") or "").lower()
    tool_or_stream_related = has_tools or stream or category in {"tool-result", "tool-heavy", "tool-light"}
    allow_tool_calls = _nested_get_bool(
        action_update.get("allow_tool_calls"),
        existing_action.get("allow_tool_calls"),
        has_tools,
        default=False,
    )
    safe_invalidation = _nested_get_bool(
        action_update.get("safe_invalidation"),
        action_update.get("safe_invalidation_evidence"),
        local_update.get("safe_invalidation"),
        local_update.get("safe_invalidation_evidence"),
        local_update.get("file_dependency_evidence_available"),
        action.get("safe_invalidation_evidence"),
        action.get("file_dependency_evidence_available"),
        existing_action.get("safe_invalidation"),
        existing_action.get("safe_invalidation_evidence"),
        default=False,
    )
    if tool_or_stream_related and not safe_invalidation and not disabled:
        return None, "missing-safe-invalidation-evidence"
    if (has_tools or category.startswith("tool-")) and not allow_tool_calls and not disabled:
        return None, "tool-cache-not-allowed"

    conditions["has_tools"] = has_tools
    conditions["stream"] = stream
    pattern_action: dict[str, Any] = {
        "type": str(action_update.get("type") or existing_action.get("type") or "exact_cache_pattern"),
        "allow_tool_calls": bool(allow_tool_calls),
        "safe_invalidation": bool(safe_invalidation),
        "safe_invalidation_evidence": bool(safe_invalidation),
        "streaming": bool(
            _nested_get_bool(
                action_update.get("streaming"),
                existing_action.get("streaming"),
                stream,
                default=False,
            )
        ),
    }
    if pattern_action["type"] not in {"exact_cache", "exact_cache_pattern"}:
        pattern_action["type"] = "exact_cache_pattern"
    estimated_saved = action_update.get("estimated_saved_cost_usd")
    if estimated_saved is None and isinstance(action.get("evidence_summary"), dict):
        estimated_saved = action["evidence_summary"].get("projected_savings_usd")
    if estimated_saved is not None:
        try:
            pattern_action["estimated_saved_cost_usd"] = max(0.0, float(estimated_saved))
        except (TypeError, ValueError):
            pass

    safety_update = local_update.get("safety_stop") if isinstance(local_update.get("safety_stop"), dict) else {}
    existing_rollout = existing.get("rollout") if isinstance(existing, dict) and isinstance(existing.get("rollout"), dict) else {}
    rollout: dict[str, Any] = {
        "schema": "tokenclaw.pattern_policy_rollout.v1",
        "recommendation_mode": "canary",
        "canary_enabled": not disabled,
        "canary_fraction": 0.0 if disabled else _as_float(action.get("canary_fraction")),
        "holdout_fraction": 0.0 if disabled else _as_float(action.get("holdout_fraction")),
        "canary_salt": str(action.get("action_id") or action.get("target_candidate_id") or action.get("target_rule_id") or ""),
        "canary_unit": str(local_update.get("canary_unit") or existing_rollout.get("canary_unit") or "request_fingerprint"),
        "rollback_threshold": safety_update.get(
            "rollback_threshold",
            safety_update.get("max_regression_rate", existing_rollout.get("rollback_threshold", 0.2)),
        ),
        "min_outcome_samples": _as_int(
            safety_update.get("min_outcome_samples", safety_update.get("min_samples", existing_rollout.get("min_outcome_samples", 5))),
            5,
        ),
        "local_feedback_fields": [
            "cache.status",
            "cache.reason",
            "cache.pattern_rule.canary.cohort",
            "status_code",
            "retry_count",
            "latency_ms",
        ],
    }

    rule: dict[str, Any] = {
        "id": str(action.get("target_rule_id") or action.get("target_candidate_id") or (existing or {}).get("id") or "promotion-cache-canary"),
        "enabled": not disabled,
        "policy_source": "managed-recommended",
        "candidate_id": action.get("target_candidate_id") or (existing or {}).get("candidate_id"),
        "promotion_action_id": action.get("action_id"),
        "conditions": conditions,
        "action": pattern_action,
        "rollout": rollout,
        "rollout_action": {
            "schema": ACTION_SCHEMA,
            "action_type": action.get("action_type"),
            "target_candidate_id": action.get("target_candidate_id"),
            "target_rule_id": action.get("target_rule_id"),
            "canary_fraction": rollout["canary_fraction"],
            "holdout_fraction": rollout["holdout_fraction"],
            "managed_enforced": False,
        },
    }
    module_family = local_update.get("module_family") or local_update.get("candidate_profile") or action.get("candidate_profile")
    if module_family is not None:
        rule["module_family"] = str(module_family)
    description = local_update.get("description") or (existing or {}).get("description")
    if description is not None:
        rule["description"] = str(description)
    return rule, None


def _is_old_context_summary_action(action: dict[str, Any]) -> bool:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    values = [
        action.get("action_family"),
        action.get("optimization_family"),
        local_update.get("kind"),
        local_update.get("module_family"),
        local_update.get("candidate_profile"),
    ]
    if isinstance(local_update.get("old_context_summarization"), dict):
        return True
    normalized = [str(value or "").strip().lower().replace("_", "-") for value in values]
    return any("old-context" in value or "old-context-summary" in value or "old-context-summarization" in value for value in normalized)


def _find_summary_rule(summary: Any, action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    rule_id = str(summary.get("rule_id") or "").strip()
    candidate_id = str(summary.get("candidate_id") or "").strip()
    if target_rule_id and rule_id and target_rule_id != rule_id:
        return None
    if target_candidate_id and candidate_id and target_candidate_id != candidate_id:
        return None
    return summary


def _old_context_summary_policy(action: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    summary_update = (
        local_update.get("old_context_summarization")
        if isinstance(local_update.get("old_context_summarization"), dict)
        else {}
    )
    disabled = str(action.get("action_type") or "") == "rollback"
    existing = existing if isinstance(existing, dict) else {}
    summary = dict(existing)
    summary["enabled"] = False if disabled else _as_bool(summary_update.get("enabled"), _as_bool(existing.get("enabled"), True))
    summary["rule_id"] = str(action.get("target_rule_id") or summary_update.get("rule_id") or existing.get("rule_id") or "promotion-old-context-summary")
    summary["candidate_id"] = str(action.get("target_candidate_id") or summary_update.get("candidate_id") or existing.get("candidate_id") or "")
    summary["policy_source"] = "managed-recommended"
    summary["promotion_action_id"] = action.get("action_id")
    for key in ("model", "model_hint", "profile", "placement", "excluded_categories"):
        value = summary_update.get(key)
        if value is None and key == "model":
            value = local_update.get("candidate_target_model")
        if value is not None:
            target_key = "model" if key == "model_hint" else key
            summary[target_key] = value
    for key in (
        "min_request_chars",
        "min_summarized_chars",
        "max_turns",
        "keep_recent_turns",
        "max_summary_chars",
        "max_source_chars",
    ):
        value = summary_update.get(key)
        if value is not None:
            summary[key] = _as_int(value)
    if summary_update.get("max_summary_cost_usd") is not None:
        try:
            summary["max_summary_cost_usd"] = max(0.0, float(summary_update["max_summary_cost_usd"]))
        except (TypeError, ValueError):
            pass
    for key in ("block_tool_protocol", "block_thinking"):
        if summary_update.get(key) is not None:
            summary[key] = _as_bool(summary_update.get(key), True)

    existing_canary = existing.get("canary") if isinstance(existing.get("canary"), dict) else {}
    update_canary = local_update.get("canary") if isinstance(local_update.get("canary"), dict) else {}
    canary_fraction = _as_float(update_canary.get("fraction", update_canary.get("canary_fraction", action.get("canary_fraction"))))
    holdout_fraction = _as_float(update_canary.get("holdout_fraction", action.get("holdout_fraction")))
    summary["canary"] = {
        **existing_canary,
        "enabled": False if disabled else _as_bool(update_canary.get("enabled"), True),
        "fraction": 0.0 if disabled else canary_fraction,
        "holdout_fraction": 0.0 if disabled else holdout_fraction,
        "salt": str(update_canary.get("salt") or update_canary.get("canary_salt") or action.get("action_id") or summary["rule_id"]),
        "unit": str(update_canary.get("unit") or update_canary.get("canary_unit") or existing_canary.get("unit") or "source_hash"),
    }

    existing_safety = existing.get("safety_stop") if isinstance(existing.get("safety_stop"), dict) else {}
    update_safety = local_update.get("safety_stop") if isinstance(local_update.get("safety_stop"), dict) else {}
    safety = dict(existing_safety)
    safety["enabled"] = _as_bool(update_safety.get("enabled"), _as_bool(existing_safety.get("enabled"), True))
    for key in ("min_outcome_samples", "window"):
        value = update_safety.get(key)
        if value is not None:
            safety[key] = _as_int(value)
    for key in (
        "max_error_rate",
        "max_retry_rate",
        "max_negative_net_savings_rate",
        "max_summary_failure_rate",
        "max_error_rate_delta",
    ):
        value = update_safety.get(key)
        if value is not None:
            try:
                safety[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    summary["safety_stop"] = safety
    summary["rollout_action"] = {
        "schema": ACTION_SCHEMA,
        "action_type": action.get("action_type"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id"),
        "canary_fraction": summary["canary"]["fraction"],
        "holdout_fraction": summary["canary"]["holdout_fraction"],
        "managed_enforced": False,
    }
    return {key: value for key, value in summary.items() if value is not None}


def _find_pattern_rule(rules: list[Any], action: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    target_rule_id = str(action.get("target_rule_id") or "").strip()
    target_candidate_id = str(action.get("target_candidate_id") or "").strip()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        candidate_id = str(rule.get("candidate_id") or rule.get("recommendation_id") or rule.get("policy_id") or "").strip()
        if target_rule_id and rule_id == target_rule_id:
            return index, rule
        if target_candidate_id and candidate_id == target_candidate_id:
            return index, rule
    return None, None


def apply_optimization_promotion_canaries(
    bundle: Any,
    *,
    config_dir: str | Path,
    dry_run: bool = True,
    sections: list[str] | tuple[str, ...] | None = None,
    backup_id: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_dir).expanduser()
    requested_sections = set(sections or _POLICY_SECTION_FILES)
    errors: list[dict[str, str]] = []
    actions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    source_validation: dict[str, Any] | None = None
    normalized, validation = normalize_promotion_canary_bundle(bundle)
    if validation is not None:
        source_validation = validation
        if normalized is None:
            errors.extend(validation.get("errors", []))
            raw_actions = []
        else:
            bundle = normalized
    if not isinstance(bundle, dict) or bundle.get("schema") != PROMOTION_ACTIONS_SCHEMA:
        errors.append({"path": "$.schema", "message": f"expected {PROMOTION_ACTIONS_SCHEMA}"})
    raw_actions = bundle.get("actions") if isinstance(bundle, dict) else None
    if not isinstance(raw_actions, list):
        raw_actions = []
        errors.append({"path": "$.actions", "message": "expected list"})
    invalid_sections = sorted(requested_sections - set(_POLICY_SECTION_FILES))
    for section in invalid_sections:
        errors.append({"path": f"$.sections.{section}", "message": "unsupported promotion canary policy section"})
    if isinstance(bundle, dict):
        _scan_raw_like(bundle, "$", errors)

    file_plans: dict[str, dict[str, Any]] = {}

    def _file_plan(section: str) -> dict[str, Any]:
        if section not in file_plans:
            path = config_path / _POLICY_SECTION_FILES[section]
            data, old_text = _load_yaml(path)
            if section in {"crunch", "cache"} and not isinstance(data.get("pattern_rules"), list):
                data["pattern_rules"] = []
            file_plans[section] = {"section": section, "path": path, "data": data, "old_text": old_text}
        return file_plans[section]

    if not errors:
        for index, action in enumerate(raw_actions):
            if not isinstance(action, dict):
                errors.append({"path": f"$.actions[{index}]", "message": "expected action object"})
                continue
            if action.get("schema") != ACTION_SCHEMA:
                errors.append({"path": f"$.actions[{index}].schema", "message": f"expected {ACTION_SCHEMA}"})
                continue
            if _managed_enforced(action):
                errors.append({"path": f"$.actions[{index}]", "message": "managed-enforced promotion canaries are not accepted"})
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "managed-enforced-not-accepted",
                    "policy_section": action.get("policy_section"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            section = str(action.get("policy_section") or "")
            if section not in requested_sections:
                actions.append({"path": f"$.actions[{index}]", "status": "skipped", "reason": "not-requested", "policy_section": section})
                continue
            if section not in _POLICY_SECTION_FILES:
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "skipped",
                    "reason": "unsupported-policy-section",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            plan = _file_plan(section)
            if section == "routing":
                local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
                target_key = "openai_canary" if _is_openai_routing_action(action) else "phase_canary"
                existing_canary = plan["data"].get(target_key)
                if (
                    isinstance(existing_canary, dict)
                    and _as_bool(existing_canary.get("enabled"), False)
                    and str(existing_canary.get("policy_source") or "") not in {"", "managed-recommended"}
                ):
                    errors.append({"path": f"$.actions[{index}]", "message": "routing canary targets an enabled non-managed local rule"})
                    actions.append({
                        "path": f"$.actions[{index}]",
                        "status": "rejected",
                        "reason": "unsafe-policy-source",
                        "policy_section": "routing",
                        "target_candidate_id": action.get("target_candidate_id"),
                        "target_rule_id": action.get("target_rule_id"),
                    })
                    continue
                plan["data"][target_key] = _routing_openai_canary(action) if target_key == "openai_canary" else _routing_phase_canary(action)
                canary_policy = plan["data"][target_key]
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "planned",
                    "policy_section": "routing",
                    "target_local_policy": target_key,
                    "action_type": action.get("action_type"),
                    "action_family": action.get("action_family"),
                    "optimization_family": action.get("optimization_family"),
                    "source_surface": action.get("source_surface"),
                    "app_family": action.get("app_family"),
                    "policy_source": action.get("policy_source") or local_update.get("policy_source") or "managed-recommended",
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                    "action_id": action.get("action_id"),
                    "rule_id": canary_policy.get("policy_id"),
                    "requested_model_family": action.get("requested_model_family") or action.get("requested_model") or local_update.get("model_pattern"),
                    "routed_model_family": action.get("routed_model_family") or action.get("candidate_target_model") or canary_policy.get("target_model"),
                    "target_model": canary_policy.get("target_model"),
                    "canary_fraction": canary_policy["canary_fraction"],
                    "holdout_fraction": canary_policy["holdout_fraction"],
                })
                continue

            if section == "crunch" and _is_old_context_summary_action(action):
                summary = plan["data"].get("old_context_summarization")
                existing_summary = _find_summary_rule(summary, action)
                if isinstance(summary, dict) and existing_summary is None:
                    errors.append({"path": f"$.actions[{index}]", "message": "promotion canary targets an unknown old-context summarization rule"})
                    actions.append({
                        "path": f"$.actions[{index}]",
                        "status": "rejected",
                        "reason": "unknown-rule",
                        "policy_section": section,
                        "target_candidate_id": action.get("target_candidate_id"),
                        "target_rule_id": action.get("target_rule_id"),
                    })
                    continue
                if existing_summary is not None and str(existing_summary.get("policy_source") or "") != "managed-recommended":
                    errors.append({"path": f"$.actions[{index}]", "message": "promotion canary targets a non-managed-recommended old-context summarization rule"})
                    actions.append({
                        "path": f"$.actions[{index}]",
                        "status": "rejected",
                        "reason": "unsafe-policy-source",
                        "policy_section": section,
                        "target_candidate_id": action.get("target_candidate_id"),
                        "target_rule_id": action.get("target_rule_id"),
                    })
                    continue
                if str(action.get("action_type") or "") == "rollback" and existing_summary is None:
                    errors.append({"path": f"$.actions[{index}]", "message": "rollback targets an unknown old-context summarization rule"})
                    actions.append({
                        "path": f"$.actions[{index}]",
                        "status": "rejected",
                        "reason": "unknown-rule",
                        "policy_section": section,
                        "target_candidate_id": action.get("target_candidate_id"),
                        "target_rule_id": action.get("target_rule_id"),
                    })
                    continue
                summary_rule = _old_context_summary_policy(action, existing=existing_summary)
                plan["data"]["old_context_summarization"] = summary_rule
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "planned",
                    "policy_section": section,
                    "action_type": action.get("action_type"),
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                    "action_id": action.get("action_id"),
                    "rule_id": summary_rule.get("rule_id"),
                    "canary_fraction": (summary_rule.get("canary") or {}).get("fraction"),
                    "holdout_fraction": (summary_rule.get("canary") or {}).get("holdout_fraction"),
                })
                continue

            pattern_rules = plan["data"]["pattern_rules"]
            rule_index, existing_rule = _find_pattern_rule(pattern_rules, action)
            if existing_rule is not None and str(existing_rule.get("policy_source") or "") != "managed-recommended":
                errors.append({"path": f"$.actions[{index}]", "message": f"promotion canary targets a non-managed-recommended {section} rule"})
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unsafe-policy-source",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            if str(action.get("action_type") or "") == "rollback" and existing_rule is None:
                errors.append({"path": f"$.actions[{index}]", "message": f"rollback targets an unknown {section} rule"})
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": "unknown-rule",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            if section == "cache":
                rule, reason = _cache_pattern_rule(action, existing=existing_rule)
            else:
                rule, reason = _crunch_pattern_rule(action, existing=existing_rule)
            if rule is None:
                errors.append({"path": f"$.actions[{index}]", "message": reason or f"invalid {section} pattern rule"})
                actions.append({
                    "path": f"$.actions[{index}]",
                    "status": "rejected",
                    "reason": reason or f"invalid-{section}-pattern-rule",
                    "policy_section": section,
                    "target_candidate_id": action.get("target_candidate_id"),
                    "target_rule_id": action.get("target_rule_id"),
                })
                continue
            if rule_index is None:
                pattern_rules.append(rule)
            else:
                pattern_rules[rule_index] = rule
            actions.append({
                "path": f"$.actions[{index}]",
                "status": "planned",
                "policy_section": section,
                "action_type": action.get("action_type"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id"),
                "action_id": action.get("action_id"),
                "rule_id": rule.get("id"),
                "rule_index": len(pattern_rules) - 1 if rule_index is None else rule_index,
                "canary_fraction": rule["rollout"]["canary_fraction"],
                "holdout_fraction": rule["rollout"]["holdout_fraction"],
            })

    if not errors:
        planned_sections = {
            str(action.get("policy_section"))
            for action in actions
            if action.get("status") == "planned"
        }
        for section in sorted(planned_sections):
            plan = file_plans[section]
            text = yaml.safe_dump(plan["data"], sort_keys=False)
            changed = plan["old_text"] != text
            backup_path = None
            if changed and not dry_run:
                backup_path = _write_policy_file(plan["path"], text, backup_id=backup_id)
            files.append({
                "section": section,
                "path": str(plan["path"]),
                "changed": bool(changed),
                "backup_path": backup_path,
                "bytes_after": len(text.encode("utf-8")),
            })

    return {
        "schema": PROMOTION_CANARY_APPLY_SCHEMA,
        "ok": not errors,
        "generated_at": utc_now(),
        "dry_run": bool(dry_run),
        "read_only": bool(dry_run),
        "wrote_policy_files": bool(files and not dry_run and any(file.get("changed") for file in files)),
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "config_dir": str(config_path),
        "actions": actions,
        "files": files,
        "source_optimization_rollout_validation": source_validation,
        "summary": {
            "action_count": len(raw_actions),
            "planned_action_count": sum(1 for action in actions if action.get("status") == "planned"),
            "skipped_action_count": sum(1 for action in actions if action.get("status") == "skipped"),
            "error_count": len(errors),
        },
        "errors": errors,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "raw_session_ids_included": False,
            "filesystem_paths_included": False,
        },
    }
