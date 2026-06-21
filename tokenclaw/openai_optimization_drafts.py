from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenclaw.policy_files import POLICY_DRAFT_STAGE_SCHEMA, stage_policy_draft
from tokenclaw.store import utc_now


OPENAI_OPTIMIZATION_DRAFT_METADATA_SCHEMA = "tokenclaw.openai_optimization_review_draft_metadata.v1"
OPENAI_OPTIMIZATION_DRAFT_VALIDATION_SCHEMA = "tokenclaw.openai_optimization_review_draft_validation.v1"
SUPPORTED_ACTION_FAMILIES = {"routing", "old_context_summarization", "cache"}
RAW_EXACT_KEYS = {
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "content",
    "contents",
    "file_content",
    "file_path",
    "file_paths",
    "message",
    "messages",
    "password",
    "prompt",
    "prompts",
    "provider_body",
    "raw_context",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_id",
    "request_ids",
    "response",
    "secret",
    "session_id",
    "session_ids",
    "system_prompt",
    "tenant_id",
    "thread_id",
    "tool_input",
    "tool_payload",
    "tool_result",
    "transcript",
}
UNSAFE_TRUE_FLAGS = {
    "provider_forwarding",
    "server_content_processing",
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_provider_bodies_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_commands_included",
    "raw_paths_included",
    "raw_request_ids_included",
    "raw_terminal_text_included",
    "raw_body_storage",
    "request_ids_returned",
    "tenant_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
}
SAFE_RAW_FLAG_KEYS = {
    "raw_payloads_returned",
    "raw_prompts_returned",
    "raw_responses_returned",
    "raw_provider_bodies_returned",
    "raw_payloads_included",
    "raw_prompts_included",
    "raw_responses_included",
    "raw_provider_bodies_included",
    "raw_commands_included",
    "raw_paths_included",
    "raw_request_ids_included",
    "raw_terminal_text_included",
    "raw_body_storage",
    "raw_prompts_included",
    "raw_responses_included",
    "request_ids_returned",
    "tenant_ids_returned",
    "cache_keys_returned",
    "file_paths_returned",
}


def is_openai_optimization_review_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("openai_optimization"), dict):
        return True
    if payload.get("schema") == "tokenclaw.policy_bundle_fetch_review.v1":
        bundle = payload.get("bundle")
        return isinstance(bundle, dict) and isinstance(bundle.get("openai_optimization"), dict)
    return False


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _stable_id(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _expired(value: Any) -> bool:
    parsed = _parse_datetime(value)
    return parsed is not None and parsed <= datetime.now(timezone.utc)


def _family(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"old_context_summary", "old_context_summarization", "summary", "summarization"}:
        return "old_context_summarization"
    return text


def _target_candidate_id(action: dict[str, Any], index: int) -> str:
    for key in ("target_candidate_id", "candidate_id", "policy_id", "recommendation_id", "action_id"):
        value = action.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return f"openai-review-action-{index + 1}"


def _action_summary(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
    surface = action.get("local_policy_surface") if isinstance(action.get("local_policy_surface"), dict) else {}
    return {
        key: value
        for key, value in {
            "action_id": action.get("action_id"),
            "target_candidate_id": action.get("target_candidate_id") or action.get("candidate_id"),
            "action_family": action.get("action_family"),
            "candidate_family": action.get("candidate_family"),
            "policy_section": action.get("policy_section"),
            "decision": action.get("decision"),
            "conflict_key": action.get("conflict_key"),
            "reason_codes": action.get("reason_codes") if isinstance(action.get("reason_codes"), list) else [],
            "suppressed_by": action.get("suppressed_by") if isinstance(action.get("suppressed_by"), dict) else None,
            "compatible": compatibility.get("compatible"),
            "compatibility_reason_codes": compatibility.get("reason_codes") if isinstance(compatibility.get("reason_codes"), list) else [],
            "policy_file": surface.get("policy_file"),
            "expected_impact": action.get("expected_impact") if isinstance(action.get("expected_impact"), dict) else {},
        }.items()
        if value not in (None, "", [], {})
    }


def _metadata(bundle: dict[str, Any], selected: list[dict[str, Any]], suppressed: list[Any], omitted: list[Any]) -> dict[str, Any]:
    openai_review = bundle.get("openai_optimization") if isinstance(bundle.get("openai_optimization"), dict) else {}
    recommendation = bundle.get("recommendation") if isinstance(bundle.get("recommendation"), dict) else {}
    counts: dict[str, dict[str, int]] = {}
    for decision, actions in (("selected", selected), ("suppressed", suppressed), ("omitted", omitted)):
        for action in actions:
            if not isinstance(action, dict):
                continue
            family = _family(action.get("action_family") or "unknown")
            row = counts.setdefault(family, {"selected": 0, "suppressed": 0, "omitted": 0})
            row[decision] += 1
    return {
        "schema": OPENAI_OPTIMIZATION_DRAFT_METADATA_SCHEMA,
        "created_at": utc_now(),
        "source": "openai_optimization_review_bundle",
        "review_bundle_schema": openai_review.get("schema") or recommendation.get("openai_optimization_schema"),
        "selected_action_count": len(selected),
        "suppressed_action_count": len([action for action in suppressed if isinstance(action, dict)]),
        "omitted_action_count": len([action for action in omitted if isinstance(action, dict)]),
        "staged_action_count": len(selected),
        "staged_policy_sections": sorted({_section_for_action(action) for action in selected}),
        "counts_by_family": dict(sorted(counts.items())),
        "conflict_summary": recommendation.get("conflict_summary") if isinstance(recommendation.get("conflict_summary"), dict) else {},
        "filters": openai_review.get("filters") if isinstance(openai_review.get("filters"), dict) else {},
        "thresholds": openai_review.get("thresholds") if isinstance(openai_review.get("thresholds"), dict) else {},
        "selected_actions": [_action_summary(action) for action in selected],
        "suppressed_actions": [_action_summary(action) for action in suppressed],
        "omitted_actions": [_action_summary(action) for action in omitted],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }


def _scan_safety(value: Any, errors: list[dict[str, str]], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if lowered in UNSAFE_TRUE_FLAGS and bool(child):
                errors.append({"path": child_path, "message": "OpenAI optimization review draft requires local-only metadata and no provider forwarding/raw payload flags"})
                continue
            if lowered in SAFE_RAW_FLAG_KEYS:
                _scan_safety(child, errors, child_path)
                continue
            if lowered in RAW_EXACT_KEYS or lowered.startswith("raw_"):
                errors.append({"path": child_path, "message": "raw or local-identifier fields are not accepted in OpenAI optimization review drafts"})
                continue
            if lowered in {"managed_enforced", "managed-enforced"} and bool(child):
                errors.append({"path": child_path, "message": "managed-enforced actions are not accepted for local review draft staging"})
                continue
            if lowered == "policy_source" and child == "managed-enforced":
                errors.append({"path": child_path, "message": "managed-enforced actions are not accepted for local review draft staging"})
                continue
            if lowered == "locally_executed" and child is False:
                errors.append({"path": child_path, "message": "OpenAI optimization review actions must be locally executed"})
                continue
            if lowered == "feature_only" and child is False:
                errors.append({"path": child_path, "message": "OpenAI optimization review actions must be feature-only"})
                continue
            if lowered in {"expires_at", "expiry"} and _expired(child):
                errors.append({"path": child_path, "message": "OpenAI optimization review bundle or action is expired"})
                continue
            if lowered == "expired" and child is True:
                errors.append({"path": child_path, "message": "OpenAI optimization review bundle or action is expired"})
                continue
            _scan_safety(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_safety(child, errors, f"{path}[{index}]")


def _section_for_action(action: dict[str, Any]) -> str:
    family = _family(action.get("action_family"))
    section = str(action.get("policy_section") or "").strip()
    if family == "routing" or section == "routing":
        return "routing"
    if family == "cache" or section == "cache":
        return "cache"
    return "crunch"


def _validate_review_bundle(bundle: Any) -> dict[str, Any]:
    from tokenclaw.policy_bundle import validate_policy_bundle

    errors: list[dict[str, str]] = []
    policy_validation = validate_policy_bundle(bundle)
    if not policy_validation.get("ok"):
        for error in policy_validation.get("errors", []):
            if isinstance(error, dict):
                errors.append({"path": str(error.get("path") or "$"), "message": str(error.get("message") or "policy bundle is invalid")})
    if not isinstance(bundle, dict):
        errors.append({"path": "$", "message": "expected OpenAI optimization review policy bundle object"})
        return {
            "schema": OPENAI_OPTIMIZATION_DRAFT_VALIDATION_SCHEMA,
            "ok": False,
            "policy_validation": policy_validation,
            "errors": errors,
        }
    openai_review = bundle.get("openai_optimization")
    if not isinstance(openai_review, dict):
        errors.append({"path": "$.openai_optimization", "message": "expected OpenAI optimization review bundle"})
        openai_review = {}
    selected = openai_review.get("selected_actions") if isinstance(openai_review.get("selected_actions"), list) else []
    if not selected:
        errors.append({"path": "$.openai_optimization.selected_actions", "message": "expected at least one selected OpenAI optimization review action"})

    top_compat = bundle.get("local_executor_compatibility")
    if not isinstance(top_compat, dict) or top_compat.get("compatible") is not True:
        errors.append({"path": "$.local_executor_compatibility", "message": "missing compatible local executor compatibility metadata"})

    _scan_safety(bundle, errors)
    for index, action in enumerate(selected):
        path = f"$.openai_optimization.selected_actions[{index}]"
        if not isinstance(action, dict):
            errors.append({"path": path, "message": "expected selected action object"})
            continue
        family = _family(action.get("action_family"))
        if family not in SUPPORTED_ACTION_FAMILIES:
            errors.append({"path": f"{path}.action_family", "message": f"unsupported OpenAI optimization action family: {action.get('action_family')}"})
        if action.get("decision") not in (None, "selected"):
            errors.append({"path": f"{path}.decision", "message": "only selected OpenAI optimization actions may be staged"})
        compat = action.get("local_executor_compatibility")
        if not isinstance(compat, dict) or compat.get("compatible") is not True:
            errors.append({"path": f"{path}.local_executor_compatibility", "message": "selected action is missing compatible local executor metadata"})
        else:
            supported = compat.get("supported_local_action_families")
            if isinstance(supported, list) and family not in {_family(item) for item in supported}:
                errors.append({"path": f"{path}.local_executor_compatibility.supported_local_action_families", "message": f"local executor does not support action family: {family}"})

    return {
        "schema": OPENAI_OPTIMIZATION_DRAFT_VALIDATION_SCHEMA,
        "ok": not errors,
        "policy_validation": policy_validation,
        "errors": errors,
    }


def _managed_meta(action: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "schema": "tokenclaw.openai_optimization_review_action_local_metadata.v1",
            "action_id": action.get("action_id"),
            "target_candidate_id": _target_candidate_id(action, index),
            "action_family": action.get("action_family"),
            "candidate_family": action.get("candidate_family"),
            "policy_source": "managed-recommended",
            "decision": action.get("decision") or "selected",
            "conflict_key": action.get("conflict_key"),
            "reason_codes": action.get("reason_codes") if isinstance(action.get("reason_codes"), list) else [],
            "expected_impact": action.get("expected_impact") if isinstance(action.get("expected_impact"), dict) else {},
            "evidence_freshness": action.get("evidence_freshness") if isinstance(action.get("evidence_freshness"), dict) else {},
        }.items()
        if value not in (None, "", [], {})
    }


def _coerce_fraction(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(1.0, max(0.0, number)), 6)


def _routing_canary(action: dict[str, Any], index: int) -> dict[str, Any]:
    update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    canary = (
        update.get("openai_canary")
        if isinstance(update.get("openai_canary"), dict)
        else update.get("routing")
        if isinstance(update.get("routing"), dict)
        else {}
    )
    action_canary = action.get("canary") if isinstance(action.get("canary"), dict) else {}
    target_candidate = _target_candidate_id(action, index)
    result = copy.deepcopy(canary)
    result.update({
        "enabled": False,
        "policy_id": result.get("policy_id") or action.get("target_rule_id") or _stable_id("managed-openai-routing", target_candidate),
        "target_candidate_id": target_candidate,
        "policy_source": "managed-recommended",
        "review_only": False,
        "managed_enforced": False,
        "required_local_review": True,
        "canary_fraction": 0.0,
        "holdout_fraction": 0.0,
        "salt": result.get("salt") or action_canary.get("salt") or target_candidate,
        "managed_recommendation": _managed_meta(action, index),
    })
    for source_key, target_key in (
        ("requested_model_family", "model_pattern"),
        ("model_pattern", "model_pattern"),
        ("candidate_target_model", "target_model"),
        ("recommended_target_model", "target_model"),
        ("target_model", "target_model"),
    ):
        value = update.get(source_key, action.get(source_key))
        if value not in (None, "") and target_key not in result:
            result[target_key] = value
    for key in ("eligible_categories", "excluded_categories", "allow_tools", "allow_stream", "min_text_chars", "max_text_chars", "min_input_tokens_est", "max_input_tokens_est"):
        if key in update and key not in result:
            result[key] = update[key]
        elif key in action and key not in result:
            result[key] = action[key]
    safety = result.get("safety_stop") if isinstance(result.get("safety_stop"), dict) else {}
    result["safety_stop"] = {
        "enabled": bool(safety.get("enabled", True)),
        "window_hours": int(safety.get("window_hours", 24)),
        "min_samples": int(safety.get("min_samples", 20)),
        "min_holdout_samples": int(safety.get("min_holdout_samples", 10)),
        "max_error_rate": _coerce_fraction(safety.get("max_error_rate"), 0.03),
        "max_retry_rate": _coerce_fraction(safety.get("max_retry_rate"), 0.10),
        "max_fallback_rate": _coerce_fraction(safety.get("max_fallback_rate"), 0.10),
        "max_latency_regression_ratio": float(safety.get("max_latency_regression_ratio", 1.5)),
        "limit": int(safety.get("limit", 1000)),
    }
    return result


def _summary_rule(action: dict[str, Any], index: int) -> dict[str, Any]:
    update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    summary = update.get("old_context_summarization") if isinstance(update.get("old_context_summarization"), dict) else {}
    result = copy.deepcopy(summary)
    target_candidate = _target_candidate_id(action, index)
    result.update({
        "enabled": False,
        "rule_id": result.get("rule_id") or action.get("target_rule_id") or _stable_id("managed-openai-summary", target_candidate),
        "candidate_id": target_candidate,
        "policy_source": "managed-recommended",
        "review_only": False,
        "managed_enforced": False,
        "required_local_review": True,
        "placement": result.get("placement") or "system",
        "managed_recommendation": _managed_meta(action, index),
    })
    for source_key, target_key in (
        ("summary_model", "model"),
        ("model_hint", "model"),
        ("candidate_target_model", "model"),
        ("model", "model"),
    ):
        value = update.get(source_key, action.get(source_key))
        if value not in (None, "") and target_key not in result:
            result[target_key] = value
    canary = result.get("canary") if isinstance(result.get("canary"), dict) else {}
    result["canary"] = {
        **canary,
        "enabled": False,
        "fraction": 0.0,
        "holdout_fraction": 0.0,
        "salt": canary.get("salt") or target_candidate,
        "unit": canary.get("unit") or "source_hash",
    }
    safety = result.get("safety_stop") if isinstance(result.get("safety_stop"), dict) else {}
    if safety:
        result["safety_stop"] = safety
    return result


def _cache_pattern_rule(action: dict[str, Any], index: int) -> dict[str, Any]:
    update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    target_candidate = _target_candidate_id(action, index)
    conditions = copy.deepcopy(update.get("conditions") if isinstance(update.get("conditions"), dict) else {})
    if not conditions.get("pattern_hash") and not conditions.get("pattern_hashes"):
        conditions["pattern_hash"] = f"managed:{target_candidate}"
    rule_action = copy.deepcopy(update.get("action") if isinstance(update.get("action"), dict) else {})
    rule_action.setdefault("type", "exact_cache_pattern")
    safe_invalidation = bool(rule_action.get("safe_invalidation") or rule_action.get("safe_invalidation_evidence"))
    rule_action["allow_tool_calls"] = bool(rule_action.get("allow_tool_calls") and safe_invalidation)
    rule_action.setdefault("safe_invalidation", safe_invalidation)
    rule_action.setdefault("safe_invalidation_evidence", safe_invalidation)
    rule_action.setdefault("streaming", bool(conditions.get("stream")))
    rule_action.setdefault("reason", "staged managed OpenAI cache replay recommendation")
    rule = {
        "id": action.get("target_rule_id") or _stable_id("managed-openai-cache", target_candidate),
        "candidate_id": target_candidate,
        "enabled": False,
        "policy_source": "managed-recommended",
        "review_only": False,
        "managed_enforced": False,
        "required_local_review": True,
        "conditions": conditions,
        "action": rule_action,
        "managed_recommendation": _managed_meta(action, index),
        "canary": {
            "enabled": False,
            "fraction": 0.0,
            "holdout_fraction": 0.0,
            "salt": target_candidate,
            "unit": "source_hash",
        },
    }
    return rule


def _apply_selected_action(proposed: dict[str, Any], action: dict[str, Any], index: int) -> None:
    policies = proposed.setdefault("policies", {})
    section = _section_for_action(action)
    if section == "routing":
        routing = policies.setdefault("routing", {})
        routing["policy_source"] = "managed-recommended"
        routing.setdefault("openai", {})
        if not isinstance(routing["openai"], dict):
            routing["openai"] = {}
        routing["openai"]["canary"] = _routing_canary(action, index)
        return
    if section == "crunch":
        crunch = policies.setdefault("crunch", {})
        crunch["policy_source"] = "managed-recommended"
        crunch["old_context_summarization"] = _summary_rule(action, index)
        return
    cache = policies.setdefault("cache", {})
    cache["policy_source"] = "managed-recommended"
    rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), list) else []
    cache["pattern_rules"] = [*rules, _cache_pattern_rule(action, index)]


def _attach_local_draft_provenance_if_configured(bundle: dict[str, Any]) -> dict[str, Any]:
    from tokenclaw.policy_bundle import _verification_secrets, attach_policy_bundle_provenance

    secrets = _verification_secrets()
    if not secrets:
        return bundle
    key_id = next((key for key in secrets if key), "local-draft")
    secret = secrets.get(key_id) or secrets.get("")
    if not secret:
        return bundle
    return attach_policy_bundle_provenance(
        bundle,
        secret=secret,
        issuer="tokenclaw-proxy",
        server_id="local-draft-stager",
        key_id=key_id,
    )


def _error_result(
    *,
    error_type: str,
    message: str,
    draft_id: str | None,
    workspace: str | Path | None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": POLICY_DRAFT_STAGE_SCHEMA,
        "ok": False,
        "draft": None,
        "draft_id": draft_id,
        "workspace": str(Path(workspace).expanduser()) if workspace is not None else None,
        "wrote_active_policy_files": False,
        "reloaded_modules": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "diff": None,
        "sections": [],
        "openai_optimization_review_draft": None,
        "validation": validation,
        "error": {
            "type": error_type,
            "message": message,
            "errors": validation.get("errors", []) if isinstance(validation, dict) else [],
        },
    }


async def stage_openai_optimization_review_draft(
    payload: Any,
    *,
    draft_id: str | None = None,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    from tokenclaw.policy_bundle import build_policy_bundle

    bundle = payload.get("bundle") if isinstance(payload, dict) and payload.get("schema") == "tokenclaw.policy_bundle_fetch_review.v1" else payload
    validation = _validate_review_bundle(bundle)
    if not validation["ok"]:
        return _error_result(
            error_type="openai_optimization_review_rejected",
            message="OpenAI optimization review bundle is not safe to stage as a local draft",
            draft_id=draft_id,
            workspace=workspace,
            validation=validation,
        )
    assert isinstance(bundle, dict)
    openai_review = bundle["openai_optimization"]
    selected = [action for action in openai_review.get("selected_actions", []) if isinstance(action, dict)]
    suppressed = openai_review.get("suppressed_actions") if isinstance(openai_review.get("suppressed_actions"), list) else []
    omitted = openai_review.get("omitted_actions") if isinstance(openai_review.get("omitted_actions"), list) else []
    current = await build_policy_bundle()
    proposed = _json_clone(current)
    proposed["generated_at"] = utc_now()
    proposed["managed_optimizer"] = {
        "enabled": False,
        "policy_source": "managed-recommended",
        "recommendation_mode": "local-draft-openai-optimization-review",
        "note": "Selected managed OpenAI optimization actions were staged as inactive local policy drafts.",
    }
    proposed["recommendation"] = {
        "schema": "tokenclaw.policy_bundle_recommendation.v1",
        "policy_source": "managed-recommended",
        "recommendation_mode": "local-draft-openai-optimization-review",
        "required_local_review": True,
        "selected_action_count": len(selected),
        "staged_action_count": len(selected),
    }
    metadata = _metadata(bundle, selected, suppressed, omitted)
    for index, action in enumerate(selected):
        _apply_selected_action(proposed, action, index)
    proposed = _attach_local_draft_provenance_if_configured(proposed)
    result = await stage_policy_draft(proposed, draft_id=draft_id, workspace=workspace, metadata={"openai_optimization_review": metadata})
    result["openai_optimization_review_draft"] = metadata
    return result
