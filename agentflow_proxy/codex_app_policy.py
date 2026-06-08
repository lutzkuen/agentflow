from __future__ import annotations

import os
from typing import Any


DEFAULT_CODEX_APP_UPSTREAM = "ws://127.0.0.1:4014"
CODEX_TURN_SOURCE_SURFACE = "codex_turn"
LEGACY_CODEX_APP_SOURCE_SURFACE = "codex_app_turn"
CODEX_APP_SOURCE_SURFACE = CODEX_TURN_SOURCE_SURFACE
CODEX_APP_SOURCE_SURFACE_ALIASES = frozenset({
    CODEX_TURN_SOURCE_SURFACE,
    LEGACY_CODEX_APP_SOURCE_SURFACE,
})
CODEX_APP_POLICY_CONDITION_KEYS = (
    "app_family",
    "workflow_phase",
    "model_field_state",
    "input_size_bucket",
    "cache_eligible",
    "cache_status",
    "replayability_level",
    "has_action_like_params",
)
CODEX_APP_POLICY_ACTION_KEYS = (
    "recommended_model",
    "model_hint",
    "crunch_profile",
    "cache_eligible",
    "cache_eligibility_reason",
    "pass_through_reason",
    "reason",
)

CODEX_ACTION_KEY_HINTS = {
    "approval",
    "approvalrequest",
    "approval_request",
    "apply_patch",
    "cmd",
    "command",
    "exec",
    "function_call",
    "patch",
    "shell",
    "tool_call",
    "tool_calls",
}
CODEX_ACTION_VALUE_HINTS = {
    "approval_request",
    "apply_patch",
    "command",
    "exec",
    "function_call",
    "shell",
    "tool_call",
    "tool_result",
    "tool_use",
}
CODEX_MODEL_FIELDS = ("model", "modelId", "model_id")
CODEX_MODEL_STATE_FIELDS = (
    "model",
    "modelId",
    "model_id",
    "activeModel",
    "active_model",
    "defaultModel",
    "default_model",
    "modelName",
    "model_name",
)
CODEX_MODEL_STATE_SKIP_KEYS = {
    "cmd",
    "command",
    "content",
    "input",
    "input_text",
    "inputtext",
    "instructions",
    "message",
    "messages",
    "patch",
    "prompt",
    "raw_request",
    "raw_response",
    "result",
    "response",
    "text",
    "tool_call",
    "tool_calls",
    "tool_result",
    "tool_results",
}
CODEX_SAFE_TURN_PARAM_KEYS = {
    "input",
    "instructions",
    "max_tokens",
    "maxTokens",
    "model",
    "modelId",
    "model_id",
    "temperature",
    "threadId",
    "thread_id",
    "top_p",
    "topP",
}
CODEX_TEXT_INPUT_TYPES = {"text", "input_text"}


def canonical_source_surface(value: Any) -> str:
    surface = str(value or "").strip()
    if surface in CODEX_APP_SOURCE_SURFACE_ALIASES:
        return CODEX_TURN_SOURCE_SURFACE
    return surface or "unknown"


def is_codex_turn_source_surface(value: Any) -> bool:
    return canonical_source_surface(value) == CODEX_TURN_SOURCE_SURFACE


def _normalized_model_field(value: Any) -> str:
    return str(value or "").replace("-", "_").lower()


def _normalized_model_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 100:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")
    if any(char not in allowed for char in cleaned):
        return None
    return cleaned


def codex_model_state_signal(method: Any, params: Any) -> dict[str, Any] | None:
    if not isinstance(params, dict):
        return None
    model_fields = {_normalized_model_field(field) for field in CODEX_MODEL_STATE_FIELDS}
    stack: list[Any] = [params]
    explicit_absent: dict[str, Any] | None = None
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_s = str(key)
                normalized_key = _normalized_model_field(key_s)
                if normalized_key in model_fields:
                    normalized = _normalized_model_value(value)
                    if normalized:
                        return {
                            "state": "derived_present",
                            "field": key_s,
                            "normalized_model": normalized,
                            "source_method": str(method or "unknown"),
                            "confidence": "high",
                            "reason": "metadata-model-field",
                        }
                    if value is None or value == "":
                        explicit_absent = {
                            "state": "derived_absent",
                            "field": key_s,
                            "normalized_model": None,
                            "source_method": str(method or "unknown"),
                            "confidence": "high",
                            "reason": "metadata-model-field-empty",
                        }
                elif normalized_key not in CODEX_MODEL_STATE_SKIP_KEYS and isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))
    return explicit_absent


def _env_bool(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() not in {"0", "false", "no", "off", ""}


def codex_app_optimize_enabled() -> bool:
    return _env_bool("AGENTFLOW_CODEX_APP_OPTIMIZE", True)


def codex_app_cache_enabled() -> bool:
    return _env_bool("AGENTFLOW_CODEX_APP_CACHE", False)


def codex_app_cache_namespace() -> str:
    return os.getenv("AGENTFLOW_CODEX_APP_CACHE_NAMESPACE", os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default"))


def codex_app_upstream() -> str:
    return os.getenv("AGENTFLOW_CODEX_APP_UPSTREAM", DEFAULT_CODEX_APP_UPSTREAM)


def codex_app_surface_policy_state(provider_policy_state: dict[str, Any]) -> dict[str, Any]:
    inherited_sections = {}
    reload_required_sections: list[str] = []
    for section in ("routing", "crunch", "cache"):
        policy = provider_policy_state.get(section)
        if not isinstance(policy, dict):
            continue
        file_status = policy.get("file") if isinstance(policy.get("file"), dict) else {}
        reload_required = bool(file_status.get("reload_required"))
        if reload_required:
            reload_required_sections.append(section)
        inherited_sections[section] = {
            "policy_source": policy.get("policy_source"),
            "rule_path": policy.get("rule_path"),
            "reload_required": reload_required,
            "file": file_status,
        }

    optimize_enabled = codex_app_optimize_enabled()
    cache_enabled = codex_app_cache_enabled()
    upstream = codex_app_upstream()
    namespace = codex_app_cache_namespace()
    return {
        "surface": CODEX_APP_SOURCE_SURFACE,
        "name": "Codex app-server",
        "enabled": optimize_enabled,
        "policy_source": "local-default",
        "runtime_flags": {
            "optimization_enabled": optimize_enabled,
            "cache_enabled": cache_enabled,
        },
        "optimization": {
            "enabled": optimize_enabled,
            "disabled_reason": None if optimize_enabled else "AGENTFLOW_CODEX_APP_OPTIMIZE=0",
            "scope": "metadata-only local JSON-RPC turn optimization",
        },
        "routing": inherited_sections.get("routing", {}),
        "crunch": inherited_sections.get("crunch", {}),
        "cache": {
            **inherited_sections.get("cache", {}),
            "enabled": cache_enabled,
            "exact_cache": {
                "enabled": cache_enabled,
                "namespace": namespace,
                "provider": "codex-app",
                "upstream": upstream,
                "request_basis": "jsonrpc turn/start frame with request id removed",
                "cache_url": "codex-app://turn/start",
                "replayability_level": "local-exact-response",
            },
            "disabled_reason": None if cache_enabled else "AGENTFLOW_CODEX_APP_CACHE is not 1",
        },
        "safe_turn_params": {
            "allowed_keys": sorted(CODEX_SAFE_TURN_PARAM_KEYS),
            "allowed_key_count": len(CODEX_SAFE_TURN_PARAM_KEYS),
            "model_fields": list(CODEX_MODEL_FIELDS),
            "text_input_types": sorted(CODEX_TEXT_INPUT_TYPES),
            "unknown_key_behavior": "skip-cache-and-keep-features-only",
        },
        "action_like_skip_behavior": {
            "enabled": True,
            "reason": "action-like-params",
            "applies_before": ["routing", "crunch", "cache"],
            "key_hints": sorted(CODEX_ACTION_KEY_HINTS),
            "value_hints": sorted(CODEX_ACTION_VALUE_HINTS),
        },
        "file_backed_policy_sections": inherited_sections,
        "reload_required": bool(reload_required_sections),
        "reload_required_sections": reload_required_sections,
        "managed_optimizer_required": False,
        "note": "Codex app-server runtime flags are local process settings; managed optimizer use remains opt-in.",
    }


def codex_app_bundle_policy_state() -> dict[str, Any]:
    optimize_enabled = codex_app_optimize_enabled()
    cache_enabled = codex_app_cache_enabled()
    return {
        "enabled": optimize_enabled,
        "policy_source": "local-default",
        "surface": CODEX_APP_SOURCE_SURFACE,
        "review_only": True,
        "runtime_flags": {
            "optimization_enabled": optimize_enabled,
            "cache_enabled": cache_enabled,
        },
        "rules": [],
        "supported_conditions": list(CODEX_APP_POLICY_CONDITION_KEYS),
        "supported_actions": list(CODEX_APP_POLICY_ACTION_KEYS),
        "application": {
            "status": "not-applied",
            "reason": "Codex app turn-level policies are reviewable metadata only until the Codex app proxy explicitly implements an action.",
        },
        "managed_optimizer_required": False,
    }
