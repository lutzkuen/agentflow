from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import httpx


RECOMMENDATION_PATH = "/v1/recommendation"
OUTCOME_PATH_TEMPLATE = "/v1/optimization-units/{unit_id}/outcome"

RAW_FEATURE_KEYS = {
    "body",
    "completion",
    "content",
    "developer",
    "input",
    "message",
    "messages",
    "output",
    "prompt",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "response",
    "system",
    "system_prompt",
}


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def recommendations_enabled() -> bool:
    return _as_bool(os.getenv("AGENTFLOW_RECOMMENDATION_ENABLED"), False)


def recommendation_server_url() -> str:
    return os.getenv("AGENTFLOW_RECOMMENDATION_SERVER_URL", "http://127.0.0.1:4100").rstrip("/")


def recommendation_timeout_seconds() -> float:
    return float(os.getenv("AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS", "1.5"))


def _sanitize_features(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_features(item)
            for key, item in value.items()
            if str(key).lower() not in RAW_FEATURE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_features(item) for item in value]
    return value


def _source_surface(provider: str, path: str) -> str:
    provider_l = (provider or "").lower()
    path_l = (path or "").lower()
    if provider_l == "anthropic":
        return "anthropic_messages"
    if provider_l == "openai":
        if "chat/completions" in path_l:
            return "openai_chat"
        return "openai_responses"
    return "anthropic_messages"


def _app_family(provider: str, requested_model: str, path: str) -> str:
    provider_l = (provider or "").lower()
    model_l = (requested_model or "").lower()
    if provider_l == "anthropic" and "messages" in (path or "").lower():
        return "claude_code"
    if provider_l == "openai" and "codex" in model_l:
        return "codex"
    if provider_l == "openai":
        return "generic_openai"
    return "unknown"


def build_optimization_unit(
    *,
    provider: str,
    path: str,
    requested_model: str,
    routed_model: str,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    category: str | None,
    stream: bool,
    input_tokens_est: int | None,
) -> dict[str, Any]:
    text_chars = routing_meta.get("text_chars")
    has_tools = routing_meta.get("has_tools")
    unit = {
        "source_surface": _source_surface(provider, path),
        "granularity": "provider_request",
        "app_family": _app_family(provider, requested_model, path),
        "requested_model": requested_model,
        "input_features": {
            "path": path,
            "stream": bool(stream),
            "category": category or routing_meta.get("category"),
            "text_chars": text_chars,
            "input_tokens_est": input_tokens_est,
            "local_routed_model": routed_model,
            "local_routing_reason": routing_meta.get("reason"),
            "local_routing_policy_source": routing_meta.get("policy_source"),
            "crunch_changed": bool(crunch_meta.get("changed")),
            "crunch_saved_chars": crunch_meta.get("saved_chars"),
            "cache_status": cache_meta.get("status"),
            "cache_reason": cache_meta.get("reason"),
        },
        "tool_features": {
            "has_tools": has_tools,
            "category": category or routing_meta.get("category"),
            "thinking_history_stripped": routing_meta.get("thinking_history_stripped"),
            "stripped_params": routing_meta.get("stripped_params") or [],
        },
        "outcome_features": {},
        "replayability_level": "features_only",
    }
    return _sanitize_features(unit)


def _base_meta() -> dict[str, Any]:
    return {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": RECOMMENDATION_PATH,
        "policy_source": "local-default",
    }


def disabled_recommendation_meta() -> dict[str, Any]:
    meta = _base_meta()
    meta.update({
        "status": "skipped",
        "reason": "disabled",
        "fallback": "local-policy",
        "applied": False,
    })
    return meta


def _normalize_recommendation(body: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(body, dict):
        return None, "response was not a JSON object"
    target_model = body.get("target_model")
    confidence = body.get("confidence")
    policy_id = body.get("policy_id")
    reason = body.get("reason")
    if not isinstance(target_model, str) or not target_model:
        return None, "missing target_model"
    if not isinstance(confidence, (int, float)):
        return None, "missing confidence"
    if not isinstance(policy_id, str) or not policy_id:
        return None, "missing policy_id"
    if not isinstance(reason, str) or not reason:
        return None, "missing reason"

    replacement_prompt = body.get("replacement_prompt")
    normalized = {
        "target_model": target_model,
        "confidence": float(confidence),
        "policy_id": policy_id,
        "reason": reason,
        "replacement_prompt_present": isinstance(replacement_prompt, str) and bool(replacement_prompt),
    }
    optimization_unit_id = body.get("optimization_unit_id")
    if isinstance(optimization_unit_id, int):
        normalized["optimization_unit_id"] = optimization_unit_id
    recommendation_id = body.get("recommendation_id")
    if isinstance(recommendation_id, (int, str)):
        normalized["recommendation_id"] = recommendation_id
    if normalized["replacement_prompt_present"]:
        normalized["replacement_prompt_sha256"] = hashlib.sha256(replacement_prompt.encode("utf-8")).hexdigest()
    return normalized, None


async def fetch_recommendation(unit: dict[str, Any]) -> dict[str, Any]:
    if not recommendations_enabled():
        return disabled_recommendation_meta()

    meta = _base_meta()
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(recommendation_server_url() + RECOMMENDATION_PATH, json=unit)
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        if response.status_code >= 400:
            meta.update({
                "status": "error",
                "reason": "server-error",
                "error": response.text[:500],
                "fallback": "local-policy",
                "applied": False,
            })
            return meta
        try:
            body = response.json()
        except Exception as exc:
            meta.update({
                "status": "invalid",
                "reason": "invalid-json",
                "error": repr(exc),
                "fallback": "local-policy",
                "applied": False,
            })
            return meta
        recommendation, error = _normalize_recommendation(body)
        if recommendation is None:
            meta.update({
                "status": "invalid",
                "reason": error or "invalid-response",
                "fallback": "local-policy",
                "applied": False,
            })
            return meta
        meta.update(recommendation)
        meta.update({
            "status": "received",
            "policy_source": "managed-recommended",
        })
        return meta
    except Exception as exc:
        meta.update({
            "latency_ms": int((time.time() - started) * 1000),
            "status": "error",
            "reason": "request-failed",
            "error": repr(exc),
            "fallback": "local-policy",
            "applied": False,
        })
        return meta


def _provider_compatible(provider: str, target_model: str) -> bool:
    target_l = target_model.lower()
    if provider == "anthropic":
        return target_l.startswith("claude-")
    if provider == "openai":
        return not target_l.startswith("claude-")
    return False


def apply_recommendation_to_body(
    *,
    provider: str,
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    recommendation_meta: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(recommendation_meta)
    if meta.get("status") != "received":
        meta.setdefault("applied", False)
        return meta

    target_model = str(meta.get("target_model") or "")
    current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
    meta["local_model_before_recommendation"] = current_model
    if not target_model:
        meta.update({"applied": False, "apply_reason": "missing-target-model", "fallback": "local-policy"})
        return meta
    if not _provider_compatible(provider, target_model):
        meta.update({"applied": False, "apply_reason": "provider-mismatch", "fallback": "local-policy"})
        return meta
    if target_model != current_model and "thinking request" in str(routing_meta.get("reason") or ""):
        meta.update({"applied": False, "apply_reason": "local-thinking-safety-guard", "fallback": "local-policy"})
        return meta

    body["model"] = target_model
    routing_meta["routed_model"] = target_model
    routing_meta["final_policy_source"] = "managed-recommended"
    routing_meta["managed_policy_id"] = meta.get("policy_id")
    routing_meta["managed_reason"] = meta.get("reason")
    meta["applied"] = True
    meta["changed_model"] = target_model != current_model
    if meta.get("replacement_prompt_present"):
        meta["replacement_prompt_applied"] = False
        meta["replacement_prompt_apply_reason"] = "not-supported-by-local-bridge"
    return meta


def _compact_error(error: str | None, status_code: int | None) -> dict[str, Any]:
    if not error:
        return {}
    error_text = str(error)
    error_class = "upstream_error" if status_code and status_code >= 400 else "proxy_error"
    try:
        body = json.loads(error_text)
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and isinstance(err.get("type"), str):
                error_class = err["type"]
            elif isinstance(body.get("type"), str):
                error_class = body["type"]
    except Exception:
        head = error_text.split(":", 1)[0].strip()
        if head and len(head) <= 80:
            error_class = head
    return {
        "error_class": error_class,
        "error_message_prefix": error_text[:200],
    }


def build_outcome_feedback(
    *,
    provider: str,
    path: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    input_tokens_est: int | None,
    output_tokens_est: int | None,
    actual_input_tokens: int | None,
    actual_output_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    thinking_output_tokens: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    cache_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    category: str | None,
    session_id: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    managed = routing_meta.get("managed_recommendation") if isinstance(routing_meta, dict) else None
    if not isinstance(managed, dict):
        managed = {}
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16] if session_id else None
    features: dict[str, Any] = {
        "provider": provider,
        "source_surface": _source_surface(provider, path),
        "path": path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "retry_count": retry_count or 0,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "input_tokens": actual_input_tokens if actual_input_tokens is not None else input_tokens_est,
        "output_tokens": actual_output_tokens if actual_output_tokens is not None else output_tokens_est,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "actual_input_tokens": actual_input_tokens,
        "actual_output_tokens": actual_output_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens or 0,
        "cache_read_input_tokens": cache_read_input_tokens or 0,
        "thinking_output_tokens": thinking_output_tokens,
        "cost_est_usd": cost_est_usd,
        "cost_baseline_usd": cost_baseline_usd,
        "cache_decision": {
            key: cache_meta.get(key)
            for key in ("status", "reason", "hit_type", "invalidated", "policy_source")
            if cache_meta.get(key) is not None
        },
        "routing_decision": {
            "reason": routing_meta.get("reason"),
            "policy_source": routing_meta.get("policy_source"),
            "final_policy_source": routing_meta.get("final_policy_source"),
            "managed_policy_id": routing_meta.get("managed_policy_id"),
            "managed_reason": routing_meta.get("managed_reason"),
            "fallback_reason": routing_meta.get("fallback_reason"),
        },
        "crunch_saved_chars": crunch_meta.get("saved_chars"),
        "crunch_tokens_saved_est": crunch_meta.get("tokens_saved_est"),
        "category": category or routing_meta.get("category"),
        "session": {
            "present": bool(session_id),
            "id_hash": session_hash,
        },
        "managed_recommendation": {
            "optimization_unit_id": managed.get("optimization_unit_id"),
            "recommendation_id": managed.get("recommendation_id"),
            "policy_id": managed.get("policy_id"),
            "target_model": managed.get("target_model"),
            "applied": bool(managed.get("applied")),
            "changed_model": bool(managed.get("changed_model")),
            "apply_reason": managed.get("apply_reason"),
            "target_model_normalized": managed.get("target_model_normalized"),
        },
    }
    features.update(_compact_error(error, status_code))
    return _sanitize_features(features)


def disabled_outcome_feedback_meta() -> dict[str, Any]:
    return {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": OUTCOME_PATH_TEMPLATE,
        "status": "skipped",
        "reason": "disabled",
    }


async def send_outcome_feedback(recommendation_meta: dict[str, Any], outcome_features: dict[str, Any]) -> dict[str, Any]:
    if not recommendations_enabled():
        return disabled_outcome_feedback_meta()

    unit_id = recommendation_meta.get("optimization_unit_id")
    if not isinstance(unit_id, int):
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "endpoint": OUTCOME_PATH_TEMPLATE,
            "status": "skipped",
            "reason": "missing-optimization-unit-id",
        }

    endpoint = OUTCOME_PATH_TEMPLATE.format(unit_id=unit_id)
    url = recommendation_server_url() + endpoint
    payload = _sanitize_features(outcome_features)
    meta: dict[str, Any] = {
        "enabled": True,
        "server_url": recommendation_server_url(),
        "endpoint": endpoint,
        "optimization_unit_id": unit_id,
    }
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
            response = await client.patch(url, json=payload)
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        if response.status_code >= 400:
            meta.update({
                "status": "error",
                "reason": "server-error",
                "error": response.text[:500],
            })
            return meta
        meta.update({
            "status": "sent",
            "reason": "accepted",
        })
        return meta
    except Exception as exc:
        meta.update({
            "latency_ms": int((time.time() - started) * 1000),
            "status": "error",
            "reason": "request-failed",
            "error": repr(exc),
        })
        return meta
