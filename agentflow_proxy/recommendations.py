from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agentflow_proxy.codex_app_policy import CODEX_APP_SOURCE_SURFACE
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.quality import derive_codex_turn_quality_signals, derive_provider_quality_signals
from agentflow_proxy.store import stable_json, utc_now


RECOMMENDATION_PATH = "/v1/recommendation"
OUTCOME_PATH_TEMPLATE = "/v1/optimization-units/{unit_id}/outcome"
POLICY_EVENTS_PATH = "/v1/policy-events"
FEATURE_SCHEMA_VERSION = "agentflow.optimization_unit_features.v1"
MANAGED_API_KEY_ENV = "AGENTFLOW_MANAGED_API_KEY"
RECOMMENDATION_SERVER_URL_ENV = "AGENTFLOW_RECOMMENDATION_SERVER_URL"
RECOMMENDATION_TIMEOUT_ENV = "AGENTFLOW_RECOMMENDATION_TIMEOUT_SECONDS"
RECOMMENDATION_FAILURE_MODE_ENV = "AGENTFLOW_RECOMMENDATION_FAILURE_MODE"
DEFAULT_RECOMMENDATION_SERVER_URL = "http://127.0.0.1:4100"
ROLLOUT_ACTION_LIFECYCLE_SOURCE_SURFACE = "rollout_action_lifecycle"
OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE = "old_context_summary_lifecycle"
OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE = "old_context_summary_outcome"

RAW_FEATURE_KEYS = {
    "account_id",
    "arguments",
    "api_key",
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "command",
    "command_text",
    "commands",
    "completion",
    "content",
    "developer",
    "file_content",
    "file_contents",
    "generated_summary",
    "generated_summaries",
    "input",
    "local_file",
    "local_session_id",
    "local_session_ids",
    "message",
    "messages",
    "output",
    "params",
    "pattern_text",
    "prompt",
    "provider_body",
    "provider_request",
    "provider_response",
    "payload",
    "payloads",
    "raw_context",
    "raw_context_turns",
    "raw_messages",
    "raw_old_context",
    "raw_payload",
    "raw_payloads",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request",
    "request_body",
    "request_id",
    "request_ids",
    "response",
    "response_body",
    "secret",
    "session_id",
    "session_ids",
    "summary_prompt",
    "summary_prompts",
    "summary_text",
    "system",
    "system_prompt",
    "tenant_id",
    "tenant_ids",
    "token",
    "tool_input",
    "tool_output",
    "tool_payload",
    "tool_payloads",
    "transcript",
    "transcripts",
}

LIFECYCLE_METADATA_COMMAND_SCHEMAS = {
    "agentflow.old_context_summary_lifecycle_metadata.v1",
    "agentflow.rollout_action_lifecycle_metadata.v1",
}

TOKEN_CHARS = 4
CHAR_BUCKETS = (
    (2_000, "lt_2k_chars"),
    (8_000, "2k_8k_chars"),
    (32_000, "8k_32k_chars"),
    (128_000, "32k_128k_chars"),
)
TOKEN_BUCKETS = (
    (1_000, "lt_1k_tokens"),
    (4_000, "1k_4k_tokens"),
    (16_000, "4k_16k_tokens"),
    (64_000, "16k_64k_tokens"),
)


def _hash_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_only_privacy_summary() -> dict[str, Any]:
    return {
        "telemetry_profile": "metadata-only",
        "raw_body_storage": False,
        "metadata_only": True,
        "aggregate_only": False,
        "raw_payload_included": False,
    }


def _compact_grouping_identifiers(values: dict[str, str | None]) -> dict[str, str]:
    return {
        key: hashed
        for key, value in values.items()
        if (hashed := _hash_identifier(value)) is not None
    }


def _bucket_number(value: Any, buckets: tuple[tuple[int, str], ...], fallback: str) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 0:
        return "unknown"
    for upper, label in buckets:
        if number < upper:
            return label
    return fallback


def _text_bucket(value: Any) -> str:
    return _bucket_number(value, CHAR_BUCKETS, "gte_128k_chars")


def _token_bucket(value: Any) -> str:
    return _bucket_number(value, TOKEN_BUCKETS, "gte_64k_tokens")


def _latency_bucket(value: Any) -> str:
    return _bucket_number(
        value,
        (
            (500, "lt_500ms"),
            (2_000, "500ms_2s"),
            (10_000, "2s_10s"),
            (30_000, "10s_30s"),
        ),
        "gte_30s",
    )


def _retry_bucket(value: Any) -> str:
    return _bucket_number(
        value,
        (
            (1, "none"),
            (2, "one"),
            (4, "two_three"),
        ),
        "gte_4",
    )


def _net_savings_bucket(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if amount < 0:
        return "negative"
    if amount == 0:
        return "zero"
    if amount < 0.001:
        return "lt_0_001_usd"
    if amount < 0.01:
        return "0_001_0_01_usd"
    if amount < 0.1:
        return "0_01_0_1_usd"
    return "gte_0_1_usd"


def _model_family(model: str | None) -> str | None:
    if not model:
        return None
    model_l = model.lower()
    for family in ("haiku", "sonnet", "opus", "codex", "gpt-5", "gpt-4", "gpt-3"):
        if family in model_l:
            return family
    return "other"


def _decision_status(meta: dict[str, Any], *, default: str = "unknown") -> str:
    status = meta.get("status")
    if isinstance(status, str) and status:
        return status
    if meta.get("applied") is True:
        return "applied"
    if meta.get("changed") is True:
        return "applied"
    if meta:
        return default
    return "missing"


def _pattern_hash(descriptor: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(stable_json(descriptor).encode("utf-8")).hexdigest()


def _codex_pattern_summaries(crunch_meta: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = crunch_meta.get("codex_patterns")
    if not isinstance(patterns, list):
        repeated = crunch_meta.get("codex_repeated_scaffolding")
        patterns = repeated.get("patterns") if isinstance(repeated, dict) else []
    summaries: list[dict[str, Any]] = []
    if not isinstance(patterns, list):
        return summaries
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        pattern_type = pattern.get("type")
        if not isinstance(pattern_type, str) or not pattern_type:
            continue
        summaries.append({
            "type": pattern_type,
            "count_bucket": _bucket_number(pattern.get("count"), ((1, "zero"), (2, "one"), (5, "two_four")), "gte_5"),
            "saved_chars_bucket": _text_bucket(pattern.get("saved_chars_est")),
        })
    return sorted(summaries, key=lambda item: (item["type"], item["count_bucket"], item["saved_chars_bucket"]))


def _pattern_features(
    *,
    source_surface: str,
    granularity: str,
    app_family: str,
    requested_model: str | None,
    candidate_target_model: str | None,
    category: str | None,
    workflow_phase: str | None,
    text_chars: Any,
    input_tokens_est: Any,
    has_tools: Any,
    stream: bool | None,
    replayability_level: str,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
) -> dict[str, Any]:
    routing_status = _decision_status(routing_meta, default="skipped")
    if (
        routing_status == "skipped"
        and isinstance(requested_model, str)
        and isinstance(candidate_target_model, str)
        and candidate_target_model
        and candidate_target_model != requested_model
    ):
        routing_status = "applied"
    crunch_status = _decision_status(crunch_meta, default="skipped")
    cache_status = str(cache_meta.get("status") or _decision_status(cache_meta, default="skipped"))
    pattern_summaries = _codex_pattern_summaries(crunch_meta)
    pattern_types = sorted({item["type"] for item in pattern_summaries})
    descriptor: dict[str, Any] = {
        "schema": "agentflow.normalized_pattern_descriptor.v1",
        "source_surface": source_surface,
        "granularity": granularity,
        "app_family": app_family,
        "requested_model_family": _model_family(requested_model),
        "candidate_target_model_family": _model_family(candidate_target_model),
        "category": category or "unknown",
        "workflow_phase": workflow_phase or category or "unknown",
        "text_bucket": _text_bucket(text_chars),
        "token_bucket": _token_bucket(input_tokens_est),
        "has_tools": bool(has_tools),
        "stream": bool(stream),
        "replayability_level": replayability_level,
        "routing_status": routing_status,
        "crunch_status": crunch_status,
        "cache_status": cache_status,
        "cache_eligible": bool(cache_meta.get("eligible")),
        "crunch_changed": bool(crunch_meta.get("changed")),
        "pattern_types": pattern_types,
        "codex_pattern_summaries": pattern_summaries,
    }
    base_hash = _pattern_hash({**descriptor, "pattern_family": "general"})
    crunch_hash = _pattern_hash({**descriptor, "pattern_family": "crunch"})
    cache_hash = _pattern_hash({**descriptor, "pattern_family": "cache"})
    hashes = sorted({base_hash, crunch_hash, cache_hash})
    return {
        "schema": "agentflow.pattern_features.v1",
        "source_surface": source_surface,
        "granularity": granularity,
        "app_family": app_family,
        "category": category or "unknown",
        "workflow_phase": workflow_phase or category or "unknown",
        "text_bucket": descriptor["text_bucket"],
        "token_bucket": descriptor["token_bucket"],
        "has_tools": descriptor["has_tools"],
        "stream": descriptor["stream"],
        "replayability_level": replayability_level,
        "routing_status": routing_status,
        "crunch_status": crunch_status,
        "cache_status": cache_status,
        "local_decision_status": f"routing:{routing_status}|crunch:{crunch_status}|cache:{cache_status}",
        "pattern_types": pattern_types,
        "codex_pattern_summaries": pattern_summaries,
        "pattern_hash": base_hash,
        "normalized_pattern_hash": base_hash,
        "crunch_pattern_hash": crunch_hash,
        "cache_pattern_hash": cache_hash,
        "pattern_hashes": hashes,
        "pattern_hash_count": len(hashes),
        "pattern_hash_algorithm": "sha256",
        "hash_basis": "normalized-structure-and-size-buckets",
        "raw_pattern_storage": False,
        "raw_pattern_strings_included": False,
    }


def pattern_feature_diagnostics(unit: dict[str, Any]) -> dict[str, Any]:
    input_features = unit.get("input_features") if isinstance(unit, dict) else None
    pattern_features = input_features.get("pattern_features") if isinstance(input_features, dict) else None
    if not isinstance(pattern_features, dict):
        pattern_features = unit.get("pattern_features") if isinstance(unit, dict) else None
    if not isinstance(pattern_features, dict):
        return {
            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
            "present": False,
            "pattern_hash_count": 0,
            "raw_pattern_strings_included": False,
        }
    hashes = [
        pattern_features.get("pattern_hash"),
        pattern_features.get("crunch_pattern_hash"),
        pattern_features.get("cache_pattern_hash"),
    ]
    hashes = [str(item) for item in hashes if isinstance(item, str) and item.startswith("sha256:")]
    return {
        "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
        "present": bool(hashes),
        "pattern_hash_count": len(sorted(set(hashes))),
        "pattern_hashes": sorted(set(hashes)),
        "hash_basis": pattern_features.get("hash_basis"),
        "text_bucket": pattern_features.get("text_bucket"),
        "token_bucket": pattern_features.get("token_bucket"),
        "workflow_phase": pattern_features.get("workflow_phase"),
        "category": pattern_features.get("category"),
        "source_surface": pattern_features.get("source_surface"),
        "app_family": pattern_features.get("app_family"),
        "requested_model": unit.get("requested_model"),
        "candidate_target_model": unit.get("candidate_target_model"),
        "replayability_level": pattern_features.get("replayability_level"),
        "has_tools": pattern_features.get("has_tools"),
        "stream": pattern_features.get("stream"),
        "pattern_types": pattern_features.get("pattern_types") or [],
        "raw_pattern_strings_included": False,
    }


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def recommendations_enabled() -> bool:
    return _as_bool(os.getenv("AGENTFLOW_RECOMMENDATION_ENABLED"), False)


def recommendation_server_url() -> str:
    raw = os.getenv(RECOMMENDATION_SERVER_URL_ENV)
    if raw is None:
        raw = DEFAULT_RECOMMENDATION_SERVER_URL
    return raw.strip().rstrip("/")


def recommendation_server_configured() -> bool:
    return bool(recommendation_server_url())


def recommendation_timeout_seconds() -> float:
    try:
        return max(0.05, float(os.getenv(RECOMMENDATION_TIMEOUT_ENV, "1.5")))
    except ValueError:
        return 1.5


def recommendation_failure_mode() -> str:
    mode = os.getenv(RECOMMENDATION_FAILURE_MODE_ENV, "fallback-local").strip().lower()
    return mode if mode in {"fallback-local"} else "fallback-local"


def managed_auth_configured() -> bool:
    return bool(os.getenv(MANAGED_API_KEY_ENV))


def _managed_headers() -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-agentflow-local-fallback": "local-policy",
    }
    api_key = os.getenv(MANAGED_API_KEY_ENV)
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def outcome_feedback_queue_max_attempts() -> int:
    try:
        return max(1, int(os.getenv("AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS", "3")))
    except ValueError:
        return 3


def outcome_feedback_queue_retry_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS", "60")))
    except ValueError:
        return 60.0


def _future_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))).isoformat()


def _sanitize_features(value: Any) -> Any:
    if isinstance(value, dict):
        allow_command = value.get("schema") in LIFECYCLE_METADATA_COMMAND_SCHEMAS
        return {
            str(key): _sanitize_features(item)
            for key, item in value.items()
            if str(key).lower() not in RAW_FEATURE_KEYS or (allow_command and str(key).lower() == "command")
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
    session_id: str | None = None,
) -> dict[str, Any]:
    text_chars = routing_meta.get("text_chars")
    has_tools = routing_meta.get("has_tools")
    source_surface = _source_surface(provider, path)
    granularity = "provider_request"
    app_family = _app_family(provider, requested_model, path)
    candidate_target_model = routed_model if routed_model else None
    replayability_level = "features_only"
    workflow_phase = routing_meta.get("workflow_phase") or category or routing_meta.get("category")
    workflow_phase_reason = routing_meta.get("workflow_phase_reason")
    pattern_features = _pattern_features(
        source_surface=source_surface,
        granularity=granularity,
        app_family=app_family,
        requested_model=requested_model,
        candidate_target_model=candidate_target_model,
        category=category or routing_meta.get("category"),
        workflow_phase=workflow_phase,
        text_chars=text_chars,
        input_tokens_est=input_tokens_est,
        has_tools=has_tools,
        stream=stream,
        replayability_level=replayability_level,
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )
    unit = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_surface": source_surface,
        "granularity": granularity,
        "app_family": app_family,
        "requested_model": requested_model,
        "candidate_target_model": candidate_target_model,
        "input_features": {
            "path": path,
            "stream": bool(stream),
            "category": category or routing_meta.get("category"),
            "workflow_phase": workflow_phase,
            "workflow_phase_reason": workflow_phase_reason,
            "workflow_phase_confidence": routing_meta.get("workflow_phase_confidence"),
            "text_bucket": pattern_features["text_bucket"],
            "input_token_bucket": pattern_features["token_bucket"],
            "text_chars": text_chars,
            "input_tokens_est": input_tokens_est,
            "local_routed_model": routed_model,
            "local_routing_reason": routing_meta.get("reason"),
            "local_routing_policy_source": routing_meta.get("policy_source"),
            "crunch_changed": bool(crunch_meta.get("changed")),
            "crunch_saved_chars": crunch_meta.get("saved_chars"),
            "cache_status": cache_meta.get("status"),
            "cache_reason": cache_meta.get("reason"),
            "pattern_features": pattern_features,
            "pattern_hash": pattern_features["pattern_hash"],
            "normalized_pattern_hash": pattern_features["normalized_pattern_hash"],
            "crunch_pattern_hash": pattern_features["crunch_pattern_hash"],
            "cache_pattern_hash": pattern_features["cache_pattern_hash"],
        },
        "tool_features": {
            "has_tools": has_tools,
            "category": category or routing_meta.get("category"),
            "workflow_phase": workflow_phase,
            "thinking_history_stripped": routing_meta.get("thinking_history_stripped"),
            "stripped_params": routing_meta.get("stripped_params") or [],
        },
        "outcome_features": {},
        "grouping_identifiers": _compact_grouping_identifiers({
            "session_id_hash": session_id,
        }),
        "privacy_summary": _metadata_only_privacy_summary(),
        "replayability_level": replayability_level,
        "pattern_features": pattern_features,
    }
    return _sanitize_features(unit)


def _codex_model_state(routing_meta: dict[str, Any]) -> tuple[str, str | None, str | None, str | None]:
    requested_model = routing_meta.get("requested_model")
    routed_model = routing_meta.get("routed_model")
    model_field = routing_meta.get("model_field")
    if isinstance(requested_model, str) and requested_model:
        return "present", model_field if isinstance(model_field, str) else None, requested_model, routed_model if isinstance(routed_model, str) else requested_model
    if routing_meta.get("reason") == "codex-turn-start-model-field-absent":
        return "absent", None, None, None
    return "unknown", model_field if isinstance(model_field, str) else None, None, None


def build_codex_turn_optimization_unit(
    *,
    method: str,
    request_id_present: bool,
    thread_id_present: bool,
    params_chars: int | None,
    input_items: int | None,
    input_text_chars: int | None,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    workflow_phase: str | None = None,
    workflow_phase_reason: str | None = None,
    request_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    model_state, model_field, requested_model, routed_model = _codex_model_state(routing_meta)
    input_tokens_est = max(1, int(input_text_chars / TOKEN_CHARS)) if input_text_chars else 0
    workflow_phase_value = workflow_phase or routing_meta.get("workflow_phase") or "unknown"
    replayability_level = str(cache_meta.get("replayability_level") or "features_only")
    pattern_features = _pattern_features(
        source_surface=CODEX_APP_SOURCE_SURFACE,
        granularity="agent_turn",
        app_family="codex",
        requested_model=requested_model,
        candidate_target_model=routed_model,
        category=routing_meta.get("category") or "codex_turn",
        workflow_phase=workflow_phase_value,
        text_chars=input_text_chars,
        input_tokens_est=input_tokens_est,
        has_tools=False,
        stream=False,
        replayability_level=replayability_level,
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )
    unit = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "app_family": "codex",
        "requested_model": requested_model,
        "candidate_target_model": routed_model,
        "input_features": {
            "jsonrpc_method": method,
            "request_id_present": bool(request_id_present),
            "thread_id_present": bool(thread_id_present),
            "params_chars": params_chars,
            "input_items": input_items,
            "input_text_chars": input_text_chars,
            "input_tokens_est": input_tokens_est,
            "text_bucket": pattern_features["text_bucket"],
            "input_token_bucket": pattern_features["token_bucket"],
            "workflow_phase": workflow_phase_value,
            "workflow_phase_reason": workflow_phase_reason or routing_meta.get("workflow_phase_reason"),
            "model_field_state": model_state,
            "model_field_name": model_field,
            "local_routed_model": routed_model,
            "local_routing_reason": routing_meta.get("reason"),
            "local_routing_policy_source": routing_meta.get("policy_source"),
            "routing_status": routing_meta.get("status"),
            "crunch_status": crunch_meta.get("status"),
            "crunch_changed": bool(crunch_meta.get("changed")),
            "crunch_saved_chars": crunch_meta.get("saved_chars"),
            "cache_status": cache_meta.get("status"),
            "cache_reason": cache_meta.get("reason"),
            "cache_eligible": bool(cache_meta.get("eligible")),
            "cache_replayability_level": cache_meta.get("replayability_level"),
            "pattern_features": pattern_features,
            "pattern_hash": pattern_features["pattern_hash"],
            "normalized_pattern_hash": pattern_features["normalized_pattern_hash"],
            "crunch_pattern_hash": pattern_features["crunch_pattern_hash"],
            "cache_pattern_hash": pattern_features["cache_pattern_hash"],
        },
        "tool_features": {
            "category": routing_meta.get("category") or "codex_turn",
            "action_like_skipped": routing_meta.get("reason") == "action-like-params",
            "unknown_param_shape": cache_meta.get("reason") == "unknown-param-shape",
            "non_text_input": cache_meta.get("reason") == "non-text-input",
            "safe_param_policy_source": routing_meta.get("policy_source") or cache_meta.get("policy_source"),
        },
        "outcome_features": {},
        "grouping_identifiers": _compact_grouping_identifiers({
            "request_id_hash": request_id,
            "thread_id_hash": thread_id,
        }),
        "privacy_summary": _metadata_only_privacy_summary(),
        "replayability_level": replayability_level,
        "pattern_features": pattern_features,
    }
    return _sanitize_features(unit)


def _base_meta() -> dict[str, Any]:
    return {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": RECOMMENDATION_PATH,
        "timeout_seconds": recommendation_timeout_seconds(),
        "failure_mode": recommendation_failure_mode(),
        "auth_configured": managed_auth_configured(),
        "api_key_value_included": False,
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
    if not recommendation_server_configured():
        meta.update({
            "status": "skipped",
            "reason": "server-url-not-configured",
            "fallback": "local-policy",
            "applied": False,
        })
        return meta

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + RECOMMENDATION_PATH,
                json=unit,
                headers=_managed_headers(),
            )
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
                "reason": "invalid-schema",
                "schema_error": error or "invalid-response",
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
    except httpx.TimeoutException as exc:
        meta.update({
            "latency_ms": int((time.time() - started) * 1000),
            "status": "error",
            "reason": "timeout",
            "error": repr(exc),
            "fallback": "local-policy",
            "applied": False,
        })
        return meta
    except httpx.NetworkError as exc:
        meta.update({
            "latency_ms": int((time.time() - started) * 1000),
            "status": "error",
            "reason": "unreachable",
            "error": repr(exc),
            "fallback": "local-policy",
            "applied": False,
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


def _supported_target_model(provider: str, target_model: str) -> bool:
    if not _provider_compatible(provider, target_model):
        return False
    target_l = target_model.lower()
    if provider == "anthropic":
        return any(tier in target_l for tier in ("haiku", "sonnet", "opus"))
    return bool(target_l)


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
    if not _supported_target_model(provider, target_model):
        meta.update({"applied": False, "apply_reason": "unsupported-target-model", "fallback": "local-policy"})
        return meta
    if meta.get("replacement_prompt_present"):
        meta.update({
            "applied": False,
            "apply_reason": "unsafe-replacement-prompt",
            "fallback": "local-policy",
            "replacement_prompt_applied": False,
        })
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


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pattern_outcome(status_code: int | None, *, applied: bool, bypassed: bool = False) -> str:
    if status_code is not None and int(status_code) >= 400:
        return "errored"
    if applied:
        return "applied"
    if bypassed:
        return "bypassed"
    return "skipped"


def _pattern_canary_cohort(meta: dict[str, Any]) -> str | None:
    canary = meta.get("canary")
    if not isinstance(canary, dict) or not canary.get("enabled"):
        return None
    cohort = canary.get("cohort")
    if isinstance(cohort, str) and cohort:
        return cohort
    status = canary.get("status")
    if status == "holdout":
        return "canary_holdout"
    if status == "applied":
        return "canary_applied"
    return None


def _copy_canary_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    canary = meta.get("canary")
    if not isinstance(canary, dict) or not canary.get("enabled"):
        return None
    return {
        key: canary.get(key)
        for key in (
            "schema",
            "enabled",
            "selected",
            "status",
            "cohort",
            "reason",
            "fraction",
            "salt",
            "unit",
            "bucket",
            "threshold",
            "cohort_key_hash",
        )
        if canary.get(key) is not None
    }


def _pattern_cost_savings(
    *,
    provider: str,
    model: str | None,
    saved_tokens: int,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
) -> float:
    if cost_est_usd is not None and cost_baseline_usd is not None:
        return max(_safe_float(cost_baseline_usd) - _safe_float(cost_est_usd), 0.0)
    if not model or saved_tokens <= 0:
        return 0.0
    return float(estimate_cost(model, saved_tokens, 0, provider=provider) or 0.0)


def pattern_decision_summaries(
    *,
    provider: str,
    path: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    cache_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    category: str | None,
) -> list[dict[str, Any]]:
    """Build metadata-only pattern decision summaries for local stats and managed feedback."""
    source_surface = _source_surface(provider, path)
    app_family = _app_family(provider, requested_model or "", path)
    workflow_phase = category or routing_meta.get("category") or "unknown"
    model = routed_model or requested_model
    rows: list[dict[str, Any]] = []

    pattern_rules = crunch_meta.get("pattern_rules") if isinstance(crunch_meta, dict) else None
    if isinstance(pattern_rules, dict) and pattern_rules.get("configured_count"):
        for rule in pattern_rules.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            saved_chars = _safe_int(rule.get("saved_chars"))
            saved_tokens = max(0, saved_chars // TOKEN_CHARS)
            applied = _safe_int(rule.get("applied_count")) > 0
            base = {
                "schema": "agentflow.pattern_decision_summary.v1",
                "decision_type": "crunch",
                "source_surface": source_surface,
                "app_family": app_family,
                "category": category or routing_meta.get("category") or pattern_rules.get("category") or "unknown",
                "workflow_phase": workflow_phase,
                "rule_id": rule.get("rule_id"),
                "candidate_id": rule.get("candidate_id"),
                "policy_source": rule.get("policy_source") or pattern_rules.get("policy_source") or crunch_meta.get("policy_source"),
                "status": "applied" if applied else "skipped",
                "outcome": _pattern_outcome(status_code, applied=applied),
                "action": rule.get("action"),
                "applied_count": _safe_int(rule.get("applied_count")),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "estimated_cost_savings_usd": round(_pattern_cost_savings(
                    provider=provider,
                    model=model,
                    saved_tokens=saved_tokens,
                    cost_est_usd=None,
                    cost_baseline_usd=None,
                ), 8),
                "before_chars": pattern_rules.get("before_chars"),
                "after_chars": pattern_rules.get("after_chars"),
                "safety_gates": {
                    "safe_text_only": True,
                    "tool_or_action_payloads_skipped": any(
                        isinstance(item, dict) and item.get("reason") == "unsafe-tool-or-action-payload"
                        for item in rule.get("skip_reasons") or []
                    ),
                    "raw_pattern_strings_included": False,
                },
            }
            canary_meta = _copy_canary_meta(rule)
            if canary_meta:
                base["canary"] = canary_meta
                base["cohort"] = _pattern_canary_cohort(rule)
                if canary_meta.get("status") == "holdout":
                    base["status"] = "holdout"
                    base["reason"] = "canary_holdout"
                    base["outcome"] = "holdout"
            matched_hashes = [
                str(item)
                for item in rule.get("matched_hashes") or []
                if isinstance(item, str) and item.startswith("sha256:")
            ]
            if matched_hashes:
                base["pattern_hash"] = matched_hashes[0]
                base["pattern_hashes"] = matched_hashes
            skip_reasons = []
            for item in rule.get("skip_reasons") or []:
                if isinstance(item, dict):
                    skip_reasons.append({
                        "reason": item.get("reason"),
                        "count": _safe_int(item.get("count")),
                        "pattern_hash": item.get("pattern_hash") if isinstance(item.get("pattern_hash"), str) and str(item.get("pattern_hash")).startswith("sha256:") else None,
                    })
            if skip_reasons:
                base["skip_reasons"] = skip_reasons
                if not applied:
                    base["reason"] = str(skip_reasons[0].get("reason") or "skipped")
                    if base["reason"] == "local-canary-safety-stop":
                        base["status"] = "bypass"
                        base["outcome"] = "bypassed"
            rows.append(base)

        for item in pattern_rules.get("skip_reasons") or []:
            if not isinstance(item, dict):
                continue
            if any(row.get("rule_id") == item.get("rule_id") for row in rows):
                continue
            rows.append({
                "schema": "agentflow.pattern_decision_summary.v1",
                "decision_type": "crunch",
                "source_surface": source_surface,
                "app_family": app_family,
                "category": category or routing_meta.get("category") or pattern_rules.get("category") or "unknown",
                "workflow_phase": workflow_phase,
                "rule_id": item.get("rule_id"),
                "policy_source": pattern_rules.get("policy_source") or crunch_meta.get("policy_source"),
                "status": "skipped",
                "reason": item.get("reason"),
                "outcome": _pattern_outcome(status_code, applied=False),
                "applied_count": 0,
                "saved_chars": 0,
                "tokens_saved_est": 0,
                "estimated_cost_savings_usd": 0.0,
                "safety_gates": {
                    "safe_text_only": True,
                    "raw_pattern_strings_included": False,
                },
            })

    if isinstance(cache_meta, dict) and cache_meta:
        cache_status = str(cache_meta.get("status") or "missing")
        cache_reason = str(cache_meta.get("reason") or "unknown")
        cache_applied = cache_status == "hit"
        cache_bypassed = cache_status in {"bypass", "bypassed"} or "bypass" in cache_reason or "disabled" in cache_reason
        pattern_rule = cache_meta.get("pattern_rule") if isinstance(cache_meta.get("pattern_rule"), dict) else {}
        pattern_rules = cache_meta.get("pattern_rules") if isinstance(cache_meta.get("pattern_rules"), dict) else {}
        descriptor = {
            "schema": "agentflow.cache_pattern_decision_basis.v1",
            "source_surface": source_surface,
            "app_family": app_family,
            "category": category or routing_meta.get("category") or "unknown",
            "workflow_phase": workflow_phase,
            "requested_model_family": _model_family(requested_model),
            "routed_model_family": _model_family(routed_model or requested_model),
            "status": cache_status,
            "reason": cache_reason,
            "hit_type": cache_meta.get("hit_type"),
            "policy_source": cache_meta.get("policy_source"),
            "eligible": bool(cache_meta.get("eligible")),
            "replayability_level": cache_meta.get("replayability_level"),
        }
        rows.append({
            "schema": "agentflow.pattern_decision_summary.v1",
            "decision_type": "cache",
            "source_surface": source_surface,
            "app_family": app_family,
            "category": descriptor["category"],
            "workflow_phase": workflow_phase,
            "rule_id": pattern_rule.get("rule_id") or cache_meta.get("rule_id") or cache_meta.get("policy_id") or cache_status,
            "candidate_id": pattern_rule.get("candidate_id") or cache_meta.get("candidate_id"),
            "pattern_hash": (
                pattern_rule.get("matched_hashes", [None])[0]
                if isinstance(pattern_rule.get("matched_hashes"), list) and pattern_rule.get("matched_hashes")
                else cache_meta.get("pattern_hash") if isinstance(cache_meta.get("pattern_hash"), str) and str(cache_meta.get("pattern_hash")).startswith("sha256:")
                else _pattern_hash(descriptor)
            ),
            "pattern_hashes": pattern_rule.get("matched_hashes") if isinstance(pattern_rule.get("matched_hashes"), list) else None,
            "policy_source": pattern_rule.get("policy_source") or cache_meta.get("policy_source"),
            "status": cache_status,
            "reason": cache_reason,
            "outcome": _pattern_outcome(status_code, applied=cache_applied, bypassed=cache_bypassed),
            "hit_type": cache_meta.get("hit_type"),
            "applied_count": 1 if cache_applied else 0,
            "saved_chars": 0,
            "tokens_saved_est": 0,
            "estimated_cost_savings_usd": round(_pattern_cost_savings(
                provider=provider,
                model=model,
                saved_tokens=0,
                cost_est_usd=cost_est_usd if cache_applied else None,
                cost_baseline_usd=cost_baseline_usd if cache_applied else None,
            ), 8),
            "safety_gates": {
                "exact_enabled": bool(cache_meta.get("exact_enabled")),
                "semantic_enabled": bool(cache_meta.get("semantic_enabled")),
                "tool_cache_enabled": bool(cache_meta.get("tool_cache_enabled")),
                "file_watch_enabled": bool(cache_meta.get("file_watch_enabled")),
                "eligible": bool(cache_meta.get("eligible")),
                "raw_pattern_strings_included": False,
            },
        })
        if pattern_rule:
            canary_meta = _copy_canary_meta(pattern_rule)
            if canary_meta:
                rows[-1]["canary"] = canary_meta
                rows[-1]["cohort"] = _pattern_canary_cohort(pattern_rule)

        for item in pattern_rules.get("skip_reasons") or []:
            if not isinstance(item, dict) or item.get("reason") not in {"canary_holdout", "local-canary-safety-stop"}:
                continue
            safety_stopped = item.get("reason") == "local-canary-safety-stop"
            canary_meta = _copy_canary_meta(item)
            rows.append({
                "schema": "agentflow.pattern_decision_summary.v1",
                "decision_type": "cache",
                "source_surface": source_surface,
                "app_family": app_family,
                "category": descriptor["category"],
                "workflow_phase": workflow_phase,
                "rule_id": item.get("rule_id"),
                "candidate_id": item.get("candidate_id"),
                "pattern_hash": item.get("matched_hashes", [None])[0] if isinstance(item.get("matched_hashes"), list) and item.get("matched_hashes") else _pattern_hash(descriptor),
                "pattern_hashes": item.get("matched_hashes") if isinstance(item.get("matched_hashes"), list) else None,
                "policy_source": item.get("policy_source") or cache_meta.get("policy_source"),
                "status": "bypass" if safety_stopped else "holdout",
                "reason": "local-canary-safety-stop" if safety_stopped else "canary_holdout",
                "outcome": "bypassed" if safety_stopped else "holdout",
                "hit_type": cache_meta.get("hit_type"),
                "applied_count": 0,
                "saved_chars": 0,
                "tokens_saved_est": 0,
                "estimated_cost_savings_usd": 0.0,
                "canary": canary_meta,
                "cohort": _pattern_canary_cohort(item),
                "safety_stop": item.get("safety_stop") if isinstance(item.get("safety_stop"), dict) else None,
                "safety_gates": {
                    "exact_enabled": bool(cache_meta.get("exact_enabled")),
                    "semantic_enabled": bool(cache_meta.get("semantic_enabled")),
                    "tool_cache_enabled": bool(cache_meta.get("tool_cache_enabled")),
                    "file_watch_enabled": bool(cache_meta.get("file_watch_enabled")),
                    "raw_pattern_strings_included": False,
                },
            })

    return _sanitize_features(rows)


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
    source_surface = _source_surface(provider, path)
    quality_signals = derive_provider_quality_signals(
        source_surface=source_surface,
        status_code=status_code,
        retry_count=retry_count,
        latency_ms=latency_ms,
        error=error,
        requested_model=requested_model,
        routed_model=routed_model,
        cache_hit=cache_meta.get("status") == "hit",
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )
    features: dict[str, Any] = {
        "provider": provider,
        "source_surface": source_surface,
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
        "pattern_decisions": pattern_decision_summaries(
            provider=provider,
            path=path,
            requested_model=requested_model,
            routed_model=routed_model,
            status_code=status_code,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
        ),
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
        "quality_signals": quality_signals,
    }
    summary_feedback = build_old_context_summary_outcome_feedback(
        provider=provider,
        path=path,
        requested_model=requested_model,
        routed_model=routed_model,
        status_code=status_code,
        latency_ms=latency_ms,
        retry_count=retry_count,
        cache_hit=cache_meta.get("status") == "hit",
        crunch_meta=crunch_meta,
        category=category,
        error=error,
    )
    if summary_feedback is not None:
        features["old_context_summarization"] = summary_feedback
    experiment = routing_meta.get("routing_experiment") if isinstance(routing_meta, dict) else None
    if isinstance(experiment, dict):
        feedback_features = experiment.get("optimization_feedback")
        if isinstance(feedback_features, dict):
            features["routing_experiment"] = feedback_features
    features.update(_compact_error(error, status_code))
    return _sanitize_features(features)


def build_old_context_summary_outcome_feedback(
    *,
    provider: str,
    path: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    cache_hit: bool,
    crunch_meta: dict[str, Any],
    category: str | None,
    error: str | None = None,
) -> dict[str, Any] | None:
    summary_meta = crunch_meta.get("old_context_summarization") if isinstance(crunch_meta, dict) else None
    if not isinstance(summary_meta, dict) or not summary_meta.get("enabled"):
        return None
    status = str(summary_meta.get("status") or "")
    reason = str(summary_meta.get("reason") or "")
    canary = summary_meta.get("canary") if isinstance(summary_meta.get("canary"), dict) else {}
    relevant = (
        status in {"applied", "bypass"}
        or reason in {"canary_holdout", "local-canary-safety-stop"}
        or (canary.get("enabled") and canary.get("cohort") in {"canary_applied", "canary_holdout"})
    )
    if not relevant:
        return None

    try:
        eligible_chars = int(summary_meta.get("eligible_chars") or 0)
    except (TypeError, ValueError):
        eligible_chars = 0
    try:
        eligible_turns = int(summary_meta.get("eligible_turns") or 0)
    except (TypeError, ValueError):
        eligible_turns = 0
    try:
        saved_chars = int(summary_meta.get("saved_chars") or 0)
    except (TypeError, ValueError):
        saved_chars = 0
    try:
        saved_tokens = int(summary_meta.get("tokens_saved_est") or 0)
    except (TypeError, ValueError):
        saved_tokens = max(0, saved_chars // TOKEN_CHARS)
    try:
        summary_cost = float(summary_meta.get("summary_cost_est_usd") or 0.0)
    except (TypeError, ValueError):
        summary_cost = 0.0
    try:
        net_savings = float(summary_meta.get("estimated_net_savings_usd") or 0.0)
    except (TypeError, ValueError):
        net_savings = 0.0

    safety_stop = summary_meta.get("safety_stop") if isinstance(summary_meta.get("safety_stop"), dict) else {}
    feedback = {
        "schema": "agentflow.old_context_summary_outcome_feedback.v1",
        "source_surface": _source_surface(provider, path),
        "category": category or summary_meta.get("category") or "unknown",
        "status": status or "unknown",
        "reason": reason or None,
        "outcome": (
            "errored"
            if status_code is not None and int(status_code) >= 400
            else "applied"
            if status == "applied"
            else "holdout"
            if reason == "canary_holdout"
            else "bypassed"
            if status == "bypass"
            else "skipped"
        ),
        "requested_model_tier": _model_family(requested_model),
        "routed_model_tier": _model_family(routed_model),
        "summary_policy_id": summary_meta.get("rule_id"),
        "rule_id": summary_meta.get("rule_id"),
        "candidate_id": summary_meta.get("candidate_id"),
        "policy_source": summary_meta.get("policy_source"),
        "canary": {
            "enabled": bool(canary.get("enabled")),
            "selected": canary.get("selected"),
            "status": canary.get("status"),
            "cohort": canary.get("cohort"),
            "fraction": canary.get("fraction"),
            "unit": canary.get("unit"),
        },
        "canary_cohort": canary.get("cohort"),
        "eligible_turns": eligible_turns,
        "eligible_chars": eligible_chars,
        "eligible_chars_bucket": _text_bucket(eligible_chars),
        "request_chars_bucket": _text_bucket(summary_meta.get("before_chars")),
        "saved_chars": saved_chars,
        "saved_tokens_est": saved_tokens,
        "saved_tokens_bucket": _token_bucket(saved_tokens),
        "summary_cost_est_usd": round(summary_cost, 8),
        "summary_cache_hit": bool(summary_meta.get("summary_cache_hit")),
        "provider_cache_hit": bool(cache_hit),
        "net_savings_bucket": _net_savings_bucket(net_savings),
        "status_code": status_code,
        "latency_bucket": _latency_bucket(latency_ms),
        "retry_bucket": _retry_bucket(retry_count),
        "error_bucket": (_compact_error(error, status_code).get("error_class") if error else None),
        "safety_stop": {
            "stopped": bool(safety_stop.get("stopped")),
            "reason": safety_stop.get("reason"),
            "trigger_metrics": [
                item.get("metric")
                for item in safety_stop.get("triggers", [])
                if isinstance(item, dict) and item.get("metric")
            ],
        } if safety_stop else None,
        "privacy": {
            "metadata_only": True,
            "raw_old_turns_included": False,
            "raw_summary_included": False,
            "summary_request_content_included": False,
            "provider_request_included": False,
            "provider_response_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
        },
    }
    return _sanitize_features({key: value for key, value in feedback.items() if value not in (None, "", [], {})})


def build_old_context_summary_outcome_event(
    outcome_feedback: dict[str, Any],
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    from agentflow_proxy import __version__

    basis = {
        "rule_id": outcome_feedback.get("rule_id"),
        "candidate_id": outcome_feedback.get("candidate_id"),
        "category": outcome_feedback.get("category"),
        "canary_cohort": outcome_feedback.get("canary_cohort"),
        "outcome": outcome_feedback.get("outcome"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
    return _sanitize_features({
        "event_type": "outcome",
        "occurred_at": occurred_at or utc_now(),
        "recommendation_id": f"old-context-summary:{digest}",
        "bundle_hash": None,
        "policy_sections": ["crunch"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": {
            "schema": "agentflow.old_context_summary_outcome_event_metadata.v1",
            "lifecycle_kind": "old_context_summarization",
            "outcome": outcome_feedback,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "raw_responses_included": False,
                "raw_transcripts_included": False,
                "summary_text_included": False,
                "summary_request_content_included": False,
                "cache_keys_included": False,
                "request_ids_included": False,
                "local_session_ids_included": False,
                "file_paths_included": False,
            },
        },
    })


def build_codex_turn_outcome_feedback(
    *,
    recommendation_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    result_chars: int | None,
    error_code: int | None,
    error_message: str | None,
    latency_ms: int | None,
    input_text_chars: int | None,
    session_id: str | None,
) -> dict[str, Any]:
    input_tokens_est = max(1, int(input_text_chars / TOKEN_CHARS)) if input_text_chars else 0
    output_tokens_est = max(1, int(result_chars / TOKEN_CHARS)) if result_chars else 0
    model = (
        routing_meta.get("routed_model")
        or routing_meta.get("requested_model")
        or codex_app_model()
    )
    cost_est = estimate_cost(
        str(model),
        input_tokens_est,
        output_tokens_est,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    baseline_model = routing_meta.get("requested_model") or model
    baseline_cost = estimate_cost(
        str(baseline_model),
        input_tokens_est,
        output_tokens_est,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16] if session_id else None
    status = "error" if error_code is not None else "success"
    quality_signals = derive_codex_turn_quality_signals(
        response_event_id="managed-feedback-response",
        error_code=error_code,
        error_message=error_message,
        latency_ms=latency_ms,
        routing_meta=routing_meta,
        crunch_meta=crunch_meta,
        cache_meta=cache_meta,
    )
    features: dict[str, Any] = {
        "provider": "codex-app",
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "app_family": "codex",
        "status": status,
        "status_code": 500 if error_code is not None else 200,
        "jsonrpc_error_code": error_code,
        "latency_ms": latency_ms,
        "result_chars": result_chars,
        "input_tokens": input_tokens_est,
        "output_tokens": output_tokens_est,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "cost_est_usd": cost_est,
        "cost_baseline_usd": baseline_cost,
        "cache_decision": {
            key: cache_meta.get(key)
            for key in ("status", "reason", "hit_type", "policy_source", "replayability_level")
            if cache_meta.get(key) is not None
        },
        "routing_decision": {
            "reason": routing_meta.get("reason"),
            "status": routing_meta.get("status"),
            "policy_source": routing_meta.get("policy_source"),
            "model_field": routing_meta.get("model_field"),
            "requested_model": routing_meta.get("requested_model"),
            "routed_model": routing_meta.get("routed_model"),
            "managed_policy_id": routing_meta.get("managed_policy_id"),
        },
        "crunch_decision": {
            "status": crunch_meta.get("status"),
            "reason": crunch_meta.get("reason"),
            "policy_source": crunch_meta.get("policy_source"),
            "saved_chars": crunch_meta.get("saved_chars"),
            "tokens_saved_est": crunch_meta.get("tokens_saved_est"),
        },
        "pattern_decisions": pattern_decision_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing_meta.get("requested_model") or codex_app_model(),
            routed_model=routing_meta.get("routed_model") or routing_meta.get("requested_model") or codex_app_model(),
            status_code=500 if error_code is not None else 200,
            cost_est_usd=cost_est,
            cost_baseline_usd=baseline_cost,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=routing_meta.get("category") or "codex_turn",
        ),
        "session": {
            "present": bool(session_id),
            "id_hash": session_hash,
        },
        "managed_recommendation": {
            "optimization_unit_id": recommendation_meta.get("optimization_unit_id"),
            "recommendation_id": recommendation_meta.get("recommendation_id"),
            "policy_id": recommendation_meta.get("policy_id"),
            "target_model": recommendation_meta.get("target_model"),
            "applied": bool(recommendation_meta.get("applied")),
            "apply_reason": recommendation_meta.get("apply_reason"),
        },
        "quality_signals": quality_signals,
    }
    if error_code is not None:
        features["error_class"] = "jsonrpc_error"
    return _sanitize_features(features)


def disabled_outcome_feedback_meta() -> dict[str, Any]:
    return {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": OUTCOME_PATH_TEMPLATE,
        "timeout_seconds": recommendation_timeout_seconds(),
        "failure_mode": recommendation_failure_mode(),
        "auth_configured": managed_auth_configured(),
        "api_key_value_included": False,
        "status": "skipped",
        "reason": "disabled",
    }


def _outcome_target(recommendation_meta: dict[str, Any]) -> tuple[int | None, str | None]:
    unit_id = recommendation_meta.get("optimization_unit_id")
    if not isinstance(unit_id, int):
        return None, None
    endpoint = OUTCOME_PATH_TEMPLATE.format(unit_id=unit_id)
    return unit_id, endpoint


async def _send_outcome_payload(*, endpoint: str, payload: dict[str, Any], unit_id: int) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "enabled": True,
        "server_url": recommendation_server_url(),
        "endpoint": endpoint,
        "optimization_unit_id": unit_id,
        "timeout_seconds": recommendation_timeout_seconds(),
        "failure_mode": recommendation_failure_mode(),
        "auth_configured": managed_auth_configured(),
        "api_key_value_included": False,
    }
    if not recommendation_server_configured():
        meta.update({
            "status": "skipped",
            "reason": "server-url-not-configured",
        })
        return meta
    url = recommendation_server_url() + endpoint
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
            response = await client.patch(url, json=payload, headers=_managed_headers())
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


async def send_outcome_feedback(recommendation_meta: dict[str, Any], outcome_features: dict[str, Any]) -> dict[str, Any]:
    if not recommendations_enabled():
        return disabled_outcome_feedback_meta()

    unit_id, endpoint = _outcome_target(recommendation_meta)
    if unit_id is None or endpoint is None:
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "endpoint": OUTCOME_PATH_TEMPLATE,
            "status": "skipped",
            "reason": "missing-optimization-unit-id",
        }

    payload = _sanitize_features(outcome_features)
    return await _send_outcome_payload(endpoint=endpoint, payload=payload, unit_id=unit_id)


async def _send_policy_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    meta = {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": POLICY_EVENTS_PATH,
        "timeout_seconds": recommendation_timeout_seconds(),
        "failure_mode": recommendation_failure_mode(),
        "auth_configured": managed_auth_configured(),
        "api_key_value_included": False,
    }
    if not recommendations_enabled():
        meta.update({"status": "disabled", "reason": "disabled"})
        return meta
    if not recommendation_server_configured():
        meta.update({"status": "skipped", "reason": "server-url-not-configured"})
        return meta

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + POLICY_EVENTS_PATH,
                json=_sanitize_features(payload),
                headers=_managed_headers(),
            )
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


def _queue_error_status(meta: dict[str, Any], attempts: int) -> str:
    if attempts >= outcome_feedback_queue_max_attempts():
        return "dropped-after-limit"
    return "retryable-error"


def _queued_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "server_url": recommendation_server_url(),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": row.get("optimization_unit_id"),
        "queue_id": row.get("id"),
        "status": row.get("status") or "queued",
        "reason": row.get("last_error") and "queued-after-error" or "queued",
        "attempts": row.get("attempts") or 0,
    }


async def _flush_claimed_outcome_feedback(store_obj: Any, row: dict[str, Any]) -> dict[str, Any]:
    queue_id = str(row.get("id") or "")
    endpoint = str(row.get("endpoint") or "")
    unit_id = row.get("optimization_unit_id")
    attempts = int(row.get("attempts") or 0)
    try:
        payload = json.loads(row.get("payload_json") or "{}")
    except Exception as exc:
        status = "dropped-after-limit"
        error = f"invalid queued payload: {exc!r}"
        if hasattr(store_obj, "mark_managed_outcome_feedback_retry"):
            store_obj.mark_managed_outcome_feedback_retry(
                queue_id,
                status=status,
                error=error,
                status_code=None,
                next_attempt_at=utc_now(),
            )
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "endpoint": endpoint,
            "optimization_unit_id": unit_id,
            "queue_id": queue_id,
            "status": status,
            "reason": "invalid-payload",
            "attempts": attempts,
            "error": error,
        }

    if endpoint == POLICY_EVENTS_PATH:
        meta = await _send_policy_event_payload(_sanitize_features(payload))
    else:
        meta = await _send_outcome_payload(endpoint=endpoint, payload=_sanitize_features(payload), unit_id=int(unit_id))
    if meta.get("status") == "sent":
        if hasattr(store_obj, "mark_managed_outcome_feedback_sent"):
            store_obj.mark_managed_outcome_feedback_sent(queue_id, status_code=meta.get("status_code"))
        meta.update({
            "queue_id": queue_id,
            "attempts": attempts,
        })
        return meta

    status = _queue_error_status(meta, attempts)
    retry_delay = 0.0 if status == "dropped-after-limit" else outcome_feedback_queue_retry_delay_seconds()
    reason = "attempt-limit-reached" if status == "dropped-after-limit" else meta.get("reason") or "request-failed"
    error = meta.get("error")
    if hasattr(store_obj, "mark_managed_outcome_feedback_retry"):
        store_obj.mark_managed_outcome_feedback_retry(
            queue_id,
            status=status,
            error=error,
            status_code=meta.get("status_code"),
            next_attempt_at=_future_iso(retry_delay),
        )
    meta.update({
        "queue_id": queue_id,
        "status": status,
        "reason": reason,
        "attempts": attempts,
    })
    return meta


async def flush_queued_outcome_feedback(
    store_obj: Any,
    *,
    limit: int = 5,
    source_surface: str | None = None,
) -> list[dict[str, Any]]:
    if not recommendations_enabled():
        return []
    if not hasattr(store_obj, "claim_due_managed_outcome_feedback"):
        return []
    results: list[dict[str, Any]] = []
    for row in store_obj.claim_due_managed_outcome_feedback(
        limit=max(1, limit),
        now=utc_now(),
        source_surface=source_surface,
    ):
        results.append(await _flush_claimed_outcome_feedback(store_obj, row))
    return results


async def queue_outcome_feedback(
    store_obj: Any,
    recommendation_meta: dict[str, Any],
    outcome_features: dict[str, Any],
    *,
    source_surface: str | None = None,
) -> dict[str, Any]:
    if not recommendations_enabled():
        return {
            **disabled_outcome_feedback_meta(),
            "status": "disabled",
        }
    unit_id, endpoint = _outcome_target(recommendation_meta)
    if unit_id is None or endpoint is None:
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "endpoint": OUTCOME_PATH_TEMPLATE,
            "status": "skipped",
            "reason": "missing-optimization-unit-id",
        }
    if not hasattr(store_obj, "enqueue_managed_outcome_feedback"):
        return await send_outcome_feedback(recommendation_meta, outcome_features)

    payload = _sanitize_features(outcome_features)
    surface = (
        source_surface
        or str(payload.get("source_surface") or "")
        or str(recommendation_meta.get("source_surface") or "")
        or CODEX_APP_SOURCE_SURFACE
    )
    queue_id = str(uuid.uuid4())
    row = {
        "id": queue_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "source_surface": surface,
        "endpoint": endpoint,
        "optimization_unit_id": unit_id,
        "payload_json": stable_json(payload),
        "status": "queued",
        "attempts": 0,
        "next_attempt_at": utc_now(),
    }
    store_obj.enqueue_managed_outcome_feedback(**row)
    claimed = (
        store_obj.claim_managed_outcome_feedback(queue_id, now=utc_now())
        if hasattr(store_obj, "claim_managed_outcome_feedback")
        else None
    )
    if not claimed:
        stored = store_obj.get_managed_outcome_feedback(queue_id) if hasattr(store_obj, "get_managed_outcome_feedback") else row
        return _queued_meta(stored or row)

    meta = await _flush_claimed_outcome_feedback(store_obj, claimed)
    return meta


async def queue_policy_event_feedback(
    store_obj: Any,
    event_payload: dict[str, Any],
    *,
    source_surface: str = ROLLOUT_ACTION_LIFECYCLE_SOURCE_SURFACE,
) -> dict[str, Any]:
    if not recommendations_enabled():
        return {
            **disabled_outcome_feedback_meta(),
            "endpoint": POLICY_EVENTS_PATH,
            "status": "disabled",
        }
    if not hasattr(store_obj, "enqueue_managed_outcome_feedback"):
        return await _send_policy_event_payload(event_payload)

    payload = _sanitize_features(event_payload)
    queue_id = str(uuid.uuid4())
    now = utc_now()
    row = {
        "id": queue_id,
        "created_at": now,
        "updated_at": now,
        "source_surface": source_surface,
        "endpoint": POLICY_EVENTS_PATH,
        "optimization_unit_id": 0,
        "payload_json": stable_json(payload),
        "status": "queued",
        "attempts": 0,
        "next_attempt_at": now,
    }
    store_obj.enqueue_managed_outcome_feedback(**row)
    claimed = (
        store_obj.claim_managed_outcome_feedback(queue_id, now=utc_now())
        if hasattr(store_obj, "claim_managed_outcome_feedback")
        else None
    )
    if not claimed:
        stored = store_obj.get_managed_outcome_feedback(queue_id) if hasattr(store_obj, "get_managed_outcome_feedback") else row
        return _queued_meta(stored or row)
    return await _flush_claimed_outcome_feedback(store_obj, claimed)


async def queue_codex_outcome_feedback(
    store_obj: Any,
    recommendation_meta: dict[str, Any],
    outcome_features: dict[str, Any],
) -> dict[str, Any]:
    return await queue_outcome_feedback(
        store_obj,
        recommendation_meta,
        outcome_features,
        source_surface=CODEX_APP_SOURCE_SURFACE,
    )
