from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from tokenclaw.http_client import async_client

from tokenclaw import __version__ as TOKENCLAW_VERSION
from tokenclaw.action_executor import ActionExecutor
from tokenclaw.codex_turn_policy import CODEX_APP_SOURCE_SURFACE
from tokenclaw.client_contract import (
    ContractClient,
    ClientContractRequest,
    fetch_or_get_client_contract,
    filter_payload_by_client_contract,
)
from tokenclaw.env import env
from tokenclaw.managed_mode import managed_product_mode
from tokenclaw.managed_egress import (
    LIFECYCLE_METADATA_COMMAND_SCHEMAS,
    RAW_FEATURE_KEYS,
    ManagedEgressBlocked,
    assert_managed_egress_safe,
    managed_egress_blocked_meta,
)
from tokenclaw.managed_measurements import execute_preflight_measurement_plan
from tokenclaw.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from tokenclaw.prompt_features import PROMPT_DIFFICULTY_FEATURE_SCHEMA
from tokenclaw.quality import derive_codex_turn_quality_signals, derive_provider_quality_signals
from tokenclaw.store import stable_json, utc_now
from tokenclaw.routing_experiments import _public_label, routing_pathway_policy_decision
from tokenclaw.terminal_features import TERMINAL_LOG_FEATURE_SCHEMA


RECOMMENDATION_PATH = "/v1/recommendation"
POLICY_DECISION_PATH = "/v1/policy-decision"
OUTCOME_PATH_TEMPLATE = "/v1/optimization-units/{unit_id}/outcome"
POLICY_EVENTS_PATH = "/v1/policy-events"
PROMOTION_BLOCKER_ACTION_OUTCOME_ROLLUPS_PATH = "/v1/promotion-blocker-action-outcome-rollups"
FEATURE_SCHEMA_VERSION = "tokenclaw.optimization_unit_features.v1"
REQUEST_FACTS_SCHEMA = "tokenclaw.request_facts.v1"
REQUEST_FACTS_ENVELOPE_SCHEMA = "tokenclaw.request_facts_envelope.v1"
POLICY_DECISION_PREFLIGHT_SCHEMA = "tokenclaw.policy_decision_preflight.v1"
POLICY_DECISION_SCHEMA = "tokenclaw.policy_decision.v1"
POLICY_DECISION_RESPONSE_SCHEMAS = {
    POLICY_DECISION_SCHEMA,
    "agentflow.policy_decision.v1",
}
MANAGED_API_KEY_ENV = "TOKENCLAW_MANAGED_API_KEY"
RECOMMENDATION_ENABLED_ENV = "TOKENCLAW_RECOMMENDATION_ENABLED"
RECOMMENDATIONS_ENABLED_ENV = "TOKENCLAW_RECOMMENDATIONS_ENABLED"
RECOMMENDATION_SERVER_URL_ENV = "TOKENCLAW_RECOMMENDATION_SERVER_URL"
RECOMMENDATION_TIMEOUT_ENV = "TOKENCLAW_RECOMMENDATION_TIMEOUT_SECONDS"
RECOMMENDATION_FAILURE_MODE_ENV = "TOKENCLAW_RECOMMENDATION_FAILURE_MODE"
POLICY_DECISION_ENABLED_ENV = "TOKENCLAW_POLICY_DECISION_ENABLED"
POLICY_DECISIONS_ENABLED_ENV = "TOKENCLAW_POLICY_DECISIONS_ENABLED"
POLICY_DECISION_MIN_CONFIDENCE_ENV = "TOKENCLAW_POLICY_DECISION_MIN_CONFIDENCE"
POLICY_DECISION_CANARY_FRACTION_ENV = "TOKENCLAW_POLICY_DECISION_CANARY_FRACTION"
POLICY_DECISION_CANARY_SALT_ENV = "TOKENCLAW_POLICY_DECISION_CANARY_SALT"
DEFAULT_RECOMMENDATION_SERVER_URL = ""
ROLLOUT_ACTION_LIFECYCLE_SOURCE_SURFACE = "rollout_action_lifecycle"
OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE = "old_context_summary_lifecycle"
OLD_CONTEXT_SUMMARY_OUTCOME_SOURCE_SURFACE = "old_context_summary_outcome"
PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE = "phase_routing_lifecycle"
PHASE_ROUTING_OUTCOME_SOURCE_SURFACE = "phase_routing_outcome"
OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE = "optimization_promotion_lifecycle"
CODEX_APP_CANARY_LIFECYCLE_SOURCE_SURFACE = "codex_app_canary_lifecycle"
CACHE_REPLAY_LIFECYCLE_SOURCE_SURFACE = "cache_replay_lifecycle"
TERMINAL_OUTPUT_COMPACTION_LIFECYCLE_SOURCE_SURFACE = "terminal_output_compaction_lifecycle"

TOKEN_CHARS = 4
POLICY_DECISION_NON_FEATURE_INPUT_KEYS = {
    "body",
    "cache_key",
    "cache_keys",
    "chat_completion",
    "chat_completions",
    "choice",
    "choices",
    "completion",
    "command",
    "commands",
    "content",
    "developer",
    "file_content",
    "file_contents",
    "function_call",
    "function_calls",
    "input",
    "message",
    "messages",
    "normalized_pattern",
    "old_context",
    "old_context_summary",
    "old_context_summary_text",
    "old_contexts",
    "openai_request",
    "openai_requests",
    "openai_response",
    "openai_responses",
    "output",
    "path",
    "pattern_body",
    "pattern_text",
    "policy_content",
    "policy_contents",
    "policy_file_content",
    "policy_file_contents",
    "prompt",
    "provider_bodies",
    "provider_body",
    "provider_request",
    "provider_response",
    "raw_old_context",
    "raw_payload",
    "raw_payloads",
    "raw_policy",
    "raw_policy_file",
    "raw_policy_yaml",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "raw_session_id",
    "raw_session_ids",
    "request",
    "request_body",
    "request_fingerprint",
    "request_fingerprints",
    "request_id",
    "request_ids",
    "response",
    "response_body",
    "rules_yaml",
    "session_id",
    "session_ids",
    "summary_content",
    "summary_prompt",
    "summary_prompts",
    "summary_request",
    "summary_text",
    "system",
    "system_prompt",
    "text",
    "tool_payload",
    "tool_payloads",
    "transcript",
    "transcripts",
}
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
OLD_CONTEXT_SUMMARY_FAIL_CLOSED_REASONS = {
    "fallback-not-configured",
    "summary-fetch-error",
    "summary-empty",
    "summary-error",
    "summary-cost-too-high",
    "summary-apply-error",
    "tool-protocol-reconstruction-mismatch",
}


def _hash_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    text_l = text.lower()
    unsafe_terms = {
        "account",
        "apikey",
        "api_key",
        "authorization",
        "body",
        "cache_key",
        "content",
        "file",
        "message",
        "path",
        "payload",
        "prompt",
        "request",
        "response",
        "secret",
        "session",
        "summary_text",
        "tenant",
        "tool",
        "transcript",
    }
    if (
        len(text) > 128
        or any(char.isspace() for char in text)
        or any(char in text for char in ("/", "\\", "{", "}", "[", "]", "\"", "'"))
        or any(term in text_l for term in unsafe_terms)
    ):
        return _hash_identifier(text)
    return text


def _metadata_only_privacy_summary() -> dict[str, Any]:
    return {
        "telemetry_profile": "metadata-only",
        "raw_body_storage": False,
        "metadata_only": True,
        "aggregate_only": False,
        "raw_payload_included": False,
    }


def _env_enabled(name: str, default: str = "0") -> bool:
    return env(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _compact_grouping_identifiers(values: dict[str, str | None]) -> dict[str, str]:
    return {
        key: hashed
        for key, value in values.items()
        if (hashed := _hash_identifier(value)) is not None
    }


def _provider_family(value: str | None) -> str:
    provider = (value or "").strip().lower()
    if provider in {"anthropic", "openai"}:
        return provider
    return "unknown"


def _endpoint_label(path: str | None) -> str:
    cleaned = str(path or "").strip().strip("/")
    if not cleaned:
        return "unknown"
    parts = [part for part in cleaned.split("/") if part and not part.startswith("v")]
    if not parts:
        return "unknown"
    return "_".join(part.replace("-", "_") for part in parts[-2:])[:96] or "unknown"


def _metadata_text_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_metadata_text_char_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_metadata_text_char_count(item) for item in value.values())
    return 0


def _request_uses_thinking(body: dict[str, Any]) -> bool:
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        return str(thinking.get("type") or "").strip().lower() == "enabled"
    return False


def _message_item_counts(body: dict[str, Any]) -> dict[str, int]:
    messages = body.get("messages")
    input_items = body.get("input")
    return {
        "message_item_count": len(messages) if isinstance(messages, list) else 0,
        "input_item_count": len(input_items) if isinstance(input_items, list) else (1 if input_items is not None else 0),
    }


def _tool_fact_counts(body: dict[str, Any]) -> dict[str, Any]:
    tools = body.get("tools")
    functions = body.get("functions")
    tool_count = (len(tools) if isinstance(tools, list) else 0) + (len(functions) if isinstance(functions, list) else 0)
    has_tool_context = tool_count > 0 or bool(body.get("tool_choice") or body.get("function_call"))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    input_items = body.get("input") if isinstance(body.get("input"), list) else []
    for item in input_items:
        if isinstance(item, dict) and str(item.get("type") or "").lower() in {
            "function_call",
            "function_call_output",
            "tool_call",
            "tool_result",
            "computer_call",
            "file_search_call",
            "web_search_call",
        }:
            has_tool_context = True
            break
    if not has_tool_context:
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict) and str(block.get("type") or "").lower() == "tool_result"
                for block in content
            ):
                has_tool_context = True
                break
            if isinstance(message.get("tool_calls"), list) or message.get("tool_call_id"):
                has_tool_context = True
                break
    return {
        "has_tools": bool(has_tool_context),
        "tool_count": tool_count,
        "tool_context_present": bool(has_tool_context),
    }


def build_request_facts_envelope(
    *,
    provider: str | None = None,
    path: str | None = None,
    body: dict[str, Any] | None = None,
    requested_model: str | None = None,
    stream: bool | None = None,
    input_tokens_est: int | None = None,
    session_id: str | None = None,
    thread_id: str | None = None,
    request_id: str | None = None,
    local_executor_capabilities: dict[str, Any] | None = None,
    category: str | None = None,
    workflow_phase: str | None = None,
) -> dict[str, Any]:
    """Build a thin managed-optimizer envelope from directly known request facts."""

    raw_body = body if isinstance(body, dict) else {}
    model = requested_model if requested_model is not None else raw_body.get("model")
    model_text = str(model).strip() if model not in (None, "") else None
    text_chars = _metadata_text_char_count(raw_body)
    token_estimate = input_tokens_est
    if token_estimate is None and text_chars > 0:
        token_estimate = max(1, text_chars // TOKEN_CHARS)
    counts = _message_item_counts(raw_body)
    tool_facts = _tool_fact_counts(raw_body)
    capabilities = dict(local_executor_capabilities or {})
    capabilities.setdefault("schema", "tokenclaw.local_executor_capabilities.v1")
    capabilities.setdefault("supported_local_action_families", ["routing", "crunch", "cache", "old_context_summarization"])
    capabilities.setdefault("locally_executed", True)
    capabilities.setdefault("server_content_processing", False)
    facts = {
        "schema": REQUEST_FACTS_SCHEMA,
        "provider_family": _provider_family(provider),
        "endpoint": _endpoint_label(path),
        "endpoint_known": bool(path),
        "requested_model": model_text,
        "requested_model_present": bool(model_text),
        "stream": bool(raw_body.get("stream") if stream is None else stream),
        "text_chars": text_chars,
        "input_tokens_est": token_estimate,
        "text_bucket": _text_bucket(text_chars),
        "input_token_bucket": _token_bucket(token_estimate),
        "message_item_count": counts["message_item_count"],
        "input_item_count": counts["input_item_count"],
        "has_tools": tool_facts["has_tools"],
        "tool_count": tool_facts["tool_count"],
        "tool_context_present": tool_facts["tool_context_present"],
        "response_format_present": bool(raw_body.get("response_format")),
        # The routing predictor's rules key on category/workflow_phase (and thinking
        # state), so the decision-path facts must carry them or the server-stored
        # unit degrades to "unknown" and no trained rule can ever match live traffic.
        # These are the proxy's own short classifier labels, not content.
        "category": _public_label(category, fallback="") or None,
        "workflow_phase": _public_label(workflow_phase, fallback="") or None,
        "uses_thinking": _request_uses_thinking(raw_body),
        "local_executor_capabilities": capabilities,
        "grouping_identifiers": _compact_grouping_identifiers({
            "session_id_hash": session_id,
            "thread_id_hash": thread_id,
            "request_id_hash": request_id,
        }),
        "raw_payload_included": False,
        "raw_body_storage": False,
    }
    envelope = {
        "schema": REQUEST_FACTS_ENVELOPE_SCHEMA,
        "request_facts_schema": REQUEST_FACTS_SCHEMA,
        "feature_only": True,
        "locally_executed": True,
        "server_content_processing": False,
        "provider_forwarding": False,
        "request_facts": facts,
        "privacy_summary": _metadata_only_privacy_summary(),
        "raw_payload_included": False,
    }
    envelope = _sanitize_features(envelope)
    assert_managed_egress_safe(envelope)
    return envelope


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


def _status_code_bucket(value: Any) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if code < 200:
        return "lt_2xx"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


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
    for family in ("haiku", "sonnet", "opus", "fable", "mythos", "codex", "gpt-5", "gpt-4", "gpt-3"):
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
    local_pattern_modules = crunch_meta.get("pattern_modules") if isinstance(crunch_meta, dict) else None
    local_module_features = (
        local_pattern_modules.get("server_features")
        if isinstance(local_pattern_modules, dict) and isinstance(local_pattern_modules.get("server_features"), dict)
        else {}
    )
    raw_local_module_entries = local_module_features.get("features") if isinstance(local_module_features, dict) else []
    local_module_entries = raw_local_module_entries if isinstance(raw_local_module_entries, list) else []
    local_module_families = sorted({
        str(item.get("family"))
        for item in local_module_entries
        if isinstance(item, dict) and item.get("family")
    })
    cacheability_features = _cacheability_pattern_features(local_module_entries)
    pattern_types = sorted({item["type"] for item in pattern_summaries} | set(local_module_families))
    descriptor: dict[str, Any] = {
        "schema": "tokenclaw.normalized_pattern_descriptor.v1",
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
        "local_pattern_module_families": local_module_families,
        "local_pattern_module_count": len(local_module_families),
    }
    base_hash = _pattern_hash({**descriptor, "pattern_family": "general"})
    crunch_hash = _pattern_hash({**descriptor, "pattern_family": "crunch"})
    cache_hash = _pattern_hash({**descriptor, "pattern_family": "cache"})
    hashes = sorted({base_hash, crunch_hash, cache_hash})
    result = {
        "schema": "tokenclaw.pattern_features.v1",
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
        "local_pattern_module_families": local_module_families,
        "local_pattern_module_count": len(local_module_families),
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
    if cacheability_features:
        result["cacheability"] = cacheability_features
        result["cacheability_bucket"] = cacheability_features.get("cacheability_bucket")
        result["static_information_hint"] = bool(cacheability_features.get("static_information_hint"))
        result["time_sensitive_hint"] = bool(cacheability_features.get("time_sensitive_hint"))
        result["user_specific_hint"] = bool(cacheability_features.get("user_specific_hint"))
        result["exact_cache_candidate_hint"] = bool(cacheability_features.get("exact_cache_candidate_hint"))
    return result


def _cacheability_pattern_features(local_module_entries: list[Any]) -> dict[str, Any]:
    for item in local_module_entries:
        if not isinstance(item, dict) or item.get("family") != "cacheability":
            continue
        features = item.get("features")
        if not isinstance(features, dict):
            continue
        return {
            key: features.get(key)
            for key in (
                "schema",
                "module_family",
                "module_version",
                "cacheability_bucket",
                "deterministic_answer_likelihood_bucket",
                "static_information_hint",
                "time_sensitive_hint",
                "user_specific_hint",
                "exact_lookup_requested_hint",
                "aggregation_requested_hint",
                "reconciliation_requested_hint",
                "provider_tool_context_hint",
                "exact_cache_candidate_hint",
                "cache_preserved_by_default_reason",
            )
            if key in features
        }
    return {}


def pattern_feature_diagnostics(unit: dict[str, Any]) -> dict[str, Any]:
    input_features = unit.get("input_features") if isinstance(unit, dict) else None
    pattern_features = input_features.get("pattern_features") if isinstance(input_features, dict) else None
    if not isinstance(pattern_features, dict):
        pattern_features = unit.get("pattern_features") if isinstance(unit, dict) else None
    if not isinstance(pattern_features, dict):
        return {
            "schema": "tokenclaw.managed_pattern_feature_diagnostics.v1",
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
    diagnostics = {
        "schema": "tokenclaw.managed_pattern_feature_diagnostics.v1",
        "present": bool(hashes),
        "pattern_hash_count": len(sorted(set(hashes))),
        "pattern_hashes": sorted(set(hashes)),
        "hash_basis": pattern_features.get("hash_basis"),
        "text_bucket": pattern_features.get("text_bucket"),
        "token_bucket": pattern_features.get("token_bucket"),
        "workflow_phase": pattern_features.get("workflow_phase"),
        "category": pattern_features.get("category"),
        "source_surface": pattern_features.get("source_surface"),
        "endpoint": unit.get("endpoint") or input_features.get("endpoint"),
        "app_family": pattern_features.get("app_family"),
        "requested_model": unit.get("requested_model"),
        "candidate_target_model": unit.get("candidate_target_model"),
        "pattern_hash": pattern_features.get("pattern_hash"),
        "normalized_pattern_hash": pattern_features.get("normalized_pattern_hash"),
        "crunch_pattern_hash": pattern_features.get("crunch_pattern_hash"),
        "cache_pattern_hash": pattern_features.get("cache_pattern_hash"),
        "replayability_level": pattern_features.get("replayability_level"),
        "rollout_unit_hash": pattern_features.get("rollout_unit_hash"),
        "rollout_unit_hash_included": bool(pattern_features.get("rollout_unit_hash")),
        "raw_request_fingerprint_included": False,
        "has_tools": pattern_features.get("has_tools"),
        "stream": pattern_features.get("stream"),
        "pattern_types": pattern_features.get("pattern_types") or [],
        "local_pattern_module_families": pattern_features.get("local_pattern_module_families") or [],
        "local_pattern_module_count": pattern_features.get("local_pattern_module_count") or 0,
        "raw_pattern_strings_included": False,
    }
    cacheability = pattern_features.get("cacheability")
    if isinstance(cacheability, dict):
        diagnostics["cacheability"] = cacheability
        diagnostics["cacheability_bucket"] = cacheability.get("cacheability_bucket")
        diagnostics["static_information_hint"] = bool(cacheability.get("static_information_hint"))
        diagnostics["time_sensitive_hint"] = bool(cacheability.get("time_sensitive_hint"))
        diagnostics["user_specific_hint"] = bool(cacheability.get("user_specific_hint"))
        diagnostics["exact_cache_candidate_hint"] = bool(cacheability.get("exact_cache_candidate_hint"))
    return diagnostics


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def recommendations_enabled() -> bool:
    product_mode = managed_product_mode()
    if product_mode.local_rules_only or product_mode.mode == "local_only":
        return False
    raw = env(RECOMMENDATIONS_ENABLED_ENV)
    if raw is not None:
        return _as_bool(raw, False)
    legacy = env(RECOMMENDATION_ENABLED_ENV)
    if legacy is not None:
        return _as_bool(legacy, False)
    return product_mode.server_calls_enabled


def recommendation_server_url() -> str:
    raw = env(RECOMMENDATION_SERVER_URL_ENV)
    if raw is None:
        raw = DEFAULT_RECOMMENDATION_SERVER_URL
    return raw.strip().rstrip("/")


def recommendation_server_configured() -> bool:
    return bool(recommendation_server_url())


def recommendation_timeout_seconds() -> float:
    try:
        return max(0.05, float(env(RECOMMENDATION_TIMEOUT_ENV, "1.5")))
    except ValueError:
        return 1.5


def recommendation_failure_mode() -> str:
    mode = env(RECOMMENDATION_FAILURE_MODE_ENV, "fallback-local").strip().lower()
    return mode if mode in {"fallback-local"} else "fallback-local"


def managed_loopback_auth_allowed() -> bool:
    parsed = urlparse(recommendation_server_url())
    if parsed.scheme != "http":
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def managed_auth_configured() -> bool:
    return bool(env(MANAGED_API_KEY_ENV)) or managed_loopback_auth_allowed()


def managed_auth_source() -> str | None:
    api_key = env(MANAGED_API_KEY_ENV)
    if api_key:
        return MANAGED_API_KEY_ENV
    if managed_loopback_auth_allowed():
        return "loopback-unauthenticated-dev"
    return None


def _managed_headers() -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-tokenclaw-local-fallback": "local-policy",
    }
    api_key = env(MANAGED_API_KEY_ENV)
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def outcome_feedback_queue_max_attempts() -> int:
    try:
        return max(1, int(env("TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS", "3")))
    except ValueError:
        return 3


def outcome_feedback_queue_retry_delay_seconds() -> float:
    try:
        return max(0.0, float(env("TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS", "60")))
    except ValueError:
        return 60.0


def outcome_feedback_queue_max_retry_delay_seconds() -> float:
    base = outcome_feedback_queue_retry_delay_seconds()
    try:
        return max(base, float(env("TOKENCLAW_OUTCOME_FEEDBACK_QUEUE_MAX_RETRY_DELAY_SECONDS", "3600")))
    except ValueError:
        return max(base, 3600.0)


def outcome_feedback_queue_retry_delay_for_attempt(attempts: int) -> float:
    base = outcome_feedback_queue_retry_delay_seconds()
    if base <= 0:
        return 0.0
    exponent = max(0, int(attempts or 1) - 1)
    return min(outcome_feedback_queue_max_retry_delay_seconds(), base * (2 ** exponent))


def _future_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))).isoformat()


def _response_retry_after_seconds(response: httpx.Response) -> float | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


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
    prompt_difficulty_features = routing_meta.get("prompt_difficulty_features")
    if not (
        isinstance(prompt_difficulty_features, dict)
        and prompt_difficulty_features.get("schema") == PROMPT_DIFFICULTY_FEATURE_SCHEMA
    ):
        prompt_difficulty_features = None
    local_pattern_modules = crunch_meta.get("pattern_modules") if isinstance(crunch_meta, dict) else None
    local_pattern_module_features = (
        local_pattern_modules.get("server_features")
        if isinstance(local_pattern_modules, dict) and isinstance(local_pattern_modules.get("server_features"), dict)
        else None
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
            "local_pattern_module_features": local_pattern_module_features,
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
        "outcome_features": {
            "routing_outcome_label": str(routing_meta.get("routing_outcome_label") or "unknown"),
        },
        "grouping_identifiers": _compact_grouping_identifiers({
            "session_id_hash": session_id,
        }),
        "privacy_summary": _metadata_only_privacy_summary(),
        "replayability_level": replayability_level,
        "pattern_features": pattern_features,
    }
    if prompt_difficulty_features is not None:
        unit["input_features"]["prompt_difficulty_features"] = prompt_difficulty_features
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
    terminal_log_features: dict[str, Any] | None = None,
    prompt_difficulty_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_state, model_field, requested_model, routed_model = _codex_model_state(routing_meta)
    candidate_target_model = (
        routing_meta.get("candidate_target_model")
        or routing_meta.get("shadow_model")
        or routing_meta.get("managed_route_candidate_model")
        or routed_model
    )
    if not isinstance(candidate_target_model, str) or not candidate_target_model:
        candidate_target_model = routed_model
    input_tokens_est = max(1, int(input_text_chars / TOKEN_CHARS)) if input_text_chars else 0
    workflow_phase_value = workflow_phase or routing_meta.get("workflow_phase") or "unknown"
    replayability_level = str(cache_meta.get("replayability_level") or "features_only")
    pattern_features = _pattern_features(
        source_surface=CODEX_APP_SOURCE_SURFACE,
        granularity="agent_turn",
        app_family="codex",
        requested_model=requested_model,
        candidate_target_model=candidate_target_model,
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
    terminal_features = terminal_log_features if isinstance(terminal_log_features, dict) else routing_meta.get("terminal_log_features")
    if not isinstance(terminal_features, dict) or terminal_features.get("schema") != TERMINAL_LOG_FEATURE_SCHEMA:
        terminal_features = None
    difficulty_features = (
        prompt_difficulty_features
        if isinstance(prompt_difficulty_features, dict)
        else routing_meta.get("prompt_difficulty_features")
    )
    if not isinstance(difficulty_features, dict) or difficulty_features.get("schema") != PROMPT_DIFFICULTY_FEATURE_SCHEMA:
        difficulty_features = None
    unit = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "granularity": "agent_turn",
        "app_family": "codex",
        "requested_model": requested_model,
        "candidate_target_model": candidate_target_model,
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
            "candidate_target_model": candidate_target_model,
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
    if terminal_features is not None:
        unit["input_features"]["terminal_log_features"] = terminal_features
    if difficulty_features is not None:
        unit["input_features"]["prompt_difficulty_features"] = difficulty_features
    return _sanitize_features(unit)


def _base_meta() -> dict[str, Any]:
    product_mode = managed_product_mode()
    return {
        "enabled": recommendations_enabled(),
        "server_url": recommendation_server_url(),
        "endpoint": RECOMMENDATION_PATH,
        "timeout_seconds": recommendation_timeout_seconds(),
        "failure_mode": recommendation_failure_mode(),
        "auth_configured": managed_auth_configured(),
        "auth_source": managed_auth_source(),
        "loopback_unauthenticated_allowed": managed_loopback_auth_allowed(),
        "api_key_value_included": False,
        "policy_source": "local-default",
        "product_mode": product_mode.public_meta(),
    }


def _provider_from_source_surface(source_surface: str | None) -> str:
    surface = (source_surface or "").strip().lower()
    if surface.startswith("openai") or surface == CODEX_APP_SOURCE_SURFACE:
        return "openai"
    if surface.startswith("anthropic"):
        return "anthropic"
    return "unknown"


def _contract_scope_from_payload(payload: dict[str, Any]) -> ClientContractRequest:
    source_surface = str(payload.get("source_surface") or "unknown")
    app_family = str(payload.get("app_family") or "unknown")
    provider = str(payload.get("provider") or payload.get("provider_family") or "")
    if not provider or provider == "unknown":
        input_features = payload.get("input_features") if isinstance(payload.get("input_features"), dict) else {}
        request_facts = payload.get("request_facts") if isinstance(payload.get("request_facts"), dict) else {}
        provider = str(
            input_features.get("provider_family")
            or input_features.get("provider")
            or request_facts.get("provider_family")
            or _provider_from_source_surface(source_surface)
        )
    return ClientContractRequest(
        provider=provider,
        source_surface=source_surface,
        app_family=app_family,
        client_version=TOKENCLAW_VERSION,
    )


async def _client_contract_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = _contract_scope_from_payload(payload)
    client = ContractClient(
        base_url=recommendation_server_url(),
        headers=_managed_headers(),
        timeout_seconds=recommendation_timeout_seconds(),
        async_client_factory=httpx.AsyncClient,
    )
    return await fetch_or_get_client_contract(
        request,
        enabled=recommendations_enabled(),
        server_url=recommendation_server_url(),
        auth_configured=managed_auth_configured(),
        auth_source=managed_auth_source(),
        client=client,
    )


def policy_decisions_enabled() -> bool:
    product_mode = managed_product_mode()
    if not product_mode.server_calls_enabled:
        return False
    if env(POLICY_DECISIONS_ENABLED_ENV) is not None:
        return _as_bool(env(POLICY_DECISIONS_ENABLED_ENV), False)
    if env(POLICY_DECISION_ENABLED_ENV) is not None:
        return _env_enabled(POLICY_DECISION_ENABLED_ENV)
    return product_mode.server_calls_enabled if product_mode.configured else False


def policy_decision_min_confidence() -> float:
    raw = env(POLICY_DECISION_MIN_CONFIDENCE_ENV, "0.75")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.75


def policy_decision_canary_fraction() -> float:
    raw = env(POLICY_DECISION_CANARY_FRACTION_ENV, "0.0")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def policy_decision_canary_salt() -> str:
    return env(POLICY_DECISION_CANARY_SALT_ENV, "tokenclaw-policy-decision-canary-v1")


def _policy_decision_base_meta() -> dict[str, Any]:
    meta = _base_meta()
    meta.update({
        "enabled": recommendations_enabled() and policy_decisions_enabled(),
        "endpoint": POLICY_DECISION_PATH,
        "policy_decision_enabled": policy_decisions_enabled(),
    })
    return meta


def disabled_recommendation_meta() -> dict[str, Any]:
    meta = _base_meta()
    meta.update({
        "status": "skipped",
        "reason": "disabled",
        "fallback": "local-policy",
        "applied": False,
    })
    return meta


def _policy_decision_preflight_payload(unit: dict[str, Any]) -> dict[str, Any]:
    raw_input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
    path_hint = raw_input_features.get("path")
    input_features = _policy_decision_input_features(raw_input_features)
    requested_actions = input_features.get("requested_local_actions")
    if not isinstance(requested_actions, list):
        requested_actions = ["routing"]
        if isinstance(unit.get("crunch"), dict):
            requested_actions.append("crunch")
        if isinstance(unit.get("cache"), dict):
            requested_actions.append("cache")
    input_features["requested_local_actions"] = sorted({str(item) for item in requested_actions if item})
    input_features.setdefault("use_routing_predictor", True)
    endpoint_hint = unit.get("endpoint") or unit.get("provider_endpoint") or path_hint
    if isinstance(endpoint_hint, str) and endpoint_hint:
        endpoint_hint = endpoint_hint.strip("/").replace("/", "_")
    input_features.setdefault("api_endpoint", endpoint_hint)

    payload = {
        "schema": POLICY_DECISION_PREFLIGHT_SCHEMA,
        "feature_schema_version": unit.get("feature_schema_version") or FEATURE_SCHEMA_VERSION,
        "source_surface": unit.get("source_surface") or "anthropic_messages",
        "granularity": unit.get("granularity") or "provider_request",
        "app_family": unit.get("app_family") or "unknown",
        "requested_model": unit.get("requested_model"),
        "candidate_target_model": unit.get("candidate_target_model"),
        "input_features": input_features,
        "tool_features": dict(unit.get("tool_features") or {}),
        "outcome_features": dict(unit.get("outcome_features") or {}),
        "grouping_identifiers": dict(unit.get("grouping_identifiers") or {}),
        "replayability_level": "features_only",
    }
    return _sanitize_features(payload)


def _client_contract_preflight_meta(contract_meta: dict[str, Any] | None) -> dict[str, Any]:
    contract = contract_meta.get("contract") if isinstance(contract_meta, dict) else None
    if not isinstance(contract, dict):
        contract = {}
    measurement_plan = contract.get("measurement_plan") if isinstance(contract.get("measurement_plan"), dict) else {}
    preflight_paths = measurement_plan.get("preflight") if isinstance(measurement_plan.get("preflight"), list) else []
    contract_hash = "sha256:" + hashlib.sha256(stable_json({
        "schema": contract.get("schema"),
        "contract_id": contract.get("contract_id"),
        "expires_at": contract.get("expires_at"),
        "measurement_plan": measurement_plan,
        "allowed_action_families": contract.get("allowed_action_families"),
    }).encode("utf-8")).hexdigest() if contract else None
    meta = {
        "schema": "tokenclaw.policy_decision_client_contract_ref.v1",
        "contract_id": contract.get("contract_id") or (contract_meta or {}).get("contract_id"),
        "contract_hash": contract_hash,
        "contract_status": (contract_meta or {}).get("status"),
        "contract_reason": (contract_meta or {}).get("reason"),
        "contract_cache_status": (contract_meta or {}).get("cache_status"),
        "expires_at": contract.get("expires_at") or (contract_meta or {}).get("expires_at"),
        "measurement_plan_id": measurement_plan.get("plan_id") or measurement_plan.get("id"),
        "measurement_plan_version": measurement_plan.get("version"),
        "measurement_path_count": len(preflight_paths),
        "allowed_action_families": contract.get("allowed_action_families") or [],
        "metadata_only": True,
        "raw_payload_included": False,
    }
    return _sanitize_features({key: value for key, value in meta.items() if value is not None})


def _attach_client_contract_to_preflight(
    payload: dict[str, Any],
    contract_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["client_contract"] = _client_contract_preflight_meta(contract_meta)
    enriched["local_client_version"] = TOKENCLAW_VERSION
    return _sanitize_features(enriched)


def _policy_decision_contract_skip_reason(contract_meta: dict[str, Any] | None) -> str | None:
    if isinstance(contract_meta, dict) and contract_meta.get("active") is True:
        return None
    if not isinstance(contract_meta, dict):
        return "missing-client-contract"
    schema_error = str(contract_meta.get("schema_error") or "")
    if schema_error == "expired":
        return "expired-contract"
    status = str(contract_meta.get("status") or "")
    reason = str(contract_meta.get("reason") or "")
    if status == "error" or reason in {"timeout", "unreachable", "server-error", "fetch-error"}:
        return "server-unavailable"
    if status == "invalid":
        return "invalid-client-contract"
    return reason or "no-active-client-contract"


def _request_facts_from_envelope(envelope: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(envelope, dict):
        return None
    if envelope.get("schema") == REQUEST_FACTS_SCHEMA:
        return envelope
    facts = envelope.get("request_facts")
    if isinstance(facts, dict) and facts.get("schema") == REQUEST_FACTS_SCHEMA:
        return facts
    return None


def _policy_decision_payload_from_request_facts(envelope: dict[str, Any]) -> dict[str, Any]:
    facts = _request_facts_from_envelope(envelope)
    if facts is None:
        raise ValueError("request facts envelope is missing tokenclaw.request_facts.v1 facts")
    provider = str(facts.get("provider_family") or "unknown")
    endpoint = str(facts.get("endpoint") or "unknown")
    requested_model = facts.get("requested_model")
    input_features = {
        "request_facts_schema": facts.get("schema"),
        "thin_request_facts": True,
        "api_endpoint": endpoint,
        "provider_family": provider,
        "stream": bool(facts.get("stream")),
        "text_chars": facts.get("text_chars"),
        "input_tokens_est": facts.get("input_tokens_est"),
        "text_bucket": facts.get("text_bucket"),
        "input_token_bucket": facts.get("input_token_bucket"),
        "message_item_count": facts.get("message_item_count"),
        "input_item_count": facts.get("input_item_count"),
        "response_format_present": bool(facts.get("response_format_present")),
        "requested_local_actions": ["cache", "crunch", "routing"],
        "use_routing_predictor": True,
    }
    # Short classifier labels ride in input_features because the client-contract
    # filter resolves single-name plan fields (category / workflow_phase /
    # uses_thinking) by searching sections in _SHORT_FIELD_SECTION_ORDER, which
    # starts at input_features. Absent here, the stored unit degrades to
    # "unknown" and no trained predictor rule can match it.
    for fact_key in ("category", "workflow_phase", "uses_thinking"):
        if facts.get(fact_key) is not None:
            input_features[fact_key] = facts[fact_key]
    capabilities = facts.get("local_executor_capabilities")
    if isinstance(capabilities, dict):
        supported = capabilities.get("supported_local_action_families")
        if isinstance(supported, list):
            input_features["requested_local_actions"] = sorted({str(item) for item in supported if item})
        input_features["local_executor_capabilities"] = capabilities
    payload = {
        "schema": POLICY_DECISION_PREFLIGHT_SCHEMA,
        "feature_schema_version": REQUEST_FACTS_SCHEMA,
        "source_surface": _source_surface(provider, endpoint) if provider in {"anthropic", "openai"} else "unknown",
        "granularity": "provider_request",
        "app_family": _app_family(provider, str(requested_model or ""), endpoint) if provider in {"anthropic", "openai"} else "unknown",
        "requested_model": requested_model if isinstance(requested_model, str) else None,
        "candidate_target_model": None,
        "input_features": input_features,
        "tool_features": {
            "has_tools": bool(facts.get("has_tools")),
            "tool_count": facts.get("tool_count"),
            "tool_context_present": bool(facts.get("tool_context_present")),
        },
        "outcome_features": {},
        "grouping_identifiers": dict(facts.get("grouping_identifiers") or {}),
        "replayability_level": "features_only",
        "request_facts": facts,
    }
    return _sanitize_features(payload)


def _policy_decision_extra_input_features(unit: dict[str, Any]) -> dict[str, Any]:
    raw_input_features = unit.get("input_features") if isinstance(unit.get("input_features"), dict) else {}
    safe_input_features = _policy_decision_input_features(raw_input_features)
    # Opt in to server-side trained-routing-predictor serving. The request-facts
    # payload path (used for live traffic) does not go through
    # _policy_decision_preflight_payload, so without this the predictor is never
    # consulted and every request falls back to the legacy hold.
    extra: dict[str, Any] = {"use_routing_predictor": True}
    thinking_tail_freshness = safe_input_features.get("thinking_tail_feedback_freshness")
    if isinstance(thinking_tail_freshness, dict):
        extra["thinking_tail_feedback_freshness"] = thinking_tail_freshness
    return _sanitize_features(extra)


def _policy_decision_input_features(value: Any) -> dict[str, Any]:
    """Return endpoint-safe feature fields for /v1/policy-decision input_features."""

    def prune(item: Any) -> Any:
        if isinstance(item, dict):
            cleaned: dict[str, Any] = {}
            for key, child in item.items():
                key_text = str(key)
                if key_text.lower() in POLICY_DECISION_NON_FEATURE_INPUT_KEYS:
                    continue
                cleaned[key_text] = prune(child)
            return cleaned
        if isinstance(item, list):
            return [prune(child) for child in item]
        return item

    return prune(value) if isinstance(value, dict) else {}


def _copy_policy_decision_response_fields(body: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in (
        "decision_id",
        "generated_at",
        "expires_at",
        "expired",
        "optimization_unit_id",
        "source_surface",
        "granularity",
        "app_family",
        "policy_id",
        "confidence",
        "reason_codes",
        "local_action_requirements",
        "provider_capability_matrix_schema",
    ):
        value = body.get(key)
        if value is not None:
            normalized[key] = _sanitize_features(value)
    for key in (
        "routing",
        "routing_action",
        "crunch",
        "cache",
        "canary",
        "shadow",
        "omitted_actions",
        "privacy_summary",
        "provenance",
        "local_executor_compatibility",
        "provider_capabilities",
        "capability_audit",
    ):
        value = body.get(key)
        if value is not None:
            normalized[key] = _sanitize_features(value)
    return normalized


def _normalize_policy_decision(body: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(body, dict):
        return None, "response was not a JSON object"
    if body.get("schema") not in POLICY_DECISION_RESPONSE_SCHEMAS:
        return None, "schema-mismatch"
    privacy = body.get("privacy_summary")
    if isinstance(privacy, dict):
        if privacy.get("metadata_only") is False or privacy.get("raw_payload_included") is True:
            return None, "privacy-not-metadata-only"
    if body.get("provider_forwarding") is True or body.get("server_content_processing") is True:
        return None, "managed-server-forwarding-not-allowed"

    routing = body.get("routing") if isinstance(body.get("routing"), dict) else {}
    routing_action = body.get("routing_action")
    if not isinstance(routing_action, dict):
        routing_action = routing.get("routing_action") if isinstance(routing.get("routing_action"), dict) else {}
    explicit_routing_action = bool(routing_action.get("recommended") is True)
    if not routing and not routing_action:
        return None, "missing-routing-section"
    confidence = body.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = routing_action.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = routing.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    route_to = body.get("route_to") or routing_action.get("target_model") or routing.get("route_to")
    target_model = routing_action.get("target_model") or routing.get("target_model") or route_to
    policy_id = (
        body.get("policy_id")
        or routing_action.get("policy_id")
        or routing.get("policy_id")
        or body.get("decision_id")
        or "managed-policy-decision"
    )
    reason_codes = routing.get("reason_codes") if isinstance(routing.get("reason_codes"), list) else []
    action_reason = routing_action.get("reason")
    reason = (
        str(action_reason)
        if isinstance(action_reason, str) and action_reason
        else ", ".join(str(item) for item in reason_codes if item) or "managed policy decision"
    )
    recommended_mode = routing_action.get("recommended_mode") or routing.get("recommended_mode")
    if explicit_routing_action and not recommended_mode:
        recommended_mode = "route_to"
    traffic_treatment = routing_action.get("traffic_treatment") or routing.get("traffic_treatment")
    route_selected = routing_action.get("route_selected") if "route_selected" in routing_action else routing.get("route_selected")
    normalized = _copy_policy_decision_response_fields(body)
    normalized.update({
        "schema": POLICY_DECISION_SCHEMA,
        "policy_decision_schema": POLICY_DECISION_SCHEMA,
        "target_model": target_model if isinstance(target_model, str) else None,
        "route_to": route_to if isinstance(route_to, str) else None,
        "route_to_present": isinstance(route_to, str) and bool(route_to),
        "confidence": float(confidence),
        "policy_id": str(policy_id),
        "reason": reason,
        "routing_status": routing.get("status") or ("recommended" if explicit_routing_action else None),
        "recommended_mode": recommended_mode,
        "traffic_treatment": traffic_treatment if isinstance(traffic_treatment, str) else None,
        "server_traffic_treatment": traffic_treatment if isinstance(traffic_treatment, str) else None,
        "route_selected": route_selected if isinstance(route_selected, bool) else None,
        "route_down_probability": routing_action.get("route_down_probability", routing.get("route_down_probability")),
        "model_artifact_version": routing_action.get("model_artifact_version", routing.get("model_artifact_version")),
        "model_evidence_hash": routing_action.get("model_evidence_hash", routing.get("model_evidence_hash")),
        "predictor_rule_id": routing_action.get("predictor_rule_id", routing.get("predictor_rule_id")),
        "explicit_routing_action": explicit_routing_action,
        "required_local_gates": routing.get("required_local_gates") if isinstance(routing.get("required_local_gates"), list) else [],
        "replacement_prompt_present": False,
        "raw_payload_included": False,
        "status": "received",
        "policy_source": "managed-recommended",
    })
    return normalized, None


async def fetch_policy_decision(unit: dict[str, Any], *, request_facts: dict[str, Any] | None = None) -> dict[str, Any]:
    if not recommendations_enabled() or not policy_decisions_enabled():
        meta = _policy_decision_base_meta()
        meta.update({
            "status": "skipped",
            "reason": "disabled",
            "fallback": "local-policy",
            "applied": False,
        })
        return meta

    meta = _policy_decision_base_meta()
    try:
        payload = (
            _policy_decision_payload_from_request_facts(request_facts)
            if request_facts is not None
            else _policy_decision_preflight_payload(unit)
        )
        if request_facts is not None:
            payload.setdefault("input_features", {}).update(_policy_decision_extra_input_features(unit))
        contract_meta = await _client_contract_for_payload(payload)
        payload = _attach_client_contract_to_preflight(payload, contract_meta)
        measurement = execute_preflight_measurement_plan(payload, contract_meta)
        contract_skip_reason = _policy_decision_contract_skip_reason(contract_meta)
        if contract_skip_reason is not None:
            _, contract_diagnostics = filter_payload_by_client_contract(
                payload,
                contract_meta,
                stage="preflight",
            )
            meta["client_contract"] = contract_diagnostics
            meta["managed_measurement"] = measurement
            meta.update({
                "status": "skipped",
                "reason": contract_skip_reason,
                "fallback": "local-policy",
                "applied": False,
            })
            return meta
        payload, contract_diagnostics = filter_payload_by_client_contract(
            payload,
            contract_meta,
            stage="preflight",
        )
        meta["client_contract"] = contract_diagnostics
        meta["managed_measurement"] = measurement
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update(managed_egress_blocked_meta(endpoint=POLICY_DECISION_PATH, violations=exc.violations))
        return meta
    except ValueError as exc:
        meta.update({
            "status": "skipped",
            "reason": "invalid-request-facts",
            "error": str(exc),
            "fallback": "local-policy",
            "applied": False,
        })
        return meta

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
        async with async_client(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + POLICY_DECISION_PATH,
                json=payload,
                headers=_managed_headers(),
            )
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        retry_after = _response_retry_after_seconds(response)
        if retry_after is not None:
            meta["retry_after_seconds"] = retry_after
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
        decision, error = _normalize_policy_decision(body)
        if decision is None:
            meta.update({
                "status": "invalid",
                "reason": "invalid-schema",
                "schema_error": error or "invalid-response",
                "fallback": "local-policy",
                "applied": False,
            })
            return meta
        meta.update(decision)
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


def _normalize_recommendation(body: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(body, dict):
        return None, "response was not a JSON object"
    routing = body.get("routing") if isinstance(body.get("routing"), dict) else {}
    target_model = body.get("target_model") or routing.get("target_model")
    confidence = body.get("confidence")
    policy_id = body.get("policy_id") or routing.get("policy_id")
    reason = body.get("reason") or routing.get("reason")
    has_policy_sections = any(isinstance(body.get(section), dict) for section in ("routing", "crunch", "cache"))
    if not has_policy_sections and (not isinstance(target_model, str) or not target_model):
        return None, "missing target_model"
    if not isinstance(confidence, (int, float)):
        confidence = 0.0 if has_policy_sections else None
    if confidence is None:
        return None, "missing confidence"
    if not isinstance(policy_id, str) or not policy_id:
        policy_id = str(body.get("recommendation_id") or "managed-policy-decision") if has_policy_sections else ""
    if not policy_id:
        return None, "missing policy_id"
    if not isinstance(reason, str) or not reason:
        reason = "managed local action policy decision" if has_policy_sections else ""
    if not reason:
        return None, "missing reason"

    replacement_prompt = body.get("replacement_prompt")
    normalized = {
        "target_model": target_model if isinstance(target_model, str) else None,
        "confidence": float(confidence),
        "policy_id": policy_id,
        "reason": reason,
        "replacement_prompt_present": isinstance(replacement_prompt, str) and bool(replacement_prompt),
    }
    for key in ("schema", "provider", "source_surface", "expires_at", "expiry"):
        value = body.get(key)
        if isinstance(value, str) and value:
            normalized[key] = value
    for key in ("routing", "crunch", "cache", "privacy_summary", "provenance"):
        value = body.get(key)
        if isinstance(value, dict):
            normalized[key] = _sanitize_features(value)
    actions = body.get("actions")
    if isinstance(actions, list):
        normalized["actions"] = _sanitize_features(actions)
    for key in ("signed", "signature_required", "requires_signature", "raw_payload_included"):
        if isinstance(body.get(key), bool):
            normalized[key] = bool(body[key])
    optimization_unit_id = body.get("optimization_unit_id")
    if isinstance(optimization_unit_id, int):
        normalized["optimization_unit_id"] = optimization_unit_id
    recommendation_id = body.get("recommendation_id")
    if isinstance(recommendation_id, (int, str)):
        normalized["recommendation_id"] = recommendation_id
    for key in (
        "source_surface",
        "app_family",
        "recommendation_family",
        "candidate_id",
        "routing_rule_id",
        "projected_latency_basis",
    ):
        value = body.get(key)
        if isinstance(value, str) and value:
            normalized[key] = value
    for key in (
        "sample_count",
        "matched_sample_count",
        "baseline_sample_count",
        "candidate_sample_count",
        "projected_latency_ms",
        "latency_ms_p50",
        "candidate_latency_ms_p50",
    ):
        value = body.get(key)
        if isinstance(value, int):
            normalized[key] = value
    for key in (
        "error_rate",
        "retry_rate",
        "fallback_rate",
        "latency_regression_ratio",
        "observed_savings_usd",
        "projected_savings_usd",
    ):
        value = body.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = float(value)
    if normalized["replacement_prompt_present"]:
        normalized["replacement_prompt_sha256"] = hashlib.sha256(replacement_prompt.encode("utf-8")).hexdigest()
    return normalized, None


async def fetch_recommendation(unit: dict[str, Any]) -> dict[str, Any]:
    if not recommendations_enabled():
        return disabled_recommendation_meta()

    meta = _base_meta()
    try:
        contract_meta = await _client_contract_for_payload(unit)
        measurement = execute_preflight_measurement_plan(unit, contract_meta)
        payload, contract_diagnostics = filter_payload_by_client_contract(
            unit,
            contract_meta,
            stage="preflight",
        )
        meta["client_contract"] = contract_diagnostics
        meta["managed_measurement"] = measurement
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update(managed_egress_blocked_meta(endpoint=RECOMMENDATION_PATH, violations=exc.violations))
        return meta

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
        async with async_client(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + RECOMMENDATION_PATH,
                json=payload,
                headers=_managed_headers(),
            )
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        retry_after = _response_retry_after_seconds(response)
        if retry_after is not None:
            meta["retry_after_seconds"] = retry_after
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


def policy_decision_canary_sample(meta: dict[str, Any], *, current_model: str, target_model: str) -> dict[str, Any]:
    fraction = policy_decision_canary_fraction()
    basis = {
        "policy_id": meta.get("policy_id"),
        "decision_id": meta.get("decision_id"),
        "optimization_unit_id": meta.get("optimization_unit_id"),
        "model_artifact_version": meta.get("model_artifact_version"),
        "model_evidence_hash": meta.get("model_evidence_hash"),
        "predictor_rule_id": meta.get("predictor_rule_id"),
        "current_model": current_model,
        "target_model": target_model,
    }
    digest = hashlib.sha256(f"{policy_decision_canary_salt()}:{stable_json(basis)}".encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    selected = score < fraction
    return {
        "schema": "tokenclaw.policy_decision_local_canary.v1",
        "enabled": fraction > 0.0,
        "fraction": fraction,
        "salt": policy_decision_canary_salt(),
        "unit": "policy_decision",
        "bucket": round(score, 8),
        "threshold": fraction,
        "selected": selected,
        "cohort": "canary_applied" if selected else "canary_holdout",
        "cohort_key_hash": f"sha256:{digest}",
    }


def _has_trained_routing_predictor_evidence(meta: dict[str, Any]) -> bool:
    routing = meta.get("routing") if isinstance(meta.get("routing"), dict) else {}
    artifact = str(meta.get("model_artifact_version") or routing.get("model_artifact_version") or "").strip()
    if not artifact.startswith("routing-predictor-"):
        return False
    if str(meta.get("predictor_rule_id") or routing.get("predictor_rule_id") or "").strip():
        return True
    reason_codes = set()
    for source in (meta.get("reason_codes"), routing.get("reason_codes")):
        if isinstance(source, list):
            reason_codes.update(str(item) for item in source)
    return "active-routing-predictor-model" in reason_codes


def _policy_decision_routing_gate(meta: dict[str, Any], *, target_model: str, current_model: str) -> str | None:
    if meta.get("routing_status") != "recommended":
        return "routing-not-recommended"
    if not target_model:
        return "missing-target-model"
    if target_model == current_model:
        return "target-model-already-selected"
    server_treatment = str(meta.get("traffic_treatment") or meta.get("server_traffic_treatment") or "").strip().lower()
    recommended_mode = str(meta.get("recommended_mode") or "").strip().lower()
    if (
        server_treatment in {"live", "canary"}
        or recommended_mode in {"live", "apply", "applied", "canary", "route_to"}
        or meta.get("route_to_present") is True
    ) and not _has_trained_routing_predictor_evidence(meta):
        return "routing-predictor-evidence-required"
    if server_treatment in {"live", "canary"} and meta.get("route_selected") is not False:
        return None
    try:
        confidence = float(meta.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    probability = meta.get("route_down_probability")
    if isinstance(probability, (int, float)):
        confidence = min(confidence, float(probability))
    if confidence < policy_decision_min_confidence():
        return "routing-predictor-confidence-too-low"
    if (
        not meta.get("explicit_routing_action")
        and (not isinstance(meta.get("model_artifact_version"), str) or not meta.get("model_artifact_version"))
    ):
        return "routing-predictor-model-version-missing"
    return None


def apply_policy_decision_routing_to_body(
    *,
    provider: str,
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    recommendation_meta: dict[str, Any],
    store_obj: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    meta = dict(recommendation_meta)
    executor = ActionExecutor(provider=provider, store_obj=store_obj, session_id=session_id)
    if meta.get("status") != "received":
        meta.setdefault("applied", False)
        meta.setdefault("local_action_taken", "fallback")
        return meta
    target_model = str(meta.get("target_model") or "")
    current_model = str(body.get("model") or routing_meta.get("routed_model") or "")
    meta["local_model_before_recommendation"] = current_model
    meta["local_policy_decision_mode"] = meta.get("recommended_mode") or "observe"
    meta["min_confidence"] = policy_decision_min_confidence()
    meta["canary_fraction"] = policy_decision_canary_fraction()
    client_policy = routing_pathway_policy_decision(
        provider=provider,
        requested_model=str(routing_meta.get("requested_model") or current_model),
        current_model=current_model,
        target_model=target_model,
        source_surface=str(meta.get("source_surface") or routing_meta.get("source_surface") or ""),
        category=str(routing_meta.get("category") or ""),
        workflow_phase=str(routing_meta.get("workflow_phase") or ""),
        stream=bool(routing_meta.get("stream")),
        pathway_id=str(meta.get("policy_id") or meta.get("decision_id") or ""),
    )
    meta["client_routing_policy"] = client_policy
    if not client_policy.get("allowed", True):
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "client-routing-blocklist",
            "fallback": "local-policy",
            "would_route_model": target_model or None,
            "local_action_taken": "noop",
        })
        return meta
    policy_target = client_policy.get("target_model")
    if isinstance(policy_target, str) and policy_target:
        target_model = policy_target
        meta["target_model_after_client_policy"] = target_model

    if not _provider_compatible(provider, target_model):
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "provider-mismatch" if target_model else "missing-target-model",
            "fallback": "local-policy",
            "local_action_taken": "noop",
        })
        return meta
    if not _supported_target_model(provider, target_model):
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "unsupported-target-model",
            "fallback": "local-policy",
            "local_action_taken": "noop",
        })
        return meta
    if meta.get("replacement_prompt_present"):
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "unsafe-replacement-prompt",
            "fallback": "local-policy",
            "replacement_prompt_applied": False,
            "local_action_taken": "noop",
        })
        return meta
    if target_model != current_model and "thinking request" in str(routing_meta.get("reason") or ""):
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "local-thinking-safety-guard",
            "fallback": "local-policy",
            "local_action_taken": "noop",
        })
        return meta

    gate_reason = _policy_decision_routing_gate(meta, target_model=target_model, current_model=current_model)
    if gate_reason is not None:
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": gate_reason,
            "fallback": "local-policy",
            "would_route_model": target_model,
            "local_action_taken": "noop",
        })
        return meta

    recommended_mode = str(meta.get("recommended_mode") or "observe").lower()
    if meta.get("route_to_present") and recommended_mode in {"", "observe", "shadow", "none", "route_to"}:
        canary = policy_decision_canary_sample(meta, current_model=current_model, target_model=target_model)
        if policy_decision_canary_fraction() > 0.0:
            meta["local_canary"] = canary
            if not canary["selected"]:
                meta.update({
                    "applied": False,
                    "changed_model": False,
                    "apply_reason": "local-canary-holdout",
                    "fallback": "local-policy",
                    "would_route_model": target_model,
                    "local_action_taken": "canary_holdout",
                    "local_policy_decision_mode": "route_to",
                })
                return meta
        meta["target_model_after_client_policy"] = target_model
        execution = executor.execute(
            body=body,
            routing_meta=routing_meta,
            decision=meta,
            application_enabled=True,
            source_surface=str(routing_meta.get("source_surface") or meta.get("source_surface") or ""),
        )
        if execution.get("status") == "vetoed":
            meta.update({
                "applied": False,
                "changed_model": False,
                "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
                "fallback": "local-policy",
                "would_route_model": target_model,
                "local_action_taken": "vetoed",
                "action_executor": execution,
                "local_actions": execution,
            })
            routing_meta["managed_local_actions"] = execution
            return meta
        routing_meta["managed_routing_confidence"] = meta.get("confidence")
        routing_meta["managed_route_recommended_mode"] = "route_to"
        meta.update({
            "applied": bool(execution.get("changed_model")),
            "changed_model": bool(execution.get("changed_model")),
            "fallback": execution.get("fallback"),
            "apply_reason": "local-canary-selected" if policy_decision_canary_fraction() > 0.0 and execution.get("changed_model") else "route-to-local-safety-gate-passed",
            "local_action_taken": "canary_applied" if policy_decision_canary_fraction() > 0.0 and execution.get("changed_model") else "route_to",
            "local_policy_decision_mode": "route_to",
            "action_executor": execution,
            "local_actions": execution,
        })
        routing_meta["managed_local_actions"] = execution
        if policy_decision_canary_fraction() > 0.0:
            meta["local_canary"] = canary
        return meta

    if recommended_mode in {"observe", "shadow"}:
        execution = executor.execute(
            body=body,
            routing_meta=routing_meta,
            decision=meta,
            application_enabled=True,
            shadow_only=True,
            source_surface=str(routing_meta.get("source_surface") or meta.get("source_surface") or ""),
        )
        if execution.get("status") == "vetoed":
            meta.update({
                "applied": False,
                "changed_model": False,
                "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
                "fallback": "local-policy",
                "would_route_model": target_model,
                "local_action_taken": "vetoed",
                "action_executor": execution,
                "local_actions": execution,
            })
            routing_meta["managed_local_actions"] = execution
            return meta
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": f"{recommended_mode}-only",
            "fallback": "local-policy",
            "would_route_model": target_model,
            "local_action_taken": recommended_mode,
            "action_executor": execution,
            "local_actions": execution,
        })
        routing_meta["managed_route_candidate_model"] = target_model
        routing_meta["managed_route_candidate_reason"] = meta.get("reason")
        routing_meta["managed_route_recommended_mode"] = recommended_mode
        routing_meta["managed_local_actions"] = execution
        return meta
    if recommended_mode in {"live", "apply", "applied"}:
        meta["target_model_after_client_policy"] = target_model
        execution = executor.execute(
            body=body,
            routing_meta=routing_meta,
            decision=meta,
            application_enabled=True,
            source_surface=str(routing_meta.get("source_surface") or meta.get("source_surface") or ""),
        )
        if execution.get("status") == "vetoed":
            meta.update({
                "applied": False,
                "changed_model": False,
                "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
                "fallback": "local-policy",
                "would_route_model": target_model,
                "local_action_taken": "vetoed",
                "action_executor": execution,
                "local_actions": execution,
            })
            routing_meta["managed_local_actions"] = execution
            return meta
        routing_meta["managed_routing_confidence"] = meta.get("confidence")
        routing_meta["managed_route_recommended_mode"] = recommended_mode
        meta.update({
            "applied": bool(execution.get("changed_model")),
            "changed_model": bool(execution.get("changed_model")),
            "fallback": execution.get("fallback"),
            "apply_reason": "server-live-selected" if execution.get("changed_model") else execution.get("apply_reason"),
            "local_action_taken": "live_applied" if execution.get("changed_model") else "noop",
            "local_policy_decision_mode": recommended_mode,
            "action_executor": execution,
            "local_actions": execution,
        })
        routing_meta["managed_local_actions"] = execution
        return meta
    if recommended_mode != "canary":
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": f"recommended-mode-{recommended_mode or 'missing'}",
            "fallback": "local-policy",
            "would_route_model": target_model,
            "local_action_taken": "noop",
        })
        return meta

    canary = policy_decision_canary_sample(meta, current_model=current_model, target_model=target_model)
    meta["local_canary"] = canary
    if not canary["selected"]:
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": "local-canary-holdout",
            "fallback": "local-policy",
            "would_route_model": target_model,
            "local_action_taken": "canary_holdout",
        })
        return meta

    meta["target_model_after_client_policy"] = target_model
    execution = executor.execute(
        body=body,
        routing_meta=routing_meta,
        decision=meta,
        application_enabled=True,
        source_surface=str(routing_meta.get("source_surface") or meta.get("source_surface") or ""),
    )
    if execution.get("status") == "vetoed":
        meta.update({
            "applied": False,
            "changed_model": False,
            "apply_reason": execution.get("apply_reason") or "local-action-vetoed",
            "fallback": "local-policy",
            "would_route_model": target_model,
            "local_action_taken": "vetoed",
            "action_executor": execution,
            "local_actions": execution,
        })
        routing_meta["managed_local_actions"] = execution
        return meta
    routing_meta["managed_routing_confidence"] = meta.get("confidence")
    routing_meta["managed_route_recommended_mode"] = recommended_mode
    meta.update({
        "applied": bool(execution.get("changed_model")),
        "changed_model": bool(execution.get("changed_model")),
        "fallback": execution.get("fallback"),
        "apply_reason": "local-canary-selected" if execution.get("changed_model") else execution.get("apply_reason"),
        "local_action_taken": "canary_applied" if execution.get("changed_model") else "noop",
        "action_executor": execution,
        "local_actions": execution,
    })
    routing_meta["managed_local_actions"] = execution
    return meta


def apply_recommendation_to_body(
    *,
    provider: str,
    body: dict[str, Any],
    routing_meta: dict[str, Any],
    recommendation_meta: dict[str, Any],
    store_obj: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    if recommendation_meta.get("policy_decision_schema") == POLICY_DECISION_SCHEMA:
        return apply_policy_decision_routing_to_body(
            provider=provider,
            body=body,
            routing_meta=routing_meta,
            recommendation_meta=recommendation_meta,
            store_obj=store_obj,
            session_id=session_id,
        )

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


def attach_observed_savings_to_routing_meta(
    routing_meta: dict[str, Any],
    *,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    status_code: int | None = None,
) -> None:
    """Annotate routing metadata with the observed call-level cost delta."""

    if cost_est_usd is None or cost_baseline_usd is None:
        observed_savings = 0.0
        cost_known = False
    else:
        observed_savings = max(float(cost_baseline_usd) - float(cost_est_usd), 0.0)
        cost_known = True

    success = status_code is None or int(status_code) < 400
    if not success:
        observed_savings = 0.0

    observed_savings = round(observed_savings, 8)
    routing_meta["observed_savings_usd"] = observed_savings
    routing_meta["observed_savings_basis"] = "calls.cost_baseline_usd-minus-cost_est_usd"
    routing_meta["observed_savings_cost_known"] = cost_known

    managed = routing_meta.get("managed_recommendation")
    if not isinstance(managed, dict):
        return

    attributed_to_managed = bool(managed.get("applied") and managed.get("changed_model"))
    managed["observed_savings_usd"] = observed_savings if attributed_to_managed else 0.0
    managed["observed_savings_basis"] = "calls.cost_baseline_usd-minus-cost_est_usd"
    managed["observed_savings_cost_known"] = cost_known
    managed["observed_savings_status_code"] = status_code
    managed["observed_savings_attributed_to_managed"] = attributed_to_managed
    managed["observed_savings_attribution"] = (
        "managed-recommendation-model-change"
        if attributed_to_managed
        else "not-attributed-without-managed-model-change"
    )
    if cost_known:
        managed["cost_est_usd"] = round(float(cost_est_usd), 8)
        managed["cost_baseline_usd"] = round(float(cost_baseline_usd), 8)


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
        if head and len(head) <= 80 and all(ch.isalnum() or ch in {"_", "-"} for ch in head):
            error_class = head
    return {
        "error_class": error_class,
        "error_present": True,
        "error_chars_bucket": _text_bucket(len(error_text)),
        "error_sha256": "sha256:" + hashlib.sha256(error_text.encode("utf-8")).hexdigest(),
        "raw_error_included": False,
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


def _pattern_evidence_cohort(decision: dict[str, Any]) -> str:
    cohort = decision.get("cohort")
    if cohort in {"canary_applied", "canary_holdout", "bypassed", "skipped"}:
        return str(cohort)
    status = str(decision.get("status") or "").lower()
    outcome = str(decision.get("outcome") or "").lower()
    reason = str(decision.get("reason") or "").lower()
    if status == "holdout" or outcome == "holdout" or reason == "canary_holdout":
        return "canary_holdout"
    if status in {"bypass", "bypassed"} or outcome == "bypassed" or "bypass" in reason or "disabled" in reason:
        return "bypassed"
    if status in {"applied", "hit"} or _safe_int(decision.get("applied_count")) > 0:
        return "canary_applied"
    return "skipped"


def _pattern_evidence_outcome(decision: dict[str, Any], status_code: int | None) -> str:
    if status_code is not None and int(status_code) >= 400:
        return "failed"
    outcome = decision.get("outcome")
    if isinstance(outcome, str) and outcome:
        if outcome == "errored":
            return "failed"
        return outcome
    status = decision.get("status")
    if isinstance(status, str) and status:
        return status
    return "unknown"


def _pattern_evidence_action_family(decision: dict[str, Any]) -> str:
    decision_type = str(decision.get("decision_type") or "")
    if decision_type in {"routing", "crunch", "cache"}:
        return decision_type
    pattern_family = str(decision.get("pattern_family") or "")
    if pattern_family in {"routing", "crunch", "cache"}:
        return pattern_family
    if decision_type == "local_pattern_fingerprint":
        return "fingerprint"
    return decision_type or "unknown"


def _pattern_evidence_hashes(decision: dict[str, Any]) -> tuple[str | None, list[str]]:
    hashes: list[str] = []
    for item in decision.get("pattern_hashes") or []:
        if isinstance(item, str) and item.startswith("sha256:"):
            hashes.append(item)
    pattern_hash = decision.get("pattern_hash")
    if isinstance(pattern_hash, str) and pattern_hash.startswith("sha256:"):
        hashes.insert(0, pattern_hash)
    deduped = sorted(set(hashes))
    primary = pattern_hash if isinstance(pattern_hash, str) and pattern_hash.startswith("sha256:") else (deduped[0] if deduped else None)
    return primary, deduped


def _pattern_evidence_row(
    decision: dict[str, Any],
    *,
    action_family: str | None = None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    savings_usd: float | None = None,
) -> dict[str, Any] | None:
    pattern_hash, pattern_hashes = _pattern_evidence_hashes(decision)
    if pattern_hash is None:
        return None
    savings = savings_usd if savings_usd is not None else _safe_float(decision.get("estimated_cost_savings_usd"))
    row = {
        "schema": "tokenclaw.managed_pattern_policy_evidence.v1",
        "source_surface": decision.get("source_surface"),
        "app_family": decision.get("app_family"),
        "action_family": action_family or _pattern_evidence_action_family(decision),
        "pattern_family": decision.get("pattern_family") or action_family or _pattern_evidence_action_family(decision),
        "pattern_hash": pattern_hash,
        "pattern_hashes": pattern_hashes,
        "candidate_id": decision.get("candidate_id"),
        "rule_id": decision.get("rule_id"),
        "policy_source": decision.get("policy_source"),
        "cohort": _pattern_evidence_cohort(decision),
        "local_decision_status": decision.get("status"),
        "local_outcome": decision.get("outcome"),
        "outcome": _pattern_evidence_outcome(decision, status_code),
        "status_code_bucket": _status_code_bucket(status_code),
        "retry_bucket": _retry_bucket(retry_count),
        "latency_bucket": _latency_bucket(latency_ms),
        "savings_bucket": _net_savings_bucket(savings),
        "tokens_saved_bucket": _token_bucket(decision.get("tokens_saved_est")),
        "category": decision.get("category"),
        "workflow_phase": decision.get("workflow_phase"),
        "text_bucket": decision.get("text_bucket"),
        "token_bucket": decision.get("token_bucket"),
        "replayability_level": decision.get("replayability_level"),
        "evidence_only": bool(decision.get("evidence_only")),
        "raw_pattern_strings_included": False,
        "raw_payload_included": False,
    }
    for key in (
        "pattern_types",
        "local_pattern_module_families",
        "local_pattern_module_count",
        "hit_type",
    ):
        if decision.get(key) is not None:
            row[key] = decision.get(key)
    return {key: value for key, value in row.items() if value is not None}


def pattern_policy_evidence_summaries(
    *,
    provider: str,
    path: str,
    requested_model: str | None,
    routed_model: str | None,
    status_code: int | None,
    latency_ms: int | None,
    retry_count: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    cache_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    category: str | None,
    pattern_decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build feature-only managed policy evidence from local pattern outcomes."""
    decisions = pattern_decisions if pattern_decisions is not None else pattern_decision_summaries(
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
    )
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        row = _pattern_evidence_row(
            decision,
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
        )
        if row is not None:
            rows.append(row)

    diagnostics = routing_meta.get("managed_pattern_features") if isinstance(routing_meta, dict) else None
    if isinstance(diagnostics, dict) and diagnostics.get("present"):
        pattern_hash = diagnostics.get("pattern_hash") or diagnostics.get("normalized_pattern_hash")
        if isinstance(pattern_hash, str) and pattern_hash.startswith("sha256:"):
            routing_changed = bool(requested_model and routed_model and requested_model != routed_model)
            managed = routing_meta.get("managed_recommendation") if isinstance(routing_meta.get("managed_recommendation"), dict) else {}
            routing_decision = {
                "schema": "tokenclaw.pattern_decision_summary.v1",
                "decision_type": "routing",
                "source_surface": diagnostics.get("source_surface") or _source_surface(provider, path),
                "app_family": diagnostics.get("app_family") or _app_family(provider, requested_model or "", path),
                "category": diagnostics.get("category") or category or routing_meta.get("category") or "unknown",
                "workflow_phase": diagnostics.get("workflow_phase") or routing_meta.get("workflow_phase") or category or "unknown",
                "rule_id": routing_meta.get("rule_id") or routing_meta.get("routing_rule_id") or routing_meta.get("matched_rule_id"),
                "candidate_id": managed.get("candidate_id") or routing_meta.get("managed_policy_id") or managed.get("policy_id"),
                "pattern_hash": pattern_hash,
                "pattern_hashes": [pattern_hash],
                "pattern_family": "routing",
                "pattern_types": diagnostics.get("pattern_types") or [],
                "local_pattern_module_families": diagnostics.get("local_pattern_module_families") or [],
                "local_pattern_module_count": _safe_int(diagnostics.get("local_pattern_module_count")),
                "policy_source": routing_meta.get("final_policy_source") or routing_meta.get("policy_source") or "local-default",
                "status": "applied" if routing_changed else "skipped",
                "outcome": _pattern_outcome(status_code, applied=routing_changed),
                "applied_count": 1 if routing_changed else 0,
                "estimated_cost_savings_usd": max(_safe_float(cost_baseline_usd) - _safe_float(cost_est_usd), 0.0) if routing_changed else 0.0,
                "text_bucket": diagnostics.get("text_bucket"),
                "token_bucket": diagnostics.get("token_bucket"),
                "replayability_level": diagnostics.get("replayability_level"),
                "evidence_only": True,
            }
            row = _pattern_evidence_row(
                routing_decision,
                action_family="routing",
                status_code=status_code,
                latency_ms=latency_ms,
                retry_count=retry_count,
            )
            if row is not None:
                rows.append(row)

    unique: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("action_family")), str(row.get("pattern_hash")), row.get("rule_id"))
        unique[key] = row
    return _sanitize_features(list(unique.values()))


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

    pattern_diagnostics = routing_meta.get("managed_pattern_features") if isinstance(routing_meta, dict) else None
    if isinstance(pattern_diagnostics, dict) and pattern_diagnostics.get("present"):
        raw_hashes = [
            ("general", pattern_diagnostics.get("pattern_hash") or pattern_diagnostics.get("normalized_pattern_hash")),
            ("crunch", pattern_diagnostics.get("crunch_pattern_hash")),
            ("cache", pattern_diagnostics.get("cache_pattern_hash")),
        ]
        for item in pattern_diagnostics.get("pattern_hashes") or []:
            raw_hashes.append(("observed", item))
        seen_hashes: set[str] = set()
        for pattern_family, pattern_hash in raw_hashes:
            if not isinstance(pattern_hash, str) or not pattern_hash.startswith("sha256:"):
                continue
            if pattern_hash in seen_hashes:
                continue
            seen_hashes.add(pattern_hash)
            rows.append({
                "schema": "tokenclaw.pattern_decision_summary.v1",
                "decision_type": "local_pattern_fingerprint",
                "source_surface": pattern_diagnostics.get("source_surface") or source_surface,
                "app_family": pattern_diagnostics.get("app_family") or app_family,
                "category": pattern_diagnostics.get("category") or category or routing_meta.get("category") or "unknown",
                "workflow_phase": pattern_diagnostics.get("workflow_phase") or workflow_phase,
                "rule_id": None,
                "candidate_id": None,
                "pattern_hash": pattern_hash,
                "pattern_hashes": [pattern_hash],
                "pattern_family": pattern_family,
                "pattern_types": pattern_diagnostics.get("pattern_types") or [],
                "local_pattern_module_families": pattern_diagnostics.get("local_pattern_module_families") or [],
                "local_pattern_module_count": _safe_int(pattern_diagnostics.get("local_pattern_module_count")),
                "policy_source": "local-default",
                "status": "observed",
                "reason": "local-pattern-fingerprint-observed",
                "outcome": "errored" if status_code is not None and int(status_code) >= 400 else "observed",
                "applied_count": 0,
                "saved_chars": 0,
                "tokens_saved_est": 0,
                "estimated_cost_savings_usd": 0.0,
                "text_bucket": pattern_diagnostics.get("text_bucket"),
                "token_bucket": pattern_diagnostics.get("token_bucket"),
                "replayability_level": pattern_diagnostics.get("replayability_level"),
                "evidence_only": True,
                "raw_pattern_strings_included": False,
                "safety_gates": {
                    "feature_only": True,
                    "raw_pattern_strings_included": False,
                },
            })

    pattern_rules = crunch_meta.get("pattern_rules") if isinstance(crunch_meta, dict) else None
    if isinstance(pattern_rules, dict) and pattern_rules.get("configured_count"):
        for rule in pattern_rules.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            saved_chars = _safe_int(rule.get("saved_chars"))
            saved_tokens = max(0, saved_chars // TOKEN_CHARS)
            applied = _safe_int(rule.get("applied_count")) > 0
            base = {
                "schema": "tokenclaw.pattern_decision_summary.v1",
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
                "schema": "tokenclaw.pattern_decision_summary.v1",
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
            "schema": "tokenclaw.cache_pattern_decision_basis.v1",
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
            "schema": "tokenclaw.pattern_decision_summary.v1",
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
                "schema": "tokenclaw.pattern_decision_summary.v1",
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


def _terminal_log_features_from_routing_meta(routing_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(routing_meta, dict):
        return None
    candidates: list[Any] = [routing_meta.get("terminal_log_features")]
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        summary = routing_meta.get(key)
        if isinstance(summary, dict):
            candidates.append(summary.get("terminal_log_features"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("schema") == TERMINAL_LOG_FEATURE_SCHEMA:
            return candidate
    return None


def _prompt_difficulty_features_from_routing_meta(routing_meta: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(routing_meta, dict):
        return None
    candidates: list[Any] = [routing_meta.get("prompt_difficulty_features")]
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        summary = routing_meta.get(key)
        if isinstance(summary, dict):
            candidates.append(summary.get("prompt_difficulty_features"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("schema") == PROMPT_DIFFICULTY_FEATURE_SCHEMA:
            return candidate
    return None


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
    provider_adoption_windows: list[dict[str, Any]] | None = None,
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
        provider_adoption_windows=provider_adoption_windows,
    )
    pattern_decisions = pattern_decision_summaries(
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
        "routing_outcome_label": str(routing_meta.get("routing_outcome_label") or "unknown"),
        "crunch_saved_chars": crunch_meta.get("saved_chars"),
        "crunch_tokens_saved_est": crunch_meta.get("tokens_saved_est"),
        "pattern_decisions": pattern_decisions,
        "pattern_policy_evidence": pattern_policy_evidence_summaries(
            provider=provider,
            path=path,
            requested_model=requested_model,
            routed_model=routed_model,
            status_code=status_code,
            latency_ms=latency_ms,
            retry_count=retry_count,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=category,
            pattern_decisions=pattern_decisions,
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
            "mode": managed.get("mode"),
            "status": managed.get("status"),
            "lifecycle_event": managed.get("lifecycle_event"),
            "applied": bool(managed.get("applied")),
            "changed_model": bool(managed.get("changed_model")),
            "apply_reason": managed.get("apply_reason"),
            "target_model_normalized": managed.get("target_model_normalized"),
            "canary_cohort": (
                managed.get("canary", {}).get("cohort")
                if isinstance(managed.get("canary"), dict)
                else None
            ),
        },
        "quality_signals": quality_signals,
    }
    terminal_log_features = _terminal_log_features_from_routing_meta(routing_meta)
    if terminal_log_features is not None:
        features["terminal_log_features"] = terminal_log_features
    prompt_difficulty_features = _prompt_difficulty_features_from_routing_meta(routing_meta)
    if prompt_difficulty_features is not None:
        features["prompt_difficulty_features"] = prompt_difficulty_features
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
    reason_codes = [str(item) for item in summary_meta.get("reason_codes") or [] if str(item)]
    reason = str(summary_meta.get("reason") or (reason_codes[0] if reason_codes else ""))
    canary = summary_meta.get("canary") if isinstance(summary_meta.get("canary"), dict) else {}
    canary_cohort = str(canary.get("cohort") or "")
    fail_closed_reasons = OLD_CONTEXT_SUMMARY_FAIL_CLOSED_REASONS | {
        "summary_fetch_error",
        "summary_empty_or_malformed",
        "summary_cost_over_budget",
        "tool_function_protocol_ambiguous",
        "file_reference_in_source_window",
        "preservation_mismatch",
    }
    relevant = (
        status in {"applied", "bypass", "error"}
        or reason in {"canary_holdout", "local-canary-safety-stop"}
        or reason in fail_closed_reasons
        or bool(set(reason_codes) & fail_closed_reasons)
        or canary_cohort in {"canary_applied", "canary_holdout", "holdout"}
        or (canary.get("enabled") and canary_cohort in {"canary_applied", "canary_holdout"})
    )
    if not relevant:
        return None

    try:
        eligible_chars = int(summary_meta.get("eligible_chars") or summary_meta.get("source_chars") or 0)
    except (TypeError, ValueError):
        eligible_chars = 0
    try:
        eligible_turns = int(summary_meta.get("eligible_turns") or summary_meta.get("source_item_count") or 0)
    except (TypeError, ValueError):
        eligible_turns = 0
    try:
        saved_chars = int(
            summary_meta.get("saved_chars")
            or summary_meta.get("estimated_chars_saved")
            or summary_meta.get("actual_chars_saved_est")
            or 0
        )
    except (TypeError, ValueError):
        saved_chars = 0
    try:
        saved_tokens = int(summary_meta.get("tokens_saved_est") or summary_meta.get("estimated_tokens_saved") or 0)
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
    enhanced_provider = (
        summary_meta.get("enhanced_crunch_provider")
        if isinstance(summary_meta.get("enhanced_crunch_provider"), dict)
        else {}
    )
    feedback = {
        "schema": "tokenclaw.old_context_summary_outcome_feedback.v1",
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
        "summary_policy_id": _metadata_identifier(summary_meta.get("rule_id")),
        "rule_id": _metadata_identifier(summary_meta.get("rule_id")),
        "candidate_id": _metadata_identifier(summary_meta.get("candidate_id")),
        "policy_source": summary_meta.get("policy_source"),
        "enhanced_crunch": {
            "state": summary_meta.get("enhanced_crunch_state") or enhanced_provider.get("state"),
            "mode": enhanced_provider.get("mode"),
            "profile": enhanced_provider.get("profile"),
            "configured": enhanced_provider.get("configured"),
            "recommended": enhanced_provider.get("recommended"),
            "model_family": enhanced_provider.get("model_family") or _model_family(summary_meta.get("model")),
            "model_tier": _model_family(summary_meta.get("model")),
            "failure_state": reason if status != "applied" else None,
            "safety_stop_status": summary_meta.get("safety_stop_state"),
        },
        "canary": {
            "enabled": bool(canary.get("enabled")),
            "selected": canary.get("selected") if canary.get("selected") is not None else canary_cohort == "canary_applied",
            "status": canary.get("status"),
            "cohort": canary_cohort or None,
            "fraction": canary.get("fraction") if canary.get("fraction") is not None else canary.get("canary_fraction"),
            "unit": canary.get("unit"),
        },
        "canary_cohort": canary_cohort or None,
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
    from tokenclaw import __version__

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
            "schema": "tokenclaw.old_context_summary_outcome_event_metadata.v1",
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


def _phase_canary_outcome(canary: dict[str, Any], status_code: int | None) -> str:
    if status_code is not None and int(status_code) >= 400:
        return "errored"
    status = str(canary.get("status") or "")
    reason = str(canary.get("reason") or "")
    if status == "applied":
        return "applied"
    if status == "holdout":
        return "holdout"
    if status == "safety_stopped" or reason == "safety-stop-tripped":
        return "safety_stopped"
    if reason == "rate_limited":
        return "fallback"
    if status in {"ineligible", "not_selected"}:
        return "skipped"
    return status or "unknown"


def _phase_savings_bucket(cost_est_usd: float | None, cost_baseline_usd: float | None) -> str:
    if cost_est_usd is None or cost_baseline_usd is None:
        return "unknown"
    return _net_savings_bucket(float(cost_baseline_usd) - float(cost_est_usd))


def build_phase_routing_outcome_feedback(
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
    thinking_output_tokens: int | None,
    cost_est_usd: float | None,
    cost_baseline_usd: float | None,
    cache_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    routing_meta: dict[str, Any],
    category: str | None,
    error: str | None = None,
) -> dict[str, Any] | None:
    canary = routing_meta.get("phase_canary") if isinstance(routing_meta, dict) else None
    if not isinstance(canary, dict) or not canary.get("enabled"):
        return None
    status = str(canary.get("status") or "")
    reason = str(canary.get("reason") or "")
    if status not in {"applied", "holdout", "safety_stopped"} and reason not in {"selected-canary", "selected-holdout", "safety-stop-tripped"}:
        return None

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
    safety_stop = canary.get("safety_stop") if isinstance(canary.get("safety_stop"), dict) else {}
    feedback = {
        "schema": "tokenclaw.phase_routing_outcome_feedback.v1",
        "source_surface": source_surface,
        "app_family": _app_family(provider, requested_model or "", path),
        "category": category or routing_meta.get("category") or canary.get("category") or "unknown",
        "workflow_phase": routing_meta.get("workflow_phase") or canary.get("workflow_phase") or "unknown",
        "workflow_phase_confidence": routing_meta.get("workflow_phase_confidence") or canary.get("workflow_phase_confidence"),
        "status": status or "unknown",
        "reason": reason or None,
        "outcome": _phase_canary_outcome(canary, status_code),
        "requested_model_tier": _model_family(requested_model),
        "routed_model_tier": _model_family(routed_model),
        "target_model_tier": _model_family(canary.get("target_model")),
        "policy_id": canary.get("policy_id"),
        "policy_source": canary.get("policy_source") or routing_meta.get("policy_source"),
        "cohort": canary.get("cohort"),
        "canary": {
            "enabled": bool(canary.get("enabled")),
            "status": status or None,
            "cohort": canary.get("cohort"),
            "fraction": canary.get("canary_fraction"),
            "holdout_fraction": canary.get("holdout_fraction"),
            "cohort_hash": canary.get("cohort_hash"),
        },
        "dimensions": {
            "text_bucket": canary.get("text_bucket") or _text_bucket(routing_meta.get("text_chars")),
            "input_token_bucket": _token_bucket(actual_input_tokens if actual_input_tokens is not None else input_tokens_est),
            "output_token_bucket": _token_bucket(actual_output_tokens if actual_output_tokens is not None else output_tokens_est),
            "has_tools": bool(canary.get("has_tools") if canary.get("has_tools") is not None else routing_meta.get("has_tools")),
            "thinking": bool(thinking_output_tokens or routing_meta.get("workflow_phase") == "thinking"),
            "stream": None,
        },
        "provider_outcome": {
            "status_code": status_code,
            "status_bucket": "ok" if status_code is not None and int(status_code) < 400 else ("unknown" if status_code is None else f"http_{int(status_code)}"),
            "latency_bucket": _latency_bucket(latency_ms),
            "retry_bucket": _retry_bucket(retry_count),
            "fallback_reason": routing_meta.get("fallback_reason") or canary.get("fallback_reason"),
            "cache_status": cache_meta.get("status"),
            "crunch_changed": bool(crunch_meta.get("changed")),
        },
        "cost": {
            "input_tokens_bucket": _token_bucket(actual_input_tokens if actual_input_tokens is not None else input_tokens_est),
            "output_tokens_bucket": _token_bucket(actual_output_tokens if actual_output_tokens is not None else output_tokens_est),
            "cost_bucket": _net_savings_bucket(cost_est_usd),
            "baseline_cost_bucket": _net_savings_bucket(cost_baseline_usd),
            "savings_bucket": _phase_savings_bucket(cost_est_usd, cost_baseline_usd),
        },
        "quality_signals": {
            "schema": quality_signals.get("schema"),
            "status": quality_signals.get("status"),
            "optimized": bool(quality_signals.get("optimized")),
            "signal_codes": quality_signals.get("signal_codes") or [],
            "risk_level": quality_signals.get("risk_level"),
        },
        "safety_stop": {
            "enabled": bool(safety_stop.get("enabled")),
            "status": safety_stop.get("status"),
            "tripped": bool(safety_stop.get("tripped")),
            "reason_codes": safety_stop.get("reason_codes") or [],
            "sample_count": safety_stop.get("sample_count"),
            "holdout_sample_count": safety_stop.get("holdout_sample_count"),
        } if safety_stop else None,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "tool_payloads_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "secrets_included": False,
        },
    }
    error_class = _compact_error(error, status_code).get("error_class") if error else None
    if error_class:
        feedback["provider_outcome"]["error_class"] = error_class
    return _sanitize_features({key: value for key, value in feedback.items() if value not in (None, "", [], {})})


def build_phase_routing_outcome_event(
    outcome_feedback: dict[str, Any],
    *,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    from tokenclaw import __version__

    basis = {
        "policy_id": outcome_feedback.get("policy_id"),
        "workflow_phase": outcome_feedback.get("workflow_phase"),
        "category": outcome_feedback.get("category"),
        "cohort": outcome_feedback.get("cohort"),
        "outcome": outcome_feedback.get("outcome"),
    }
    digest = hashlib.sha256(stable_json(basis).encode("utf-8")).hexdigest()[:24]
    return _sanitize_features({
        "event_type": "outcome",
        "occurred_at": occurred_at or utc_now(),
        "recommendation_id": f"phase-routing:{digest}",
        "bundle_hash": None,
        "policy_sections": ["routing"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": {
            "schema": "tokenclaw.phase_routing_outcome_event_metadata.v1",
            "lifecycle_kind": "phase_routing",
            "outcome": outcome_feedback,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "raw_responses_included": False,
                "raw_transcripts_included": False,
                "tool_payloads_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "local_session_ids_included": False,
                "file_paths_included": False,
                "cache_keys_included": False,
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
    pattern_decisions = pattern_decision_summaries(
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
        "pattern_decisions": pattern_decisions,
        "pattern_policy_evidence": pattern_policy_evidence_summaries(
            provider="openai",
            path="codex-app://turn/start",
            requested_model=routing_meta.get("requested_model") or codex_app_model(),
            routed_model=routing_meta.get("routed_model") or routing_meta.get("requested_model") or codex_app_model(),
            status_code=500 if error_code is not None else 200,
            latency_ms=latency_ms,
            retry_count=0,
            cost_est_usd=cost_est,
            cost_baseline_usd=baseline_cost,
            cache_meta=cache_meta,
            crunch_meta=crunch_meta,
            routing_meta=routing_meta,
            category=routing_meta.get("category") or "codex_turn",
            pattern_decisions=pattern_decisions,
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
    terminal_log_features = _terminal_log_features_from_routing_meta(routing_meta)
    if terminal_log_features is not None:
        features["terminal_log_features"] = terminal_log_features
    prompt_difficulty_features = _prompt_difficulty_features_from_routing_meta(routing_meta)
    if prompt_difficulty_features is not None:
        features["prompt_difficulty_features"] = prompt_difficulty_features
    if error_code is not None:
        features["error_class"] = "jsonrpc_error"
    return _sanitize_features(features)


def _codex_lifecycle_outcome(
    *,
    action_family: str,
    routing_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    error_code: int | None,
) -> str:
    if error_code is not None:
        return "error"
    if action_family == "routing":
        status = str(routing_meta.get("status") or "")
        reason = str(routing_meta.get("reason") or "")
        safety_stop = routing_meta.get("safety_stop") if isinstance(routing_meta.get("safety_stop"), dict) else {}
        if status == "safety_stopped" or reason == "local-canary-safety-stop" or safety_stop.get("tripped"):
            return "safety_stopped"
        if status == "applied":
            return "applied"
        if status == "holdout" or reason == "summary-model-hint-canary-holdout":
            return "holdout"
        if status in {"skipped", "eligible-skipped"}:
            return "skipped"
        return status or "unknown"
    status = str(cache_meta.get("status") or "")
    reason = str(cache_meta.get("reason") or "")
    safety_stop = cache_meta.get("safety_stop") if isinstance(cache_meta.get("safety_stop"), dict) else {}
    if reason == "local-canary-safety-stop" or safety_stop.get("tripped"):
        return "safety_stopped"
    if status == "hit":
        return "hit"
    if status == "miss":
        return "invalidated" if reason in {"dependency-changed", "dependency-deleted", "codex-cache-ttl-expired"} else "miss"
    if status == "holdout" or reason == "codex-app-cache-canary-holdout":
        return "holdout"
    if status in {"skipped", "unsafe-skip"}:
        return "skipped"
    return status or "unknown"


def _codex_canary_for_action(action_family: str, routing_meta: dict[str, Any], cache_meta: dict[str, Any]) -> dict[str, Any]:
    if action_family == "routing":
        sample = routing_meta.get("canary_sample") if isinstance(routing_meta.get("canary_sample"), dict) else {}
        policy = routing_meta.get("canary_policy") if isinstance(routing_meta.get("canary_policy"), dict) else {}
        name = "codex-app-rule" if routing_meta.get("canary") == "codex-app-rule" else "codex-app-summary-model-hint"
        return {
            "name": name,
            "enabled": bool(routing_meta.get("canary_enabled") or sample.get("enabled") or routing_meta.get("canary") == "codex-app-rule"),
            "cohort": routing_meta.get("canary_cohort") or sample.get("cohort"),
            "status": sample.get("status") or routing_meta.get("status"),
            "fraction": sample.get("fraction") if sample.get("fraction") is not None else policy.get("fraction"),
            "holdout_fraction": (
                sample.get("holdout_fraction")
                if sample.get("holdout_fraction") is not None
                else policy.get("holdout_fraction")
            ),
            "sample_unit": sample.get("sample_unit") or policy.get("sample_unit"),
            "raw_basis_included": False,
        }
    sample = cache_meta.get("canary_sample") if isinstance(cache_meta.get("canary_sample"), dict) else {}
    name = (
        "codex-app-rule"
        if cache_meta.get("canary") == "codex-app-rule" or isinstance(cache_meta.get("codex_app_rule"), dict)
        else "codex-app-exact-cache"
    )
    return {
        "name": name,
        "enabled": bool(sample.get("enabled")),
        "cohort": cache_meta.get("canary_cohort") or sample.get("cohort"),
        "status": sample.get("status") or cache_meta.get("status"),
        "fraction": sample.get("fraction"),
        "holdout_fraction": sample.get("holdout_fraction"),
        "sample_unit": sample.get("sample_unit"),
        "raw_basis_included": False,
    }


def build_codex_app_canary_lifecycle_feedback(
    *,
    action_family: str,
    routing_meta: dict[str, Any],
    crunch_meta: dict[str, Any],
    cache_meta: dict[str, Any],
    result_chars: int | None,
    error_code: int | None,
    error_message: str | None,
    latency_ms: int | None,
    input_text_chars: int | None,
) -> dict[str, Any] | None:
    if action_family not in {"routing", "cache"}:
        return None
    if action_family == "routing":
        if routing_meta.get("canary") not in {"codex-app-summary-model-hint", "codex-app-rule"}:
            return None
        rule_meta = routing_meta.get("codex_app_rule") if isinstance(routing_meta.get("codex_app_rule"), dict) else {}
        policy_id = (
            rule_meta.get("policy_id")
            or routing_meta.get("policy_id")
            or rule_meta.get("candidate_id")
            or rule_meta.get("rule_id")
            or "local-codex-app-summary-model-hint-canary"
        )
        rule_id = rule_meta.get("rule_id")
        candidate_id = rule_meta.get("candidate_id") or policy_id
        target_model = routing_meta.get("target_model") or routing_meta.get("routed_model")
        decision_status = routing_meta.get("status")
        decision_reason = routing_meta.get("reason")
        replayability_level = "features_only"
        cache_status = cache_meta.get("status")
    else:
        if cache_meta.get("canary") != "codex-app-exact-cache" and not isinstance(cache_meta.get("canary_sample"), dict):
            return None
        rule_meta = cache_meta.get("codex_app_rule") if isinstance(cache_meta.get("codex_app_rule"), dict) else {}
        policy_id = (
            rule_meta.get("policy_id")
            or rule_meta.get("candidate_id")
            or rule_meta.get("rule_id")
            or "local-codex-app-exact-cache-canary"
        )
        rule_id = rule_meta.get("rule_id")
        candidate_id = rule_meta.get("candidate_id") or policy_id
        target_model = routing_meta.get("routed_model") or routing_meta.get("requested_model")
        decision_status = cache_meta.get("status")
        decision_reason = cache_meta.get("reason")
        replayability_level = cache_meta.get("replayability_level") or "features_only"
        cache_status = cache_meta.get("status")

    input_tokens_est = max(0, int((input_text_chars or 0) / TOKEN_CHARS))
    output_tokens_est = max(0, int((result_chars or 0) / TOKEN_CHARS))
    requested_model = routing_meta.get("requested_model")
    routed_model = routing_meta.get("routed_model") or requested_model
    routed_cost = estimate_cost(
        str(routed_model or codex_app_model()),
        input_tokens_est,
        output_tokens_est,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    requested_cost = estimate_cost(
        str(requested_model or routed_model or codex_app_model()),
        input_tokens_est,
        output_tokens_est,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    )
    cost_delta = (
        float(requested_cost) - float(routed_cost)
        if requested_cost is not None and routed_cost is not None
        else None
    )
    error_bits = _compact_error(error_message, 500 if error_code is not None else 200) if error_code is not None else {}
    safety_stop = routing_meta.get("safety_stop") if isinstance(routing_meta.get("safety_stop"), dict) else {}
    canary = _codex_canary_for_action(action_family, routing_meta, cache_meta)
    feedback = {
        "schema": "tokenclaw.codex_app_canary_lifecycle_feedback.v1",
        "event_type": "codex_app_canary_lifecycle",
        "occurred_at": utc_now(),
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "app_family": "codex",
        "lifecycle_kind": "codex_app_canary",
        "action_family": action_family,
        "policy_id": policy_id,
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "workflow_phase": (
            routing_meta.get("workflow_phase")
            or cache_meta.get("workflow_phase")
            or routing_meta.get("category")
            or "unknown"
        ),
        "workflow_phase_reason": routing_meta.get("workflow_phase_reason") or cache_meta.get("workflow_phase_reason"),
        "model_family_bucket": _model_family(routed_model or requested_model),
        "requested_model_family": _model_family(requested_model),
        "target_model_family": _model_family(target_model),
        "dimensions": {
            "text_bucket": _text_bucket(input_text_chars),
            "input_token_bucket": _token_bucket(input_tokens_est),
            "output_token_bucket": _token_bucket(output_tokens_est),
            "model_field_state": "present" if routing_meta.get("model_field") else "unknown",
            "cache_status": cache_status,
            "replayability_level": replayability_level,
        },
        "canary": canary,
        "canary_cohort": canary.get("cohort"),
        "decision": {
            "status": decision_status,
            "reason": decision_reason,
            "policy_source": (
                routing_meta.get("policy_source")
                if action_family == "routing"
                else cache_meta.get("policy_source")
            ),
            "rule_path_included": False,
            "applied": bool(routing_meta.get("applied") if action_family == "routing" else cache_meta.get("status") == "hit"),
            "cache_key_included": False,
        },
        "outcome": {
            "status": _codex_lifecycle_outcome(
                action_family=action_family,
                routing_meta=routing_meta,
                cache_meta=cache_meta,
                error_code=error_code,
            ),
            "status_class": "error" if error_code is not None else "success",
            "jsonrpc_error": bool(error_code is not None),
            "latency_bucket": _latency_bucket(latency_ms),
            "cost_delta_bucket": _net_savings_bucket(cost_delta),
            "cache_status": cache_status,
            "crunch_changed": bool(crunch_meta.get("changed")),
            "safety_stop_reasons": safety_stop.get("reason_codes") or safety_stop.get("trigger_metrics") or [],
        },
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_params_included": False,
            "raw_transcripts_included": False,
            "raw_commands_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "thread_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "secrets_included": False,
        },
    }
    if error_bits:
        feedback["outcome"]["error_bucket"] = error_bits.get("error_class")
    return _sanitize_features({key: value for key, value in feedback.items() if value not in (None, "", [], {})})


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
    try:
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update(managed_egress_blocked_meta(
            endpoint=endpoint,
            violations=exc.violations,
            optimization_unit_id=unit_id,
        ))
        return meta

    if not recommendation_server_configured():
        meta.update({
            "status": "skipped",
            "reason": "server-url-not-configured",
        })
        return meta
    url = recommendation_server_url() + endpoint
    started = time.time()
    try:
        async with async_client(timeout=recommendation_timeout_seconds()) as client:
            if endpoint == PROMOTION_BLOCKER_ACTION_OUTCOME_ROLLUPS_PATH:
                response = await client.post(url, json=payload, headers=_managed_headers())
            else:
                response = await client.patch(url, json=payload, headers=_managed_headers())
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        retry_after = _response_retry_after_seconds(response)
        if retry_after is not None:
            meta["retry_after_seconds"] = retry_after
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

    try:
        assert_managed_egress_safe(outcome_features)
    except ManagedEgressBlocked as exc:
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "timeout_seconds": recommendation_timeout_seconds(),
            "failure_mode": recommendation_failure_mode(),
            "auth_configured": managed_auth_configured(),
            "api_key_value_included": False,
            **managed_egress_blocked_meta(
                endpoint=endpoint,
                violations=exc.violations,
                optimization_unit_id=unit_id,
            ),
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
    try:
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        meta.update(managed_egress_blocked_meta(endpoint=POLICY_EVENTS_PATH, violations=exc.violations))
        return meta
    if not recommendation_server_configured():
        meta.update({"status": "skipped", "reason": "server-url-not-configured"})
        return meta

    started = time.time()
    try:
        async with async_client(timeout=recommendation_timeout_seconds()) as client:
            response = await client.post(
                recommendation_server_url() + POLICY_EVENTS_PATH,
                json=_sanitize_features(payload),
                headers=_managed_headers(),
            )
        meta["latency_ms"] = int((time.time() - started) * 1000)
        meta["status_code"] = response.status_code
        retry_after = _response_retry_after_seconds(response)
        if retry_after is not None:
            meta["retry_after_seconds"] = retry_after
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
    status_code = meta.get("status_code")
    try:
        code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        code = None
    if code is not None and 400 <= code < 500 and code not in {408, 409, 425, 429}:
        return "dropped-after-limit"
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

    try:
        assert_managed_egress_safe(payload)
    except ManagedEgressBlocked as exc:
        status = "dropped-after-limit"
        error = "unsafe managed egress payload blocked"
        if hasattr(store_obj, "mark_managed_outcome_feedback_retry"):
            store_obj.mark_managed_outcome_feedback_retry(
                queue_id,
                status=status,
                error=error,
                status_code=None,
                next_attempt_at=utc_now(),
            )
        meta = managed_egress_blocked_meta(
            endpoint=endpoint,
            violations=exc.violations,
            optimization_unit_id=int(unit_id) if isinstance(unit_id, int) else None,
            queue_id=queue_id,
        )
        meta.update({
            "enabled": True,
            "server_url": recommendation_server_url(),
            "status": status,
            "attempts": attempts,
        })
        return meta

    if endpoint == POLICY_EVENTS_PATH:
        meta = await _send_policy_event_payload(payload)
    else:
        meta = await _send_outcome_payload(endpoint=endpoint, payload=payload, unit_id=int(unit_id))
    if meta.get("status") == "sent":
        if hasattr(store_obj, "mark_managed_outcome_feedback_sent"):
            store_obj.mark_managed_outcome_feedback_sent(queue_id, status_code=meta.get("status_code"))
        meta.update({
            "queue_id": queue_id,
            "attempts": attempts,
        })
        return meta

    status = _queue_error_status(meta, attempts)
    retry_after = meta.get("retry_after_seconds")
    try:
        retry_after_delay = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_after_delay = None
    retry_delay = 0.0 if status == "dropped-after-limit" else (
        retry_after_delay if retry_after_delay is not None else outcome_feedback_queue_retry_delay_for_attempt(attempts)
    )
    status_code = meta.get("status_code")
    permanent_client_error = False
    try:
        code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        code = None
    if code is not None and 400 <= code < 500 and code not in {408, 409, 425, 429}:
        permanent_client_error = True
    if permanent_client_error:
        reason = "permanent-client-error"
    elif status == "dropped-after-limit":
        reason = "attempt-limit-reached"
    else:
        reason = meta.get("reason") or "request-failed"
    error = meta.get("error")
    if not error:
        error = f"{meta.get('status') or 'error'}: {reason}"
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


async def flush_selected_outcome_feedback(
    store_obj: Any,
    *,
    queue_ids: list[str],
) -> list[dict[str, Any]]:
    if not recommendations_enabled():
        return []
    if not hasattr(store_obj, "claim_managed_outcome_feedback"):
        return []
    results: list[dict[str, Any]] = []
    for queue_id in queue_ids:
        claimed = store_obj.claim_managed_outcome_feedback(str(queue_id), now=utc_now())
        if not claimed:
            continue
        results.append(await _flush_claimed_outcome_feedback(store_obj, claimed))
    return results


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

    try:
        assert_managed_egress_safe(outcome_features)
    except ManagedEgressBlocked as exc:
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "timeout_seconds": recommendation_timeout_seconds(),
            "failure_mode": recommendation_failure_mode(),
            "auth_configured": managed_auth_configured(),
            "api_key_value_included": False,
            **managed_egress_blocked_meta(
                endpoint=endpoint,
                violations=exc.violations,
                optimization_unit_id=unit_id,
            ),
        }

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
    queue_when_disabled: bool = False,
    flush_immediately: bool = True,
) -> dict[str, Any]:
    if not recommendations_enabled():
        if queue_when_disabled and hasattr(store_obj, "enqueue_managed_outcome_feedback"):
            try:
                assert_managed_egress_safe(event_payload)
            except ManagedEgressBlocked as exc:
                return {
                    "enabled": False,
                    "server_url": recommendation_server_url(),
                    "timeout_seconds": recommendation_timeout_seconds(),
                    "failure_mode": recommendation_failure_mode(),
                    "auth_configured": managed_auth_configured(),
                    "api_key_value_included": False,
                    **managed_egress_blocked_meta(endpoint=POLICY_EVENTS_PATH, violations=exc.violations),
                }
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
            stored = store_obj.get_managed_outcome_feedback(queue_id) if hasattr(store_obj, "get_managed_outcome_feedback") else row
            meta = _queued_meta(stored or row)
            meta.update({
                "enabled": False,
                "reason": "queued-managed-disabled",
            })
            return meta
        return {
            **disabled_outcome_feedback_meta(),
            "endpoint": POLICY_EVENTS_PATH,
            "status": "disabled",
        }
    if not hasattr(store_obj, "enqueue_managed_outcome_feedback"):
        return await _send_policy_event_payload(event_payload)

    try:
        assert_managed_egress_safe(event_payload)
    except ManagedEgressBlocked as exc:
        return {
            "enabled": True,
            "server_url": recommendation_server_url(),
            "timeout_seconds": recommendation_timeout_seconds(),
            "failure_mode": recommendation_failure_mode(),
            "auth_configured": managed_auth_configured(),
            "api_key_value_included": False,
            **managed_egress_blocked_meta(endpoint=POLICY_EVENTS_PATH, violations=exc.violations),
        }

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
    if not flush_immediately:
        stored = store_obj.get_managed_outcome_feedback(queue_id) if hasattr(store_obj, "get_managed_outcome_feedback") else row
        return _queued_meta(stored or row)
    claimed = (
        store_obj.claim_managed_outcome_feedback(queue_id, now=utc_now())
        if hasattr(store_obj, "claim_managed_outcome_feedback")
        else None
    )
    if not claimed:
        stored = store_obj.get_managed_outcome_feedback(queue_id) if hasattr(store_obj, "get_managed_outcome_feedback") else row
        return _queued_meta(stored or row)
    return await _flush_claimed_outcome_feedback(store_obj, claimed)


async def queue_codex_app_canary_lifecycle_feedback(
    store_obj: Any,
    event_payload: dict[str, Any],
    *,
    flush_immediately: bool = False,
) -> dict[str, Any]:
    return await queue_policy_event_feedback(
        store_obj,
        event_payload,
        source_surface=CODEX_APP_CANARY_LIFECYCLE_SOURCE_SURFACE,
        flush_immediately=flush_immediately,
    )


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
