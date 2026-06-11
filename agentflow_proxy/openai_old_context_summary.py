from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from agentflow_proxy.openai_old_context_summary_dry_run import (
    DEFAULT_BLOCKED_CATEGORIES,
    DEFAULT_KEEP_RECENT_ITEMS,
    DEFAULT_MAX_SOURCE_CHARS,
    DEFAULT_MAX_SUMMARY_CHARS,
    DEFAULT_MAX_SUMMARY_COST_USD,
    DEFAULT_MIN_REQUEST_CHARS,
    DEFAULT_MIN_SOURCE_CHARS,
    DEFAULT_SUMMARY_COMPRESSION_RATIO,
    DEFAULT_SUMMARY_MODEL,
    DEFAULT_SUMMARY_PROVIDER,
    DEFAULT_SUPPORTED_ENDPOINTS,
    INSTRUCTION_ROLES,
    TOKEN_CHARS,
    _as_float,
    _as_int,
    _has_file_reference,
    _has_tool_protocol,
    _metadata_text_chars,
    _projection,
)
from agentflow_proxy.optimization.openai_features import openai_endpoint
from agentflow_proxy.store import stable_json


SCHEMA = "agentflow.openai_old_context_summary.v1"
SUMMARY_CACHE_SCHEMA = "agentflow.openai_old_context_summary_cache.v1"
SUMMARY_MARKER = "AgentFlow summary of earlier OpenAI context"

SummaryFetcher = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
CacheGetter = Callable[[str], dict[str, Any] | None]
CacheSetter = Callable[[str, dict[str, Any]], None]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _env_bool(name: str, default: bool) -> bool:
    return _as_bool(os.getenv(name), default)


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    candidates.append(Path(__file__).parent / filename)
    return candidates


def default_openai_old_context_summary_policy() -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_old_context_summary_policy.v1",
        "enabled": False,
        "rule_id": "local-openai-old-context-summary",
        "summary_provider": DEFAULT_SUMMARY_PROVIDER,
        "summary_model": DEFAULT_SUMMARY_MODEL,
        "min_request_chars": DEFAULT_MIN_REQUEST_CHARS,
        "min_source_chars": DEFAULT_MIN_SOURCE_CHARS,
        "max_source_chars": DEFAULT_MAX_SOURCE_CHARS,
        "keep_recent_items": DEFAULT_KEEP_RECENT_ITEMS,
        "max_summary_chars": DEFAULT_MAX_SUMMARY_CHARS,
        "summary_compression_ratio": DEFAULT_SUMMARY_COMPRESSION_RATIO,
        "max_summary_cost_usd": DEFAULT_MAX_SUMMARY_COST_USD,
        "supported_endpoints": list(DEFAULT_SUPPORTED_ENDPOINTS),
        "blocked_categories": list(DEFAULT_BLOCKED_CATEGORIES),
        "block_tool_protocol": True,
        "block_file_references": True,
        "canary": {
            "enabled": True,
            "canary_fraction": 0.0,
            "holdout_fraction": 1.0,
            "salt": "",
            "cohort_basis": "deterministic-candidate-id-hash",
        },
        "policy_source": "local-default",
        "rule_path": None,
    }


def _string_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(default)


def _apply_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> None:
    for key in ("enabled", "block_tool_protocol", "block_file_references"):
        if data.get(key) is not None:
            policy[key] = _as_bool(data.get(key), bool(policy[key]))
    for key in ("rule_id", "summary_provider", "summary_model"):
        if data.get(key) is not None:
            policy[key] = str(data[key])
    if data.get("model") is not None:
        policy["summary_model"] = str(data["model"])
    for key in (
        "min_request_chars",
        "min_source_chars",
        "max_source_chars",
        "keep_recent_items",
        "max_summary_chars",
    ):
        if data.get(key) is not None:
            policy[key] = max(0, int(data[key]))
    for key in ("summary_compression_ratio", "max_summary_cost_usd"):
        if data.get(key) is not None:
            policy[key] = max(0.0, float(data[key]))
    policy["supported_endpoints"] = _string_list(
        data.get("supported_endpoints") or data.get("endpoints"),
        policy["supported_endpoints"],
    )
    policy["blocked_categories"] = _string_list(
        data.get("blocked_categories") or data.get("excluded_categories"),
        policy["blocked_categories"],
    )

    canary = data.get("canary") if isinstance(data.get("canary"), dict) else {}
    target_canary = policy["canary"]
    if canary:
        target_canary["enabled"] = _as_bool(canary.get("enabled"), bool(target_canary["enabled"]))
        for source_key, target_key in (
            ("canary_fraction", "canary_fraction"),
            ("fraction", "canary_fraction"),
            ("rollout_fraction", "canary_fraction"),
            ("holdout_fraction", "holdout_fraction"),
        ):
            if canary.get(source_key) is not None:
                target_canary[target_key] = max(0.0, min(1.0, float(canary[source_key])))
        if canary.get("salt") is not None:
            target_canary["salt"] = str(canary["salt"])


def load_openai_old_context_summary_policy() -> dict[str, Any]:
    policy = default_openai_old_context_summary_policy()
    for path in _manual_rule_candidates("crunch_rules.yaml", "AGENTFLOW_CRUNCH_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        section = (
            data.get("openai_old_context_summarization")
            or data.get("openai_old_context_summary")
            or (
                (data.get("openai") or {}).get("old_context_summarization")
                if isinstance(data.get("openai"), dict)
                else None
            )
        )
        if isinstance(section, dict):
            _apply_policy_yaml(policy, section)
            policy["policy_source"] = (
                "local-manual"
                if path != Path(__file__).parent / "crunch_rules.yaml"
                else "local-default"
            )
            policy["rule_path"] = str(path)
            break

    if os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED") is not None:
        policy["enabled"] = _env_bool("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_ENABLED", bool(policy["enabled"]))
    if os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MODEL"):
        policy["summary_model"] = str(os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MODEL"))
    env_thresholds = {
        "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MIN_REQUEST_CHARS": "min_request_chars",
        "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MIN_SOURCE_CHARS": "min_source_chars",
        "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SOURCE_CHARS": "max_source_chars",
        "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_KEEP_RECENT_ITEMS": "keep_recent_items",
        "AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_CHARS": "max_summary_chars",
    }
    for env_name, key in env_thresholds.items():
        if os.getenv(env_name) is not None:
            policy[key] = max(0, int(os.getenv(env_name, "0")))
    if os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_COST_USD") is not None:
        policy["max_summary_cost_usd"] = max(
            0.0,
            float(os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_COST_USD", "0")),
        )
    if os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_CANARY_FRACTION") is not None:
        policy["canary"]["canary_fraction"] = max(
            0.0,
            min(1.0, float(os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_CANARY_FRACTION", "0"))),
        )
    if os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_HOLDOUT_FRACTION") is not None:
        policy["canary"]["holdout_fraction"] = max(
            0.0,
            min(1.0, float(os.getenv("AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_HOLDOUT_FRACTION", "1"))),
        )
    return policy


def _summary_meta_base(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "enabled": bool(policy.get("enabled")),
        "status": "disabled" if not policy.get("enabled") else "not_evaluated",
        "applied": False,
        "changed": False,
        "reason_codes": ["disabled"] if not policy.get("enabled") else [],
        "policy_source": str(policy.get("policy_source") or "local-default"),
        "rule_id": str(policy.get("rule_id") or "local-openai-old-context-summary"),
        "rule_path": policy.get("rule_path"),
        "summary_provider": str(policy.get("summary_provider") or DEFAULT_SUMMARY_PROVIDER),
        "summary_model": str(policy.get("summary_model") or DEFAULT_SUMMARY_MODEL),
        "summary_cache_hit": False,
        "summary_cost_est_usd": 0.0,
        "estimated_tokens_saved": 0,
        "estimated_chars_saved": 0,
        "estimated_gross_savings_usd": 0.0,
        "estimated_net_savings_usd": 0.0,
        "preservation": {
            "system_developer_instructions_preserved": True,
            "recent_items_preserved": 0,
            "tool_function_protocol_preserved": True,
            "attachments_file_references_preserved": True,
            "response_format_constraints_preserved": True,
            "streaming_compatible": True,
        },
        "privacy": {
            "raw_source_included": False,
            "raw_summary_included": False,
            "raw_request_body_included": False,
            "summary_text_included": False,
            "tool_payloads_included": False,
            "function_arguments_included": False,
            "file_paths_included": False,
            "cache_key_included": False,
            "session_id_included": False,
        },
    }


def _endpoint_for_path(path: str) -> str:
    return openai_endpoint(path)


def _copy_item(item: Any) -> Any:
    return copy.deepcopy(item)


def _source_and_preserved(body: dict[str, Any], endpoint: str, keep_recent: int) -> dict[str, Any]:
    key = "messages" if endpoint == "chat_completions" else "input"
    items = body.get(key)
    if not isinstance(items, list):
        return {"supported": False, "reason": "unsupported_request_shape", "key": key}
    old_limit = max(0, len(items) - keep_recent)
    source: list[Any] = []
    preserved: list[Any] = []
    insert_at = 0
    source_started = False
    instruction_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return {"supported": False, "reason": "unsupported_request_shape", "key": key}
        role = str(item.get("role") or "").lower()
        if role in INSTRUCTION_ROLES:
            instruction_count += 1
            preserved.append(_copy_item(item))
            if not source_started:
                insert_at = len(preserved)
            continue
        if index < old_limit:
            source_started = True
            source.append(_copy_item(item))
            continue
        preserved.append(_copy_item(item))
    return {
        "supported": True,
        "key": key,
        "source_items": source,
        "source_item_count": len(source),
        "preserved_items": preserved,
        "insert_at": insert_at,
        "conversation_item_count": len(items),
        "recent_item_count": min(keep_recent, len(items)),
        "preserved_instruction_count": instruction_count,
        "response_format_present": bool(body.get("text") or body.get("response_format")),
        "top_level_instruction_present": bool(body.get("instructions")),
    }


def _source_hash(source_items: list[Any]) -> str:
    return hashlib.sha256(stable_json(source_items).encode("utf-8")).hexdigest()


def _candidate_id(*, source_hash: str, endpoint: str, requested_model: str, policy: dict[str, Any]) -> str:
    salt = str((policy.get("canary") or {}).get("salt") or "")
    basis = "|".join([salt, str(policy.get("rule_id") or ""), endpoint, requested_model, source_hash])
    return "openai-old-context-summary-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def _cohort(candidate_id: str, policy: dict[str, Any]) -> str:
    canary = policy.get("canary") if isinstance(policy.get("canary"), dict) else {}
    if not _as_bool(canary.get("enabled"), True):
        return "not_selected"
    canary_fraction = _as_float(canary.get("canary_fraction"))
    holdout_fraction = _as_float(canary.get("holdout_fraction"), 1.0)
    value = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < canary_fraction:
        return "canary_applied"
    if value < canary_fraction + holdout_fraction:
        return "holdout"
    return "not_selected"


def _summary_cache_key(*, source_hash: str, policy: dict[str, Any]) -> str:
    basis = "|".join([
        SCHEMA,
        str(policy.get("summary_model") or DEFAULT_SUMMARY_MODEL),
        str(policy.get("max_summary_chars") or DEFAULT_MAX_SUMMARY_CHARS),
        source_hash,
    ])
    return "openai-old-context-summary:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _summary_prompt(source_items: list[Any], max_summary_chars: int, max_source_chars: int) -> str:
    source_payload = stable_json(source_items)
    if max_source_chars > 0 and len(source_payload) > max_source_chars:
        source_payload = source_payload[:max_source_chars].rstrip()
    return "\n".join([
        "Summarize the older OpenAI conversation context below for a coding agent.",
        "Preserve durable user intent, decisions, constraints, unresolved tasks, and facts needed to continue.",
        "Do not include tool/function call payloads, file attachments, or secrets beyond what is necessary "
        "for continuity.",
        f"Keep the summary under {max_summary_chars} characters.",
        "",
        source_payload,
    ])


def build_summary_request(source_items: list[Any], policy: dict[str, Any]) -> dict[str, Any]:
    max_summary_chars = _as_int(policy.get("max_summary_chars"), DEFAULT_MAX_SUMMARY_CHARS)
    max_source_chars = _as_int(policy.get("max_source_chars"), DEFAULT_MAX_SOURCE_CHARS)
    return {
        "model": str(policy.get("summary_model") or DEFAULT_SUMMARY_MODEL),
        "input": _summary_prompt(source_items, max_summary_chars, max_source_chars),
        "max_output_tokens": max(1, max_summary_chars // TOKEN_CHARS),
    }


def _summary_item(endpoint: str, summary_text: str) -> dict[str, Any]:
    text = (
        f"{SUMMARY_MARKER}:\n\n"
        f"{summary_text.strip()}\n\n"
        "Recent unsummarized messages below remain authoritative."
    )
    if endpoint == "chat_completions":
        return {"role": "system", "content": text}
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _apply_summary_body(
    body: dict[str, Any],
    endpoint: str,
    window: dict[str, Any],
    summary_text: str,
) -> dict[str, Any]:
    new_body = copy.deepcopy(body)
    key = str(window["key"])
    preserved = [_copy_item(item) for item in window["preserved_items"]]
    insert_at = max(0, min(_as_int(window.get("insert_at")), len(preserved)))
    preserved.insert(insert_at, _summary_item(endpoint, summary_text))
    new_body[key] = preserved
    return new_body


def _text_chars(body: dict[str, Any]) -> int:
    return _metadata_text_chars(body)


async def maybe_apply_openai_old_context_summary(
    *,
    body: dict[str, Any],
    path: str,
    requested_model: str,
    category: str | None,
    stream: bool,
    fetch_summary: SummaryFetcher,
    get_cached_summary: CacheGetter,
    set_cached_summary: CacheSetter,
    policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = copy.deepcopy(policy or load_openai_old_context_summary_policy())
    meta = _summary_meta_base(policy)
    if not policy.get("enabled"):
        return body, meta

    endpoint = _endpoint_for_path(path)
    meta["endpoint"] = endpoint
    reason_codes: list[str] = []
    supported = {str(item) for item in policy.get("supported_endpoints") or []}
    if endpoint not in supported:
        reason_codes.append("unsupported_endpoint")
    if str(category or "unknown") in {str(item) for item in policy.get("blocked_categories") or []}:
        reason_codes.append("blocked_category")
    request_chars = _text_chars(body)
    if request_chars < _as_int(policy.get("min_request_chars"), DEFAULT_MIN_REQUEST_CHARS):
        reason_codes.append("request_below_min_chars")

    window = _source_and_preserved(body, endpoint, _as_int(policy.get("keep_recent_items"), DEFAULT_KEEP_RECENT_ITEMS))
    if not window.get("supported"):
        reason_codes.append(str(window.get("reason") or "unsupported_request_shape"))
    source_items = window.get("source_items") if isinstance(window.get("source_items"), list) else []
    source_chars = _metadata_text_chars(source_items)
    if source_chars < _as_int(policy.get("min_source_chars"), DEFAULT_MIN_SOURCE_CHARS):
        reason_codes.append("source_below_min_chars")
    if source_chars > _as_int(policy.get("max_source_chars"), DEFAULT_MAX_SOURCE_CHARS):
        source_items = source_items[:]
    if _as_bool(policy.get("block_tool_protocol"), True) and (
        _has_tool_protocol(source_items)
        or _has_tool_protocol(body.get("tools"))
        or _has_tool_protocol(body.get("functions"))
    ):
        reason_codes.append("tool_function_protocol_ambiguous")
    if _as_bool(policy.get("block_file_references"), True) and _has_file_reference(source_items):
        reason_codes.append("file_reference_in_source_window")

    projection = _projection(
        requested_model=requested_model,
        endpoint=endpoint,
        source_chars=source_chars,
        source_item_count=len(source_items),
        policy=policy,
    )
    meta.update({
        "request_chars": request_chars,
        "source_item_count": len(source_items),
        "source_chars": min(source_chars, _as_int(policy.get("max_source_chars"), DEFAULT_MAX_SOURCE_CHARS)),
        "estimated_chars_saved": projection["expected_saved_chars"],
        "estimated_tokens_saved": projection["expected_saved_tokens"],
        "estimated_gross_savings_usd": projection["projected_gross_savings_usd"],
        "estimated_net_savings_usd": projection["projected_net_savings_usd"],
        "projected_summary_cost_usd": projection["estimated_summary_cost_usd"],
    })
    if projection["estimated_summary_cost_usd"] > _as_float(
        policy.get("max_summary_cost_usd"),
        DEFAULT_MAX_SUMMARY_COST_USD,
    ):
        reason_codes.append("summary_cost_over_budget")

    source_digest = _source_hash(source_items) if source_items else ""
    candidate_id = _candidate_id(
        source_hash=source_digest,
        endpoint=endpoint,
        requested_model=requested_model,
        policy=policy,
    )
    cohort = _cohort(candidate_id, policy)
    meta["candidate_id"] = candidate_id
    meta["canary"] = {
        "cohort": cohort if not reason_codes else "blocked",
        "canary_fraction": _as_float((policy.get("canary") or {}).get("canary_fraction")),
        "holdout_fraction": _as_float((policy.get("canary") or {}).get("holdout_fraction"), 1.0),
    }
    meta["preservation"] = {
        "system_developer_instructions_preserved": True,
        "preserved_instruction_count": _as_int(window.get("preserved_instruction_count")),
        "recent_items_preserved": _as_int(window.get("recent_item_count")),
        "tool_function_protocol_preserved": "tool_function_protocol_ambiguous" not in reason_codes,
        "attachments_file_references_preserved": "file_reference_in_source_window" not in reason_codes,
        "response_format_constraints_preserved": True,
        "streaming_compatible": True,
    }

    if reason_codes:
        meta["status"] = "skipped"
        meta["reason_codes"] = sorted(set(reason_codes))
        return body, meta
    if cohort != "canary_applied":
        meta["status"] = "holdout" if cohort == "holdout" else "skipped"
        meta["reason_codes"] = [cohort]
        return body, meta

    cache_key = _summary_cache_key(source_hash=source_digest, policy=policy)
    cached = get_cached_summary(cache_key)
    summary_text = ""
    summary_cost = 0.0
    if isinstance(cached, dict) and isinstance(cached.get("summary"), str) and cached["summary"].strip():
        summary_text = cached["summary"].strip()
        meta["summary_cache_hit"] = True
        summary_cost = _as_float(cached.get("summary_cost_est_usd"))
        meta["summary_input_tokens"] = cached.get("summary_input_tokens")
        meta["summary_output_tokens"] = cached.get("summary_output_tokens")
    else:
        try:
            summary_response = await fetch_summary(build_summary_request(source_items, policy))
        except Exception as exc:
            meta["status"] = "skipped"
            meta["reason_codes"] = ["summary_fetch_error"]
            meta["summary_error_type"] = type(exc).__name__
            return body, meta
        if not isinstance(summary_response, dict):
            summary_response = {}
        raw_summary = summary_response.get("summary")
        summary_text = raw_summary.strip() if isinstance(raw_summary, str) else ""
        summary_cost = _as_float(summary_response.get("summary_cost_est_usd"))
        meta["summary_status_code"] = summary_response.get("summary_status_code")
        meta["summary_input_tokens"] = summary_response.get("summary_input_tokens")
        meta["summary_output_tokens"] = summary_response.get("summary_output_tokens")
        if not summary_text:
            meta["status"] = "skipped"
            meta["reason_codes"] = ["summary_empty_or_malformed"]
            if summary_response.get("summary_error"):
                meta["summary_error"] = str(summary_response.get("summary_error"))[:500]
            return body, meta
        set_cached_summary(
            cache_key,
            {
                "schema": SUMMARY_CACHE_SCHEMA,
                "summary": summary_text,
                "summary_model": str(policy.get("summary_model") or DEFAULT_SUMMARY_MODEL),
                "summary_cost_est_usd": summary_cost,
                "summary_input_tokens": meta.get("summary_input_tokens"),
                "summary_output_tokens": meta.get("summary_output_tokens"),
            },
        )

    max_summary_chars = _as_int(policy.get("max_summary_chars"), DEFAULT_MAX_SUMMARY_CHARS)
    if len(summary_text) > max_summary_chars:
        summary_text = summary_text[:max_summary_chars].rstrip()
        meta["summary_truncated"] = True
    new_body = _apply_summary_body(body, endpoint, window, summary_text)
    new_request_chars = _text_chars(new_body)
    meta.update({
        "status": "applied",
        "applied": True,
        "changed": True,
        "reason_codes": ["applied"],
        "summary_cost_est_usd": round(summary_cost, 8),
        "summary_chars": len(summary_text),
        "summary_sha256": hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
        "request_chars_after": new_request_chars,
        "actual_chars_saved_est": max(0, request_chars - new_request_chars),
    })
    if summary_cost:
        gross = _as_float(meta.get("estimated_gross_savings_usd"))
        meta["estimated_net_savings_usd"] = round(gross - summary_cost, 8)
    return new_body, meta


def add_summary_cost(cost: float | None, summary_meta: dict[str, Any]) -> float | None:
    extra = _as_float(summary_meta.get("summary_cost_est_usd"))
    if cost is None:
        return extra if extra else None
    return cost + extra


def input_tokens_after_summary(body: dict[str, Any]) -> int:
    return max(1, _text_chars(body) // TOKEN_CHARS)
