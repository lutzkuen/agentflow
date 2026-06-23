from __future__ import annotations

import os
import re
import yaml
from pathlib import Path
from typing import Any

from tokenclaw.env import env
from tokenclaw.policy_files import policy_file_snapshot, utc_now
from tokenclaw.paths import tokenclaw_config_path, safe_expanduser
from tokenclaw.store import stable_json

HAIKU_DEFAULT = env("TOKENCLAW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_DEFAULT = env("TOKENCLAW_SONNET_MODEL", "claude-sonnet-4-6")
OPUS_DEFAULT = env("TOKENCLAW_OPUS_MODEL", "claude-opus-4-5")
OPENAI_LARGE_DEFAULT = env("TOKENCLAW_OPENAI_LARGE_MODEL", "gpt-5-codex")
OPENAI_SMALL_DEFAULT = env("TOKENCLAW_OPENAI_SMALL_MODEL", "gpt-5-mini")
OPENAI_TINY_DEFAULT = env("TOKENCLAW_OPENAI_TINY_MODEL", "gpt-5-nano")

ROUTING_ENABLED = env("TOKENCLAW_ROUTING", "1") != "0"
OPENAI_ROUTING_ENABLED = env("TOKENCLAW_OPENAI_ROUTING", "0") == "1"
ROUTING_RULES_PATH = env("TOKENCLAW_ROUTING_RULES", str(tokenclaw_config_path("routing_rules.yaml")))
STRIP_THINKING_HISTORY = env("TOKENCLAW_STRIP_THINKING_HISTORY", "0") == "1"


def _env_flag_enabled(name: str) -> bool:
    new_name = name.replace("TOKENCLAW_", "TOKENCLAW_", 1) if name.startswith("TOKENCLAW_") else name
    return env(new_name, "0").strip().lower() in {"1", "true", "yes", "on"}


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


def extract_text(obj: Any) -> str:
    parts: list[str] = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            parts.append(extract_text(x))
    elif isinstance(obj, dict):
        # Only text-ish values for token estimate. Avoid binary/source blobs where possible.
        for k, v in obj.items():
            if k in {"text", "content", "input", "system", "name", "type"}:
                parts.append(extract_text(v))
            elif isinstance(v, (list, dict)):
                parts.append(extract_text(v))
    return "\n".join(p for p in parts if p)


def has_tools(body: dict[str, Any]) -> bool:
    if body.get("tools"):
        return True
    text = stable_json(body.get("messages", []))
    return "tool_use" in text or "tool_result" in text


def categorize_request(body: dict[str, Any]) -> str:
    tools = has_tools(body)
    text_chars = len(extract_text(body))
    msg_count = len(body.get("messages") or [])
    has_code = "```" in extract_text(body)

    messages = body.get("messages") or []
    if messages:
        last = messages[-1]
        if last.get("role") == "user":
            content = last.get("content", [])
            if isinstance(content, list) and content and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                return "tool-result"

    if tools and text_chars > 16000:
        return "tool-heavy"
    if tools:
        return "tool-light"
    if text_chars > 32000:
        return "long-context"
    if text_chars < 1500 and msg_count <= 2:
        return "short-completion"
    if has_code:
        return "code-gen"
    return "chat"


def _message_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _last_user_message(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def _last_user_text(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    return extract_text(message.get("content"))


def _assistant_tool_use_seen(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if any(block.get("type") == "tool_use" for block in _message_content_blocks(message)):
            return True
    return False


def _has_assistant_thinking_history(body: dict[str, Any]) -> bool:
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            if any(isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in msg["content"]):
                return True
    return False


def classify_workflow_phase(body: dict[str, Any], category: str | None = None) -> dict[str, str]:
    """Classify Anthropic workflow phase using metadata-only request shape signals."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return {
            "workflow_phase": "unknown",
            "workflow_phase_reason": "missing-messages",
            "workflow_phase_confidence": "low",
        }

    category = category or categorize_request(body)
    text_chars = len(extract_text(body))
    msg_count = len([message for message in messages if isinstance(message, dict)])
    last_user = _last_user_message(messages)
    last_user_blocks = _message_content_blocks(last_user or {})
    last_user_text = _last_user_text(last_user).lower()
    last_user_block_types = {
        str(block.get("type") or "")
        for block in last_user_blocks
        if isinstance(block.get("type"), str)
    }

    if _has_top_level_thinking(body):
        return {
            "workflow_phase": "thinking",
            "workflow_phase_reason": "thinking-current-request",
            "workflow_phase_confidence": "high",
        }

    if "tool_result" in last_user_block_types:
        return {
            "workflow_phase": "tool-execution",
            "workflow_phase_reason": "last-user-tool-result",
            "workflow_phase_confidence": "high",
        }

    if _has_assistant_thinking_history(body):
        return {
            "workflow_phase": "thinking",
            "workflow_phase_reason": "thinking-history",
            "workflow_phase_confidence": "high",
        }

    summary_terms = ("summary", "summarize", "summarise", "recap", "final answer", "report back")
    if any(term in last_user_text for term in summary_terms):
        return {
            "workflow_phase": "summary",
            "workflow_phase_reason": "summary-intent-text",
            "workflow_phase_confidence": "medium",
        }
    if category == "short-completion" and msg_count > 2 and text_chars < 2000:
        return {
            "workflow_phase": "summary",
            "workflow_phase_reason": "short-late-conversation",
            "workflow_phase_confidence": "medium",
        }

    verification_terms = ("verify", "verification", "test", "tests", "failing", "failure", "error", "regression")
    if any(term in last_user_text for term in verification_terms):
        return {
            "workflow_phase": "verification",
            "workflow_phase_reason": "verification-intent-text",
            "workflow_phase_confidence": "medium",
        }
    if category == "code-gen":
        return {
            "workflow_phase": "verification",
            "workflow_phase_reason": "code-context",
            "workflow_phase_confidence": "low",
        }

    if has_tools(body) and msg_count <= 2 and not _assistant_tool_use_seen(messages):
        return {
            "workflow_phase": "planning",
            "workflow_phase_reason": "early-tool-capable-user-turn",
            "workflow_phase_confidence": "medium",
        }

    if category == "chat":
        return {
            "workflow_phase": "chat",
            "workflow_phase_reason": "non-tool-chat-category",
            "workflow_phase_confidence": "medium",
        }

    return {
        "workflow_phase": "unknown",
        "workflow_phase_reason": f"category-{category or 'unknown'}",
        "workflow_phase_confidence": "low",
    }


def uses_thinking(body: dict[str, Any]) -> bool:
    return _has_top_level_thinking(body) or _has_assistant_thinking_history(body)


def _has_top_level_thinking(body: dict[str, Any]) -> bool:
    if body.get("effort"):
        return True
    if body.get("interleaved_thinking"):
        return True
    thinking = body.get("thinking")
    if thinking:
        if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
            pass
        else:
            return True
    return False


def _thinking_gate_meta(body: dict[str, Any], category: str) -> dict[str, Any]:
    top_level = _has_top_level_thinking(body)
    historical = _has_assistant_thinking_history(body)
    if top_level or historical:
        return {
            "status": "blocked",
            "reason": "current-thinking-request" if top_level else "assistant-thinking-history",
            "top_level_thinking": top_level,
            "assistant_thinking_history": historical,
            "metadata_only": True,
        }
    return {
        "status": "clear",
        "reason": "no-thinking-signals",
        "top_level_thinking": False,
        "assistant_thinking_history": False,
        "metadata_only": True,
    }


def strip_thinking_history_blocks(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove thinking-history blocks from assistant messages; returns (modified_body, n_stripped)."""
    import copy
    body = copy.deepcopy(body)
    n_stripped = 0
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            filtered = [
                b
                for b in msg["content"]
                if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})
            ]
            n_stripped += len(msg["content"]) - len(filtered)
            msg["content"] = filtered
    return body, n_stripped


def _retired_phase_canary_policy(policy_source: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "policy_id": "local-phase-routing-retired",
        "policy_source": policy_source,
        "router_runtime_status": "retired",
        "router_runtime_reason": "managed policy decisions own canary, holdout, and cohort routing",
    }


def _retired_openai_canary_policy(policy_source: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "policy_id": "local-openai-routing-retired",
        "policy_source": policy_source,
        "router_runtime_status": "retired",
        "router_runtime_reason": "managed policy decisions own canary, holdout, and cohort routing",
    }


def _load_routing_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return {"rules": data}
    if isinstance(data, dict):
        return data
    return {"rules": []}


def _rules_list(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rules = data.get("rules")
    if not isinstance(rules, list):
        return []
    return [rule for rule in rules if isinstance(rule, dict)]


def _hard_rule_count(rules: list[dict[str, Any]]) -> int:
    count = 0
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("enabled") is False:
            continue
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        if str(action.get("route_to") or "").strip():
            count += 1
    return count


def _load_routing_rules() -> tuple[list[dict], str, str, int]:
    defaults_path = Path(__file__).parent / "routing_rules.yaml"
    defaults = _load_routing_yaml(defaults_path) or {"rules": []}
    env_path = env("TOKENCLAW_ROUTING_RULES")
    if env_path:
        p = safe_expanduser(env_path)
        data = _load_routing_yaml(p)
        if data is not None:
            rules = _rules_list(data)
            return rules, "local-manual", str(p), _hard_rule_count(rules)

    local_path = tokenclaw_config_path("routing_rules.yaml")
    local = _load_routing_yaml(local_path)
    if local is not None:
        local_rules = _rules_list(local)
        return local_rules + _rules_list(defaults), "local-manual", str(local_path), _hard_rule_count(local_rules)

    return _rules_list(defaults), "local-default", str(defaults_path), 0


ROUTING_RULES, ROUTING_RULES_SOURCE, ROUTING_RULES_PATH, ROUTING_MANUAL_HARD_RULE_COUNT = _load_routing_rules()
ROUTING_PHASE_CANARY = _retired_phase_canary_policy(ROUTING_RULES_SOURCE)
ROUTING_OPENAI_CANARY = _retired_openai_canary_policy(ROUTING_RULES_SOURCE)
ROUTING_OPENAI_CANARIES = [ROUTING_OPENAI_CANARY]
ROUTING_RULES_LOADED_AT = utc_now()
ROUTING_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_RULES_PATH)

_TIER_MAP = {"haiku": HAIKU_DEFAULT, "sonnet": SONNET_DEFAULT, "opus": OPUS_DEFAULT}
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _input_tokens_est(text_chars: int) -> int:
    return max(1, text_chars // 4)


def _safe_openai_target_model(target_model: Any) -> tuple[str | None, str | None]:
    if not isinstance(target_model, str) or not target_model.strip():
        return None, "missing-target-model"
    target = target_model.strip()
    lowered = target.lower()
    if lowered.startswith("claude-") or any(part in lowered for part in ("haiku", "sonnet", "opus")):
        return None, "provider-mismatch"
    if lowered.startswith(("gpt-", "o", "text-", "computer-use-", "codex")) or "gpt" in lowered:
        return target, None
    return None, "unsupported-openai-target-model"


def _is_promoted_permanent_rule(rule: dict[str, Any]) -> bool:
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    if metadata.get("promoted_from_canary"):
        return True
    if metadata.get("source") == "claude-canary-promote":
        return True
    return str(rule.get("policy_source") or metadata.get("policy_source") or "") == "local-promoted"


def _rule_matches(
    rule: dict[str, Any],
    *,
    requested_l: str,
    text_chars: int,
    input_tokens_est: int,
    tools: bool,
    max_tokens: Any,
    category: str,
    phase_meta: dict[str, str],
    stream: bool,
    session_id: str | None,
    requested_model: str,
    provider: str = "anthropic",
    source_surface: str = "anthropic_messages",
    endpoint: str = "messages",
) -> bool:
    if rule.get("enabled") is False:
        return False
    cond = rule.get("conditions") or {}
    if "provider" in cond and str(cond["provider"]).lower() != provider.lower():
        return False
    if "source_surface" in cond and str(cond["source_surface"]).lower() != source_surface.lower():
        return False
    if "endpoint" in cond and str(cond["endpoint"]).lower() != endpoint.lower():
        return False
    if "model_pattern" in cond and str(cond["model_pattern"]).lower() not in requested_l:
        return False
    if "text_chars_lt" in cond and not (text_chars < int(cond["text_chars_lt"])):
        return False
    if "text_chars_gt" in cond and not (text_chars > int(cond["text_chars_gt"])):
        return False
    if "text_chars_lte" in cond and not (text_chars <= int(cond["text_chars_lte"])):
        return False
    if "text_chars_gte" in cond and not (text_chars >= int(cond["text_chars_gte"])):
        return False
    if "max_text_chars" in cond and not (text_chars <= int(cond["max_text_chars"])):
        return False
    if "min_text_chars" in cond and not (text_chars >= int(cond["min_text_chars"])):
        return False
    if "max_input_tokens_est" in cond and not (input_tokens_est <= int(cond["max_input_tokens_est"])):
        return False
    if "min_input_tokens_est" in cond and not (input_tokens_est >= int(cond["min_input_tokens_est"])):
        return False
    if "has_tools" in cond and bool(cond["has_tools"]) != tools:
        return False
    if "stream" in cond and bool(cond["stream"]) != stream:
        return False
    if "env_flag" in cond and not _env_flag_enabled(str(cond["env_flag"])):
        return False
    # A missing max_tokens value is unknown, not safely bounded.
    if "max_tokens_lte" in cond:
        if max_tokens is None or not (int(max_tokens) <= int(cond["max_tokens_lte"])):
            return False
    if "category" in cond and cond["category"] != category:
        return False
    if "category_not_in" in cond:
        excluded = cond["category_not_in"]
        if isinstance(excluded, str):
            excluded = [excluded]
        if category in set(excluded):
            return False
    if "workflow_phase" in cond and cond["workflow_phase"] != phase_meta.get("workflow_phase"):
        return False
    if "workflow_phase_confidence_gte" in cond:
        confidence = str(phase_meta.get("workflow_phase_confidence") or "low")
        required = str(cond.get("workflow_phase_confidence_gte") or "medium")
        if _CONFIDENCE_ORDER.get(confidence, 0) < _CONFIDENCE_ORDER.get(required, 1):
            return False
    if "session_memory" in cond:
        return False
    return True


def _route_from_rule(
    rule: dict[str, Any],
    *,
    requested: str,
    text_chars: int,
    tools: bool,
    category: str,
    phase_meta: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    action = rule.get("action") or {}
    route_key = str(action.get("route_to", ""))
    routed = _TIER_MAP.get(route_key, route_key) if route_key else requested
    reason = str(action.get("reason", "matched routing rule"))
    meta = {
        "enabled": True,
        "requested_model": requested,
        "routed_model": routed,
        "reason": reason,
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": str(rule.get("policy_source") or ROUTING_RULES_SOURCE),
        "routing_source": "explicit-local-rule",
    }
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    if metadata.get("promoted_from_canary"):
        meta["canary_cohort"] = None
        meta["routing_rule"] = {
            "status": "applied",
            "source": metadata.get("source") or "claude-canary-promote",
            "promoted_from_canary": True,
            "rule_id": rule.get("id") or metadata.get("rule_id"),
            "target_candidate_id": metadata.get("target_candidate_id"),
            "promotion_source_policy_id": metadata.get("promotion_source_policy_id"),
        }
    return routed, meta


def _route_openai_from_rule(
    rule: dict[str, Any],
    *,
    requested: str,
    text_chars: int,
    input_tokens_est: int,
    tools: bool,
    stream: bool,
    category: str,
    source_surface: str,
    endpoint: str,
) -> tuple[str | None, dict[str, Any] | None]:
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    target, target_error = _safe_openai_target_model(action.get("route_to"))
    if target_error or target is None:
        return None, None
    reason = str(action.get("reason") or "matched OpenAI routing rule")
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    meta: dict[str, Any] = {
        "enabled": True,
        "requested_model": requested,
        "routed_model": target,
        "reason": reason,
        "text_chars": text_chars,
        "input_tokens_est": input_tokens_est,
        "has_tools": tools,
        "stream": stream,
        "category": category,
        "workflow_phase": category,
        "workflow_phase_reason": "openai-request-category",
        "workflow_phase_confidence": "medium",
        "policy_source": str(rule.get("policy_source") or ROUTING_RULES_SOURCE),
        "routing_source": "explicit-local-rule",
        "provider": "openai",
        "source_surface": source_surface,
        "endpoint": endpoint,
    }
    if metadata.get("promoted_from_canary"):
        meta["openai_routing_rule"] = {
            "status": "applied",
            "source": metadata.get("source") or "openai_routing_promotion_decision",
            "promoted_from_canary": True,
            "rule_id": rule.get("id") or metadata.get("rule_id"),
            "target_candidate_id": metadata.get("target_candidate_id"),
            "promotion_source_policy_id": metadata.get("promotion_source_policy_id"),
            "target_local_policy_section": metadata.get("target_local_policy_section") or "routing.rules",
            "target_local_rule_file": metadata.get("target_local_rule_file") or "routing_rules.yaml",
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "provider_bodies_included": False,
                "tool_payloads_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
                "file_paths_included": False,
            },
        }
    return target, meta


def _openai_promoted_rule_semantic_gate_passed(rule: dict[str, Any]) -> bool:
    if not _is_promoted_permanent_rule(rule):
        return True
    metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
    promotion_decision = (
        metadata.get("promotion_decision")
        if isinstance(metadata.get("promotion_decision"), dict)
        else {}
    )
    semantic_quality = (
        metadata.get("semantic_quality")
        if isinstance(metadata.get("semantic_quality"), dict)
        else promotion_decision.get("semantic_quality")
        if isinstance(promotion_decision.get("semantic_quality"), dict)
        else {}
    )
    quality_gates = (
        metadata.get("quality_gates")
        if isinstance(metadata.get("quality_gates"), dict)
        else promotion_decision.get("quality_gates")
        if isinstance(promotion_decision.get("quality_gates"), dict)
        else {}
    )
    if quality_gates.get("requires_semantic_quality_pass") is False:
        return True
    return bool(semantic_quality.get("gate_passed"))


def routing_has_manual_hard_rules() -> bool:
    return ROUTING_MANUAL_HARD_RULE_COUNT > 0


def routing_is_unbacked_default_policy() -> bool:
    return ROUTING_RULES_SOURCE == "local-default" and not routing_has_manual_hard_rules()


def _unbacked_routing_meta(
    *,
    requested: str,
    text_chars: int,
    tools: bool,
    category: str,
    phase_meta: dict[str, str],
    policy_source: str,
    stream: bool | None = None,
    input_tokens_est: int | None = None,
    provider: str = "anthropic",
    source_surface: str | None = None,
    endpoint: str | None = None,
    thinking_gate: dict[str, Any] | None = None,
    reason: str = "routing off: no managed server or manual hard rules",
    backing_reason: str = "no-manual-hard-rules",
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "enabled": False,
        "requested_model": requested,
        "routed_model": requested,
        "reason": reason,
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": policy_source,
        "routing_source": "local-rules",
        "routing_backing": {
            "schema": "tokenclaw.routing_backing.v1",
            "backed": False,
            "manual_hard_rule_count": ROUTING_MANUAL_HARD_RULE_COUNT,
            "managed_policy_decision_checked_later": True,
            "reason": backing_reason,
        },
    }
    if stream is not None:
        meta["stream"] = stream
    if input_tokens_est is not None:
        meta["input_tokens_est"] = input_tokens_est
    if provider:
        meta["provider"] = provider
    if source_surface is not None:
        meta["source_surface"] = source_surface
    if endpoint is not None:
        meta["endpoint"] = endpoint
    if thinking_gate is not None:
        meta["thinking_gate"] = thinking_gate
    return meta


def route_model(body: dict[str, Any], *, session_id: str | None = None) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    text_chars = len(extract_text(body))
    tools = has_tools(body)
    max_tokens = body.get("max_tokens")  # None when caller didn't set it
    category = categorize_request(body)
    phase_meta = classify_workflow_phase(body, category)
    stream = _as_bool(body.get("stream"), False)
    thinking_gate = _thinking_gate_meta(body, category)
    if not ROUTING_ENABLED:
        return requested, {
            "enabled": False,
            "requested_model": requested,
            "routed_model": requested,
            "reason": "routing disabled",
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            **phase_meta,
            "policy_source": "local-default",
            "routing_source": "disabled",
            "thinking_gate": thinking_gate,
        }

    if routing_is_unbacked_default_policy():
        return requested, _unbacked_routing_meta(
            requested=requested,
            text_chars=text_chars,
            tools=tools,
            category=category,
            phase_meta=phase_meta,
            policy_source=ROUTING_RULES_SOURCE,
            stream=stream,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            thinking_gate=thinking_gate,
        )

    requested_l = requested.lower()

    if uses_thinking(body):
        return requested, {
            "enabled": False,
            "requested_model": requested,
            "routed_model": requested,
            "reason": "keep requested model for thinking request",
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            **phase_meta,
            "policy_source": ROUTING_RULES_SOURCE,
            "routing_source": "local-helper-safety-gate",
            "thinking_gate": thinking_gate,
        }

    for rule in ROUTING_RULES:
        matched = _rule_matches(
            rule,
            requested_l=requested_l,
            text_chars=text_chars,
            input_tokens_est=_input_tokens_est(text_chars),
            tools=tools,
            max_tokens=max_tokens,
            category=category,
            phase_meta=phase_meta,
            stream=stream,
            session_id=session_id,
            requested_model=requested,
        )
        if matched:
            routed, meta = _route_from_rule(
                rule,
                requested=requested,
                text_chars=text_chars,
                tools=tools,
                category=category,
                phase_meta=phase_meta,
            )
            meta["thinking_gate"] = thinking_gate
            return routed, meta

    meta = {
        "enabled": False,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "routing off: no matching manual hard rule",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        **phase_meta,
        "policy_source": ROUTING_RULES_SOURCE,
        "routing_source": "local-rules",
        "thinking_gate": thinking_gate,
        "routing_backing": {
            "schema": "tokenclaw.routing_backing.v1",
            "backed": False,
            "manual_hard_rule_count": ROUTING_MANUAL_HARD_RULE_COUNT,
            "managed_policy_decision_checked_later": True,
            "reason": "no-matching-manual-hard-rule",
        },
    }
    return requested, meta


def route_openai_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or OPENAI_LARGE_DEFAULT)
    text_chars = len(extract_text(body))
    tools = bool(body.get("tools"))
    stream = bool(body.get("stream"))
    category = categorize_request(body)
    input_tokens_est = _input_tokens_est(text_chars)
    source_surface = str(body.get("source_surface") or body.get("_tokenclaw_source_surface") or "openai_responses")
    endpoint = str(body.get("endpoint") or body.get("_tokenclaw_endpoint") or "responses")
    app_family = str(body.get("app_family") or body.get("_tokenclaw_app_family") or "generic_openai")
    workflow_phase = str(body.get("workflow_phase") or body.get("_tokenclaw_workflow_phase") or category)
    workflow_phase_confidence = str(
        body.get("workflow_phase_confidence")
        or body.get("_tokenclaw_workflow_phase_confidence")
        or "medium"
    )
    meta = {
        "enabled": bool(OPENAI_ROUTING_ENABLED),
        "requested_model": requested,
        "routed_model": requested,
        "reason": "openai routing disabled",
        "text_chars": text_chars,
        "input_tokens_est": input_tokens_est,
        "has_tools": tools,
        "stream": stream,
        "category": category,
        "policy_source": "local-default",
        "routing_source": "local-rules",
        "provider": "openai",
        "source_surface": source_surface,
        "endpoint": endpoint,
    }
    if not ROUTING_ENABLED:
        meta["enabled"] = False
        meta["reason"] = "routing disabled"
        return requested, meta
    if routing_is_unbacked_default_policy():
        return requested, _unbacked_routing_meta(
            requested=requested,
            text_chars=text_chars,
            tools=tools,
            category=category,
            phase_meta={
                "workflow_phase": workflow_phase,
                "workflow_phase_reason": "openai-request-category",
                "workflow_phase_confidence": workflow_phase_confidence,
            },
            policy_source=ROUTING_RULES_SOURCE,
            stream=stream,
            input_tokens_est=input_tokens_est,
            provider="openai",
            source_surface=source_surface,
            endpoint=endpoint,
        )

    requested_l = requested.lower()
    for rule in ROUTING_RULES:
        matched = _rule_matches(
            rule,
            requested_l=requested_l,
            text_chars=text_chars,
            input_tokens_est=input_tokens_est,
            tools=tools,
            max_tokens=body.get("max_tokens"),
            category=category,
            phase_meta={
                "workflow_phase": category,
                "workflow_phase_reason": "openai-request-category",
                "workflow_phase_confidence": "medium",
            },
            stream=stream,
            session_id=None,
            requested_model=requested,
            provider="openai",
            source_surface=source_surface,
            endpoint=endpoint,
        )
        if not matched:
            continue
        if not _openai_promoted_rule_semantic_gate_passed(rule):
            metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
            meta.update({
                "enabled": True,
                "reason": "promoted OpenAI routing rule blocked by semantic quality gate",
                "policy_source": "local-promoted-review",
                "openai_routing_rule": {
                    "status": "blocked",
                    "reason": "semantic-quality-gate-not-passed",
                    "source": metadata.get("source") or "openai_routing_promotion_decision",
                    "promoted_from_canary": True,
                    "rule_id": rule.get("id") or metadata.get("rule_id"),
                    "promotion_source_policy_id": metadata.get("promotion_source_policy_id"),
                    "target_local_policy_section": metadata.get("target_local_policy_section") or "routing.rules",
                    "target_local_rule_file": metadata.get("target_local_rule_file") or "routing_rules.yaml",
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "provider_bodies_included": False,
                    },
                },
            })
            continue
        routed, rule_meta = _route_openai_from_rule(
            rule,
            requested=requested,
            text_chars=text_chars,
            input_tokens_est=input_tokens_est,
            tools=tools,
            stream=stream,
            category=category,
            source_surface=source_surface,
            endpoint=endpoint,
        )
        if routed and rule_meta:
            return routed, rule_meta

    meta["enabled"] = False
    meta["reason"] = "routing off: no matching manual hard rule"
    meta["policy_source"] = ROUTING_RULES_SOURCE
    meta["routing_backing"] = {
        "schema": "tokenclaw.routing_backing.v1",
        "backed": False,
        "manual_hard_rule_count": ROUTING_MANUAL_HARD_RULE_COUNT,
        "managed_policy_decision_checked_later": True,
        "reason": "no-matching-manual-hard-rule",
    }
    return requested, meta
