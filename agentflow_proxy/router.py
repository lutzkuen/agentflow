from __future__ import annotations

import os
import re
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.store import stable_json

HAIKU_DEFAULT = os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_DEFAULT = os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
OPUS_DEFAULT = os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")
OPENAI_LARGE_DEFAULT = os.getenv("AGENTFLOW_OPENAI_LARGE_MODEL", "gpt-5-codex")
OPENAI_SMALL_DEFAULT = os.getenv("AGENTFLOW_OPENAI_SMALL_MODEL", "gpt-5-mini")
OPENAI_TINY_DEFAULT = os.getenv("AGENTFLOW_OPENAI_TINY_MODEL", "gpt-5-nano")

ROUTING_ENABLED = os.getenv("AGENTFLOW_ROUTING", "1") != "0"
OPENAI_ROUTING_ENABLED = os.getenv("AGENTFLOW_OPENAI_ROUTING", "0") == "1"
ROUTING_RULES_PATH = os.getenv("AGENTFLOW_ROUTING_RULES", str(Path.home() / ".agentflow" / "routing_rules.yaml"))
STRIP_THINKING_HISTORY = os.getenv("AGENTFLOW_STRIP_THINKING_HISTORY", "0") == "1"


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


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


def uses_thinking(body: dict[str, Any]) -> bool:
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
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            if any(isinstance(b, dict) and b.get("type") == "thinking" for b in msg["content"]):
                return True
    return False


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


def strip_thinking_history_blocks(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove type=='thinking' blocks from assistant messages; returns (modified_body, n_stripped)."""
    import copy
    body = copy.deepcopy(body)
    n_stripped = 0
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            filtered = [b for b in msg["content"] if not (isinstance(b, dict) and b.get("type") == "thinking")]
            n_stripped += len(msg["content"]) - len(filtered)
            msg["content"] = filtered
    return body, n_stripped


def _load_routing_rules() -> tuple[list[dict], str, str]:
    p = Path(ROUTING_RULES_PATH)
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data, "local-manual", str(p)
        if isinstance(data, dict) and "rules" in data:
            return list(data["rules"]), "local-manual", str(p)
    defaults = Path(__file__).parent / "routing_rules.yaml"
    with open(defaults) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data, "local-default", str(defaults)
    return list(data.get("rules", [])), "local-default", str(defaults)


ROUTING_RULES, ROUTING_RULES_SOURCE, ROUTING_RULES_PATH = _load_routing_rules()
ROUTING_RULES_LOADED_AT = utc_now()
ROUTING_RULES_LOADED_FILE = policy_file_snapshot(ROUTING_RULES_PATH)

_TIER_MAP = {"haiku": HAIKU_DEFAULT, "sonnet": SONNET_DEFAULT, "opus": OPUS_DEFAULT}


def route_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    if not ROUTING_ENABLED:
        return requested, {"enabled": False, "requested_model": requested, "routed_model": requested, "reason": "routing disabled", "policy_source": "local-default"}

    requested_l = requested.lower()
    text_chars = len(extract_text(body))
    tools = has_tools(body)
    max_tokens = body.get("max_tokens")  # None when caller didn't set it
    category = categorize_request(body)

    if uses_thinking(body):
        return requested, {
            "enabled": True,
            "requested_model": requested,
            "routed_model": requested,
            "reason": "keep requested model for thinking request",
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            "policy_source": ROUTING_RULES_SOURCE,
        }

    for rule in ROUTING_RULES:
        cond = rule.get("conditions") or {}
        if "model_pattern" in cond and cond["model_pattern"].lower() not in requested_l:
            continue
        if "text_chars_lt" in cond and not (text_chars < int(cond["text_chars_lt"])):
            continue
        if "text_chars_gt" in cond and not (text_chars > int(cond["text_chars_gt"])):
            continue
        if "text_chars_lte" in cond and not (text_chars <= int(cond["text_chars_lte"])):
            continue
        if "text_chars_gte" in cond and not (text_chars >= int(cond["text_chars_gte"])):
            continue
        if "has_tools" in cond and bool(cond["has_tools"]) != tools:
            continue
        if "env_flag" in cond and not _env_flag_enabled(str(cond["env_flag"])):
            continue
        # A missing max_tokens value is unknown, not safely bounded.
        if "max_tokens_lte" in cond:
            if max_tokens is None or not (int(max_tokens) <= int(cond["max_tokens_lte"])):
                continue
        if "category" in cond and cond["category"] != category:
            continue
        if "category_not_in" in cond:
            excluded = cond["category_not_in"]
            if isinstance(excluded, str):
                excluded = [excluded]
            if category in set(excluded):
                continue

        action = rule.get("action") or {}
        route_key = str(action.get("route_to", ""))
        routed = _TIER_MAP.get(route_key, route_key) if route_key else requested
        reason = str(action.get("reason", "matched routing rule"))
        return routed, {
            "enabled": True,
            "requested_model": requested,
            "routed_model": routed,
            "reason": reason,
            "text_chars": text_chars,
            "has_tools": tools,
            "category": category,
            "policy_source": ROUTING_RULES_SOURCE,
        }

    return requested, {
        "enabled": True,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "keep requested model",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        "policy_source": ROUTING_RULES_SOURCE,
    }


def route_openai_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or OPENAI_LARGE_DEFAULT)
    text_chars = len(extract_text(body))
    tools = bool(body.get("tools"))
    category = categorize_request(body)
    meta = {
        "enabled": OPENAI_ROUTING_ENABLED,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "openai routing disabled",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
        "policy_source": "local-default",
        "provider": "openai",
    }
    if not ROUTING_ENABLED or not OPENAI_ROUTING_ENABLED:
        return requested, meta

    requested_l = requested.lower()
    if tools:
        meta["reason"] = "keep requested OpenAI model for tool request"
        meta["enabled"] = True
        return requested, meta

    routed = requested
    reason = "keep requested OpenAI model"
    if requested_l == OPENAI_LARGE_DEFAULT.lower() and text_chars < int(os.getenv("AGENTFLOW_OPENAI_SMALL_TEXT_CHARS_LT", "6000")):
        routed = OPENAI_SMALL_DEFAULT
        reason = "small non-tool OpenAI request"
    elif requested_l == OPENAI_SMALL_DEFAULT.lower() and text_chars < int(os.getenv("AGENTFLOW_OPENAI_TINY_TEXT_CHARS_LT", "1500")):
        routed = OPENAI_TINY_DEFAULT
        reason = "tiny non-tool OpenAI request"

    meta.update({
        "enabled": True,
        "routed_model": routed,
        "reason": reason,
    })
    return routed, meta
