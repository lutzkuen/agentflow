from __future__ import annotations

import os
import re
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.store import stable_json

HAIKU_DEFAULT = os.getenv("AGENTFLOW_HAIKU_MODEL", "claude-haiku-4-5-20251001")
SONNET_DEFAULT = os.getenv("AGENTFLOW_SONNET_MODEL", "claude-sonnet-4-6")
OPUS_DEFAULT = os.getenv("AGENTFLOW_OPUS_MODEL", "claude-opus-4-5")

ROUTING_ENABLED = os.getenv("AGENTFLOW_ROUTING", "1") != "0"
ROUTING_RULES_PATH = os.getenv("AGENTFLOW_ROUTING_RULES", str(Path.home() / ".agentflow" / "routing_rules.yaml"))


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
    thinking = body.get("thinking")
    if not thinking:
        return False
    if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
        return False
    return True


def _load_routing_rules() -> list[dict]:
    p = Path(ROUTING_RULES_PATH)
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "rules" in data:
            return list(data["rules"])
    defaults = Path(__file__).parent / "routing_rules.yaml"
    with open(defaults) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    return list(data.get("rules", []))


ROUTING_RULES: list[dict] = _load_routing_rules()

_TIER_MAP = {"haiku": HAIKU_DEFAULT, "sonnet": SONNET_DEFAULT, "opus": OPUS_DEFAULT}


def route_model(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(body.get("model") or SONNET_DEFAULT)
    if not ROUTING_ENABLED:
        return requested, {"enabled": False, "requested_model": requested, "routed_model": requested, "reason": "routing disabled"}

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
        }

    for rule in ROUTING_RULES:
        cond = rule.get("conditions") or {}
        if "model_pattern" in cond and cond["model_pattern"].lower() not in requested_l:
            continue
        if "text_chars_lt" in cond and not (text_chars < int(cond["text_chars_lt"])):
            continue
        if "text_chars_gt" in cond and not (text_chars > int(cond["text_chars_gt"])):
            continue
        if "has_tools" in cond and bool(cond["has_tools"]) != tools:
            continue
        # absent max_tokens means unconstrained — treat as matching any max_tokens_lte rule
        if "max_tokens_lte" in cond and max_tokens is not None and not (int(max_tokens) <= int(cond["max_tokens_lte"])):
            continue
        if "category" in cond and cond["category"] != category:
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
        }

    return requested, {
        "enabled": True,
        "requested_model": requested,
        "routed_model": requested,
        "reason": "keep requested model",
        "text_chars": text_chars,
        "has_tools": tools,
        "category": category,
    }
