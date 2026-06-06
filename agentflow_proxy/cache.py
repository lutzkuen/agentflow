from __future__ import annotations

import os
from typing import Any

from agentflow_proxy.crunch import sha256_text
from agentflow_proxy.store import stable_json

CACHE_ENABLED = os.getenv("AGENTFLOW_CACHE", "1") != "0"
# Avoid caching tool-using agent turns by default. Exact cache can be dangerous when tools reflect filesystem state.
CACHE_TOOL_CALLS = os.getenv("AGENTFLOW_CACHE_TOOL_CALLS", "0") == "1"
SEMANTIC_CACHE_ENABLED = os.getenv("AGENTFLOW_SEMANTIC_CACHE", "0") == "1"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("AGENTFLOW_SEMANTIC_THRESHOLD", "0.95"))


def cache_decision_meta(
    status: str,
    reason: str,
    *,
    hit_type: str | None = None,
    enabled: bool | None = None,
    exact_enabled: bool | None = None,
    semantic_enabled: bool | None = None,
    tool_cache_enabled: bool | None = None,
) -> dict[str, Any]:
    exact = CACHE_ENABLED if exact_enabled is None else exact_enabled
    semantic = SEMANTIC_CACHE_ENABLED if semantic_enabled is None else semantic_enabled
    tool_cache = CACHE_TOOL_CALLS if tool_cache_enabled is None else tool_cache_enabled
    overall_enabled = (CACHE_ENABLED or SEMANTIC_CACHE_ENABLED) if enabled is None else enabled
    meta = {
        "enabled": bool(overall_enabled),
        "status": status,
        "reason": reason,
        "policy_source": "local-default",
        "exact_enabled": bool(exact),
        "semantic_enabled": bool(semantic),
        "tool_cache_enabled": bool(tool_cache),
    }
    if hit_type:
        meta["hit_type"] = hit_type
    return meta


def cache_lookup_meta(has_tool_blocks: bool) -> tuple[bool, bool, dict[str, Any]]:
    exact_enabled = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tool_blocks)
    semantic_enabled = SEMANTIC_CACHE_ENABLED and not has_tool_blocks
    if exact_enabled or semantic_enabled:
        if exact_enabled and semantic_enabled:
            reason = "exact-and-semantic-miss"
        elif exact_enabled:
            reason = "exact-miss"
        else:
            reason = "semantic-miss"
        status = "miss"
    elif has_tool_blocks and (CACHE_ENABLED or SEMANTIC_CACHE_ENABLED):
        status = "skipped"
        reason = "tools-disabled"
    else:
        status = "skipped"
        reason = "cache-disabled"
    return exact_enabled, semantic_enabled, cache_decision_meta(
        status,
        reason,
        enabled=CACHE_ENABLED or SEMANTIC_CACHE_ENABLED,
        exact_enabled=exact_enabled,
        semantic_enabled=semantic_enabled,
    )


def cache_key_for(body: dict[str, Any], path: str) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    return sha256_text(path + "\n" + stable_json(body))


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
