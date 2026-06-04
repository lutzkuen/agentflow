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


def cache_key_for(body: dict[str, Any], path: str) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    return sha256_text(path + "\n" + stable_json(body))


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
