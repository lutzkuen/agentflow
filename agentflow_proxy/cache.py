from __future__ import annotations

import base64
import os
import yaml
from pathlib import Path
from typing import Any

from agentflow_proxy.crunch import sha256_text
from agentflow_proxy.store import stable_json


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _default_cache_policy() -> dict[str, Any]:
    return {
        "exact_cache": {
            "enabled": True,
            # Avoid caching tool-using agent turns by default. Exact cache can be dangerous when tools reflect filesystem state.
            "cache_tool_calls": False,
        },
        "semantic_cache": {
            "enabled": False,
            "threshold": 0.95,
        },
    }


def _manual_rule_candidates(filename: str, env_name: str) -> list[Path]:
    env_path = os.getenv(env_name)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / "config" / filename)
    candidates.append(Path.home() / ".agentflow" / filename)
    return candidates


def _apply_cache_policy_yaml(policy: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    exact = data.get("exact_cache") or {}
    if isinstance(exact, dict):
        policy["exact_cache"]["enabled"] = _as_bool(exact.get("enabled"), policy["exact_cache"]["enabled"])
        policy["exact_cache"]["cache_tool_calls"] = _as_bool(
            exact.get("cache_tool_calls"),
            policy["exact_cache"]["cache_tool_calls"],
        )
    semantic = data.get("semantic_cache") or {}
    if isinstance(semantic, dict):
        policy["semantic_cache"]["enabled"] = _as_bool(
            semantic.get("enabled"),
            policy["semantic_cache"]["enabled"],
        )
        if semantic.get("threshold") is not None:
            policy["semantic_cache"]["threshold"] = float(semantic["threshold"])
    return policy


def _load_cache_policy() -> tuple[dict[str, Any], str, str]:
    for path in _manual_rule_candidates("cache_rules.yaml", "AGENTFLOW_CACHE_RULES"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return _apply_cache_policy_yaml(_default_cache_policy(), data), "local-manual", str(path)

    defaults_path = Path(__file__).parent / "cache_rules.yaml"
    policy = _default_cache_policy()
    if defaults_path.exists():
        with open(defaults_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            policy = _apply_cache_policy_yaml(policy, data)
    policy["exact_cache"]["enabled"] = os.getenv("AGENTFLOW_CACHE", "1") != "0"
    policy["exact_cache"]["cache_tool_calls"] = os.getenv("AGENTFLOW_CACHE_TOOL_CALLS", "0") == "1"
    policy["semantic_cache"]["enabled"] = os.getenv("AGENTFLOW_SEMANTIC_CACHE", "0") == "1"
    policy["semantic_cache"]["threshold"] = float(
        os.getenv("AGENTFLOW_SEMANTIC_THRESHOLD", str(policy["semantic_cache"]["threshold"]))
    )
    return policy, "local-default", str(defaults_path)


CACHE_POLICY, CACHE_POLICY_SOURCE, CACHE_RULES_PATH = _load_cache_policy()
CACHE_ENABLED = bool(CACHE_POLICY["exact_cache"]["enabled"])
CACHE_TOOL_CALLS = bool(CACHE_POLICY["exact_cache"]["cache_tool_calls"])
SEMANTIC_CACHE_ENABLED = bool(CACHE_POLICY["semantic_cache"]["enabled"])
SEMANTIC_CACHE_THRESHOLD = float(CACHE_POLICY["semantic_cache"]["threshold"])


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
        "policy_source": CACHE_POLICY_SOURCE,
        "rule_path": CACHE_RULES_PATH,
        "exact_enabled": bool(exact),
        "semantic_enabled": bool(semantic),
        "tool_cache_enabled": bool(tool_cache),
        "semantic_threshold": SEMANTIC_CACHE_THRESHOLD,
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


def streaming_cache_lookup_meta(has_tool_blocks: bool) -> tuple[bool, dict[str, Any]]:
    exact_enabled = CACHE_ENABLED and (CACHE_TOOL_CALLS or not has_tool_blocks)
    if exact_enabled:
        status = "miss"
        reason = "streaming-exact-miss"
    elif has_tool_blocks and CACHE_ENABLED:
        status = "skipped"
        reason = "streaming-tools-disabled"
    else:
        status = "skipped"
        reason = "streaming-cache-disabled"
    return exact_enabled, cache_decision_meta(
        status,
        reason,
        enabled=CACHE_ENABLED,
        exact_enabled=exact_enabled,
        semantic_enabled=False,
    )


def stream_cache_payload(
    frames: list[bytes],
    *,
    provider: str,
    usage: dict[str, Any] | None = None,
    output_text: str | None = None,
) -> dict[str, Any]:
    return {
        "agentflow_cache_type": "sse-stream",
        "version": 1,
        "provider": provider,
        "frames_b64": [base64.b64encode(frame).decode("ascii") for frame in frames],
        "usage": usage or {},
        "output_text": output_text or "",
    }


def is_stream_cache_payload(payload: Any, *, provider: str | None = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("agentflow_cache_type") != "sse-stream":
        return False
    if provider is not None and payload.get("provider") != provider:
        return False
    return isinstance(payload.get("frames_b64"), list)


def stream_cache_frames(payload: dict[str, Any]) -> list[bytes]:
    frames: list[bytes] = []
    for item in payload.get("frames_b64") or []:
        if isinstance(item, str):
            frames.append(base64.b64decode(item.encode("ascii")))
    return frames


def _default_cache_provider() -> str:
    return os.getenv("AGENTFLOW_PROVIDER", "anthropic").lower()


def _default_cache_upstream(provider: str) -> str:
    if provider == "openai":
        return os.getenv("AGENTFLOW_OPENAI_UPSTREAM", "https://api.openai.com").rstrip("/")
    return os.getenv("AGENTFLOW_ANTHROPIC_UPSTREAM", "https://api.anthropic.com").rstrip("/")


def cache_key_for(
    body: dict[str, Any],
    path: str,
    *,
    provider: str | None = None,
    upstream: str | None = None,
    namespace: str | None = None,
) -> str:
    # Do not include auth. Include endpoint and body after crunch/routing.
    # Namespacing prevents cache reuse across providers, upstreams, or user-selected projects.
    provider = (provider or _default_cache_provider()).lower()
    upstream = (upstream or _default_cache_upstream(provider)).rstrip("/")
    namespace = namespace if namespace is not None else os.getenv("AGENTFLOW_CACHE_NAMESPACE", "default")
    key_material = stable_json({
        "version": 2,
        "namespace": namespace,
        "provider": provider,
        "upstream": upstream,
        "path": path,
        "body": body,
    })
    return sha256_text(key_material)


def response_output_text(resp: dict[str, Any]) -> str:
    parts = []
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)
