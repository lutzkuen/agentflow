from __future__ import annotations

import base64
import os
import re
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
        "file_watch": {
            "enabled": True,
            "root": ".",
            "max_paths": 128,
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
    file_watch = data.get("file_watch") or {}
    if isinstance(file_watch, dict):
        policy["file_watch"]["enabled"] = _as_bool(
            file_watch.get("enabled"),
            policy["file_watch"]["enabled"],
        )
        if file_watch.get("root") is not None:
            policy["file_watch"]["root"] = str(file_watch["root"])
        if file_watch.get("max_paths") is not None:
            policy["file_watch"]["max_paths"] = int(file_watch["max_paths"])
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
    policy["file_watch"]["enabled"] = _as_bool(
        os.getenv("AGENTFLOW_CACHE_FILE_WATCH"),
        policy["file_watch"]["enabled"],
    )
    policy["file_watch"]["root"] = os.getenv(
        "AGENTFLOW_CACHE_WATCH_ROOT",
        str(policy["file_watch"]["root"]),
    )
    policy["file_watch"]["max_paths"] = int(
        os.getenv("AGENTFLOW_CACHE_WATCH_MAX_PATHS", str(policy["file_watch"]["max_paths"]))
    )
    return policy, "local-default", str(defaults_path)


CACHE_POLICY, CACHE_POLICY_SOURCE, CACHE_RULES_PATH = _load_cache_policy()
CACHE_ENABLED = bool(CACHE_POLICY["exact_cache"]["enabled"])
CACHE_TOOL_CALLS = bool(CACHE_POLICY["exact_cache"]["cache_tool_calls"])
SEMANTIC_CACHE_ENABLED = bool(CACHE_POLICY["semantic_cache"]["enabled"])
SEMANTIC_CACHE_THRESHOLD = float(CACHE_POLICY["semantic_cache"]["threshold"])
CACHE_FILE_WATCH_ENABLED = bool(CACHE_POLICY["file_watch"]["enabled"])
CACHE_FILE_WATCH_ROOT = str(CACHE_POLICY["file_watch"]["root"])
CACHE_FILE_WATCH_MAX_PATHS = int(CACHE_POLICY["file_watch"]["max_paths"])


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
        "file_watch_enabled": CACHE_FILE_WATCH_ENABLED,
    }
    if hit_type:
        meta["hit_type"] = hit_type
    return meta


_PATH_TRAILING_JUNK = ".,;:)\\]}>\"'"
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _walk_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_walk_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(_walk_strings(item))
    return strings


def _candidate_path_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.split(r"\s+", text):
        token = raw.strip("`'\"(<[{")
        token = token.rstrip(_PATH_TRAILING_JUNK)
        if not token or "://" in token or token.startswith("data:"):
            continue
        if "\x00" in token or "*" in token or "?" in token:
            continue
        if ":" in token and not _WINDOWS_DRIVE_RE.match(token):
            before, after = token.rsplit(":", 1)
            if after.isdigit():
                token = before.rstrip(_PATH_TRAILING_JUNK)
        if not token:
            continue
        path_like = (
            token.startswith(("/", "./", "../", "~/"))
            or _WINDOWS_DRIVE_RE.match(token) is not None
            or "/" in token
            or "\\" in token
        )
        if path_like:
            tokens.append(token)
    return tokens


def _resolve_under_root(token: str, root: Path) -> Path | None:
    expanded = Path(token).expanduser()
    if expanded.is_absolute():
        resolved = expanded.resolve(strict=False)
    else:
        resolved = (root / expanded).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def cache_file_dependency_snapshots(
    body: dict[str, Any],
    *,
    root: str | Path | None = None,
    max_paths: int | None = None,
) -> list[dict[str, Any]]:
    if not CACHE_FILE_WATCH_ENABLED:
        return []
    watch_root = Path(root if root is not None else CACHE_FILE_WATCH_ROOT).expanduser().resolve(strict=False)
    limit = CACHE_FILE_WATCH_MAX_PATHS if max_paths is None else max_paths
    paths: dict[str, Path] = {}
    for text in _walk_strings(body):
        for token in _candidate_path_tokens(text):
            resolved = _resolve_under_root(token, watch_root)
            if resolved is None:
                continue
            paths[str(resolved)] = resolved
            if len(paths) >= limit:
                break
        if len(paths) >= limit:
            break

    snapshots: list[dict[str, Any]] = []
    for path in sorted(paths):
        file_path = paths[path]
        try:
            stat = file_path.stat()
            exists = file_path.is_file()
        except OSError:
            stat = None
            exists = False
        snapshots.append({
            "path": path,
            "exists": bool(exists),
            "mtime_ns": int(stat.st_mtime_ns) if stat and exists else None,
            "size": int(stat.st_size) if stat and exists else None,
        })
    return snapshots


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
