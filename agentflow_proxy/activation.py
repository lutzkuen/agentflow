from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
from typing import Any


SCHEMA = "agentflow.activation_config.v1"
DEFAULT_CONFIG_FILENAME = "activation.json"
DEFAULT_OPENAI_LOCAL_BASE_URL = "http://127.0.0.1:4003/v1"
DEFAULT_OPENAI_HEALTH_URL = "http://127.0.0.1:4003/health"
DEFAULT_OPENAI_UPSTREAM_BASE_URL = "https://api.openai.com"
DEFAULT_CLAUDE_LOCAL_BASE_URL = "http://127.0.0.1:4000"
DEFAULT_CLAUDE_HEALTH_URL = "http://127.0.0.1:4000/health"
DEFAULT_CLAUDE_UPSTREAM_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_AUTH_MODE = "client"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_OPENAI_PORT = 4003
DEFAULT_CLAUDE_PORT = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_config_dir() -> Path:
    return Path(os.getenv("AGENTFLOW_CONFIG_DIR") or Path.home() / ".agentflow")


def activation_config_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return base / DEFAULT_CONFIG_FILENAME


def empty_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "targets": {},
    }


def load_activation_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    path = activation_config_path(config_dir)
    if not path.exists():
        return empty_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return empty_config()
    if not isinstance(payload, dict):
        return empty_config()
    if not isinstance(payload.get("targets"), dict):
        payload["targets"] = {}
    payload.setdefault("schema", SCHEMA)
    return payload


def write_activation_config(config: dict[str, Any], config_dir: str | Path | None = None) -> Path:
    path = activation_config_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _target_port(target: str) -> int:
    return DEFAULT_OPENAI_PORT if target == "openai" else DEFAULT_CLAUDE_PORT


def _target_host(_: str) -> str:
    return DEFAULT_HOST


def _profile_command_args(profile: dict[str, Any]) -> list[str]:
    target = str(profile.get("id") or "")
    provider = str(profile.get("provider") or "")
    if target == "openai" and provider == "openai":
        return [
            "--provider",
            "openai",
            "--host",
            str(profile.get("host") or DEFAULT_HOST),
            "--port",
            str(profile.get("port") or DEFAULT_OPENAI_PORT),
            "--openai-upstream",
            str(profile.get("upstream_base_url") or DEFAULT_OPENAI_UPSTREAM_BASE_URL),
            "--openai-auth-mode",
            str(profile.get("openai_auth_mode") or DEFAULT_OPENAI_AUTH_MODE),
        ]
    if target == "claude" and provider == "anthropic":
        return [
            "--provider",
            "anthropic",
            "--host",
            str(profile.get("host") or DEFAULT_HOST),
            "--port",
            str(profile.get("port") or DEFAULT_CLAUDE_PORT),
            "--anthropic-upstream",
            str(profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL),
        ]
    raise ValueError(f"unknown activation target: {target or provider or '<missing>'}")


def proxy_args_for_target(config: dict[str, Any], target: str) -> list[str]:
    profile = (config.get("targets") or {}).get(target)
    if not isinstance(profile, dict) or not profile.get("configured"):
        raise KeyError(target)
    return _profile_command_args(profile)


def shell_command_for_profile(profile: dict[str, Any]) -> str:
    return " ".join(["agentflow-proxy", *[shlex.quote(arg) for arg in _profile_command_args(profile)]])


def activation_profile(
    target: str,
    *,
    openai_base_url: str | None = None,
    anthropic_base_url: str | None = None,
    local_base_url: str | None = None,
    health_url: str | None = None,
    openai_auth_mode: str = DEFAULT_OPENAI_AUTH_MODE,
) -> dict[str, Any]:
    target = target.lower()
    if target == "openai":
        auth_mode = openai_auth_mode.lower()
        if auth_mode not in {"client", "proxy"}:
            raise ValueError("openai auth mode must be 'client' or 'proxy'")
        profile = {
            "id": "openai",
            "configured": True,
            "provider": "openai",
            "local_base_url": local_base_url or DEFAULT_OPENAI_LOCAL_BASE_URL,
            "health_url": health_url or DEFAULT_OPENAI_HEALTH_URL,
            "upstream_base_url": openai_base_url or DEFAULT_OPENAI_UPSTREAM_BASE_URL,
            "openai_auth_mode": auth_mode,
            "host": _target_host(target),
            "port": _target_port(target),
            "updated_at": utc_now(),
        }
    elif target == "claude":
        profile = {
            "id": "claude",
            "configured": True,
            "provider": "anthropic",
            "local_base_url": local_base_url or DEFAULT_CLAUDE_LOCAL_BASE_URL,
            "health_url": health_url or DEFAULT_CLAUDE_HEALTH_URL,
            "upstream_base_url": anthropic_base_url or DEFAULT_CLAUDE_UPSTREAM_BASE_URL,
            "host": _target_host(target),
            "port": _target_port(target),
            "updated_at": utc_now(),
        }
    else:
        raise ValueError("target must be 'openai' or 'claude'")
    profile["command_profile"] = {
        "entrypoint": "agentflow-proxy",
        "argv": _profile_command_args(profile),
    }
    return profile


def apply_activation_profile(
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(config)
    targets = dict(updated.get("targets") or {})
    targets[str(profile["id"])] = profile
    updated["schema"] = SCHEMA
    updated["targets"] = targets
    updated["updated_at"] = profile["updated_at"]
    return updated


def activation_result(
    *,
    config: dict[str, Any],
    profile: dict[str, Any],
    config_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema": "agentflow.activation_result.v1",
        "ok": True,
        "dry_run": dry_run,
        "target": profile["id"],
        "configured": True,
        "config_path": str(config_path),
        "local_base_url": profile["local_base_url"],
        "health_url": profile["health_url"],
        "upstream_base_url": profile["upstream_base_url"],
        "run_command": f"agentflow run {profile['id']}",
        "proxy_command": shell_command_for_profile(profile),
        "profile": profile,
        "target_count": len(config.get("targets") or {}),
    }
