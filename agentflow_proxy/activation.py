from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
from typing import Any

from agentflow_proxy.upstream_url import normalize_openai_upstream_base_url, redact_url


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
CODEX_CONFIG_RELATIVE_PATH = Path(".codex") / "config.toml"
CLAUDE_VSCODE_ENV_FILENAME = "claude-vscode.env"


class ActivationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_config_dir() -> Path:
    return Path(os.getenv("AGENTFLOW_CONFIG_DIR") or Path.home() / ".agentflow")


def activation_config_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return base / DEFAULT_CONFIG_FILENAME


def default_codex_config_path() -> Path:
    return Path.home() / CODEX_CONFIG_RELATIVE_PATH


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


def _profile_command_args(profile: dict[str, Any], *, redact: bool = False) -> list[str]:
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
            str(
                redact_url(profile.get("upstream_base_url") or DEFAULT_OPENAI_UPSTREAM_BASE_URL)
                if redact
                else profile.get("upstream_base_url") or DEFAULT_OPENAI_UPSTREAM_BASE_URL
            ),
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
            str(
                redact_url(profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL)
                if redact
                else profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL
            ),
        ]
    raise ValueError(f"unknown activation target: {target or provider or '<missing>'}")


def proxy_args_for_target(config: dict[str, Any], target: str) -> list[str]:
    profile = (config.get("targets") or {}).get(target)
    if not isinstance(profile, dict) or not profile.get("configured"):
        raise KeyError(target)
    return _profile_command_args(profile)


def shell_command_for_profile(profile: dict[str, Any], *, redact: bool = False) -> str:
    return " ".join(["agentflow-proxy", *[shlex.quote(arg) for arg in _profile_command_args(profile, redact=redact)]])


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
        upstream_base_url = normalize_openai_upstream_base_url(openai_base_url or DEFAULT_OPENAI_UPSTREAM_BASE_URL)
        profile = {
            "id": "openai",
            "configured": True,
            "provider": "openai",
            "local_base_url": local_base_url or DEFAULT_OPENAI_LOCAL_BASE_URL,
            "health_url": health_url or DEFAULT_OPENAI_HEALTH_URL,
            "upstream_base_url": upstream_base_url,
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


def _openai_profile_from_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    targets = config.get("targets") if isinstance(config.get("targets"), dict) else {}
    existing = targets.get("openai") if isinstance(targets.get("openai"), dict) else None
    if existing and existing.get("configured"):
        return existing, False
    return activation_profile("openai"), True


def _claude_profile_from_config(
    config: dict[str, Any],
    *,
    auto_configure: bool = True,
) -> tuple[dict[str, Any], bool]:
    targets = config.get("targets") if isinstance(config.get("targets"), dict) else {}
    existing = targets.get("claude") if isinstance(targets.get("claude"), dict) else None
    if existing and existing.get("configured"):
        return existing, False
    if not auto_configure:
        raise ActivationError(
            "AgentFlow target is not configured: claude. Run `agentflow activate claude` first, "
            "or rerun `agentflow activate claude-vscode` without `--no-auto-claude` to create "
            "the default Claude target."
        )
    return activation_profile("claude"), True


def codex_activation_profile(*, openai_profile: dict[str, Any], codex_config_path: Path) -> dict[str, Any]:
    return {
        "id": "codex",
        "configured": True,
        "provider": "openai",
        "app": "codex",
        "depends_on": "openai",
        "local_base_url": str(openai_profile.get("local_base_url") or DEFAULT_OPENAI_LOCAL_BASE_URL),
        "codex_config_path": str(codex_config_path),
        "updated_at": utc_now(),
    }


def claude_vscode_activation_profile(
    *,
    claude_profile: dict[str, Any],
    env_file_path: Path,
) -> dict[str, Any]:
    return {
        "id": "claude-vscode",
        "configured": True,
        "provider": "anthropic",
        "app": "claude-vscode",
        "depends_on": "claude",
        "local_base_url": str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL),
        "upstream_base_url": str(claude_profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL),
        "env_file_path": str(env_file_path),
        "safe_env": {
            "ANTHROPIC_BASE_URL": str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL),
        },
        "launch_command": "code .",
        "updated_at": utc_now(),
    }


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
        "upstream_base_url_redacted": redact_url(str(profile["upstream_base_url"])),
        "run_command": f"agentflow run {profile['id']}",
        "proxy_command": shell_command_for_profile(profile, redact=True),
        "profile": profile,
        "target_count": len(config.get("targets") or {}),
    }


_OPENAI_BASE_URL_RE = re.compile(r'^(\s*openai_base_url\s*=\s*)(".*?"|\'.*?\'|[^#\n]*?)(\s+#.*)?(\r?\n)?$')


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _is_table_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") and not stripped.startswith("#")


def _update_codex_toml(raw: str, local_base_url: str) -> tuple[str, bool]:
    replacement_value = _toml_string(local_base_url)
    newline = "\r\n" if "\r\n" in raw else "\n"
    if raw and not raw.endswith(("\n", "\r")):
        raw += newline
    replacement_line = f"openai_base_url = {replacement_value}{newline}"
    lines = raw.splitlines(keepends=True)

    for index, line in enumerate(lines):
        if _is_table_header(line):
            break
        match = _OPENAI_BASE_URL_RE.match(line)
        if not match:
            continue
        updated_line = f"{match.group(1)}{replacement_value}{match.group(3) or ''}{match.group(4) or newline}"
        if updated_line == line:
            return raw, False
        updated = list(lines)
        updated[index] = updated_line
        return "".join(updated), True

    insert_at = len(lines)
    for index, line in enumerate(lines):
        if _is_table_header(line):
            insert_at = index
            break
    updated = list(lines)
    if insert_at > 0 and updated[insert_at - 1].strip():
        updated.insert(insert_at, newline)
        updated.insert(insert_at, replacement_line)
    else:
        updated.insert(insert_at, replacement_line)
    return "".join(updated), True


def _next_backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".agentflow.bak")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = path.with_name(path.name + f".agentflow.bak.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _is_project_local_codex_config(path: Path, cwd: Path | None = None) -> bool:
    base = cwd or Path.cwd()
    try:
        return path.expanduser().resolve(strict=False) == (base / CODEX_CONFIG_RELATIVE_PATH).resolve(strict=False)
    except OSError:
        return False


def claude_vscode_env_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return base / CLAUDE_VSCODE_ENV_FILENAME


def _claude_vscode_env_contents(local_base_url: str) -> str:
    return (
        "# AgentFlow-managed non-secret routing values for Claude in VS Code.\n"
        "# Keep Claude API keys in your shell or OS secret manager, not in this file.\n"
        f"ANTHROPIC_BASE_URL={local_base_url}\n"
    )


def activate_claude_vscode(
    *,
    config_dir: str | Path | None = None,
    dry_run: bool = False,
    auto_configure_claude: bool = True,
) -> dict[str, Any]:
    config = load_activation_config(config_dir)
    claude_profile, created_claude = _claude_profile_from_config(config, auto_configure=auto_configure_claude)
    local_base_url = str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL)
    upstream_base_url = str(claude_profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL)
    env_path = claude_vscode_env_path(config_dir)
    env_contents = _claude_vscode_env_contents(local_base_url)
    existing_env = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    env_changed = existing_env != env_contents

    vscode_profile = claude_vscode_activation_profile(claude_profile=claude_profile, env_file_path=env_path)
    updated_config = apply_activation_profile(config, claude_profile)
    updated_config = apply_activation_profile(updated_config, vscode_profile)
    config_path = activation_config_path(config_dir)

    if not dry_run:
        if env_changed:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(env_contents, encoding="utf-8")
        config_path = write_activation_config(updated_config, config_dir)

    shell_exports = [f"export ANTHROPIC_BASE_URL={shlex.quote(local_base_url)}"]
    routing_snippet = "\n".join([*shell_exports, "code ."])
    return {
        "schema": "agentflow.claude_vscode_activation_result.v1",
        "ok": True,
        "dry_run": bool(dry_run),
        "target": "claude-vscode",
        "configured": True,
        "config_path": str(config_path),
        "env_file_path": str(env_path),
        "env_file_changed": env_changed,
        "local_base_url": local_base_url,
        "upstream_base_url": upstream_base_url,
        "depends_on": "claude",
        "claude_target_created": created_claude,
        "run_command": "agentflow run claude",
        "safe_env": dict(vscode_profile["safe_env"]),
        "routing_snippet": routing_snippet,
        "profile": vscode_profile,
        "target_count": len(updated_config.get("targets") or {}),
    }


def activate_codex(
    *,
    config_dir: str | Path | None = None,
    codex_config_path: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    config = load_activation_config(config_dir)
    openai_profile, created_openai = _openai_profile_from_config(config)
    path = Path(codex_config_path).expanduser() if codex_config_path is not None else default_codex_config_path()
    is_user_level = path.resolve(strict=False) == default_codex_config_path().resolve(strict=False)
    if _is_project_local_codex_config(path, cwd=cwd) and not is_user_level:
        raise ActivationError(
            "refusing to modify project-local .codex/config.toml; use the user-level Codex config instead"
        )

    local_base_url = str(openai_profile.get("local_base_url") or DEFAULT_OPENAI_LOCAL_BASE_URL)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    updated_toml, codex_changed = _update_codex_toml(original, local_base_url)
    backup_path: Path | None = None

    codex_profile = codex_activation_profile(openai_profile=openai_profile, codex_config_path=path)
    updated_config = apply_activation_profile(config, openai_profile)
    updated_config = apply_activation_profile(updated_config, codex_profile)
    config_path = activation_config_path(config_dir)

    if not dry_run:
        if codex_changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                backup_path = _next_backup_path(path)
                shutil.copy2(path, backup_path)
            path.write_text(updated_toml, encoding="utf-8")
        config_path = write_activation_config(updated_config, config_dir)

    return {
        "schema": "agentflow.codex_activation_result.v1",
        "ok": True,
        "dry_run": bool(dry_run),
        "force": bool(force),
        "target": "codex",
        "configured": True,
        "config_path": str(config_path),
        "codex_config_path": str(path),
        "codex_config_changed": codex_changed,
        "codex_config_backup_path": str(backup_path) if backup_path is not None else None,
        "local_base_url": local_base_url,
        "depends_on": "openai",
        "openai_target_created": created_openai,
        "run_command": "agentflow run openai",
        "profile": codex_profile,
        "target_count": len(updated_config.get("targets") or {}),
    }
