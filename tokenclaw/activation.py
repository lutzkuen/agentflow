from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from tokenclaw.env import env
from tokenclaw.paths import default_config_dir as default_agentflow_config_dir, safe_home_dir
from tokenclaw.upstream_url import normalize_openai_upstream_base_url, redact_url


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
CLAUDE_DESKTOP_USER_DESKTOP_PATH = Path(".local") / "share" / "applications" / "claude-desktop.desktop"
CLAUDE_DESKTOP_SYSTEM_DESKTOP_PATH = Path("/usr/share/applications/claude-desktop.desktop")
ACTIVATION_TARGETS = ("openai", "claude", "codex", "claude-vscode", "claude-desktop")
SHELL_PROFILE_CANDIDATES = (".zshrc", ".bashrc", ".profile")


class ActivationError(ValueError):
    pass


class ActivationConfigError(ActivationError):
    def __init__(self, message: str, *, path: Path | None = None, errors: list[dict[str, str]] | None = None):
        self.path = path
        self.errors = errors or [{"path": "$", "message": message}]
        location = f" at {path}" if path is not None else ""
        super().__init__(f"Invalid AgentFlow activation config{location}: {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_config_dir() -> Path:
    return default_agentflow_config_dir()


def activation_config_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return base / DEFAULT_CONFIG_FILENAME


def default_codex_config_path() -> Path:
    return safe_home_dir() / CODEX_CONFIG_RELATIVE_PATH


def default_claude_desktop_file_path() -> Path | None:
    user_path = safe_home_dir() / CLAUDE_DESKTOP_USER_DESKTOP_PATH
    if user_path.exists():
        return user_path
    if CLAUDE_DESKTOP_SYSTEM_DESKTOP_PATH.exists():
        return CLAUDE_DESKTOP_SYSTEM_DESKTOP_PATH
    return None


def empty_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "targets": {},
    }


def _sanitize_url_for_config(raw_url: str | None) -> str | None:
    if not raw_url:
        return raw_url
    parsed = urlparse(str(raw_url))
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in {"api-key", "api_key", "apikey", "access_token", "authorization", "client_secret", "code", "key", "sig", "signature", "token"}:
            query_items.append((key, "[redacted]"))
        else:
            query_items.append((key, value))
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, urlencode(query_items, doseq=True), ""))


def _validate_url_field(errors: list[dict[str, str]], payload: dict[str, Any], key: str, path: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append({"path": f"{path}.{key}", "message": "must be a non-empty string"})
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        errors.append({"path": f"{path}.{key}", "message": "must be an http(s) URL with a host"})
        return
    if parsed.username or parsed.password:
        errors.append({"path": f"{path}.{key}", "message": "must not include URL userinfo credentials"})
        return
    sensitive_keys = {"api-key", "api_key", "apikey", "access_token", "authorization", "client_secret", "code", "key", "sig", "signature", "token"}
    for query_key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if query_key.lower() in sensitive_keys and query_value not in {"[redacted]", "%5Bredacted%5D"}:
            errors.append({"path": f"{path}.{key}", "message": f"must not include secret query parameter {query_key!r}"})
            return


def validate_activation_config(config: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not isinstance(config, dict):
        return [{"path": "$", "message": "must be a JSON object"}]
    if config.get("schema") != SCHEMA:
        errors.append({"path": "$.schema", "message": f"must be {SCHEMA!r}"})
    targets = config.get("targets")
    if not isinstance(targets, dict):
        errors.append({"path": "$.targets", "message": "must be an object"})
        return errors
    for target_name, profile in targets.items():
        target_path = f"$.targets.{target_name}"
        if not isinstance(profile, dict):
            errors.append({"path": target_path, "message": "must be an object"})
            continue
        if profile.get("id") not in {None, target_name}:
            errors.append({"path": f"{target_path}.id", "message": "must match the target key"})
        if not isinstance(profile.get("configured"), bool):
            errors.append({"path": f"{target_path}.configured", "message": "must be a boolean"})
            continue
        if not profile.get("configured"):
            continue
        if target_name not in ACTIVATION_TARGETS:
            errors.append({"path": target_path, "message": "unknown activation target"})
        provider = profile.get("provider")
        if provider not in {"openai", "anthropic"}:
            errors.append({"path": f"{target_path}.provider", "message": "must be 'openai' or 'anthropic'"})
        _validate_url_field(errors, profile, "local_base_url", target_path)
        if target_name in {"openai", "claude"}:
            _validate_url_field(errors, profile, "health_url", target_path)
        if target_name in {"openai", "claude", "claude-vscode"} and "upstream_base_url" in profile:
            _validate_url_field(errors, profile, "upstream_base_url", target_path)
        if target_name == "openai" and profile.get("openai_auth_mode") not in {"client", "proxy"}:
            errors.append({"path": f"{target_path}.openai_auth_mode", "message": "must be 'client' or 'proxy'"})
        if target_name == "codex" and not isinstance(profile.get("codex_config_path"), str):
            errors.append({"path": f"{target_path}.codex_config_path", "message": "must be a string"})
        if target_name == "claude-vscode" and not isinstance(profile.get("env_file_path"), str):
            errors.append({"path": f"{target_path}.env_file_path", "message": "must be a string"})
        if target_name == "claude-desktop" and not isinstance(profile.get("desktop_file_path"), str):
            errors.append({"path": f"{target_path}.desktop_file_path", "message": "must be a string"})
    return errors


def _raise_config_error(path: Path, errors: list[dict[str, str]]) -> None:
    first = errors[0] if errors else {"path": "$", "message": "invalid config"}
    raise ActivationConfigError(f"{first['path']} {first['message']}", path=path, errors=errors)


def load_activation_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    path = activation_config_path(config_dir)
    if not path.exists():
        return empty_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ActivationConfigError(f"invalid JSON: {exc}", path=path) from exc
    if not isinstance(payload, dict):
        raise ActivationConfigError("root must be a JSON object", path=path)
    errors = validate_activation_config(payload)
    if errors:
        _raise_config_error(path, errors)
    return payload


def write_activation_config(config: dict[str, Any], config_dir: str | Path | None = None) -> Path:
    path = activation_config_path(config_dir)
    errors = validate_activation_config(config)
    if errors:
        _raise_config_error(path, errors)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
    return path


def _config_file_paths_for_profile(profile: dict[str, Any], config_dir: str | Path | None = None) -> list[str]:
    paths = [str(activation_config_path(config_dir))]
    for key in ("codex_config_path", "env_file_path", "desktop_file_path"):
        value = profile.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


def activation_status_from_config(
    config: dict[str, Any],
    *,
    config_dir: str | Path | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    requested_targets = [target] if target else list(ACTIVATION_TARGETS)
    profiles = config.get("targets") if isinstance(config.get("targets"), dict) else {}
    targets: dict[str, Any] = {}
    for name in requested_targets:
        raw_profile = profiles.get(name) if isinstance(profiles.get(name), dict) else {}
        configured = bool(raw_profile.get("configured"))
        status: dict[str, Any] = {
            "target": name,
            "configured": configured,
            "config_path": str(activation_config_path(config_dir)),
        }
        if configured:
            status.update(
                {
                    "provider": raw_profile.get("provider"),
                    "local_base_url": raw_profile.get("local_base_url"),
                    "health_url": raw_profile.get("health_url"),
                    "upstream_base_url": redact_url(raw_profile.get("upstream_base_url")),
                    "auth_mode": raw_profile.get("openai_auth_mode"),
                    "depends_on": raw_profile.get("depends_on"),
                    "config_file_paths": list(raw_profile.get("config_file_paths") or _config_file_paths_for_profile(raw_profile, config_dir)),
                    "last_activation_at": raw_profile.get("last_activation_at") or raw_profile.get("updated_at"),
                }
            )
            for key in ("codex_config_path", "env_file_path", "desktop_file_path"):
                if raw_profile.get(key):
                    status[key] = raw_profile.get(key)
        targets[name] = status
    return {
        "schema": "agentflow.activation_status.v1",
        "ok": True,
        "config_path": str(activation_config_path(config_dir)),
        "targets": targets,
    }


def activation_status(config_dir: str | Path | None = None, *, target: str | None = None) -> dict[str, Any]:
    config = load_activation_config(config_dir)
    return activation_status_from_config(config, config_dir=config_dir, target=target)


def _target_port(target: str) -> int:
    return DEFAULT_OPENAI_PORT if target == "openai" else DEFAULT_CLAUDE_PORT


def _target_host(_: str) -> str:
    return DEFAULT_HOST


def _profile_command_args(profile: dict[str, Any], *, redact: bool = False) -> list[str]:
    target = str(profile.get("id") or "")
    provider = str(profile.get("provider") or "")
    host = env("TOKENCLAW_HOST") or str(profile.get("host") or DEFAULT_HOST)
    port = env("TOKENCLAW_PORT")
    if target == "openai" and provider == "openai":
        return [
            "--provider",
            "openai",
            "--host",
            host,
            "--port",
            str(port or profile.get("port") or DEFAULT_OPENAI_PORT),
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
            host,
            "--port",
            str(port or profile.get("port") or DEFAULT_CLAUDE_PORT),
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
        upstream_base_url = _sanitize_url_for_config(
            normalize_openai_upstream_base_url(openai_base_url or DEFAULT_OPENAI_UPSTREAM_BASE_URL)
        )
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
        }
    elif target == "claude":
        profile = {
            "id": "claude",
            "configured": True,
            "provider": "anthropic",
            "local_base_url": local_base_url or DEFAULT_CLAUDE_LOCAL_BASE_URL,
            "health_url": health_url or DEFAULT_CLAUDE_HEALTH_URL,
            "upstream_base_url": _sanitize_url_for_config(anthropic_base_url or DEFAULT_CLAUDE_UPSTREAM_BASE_URL),
            "host": _target_host(target),
            "port": _target_port(target),
        }
    else:
        raise ValueError("target must be 'openai' or 'claude'")
    activated_at = utc_now()
    profile["last_activation_at"] = activated_at
    profile["updated_at"] = activated_at
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
    profile = {
        "id": "codex",
        "configured": True,
        "provider": "openai",
        "app": "codex",
        "depends_on": "openai",
        "local_base_url": str(openai_profile.get("local_base_url") or DEFAULT_OPENAI_LOCAL_BASE_URL),
        "codex_config_path": str(codex_config_path),
        "last_activation_at": utc_now(),
    }
    profile["updated_at"] = profile["last_activation_at"]
    return profile


def claude_vscode_activation_profile(
    *,
    claude_profile: dict[str, Any],
    env_file_path: Path,
) -> dict[str, Any]:
    profile = {
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
        "last_activation_at": utc_now(),
    }
    profile["updated_at"] = profile["last_activation_at"]
    return profile


def claude_desktop_activation_profile(
    *,
    claude_profile: dict[str, Any],
    desktop_file_path: Path,
) -> dict[str, Any]:
    profile = {
        "id": "claude-desktop",
        "configured": True,
        "provider": "anthropic",
        "app": "claude-desktop",
        "depends_on": "claude",
        "local_base_url": str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL),
        "desktop_file_path": str(desktop_file_path),
        "safe_env": {
            "ANTHROPIC_BASE_URL": str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL),
        },
        "last_activation_at": utc_now(),
    }
    profile["updated_at"] = profile["last_activation_at"]
    return profile


def apply_activation_profile(
    config: dict[str, Any],
    profile: dict[str, Any],
    *,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    updated = dict(config)
    targets = dict(updated.get("targets") or {})
    profile = dict(profile)
    activated_at = str(profile.get("last_activation_at") or profile.get("updated_at") or utc_now())
    profile["last_activation_at"] = activated_at
    profile["updated_at"] = activated_at
    profile["config_file_paths"] = _config_file_paths_for_profile(profile, config_dir)
    targets[str(profile["id"])] = profile
    updated["schema"] = SCHEMA
    updated["targets"] = targets
    updated["updated_at"] = activated_at
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


def _desktop_backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".agentflow.bak")


def _split_exec_value(value: str) -> list[str]:
    try:
        return shlex.split(value, posix=True)
    except ValueError as exc:
        raise ActivationError(f"Could not parse Claude Desktop Exec line: {exc}") from exc


def _update_desktop_exec_value(value: str, local_base_url: str) -> tuple[str, bool]:
    tokens = _split_exec_value(value)
    if not tokens:
        raise ActivationError("Claude Desktop Exec line is empty")
    env_token = f"ANTHROPIC_BASE_URL={local_base_url}"
    updated = list(tokens)
    for index, token in enumerate(updated):
        if token.startswith("ANTHROPIC_BASE_URL="):
            if token == env_token:
                return value, False
            updated[index] = env_token
            return shlex.join(updated), True
    if updated[0] == "env":
        updated.insert(1, env_token)
    else:
        updated = ["env", env_token, *updated]
    return shlex.join(updated), True


def update_claude_desktop_desktop_file(raw: str, local_base_url: str) -> tuple[str, bool]:
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines(keepends=True)
    if raw and not raw.endswith(("\n", "\r")):
        lines.append("")
    for index, line in enumerate(lines):
        line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(line_ending)] if line_ending else line
        if not body.startswith("Exec="):
            continue
        updated_value, changed = _update_desktop_exec_value(body[len("Exec="):], local_base_url)
        if not changed:
            return raw, False
        updated = list(lines)
        updated[index] = f"Exec={updated_value}{line_ending or newline}"
        return "".join(updated), True
    raise ActivationError("Claude Desktop .desktop file does not contain an Exec= line")


def claude_desktop_base_url_from_desktop_file(raw: str) -> str | None:
    for line in raw.splitlines():
        if not line.startswith("Exec="):
            continue
        tokens = _split_exec_value(line[len("Exec="):])
        for token in tokens:
            if token.startswith("ANTHROPIC_BASE_URL="):
                return token.split("=", 1)[1]
        return None
    return None


def _is_project_local_codex_config(path: Path, cwd: Path | None = None) -> bool:
    base = cwd or Path.cwd()
    try:
        return path.expanduser().resolve(strict=False) == (base / CODEX_CONFIG_RELATIVE_PATH).resolve(strict=False)
    except OSError:
        return False


def claude_vscode_env_path(config_dir: str | Path | None = None) -> Path:
    base = Path(config_dir) if config_dir is not None else default_config_dir()
    return base / CLAUDE_VSCODE_ENV_FILENAME


def default_shell_profile_path() -> Path:
    home = safe_home_dir()
    for name in SHELL_PROFILE_CANDIDATES:
        path = home / name
        if path.exists():
            return path
    return home / ".profile"


def _claude_vscode_env_contents(local_base_url: str) -> str:
    return (
        "# AgentFlow-managed non-secret routing values for Claude in VS Code.\n"
        "# Keep Claude API keys in your shell or OS secret manager, not in this file.\n"
        f"ANTHROPIC_BASE_URL={local_base_url}\n"
    )


def _shell_profile_source_line(env_path: Path) -> str:
    try:
        display_path = env_path.expanduser().resolve(strict=False)
    except OSError:
        display_path = env_path.expanduser()
    return f"source {shlex.quote(str(display_path))}"


def _shell_profile_source_variants(env_path: Path) -> set[str]:
    variants = {_shell_profile_source_line(env_path)}
    home = safe_home_dir()
    try:
        relative = env_path.expanduser().resolve(strict=False).relative_to(home.expanduser().resolve(strict=False))
    except (OSError, ValueError):
        relative = None
    if relative is not None:
        tilde_path = "~/" + relative.as_posix()
        variants.add(f"source {tilde_path}")
        variants.add(f". {tilde_path}")
    variants.add(f"source {shlex.quote(str(env_path.expanduser()))}")
    variants.add(f". {shlex.quote(str(env_path.expanduser()))}")
    return variants


def _normalized_shell_source_path(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if expanded == "~":
        path = safe_home_dir()
    elif expanded.startswith("~/"):
        path = safe_home_dir() / expanded[2:]
    else:
        path = Path(expanded).expanduser()
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def _shell_profile_sources_env(profile_contents: str, env_path: Path) -> bool:
    wanted = _shell_profile_source_variants(env_path)
    wanted_path = _normalized_shell_source_path(str(env_path))
    for raw_line in profile_contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in wanted:
            return True
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) >= 2 and tokens[0] in {"source", "."} and _normalized_shell_source_path(tokens[1]) == wanted_path:
            return True
    return False


def _shell_profile_append_block(env_path: Path) -> str:
    return f"# AgentFlow\n{_shell_profile_source_line(env_path)}\n"


def activate_claude_vscode(
    *,
    config_dir: str | Path | None = None,
    dry_run: bool = False,
    auto_configure_claude: bool = True,
    shell_profile: bool = True,
) -> dict[str, Any]:
    config = load_activation_config(config_dir)
    claude_profile, created_claude = _claude_profile_from_config(config, auto_configure=auto_configure_claude)
    local_base_url = str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL)
    upstream_base_url = str(claude_profile.get("upstream_base_url") or DEFAULT_CLAUDE_UPSTREAM_BASE_URL)
    env_path = claude_vscode_env_path(config_dir)
    env_contents = _claude_vscode_env_contents(local_base_url)
    existing_env = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    env_changed = existing_env != env_contents
    shell_profile_path = default_shell_profile_path()
    existing_shell_profile = shell_profile_path.read_text(encoding="utf-8") if shell_profile_path.exists() else ""
    shell_profile_has_source = _shell_profile_sources_env(existing_shell_profile, env_path)
    shell_profile_append = _shell_profile_append_block(env_path)
    shell_profile_changed = bool(shell_profile and not shell_profile_has_source)

    vscode_profile = claude_vscode_activation_profile(claude_profile=claude_profile, env_file_path=env_path)
    if shell_profile:
        vscode_profile["shell_profile_path"] = str(shell_profile_path)
    updated_config = apply_activation_profile(config, claude_profile, config_dir=config_dir)
    updated_config = apply_activation_profile(updated_config, vscode_profile, config_dir=config_dir)
    config_path = activation_config_path(config_dir)

    if not dry_run:
        if env_changed:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(env_contents, encoding="utf-8")
        if shell_profile_changed:
            shell_profile_path.parent.mkdir(parents=True, exist_ok=True)
            prefix = "" if not existing_shell_profile or existing_shell_profile.endswith("\n") else "\n"
            shell_profile_path.write_text(existing_shell_profile + prefix + shell_profile_append, encoding="utf-8")
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
        "shell_profile_enabled": bool(shell_profile),
        "shell_profile_path": str(shell_profile_path) if shell_profile else None,
        "shell_profile_changed": shell_profile_changed,
        "shell_profile_append": shell_profile_append if shell_profile_changed else "",
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


def activate_claude_desktop(
    *,
    config_dir: str | Path | None = None,
    desktop_file_path: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    config = load_activation_config(config_dir)
    claude_profile, created_claude = _claude_profile_from_config(config, auto_configure=True)
    local_base_url = str(claude_profile.get("local_base_url") or DEFAULT_CLAUDE_LOCAL_BASE_URL)

    discovered = False
    if desktop_file_path is not None:
        path = Path(desktop_file_path).expanduser()
    else:
        found = default_claude_desktop_file_path()
        if found is None:
            raise ActivationError(
                "Claude Desktop .desktop file not found. Expected "
                f"{safe_home_dir() / CLAUDE_DESKTOP_USER_DESKTOP_PATH} or {CLAUDE_DESKTOP_SYSTEM_DESKTOP_PATH}; "
                "pass --desktop-file PATH if Claude Desktop is installed elsewhere."
            )
        path = found
        discovered = True

    system_path = path.resolve(strict=False) == CLAUDE_DESKTOP_SYSTEM_DESKTOP_PATH.resolve(strict=False)
    if not path.exists():
        raise ActivationError(f"Claude Desktop .desktop file not found: {path}")
    if system_path and not force:
        raise ActivationError(
            f"Claude Desktop launcher is system-level at {path}. Copy it to "
            f"{safe_home_dir() / CLAUDE_DESKTOP_USER_DESKTOP_PATH}, or rerun with --force if you intend to patch "
            "the system-level file and have write permission."
        )
    if system_path and force and not os.access(path, os.W_OK):
        raise ActivationError(
            f"Claude Desktop system launcher is not writeable: {path}. Rerun with suitable permissions, "
            "copy it to the user applications directory, or patch it manually."
        )

    original = path.read_text(encoding="utf-8")
    updated_desktop, desktop_changed = update_claude_desktop_desktop_file(original, local_base_url)
    backup_path = _desktop_backup_path(path)

    desktop_profile = claude_desktop_activation_profile(claude_profile=claude_profile, desktop_file_path=path)
    updated_config = apply_activation_profile(config, claude_profile, config_dir=config_dir)
    updated_config = apply_activation_profile(updated_config, desktop_profile, config_dir=config_dir)
    config_path = activation_config_path(config_dir)

    if not dry_run:
        if desktop_changed:
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
            path.write_text(updated_desktop, encoding="utf-8")
        config_path = write_activation_config(updated_config, config_dir)

    return {
        "schema": "agentflow.claude_desktop_activation_result.v1",
        "ok": True,
        "dry_run": bool(dry_run),
        "force": bool(force),
        "target": "claude-desktop",
        "configured": True,
        "config_path": str(config_path),
        "desktop_file_path": str(path),
        "desktop_file_changed": desktop_changed,
        "desktop_file_backup_path": str(backup_path) if desktop_changed or backup_path.exists() else None,
        "desktop_file_discovered": discovered,
        "desktop_file_system_level": system_path,
        "local_base_url": local_base_url,
        "depends_on": "claude",
        "claude_target_created": created_claude,
        "run_command": "agentflow run claude",
        "safe_env": dict(desktop_profile["safe_env"]),
        "profile": desktop_profile,
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
    updated_config = apply_activation_profile(config, openai_profile, config_dir=config_dir)
    updated_config = apply_activation_profile(updated_config, codex_profile, config_dir=config_dir)
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
