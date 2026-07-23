from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from tokenclaw.cli_common import (
    default_config_dir,
    default_db_path,
    default_stats_url,
    is_loopback_url as _is_loopback_url,
    open_metadata_report_store_for_db,
    open_store_for_db as _open_store_for_db,
    write_json as _write_json,
)
from tokenclaw.upstream_url import redact_url as _redact_url


ONBOARDING_TARGETS = ("openai", "claude", "codex", "claude-vscode", "claude-desktop")
UNSUPPORTED_ONBOARDING_TARGETS = ("copilot",)
RUN_TARGETS = ("openai", "claude")
DOCTOR_TARGETS = (*ONBOARDING_TARGETS, "start")
DEFAULT_STATS_URL = "http://127.0.0.1:4002/tokenclaw/stats"
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:4002/tokenclaw/dashboard"
DEFAULT_DASHBOARD_HOST = "0.0.0.0"
DEFAULT_DASHBOARD_PORT = 4002
ROUTING_EXPERIMENTS_ENV = "TOKENCLAW_ROUTING_EXPERIMENTS"
ROUTING_EXPERIMENTS_STRICT_ENV = "TOKENCLAW_ROUTING_EXPERIMENTS_STRICT"
DEFAULT_CLAUDE_PROD_PORT = 4000


def _local_url_with_path(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    if base_path == "/v1":
        base_path = ""
    target_path = "/" + path.lstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, base_path + target_path, "", ""))


def _local_sqlite_db_path() -> Path | None:
    raw = os.environ.get("TOKENCLAW_DATABASE_URL") or os.environ.get("TOKENCLAW_DB")
    if raw and not raw.startswith("sqlite:///") and "://" in raw:
        return None
    if raw:
        return Path(raw.removeprefix("sqlite:///")).expanduser()
    return Path(default_db_path()).expanduser()


def _calls_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("select count(*) from calls").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0] or 0) if row else None


def _proxy_arg_value(proxy_args: Sequence[str], flag: str) -> str | None:
    try:
        index = list(proxy_args).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(proxy_args):
        return None
    return str(proxy_args[index + 1])


def _configured_proxy_port(proxy_args: Sequence[str]) -> int | None:
    raw = _proxy_arg_value(proxy_args, "--port")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _configured_proxy_health_url(proxy_args: Sequence[str]) -> str | None:
    host = _proxy_arg_value(proxy_args, "--host") or "127.0.0.1"
    port = _configured_proxy_port(proxy_args)
    if port is None:
        return None
    return f"http://{host}:{port}/health"


def _prod_routing_experiments_path(
    *,
    target: str,
    proxy_args: Sequence[str],
    config_dir: str | Path,
) -> Path | None:
    if target != "claude":
        return None
    if _configured_proxy_port(proxy_args) != DEFAULT_CLAUDE_PROD_PORT:
        return None
    return Path(config_dir).expanduser() / "routing_experiments.yaml"


def _start_selected_provider_targets(args: Any) -> list[str]:
    if bool(getattr(args, "dashboard_only", False)):
        return []
    selected = []
    if bool(getattr(args, "openai", False)):
        selected.append("openai")
    if bool(getattr(args, "claude", False)):
        selected.append("claude")
    return selected or list(RUN_TARGETS)


def _ensure_start_activation_config(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    targets: Sequence[str],
    dry_run: bool,
) -> tuple[dict[str, Any], list[str], str]:
    from tokenclaw import activation

    updated = json.loads(json.dumps(config))
    configured_targets: list[str] = []
    changed = False
    for target in targets:
        profiles = updated.setdefault("targets", {})
        existing = profiles.get(target) if isinstance(profiles.get(target), dict) else None
        if existing and existing.get("configured"):
            continue
        profiles[target] = activation.activation_profile(target)
        configured_targets.append(target)
        changed = True
    config_path = str(activation.activation_config_path(config_dir))
    if changed and not dry_run:
        config_path = str(activation.write_activation_config(updated, config_dir))
    return updated, configured_targets, config_path


def _health_probe(url: str, *, timeout: float) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not _is_loopback_url(url):
        return False, None, "non-loopback-url"
    try:
        response = httpx.get(url, timeout=timeout)
    except Exception as exc:
        return False, None, type(exc).__name__
    if response.status_code < 200 or response.status_code >= 300:
        return False, {"status_code": response.status_code}, "health-non-2xx"
    try:
        payload = response.json()
    except ValueError:
        return False, {"status_code": response.status_code}, "health-invalid-json"
    return True, payload if isinstance(payload, dict) else {}, None


def _provider_start_command(proxy_args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "tokenclaw.server", *list(proxy_args)]


def _provider_launch_plan(
    *,
    target: str,
    proxy_args: Sequence[str],
    config_dir: str | Path,
) -> dict[str, Any]:
    durable_routing_experiments = _prod_routing_experiments_path(
        target=target,
        proxy_args=proxy_args,
        config_dir=config_dir,
    )
    env_overrides: dict[str, str] = {}
    if durable_routing_experiments is not None:
        env_overrides[ROUTING_EXPERIMENTS_ENV] = str(durable_routing_experiments)
        env_overrides[ROUTING_EXPERIMENTS_STRICT_ENV] = "1"
    env_display = dict(env_overrides)
    inherited_routing_experiments = os.environ.get(ROUTING_EXPERIMENTS_ENV)
    inherited_routing_experiments_strict = os.environ.get(ROUTING_EXPERIMENTS_STRICT_ENV)
    if ROUTING_EXPERIMENTS_ENV not in env_display and inherited_routing_experiments:
        env_display[ROUTING_EXPERIMENTS_ENV] = inherited_routing_experiments
    if ROUTING_EXPERIMENTS_STRICT_ENV not in env_display and inherited_routing_experiments_strict:
        env_display[ROUTING_EXPERIMENTS_STRICT_ENV] = inherited_routing_experiments_strict
    return {
        "target": target,
        "proxy_args": list(proxy_args),
        "port": _configured_proxy_port(proxy_args),
        "durable_routing_experiments": str(durable_routing_experiments) if durable_routing_experiments is not None else None,
        "routing_experiments": env_display.get(ROUTING_EXPERIMENTS_ENV),
        "env_overrides": env_overrides,
        "env_display": env_display,
    }


def _command_env_prefix(env_overrides: dict[str, str]) -> list[str]:
    if not env_overrides:
        return []
    return ["env", *[f"{key}={env_overrides[key]}" for key in sorted(env_overrides)]]


def _redacted_command_with_env(command: Sequence[str], env_overrides: dict[str, str] | None = None) -> str:
    return _redacted_command([*_command_env_prefix(env_overrides or {}), *list(command)])


def _apply_launch_env_overrides(env_overrides: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in env_overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_launch_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_provider_launch_log(stderr: Any, *, brand: str, launch_plan: dict[str, Any]) -> None:
    launched_path = launch_plan.get("routing_experiments")
    if launched_path:
        stderr.write(
            f"{brand} run launch: target={launch_plan.get('target')} port={launch_plan.get('port')} "
            f"routing_experiments={launched_path}\n"
        )


def _dashboard_start_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "tokenclaw.dashboard",
        "--host",
        DEFAULT_DASHBOARD_HOST,
        "--port",
        str(DEFAULT_DASHBOARD_PORT),
    ]


def _redacted_command(command: Sequence[str]) -> str:
    return " ".join(shlex_quote(part) for part in command)


def shlex_quote(value: object) -> str:
    import shlex

    return shlex.quote(str(value))


def _launch_start_service(
    *,
    name: str,
    command: Sequence[str],
    env_overrides: dict[str, str] | None = None,
    health_url: str | None,
    timeout: float,
    dry_run: bool,
) -> tuple[dict[str, Any], subprocess.Popen[Any] | None]:
    env_overrides = dict(env_overrides or {})
    result: dict[str, Any] = {
        "name": name,
        "command": _redacted_command_with_env(command, env_overrides),
        "status": "not-started",
        "started": False,
        "already_running": False,
    }
    if env_overrides:
        result["env_overrides"] = dict(env_overrides)
    if health_url:
        result["health_url"] = health_url
        healthy, payload, reason = _health_probe(health_url, timeout=timeout)
        if healthy and (not payload or bool(payload.get("ok", True))):
            result["status"] = "already running"
            result["already_running"] = True
            result["health"] = payload or {}
            return result, None
        if reason:
            result["preflight_reason"] = reason

    if dry_run:
        result["status"] = "would start"
        return result, None

    try:
        process_env = None
        if env_overrides:
            process_env = os.environ.copy()
            process_env.update(env_overrides)
        process = subprocess.Popen(list(command), env=process_env)
    except OSError as exc:
        result["status"] = "failed to start"
        result["error"] = type(exc).__name__
        return result, None

    time.sleep(0.2)
    returncode = process.poll()
    result["pid"] = process.pid
    if returncode is not None:
        result["status"] = "exited"
        result["returncode"] = returncode
        return result, None
    result["status"] = "started"
    result["started"] = True
    return result, process


def _wait_for_start_processes(processes: Sequence[subprocess.Popen[Any]]) -> int:
    active = list(processes)
    if not active:
        return 0
    exit_code = 0
    try:
        while active:
            for process in list(active):
                returncode = process.poll()
                if returncode is not None:
                    active.remove(process)
                    if returncode and exit_code == 0:
                        exit_code = int(returncode)
            if active:
                time.sleep(0.5)
    except KeyboardInterrupt:
        for process in active:
            if process.poll() is None:
                process.terminate()
        for process in active:
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        return 130
    return exit_code


def _savings_line_local(label: str, enabled: bool | None) -> str:
    if enabled is None:
        return f"- {label}: unknown"
    return f"- {label}: {'on' if enabled else 'off'}"


def _resolve_local_savings_enabled() -> tuple[bool | None, bool | None]:
    """Return (crunching_enabled, cache_enabled) from the resolved local policy.

    Both are cheap module-level booleans derived from the active policy at import
    time. Failure to import degrades to ``None`` (rendered as "unknown") so the
    start summary never overstates a feature as "on".
    """
    try:
        from tokenclaw import crunch

        crunch_enabled: bool | None = bool(crunch.CRUNCH_ENABLED)
    except Exception:
        crunch_enabled = None
    try:
        from tokenclaw import cache

        cache_enabled: bool | None = bool(cache.CACHE_ENABLED)
    except Exception:
        cache_enabled = None
    return crunch_enabled, cache_enabled


def _resolve_managed_recommendations_line() -> str:
    """Resolve the actual managed-recommendations state for the start summary.

    Per the standalone-first contract, managed recommendations are honestly OFF
    unless they are both enabled and pointed at a configured server URL. Enabled
    without a URL is effectively off because local policy stays authoritative
    (mirrors the dashboard warning in stats.py).
    """
    try:
        from tokenclaw import recommendations
        from tokenclaw.managed_mode import managed_product_mode

        product_mode = managed_product_mode()
        managed_enabled = (
            recommendations.recommendations_enabled()
            and recommendations.policy_decisions_enabled()
        )
        server_configured = recommendations.recommendation_server_configured()
        server_url = recommendations.recommendation_server_url()
    except Exception:
        return "- managed recommendations: unknown"

    if product_mode.mode == "local_only" and (product_mode.configured or product_mode.local_rules_only):
        return f"- managed recommendations: off ({product_mode.reason})"
    if managed_enabled and server_configured:
        host = urlsplit(server_url).netloc or server_url
        if product_mode.configured:
            return f"- managed recommendations: on (mode: {product_mode.mode.replace('_', '-')}, server: {host})"
        return f"- managed recommendations: on (server: {host})"
    if managed_enabled and not server_configured:
        return "- managed recommendations: off (enabled, but no server URL configured)"
    return "- managed recommendations: off"


def _write_start_summary(stdout: Any, result: dict[str, Any]) -> None:
    services = result.get("services") if isinstance(result.get("services"), dict) else {}
    openai = services.get("openai") if isinstance(services.get("openai"), dict) else {}
    claude = services.get("claude") if isinstance(services.get("claude"), dict) else {}
    dashboard = services.get("dashboard") if isinstance(services.get("dashboard"), dict) else {}

    if openai.get("local_base_url"):
        stdout.write(f"OpenAI-compatible proxy: {openai['local_base_url']}\n")
    if claude.get("local_base_url"):
        stdout.write(f"Anthropic-compatible proxy: {claude['local_base_url']}\n")
    if dashboard.get("url"):
        stdout.write(f"Dashboard: {dashboard['url']}\n")
    crunch_enabled, cache_enabled = _resolve_local_savings_enabled()
    stdout.write("\nSavings active:\n")
    stdout.write(_savings_line_local("local crunching", crunch_enabled) + "\n")
    stdout.write(_savings_line_local("local cache", cache_enabled) + "\n")
    stdout.write(_resolve_managed_recommendations_line() + "\n")
    stdout.write("\n")
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        status = service.get("status") or "unknown"
        stdout.write(f"{name}: {status}\n")
        if service.get("command"):
            stdout.write(f"  command: {service['command']}\n")
        if service.get("routing_experiments"):
            stdout.write(f"  routing_experiments: {service['routing_experiments']}\n")
        if service.get("reason"):
            stdout.write(f"  reason: {service['reason']}\n")
    if result.get("dry_run"):
        stdout.write("Dry run only; no services were started.\n")
    elif result.get("started_process_count", 0):
        stdout.write("Press Ctrl-C to stop services started by this command.\n")
    stdout.flush()


def _start_result(
    *,
    config: dict[str, Any],
    config_dir: str | Path,
    targets: Sequence[str],
    start_dashboard: bool,
    timeout: float,
    dry_run: bool,
) -> tuple[dict[str, Any], list[subprocess.Popen[Any]]]:
    from tokenclaw import activation

    result: dict[str, Any] = {
        "schema": "tokenclaw.start.v1",
        "ok": True,
        "dry_run": dry_run,
        "config_path": str(activation.activation_config_path(config_dir)),
        "services": {},
        "auto_configured_targets": [],
        "started_process_count": 0,
        "privacy": {
            "secrets_printed": False,
            "provider_credentials_stored": False,
            "managed_server_required": False,
        },
    }
    processes: list[subprocess.Popen[Any]] = []
    updated, configured_targets, config_path = _ensure_start_activation_config(
        config,
        config_dir=config_dir,
        targets=targets,
        dry_run=dry_run,
    )
    result["config_path"] = config_path
    result["auto_configured_targets"] = configured_targets

    for target in targets:
        profile = (updated.get("targets") or {}).get(target) if isinstance(updated.get("targets"), dict) else None
        if not isinstance(profile, dict) or not profile.get("configured"):
            result["services"][target] = {
                "name": target,
                "status": "disabled",
                "reason": "activation-profile-missing",
            }
            continue
        try:
            proxy_args = activation.proxy_args_for_target(updated, target)
        except Exception as exc:
            result["services"][target] = {
                "name": target,
                "status": "disabled",
                "reason": type(exc).__name__,
            }
            continue
        launch_plan = _provider_launch_plan(target=target, proxy_args=proxy_args, config_dir=config_dir)
        service, process = _launch_start_service(
            name=target,
            command=_provider_start_command(proxy_args),
            env_overrides=launch_plan["env_overrides"],
            health_url=_configured_proxy_health_url(proxy_args) or str(profile.get("health_url") or ""),
            timeout=timeout,
            dry_run=dry_run,
        )
        env_prefix = _redacted_command(_command_env_prefix(launch_plan["env_display"]))
        profile_command = activation.shell_command_for_profile(profile, redact=True)
        service["command"] = f"{env_prefix} {profile_command}" if env_prefix else profile_command
        service["local_base_url"] = profile.get("local_base_url")
        service["provider"] = profile.get("provider")
        service["routing_experiments"] = launch_plan.get("routing_experiments")
        service["port"] = launch_plan.get("port")
        result["services"][target] = service
        if process is not None:
            processes.append(process)

    if start_dashboard:
        service, process = _launch_start_service(
            name="dashboard",
            command=_dashboard_start_command(),
            env_overrides=None,
            health_url=DEFAULT_STATS_URL,
            timeout=timeout,
            dry_run=dry_run,
        )
        service["url"] = DEFAULT_DASHBOARD_URL
        result["services"]["dashboard"] = service
        if process is not None:
            processes.append(process)
    else:
        result["services"]["dashboard"] = {
            "name": "dashboard",
            "status": "disabled",
            "reason": "no-dashboard",
            "url": DEFAULT_DASHBOARD_URL,
        }

    result["started_process_count"] = len(processes)
    failed = [
        name
        for name, service in result["services"].items()
        if isinstance(service, dict) and service.get("status") in {"failed to start", "exited"}
    ]
    result["ok"] = not failed
    if failed:
        result["failed_services"] = failed
    return result, processes


def _doctor_dashboard_status(*, timeout: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "dashboard",
        "ok": False,
        "status": "configured but not running",
        "url": DEFAULT_DASHBOARD_URL,
        "stats_url": DEFAULT_STATS_URL,
        "reasons": [],
    }
    healthy, payload, reason = _health_probe(DEFAULT_STATS_URL, timeout=timeout)
    if not healthy:
        if reason:
            result["reasons"].append(reason)
        return result
    result["ok"] = True
    result["status"] = "healthy"
    result["stats"] = {
        "ok": bool(payload.get("ok", True)) if isinstance(payload, dict) else True,
        "db": payload.get("db") if isinstance(payload, dict) else None,
    }
    return result


def _start_doctor_result(
    config: dict[str, Any],
    *,
    config_dir: str | Path | None,
    timeout: float,
) -> dict[str, Any]:
    services: dict[str, dict[str, Any]] = {}
    for name in RUN_TARGETS:
        base = _target_activation_base(config, config_dir=config_dir, target=name)
        profile = _profile_for_target(config, name)
        services[name] = _doctor_provider_target(base, profile, timeout=timeout)
    services["dashboard"] = _doctor_dashboard_status(timeout=timeout)
    return {
        "schema": "tokenclaw.start_doctor.v1",
        "ok": all(bool(service.get("ok")) for service in services.values()),
        "status": "healthy" if all(bool(service.get("ok")) for service in services.values()) else "issue",
        "services": services,
        "dashboard_url": DEFAULT_DASHBOARD_URL,
    }


def _claude_desktop_routing_verification(result: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    local_base_url = str(result.get("local_base_url") or "")
    verification: dict[str, Any] = {
        "schema": "tokenclaw.claude_desktop_routing_verification.v1",
        "proxy_reachable": False,
        "health_status": "skipped",
        "test_call_status": "skipped",
        "db_entry_status": "skipped",
        "reasons": [],
    }
    if not _is_loopback_url(local_base_url):
        verification["reasons"].append("local-base-url-not-loopback")
        return verification

    health_url = _local_url_with_path(local_base_url, "/health")
    verification["health_url"] = health_url
    try:
        response = httpx.get(health_url, timeout=timeout)
    except Exception as exc:
        verification["health_status"] = "unreachable"
        verification["health_error"] = type(exc).__name__
        verification["reasons"].append("proxy-unreachable")
        return verification
    verification["health_status_code"] = response.status_code
    if response.status_code < 200 or response.status_code >= 300:
        verification["health_status"] = "unhealthy"
        verification["reasons"].append("health-non-2xx")
        return verification
    verification["health_status"] = "reachable"
    verification["proxy_reachable"] = True

    if os.environ.get("ANTHROPIC_BASE_URL") != local_base_url:
        verification["reasons"].append("shell-anthropic-base-url-not-tokenclaw")
        return verification

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key and not auth_token:
        verification["reasons"].append("anthropic-credential-env-missing")
        return verification

    db_path = _local_sqlite_db_path()
    before_count = _calls_count(db_path) if db_path is not None else None
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    elif auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ok"}],
    }
    messages_url = _local_url_with_path(local_base_url, "/v1/messages")
    verification["messages_url"] = messages_url
    try:
        smoke_response = httpx.post(messages_url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        verification["test_call_status"] = "failed"
        verification["test_call_error"] = type(exc).__name__
        verification["reasons"].append("smoke-call-failed")
        return verification
    verification["test_call_status_code"] = smoke_response.status_code
    if smoke_response.status_code < 200 or smoke_response.status_code >= 300:
        verification["test_call_status"] = "failed"
        verification["reasons"].append("smoke-call-non-2xx")
        return verification
    verification["test_call_status"] = "succeeded"

    if db_path is None:
        verification["db_entry_status"] = "skipped"
        verification["reasons"].append("db-not-local-sqlite")
        return verification
    after_count = _calls_count(db_path)
    verification["db_path"] = str(db_path)
    if before_count is None or after_count is None:
        verification["db_entry_status"] = "skipped"
        verification["reasons"].append("db-unavailable")
    elif after_count > before_count:
        verification["db_entry_status"] = "confirmed"
    else:
        verification["db_entry_status"] = "not-confirmed"
        verification["reasons"].append("db-entry-not-confirmed")
    return verification


def _write_activation_summary(stdout: Any, result: dict[str, Any], *, brand: str = "TokenClaw") -> None:
    if result["target"] == "claude-vscode":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} {brand} target: claude-vscode\n")
        stdout.write(f"Claude VS Code local {brand} base URL: {result['local_base_url']}\n")
        stdout.write(f"Upstream Anthropic base URL used by {brand}: {_redact_url(result['upstream_base_url'])}\n")
        stdout.write(f"{brand}-managed non-secret env file: {result['env_file_path']}\n")
        stdout.write(f"Env file changed: {str(result['env_file_changed']).lower()}\n")
        stdout.write(f"Systemd user env file: {result['systemd_env_file_path']}\n")
        stdout.write(f"Systemd user env file changed: {str(result['systemd_env_file_changed']).lower()}\n")
        stdout.write(f"Shell profile: {result.get('shell_profile_path') or 'skipped'}\n")
        stdout.write(f"Shell profile changed: {str(result.get('shell_profile_changed')).lower()}\n")
        if result.get("dry_run") and result.get("shell_profile_append"):
            stdout.write("Shell profile append:\n")
            stdout.write(result["shell_profile_append"])
        stdout.write(f"Depends on {brand} target: {result['depends_on']}\n")
        if result.get("claude_target_created"):
            stdout.write("Claude target was not configured; created the default Claude activation profile.\n")
        stdout.write(
            "Activation writes future launch configuration only; already-running VS Code windows "
            "and extension hosts keep their launch-time environment.\n"
        )
        stdout.write("Immediate no-logout relaunch from a terminal that already has your Claude API key:\n")
        stdout.write(result["routing_snippet"] + "\n")
        stdout.write(f"Run configured proxy: {result['run_command']}\n")
        stdout.write(f"Config file: {result['config_path']}\n")
        stdout.write(
            "GNOME and other graphical launchers read the systemd user env file at login; "
            "log out and back in, or reboot, before launching VS Code from the desktop.\n"
        )
        stdout.write(
            "For terminal-launched VS Code, open a new shell or source the shell profile before running code .\n"
        )
        stdout.write(f"{brand} does not store or print ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or token values.\n")
        return

    if result["target"] == "claude-desktop":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} {brand} target: claude-desktop\n")
        stdout.write(f"Claude Desktop file: {result['desktop_file_path']}\n")
        stdout.write(f"Desktop file changed: {str(result['desktop_file_changed']).lower()}\n")
        stdout.write(f"Systemd user env file: {result['env_file_path']}\n")
        stdout.write(f"Systemd user env file changed: {str(result['env_file_changed']).lower()}\n")
        if result.get("desktop_file_backup_path"):
            stdout.write(f"Backup: {result['desktop_file_backup_path']}\n")
        verification = result.get("routing_verification") if isinstance(result.get("routing_verification"), dict) else {}
        if verification:
            stdout.write(f"Proxy reachable: {str(bool(verification.get('proxy_reachable'))).lower()}\n")
            stdout.write(f"Test call: {verification.get('test_call_status', 'skipped')}\n")
            stdout.write(f"DB entry: {verification.get('db_entry_status', 'skipped')}\n")
        stdout.write(f"Depends on {brand} target: {result['depends_on']}\n")
        if result.get("claude_target_created"):
            stdout.write("Claude target was not configured; created the default Claude activation profile.\n")
        if not result.get("dry_run"):
            stdout.write(
                "Session environment updated. Log out and back in (or run:\n"
                "  systemctl --user import-environment ANTHROPIC_BASE_URL\n"
                ") for Claude Desktop subprocesses to pick up the new URL.\n"
            )
        stdout.write(f"Run configured proxy: {result['run_command']}\n")
        stdout.write(f"Config file: {result['config_path']}\n")
        stdout.write(f"{brand} does not store or print ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or token values.\n")
        return

    if result["target"] == "codex":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} {brand} target: codex\n")
        stdout.write(f"Codex OpenAI base URL: {result['local_base_url']}\n")
        stdout.write(f"Codex config file: {result['codex_config_path']}\n")
        stdout.write(f"Codex config changed: {str(result['codex_config_changed']).lower()}\n")
        if result.get("codex_config_backup_path"):
            stdout.write(f"Backup file: {result['codex_config_backup_path']}\n")
        stdout.write(f"Depends on {brand} target: {result['depends_on']}\n")
        if result.get("openai_target_created"):
            stdout.write("OpenAI target was not configured; created the default OpenAI activation profile.\n")
        stdout.write(f"Run configured proxy: {result['run_command']}\n")
        stdout.write(f"Config file: {result['config_path']}\n")
        stdout.write("API keys are not stored or printed by activation.\n")
        return

    prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
    stdout.write(f"{prefix} {brand} target: {result['target']}\n")
    stdout.write(f"Local base URL for clients: {result['local_base_url']}\n")
    stdout.write(f"Health URL: {result['health_url']}\n")
    stdout.write(f"Upstream provider base URL: {_redact_url(result['upstream_base_url'])}\n")
    stdout.write(f"Run configured proxy: {result['run_command']}\n")
    stdout.write(f"Equivalent proxy command: {result['proxy_command']}\n")
    stdout.write(f"Config file: {result['config_path']}\n")
    stdout.write("API keys are not stored or printed by activation.\n")


def _write_deactivation_summary(stdout: Any, result: dict[str, Any], *, brand: str = "TokenClaw") -> None:
    prefix = "Dry run: would deactivate" if result.get("dry_run") else "Deactivated"
    if result.get("target"):
        if result.get("configured_before"):
            stdout.write(f"{prefix} {brand} target: {result['target']}\n")
        else:
            stdout.write(f"{brand} target was not active: {result['target']}\n")
    for action in result.get("actions") or []:
        action_name = str(action.get("action") or "cleanup")
        path = str(action.get("path") or "")
        changed = str(bool(action.get("changed"))).lower()
        reason = str(action.get("reason") or "")
        stdout.write(f"{action_name}: changed={changed}")
        if path:
            stdout.write(f", path={path}")
        if reason:
            stdout.write(f", reason={reason}")
        stdout.write("\n")
    stdout.write(f"Config file: {result['config_path']}\n")
    if result.get("target") in {"claude-vscode", "claude-desktop"} and result.get("changed"):
        stdout.write("Open a new shell, relaunch the app, or log out and back in for inherited environment changes to clear.\n")
    if not result.get("configured_before"):
        stdout.write(f"No {brand}-managed activation profile needed removal.\n")


def _write_activation_config_error(stderr: Any, exc: Exception, *, command: str) -> None:
    stderr.write(str(exc) + "\n")
    if command == "activate":
        stderr.write(
            "Activation did not overwrite this file automatically. Move it aside, fix the JSON, "
            "or pass --config-dir to write an isolated TokenClaw config.\n"
        )


def _resolve_downroute_pocket(raw: str) -> tuple[str, str, str] | None:
    """(pocket_key, requested_family, target_family) from operator input, which
    may be a requested family ("opus") or the full key ("opus->sonnet"). None
    when it does not name a canonical pocket. We do not route the input through
    _model_family here: "opus->sonnet" contains "sonnet", so family detection
    would mis-parse the full key — we split on "->" explicitly instead."""
    from tokenclaw import downroute

    text = str(raw or "").strip().lower()
    if not text:
        return None
    if "->" in text:
        req, _, tgt = text.partition("->")
        req, tgt = req.strip(), tgt.strip()
    else:
        req, tgt = text, downroute.POCKET_TARGET_FAMILY.get(text, "")
    if not req or not tgt or downroute.POCKET_TARGET_FAMILY.get(req) != tgt:
        return None
    return (downroute.pocket_key(req, tgt), req, tgt)


def _downroute_known_pockets() -> list[str]:
    from tokenclaw import downroute

    return [downroute.pocket_key(req, tgt) for req, tgt in downroute.POCKET_TARGET_FAMILY.items()]


def _downroute_pocket_view(row: dict[str, Any]) -> dict[str, Any]:
    f = float(row.get("f") or 0.0)
    return {
        "pocket": row.get("pocket"),
        "requested_family": row.get("requested_family"),
        "target_family": row.get("target_family"),
        "f": f,
        "armed": bool(row.get("armed_at")) and f > 0.0,
        "eligible_count": int(row.get("eligible_count") or 0),
        "applied_count": int(row.get("applied_count") or 0),
        "clean_count": int(row.get("clean_count") or 0),
        "harm_count": int(row.get("harm_count") or 0),
        "harm_error_count": int(row.get("harm_error_count") or 0),
        "harm_repair_count": int(row.get("harm_repair_count") or 0),
        "last_action": row.get("last_action"),
        "armed_at": row.get("armed_at"),
        "updated_at": row.get("updated_at"),
    }


def _downroute_status_result(rows: list[dict[str, Any]], *, controller_enabled: bool) -> dict[str, Any]:
    """Overlay stored pocket rows onto the canonical pocket map so status always
    lists every armable pocket, even ones an operator has never touched (f=0)."""
    from tokenclaw import downroute

    by_pocket = {str(r.get("pocket")): r for r in rows}
    pockets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for req, tgt in downroute.POCKET_TARGET_FAMILY.items():
        key = downroute.pocket_key(req, tgt)
        seen.add(key)
        row = by_pocket.get(key)
        if row is None:
            pockets.append(
                {
                    "pocket": key,
                    "requested_family": req,
                    "target_family": tgt,
                    "f": 0.0,
                    "armed": False,
                    "eligible_count": 0,
                    "applied_count": 0,
                    "clean_count": 0,
                    "harm_count": 0,
                    "harm_error_count": 0,
                    "harm_repair_count": 0,
                    "last_action": None,
                    "armed_at": None,
                    "updated_at": None,
                }
            )
        else:
            pockets.append(_downroute_pocket_view(row))
    for key, row in by_pocket.items():
        if key not in seen:
            pockets.append(_downroute_pocket_view(row))
    return {
        "ok": True,
        "schema": "tokenclaw.downroute_status.v1",
        "controller_enabled": controller_enabled,
        "pockets": pockets,
    }


def _write_downroute_status_summary(stdout: Any, result: dict[str, Any], *, brand: str) -> None:
    controller = "on" if result.get("controller_enabled") else "off"
    stdout.write(f"{brand} downroute pockets (controller={controller}):\n")
    pockets = result.get("pockets") or []
    for p in pockets:
        applied = int(p.get("applied_count") or 0)
        harm = int(p.get("harm_count") or 0)
        harm_rate = f"{(harm / applied):.3f}" if applied else "n/a"
        state = "ARMED" if p.get("armed") else "off"
        stdout.write(
            f"  {str(p.get('pocket')):<16} {state:<5} f={float(p.get('f') or 0.0):.3f} "
            f"applied={applied} clean={int(p.get('clean_count') or 0)} "
            f"harm={harm}(err={int(p.get('harm_error_count') or 0)}/rep={int(p.get('harm_repair_count') or 0)}) "
            f"harm_rate={harm_rate}\n"
        )
    if not pockets:
        stdout.write("  (no pockets)\n")


def _write_downroute_pocket_action_summary(stdout: Any, result: dict[str, Any], *, brand: str) -> None:
    p = result.get("pocket") or {}
    controller = "on" if result.get("controller_enabled") else "off"
    state = "ARMED" if p.get("armed") else "off"
    stdout.write(
        f"{brand} downroute {result.get('command')}: {p.get('pocket')} -> "
        f"{state} f={float(p.get('f') or 0.0):.3f} (controller={controller})\n"
    )
    if p.get("armed") and controller == "off":
        stdout.write(
            "  Controller is off: f holds at this value until you change it "
            "(set TOKENCLAW_DOWNROUTE_CONTROLLER=1 to auto-tune within [f_min, f_max]).\n"
        )


def _onboarding_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
    command_name: str = "tokenclaw",
    brand: str = "TokenClaw",
) -> int:
    from tokenclaw import activation
    from tokenclaw import __version__

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    config_help = f"Local {brand} config directory, default: TOKENCLAW_CONFIG_DIR or ~/.tokenclaw."

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )

    parser = argparse.ArgumentParser(
        prog=command_name,
        description=f"{brand} local proxy onboarding and runtime commands",
        epilog=(
            "Onboarding targets: "
            + ", ".join(ONBOARDING_TARGETS)
            + ". Runtime proxy targets: "
            + ", ".join(RUN_TARGETS)
            + ". Defaults bind provider proxies to 127.0.0.1."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{command_name} {__version__}")
    parser.add_argument(
        "--config-dir",
        default=None,
        help=config_help,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        parents=[config_parent],
        help=f"Start the local {brand} savings stack from one terminal.",
    )
    start_parser.add_argument("--openai", action="store_true", help="Start only the OpenAI-compatible proxy plus dashboard.")
    start_parser.add_argument("--claude", action="store_true", help="Start only the Anthropic-compatible proxy plus dashboard.")
    start_parser.add_argument("--dashboard-only", action="store_true", help="Start only the read-only dashboard.")
    start_parser.add_argument("--no-dashboard", action="store_true", help="Start provider proxies without the dashboard.")
    start_parser.add_argument("--dry-run", action="store_true", help="Print what would start without launching services.")
    start_parser.add_argument("--timeout", type=float, default=1.0, help="Loopback health probe timeout in seconds.")

    activate_parser = subparsers.add_parser(
        "activate",
        parents=[config_parent],
        help=f"Configure a local client/API target to use {brand}.",
    )
    activate_subparsers = activate_parser.add_subparsers(dest="target", required=True)
    activate_openai = activate_subparsers.add_parser("openai", help="Configure OpenAI-compatible API traffic.")
    activate_openai.add_argument(
        "--openai-base-url",
        default=None,
        help=f"Upstream OpenAI-compatible provider base URL. The local client URL stays on {brand}.",
    )
    activate_openai.add_argument(
        "--openai-auth-mode",
        choices=("client", "proxy"),
        default=activation.DEFAULT_OPENAI_AUTH_MODE,
        help="Forward client Authorization headers or use proxy environment credentials. Default: client.",
    )
    activate_openai.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    activate_openai.add_argument("--local-base-url", default=None, help=argparse.SUPPRESS)
    activate_openai.add_argument("--health-url", default=None, help=argparse.SUPPRESS)
    activate_openai.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")

    activate_claude = activate_subparsers.add_parser("claude", help="Configure Claude/Anthropic-compatible API traffic.")
    activate_claude.add_argument(
        "--anthropic-base-url",
        default=None,
        help="Upstream Anthropic-compatible provider base URL.",
    )
    activate_claude.add_argument(
        "--claude-base-url",
        default=None,
        help="Alias for --anthropic-base-url.",
    )
    activate_claude.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    activate_claude.add_argument("--local-base-url", default=None, help=argparse.SUPPRESS)
    activate_claude.add_argument("--health-url", default=None, help=argparse.SUPPRESS)
    activate_claude.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")

    activate_claude_vscode = activate_subparsers.add_parser(
        "claude-vscode",
        aliases=["claude-code"],
        help=f"Configure Claude/Claude Code in VS Code to inherit {brand} Anthropic routing.",
    )
    activate_claude_vscode.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    activate_claude_vscode.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")
    activate_claude_vscode.add_argument(
        "--no-auto-claude",
        action="store_true",
        help=f"Require an existing `{command_name} activate claude` profile instead of creating the default one.",
    )
    activate_claude_vscode.add_argument(
        "--no-shell-profile",
        action="store_true",
        help="Skip adding the Claude Code env file source line to your shell profile.",
    )

    activate_claude_desktop = activate_subparsers.add_parser(
        "claude-desktop",
        help=f"Configure the Linux Claude Desktop launcher to use {brand} Anthropic routing.",
    )
    activate_claude_desktop.add_argument(
        "--desktop-file",
        default=None,
        help="Claude Desktop .desktop file path, default: user launcher then system launcher.",
    )
    activate_claude_desktop.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    activate_claude_desktop.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")
    activate_claude_desktop.add_argument(
        "--force",
        action="store_true",
        help="Allow patching the system-level Claude Desktop launcher if it is writeable.",
    )

    activate_codex = activate_subparsers.add_parser("codex", help="Configure Codex VS Code/Codex CLI OpenAI base URL.")
    activate_codex.add_argument(
        "--codex-config",
        default=None,
        help="User-level Codex config.toml path, default: ~/.codex/config.toml.",
    )
    activate_codex.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    activate_codex.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")
    activate_codex.add_argument(
        "--force",
        action="store_true",
        help="Confirm an explicit Codex config update while still refusing project-local .codex/config.toml.",
    )
    activate_copilot = activate_subparsers.add_parser(
        "copilot",
        help="Unsupported: GitHub Copilot is not a base-url activation target.",
    )
    activate_copilot.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    deactivate_parser = subparsers.add_parser(
        "deactivate",
        parents=[config_parent],
        help=f"Remove {brand}-managed activation profiles and local client routing hooks.",
    )
    deactivate_parser.add_argument(
        "target",
        nargs="?",
        choices=(*ONBOARDING_TARGETS, "claude-code"),
        help="Optional onboarding target label; defaults to all configured targets.",
    )
    deactivate_parser.add_argument("--dry-run", action="store_true", help="Show intended cleanup without writing files.")

    run_parser = subparsers.add_parser(
        "run",
        parents=[config_parent],
        help=f"Run a configured {brand} proxy target.",
    )
    run_parser.add_argument("target", choices=RUN_TARGETS)
    run_parser.add_argument(
        "--dry-run",
        "--print-command",
        action="store_true",
        help="Print the proxy command that would run without starting the server.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        parents=[config_parent],
        help=f"Check a configured {brand} target without exposing secrets.",
    )
    doctor_parser.add_argument("target", nargs="?", choices=DOCTOR_TARGETS)
    doctor_parser.add_argument("--timeout", type=float, default=5.0, help="Health request timeout in seconds.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    stats_parser = subparsers.add_parser(
        "stats",
        parents=[config_parent],
        help=f"Show configured {brand} activation targets.",
    )
    stats_parser.add_argument(
        "target",
        nargs="?",
        choices=ONBOARDING_TARGETS,
        help="Optional onboarding target label; defaults to all targets.",
    )
    stats_parser.add_argument(
        "--url",
        default=default_stats_url(),
        help=argparse.SUPPRESS,
    )
    stats_parser.add_argument("--timeout", type=float, default=5.0, help=argparse.SUPPRESS)
    stats_parser.add_argument("--json", action="store_true", help="Print full stats JSON.")

    db_parser = subparsers.add_parser(
        "db",
        help=f"Inspect or maintain local {brand} SQLite metadata.",
    )
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    adopt_legacy_parser = db_subparsers.add_parser(
        "adopt-legacy",
        help="Adopt legacy agentflow.sqlite3 evidence into the canonical tokenclaw.sqlite3 DB.",
    )
    adopt_legacy_parser.add_argument(
        "--db",
        default=default_db_path(),
        help="Canonical TokenClaw SQLite DB path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    adopt_legacy_parser.add_argument(
        "--from",
        dest="legacy_db",
        default=None,
        help="Legacy AgentFlow SQLite DB path, default: sibling agentflow.sqlite3 next to the canonical DB.",
    )
    adopt_legacy_parser.add_argument("--dry-run", action="store_true", help="Report rows that would be adopted without writing.")
    adopt_legacy_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    savings_parser = subparsers.add_parser(
        "savings",
        parents=[config_parent],
        help=f"Show savings opportunity report for configured {brand} targets.",
    )
    savings_subparsers = savings_parser.add_subparsers(dest="savings_command", required=True)
    savings_report_parser = savings_subparsers.add_parser(
        "report",
        help="Ranked savings opportunities from local activation config and metadata.",
    )
    savings_report_parser.add_argument(
        "--db",
        default=None,
        help="Local TokenClaw SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    savings_report_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to scan, default: 1000.",
    )
    savings_report_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    savings_report_parser.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )
    savings_loop_parser = savings_subparsers.add_parser(
        "loop-bottlenecks",
        help="Report source-traffic, legacy DB, rollup freshness, and stale-policy blockers.",
    )
    savings_loop_parser.add_argument(
        "--db",
        default=None,
        help="Canonical TokenClaw SQLite DB path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3.",
    )
    savings_loop_parser.add_argument(
        "--legacy-db",
        default=None,
        help="Legacy AgentFlow SQLite DB path, default: sibling agentflow.sqlite3 next to the canonical DB.",
    )
    savings_loop_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Bounded recent metadata rows to inspect for policy traffic, default: 1000.",
    )
    savings_loop_parser.add_argument("--active-window-hours", type=float, default=24.0, help="Source traffic window, default: 24.")
    savings_loop_parser.add_argument("--activation-min-source-rows", type=int, default=10, help="Minimum source rows before activation is considered alive, default: 10.")
    savings_loop_parser.add_argument("--rollup-max-age-hours", type=float, default=72.0, help="Maximum rollup/snapshot evidence age, default: 72.")
    savings_loop_parser.add_argument("--policy-max-age-hours", type=float, default=72.0, help="Maximum staged policy evidence age, default: 72.")
    savings_loop_parser.add_argument(
        "--adopt-legacy-preflight",
        action="store_true",
        help="Before reporting, adopt richer sibling legacy SQLite metadata into the canonical DB.",
    )
    savings_loop_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    savings_loop_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    savings_loop_parser.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help=config_help,
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help=f"Run deterministic no-provider {brand} demos.",
    )
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command", required=True)
    golden_path_parser = demo_subparsers.add_parser(
        "golden-path",
        help="Prove OpenAI/Codex local savings with fixture-backed metadata only.",
    )
    golden_path_parser.add_argument(
        "--db",
        default=None,
        help=f"Optional local {brand} SQLite path to include live OpenAI/Codex metadata evidence.",
    )
    golden_path_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for live evidence, default: 1000.",
    )
    golden_path_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    savings_demo_parser = demo_subparsers.add_parser(
        "savings",
        help="Show the no-provider OpenAI/Codex savings loop demo.",
    )
    savings_demo_parser.add_argument(
        "--db",
        default=None,
        help=f"Optional local {brand} SQLite path to include live OpenAI/Codex metadata evidence.",
    )
    savings_demo_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for live evidence, default: 1000.",
    )
    savings_demo_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    rule_drill_parser = demo_subparsers.add_parser(
        "rule-drill",
        help="Run a no-provider apply/rollback drill for one local savings rule.",
    )
    rule_drill_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    version_parser = subparsers.add_parser("version", help=f"Print the {brand} CLI version.")
    version_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    downroute_db_help = "Local TokenClaw SQLite path, default: TOKENCLAW_DB or ~/.tokenclaw/tokenclaw.sqlite3."
    downroute_pocket_help = "Pocket as a requested family (e.g. 'opus') or full key (e.g. 'opus->sonnet')."
    downroute_parser = subparsers.add_parser(
        "downroute",
        help="Inspect and arm per-pocket read-only downrouting dials (default off).",
    )
    downroute_subparsers = downroute_parser.add_subparsers(dest="downroute_command", required=True)
    downroute_status_parser = downroute_subparsers.add_parser(
        "status",
        help="Show each downroute pocket's f, armed state, and harm evidence.",
    )
    downroute_status_parser.add_argument("--db", default=None, help=downroute_db_help)
    downroute_status_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    for _sub, _sub_help in (
        ("arm", "Arm a pocket: set f to TOKENCLAW_DOWNROUTE_F_START and reset its evidence window."),
        ("disarm", "Disarm a pocket: set f to 0 (no downrouting) and reset its evidence window."),
    ):
        _p = downroute_subparsers.add_parser(_sub, help=_sub_help)
        _p.add_argument("pocket", help=downroute_pocket_help)
        _p.add_argument("--db", default=None, help=downroute_db_help)
        _p.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    downroute_setf_parser = downroute_subparsers.add_parser(
        "set-f",
        help="Set a pocket's downroute fraction f directly (operator override).",
    )
    downroute_setf_parser.add_argument("pocket", help=downroute_pocket_help)
    downroute_setf_parser.add_argument("--f", type=float, required=True, help="Downroute fraction in [0,1].")
    downroute_setf_parser.add_argument("--db", default=None, help=downroute_db_help)
    downroute_setf_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    args = parser.parse_args(argv)
    if not hasattr(args, "config_dir") or args.config_dir is None:
        args.config_dir = default_config_dir()

    if args.command == "start":
        if args.dashboard_only and args.no_dashboard:
            stderr.write("--dashboard-only and --no-dashboard cannot be combined.\n")
            return 2
        if args.dashboard_only and (args.openai or args.claude):
            stderr.write("--dashboard-only cannot be combined with --openai or --claude.\n")
            return 2
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            _write_activation_config_error(stderr, exc, command="start")
            return 2
        targets = _start_selected_provider_targets(args)
        result, processes = _start_result(
            config=config,
            config_dir=args.config_dir,
            targets=targets,
            start_dashboard=not bool(args.no_dashboard),
            timeout=float(args.timeout),
            dry_run=bool(args.dry_run),
        )
        _write_start_summary(stdout, result)
        if args.dry_run:
            return 0 if result.get("ok") else 1
        wait_code = _wait_for_start_processes(processes)
        return wait_code if wait_code else (0 if result.get("ok") else 1)

    if args.command == "activate":
        if args.target in UNSUPPORTED_ONBOARDING_TARGETS:
            stderr.write("unsupported: GitHub Copilot is not a base-url target; see README.md\n")
            return 2

        if args.target in {"claude-vscode", "claude-code"}:
            try:
                result = activation.activate_claude_vscode(
                    config_dir=args.config_dir,
                    dry_run=bool(args.dry_run),
                    auto_configure_claude=not bool(args.no_auto_claude),
                    shell_profile=not bool(args.no_shell_profile),
                )
            except activation.ActivationError as exc:
                _write_activation_config_error(stderr, exc, command="activate")
                return 2
            _write_activation_summary(stdout, result, brand=brand)
            return 0

        if args.target == "codex":
            try:
                result = activation.activate_codex(
                    config_dir=args.config_dir,
                    codex_config_path=args.codex_config,
                    dry_run=bool(args.dry_run),
                    force=bool(args.force),
                )
            except activation.ActivationError as exc:
                _write_activation_config_error(stderr, exc, command="activate")
                return 2
            _write_activation_summary(stdout, result, brand=brand)
            return 0

        if args.target == "claude-desktop":
            try:
                result = activation.activate_claude_desktop(
                    config_dir=args.config_dir,
                    desktop_file_path=args.desktop_file,
                    dry_run=bool(args.dry_run),
                    force=bool(args.force),
                )
            except activation.ActivationError as exc:
                _write_activation_config_error(stderr, exc, command="activate")
                return 2
            if not result.get("dry_run"):
                result["routing_verification"] = _claude_desktop_routing_verification(result)
            _write_activation_summary(stdout, result, brand=brand)
            return 0

        if args.target == "claude" and args.anthropic_base_url and args.claude_base_url:
            stderr.write("--anthropic-base-url and --claude-base-url are aliases; pass only one.\n")
            return 2
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            _write_activation_config_error(stderr, exc, command="activate")
            return 2
        try:
            profile = activation.activation_profile(
                args.target,
                openai_base_url=getattr(args, "openai_base_url", None),
                anthropic_base_url=getattr(args, "anthropic_base_url", None) or getattr(args, "claude_base_url", None),
                local_base_url=getattr(args, "local_base_url", None),
                health_url=getattr(args, "health_url", None),
                openai_auth_mode=getattr(args, "openai_auth_mode", activation.DEFAULT_OPENAI_AUTH_MODE),
            )
        except ValueError as exc:
            stderr.write(str(exc) + "\n")
            return 2
        updated = activation.apply_activation_profile(config, profile, config_dir=args.config_dir)
        config_path = activation.activation_config_path(args.config_dir)
        if not args.dry_run:
            config_path = activation.write_activation_config(updated, args.config_dir)
        result = activation.activation_result(
            config=updated,
            profile=profile,
            config_path=config_path,
            dry_run=bool(args.dry_run),
        )
        _write_activation_summary(stdout, result, brand=brand)
        return 0

    if args.command == "deactivate":
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            _write_activation_config_error(stderr, exc, command="deactivate")
            return 2
        targets = [args.target] if args.target else ["claude-vscode", "claude-desktop", "codex", "openai", "claude"]
        current_config = config
        results = []
        for target in targets:
            try:
                current_config, result = activation.deactivate_target(
                    current_config,
                    target,
                    config_dir=args.config_dir,
                    dry_run=bool(args.dry_run),
                )
            except activation.ActivationError as exc:
                stderr.write(str(exc) + "\n")
                return 2
            results.append(result)
            _write_deactivation_summary(stdout, result, brand=brand)
        return 0

    if args.command == "run":
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            _write_activation_config_error(stderr, exc, command="run")
            return 2
        try:
            proxy_args = activation.proxy_args_for_target(config, args.target)
        except KeyError:
            stderr.write(f"{brand} target is not configured: {args.target}. Run `{command_name} activate {args.target}` first.\n")
            return 1
        profile = config["targets"][args.target]
        launch_plan = _provider_launch_plan(
            target=args.target,
            proxy_args=proxy_args,
            config_dir=args.config_dir,
        )
        if args.dry_run:
            env_prefix = _redacted_command(_command_env_prefix(launch_plan["env_display"]))
            command = activation.shell_command_for_profile(profile, redact=True)
            stdout.write(f"{env_prefix} {command}\n" if env_prefix else command + "\n")
            return 0
        from tokenclaw import server

        previous_env = _apply_launch_env_overrides(launch_plan["env_overrides"])
        launch_plan["routing_experiments"] = (
            launch_plan["env_overrides"].get(ROUTING_EXPERIMENTS_ENV)
            or os.environ.get(ROUTING_EXPERIMENTS_ENV)
        )
        _write_provider_launch_log(stderr, brand=brand, launch_plan=launch_plan)
        try:
            server.main(proxy_args)
        finally:
            _restore_launch_env(previous_env)
        return 0

    if args.command == "doctor":
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            result = _activation_config_error_result("tokenclaw.activation_doctor.v1", args.config_dir, args.target, exc)
            if args.json:
                _write_json(stdout, result)
            else:
                stderr.write(str(exc) + "\n")
            return 1
        result = _activation_doctor_result(config, config_dir=args.config_dir, target=args.target, timeout=float(args.timeout))
        if args.json:
            _write_json(stdout, result)
        else:
            _write_activation_doctor_summary(stdout, result)
        return 0 if result["ok"] else 1

    if args.command == "stats":
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            result = _activation_config_error_result("tokenclaw.activation_stats.v1", args.config_dir, args.target, exc)
        else:
            result = _activation_stats_result(config, config_dir=args.config_dir, target=args.target)
        if args.json:
            _write_json(stdout, result)
        elif result["ok"]:
            _write_activation_stats_summary(stdout, result)
        else:
            _write_json(stderr, result)
        return 0 if result["ok"] else 1

    if args.command == "db":
        if args.db_command == "adopt-legacy":
            from tokenclaw.db_adoption import adopt_legacy_sqlite_evidence

            result = adopt_legacy_sqlite_evidence(
                canonical_db=args.db,
                legacy_db=args.legacy_db,
                dry_run=bool(args.dry_run),
            )
            if args.pretty:
                stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                _write_json(stdout, result)
            # A clean install has no legacy DB; "nothing to adopt" is a successful
            # no-op, not an error. Real adoption failures raise instead.
            return 0 if result.get("ok") or result.get("status") == "legacy-db-missing" else 1

    if args.command == "downroute":
        from tokenclaw import downroute as _downroute_mod

        db_path = getattr(args, "db", None) or default_db_path()
        controller_enabled = _downroute_mod.DownrouteConfig.from_env().controller_enabled
        store = _open_store_for_db(db_path)
        try:
            if args.downroute_command == "status":
                result = _downroute_status_result(
                    store.list_downroute_pockets(),
                    controller_enabled=controller_enabled,
                )
                if args.json:
                    _write_json(stdout, result)
                else:
                    _write_downroute_status_summary(stdout, result, brand=brand)
                return 0

            resolved = _resolve_downroute_pocket(args.pocket)
            if resolved is None:
                _write_json(
                    stderr,
                    {
                        "ok": False,
                        "schema": "tokenclaw.downroute_pocket.v1",
                        "error": "unknown-pocket",
                        "pocket": args.pocket,
                        "known": _downroute_known_pockets(),
                    },
                )
                return 2
            pocket_key, req_fam, tgt_fam = resolved
            if args.downroute_command == "arm":
                row = store.set_downroute_pocket_f(
                    pocket=pocket_key,
                    f=_downroute_mod.DownrouteConfig.from_env().f_start,
                    requested_family=req_fam,
                    target_family=tgt_fam,
                    action="arm",
                    reset_window=True,
                )
            elif args.downroute_command == "disarm":
                row = store.set_downroute_pocket_f(
                    pocket=pocket_key,
                    f=0.0,
                    requested_family=req_fam,
                    target_family=tgt_fam,
                    action="disarm",
                    reset_window=True,
                )
            else:  # set-f
                f_val = float(args.f)
                if not (0.0 <= f_val <= 1.0):
                    _write_json(
                        stderr,
                        {
                            "ok": False,
                            "schema": "tokenclaw.downroute_pocket.v1",
                            "error": "f-out-of-range",
                            "pocket": pocket_key,
                            "f": f_val,
                        },
                    )
                    return 2
                row = store.set_downroute_pocket_f(
                    pocket=pocket_key,
                    f=f_val,
                    requested_family=req_fam,
                    target_family=tgt_fam,
                    action="set",
                )
            result = {
                "ok": True,
                "schema": "tokenclaw.downroute_pocket.v1",
                "command": args.downroute_command,
                "controller_enabled": controller_enabled,
                "pocket": _downroute_pocket_view(row),
            }
            if args.json:
                _write_json(stdout, result)
            else:
                _write_downroute_pocket_action_summary(stdout, result, brand=brand)
            return 0
        finally:
            try:
                store.conn.close()
            except Exception:
                pass

    if args.command == "savings":
        if args.savings_command == "loop-bottlenecks":
            from tokenclaw.savings_loop_bottlenecks import build_savings_loop_bottlenecks_report

            db_path = getattr(args, "db", None) or default_db_path()
            store = open_metadata_report_store_for_db(db_path)
            try:
                result = build_savings_loop_bottlenecks_report(
                    store,
                    db_path=db_path,
                    legacy_db=getattr(args, "legacy_db", None),
                    config_dir=args.config_dir,
                    active_window_hours=float(args.active_window_hours),
                    activation_min_source_rows=int(args.activation_min_source_rows),
                    rollup_max_age_hours=float(args.rollup_max_age_hours),
                    policy_max_age_hours=float(args.policy_max_age_hours),
                    policy_scan_limit=int(args.limit),
                    adopt_legacy_preflight=bool(args.adopt_legacy_preflight),
                )
            finally:
                store.conn.close()
            if args.pretty:
                stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                _write_json(stdout, result)
            return 0

        from tokenclaw.savings_report import build_savings_report

        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            result = _activation_config_error_result("tokenclaw.savings_report.v1", args.config_dir, None, exc)
            _write_json(stderr, result)
            return 1

        db_path = getattr(args, "db", None) or default_db_path()
        store = None
        db_path_obj = Path(db_path)
        if db_path_obj.exists():
            try:
                store = _open_store_for_db(db_path)
            except Exception:
                pass

        try:
            result = build_savings_report(config, store=store, limit=int(getattr(args, "limit", 1000)))
        finally:
            if store is not None:
                try:
                    store.conn.close()
                except Exception:
                    pass

        if args.json:
            _write_json(stdout, result)
        else:
            summary_brand = "AgentFlow" if command_name == "tokenclaw" else brand
            _write_savings_report_summary(stdout, result, brand=summary_brand)
        return 0 if result.get("ok") else 1

    if args.command == "demo":
        if args.demo_command == "rule-drill":
            from tokenclaw.local_savings_rule_drill import build_local_savings_rule_drill_summary

            result = build_local_savings_rule_drill_summary()
            if args.json:
                _write_json(stdout, result)
            else:
                demo_brand = "AgentFlow" if command_name == "tokenclaw" else brand
                stdout.write(
                    f"{demo_brand} local savings rule drill: "
                    f"{result.get('status')} "
                    f"{result.get('rule_family')} "
                    f"applied={str(bool(result.get('applied'))).lower()} "
                    f"rollback_available={str(bool(result.get('rollback_available'))).lower()} "
                    f"rollback_success={str(bool(result.get('rollback_success'))).lower()} "
                    f"before={result.get('before_decision_state')} "
                    f"after_apply={result.get('after_apply_decision_state')} "
                    f"after_rollback={result.get('after_rollback_decision_state')}\n"
                )
            return 0 if result.get("ok") else 1

        from tokenclaw.golden_path import build_golden_path_summary

        db_path = getattr(args, "db", None)
        store = None
        if db_path:
            db_path_obj = Path(db_path)
            if db_path_obj.exists():
                try:
                    store = _open_store_for_db(db_path)
                except Exception:
                    store = None
        try:
            result = build_golden_path_summary(store=store, limit=int(getattr(args, "limit", 1000)))
        finally:
            if store is not None:
                try:
                    store.conn.close()
                except Exception:
                    pass
        if args.json:
            _write_json(stdout, result)
        else:
            demo_brand = "AgentFlow" if command_name == "tokenclaw" else brand
            heading = f"{demo_brand} savings demo" if args.demo_command == "savings" else f"{demo_brand} golden path"
            savings_label = "tokenclaw_saved" if command_name == "tokenclaw" else "tokenclaw_saved"
            stdout.write(
                f"{heading}: "
                f"{result.get('decision_status')} "
                f"{result.get('local_action_family')} "
                f"{savings_label}=${float(result.get('estimated_tokenclaw_savings_usd') or 0.0):.6f} "
                f"provider_prompt_cache_discount=${float(result.get('provider_prompt_cache_discount_usd') or 0.0):.6f} "
                f"managed_server_required={str(bool(result.get('managed_server_required'))).lower()}\n"
            )
        return 0 if result.get("ok") else 1

    if args.command == "version":
        result = {
            "schema": "tokenclaw.version.v1",
            "ok": True,
            "version": __version__,
            "package": "tokenclaw",
                "command": command_name,
        }
        if args.json:
            _write_json(stdout, result)
        else:
            stdout.write(f"{command_name} {__version__}\n")
        return 0

    parser.error("unknown command")
    return 2


def tokenclaw_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    return _onboarding_cli(argv, stdout=stdout, stderr=stderr, command_name="tokenclaw", brand="TokenClaw")


def tokenclaw_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    return _onboarding_cli(argv, stdout=stdout, stderr=stderr, command_name="tokenclaw", brand="TokenClaw")


_CODEX_OPENAI_BASE_URL_RE = re.compile(r'^(\s*openai_base_url\s*=\s*)(".*?"|\'.*?\'|[^#\n]*?)(\s+#.*)?(\r?\n)?$')


def _activation_config_error_result(
    schema: str,
    config_dir: str | Path | None,
    target: str | None,
    exc: Exception,
) -> dict[str, Any]:
    from tokenclaw import activation

    return {
        "schema": schema,
        "ok": False,
        "target": target,
        "config_path": str(activation.activation_config_path(config_dir)),
        "targets": {},
        "issues": [
            {
                "code": "activation-config-invalid",
                "message": str(exc),
                "errors": getattr(exc, "errors", []),
            }
        ],
    }


def _selected_activation_targets(target: str | None) -> list[str]:
    if target == "start":
        return []
    return [target] if target else list(ONBOARDING_TARGETS)


def _profile_for_target(config: dict[str, Any], target: str) -> dict[str, Any]:
    targets = config.get("targets") if isinstance(config.get("targets"), dict) else {}
    profile = targets.get(target) if isinstance(targets.get(target), dict) else {}
    return profile if isinstance(profile, dict) else {}


def _target_activation_base(
    config: dict[str, Any],
    *,
    config_dir: str | Path | None,
    target: str,
) -> dict[str, Any]:
    from tokenclaw import activation

    profile = _profile_for_target(config, target)
    configured = bool(profile.get("configured"))
    upstream = profile.get("upstream_base_url")
    result: dict[str, Any] = {
        "target": target,
        "status": "configured" if configured else "not configured",
        "configured": configured,
        "provider": profile.get("provider") if configured else None,
        "local_base_url": _redact_url(str(profile.get("local_base_url"))) if configured and profile.get("local_base_url") else None,
        "health_url": str(profile.get("health_url")) if configured and profile.get("health_url") else None,
        "upstream_base_url": _redact_url(str(upstream)) if configured and upstream else None,
        "reasons": [],
    }
    if configured:
        for key in (
            "codex_config_path",
            "env_file_path",
            "systemd_env_file_path",
            "desktop_file_path",
            "shell_profile_path",
            "depends_on",
        ):
            if profile.get(key):
                result[key] = str(profile.get(key))
    else:
        result["reasons"].append("activation-profile-missing")
    if target == "codex" and configured and not result.get("upstream_base_url"):
        openai_profile = _profile_for_target(config, "openai")
        upstream = openai_profile.get("upstream_base_url")
        if upstream:
            result["upstream_base_url"] = _redact_url(str(upstream))
    if target in {"claude-vscode", "claude-desktop"} and configured and not result.get("upstream_base_url"):
        claude_profile = _profile_for_target(config, "claude")
        upstream = claude_profile.get("upstream_base_url")
        if upstream:
            result["upstream_base_url"] = _redact_url(str(upstream))
    result["config_path"] = str(activation.activation_config_path(config_dir))
    return result


def _activation_stats_result(
    config: dict[str, Any],
    *,
    config_dir: str | Path | None,
    target: str | None,
) -> dict[str, Any]:
    from tokenclaw import activation
    from tokenclaw.managed_mode import managed_mode_public_meta

    targets = {
        name: _target_activation_base(config, config_dir=config_dir, target=name)
        for name in _selected_activation_targets(target)
    }
    return {
        "schema": "tokenclaw.activation_stats.v1",
        "ok": True,
        "target": target,
        "config_path": str(activation.activation_config_path(config_dir)),
        "targets": targets,
        "managed_mode": managed_mode_public_meta(),
        "activation_successor_queue_health": _activation_successor_queue_health(),
    }


def _activation_doctor_result(
    config: dict[str, Any],
    *,
    config_dir: str | Path | None,
    target: str | None,
    timeout: float,
) -> dict[str, Any]:
    from tokenclaw import activation
    from tokenclaw.managed_mode import managed_mode_public_meta
    from tokenclaw.managed_readiness import build_managed_readiness

    managed_readiness = build_managed_readiness(timeout=timeout)
    managed_ready = bool(managed_readiness.get("ok")) or managed_readiness.get("state") == "local_only"

    if target == "start":
        start = _start_doctor_result(config, config_dir=config_dir, timeout=timeout)
        return {
            "schema": "tokenclaw.activation_doctor.v1",
            "ok": bool(start.get("ok")) and managed_ready,
            "target": target,
            "config_path": str(activation.activation_config_path(config_dir)),
            "targets": {},
            "managed_mode": managed_mode_public_meta(),
            "managed_readiness": managed_readiness,
            "start": start,
            "activation_successor_queue_health": _activation_successor_queue_health(),
        }

    targets: dict[str, dict[str, Any]] = {}
    for name in _selected_activation_targets(target):
        base = _target_activation_base(config, config_dir=config_dir, target=name)
        profile = _profile_for_target(config, name)
        if name in RUN_TARGETS:
            checked = _doctor_provider_target(base, profile, timeout=timeout)
        elif name == "codex":
            checked = _doctor_codex_target(base, config)
        elif name == "claude-vscode":
            checked = _doctor_claude_vscode_target(base)
        elif name == "claude-desktop":
            checked = _doctor_claude_desktop_target(base)
        else:
            checked = base
            checked["ok"] = False
            checked["status"] = "unhealthy"
            checked["reasons"].append("unknown-target")
        targets[name] = checked
    return {
        "schema": "tokenclaw.activation_doctor.v1",
        "ok": all(bool(item.get("ok")) for item in targets.values()) and managed_ready,
        "target": target,
        "config_path": str(activation.activation_config_path(config_dir)),
        "targets": targets,
        "managed_mode": managed_mode_public_meta(),
        "managed_readiness": managed_readiness,
        "activation_successor_queue_health": _activation_successor_queue_health(),
    }


def _activation_successor_queue_health() -> dict[str, Any]:
    try:
        from tokenclaw.stats import build_activation_successor_queue_health

        return build_activation_successor_queue_health(limit=5)
    except Exception as exc:
        return {
            "schema": "tokenclaw.activation_successor_queue_health.v1",
            "status": "unavailable",
            "status_reason": f"activation successor queue health could not be loaded: {type(exc).__name__}",
            "summary": {
                "queued_action_count": 0,
                "successor_action_count": 0,
                "successor_decision_count": 0,
                "top_blocker": None,
                "top_next_action": None,
                "preview_gate_status_counts": [],
                "blocker_counts": [],
                "next_action_counts": [],
            },
            "top_entries": [],
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "absolute_paths_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "tenant_ids_included": False,
                "tool_payloads_included": False,
                "cache_keys_included": False,
                "file_paths_included": False,
                "individual_candidate_ids_included": False,
                "managed_server_calls_made": False,
                "provider_calls_made": False,
                "artifact_path_included": False,
            },
        }


def _doctor_provider_target(base: dict[str, Any], profile: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    result = dict(base)
    result["ok"] = False
    if not result.get("configured"):
        result["status"] = "not configured"
        return result
    health_url = str(result.get("health_url") or "")
    if not _is_loopback_url(health_url):
        result["status"] = "unhealthy"
        result["reasons"].append("health-url-not-loopback")
        return result

    expected_provider = "openai" if profile.get("provider") == "openai" else "anthropic"
    try:
        response = httpx.get(health_url, timeout=timeout)
    except Exception as exc:
        result["status"] = "configured but not running"
        result["reasons"].append("health-unreachable")
        result["health_error"] = type(exc).__name__
        return result

    result["health_status_code"] = response.status_code
    if response.status_code < 200 or response.status_code >= 300:
        result["status"] = "unhealthy"
        result["reasons"].append("health-non-2xx")
        return result
    try:
        health = response.json()
    except ValueError:
        result["status"] = "unhealthy"
        result["reasons"].append("health-invalid-json")
        return result
    if not isinstance(health, dict):
        result["status"] = "unhealthy"
        result["reasons"].append("health-invalid-payload")
        return result

    health_provider = str(health.get("provider") or "")
    health_upstream = _redact_url(str(health.get("upstream") or "")) if health.get("upstream") else None
    result["health"] = {
        "ok": bool(health.get("ok")),
        "provider": health_provider,
        "upstream_base_url": health_upstream,
        "openai_auth_mode": health.get("openai_auth_mode"),
    }
    if health_provider != expected_provider:
        result["status"] = "provider mismatch"
        result["reasons"].append("provider-mismatch")
        result["expected_provider"] = expected_provider
        return result
    configured_upstream = result.get("upstream_base_url")
    if configured_upstream and health_upstream and configured_upstream != health_upstream:
        result["status"] = "stale base url"
        result["reasons"].append("upstream-mismatch")
        result["running_upstream_base_url"] = health_upstream
        return result
    if not health.get("ok"):
        result["status"] = "unhealthy"
        result["reasons"].append("health-ok-false")
        return result

    result["status"] = "healthy"
    result["ok"] = True
    return result


def _decode_toml_string(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            decoded = json.loads(value)
        except ValueError:
            return value[1:-1]
        return decoded if isinstance(decoded, str) else str(decoded)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _codex_openai_base_url_from_toml(raw: str) -> str | None:
    for line in raw.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and not stripped.startswith("#"):
            return None
        match = _CODEX_OPENAI_BASE_URL_RE.match(line)
        if match:
            return _decode_toml_string(match.group(2))
    return None


def _doctor_codex_target(base: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    result["ok"] = False
    if not result.get("configured"):
        result["status"] = "not configured"
        return result
    expected = str(result.get("local_base_url") or "")
    path_value = result.get("codex_config_path")
    if not path_value:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("codex-config-path-missing")
        return result
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("codex-config-missing")
        return result
    try:
        configured_base_url = _codex_openai_base_url_from_toml(path.read_text(encoding="utf-8"))
    except OSError as exc:
        result["status"] = "unhealthy"
        result["reasons"].append("codex-config-unreadable")
        result["config_error"] = type(exc).__name__
        return result
    result["codex_openai_base_url"] = _redact_url(configured_base_url) if configured_base_url else None
    if not configured_base_url:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("codex-openai-base-url-missing")
        return result
    if configured_base_url != expected:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("codex-openai-base-url-mismatch")
        return result
    openai_profile = _profile_for_target(config, "openai")
    if openai_profile and not openai_profile.get("configured"):
        result["reasons"].append("openai-profile-not-configured")
    result["status"] = "healthy"
    result["ok"] = True
    return result


def _env_file_value(path: Path, key: str) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _doctor_claude_vscode_target(base: dict[str, Any]) -> dict[str, Any]:
    from tokenclaw import activation

    result = dict(base)
    result["ok"] = False
    if not result.get("configured"):
        result["status"] = "not configured"
        return result
    expected = str(result.get("local_base_url") or "")
    env_path_value = result.get("env_file_path")
    env_file_base_url = None
    if env_path_value:
        env_path = Path(str(env_path_value)).expanduser()
        result["env_file_exists"] = env_path.exists()
        env_file_base_url = _env_file_value(env_path, "ANTHROPIC_BASE_URL") if env_path.exists() else None
        result["env_file_base_url"] = _redact_url(env_file_base_url) if env_file_base_url else None
    else:
        result["env_file_exists"] = False
    if env_file_base_url != expected:
        result["status"] = "not routed via tokenclaw" if not env_file_base_url else "stale base url"
        result["reasons"].append("claude-vscode-env-file-missing" if not env_file_base_url else "claude-vscode-env-file-mismatch")
        return result

    systemd_env_file_base_url = None
    systemd_env_path_value = result.get("systemd_env_file_path")
    if systemd_env_path_value:
        systemd_env_path = Path(str(systemd_env_path_value)).expanduser()
        result["systemd_env_file_exists"] = systemd_env_path.exists()
        systemd_env_file_base_url = _env_file_value(systemd_env_path, "ANTHROPIC_BASE_URL") if systemd_env_path.exists() else None
        result["systemd_env_file_base_url"] = _redact_url(systemd_env_file_base_url) if systemd_env_file_base_url else None
        if systemd_env_file_base_url != expected:
            result["reasons"].append(
                "systemd-env-file-missing" if not systemd_env_file_base_url else "systemd-env-file-mismatch"
            )
    else:
        result["systemd_env_file_exists"] = False
        result["reasons"].append("systemd-env-file-path-missing")

    configured_shell_path_value = result.get("shell_profile_path")
    configured_shell_path = Path(str(configured_shell_path_value)).expanduser() if configured_shell_path_value else None
    current_shell_path = activation.default_shell_profile_path()
    result["current_shell_profile_path"] = str(current_shell_path)
    result["current_shell_profile_exists"] = current_shell_path.exists()
    result["current_shell_profile_matches_activation"] = (
        configured_shell_path is not None
        and configured_shell_path.expanduser().resolve(strict=False) == current_shell_path.expanduser().resolve(strict=False)
    )
    current_shell_profile_raw = current_shell_path.read_text(encoding="utf-8") if current_shell_path.exists() else ""
    result["current_shell_profile_sources_env_file"] = bool(
        env_path_value and activation._shell_profile_sources_env(current_shell_profile_raw, Path(str(env_path_value)).expanduser())
    )
    if not result["current_shell_profile_sources_env_file"]:
        result["reasons"].append("current-shell-profile-does-not-source-tokenclaw-env")
    elif not result["current_shell_profile_matches_activation"]:
        result["reasons"].append("current-shell-profile-differs-from-activation-profile")

    current_shell_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    result["current_shell_base_url"] = _redact_url(current_shell_base_url) if current_shell_base_url else None
    if current_shell_base_url and current_shell_base_url != expected:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("current-shell-anthropic-base-url-mismatch")
        result["reasons"].append("vscode-runtime-env-uncertain")
        result["next_steps"] = [
            f"Run `ANTHROPIC_BASE_URL={expected} code .` from a terminal to relaunch VS Code through TokenClaw now.",
            "Log out and back in, or reboot, before launching VS Code from GNOME or another desktop launcher.",
        ]
        return result
    if current_shell_base_url == expected:
        result["status"] = "healthy"
        result["reasons"].append("current-shell-routed")
        result["ok"] = True
    else:
        result["status"] = "configured on disk; current session not routed"
        result["reasons"].append("shell-env-missing")
        result["reasons"].append("activated-on-disk-runtime-env-missing")
        result["next_steps"] = [
            f"Run `ANTHROPIC_BASE_URL={expected} code .` from a terminal to relaunch VS Code through TokenClaw now.",
            "Log out and back in, or reboot, before launching VS Code from GNOME or another desktop launcher.",
            "Already-running VS Code windows and extension hosts must be fully quit and relaunched.",
        ]
    result["reasons"].append("vscode-runtime-env-uncertain")
    return result


def _doctor_claude_desktop_target(base: dict[str, Any]) -> dict[str, Any]:
    from tokenclaw import activation

    result = dict(base)
    result["ok"] = False
    if not result.get("configured"):
        result["status"] = "not configured"
        return result
    expected = str(result.get("local_base_url") or "")
    desktop_path_value = result.get("desktop_file_path")
    if not desktop_path_value:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("claude-desktop-file-path-missing")
        return result
    path = Path(str(desktop_path_value)).expanduser()
    result["desktop_file_exists"] = path.exists()
    if not path.exists():
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("claude-desktop-file-missing")
        return result
    try:
        configured_base_url = activation.claude_desktop_base_url_from_desktop_file(path.read_text(encoding="utf-8"))
    except activation.ActivationError as exc:
        result["status"] = "unhealthy"
        result["reasons"].append("claude-desktop-exec-parse-error")
        result["config_error"] = str(exc)
        return result
    except OSError as exc:
        result["status"] = "unhealthy"
        result["reasons"].append("claude-desktop-file-unreadable")
        result["config_error"] = type(exc).__name__
        return result
    result["desktop_file_base_url"] = _redact_url(configured_base_url) if configured_base_url else None
    if not configured_base_url:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("claude-desktop-base-url-missing")
        return result
    if configured_base_url != expected:
        result["status"] = "stale base url"
        result["reasons"].append("claude-desktop-base-url-mismatch")
        return result
    env_path_value = result.get("env_file_path") or str(activation.default_claude_desktop_systemd_env_path())
    env_path = Path(str(env_path_value)).expanduser()
    result["env_file_path"] = str(env_path)
    result["env_file_exists"] = env_path.exists()
    if not env_path.exists():
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("env-file-missing")
        return result
    try:
        env_base_url = activation.claude_desktop_base_url_from_systemd_env_file(env_path.read_text(encoding="utf-8"))
    except OSError as exc:
        result["status"] = "unhealthy"
        result["reasons"].append("env-file-unreadable")
        result["config_error"] = type(exc).__name__
        return result
    result["env_file_base_url"] = _redact_url(env_base_url) if env_base_url else None
    if not env_base_url:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("env-file-base-url-missing")
        return result
    if env_base_url != expected:
        result["status"] = "stale base url"
        result["reasons"].append("env-file-base-url-mismatch")
        return result
    result["status"] = "healthy"
    result["ok"] = True
    return result


def _write_activation_stats_summary(stdout: Any, result: dict[str, Any]) -> None:
    managed = result.get("managed_mode") if isinstance(result.get("managed_mode"), dict) else {}
    if managed and (managed.get("configured") or managed.get("local_rules_only") or managed.get("server_calls_enabled")):
        stdout.write(
            f"managed: {managed.get('mode')} "
            f"(server calls: {str(bool(managed.get('server_calls_enabled'))).lower()}, "
            f"local apply: {str(bool(managed.get('local_application_enabled'))).lower()})\n"
        )
    targets = result.get("targets") if isinstance(result.get("targets"), dict) else {}
    for name in _selected_activation_targets(result.get("target")):
        target = targets.get(name) if isinstance(targets.get(name), dict) else {}
        if not target.get("configured"):
            stdout.write(f"{name}: not configured\n")
            continue
        parts = [f"{name}: configured"]
        if target.get("local_base_url"):
            parts.append(f"base url: {target['local_base_url']}")
        if target.get("upstream_base_url"):
            parts.append(f"upstream: {target['upstream_base_url']}")
        if target.get("desktop_file_path"):
            parts.append(f"desktop file: {target['desktop_file_path']}")
        stdout.write(", ".join(parts) + "\n")


def _write_activation_doctor_summary(stdout: Any, result: dict[str, Any]) -> None:
    managed = result.get("managed_mode") if isinstance(result.get("managed_mode"), dict) else {}
    if managed and (managed.get("configured") or managed.get("local_rules_only") or managed.get("server_calls_enabled")):
        stdout.write(
            f"managed: {managed.get('mode')} "
            f"(server calls: {str(bool(managed.get('server_calls_enabled'))).lower()}, "
            f"local apply: {str(bool(managed.get('local_application_enabled'))).lower()})\n"
        )
    readiness = result.get("managed_readiness") if isinstance(result.get("managed_readiness"), dict) else {}
    if readiness and (managed.get("configured") or managed.get("local_rules_only") or managed.get("server_calls_enabled")):
        reasons = readiness.get("reason_codes") if isinstance(readiness.get("reason_codes"), list) else []
        line = f"managed readiness: {readiness.get('state') or readiness.get('status') or 'unknown'}"
        if reasons:
            line += " (" + ", ".join(str(reason) for reason in reasons) + ")"
        stdout.write(line + "\n")
    if result.get("target") == "start":
        start = result.get("start") if isinstance(result.get("start"), dict) else {}
        stdout.write(f"start: {start.get('status') or 'unknown'}\n")
        services = start.get("services") if isinstance(start.get("services"), dict) else {}
        for name in ("openai", "claude", "dashboard"):
            service = services.get(name) if isinstance(services.get(name), dict) else {}
            status = service.get("status") or "unknown"
            stdout.write(f"{name}: {status}\n")
            if service.get("local_base_url"):
                stdout.write(f"  base url: {service['local_base_url']}\n")
            if service.get("url"):
                stdout.write(f"  url: {service['url']}\n")
            reasons = service.get("reasons") if isinstance(service.get("reasons"), list) else []
            if reasons:
                stdout.write("  reasons: " + ", ".join(str(reason) for reason in reasons) + "\n")
        return

    targets = result.get("targets") if isinstance(result.get("targets"), dict) else {}
    for name in _selected_activation_targets(result.get("target")):
        target = targets.get(name) if isinstance(targets.get(name), dict) else {}
        status = target.get("status") or "unknown"
        parts = [f"{name}: {status}"]
        if target.get("local_base_url"):
            parts.append(f"base url: {target['local_base_url']}")
        if target.get("health_url"):
            parts.append(f"health url: {target['health_url']}")
        if target.get("upstream_base_url"):
            parts.append(f"upstream: {target['upstream_base_url']}")
        if target.get("running_upstream_base_url"):
            parts.append(f"running upstream: {target['running_upstream_base_url']}")
        if target.get("desktop_file_path"):
            parts.append(f"desktop file: {target['desktop_file_path']}")
        if target.get("env_file_path"):
            parts.append(f"env file: {target['env_file_path']}")
        if target.get("reasons"):
            parts.append("reasons: " + ", ".join(str(reason) for reason in target["reasons"]))
        stdout.write(", ".join(parts) + "\n")
        next_steps = target.get("next_steps") if isinstance(target.get("next_steps"), list) else []
        for step in next_steps:
            stdout.write(f"  next: {step}\n")


def _write_savings_report_summary(stdout: Any, result: dict[str, Any], *, brand: str = "TokenClaw") -> None:
    opportunities = result.get("opportunities") if isinstance(result.get("opportunities"), list) else []
    count = len(opportunities)
    stdout.write(f"{brand} savings report: {count} opportunit{'y' if count == 1 else 'ies'}\n")
    for opp in opportunities:
        if not isinstance(opp, dict):
            continue
        family = str(opp.get("opportunity_family") or "unknown")
        target = str(opp.get("target") or "unknown")
        bucket = str(opp.get("projected_savings_bucket") or "unknown")
        blockers = opp.get("blocker_codes") or []
        blocker_str = ", ".join(str(b) for b in blockers) if blockers else "none"
        suggested = opp.get("suggested_command")
        parts = [f"  {family} ({target}): bucket={bucket}, blockers=[{blocker_str}]"]
        if suggested:
            parts.append(f"next: {suggested}")
        stdout.write(", ".join(parts) + "\n")


def _fetch_tokenclaw_stats(*, url: str = DEFAULT_STATS_URL, timeout: float = 5.0, target: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "tokenclaw.stats_cli.v1",
        "ok": False,
        "target": target,
        "url": url,
        "issues": [],
    }
    if not _is_loopback_url(url):
        result["issues"].append({
            "code": "non-loopback-url",
            "message": "tokenclaw stats only reads loopback URLs by default.",
        })
        return result
    try:
        response = httpx.get(url, timeout=timeout)
    except Exception as exc:
        result["issues"].append({
            "code": "stats-unreachable",
            "message": f"Could not reach TokenClaw stats URL: {type(exc).__name__}",
        })
        return result
    result["status_code"] = response.status_code
    if response.status_code >= 400:
        result["issues"].append({
            "code": "stats-error",
            "message": f"TokenClaw stats returned HTTP {response.status_code}.",
        })
        return result
    try:
        payload = response.json()
    except ValueError:
        result["issues"].append({"code": "stats-invalid-json", "message": "TokenClaw stats did not return JSON."})
        return result
    if not isinstance(payload, dict):
        result["issues"].append({"code": "stats-invalid-payload", "message": "TokenClaw stats payload is not an object."})
        return result
    result["ok"] = True
    result["stats"] = payload
    return result


def _write_stats_summary(stdout: Any, result: dict[str, Any]) -> None:
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    stdout.write("TokenClaw stats status=ok\n")
    stdout.write(f"Stats URL: {result.get('url')}\n")
    if result.get("target"):
        stdout.write(f"Target: {result.get('target')}\n")
    for label, key in (
        ("Calls", "calls"),
        ("Cache hit rate", "cache_hit_rate"),
        ("DB", "db"),
    ):
        if key in stats:
            stdout.write(f"{label}: {stats.get(key)}\n")
    activation_status = result.get("activation") if isinstance(result.get("activation"), dict) else {}
    targets = activation_status.get("targets") if isinstance(activation_status.get("targets"), dict) else {}
    if result.get("target") and result.get("target") in targets:
        target_status = targets[result["target"]]
        stdout.write(f"Configured: {str(bool(target_status.get('configured'))).lower()}\n")
        if target_status.get("local_base_url"):
            stdout.write(f"Local base URL: {target_status.get('local_base_url')}\n")


def _doctor_activation_target(profile: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    expected_provider = "openai" if profile.get("provider") == "openai" else "anthropic"
    configured_upstream = _redact_url(str(profile.get("upstream_base_url") or ""))
    health_url = str(profile.get("health_url") or "")
    result: dict[str, Any] = {
        "schema": "tokenclaw.activation_doctor.v1",
        "target": profile.get("id"),
        "ok": False,
        "configured": True,
        "provider": expected_provider,
        "local_base_url": profile.get("local_base_url"),
        "health_url": health_url,
        "configured_upstream": configured_upstream,
        "issues": [],
    }
    try:
        response = httpx.get(health_url, timeout=timeout)
    except Exception as exc:
        result["issues"].append({
            "code": "health-unreachable",
            "message": f"Could not reach TokenClaw health URL: {type(exc).__name__}",
        })
        return result
    result["health_status_code"] = response.status_code
    if response.status_code >= 400:
        result["issues"].append({
            "code": "health-error",
            "message": f"TokenClaw health returned HTTP {response.status_code}.",
        })
        return result
    try:
        health = response.json()
    except ValueError:
        result["issues"].append({"code": "health-invalid-json", "message": "TokenClaw health did not return JSON."})
        return result
    health_provider = str(health.get("provider") or "")
    health_upstream = _redact_url(str(health.get("upstream") or ""))
    result["health"] = {
        "ok": bool(health.get("ok")),
        "provider": health_provider,
        "upstream": health_upstream,
        "openai_auth_mode": health.get("openai_auth_mode"),
    }
    if health_provider != expected_provider:
        result["issues"].append({
            "code": "provider-mismatch",
            "message": f"Running proxy provider is {health_provider or '<missing>'}, expected {expected_provider}.",
        })
    if configured_upstream and health_upstream and configured_upstream != health_upstream:
        result["issues"].append({
            "code": "upstream-mismatch",
            "message": "Running proxy upstream does not match the activation profile.",
            "configured_upstream": configured_upstream,
            "running_upstream": health_upstream,
        })
    if not health.get("ok"):
        result["issues"].append({"code": "health-not-ok", "message": "TokenClaw health reports ok=false."})
    result["ok"] = not result["issues"]
    return result


def _write_doctor_summary(stdout: Any, result: dict[str, Any]) -> None:
    status = "ok" if result["ok"] else "issue"
    stdout.write(f"TokenClaw doctor target={result['target']} status={status}\n")
    stdout.write(f"Local base URL for clients: {result.get('local_base_url')}\n")
    stdout.write(f"Health URL: {result.get('health_url')}\n")
    stdout.write(f"Configured upstream: {result.get('configured_upstream')}\n")
    health = result.get("health") if isinstance(result.get("health"), dict) else {}
    if health:
        stdout.write(f"Running provider: {health.get('provider')}\n")
        stdout.write(f"Running upstream: {health.get('upstream')}\n")
    for issue in result.get("issues") or []:
        stdout.write(f"- {issue.get('code')}: {issue.get('message')}\n")
