from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx

from agentflow_proxy.optimization.cli_support import (
    open_store_for_db as _open_store_for_db,
    redact_secret as _redact_secret,
    write_json as _write_json,
)
from agentflow_proxy.upstream_url import redact_url as _redact_url


POLICY_RELOAD_PATH = "/agentflow/admin/reload-policies"
POLICY_BUNDLE_RECOMMENDATION_URL_ENV = "AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL"
PATTERN_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_PATTERN_ROLLOUT_ACTIONS_URL"
OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_OPTIMIZATION_ROLLOUT_ACTIONS_URL"
SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV = "AGENTFLOW_SCAFFOLD_ROLLOUT_ACTIONS_URL"
MANAGED_POLICY_API_KEY_ENV = "AGENTFLOW_MANAGED_API_KEY"
ONBOARDING_TARGETS = ("openai", "claude", "codex", "claude-vscode")
RUN_TARGETS = ("openai", "claude")
DEFAULT_STATS_URL = "http://127.0.0.1:4002/agentflow/stats"


def _write_activation_summary(stdout: Any, result: dict[str, Any]) -> None:
    if result["target"] == "claude-vscode":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} AgentFlow target: claude-vscode\n")
        stdout.write(f"Claude VS Code local AgentFlow base URL: {result['local_base_url']}\n")
        stdout.write(f"Upstream Anthropic base URL used by AgentFlow: {_redact_url(result['upstream_base_url'])}\n")
        stdout.write(f"AgentFlow-managed non-secret env file: {result['env_file_path']}\n")
        stdout.write(f"Env file changed: {str(result['env_file_changed']).lower()}\n")
        stdout.write(f"Depends on AgentFlow target: {result['depends_on']}\n")
        if result.get("claude_target_created"):
            stdout.write("Claude target was not configured; created the default Claude activation profile.\n")
        stdout.write("Routing snippet for a terminal that already has your Claude API key:\n")
        stdout.write(result["routing_snippet"] + "\n")
        stdout.write(f"Run configured proxy: {result['run_command']}\n")
        stdout.write(f"Config file: {result['config_path']}\n")
        stdout.write(
            "VS Code extensions usually inherit environment variables only from the VS Code process; "
            "restart VS Code from that terminal if it was opened from the desktop.\n"
        )
        stdout.write("AgentFlow does not store or print ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or token values.\n")
        return

    if result["target"] == "codex":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} AgentFlow target: codex\n")
        stdout.write(f"Codex OpenAI base URL: {result['local_base_url']}\n")
        stdout.write(f"Codex config file: {result['codex_config_path']}\n")
        stdout.write(f"Codex config changed: {str(result['codex_config_changed']).lower()}\n")
        if result.get("codex_config_backup_path"):
            stdout.write(f"Backup file: {result['codex_config_backup_path']}\n")
        stdout.write(f"Depends on AgentFlow target: {result['depends_on']}\n")
        if result.get("openai_target_created"):
            stdout.write("OpenAI target was not configured; created the default OpenAI activation profile.\n")
        stdout.write(f"Run configured proxy: {result['run_command']}\n")
        stdout.write(f"Config file: {result['config_path']}\n")
        stdout.write("API keys are not stored or printed by activation.\n")
        return

    prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
    stdout.write(f"{prefix} AgentFlow target: {result['target']}\n")
    stdout.write(f"Local base URL for clients: {result['local_base_url']}\n")
    stdout.write(f"Health URL: {result['health_url']}\n")
    stdout.write(f"Upstream provider base URL: {_redact_url(result['upstream_base_url'])}\n")
    stdout.write(f"Run configured proxy: {result['run_command']}\n")
    stdout.write(f"Equivalent proxy command: {result['proxy_command']}\n")
    stdout.write(f"Config file: {result['config_path']}\n")
    stdout.write("API keys are not stored or printed by activation.\n")


def agentflow_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    from agentflow_proxy import activation
    from agentflow_proxy import __version__

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
    )

    parser = argparse.ArgumentParser(
        description="AgentFlow local proxy onboarding and runtime commands",
        epilog=(
            "Onboarding targets: "
            + ", ".join(ONBOARDING_TARGETS)
            + ". Runtime proxy targets: "
            + ", ".join(RUN_TARGETS)
            + ". Defaults bind provider proxies to 127.0.0.1."
        ),
    )
    parser.add_argument("--version", action="version", version=f"agentflow {__version__}")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    activate_parser = subparsers.add_parser(
        "activate",
        parents=[config_parent],
        help="Configure a local client/API target to use AgentFlow.",
    )
    activate_subparsers = activate_parser.add_subparsers(dest="target", required=True)
    activate_openai = activate_subparsers.add_parser("openai", help="Configure OpenAI-compatible API traffic.")
    activate_openai.add_argument(
        "--openai-base-url",
        default=None,
        help="Upstream OpenAI-compatible provider base URL. The local client URL stays on AgentFlow.",
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
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
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
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
    )
    activate_claude.add_argument("--local-base-url", default=None, help=argparse.SUPPRESS)
    activate_claude.add_argument("--health-url", default=None, help=argparse.SUPPRESS)
    activate_claude.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")

    activate_claude_vscode = activate_subparsers.add_parser(
        "claude-vscode",
        aliases=["claude-code"],
        help="Configure Claude/Claude Code in VS Code to inherit AgentFlow Anthropic routing.",
    )
    activate_claude_vscode.add_argument(
        "--config-dir",
        default=argparse.SUPPRESS,
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
    )
    activate_claude_vscode.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")
    activate_claude_vscode.add_argument(
        "--no-auto-claude",
        action="store_true",
        help="Require an existing `agentflow activate claude` profile instead of creating the default one.",
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
        help="Local AgentFlow config directory, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow.",
    )
    activate_codex.add_argument("--dry-run", action="store_true", help="Show intended changes without writing config.")
    activate_codex.add_argument(
        "--force",
        action="store_true",
        help="Confirm an explicit Codex config update while still refusing project-local .codex/config.toml.",
    )

    run_parser = subparsers.add_parser(
        "run",
        parents=[config_parent],
        help="Run a configured AgentFlow proxy target.",
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
        help="Check a configured AgentFlow target without exposing secrets.",
    )
    doctor_parser.add_argument("target", choices=ONBOARDING_TARGETS)
    doctor_parser.add_argument("--timeout", type=float, default=5.0, help="Health request timeout in seconds.")
    doctor_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    stats_parser = subparsers.add_parser(
        "stats",
        help="Fetch local AgentFlow stats from the read-only dashboard API.",
    )
    stats_parser.add_argument(
        "target",
        nargs="?",
        choices=ONBOARDING_TARGETS,
        help="Optional onboarding target label for docs/scripts; stats are read from the local API.",
    )
    stats_parser.add_argument(
        "--url",
        default=os.getenv("AGENTFLOW_STATS_URL", DEFAULT_STATS_URL),
        help=f"Local stats URL, default: {DEFAULT_STATS_URL}.",
    )
    stats_parser.add_argument("--timeout", type=float, default=5.0, help="Stats request timeout in seconds.")
    stats_parser.add_argument("--json", action="store_true", help="Print full stats JSON.")

    version_parser = subparsers.add_parser("version", help="Print the AgentFlow CLI version.")
    version_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    args = parser.parse_args(argv)
    if not hasattr(args, "config_dir") or args.config_dir is None:
        args.config_dir = os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow"))

    if args.command == "activate":
        if args.target in {"claude-vscode", "claude-code"}:
            try:
                result = activation.activate_claude_vscode(
                    config_dir=args.config_dir,
                    dry_run=bool(args.dry_run),
                    auto_configure_claude=not bool(args.no_auto_claude),
                )
            except activation.ActivationError as exc:
                stderr.write(str(exc) + "\n")
                return 2
            _write_activation_summary(stdout, result)
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
                stderr.write(str(exc) + "\n")
                return 2
            _write_activation_summary(stdout, result)
            return 0

        if args.target == "claude" and args.anthropic_base_url and args.claude_base_url:
            stderr.write("--anthropic-base-url and --claude-base-url are aliases; pass only one.\n")
            return 2
        config = activation.load_activation_config(args.config_dir)
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
        updated = activation.apply_activation_profile(config, profile)
        config_path = activation.activation_config_path(args.config_dir)
        if not args.dry_run:
            config_path = activation.write_activation_config(updated, args.config_dir)
        result = activation.activation_result(
            config=updated,
            profile=profile,
            config_path=config_path,
            dry_run=bool(args.dry_run),
        )
        _write_activation_summary(stdout, result)
        return 0

    if args.command == "run":
        config = activation.load_activation_config(args.config_dir)
        try:
            proxy_args = activation.proxy_args_for_target(config, args.target)
        except KeyError:
            stderr.write(f"AgentFlow target is not configured: {args.target}. Run `agentflow activate {args.target}` first.\n")
            return 1
        profile = config["targets"][args.target]
        if args.dry_run:
            stdout.write(activation.shell_command_for_profile(profile, redact=True) + "\n")
            return 0
        from agentflow_proxy import server

        server.main(proxy_args)
        return 0

    if args.command == "doctor":
        if args.target not in RUN_TARGETS:
            stderr.write(
                f"AgentFlow doctor target is not implemented yet: {args.target}. "
                "Supported runtime targets are: openai, claude.\n"
            )
            return 2
        config = activation.load_activation_config(args.config_dir)
        profile = (config.get("targets") or {}).get(args.target)
        if not isinstance(profile, dict) or not profile.get("configured"):
            stderr.write(f"AgentFlow target is not configured: {args.target}. Run `agentflow activate {args.target}` first.\n")
            return 1
        result = _doctor_activation_target(profile, timeout=float(args.timeout))
        if args.json:
            _write_json(stdout, result)
        else:
            _write_doctor_summary(stdout, result)
        return 0 if result["ok"] else 1

    if args.command == "stats":
        result = _fetch_agentflow_stats(url=str(args.url), timeout=float(args.timeout), target=args.target)
        if args.json:
            _write_json(stdout, result)
        elif result["ok"]:
            _write_stats_summary(stdout, result)
        else:
            _write_json(stderr, result)
        return 0 if result["ok"] else 1

    if args.command == "version":
        result = {
            "schema": "agentflow.version.v1",
            "ok": True,
            "version": __version__,
            "package": "agentflow-proxy",
            "command": "agentflow",
        }
        if args.json:
            _write_json(stdout, result)
        else:
            stdout.write(f"agentflow {__version__}\n")
        return 0

    parser.error("unknown command")
    return 2


def _fetch_agentflow_stats(*, url: str = DEFAULT_STATS_URL, timeout: float = 5.0, target: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "agentflow.stats_cli.v1",
        "ok": False,
        "target": target,
        "url": url,
        "issues": [],
    }
    if not _is_loopback_url(url):
        result["issues"].append({
            "code": "non-loopback-url",
            "message": "agentflow stats only reads loopback URLs by default.",
        })
        return result
    try:
        response = httpx.get(url, timeout=timeout)
    except Exception as exc:
        result["issues"].append({
            "code": "stats-unreachable",
            "message": f"Could not reach AgentFlow stats URL: {type(exc).__name__}",
        })
        return result
    result["status_code"] = response.status_code
    if response.status_code >= 400:
        result["issues"].append({
            "code": "stats-error",
            "message": f"AgentFlow stats returned HTTP {response.status_code}.",
        })
        return result
    try:
        payload = response.json()
    except ValueError:
        result["issues"].append({"code": "stats-invalid-json", "message": "AgentFlow stats did not return JSON."})
        return result
    if not isinstance(payload, dict):
        result["issues"].append({"code": "stats-invalid-payload", "message": "AgentFlow stats payload is not an object."})
        return result
    result["ok"] = True
    result["stats"] = payload
    return result


def _write_stats_summary(stdout: Any, result: dict[str, Any]) -> None:
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    stdout.write("AgentFlow stats status=ok\n")
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


def _doctor_activation_target(profile: dict[str, Any], *, timeout: float = 5.0) -> dict[str, Any]:
    expected_provider = "openai" if profile.get("provider") == "openai" else "anthropic"
    configured_upstream = _redact_url(str(profile.get("upstream_base_url") or ""))
    health_url = str(profile.get("health_url") or "")
    result: dict[str, Any] = {
        "schema": "agentflow.activation_doctor.v1",
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
            "message": f"Could not reach AgentFlow health URL: {type(exc).__name__}",
        })
        return result
    result["health_status_code"] = response.status_code
    if response.status_code >= 400:
        result["issues"].append({
            "code": "health-error",
            "message": f"AgentFlow health returned HTTP {response.status_code}.",
        })
        return result
    try:
        health = response.json()
    except ValueError:
        result["issues"].append({"code": "health-invalid-json", "message": "AgentFlow health did not return JSON."})
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
        result["issues"].append({"code": "health-not-ok", "message": "AgentFlow health reports ok=false."})
    result["ok"] = not result["issues"]
    return result


def _write_doctor_summary(stdout: Any, result: dict[str, Any]) -> None:
    status = "ok" if result["ok"] else "issue"
    stdout.write(f"AgentFlow doctor target={result['target']} status={status}\n")
    stdout.write(f"Local base URL for clients: {result.get('local_base_url')}\n")
    stdout.write(f"Health URL: {result.get('health_url')}\n")
    stdout.write(f"Configured upstream: {result.get('configured_upstream')}\n")
    health = result.get("health") if isinstance(result.get("health"), dict) else {}
    if health:
        stdout.write(f"Running provider: {health.get('provider')}\n")
        stdout.write(f"Running upstream: {health.get('upstream')}\n")
    for issue in result.get("issues") or []:
        stdout.write(f"- {issue.get('code')}: {issue.get('message')}\n")


def _default_policy_reload_url() -> str:
    port = os.getenv("AGENTFLOW_ADMIN_PORT") or os.getenv("AGENTFLOW_PORT", "4000")
    return f"http://127.0.0.1:{port}{POLICY_RELOAD_PATH}"


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def policy_reload_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Reload local AgentFlow policy files through the loopback admin API")
    parser.add_argument(
        "--url",
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_ADMIN_TIMEOUT", "10")),
        help="HTTP timeout in seconds, default: 10",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.allow_non_loopback and not _is_loopback_url(args.url):
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "reload",
            ok=False,
            details={"source": "cli", "url": args.url, "error_type": "unsafe_url", "exit_code": 2},
        )
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {
                    "type": "unsafe_url",
                    "message": "policy reload CLI only posts to loopback URLs unless --allow-non-loopback is set",
                },
                "url": args.url,
            },
        )
        return 2

    try:
        response = httpx.post(args.url, timeout=args.timeout)
    except httpx.HTTPError as exc:
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "reload",
            ok=False,
            details={"source": "cli", "url": args.url, "error_type": exc.__class__.__name__, "exit_code": 1},
        )
        _write_json(
            stderr,
            {
                "ok": False,
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "url": args.url,
            },
        )
        return 1

    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.is_success,
            "status_code": response.status_code,
            "body": response.text,
            "url": args.url,
        }

    if response.is_success:
        from agentflow_proxy.policy_events import log_policy_event

        details = {
            "source": "cli",
            "url": args.url,
            "status_code": response.status_code,
            "exit_code": 0,
        }
        if isinstance(payload, dict):
            details["reloaded_modules"] = payload.get("reloaded_modules", [])
            details["policies"] = payload.get("policies")
        log_policy_event("reload", ok=True, details=details)
        _write_json(stdout, payload if isinstance(payload, dict) else {"ok": True, "response": payload})
        return 0

    error_payload = payload if isinstance(payload, dict) else {"ok": False, "response": payload}
    error_payload.setdefault("ok", False)
    error_payload.setdefault("status_code", response.status_code)
    error_payload.setdefault("url", args.url)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "reload",
        ok=False,
        details={"source": "cli", "url": args.url, "status_code": response.status_code, "exit_code": 1},
    )
    _write_json(stderr, error_payload)
    return 1


def policy_export_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export the effective local AgentFlow policy bundle as JSON")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.policy_bundle import build_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event

    bundle = asyncio.run(build_policy_bundle())
    log_policy_event("export", ok=True, details={"source": "cli", "exit_code": 0, "policies": bundle.get("policies")})
    if args.pretty:
        stdout.write(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, bundle)
    return 0


def _validation_result_error(message: str, *, path: str = "$") -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_validation.v1",
        "ok": False,
        "bundle_schema": None,
        "errors": [{"path": path, "message": message}],
        "warnings": [],
    }


def policy_validate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Validate an AgentFlow policy bundle JSON file offline")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print validation JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.path == "-":
        raw = stdin.read()
    else:
        try:
            raw = Path(args.path).read_text(encoding="utf-8")
        except OSError as exc:
            result = _validation_result_error(str(exc), path=args.path)
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "validate",
                ok=False,
                details={"source": "cli", "path": args.path, "error_count": 1, "warning_count": 0, "exit_code": 1},
            )
            _write_validation_result(stdout, result, pretty=args.pretty)
            return 1

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        result = _validation_result_error(f"invalid JSON: {exc}", path="$")
        from agentflow_proxy.policy_events import log_policy_event

        log_policy_event(
            "validate",
            ok=False,
            details={"source": "cli", "path": args.path, "error_count": 1, "warning_count": 0, "exit_code": 1},
        )
        _write_validation_result(stdout, result, pretty=args.pretty)
        return 1

    from agentflow_proxy.policy_bundle import validate_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event

    result = validate_policy_bundle(payload)
    log_policy_event(
        "validate",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "bundle_schema": result.get("bundle_schema"),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "error_count": len(result.get("errors", [])),
            "warning_count": len(result.get("warnings", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_validation_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _read_policy_json_arg(path: str, *, stdin: Any, stdin_used: bool) -> tuple[Any, dict[str, Any] | None, bool]:
    if path == "-":
        if stdin_used:
            return None, _validation_result_error("stdin can only be used for one policy bundle input"), stdin_used
        raw = stdin.read()
        stdin_used = True
    else:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return None, _validation_result_error(str(exc), path=path), stdin_used

    try:
        return json.loads(raw), None, stdin_used
    except ValueError as exc:
        return None, _validation_result_error(f"invalid JSON: {exc}", path="$"), stdin_used


def _policy_diff_error_result(
    before_validation: dict[str, Any],
    after_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_diff.v1",
        "ok": False,
        "changed": False,
        "changed_sections": [],
        "change_count": 0,
        "changes": [],
        "before_validation": before_validation,
        "after_validation": after_validation,
    }


def policy_diff_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Compare two AgentFlow policy bundle JSON files offline")
    parser.add_argument("before", help="Earlier policy bundle JSON path, or '-' for stdin.")
    parser.add_argument("after", help="Later policy bundle JSON path, or '-' for stdin.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print diff JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    before, before_error, stdin_used = _read_policy_json_arg(args.before, stdin=stdin, stdin_used=False)
    after, after_error, _stdin_used = _read_policy_json_arg(args.after, stdin=stdin, stdin_used=stdin_used)

    if before_error or after_error:
        result = _policy_diff_error_result(
            before_error or _validation_result_error("not validated because the other input could not be read"),
            after_error or _validation_result_error("not validated because the other input could not be read"),
        )
    else:
        from agentflow_proxy.policy_bundle import compare_policy_bundles

        result = compare_policy_bundles(before, after)

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "diff",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "before": args.before,
            "after": args.after,
            "changed": result.get("changed"),
            "changed_sections": result.get("changed_sections", []),
            "change_count": result.get("change_count", 0),
            "before_error_count": len(result.get("before_validation", {}).get("errors", [])),
            "after_error_count": len(result.get("after_validation", {}).get("errors", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_diff_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _policy_review_read_error_result(proposed_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_review.v1",
        "ok": False,
        "changed": False,
        "changed_sections": [],
        "change_count": 0,
        "safety_warning_count": 0,
        "safety_warnings": [],
        "current_validation": None,
        "proposed_validation": proposed_validation,
        "diff": {
            "schema": "agentflow.policy_bundle_diff.v1",
            "ok": False,
            "changed": False,
            "changed_sections": [],
            "change_count": 0,
            "changes": [],
        },
    }


def policy_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review a proposed AgentFlow policy bundle against current local policy")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Proposed policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print review JSON instead of emitting one compact line.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="SQLite metadata database path for local impact simulation, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Maximum recent calls to scan for metadata-only impact simulation, default: 1000.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _policy_review_read_error_result(read_error)
    else:
        from agentflow_proxy.policy_bundle import build_policy_bundle, review_policy_bundle

        current = asyncio.run(build_policy_bundle())
        result = review_policy_bundle(current, proposed, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "review",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "changed": result.get("changed"),
            "changed_sections": result.get("changed_sections", []),
            "change_count": result.get("change_count", 0),
            "safety_warning_count": result.get("safety_warning_count", 0),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "impact_status": (result.get("impact_summary") or {}).get("status"),
            "proposed_error_count": len((result.get("proposed_validation") or {}).get("errors", [])),
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="review", db_path=str(args.db))
    _write_policy_review_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def _policy_fetch_review_error_result(
    *,
    error_type: str,
    message: str,
    url: str | None,
    auth_configured: bool,
    reason: str,
    status_code: int | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    fetch: dict[str, Any] = {
        "status": "skipped" if status_code is None else "error",
        "reason": reason,
        "url": _redact_url(url),
        "auth_configured": bool(auth_configured),
        "status_code": status_code,
    }
    if body is not None:
        fetch["body"] = body[:500]
    return {
        "schema": "agentflow.policy_bundle_fetch_review.v1",
        "ok": False,
        "applied": False,
        "wrote_local_files": False,
        "fetch": fetch,
        "validation": None,
        "review": None,
        "recommendation": {},
        "bundle": None,
        "next_manual_command": None,
        "error": {"type": error_type, "message": message},
    }


def _managed_policy_auth(args: argparse.Namespace) -> tuple[dict[str, str], bool, str]:
    api_key = args.api_key
    source = "argument" if api_key else ""
    if not api_key and args.api_key_env:
        api_key = os.getenv(args.api_key_env)
        if api_key:
            source = f"env:{args.api_key_env}"

    headers: dict[str, str] = {}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    if args.tenant:
        headers["x-agentflow-tenant"] = args.tenant
    if args.account:
        headers["x-agentflow-account"] = args.account
    if api_key:
        return headers, True, source
    if args.allow_unauthenticated:
        return headers, False, "unauthenticated-explicit"
    return headers, False, ""


def _managed_policy_query(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "min_samples": args.min_samples,
        "max_error_rate": args.max_error_rate,
        "limit": args.limit,
    }
    for key in ("source_surface", "app_family", "category"):
        value = getattr(args, key)
        if value:
            params[key] = value
    if _openai_optimization_review_requested(args):
        for key in (
            "provider_endpoint",
            "requested_model_family",
            "max_retry_rate",
            "max_latency_regression_ms",
            "max_invalidation_rate",
        ):
            value = getattr(args, key, None)
            if value is not None and value != "":
                params[key] = value
        params["supported_local_action_families"] = _supported_openai_optimization_action_families(args)
    return params


def _openai_optimization_review_requested(args: argparse.Namespace) -> bool:
    url = str(getattr(args, "url", "") or "")
    if "openai-optimization-review-bundle" in url:
        return True
    for key in (
        "provider_endpoint",
        "requested_model_family",
        "max_retry_rate",
        "max_latency_regression_ms",
        "max_invalidation_rate",
    ):
        value = getattr(args, key, None)
        if value is not None and value != "":
            return True
    return bool(getattr(args, "supported_local_action_families", None))


def _supported_openai_optimization_action_families(args: argparse.Namespace) -> list[str]:
    configured = getattr(args, "supported_local_action_families", None) or []
    if configured:
        return sorted({str(value).strip() for value in configured if str(value).strip()})
    return ["cache", "old_context_summarization", "routing"]


def _managed_policy_capability_headers(args: argparse.Namespace) -> dict[str, str]:
    if not _openai_optimization_review_requested(args):
        return {}
    from agentflow_proxy import __version__

    families = _supported_openai_optimization_action_families(args)
    return {
        "x-agentflow-local-version": __version__,
        "x-agentflow-supported-local-action-families": ",".join(families),
    }


def _count_openai_review_actions_by_family(openai_review: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for decision, key in (
        ("selected", "selected_actions"),
        ("suppressed", "suppressed_actions"),
        ("omitted", "omitted_actions"),
    ):
        actions = openai_review.get(key) if isinstance(openai_review.get(key), list) else []
        for action in actions:
            if not isinstance(action, dict):
                continue
            family = str(action.get("action_family") or "unknown")
            row = counts.setdefault(family, {"selected": 0, "suppressed": 0, "omitted": 0})
            row[decision] += 1
    return dict(sorted(counts.items()))


def _openai_review_action_summary(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
    surface = action.get("local_policy_surface") if isinstance(action.get("local_policy_surface"), dict) else {}
    return {
        "action_id": action.get("action_id"),
        "target_candidate_id": action.get("target_candidate_id"),
        "action_family": action.get("action_family"),
        "candidate_family": action.get("candidate_family"),
        "policy_section": action.get("policy_section"),
        "decision": action.get("decision"),
        "reason_codes": action.get("reason_codes", []),
        "compatible": compatibility.get("compatible"),
        "compatibility_reason_codes": compatibility.get("reason_codes", []),
        "policy_file": surface.get("policy_file"),
        "expected_impact": action.get("expected_impact") if isinstance(action.get("expected_impact"), dict) else {},
    }


def _openai_optimization_review_summary(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {"schema": "agentflow.openai_optimization_review_summary.v1", "status": "missing"}
    openai_review = bundle.get("openai_optimization") if isinstance(bundle.get("openai_optimization"), dict) else {}
    recommendation = bundle.get("recommendation") if isinstance(bundle.get("recommendation"), dict) else {}
    if not openai_review:
        return {"schema": "agentflow.openai_optimization_review_summary.v1", "status": "missing"}
    selected = openai_review.get("selected_actions") if isinstance(openai_review.get("selected_actions"), list) else []
    suppressed = openai_review.get("suppressed_actions") if isinstance(openai_review.get("suppressed_actions"), list) else []
    omitted = openai_review.get("omitted_actions") if isinstance(openai_review.get("omitted_actions"), list) else []
    all_actions = [*selected, *suppressed, *omitted]
    local_gaps: list[dict[str, Any]] = []
    for action in all_actions:
        if not isinstance(action, dict):
            continue
        compatibility = action.get("local_executor_compatibility") if isinstance(action.get("local_executor_compatibility"), dict) else {}
        reason_codes = [
            str(reason)
            for reason in [
                *(action.get("reason_codes") if isinstance(action.get("reason_codes"), list) else []),
                *(compatibility.get("reason_codes") if isinstance(compatibility.get("reason_codes"), list) else []),
            ]
            if reason
        ]
        if compatibility.get("compatible") is False or any("unsupported" in reason for reason in reason_codes):
            local_gaps.append({
                "target_candidate_id": action.get("target_candidate_id"),
                "action_family": action.get("action_family"),
                "decision": action.get("decision"),
                "reason_codes": sorted(set(reason_codes)),
            })
    return {
        "schema": "agentflow.openai_optimization_review_summary.v1",
        "status": "present",
        "review_bundle_schema": openai_review.get("schema") or recommendation.get("openai_optimization_schema"),
        "selected_action_count": len(selected),
        "suppressed_action_count": len(suppressed),
        "omitted_action_count": len(omitted),
        "counts_by_family": _count_openai_review_actions_by_family(openai_review),
        "conflict_summary": recommendation.get("conflict_summary") if isinstance(recommendation.get("conflict_summary"), dict) else {},
        "local_capability_gaps": local_gaps,
        "selected_actions": [_openai_review_action_summary(action) for action in selected],
        "suppressed_actions": [_openai_review_action_summary(action) for action in suppressed],
        "omitted_actions": [_openai_review_action_summary(action) for action in omitted],
    }


def _managed_recommendation_summary(bundle: Any) -> dict[str, Any]:
    from agentflow_proxy.recommendation_health import summarize_recommendation_health

    if not isinstance(bundle, dict):
        return {}
    recommendation = bundle.get("recommendation")
    if not isinstance(recommendation, dict):
        recommendation = {}
    policies = bundle.get("policies")
    routing = policies.get("routing") if isinstance(policies, dict) and isinstance(policies.get("routing"), dict) else {}
    codex_app = policies.get("codex_app") if isinstance(policies, dict) and isinstance(policies.get("codex_app"), dict) else {}
    crunch = policies.get("crunch") if isinstance(policies, dict) and isinstance(policies.get("crunch"), dict) else {}
    cache = policies.get("cache") if isinstance(policies, dict) and isinstance(policies.get("cache"), dict) else {}
    routing_recommendation = (
        routing.get("recommendation")
        if isinstance(routing, dict) and isinstance(routing.get("recommendation"), dict)
        else {}
    )
    rules = routing.get("rules") if isinstance(routing, dict) and isinstance(routing.get("rules"), list) else []
    candidates: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        managed = rule.get("managed_recommendation")
        if not isinstance(managed, dict):
            continue
        candidates.append({
            key: managed.get(key)
            for key in (
                "candidate_id",
                "confidence",
                "sample_count",
                "success_count",
                "error_count",
                "error_rate",
                "estimated_savings_usd",
                "baseline_sample_count",
                "requested_model",
                "recommended_target_model",
                "source_surface",
                "app_family",
                "category",
                "text_bucket",
                "token_bucket",
            )
            if key in managed
        })

    codex_app_summary: dict[str, Any] = {}
    if codex_app:
        from agentflow_proxy.policy_bundle import codex_app_policy_review_summary

        codex_app_summary = codex_app_policy_review_summary(codex_app)
    pattern_summaries: dict[str, Any] = {}
    old_context_summary: dict[str, Any] = {}
    if crunch or cache:
        from agentflow_proxy.policy_bundle import old_context_summary_policy_review_summary, pattern_policy_review_summary

        if crunch:
            pattern_summaries["crunch"] = pattern_policy_review_summary(crunch, section="crunch")
            old_context_summary = old_context_summary_policy_review_summary(crunch)
        if cache:
            pattern_summaries["cache"] = pattern_policy_review_summary(cache, section="cache")
    pattern_candidate_ids = [
        candidate_id
        for summary in pattern_summaries.values()
        for candidate_id in summary.get("candidate_ids", [])
    ]

    return {
        "schema": recommendation.get("schema"),
        "policy_source": recommendation.get("policy_source"),
        "candidate_ids": recommendation.get("candidate_ids", []),
        "candidate_count": recommendation.get("candidate_count", len(candidates)),
        "routing_rule_count": recommendation.get("routing_rule_count", len(candidates)),
        "codex_app_candidate_ids": codex_app_summary.get("candidate_ids", []),
        "codex_app_candidate_count": codex_app_summary.get("candidate_count", 0),
        "codex_app_review_only": codex_app_summary.get("review_only", False),
        "codex_app_application_status": (codex_app_summary.get("application") or {}).get("status"),
        "pattern_candidate_ids": pattern_candidate_ids,
        "pattern_candidate_count": sum(summary.get("candidate_count", 0) for summary in pattern_summaries.values()),
        "crunch_pattern_candidate_count": pattern_summaries.get("crunch", {}).get("candidate_count", 0),
        "cache_pattern_candidate_count": pattern_summaries.get("cache", {}).get("candidate_count", 0),
        "pattern_review_only_candidate_count": sum(summary.get("review_only_candidate_count", 0) for summary in pattern_summaries.values()),
        "pattern_omitted_candidate_count": sum(summary.get("omitted_candidate_count", 0) for summary in pattern_summaries.values()),
        "old_context_summary_candidate_ids": old_context_summary.get("candidate_ids", []),
        "old_context_summary_candidate_count": old_context_summary.get("candidate_count", 0),
        "old_context_summary_application_status": (old_context_summary.get("application") or {}).get("status"),
        "old_context_summary_warning_codes": old_context_summary.get("warning_codes", []),
        "omitted_candidate_count": recommendation.get(
            "omitted_candidate_count",
            routing_recommendation.get("omitted_candidate_count", 0),
        ),
        "filters": recommendation.get("filters", {}),
        "candidates": candidates,
        "codex_app": codex_app_summary,
        "patterns": pattern_summaries,
        "old_context_summarization": old_context_summary,
        "health": summarize_recommendation_health(bundle),
    }


def _write_policy_fetch_review_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def policy_fetch_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a managed AgentFlow policy bundle recommendation and review it without applying it"
    )
    parser.add_argument(
        "--url",
        default=os.getenv(POLICY_BUNDLE_RECOMMENDATION_URL_ENV),
        help=f"Full managed policy bundle recommendation URL. May also be set with {POLICY_BUNDLE_RECOMMENDATION_URL_ENV}.",
    )
    parser.add_argument(
        "--api-key",
        help=f"Managed optimizer API key. Prefer --api-key-env or {MANAGED_POLICY_API_KEY_ENV} for shell history safety.",
    )
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Fetch without an API key. Intended only for local/dev managed servers.",
    )
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum candidates to request.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument("--provider-endpoint", help="Optional OpenAI optimization provider endpoint filter, such as responses or chat_completions.")
    parser.add_argument("--requested-model-family", help="Optional OpenAI optimization requested model family filter.")
    parser.add_argument("--max-retry-rate", type=float, default=None, help="Maximum OpenAI optimization retry rate to request.")
    parser.add_argument(
        "--max-latency-regression-ms",
        type=float,
        default=None,
        help="Maximum OpenAI optimization latency regression in milliseconds to request.",
    )
    parser.add_argument("--max-invalidation-rate", type=float, default=None, help="Maximum OpenAI cache replay invalidation rate to request.")
    parser.add_argument(
        "--supported-local-action-families",
        action="append",
        choices=("routing", "crunch", "cache", "old_context_summarization"),
        help="Local OpenAI optimization action family supported by this executor. Repeat to send multiple values.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="SQLite metadata database path for local impact simulation, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Maximum recent calls to scan for metadata-only impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print fetch/review JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    headers, auth_configured, auth_source = _managed_policy_auth(args)
    headers.update(_managed_policy_capability_headers(args))
    safe_url = _redact_url(args.url)
    from agentflow_proxy.policy_events import log_policy_event

    if not args.url:
        result = _policy_fetch_review_error_result(
            error_type="missing_url",
            message=f"set --url or {POLICY_BUNDLE_RECOMMENDATION_URL_ENV} to enable managed fetch/review",
            url=None,
            auth_configured=auth_configured,
            reason="missing-url",
        )
        log_policy_event(
            "fetch-review",
            ok=False,
            details={"source": "cli", "url": None, "auth_configured": auth_configured, "exit_code": 2},
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 2
    if not auth_configured and not args.allow_unauthenticated:
        result = _policy_fetch_review_error_result(
            error_type="missing_auth",
            message=f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated",
            url=args.url,
            auth_configured=False,
            reason="missing-auth",
        )
        log_policy_event(
            "fetch-review",
            ok=False,
            details={"source": "cli", "url": safe_url, "auth_configured": False, "exit_code": 2},
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 2

    started = time.time()
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    try:
        response = httpx.get(args.url, headers=headers, params=_managed_policy_query(args), timeout=args.timeout)
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        result = _policy_fetch_review_error_result(
            error_type=exc.__class__.__name__,
            message=_redact_secret(str(exc), secret),
            url=args.url,
            auth_configured=auth_configured,
            reason="request-failed",
        )
        result["fetch"]["latency_ms"] = int((time.time() - started) * 1000)
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "error_type": exc.__class__.__name__,
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    if response.status_code >= 400:
        result = _policy_fetch_review_error_result(
            error_type="server_error",
            message="managed server returned an error response",
            url=args.url,
            auth_configured=auth_configured,
            reason="server-error",
            status_code=response.status_code,
            body=_redact_secret(response.text, secret),
        )
        result["fetch"]["latency_ms"] = latency_ms
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "status_code": response.status_code,
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    try:
        bundle = response.json()
    except ValueError as exc:
        validation = _validation_result_error(f"invalid JSON: {exc}", path="$")
        result = _policy_fetch_review_error_result(
            error_type="invalid_json",
            message=f"managed server response was not valid JSON: {exc}",
            url=args.url,
            auth_configured=auth_configured,
            reason="invalid-json",
            status_code=response.status_code,
        )
        result["fetch"]["latency_ms"] = latency_ms
        result["validation"] = validation
        result["review"] = _policy_review_read_error_result(validation)
        log_policy_event(
            "fetch-review",
            ok=False,
            details={
                "source": "cli",
                "url": safe_url,
                "auth_configured": auth_configured,
                "auth_source": auth_source,
                "status_code": response.status_code,
                "proposed_error_count": len(validation["errors"]),
                "exit_code": 1,
            },
        )
        _write_policy_fetch_review_result(stderr, result, pretty=args.pretty)
        return 1

    from agentflow_proxy.policy_bundle import build_policy_bundle, review_policy_bundle, validate_policy_bundle
    from agentflow_proxy.recommendation_health import strip_raw_payload_fields

    validation = validate_policy_bundle(bundle)
    current = asyncio.run(build_policy_bundle())
    review = review_policy_bundle(current, bundle, impact_db_path=args.db, impact_limit=max(0, args.impact_limit))
    recommendation = _managed_recommendation_summary(bundle)
    openai_review = _openai_optimization_review_summary(bundle)
    next_manual_commands = ["agentflow-policy-apply reviewed-bundle.json --dry-run --pretty"]
    if openai_review.get("status") == "present":
        next_manual_commands = [
            "agentflow-policy-draft-stage reviewed-bundle.json --pretty",
            "agentflow-policy-draft-validate <draft-id> --pretty",
        ]
    ok = bool(validation["ok"] and review["ok"])
    result = {
        "schema": "agentflow.policy_bundle_fetch_review.v1",
        "ok": ok,
        "applied": False,
        "wrote_local_files": False,
        "fetch": {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "query": _managed_policy_query(args),
        },
        "validation": validation,
        "review": review,
        "provenance": validation.get("provenance"),
        "recommendation": recommendation,
        "openai_optimization_review": openai_review,
        "bundle": strip_raw_payload_fields(bundle),
        "next_manual_command": next_manual_commands[0],
        "next_manual_commands": next_manual_commands,
        "error": None if ok else {"type": "validation_failed", "message": "managed policy bundle is invalid"},
    }
    result = _redact_secret(result, secret)
    log_policy_event(
        "fetch-review",
        ok=ok,
        details={
            "source": "cli",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "changed": review.get("changed"),
            "changed_sections": review.get("changed_sections", []),
            "change_count": review.get("change_count", 0),
            "safety_warning_count": review.get("safety_warning_count", 0),
            "provenance_status": (validation.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (validation.get("provenance") or {}).get("managed_bundle"),
            "recommendation_health": recommendation.get("health", {}),
            "impact_status": (review.get("impact_summary") or {}).get("status"),
            "proposed_error_count": len(validation.get("errors", [])),
            "candidate_ids": result.get("recommendation", {}).get("candidate_ids", []),
            "candidate_count": result.get("recommendation", {}).get("candidate_count", 0),
            "openai_optimization_review": {
                "status": openai_review.get("status"),
                "selected_action_count": openai_review.get("selected_action_count", 0),
                "suppressed_action_count": openai_review.get("suppressed_action_count", 0),
                "omitted_action_count": openai_review.get("omitted_action_count", 0),
                "local_capability_gap_count": len(openai_review.get("local_capability_gaps", [])),
            },
            "exit_code": 0 if ok else 1,
        },
    )
    _write_policy_fetch_review_result(stdout if ok else stderr, result, pretty=args.pretty)
    return 0 if ok else 1


def _policy_apply_read_error_result(read_error: dict[str, Any], *, config_dir: str, dry_run: bool) -> dict[str, Any]:
    return {
        "schema": "agentflow.policy_bundle_apply.v1",
        "ok": False,
        "dry_run": bool(dry_run),
        "config_dir": config_dir,
        "applied_sections": [],
        "skipped_sections": [],
        "files": [],
        "validation": read_error,
        "safety_warning_count": 0,
        "safety_warnings": [],
        "error": {"type": "read_failed", "message": "policy bundle could not be read"},
    }


def policy_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply an AgentFlow policy bundle to local YAML rule files offline")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the files that would change without writing them.",
    )
    parser.add_argument(
        "--allow-risky",
        action="store_true",
        help="Apply bundles with safety warnings. The warnings are still included in the JSON result.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print apply JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _policy_apply_read_error_result(read_error, config_dir=args.config_dir, dry_run=args.dry_run)
    else:
        from agentflow_proxy.policy_bundle import apply_policy_bundle

        result = apply_policy_bundle(
            bundle,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
            allow_risky=args.allow_risky,
            sections=args.section,
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "apply",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "allow_risky": args.allow_risky,
            "applied_sections": result.get("applied_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "safety_warning_count": result.get("safety_warning_count", 0),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "provenance_managed_bundle": (result.get("provenance") or {}).get("managed_bundle"),
            "old_context_summarization": result.get("old_context_summarization"),
            "codex_app": result.get("codex_app"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_apply_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def policy_draft_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Stage a local AgentFlow policy draft and return structured diffs without touching active rules"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON/YAML path, section YAML path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--section",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Treat input as one local policy section payload and patch it into the current policy bundle.",
    )
    parser.add_argument(
        "--draft-id",
        help="Optional local draft ID. Unsafe path characters are stripped.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print draft JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.path == "-":
        raw = stdin.read()
    else:
        try:
            raw = Path(args.path).read_text(encoding="utf-8")
        except OSError as exc:
            result = {
                "schema": "agentflow.policy_draft_stage.v1",
                "ok": False,
                "draft": None,
                "draft_id": args.draft_id,
                "workspace": args.workspace,
                "wrote_active_policy_files": False,
                "reloaded_modules": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "diff": None,
                "sections": [],
                "error": {
                    "type": "read_failed",
                    "message": str(exc),
                    "errors": [{"path": args.path, "message": str(exc)}],
                },
            }
            _write_policy_draft_stage_result(stdout, result, pretty=args.pretty)
            return 1

    from agentflow_proxy.openai_optimization_drafts import (
        is_openai_optimization_review_payload,
        stage_openai_optimization_review_draft,
    )
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_files import parse_policy_payload, stage_policy_draft

    payload, parse_error = parse_policy_payload(raw)
    if parse_error:
        result = {
            "schema": "agentflow.policy_draft_stage.v1",
            "ok": False,
            "draft": None,
            "draft_id": args.draft_id,
            "workspace": args.workspace,
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "diff": None,
            "sections": [],
            "error": {"type": "parse_failed", "message": "policy draft payload could not be parsed", "errors": parse_error["errors"]},
        }
    elif is_openai_optimization_review_payload(payload):
        result = asyncio.run(stage_openai_optimization_review_draft(
            payload,
            draft_id=args.draft_id,
            workspace=args.workspace,
        ))
    else:
        result = asyncio.run(stage_policy_draft(
            payload,
            section=args.section,
            draft_id=args.draft_id,
            workspace=args.workspace,
        ))

    log_policy_event(
        "draft-stage",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.path,
            "section": args.section,
            "draft_id": result.get("draft_id"),
            "workspace": result.get("workspace"),
            "changed": (result.get("diff") or {}).get("changed") if isinstance(result.get("diff"), dict) else None,
            "changed_sections": (result.get("diff") or {}).get("changed_sections", []) if isinstance(result.get("diff"), dict) else [],
            "change_count": (result.get("diff") or {}).get("change_count", 0) if isinstance(result.get("diff"), dict) else 0,
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_policy_draft_stage_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def policy_draft_validate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and dry-run a staged local AgentFlow policy draft before apply"
    )
    parser.add_argument(
        "draft",
        help="Staged draft ID, draft directory, draft.json path, or policy_bundle.json path.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files used by the dry-run apply projection, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only impact simulation.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Recent provider metadata rows to inspect for impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--codex-recent-limit",
        type=int,
        default=200,
        help="Recent Codex app metadata rows to inspect for Codex app dry-run projection, default: 200.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print validation JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.policy_workbench import validate_staged_policy_draft

    result = asyncio.run(validate_staged_policy_draft(
        args.draft,
        workspace=args.workspace,
        config_dir=args.config_dir,
        db_path=args.db,
        impact_limit=args.impact_limit,
        codex_recent_limit=args.codex_recent_limit,
    ))

    log_policy_event(
        "draft-validate",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "draft": args.draft,
            "workspace": args.workspace,
            "config_dir": args.config_dir,
            "db": args.db,
            "status": result.get("status"),
            "can_apply": result.get("can_apply"),
            "apply_blocked": result.get("apply_blocked"),
            "changed_sections": (result.get("draft") or {}).get("changed_sections", []) if isinstance(result.get("draft"), dict) else [],
            "section_verdicts": {
                section.get("section"): section.get("verdict")
                for section in result.get("sections", [])
                if isinstance(section, dict)
            },
            "blocker_reason_codes": (result.get("apply_prerequisites") or {}).get("blocker_reason_codes", [])
            if isinstance(result.get("apply_prerequisites"), dict)
            else [],
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_policy_draft_validate_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def openai_optimization_draft_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run a staged managed OpenAI optimization draft through the local governor"
    )
    parser.add_argument(
        "draft",
        help="Staged draft ID, draft directory, draft.json path, or policy_bundle.json path.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only OpenAI dry-run rows.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Recent local call metadata rows to inspect, default: 1000.")
    parser.add_argument(
        "--canary-fraction",
        type=float,
        default=1.0,
        help="Projected per-action and governor canary fraction for the dry-run, default: 1.0.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Projected per-action and governor holdout fraction for the dry-run, default: 0.0.",
    )
    parser.add_argument(
        "--queue-feedback",
        action="store_true",
        help="Queue sanitized dry-run lifecycle feedback in the local metadata queue.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_optimization_draft_dry_run import dry_run_openai_optimization_draft
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.store import Store

    store = None
    try:
        db_path = Path(args.db).expanduser()
        if db_path.exists():
            store = Store(str(db_path))
        result = asyncio.run(
            dry_run_openai_optimization_draft(
                args.draft,
                workspace=args.workspace,
                store_obj=store,
                limit=args.limit,
                canary_fraction=args.canary_fraction,
                holdout_fraction=args.holdout_fraction,
                queue_feedback=args.queue_feedback,
            )
        )
    finally:
        if store is not None:
            store.conn.close()

    log_policy_event(
        "openai-optimization-draft-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "draft": args.draft,
            "workspace": args.workspace,
            "db": args.db,
            "limit": args.limit,
            "queue_feedback": bool(args.queue_feedback),
            "openai_rows_considered": (result.get("summary") or {}).get("openai_rows_considered")
            if isinstance(result.get("summary"), dict)
            else None,
            "applied_if_enabled_total": (result.get("summary") or {}).get("applied_if_enabled_total")
            if isinstance(result.get("summary"), dict)
            else None,
            "suppressed_total": (result.get("summary") or {}).get("suppressed_total")
            if isinstance(result.get("summary"), dict)
            else None,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "active_policy_files_written": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    stdout.write(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    return 0 if result.get("ok") else 1


def openai_optimization_draft_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a staged OpenAI optimization draft as bounded local canaries, with backups and rollback metadata"
    )
    parser.add_argument("draft", help="Staged OpenAI optimization draft ID, draft directory, draft.json path, or policy_bundle.json path.")
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only dry-run projection and lifecycle feedback.",
    )
    parser.add_argument("--section", action="append", choices=["routing", "crunch", "cache"], help="Apply only one policy section. Repeat to apply multiple sections.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview canary policy edits without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write canary edits to local YAML policy files.")
    parser.add_argument("--canary-fraction", type=float, default=0.10, help="Deterministic applied canary fraction, default: 0.10.")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Deterministic holdout fraction, default: 0.10.")
    parser.add_argument("--impact-limit", type=int, default=1000, help="Recent OpenAI metadata rows to inspect for governor conflicts, default: 1000.")
    parser.add_argument(
        "--require-verified-provenance",
        action="store_true",
        help="Require staged draft provenance to verify with the configured managed policy secret before applying.",
    )
    parser.add_argument("--queue-feedback", action="store_true", help="Queue sanitized apply lifecycle feedback in the local metadata queue.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from agentflow_proxy.openai_optimization_draft_apply import apply_openai_optimization_draft
    from agentflow_proxy.store import Store

    store = None
    try:
        db_path = Path(args.db).expanduser()
        if db_path.exists():
            store = Store(str(db_path))
        result = asyncio.run(
            apply_openai_optimization_draft(
                args.draft,
                workspace=args.workspace,
                config_dir=args.config_dir,
                store_obj=store,
                dry_run=args.dry_run,
                canary_fraction=args.canary_fraction,
                holdout_fraction=args.holdout_fraction,
                impact_limit=args.impact_limit,
                sections=args.section,
                require_verified_provenance=args.require_verified_provenance,
                queue_feedback=args.queue_feedback,
            )
        )
    finally:
        if store is not None:
            store.conn.close()

    _write_policy_draft_apply_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


async def _reload_policy_state_via_url(url: str, *, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url)
    try:
        payload = response.json()
    except ValueError:
        payload = {
            "ok": response.is_success,
            "status_code": response.status_code,
            "body": response.text,
            "url": url,
        }
    if isinstance(payload, dict):
        payload.setdefault("status_code", response.status_code)
        payload.setdefault("url", url)
        if not response.is_success:
            payload.setdefault("ok", False)
        return payload
    return {
        "ok": response.is_success,
        "status_code": response.status_code,
        "response": payload,
        "url": url,
    }


def policy_draft_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a validated staged AgentFlow policy draft transactionally, with backup, reload, and verification"
    )
    parser.add_argument(
        "draft",
        help="Staged draft ID, draft directory, draft.json path, or policy_bundle.json path.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local SQLite DB path for metadata-only impact simulation.",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument(
        "--reload-url",
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_ADMIN_TIMEOUT", "10")),
        help="HTTP timeout in seconds for the loopback reload call, default: 10.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting reload to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=1000,
        help="Recent provider metadata rows to inspect for impact simulation, default: 1000.",
    )
    parser.add_argument(
        "--codex-recent-limit",
        type=int,
        default=200,
        help="Recent Codex app metadata rows to inspect for Codex app dry-run projection, default: 200.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print apply JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if not args.allow_non_loopback and not _is_loopback_url(args.reload_url):
        from agentflow_proxy.policy_events import log_policy_event
        from agentflow_proxy.policy_workbench import POLICY_DRAFT_APPLY_SCHEMA

        result = {
            "schema": POLICY_DRAFT_APPLY_SCHEMA,
            "ok": False,
            "status": "blocked",
            "draft_id": args.draft,
            "apply_id": None,
            "backup_id": None,
            "config_dir": args.config_dir,
            "requested_sections": args.section or ["routing", "crunch", "cache", "routing_experiments", "codex_app"],
            "applied_sections": [],
            "changed_sections": [],
            "files": [],
            "backups": [],
            "reloaded_modules": False,
            "reload": None,
            "verification": None,
            "validation": None,
            "restored": False,
            "restore": None,
            "rollback_command": None,
            "privacy": {"provider_calls_made": False, "managed_server_calls_made": False, "loopback_admin_calls_made": False},
            "error": {
                "type": "unsafe_url",
                "message": "policy draft apply only posts reloads to loopback URLs unless --allow-non-loopback is set",
                "url": args.reload_url,
            },
        }
        log_policy_event(
            "draft-apply",
            ok=False,
            details={"source": "cli", "draft_id": args.draft, "error_type": "unsafe_url", "exit_code": 2},
        )
        _write_policy_draft_apply_result(stderr, result, pretty=args.pretty)
        return 2

    from agentflow_proxy.policy_workbench import apply_validated_policy_draft

    async def reload_state() -> dict[str, Any]:
        return await _reload_policy_state_via_url(args.reload_url, timeout=args.timeout)

    result = asyncio.run(apply_validated_policy_draft(
        args.draft,
        workspace=args.workspace,
        config_dir=args.config_dir,
        db_path=args.db,
        impact_limit=args.impact_limit,
        codex_recent_limit=args.codex_recent_limit,
        sections=args.section,
        reload_policy_state=reload_state,
        event_source="cli",
        loopback_admin_calls_made=True,
    ))

    _write_policy_draft_apply_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


async def _queue_codex_app_dry_run_lifecycle_events(store: Any, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy.recommendations import queue_policy_event_feedback

    queued: dict[str, Any] = {
        "schema": "agentflow.codex_app_canary_lifecycle_queue_meta.v1",
        "source_surface": "codex_app_canary_lifecycle",
        "event_phase": "dry_run",
        "results": {},
        "payload_included": False,
    }
    candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        rule_id = str(candidate.get("rule_id") or candidate.get("candidate_id") or "unknown-codex-app-rule")
        candidate_id = str(candidate.get("candidate_id") or rule_id)
        event = {
            "schema": "agentflow.codex_app_canary_lifecycle_feedback.v1",
            "event_type": "codex_app_canary_lifecycle",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "source_surface": "codex_turn",
            "app_family": "codex",
            "lifecycle_kind": "codex_app_canary",
            "lifecycle_phase": "dry_run",
            "action_family": "codex_app",
            "policy_id": candidate.get("policy_id") or candidate_id,
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "canary_cohort": "projected",
            "decision": {
                "status": "dry_run",
                "reason": "codex-app-policy-dry-run",
                "policy_source": candidate.get("policy_source") or "managed-recommended",
                "applied": False,
                "rule_path_included": False,
                "cache_key_included": False,
            },
            "outcome": {
                "status": "dry_run",
                "status_class": "metadata",
                "projected_applied_count": candidate.get("projected_applied_count"),
                "projected_holdout_count": candidate.get("projected_holdout_count"),
                "projected_skip_count": candidate.get("projected_skip_count"),
                "projected_savings_usd": candidate.get("projected_savings_usd"),
            },
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_params_included": False,
                "raw_transcripts_included": False,
                "raw_commands_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "thread_ids_included": False,
                "local_session_ids_included": False,
                "cache_keys_included": False,
                "rule_path_included": False,
            },
        }
        meta = await queue_policy_event_feedback(
            store,
            event,
            source_surface="codex_app_canary_lifecycle",
            queue_when_disabled=True,
            flush_immediately=False,
        )
        queued["results"][candidate_id] = {
            "enabled": bool(meta.get("enabled")),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "endpoint": meta.get("endpoint"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "payload_included": False,
        }
    return queued


def codex_app_policy_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed Codex app policy rules against local metadata-only turn windows")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Policy bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(POLICY_BUNDLE_RECOMMENDATION_URL_ENV),
        help=f"Managed policy bundle recommendation URL. May also be set with {POLICY_BUNDLE_RECOMMENDATION_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum candidates to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="SQLite metadata database path for recent Codex turn windows, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument("--recent-limit", type=int, default=200, help="Maximum recent Codex turn/start rows to project, default: 200.")
    parser.add_argument("--fixture", help="Optional JSON fixture rows/features to include in the projection.")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Do not include the built-in synthetic summary-turn fixture row.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    managed_server_calls_made = False
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        managed_server_calls_made = fetch.get("status") not in {"skipped", None} if isinstance(fetch, dict) else False
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.codex_app_policy_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "applied": False,
                "wrote_local_policy_files": False,
                "cache_table_mutated": False,
                "provider_calls_made": False,
                "managed_server_calls_made": managed_server_calls_made,
                "fetch": fetch,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        fetch = {"status": "skipped", "reason": "local-input"}
        if read_error:
            result = {
                "schema": "agentflow.codex_app_policy_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "applied": False,
                "wrote_local_policy_files": False,
                "cache_table_mutated": False,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "fetch": fetch,
                "validation": read_error,
                "error": {"type": "read_failed", "message": "policy bundle could not be read"},
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return 1

    from agentflow_proxy.codex_app_dry_run import dry_run_codex_app_policy, load_codex_app_fixture_features
    from agentflow_proxy.policy_bundle import validate_policy_bundle
    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.store import Store

    validation = validate_policy_bundle(bundle)
    if not validation["ok"]:
        result = {
            "schema": "agentflow.codex_app_policy_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "applied": False,
            "wrote_local_policy_files": False,
            "cache_table_mutated": False,
            "provider_calls_made": False,
            "managed_server_calls_made": managed_server_calls_made,
            "fetch": fetch,
            "validation": validation,
            "error": {"type": "validation_failed", "message": "policy bundle is invalid"},
        }
        log_policy_event(
            "codex-app-policy-dry-run",
            ok=False,
            details={"source": "cli", "validation_error_count": len(validation.get("errors", [])), "exit_code": 1},
        )
        _write_rollout_actions_result(stderr, result, pretty=args.pretty)
        return 1

    fixture_features = load_codex_app_fixture_features(args.fixture) if args.fixture else []
    store = Store(args.db)
    try:
        result = dry_run_codex_app_policy(
            bundle,
            store=store,
            recent_limit=max(0, args.recent_limit),
            fixture_features=fixture_features,
            include_synthetic=not args.no_synthetic,
        )
        result["managed_lifecycle_feedback"] = asyncio.run(
            _queue_codex_app_dry_run_lifecycle_events(store, result)
        )
    finally:
        store.conn.close()
    result["validation"] = validation
    result["fetch"] = fetch
    result["managed_server_calls_made"] = managed_server_calls_made
    result["privacy"]["managed_server_calls_made"] = managed_server_calls_made
    log_policy_event(
        "codex-app-policy-dry-run",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "candidate_count": result.get("summary", {}).get("candidate_count", 0),
            "projected_applied_count": result.get("summary", {}).get("projected_applied_count", 0),
            "projected_holdout_count": result.get("summary", {}).get("projected_holdout_count", 0),
            "projected_skip_count": result.get("summary", {}).get("projected_skip_count", 0),
            "projected_savings_usd": result.get("summary", {}).get("projected_savings_usd", 0.0),
            "provider_calls_made": False,
            "managed_server_calls_made": managed_server_calls_made,
            "exit_code": 0,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0


def _read_rollout_actions_from_url(args: argparse.Namespace) -> tuple[Any, dict[str, Any] | None, int | None]:
    headers, auth_configured, auth_source = _managed_policy_auth(args)
    safe_url = _redact_url(args.url)
    if not auth_configured and not args.allow_unauthenticated:
        return None, {
            "status": "skipped",
            "reason": "missing-auth",
            "url": safe_url,
            "auth_configured": False,
            "auth_source": "",
            "error": {"type": "missing_auth", "message": f"set --api-key, --api-key-env, {MANAGED_POLICY_API_KEY_ENV}, or --allow-unauthenticated"},
        }, 2

    started = time.time()
    secret = args.api_key or (os.getenv(args.api_key_env) if args.api_key_env else None)
    try:
        response = httpx.get(args.url, headers=headers, params=_managed_policy_query(args), timeout=args.timeout)
        latency_ms = int((time.time() - started) * 1000)
    except httpx.HTTPError as exc:
        return None, {
            "status": "error",
            "reason": "request-failed",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "latency_ms": int((time.time() - started) * 1000),
            "error": {"type": exc.__class__.__name__, "message": _redact_secret(str(exc), secret)},
        }, 1
    if response.status_code >= 400:
        return None, {
            "status": "error",
            "reason": "server-error",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "body": _redact_secret(response.text[:500], secret),
            "error": {"type": "server_error", "message": "managed server returned an error response"},
        }, 1
    try:
        return response.json(), {
            "status": "received",
            "reason": "ok",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "query": _managed_policy_query(args),
        }, None
    except ValueError as exc:
        return None, {
            "status": "error",
            "reason": "invalid-json",
            "url": safe_url,
            "auth_configured": auth_configured,
            "auth_source": auth_source,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "error": {"type": "invalid_json", "message": f"managed server response was not valid JSON: {exc}"},
        }, 1


def managed_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed pattern rollout actions against local crunch/cache rules")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(PATTERN_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed pattern rollout actions URL. May also be set with {PATTERN_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Review only one policy section. Repeat to review multiple sections.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            from agentflow_proxy.policy_events import log_policy_event

            result = {
                "schema": "agentflow.pattern_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "rollout-actions-review",
                ok=False,
                details={"source": "cli", "url": _redact_url(args.url), "fetch_status": fetch.get("status"), "exit_code": fetch_exit},
            )
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.policy_events import log_policy_event

    if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
        from agentflow_proxy.openai_cache_replay_rollout_actions import review_openai_cache_replay_rollout_actions

        review = review_openai_cache_replay_rollout_actions(bundle, config_dir=args.config_dir)
    else:
        from agentflow_proxy.rollout_actions import plan_rollout_actions

        review = plan_rollout_actions(bundle, config_dir=args.config_dir, sections=args.section)
    if fetch:
        review["fetch"] = fetch
    log_policy_event(
        "rollout-actions-review",
        ok=bool(review["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "config_dir": args.config_dir,
            "action_count": review.get("action_count", 0),
            "planned_action_count": review.get("planned_action_count", 0),
            "changed_action_count": review.get("changed_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "exit_code": 0 if review["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(review, command="review", db_path=str(args.db))
    _attach_terminal_output_compaction_lifecycle_feedback(
        review,
        command="review",
        db_path=str(args.db),
        result_key="managed_terminal_output_compaction_lifecycle_feedback",
    )
    _write_rollout_actions_result(stdout if review["ok"] else stderr, review, pretty=args.pretty)
    return 0 if review["ok"] else 1


def optimization_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed optimization rollout actions before local apply")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed optimization rollout actions URL. May also be set with {OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            from agentflow_proxy.policy_events import log_policy_event

            result = {
                "schema": "agentflow.optimization_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "wrote_local_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "optimization-rollout-actions-review",
                ok=False,
                details={"source": "cli", "url": _redact_url(args.url), "fetch_status": fetch.get("status"), "exit_code": fetch_exit},
            )
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.optimization_rollout_review import review_optimization_rollout_actions
    from agentflow_proxy.policy_events import log_policy_event

    review = review_optimization_rollout_actions(bundle)
    if fetch:
        review["fetch"] = fetch
    log_policy_event(
        "optimization-rollout-actions-review",
        ok=bool(review["ok"]),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "action_count": review.get("summary", {}).get("action_count", 0),
            "accepted_action_count": review.get("summary", {}).get("accepted_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "wrote_local_policy_files": False,
            "exit_code": 0 if review["ok"] else 1,
        },
    )
    _write_rollout_actions_result(stdout if review["ok"] else stderr, review, pretty=args.pretty)
    return 0 if review["ok"] else 1


def optimization_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed optimization rollout actions as local canary policy files")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Managed optimization rollout actions URL. May also be set with {OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache"],
        help="Policy section to apply. May be repeated. Default: routing, crunch, and cache.",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write reviewed canary edits to local YAML policy files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    fetch_url = args.url or os.getenv(OPTIMIZATION_ROLLOUT_ACTIONS_URL_ENV)
    if fetch_url:
        args.url = fetch_url
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.optimization_rollout_actions_apply.v1",
                "ok": False,
                "dry_run": bool(args.dry_run),
                "fetch": fetch,
                "actions": [],
                "files": [],
                "wrote_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.optimization_promotion_canary import apply_optimization_promotion_canaries

    result = apply_optimization_promotion_canaries(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        sections=args.section,
    )
    result["schema"] = "agentflow.optimization_rollout_actions_apply.v1"
    result["source_command"] = "agentflow-optimization-rollout-actions-apply"
    if fetch:
        result["fetch"] = fetch
    _attach_optimization_promotion_lifecycle_feedback(
        result,
        command="dry-run" if args.dry_run else "apply",
        db_path=str(args.db),
    )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "optimization-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": None if fetch_url else args.path,
            "url": _redact_url(fetch_url),
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count"),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def scaffold_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed repeated-scaffold rollout actions before local canary apply")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV),
        help=f"Managed scaffold rollout actions URL. May also be set with {SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print review JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    if args.url:
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            from agentflow_proxy.policy_events import log_policy_event

            result = {
                "schema": "agentflow.scaffold_rollout_actions_fetch_review.v1",
                "ok": False,
                "fetch": fetch,
                "review": None,
                "wrote_local_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            log_policy_event(
                "scaffold-rollout-actions-review",
                ok=False,
                details={"source": "cli", "url": _redact_url(args.url), "fetch_status": fetch.get("status"), "exit_code": fetch_exit},
            )
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.scaffold_rollout_review import review_scaffold_rollout_actions

    review = review_scaffold_rollout_actions(bundle)
    if fetch:
        review["fetch"] = fetch
        review["managed_server_calls_made"] = fetch.get("status") != "skipped"
    log_policy_event(
        "scaffold-rollout-actions-review",
        ok=bool(review.get("ok")),
        details={
            "source": "cli",
            "path": None if args.url else args.path,
            "url": _redact_url(args.url),
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else ("skipped" if not args.url else None),
            "action_count": review.get("action_count", 0),
            "accepted_action_count": review.get("accepted_action_count", 0),
            "provenance_status": (review.get("provenance") or {}).get("status"),
            "error_count": len(review.get("errors", [])),
            "wrote_local_policy_files": False,
            "exit_code": 0 if review.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if review.get("ok") else stderr, review, pretty=args.pretty)
    return 0 if review.get("ok") else 1


def scaffold_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed repeated-scaffold rollout actions as a local crunch canary overlay")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Optimization rollout action bundle JSON path, or '-' for stdin. Ignored when --url is set.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help=f"Managed scaffold rollout actions URL. May also be set with {SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV}.",
    )
    parser.add_argument("--api-key", help="Managed optimizer API key. Prefer --api-key-env for shell history safety.")
    parser.add_argument(
        "--api-key-env",
        default=MANAGED_POLICY_API_KEY_ENV,
        help=f"Environment variable containing the managed optimizer API key, default: {MANAGED_POLICY_API_KEY_ENV}.",
    )
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Fetch without an API key for local/dev managed servers.")
    parser.add_argument("--tenant", help="Optional x-agentflow-tenant header for tenant-bound managed keys.")
    parser.add_argument("--account", help="Optional x-agentflow-account header for account metadata.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum candidate samples to request when fetching.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum candidate error rate to request when fetching.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum actions to request when fetching.")
    parser.add_argument("--source-surface", help="Optional managed server source_surface filter.")
    parser.add_argument("--app-family", help="Optional managed server app_family filter.")
    parser.add_argument("--category", help="Optional managed server category filter.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_MANAGED_POLICY_TIMEOUT_SECONDS", "10")),
        help="HTTP timeout in seconds, default: 10.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local AgentFlow policy overlays, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview the scaffold overlay without writing YAML.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    fetch = None
    fetch_url = args.url or os.getenv(SCAFFOLD_ROLLOUT_ACTIONS_URL_ENV)
    if fetch_url:
        args.url = fetch_url
        bundle, fetch, fetch_exit = _read_rollout_actions_from_url(args)
        if fetch_exit is not None:
            result = {
                "schema": "agentflow.scaffold_rollout_actions_apply.v1",
                "ok": False,
                "dry_run": bool(args.dry_run),
                "fetch": fetch,
                "actions": [],
                "files": [],
                "wrote_policy_files": False,
                "provider_calls_made": False,
                "managed_server_calls_made": fetch.get("status") != "skipped" if isinstance(fetch, dict) else False,
                "error": fetch.get("error") if isinstance(fetch, dict) else None,
            }
            _write_rollout_actions_result(stderr, result, pretty=args.pretty)
            return fetch_exit
    else:
        bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            bundle = {"schema": "invalid"}
            fetch = {"status": "skipped", "reason": "local-input", "error": read_error}

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.scaffold_rollout_review import apply_scaffold_rollout_actions

    result = apply_scaffold_rollout_actions(bundle, config_dir=args.config_dir, dry_run=args.dry_run)
    if fetch:
        result["fetch"] = fetch
        result["managed_server_calls_made"] = fetch.get("status") != "skipped"
    log_policy_event(
        "scaffold-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": None if fetch_url else args.path,
            "url": _redact_url(fetch_url),
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "fetch_status": fetch.get("status") if isinstance(fetch, dict) else ("skipped" if not fetch_url else None),
            "action_count": result.get("action_count", 0),
            "accepted_action_count": result.get("accepted_action_count", 0),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def managed_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed managed pattern rollout actions to local YAML rule files")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Apply only one policy section. Repeat to apply multiple sections.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report exact YAML edits without writing files.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print apply JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_apply.v1",
            "ok": False,
            "dry_run": bool(args.dry_run),
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action bundle could not be read"},
            "files": [],
            "actions": [],
        }
    else:
        if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
            from agentflow_proxy.openai_cache_replay_rollout_actions import apply_openai_cache_replay_rollout_actions

            result = apply_openai_cache_replay_rollout_actions(
                bundle,
                config_dir=args.config_dir,
                dry_run=args.dry_run,
            )
        else:
            from agentflow_proxy.rollout_actions import apply_rollout_actions

            result = apply_rollout_actions(
                bundle,
                config_dir=args.config_dir,
                dry_run=args.dry_run,
                sections=args.section,
            )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "rollout-actions-apply",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "applied_sections": result.get("applied_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(result, command="apply", db_path=str(args.db))
    _attach_terminal_output_compaction_lifecycle_feedback(
        result,
        command="apply",
        db_path=str(args.db),
        result_key="managed_terminal_output_compaction_lifecycle_feedback",
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def managed_rollout_actions_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed pattern rollout actions against recent local traffic metadata")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action bundle JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls and Codex turns to inspect, default: 500.")
    parser.add_argument(
        "--section",
        action="append",
        choices=["crunch", "cache"],
        help="Dry-run only one policy section. Repeat to include multiple sections.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        if isinstance(bundle, dict) and bundle.get("schema") == "agentflow.openai_cache_replay_rollout_actions.v1":
            from agentflow_proxy.openai_cache_replay_rollout_actions import dry_run_openai_cache_replay_rollout_actions

            result = dry_run_openai_cache_replay_rollout_actions(bundle, config_dir=args.config_dir)
            result["db_path"] = args.db
        else:
            from agentflow_proxy.rollout_actions import dry_run_rollout_actions

            store = _open_store_for_db(args.db)
            try:
                result = dry_run_rollout_actions(
                    bundle,
                    store_obj=store,
                    config_dir=args.config_dir,
                    sections=args.section,
                    limit=args.limit,
                )
                result["db_path"] = args.db
            finally:
                store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "rollout-actions-dry-run",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "path": args.path,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "dry_run": True,
            "action_count": len(result.get("actions", [])),
            "affected_metadata_row_count": (result.get("summary") or {}).get("affected_metadata_row_count"),
            "provenance_status": (result.get("provenance") or {}).get("status"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _attach_rollout_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def old_context_summary_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run old-context summarization against recent local traffic")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional policy bundle JSON path, '-' for stdin, or omit to use the currently loaded local crunch policy.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500.")
    parser.add_argument(
        "--profile",
        choices=("current-policy", "tool-protocol-aware"),
        default="current-policy",
        help=(
            "Dry-run profile to simulate. tool-protocol-aware enables a read-only overlay that "
            "summarizes only old non-tool text turns while preserving tool protocol messages."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print dry-run JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    bundle_or_policy: Any = None
    read_error: dict[str, Any] | None = None
    if args.path is not None:
        bundle_or_policy, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)

    if read_error:
        from agentflow_proxy.old_context_summary_dry_run import OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_DRY_RUN_SCHEMA,
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "db_path": args.db,
            "lookback_call_limit": max(0, args.limit),
            "validation": read_error,
            "error": {"type": "read_failed", "message": "policy bundle could not be read"},
            "groups": [],
        }
    else:
        from agentflow_proxy.old_context_summary_dry_run import dry_run_old_context_summary

        result = dry_run_old_context_summary(
            bundle_or_policy,
            db_path=str(args.db),
            limit=max(0, args.limit),
            profile=args.profile,
        )

    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.path or "current-policy",
            "profile": args.profile,
            "db_path": args.db,
            "dry_run": True,
            "eligible_call_count": (result.get("summary") or {}).get("eligible_call_count"),
            "projected_saved_tokens": (result.get("summary") or {}).get("projected_saved_tokens"),
            "projected_net_savings_usd": (result.get("summary") or {}).get("projected_net_savings_usd"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def old_context_summary_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply old-context summarization canary impact against a dry-run projection")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Old-context summary dry-run or policy review JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500.")
    parser.add_argument(
        "--since",
        help="Only count metadata at or after this ISO-8601 post-apply timestamp. Defaults to dry-run generated_at.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print impact JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        from agentflow_proxy.old_context_summary_impact import OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_IMPACT_SCHEMA,
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary dry-run or review report could not be read"},
            "summary": {},
            "privacy": {
                "metadata_only": True,
                "raw_old_context_included": False,
                "generated_summaries_included": False,
                "raw_request_bodies_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "local_session_ids_included": False,
                "cache_keys_included": False,
            },
        }
    else:
        from agentflow_proxy.old_context_summary_impact import measure_old_context_summary_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_old_context_summary_impact(
                report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    log_policy_event(
        "old-context-summary-impact",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "projected_affected_metadata_row_count": summary.get("projected_affected_metadata_row_count"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "quality_gate_verdict": quality_gate.get("verdict"),
            "quality_gate_reason_codes": quality_gate.get("reason_codes"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    _attach_old_context_summary_lifecycle_feedback(result, command="impact", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def old_context_summary_quality_gate_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local old-context summary canary quality gate")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Old-context summary dry-run or policy review JSON path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500.")
    parser.add_argument(
        "--since",
        help="Only count metadata at or after this ISO-8601 post-apply timestamp. Defaults to dry-run generated_at.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print quality gate JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        from agentflow_proxy.old_context_summary_impact import OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA

        result = {
            "schema": OLD_CONTEXT_SUMMARY_QUALITY_GATE_SCHEMA,
            "ok": False,
            "read_only": True,
            "wrote_policy_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "verdict": "hold",
            "reason_codes": ["input-read-failed"],
            "warning_codes": [],
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary dry-run or review report could not be read"},
            "privacy": {
                "metadata_only": True,
                "raw_old_context_included": False,
                "generated_summaries_included": False,
                "raw_request_bodies_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "local_session_ids_included": False,
                "cache_keys_included": False,
                "file_paths_included": False,
            },
        }
    else:
        from agentflow_proxy.old_context_summary_impact import build_old_context_summary_quality_gate

        store = _open_store_for_db(args.db)
        try:
            result = build_old_context_summary_quality_gate(
                report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "old-context-summary-quality-gate",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "candidate_id": result.get("candidate_id"),
            "rule_id": result.get("rule_id"),
            "verdict": result.get("verdict"),
            "reason_codes": result.get("reason_codes"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else result.get("source_error_type"),
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if result.get("ok"):
        _attach_old_context_summary_lifecycle_feedback(result, command="quality-gate", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def _summary_rollout_event_details(command: str, result: dict[str, Any], *, path: str | None, config_dir: str | None = None, db_path: str | None = None) -> dict[str, Any]:
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    files = [item for item in result.get("files", []) if isinstance(item, dict)]

    snapshots = []
    for action in actions:
        edit = action.get("proposed_edit") if isinstance(action.get("proposed_edit"), dict) else {}
        gate = action.get("quality_gate") if isinstance(action.get("quality_gate"), dict) else {}
        snapshots.append({
            key: value
            for key, value in {
                "action_id": action.get("action_id"),
                "target_candidate_id": action.get("target_candidate_id"),
                "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
                "action_type": action.get("action_type"),
                "current_fraction": edit.get("current_fraction") if edit else action.get("current_fraction"),
                "recommended_fraction": edit.get("recommended_fraction") if edit else action.get("projected_fraction"),
                "quality_gate_verdict": gate.get("verdict"),
                "quality_gate_reason_codes": gate.get("reason_codes"),
                "confidence": action.get("confidence"),
                "required_local_review": True,
                "managed_enforced": False,
            }.items()
            if value not in (None, "", [], {})
        })

    return {
        "source": "cli",
        "command": f"old-context-summary-rollout-actions-{command}",
        "input_source": "stdin" if path == "-" else "file",
        "config_dir_configured": bool(config_dir),
        "db_configured": bool(db_path),
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "action_count": len(actions) or result.get("action_count") or validation.get("action_count") or 0,
        "planned_action_count": result.get("planned_action_count") or review.get("planned_action_count") or 0,
        "changed_action_count": result.get("changed_action_count") or review.get("changed_action_count") or 0,
        "rejected_action_count": result.get("rejected_action_count") or review.get("rejected_action_count") or 0,
        "action_snapshots": snapshots,
        "affected_metadata_row_count": summary.get("affected_metadata_row_count") or summary.get("actual_matched_metadata_row_count"),
        "projected_additional_applied_count": summary.get("projected_additional_applied_count"),
        "projected_local_bypass_or_disable_count": summary.get("projected_local_bypass_or_disable_count"),
        "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
        "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
        "changed_file_count": sum(1 for item in files if item.get("changed")),
        "changed_sections": sorted({str(item.get("section")) for item in files if item.get("changed") and item.get("section")}),
        "provenance_status": provenance.get("status"),
        "computed_bundle_hash": provenance.get("computed_bundle_hash"),
        "validation_error_count": len(validation.get("errors", []) if isinstance(validation.get("errors"), list) else []),
        "validation_warning_count": len(validation.get("warnings", []) if isinstance(validation.get("warnings"), list) else []),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "exit_code": 0 if result.get("ok") else 1,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_old_context_included": False,
            "generated_summaries_included": False,
            "raw_messages_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "cache_keys_included": False,
            "yaml_contents_included": False,
        },
    }


def old_context_summary_rollout_actions_review_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Review managed old-context summary rollout actions against local crunch rules")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_review.v1",
            "ok": False,
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import plan_summary_rollout_actions

        result = plan_summary_rollout_actions(bundle, config_dir=args.config_dir)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-review",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("review", result, path=args.path, config_dir=args.config_dir),
    )
    _write_rollout_actions_result(stdout if result.get("ok") else stderr, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply managed old-context summary rollout actions to local crunch rules")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_apply.v1",
            "ok": False,
            "dry_run": bool(args.dry_run),
            "config_dir": args.config_dir,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "files": [],
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import apply_summary_rollout_actions

        result = apply_summary_rollout_actions(bundle, config_dir=args.config_dir, dry_run=args.dry_run)
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-apply",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("apply", result, path=args.path, config_dir=args.config_dir),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run managed old-context summary rollout actions against recent local metadata")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action bundle JSON path, or '-' for stdin.")
    parser.add_argument("--config-dir", default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")))
    parser.add_argument("--db", default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    bundle, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_dry_run.v1",
            "ok": False,
            "dry_run": True,
            "read_only": True,
            "config_dir": args.config_dir,
            "db_path": args.db,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action bundle could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import dry_run_summary_rollout_actions

        store = _open_store_for_db(args.db)
        try:
            result = dry_run_summary_rollout_actions(
                bundle,
                store_obj=store,
                config_dir=args.config_dir,
                limit=args.limit,
            )
            result["db_path"] = args.db
        finally:
            store.conn.close()
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-dry-run",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("dry-run", result, path=args.path, config_dir=args.config_dir, db_path=args.db),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def old_context_summary_rollout_actions_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply old-context summary rollout-action impact against a dry-run projection")
    parser.add_argument("path", nargs="?", default="-", help="Old-context summary rollout action dry-run JSON path, or '-' for stdin.")
    parser.add_argument("--db", default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--since")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.old_context_summary_rollout_actions_impact.v1",
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "old-context summary rollout action dry-run report could not be read"},
            "actions": [],
        }
    else:
        from agentflow_proxy.old_context_summary_rollout_actions import measure_summary_rollout_action_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_summary_rollout_action_impact(report, store_obj=store, limit=args.limit, since=args.since)
        finally:
            store.conn.close()
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "old-context-summary-rollout-actions-impact",
        ok=bool(result.get("ok")),
        details=_summary_rollout_event_details("impact", result, path=args.path, db_path=args.db),
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result.get("ok") else 1


def managed_rollout_actions_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Measure post-apply rollout-action impact against a dry-run projection")
    parser.add_argument(
        "path",
        nargs="?",
        default="-",
        help="Rollout action dry-run JSON report path, or '-' for stdin. Default: stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="Local AgentFlow SQLite DB path, default: ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls and Codex turns to inspect, default: 500.")
    parser.add_argument(
        "--since",
        help="Only count provider/Codex metadata at or after this ISO-8601 post-apply timestamp. Defaults to dry-run generated_at.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print impact JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    dry_run_report, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = {
            "schema": "agentflow.pattern_rollout_actions_impact.v1",
            "ok": False,
            "read_only": True,
            "validation": read_error,
            "error": {"type": "read_failed", "message": "rollout action dry-run report could not be read"},
            "actions": [],
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_messages_included": False,
                "raw_responses_included": False,
                "tool_payloads_included": False,
                "request_ids_included": False,
                "local_session_ids_included": False,
                "file_paths_included": False,
                "yaml_contents_included": False,
            },
        }
    else:
        from agentflow_proxy.rollout_actions import measure_rollout_action_impact

        store = _open_store_for_db(args.db)
        try:
            result = measure_rollout_action_impact(
                dry_run_report,
                store_obj=store,
                limit=args.limit,
                since=args.since,
            )
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "rollout-actions-impact",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "input_source": "stdin" if args.path == "-" else "file",
            "db_configured": bool(args.db),
            "since": args.since,
            "action_count": len(result.get("actions", [])),
            "projected_affected_metadata_row_count": summary.get("projected_affected_metadata_row_count"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_matched_provider_call_count": summary.get("actual_matched_provider_call_count"),
            "actual_matched_codex_turn_count": summary.get("actual_matched_codex_turn_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_estimated_cost_savings_usd": summary.get("actual_estimated_cost_savings_usd"),
            "actions_without_post_apply_matches": summary.get("actions_without_post_apply_matches"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_rollout_actions_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def policy_rollback_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Rollback local AgentFlow policy YAML files from apply backups")
    parser.add_argument(
        "--apply-id",
        help="Rollback the exact policy workbench apply transaction with this apply ID.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local rule files, default: ~/.agentflow",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache", "routing_experiments", "codex_app"],
        help="Rollback only one policy section. Repeat to rollback multiple sections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the backups that would be restored without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow CLI-only partial recovery when an apply event is missing or backup metadata is incomplete.",
    )
    parser.add_argument(
        "--reload-url",
        default=os.getenv("AGENTFLOW_ADMIN_URL", _default_policy_reload_url()),
        help=f"Admin reload URL for --apply-id rollback, default: {_default_policy_reload_url()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("AGENTFLOW_ADMIN_TIMEOUT", "10")),
        help="HTTP timeout in seconds for the loopback reload call, default: 10.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow posting reload to a non-loopback URL. Use only for explicit trusted tunnels.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print rollback JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    if args.apply_id:
        from agentflow_proxy.policy_events import log_policy_event
        from agentflow_proxy.policy_workbench import POLICY_DRAFT_ROLLBACK_SCHEMA, rollback_policy_apply

        if not args.dry_run and not args.allow_non_loopback and not _is_loopback_url(args.reload_url):
            result = {
                "schema": POLICY_DRAFT_ROLLBACK_SCHEMA,
                "ok": False,
                "status": "blocked",
                "apply_id": args.apply_id,
                "backup_id": args.apply_id,
                "dry_run": bool(args.dry_run),
                "force": bool(args.force),
                "config_dir": args.config_dir,
                "manifest_source": None,
                "apply_event_found": None,
                "requested_sections": args.section or ["routing", "crunch", "cache", "routing_experiments", "codex_app"],
                "restored_sections": [],
                "skipped_sections": [],
                "files": [],
                "current_backups": [],
                "reloaded_modules": False,
                "reload": None,
                "verification": None,
                "privacy": {"provider_calls_made": False, "managed_server_calls_made": False, "loopback_admin_calls_made": False},
                "error": {
                    "type": "unsafe_url",
                    "message": "policy apply rollback only posts reloads to loopback URLs unless --allow-non-loopback is set",
                    "url": args.reload_url,
                },
            }
            log_policy_event(
                "rollback",
                ok=False,
                details={"source": "cli", "apply_id": args.apply_id, "error_type": "unsafe_url", "exit_code": 2},
            )
            _write_policy_rollback_result(stdout, result, pretty=args.pretty)
            return 2

        async def reload_state() -> dict[str, Any]:
            return await _reload_policy_state_via_url(args.reload_url, timeout=args.timeout)

        result = asyncio.run(rollback_policy_apply(
            args.apply_id,
            config_dir=args.config_dir,
            sections=args.section,
            dry_run=args.dry_run,
            force=args.force,
            reload_policy_state=reload_state,
            event_source="cli",
            loopback_admin_calls_made=not args.dry_run,
        ))
        _write_policy_rollback_result(stdout, result, pretty=args.pretty)
        return 0 if result.get("ok") else 1

    from agentflow_proxy.policy_bundle import rollback_policy_files
    from agentflow_proxy.policy_events import log_policy_event

    result = rollback_policy_files(config_dir=args.config_dir, dry_run=args.dry_run, sections=args.section)
    log_policy_event(
        "rollback",
        ok=bool(result["ok"]),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "restored_sections": result.get("restored_sections", []),
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "old_context_summarization": result.get("old_context_summarization"),
            "codex_app": result.get("codex_app"),
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result["ok"] else 1,
        },
    )
    _write_policy_rollback_result(stdout, result, pretty=args.pretty)
    return 0 if result["ok"] else 1


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.feedback import managed_feedback_status_cli as _managed_feedback_status_cli

    return _managed_feedback_status_cli(argv, stdout=stdout)


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.feedback import managed_feedback_flush_cli as _managed_feedback_flush_cli

    return _managed_feedback_flush_cli(argv, stdout=stdout)


def sqlite_maintenance_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run local SQLite retention maintenance for AgentFlow metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow SQLite DB path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Retention window in days. Defaults to AGENTFLOW_SQLITE_RETENTION_DAYS or 7.",
    )
    parser.add_argument("--disable-retention", action="store_true", help="Record disabled maintenance without deleting rows.")
    parser.add_argument("--dry-run", action="store_true", help="Report rows that would be purged without deleting them.")
    parser.add_argument("--no-analyze", action="store_true", help="Skip ANALYZE after a purge.")
    parser.add_argument("--no-optimize", action="store_true", help="Skip PRAGMA optimize after maintenance.")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after the purge. Use only during an explicit maintenance window.")
    args = parser.parse_args(argv)
    store = _open_store_for_db(str(args.db))
    try:
        result = store.run_sqlite_maintenance(
            retention_days=None if args.retention_days is None and not args.disable_retention else (0 if args.disable_retention else args.retention_days),
            dry_run=bool(args.dry_run),
            analyze=not bool(args.no_analyze),
            optimize=not bool(args.no_optimize),
            vacuum=bool(args.vacuum),
        )
    finally:
        try:
            store.conn.close()
        except Exception:
            pass
    _write_json(stdout or sys.stdout, result)
    return 0


def _phase_routing_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if command != "dry-run" or not isinstance(result, dict) or result.get("schema") != "agentflow.phase_routing_dry_run.v1":
        return None
    from agentflow_proxy import __version__

    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    rules = [item for item in result.get("rules", []) if isinstance(item, dict)]
    candidate_ids = [
        str(item.get("candidate_id") or item.get("rule_id"))
        for item in rules
        if item.get("candidate_id") or item.get("rule_id")
    ]
    basis = {
        "command": command,
        "candidate_ids": candidate_ids,
        "matched_count": summary.get("matched_count"),
        "projected_candidate_count": summary.get("projected_candidate_count"),
        "projected_savings_usd": summary.get("projected_savings_usd"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    excluded: dict[str, int] = {}
    for item in summary.get("excluded_count_by_reason") or []:
        if isinstance(item, dict):
            reason = str(item.get("reason") or "unknown")
            excluded[reason] = excluded.get(reason, 0) + int(item.get("count") or 0)
    metadata = {
        "schema": "agentflow.phase_routing_lifecycle_metadata.v1",
        "lifecycle_kind": "phase_routing",
        "command": "phase-routing-dry-run",
        "local_result_status": "ok",
        "dry_run": True,
        "read_only": True,
        "wrote_local_files": bool(result.get("wrote_local_files")),
        "altered_provider_routing": bool(result.get("altered_provider_routing")),
        "policy_source": result.get("policy_source"),
        "sampled_call_count": result.get("sampled_call_count"),
        "rule_count": result.get("rule_count"),
        "matched_count": summary.get("matched_count"),
        "projected_candidate_count": summary.get("projected_candidate_count"),
        "excluded_count": summary.get("excluded_count"),
        "projected_savings_usd": summary.get("projected_savings_usd"),
        "projected_target_cost_usd": summary.get("projected_target_cost_usd"),
        "risk_warning_count": summary.get("risk_warning_count"),
        "candidate_rule_ids": candidate_ids,
        "excluded_count_by_reason": excluded,
        "settings": result.get("settings") if isinstance(result.get("settings"), dict) else {},
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "provider_bodies_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "tenant_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "db_path_included": False,
            "policy_file_contents_included": False,
            "secrets_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": "dry-run" if result.get("ok", True) else "rejected",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": f"phase-routing:{digest}",
        "bundle_hash": None,
        "policy_sections": ["routing"],
        "validation_warning_count": 0,
        "review_warning_count": int(summary.get("risk_warning_count") or 0),
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_phase_routing_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _phase_routing_lifecycle_payload(command, result)
    if payload is None:
        return
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.PHASE_ROUTING_LIFECYCLE_SOURCE_SURFACE,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        if store is not None:
            store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def codex_diagnose_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Codex app-server routing, crunching, and cache effectiveness from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent turn/start rows to inspect, default: 500, max: 5000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_codex_effectiveness

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_codex_effectiveness(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def codex_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report managed Codex app canary impact and lifecycle evidence by rule")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent turn/start rows to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_codex_canary_impact

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_codex_canary_impact(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_scoreboard_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.optimization.scoreboard import openai_scoreboard_cli as _openai_scoreboard_cli

    return _openai_scoreboard_cli(argv, stdout=stdout)


def openai_routing_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI local routing opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_routing_report import build_openai_routing_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_routing_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_old_context_summary_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI old-context summary opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_old_context_summary_report import build_openai_old_context_summary_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_old_context_summary_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_cache_replay_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure OpenAI cache replay opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_report import build_openai_cache_replay_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def optimization_action_ledger_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize cross-family optimization eligibility from local call metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.optimization_action_ledger import build_optimization_action_ledger_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_optimization_action_ledger_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def optimization_coordinator_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run cross-family optimization coordinator decisions against local metadata")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional managed rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument("--provider", help="Only inspect rows for this provider, for example openai or anthropic.")
    parser.add_argument("--source-surface", help="Only inspect rows for this source surface, for example openai_responses.")
    parser.add_argument("--canary-fraction", type=float, help="Coordinator canary fraction override for the dry run.")
    parser.add_argument("--holdout-fraction", type=float, help="Coordinator holdout fraction override for the dry run.")
    parser.add_argument(
        "--local-salt",
        help="Coordinator cohort salt override. The salt is used for hashing but is not emitted in the report.",
    )
    parser.add_argument("--examples", type=int, default=20, help="Maximum sanitized example decisions to include, default: 20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    rollout_actions = None
    if args.path is not None:
        rollout_actions, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
        if read_error:
            result = {
                "schema": "agentflow.optimization_coordinator_dry_run.v1",
                "ok": False,
                "dry_run": True,
                "read_only": True,
                "db_path": args.db,
                "validation": read_error,
                "error": {"type": "read_failed", "message": "managed rollout action bundle could not be read"},
                "privacy": {
                    "metadata_only": True,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "policy_files_changed": False,
                    "provider_body_changed": False,
                },
            }
            if args.pretty:
                stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                _write_json(stdout, result)
            return 1

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.optimization_coordinator_dry_run import build_optimization_coordinator_dry_run

    store = open_store_for_db(str(args.db))
    try:
        result = build_optimization_coordinator_dry_run(
            store,
            rollout_actions=rollout_actions,
            limit=args.limit,
            provider=args.provider,
            source_surface=args.source_surface,
            local_salt=args.local_salt,
            canary_fraction=args.canary_fraction,
            holdout_fraction=args.holdout_fraction,
            examples=args.examples,
        )
    finally:
        store.conn.close()
    result["db_path"] = args.db
    from agentflow_proxy.policy_events import log_policy_event

    log_policy_event(
        "optimization-coordinator-dry-run",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "db_path": args.db,
            "path": args.path,
            "dry_run": True,
            "sampled_call_count": result.get("sampled_call_count"),
            "decision_count": result.get("decision_count"),
            "projected_savings_usd_est": result.get("projected_savings_usd_est"),
            "exit_code": 0,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def provider_tool_adoption_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    from agentflow_proxy.provider_adoption import provider_tool_adoption_report_cli as _provider_tool_adoption_report_cli

    return _provider_tool_adoption_report_cli(argv, stdout=stdout)


def repeated_scaffold_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure repeated provider-message scaffolding crunch opportunity")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-repeated-rows",
        type=int,
        default=2,
        help="Minimum rows per hidden scaffold fingerprint before a group is considered repeated, default: 2",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.repeated_scaffold_report import build_repeated_scaffold_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_opportunity_report(
            store,
            limit=args.limit,
            min_repeated_rows=args.min_repeated_rows,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure instruction-section deduplication opportunity")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls and Codex app events to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-repeated-rows",
        type=int,
        default=2,
        help="Minimum rows per hidden instruction fingerprint before a group is considered repeated, default: 2",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_report import build_instruction_dedup_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_opportunity_report(
            store,
            limit=args.limit,
            min_repeated_rows=args.min_repeated_rows,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run instruction-section deduplication plans")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls and Codex app events to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        help="Maximum dry-run plan samples to emit, default: 20, max: 200",
    )
    parser.add_argument(
        "--local-salt",
        default=None,
        help="Optional local cohort salt for deterministic canary/holdout assignment; never emitted.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_dry_run import build_instruction_dedup_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_dry_run(
            store,
            limit=args.limit,
            examples=args.examples,
            local_salt=args.local_salt,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def instruction_dedup_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report instruction-section deduplication canary impact and lifecycle gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Recent provider calls to inspect, default: 500, max: 10000")
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for widen, default: 0.")
    parser.add_argument(
        "--max-negative-net-savings-rate",
        type=float,
        default=0.0,
        help="Applied negative net savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.instruction_dedup_impact import build_instruction_dedup_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_instruction_dedup_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_negative_net_savings_rate=args.max_negative_net_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_instruction_dedup_lifecycle_feedback(result, store=store)
        from agentflow_proxy.instruction_dedup_feedback import SOURCE_SURFACE as INSTRUCTION_DEDUP_LIFECYCLE_SOURCE_SURFACE
        from agentflow_proxy.optimization.feedback import managed_feedback_status_result

        queue_status = managed_feedback_status_result(
            store,
            source_surface=INSTRUCTION_DEDUP_LIFECYCLE_SOURCE_SURFACE,
            sample_limit=5,
        )
        result["managed_lifecycle_feedback_queue"] = queue_status
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure terminal-output compaction opportunity for plateaued tool-result sessions")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum adjacent call text size for plateau detection, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_report import build_terminal_output_compaction_opportunity_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_opportunity_report(
            store,
            limit=args.limit,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Codex terminal-transcript compaction opportunity from local event windows")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent Codex turn/start rows to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-input-chars",
        type=int,
        default=8000,
        help="Minimum Codex turn input text size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--min-terminal-chars",
        type=int,
        default=2000,
        help="Minimum estimated terminal-transcript chars for candidate rows, default: 2000",
    )
    parser.add_argument(
        "--compaction-ratio",
        type=float,
        default=0.65,
        help="Projected removable fraction of terminal transcript chars after preserving diagnostics, default: 0.65",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_report import build_codex_terminal_transcript_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_opportunity_report(
            store,
            limit=args.limit,
            min_input_chars=args.min_input_chars,
            min_terminal_chars=args.min_terminal_chars,
            compaction_ratio=args.compaction_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_opportunity_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Anthropic thinking-session compaction opportunity from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum thinking-session input size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--metadata-compaction-ratio",
        type=float,
        default=0.20,
        help="Projected removable fraction of input chars when raw bodies are unavailable, default: 0.20",
    )
    parser.add_argument(
        "--body-compaction-ratio",
        type=float,
        default=0.65,
        help="Projected removable fraction of detected thinking-history chars when local bodies are available, default: 0.65",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_report import build_anthropic_thinking_compaction_opportunity_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_opportunity_report(
            store,
            limit=args.limit,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
            metadata_compaction_ratio=args.metadata_compaction_ratio,
            body_compaction_ratio=args.body_compaction_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Anthropic thinking-history compaction canary impact from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", default=None, help="Optional inclusive UTC timestamp lower bound.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied samples for budget feedback.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout samples for budget feedback.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def anthropic_thinking_compaction_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Anthropic thinking-history compaction plans from local request metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=50,
        help="Maximum plans to include in output, default: 50, max: 500",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum request text size for candidate rows, default: 8000",
    )
    parser.add_argument(
        "--min-block-chars",
        type=int,
        default=2000,
        help="Minimum thinking block size for a compaction target, default: 2000",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Near-duplicate shingle similarity threshold, default: 0.95",
    )
    parser.add_argument(
        "--canary-fraction",
        type=float,
        default=1.0,
        help="Deterministic dry-run canary fraction, default: 1.0",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Deterministic dry-run holdout fraction, default: 0.0",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.anthropic_thinking_compaction_dry_run import build_anthropic_thinking_compaction_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_anthropic_thinking_compaction_dry_run(
            store,
            limit=args.limit,
            examples=args.examples,
            min_text_chars=args.min_text_chars,
            min_block_chars=args.min_block_chars,
            similarity_threshold=args.similarity_threshold,
            canary_fraction=args.canary_fraction,
            holdout_fraction=args.holdout_fraction,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Codex terminal-transcript compaction plans from local event windows")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Codex turn/start rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_dry_run import build_codex_terminal_transcript_compaction_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_compaction_dry_run(
            store,
            limit=args.limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def codex_terminal_transcript_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Codex terminal-transcript compaction canary impact and lifecycle gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Codex turn/start rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect rows at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for promote, default: 0.")
    parser.add_argument(
        "--max-negative-savings-rate",
        type=float,
        default=0.0,
        help="Applied negative savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.codex_terminal_compaction_impact import build_codex_terminal_transcript_compaction_impact_report
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_codex_terminal_transcript_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_negative_savings_rate=args.max_negative_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_codex_terminal_transcript_lifecycle_feedback(result, store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run terminal-output compaction plans for Anthropic tool-result history")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--keep-recent-turns",
        type=int,
        default=2,
        help="Newest message turns to preserve untouched, default: 2",
    )
    parser.add_argument(
        "--min-block-chars",
        type=int,
        default=2000,
        help="Minimum terminal/log text block size eligible for compaction, default: 2000",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=8000,
        help="Minimum adjacent call text size for plateau detection, default: 8000",
    )
    parser.add_argument(
        "--max-plateau-delta-ratio",
        type=float,
        default=0.03,
        help="Maximum adjacent text size delta ratio for plateau detection, default: 0.03",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_dry_run import build_terminal_output_compaction_dry_run

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_dry_run(
            store,
            limit=args.limit,
            keep_recent_turns=args.keep_recent_turns,
            min_block_chars=args.min_block_chars,
            min_text_chars=args.min_text_chars,
            max_plateau_delta_ratio=args.max_plateau_delta_ratio,
        )
        _attach_terminal_output_compaction_lifecycle_feedback(result, command="review", store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def terminal_output_compaction_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report terminal-output compaction canary impact and rollback gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument("--min-net-savings-usd", type=float, default=0.0, help="Minimum applied net savings for promote, default: 0.")
    parser.add_argument(
        "--max-non-positive-savings-rate",
        type=float,
        default=0.0,
        help="Applied non-positive savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.terminal_compaction_impact import build_terminal_output_compaction_impact_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_terminal_output_compaction_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            min_net_savings_usd=args.min_net_savings_usd,
            max_non_positive_savings_rate=args.max_non_positive_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_terminal_output_compaction_lifecycle_feedback(result, command="impact", store=store)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def repeated_scaffold_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report repeated-scaffold crunch canary impact and rollback gates")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied rows per candidate, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout rows per candidate, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Evidence staleness threshold, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Applied error rate hold threshold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Applied-minus-holdout error regression rollback threshold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Applied-minus-holdout retry regression rollback threshold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Applied-minus-holdout latency hold threshold, default: 2000.")
    parser.add_argument(
        "--max-non-positive-savings-rate",
        type=float,
        default=0.0,
        help="Applied non-positive savings rate hold threshold, default: 0.0.",
    )
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Absolute applied error rate rollback threshold, default: 0.20.")
    parser.add_argument(
        "--flush-feedback",
        action="store_true",
        help="Flush due repeated-scaffold lifecycle feedback after queueing, bounded by --feedback-limit.",
    )
    parser.add_argument(
        "--feedback-dry-run",
        action="store_true",
        help="Preview due repeated-scaffold lifecycle feedback flush rows without claiming or sending.",
    )
    parser.add_argument("--feedback-limit", type=int, default=5, help="Maximum repeated-scaffold feedback rows to flush or sample, default: 5, max: 100.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.repeated_scaffold_impact import build_repeated_scaffold_impact_report
    from agentflow_proxy.repeated_scaffold_feedback import (
        build_repeated_scaffold_lifecycle_feedback_status,
        flush_repeated_scaffold_lifecycle_feedback,
    )

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            max_non_positive_savings_rate=args.max_non_positive_savings_rate,
            rollback_error_rate=args.rollback_error_rate,
        )
        _attach_repeated_scaffold_lifecycle_feedback(result, store)
        feedback_limit = max(1, min(int(args.feedback_limit or 5), 100))
        if args.flush_feedback or args.feedback_dry_run:
            result["managed_lifecycle_feedback_flush"] = asyncio.run(
                flush_repeated_scaffold_lifecycle_feedback(
                    store,
                    limit=feedback_limit,
                    dry_run=bool(args.feedback_dry_run),
                )
            )
        result["managed_lifecycle_feedback_queue"] = build_repeated_scaffold_lifecycle_feedback_status(
            store,
            sample_limit=feedback_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def repeated_scaffold_activation_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Anthropic repeated-scaffold policy-decision activation coverage")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent Anthropic provider calls to inspect, default: 500, max: 10000",
    )
    parser.add_argument("--since", help="Only inspect calls at or after this ISO timestamp.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json
    from agentflow_proxy.repeated_scaffold_activation import build_repeated_scaffold_activation_report

    store = open_store_for_db(str(args.db))
    try:
        result = build_repeated_scaffold_activation_report(
            store,
            limit=args.limit,
            since=args.since,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def _attach_repeated_scaffold_lifecycle_feedback(result: dict[str, Any], store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.repeated_scaffold_feedback import (
        SOURCE_SURFACE as REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
        build_repeated_scaffold_lifecycle_feedback,
    )

    payload = build_repeated_scaffold_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-repeated-scaffold-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_instruction_dedup_lifecycle_feedback(result: dict[str, Any], store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.instruction_dedup_feedback import (
        build_instruction_dedup_lifecycle_feedback,
        queue_instruction_dedup_lifecycle_feedback,
    )

    payload = build_instruction_dedup_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-instruction-dedup-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            queue_instruction_dedup_lifecycle_feedback(
                store,
                result,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_terminal_output_compaction_lifecycle_feedback(
    result: dict[str, Any],
    *,
    command: str,
    store: Any | None = None,
    db_path: str | None = None,
    result_key: str = "managed_lifecycle_feedback",
) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.terminal_compaction_feedback import (
        build_terminal_output_compaction_lifecycle_feedback,
        queue_terminal_output_compaction_lifecycle_feedback,
    )

    payload = build_terminal_output_compaction_lifecycle_feedback(result, command=command)
    if payload is None:
        return

    close_store = False
    local_store = store
    if local_store is None:
        if not db_path:
            return
        local_store = _open_store_for_db(str(db_path))
        close_store = True
    try:
        meta = asyncio.run(
            queue_terminal_output_compaction_lifecycle_feedback(
                local_store,
                result,
                command=command,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }
    finally:
        if close_store and local_store is not None:
            local_store.conn.close()

    result[result_key] = _public_lifecycle_feedback_meta(meta)


def _attach_codex_terminal_transcript_lifecycle_feedback(result: dict[str, Any], *, store: Any) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.codex_terminal_compaction_feedback import (
        build_codex_terminal_transcript_lifecycle_feedback,
        queue_codex_terminal_transcript_lifecycle_feedback,
    )

    payload = build_codex_terminal_transcript_lifecycle_feedback(result)
    if payload is None:
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "skipped",
            "reason": "no-codex-terminal-transcript-lifecycle-candidates",
            "auth_configured": recommendations.managed_auth_configured(),
        })
        return

    try:
        meta = asyncio.run(
            queue_codex_terminal_transcript_lifecycle_feedback(
                store,
                result,
                flush_immediately=False,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
            "auth_configured": recommendations.managed_auth_configured(),
        }

    result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta(meta)


def _attach_openai_cache_replay_lifecycle_feedback(result: dict[str, Any], *, db_path: str) -> None:
    from agentflow_proxy import recommendations
    from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_lifecycle_feedback

    payload = build_openai_cache_replay_lifecycle_feedback(result)
    if payload is None:
        return
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.CACHE_REPLAY_LIFECYCLE_SOURCE_SURFACE,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        if store is not None:
            store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def openai_cache_replay_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report OpenAI cache replay canary impact and safety gates from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent OpenAI calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied cache replay samples before promotion, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout cache replay samples before promotion, default: 1.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--max-invalidation-rate", type=float, default=0.02, help="Invalidation rate that triggers rollback, default: 0.02.")
    parser.add_argument("--min-cache-hit-rate", type=float, default=0.01, help="Minimum applied cache hit rate before promotion, default: 0.01.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--min-savings-realization-ratio", type=float, default=0.50, help="Minimum observed/projected savings ratio before hold, default: 0.50.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_impact import build_openai_cache_replay_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            max_invalidation_rate=args.max_invalidation_rate,
            min_cache_hit_rate=args.min_cache_hit_rate,
            rollback_error_rate=args.rollback_error_rate,
            min_savings_realization_ratio=args.min_savings_realization_ratio,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()
    _attach_openai_cache_replay_lifecycle_feedback(result, db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_cache_replay_readiness_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Show OpenAI cache replay readiness from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for replay opportunity, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=500,
        help="Recent OpenAI calls to scan for replay impact evidence, default: 500, max: 10000.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_readiness import build_openai_cache_replay_readiness_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_readiness_report(
            store,
            opportunity_limit=args.opportunity_limit,
            impact_limit=args.impact_limit,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_cache_replay_apply_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Graduate ready OpenAI cache replay candidates into a local cache canary overlay")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR") or os.getenv("AGENTFLOW_POLICY_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory for local AgentFlow policy overlays, default: AGENTFLOW_CONFIG_DIR, AGENTFLOW_POLICY_CONFIG_DIR, or ~/.agentflow",
    )
    parser.add_argument(
        "--opportunity-limit",
        type=int,
        default=1000,
        help="Recent OpenAI calls to scan for replay opportunity, default: 1000, max: 10000.",
    )
    parser.add_argument(
        "--impact-limit",
        type=int,
        default=500,
        help="Recent OpenAI calls to scan for replay impact evidence, default: 500, max: 10000.",
    )
    parser.add_argument(
        "--min-observed-savings-usd",
        type=float,
        default=0.0,
        help="Minimum observed applied savings required before writing a canary, default: 0.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
        help="Fraction of matching traffic to keep as holdout, default: 0.20.",
    )
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum ready candidates to write, default: 10.")
    parser.add_argument("--dry-run", action="store_true", help="Preview cache canary policy without writing YAML.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_cache_replay_apply import apply_openai_cache_replay_candidates
    from agentflow_proxy.policy_events import log_policy_event

    store = _open_store_for_db(str(args.db))
    try:
        result = apply_openai_cache_replay_candidates(
            store,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
            opportunity_limit=args.opportunity_limit,
            impact_limit=args.impact_limit,
            min_observed_savings_usd=args.min_observed_savings_usd,
            holdout_fraction=args.holdout_fraction,
            max_candidates=args.max_candidates,
        )
    finally:
        store.conn.close()
    log_policy_event(
        "openai-cache-replay-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "db": str(args.db),
            "config_dir": str(args.config_dir),
            "dry_run": bool(args.dry_run),
            "accepted_candidate_count": (result.get("summary") or {}).get("accepted_candidate_count"),
            "policy_rule_count": (result.get("summary") or {}).get("policy_rule_count"),
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def _openai_cache_replay_dry_run_read_error_result(read_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.openai_cache_replay_dry_run.v1",
        "ok": False,
        "summary": {
            "openai_rows_considered": 0,
            "policy_rule_count": 0,
            "projected_applied_rows": 0,
            "holdout_rows": 0,
            "blocked_rows": 0,
            "projected_hits": 0,
            "projected_savings_usd": 0.0,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
        },
        "read_error": read_error,
        "rows": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "pattern_hashes_included": False,
        },
    }


def openai_cache_replay_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run OpenAI cache replay pattern rules against local metadata")
    parser.add_argument(
        "path",
        help="Proposed cache policy JSON path, policy bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent OpenAI provider calls to inspect and aggregate rows to return, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _openai_cache_replay_dry_run_read_error_result(read_error)
        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 1

    from agentflow_proxy.openai_cache_replay_dry_run import build_openai_cache_replay_dry_run

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_cache_replay_dry_run(store, proposed, limit=args.limit)
    finally:
        store.conn.close()
    result["ok"] = True
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def openai_old_context_summary_dry_run_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run OpenAI old-context summary plans with protocol preservation checks")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_old_context_summary_dry_run import build_openai_old_context_summary_dry_run
    from agentflow_proxy.optimization.cli_support import open_store_for_db, write_json

    store = open_store_for_db(str(args.db))
    try:
        result = build_openai_old_context_summary_dry_run(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def openai_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report OpenAI local routing canary impact and promotion verdicts from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent OpenAI calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-fallback-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout fallback-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--min-projection-realization-ratio", type=float, default=0.50, help="Minimum observed/projected savings ratio before hold, default: 0.50.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.openai_canary_impact import build_openai_canary_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_openai_canary_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_fallback_rate_delta=args.max_fallback_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            rollback_error_rate=args.rollback_error_rate,
            min_projection_realization_ratio=args.min_projection_realization_ratio,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_impact_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report Claude local routing canary impact and promotion verdicts from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate before hold, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-retry-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout retry-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-fallback-rate-delta", type=float, default=0.10, help="Maximum applied-minus-holdout fallback-rate delta before hold, default: 0.10.")
    parser.add_argument("--max-rate-limit-fallback-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout rate-limit fallback-rate delta before hold, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression before hold, default: 2000.")
    parser.add_argument("--rollback-error-rate", type=float, default=0.20, help="Applied error rate that triggers rollback, default: 0.20.")
    parser.add_argument("--rollback-fallback-rate", type=float, default=0.50, help="Applied fallback rate that triggers rollback, default: 0.50.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_claude_canary_impact_report(
            store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
            max_error_rate=args.max_error_rate,
            max_error_rate_delta=args.max_error_rate_delta,
            max_retry_rate_delta=args.max_retry_rate_delta,
            max_fallback_rate_delta=args.max_fallback_rate_delta,
            max_rate_limit_fallback_rate_delta=args.max_rate_limit_fallback_rate_delta,
            max_latency_regression_ms=args.max_latency_regression_ms,
            rollback_error_rate=args.rollback_error_rate,
            rollback_fallback_rate=args.rollback_fallback_rate,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_actions_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Emit reviewable local Claude routing canary widen, hold, rollback, and more-samples actions")
    parser.add_argument(
        "impact_report",
        nargs="?",
        help="Optional Claude canary impact report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh impact report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan when building a fresh report, default: 500.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp when building a fresh report.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--widen-step", type=float, default=0.25, help="Fraction added when widening an existing Claude canary, default: 0.25.")
    parser.add_argument("--max-canary-fraction", type=float, default=1.0, help="Maximum Claude canary fraction, default: 1.0.")
    parser.add_argument("--preserved-holdout-fraction", type=float, default=0.10, help="Minimum holdout fraction preserved while widening, default: 0.10.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.impact_report:
            report = _read_json_input(str(args.impact_report), stdin=stdin)
        else:
            from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_claude_canary_impact_report(
                    store,
                    limit=args.limit,
                    since=args.since,
                    min_applied_samples=args.min_applied_samples,
                    min_holdout_samples=args.min_holdout_samples,
                    max_evidence_age_hours=args.max_evidence_age_hours,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.claude_canary_rollout_actions_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.claude_canary_actions import build_claude_canary_actions

    result = build_claude_canary_actions(
        report,
        widen_step=args.widen_step,
        max_canary_fraction=args.max_canary_fraction,
        preserved_holdout_fraction=args.preserved_holdout_fraction,
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def claude_canary_actions_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed Claude routing canary actions to the local routing policy file")
    parser.add_argument(
        "actions",
        nargs="?",
        default="-",
        help="Claude canary rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without writing local YAML files.")
    parser.add_argument("--write", dest="dry_run", action="store_false", help="Write reviewed Claude canary edits to the local routing YAML file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.claude_canary_rollout_actions_apply_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.claude_canary_actions import apply_claude_canary_actions
    from agentflow_proxy.policy_events import log_policy_event

    result = apply_claude_canary_actions(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
    )
    event = log_policy_event(
        "claude-canary-actions-apply",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": args.dry_run,
            "action_count": (result.get("summary") or {}).get("action_count") if isinstance(result.get("summary"), dict) else None,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count") if isinstance(result.get("summary"), dict) else None,
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if event is not None:
        result["policy_event"] = {
            "id": event.get("id"),
            "action": event.get("action"),
            "created_at": event.get("created_at"),
            "ok": event.get("ok"),
        }
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def routing_canary_promote_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Promote winning Claude routing canaries to permanent local routing rules")
    parser.add_argument(
        "impact_report",
        nargs="?",
        help="Optional Claude canary impact report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh impact report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent Claude calls to scan when building a fresh report, default: 500.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp when building a fresh report.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples required for promotion, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout canary samples required for promotion, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours when building a fresh report, default: 72.")
    parser.add_argument("--max-error-rate", type=float, default=0.05, help="Maximum applied error rate for promotion, default: 0.05.")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.05, help="Maximum applied-minus-holdout error-rate delta for promotion, default: 0.05.")
    parser.add_argument("--max-latency-regression-ms", type=int, default=2000, help="Maximum applied-minus-holdout latency regression for promotion, default: 2000.")
    parser.add_argument("--apply", action="store_true", help="Write permanent rules and disable matching local canary entries.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.impact_report:
            report = _read_json_input(str(args.impact_report), stdin=stdin)
        else:
            from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_claude_canary_impact_report(
                    store,
                    limit=args.limit,
                    since=args.since,
                    min_applied_samples=args.min_applied_samples,
                    min_holdout_samples=args.min_holdout_samples,
                    max_evidence_age_hours=args.max_evidence_age_hours,
                    max_error_rate=args.max_error_rate,
                    max_error_rate_delta=args.max_error_rate_delta,
                    max_latency_regression_ms=args.max_latency_regression_ms,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.routing_canary_promotion_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.routing_canary_promote import (
        apply_routing_canary_promotion_plan,
        build_routing_canary_promotion_plan,
    )

    plan = build_routing_canary_promotion_plan(
        report,
        config_dir=args.config_dir,
        min_applied_samples=args.min_applied_samples,
        min_holdout_samples=args.min_holdout_samples,
        max_error_rate=args.max_error_rate,
        max_error_rate_delta=args.max_error_rate_delta,
        max_latency_regression_ms=args.max_latency_regression_ms,
    )
    result = (
        apply_routing_canary_promotion_plan(plan, config_dir=args.config_dir, dry_run=not args.apply)
        if plan.get("ok")
        else plan
    )
    event = log_policy_event(
        "routing-canary-promote",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "config_dir": args.config_dir,
            "dry_run": not args.apply,
            "action_count": (result.get("summary") or {}).get("action_count") if isinstance(result.get("summary"), dict) else None,
            "planned_action_count": (result.get("summary") or {}).get("planned_action_count") if isinstance(result.get("summary"), dict) else None,
            "promotion_action_count": (plan.get("summary") or {}).get("promotion_action_count") if isinstance(plan.get("summary"), dict) else None,
            "omitted_count": (plan.get("summary") or {}).get("omitted_count") if isinstance(plan.get("summary"), dict) else None,
            "changed_files": [
                file.get("path")
                for file in result.get("files", [])
                if isinstance(file, dict) and file.get("changed")
            ],
            "wrote_policy_files": bool(result.get("wrote_policy_files")),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if event is not None:
        result["policy_event"] = {
            "id": event.get("id"),
            "action": event.get("action"),
            "created_at": event.get("created_at"),
            "ok": event.get("ok"),
        }
    result["promotion_plan"] = {
        "schema": plan.get("schema"),
        "ok": plan.get("ok"),
        "summary": plan.get("summary"),
    }
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def routing_experiment_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local budgeted routing A/B experiment results from metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum candidate rows to include, default: 20.")
    parser.add_argument("--since", help="Only include post-fix shadow yield rows at or after this ISO timestamp.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Rolling post-fix shadow-yield window when --since is omitted, default: 24. Use 0 for all history.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.routing_experiments import build_routing_experiment_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_routing_experiment_report(
            store,
            limit=args.limit,
            since=args.since,
            window_hours=args.window_hours,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def routing_promotion_draft_stage_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Stage local policy drafts from promoted routing experiment candidates")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Routing experiment promotion report JSON path, or '-' for stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum candidate rows when building a fresh report, default: 20.")
    parser.add_argument(
        "--draft-id",
        help="Optional local draft ID. When multiple promoted candidates are staged, a numeric suffix is added.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AGENTFLOW_POLICY_DRAFT_DIR", str(Path.home() / ".agentflow" / "policy_drafts")),
        help="Local draft workspace directory, default: ~/.agentflow/policy_drafts.",
    )
    parser.add_argument(
        "--initial-canary-fraction",
        type=float,
        default=0.10,
        help="Deterministic applied canary fraction for staged routing canaries, default: 0.10.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.10,
        help="Deterministic holdout fraction for staged routing canaries, default: 0.10.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    if args.promotion_report:
        report = _read_json_input(str(args.promotion_report), stdin=stdin)
    else:
        from agentflow_proxy.routing_experiments import build_routing_experiment_report

        store = _open_store_for_db(str(args.db))
        try:
            report = build_routing_experiment_report(store, limit=args.limit)
        finally:
            store.conn.close()

    from agentflow_proxy.policy_events import log_policy_event
    from agentflow_proxy.routing_promotion_drafts import stage_routing_promotion_drafts

    result = asyncio.run(stage_routing_promotion_drafts(
        report,
        draft_id=args.draft_id,
        workspace=args.workspace,
        initial_canary_fraction=args.initial_canary_fraction,
        holdout_fraction=args.holdout_fraction,
    ))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    log_policy_event(
        "routing-promotion-draft-stage",
        ok=bool(result.get("ok")),
        details={
            "source": "cli",
            "path": args.promotion_report,
            "workspace": args.workspace,
            "candidate_count": summary.get("candidate_count", 0),
            "promoted_candidate_count": summary.get("promoted_candidate_count", 0),
            "staged_count": summary.get("staged_count", 0),
            "omitted_count": summary.get("omitted_count", 0),
            "wrote_active_policy_files": False,
            "reloaded_modules": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
            "exit_code": 0 if result.get("ok") else 1,
        },
    )
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0 if result.get("ok") else 1


def cache_replayability_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure replay-safe cache opportunity and blockers from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Replayability groups to return, default: 25",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_cache_replayability

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_cache_replayability(store, limit=args.limit))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def cache_replay_cohorts_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Rank replay-ready plateau cohorts from local cache metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Recent provider and Codex rows per surface to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Ranked cohorts to return, default: 25",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_cache_replay_cohort_ranking

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            stats_cache_replay_cohort_ranking(
                store,
                limit=args.limit,
                row_limit=args.scan_limit,
            )
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def cache_smoke_diagnostic_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether the local exact cache can serve hits from metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Newest cache rows to summarize, default: 10",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=5000,
        help="Recent call rows to inspect for cache decisions, default: 5000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.cache_smoke import build_cache_smoke_diagnostic

    store = _open_store_for_db(str(args.db))
    try:
        result = build_cache_smoke_diagnostic(store, limit=args.limit, scan_limit=args.scan_limit)
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _cache_replay_dry_run_read_error_result(read_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "agentflow.cache_replay_dry_run.v1",
        "ok": False,
        "summary": {
            "rows_considered": 0,
            "policy_rule_count": 0,
            "candidate_rows": 0,
            "projected_exact_hits": 0,
            "projected_streaming_hits": 0,
            "estimated_saved_cost_usd": 0.0,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
        },
        "read_error": read_error,
        "rows": [],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "cache_keys_included": False,
            "pattern_hashes_included": False,
        },
    }


def cache_replay_dry_run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Dry-run proposed cache replay pattern rules against recent local metadata")
    parser.add_argument(
        "path",
        help="Proposed cache policy JSON path, policy bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=1000,
        help="Recent provider and Codex rows per surface to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Dry-run aggregate rows to return, default: 50",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    proposed, read_error, _stdin_used = _read_policy_json_arg(args.path, stdin=stdin, stdin_used=False)
    if read_error:
        result = _cache_replay_dry_run_read_error_result(read_error)
        if args.pretty:
            stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            _write_json(stdout, result)
        return 1

    from agentflow_proxy.stats import stats_cache_replay_dry_run

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            stats_cache_replay_dry_run(
                store,
                proposed,
                limit=args.limit,
                row_limit=args.scan_limit,
            )
        )
    finally:
        store.conn.close()
    result["ok"] = True
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def phase_routing_report_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Measure or dry-run Anthropic phase-routing policy from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent Anthropic provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    parser.add_argument(
        "--dry-run-policy",
        help="Proposed routing YAML or policy bundle JSON/YAML to simulate without writing local policy files.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum matched safe samples required before projecting savings in dry-run mode, default: 1.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Historical matched error-rate warning threshold for dry-run mode, default: 0.05.",
    )
    parser.add_argument(
        "--stale-hours",
        type=int,
        default=168,
        help="Treat evidence older than this as stale in dry-run mode, default: 168 hours.",
    )
    parser.add_argument(
        "--require-shadow-support",
        action="store_true",
        help="Exclude streaming calls when the proposed phase policy requires shadow/holdout comparison support.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.phase_routing_report import (
        build_phase_routing_dry_run,
        build_phase_routing_report,
        load_phase_routing_policy,
    )

    store = _open_store_for_db(str(args.db))
    try:
        if args.dry_run_policy:
            proposed = load_phase_routing_policy(args.dry_run_policy)
            result = build_phase_routing_dry_run(
                store,
                proposed,
                limit=args.limit,
                min_samples=args.min_samples,
                max_error_rate=args.max_error_rate,
                stale_hours=args.stale_hours,
                require_shadow_support=args.require_shadow_support,
            )
        else:
            result = build_phase_routing_report(store, limit=args.limit)
    finally:
        store.conn.close()
    if args.dry_run_policy:
        _attach_phase_routing_lifecycle_feedback(result, command="dry-run", db_path=str(args.db))
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def session_phase_memory_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build metadata-only session phase memory rollups from local calls")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Recent provider calls to inspect, default: 1000, max: 10000",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Recent calls per session to include in each memory window, default: 20, max: 200",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.session_phase_memory import build_session_phase_memory

    store = _open_store_for_db(str(args.db))
    try:
        result = build_session_phase_memory(
            store,
            limit=args.limit,
            window_size=args.window_size,
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def managed_pattern_rollups_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export metadata-only managed pattern canary cohort outcome rollups")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider calls and Codex turn/start rows to inspect per surface, default: 500, max: 5000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required before a cohort bucket is marked ready, default: 10",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.stats import stats_managed_pattern_rollups

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(stats_managed_pattern_rollups(store, limit=args.limit, min_samples=args.min_samples))
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_eval_plan_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Export family-agnostic optimization eval plans from local metadata")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Recent provider and Codex rows to inspect per source report, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples for managed pattern readiness normalization, default: 1",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout

    from agentflow_proxy.optimization_eval_plan import build_optimization_eval_plan

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            build_optimization_eval_plan(
                store,
                limit=args.limit,
                min_samples=args.min_samples,
            )
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def _read_json_input(path: str, *, stdin: Any = None) -> dict[str, Any]:
    if path == "-":
        source = stdin if stdin is not None else sys.stdin
        return json.loads(source.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def optimization_shadow_eval_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run opt-in local metadata-only shadow evals from an optimization eval plan")
    parser.add_argument(
        "plan",
        help="Optimization eval plan JSON path, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for result records, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Optional path to write sanitized per-candidate result records as JSONL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to provider-call execution. Requires --budget-usd > 0; rows without replayable local inputs remain blocked.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=0.0,
        help="Maximum provider-call spend allowed when --execute is set. Default 0, which disables execution.",
    )
    parser.add_argument(
        "--min-output-similarity",
        type=float,
        default=0.9,
        help="Minimum offline fixture output similarity/quality score required for a pass verdict, default: 0.9",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum plan rows to evaluate, default: 100, max: 1000",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if args.execute and args.budget_usd <= 0:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_shadow_eval_error.v1",
                "error": {
                    "type": "missing_budget_cap",
                    "message": "--execute requires --budget-usd greater than 0",
                },
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 2

    from agentflow_proxy.optimization_shadow_eval import run_optimization_shadow_eval

    try:
        plan = _read_json_input(str(args.plan), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_shadow_eval_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    store = _open_store_for_db(str(args.db))
    try:
        result = run_optimization_shadow_eval(
            plan,
            store=store,
            execute=bool(args.execute),
            budget_usd=float(args.budget_usd or 0.0),
            min_output_similarity=float(args.min_output_similarity or 0.9),
            max_candidates=int(args.max_candidates or 100),
            results_jsonl_path=args.results_jsonl,
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_eval_queue_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded batch from the local optimization eval queue")
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queue selection and result records, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--family",
        help="Optional optimization family filter, for example phase_routing or cache_replayability.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum queue candidates to evaluate, default: 25, max: 1000",
    )
    parser.add_argument(
        "--plan-limit",
        type=int,
        default=500,
        help="Recent provider and Codex rows to inspect while building the queue, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples for managed pattern readiness normalization, default: 1",
    )
    parser.add_argument(
        "--max-candidate-age-hours",
        type=int,
        help="Record candidates older than this many hours as blocked with candidate-stale.",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Optional path to write sanitized per-candidate result records as JSONL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to provider-call execution. Requires --budget-usd > 0; rows without replayable local inputs remain blocked.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=0.0,
        help="Maximum provider-call spend allowed when --execute is set. Default 0, which disables execution.",
    )
    parser.add_argument(
        "--min-output-similarity",
        type=float,
        default=0.9,
        help="Minimum offline fixture output similarity/quality score required for a pass verdict, default: 0.9",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    if args.execute and args.budget_usd <= 0:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_eval_queue_error.v1",
                "error": {
                    "type": "missing_budget_cap",
                    "message": "--execute requires --budget-usd greater than 0",
                },
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 2

    from agentflow_proxy.optimization_eval_queue import run_optimization_eval_queue

    store = _open_store_for_db(str(args.db))
    try:
        result = asyncio.run(
            run_optimization_eval_queue(
                store,
                family=args.family,
                limit=int(args.limit or 25),
                max_candidate_age_hours=args.max_candidate_age_hours,
                execute=bool(args.execute),
                budget_usd=float(args.budget_usd or 0.0),
                min_output_similarity=float(args.min_output_similarity or 0.9),
                plan_limit=int(args.plan_limit or 500),
                min_samples=int(args.min_samples or 1),
                results_jsonl_path=args.results_jsonl,
            )
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_promotion_report_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Score local eval, canary, and holdout evidence into optimization promotion verdicts")
    parser.add_argument(
        "plan",
        nargs="?",
        help="Optional optimization eval plan JSON path, or '-' to read from stdin. If omitted, a fresh plan is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--evidence-report",
        action="append",
        default=[],
        help="Optional sanitized post-apply impact or feedback report JSON to merge by candidate_id. May be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum candidates to score and recent eval result rows to inspect, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples when building a fresh eval plan, default: 1",
    )
    parser.add_argument(
        "--min-eval-pass-count",
        type=int,
        default=1,
        help="Minimum passing local eval results required before widening, default: 1",
    )
    parser.add_argument(
        "--min-canary-applied-samples",
        type=int,
        default=2,
        help="Minimum applied canary samples required before widening, default: 2",
    )
    parser.add_argument(
        "--min-canary-holdout-samples",
        type=int,
        default=1,
        help="Minimum holdout samples required before widening, default: 1",
    )
    parser.add_argument(
        "--max-evidence-age-hours",
        type=int,
        default=168,
        help="Mark eval evidence stale after this many hours, default: 168",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting one compact line.",
    )
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        plan = _read_json_input(str(args.plan), stdin=stdin) if args.plan else None
        evidence_reports = [_read_json_input(str(path), stdin=stdin) for path in args.evidence_report]
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_report_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report

    store = _open_store_for_db(str(args.db))
    try:
        result = build_optimization_promotion_report(
            store,
            plan=plan,
            evidence_reports=evidence_reports,
            limit=args.limit,
            min_samples=args.min_samples,
            min_eval_pass_count=args.min_eval_pass_count,
            min_canary_applied_samples=args.min_canary_applied_samples,
            min_canary_holdout_samples=args.min_canary_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_promotion_actions_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Emit local rollout actions from optimization promotion verdicts")
    parser.add_argument(
        "promotion_report",
        nargs="?",
        help="Optional optimization promotion report JSON path, or '-' to read from stdin. If omitted, a fresh report is built from local metadata.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path when building a fresh promotion report, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum candidates to score when building a fresh promotion report, default: 500, max: 10000",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples when building a fresh eval plan for a fresh promotion report, default: 1",
    )
    parser.add_argument("--initial-canary-fraction", type=float, default=0.10, help="Canary fraction for first local rollout actions, default: 0.10")
    parser.add_argument("--widen-step", type=float, default=0.25, help="Fraction added when widening an existing canary, default: 0.25")
    parser.add_argument("--max-canary-fraction", type=float, default=1.0, help="Maximum recommended canary fraction, default: 1.0")
    parser.add_argument("--holdout-fraction", type=float, default=0.10, help="Deterministic holdout fraction to preserve, default: 0.10")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        if args.promotion_report:
            report = _read_json_input(str(args.promotion_report), stdin=stdin)
        else:
            from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report

            store = _open_store_for_db(str(args.db))
            try:
                report = build_optimization_promotion_report(
                    store,
                    limit=args.limit,
                    min_samples=args.min_samples,
                )
            finally:
                store.conn.close()
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_rollout_actions_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_local_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_actions import build_optimization_promotion_actions

    result = build_optimization_promotion_actions(
        report,
        initial_canary_fraction=args.initial_canary_fraction,
        widen_step=args.widen_step,
        max_canary_fraction=args.max_canary_fraction,
        holdout_fraction=args.holdout_fraction,
    )

    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stdout, result)
    return 0


def optimization_promotion_canary_apply_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply optimization promotion canaries to local routing, crunch, and cache policy files")
    parser.add_argument(
        "promotion_actions",
        nargs="?",
        default="-",
        help="Optimization promotion rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--config-dir",
        default=os.getenv("AGENTFLOW_CONFIG_DIR", str(Path.home() / ".agentflow")),
        help="Directory containing local AgentFlow YAML policy files, default: AGENTFLOW_CONFIG_DIR or ~/.agentflow",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path for queued managed lifecycle feedback, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["routing", "crunch", "cache"],
        help="Policy section to apply. May be repeated. Default: routing, crunch, and cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without writing local YAML files.",
    )
    parser.add_argument(
        "--write",
        dest="dry_run",
        action="store_false",
        help="Write reviewed promotion canary edits to local YAML policy files.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.promotion_actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_canary_apply_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_canary import apply_optimization_promotion_canaries

    result = apply_optimization_promotion_canaries(
        bundle,
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        sections=args.section,
    )
    _attach_optimization_promotion_lifecycle_feedback(
        result,
        command="dry-run" if args.dry_run else "apply",
        db_path=str(args.db),
    )
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def optimization_promotion_impact_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description="Report post-apply optimization promotion canary impact from local metadata")
    parser.add_argument(
        "promotion_actions",
        nargs="?",
        default="-",
        help="Optimization promotion rollout action bundle JSON path, or '-' for stdin.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("AGENTFLOW_DATABASE_URL") or os.getenv("AGENTFLOW_DB", str(Path.home() / ".agentflow" / "agentflow.sqlite3")),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--limit", type=int, default=500, help="Maximum recent calls to scan, default: 500, max: 10000.")
    parser.add_argument("--since", help="Only scan calls at or after this ISO-8601 timestamp. Defaults to the action bundle generated_at.")
    parser.add_argument("--min-applied-samples", type=int, default=2, help="Minimum applied canary samples before widening, default: 2.")
    parser.add_argument("--min-holdout-samples", type=int, default=1, help="Minimum holdout samples before widening, default: 1.")
    parser.add_argument("--max-evidence-age-hours", type=float, default=72.0, help="Mark evidence stale after this many hours, default: 72.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    try:
        bundle = _read_json_input(str(args.promotion_actions), stdin=stdin)
    except (OSError, json.JSONDecodeError) as exc:
        _write_json(
            stderr,
            {
                "ok": False,
                "schema": "agentflow.optimization_promotion_impact_error.v1",
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "wrote_policy_files": False,
                "wrote_store": False,
            },
        )
        return 1

    from agentflow_proxy.optimization_promotion_impact import measure_optimization_promotion_impact

    store = _open_store_for_db(str(args.db))
    try:
        result = measure_optimization_promotion_impact(
            bundle,
            store_obj=store,
            limit=args.limit,
            since=args.since,
            min_applied_samples=args.min_applied_samples,
            min_holdout_samples=args.min_holdout_samples,
            max_evidence_age_hours=args.max_evidence_age_hours,
        )
    finally:
        store.conn.close()

    _attach_optimization_promotion_lifecycle_feedback(result, command="impact", db_path=str(args.db))
    stream = stdout if result.get("ok") else stderr
    if args.pretty:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, result)
    return 0 if result.get("ok") else 1


def _write_validation_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_diff_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_review_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_apply_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_stage_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_validate_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_policy_draft_apply_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _write_rollout_actions_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def _rollout_lifecycle_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rollout_lifecycle_nested_counts(items: list[dict[str, Any]], object_key: str, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        nested = item.get(object_key) if isinstance(item.get(object_key), dict) else {}
        value = nested.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rollout_lifecycle_rejection_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        reason = str(action.get("reason") or "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
        family_validation = action.get("family_validation") if isinstance(action.get("family_validation"), dict) else {}
        for error in family_validation.get("errors") or []:
            if isinstance(error, dict):
                message = str(error.get("message") or "family-specific-validation-failed")
                counts[message] = counts.get(message, 0) + 1
    return dict(sorted(counts.items()))


def _rollout_action_id(action: dict[str, Any]) -> str:
    basis = {
        "policy_section": action.get("policy_section"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "pattern_hash": action.get("pattern_hash"),
        "action_type": action.get("action_type"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"rollout-action:{digest[:24]}"


def _rollout_action_snapshot(action: dict[str, Any]) -> dict[str, Any]:
    edit = action.get("proposed_edit") if isinstance(action.get("proposed_edit"), dict) else {}
    current_rule = action.get("current_rule") if isinstance(action.get("current_rule"), dict) else {}
    family_validation = action.get("family_validation") if isinstance(action.get("family_validation"), dict) else {}
    snapshot = {
        "action_id": _rollout_action_id(action),
        "status": action.get("status"),
        "reason": action.get("reason"),
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "policy_section": action.get("policy_section"),
        "policy_source": current_rule.get("policy_source"),
        "pattern_family": family_validation.get("family"),
        "policy_profile": family_validation.get("policy_profile"),
        "family_validation_status": family_validation.get("status"),
        "pattern_hash": action.get("pattern_hash"),
        "action_type": action.get("action_type"),
        "current_fraction": edit.get("current_fraction") if edit else action.get("current_fraction"),
        "recommended_fraction": edit.get("recommended_fraction") if edit else action.get("projected_fraction"),
        "confidence": action.get("confidence"),
        "blockers": action.get("blockers") if isinstance(action.get("blockers"), list) else [],
        "required_local_review": True,
        "managed_enforced": False,
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _rollout_lifecycle_event_type(command: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return "rejected"
    if command == "review":
        return "reviewed"
    if command == "dry-run" or result.get("dry_run"):
        return "dry-run"
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    action_types = {str(item.get("action_type") or "") for item in actions}
    if action_types & {"rollback", "retire", "disable"}:
        return "rollback"
    return "applied"


def _rollout_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy import __version__

    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    snapshots = [_rollout_action_snapshot(item) for item in actions]
    action_ids = sorted({str(item["action_id"]) for item in snapshots if item.get("action_id")})
    candidate_ids = sorted({str(item.get("target_candidate_id")) for item in snapshots if item.get("target_candidate_id")})
    rule_ids = sorted({str(item.get("target_rule_id")) for item in snapshots if item.get("target_rule_id")})
    pattern_hashes = sorted({str(item.get("pattern_hash")) for item in snapshots if str(item.get("pattern_hash") or "").startswith("sha256:")})
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    provenance = result.get("provenance") if isinstance(result.get("provenance"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    files = [item for item in result.get("files", []) if isinstance(item, dict)]
    event_type = _rollout_lifecycle_event_type(command, result)
    metadata: dict[str, Any] = {
        "schema": "agentflow.rollout_action_lifecycle_metadata.v1",
        "lifecycle_kind": "pattern_rollout_actions",
        "command": f"rollout-actions-{command}",
        "local_result_status": "ok" if result.get("ok") else "error",
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "action_count": len(actions) or result.get("action_count") or validation.get("action_count") or 0,
        "planned_action_count": result.get("planned_action_count") or review.get("planned_action_count") or 0,
        "changed_action_count": result.get("changed_action_count") or review.get("changed_action_count") or 0,
        "rejected_action_count": result.get("rejected_action_count") or review.get("rejected_action_count") or 0,
        "action_type_counts": _rollout_lifecycle_counts(actions, "action_type"),
        "policy_section_counts": _rollout_lifecycle_counts(actions, "policy_section"),
        "local_status_counts": _rollout_lifecycle_counts(actions, "status"),
        "pattern_family_counts": _rollout_lifecycle_nested_counts(actions, "family_validation", "family"),
        "family_validation_status_counts": _rollout_lifecycle_nested_counts(actions, "family_validation", "status"),
        "rejection_reason_counts": _rollout_lifecycle_rejection_counts(actions),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "pattern_hashes": pattern_hashes,
        "rollout_action_snapshots": snapshots,
        "validation_error_count": len(validation.get("errors", []) if isinstance(validation.get("errors"), list) else []),
        "validation_warning_count": len(validation.get("warnings", []) if isinstance(validation.get("warnings"), list) else []),
        "review_error_count": len(review.get("errors", []) if isinstance(review.get("errors"), list) else []),
        "review_warning_count": len(review.get("warnings", []) if isinstance(review.get("warnings"), list) else []),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "provenance_status": provenance.get("status"),
        "provenance_bundle_hash": provenance.get("bundle_hash"),
        "computed_bundle_hash": provenance.get("computed_bundle_hash"),
        "changed_file_count": sum(1 for item in files if item.get("changed")),
        "changed_sections": sorted({str(item.get("section")) for item in files if item.get("changed") and item.get("section")}),
        "affected_metadata_row_count": summary.get("affected_metadata_row_count"),
        "affected_provider_call_count": summary.get("affected_provider_call_count"),
        "affected_codex_turn_count": summary.get("affected_codex_turn_count"),
        "projected_additional_applied_count": summary.get("projected_additional_applied_count"),
        "projected_local_bypass_or_disable_count": summary.get("projected_local_bypass_or_disable_count"),
        "historical_tokens_saved_est": summary.get("historical_tokens_saved_est"),
        "historical_estimated_cost_savings_usd": summary.get("historical_estimated_cost_savings_usd"),
        "safety_stop_reason_counts": _rollout_lifecycle_counts(
            [
                reason
                for action in actions
                for reason in (action.get("local_bypass_reasons") or [])
                if isinstance(reason, dict)
            ],
            "value",
        ),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "raw_params_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": action_ids[0] if len(action_ids) == 1 else None,
        "bundle_hash": provenance.get("computed_bundle_hash") or provenance.get("bundle_hash"),
        "policy_sections": sorted(_rollout_lifecycle_counts(actions, "policy_section")),
        "validation_warning_count": metadata.get("validation_warning_count", 0),
        "review_warning_count": metadata.get("review_warning_count", 0),
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _public_lifecycle_feedback_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "enabled": meta.get("enabled"),
            "server_url": _redact_url(meta.get("server_url")),
            "endpoint": meta.get("endpoint"),
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "queue_id": meta.get("queue_id"),
            "attempts": meta.get("attempts"),
            "status_code": meta.get("status_code"),
            "latency_ms": meta.get("latency_ms"),
            "auth_configured": meta.get("auth_configured"),
            "api_key_value_included": False,
            "payload_included": False,
        }.items()
        if value is not None
    }


def _attach_rollout_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _rollout_lifecycle_payload(command, result)
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = _open_store_for_db(str(db_path))
    try:
        meta = asyncio.run(recommendations.queue_policy_event_feedback(store, payload))
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if command in {"dry-run", "impact"} and public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _optimization_promotion_event_type(command: str, result: dict[str, Any]) -> str:
    if command == "impact":
        return "impact"
    if command == "dry-run" or result.get("dry_run"):
        return "dry-run"
    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    action_types = {str(item.get("action_type") or "") for item in actions}
    if action_types & {"rollback", "retire", "disable"}:
        return "rollback"
    return "apply"


def _optimization_promotion_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optimization_promotion_reason_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        for key in ("reason",):
            value = action.get(key)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
        next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
        for key in ("reason_codes", "warning_codes"):
            for value in next_step.get(key) or []:
                if isinstance(value, str) and value:
                    counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optimization_promotion_action_snapshot(action: dict[str, Any]) -> dict[str, Any]:
    local_update = action.get("local_policy_update") if isinstance(action.get("local_policy_update"), dict) else {}
    projection = action.get("projection") if isinstance(action.get("projection"), dict) else {}
    actual = action.get("actual") if isinstance(action.get("actual"), dict) else {}
    next_step = action.get("next_step") if isinstance(action.get("next_step"), dict) else {}
    cohorts = actual.get("cohorts") if isinstance(actual.get("cohorts"), dict) else {}
    evidence = action.get("evidence_summary") if isinstance(action.get("evidence_summary"), dict) else {}
    evidence_counts = evidence.get("cohort_counts") if isinstance(evidence.get("cohort_counts"), dict) else {}
    policy_source = (
        action.get("policy_source")
        or local_update.get("policy_source")
        or "managed-recommended"
    )
    requested_model_family = (
        action.get("requested_model_family")
        or action.get("requested_model")
        or local_update.get("model_pattern")
    )
    routed_model_family = (
        action.get("routed_model_family")
        or action.get("candidate_target_model")
        or action.get("target_model")
        or local_update.get("candidate_target_model")
        or local_update.get("target_model")
    )
    actual_counts = {
        "canary_applied": actual.get("actual_canary_applied_count"),
        "canary_holdout": actual.get("actual_canary_holdout_count"),
        "skipped": actual.get("actual_skipped_count"),
        "bypassed_or_disabled": actual.get("actual_bypassed_or_disabled_count"),
        "safety_stopped": actual.get("actual_safety_stopped_count"),
    }
    projected_counts = {
        "canary_applied": projection.get("current_canary_applied_count", evidence_counts.get("canary_applied")),
        "canary_holdout": projection.get("current_canary_holdout_count", evidence_counts.get("canary_holdout")),
        "bypassed_or_disabled": projection.get("current_bypassed_or_disabled_count", evidence_counts.get("bypassed_or_disabled")),
    }
    snapshot = {
        "action_id": action.get("action_id"),
        "action_type": action.get("action_type"),
        "status": action.get("status"),
        "reason": action.get("reason"),
        "action_family": action.get("action_family"),
        "optimization_family": action.get("optimization_family"),
        "source_surface": action.get("source_surface"),
        "app_family": action.get("app_family"),
        "policy_section": action.get("policy_section"),
        "policy_source": policy_source,
        "target_candidate_id": action.get("target_candidate_id"),
        "target_rule_id": action.get("target_rule_id") or action.get("rule_id"),
        "requested_model_family": requested_model_family,
        "routed_model_family": routed_model_family,
        "model_family_pair": f"{requested_model_family or 'unknown'}->{routed_model_family}" if routed_model_family else None,
        "target_model": routed_model_family,
        "canary_fraction": action.get("canary_fraction") or projection.get("target_canary_fraction"),
        "holdout_fraction": action.get("holdout_fraction") or projection.get("target_holdout_fraction"),
        "projected_savings_usd": projection.get("projected_savings_usd"),
        "observed_savings_usd": actual.get("observed_savings_usd"),
        "projected_cohort_counts": {key: value for key, value in projected_counts.items() if value not in (None, "", [], {})},
        "actual_cohort_counts": {key: value for key, value in actual_counts.items() if value not in (None, "", [], {})},
        "actual_canary_applied_count": actual.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": actual.get("actual_canary_holdout_count"),
        "actual_skipped_count": actual.get("actual_skipped_count"),
        "actual_bypassed_or_disabled_count": actual.get("actual_bypassed_or_disabled_count"),
        "actual_safety_stopped_count": actual.get("actual_safety_stopped_count"),
        "error_rate_delta": actual.get("applied_minus_holdout_error_rate"),
        "retry_rate_delta": actual.get("applied_minus_holdout_retry_rate"),
        "latency_avg_delta_ms": actual.get("applied_minus_holdout_latency_avg_ms"),
        "applied_error_rate": (cohorts.get("canary_applied") or {}).get("error_rate") if isinstance(cohorts.get("canary_applied"), dict) else None,
        "holdout_error_rate": (cohorts.get("canary_holdout") or {}).get("error_rate") if isinstance(cohorts.get("canary_holdout"), dict) else None,
        "applied_retry_rate": (cohorts.get("canary_applied") or {}).get("retry_rate") if isinstance(cohorts.get("canary_applied"), dict) else None,
        "holdout_retry_rate": (cohorts.get("canary_holdout") or {}).get("retry_rate") if isinstance(cohorts.get("canary_holdout"), dict) else None,
        "status_buckets": actual.get("status_buckets") or [],
        "reason_buckets": actual.get("reason_buckets") or [],
        "error_buckets": actual.get("error_buckets") or [],
        "latency_buckets": actual.get("latency_buckets") or [],
        "next_step_verdict": next_step.get("verdict"),
        "next_step_reason_codes": next_step.get("reason_codes") or [],
        "next_step_warning_codes": next_step.get("warning_codes") or [],
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in (None, "", [], {})
    }


def _optimization_promotion_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy import __version__

    actions = [item for item in result.get("actions", []) if isinstance(item, dict)]
    snapshots = [_optimization_promotion_action_snapshot(item) for item in actions]
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    source_bundle = result.get("source_action_bundle") if isinstance(result.get("source_action_bundle"), dict) else {}
    action_ids = sorted({str(item.get("action_id")) for item in snapshots if item.get("action_id")})
    candidate_ids = sorted({str(item.get("target_candidate_id")) for item in snapshots if item.get("target_candidate_id")})
    rule_ids = sorted({str(item.get("target_rule_id")) for item in snapshots if item.get("target_rule_id")})
    policy_sections = sorted(_optimization_promotion_counts(actions, "policy_section"))
    source_surface_counts = _optimization_promotion_counts(snapshots, "source_surface")
    model_family_pair_counts = _optimization_promotion_counts(snapshots, "model_family_pair")
    event_type = _optimization_promotion_event_type(command, result)
    basis = {
        "command": command,
        "generated_at": result.get("generated_at"),
        "source_generated_at": source_bundle.get("generated_at"),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "status": result.get("status"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    metadata: dict[str, Any] = {
        "schema": "agentflow.optimization_promotion_lifecycle_feedback.v1",
        "lifecycle_kind": "optimization_promotion_canary",
        "command": f"optimization-promotion-{command}",
        "local_result_status": "ok" if result.get("ok") else "error",
        "dry_run": bool(result.get("dry_run")),
        "read_only": bool(result.get("read_only")),
        "wrote_policy_files": bool(result.get("wrote_policy_files")),
        "action_count": len(actions) or summary.get("action_count") or source_bundle.get("action_count") or 0,
        "planned_action_count": summary.get("planned_action_count"),
        "skipped_action_count": summary.get("skipped_action_count"),
        "error_count": summary.get("error_count"),
        "action_type_counts": _optimization_promotion_counts(actions, "action_type"),
        "policy_section_counts": _optimization_promotion_counts(actions, "policy_section"),
        "source_surface_counts": source_surface_counts,
        "model_family_pair_counts": model_family_pair_counts,
        "local_status_counts": _optimization_promotion_counts(actions, "status"),
        "reason_code_counts": _optimization_promotion_reason_counts(actions),
        "action_ids": action_ids,
        "candidate_ids": candidate_ids,
        "rule_ids": rule_ids,
        "action_snapshots": snapshots,
        "sampled_call_count": summary.get("sampled_call_count"),
        "observed_promotion_metadata_row_count": summary.get("observed_promotion_metadata_row_count"),
        "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
        "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
        "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
        "actual_skipped_count": summary.get("actual_skipped_count"),
        "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
        "actual_safety_stopped_count": summary.get("actual_safety_stopped_count"),
        "observed_savings_usd": summary.get("observed_savings_usd"),
        "next_step_counts": summary.get("next_step_counts"),
        "stale_evidence_action_count": summary.get("stale_evidence_action_count"),
        "error_type": (result.get("error") or {}).get("type") if isinstance(result.get("error"), dict) else None,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_responses_included": False,
            "raw_transcripts_included": False,
            "raw_params_included": False,
            "tool_payloads_included": False,
            "request_ids_included": False,
            "local_session_ids_included": False,
            "file_paths_included": False,
            "yaml_contents_included": False,
            "db_path_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": action_ids[0] if len(action_ids) == 1 else f"optimization-promotion:{digest}",
        "bundle_hash": f"sha256:{hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}",
        "policy_sections": policy_sections,
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_optimization_promotion_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _optimization_promotion_lifecycle_payload(command, result)
    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.OPTIMIZATION_PROMOTION_LIFECYCLE_SOURCE_SURFACE,
                queue_when_disabled=True,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": recommendations.recommendations_enabled(),
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        if store is not None:
            store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _old_context_summary_lifecycle_result(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if command == "review":
        impact = result.get("impact_summary") if isinstance(result.get("impact_summary"), dict) else {}
        sections = impact.get("sections") if isinstance(impact.get("sections"), dict) else {}
        crunch = sections.get("crunch") if isinstance(sections.get("crunch"), dict) else {}
        dry_run = crunch.get("old_context_summary_dry_run")
        return dry_run if isinstance(dry_run, dict) else None
    return result


def _old_context_summary_lifecycle_event_type(command: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return "rejected"
    if command == "quality-gate":
        return "quality-gate"
    if command == "impact":
        return "impact"
    if command == "review":
        return "reviewed"
    return "dry-run"


def _old_context_summary_metadata_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    text_l = text.lower()
    unsafe_terms = {
        "account",
        "apikey",
        "api_key",
        "authorization",
        "body",
        "cache_key",
        "content",
        "file",
        "message",
        "path",
        "payload",
        "prompt",
        "request",
        "response",
        "secret",
        "session",
        "summary_text",
        "tenant",
        "tool",
        "transcript",
    }
    if (
        len(text) > 128
        or any(char.isspace() for char in text)
        or any(char in text for char in ("/", "\\", "{", "}", "[", "]", "\"", "'"))
        or any(term in text_l for term in unsafe_terms)
    ):
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text


def _old_context_summary_quality_gate_feedback(
    *,
    quality_gate: dict[str, Any],
    policy: dict[str, Any],
    summary: dict[str, Any],
    actual: dict[str, Any],
    delta: dict[str, Any],
    local_tool_version: str,
) -> dict[str, Any] | None:
    if not isinstance(quality_gate, dict) or not quality_gate:
        return None

    metrics = quality_gate.get("metrics") if isinstance(quality_gate.get("metrics"), dict) else {}
    cohorts = quality_gate.get("cohorts") if isinstance(quality_gate.get("cohorts"), dict) else {}
    applied = cohorts.get("canary_applied") if isinstance(cohorts.get("canary_applied"), dict) else {}
    holdout = cohorts.get("canary_holdout") if isinstance(cohorts.get("canary_holdout"), dict) else {}
    bypassed = cohorts.get("bypassed_or_disabled") if isinstance(cohorts.get("bypassed_or_disabled"), dict) else {}
    matched_count = int(metrics.get("matched_metadata_row_count") or summary.get("actual_matched_metadata_row_count") or 0)
    summary_failure_count = int(metrics.get("summary_failure_count") or summary.get("summary_failure_count") or 0)
    safety_stop_count = int(applied.get("safety_stop_count") or 0) + int(bypassed.get("safety_stop_count") or 0)
    latency = actual.get("latency") if isinstance(actual.get("latency"), dict) else {}
    privacy = quality_gate.get("privacy") if isinstance(quality_gate.get("privacy"), dict) else {}

    return {
        "schema": "agentflow.old_context_summary_quality_gate_feedback.v1",
        "quality_gate_schema": quality_gate.get("schema"),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "policy_source": policy.get("policy_source"),
        "local_tool_version": local_tool_version,
        "verdict": quality_gate.get("verdict"),
        "reason_codes": quality_gate.get("reason_codes") or [],
        "warning_codes": quality_gate.get("warning_codes") or [],
        "thresholds": quality_gate.get("thresholds") if isinstance(quality_gate.get("thresholds"), dict) else {},
        "cohorts": {
            "canary_applied": applied,
            "canary_holdout": holdout,
            "bypassed_or_disabled": bypassed,
        },
        "cohort_counts": {
            "matched": matched_count,
            "canary_applied": int(metrics.get("canary_applied_count") or summary.get("actual_canary_applied_count") or 0),
            "canary_holdout": int(metrics.get("canary_holdout_count") or summary.get("actual_canary_holdout_count") or 0),
            "bypassed_or_disabled": int(metrics.get("bypassed_or_disabled_count") or summary.get("actual_bypassed_or_disabled_count") or 0),
        },
        "aggregate_rates": {
            "error_rate": summary.get("error_rate"),
            "retry_rate": summary.get("retry_rate"),
            "summary_failure_rate": round(summary_failure_count / matched_count, 6) if matched_count else 0.0,
        },
        "aggregate_deltas": {
            "applied_minus_holdout_error_rate": metrics.get("applied_minus_holdout_error_rate"),
            "applied_minus_holdout_retry_rate": metrics.get("applied_minus_holdout_retry_rate"),
            "applied_minus_holdout_latency_avg_ms": metrics.get("applied_minus_holdout_latency_avg_ms"),
            "latency_applied_minus_holdout_avg_ms": latency.get("applied_minus_holdout_avg_ms"),
            "matched_vs_projected_affected_delta": delta.get("matched_vs_projected_affected_delta"),
            "applied_vs_projected_delta": delta.get("applied_vs_projected_delta"),
            "holdout_vs_projected_delta": delta.get("holdout_vs_projected_delta"),
            "bypass_or_disabled_vs_projected_delta": delta.get("bypass_or_disabled_vs_projected_delta"),
            "net_savings_vs_projection_delta_usd": delta.get("net_savings_vs_projection_delta_usd") or summary.get("net_savings_vs_projection_delta_usd"),
        },
        "savings": {
            "net_savings_usd": metrics.get("net_savings_usd") or summary.get("actual_net_savings_usd"),
            "gross_savings_usd": metrics.get("gross_savings_usd") or summary.get("actual_gross_savings_usd"),
            "summary_model_cost_usd": metrics.get("summary_model_cost_usd") or summary.get("actual_summary_model_cost_usd"),
            "payback_ratio": metrics.get("payback_ratio"),
            "projection_realization_ratio": metrics.get("projection_realization_ratio"),
        },
        "safety": {
            "summary_failure_count": summary_failure_count,
            "summary_failure_rate": round(summary_failure_count / matched_count, 6) if matched_count else 0.0,
            "safety_stop_count": safety_stop_count,
            "applied_safety_stop_count": int(applied.get("safety_stop_count") or 0),
            "bypassed_safety_stop_count": int(bypassed.get("safety_stop_count") or 0),
        },
        "privacy": {
            "metadata_only": True,
            "raw_old_context_included": bool(privacy.get("raw_old_context_included", False)),
            "generated_summaries_included": bool(privacy.get("generated_summaries_included", False)),
            "summary_prompts_included": False,
            "raw_messages_included": bool(privacy.get("raw_messages_included", False)),
            "raw_transcripts_included": bool(privacy.get("raw_transcripts_included", False)),
            "provider_bodies_included": bool(privacy.get("provider_bodies_included", False)),
            "file_contents_included": False,
            "request_ids_included": bool(privacy.get("request_ids_included", False)),
            "tenant_ids_included": bool(privacy.get("tenant_ids_included", False)),
            "local_session_ids_included": bool(privacy.get("local_session_ids_included", False)),
            "cache_keys_included": bool(privacy.get("cache_keys_included", False)),
            "raw_payload_strings_included": False,
        },
    }


def _old_context_summary_lifecycle_payload(command: str, result: dict[str, Any]) -> dict[str, Any] | None:
    from agentflow_proxy import __version__

    dry_run = _old_context_summary_lifecycle_result(command, result)
    if not isinstance(dry_run, dict):
        return None
    if command in {"impact", "quality-gate"}:
        dry_run_meta = dry_run.get("dry_run") if isinstance(dry_run.get("dry_run"), dict) else {}
        if command == "quality-gate":
            policy = {
                "rule_id": dry_run.get("rule_id"),
                "candidate_id": dry_run.get("candidate_id"),
                "policy_source": dry_run.get("policy_source"),
                "model": dry_run.get("model"),
                "canary": dry_run.get("canary") if isinstance(dry_run.get("canary"), dict) else {},
                "safety_gates": dry_run.get("safety_gates") if isinstance(dry_run.get("safety_gates"), dict) else {},
            }
            projection = dry_run.get("projection") if isinstance(dry_run.get("projection"), dict) else {}
        else:
            policy = dry_run_meta.get("policy") if isinstance(dry_run_meta.get("policy"), dict) else {}
            projection = dry_run_meta.get("projection") if isinstance(dry_run_meta.get("projection"), dict) else {}
        summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
        actual = dry_run.get("actual") if isinstance(dry_run.get("actual"), dict) else {}
        delta = dry_run.get("delta") if isinstance(dry_run.get("delta"), dict) else {}
        quality_gate = dry_run.get("quality_gate") if isinstance(dry_run.get("quality_gate"), dict) else {}
        if command == "quality-gate" and not quality_gate:
            quality_gate = dry_run
        quality_gate_feedback = _old_context_summary_quality_gate_feedback(
            quality_gate=quality_gate,
            policy=policy,
            summary=summary,
            actual=actual,
            delta=delta,
            local_tool_version=__version__,
        )
        basis = {
            "command": command,
            "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
            "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
        }
        digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
        metadata = {
            "schema": "agentflow.old_context_summary_lifecycle_metadata.v1",
            "lifecycle_kind": "old_context_summarization",
            "command": "old-context-summary-quality-gate" if command == "quality-gate" else "old-context-summary-impact",
            "local_result_status": "ok" if dry_run.get("ok") else "error",
            "dry_run": False,
            "read_only": bool(dry_run.get("read_only", True)),
            "policy_source": policy.get("policy_source"),
            "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
            "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
            "model": policy.get("model"),
            "canary_enabled": ((policy.get("canary") or {}).get("enabled") if isinstance(policy.get("canary"), dict) else None),
            "canary_fraction": ((policy.get("canary") or {}).get("fraction") if isinstance(policy.get("canary"), dict) else None),
            "safety_stop_enabled": ((policy.get("safety_stop") or {}).get("enabled") if isinstance(policy.get("safety_stop"), dict) else None),
            "projected_affected_metadata_row_count": projection.get("projected_affected_metadata_row_count"),
            "projected_canary_applied_count": projection.get("projected_canary_applied_count"),
            "projected_canary_holdout_count": projection.get("projected_canary_holdout_count"),
            "projected_saved_tokens": projection.get("projected_saved_tokens"),
            "projected_net_savings_usd": projection.get("projected_net_savings_usd"),
            "actual_matched_metadata_row_count": summary.get("actual_matched_metadata_row_count"),
            "actual_canary_applied_count": summary.get("actual_canary_applied_count"),
            "actual_canary_holdout_count": summary.get("actual_canary_holdout_count"),
            "actual_bypassed_or_disabled_count": summary.get("actual_bypassed_or_disabled_count"),
            "summary_failure_count": summary.get("summary_failure_count"),
            "error_rate": summary.get("error_rate"),
            "retry_rate": summary.get("retry_rate"),
            "actual_tokens_saved_est": summary.get("actual_tokens_saved_est"),
            "actual_gross_savings_usd": summary.get("actual_gross_savings_usd"),
            "actual_summary_model_cost_usd": summary.get("actual_summary_model_cost_usd"),
            "actual_net_savings_usd": summary.get("actual_net_savings_usd"),
            "net_savings_vs_projection_delta_usd": summary.get("net_savings_vs_projection_delta_usd"),
            "latency": actual.get("latency") if isinstance(actual.get("latency"), dict) else None,
            "status_buckets": actual.get("status_buckets"),
            "summary_decision_status_buckets": actual.get("summary_decision_status_buckets"),
            "summary_reason_buckets": actual.get("summary_reason_buckets"),
            "summary_cache_buckets": actual.get("summary_cache_buckets"),
            "safety_stop_buckets": actual.get("safety_stop_buckets"),
            "delta": delta,
            "old_context_summary_quality_gate": quality_gate_feedback,
            "quality_gate": {
                "schema": quality_gate.get("schema"),
                "verdict": quality_gate.get("verdict"),
                "reason_codes": quality_gate.get("reason_codes"),
                "warning_codes": quality_gate.get("warning_codes"),
                "metrics": quality_gate.get("metrics"),
                "thresholds": quality_gate.get("thresholds"),
            } if quality_gate else None,
            "error_type": (dry_run.get("error") or {}).get("type") if isinstance(dry_run.get("error"), dict) else None,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_old_turns_included": False,
                "raw_summaries_included": False,
                "provider_bodies_included": False,
                "raw_session_ids_included": False,
                "request_ids_included": False,
                "tenant_ids_included": False,
                "cache_keys_included": False,
                "file_paths_included": False,
                "db_path_included": False,
            },
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
        return {
            "event_type": _old_context_summary_lifecycle_event_type(command, dry_run),
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "recommendation_id": f"old-context-summary:{digest}",
            "bundle_hash": None,
            "policy_sections": ["crunch"],
            "validation_warning_count": 0,
            "review_warning_count": 0,
            "applied_files": [],
            "local_tool_version": __version__,
            "metadata": metadata,
        }

    policy = dry_run.get("policy") if isinstance(dry_run.get("policy"), dict) else {}
    summary = dry_run.get("summary") if isinstance(dry_run.get("summary"), dict) else {}
    groups = [item for item in dry_run.get("groups", []) if isinstance(item, dict)]
    eligible_groups = [item for item in groups if item.get("blocker") == "eligible"]
    group_counts: dict[str, int] = {}
    for group in groups:
        blocker = str(group.get("blocker") or "unknown")
        group_counts[blocker] = group_counts.get(blocker, 0) + int(group.get("call_count") or 0)
    basis = {
        "command": command,
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
        "eligible_call_count": summary.get("eligible_call_count"),
        "projected_saved_tokens": summary.get("projected_saved_tokens"),
    }
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    event_type = _old_context_summary_lifecycle_event_type(command, dry_run)
    metadata = {
        "schema": "agentflow.old_context_summary_lifecycle_metadata.v1",
        "lifecycle_kind": "old_context_summarization",
        "command": "policy-review" if command == "review" else "old-context-summary-dry-run",
        "local_result_status": "ok" if dry_run.get("ok") else "error",
        "dry_run": True,
        "read_only": bool(dry_run.get("read_only", True)),
        "policy_source": policy.get("policy_source"),
        "rule_id": _old_context_summary_metadata_identifier(policy.get("rule_id")),
        "candidate_id": _old_context_summary_metadata_identifier(policy.get("candidate_id")),
        "model": policy.get("model"),
        "placement": policy.get("placement"),
        "canary_enabled": ((policy.get("canary") or {}).get("enabled") if isinstance(policy.get("canary"), dict) else None),
        "canary_fraction": ((policy.get("canary") or {}).get("fraction") if isinstance(policy.get("canary"), dict) else None),
        "safety_stop_enabled": ((policy.get("safety_stop") or {}).get("enabled") if isinstance(policy.get("safety_stop"), dict) else None),
        "sampled_call_count": summary.get("sampled_call_count"),
        "sampled_provider_call_count": summary.get("sampled_provider_call_count"),
        "request_body_available_count": summary.get("request_body_available_count"),
        "request_body_replayed_count": summary.get("request_body_replayed_count"),
        "eligible_call_count": summary.get("eligible_call_count"),
        "summary_cache_hit_count": summary.get("summary_cache_hit_count"),
        "eligible_old_turns": summary.get("eligible_old_turns"),
        "eligible_chars": summary.get("eligible_chars"),
        "projected_saved_chars": summary.get("projected_saved_chars"),
        "projected_saved_tokens": summary.get("projected_saved_tokens"),
        "estimated_summary_cost_usd": summary.get("estimated_summary_cost_usd"),
        "projected_gross_savings_usd": summary.get("projected_gross_savings_usd"),
        "projected_net_savings_usd": summary.get("projected_net_savings_usd"),
        "eligible_group_count": len(eligible_groups),
        "blocker_counts": dict(sorted(group_counts.items())),
        "reload_required": dry_run.get("reload_required"),
        "error_type": (dry_run.get("error") or {}).get("type") if isinstance(dry_run.get("error"), dict) else None,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_old_turns_included": False,
            "raw_summaries_included": False,
            "provider_bodies_included": False,
            "raw_session_ids_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "file_paths_included": False,
            "db_path_included": False,
        },
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {
        "event_type": event_type,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": f"old-context-summary:{digest}",
        "bundle_hash": None,
        "policy_sections": ["crunch"],
        "validation_warning_count": 0,
        "review_warning_count": 0,
        "applied_files": [],
        "local_tool_version": __version__,
        "metadata": metadata,
    }


def _attach_old_context_summary_lifecycle_feedback(result: dict[str, Any], *, command: str, db_path: str) -> None:
    from agentflow_proxy import recommendations

    payload = _old_context_summary_lifecycle_payload(command, result)
    if payload is None:
        return
    if not recommendations.recommendations_enabled():
        result["managed_lifecycle_feedback"] = _public_lifecycle_feedback_meta({
            **recommendations.disabled_outcome_feedback_meta(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "disabled",
        })
        return

    store = None
    try:
        store = _open_store_for_db(str(db_path))
        meta = asyncio.run(
            recommendations.queue_policy_event_feedback(
                store,
                payload,
                source_surface=recommendations.OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE,
            )
        )
    except Exception as exc:
        meta = {
            "enabled": True,
            "server_url": recommendations.recommendation_server_url(),
            "endpoint": recommendations.POLICY_EVENTS_PATH,
            "status": "error",
            "reason": "queue-failed",
            "error": repr(exc),
        }
    finally:
        if store is not None:
            store.conn.close()

    public_meta = _public_lifecycle_feedback_meta(meta)
    result["managed_lifecycle_feedback"] = public_meta
    if command in {"dry-run", "impact"} and public_meta.get("status") in {"sent", "retryable-error", "dropped-after-limit", "error"}:
        result["managed_server_calls_made"] = True


def _write_policy_rollback_result(stream: Any, payload: dict[str, Any], *, pretty: bool) -> None:
    if pretty:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        _write_json(stream, payload)


def orchestrator_research_cli(argv: Sequence[str] | None = None, *, stdout: Any = None, stderr: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Build a sanitized data-backed AgentFlow research-mode backlog plan")
    parser.add_argument(
        "--issues-json",
        required=True,
        help="Path to a GitHub issue-list JSON array. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--stats-json",
        help="Optional path to local stats JSON from the dashboard/API.",
    )
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="Recent orchestrator log path to inspect. Can be repeated.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.getenv("AGENTFLOW_RESEARCH_BACKLOG_THRESHOLD", "3")),
        help="Minimum status:ready actionable issue count before research mode is skipped.",
    )
    parser.add_argument(
        "--trusted-author",
        default=os.getenv("AGENTFLOW_GITHUB_TRUSTED_AUTHOR", "lutzkuen"),
        help="Only issues from this GitHub author are considered unattended-actionable.",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=14,
        help="Age in days before a blocked issue is treated as stale for research comments.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr

    from agentflow_proxy.orchestrator_research import build_research_plan, load_json_file, write_json

    try:
        if args.issues_json == "-":
            issues = json.loads(sys.stdin.read())
        else:
            issues = load_json_file(args.issues_json)
        stats = load_json_file(args.stats_json) if args.stats_json else None
    except (OSError, ValueError, TypeError) as exc:
        _write_json(stderr, {"ok": False, "error": {"type": exc.__class__.__name__, "message": str(exc)}})
        return 1

    if not isinstance(issues, list):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_issues_json", "message": "issues JSON must be an array"}})
        return 1
    if stats is not None and not isinstance(stats, dict):
        _write_json(stderr, {"ok": False, "error": {"type": "invalid_stats_json", "message": "stats JSON must be an object"}})
        return 1

    plan = build_research_plan(
        issues=issues,
        stats=stats,
        log_sources=args.log,
        threshold=args.threshold,
        trusted_author=args.trusted_author,
        stale_days=args.stale_days,
    )
    write_json(stdout, plan, pretty=args.pretty)
    return 0


def proxy_main() -> None:
    # The provider proxy forwards real API credentials and request bodies upstream.
    # Keep installed CLI defaults localhost-only unless the user explicitly opts in
    # to a different bind address through AGENTFLOW_HOST or --host.
    os.environ.setdefault("AGENTFLOW_HOST", "127.0.0.1")

    from agentflow_proxy.server import main

    main()


def agentflow_main() -> None:
    raise SystemExit(agentflow_cli())


def policy_reload_main() -> None:
    raise SystemExit(policy_reload_cli())


def policy_export_main() -> None:
    raise SystemExit(policy_export_cli())


def policy_validate_main() -> None:
    raise SystemExit(policy_validate_cli())


def policy_diff_main() -> None:
    raise SystemExit(policy_diff_cli())


def policy_review_main() -> None:
    raise SystemExit(policy_review_cli())


def policy_fetch_review_main() -> None:
    raise SystemExit(policy_fetch_review_cli())


def policy_apply_main() -> None:
    raise SystemExit(policy_apply_cli())


def policy_draft_stage_main() -> None:
    raise SystemExit(policy_draft_stage_cli())


def policy_draft_validate_main() -> None:
    raise SystemExit(policy_draft_validate_cli())


def openai_optimization_draft_dry_run_main() -> None:
    raise SystemExit(openai_optimization_draft_dry_run_cli())


def openai_optimization_draft_apply_main() -> None:
    raise SystemExit(openai_optimization_draft_apply_cli())


def policy_draft_apply_main() -> None:
    raise SystemExit(policy_draft_apply_cli())


def codex_app_policy_dry_run_main() -> None:
    raise SystemExit(codex_app_policy_dry_run_cli())


def managed_rollout_actions_review_main() -> None:
    raise SystemExit(managed_rollout_actions_review_cli())


def optimization_rollout_actions_review_main() -> None:
    raise SystemExit(optimization_rollout_actions_review_cli())


def optimization_rollout_actions_apply_main() -> None:
    raise SystemExit(optimization_rollout_actions_apply_cli())


def scaffold_rollout_actions_review_main() -> None:
    raise SystemExit(scaffold_rollout_actions_review_cli())


def scaffold_rollout_actions_apply_main() -> None:
    raise SystemExit(scaffold_rollout_actions_apply_cli())


def managed_rollout_actions_apply_main() -> None:
    raise SystemExit(managed_rollout_actions_apply_cli())


def managed_rollout_actions_dry_run_main() -> None:
    raise SystemExit(managed_rollout_actions_dry_run_cli())


def old_context_summary_dry_run_main() -> None:
    raise SystemExit(old_context_summary_dry_run_cli())


def old_context_summary_impact_main() -> None:
    raise SystemExit(old_context_summary_impact_cli())


def old_context_summary_quality_gate_main() -> None:
    raise SystemExit(old_context_summary_quality_gate_cli())


def old_context_summary_rollout_actions_review_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_review_cli())


def old_context_summary_rollout_actions_apply_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_apply_cli())


def old_context_summary_rollout_actions_dry_run_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_dry_run_cli())


def old_context_summary_rollout_actions_impact_main() -> None:
    raise SystemExit(old_context_summary_rollout_actions_impact_cli())


def managed_rollout_actions_impact_main() -> None:
    raise SystemExit(managed_rollout_actions_impact_cli())


def policy_rollback_main() -> None:
    raise SystemExit(policy_rollback_cli())


def codex_diagnose_main() -> None:
    raise SystemExit(codex_diagnose_cli())


def codex_canary_impact_main() -> None:
    raise SystemExit(codex_canary_impact_cli())


def openai_scoreboard_main() -> None:
    raise SystemExit(openai_scoreboard_cli())


def openai_routing_report_main() -> None:
    raise SystemExit(openai_routing_report_cli())


def openai_old_context_summary_report_main() -> None:
    raise SystemExit(openai_old_context_summary_report_cli())


def openai_cache_replay_report_main() -> None:
    raise SystemExit(openai_cache_replay_report_cli())


def optimization_action_ledger_main() -> None:
    raise SystemExit(optimization_action_ledger_cli())


def optimization_coordinator_dry_run_main() -> None:
    raise SystemExit(optimization_coordinator_dry_run_cli())


def provider_tool_adoption_report_main() -> None:
    raise SystemExit(provider_tool_adoption_report_cli())


def repeated_scaffold_opportunity_main() -> None:
    raise SystemExit(repeated_scaffold_opportunity_cli())


def instruction_dedup_opportunity_main() -> None:
    raise SystemExit(instruction_dedup_opportunity_cli())


def instruction_dedup_dry_run_main() -> None:
    raise SystemExit(instruction_dedup_dry_run_cli())


def instruction_dedup_impact_main() -> None:
    raise SystemExit(instruction_dedup_impact_cli())


def terminal_output_compaction_opportunity_main() -> None:
    raise SystemExit(terminal_output_compaction_opportunity_cli())


def codex_terminal_transcript_opportunity_main() -> None:
    raise SystemExit(codex_terminal_transcript_opportunity_cli())


def anthropic_thinking_compaction_opportunity_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_opportunity_cli())


def anthropic_thinking_compaction_impact_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_impact_cli())


def anthropic_thinking_compaction_dry_run_main() -> None:
    raise SystemExit(anthropic_thinking_compaction_dry_run_cli())


def codex_terminal_transcript_dry_run_main() -> None:
    raise SystemExit(codex_terminal_transcript_dry_run_cli())


def codex_terminal_transcript_impact_main() -> None:
    raise SystemExit(codex_terminal_transcript_impact_cli())


def terminal_output_compaction_dry_run_main() -> None:
    raise SystemExit(terminal_output_compaction_dry_run_cli())


def terminal_output_compaction_impact_main() -> None:
    raise SystemExit(terminal_output_compaction_impact_cli())


def repeated_scaffold_impact_main() -> None:
    raise SystemExit(repeated_scaffold_impact_cli())


def repeated_scaffold_activation_main() -> None:
    raise SystemExit(repeated_scaffold_activation_cli())


def openai_cache_replay_impact_main() -> None:
    raise SystemExit(openai_cache_replay_impact_cli())


def openai_cache_replay_readiness_main() -> None:
    raise SystemExit(openai_cache_replay_readiness_cli())


def openai_cache_replay_apply_main() -> None:
    raise SystemExit(openai_cache_replay_apply_cli())


def openai_cache_replay_dry_run_main() -> None:
    raise SystemExit(openai_cache_replay_dry_run_cli())


def openai_old_context_summary_dry_run_main() -> None:
    raise SystemExit(openai_old_context_summary_dry_run_cli())


def openai_canary_impact_main() -> None:
    raise SystemExit(openai_canary_impact_cli())


def claude_canary_impact_main() -> None:
    raise SystemExit(claude_canary_impact_cli())


def claude_canary_actions_main() -> None:
    raise SystemExit(claude_canary_actions_cli())


def claude_canary_actions_apply_main() -> None:
    raise SystemExit(claude_canary_actions_apply_cli())


def routing_canary_promote_main() -> None:
    raise SystemExit(routing_canary_promote_cli())


def routing_experiment_report_main() -> None:
    raise SystemExit(routing_experiment_report_cli())


def routing_promotion_draft_stage_main() -> None:
    raise SystemExit(routing_promotion_draft_stage_cli())


def phase_routing_report_main() -> None:
    raise SystemExit(phase_routing_report_cli())


def session_phase_memory_main() -> None:
    raise SystemExit(session_phase_memory_cli())


def cache_replayability_report_main() -> None:
    raise SystemExit(cache_replayability_report_cli())


def cache_replay_cohorts_main() -> None:
    raise SystemExit(cache_replay_cohorts_cli())


def cache_smoke_diagnostic_main() -> None:
    raise SystemExit(cache_smoke_diagnostic_cli())


def cache_replay_dry_run_main() -> None:
    raise SystemExit(cache_replay_dry_run_cli())


def managed_pattern_rollups_main() -> None:
    raise SystemExit(managed_pattern_rollups_cli())


def optimization_eval_plan_main() -> None:
    raise SystemExit(optimization_eval_plan_cli())


def optimization_shadow_eval_main() -> None:
    raise SystemExit(optimization_shadow_eval_cli())


def optimization_eval_queue_main() -> None:
    raise SystemExit(optimization_eval_queue_cli())


def optimization_promotion_report_main() -> None:
    raise SystemExit(optimization_promotion_report_cli())


def optimization_promotion_actions_main() -> None:
    raise SystemExit(optimization_promotion_actions_cli())


def optimization_promotion_canary_apply_main() -> None:
    raise SystemExit(optimization_promotion_canary_apply_cli())


def optimization_promotion_impact_main() -> None:
    raise SystemExit(optimization_promotion_impact_cli())


def managed_feedback_status_main() -> None:
    raise SystemExit(managed_feedback_status_cli())


def managed_feedback_flush_main() -> None:
    raise SystemExit(managed_feedback_flush_cli())


def sqlite_maintenance_main() -> None:
    raise SystemExit(sqlite_maintenance_cli())


def orchestrator_research_main() -> None:
    raise SystemExit(orchestrator_research_cli())
