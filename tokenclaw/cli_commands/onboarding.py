from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import httpx

from tokenclaw.cli_common import (
    default_config_dir,
    default_db_path,
    default_stats_url,
    is_loopback_url as _is_loopback_url,
    open_store_for_db as _open_store_for_db,
    write_json as _write_json,
)
from tokenclaw.upstream_url import redact_url as _redact_url


ONBOARDING_TARGETS = ("openai", "claude", "codex", "claude-vscode", "claude-desktop")
UNSUPPORTED_ONBOARDING_TARGETS = ("copilot",)
RUN_TARGETS = ("openai", "claude")
DEFAULT_STATS_URL = "http://127.0.0.1:4002/tokenclaw/stats"


def _write_activation_summary(stdout: Any, result: dict[str, Any], *, brand: str = "TokenClaw") -> None:
    if result["target"] == "claude-vscode":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} {brand} target: claude-vscode\n")
        stdout.write(f"Claude VS Code local {brand} base URL: {result['local_base_url']}\n")
        stdout.write(f"Upstream Anthropic base URL used by {brand}: {_redact_url(result['upstream_base_url'])}\n")
        stdout.write(f"{brand}-managed non-secret env file: {result['env_file_path']}\n")
        stdout.write(f"Env file changed: {str(result['env_file_changed']).lower()}\n")
        stdout.write(f"Shell profile: {result.get('shell_profile_path') or 'skipped'}\n")
        stdout.write(f"Shell profile changed: {str(result.get('shell_profile_changed')).lower()}\n")
        if result.get("dry_run") and result.get("shell_profile_append"):
            stdout.write("Shell profile append:\n")
            stdout.write(result["shell_profile_append"])
        stdout.write(f"Depends on {brand} target: {result['depends_on']}\n")
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
        stdout.write(f"{brand} does not store or print ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or token values.\n")
        return

    if result["target"] == "claude-desktop":
        prefix = "Dry run: would configure" if result["dry_run"] else "Configured"
        stdout.write(f"{prefix} {brand} target: claude-desktop\n")
        stdout.write(f"Claude Desktop file: {result['desktop_file_path']}\n")
        stdout.write(f"Desktop file changed: {str(result['desktop_file_changed']).lower()}\n")
        if result.get("desktop_file_backup_path"):
            stdout.write(f"Backup: {result['desktop_file_backup_path']}\n")
        stdout.write(f"Depends on {brand} target: {result['depends_on']}\n")
        if result.get("claude_target_created"):
            stdout.write("Claude target was not configured; created the default Claude activation profile.\n")
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


def _write_activation_config_error(stderr: Any, exc: Exception, *, command: str) -> None:
    stderr.write(str(exc) + "\n")
    if command == "activate":
        stderr.write(
            "Activation did not overwrite this file automatically. Move it aside, fix the JSON, "
            "or pass --config-dir to write an isolated TokenClaw config.\n"
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
    doctor_parser.add_argument("target", nargs="?", choices=ONBOARDING_TARGETS)
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

    args = parser.parse_args(argv)
    if not hasattr(args, "config_dir") or args.config_dir is None:
        args.config_dir = default_config_dir()

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
        if args.dry_run:
            stdout.write(activation.shell_command_for_profile(profile, redact=True) + "\n")
            return 0
        from tokenclaw import server

        server.main(proxy_args)
        return 0

    if args.command == "doctor":
        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            result = _activation_config_error_result("agentflow.activation_doctor.v1", args.config_dir, args.target, exc)
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
            result = _activation_config_error_result("agentflow.activation_stats.v1", args.config_dir, args.target, exc)
        else:
            result = _activation_stats_result(config, config_dir=args.config_dir, target=args.target)
        if args.json:
            _write_json(stdout, result)
        elif result["ok"]:
            _write_activation_stats_summary(stdout, result)
        else:
            _write_json(stderr, result)
        return 0 if result["ok"] else 1

    if args.command == "savings":
        from tokenclaw.savings_report import build_savings_report

        try:
            config = activation.load_activation_config(args.config_dir)
        except activation.ActivationConfigError as exc:
            result = _activation_config_error_result("agentflow.savings_report.v1", args.config_dir, None, exc)
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
            _write_savings_report_summary(stdout, result)
        return 0 if result.get("ok") else 1

    if args.command == "demo":
        if args.demo_command == "rule-drill":
            from tokenclaw.local_savings_rule_drill import build_local_savings_rule_drill_summary

            result = build_local_savings_rule_drill_summary()
            if args.json:
                _write_json(stdout, result)
            else:
                stdout.write(
                    f"{brand} local savings rule drill: "
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
            heading = f"{brand} savings demo" if args.demo_command == "savings" else f"{brand} golden path"
            stdout.write(
                f"{heading}: "
                f"{result.get('decision_status')} "
                f"{result.get('local_action_family')} "
                f"tokenclaw_saved=${float(result.get('estimated_agentflow_savings_usd') or 0.0):.6f} "
                f"provider_prompt_cache_discount=${float(result.get('provider_prompt_cache_discount_usd') or 0.0):.6f} "
                f"managed_server_required={str(bool(result.get('managed_server_required'))).lower()}\n"
            )
        return 0 if result.get("ok") else 1

    if args.command == "version":
        result = {
            "schema": "agentflow.version.v1",
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


def agentflow_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    return _onboarding_cli(argv, stdout=stdout, stderr=stderr, command_name="agentflow", brand="TokenClaw")


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
        for key in ("codex_config_path", "env_file_path", "desktop_file_path", "shell_profile_path", "depends_on"):
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

    targets = {
        name: _target_activation_base(config, config_dir=config_dir, target=name)
        for name in _selected_activation_targets(target)
    }
    return {
        "schema": "agentflow.activation_stats.v1",
        "ok": True,
        "target": target,
        "config_path": str(activation.activation_config_path(config_dir)),
        "targets": targets,
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
        "schema": "agentflow.activation_doctor.v1",
        "ok": all(bool(item.get("ok")) for item in targets.values()),
        "target": target,
        "config_path": str(activation.activation_config_path(config_dir)),
        "targets": targets,
        "activation_successor_queue_health": _activation_successor_queue_health(),
    }


def _activation_successor_queue_health() -> dict[str, Any]:
    try:
        from tokenclaw.stats import build_activation_successor_queue_health

        return build_activation_successor_queue_health(limit=5)
    except Exception as exc:
        return {
            "schema": "agentflow.activation_successor_queue_health.v1",
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

    current_shell_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    result["current_shell_base_url"] = _redact_url(current_shell_base_url) if current_shell_base_url else None
    if current_shell_base_url and current_shell_base_url != expected:
        result["status"] = "not routed via tokenclaw"
        result["reasons"].append("current-shell-anthropic-base-url-mismatch")
        result["reasons"].append("vscode-runtime-env-uncertain")
        return result
    if current_shell_base_url == expected:
        result["status"] = "healthy"
        result["reasons"].append("current-shell-routed")
    else:
        result["status"] = "configured"
        result["reasons"].append("shell-env-missing")
    result["reasons"].append("vscode-runtime-env-uncertain")
    result["ok"] = True
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
    result["status"] = "healthy"
    result["ok"] = True
    return result


def _write_activation_stats_summary(stdout: Any, result: dict[str, Any]) -> None:
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
        if target.get("reasons"):
            parts.append("reasons: " + ", ".join(str(reason) for reason in target["reasons"]))
        stdout.write(", ".join(parts) + "\n")


def _write_savings_report_summary(stdout: Any, result: dict[str, Any]) -> None:
    opportunities = result.get("opportunities") if isinstance(result.get("opportunities"), list) else []
    count = len(opportunities)
    stdout.write(f"TokenClaw savings report: {count} opportunit{'y' if count == 1 else 'ies'}\n")
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
