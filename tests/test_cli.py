import asyncio
import importlib
import io
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import yaml

from agentflow_proxy import activation
from agentflow_proxy import cli
from agentflow_proxy.cli_commands import onboarding as onboarding_cli
from agentflow_proxy.cli_commands import optimization_reports as optimization_reports_cli
from agentflow_proxy.cli_commands import policy_bundle as policy_bundle_cli
from agentflow_proxy.cli_commands import policy_workbench as policy_workbench_cli


class ManagedFeedbackFlushClient:
    calls = []
    status_code = 200
    text = '{"ok":true}'

    def __init__(self, *, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def patch(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)


class AgentflowActivationCliTests(unittest.TestCase):
    def test_public_agentflow_help_lists_onboarding_commands_and_targets(self):
        stdout = io.StringIO()

        with patch("sys.stdout", stdout), self.assertRaises(SystemExit) as raised:
            cli.agentflow_cli(["--help"])

        self.assertEqual(raised.exception.code, 0)
        output = stdout.getvalue()
        for expected in ("activate", "stats", "doctor", "run", "version"):
            self.assertIn(expected, output)
        for target in ("openai", "claude", "codex", "claude-vscode"):
            self.assertIn(target, output)
        self.assertIn("127.0.0.1", output)

    def test_public_agentflow_subcommand_help_works(self):
        commands = [
            ("activate", "--help"),
            ("stats", "--help"),
            ("doctor", "--help"),
            ("run", "--help"),
        ]
        for command in commands:
            with self.subTest(command=command):
                stdout = io.StringIO()
                with patch("sys.stdout", stdout), self.assertRaises(SystemExit) as raised:
                    cli.agentflow_cli(list(command))
                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", stdout.getvalue())

    def test_agentflow_cli_reexports_onboarding_command_group(self):
        self.assertIs(cli.agentflow_cli, onboarding_cli.agentflow_cli)
        self.assertIs(cli._activation_stats_result, onboarding_cli._activation_stats_result)
        self.assertIs(cli._doctor_codex_target, onboarding_cli._doctor_codex_target)

    def test_agentflow_cli_reexports_optimization_report_commands(self):
        moved_commands = [
            "openai_optimization_draft_dry_run_cli",
            "openai_optimization_draft_apply_cli",
            "managed_rollout_actions_review_cli",
            "optimization_rollout_actions_review_cli",
            "optimization_rollout_actions_apply_cli",
            "openai_routing_report_cli",
            "anthropic_routing_lifecycle_report_cli",
            "openai_cache_replay_report_cli",
            "optimization_eval_plan_cli",
            "optimization_promotion_report_cli",
            "optimization_promotion_blocker_review_cli",
            "repeated_scaffold_opportunity_cli",
            "instruction_dedup_opportunity_cli",
            "terminal_output_compaction_opportunity_cli",
        ]

        for command in moved_commands:
            with self.subTest(command=command):
                self.assertIs(getattr(cli, command), getattr(optimization_reports_cli, command))

    def test_optimization_report_console_script_entrypoints_import(self):
        script_lines = Path("pyproject.toml").read_text(encoding="utf-8").splitlines()
        affected_tokens = ("agentflow-openai-", "agentflow-optimization-")
        report_tokens = ("opportunity", "compaction", "report", "rollout-actions")
        checked = []

        for line in script_lines:
            stripped = line.strip()
            if not stripped.startswith("agentflow-") or "=" not in stripped:
                continue
            script_name, target = [part.strip().strip('"') for part in stripped.split("=", 1)]
            if not (
                script_name.startswith(affected_tokens)
                or any(token in script_name for token in report_tokens)
            ):
                continue
            module_name, attr = target.split(":", 1)
            module = importlib.import_module(module_name)
            self.assertTrue(callable(getattr(module, attr)), script_name)
            checked.append(script_name)

        self.assertIn("agentflow-openai-routing-report", checked)
        self.assertIn("agentflow-optimization-eval-plan", checked)
        self.assertIn("agentflow-repeated-scaffold-opportunity", checked)

    def test_optimization_reports_module_openai_routing_report_empty_db(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            stdout = io.StringIO()

            code = optimization_reports_cli.openai_routing_report_cli(
                ["--db", db_path, "--limit", "10"],
                stdout=stdout,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(payload["summary"]["candidate_count"], 0)

    def test_onboarding_module_version_json_path(self):
        from agentflow_proxy import __version__

        stdout = io.StringIO()

        code = onboarding_cli.agentflow_cli(["version", "--json"], stdout=stdout)

        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "schema": "agentflow.version.v1",
                "ok": True,
                "version": __version__,
                "package": "agentflow-proxy",
                "command": "agentflow",
            },
        )

    def test_readme_onboarding_happy_path_stays_primary_and_private(self):
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertLess(readme.index("## Quick start"), readme.index("## Manual proxy fallback"))
        self.assertLess(readme.index("agentflow activate openai"), readme.index("agentflow-proxy --provider"))
        for expected in (
            "agentflow activate openai",
            "agentflow activate claude",
            "agentflow activate codex",
            "agentflow activate claude-vscode",
            "agentflow stats",
            "agentflow doctor",
            "openai_base_url = \"http://127.0.0.1:4003/v1\"",
            "ANTHROPIC_BASE_URL=http://127.0.0.1:4000",
            "agentflow_server",
            "not a provider proxy",
            "GitHub Copilot non-goal",
            "unsupported: GitHub Copilot is not a base-url target",
        ):
            self.assertIn(expected, readme)
        self.assertNotIn("sk-", readme)

    def test_readme_onboarding_commands_smoke_without_credentials(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "codex-config.toml"
            smoke_commands = [
                ["activate", "openai", "--config-dir", str(config_dir), "--dry-run"],
                ["activate", "claude", "--config-dir", str(config_dir), "--dry-run"],
                [
                    "activate",
                    "codex",
                    "--config-dir",
                    str(config_dir),
                    "--codex-config",
                    str(codex_config),
                    "--dry-run",
                ],
                ["activate", "claude-vscode", "--config-dir", str(config_dir), "--dry-run"],
                ["stats", "--config-dir", str(config_dir)],
                ["stats", "--config-dir", str(config_dir), "--json"],
            ]

            for command in smoke_commands:
                with self.subTest(command=command):
                    code = cli.agentflow_cli(command, stdout=io.StringIO(), stderr=io.StringIO())
                    self.assertEqual(code, 0)

            cli.agentflow_cli(["activate", "openai", "--config-dir", str(config_dir)], stdout=io.StringIO())
            cli.agentflow_cli(["activate", "claude", "--config-dir", str(config_dir)], stdout=io.StringIO())
            run_stdout = io.StringIO()
            self.assertEqual(
                cli.agentflow_cli(["run", "openai", "--config-dir", str(config_dir), "--dry-run"], stdout=run_stdout),
                0,
            )
            self.assertIn("--provider openai", run_stdout.getvalue())

            with patch("agentflow_proxy.cli.httpx.get") as http_get:
                http_get.return_value = httpx.Response(
                    200,
                    json={"ok": True, "provider": "openai", "upstream": "https://api.openai.com"},
                )
                doctor_stdout = io.StringIO()
                code = cli.agentflow_cli(["doctor", "openai", "--config-dir", str(config_dir)], stdout=doctor_stdout)
            self.assertEqual(code, 0)
            self.assertIn("openai: healthy", doctor_stdout.getvalue())

    def test_activate_copilot_fails_as_unsupported_base_url_target(self):
        stderr = io.StringIO()

        code = cli.agentflow_cli(["activate", "copilot"], stdout=io.StringIO(), stderr=stderr)

        self.assertEqual(code, 2)
        self.assertIn("unsupported: GitHub Copilot is not a base-url target", stderr.getvalue())

    def test_agentflow_version_command_prints_version(self):
        from agentflow_proxy import __version__

        stdout = io.StringIO()

        code = cli.agentflow_cli(["version"], stdout=stdout)

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), f"agentflow {__version__}\n")

    def test_agentflow_stats_prints_activation_targets_without_http(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO())

            with patch("agentflow_proxy.cli.httpx.get") as http_get:
                code = cli.agentflow_cli(["stats", "--config-dir", tmp, "openai"], stdout=stdout)

        self.assertEqual(code, 0)
        http_get.assert_not_called()
        output = stdout.getvalue()
        self.assertEqual(output, "openai: configured, base url: http://127.0.0.1:4003/v1, upstream: https://api.openai.com\n")

    def test_agentflow_stats_lists_all_known_targets(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO())

            code = cli.agentflow_cli(["stats", "--config-dir", tmp], stdout=stdout)

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("openai: configured, base url: http://127.0.0.1:4003/v1", output)
        self.assertIn("claude: not configured", output)
        self.assertIn("codex: not configured", output)
        self.assertIn("claude-vscode: not configured", output)

    def test_agentflow_doctor_codex_not_configured(self):
        stdout = io.StringIO()

        code = cli.agentflow_cli(["doctor", "codex"], stdout=stdout)

        self.assertEqual(code, 1)
        self.assertIn("codex: not configured", stdout.getvalue())

    def test_unsupported_subcommand_has_stable_argparse_exit(self):
        stderr = io.StringIO()

        with patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
            cli.agentflow_cli(["nope"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_pyproject_scripts_keep_public_and_legacy_entry_points(self):
        raw = Path("pyproject.toml").read_text(encoding="utf-8").splitlines()
        scripts: dict[str, str] = {}
        in_scripts = False
        for line in raw:
            stripped = line.strip()
            if stripped == "[project.scripts]":
                in_scripts = True
                continue
            if in_scripts and stripped.startswith("["):
                break
            if in_scripts and "=" in stripped:
                name, value = stripped.split("=", 1)
                scripts[name.strip()] = value.strip().strip('"')

        self.assertEqual(scripts["agentflow"], "agentflow_proxy.cli:agentflow_main")
        self.assertEqual(scripts["agentflow-proxy"], "agentflow_proxy.cli:proxy_main")
        self.assertEqual(scripts["agentflow-claude-proxy"], "agentflow_proxy.cli:proxy_main")
        self.assertEqual(scripts["agentflow-dashboard"], "agentflow_proxy.dashboard:main")
        self.assertGreater(len(scripts), 80)

        for name, target in scripts.items():
            with self.subTest(script=name, target=target):
                module_name, attr_name = target.split(":", 1)
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, attr_name)))

        cli_exceptions = {
            "agentflow-codex-app-proxy": "agentflow_proxy.codex_app_proxy:main",
            "agentflow-codex-app-client": "agentflow_proxy.codex_app_client:main",
            "agentflow-dashboard": "agentflow_proxy.dashboard:main",
        }
        for name, target in scripts.items():
            if name in cli_exceptions:
                self.assertEqual(target, cli_exceptions[name])
            else:
                self.assertTrue(target.startswith("agentflow_proxy.cli:"), f"{name} moved away from cli.py")

    def test_activate_openai_writes_default_profile_idempotently(self):
        with TemporaryDirectory() as tmp:
            stdout_one = io.StringIO()
            stdout_two = io.StringIO()

            code_one = cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=stdout_one)
            code_two = cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=stdout_two)

            self.assertEqual(code_one, 0)
            self.assertEqual(code_two, 0)
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["targets"].keys()), ["openai"])
            profile = config["targets"]["openai"]
            self.assertTrue(profile["configured"])
            self.assertEqual(profile["local_base_url"], "http://127.0.0.1:4003/v1")
            self.assertEqual(profile["health_url"], "http://127.0.0.1:4003/health")
            self.assertEqual(profile["upstream_base_url"], "https://api.openai.com")
            self.assertEqual(profile["openai_auth_mode"], "client")
            self.assertIn("last_activation_at", profile)
            self.assertEqual(profile["config_file_paths"], [str(Path(tmp) / "activation.json")])
            self.assertIn("--provider", profile["command_profile"]["argv"])
            self.assertIn("openai", profile["command_profile"]["argv"])
            self.assertIn("Local base URL for clients: http://127.0.0.1:4003/v1", stdout_one.getvalue())
            self.assertIn("Upstream provider base URL: https://api.openai.com", stdout_one.getvalue())
            self.assertIn("Run configured proxy: agentflow run openai", stdout_one.getvalue())

    def test_activate_openai_custom_upstream_preserves_local_base_url(self):
        upstream = "https://example-resource.openai.azure.com/openai/deployments/my-deployment?api-version=2024-10-21"
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(
                [
                    "activate",
                    "openai",
                    "--config-dir",
                    tmp,
                    "--openai-base-url",
                    upstream,
                    "--openai-auth-mode",
                    "proxy",
                ],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            profile = config["targets"]["openai"]
            self.assertEqual(profile["local_base_url"], "http://127.0.0.1:4003/v1")
            self.assertEqual(profile["upstream_base_url"], upstream)
            self.assertEqual(profile["openai_auth_mode"], "proxy")
            self.assertIn(upstream, stdout.getvalue())
            self.assertIn("Local base URL for clients: http://127.0.0.1:4003/v1", stdout.getvalue())

    def test_activate_openai_redacts_sensitive_custom_upstream_in_output(self):
        upstream = "https://user:pass@example-resource.openai.azure.com/openai/deployments/my-deployment?api-key=secret&api-version=2024-10-21"
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(
                [
                    "activate",
                    "openai",
                    "--config-dir",
                    tmp,
                    "--openai-base-url",
                    upstream,
                ],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            raw_config = (Path(tmp) / "activation.json").read_text(encoding="utf-8")
            self.assertNotIn("user:pass", raw_config)
            self.assertNotIn("api-key=secret", raw_config)
            self.assertIn("api-key=%5Bredacted%5D", raw_config)
            output = stdout.getvalue()
            self.assertNotIn("user:pass", output)
            self.assertNotIn("api-key=secret", output)
            self.assertIn("https://example-resource.openai.azure.com/openai/deployments/my-deployment", output)
            self.assertIn("api-key=%5Bredacted%5D", output)

    def test_missing_activation_config_returns_default_status(self):
        with TemporaryDirectory() as tmp:
            status = activation.activation_status(tmp)

        self.assertEqual(status["schema"], "agentflow.activation_status.v1")
        self.assertTrue(status["ok"])
        self.assertFalse(status["targets"]["openai"]["configured"])
        self.assertFalse(status["targets"]["claude"]["configured"])
        self.assertFalse(status["targets"]["codex"]["configured"])
        self.assertFalse(status["targets"]["claude-vscode"]["configured"])

    def test_activation_config_round_trips_with_atomic_write(self):
        with TemporaryDirectory() as tmp:
            profile = activation.activation_profile("openai")
            config = activation.apply_activation_profile(activation.empty_config(), profile, config_dir=tmp)
            path = activation.write_activation_config(config, tmp)
            loaded = activation.load_activation_config(tmp)

            self.assertEqual(loaded, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(loaded["targets"]["openai"]["config_file_paths"], [str(Path(tmp) / "activation.json")])
            self.assertFalse(list(Path(tmp).glob(".activation.json.*.tmp")))

    def test_doctor_reports_invalid_activation_config_schema(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "activation.json").write_text('{"schema":"bad","targets":{}}\n', encoding="utf-8")
            stderr = io.StringIO()

            code = cli.agentflow_cli(["doctor", "openai", "--config-dir", tmp], stdout=io.StringIO(), stderr=stderr)

        self.assertEqual(code, 1)
        self.assertIn("Invalid AgentFlow activation config", stderr.getvalue())
        self.assertIn("$.schema", stderr.getvalue())

    def test_doctor_json_reports_invalid_activation_config_schema(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "activation.json").write_text('{"schema":"bad","targets":{}}\n', encoding="utf-8")
            stdout = io.StringIO()

            code = cli.agentflow_cli(["doctor", "openai", "--config-dir", tmp, "--json"], stdout=stdout)

        self.assertEqual(code, 1)
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["ok"])
        self.assertEqual(result["issues"][0]["code"], "activation-config-invalid")
        self.assertEqual(result["issues"][0]["errors"][0]["path"], "$.schema")

    def test_activate_refuses_to_overwrite_invalid_activation_config(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "activation.json"
            path.write_text('{"schema":"bad","targets":{}}\n', encoding="utf-8")
            stderr = io.StringIO()

            code = cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO(), stderr=stderr)

            self.assertEqual(code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"schema":"bad","targets":{}}\n')
            self.assertIn("Activation did not overwrite this file automatically", stderr.getvalue())

    def test_stats_json_includes_activation_status_snapshot(self):
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO())
            stdout = io.StringIO()

            with patch("agentflow_proxy.cli.httpx.get") as http_get:
                code = cli.agentflow_cli(["stats", "--config-dir", tmp, "openai", "--json"], stdout=stdout)

        self.assertEqual(code, 0)
        http_get.assert_not_called()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.activation_stats.v1")
        status = result["targets"]["openai"]
        self.assertEqual(status["status"], "configured")
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "openai")
        self.assertEqual(status["local_base_url"], "http://127.0.0.1:4003/v1")
        self.assertEqual(status["upstream_base_url"], "https://api.openai.com")
        for field in ("status", "configured", "local_base_url", "health_url", "upstream_base_url", "reasons"):
            self.assertIn(field, status)

    def test_activate_openai_invalid_upstream_fails_actionably(self):
        with TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            code = cli.agentflow_cli(
                ["activate", "openai", "--config-dir", tmp, "--openai-base-url", "api.openai.com"],
                stdout=io.StringIO(),
                stderr=stderr,
            )

            self.assertEqual(code, 2)
            self.assertFalse((Path(tmp) / "activation.json").exists())
            self.assertIn("must start with http:// or https://", stderr.getvalue())

    def test_activate_claude_writes_default_profile_idempotently(self):
        with TemporaryDirectory() as tmp:
            stdout_one = io.StringIO()
            stdout_two = io.StringIO()

            code_one = cli.agentflow_cli(["activate", "claude", "--config-dir", tmp], stdout=stdout_one)
            code_two = cli.agentflow_cli(["activate", "claude", "--config-dir", tmp], stdout=stdout_two)

            self.assertEqual(code_one, 0)
            self.assertEqual(code_two, 0)
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["targets"].keys()), ["claude"])
            profile = config["targets"]["claude"]
            self.assertTrue(profile["configured"])
            self.assertEqual(profile["provider"], "anthropic")
            self.assertEqual(profile["local_base_url"], "http://127.0.0.1:4000")
            self.assertEqual(profile["health_url"], "http://127.0.0.1:4000/health")
            self.assertEqual(profile["upstream_base_url"], "https://api.anthropic.com")
            self.assertIn("Local base URL for clients: http://127.0.0.1:4000", stdout_one.getvalue())
            self.assertIn("Upstream provider base URL: https://api.anthropic.com", stdout_one.getvalue())
            self.assertIn("Run configured proxy: agentflow run claude", stdout_one.getvalue())

    def test_activate_claude_vscode_writes_env_file_and_default_claude_dependency(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 0)
            config_path = Path(tmp) / "activation.json"
            env_path = Path(tmp) / "claude-vscode.env"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["targets"].keys()), ["claude", "claude-vscode"])
            self.assertEqual(config["targets"]["claude-vscode"]["depends_on"], "claude")
            self.assertEqual(config["targets"]["claude-vscode"]["local_base_url"], "http://127.0.0.1:4000")
            self.assertEqual(config["targets"]["claude-vscode"]["upstream_base_url"], "https://api.anthropic.com")
            self.assertEqual(config["targets"]["claude-vscode"]["env_file_path"], str(env_path))
            self.assertEqual(config["targets"]["claude-vscode"]["safe_env"], {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000"})
            self.assertEqual(
                env_path.read_text(encoding="utf-8"),
                "# AgentFlow-managed non-secret routing values for Claude in VS Code.\n"
                "# Keep Claude API keys in your shell or OS secret manager, not in this file.\n"
                "ANTHROPIC_BASE_URL=http://127.0.0.1:4000\n",
            )
            output = stdout.getvalue()
            self.assertIn("Claude VS Code local AgentFlow base URL: http://127.0.0.1:4000", output)
            self.assertIn("Upstream Anthropic base URL used by AgentFlow: https://api.anthropic.com", output)
            self.assertIn("Claude target was not configured; created the default Claude activation profile.", output)
            self.assertIn("export ANTHROPIC_BASE_URL=http://127.0.0.1:4000\ncode .", output)
            self.assertIn("VS Code extensions usually inherit environment variables only from the VS Code process", output)

    def test_activate_claude_vscode_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            first = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=io.StringIO())
            second_stdout = io.StringIO()
            second = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=second_stdout)

            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            self.assertIn("Env file changed: false", second_stdout.getvalue())
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["targets"].keys()), ["claude", "claude-vscode"])

    def test_activate_claude_vscode_uses_existing_claude_local_base_url(self):
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(
                ["activate", "claude", "--config-dir", tmp, "--local-base-url", "http://127.0.0.1:4998"],
                stdout=io.StringIO(),
            )
            stdout = io.StringIO()

            code = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 0)
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(config["targets"]["claude-vscode"]["local_base_url"], "http://127.0.0.1:4998")
            self.assertEqual((Path(tmp) / "claude-vscode.env").read_text(encoding="utf-8").splitlines()[-1], "ANTHROPIC_BASE_URL=http://127.0.0.1:4998")
            self.assertIn("Claude VS Code local AgentFlow base URL: http://127.0.0.1:4998", stdout.getvalue())
            self.assertIn("Upstream Anthropic base URL used by AgentFlow: https://api.anthropic.com", stdout.getvalue())

    def test_activate_claude_vscode_alias_claude_code(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(["activate", "claude-code", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 0)
            config = json.loads((Path(tmp) / "activation.json").read_text(encoding="utf-8"))
            self.assertIn("claude-vscode", config["targets"])
            self.assertIn("Configured AgentFlow target: claude-vscode", stdout.getvalue())

    def test_activate_claude_vscode_dry_run_does_not_write_files(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp, "--dry-run"], stdout=stdout)

            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "activation.json").exists())
            self.assertFalse((Path(tmp) / "claude-vscode.env").exists())
            self.assertIn("Dry run: would configure AgentFlow target: claude-vscode", stdout.getvalue())
            self.assertIn("Env file changed: true", stdout.getvalue())

    def test_activate_claude_vscode_does_not_persist_secret_env_vars(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "sk-ant-secret",
                    "ANTHROPIC_AUTH_TOKEN": "sk-ant-token-secret",
                },
                clear=False,
            ):
                code = cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 0)
            activation_raw = (Path(tmp) / "activation.json").read_text(encoding="utf-8")
            env_raw = (Path(tmp) / "claude-vscode.env").read_text(encoding="utf-8")
            persisted = activation_raw + "\n" + env_raw
            self.assertNotIn("ANTHROPIC_API_KEY", persisted)
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN", persisted)
            self.assertNotIn("sk-ant-secret", persisted)
            self.assertNotIn("sk-ant-token-secret", persisted)
            output = stdout.getvalue()
            self.assertNotIn("sk-ant-secret", output)
            self.assertNotIn("sk-ant-token-secret", output)

    def test_activate_claude_vscode_no_auto_claude_prints_remediation(self):
        with TemporaryDirectory() as tmp:
            stderr = io.StringIO()

            code = cli.agentflow_cli(
                ["activate", "claude-vscode", "--config-dir", tmp, "--no-auto-claude"],
                stdout=io.StringIO(),
                stderr=stderr,
            )

            self.assertEqual(code, 2)
            self.assertFalse((Path(tmp) / "activation.json").exists())
            self.assertFalse((Path(tmp) / "claude-vscode.env").exists())
            self.assertIn("AgentFlow target is not configured: claude", stderr.getvalue())
            self.assertIn("agentflow activate claude", stderr.getvalue())

    def test_activate_dry_run_does_not_write_config(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(["activate", "openai", "--config-dir", tmp, "--dry-run"], stdout=stdout)

            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "activation.json").exists())
            self.assertIn("Dry run: would configure AgentFlow target: openai", stdout.getvalue())

    def test_top_level_config_dir_is_preserved(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.agentflow_cli(["--config-dir", tmp, "activate", "openai"], stdout=stdout)

            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "activation.json").exists())
            self.assertIn(f"Config file: {Path(tmp) / 'activation.json'}", stdout.getvalue())

    def test_activation_does_not_write_or_print_api_keys(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-openai-secret",
                    "ANTHROPIC_API_KEY": "sk-ant-secret",
                    "AGENTFLOW_OPENAI_API_KEY": "sk-proxy-secret",
                },
                clear=False,
            ):
                code = cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 0)
            raw_config = (Path(tmp) / "activation.json").read_text(encoding="utf-8")
            output = stdout.getvalue()
            for secret in ("sk-openai-secret", "sk-ant-secret", "sk-proxy-secret"):
                self.assertNotIn(secret, raw_config)
                self.assertNotIn(secret, output)

    def test_agentflow_run_translates_openai_profile_to_proxy_flags(self):
        upstream = "https://azure.example/openai"
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(
                [
                    "activate",
                    "openai",
                    "--config-dir",
                    tmp,
                    "--openai-base-url",
                    upstream,
                    "--openai-auth-mode",
                    "proxy",
                ],
                stdout=io.StringIO(),
            )

            with patch("agentflow_proxy.server.main") as server_main:
                code = cli.agentflow_cli(["run", "openai", "--config-dir", tmp], stdout=io.StringIO())

            self.assertEqual(code, 0)
            server_main.assert_called_once_with([
                "--provider",
                "openai",
                "--host",
                "127.0.0.1",
                "--port",
                "4003",
                "--openai-upstream",
                upstream,
                "--openai-auth-mode",
                "proxy",
            ])

    def test_agentflow_doctor_reports_redacted_upstream_mismatch(self):
        upstream = "https://user:pass@example-resource.openai.azure.com/openai/deployments/a?api-key=secret&api-version=2024-10-21"
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(
                ["activate", "openai", "--config-dir", tmp, "--openai-base-url", upstream],
                stdout=io.StringIO(),
            )
            stdout = io.StringIO()

            with patch("agentflow_proxy.cli.httpx.get") as http_get:
                http_get.return_value = httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "provider": "openai",
                        "upstream": "https://api.openai.com",
                        "openai_auth_mode": "client",
                    },
                )
                code = cli.agentflow_cli(["doctor", "openai", "--config-dir", tmp], stdout=stdout)

            self.assertEqual(code, 1)
            output = stdout.getvalue()
            self.assertIn("upstream-mismatch", output)
            self.assertIn("upstream: https://example-resource.openai.azure.com/openai/deployments/a", output)
            self.assertIn("api-key=%5Bredacted%5D", output)
            self.assertNotIn("user:pass", output)
            self.assertNotIn("api-key=secret", output)

    def test_agentflow_doctor_json_covers_provider_health_statuses(self):
        cases = [
            (
                "healthy",
                httpx.Response(200, json={"ok": True, "provider": "openai", "upstream": "https://api.openai.com"}),
                0,
                "healthy",
                [],
            ),
            (
                "connection-refused",
                httpx.ConnectError("refused"),
                1,
                "configured but not running",
                ["health-unreachable"],
            ),
            (
                "non-2xx",
                httpx.Response(503, json={"ok": False}),
                1,
                "unhealthy",
                ["health-non-2xx"],
            ),
            (
                "ok-false",
                httpx.Response(200, json={"ok": False, "provider": "openai", "upstream": "https://api.openai.com"}),
                1,
                "unhealthy",
                ["health-ok-false"],
            ),
            (
                "provider-mismatch",
                httpx.Response(200, json={"ok": True, "provider": "anthropic", "upstream": "https://api.openai.com"}),
                1,
                "provider mismatch",
                ["provider-mismatch"],
            ),
            (
                "upstream-mismatch",
                httpx.Response(200, json={"ok": True, "provider": "openai", "upstream": "https://other.example"}),
                1,
                "stale base url",
                ["upstream-mismatch"],
            ),
            (
                "invalid-json",
                httpx.Response(200, text="not json"),
                1,
                "unhealthy",
                ["health-invalid-json"],
            ),
        ]
        for name, response_or_exc, expected_code, expected_status, expected_reasons in cases:
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO())
                stdout = io.StringIO()

                with patch("agentflow_proxy.cli.httpx.get", side_effect=[response_or_exc]):
                    code = cli.agentflow_cli(["doctor", "openai", "--config-dir", tmp, "--json"], stdout=stdout)

                self.assertEqual(code, expected_code)
                result = json.loads(stdout.getvalue())
                target = result["targets"]["openai"]
                self.assertEqual(target["status"], expected_status)
                self.assertEqual(target["configured"], True)
                self.assertEqual(target["local_base_url"], "http://127.0.0.1:4003/v1")
                self.assertEqual(target["health_url"], "http://127.0.0.1:4003/health")
                self.assertEqual(target["upstream_base_url"], "https://api.openai.com")
                for reason in expected_reasons:
                    self.assertIn(reason, target["reasons"])

    def test_agentflow_doctor_provider_mismatch_on_openai_port(self):
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(["activate", "openai", "--config-dir", tmp], stdout=io.StringIO())
            stdout = io.StringIO()

            with patch("agentflow_proxy.cli.httpx.get") as http_get:
                http_get.return_value = httpx.Response(
                    200,
                    json={"ok": True, "provider": "anthropic", "upstream": "https://api.openai.com"},
                )
                code = cli.agentflow_cli(["doctor", "openai", "--config-dir", tmp], stdout=stdout)

        self.assertEqual(code, 1)
        self.assertIn("openai: provider mismatch", stdout.getvalue())
        self.assertIn("provider-mismatch", stdout.getvalue())

    def test_agentflow_doctor_codex_reports_not_routed_when_config_missing_or_stale(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "config.toml"
            cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )
            codex_config.unlink()
            missing_stdout = io.StringIO()

            missing_code = cli.agentflow_cli(["doctor", "codex", "--config-dir", str(config_dir), "--json"], stdout=missing_stdout)

            codex_config.write_text('openai_base_url = "https://api.openai.com/v1"\n', encoding="utf-8")
            stale_stdout = io.StringIO()
            stale_code = cli.agentflow_cli(["doctor", "codex", "--config-dir", str(config_dir), "--json"], stdout=stale_stdout)

        self.assertEqual(missing_code, 1)
        missing = json.loads(missing_stdout.getvalue())["targets"]["codex"]
        self.assertEqual(missing["status"], "not routed via agentflow")
        self.assertIn("codex-config-missing", missing["reasons"])
        self.assertEqual(stale_code, 1)
        stale = json.loads(stale_stdout.getvalue())["targets"]["codex"]
        self.assertEqual(stale["status"], "not routed via agentflow")
        self.assertEqual(stale["codex_openai_base_url"], "https://api.openai.com/v1")
        self.assertIn("codex-openai-base-url-mismatch", stale["reasons"])

    def test_agentflow_doctor_codex_reports_healthy_when_toml_points_to_agentflow(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "config.toml"
            cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )
            stdout = io.StringIO()

            code = cli.agentflow_cli(["doctor", "codex", "--config-dir", str(config_dir), "--json"], stdout=stdout)

        self.assertEqual(code, 0)
        target = json.loads(stdout.getvalue())["targets"]["codex"]
        self.assertEqual(target["status"], "healthy")
        self.assertEqual(target["codex_openai_base_url"], "http://127.0.0.1:4003/v1")

    def test_agentflow_doctor_claude_vscode_reports_environment_uncertainty(self):
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=io.StringIO())
            stdout = io.StringIO()

            with patch.dict(os.environ, {}, clear=True):
                code = cli.agentflow_cli(["doctor", "claude-vscode", "--config-dir", tmp, "--json"], stdout=stdout)

        self.assertEqual(code, 0)
        target = json.loads(stdout.getvalue())["targets"]["claude-vscode"]
        self.assertEqual(target["status"], "configured")
        self.assertEqual(target["env_file_base_url"], "http://127.0.0.1:4000")
        self.assertIn("current-shell-env-missing", target["reasons"])
        self.assertIn("vscode-runtime-env-uncertain", target["reasons"])

    def test_agentflow_doctor_claude_vscode_detects_shell_base_url_mismatch(self):
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(["activate", "claude-vscode", "--config-dir", tmp], stdout=io.StringIO())
            stdout = io.StringIO()

            with patch.dict(os.environ, {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}, clear=True):
                code = cli.agentflow_cli(["doctor", "claude-vscode", "--config-dir", tmp, "--json"], stdout=stdout)

        self.assertEqual(code, 1)
        target = json.loads(stdout.getvalue())["targets"]["claude-vscode"]
        self.assertEqual(target["status"], "not routed via agentflow")
        self.assertIn("current-shell-anthropic-base-url-mismatch", target["reasons"])

    def test_agentflow_run_translates_claude_profile_to_proxy_flags(self):
        upstream = "https://anthropic.example"
        with TemporaryDirectory() as tmp:
            cli.agentflow_cli(
                ["activate", "claude", "--config-dir", tmp, "--claude-base-url", upstream],
                stdout=io.StringIO(),
            )

            with patch("agentflow_proxy.server.main") as server_main:
                code = cli.agentflow_cli(["run", "claude", "--config-dir", tmp], stdout=io.StringIO())

            self.assertEqual(code, 0)
            server_main.assert_called_once_with([
                "--provider",
                "anthropic",
                "--host",
                "127.0.0.1",
                "--port",
                "4000",
                "--anthropic-upstream",
                upstream,
            ])

    def test_activate_codex_writes_user_config_and_default_openai_dependency(self):
        with TemporaryDirectory() as tmp:
            codex_config = Path(tmp) / "home" / ".codex" / "config.toml"
            stdout = io.StringIO()

            code = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(Path(tmp) / "agentflow"), "--codex-config", str(codex_config)],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            self.assertEqual(codex_config.read_text(encoding="utf-8"), 'openai_base_url = "http://127.0.0.1:4003/v1"\n')
            config = json.loads((Path(tmp) / "agentflow" / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(config["targets"].keys()), ["codex", "openai"])
            self.assertEqual(config["targets"]["codex"]["depends_on"], "openai")
            self.assertEqual(config["targets"]["codex"]["codex_config_path"], str(codex_config))
            self.assertEqual(config["targets"]["codex"]["local_base_url"], "http://127.0.0.1:4003/v1")
            self.assertIn("OpenAI target was not configured; created the default OpenAI activation profile.", stdout.getvalue())
            self.assertNotIn("sk-", codex_config.read_text(encoding="utf-8"))

    def test_activate_codex_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "config.toml"

            first = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )
            second_stdout = io.StringIO()
            second = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=second_stdout,
            )

            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            self.assertEqual(codex_config.read_text(encoding="utf-8"), 'openai_base_url = "http://127.0.0.1:4003/v1"\n')
            self.assertFalse(list(codex_config.parent.glob("config.toml.agentflow.bak*")))
            self.assertIn("Codex config changed: false", second_stdout.getvalue())

    def test_activate_codex_preserves_unrelated_toml_and_comments(self):
        with TemporaryDirectory() as tmp:
            codex_config = Path(tmp) / "config.toml"
            codex_config.write_text(
                '# keep this comment\nmodel = "gpt-5-codex"\nopenai_base_url = "https://api.openai.com/v1" # route\n[projects]\ntrusted = true\n',
                encoding="utf-8",
            )

            code = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(Path(tmp) / "agentflow"), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )

            self.assertEqual(code, 0)
            updated = codex_config.read_text(encoding="utf-8")
            self.assertIn('# keep this comment\nmodel = "gpt-5-codex"\n', updated)
            self.assertIn('openai_base_url = "http://127.0.0.1:4003/v1" # route\n', updated)
            self.assertIn("[projects]\ntrusted = true\n", updated)

    def test_activate_codex_replaces_stale_url_when_openai_local_url_changes(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "config.toml"
            cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )
            cli.agentflow_cli(
                ["activate", "openai", "--config-dir", str(config_dir), "--local-base-url", "http://127.0.0.1:4999/v1"],
                stdout=io.StringIO(),
            )

            code = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(config_dir), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )

            self.assertEqual(code, 0)
            self.assertIn('openai_base_url = "http://127.0.0.1:4999/v1"', codex_config.read_text(encoding="utf-8"))
            config = json.loads((config_dir / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(config["targets"]["codex"]["local_base_url"], "http://127.0.0.1:4999/v1")

    def test_activate_codex_dry_run_does_not_write_files(self):
        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "agentflow"
            codex_config = Path(tmp) / "config.toml"
            stdout = io.StringIO()

            code = cli.agentflow_cli(
                [
                    "activate",
                    "codex",
                    "--config-dir",
                    str(config_dir),
                    "--codex-config",
                    str(codex_config),
                    "--dry-run",
                ],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            self.assertFalse(codex_config.exists())
            self.assertFalse((config_dir / "activation.json").exists())
            self.assertIn(f"Codex config file: {codex_config}", stdout.getvalue())
            self.assertIn("Codex OpenAI base URL: http://127.0.0.1:4003/v1", stdout.getvalue())

    def test_activate_codex_creates_backup_before_modifying_existing_config(self):
        with TemporaryDirectory() as tmp:
            codex_config = Path(tmp) / "config.toml"
            original = 'model = "gpt-5-codex"\n'
            codex_config.write_text(original, encoding="utf-8")

            code = cli.agentflow_cli(
                ["activate", "codex", "--config-dir", str(Path(tmp) / "agentflow"), "--codex-config", str(codex_config)],
                stdout=io.StringIO(),
            )

            self.assertEqual(code, 0)
            backup = codex_config.with_name("config.toml.agentflow.bak")
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8"), original)
            self.assertIn("openai_base_url", codex_config.read_text(encoding="utf-8"))

    def test_activate_codex_does_not_write_or_print_api_keys(self):
        with TemporaryDirectory() as tmp:
            codex_config = Path(tmp) / "config.toml"
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-openai-secret",
                    "AGENTFLOW_OPENAI_API_KEY": "sk-proxy-secret",
                },
                clear=False,
            ):
                code = cli.agentflow_cli(
                    ["activate", "codex", "--config-dir", str(Path(tmp) / "agentflow"), "--codex-config", str(codex_config)],
                    stdout=stdout,
                )

            self.assertEqual(code, 0)
            activation_raw = (Path(tmp) / "agentflow" / "activation.json").read_text(encoding="utf-8")
            codex_raw = codex_config.read_text(encoding="utf-8")
            output = stdout.getvalue()
            for secret in ("sk-openai-secret", "sk-proxy-secret"):
                self.assertNotIn(secret, activation_raw)
                self.assertNotIn(secret, codex_raw)
                self.assertNotIn(secret, output)

    def test_activate_codex_default_path_does_not_touch_project_local_config(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            project_config = workspace / ".codex" / "config.toml"
            project_config.parent.mkdir()
            project_config.write_text('openai_base_url = "https://api.openai.com/v1"\n', encoding="utf-8")
            home = Path(tmp) / "home"
            home.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                    code = cli.agentflow_cli(
                        ["activate", "codex", "--config-dir", str(Path(tmp) / "agentflow")],
                        stdout=io.StringIO(),
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 0)
            self.assertEqual(project_config.read_text(encoding="utf-8"), 'openai_base_url = "https://api.openai.com/v1"\n')
            self.assertEqual(
                (home / ".codex" / "config.toml").read_text(encoding="utf-8"),
                'openai_base_url = "http://127.0.0.1:4003/v1"\n',
            )

    def test_activate_codex_refuses_explicit_project_local_config(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            project_config = workspace / ".codex" / "config.toml"
            project_config.parent.mkdir()
            project_config.write_text('openai_base_url = "https://api.openai.com/v1"\n', encoding="utf-8")
            stderr = io.StringIO()
            previous_cwd = Path.cwd()
            try:
                os.chdir(workspace)
                code = cli.agentflow_cli(
                    [
                        "activate",
                        "codex",
                        "--config-dir",
                        str(Path(tmp) / "agentflow"),
                        "--codex-config",
                        str(project_config),
                    ],
                    stdout=io.StringIO(),
                    stderr=stderr,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(code, 2)
            self.assertIn("refusing to modify project-local .codex/config.toml", stderr.getvalue())
            self.assertEqual(project_config.read_text(encoding="utf-8"), 'openai_base_url = "https://api.openai.com/v1"\n')


class PolicyReloadCliTests(unittest.TestCase):
    def setUp(self):
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 200
        ManagedFeedbackFlushClient.text = '{"ok":true}'
        self.tmp = TemporaryDirectory()
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")
        self._old_provenance_env = {
            key: os.environ.get(key)
            for key in (
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET",
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS",
                "AGENTFLOW_MANAGED_POLICY_HMAC_SECRET",
            )
        }
        for key in self._old_provenance_env:
            os.environ[key] = ""

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        for key, value in self._old_provenance_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_default_policy_reload_url_uses_agentflow_port(self):
        with patch.dict(os.environ, {"AGENTFLOW_PORT": "4001"}, clear=False):
            self.assertEqual(
                cli._default_policy_reload_url(),
                "http://127.0.0.1:4001/agentflow/admin/reload-policies",
            )

    def test_cli_module_re_exports_policy_bundle_commands(self):
        self.assertIs(cli.policy_reload_cli, policy_bundle_cli.policy_reload_cli)
        self.assertIs(cli.policy_export_cli, policy_bundle_cli.policy_export_cli)
        self.assertIs(cli.policy_validate_cli, policy_bundle_cli.policy_validate_cli)
        self.assertIs(cli.policy_diff_cli, policy_bundle_cli.policy_diff_cli)
        self.assertIs(cli.policy_review_cli, policy_bundle_cli.policy_review_cli)

    def test_cli_module_re_exports_policy_workbench_commands(self):
        self.assertIs(cli.policy_fetch_review_cli, policy_workbench_cli.policy_fetch_review_cli)
        self.assertIs(cli.policy_apply_cli, policy_workbench_cli.policy_apply_cli)
        self.assertIs(cli.policy_draft_stage_cli, policy_workbench_cli.policy_draft_stage_cli)
        self.assertIs(cli.policy_draft_validate_cli, policy_workbench_cli.policy_draft_validate_cli)
        self.assertIs(cli.policy_draft_apply_cli, policy_workbench_cli.policy_draft_apply_cli)

    def test_loopback_url_validation(self):
        self.assertTrue(cli._is_loopback_url("http://127.0.0.1:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://localhost:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://[::1]:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("http://192.168.1.20:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("file:///tmp/reload"))

    def test_policy_reload_cli_prints_reload_json_on_success(self):
        payload = {
            "ok": True,
            "schema": "agentflow.policy_reload.v1",
            "policies": {"schema": "agentflow.policy_state.v1"},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli_commands.policy_bundle.httpx.post") as post:
            post.return_value = httpx.Response(200, json=payload)
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4001/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), payload)
        post.assert_called_once_with("http://127.0.0.1:4001/agentflow/admin/reload-policies", timeout=10.0)

    def test_policy_reload_cli_rejects_non_loopback_url(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli_commands.policy_bundle.httpx.post") as post:
            code = cli.policy_reload_cli(["--url", "http://192.168.1.20:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["type"], "unsafe_url")
        post.assert_not_called()

    def test_policy_draft_apply_cli_rejects_non_loopback_reload_url(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.policy_draft_apply_cli(
            ["draft-one", "--reload-url", "http://192.168.1.20:4000/agentflow/admin/reload-policies"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_draft_apply.v1")
        self.assertEqual(payload["error"]["type"], "unsafe_url")
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        self.assertFalse(payload["privacy"]["managed_server_calls_made"])
        self.assertFalse(payload["privacy"]["loopback_admin_calls_made"])

    def test_policy_rollback_apply_id_cli_rejects_non_loopback_reload_url(self):
        stdout = io.StringIO()

        code = cli.policy_rollback_cli(
            [
                "--apply-id",
                "apply-one",
                "--reload-url",
                "http://192.168.1.20:4000/agentflow/admin/reload-policies",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_draft_rollback.v1")
        self.assertEqual(payload["error"]["type"], "unsafe_url")
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        self.assertFalse(payload["privacy"]["managed_server_calls_made"])
        self.assertFalse(payload["privacy"]["loopback_admin_calls_made"])

    def test_policy_reload_cli_reports_non_success_response(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli_commands.policy_bundle.httpx.post") as post:
            post.return_value = httpx.Response(403, json={"ok": False, "error": {"type": "forbidden"}})
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status_code"], 403)

    def test_policy_export_cli_prints_policy_bundle_json(self):
        stdout = io.StringIO()

        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "1",
                "AGENTFLOW_CODEX_APP_CACHE": "0",
            },
            clear=False,
        ):
            code = cli.policy_export_cli([], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_bundle.v1")
        self.assertEqual(payload["generator"]["mode"], "local-offline")
        self.assertFalse(payload["managed_optimizer"]["enabled"])
        self.assertEqual(payload["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertIn("routing", payload["policies"])
        self.assertIn("codex_app", payload["policies"])
        self.assertFalse(payload["policies"]["codex_app"]["review_only"])
        surface = payload["policies"]["source_surfaces"]["codex_turn"]
        self.assertTrue(surface["optimization"]["enabled"])
        self.assertFalse(surface["cache"]["enabled"])
        self.assertFalse(surface["managed_optimizer_required"])

    def test_codex_diagnose_cli_reads_local_metadata_only(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_codex_app_event(
                    id="start-cli",
                    created_at="2026-06-08T10:00:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-cli",
                    thread_id="thread-cli",
                    message_chars=120,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=64,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-cli",
                    routing_json=stable_json({
                        "status": "not-applicable",
                        "reason": "codex-turn-start-model-field-absent",
                        "applied": False,
                        "policy_source": "local-default",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hash_count": 3,
                            "hash_basis": "normalized-structure-and-size-buckets",
                            "text_bucket": "lt_2k_chars",
                            "token_bucket": "lt_1k_tokens",
                            "pattern_types": ["repeated_input_section"],
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    crunch_json=stable_json({
                        "status": "applied",
                        "reason": "codex-repeated-scaffolding-crunched",
                        "applied": True,
                        "saved_chars": 48,
                        "tokens_saved_est": 12,
                        "codex_repeated_scaffolding": {
                            "status": "applied",
                            "saved_chars": 48,
                            "patterns": [
                                {"type": "repeated_input_section", "count": 1, "saved_chars_est": 48},
                            ],
                        },
                        "codex_patterns": [
                            {"type": "repeated_input_section", "count": 1, "saved_chars_est": 48},
                        ],
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True}),
                )
                store.log_codex_app_event(
                    id="plan-cli",
                    created_at="2026-06-08T10:00:01+00:00",
                    direction="server_to_client",
                    method="turn/plan/updated",
                    request_id=None,
                    thread_id="thread-cli",
                    message_chars=40,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-cli",
                )
                store.log_codex_app_event(
                    id="end-cli",
                    created_at="2026-06-08T10:00:02+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-cli",
                    thread_id="thread-cli",
                    message_chars=90,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=40,
                    error_code=None,
                    error_message=None,
                    latency_ms=25,
                    session_id="session-cli",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_diagnose_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.codex_app_effectiveness.v1")
        self.assertEqual(payload["summary"]["turn_start_rows"], 1)
        self.assertEqual(payload["summary"]["model_field_absent"], 1)
        self.assertEqual(payload["summary"]["codex_repeated_scaffolding_saved_chars"], 48)
        self.assertEqual(payload["summary"]["managed_pattern_fingerprint_rows"], 1)
        self.assertEqual(payload["managed_pattern_fingerprints"]["pattern_hash_count"], 3)
        self.assertFalse(payload["managed_pattern_fingerprints"]["raw_pattern_strings_included"])
        self.assertTrue(payload["recent_samples"][0]["managed_pattern_features"]["present"])
        self.assertEqual(payload["workflow_phase_breakdown"][0]["phase"], "planning")
        self.assertEqual(payload["crunch_pattern_breakdown"][0]["type"], "repeated_input_section")
        self.assertFalse(payload["privacy"]["raw_params_included"])

    def test_codex_canary_impact_cli_reports_rule_lifecycle_metadata_only(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            rule_meta = {
                "schema": "agentflow.codex_app_rule_execution.v1",
                "rule_id": "cli-codex-rule",
                "candidate_id": "cli-candidate",
                "policy_source": "managed-recommended",
                "condition_keys": ["workflow_phase"],
                "action_keys": ["model_hint"],
                "raw_params_included": False,
            }
            try:
                store.log_codex_app_event(
                    id="cli-canary-start",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="raw-cli-request-must-not-leak",
                    thread_id="raw-cli-thread-must-not-leak",
                    session_id="raw-cli-session-must-not-leak",
                    message_chars=100,
                    params_chars=80,
                    input_items=1,
                    input_text_chars=120,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    routing_json=stable_json({
                        "status": "applied",
                        "reason": "codex-app-rule-canary-applied",
                        "applied": True,
                        "canary": "codex-app-rule",
                        "canary_cohort": "canary_applied",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4-mini",
                        "policy_source": "managed-recommended",
                        "codex_app_rule": rule_meta,
                    }),
                    crunch_json=stable_json({"status": "skipped"}),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-rule-no-cache-action"}),
                )
                store.log_codex_app_event(
                    id="cli-canary-end",
                    created_at=utc_now(),
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="raw-cli-request-must-not-leak",
                    thread_id="raw-cli-thread-must-not-leak",
                    session_id="raw-cli-session-must-not-leak",
                    message_chars=80,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=40,
                    error_code=None,
                    error_message=None,
                    latency_ms=25,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_canary_impact_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.codex_app_canary_impact_by_rule.v1")
        self.assertEqual(payload["summary"]["applied_count"], 1)
        self.assertEqual(payload["rules"][0]["rule_id"], "cli-codex-rule")
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-cli-request-must-not-leak", encoded)
        self.assertFalse(payload["privacy"]["request_ids_included"])

    def test_routing_experiment_report_cli_reads_metadata_only_metrics(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for idx, similarity in enumerate((0.95, 0.75), start=1):
                    store.log_routing_experiment(
                        id=f"exp-{idx}",
                        call_id=f"call-{idx}",
                        created_at=utc_now(),
                        provider="anthropic",
                        source_surface="anthropic_messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001",
                        primary_model="claude-haiku-4-5-20251001",
                        shadow_model="claude-sonnet-4-6",
                        category="tool-result",
                        routing_reason="fixture route",
                        input_tokens_est=100,
                        primary_status_code=200,
                        shadow_status_code=200,
                        primary_latency_ms=40,
                        shadow_latency_ms=90,
                        primary_output_chars=12,
                        shadow_output_chars=14,
                        primary_output_sha256=f"primary-{idx}",
                        shadow_output_sha256=f"shadow-{idx}",
                        output_similarity=similarity,
                        passed_threshold=1 if similarity >= 0.86 else 0,
                        primary_cost_est_usd=0.001,
                        shadow_cost_est_usd=0.003,
                        budget_limit_usd=0.01,
                        budget_spent_before_usd=0.0,
                        budget_remaining_before_usd=0.01,
                        budget_spent_after_usd=0.003,
                        error=None,
                        routing_json=stable_json({"routing_experiment": {"sampled": True}}),
                        experiment_json=stable_json({"sampled": True}),
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.routing_experiment_report_cli(["--db", db_path], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.routing_experiment_report.v1")
        self.assertEqual(payload["summary"]["sample_count"], 2)
        self.assertEqual(payload["summary"]["comparison_count"], 2)
        self.assertAlmostEqual(payload["summary"]["pass_rate"], 0.5, places=6)
        self.assertAlmostEqual(payload["summary"]["cost_delta_usd"], -0.004, places=6)
        self.assertAlmostEqual(payload["summary"]["avg_latency_delta_ms"], -50.0, places=6)
        self.assertEqual(payload["candidates"][0]["provider"], "anthropic")
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertNotIn("private cli cache prompt", json.dumps(payload).lower())

    def test_routing_experiment_report_cli_slices_post_fix_shadow_yield(self):
        from agentflow_proxy.store import Store, stable_json

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_routing_experiment(
                    id="old-shadow-error",
                    call_id="old-shadow-call-secret",
                    created_at="2026-06-12T23:59:00+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    primary_model="claude-sonnet-4-6",
                    shadow_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    routing_reason="fixture old shadow",
                    input_tokens_est=100,
                    primary_status_code=200,
                    shadow_status_code=400,
                    output_similarity=None,
                    passed_threshold=0,
                    shadow_cost_est_usd=0.0,
                    error="shadow-http-400",
                    routing_json=stable_json({"request_id": "old-shadow-request-secret"}),
                    experiment_json=stable_json({
                        "optimization_feedback": {
                            "status": "shadow-http-400",
                            "reason_codes": ["shadow-http-400"],
                        },
                        "raw_prompt": "old raw prompt must stay excluded",
                    }),
                )
                for idx, status in enumerate(("compared", "shadow-unsupported-shape"), start=1):
                    compared = status == "compared"
                    store.log_routing_experiment(
                        id=f"recent-shadow-{idx}",
                        call_id=f"recent-shadow-call-secret-{idx}",
                        created_at=f"2026-06-13T01:0{idx}:00+00:00",
                        provider="anthropic",
                        source_surface="anthropic_messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001",
                        primary_model="claude-sonnet-4-6",
                        shadow_model="claude-haiku-4-5-20251001",
                        category="tool-result",
                        routing_reason="fixture recent shadow",
                        input_tokens_est=100,
                        primary_status_code=200,
                        shadow_status_code=200 if compared else None,
                        primary_latency_ms=40,
                        shadow_latency_ms=20 if compared else None,
                        primary_output_chars=12,
                        shadow_output_chars=12 if compared else 0,
                        primary_output_sha256=f"recent-primary-{idx}",
                        shadow_output_sha256=f"recent-shadow-{idx}",
                        output_similarity=0.95 if compared else None,
                        passed_threshold=1 if compared else 0,
                        primary_cost_est_usd=0.003,
                        shadow_cost_est_usd=0.001 if compared else 0.0,
                        budget_limit_usd=1.0,
                        budget_spent_before_usd=0.0,
                        budget_remaining_before_usd=1.0,
                        budget_spent_after_usd=0.001,
                        error=None if compared else "shadow-unsupported-shape:tool-protocol-context-blocked",
                        routing_json=stable_json({"request_id": "recent-shadow-request-secret"}),
                        experiment_json=stable_json({
                            "optimization_feedback": {
                                "status": status,
                                "reason_codes": ["passed"] if compared else [
                                    "shadow-unsupported-shape",
                                    "unsupported-shadow-shape-tool-protocol-context-blocked",
                                ],
                            },
                            "managed_feedback": {
                                "status": "queued",
                                "source_surface": "routing_experiment_outcome",
                                "payload_included": False,
                            },
                            "provider_body": {"messages": [{"content": "recent raw provider body must stay excluded"}]},
                            "tool_payload": {"secret": "recent tool payload must stay excluded"},
                        }),
                        primary_response_json=stable_json({"content": "recent raw primary response must stay excluded"}),
                        shadow_response_json=stable_json({"content": "recent raw shadow response must stay excluded"}),
                    )
                store.log_call(
                    id="recent-decision-unsampled",
                    created_at="2026-06-13T01:03:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=10,
                    input_tokens_est=10,
                    output_tokens_est=10,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    routing_json=stable_json({
                        "routing_experiment": {
                            "status": "skipped",
                            "reason": "sample-rate-not-selected",
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "requested_model": "claude-sonnet-4-6",
                            "shadow_model": "claude-haiku-4-5-20251001",
                            "request_id": "decision-request-secret",
                        }
                    }),
                    category="tool-result",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.routing_experiment_report_cli(
                ["--db", db_path, "--since", "2026-06-13T00:00:00+00:00", "--limit", "10"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        yield_report = payload["post_fix_shadow_yield"]
        self.assertEqual(yield_report["schema"], "agentflow.post_fix_shadow_yield.v1")
        self.assertEqual(yield_report["summary"]["sample_count"], 2)
        self.assertEqual(yield_report["summary"]["compared_count"], 1)
        self.assertEqual(yield_report["summary"]["eligible_unsampled_count"], 1)
        self.assertAlmostEqual(yield_report["summary"]["clean_yield"], 0.5, places=6)
        reasons = {row["reason"] for row in yield_report["reason_counts"]}
        self.assertIn("passed", reasons)
        self.assertIn("shadow-unsupported-shape", reasons)
        self.assertNotIn("shadow-http-400", reasons)
        row = yield_report["candidates"][0]
        self.assertEqual(row["provider"], "anthropic")
        self.assertEqual(row["source_surface"], "anthropic_messages")
        self.assertEqual(row["category"], "tool-result")
        self.assertEqual(row["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(row["shadow_model"], "claude-haiku-4-5-20251001")
        self.assertTrue(yield_report["privacy"]["metadata_only"])
        rendered = json.dumps(yield_report, sort_keys=True)
        for forbidden in (
            "old-shadow-call-secret",
            "recent-shadow-call-secret",
            "recent-shadow-request-secret",
            "decision-request-secret",
            "old raw prompt must stay excluded",
            "recent raw provider body must stay excluded",
            "recent tool payload must stay excluded",
            "recent raw primary response must stay excluded",
            "recent raw shadow response must stay excluded",
            '"request_id"',
            '"provider_body"',
            '"tool_payload"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replayability_report_cli_reads_local_metadata_only(self):
        from agentflow_proxy.store import Store, stable_json

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for index, cost in enumerate((0.01, 0.02)):
                    store.log_call(
                        id=f"cli-cache-{index}",
                        created_at=f"2026-06-10T01:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=1,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({
                            "pattern_modules": {
                                "server_features": {
                                    "features": [
                                        {
                                            "family": "cacheability",
                                            "features": {
                                                "cacheability_bucket": "high",
                                                "static_information_hint": True,
                                                "exact_cache_candidate_hint": True,
                                            },
                                        }
                                    ],
                                },
                            },
                        }),
                        routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 1200}),
                        cache_json=stable_json({
                            "status": "skipped",
                            "reason": "streaming",
                            "policy_source": "local-default",
                            "session_memory_hints": {
                                "dry_run_replay_proposal": {
                                    "schema": "agentflow.session_memory_cache_replay_proposal.v1",
                                    "status": "session-plateau-dry-run-eligible",
                                    "reason": "session-plateau-dry-run-eligible",
                                    "proposal_id": "session-memory-cache-replay:cli123",
                                    "proposal_fingerprint": "sha256:" + "b" * 16,
                                    "rule_id": "cli-session-memory-cache",
                                    "policy_source": "local-manual",
                                    "phase": "summary",
                                    "category": "chat",
                                    "stream": True,
                                    "has_tool_blocks": False,
                                    "thinking_present": False,
                                    "text_size_bucket": "8k_32k_chars",
                                    "projected_tokens_saved_est": 1200,
                                    "projected_savings_bucket": "1k_10k_tokens",
                                    "projected_cost_savings_bucket": "lt_1c",
                                    "blockers": [],
                                    "blocker_families": {},
                                    "review_steps": ["review metadata-only session plateau shape"],
                                    "mutation_applied": False,
                                    "cache_mutation": False,
                                    "cache_entries_written": 0,
                                    "policy_files_written": False,
                                    "provider_calls_made": 0,
                                    "managed_server_calls_made": 0,
                                    "privacy": {"metadata_only": True},
                                },
                            },
                        }),
                        request_json=stable_json({"messages": [{"content": "private cli cache prompt"}]}),
                        session_id="cli-cache-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_replayability_report_cli(["--db", db_path, "--limit", "5"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.cache_replayability.v1")
        self.assertEqual(payload["summary"]["repeated_shape_groups"], 1)
        self.assertAlmostEqual(payload["summary"]["projected_repeated_call_cost_usd"], 0.015)
        self.assertEqual(payload["groups"][0]["replay_candidate_class"], "streaming-non-tool-exact-candidate")
        self.assertEqual(payload["groups"][0]["cacheability_bucket"], "high")
        self.assertEqual(payload["summary"]["session_memory_replay_eligible_count"], 2)
        self.assertEqual(payload["session_memory_replay_proposals"][0]["status"], "session-plateau-dry-run-eligible")
        self.assertEqual(payload["session_memory_replay_proposals"][0]["rule_id"], "cli-session-memory-cache")
        evidence = payload["cache_replayability_evidence"]
        self.assertEqual(evidence["schema"], "agentflow.cache_replayability_evidence.v1")
        self.assertEqual(evidence["status"], "no-safe-replayable-cohorts")
        self.assertEqual(evidence["summary"]["total_rows_considered"], 2)
        self.assertEqual(evidence["summary"]["repeated_shape_groups"], 1)
        self.assertIn("cache hits are zero", evidence["zero_hit_explanation"])
        self.assertEqual(evidence["ranked_replayability_cohorts"][0]["provider"], "anthropic")
        self.assertEqual(evidence["ranked_replayability_cohorts"][0]["endpoint"], "messages")
        self.assertEqual(evidence["ranked_replayability_cohorts"][0]["request_shape"]["category"], "chat")
        self.assertEqual(evidence["ranked_replayability_cohorts"][0]["cache_decision_reason"], "streaming")
        self.assertIn(
            "streaming-response-cache-missing",
            evidence["ranked_replayability_cohorts"][0]["blocker_codes"],
        )
        self.assertTrue(evidence["privacy"]["aggregate_only"])
        self.assertFalse(evidence["privacy"]["raw_request_bodies_included"])
        self.assertFalse(evidence["privacy"]["cache_keys_included"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("private cli cache prompt", encoded)
        self.assertNotIn("cli-cache-session-secret", encoded)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])

    def test_cache_replay_cohorts_rank_stable_dependency_plateau_without_raw_ids(self):
        from agentflow_proxy.store import Store, stable_json

        def audit(reason=None, safe=True):
            return {
                "schema": "agentflow.cache_file_dependency_audit.v1",
                "file_watch_enabled": True,
                "snapshot_root_policy": "stored-local-paths",
                "root_path_included": False,
                "snapshot_count": 2,
                "snapshot_count_bucket": "2_5",
                "candidate_path_count_bucket": "2_5",
                "raw_candidate_path_count_bucket": "2_5",
                "distinct_candidate_path_count_bucket": "2_5",
                "dependency_capture_reason": "complete",
                "present_path_count": 2,
                "missing_path_count": 0,
                "changed_path_count": 1 if reason == "dependency-changed" else 0,
                "deleted_path_count": 0,
                "created_path_count": 0,
                "invalidation_reason": reason,
                "safe_invalidation_evidence": safe,
                "file_dependency_evidence_available": safe,
                "paths_included": False,
            }

        def proposal(fingerprint, *, status="session-plateau-dry-run-eligible", blockers=None):
            return {
                "schema": "agentflow.session_memory_cache_replay_proposal.v1",
                "status": status,
                "reason": status,
                "proposal_fingerprint": "sha256:" + fingerprint,
                "rule_id": "private raw rule id should hash",
                "policy_source": "local-manual",
                "phase": "tool-execution",
                "category": "tool-result",
                "stream": True,
                "has_tool_blocks": True,
                "thinking_present": False,
                "text_size_bucket": "32k_128k_chars",
                "projected_tokens_saved_est": 9000,
                "projected_savings_bucket": "1k_10k_tokens",
                "projected_cost_savings_bucket": "1c_5c",
                "blockers": blockers or [],
                "blocker_families": {"safe_invalidation": bool(blockers)},
                "review_steps": ["review metadata-only session plateau shape"],
                "privacy": {"metadata_only": True},
            }

        def log_plateau(store, call_id, created_at, *, session_id, cost, dep_audit, category="tool-result", proposal_hash="a" * 16):
            store.log_call(
                id=call_id,
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=100,
                input_tokens_est=4000,
                output_tokens_est=200,
                actual_input_tokens=4000,
                actual_output_tokens=200,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({
                    "category": category,
                    "workflow_phase": "tool-execution",
                    "has_tools": category.startswith("tool"),
                    "text_chars": 64000,
                    "managed_pattern_features": {
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "category": category,
                        "workflow_phase": "tool-execution",
                        "text_bucket": "32k_128k_chars",
                        "token_bucket": "1k_10k_tokens",
                        "pattern_hashes": ["sha256:" + "d" * 64],
                    },
                }),
                cache_json=stable_json({
                    "status": "skipped",
                    "reason": "streaming",
                    "policy_source": "local-default",
                    "tool_cache_enabled": False,
                    "replayability_level": "local-exact-response",
                    "replay_scope": "session",
                    "replay_scope_id_available": True,
                    "cacheability": {
                        "cacheability_bucket": "high",
                        "static_information_hint": True,
                        "exact_cache_candidate_hint": True,
                    },
                    "file_dependency_audit": dep_audit,
                    "session_memory_hints": {
                        "dry_run_replay_proposal": proposal(proposal_hash),
                    },
                }),
                request_json=stable_json({"messages": [{"content": "raw plateau prompt must not leak"}]}),
                response_json=stable_json({"content": "raw plateau response must not leak"}),
                session_id=session_id,
                category=category,
                retry_count=0,
                provider="anthropic",
            )

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                log_plateau(
                    store,
                    "stable-secret-call-1",
                    "2026-06-10T01:00:00+00:00",
                    session_id="stable-session-secret",
                    cost=0.02,
                    dep_audit=audit(safe=True),
                )
                log_plateau(
                    store,
                    "stable-secret-call-2",
                    "2026-06-10T01:01:00+00:00",
                    session_id="stable-session-secret",
                    cost=0.03,
                    dep_audit=audit(safe=True),
                )
                log_plateau(
                    store,
                    "stale-secret-call-1",
                    "2026-06-10T01:02:00+00:00",
                    session_id="stale-session-secret",
                    cost=0.05,
                    dep_audit=audit(reason="dependency-changed", safe=False),
                    proposal_hash="b" * 16,
                )
                log_plateau(
                    store,
                    "stale-secret-call-2",
                    "2026-06-10T01:03:00+00:00",
                    session_id="stale-session-secret",
                    cost=0.05,
                    dep_audit=audit(reason="dependency-changed", safe=False),
                    proposal_hash="b" * 16,
                )
                log_plateau(
                    store,
                    "oneoff-secret-call",
                    "2026-06-10T01:04:00+00:00",
                    session_id="oneoff-session-secret",
                    cost=0.04,
                    dep_audit=audit(safe=True),
                    category="chat",
                    proposal_hash="c" * 16,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_replay_cohorts_cli(["--db", db_path, "--scan-limit", "20", "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.cache_replay_plateau_cohort_ranking.v1")
        self.assertEqual(payload["summary"]["activation_ready_count"], 1)
        self.assertEqual(payload["summary"]["needs_more_evidence_count"], 1)
        self.assertEqual(payload["summary"]["blocked_count"], 1)
        self.assertEqual(payload["cohorts"][0]["readiness"], "activation-ready")
        self.assertEqual(payload["cohorts"][0]["dependency_state"], "stable")
        self.assertEqual(payload["cohorts"][0]["projected_hits"], 1)
        self.assertEqual(payload["cohorts"][0]["recommended_canary"]["safe_invalidation"], True)
        self.assertTrue(any(row["readiness"] == "needs-more-evidence" for row in payload["cohorts"]))
        blocked = [row for row in payload["cohorts"] if row["readiness"] == "blocked"][0]
        self.assertIn("dependency-invalidated", blocked["blocker_reasons"])
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw plateau prompt must not leak",
            "raw plateau response must not leak",
            "stable-session-secret",
            "stale-session-secret",
            "oneoff-session-secret",
            "stable-secret-call",
            "stale-secret-call",
            "oneoff-secret-call",
            "sha256:" + "d" * 64,
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["raw_session_ids_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])

    def test_cache_smoke_diagnostic_cli_explains_exact_cache_hits_and_skips(self):
        from agentflow_proxy.cache import cache_key_for
        from agentflow_proxy.store import Store, stable_json

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = str(tmp_path / "agentflow.sqlite3")
            dep_path = tmp_path / "watched.txt"
            dep_path.write_text("old", encoding="utf-8")
            stat = dep_path.stat()
            hit_body = {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "raw exact hit prompt must not leak"}],
            }
            hit_key = cache_key_for(hit_body, "/v1/messages", provider="anthropic")
            no_hit_key = "cache-present-no-hit-secret"
            invalidated_key = "cache-invalidated-secret"
            store = Store(db_path)
            try:
                store.set_cache(hit_key, "claude-sonnet-4-6", 80, {"content": "raw cached response must not leak"})
                store.set_cache(no_hit_key, "claude-haiku-4-5-20251001", 30, {"content": "unused"})
                store.set_cache(
                    invalidated_key,
                    "claude-sonnet-4-6",
                    120,
                    {"content": "invalidated"},
                    file_deps=[{
                        "path": str(dep_path),
                        "exists": True,
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                    }],
                )
                dep_path.write_text("new text that changes the dependency", encoding="utf-8")
                for index in range(2):
                    store.log_call(
                        id=f"cache-smoke-miss-{index}",
                        created_at=f"2026-06-10T03:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=300,
                        output_tokens_est=20,
                        cost_est_usd=0.002,
                        cost_baseline_usd=0.002,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 1000}),
                        cache_json=stable_json({"status": "miss", "reason": "exact-miss", "exact_enabled": True}),
                        request_json=stable_json({"messages": [{"content": "raw miss prompt must not leak"}]}),
                        session_id="cache-smoke-session-secret",
                        category="chat",
                        provider="anthropic",
                    )
                store.log_call(
                    id="cache-smoke-streaming",
                    created_at="2026-06-10T03:10:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=120,
                    input_tokens_est=500,
                    output_tokens_est=40,
                    cost_est_usd=0.003,
                    cost_baseline_usd=0.003,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 2000}),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming", "exact_enabled": False}),
                    request_json=stable_json({"messages": [{"content": "raw streaming prompt must not leak"}]}),
                    session_id="cache-smoke-session-secret",
                    category="chat",
                    provider="anthropic",
                )
                store.log_call(
                    id="cache-smoke-tools-disabled",
                    created_at="2026-06-10T03:20:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=130,
                    input_tokens_est=600,
                    output_tokens_est=50,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.004,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"category": "tool-result", "has_tools": True, "text_chars": 4000}),
                    cache_json=stable_json({"status": "skipped", "reason": "tools-disabled", "exact_enabled": False}),
                    request_json=stable_json({"messages": [{"content": "raw tools prompt must not leak"}]}),
                    session_id="cache-smoke-session-secret",
                    category="tool-result",
                    provider="anthropic",
                )
                store.log_call(
                    id="cache-smoke-hit",
                    created_at="2026-06-10T03:30:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=1,
                    status_code=200,
                    latency_ms=3,
                    input_tokens_est=80,
                    output_tokens_est=10,
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"category": "chat", "has_tools": False, "text_chars": 1000}),
                    cache_json=stable_json({"status": "hit", "reason": "exact-match", "hit_type": "exact", "exact_enabled": True}),
                    request_json=stable_json(hit_body),
                    session_id="cache-smoke-session-secret",
                    category="chat",
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_smoke_diagnostic_cli(["--db", db_path, "--scan-limit", "20"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.cache_smoke_diagnostic.v1")
        self.assertEqual(payload["summary"]["cache_rows"], 3)
        self.assertEqual(payload["summary"]["eligible_lookup_count"], 3)
        self.assertEqual(payload["summary"]["exact_miss_count"], 2)
        self.assertEqual(payload["summary"]["cache_hit_count"], 1)
        self.assertEqual(payload["summary"]["invalidated_cache_row_count"], 1)
        self.assertEqual(payload["duplicate_key_opportunity"]["candidate_group_count"], 1)
        self.assertEqual(payload["selected_cache_row_reconstruction"]["status"], "matched")
        skip_reasons = {row["value"] for row in payload["skip_reason_breakdown"]}
        self.assertIn("streaming", skip_reasons)
        self.assertIn("tools-disabled", skip_reasons)
        invalidation_reasons = {row["value"] for row in payload["invalidation_breakdown"]}
        self.assertIn("dependency-changed", invalidation_reasons)
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            hit_key,
            no_hit_key,
            invalidated_key,
            str(dep_path),
            "raw exact hit prompt must not leak",
            "raw cached response must not leak",
            "raw miss prompt must not leak",
            "raw streaming prompt must not leak",
            "raw tools prompt must not leak",
            "cache-smoke-session-secret",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertFalse(payload["privacy"]["file_paths_included"])
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])

    def test_zero_hit_cache_ladder_emits_ranked_issue_ready_action_candidates(self):
        from agentflow_proxy import stats
        from agentflow_proxy.store import stable_json

        def row(index, *, stream, reason, category, phase, has_tools=False, path="/v1/messages"):
            return {
                "id": f"secret-call-{index}",
                "created_at": f"2026-06-10T04:0{index}:00+00:00",
                "stream": 1 if stream else 0,
                "cache_hit": 0,
                "status_code": 200,
                "path": path,
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": None,
                "routing_json": stable_json({
                    "category": category,
                    "workflow_phase": phase,
                    "has_tools": has_tools,
                    "managed_pattern_features": {
                        "pattern_hashes": ["sha256:" + "e" * 64],
                        "workflow_phase": phase,
                    },
                }),
                "cache_json": stable_json({
                    "status": "skipped" if reason != "exact-miss" else "miss",
                    "reason": reason,
                    "policy_source": "local-default",
                    "has_tools": has_tools,
                    "replayability_level": "local-exact-response",
                    "cache_key": "raw-cache-key-secret",
                    "pattern_hash": "sha256:" + "f" * 64,
                    "file_dependency_audit": {
                        "paths_included": True,
                        "path": "/home/lutz/private/cache-source.py",
                    },
                }),
                "request_json": stable_json({"messages": [{"content": "raw cache ladder prompt must not leak"}]}),
                "session_id": "secret-session-id",
            }

        rows = [
            row(0, stream=True, reason="streaming", category="tool-result", phase="tool-execution", has_tools=True),
            row(1, stream=True, reason="streaming", category="tool-result", phase="tool-execution", has_tools=True),
            row(2, stream=True, reason="streaming", category="tool-result", phase="tool-execution", has_tools=True),
            row(3, stream=False, reason="tools-disabled", category="tool-result", phase="tool-execution", has_tools=True),
            row(4, stream=False, reason="tools-disabled", category="tool-result", phase="tool-execution", has_tools=True),
            row(5, stream=False, reason="exact-miss", category="chat", phase="chat", has_tools=False),
            row(6, stream=False, reason="unknown", category="chat", phase="chat", path="/home/lutz/private/raw-path"),
        ]

        payload = stats._cache_zero_hit_blocker_ladder(rows, scan_limit=20)

        self.assertEqual(payload["schema"], "agentflow.cache_zero_hit_blocker_ladder.v1")
        self.assertTrue(payload["summary"]["zero_hit_window"])
        self.assertGreaterEqual(payload["summary"]["action_candidate_count"], 3)
        self.assertEqual(payload["summary"]["top_action_candidate_family"], "stage-replay-policy")
        self.assertEqual(payload["ladder"][0]["blocker_code"], "skipped-streaming")
        self.assertEqual(payload["ladder"][0]["category"], "tool-result")
        self.assertEqual(payload["ladder"][0]["workflow_phase"], "tool-execution")
        top = payload["action_candidates"][0]
        self.assertIn("Stage streaming cache replay pattern rule", top["title"])
        self.assertEqual(top["local_action"]["family"], "stage-replay-policy")
        self.assertTrue(top["local_action"]["concrete"])
        self.assertIn("dry-run", top["acceptance_metric"])
        self.assertIn("backlog", top["labels"])
        self.assertIn("expected_savings_path_or_bottleneck_removed", top)
        true_miss = [
            candidate
            for candidate in payload["action_candidates"]
            if candidate["local_action"]["family"] == "accept-non-repeatable-traffic"
        ][0]
        self.assertEqual(true_miss["local_action"]["activation_mode"], "research-only")
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw cache ladder prompt must not leak",
            "secret-call-",
            "secret-session-id",
            "raw-cache-key-secret",
            "sha256:" + "e" * 64,
            "sha256:" + "f" * 64,
            "/home/lutz/private",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["file_paths_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["session_ids_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])
        self.assertFalse(payload["privacy"]["candidate_identifiers_included"])

    def test_cache_replay_dry_run_cli_reads_policy_without_mutating_cache(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "a" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            policy_path = Path(tmp) / "proposed-cache-policy.json"
            policy_path.write_text(
                json.dumps({
                    "policies": {
                        "cache": {
                            "pattern_rules": [
                                {
                                    "id": "cli-cache-dry-run-rule",
                                    "candidate_id": "cli-cache-candidate",
                                    "conditions": {
                                        "pattern_hashes": [pattern_hash],
                                        "source_surface": "anthropic_messages",
                                        "category": "chat",
                                        "has_tools": False,
                                        "stream": False,
                                    },
                                    "action": {"type": "exact_cache_pattern"},
                                }
                            ],
                        }
                    }
                }),
                encoding="utf-8",
            )
            store = Store(db_path)
            try:
                store.set_cache("existing-cli-cache-key", "claude-sonnet-4-6", 10, {"content": "cached"})
                for index, cost in enumerate((0.01, 0.03)):
                    store.log_call(
                        id=f"cli-cache-dry-{index}",
                        created_at=f"2026-06-10T02:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({
                            "text_chars": 1200,
                            "category": "chat",
                            "has_tools": False,
                            "managed_pattern_features": {
                                "pattern_hashes": [pattern_hash],
                                "source_surface": "anthropic_messages",
                                "app_family": "claude_code",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "text_bucket": "lt_2k_chars",
                                "token_bucket": "lt_1k_tokens",
                                "raw_pattern_strings_included": False,
                            },
                        }),
                        cache_json=stable_json({
                            "status": "miss",
                            "reason": "exact-miss",
                            "policy_source": "local-default",
                            "replayability_level": "local-exact-response",
                        }),
                        request_json=stable_json({"messages": [{"content": "raw cli dry run prompt must not leak"}]}),
                        session_id="cli-dry-run-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.cache_replay_dry_run_cli(
                [str(policy_path), "--db", db_path, "--scan-limit", "10", "--limit", "10"],
                stdout=stdout,
            )

            with sqlite3.connect(db_path) as conn:
                cache_rows = conn.execute("select count(*) from cache").fetchone()[0]

        self.assertEqual(code, 0)
        self.assertEqual(cache_rows, 1)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.cache_replay_dry_run.v1")
        self.assertEqual(payload["summary"]["cache_rows_before"], 1)
        self.assertEqual(payload["summary"]["cache_rows_after"], 1)
        self.assertFalse(payload["summary"]["cache_table_mutated"])
        self.assertEqual(payload["summary"]["projected_exact_hits"], 1)
        self.assertEqual(payload["summary"]["projected_streaming_hits"], 0)
        self.assertAlmostEqual(payload["summary"]["estimated_saved_cost_usd"], 0.02)
        self.assertEqual(payload["rows"][0]["rule_id"], "cli-cache-dry-run-rule")
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw cli dry run prompt must not leak", encoded)
        self.assertNotIn("cli-dry-run-session-secret", encoded)
        self.assertNotIn(pattern_hash, encoded)
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])

    def test_managed_pattern_rollups_cli_exports_metadata_only_cohorts(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "9" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="cli-pattern-call",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.006,
                    crunch_json=stable_json({
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "rules": [
                                {
                                    "rule_id": "cli-crunch-rule",
                                    "candidate_id": "cli-crunch-candidate",
                                    "policy_source": "managed-recommended",
                                    "matched_hashes": [pattern_hash],
                                    "applied_count": 1,
                                    "saved_chars": 800,
                                    "canary": {
                                        "enabled": True,
                                        "selected": True,
                                        "status": "applied",
                                        "cohort": "canary_applied",
                                    },
                                }
                            ],
                        }
                    }),
                    routing_json=stable_json({"category": "chat"}),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw cli prompt must stay local"}),
                    session_id="cli-session-secret",
                    category="chat",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_pattern_rollups_cli(["--db", db_path, "--limit", "5", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.managed_pattern_canary_cohort_rollups.v1")
        self.assertEqual(payload["summary"]["cohort_bucket_count"], 1)
        crunch = next(row for row in payload["cohorts"] if row["candidate_id"] == "cli-crunch-candidate")
        self.assertEqual(crunch["canary_cohort"], "canary_applied")
        self.assertTrue(crunch["minimum_sample_readiness"]["ready"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw cli prompt must stay local", encoded)
        self.assertNotIn("cli-session-secret", encoded)

    def test_managed_pattern_rollups_cli_exports_local_fingerprint_evidence(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "7" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="cli-local-pattern-call",
                    created_at="2026-06-10T01:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.006,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({
                        "category": "tool-result",
                        "workflow_phase": "tool-result",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hash": pattern_hash,
                            "normalized_pattern_hash": pattern_hash,
                            "pattern_hashes": [pattern_hash],
                            "pattern_hash_count": 1,
                            "hash_basis": "normalized-structure-and-size-buckets",
                            "workflow_phase": "tool-result",
                            "category": "tool-result",
                            "source_surface": "anthropic_messages",
                            "app_family": "claude_code",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "pattern_types": ["tool_results"],
                            "local_pattern_module_families": ["tool_results"],
                            "local_pattern_module_count": 1,
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw local fingerprint prompt must stay local"}),
                    session_id="local-fingerprint-session-secret",
                    category="tool-result",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_pattern_rollups_cli(["--db", db_path, "--limit", "5", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        evidence = next(row for row in payload["cohorts"] if row["pattern_hash"] == pattern_hash)
        self.assertEqual(evidence["policy_section"], "local_pattern_fingerprint")
        self.assertTrue(evidence["evidence_only"])
        self.assertEqual(evidence["sample_count"], 1)
        self.assertEqual(evidence["pattern_family"], "general")
        self.assertEqual(evidence["local_pattern_module_families"], ["tool_results"])
        self.assertTrue(evidence["minimum_sample_readiness"]["ready"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw local fingerprint prompt must stay local", encoded)
        self.assertNotIn("local-fingerprint-session-secret", encoded)

    def test_optimization_eval_plan_cli_exports_family_agnostic_metadata_only_plans(self):
        from agentflow_proxy.store import Store, stable_json

        pattern_hash = "sha256:" + "8" * 64
        cache_pattern_hash = "sha256:" + "6" * 64
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="eval-routing-call",
                    created_at="2026-06-10T01:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    cost_baseline_usd=0.0045,
                    crunch_json=stable_json({"tokens_saved_est": 0}),
                    routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-execution", "has_tools": True, "text_chars": 4000}),
                    cache_json=stable_json({"status": "skipped", "reason": "tool-cache-disabled", "policy_source": "local-default"}),
                    request_json=stable_json({"messages": [{"content": "raw eval routing prompt must stay local"}]}),
                    session_id="eval-routing-session-secret",
                    category="tool-result",
                    retry_count=0,
                    provider="anthropic",
                )
                for index in range(2):
                    store.log_call(
                        id=f"eval-cache-call-{index}",
                        created_at=f"2026-06-10T01:1{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=100,
                        input_tokens_est=1000,
                        output_tokens_est=100,
                        actual_input_tokens=1000,
                        actual_output_tokens=100,
                        cost_est_usd=0.01,
                        cost_baseline_usd=0.01,
                        crunch_json=stable_json({
                            "pattern_modules": {
                                "server_features": {
                                    "features": [
                                        {
                                            "family": "cacheability",
                                            "features": {
                                                "cacheability_bucket": "high",
                                                "static_information_hint": True,
                                                "exact_cache_candidate_hint": True,
                                            },
                                        }
                                    ],
                                },
                            },
                        }),
                        routing_json=stable_json({
                            "category": "chat",
                            "workflow_phase": "summary",
                            "has_tools": False,
                            "text_chars": 1200,
                            "managed_pattern_features": {
                                "present": True,
                                "pattern_hash": cache_pattern_hash,
                                "pattern_hashes": [cache_pattern_hash],
                                "source_surface": "anthropic_messages",
                                "app_family": "claude_code",
                                "category": "chat",
                                "workflow_phase": "summary",
                                "text_bucket": "lt_2k_chars",
                                "token_bucket": "lt_1k_tokens",
                                "raw_pattern_strings_included": False,
                            },
                        }),
                        cache_json=stable_json({
                            "status": "miss",
                            "reason": "exact-miss",
                            "policy_source": "local-default",
                            "replayability_level": "local-exact-response",
                        }),
                        request_json=stable_json({"prompt": "raw eval cache prompt must stay local"}),
                        session_id="eval-cache-session-secret",
                        category="chat",
                        retry_count=0,
                        provider="anthropic",
                    )
                store.log_call(
                    id="eval-old-context-call",
                    created_at="2026-06-10T01:30:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=100,
                    input_tokens_est=4000,
                    output_tokens_est=100,
                    actual_input_tokens=4000,
                    actual_output_tokens=100,
                    cost_est_usd=0.02,
                    cost_baseline_usd=0.03,
                    crunch_json=stable_json({
                        "old_context_summarization": {
                            "status": "applied",
                            "enabled": True,
                            "reason": "summary-created",
                            "candidate_id": "eval-old-context-candidate",
                            "rule_id": "eval-old-context-rule",
                            "policy_source": "managed-recommended",
                            "model": "claude-haiku-4-5-20251001",
                            "tokens_saved_est": 1200,
                            "summary_input_tokens": 500,
                            "summary_output_tokens": 100,
                            "summary_cost_est_usd": 0.0002,
                            "estimated_net_savings_usd": 0.009,
                            "canary": {
                                "enabled": True,
                                "fraction": 0.5,
                                "unit": "session",
                                "cohort": "canary_applied",
                            },
                        },
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "rules": [
                                {
                                    "rule_id": "eval-crunch-rule",
                                    "candidate_id": "eval-crunch-candidate",
                                    "policy_source": "managed-recommended",
                                    "matched_hashes": [pattern_hash],
                                    "applied_count": 1,
                                    "saved_chars": 800,
                                    "canary": {
                                        "enabled": True,
                                        "selected": True,
                                        "status": "applied",
                                        "cohort": "canary_applied",
                                    },
                                }
                            ],
                        },
                    }),
                    routing_json=stable_json({"category": "chat", "workflow_phase": "summary"}),
                    cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                    request_json=stable_json({"prompt": "raw eval old context prompt must stay local"}),
                    session_id="eval-old-context-session-secret",
                    category="chat",
                    retry_count=0,
                    provider="anthropic",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.optimization_eval_plan_cli(["--db", db_path, "--limit", "20", "--min-samples", "1"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_eval_plan.v1")
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertGreaterEqual(payload["summary"]["candidate_count"], 4)
        family_counts = {row["value"]: row["count"] for row in payload["summary"]["family_counts"]}
        self.assertIn("phase_routing", family_counts)
        self.assertIn("cache_replayability", family_counts)
        self.assertIn("old_context_summarization", family_counts)
        self.assertIn("managed_pattern_candidate", family_counts)
        self.assertIn("eval-old-context-candidate", {row["candidate_id"] for row in payload["plans"]})
        self.assertIn("eval-crunch-candidate", {row["candidate_id"] for row in payload["plans"]})
        self.assertTrue(any(row["blocker_reason_codes"] for row in payload["plans"]))
        self.assertTrue({row["recommended_eval_mode"] for row in payload["plans"]})
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw eval routing prompt must stay local",
            "raw eval cache prompt must stay local",
            "raw eval old context prompt must stay local",
            "eval-routing-session-secret",
            "eval-cache-session-secret",
            "eval-old-context-session-secret",
            pattern_hash,
            cache_pattern_hash,
        ):
            self.assertNotIn(forbidden, encoded)

    def test_optimization_shadow_eval_cli_scores_fixture_plan_metadata_only(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-pass",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "projected_savings_usd": 0.02,
                    "shadow_eval_fixture": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.98,
                        "baseline_cost_usd": 0.03,
                        "candidate_cost_usd": 0.01,
                        "output": "raw passing output must stay local",
                    },
                    "request_json": {"messages": [{"content": "raw passing prompt must stay local"}]},
                    "session_id": "shadow-pass-session-secret",
                },
                {
                    "candidate_id": "shadow-fail",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "offline_eval": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.42,
                    },
                },
                {
                    "candidate_id": "shadow-blocked",
                    "optimization_family": "managed_pattern_candidate",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "features_only",
                    "blocker_reason_codes": ["tool-call-disabled"],
                },
                {
                    "candidate_id": "shadow-unknown",
                    "optimization_family": "old_context_summarization",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                },
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            jsonl_path = Path(tmp) / "results.jsonl"
            stdout = io.StringIO()

            code = cli.optimization_shadow_eval_cli(
                ["--db", db_path, "--results-jsonl", str(jsonl_path), str(plan_path)],
                stdout=stdout,
            )

            conn = sqlite3.connect(db_path)
            try:
                stored_count = conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
            finally:
                conn.close()
            jsonl_text = jsonl_path.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_shadow_eval.v1")
        self.assertEqual(payload["mode"], "plan-only")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertTrue(payload["wrote_result_records"])
        status_by_candidate = {row["candidate_id"]: row["status_class"] for row in payload["results"]}
        self.assertEqual(status_by_candidate["shadow-pass"], "pass")
        self.assertEqual(status_by_candidate["shadow-fail"], "fail")
        self.assertEqual(status_by_candidate["shadow-blocked"], "blocked")
        self.assertEqual(status_by_candidate["shadow-unknown"], "unknown")
        self.assertEqual(stored_count, 4)
        self.assertEqual(len(jsonl_text.splitlines()), 4)
        rendered = stdout.getvalue() + jsonl_text
        for forbidden in (
            "raw passing prompt must stay local",
            "raw passing output must stay local",
            "shadow-pass-session-secret",
            '"request_json"',
            '"session_id"',
            '"messages"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_optimization_eval_queue_cli_builds_and_records_bounded_batch(self):
        from agentflow_proxy.store import Store

        async def fake_plan(_store, *, limit=500, min_samples=1):
            return {
                "schema": "agentflow.optimization_eval_plan.v1",
                "generated_at": "2026-06-10T02:00:00+00:00",
                "plans": [
                    {
                        "candidate_id": "queue-cli-pass",
                        "optimization_family": "phase_routing",
                        "action_family": "routing",
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "granularity": "provider_request",
                        "replayability_level": "local-exact-response",
                        "candidate_created_at": "2026-06-10T01:00:00+00:00",
                        "projected_savings_usd": 0.02,
                        "shadow_eval_fixture": {
                            "baseline_status_code": 200,
                            "candidate_status_code": 200,
                            "output_similarity": 0.98,
                            "prompt": "raw queue cli prompt must stay local",
                        },
                    }
                ],
            }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            stdout = io.StringIO()

            with patch("agentflow_proxy.optimization_eval_queue.build_optimization_eval_plan", fake_plan):
                code = cli.optimization_eval_queue_cli(
                    ["--db", db_path, "--family", "phase_routing", "--limit", "1"],
                    stdout=stdout,
                )

            conn = sqlite3.connect(db_path)
            try:
                stored_count = conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
            finally:
                conn.close()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_eval_queue_run.v1")
        self.assertEqual(payload["summary"]["selected_candidate_count"], 1)
        self.assertEqual(payload["results"][0]["candidate_id"], "queue-cli-pass")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertEqual(stored_count, 1)
        self.assertNotIn("raw queue cli prompt", stdout.getvalue())

    def test_optimization_eval_queue_cli_backfills_promotion_report_dry_run_and_apply(self):
        from agentflow_proxy.store import Store

        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "queue-promotion-high",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.10,
                    "sample_count": 8,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing", "insufficient-eval-pass-results"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                    "prompt": "raw queue promotion prompt must stay local",
                    "session_id": "queue-promotion-session-secret",
                },
                {
                    "candidate_id": "queue-promotion-widen",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "projected_savings_usd": 1.0,
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            report_path = Path(tmp) / "promotion-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            stdout = io.StringIO()
            code = cli.optimization_eval_queue_cli(
                ["--db", db_path, "--promotion-report", str(report_path), "--limit", "5"],
                stdout=stdout,
            )
            dry_payload = json.loads(stdout.getvalue())
            conn = sqlite3.connect(db_path)
            try:
                dry_count = conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
            finally:
                conn.close()

            stdout = io.StringIO()
            code_apply = cli.optimization_eval_queue_cli(
                ["--db", db_path, "--promotion-report", "-", "--apply-backfill", "--limit", "5"],
                stdin=io.StringIO(json.dumps(report)),
                stdout=stdout,
            )
            applied_payload = json.loads(stdout.getvalue())
            conn = sqlite3.connect(db_path)
            try:
                stored = conn.execute(
                    "select candidate_id, status_class, reason_codes_json, result_json from optimization_eval_results"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(code_apply, 0)
        self.assertEqual(dry_payload["schema"], "agentflow.optimization_promotion_eval_backfill.v1")
        self.assertTrue(dry_payload["dry_run"])
        self.assertFalse(dry_payload["wrote_eval_queue_rows"])
        self.assertEqual(dry_count, 0)
        self.assertEqual(dry_payload["tasks"][0]["candidate_id"], "queue-promotion-high")
        self.assertFalse(applied_payload["dry_run"])
        self.assertTrue(applied_payload["wrote_eval_queue_rows"])
        self.assertEqual(stored[0], "queue-promotion-high")
        self.assertEqual(stored[1], "queued")
        self.assertIn("eval-queued", json.loads(stored[2]))
        rendered = json.dumps(dry_payload, sort_keys=True) + json.dumps(applied_payload, sort_keys=True) + stored[3]
        self.assertNotIn("raw queue promotion prompt", rendered)
        self.assertNotIn("queue-promotion-session-secret", rendered)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"session_id"', rendered)

    def _promotion_plan_row(
        self,
        candidate_id: str,
        *,
        optimization_family: str,
        action_family: str,
        applied_count: int,
        holdout_count: int,
        applied_error_rate: float = 0.0,
        holdout_error_rate: float = 0.0,
        applied_retry_rate: float = 0.0,
        holdout_retry_rate: float = 0.0,
        applied_latency_avg_ms: int = 1000,
        holdout_latency_avg_ms: int = 1000,
        projected_savings_usd: float = 0.01,
        extra_evidence: dict | None = None,
    ) -> dict:
        evidence = {
            "canary_evidence": {
                "applied": {
                    "count": applied_count,
                    "error_rate": applied_error_rate,
                    "retry_rate": applied_retry_rate,
                    "latency_avg_ms": applied_latency_avg_ms,
                    "net_savings_usd": projected_savings_usd,
                },
                "holdout": {
                    "count": holdout_count,
                    "error_rate": holdout_error_rate,
                    "retry_rate": holdout_retry_rate,
                    "latency_avg_ms": holdout_latency_avg_ms,
                },
            },
            "prompt": "raw promotion prompt must stay local",
            "request_id": "req-promotion-secret",
            "session_id": "promotion-session-secret",
            "file_path": "/tmp/promotion-secret.py",
        }
        if extra_evidence:
            evidence.update(extra_evidence)
        return {
            "schema": "agentflow.optimization_eval_plan_row.v1",
            "candidate_id": candidate_id,
            "optimization_family": optimization_family,
            "action_family": action_family,
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "granularity": "provider_request",
            "replayability_level": "local-exact-response",
            "current_canary_count": applied_count,
            "holdout_count": holdout_count,
            "sample_count": applied_count + holdout_count,
            "projected_savings_usd": projected_savings_usd,
            "evidence": evidence,
        }

    def _log_promotion_eval_result(self, store, candidate_id: str, status: str, *, created_at: str = "2026-06-10T03:30:00+00:00") -> None:
        from agentflow_proxy.store import stable_json

        reason = "offline-fixture-passed" if status == "pass" else "output-similarity-below-threshold"
        store.log_optimization_eval_result(
            id=f"promotion-eval-{candidate_id}-{status}",
            run_id="promotion-eval-run",
            created_at=created_at,
            candidate_id=candidate_id,
            source_surface="anthropic_messages",
            optimization_family="phase_routing",
            action_family="routing",
            status_class=status,
            reason_codes_json=stable_json([reason]),
            score_json=stable_json({"output_similarity": 0.98 if status == "pass" else 0.2, "quality_score": 0.97 if status == "pass" else 0.1}),
            cost_json=stable_json({"projected_savings_usd": 0.01}),
            result_json=stable_json({
                "candidate_id": candidate_id,
                "status_class": status,
                "prompt": "raw eval prompt must stay local",
                "response": "raw eval response must stay local",
                "request_id": "req-eval-secret",
                "session_id": "session-eval-secret",
            }),
        )

    def test_optimization_promotion_report_cli_scores_verdict_paths_metadata_only(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                self._promotion_plan_row(
                    "routing-widen",
                    optimization_family="phase_routing",
                    action_family="routing",
                    applied_count=2,
                    holdout_count=2,
                    projected_savings_usd=0.02,
                ),
                self._promotion_plan_row(
                    "cache-rollback",
                    optimization_family="cache_replay_confidence",
                    action_family="cache",
                    applied_count=2,
                    holdout_count=2,
                    applied_error_rate=0.5,
                    holdout_error_rate=0.0,
                    projected_savings_usd=0.03,
                ),
                self._promotion_plan_row(
                    "old-context-hold",
                    optimization_family="old_context_summarization",
                    action_family="crunch",
                    applied_count=2,
                    holdout_count=2,
                    applied_retry_rate=0.5,
                    holdout_retry_rate=0.0,
                    projected_savings_usd=0.04,
                ),
                self._promotion_plan_row(
                    "cache-needs-eval",
                    optimization_family="cache_replayability",
                    action_family="cache",
                    applied_count=0,
                    holdout_count=0,
                    projected_savings_usd=0.05,
                ),
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_promotion_eval_result(store, "routing-widen", "pass")
                self._log_promotion_eval_result(store, "cache-rollback", "pass")
                self._log_promotion_eval_result(store, "old-context-hold", "pass")
                plan_path = Path(tmp) / "promotion-plan.json"
                plan_path.write_text(json.dumps(plan), encoding="utf-8")
                evidence_path = Path(tmp) / "old-context-impact.json"
                evidence_path.write_text(
                    json.dumps({
                        "schema": "agentflow.old_context_summary_impact.v1",
                        "policy": {"candidate_id": "old-context-hold"},
                        "quality_gate": {
                            "verdict": "hold",
                            "reason_codes": ["applied-retry-rate-above-threshold"],
                        },
                    }),
                    encoding="utf-8",
                )
                stdout = io.StringIO()

                code = cli.optimization_promotion_report_cli(
                    ["--db", db_path, "--evidence-report", str(evidence_path), str(plan_path)],
                    stdout=stdout,
                )
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_promotion_report.v1")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        by_candidate = {row["candidate_id"]: row for row in payload["candidates"]}
        self.assertEqual(by_candidate["routing-widen"]["verdict"], "widen")
        self.assertIn("promotion-thresholds-met", by_candidate["routing-widen"]["reason_codes"])
        self.assertEqual(by_candidate["cache-rollback"]["verdict"], "rollback")
        self.assertIn("rollback-error-rate", by_candidate["cache-rollback"]["reason_codes"])
        self.assertEqual(by_candidate["old-context-hold"]["verdict"], "hold")
        self.assertIn("old-context-quality-gate-hold", by_candidate["old-context-hold"]["reason_codes"])
        self.assertEqual(by_candidate["cache-needs-eval"]["verdict"], "needs_eval")
        self.assertIn("eval-results-missing", by_candidate["cache-needs-eval"]["reason_codes"])
        verdict_counts = {row["value"]: row["count"] for row in payload["summary"]["verdict_counts"]}
        self.assertEqual(verdict_counts["widen"], 1)
        self.assertEqual(verdict_counts["hold"], 1)
        self.assertEqual(verdict_counts["rollback"], 1)
        self.assertEqual(verdict_counts["needs_eval"], 1)
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw promotion prompt",
            "raw eval prompt",
            "raw eval response",
            "req-promotion-secret",
            "req-eval-secret",
            "promotion-session-secret",
            "session-eval-secret",
            "/tmp/promotion-secret.py",
            '"prompt"',
            '"response"',
            '"request_id"',
            '"session_id"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["raw_session_ids_included"])

    def test_optimization_promotion_report_cli_reports_eval_failure_rollback(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                self._promotion_plan_row(
                    "routing-eval-fail",
                    optimization_family="phase_routing",
                    action_family="routing",
                    applied_count=2,
                    holdout_count=2,
                    projected_savings_usd=0.02,
                )
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_promotion_eval_result(store, "routing-eval-fail", "fail")
                stdout = io.StringIO()

                code = cli.optimization_promotion_report_cli(
                    ["--db", db_path, "-"],
                    stdin=io.StringIO(json.dumps(plan)),
                    stdout=stdout,
                )
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["verdict"], "rollback")
        self.assertIn("eval-failed", candidate["reason_codes"])
        self.assertEqual(candidate["eval_evidence"]["fail_count"], 1)

    def test_optimization_promotion_actions_cli_emits_rollout_bundle_from_report(self):
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "routing-cli-action",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "candidate_target_model": "claude-haiku-4-5-20251001",
                    "projected_savings_usd": 0.02,
                    "sample_count": 3,
                    "cohort_counts": {"canary_applied": 2, "canary_holdout": 1, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0, "blocked_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"raw_prompts_included": False},
                    "prompt": "raw promotion action prompt must stay local",
                    "request_id": "promotion-action-request-secret",
                    "session_id": "promotion-action-session-secret",
                    "file_path": "/tmp/promotion-action-secret.py",
                },
                {
                    "candidate_id": "cache-cli-action",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "candidate_profile": "replay-safe-exact-candidate",
                    "projected_savings_usd": 0.03,
                    "sample_count": 3,
                    "cohort_counts": {"canary_applied": 2, "canary_holdout": 1, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0, "blocked_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"raw_prompts_included": False},
                },
                {
                    "candidate_id": "blocked-cli-action",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.01,
                    "sample_count": 0,
                    "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 0, "pass_count": 0, "fail_count": 0, "blocked_count": 0},
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing"],
                    "privacy": {"raw_prompts_included": False},
                },
            ],
        }
        stdout = io.StringIO()

        code = cli.optimization_promotion_actions_cli(
            ["--widen-step", "0.25", "--holdout-fraction", "0.1", "-"],
            stdin=io.StringIO(json.dumps(report)),
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_promotion_rollout_actions.v1")
        self.assertEqual(payload["summary"]["action_count"], 2)
        self.assertEqual(payload["summary"]["omitted_count"], 1)
        self.assertEqual(payload["summary"]["omission_bucket_count"], 1)
        self.assertEqual(payload["summary"]["top_omission_next_action"], "run-local-shadow-eval")
        sections = {row["policy_section"] for row in payload["actions"]}
        self.assertEqual(sections, {"routing", "cache"})
        bucket = payload["omission_buckets"][0]
        self.assertEqual(bucket["action_family"], "cache")
        self.assertEqual(bucket["next_action"], "run-local-shadow-eval")
        self.assertEqual(bucket["candidate_count"], 1)
        self.assertNotIn("blocked-cli-action", json.dumps(bucket, sort_keys=True))
        omitted = payload["omitted"][0]
        self.assertEqual(omitted["target_candidate_id"], "blocked-cli-action")
        self.assertEqual(omitted["reason"], "insufficient-eval-evidence")
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw promotion action prompt",
            "promotion-action-request-secret",
            "promotion-action-session-secret",
            "/tmp/promotion-action-secret.py",
            '"prompt"',
            '"request_id"',
            '"session_id"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])

    def test_optimization_promotion_blocker_review_cli_reads_fixture_and_scrubs_raw_fields(self):
        fixture = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [
                {
                    "recommendation_id": "promotion-blocker-next-action:openai:routing:eval",
                    "rank": 1,
                    "status": "recommended",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "openai_provider_request",
                    "provider_family": "openai",
                    "provider_endpoint": "responses",
                    "blocker_family": "eval-missing",
                    "blocker_reason_codes": ["missing-eval-evidence"],
                    "blocker_count": 30,
                    "recommendation_type": "collect-eval-evidence",
                    "next_action": "backfill-local-eval-evidence",
                    "expected_local_executor": "optimization-shadow-eval",
                    "file_backed_policy_representation": {
                        "exists": True,
                        "policy_section": "routing",
                        "policy_source": "local-manual",
                        "rule_file": "routing_rules.yaml",
                    },
                    "confidence": 0.91,
                    "projected_savings_usd": 4.5,
                    "prompt": "raw blocker prompt secret",
                    "provider_body": {"body": "raw provider body secret"},
                    "request_id": "blocker-request-id-secret",
                    "session_id": "blocker-session-id-secret",
                    "cache_key": "blocker-cache-key-secret",
                    "file_path": "/tmp/blocker-secret.py",
                },
                {
                    "recommendation_id": "promotion-blocker-next-action:codex:routing:canary",
                    "rank": 2,
                    "status": "noop",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "codex_turn",
                    "blocker_family": "canary-missing",
                    "blocker_reason_codes": ["missing-canary-evidence"],
                    "blocker_count": 3,
                    "recommendation_type": "noop",
                    "next_action": "keep-blocked",
                    "confidence": 0.5,
                    "no_op_reasons": ["provider-capability-canary_holdout-unavailable"],
                    "file_backed_policy_representation": {"exists": False},
                },
            ],
        }
        stdout = io.StringIO()

        code = cli.optimization_promotion_blocker_review_cli(
            ["-", "--pretty"],
            stdin=io.StringIO(json.dumps(fixture)),
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.promotion_blocker_recommendation_review.v1")
        self.assertEqual(payload["summary"]["review_candidate_count"], 2)
        self.assertEqual(payload["summary"]["recommended_count"], 1)
        self.assertEqual(payload["summary"]["noop_count"], 1)
        candidate = payload["groups"][0]["recommendations"][0]
        self.assertEqual(candidate["recommendation_type"], "collect-eval-evidence")
        self.assertEqual(candidate["expected_local_executor"], "optimization-shadow-eval")
        self.assertEqual(candidate["blocker_reason_codes"], ["missing-eval-evidence"])
        self.assertEqual(candidate["file_backed_policy_representation"]["rule_file"], "routing_rules.yaml")
        self.assertEqual(candidate["confidence"], 0.91)
        self.assertEqual(payload["omitted_actions"][0]["no_op_reasons"], ["provider-capability-canary-holdout-unavailable"])

        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw blocker prompt secret",
            "raw provider body secret",
            "blocker-request-id-secret",
            "blocker-session-id-secret",
            "blocker-cache-key-secret",
            "/tmp/blocker-secret.py",
            '"prompt"',
            '"provider_body"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_optimization_eval_queue_cli_queues_promotion_blocker_review_recommendations(self):
        from agentflow_proxy.store import Store

        recommendations = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [
                {
                    "recommendation_id": "promotion-blocker-next-action:openai:routing:eval-cli",
                    "rank": 1,
                    "status": "recommended",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "openai_provider_request",
                    "provider_family": "openai",
                    "provider_endpoint": "responses",
                    "blocker_family": "eval-missing",
                    "blocker_reason_codes": ["missing-eval-evidence", "eval-results-missing"],
                    "blocker_count": 30,
                    "recommendation_type": "collect-eval-evidence",
                    "next_action": "backfill-local-eval-evidence",
                    "expected_local_executor": "optimization-shadow-eval",
                    "file_backed_policy_representation": {
                        "exists": True,
                        "policy_section": "routing",
                        "policy_source": "local-manual",
                        "rule_file": "/tmp/routing-secret.yaml",
                    },
                    "confidence": 0.91,
                    "projected_savings_usd": 4.5,
                    "prompt": "raw queue blocker prompt secret",
                    "provider_body": {"body": "raw queue provider body secret"},
                    "request_id": "queue-blocker-request-id-secret",
                    "session_id": "queue-blocker-session-id-secret",
                    "cache_key": "queue-blocker-cache-key-secret",
                    "file_path": "/tmp/queue-blocker-secret.py",
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            stdout = io.StringIO()
            code = cli.optimization_eval_queue_cli(
                ["--db", db_path, "--promotion-blocker-review", "-", "--apply-backfill"],
                stdin=io.StringIO(json.dumps(recommendations)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            conn = sqlite3.connect(db_path)
            try:
                stored = conn.execute(
                    "select candidate_id, status_class, reason_codes_json, result_json from optimization_eval_results"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.promotion_recommendation_eval_queue.v1")
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["summary"]["written_task_count"], 1)
        self.assertEqual(stored[1], "queued")
        self.assertIn("eval-queued", json.loads(stored[2]))
        stored_result = json.loads(stored[3])
        self.assertEqual(stored_result["task"]["recommendation_id"], "promotion-blocker-next-action:openai:routing:eval-cli")
        rendered = json.dumps(payload, sort_keys=True) + stored[3]
        for forbidden in (
            "raw queue blocker prompt secret",
            "raw queue provider body secret",
            "queue-blocker-request-id-secret",
            "queue-blocker-session-id-secret",
            "queue-blocker-cache-key-secret",
            "/tmp/queue-blocker-secret.py",
            "/tmp/routing-secret.yaml",
            '"prompt"',
            '"provider_body"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_optimization_promotion_canary_apply_cli_dry_run_and_apply_routing_yaml(self):
        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "actions": [
                {
                    "schema": "agentflow.optimization_promotion_rollout_action.v1",
                    "action_id": "promotion-rollout-action:cli-routing",
                    "status": "planned",
                    "action_type": "widen",
                    "target_candidate_id": "routing-cli-canary",
                    "target_rule_id": "promotion-routing-cli",
                    "action_family": "routing",
                    "optimization_family": "phase_routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "policy_section": "routing",
                    "target_local_policy_section": "routing.rules",
                    "local_policy_update": {
                        "kind": "yaml-rule-canary",
                        "policy_source": "managed-recommended",
                        "managed_enforced": False,
                        "required_local_review": True,
                        "candidate_target_model": "claude-haiku-4-5-20251001",
                    },
                    "canary_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "privacy": {"metadata_only": True},
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()
            code = cli.optimization_promotion_canary_apply_cli(
                ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            dry_run = json.loads(stdout.getvalue())
            self.assertEqual(dry_run["schema"], "agentflow.optimization_promotion_canary_apply.v1")
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())

            stdout = io.StringIO()
            code = cli.optimization_promotion_canary_apply_cli(
                ["--config-dir", tmp, "--db", db_path, "--write", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            applied = json.loads(stdout.getvalue())
            self.assertTrue(applied["ok"])
            self.assertTrue(applied["wrote_policy_files"])
            data = yaml.safe_load((Path(tmp) / "routing_rules.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["phase_canary"]["policy_id"], "promotion-routing-cli")
            self.assertEqual(data["phase_canary"]["promotion_action_id"], "promotion-rollout-action:cli-routing")
            self.assertEqual(data["phase_canary"]["target_candidate_id"], "routing-cli-canary")
            self.assertEqual(data["phase_canary"]["canary_fraction"], 0.1)
            self.assertEqual(data["phase_canary"]["holdout_fraction"], 0.1)

    def test_optimization_promotion_canary_apply_queues_lifecycle_feedback_when_managed_disabled(self):
        from agentflow_proxy.store import Store

        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "actions": [
                {
                    "schema": "agentflow.optimization_promotion_rollout_action.v1",
                    "action_id": "promotion-rollout-action:cli-disabled-queue",
                    "status": "planned",
                    "action_type": "widen",
                    "target_candidate_id": "routing-cli-disabled-queue",
                    "target_rule_id": "promotion-routing-disabled-queue",
                    "action_family": "routing",
                    "optimization_family": "phase_routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "policy_section": "routing",
                    "target_local_policy_section": "routing.rules",
                    "local_policy_update": {
                        "kind": "yaml-rule-canary",
                        "policy_source": "managed-recommended",
                        "managed_enforced": False,
                        "required_local_review": True,
                        "candidate_target_model": "claude-haiku-4-5-20251001",
                    },
                    "canary_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "privacy": {"metadata_only": True},
                }
            ],
            "privacy": {"metadata_only": True},
        }
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_RECOMMENDATION_ENABLED": "0"}, clear=False):
                code = cli.optimization_promotion_canary_apply_cli(
                    ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )

            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "optimization_promotion_lifecycle"],
                stdout=status_stdout,
            )
            store = Store(db_path)
            try:
                row = store.managed_outcome_feedback_payload_rows(source_surface="optimization_promotion_lifecycle", limit=1)[0]
                queued_payload = json.loads(row["payload_json"])
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "queued")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        self.assertFalse(payload["managed_server_calls_made"])
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["queued"], 1)
        self.assertEqual(status_payload["oldest_pending"]["source_surface"], "optimization_promotion_lifecycle")
        lifecycle = status_payload["routing_promotion_lifecycle"]
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertEqual(lifecycle["action_count"], 1)
        self.assertEqual(lifecycle["candidate_id_breakdown"], [{"value": "routing-cli-disabled-queue", "count": 1}])
        self.assertEqual(lifecycle["action_type_breakdown"], [{"value": "widen", "count": 1}])
        self.assertEqual(lifecycle["model_family_pair_breakdown"], [{"value": "unknown->claude-haiku-4-5-20251001", "count": 1}])
        self.assertFalse(lifecycle["payload_json_included"])
        self.assertFalse(status_payload["privacy"]["payload_json_included"])
        self.assertEqual(queued_payload["event_type"], "dry-run")
        self.assertEqual(queued_payload["metadata"]["schema"], "agentflow.optimization_promotion_lifecycle_feedback.v1")
        self.assertEqual(queued_payload["metadata"]["candidate_ids"], ["routing-cli-disabled-queue"])
        self.assertEqual(queued_payload["metadata"]["action_snapshots"][0]["source_surface"], "anthropic_messages")
        self.assertEqual(queued_payload["metadata"]["action_snapshots"][0]["policy_source"], "managed-recommended")
        self.assertEqual(queued_payload["metadata"]["action_snapshots"][0]["routed_model_family"], "claude-haiku-4-5-20251001")
        self.assertFalse(queued_payload["metadata"]["privacy"]["file_paths_included"])

    def test_optimization_promotion_canary_apply_retries_lifecycle_feedback_without_payload_output(self):
        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "actions": [
                {
                    "schema": "agentflow.optimization_promotion_rollout_action.v1",
                    "action_id": "promotion-rollout-action:cli-retry",
                    "status": "planned",
                    "action_type": "widen",
                    "target_candidate_id": "routing-cli-retry",
                    "target_rule_id": "promotion-routing-retry",
                    "action_family": "routing",
                    "optimization_family": "phase_routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "policy_section": "routing",
                    "target_local_policy_section": "routing.rules",
                    "local_policy_update": {
                        "kind": "yaml-rule-canary",
                        "policy_source": "managed-recommended",
                        "candidate_target_model": "claude-haiku-4-5-20251001",
                    },
                    "canary_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "privacy": {"metadata_only": True},
                }
            ],
        }
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.optimization_promotion_canary_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "--write", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )

            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "optimization_promotion_lifecycle"],
                stdout=status_stdout,
            )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "apply")
        self.assertEqual(sent_payload["metadata"]["command"], "optimization-promotion-apply")
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "retryable-error")
        self.assertEqual(output["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["retryable_error"], 1)
        lifecycle = status_payload["routing_promotion_lifecycle"]
        self.assertEqual(lifecycle["queue_state_breakdown"], [{"value": "pending", "count": 1}])
        self.assertEqual(lifecycle["candidate_id_breakdown"], [{"value": "routing-cli-retry", "count": 1}])
        self.assertFalse(lifecycle["payload_json_included"])
        self.assertFalse(status_payload["privacy"]["payload_json_included"])

    def test_optimization_promotion_canary_apply_rejects_raw_like_input_without_leaking_feedback(self):
        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "actions": [
                {
                    "schema": "agentflow.optimization_promotion_rollout_action.v1",
                    "action_id": "promotion-rollout-action:cli-raw-rejected",
                    "status": "planned",
                    "action_type": "widen",
                    "target_candidate_id": "routing-cli-raw-rejected",
                    "target_rule_id": "promotion-routing-raw-rejected",
                    "action_family": "routing",
                    "optimization_family": "phase_routing",
                    "policy_section": "routing",
                    "local_policy_update": {
                        "kind": "yaml-rule-canary",
                        "policy_source": "managed-recommended",
                        "candidate_target_model": "claude-haiku-4-5-20251001",
                        "raw_prompt": "raw promotion lifecycle secret",
                    },
                    "canary_fraction": 0.1,
                    "holdout_fraction": 0.1,
                }
            ],
        }
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stderr = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.optimization_promotion_canary_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=io.StringIO(),
                        stderr=stderr,
                    )

        self.assertEqual(code, 1)
        output = json.loads(stderr.getvalue())
        self.assertFalse(output["ok"])
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["json"]["event_type"], "dry-run")
        rendered = json.dumps(ManagedFeedbackFlushClient.calls[0]["json"], sort_keys=True)
        self.assertNotIn("raw promotion lifecycle secret", rendered)
        self.assertNotIn('"raw_prompt"', rendered)

    def test_optimization_promotion_impact_cli_reports_post_apply_canary_metadata(self):
        from agentflow_proxy.store import Store, stable_json

        action = {
            "schema": "agentflow.optimization_promotion_rollout_action.v1",
            "action_id": "promotion-rollout-action:cli-impact",
            "action_type": "widen",
            "target_candidate_id": "routing-cli-impact",
            "target_rule_id": "promotion-routing-cli-impact",
            "action_family": "routing",
            "optimization_family": "phase_routing",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "policy_section": "routing",
            "target_local_policy_section": "routing.rules",
            "canary_fraction": 0.25,
            "holdout_fraction": 0.10,
            "evidence_summary": {
                "projected_savings_usd": 0.004,
                "sample_count": 2,
                "cohort_counts": {"canary_applied": 1, "canary_holdout": 1, "bypassed_or_disabled": 0},
            },
            "local_policy_update": {"policy_source": "managed-recommended"},
            "privacy": {"metadata_only": True},
        }
        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T00:00:00+00:00",
            "ok": True,
            "actions": [action],
            "privacy": {"metadata_only": True},
        }
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for cohort, suffix, cost_est, cost_baseline in (
                    ("canary_applied", "a", 0.001, 0.003),
                    ("canary_holdout", "h", 0.003, 0.003),
                ):
                    store.log_call(
                        id=f"cli-promotion-impact-{suffix}",
                        created_at=f"2026-06-10T00:10:0{0 if suffix == 'a' else 1}+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001" if cohort == "canary_applied" else "claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=1000,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost_est,
                        cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({
                            "category": "tool-result",
                            "phase_canary": {
                                "promotion_action_id": action["action_id"],
                                "target_candidate_id": action["target_candidate_id"],
                                "target_rule_id": action["target_rule_id"],
                                "policy_section": "routing",
                                "policy_source": "managed-recommended",
                                "status": "applied" if cohort == "canary_applied" else "holdout",
                                "cohort": cohort,
                                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                            },
                        }),
                        cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                        error=None,
                        request_json=None,
                        response_json=None,
                        session_id="cli-impact-session-secret",
                        category="tool-result",
                        retry_count=0,
                        provider="anthropic",
                        source_surface="anthropic_messages",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.optimization_promotion_impact_cli(
                ["--db", db_path, "--limit", "10", "--min-applied-samples", "1", "--min-holdout-samples", "1", "--max-evidence-age-hours", "999999", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            store = Store(db_path)
            try:
                feedback_rows = store.promotion_outcome_feedback_rows(limit=10)
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_promotion_impact.v1")
        self.assertTrue(payload["wrote_store"])
        self.assertEqual(payload["summary"]["actual_canary_applied_count"], 1)
        self.assertEqual(payload["summary"]["actual_canary_holdout_count"], 1)
        self.assertEqual(payload["actions"][0]["next_step"]["verdict"], "widen")
        self.assertEqual(payload["actions"][0]["next_step"]["recommendation"], "promote")
        self.assertEqual(payload["family_impacts"][0]["action_family"], "routing")
        self.assertEqual(payload["family_impacts"][0]["recommendation"], "promote")
        self.assertEqual(payload["family_impacts"][0]["cohort_metrics"]["canary_applied"]["count"], 1)
        self.assertEqual(payload["family_impacts"][0]["cohort_metrics"]["canary_holdout"]["count"], 1)
        self.assertEqual(payload["summary"]["recommendation_counts"], [{"value": "promote", "count": 1}])
        self.assertEqual(payload["promotion_outcome_feedback"]["schema"], "agentflow.promotion_outcome_feedback_ledger.v1")
        self.assertEqual(payload["promotion_outcome_feedback"]["summary"]["rows_written"], 1)
        self.assertEqual(payload["promotion_outcome_feedback"]["entries"][0]["status"], "positive")
        self.assertEqual(len(feedback_rows), 1)
        self.assertEqual(feedback_rows[0]["policy_id"], action["target_rule_id"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("cli-impact-session-secret", rendered)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])

    def test_optimization_promotion_impact_queues_lifecycle_feedback_with_next_step_summary(self):
        from agentflow_proxy.store import Store, stable_json

        action = {
            "schema": "agentflow.optimization_promotion_rollout_action.v1",
            "action_id": "promotion-rollout-action:cli-impact-feedback",
            "action_type": "widen",
            "target_candidate_id": "routing-cli-impact-feedback",
            "target_rule_id": "promotion-routing-cli-impact-feedback",
            "action_family": "routing",
            "optimization_family": "phase_routing",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "policy_section": "routing",
            "target_local_policy_section": "routing.rules",
            "canary_fraction": 0.25,
            "holdout_fraction": 0.10,
            "evidence_summary": {
                "projected_savings_usd": 0.004,
                "sample_count": 2,
                "cohort_counts": {"canary_applied": 1, "canary_holdout": 1, "bypassed_or_disabled": 0},
            },
            "local_policy_update": {"policy_source": "managed-recommended"},
            "privacy": {"metadata_only": True},
        }
        bundle = {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T00:00:00+00:00",
            "ok": True,
            "actions": [action],
            "privacy": {"metadata_only": True},
        }
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable with raw impact secret"
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for cohort, suffix, cost_est, cost_baseline in (
                    ("canary_applied", "a", 0.001, 0.003),
                    ("canary_holdout", "h", 0.003, 0.003),
                ):
                    store.log_call(
                        id=f"cli-promotion-impact-feedback-{suffix}",
                        created_at=f"2026-06-10T00:10:0{0 if suffix == 'a' else 1}+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001" if cohort == "canary_applied" else "claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=1000,
                        input_tokens_est=100,
                        output_tokens_est=10,
                        actual_input_tokens=100,
                        actual_output_tokens=10,
                        cost_est_usd=cost_est,
                        cost_baseline_usd=cost_baseline,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({
                            "phase_canary": {
                                "promotion_action_id": action["action_id"],
                                "target_candidate_id": action["target_candidate_id"],
                                "target_rule_id": action["target_rule_id"],
                                "policy_section": "routing",
                                "policy_source": "managed-recommended",
                                "status": "applied" if cohort == "canary_applied" else "holdout",
                                "cohort": cohort,
                                "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                            },
                        }),
                        cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                        error=None,
                        request_json=None,
                        response_json=None,
                        session_id="cli-impact-feedback-session-secret",
                        category="tool-result",
                        retry_count=0,
                        provider="anthropic",
                        source_surface="anthropic_messages",
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.optimization_promotion_impact_cli(
                        ["--db", db_path, "--limit", "10", "--min-applied-samples", "1", "--min-holdout-samples", "1", "--max-evidence-age-hours", "999999", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )

            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "optimization_promotion_lifecycle"],
                stdout=status_stdout,
            )

        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "retryable-error")
        self.assertTrue(output["managed_server_calls_made"])
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "impact")
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["schema"], "agentflow.optimization_promotion_lifecycle_feedback.v1")
        self.assertEqual(metadata["actual_canary_applied_count"], 1)
        self.assertEqual(metadata["actual_canary_holdout_count"], 1)
        self.assertEqual(metadata["reason_code_counts"]["promotion-impact-positive"], 1)
        self.assertEqual(metadata["action_snapshots"][0]["next_step_verdict"], "widen")
        self.assertEqual(metadata["recommendation_counts"], [{"value": "promote", "count": 1}])
        self.assertEqual(metadata["family_impacts"][0]["action_family"], "routing")
        self.assertEqual(metadata["family_impacts"][0]["recommendation"], "promote")
        self.assertFalse(metadata["privacy"]["request_ids_included"])
        rendered = json.dumps(sent_payload, sort_keys=True)
        self.assertNotIn("cli-impact-feedback-session-secret", rendered)
        self.assertNotIn("raw impact secret", rendered)
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["retryable_error"], 1)
        lifecycle = status_payload["routing_promotion_lifecycle"]
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertEqual(lifecycle["cohort_count_breakdown"], [
            {"value": "canary_applied", "count": 1},
            {"value": "canary_holdout", "count": 1},
        ])
        self.assertEqual(lifecycle["outcome_status_breakdown"], [{"value": "widen", "count": 1}])
        self.assertEqual(lifecycle["candidate_breakdown"][0]["observed_savings_usd"], 0.002)
        self.assertFalse(status_payload["privacy"]["payload_json_included"])

    def test_optimization_rollout_actions_review_cli_accepts_signed_bundle_and_logs_event(self):
        from agentflow_proxy.optimization_rollout_review import attach_optimization_rollout_provenance

        bundle = {
            "schema": "agentflow.optimization_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "expires_at": "2099-06-11T05:00:00+00:00",
            "summary": {
                "candidate_count": 1,
                "action_count": 1,
                "omitted_count": 0,
                "managed_enforced": False,
                "required_local_review": True,
                "provider_forwarding": False,
                "server_content_processing": False,
            },
            "local_executor_compatibility": {
                "minimum_local_client_version": "0.1.0",
                "compatible": True,
                "supported_local_action_families": ["routing", "crunch", "cache", "old_context_summarization"],
                "local_review_required": True,
            },
            "actions": [
                {
                    "schema": "agentflow.optimization_rollout_action.v1",
                    "action_id": "optimization-rollout-action:cli-routing",
                    "action_type": "widen",
                    "target_candidate_id": "cli-routing-candidate",
                    "action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "policy_section": "routing",
                    "source_surface": "openai_responses",
                    "provider_endpoint": "responses",
                    "confidence": 0.93,
                    "generated_at": "2026-06-10T05:00:00+00:00",
                    "expires_at": "2099-06-11T05:00:00+00:00",
                    "required_local_review": True,
                    "managed_enforced": False,
                    "local_executor_compatibility": {
                        "minimum_local_client_version": "0.1.0",
                        "compatible": True,
                        "supported_local_action_families": ["routing", "crunch", "cache", "old_context_summarization"],
                    },
                    "evidence_summary": {
                        "local_eval_verdict": {"verdict": "widen", "latest_eval_at": "2026-06-10T04:30:00+00:00"}
                    },
                    "action": {
                        "schema": "agentflow.openai_rollout_action.v1",
                        "target_rule_id": "cli-openai-routing-rule",
                        "proposed_edit": {
                            "rule_id": "cli-openai-routing-rule",
                            "changed": True,
                            "action": {"route_to": "gpt-5-mini"},
                        },
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "feature_only": True,
                        "raw_payloads_returned": False,
                        "raw_prompts_returned": False,
                        "raw_responses_returned": False,
                        "provider_bodies_returned": False,
                        "request_ids_returned": False,
                        "tenant_ids_returned": False,
                        "cache_keys_returned": False,
                        "file_paths_returned": False,
                        "provider_forwarding": False,
                        "managed_enforced": False,
                    },
                }
            ],
            "omitted_actions": [],
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
                "provider_forwarding": False,
                "managed_enforced": False,
            },
        }
        signed = attach_optimization_rollout_provenance(
            bundle,
            secret="cli-review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="cli-review",
            generated_at="2026-06-10T05:00:00+00:00",
        )
        stdout = io.StringIO()

        with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "cli-review-secret"}):
            code = cli.optimization_rollout_actions_review_cli(
                ["--pretty", "-"],
                stdin=io.StringIO(json.dumps(signed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.optimization_rollout_actions_review.v1")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertEqual(payload["provenance"]["status"], "verified")
        self.assertEqual(payload["actions"][0]["target_rule_id"], "cli-openai-routing-rule")

        events_path = Path(os.environ["AGENTFLOW_POLICY_EVENTS_LOG"])
        event = json.loads(events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(event["action"], "optimization-rollout-actions-review")
        self.assertTrue(event["ok"])
        self.assertEqual(event["details"]["accepted_action_count"], 1)
        self.assertFalse(event["details"]["wrote_local_policy_files"])

    def test_optimization_shadow_eval_cli_requires_budget_for_execute(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.optimization_shadow_eval_cli(
            ["--execute", "-"],
            stdin=io.StringIO(json.dumps({"schema": "agentflow.optimization_eval_plan.v1", "plans": []})),
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "missing_budget_cap")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])

    def test_optimization_eval_queue_cli_requires_budget_for_execute(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = cli.optimization_eval_queue_cli(
            ["--execute", "--limit", "1"],
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "missing_budget_cap")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])

    def test_optimization_shadow_eval_cli_enforces_budget_cap_without_policy_mutation(self):
        from agentflow_proxy.store import Store

        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-budget-blocked",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "evidence": {"estimated_eval_cost_usd": 0.25},
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            Store(db_path).conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            policy_path = Path(tmp) / "routing_rules.yaml"
            original_policy = "enabled: true\nrules: []\n"
            policy_path.write_text(original_policy, encoding="utf-8")
            stdout = io.StringIO()

            code = cli.optimization_shadow_eval_cli(
                ["--db", db_path, "--execute", "--budget-usd", "0.01", str(plan_path)],
                stdout=stdout,
            )

            policy_after = policy_path.read_text(encoding="utf-8")
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "select status_class, reason_codes_json, result_json from optimization_eval_results"
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(policy_after, original_policy)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "execute")
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertEqual(payload["summary"]["budget_exhausted_count"], 1)
        self.assertEqual(payload["results"][0]["status_class"], "blocked")
        self.assertIn("budget-cap-exceeded", payload["results"][0]["reason_codes"])
        self.assertEqual(row[0], "blocked")
        self.assertIn("budget-cap-exceeded", json.loads(row[1]))
        stored_result = json.loads(row[2])
        self.assertFalse(stored_result["provider_call_made"])
        self.assertNotIn("request_json", json.dumps(stored_result))

    def test_policy_validate_cli_accepts_exported_bundle_from_stdin(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO(exported.getvalue()), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(payload["errors"], [])

    def test_policy_validate_cli_rejects_invalid_json_with_structured_errors(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertIn("invalid JSON", payload["errors"][0]["message"])

    def test_policy_validate_cli_rejects_missing_file_with_structured_errors(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["/tmp/agentflow-no-such-policy-bundle.json"], stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(payload["errors"][0]["path"], "/tmp/agentflow-no-such-policy-bundle.json")
        self.assertIn("No such file", payload["errors"][0]["message"])

    def test_policy_validate_cli_rejects_malformed_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(
            ["-"],
            stdin=io.StringIO(json.dumps({"schema": "wrong"})),
            stdout=stdout,
        )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("$.schema", {error["path"] for error in payload["errors"]})

    def test_policy_diff_cli_reports_file_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["changes"][0]["path"], "$.policies.routing.enabled")

    def test_policy_diff_cli_accepts_one_stdin_input(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO(json.dumps(before)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["changed"])

    def test_policy_diff_cli_rejects_invalid_json_with_structured_errors(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(exported.getvalue(), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_diff.v1")
        self.assertIn("invalid JSON", payload["before_validation"]["errors"][0]["message"])

    def test_policy_review_cli_reports_current_to_proposed_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["enabled"] = not proposed["policies"]["routing"]["enabled"]
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["change_count"], 1)
        self.assertEqual(payload["safety_warning_count"], 0)
        self.assertIn("impact_summary", payload)

    def test_policy_review_cli_simulates_routing_impact_from_test_db(self):
        from agentflow_proxy.store import Store, stable_json

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"] = [{
            "conditions": {"model_pattern": "sonnet", "category": "chat"},
            "action": {"route_to": "haiku", "reason": "test chat downgrade"},
        }]

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="review-match-ok",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_call(
                    id="review-match-thinking",
                    created_at="2026-06-08T10:01:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1200,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                    thinking_output_tokens=40,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        impact = payload["impact_summary"]
        self.assertEqual(impact["status"], "simulated")
        self.assertTrue(impact["metadata_only"])
        self.assertFalse(impact["raw_bodies_read"])
        rule = impact["sections"]["routing"]["rules"][0]
        self.assertEqual(rule["would_match_count"], 1)
        self.assertEqual(rule["excluded_thinking_count"], 1)
        self.assertGreater(rule["estimated_savings_usd"], 0)

    def test_policy_review_cli_reports_missing_db_impact_unavailable(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())

        with TemporaryDirectory() as tmp:
            missing_db = str(Path(tmp) / "missing.sqlite3")
            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", missing_db],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        impact = json.loads(stdout.getvalue())["impact_summary"]
        self.assertEqual(impact["status"], "unavailable")
        self.assertEqual(impact["reason"], "db-not-found")

    def test_policy_review_cli_generates_high_risk_impact_warning(self):
        from agentflow_proxy.store import Store, stable_json

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"] = [{
            "conditions": {"model_pattern": "sonnet", "category": "chat"},
            "action": {"route_to": "haiku", "reason": "test risky downgrade"},
        }]

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for index, status in enumerate((200, 500)):
                    store.log_call(
                        id=f"review-risk-{index}",
                        created_at=f"2026-06-08T10:0{index}:00+00:00",
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=status,
                        latency_ms=1000,
                        input_tokens_est=1000,
                        output_tokens_est=100,
                        actual_input_tokens=1000,
                        actual_output_tokens=100,
                        cost_est_usd=0.0045,
                        routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                        cache_json=stable_json({"status": "miss"}),
                        retry_count=1 if status >= 400 else 0,
                    )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        impact = json.loads(stdout.getvalue())["impact_summary"]
        warning_codes = {warning["code"] for warning in impact["warnings"]}
        self.assertIn("high-error-rate-routing-match", warning_codes)

    def _old_context_summary_bundle(self) -> dict:
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["crunch"]["old_context_summarization"].update({
            "enabled": True,
            "rule_id": "test-old-context-dry-run",
            "model": "claude-haiku-4-5-20251001",
            "min_request_chars": 1,
            "min_summarized_chars": 10,
            "max_turns": 3,
            "keep_recent_turns": 1,
            "max_summary_chars": 80,
            "max_source_chars": 10000,
            "max_summary_cost_usd": 1.0,
            "excluded_categories": [],
            "block_tool_protocol": True,
            "block_thinking": True,
        })
        return proposed

    def _write_old_context_summary_dry_run_rows(self, db_path: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        old_text = "raw-secret-old-context " + ("durable fact " * 260)
        eligible_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": old_text},
                {"role": "assistant", "content": "decision: keep src/app.py behavior stable " * 120},
                {"role": "user", "content": "recent request stays verbatim"},
            ],
        }
        tool_blocked_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-secret", "name": "Read", "input": {"file": "secret.py"}}]},
                {"role": "user", "content": "recent request"},
            ],
        }
        store = Store(db_path)
        try:
            store.log_call(
                id="old-summary-eligible",
                created_at="2026-06-08T10:00:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=2000,
                output_tokens_est=100,
                actual_input_tokens=2000,
                actual_output_tokens=100,
                cost_est_usd=0.007,
                routing_json=stable_json({"category": "chat", "text_chars": len(stable_json(eligible_body)), "has_tools": False}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                request_json=stable_json(eligible_body),
                session_id="session-secret-should-not-leak",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
            store.log_call(
                id="old-summary-tool-blocked",
                created_at="2026-06-08T10:01:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=1000,
                actual_output_tokens=100,
                cost_est_usd=0.004,
                routing_json=stable_json({"category": "chat", "text_chars": len(stable_json(tool_blocked_body)), "has_tools": True}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "miss"}),
                request_json=stable_json(tool_blocked_body),
                session_id="blocked-session-secret",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
            store.log_call(
                id="old-summary-no-body",
                created_at="2026-06-08T10:02:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=1000,
                output_tokens_est=100,
                actual_input_tokens=1000,
                actual_output_tokens=100,
                cost_est_usd=0.004,
                routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "miss"}),
                request_json=None,
                session_id="missing-body-session-secret",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
        finally:
            store.conn.close()

    def test_old_context_summary_dry_run_cli_reports_metadata_only_projection(self):
        proposed = self._old_context_summary_bundle()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()

            code = cli.old_context_summary_dry_run_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.old_context_summary_dry_run.v1")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["policy"]["rule_id"], "test-old-context-dry-run")
        self.assertEqual(payload["summary"]["sampled_call_count"], 3)
        self.assertEqual(payload["summary"]["request_body_available_count"], 2)
        self.assertEqual(payload["summary"]["eligible_call_count"], 1)
        self.assertGreater(payload["summary"]["eligible_chars"], 10)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["estimated_summary_cost_usd"], 0)
        self.assertGreater(payload["summary"]["projected_gross_savings_usd"], 0)
        skip_reasons = {row["reason"] for row in payload["summary"]["skip_reasons"]}
        self.assertIn("tool-protocol-context-blocked", skip_reasons)
        self.assertIn("request-body-unavailable", skip_reasons)
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", encoded)
        self.assertNotIn("session-secret-should-not-leak", encoded)
        self.assertNotIn("tool-secret", encoded)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "disabled")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])

    def _write_tool_protocol_aware_old_context_rows(self, db_path: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        mixed_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "RAW_TOOL_AWARE_SECRET old durable constraint " * 1200},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_raw_should_not_leak",
                            "name": "Read",
                            "input": {"file_path": "/private/project/secret.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_raw_should_not_leak",
                            "content": "RAW_TOOL_RESULT_SECRET",
                        }
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "old visible conclusion remains true " * 1200}]},
                {"role": "user", "content": "recent plain follow-up stays in request"},
                {"role": "assistant", "content": "recent answer stays in request"},
                {"role": "user", "content": "another recent turn stays in request"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_recent_should_not_leak",
                            "content": "recent result stays in forwarded request",
                        }
                    ],
                },
            ],
        }
        thinking_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "RAW_THINKING_SECRET " * 1200},
                        {"type": "text", "text": "older visible answer " * 500},
                    ],
                },
                {"role": "user", "content": "old text that would otherwise qualify " * 1200},
                {"role": "assistant", "content": "older visible answer without thinking " * 400},
                {"role": "user", "content": "recent turn one"},
                {"role": "assistant", "content": "recent turn two"},
                {"role": "user", "content": "recent turn three"},
                {"role": "user", "content": "recent request"},
            ],
        }
        store = Store(db_path)
        try:
            for idx, body in enumerate((mixed_body, mixed_body)):
                store.log_call(
                    id=f"tool-aware-eligible-{idx}",
                    created_at=f"2026-06-08T10:0{idx}:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=len(stable_json(body)) // 4,
                    output_tokens_est=100,
                    actual_input_tokens=len(stable_json(body)) // 4,
                    actual_output_tokens=100,
                    cost_est_usd=0.05,
                    routing_json=stable_json({"category": "tool-result", "text_chars": len(stable_json(body)), "has_tools": True}),
                    crunch_json=stable_json({}),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    request_json=stable_json(body),
                    session_id="tool-aware-session-secret",
                    category="tool-result",
                    retry_count=0,
                    provider="anthropic",
                )
            store.log_call(
                id="tool-aware-thinking-blocked",
                created_at="2026-06-08T10:02:00+00:00",
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1000,
                input_tokens_est=len(stable_json(thinking_body)) // 4,
                output_tokens_est=100,
                actual_input_tokens=len(stable_json(thinking_body)) // 4,
                actual_output_tokens=100,
                cost_est_usd=0.05,
                routing_json=stable_json({"category": "chat", "text_chars": len(stable_json(thinking_body)), "has_tools": False}),
                crunch_json=stable_json({}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                request_json=stable_json(thinking_body),
                session_id="thinking-session-secret",
                category="chat",
                retry_count=0,
                provider="anthropic",
            )
        finally:
            store.conn.close()

    def test_old_context_summary_dry_run_tool_protocol_aware_profile_reports_plateau_eligibility(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_tool_protocol_aware_old_context_rows(db_path)
            stdout = io.StringIO()

            code = cli.old_context_summary_dry_run_cli(
                ["--db", db_path, "--limit", "10", "--profile", "tool-protocol-aware"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["policy"]["dry_run_profile"], "tool-protocol-aware")
        self.assertTrue(payload["policy"]["enabled"])
        self.assertFalse(payload["policy"]["block_tool_protocol"])
        self.assertEqual(
            payload["policy"]["tool_protocol_handling"]["summary_source"],
            "non-tool-text-turns-only",
        )
        self.assertEqual(payload["summary"]["eligible_call_count"], 2)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["projected_net_savings_usd"], 0)
        self.assertEqual(payload["summary"]["plateau_policy"]["min_text_chars"], 8000)

        eligible_groups = [
            group for group in payload["groups"]
            if group["blocker"] == "eligible" and group["category"] == "tool-result"
        ]
        self.assertEqual(len(eligible_groups), 1)
        self.assertEqual(eligible_groups[0]["plateau_status"], "plateau-adjacent")
        self.assertEqual(eligible_groups[0]["tool_protocol_blocker"], "preserved")
        self.assertEqual(eligible_groups[0]["thinking_blocker"], "clear")
        self.assertEqual(eligible_groups[0]["eligible_call_count"], 2)

        thinking_groups = [group for group in payload["groups"] if group["blocker"] == "thinking-context-blocked"]
        self.assertEqual(len(thinking_groups), 1)
        self.assertEqual(thinking_groups[0]["thinking_blocker"], "blocked")
        skip_reasons = {row["reason"] for row in payload["summary"]["skip_reasons"]}
        self.assertIn("thinking-context-blocked", skip_reasons)

        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "RAW_TOOL_AWARE_SECRET",
            "RAW_TOOL_RESULT_SECRET",
            "RAW_THINKING_SECRET",
            "toolu_raw_should_not_leak",
            "/private/project/secret.txt",
            "tool-aware-session-secret",
            "thinking-session-secret",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])

    def test_old_context_summary_dry_run_sends_metadata_only_lifecycle_feedback(self):
        proposed = self._old_context_summary_bundle()
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_dry_run_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(proposed)),
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "dry-run")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["lifecycle_kind"], "old_context_summarization")
        self.assertEqual(metadata["rule_id"], "test-old-context-dry-run")
        self.assertEqual(metadata["eligible_call_count"], 1)
        self.assertFalse(metadata["privacy"]["cache_keys_included"])
        rendered_payload = json.dumps(sent_payload, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", rendered_payload)
        self.assertNotIn("session-secret-should-not-leak", rendered_payload)
        self.assertNotIn("tool-secret", rendered_payload)
        self.assertNotIn("secret.py", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertEqual(output["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])

    def _old_context_summary_impact_dry_run(self) -> dict:
        return {
            "schema": "agentflow.old_context_summary_dry_run.v1",
            "ok": True,
            "dry_run": True,
            "read_only": True,
            "generated_at": "2026-06-08T09:00:00+00:00",
            "policy": {
                "policy_source": "managed-recommended",
                "rule_id": "test-old-context-impact",
                "candidate_id": "candidate-old-context-impact",
                "model": "claude-haiku-4-5-20251001",
                "canary": {"enabled": True, "fraction": 0.5, "unit": "source_hash"},
                "safety_stop": {"enabled": True},
            },
            "summary": {
                "eligible_call_count": 4,
                "projected_saved_chars": 16000,
                "projected_saved_tokens": 4000,
                "estimated_summary_cost_usd": 0.001,
                "projected_gross_savings_usd": 0.012,
                "projected_net_savings_usd": 0.011,
            },
            "groups": [
                {
                    "source_surface": "anthropic_messages",
                    "category": "chat",
                    "model_tier": "sonnet",
                    "stream": True,
                    "blocker": "eligible",
                    "call_count": 4,
                    "eligible_call_count": 4,
                }
            ],
            "privacy": {
                "metadata_only_output": True,
                "raw_prompts_included": False,
                "raw_request_bodies_included": False,
                "raw_session_ids_included": False,
                "cache_keys_included": False,
            },
        }

    def _write_old_context_summary_impact_rows(self, db_path: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        store = Store(db_path)
        try:
            rows = [
                (
                    "impact-applied",
                    "2026-06-08T10:00:00+00:00",
                    200,
                    2200,
                    0,
                    {
                        "enabled": True,
                        "status": "applied",
                        "reason": "summary-created",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "before_chars": 40000,
                        "eligible_chars": 30000,
                        "eligible_turns": 3,
                        "saved_chars": 8000,
                        "tokens_saved_est": 2000,
                        "estimated_gross_savings_usd": 0.006,
                        "summary_cost_est_usd": 0.0004,
                        "estimated_net_savings_usd": 0.0056,
                        "summary_cache_hit": False,
                        "summary_status_code": 200,
                        "canary": {"enabled": True, "cohort": "canary_applied", "selected": True, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-holdout",
                    "2026-06-08T10:01:00+00:00",
                    200,
                    1100,
                    0,
                    {
                        "enabled": True,
                        "status": "skipped",
                        "reason": "canary_holdout",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "eligible_chars": 31000,
                        "eligible_turns": 3,
                        "canary": {"enabled": True, "cohort": "canary_holdout", "selected": False, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-bypass",
                    "2026-06-08T10:02:00+00:00",
                    500,
                    900,
                    2,
                    {
                        "enabled": True,
                        "status": "bypass",
                        "reason": "local-canary-safety-stop",
                        "rule_id": "test-old-context-impact",
                        "candidate_id": "candidate-old-context-impact",
                        "policy_source": "managed-recommended",
                        "category": "chat",
                        "summary_status_code": 500,
                        "summary_error": "redacted upstream error bucket only",
                        "safety_stop_state": "stopped",
                        "canary": {"enabled": True, "cohort": "bypassed", "selected": False, "fraction": 0.5, "unit": "source_hash"},
                    },
                ),
                (
                    "impact-other",
                    "2026-06-08T10:03:00+00:00",
                    200,
                    800,
                    0,
                    {
                        "enabled": True,
                        "status": "applied",
                        "reason": "summary-created",
                        "rule_id": "different-rule",
                        "candidate_id": "different-candidate",
                        "category": "chat",
                        "canary": {"enabled": True, "cohort": "canary_applied"},
                    },
                ),
            ]
            for call_id, created_at, status_code, latency_ms, retry_count, meta in rows:
                store.log_call(
                    id=call_id,
                    created_at=created_at,
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    input_tokens_est=10000,
                    output_tokens_est=200,
                    actual_input_tokens=9000,
                    actual_output_tokens=180,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.04,
                    routing_json=stable_json({"category": "chat", "text_chars": 40000, "has_tools": False}),
                    crunch_json=stable_json({
                        "changed": meta.get("status") == "applied",
                        "old_context_summarization": meta,
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    request_json=stable_json({"messages": [{"content": "raw impact secret must not leak"}]}),
                    response_json=stable_json({"content": "generated summary secret must not leak"}),
                    session_id="impact-session-secret",
                    category="chat",
                    retry_count=retry_count,
                    provider="anthropic",
                    error="raw error secret must not leak" if status_code >= 400 else None,
                )
        finally:
            store.conn.close()

    def _old_context_summary_quality_gate_dry_run(self) -> dict:
        dry_run = self._old_context_summary_impact_dry_run()
        dry_run["policy"]["rule_id"] = "test-old-context-quality-gate"
        dry_run["policy"]["candidate_id"] = "candidate-old-context-quality-gate"
        dry_run["policy"]["safety_gates"] = {
            "min_outcome_samples": 4,
            "min_canary_applied_samples": 2,
            "min_canary_holdout_samples": 2,
            "min_net_savings_usd": 0.0,
            "min_payback_ratio": 1.0,
            "min_projection_realization_ratio": 0.5,
            "max_error_rate": 0.05,
            "max_error_rate_delta": 0.0,
            "max_retry_rate": 0.10,
            "max_retry_rate_delta": 0.10,
            "max_summary_failure_rate": 0.02,
            "max_bypass_or_disabled_rate": 0.10,
            "max_safety_stop_count": 0,
            "max_latency_regression_ms": 2000,
            "rollback_error_rate": 0.40,
            "rollback_summary_failure_rate": 0.20,
            "rollback_safety_stop_count": 1,
            "rollback_negative_net_savings_usd": 0.0,
        }
        dry_run["summary"]["eligible_call_count"] = 4
        dry_run["summary"]["projected_saved_tokens"] = 2000
        dry_run["summary"]["projected_net_savings_usd"] = 0.004
        return dry_run

    def _write_old_context_summary_quality_gate_rows(self, db_path: str, *, scenario: str) -> None:
        from agentflow_proxy.store import Store, stable_json

        def meta_for(cohort: str, *, status_code: int = 200, retry_count: int = 0) -> dict:
            applied = cohort == "canary_applied"
            meta = {
                "enabled": True,
                "status": "applied" if applied else "skipped",
                "reason": "summary-created" if applied else "canary_holdout",
                "rule_id": "test-old-context-quality-gate",
                "candidate_id": "candidate-old-context-quality-gate",
                "policy_source": "managed-recommended",
                "category": "chat",
                "eligible_chars": 32000,
                "eligible_turns": 3,
                "canary": {
                    "enabled": True,
                    "cohort": cohort,
                    "selected": applied,
                    "fraction": 0.5,
                    "unit": "source_hash",
                },
            }
            if applied:
                meta.update({
                    "before_chars": 40000,
                    "saved_chars": 4000,
                    "tokens_saved_est": 1000,
                    "estimated_gross_savings_usd": 0.003,
                    "summary_cost_est_usd": 0.001,
                    "estimated_net_savings_usd": 0.002,
                    "summary_status_code": 200 if status_code < 400 else status_code,
                    "summary_cache_hit": False,
                })
            if status_code >= 400:
                meta["summary_error"] = "redacted summary failure bucket"
            if retry_count:
                meta["retry_bucket"] = "retried"
            return meta

        if scenario == "insufficient-evidence":
            rows = [("applied-0", "canary_applied", 200, 0, 1000)]
        elif scenario == "hold":
            rows = [
                ("applied-0", "canary_applied", 200, 1, 1000),
                ("applied-1", "canary_applied", 200, 1, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]
        elif scenario == "rollback":
            rows = [
                ("applied-0", "canary_applied", 500, 0, 1000),
                ("applied-1", "canary_applied", 200, 0, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]
        else:
            rows = [
                ("applied-0", "canary_applied", 200, 0, 1000),
                ("applied-1", "canary_applied", 200, 0, 1100),
                ("holdout-0", "canary_holdout", 200, 0, 1000),
                ("holdout-1", "canary_holdout", 200, 0, 1100),
            ]

        store = Store(db_path)
        try:
            for suffix, cohort, status_code, retry_count, latency_ms in rows:
                store.log_call(
                    id=f"quality-gate-{scenario}-{suffix}",
                    created_at=f"2026-06-08T11:00:0{len(suffix)}+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    input_tokens_est=10000,
                    output_tokens_est=200,
                    actual_input_tokens=9000,
                    actual_output_tokens=180,
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.04,
                    routing_json=stable_json({"category": "chat", "text_chars": 40000, "has_tools": False}),
                    crunch_json=stable_json({
                        "changed": cohort == "canary_applied",
                        "old_context_summarization": meta_for(cohort, status_code=status_code, retry_count=retry_count),
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                    request_json=stable_json({"messages": [{"content": "raw quality secret must not leak"}]}),
                    response_json=stable_json({"content": "generated quality summary must not leak"}),
                    session_id="quality-gate-session-secret",
                    category="chat",
                    retry_count=retry_count,
                    provider="anthropic",
                    error="raw quality error must not leak" if status_code >= 400 else None,
                )
        finally:
            store.conn.close()

    def test_old_context_summary_impact_cli_reports_metadata_only_post_apply_evidence(self):
        dry_run = self._old_context_summary_impact_dry_run()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_impact_rows(db_path)
            stdout = io.StringIO()

            code = cli.old_context_summary_impact_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.old_context_summary_impact.v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["projected_affected_metadata_row_count"], 4)
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 3)
        self.assertEqual(payload["summary"]["actual_canary_applied_count"], 1)
        self.assertEqual(payload["summary"]["actual_canary_holdout_count"], 1)
        self.assertEqual(payload["summary"]["actual_bypassed_or_disabled_count"], 1)
        self.assertEqual(payload["summary"]["summary_failure_count"], 1)
        self.assertEqual(payload["summary"]["actual_tokens_saved_est"], 2000)
        self.assertGreater(payload["summary"]["actual_net_savings_usd"], 0)
        self.assertEqual(payload["actual"]["latency"]["applied_minus_holdout_avg_ms"], 1100.0)
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw impact secret", encoded)
        self.assertNotIn("generated summary secret", encoded)
        self.assertNotIn("impact-session-secret", encoded)
        self.assertNotIn("raw error secret", encoded)
        self.assertFalse(payload["privacy"]["raw_old_context_included"])
        self.assertFalse(payload["privacy"]["generated_summaries_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["tenant_ids_included"])
        self.assertFalse(payload["privacy"]["local_session_ids_included"])
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "disabled")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])

    def test_old_context_summary_quality_gate_verdicts_are_metadata_only(self):
        for scenario, expected_verdict, expected_reason in (
            ("promote", "promote", "quality-gate-passed"),
            ("hold", "hold", "applied-retry-rate-above-threshold"),
            ("rollback", "rollback", "rollback-error-rate"),
            ("insufficient-evidence", "hold", "insufficient-matched-samples"),
        ):
            dry_run = self._old_context_summary_quality_gate_dry_run()
            with self.subTest(scenario=scenario):
                with TemporaryDirectory() as tmp:
                    db_path = str(Path(tmp) / "agentflow.sqlite3")
                    self._write_old_context_summary_quality_gate_rows(db_path, scenario=scenario)
                    stdout = io.StringIO()

                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                gate = payload["quality_gate"]
                self.assertEqual(gate["schema"], "agentflow.old_context_summary_quality_gate.v1")
                self.assertEqual(gate["verdict"], expected_verdict)
                self.assertIn(expected_reason, gate["reason_codes"])
                self.assertEqual(payload["summary"]["quality_gate_verdict"], expected_verdict)
                self.assertEqual(gate["thresholds"]["min_matched_samples"], 4)
                self.assertEqual(gate["thresholds"]["max_error_rate"], 0.05)
                self.assertEqual(gate["cohorts"]["canary_applied"]["count"], 1 if scenario == "insufficient-evidence" else 2)
                self.assertFalse(gate["privacy"]["raw_old_context_included"])
                self.assertFalse(gate["privacy"]["generated_summaries_included"])
                encoded = json.dumps(payload, sort_keys=True)
                self.assertNotIn("raw quality secret", encoded)
                self.assertNotIn("generated quality summary", encoded)
                self.assertNotIn("quality-gate-session-secret", encoded)
                self.assertNotIn("raw quality error", encoded)

    def test_old_context_summary_quality_gate_cli_returns_compact_verdicts(self):
        for scenario, expected_verdict, expected_reason in (
            ("promote", "promote", "quality-gate-passed"),
            ("insufficient-evidence", "hold", "insufficient-matched-samples"),
            ("rollback", "rollback", "rollback-error-rate"),
        ):
            dry_run = self._old_context_summary_quality_gate_dry_run()
            with self.subTest(scenario=scenario):
                with TemporaryDirectory() as tmp:
                    db_path = str(Path(tmp) / "agentflow.sqlite3")
                    self._write_old_context_summary_quality_gate_rows(db_path, scenario=scenario)
                    stdout = io.StringIO()

                    code = cli.old_context_summary_quality_gate_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["schema"], "agentflow.old_context_summary_quality_gate.v1")
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["candidate_id"], "candidate-old-context-quality-gate")
                self.assertEqual(payload["rule_id"], "test-old-context-quality-gate")
                self.assertEqual(payload["verdict"], expected_verdict)
                self.assertIn(expected_reason, payload["reason_codes"])
                self.assertIn(payload["verdict"], {"promote", "hold", "rollback"})
                self.assertFalse(payload["privacy"]["raw_old_context_included"])
                self.assertFalse(payload["privacy"]["generated_summaries_included"])
                self.assertFalse(payload["privacy"]["file_paths_included"])
                self.assertFalse(payload["privacy"]["cache_keys_included"])
                self.assertFalse(payload["privacy"]["request_ids_included"])
                encoded = json.dumps(payload, sort_keys=True)
                for forbidden in (
                    "raw quality secret",
                    "generated quality summary",
                    "quality-gate-session-secret",
                    "raw quality error",
                    "/tmp/quality-gate-secret.py",
                    "cache-key-quality-secret",
                    "req_quality_secret",
                ):
                    self.assertNotIn(forbidden, encoded)

    def test_old_context_summary_impact_cli_exits_nonzero_without_matches(self):
        dry_run = self._old_context_summary_impact_dry_run()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()

            code = cli.old_context_summary_impact_cli(
                ["-", "--db", db_path, "--limit", "10"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=stdout,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "no-post-apply-matches")
        self.assertEqual(payload["error"]["type"], "no_post_apply_matches")
        self.assertEqual(payload["quality_gate"]["verdict"], "hold")
        self.assertIn("insufficient-matched-samples", payload["quality_gate"]["reason_codes"])

    def test_old_context_summary_impact_sends_metadata_only_lifecycle_feedback(self):
        dry_run = self._old_context_summary_impact_dry_run()
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_impact_rows(db_path)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "impact")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["lifecycle_kind"], "old_context_summarization")
        self.assertEqual(metadata["command"], "old-context-summary-impact")
        self.assertEqual(metadata["rule_id"], "test-old-context-impact")
        self.assertEqual(metadata["candidate_id"], "candidate-old-context-impact")
        self.assertEqual(metadata["actual_matched_metadata_row_count"], 3)
        self.assertFalse(metadata["privacy"]["cache_keys_included"])
        rendered_payload = json.dumps(sent_payload, sort_keys=True)
        self.assertNotIn("raw impact secret", rendered_payload)
        self.assertNotIn("generated summary secret", rendered_payload)
        self.assertNotIn("impact-session-secret", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertEqual(output["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])
        self.assertTrue(output["managed_server_calls_made"])

    def test_old_context_summary_quality_gate_feedback_queues_and_flushes_metadata_only_verdict(self):
        from agentflow_proxy import recommendations
        from agentflow_proxy.store import Store

        dry_run = self._old_context_summary_quality_gate_dry_run()
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable with raw quality secret"
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_quality_gate_rows(db_path, scenario="promote")
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_impact_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

                    output = json.loads(stdout.getvalue())
                    self.assertEqual(code, 0)
                    self.assertEqual(output["quality_gate"]["verdict"], "promote")
                    self.assertEqual(output["managed_lifecycle_feedback"]["status"], "retryable-error")
                    queue_id = output["managed_lifecycle_feedback"]["queue_id"]

                    store = Store(db_path)
                    try:
                        row = store.get_managed_outcome_feedback(queue_id)
                        self.assertIsNotNone(row)
                        queued_payload = json.loads(row["payload_json"])
                    finally:
                        store.conn.close()

                    metadata = queued_payload["metadata"]
                    gate = metadata["old_context_summary_quality_gate"]
                    self.assertEqual(gate["schema"], "agentflow.old_context_summary_quality_gate_feedback.v1")
                    self.assertEqual(gate["quality_gate_schema"], "agentflow.old_context_summary_quality_gate.v1")
                    self.assertEqual(gate["candidate_id"], "candidate-old-context-quality-gate")
                    self.assertEqual(gate["rule_id"], "test-old-context-quality-gate")
                    self.assertEqual(gate["policy_source"], "managed-recommended")
                    self.assertEqual(gate["verdict"], "promote")
                    self.assertEqual(gate["reason_codes"], ["quality-gate-passed"])
                    self.assertEqual(gate["cohort_counts"]["matched"], 4)
                    self.assertEqual(gate["cohort_counts"]["canary_applied"], 2)
                    self.assertEqual(gate["cohort_counts"]["canary_holdout"], 2)
                    self.assertEqual(gate["safety"]["summary_failure_count"], 0)
                    self.assertEqual(gate["safety"]["safety_stop_count"], 0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_error_rate"], 0.0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_retry_rate"], 0.0)
                    self.assertEqual(gate["aggregate_deltas"]["applied_minus_holdout_latency_avg_ms"], 0.0)
                    self.assertGreater(gate["savings"]["net_savings_usd"], 0)
                    self.assertGreater(gate["savings"]["payback_ratio"], 1.0)
                    self.assertFalse(gate["privacy"]["raw_old_context_included"])
                    self.assertFalse(gate["privacy"]["generated_summaries_included"])
                    self.assertFalse(gate["privacy"]["summary_prompts_included"])
                    self.assertFalse(gate["privacy"]["file_contents_included"])
                    self.assertFalse(gate["privacy"]["request_ids_included"])
                    self.assertFalse(gate["privacy"]["tenant_ids_included"])
                    self.assertFalse(gate["privacy"]["local_session_ids_included"])
                    self.assertFalse(gate["privacy"]["cache_keys_included"])
                    def keys_in(value):
                        if isinstance(value, dict):
                            keys = set(value)
                            for child in value.values():
                                keys.update(keys_in(child))
                            return keys
                        if isinstance(value, list):
                            keys = set()
                            for child in value:
                                keys.update(keys_in(child))
                            return keys
                        return set()

                    self.assertTrue(recommendations.RAW_FEATURE_KEYS.isdisjoint(keys_in(queued_payload) - {"command"}))
                    rendered = json.dumps(queued_payload, sort_keys=True)
                    self.assertNotIn("raw quality secret", rendered)
                    self.assertNotIn("generated quality summary", rendered)
                    self.assertNotIn("quality-gate-session-secret", rendered)
                    self.assertNotIn("raw quality error", rendered)

                    ManagedFeedbackFlushClient.status_code = 200
                    ManagedFeedbackFlushClient.text = '{"ok":true}'
                    flush_stdout = io.StringIO()
                    flush_code = cli.managed_feedback_flush_cli(
                        [
                            "--db",
                            db_path,
                            "--source-surface",
                            recommendations.OLD_CONTEXT_SUMMARY_LIFECYCLE_SOURCE_SURFACE,
                        ],
                        stdout=flush_stdout,
                    )

            self.assertEqual(flush_code, 0)
            flush_payload = json.loads(flush_stdout.getvalue())
            self.assertEqual(flush_payload["flush"]["sent"], 1)
            self.assertEqual(ManagedFeedbackFlushClient.calls[-1]["url"], "http://managed.test/v1/policy-events")
            self.assertEqual(
                ManagedFeedbackFlushClient.calls[-1]["json"]["metadata"]["old_context_summary_quality_gate"]["verdict"],
                "promote",
            )
            store = Store(db_path)
            try:
                flushed = store.get_managed_outcome_feedback(queue_id)
                self.assertEqual(flushed["status"], "sent")
            finally:
                store.conn.close()

    def test_old_context_summary_quality_gate_cli_sends_metadata_only_verdict_feedback(self):
        dry_run = self._old_context_summary_quality_gate_dry_run()
        ManagedFeedbackFlushClient.calls = []
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_quality_gate_rows(db_path, scenario="promote")
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.old_context_summary_quality_gate_cli(
                        ["-", "--db", db_path, "--limit", "10"],
                        stdin=io.StringIO(json.dumps(dry_run)),
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "quality-gate")
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["command"], "old-context-summary-quality-gate")
        gate = metadata["old_context_summary_quality_gate"]
        self.assertEqual(gate["schema"], "agentflow.old_context_summary_quality_gate_feedback.v1")
        self.assertEqual(gate["verdict"], "promote")
        self.assertEqual(gate["cohort_counts"]["matched"], 4)
        self.assertFalse(gate["privacy"]["raw_old_context_included"])
        self.assertFalse(gate["privacy"]["generated_summaries_included"])
        rendered = json.dumps(sent_payload, sort_keys=True)
        for forbidden in (
            "raw quality secret",
            "generated quality summary",
            "quality-gate-session-secret",
            "raw quality error",
            "/tmp/quality-gate-secret.py",
            "cache-key-quality-secret",
            "req_quality_secret",
        ):
            self.assertNotIn(forbidden, rendered)

    def _summary_rollout_action_bundle(self, *, action_type: str = "widen", raw_like: bool = False, verdict: str = "promote") -> dict:
        action = {
            "schema": "agentflow.old_context_summary_rollout_action.v1",
            "action_type": action_type,
            "target_candidate_id": "candidate-old-context-rollout",
            "target_rule_id": "managed-old-context-summary-rollout",
            "policy_section": "crunch",
            "current_fraction": 0.1,
            "recommended_fraction": 0.35 if action_type == "widen" else 0.0,
            "confidence": 0.92,
            "rationale": "Quality-gated metadata shows positive old-context summary canary outcomes.",
            "blockers": [] if action_type == "widen" else ["quality-gate-rollback"],
            "quality_gate": {
                "schema": "agentflow.old_context_summary_quality_gate.v1",
                "verdict": verdict,
                "reason_codes": ["quality-gate-passed"] if verdict == "promote" else ["applied-retry-rate-above-threshold"],
                "warning_codes": [],
            },
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_context_included": False,
                "generated_summaries_included": False,
            },
        }
        if raw_like:
            action["evidence"] = {"prompt": "raw old context must be rejected"}
        return {
            "schema": "agentflow.old_context_summary_rollout_actions.v1",
            "generated_at": "2026-06-09T04:10:00+00:00",
            "summary": {"action_count": 1, "managed_enforced": False, "required_local_review": True},
            "actions": [action],
            "privacy_summary": {
                "metadata_only": True,
                "raw_context_included": False,
                "generated_summaries_included": False,
            },
        }

    def _write_summary_rollout_rule(self, tmp: str, *, policy_source: str = "managed-recommended"):
        path = Path(tmp) / "crunch_rules.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "threshold_chars": 24000,
                    "old_context_summarization": {
                        "enabled": True,
                        "rule_id": "managed-old-context-summary-rollout",
                        "candidate_id": "candidate-old-context-rollout",
                        "policy_source": policy_source,
                        "model": "claude-haiku-4-5-20251001",
                        "placement": "system",
                        "min_request_chars": 32000,
                        "min_summarized_chars": 12000,
                        "max_turns": 6,
                        "keep_recent_turns": 4,
                        "max_summary_chars": 4000,
                        "max_source_chars": 80000,
                        "max_summary_cost_usd": 0.02,
                        "excluded_categories": ["tool-heavy", "tool-result"],
                        "block_tool_protocol": True,
                        "block_thinking": True,
                        "canary": {
                            "enabled": True,
                            "fraction": 0.1,
                            "salt": "summary-rollout-test",
                            "unit": "source_hash",
                        },
                        "safety_stop": {
                            "enabled": True,
                            "min_outcome_samples": 5,
                            "window": 500,
                            "max_error_rate": 0.1,
                            "max_retry_rate": 0.25,
                            "max_negative_net_savings_rate": 0.5,
                            "max_summary_failure_rate": 0.1,
                            "max_error_rate_delta": 0.05,
                        },
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _log_summary_rollout_call(
        self,
        store,
        *,
        cohort: str,
        created_at: str = "2026-06-09T04:20:00+00:00",
        id_suffix: str = "",
        status_code: int = 200,
    ):
        from agentflow_proxy.store import stable_json

        applied = cohort == "canary_applied"
        meta = {
            "enabled": True,
            "status": "applied" if applied else "skipped",
            "reason": "summary-created" if applied else ("canary_holdout" if cohort == "canary_holdout" else "local-canary-safety-stop"),
            "rule_id": "managed-old-context-summary-rollout",
            "candidate_id": "candidate-old-context-rollout",
            "policy_source": "managed-recommended",
            "category": "chat",
            "eligible_chars": 32000,
            "eligible_turns": 3,
            "saved_chars": 4000 if applied else 0,
            "tokens_saved_est": 1000 if applied else 0,
            "estimated_net_savings_usd": 0.002 if applied else 0.0,
            "summary_status_code": status_code,
            "canary": {
                "enabled": True,
                "cohort": cohort,
                "selected": applied,
                "fraction": 0.1,
                "unit": "source_hash",
            },
        }
        if cohort == "bypassed":
            meta["status"] = "bypass"
            meta["safety_stop_state"] = "stopped"
        store.log_call(
            id=f"summary-rollout-{cohort}{id_suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=1000,
            input_tokens_est=10000,
            output_tokens_est=200,
            actual_input_tokens=9000,
            actual_output_tokens=180,
            cost_est_usd=0.03,
            cost_baseline_usd=0.04,
            routing_json=stable_json({"category": "chat"}),
            crunch_json=stable_json({"changed": applied, "old_context_summarization": meta}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            request_json=stable_json({"messages": [{"content": "raw rollout secret must not leak"}]}),
            response_json=stable_json({"content": "generated rollout summary must not leak"}),
            session_id="summary-rollout-session-secret",
            category="chat",
            retry_count=0,
            provider="anthropic",
        )

    def test_old_context_summary_rollout_actions_review_requires_promote_gate(self):
        bundle = self._summary_rollout_action_bundle(verdict="hold")
        with TemporaryDirectory() as tmp:
            self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            stderr = io.StringIO()
            code = cli.old_context_summary_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("widen requires local old-context summary quality_gate verdict promote", [error["message"] for error in payload["validation"]["errors"]])

    def test_old_context_summary_rollout_actions_review_reports_fraction_edit(self):
        bundle = self._summary_rollout_action_bundle(action_type="widen")
        with TemporaryDirectory() as tmp:
            self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        edit = payload["actions"][0]["proposed_edit"]
        self.assertTrue(edit["changed"])
        self.assertEqual(edit["current_fraction"], 0.1)
        self.assertEqual(edit["recommended_fraction"], 0.35)
        self.assertEqual(edit["canary"]["fraction"], 0.35)
        self.assertEqual(payload["actions"][0]["quality_gate"]["verdict"], "promote")

    def test_signed_old_context_summary_rollout_actions_apply_rolls_back_and_creates_backup(self):
        from agentflow_proxy.old_context_summary_rollout_actions import attach_summary_rollout_action_provenance

        secret = "summary-rollout-secret"
        bundle = attach_summary_rollout_action_provenance(
            self._summary_rollout_action_bundle(action_type="rollback", verdict="rollback"),
            secret=secret,
            issuer="agentflow-server",
            server_id="local-dev",
            key_id="summary-rollout-key",
        )
        with TemporaryDirectory() as tmp:
            path = self._write_summary_rollout_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS": json.dumps({"summary-rollout-key": secret})}, clear=False):
                code = cli.old_context_summary_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
            written = yaml.safe_load(path.read_text(encoding="utf-8"))
            backups = list(Path(tmp).glob("crunch_rules.yaml.bak-*"))

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["provenance"]["status"], "verified")
        self.assertEqual(len(backups), 1)
        summary = written["old_context_summarization"]
        self.assertFalse(summary["enabled"])
        self.assertFalse(summary["canary"]["enabled"])
        self.assertEqual(summary["canary"]["fraction"], 0.0)
        self.assertEqual(summary["rollout_action"]["action_type"], "rollback")
        from agentflow_proxy.policy_events import recent_policy_events

        event = recent_policy_events(limit=1)["events"][0]
        self.assertEqual(event["action"], "old-context-summary-rollout-actions-apply")
        event_text = json.dumps(event)
        self.assertNotIn("raw rollout secret", event_text)
        self.assertNotIn("generated rollout summary", event_text)

    def test_old_context_summary_rollout_actions_reject_raw_and_unknown_before_writing(self):
        with TemporaryDirectory() as tmp:
            path = self._write_summary_rollout_rule(tmp)
            before = path.read_text(encoding="utf-8")
            raw_like = self._summary_rollout_action_bundle(raw_like=True)
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(raw_like)),
                stdout=stdout,
            )
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            self.assertIn("raw or prompt-like old-context summary rollout action payloads are not accepted", [error["message"] for error in json.loads(stdout.getvalue())["validation"]["errors"]])

            unknown = self._summary_rollout_action_bundle()
            unknown["actions"][0]["target_rule_id"] = "missing-rule"
            stdout = io.StringIO()
            code = cli.old_context_summary_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(unknown)),
                stdout=stdout,
            )
            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(json.loads(stdout.getvalue())["review"]["actions"][0]["reason"], "unknown-rule")

    def test_old_context_summary_rollout_actions_dry_run_and_impact_are_metadata_only(self):
        from agentflow_proxy.store import Store

        bundle = self._summary_rollout_action_bundle(action_type="widen")
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_summary_rollout_rule(tmp)
                self._log_summary_rollout_call(store, cohort="canary_applied")
                self._log_summary_rollout_call(store, cohort="canary_holdout")
                self._log_summary_rollout_call(store, cohort="bypassed")
                dry_stdout = io.StringIO()
                code = cli.old_context_summary_rollout_actions_dry_run_cli(
                    ["--config-dir", tmp, "--db", db_path, "--limit", "10", "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=dry_stdout,
                )
                dry_run = json.loads(dry_stdout.getvalue())
                self._log_summary_rollout_call(store, cohort="canary_applied", created_at="2026-06-09T05:00:01+00:00", id_suffix="-post")
            finally:
                store.conn.close()

            impact_stdout = io.StringIO()
            impact_code = cli.old_context_summary_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "10", "--since", "2026-06-09T05:00:00+00:00", "-"],
                stdin=io.StringIO(json.dumps(dry_run)),
                stdout=impact_stdout,
            )

        self.assertEqual(code, 0)
        self.assertTrue(dry_run["ok"])
        self.assertTrue(dry_run["read_only"])
        self.assertFalse(dry_run["wrote_policy_files"])
        self.assertEqual(dry_run["summary"]["affected_metadata_row_count"], 3)
        self.assertEqual(dry_run["actions"][0]["current_canary_applied_count"], 1)
        self.assertEqual(dry_run["actions"][0]["current_canary_holdout_count"], 1)
        self.assertEqual(dry_run["actions"][0]["current_bypassed_or_disabled_count"], 1)
        self.assertEqual(dry_run["actions"][0]["projected_fraction"], 0.35)
        self.assertFalse(dry_run["privacy"]["raw_old_context_included"])
        rendered = json.dumps(dry_run)
        self.assertNotIn("summary-rollout-session-secret", rendered)
        self.assertNotIn("raw rollout secret", rendered)

        self.assertEqual(impact_code, 0)
        impact = json.loads(impact_stdout.getvalue())
        self.assertEqual(impact["schema"], "agentflow.old_context_summary_rollout_actions_impact.v1")
        self.assertTrue(impact["ok"])
        self.assertEqual(impact["summary"]["actual_matched_metadata_row_count"], 1)
        self.assertEqual(impact["actions"][0]["actual"]["actual_canary_applied_count"], 1)
        self.assertFalse(impact["privacy"]["generated_summaries_included"])

    def test_policy_review_cli_includes_old_context_summary_dry_run(self):
        proposed = self._old_context_summary_bundle()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()

            code = cli.policy_review_cli(
                ["-", "--db", db_path, "--impact-limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        dry_run = payload["impact_summary"]["sections"]["crunch"]["old_context_summary_dry_run"]
        self.assertEqual(dry_run["schema"], "agentflow.old_context_summary_dry_run.v1")
        self.assertEqual(dry_run["summary"]["eligible_call_count"], 1)
        encoded = json.dumps(dry_run, sort_keys=True)
        self.assertNotIn("raw-secret-old-context", encoded)

    def test_policy_review_cli_surfaces_managed_old_context_summary_candidate(self):
        proposed = self._managed_policy_bundle()
        proposed["recommendation"]["candidate_ids"].append("candidate-old-context-summary")
        proposed["recommendation"]["candidate_count"] = 2
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "candidate-old-context-summary",
                    "candidate_family": "old-context-summarization-policy-rule",
                    "policy_source": "managed-recommended",
                    "confidence": 0.84,
                    "sample_count": 42,
                    "source_model": "claude-sonnet-4-6",
                    "summary_model": "claude-haiku-4-5-20251001",
                    "conditions": {
                        "min_request_chars": 1,
                        "min_summarized_chars": 10,
                        "max_source_chars": 10000,
                    },
                    "action": {
                        "max_turns": 3,
                        "keep_recent_turns": 1,
                        "max_summary_chars": 80,
                        "max_summary_cost_usd": 1.0,
                    },
                    "canary": {
                        "enabled": True,
                        "fraction": 0.25,
                        "salt": "summary-canary-test",
                        "unit": "source_hash",
                        "holdout_sample_count": 9,
                        "widening_threshold": 0.8,
                        "rollback_threshold": 0.2,
                    },
                    "safety_gates": {
                        "max_error_rate": 0.05,
                        "max_summary_failure_rate": 0.02,
                    },
                    "net_savings_evidence": {
                        "projected_net_savings_usd": 0.42,
                        "projected_saved_tokens": 1200,
                        "estimated_summary_cost_usd": 0.01,
                    },
                    "quality_evidence": {
                        "holdout_success_count": 9,
                        "eval_pass_rate": 0.98,
                    },
                    "blocker_reason_codes": ["quality-gate-passed"],
                    "local_action_requirements": {
                        "expected_policy_section": "crunch",
                        "actionability_status": "review-only-local-action",
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "raw_body_storage": False,
                        "aggregate_only": False,
                    },
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_old_context_summary_dry_run_rows(db_path)
            stdout = io.StringIO()
            code = cli.policy_review_cli(
                ["-", "--db", db_path, "--impact-limit", "10"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        old_context = payload["section_reviews"]["crunch"]["old_context_summarization"]
        self.assertEqual(old_context["schema"], "agentflow.old_context_summary_policy_review.v1")
        self.assertEqual(old_context["candidate_ids"], ["candidate-old-context-summary"])
        candidate = old_context["candidates"][0]
        self.assertEqual(candidate["candidate_family"], "old-context-summarization-policy-rule")
        self.assertEqual(candidate["source_model"], "claude-sonnet-4-6")
        self.assertEqual(candidate["summary_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(candidate["canary"]["fraction"], 0.25)
        self.assertEqual(candidate["safety_gates"]["max_error_rate"], 0.05)
        self.assertEqual(candidate["net_savings_evidence"]["projected_net_savings_usd"], 0.42)
        self.assertEqual(candidate["quality_evidence"]["holdout_success_count"], 9)
        self.assertEqual(candidate["blocker_reason_codes"], ["quality-gate-passed"])
        self.assertFalse(old_context["application"]["writes_local_policy_files"])
        self.assertFalse(old_context["application"]["provider_calls_made"])
        dry_run = payload["impact_summary"]["sections"]["crunch"]["old_context_summary_dry_run"]
        self.assertEqual(dry_run["policy"]["candidate_id"], "candidate-old-context-summary")
        self.assertEqual(dry_run["summary"]["eligible_call_count"], 1)
        rendered = stdout.getvalue()
        self.assertNotIn("raw-secret-old-context", rendered)
        self.assertIn("old-context summarization candidates: 1", " ".join(payload["human_summary"]))

    def test_policy_review_cli_rejects_raw_like_old_context_summary_fields(self):
        proposed = self._managed_policy_bundle()
        proposed["policies"]["crunch"]["old_context_summarization"]["raw_prompt"] = "managed raw prompt must not be accepted"
        proposed["policies"]["crunch"]["old_context_summarization"]["candidate_id"] = "candidate-old-context-summary"
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        errors = payload["proposed_validation"]["errors"]
        self.assertIn("$.policies.crunch.old_context_summarization.raw_prompt", {error["path"] for error in errors})
        rendered = stdout.getvalue()
        self.assertNotIn("managed raw prompt must not be accepted", rendered)

    def test_policy_impact_simulates_managed_pattern_bundle_without_mutation(self):
        from agentflow_proxy.policy_bundle import simulate_policy_bundle_impact
        from agentflow_proxy.store import Store, stable_json

        crunch_hash = "sha256:" + ("a" * 64)
        cache_hash = "sha256:" + ("b" * 64)
        blocked_cache_hash = "sha256:" + ("c" * 64)
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["pattern_rules"] = [{
            "id": "managed-crunch-pattern-test",
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": "crunch-candidate-test",
            "conditions": {
                "pattern_hashes": [crunch_hash],
                "model_pattern": "sonnet",
                "category": "tool-result",
                "workflow_phase": "tool-result",
                "min_text_chars": 1000,
            },
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "full-review",
                "canary_enabled": False,
                "canary_fraction": 1.0,
                "canary_salt": "sha256:" + ("1" * 64),
                "canary_unit": "request_fingerprint",
            },
            "action": {"type": "shorten", "head_chars": 100, "tail_chars": 100},
            "managed_recommendation": {"estimated_savings_usd": 0.12},
        }]
        proposed["policies"]["cache"]["policy_source"] = "managed-recommended"
        proposed["policies"]["cache"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 2,
            "candidates": [
                {
                    "candidate_id": "cache-canary-test",
                    "policy_source": "managed-recommended",
                    "pattern_hash": cache_hash,
                    "estimated_savings_usd": 0.25,
                    "dimensions": {"source_surface": "anthropic_messages", "app_family": "claude_code", "category": "chat", "phase": "chat"},
                    "buckets": {"has_tools": False},
                    "rollout": {
                        "schema": "agentflow.pattern_policy_rollout.v1",
                        "recommendation_mode": "canary-only",
                        "canary_enabled": True,
                        "canary_fraction": 0.0,
                        "canary_salt": "sha256:" + ("2" * 64),
                        "canary_unit": "request_fingerprint",
                    },
                    "cache_action": {"type": "exact_cache_pattern_review", "streaming": False, "pattern_hashes": [cache_hash]},
                },
                {
                    "candidate_id": "cache-features-only-test",
                    "policy_source": "managed-recommended",
                    "pattern_hash": blocked_cache_hash,
                    "estimated_savings_usd": 0.05,
                    "dimensions": {"source_surface": "codex_turn", "app_family": "codex", "category": "summary", "phase": "summary"},
                    "buckets": {"has_tools": False},
                    "rollout": {
                        "schema": "agentflow.pattern_policy_rollout.v1",
                        "recommendation_mode": "full-review",
                        "canary_enabled": False,
                        "canary_fraction": 1.0,
                        "canary_salt": "sha256:" + ("3" * 64),
                        "canary_unit": "request_fingerprint",
                    },
                    "cache_action": {"type": "exact_cache_pattern_review", "streaming": False, "pattern_hashes": [blocked_cache_hash]},
                },
            ],
            "review_only_candidate_count": 0,
            "omitted_candidate_count": 0,
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            config_path = Path(tmp) / "crunch_rules.yaml"
            config_path.write_text("enabled: true\n", encoding="utf-8")
            store = Store(db_path)
            try:
                store.set_cache("cache-key", "claude-sonnet-4-6", 10, {"content": "cached"})
                store.log_call(
                    id="pattern-crunch-call",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=2000,
                    output_tokens_est=200,
                    actual_input_tokens=2000,
                    actual_output_tokens=200,
                    cost_est_usd=0.006,
                    routing_json=stable_json({
                        "category": "tool-result",
                        "text_chars": 8000,
                        "has_tools": False,
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [crunch_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "tool-result",
                            "category": "tool-result",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_call(
                    id="pattern-cache-call",
                    created_at="2026-06-08T10:01:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=900,
                    input_tokens_est=1200,
                    output_tokens_est=120,
                    actual_input_tokens=1200,
                    actual_output_tokens=120,
                    cost_est_usd=0.004,
                    routing_json=stable_json({
                        "category": "chat",
                        "text_chars": 4800,
                        "has_tools": False,
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [cache_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "chat",
                            "category": "chat",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                )
                store.log_codex_app_event(
                    id="pattern-codex-turn",
                    created_at="2026-06-08T10:02:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-pattern",
                    thread_id="thread-pattern",
                    message_chars=160,
                    params_chars=5200,
                    input_items=1,
                    input_text_chars=5000,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="session-pattern",
                    routing_json=stable_json({
                        "category": "summary",
                        "workflow_phase": "summary",
                        "managed_pattern_features": {
                            "schema": "agentflow.managed_pattern_feature_diagnostics.v1",
                            "present": True,
                            "pattern_hashes": [blocked_cache_hash],
                            "pattern_hash_count": 1,
                            "workflow_phase": "summary",
                            "category": "summary",
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "1k_4k_tokens",
                            "replayability_level": "features_only",
                            "raw_pattern_strings_included": False,
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "features-only"}),
                )

                before_cache = store.get_cache("cache-key")
                event_log = Path(os.environ["AGENTFLOW_POLICY_EVENTS_LOG"])
                before_event_text = event_log.read_text(encoding="utf-8") if event_log.exists() else ""

                impact = simulate_policy_bundle_impact(proposed, db_path=db_path, limit=10)

                after_cache = store.get_cache("cache-key")
                after_event_text = event_log.read_text(encoding="utf-8") if event_log.exists() else ""
            finally:
                store.conn.close()

            self.assertEqual(config_path.read_text(encoding="utf-8"), "enabled: true\n")

        self.assertEqual(before_cache, {"content": "cached"})
        self.assertEqual(after_cache, {"content": "cached"})
        self.assertEqual(before_event_text, after_event_text)
        self.assertEqual(impact["status"], "simulated")
        self.assertEqual(impact["sampled_call_count"], 2)
        self.assertEqual(impact["sampled_codex_turn_count"], 1)
        crunch_patterns = impact["sections"]["crunch"]["pattern_rules"]
        self.assertEqual(crunch_patterns["would_match_count"], 1)
        self.assertEqual(crunch_patterns["would_apply_count"], 1)
        crunch_candidate = crunch_patterns["candidates"][0]
        self.assertEqual(crunch_candidate["candidate_id"], "crunch-candidate-test")
        self.assertEqual(crunch_candidate["estimated_tokens_affected"], 2000)
        cache_patterns = impact["sections"]["cache"]["pattern_rules"]
        self.assertEqual(cache_patterns["would_match_count"], 2)
        self.assertEqual(cache_patterns["would_apply_count"], 0)
        self.assertEqual(cache_patterns["would_holdout_count"], 1)
        self.assertEqual(cache_patterns["would_bypass_count"], 1)
        blocker_reasons = {
            blocker["reason"]
            for candidate in cache_patterns["candidates"]
            for blocker in candidate["safety_blockers"]
        }
        self.assertIn("features-only-cache-replay-block", blocker_reasons)
        self.assertIn("codex_turn", cache_patterns["candidates"][1]["source_surfaces"])

    def test_policy_review_cli_rejects_invalid_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps({"schema": "wrong"})), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertIn("$.schema", {error["path"] for error in payload["proposed_validation"]["errors"]})
        self.assertFalse(payload["changed"])

    def test_policy_review_cli_surfaces_risky_policy_warnings(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        proposed["policies"]["cache"]["semantic_cache"]["enabled"] = True
        proposed["policies"]["crunch"]["old_context_summarization"]["enabled"] = True
        proposed["policies"]["routing"]["policy_source"] = "managed-enforced"
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        warning_codes = {warning["code"] for warning in payload["safety_warnings"]}
        self.assertIn("tool-call-cache-enabled", warning_codes)
        self.assertIn("semantic-cache-enabled", warning_codes)
        self.assertIn("old-context-summarization-enabled", warning_codes)
        self.assertIn("managed-enforced-policy-source", warning_codes)
        self.assertEqual(payload["safety_warning_count"], 4)

    def test_policy_review_cli_records_compact_event(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        stdout = io.StringIO()

        cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "review")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["cache"])
        self.assertEqual(events[0]["details"]["safety_warning_count"], 1)

    def _managed_policy_bundle(self, *, invalid: bool = False):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        if invalid:
            return {"schema": "wrong"}
        bundle = json.loads(exported.getvalue())
        bundle["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "policy_source": "managed-recommended",
            "candidate_ids": ["candidate-route-chat"],
            "candidate_count": 1,
            "routing_rule_count": 1,
            "omitted_candidate_count": 0,
            "filters": {"min_samples": 3},
        }
        bundle["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "note": "Review-only managed recommendation.",
        }
        bundle["policies"]["routing"]["policy_source"] = "managed-recommended"
        bundle["policies"]["routing"].setdefault("rules", []).append({
            "conditions": {
                "model_pattern": "sonnet",
                "category": "chat",
                "has_tools": False,
            },
            "action": {
                "route_to": "claude-haiku-4-5-20251001",
                "reason": "managed candidate for local review",
            },
            "managed_recommendation": {
                "policy_source": "managed-recommended",
                "candidate_id": "candidate-route-chat",
                "confidence": 0.82,
                "sample_count": 24,
                "success_count": 23,
                "error_count": 1,
                "error_rate": 0.041,
                "estimated_savings_usd": 1.23,
                "requested_model": "claude-sonnet-4-6",
                "recommended_target_model": "claude-haiku-4-5-20251001",
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "category": "chat",
            },
        })
        bundle["policies"]["codex_app"] = {
            **bundle["policies"]["codex_app"],
            "policy_source": "managed-recommended",
            "review_only": True,
            "application": {
                "status": "not-applied",
                "reason": "Codex app candidates are review-only in local policy tooling.",
            },
            "rules": [
                {
                    "candidate_id": "candidate-codex-summary",
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                        "input_size_bucket": "small",
                        "cache_eligible": False,
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "crunch_profile": "pass-through",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                    "managed_recommendation": {
                        "policy_source": "managed-recommended",
                        "candidate_id": "candidate-codex-summary",
                        "confidence": 0.76,
                        "sample_count": 18,
                    },
                }
            ],
        }
        return bundle

    def _managed_old_context_summary_bundle(self, *, raw_like: bool = False):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["candidate_ids"].append("candidate-old-context-summary")
        bundle["recommendation"]["candidate_count"] = 2
        bundle["policies"]["crunch"]["policy_source"] = "managed-recommended"
        candidate = {
            "candidate_id": "candidate-old-context-summary",
            "candidate_family": "old-context-summarization-policy-rule",
            "policy_source": "managed-recommended",
            "confidence": 0.84,
            "sample_count": 42,
            "source_model": "claude-sonnet-4-6",
            "summary_model": "claude-haiku-4-5-20251001",
            "conditions": {
                "min_request_chars": 1,
                "min_summarized_chars": 10,
                "max_source_chars": 10000,
            },
            "action": {
                "max_turns": 3,
                "keep_recent_turns": 1,
                "max_summary_chars": 80,
                "max_summary_cost_usd": 1.0,
            },
            "canary": {
                "enabled": True,
                "fraction": 0.25,
                "salt": "summary-canary-test",
                "unit": "source_hash",
                "holdout_sample_count": 9,
                "widening_threshold": 0.8,
                "rollback_threshold": 0.2,
            },
            "safety_gates": {
                "min_outcome_samples": 5,
                "max_error_rate": 0.05,
                "max_summary_failure_rate": 0.02,
            },
            "net_savings_evidence": {
                "projected_net_savings_usd": 0.42,
                "projected_saved_tokens": 1200,
                "estimated_summary_cost_usd": 0.01,
            },
            "quality_evidence": {
                "holdout_success_count": 9,
                "eval_pass_rate": 0.98,
            },
            "blocker_reason_codes": ["quality-gate-passed"],
            "local_action_requirements": {
                "expected_policy_section": "crunch",
                "actionability_status": "review-only-local-action",
            },
            "privacy_summary": {
                "metadata_only": True,
                "raw_body_storage": False,
                "aggregate_only": False,
            },
            "rationale": "raw-secret-old-context should not be copied to events",
        }
        if raw_like:
            candidate["prompt"] = "raw-secret-old-context must stay out of policy events"
        bundle["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [candidate],
        }
        return bundle

    def _pattern_hash(self):
        return "sha256:" + ("a" * 64)

    def _pattern_hash_char(self, char: str):
        return "sha256:" + (char * 64)

    def _rollout_action_for_rule(
        self,
        *,
        section: str,
        rule_id: str,
        candidate_id: str,
        pattern_hash: str,
        module_family: str,
        recommended_fraction: float = 0.4,
        policy_profile: str | None = "conservative",
    ) -> dict:
        action = {
            "schema": "agentflow.pattern_rollout_action.v1",
            "action_type": "widen",
            "target_candidate_id": candidate_id,
            "target_rule_id": rule_id,
            "policy_section": section,
            "module_family": module_family,
            "pattern_hash": pattern_hash,
            "current_fraction": 0.1,
            "recommended_fraction": recommended_fraction,
            "confidence": 0.91,
            "rationale": "Family-specific canary outcomes are positive.",
            "blockers": [],
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }
        if policy_profile is not None:
            action["policy_profile"] = policy_profile
        return action

    def _rollout_bundle_for_actions(self, actions: list[dict]) -> dict:
        return {
            "schema": "agentflow.pattern_rollout_actions.v1",
            "generated_at": "2026-06-09T03:10:00+00:00",
            "tenant_scope": "local-dev",
            "filters": {"min_samples": 3},
            "thresholds": {"min_samples": 3, "max_error_rate": 0.05},
            "summary": {
                "candidate_count": len(actions),
                "action_count": len(actions),
                "action_counts": {"widen": len(actions)},
                "managed_enforced": False,
                "required_local_review": True,
            },
            "actions": actions,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }

    def _write_rollout_pattern_files(
        self,
        tmp: str,
        *,
        crunch_rules: list[dict] | None = None,
        cache_rules: list[dict] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        if crunch_rules is not None:
            path = Path(tmp) / "crunch_rules.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "enabled": True,
                        "threshold_chars": 24000,
                        "pattern_rules": crunch_rules,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            paths["crunch"] = path
        if cache_rules is not None:
            path = Path(tmp) / "cache_rules.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "exact_cache": {"enabled": True, "cache_tool_calls": False},
                        "semantic_cache": {"enabled": False, "threshold": 0.95},
                        "file_watch": {"enabled": True, "root": ".", "max_paths": 128},
                        "pattern_rules": cache_rules,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            paths["cache"] = path
        return paths

    def _family_pattern_rule(
        self,
        *,
        section: str,
        module_family: str,
        rule_id: str,
        candidate_id: str,
        pattern_hash: str,
        action: dict,
        policy_profile: str = "conservative",
        replayability_levels: list[str] | None = None,
    ) -> dict:
        conditions = {
            "pattern_hashes": [pattern_hash],
            "module_family": module_family,
        }
        if section == "crunch":
            conditions.update({
                "min_repeated_count": 2,
                "keep_recent_matches": 1,
            })
        if replayability_levels is not None:
            conditions["replayability_levels"] = replayability_levels
        return {
            "id": rule_id,
            "enabled": True,
            "policy_source": "managed-recommended",
            "candidate_id": candidate_id,
            "module_family": module_family,
            "policy_profile": policy_profile,
            "conditions": conditions,
            "action": action,
            "rollout": {
                "schema": "agentflow.pattern_policy_rollout.v1",
                "recommendation_mode": "canary",
                "canary_enabled": True,
                "canary_fraction": 0.1,
                "canary_salt": "local-dev",
                "canary_unit": "request_fingerprint",
                "rollback_threshold": 0.2,
                "widening_threshold": 0.8,
                "min_outcome_samples": 5,
            },
        }

    def _rollout_action_bundle(self, *, action_type: str = "widen", raw_like: bool = False):
        action = {
            "schema": "agentflow.pattern_rollout_action.v1",
            "action_type": action_type,
            "target_candidate_id": "candidate-crunch-rollout",
            "target_rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
            "policy_section": "crunch",
            "pattern_hash": self._pattern_hash(),
            "current_fraction": 0.1,
            "recommended_fraction": 0.25 if action_type == "widen" else 0.0,
            "confidence": 0.91,
            "rationale": "Canary-applied outcomes show positive section savings without error regression.",
            "blockers": [] if action_type == "widen" else ["pattern-applied-cohort-errors"],
            "required_local_review": True,
            "managed_enforced": False,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
            "evidence": {
                "pattern_cohorts": {
                    "outcome_counts": {"applied": 12, "bypassed": 8},
                },
            },
        }
        if raw_like:
            action["evidence"]["raw_request"] = "raw prompt body must be rejected"
        return {
            "schema": "agentflow.pattern_rollout_actions.v1",
            "generated_at": "2026-06-09T03:10:00+00:00",
            "tenant_scope": "local-dev",
            "filters": {"min_samples": 3},
            "thresholds": {"min_samples": 3, "max_error_rate": 0.05},
            "summary": {
                "candidate_count": 1,
                "action_count": 1,
                "action_counts": {action_type: 1},
                "managed_enforced": False,
                "required_local_review": True,
            },
            "actions": [action],
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
            },
        }

    def _terminal_compaction_rollout_bundle(
        self,
        *,
        raw_like: bool = False,
        unsafe_privacy: bool = False,
        unsupported_action: bool = False,
        unsupported_field: bool = False,
        expired_bundle: bool = False,
        expired_action: bool = False,
        incompatible_bundle: bool = False,
        incompatible_action: bool = False,
    ):
        proposed_edit = {
            "rule_id": "managed-terminal-output-compaction-rule",
            "policy_source": "managed-recommended",
            "conditions": {
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "phase": "tool-result",
                "category": "tool-result",
                "text_bucket": "32k_128k_chars",
                "labels": ["terminal-output", "plateau-session"],
                "expected_saved_tokens_bucket": "1k_2k_tokens",
                "model_pattern": "sonnet",
                "has_tools": True,
                "stream": True,
                "uses_thinking": False,
            },
            "action": {
                "type": "rewrite_provider_body" if unsupported_action else "terminal_output_compaction",
                "keep_recent_turns": 2,
                "min_block_chars": 700,
                "head_lines": 9,
                "tail_lines": 11,
                "max_evidence_lines": 55,
                "min_saved_chars": 250,
                "preserve_diagnostics": True,
                "preserve_tool_protocol": True,
            },
            "canary": {
                "enabled": True,
                "canary_fraction": 0.25,
                "holdout_fraction": 0.75,
                "canary_salt": "terminal-rollout-test",
                "canary_unit": "request_fingerprint",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 7,
                "window": 77,
                "max_error_rate": 0.2,
                "max_retry_rate": 0.3,
                "max_negative_savings_rate": 0.4,
                "max_error_rate_delta": 0.05,
            },
            "compatibility": {
                "minimum_local_client_version": "0.1.0",
                "supported_local_action_families": ["crunch"],
            },
            "local_action_requirements": {
                "expected_policy_section": "crunch",
                "actionability_status": "review-only-local-action",
            },
        }
        if raw_like:
            proposed_edit.update(
                {
                    "raw_request": "raw terminal output must not be accepted",
                    "provider_body": {"messages": [{"content": "raw provider body must not be accepted"}]},
                    "prompt": "raw prompt must not be accepted",
                    "message_content": "raw message content must not be accepted",
                    "terminal_lines": ["raw terminal line must not be accepted"],
                    "tool_payload": {"command": "cat /workspace/private/terminal-secret.log"},
                    "request_id": "raw-terminal-request-id",
                    "session_id": "raw-terminal-session-id",
                    "cache_key": "raw-terminal-cache-key",
                    "file_path": "/workspace/private/terminal-secret.log",
                    "tenant_id": "raw-terminal-tenant-id",
                    "api_key": "sk-terminal-secret",
                    "local_policy_file_contents": "raw local policy file contents must not be accepted",
                }
            )
        if unsupported_field:
            proposed_edit["action"]["provider_body_patch"] = {"path": "$.messages"}
        action = {
            "schema": "agentflow.pattern_rollout_action.v1",
            "action_type": "widen",
            "candidate_family": "terminal-output-compaction-crunch-policy-rule",
            "target_candidate_id": "terminal-compaction-candidate-123",
            "target_rule_id": "managed-terminal-output-compaction-rule",
            "policy_section": "crunch",
            "current_fraction": 0.0,
            "recommended_fraction": 0.25,
            "confidence": 0.91,
            "rationale": "Terminal-output canary evidence is positive without error regression.",
            "blockers": [],
            "required_local_review": True,
            "managed_enforced": False,
            "proposed_edit": proposed_edit,
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
                "raw_provider_bodies_included": bool(unsafe_privacy),
                "request_ids_returned": bool(unsafe_privacy),
                "session_ids_returned": bool(unsafe_privacy),
                "cache_keys_returned": bool(unsafe_privacy),
                "file_paths_returned": bool(unsafe_privacy),
            },
            "expires_at": "2000-01-01T00:00:00+00:00" if expired_action else "2099-06-12T00:00:00+00:00",
            "local_executor_compatibility": {
                "minimum_local_client_version": "99.0.0" if incompatible_action else "0.1.0",
                "compatible": not incompatible_action,
                "supported_local_action_families": ["routing"] if incompatible_action else ["crunch"],
            },
        }
        bundle = self._rollout_bundle_for_actions([action])
        bundle["expires_at"] = "2000-01-01T00:00:00+00:00" if expired_bundle else "2099-06-12T00:00:00+00:00"
        bundle["local_executor_compatibility"] = {
            "minimum_local_client_version": "99.0.0" if incompatible_bundle else "0.1.0",
            "compatible": not incompatible_bundle,
            "supported_local_action_families": ["routing"] if incompatible_bundle else ["crunch"],
        }
        return bundle

    def _write_terminal_compaction_crunch_file(self, tmp: str):
        path = Path(tmp) / "crunch_rules.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "threshold_chars": 24000,
                    "terminal_output_compaction": {
                        "enabled": True,
                        "rules": [
                            {
                                "id": "existing-terminal-rule",
                                "enabled": True,
                                "policy_source": "local-manual",
                                "candidate_id": "existing-terminal-candidate",
                                "conditions": {"category": "tool-result"},
                                "action": {"type": "compact_terminal_output", "keep_recent_turns": 1},
                            }
                        ],
                    },
                    "pattern_rules": [
                        {
                            "id": "unrelated-pattern-rule",
                            "enabled": True,
                            "policy_source": "managed-recommended",
                            "candidate_id": "unrelated-pattern-candidate",
                            "conditions": {"pattern_hashes": [self._pattern_hash()]},
                            "action": {"type": "shorten", "head_chars": 500, "tail_chars": 500},
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _write_rollout_crunch_rule(self, tmp: str, *, policy_source: str = "managed-recommended"):
        path = Path(tmp) / "crunch_rules.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "threshold_chars": 24000,
                    "pattern_rules": [
                        {
                            "id": "managed-crunch-pattern-candidate-crunch-rollout",
                            "enabled": True,
                            "policy_source": policy_source,
                            "candidate_id": "candidate-crunch-rollout",
                            "conditions": {
                                "pattern_hashes": [self._pattern_hash()],
                                "min_repeated_count": 2,
                                "keep_recent_matches": 1,
                            },
                            "action": {
                                "type": "shorten",
                                "head_chars": 800,
                                "tail_chars": 600,
                                "max_replacement_chars": 1800,
                            },
                            "rollout": {
                                "schema": "agentflow.pattern_policy_rollout.v1",
                                "recommendation_mode": "canary",
                                "canary_enabled": True,
                                "canary_fraction": 0.1,
                                "canary_salt": "local-dev",
                                "canary_unit": "request_fingerprint",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_managed_rollout_actions_review_reports_terminal_output_compaction_rule_edit(self):
        bundle = self._terminal_compaction_rollout_bundle()

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["planned_action_count"], 1)
        action = payload["actions"][0]
        self.assertEqual(action["rule_collection"], "terminal_output_compaction.rules")
        self.assertEqual(action["family_validation"]["status"], "accepted")
        self.assertEqual(action["family_validation"]["family"], "terminal_output_compaction")
        self.assertEqual(action["family_validation"]["compatibility"]["minimum_local_client_version"], "0.1.0")
        edit = action["proposed_edit"]
        self.assertEqual(edit["operation"], "append")
        self.assertTrue(edit["changed"])
        self.assertEqual(edit["policy_source"], "managed-recommended")
        self.assertEqual(edit["conditions"]["workflow_phase"], "tool-result")
        self.assertEqual(edit["conditions"]["expected_saved_token_bucket"], "1k_2k_tokens")
        self.assertEqual(edit["action"]["type"], "compact_terminal_output")
        self.assertEqual(edit["action"]["keep_recent_turns"], 2)
        self.assertEqual(edit["canary"]["canary_fraction"], 0.25)
        self.assertEqual(edit["canary"]["holdout_fraction"], 0.75)
        self.assertEqual(edit["safety_stop"]["min_outcome_samples"], 7)

    def test_managed_rollout_actions_apply_dry_run_reports_terminal_compaction_without_writing(self):
        bundle = self._terminal_compaction_rollout_bundle()

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "crunch_rules.yaml"
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        self.assertFalse(path.exists())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["applied_sections"], ["crunch"])
        self.assertTrue(payload["files"][0]["changed"])
        self.assertIsNone(payload["files"][0]["backup_path"])
        self.assertEqual(payload["actions"][0]["proposed_edit"]["rule"]["policy_source"], "managed-recommended")

    def test_managed_rollout_actions_apply_terminal_compaction_creates_backup_and_preserves_rules(self):
        bundle = self._terminal_compaction_rollout_bundle()

        with TemporaryDirectory() as tmp:
            path = self._write_terminal_compaction_crunch_file(tmp)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            written = yaml.safe_load(path.read_text(encoding="utf-8"))
            backup_count = len(list(Path(tmp).glob("crunch_rules.yaml.bak-*")))

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["applied_sections"], ["crunch"])
        self.assertTrue(payload["files"][0]["changed"])
        self.assertIsNotNone(payload["files"][0]["backup_path"])
        self.assertEqual(backup_count, 1)
        self.assertEqual(written["pattern_rules"][0]["id"], "unrelated-pattern-rule")
        terminal_rules = written["terminal_output_compaction"]["rules"]
        self.assertEqual([rule["id"] for rule in terminal_rules], ["existing-terminal-rule", "managed-terminal-output-compaction-rule"])
        managed_rule = terminal_rules[1]
        self.assertEqual(managed_rule["policy_source"], "managed-recommended")
        self.assertEqual(managed_rule["candidate_id"], "terminal-compaction-candidate-123")
        self.assertEqual(managed_rule["action"]["type"], "compact_terminal_output")
        self.assertEqual(managed_rule["canary"]["canary_fraction"], 0.25)
        self.assertEqual(managed_rule["safety_stop"]["max_error_rate_delta"], 0.05)

    def test_managed_rollout_actions_reject_terminal_compaction_raw_and_unsupported_before_writing(self):
        for bundle in (
            self._terminal_compaction_rollout_bundle(raw_like=True),
            self._terminal_compaction_rollout_bundle(unsafe_privacy=True),
            self._terminal_compaction_rollout_bundle(unsupported_action=True),
            self._terminal_compaction_rollout_bundle(unsupported_field=True),
            self._terminal_compaction_rollout_bundle(expired_bundle=True),
            self._terminal_compaction_rollout_bundle(expired_action=True),
            self._terminal_compaction_rollout_bundle(incompatible_bundle=True),
            self._terminal_compaction_rollout_bundle(incompatible_action=True),
        ):
            with TemporaryDirectory() as tmp:
                path = self._write_terminal_compaction_crunch_file(tmp)
                before = path.read_text(encoding="utf-8")
                stdout = io.StringIO()
                code = cli.managed_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )

                self.assertEqual(code, 1)
                self.assertEqual(path.read_text(encoding="utf-8"), before)
                self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
                payload = json.loads(stdout.getvalue())
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["type"], "validation_failed")
                self.assertNotIn("raw terminal output must not be accepted", json.dumps(payload, sort_keys=True))

    def test_managed_rollout_actions_reject_terminal_compaction_local_manual_overwrite(self):
        bundle = self._terminal_compaction_rollout_bundle()

        with TemporaryDirectory() as tmp:
            path = self._write_terminal_compaction_crunch_file(tmp)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["terminal_output_compaction"]["rules"][0]["id"] = "managed-terminal-output-compaction-rule"
            data["terminal_output_compaction"]["rules"][0]["candidate_id"] = "terminal-compaction-candidate-123"
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["review"]["actions"][0]["reason"], "unsafe-policy-source")

    def test_managed_rollout_actions_terminal_compaction_queues_feedback_when_server_unavailable(self):
        bundle = self._terminal_compaction_rollout_bundle()
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "rollout_action_lifecycle"],
                stdout=status_stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "retryable-error")
        self.assertEqual(payload["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["retryable_error"], 1)
        self.assertFalse(status_payload["terminal_output_compaction_lifecycle"]["payload_json_included"])
        rendered = json.dumps(ManagedFeedbackFlushClient.calls[0]["json"], sort_keys=True)
        self.assertIn("terminal_output_compaction", rendered)
        self.assertNotIn("raw terminal output", rendered)
        self.assertNotIn("raw-terminal-request-id", rendered)
        self.assertNotIn("raw-terminal-session-id", rendered)
        self.assertNotIn("raw-terminal-cache-key", rendered)
        self.assertNotIn("crunch_rules.yaml", rendered)

    def test_managed_rollout_actions_review_cli_reports_local_fraction_edit(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_review_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
                stderr=io.StringIO(),
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["planned_action_count"], 1)
        edit = payload["actions"][0]["proposed_edit"]
        self.assertTrue(edit["changed"])
        self.assertFalse(edit["disable"])
        self.assertEqual(edit["current_fraction"], 0.1)
        self.assertEqual(edit["recommended_fraction"], 0.25)
        self.assertEqual(edit["rollout"]["canary_fraction"], 0.25)
        self.assertEqual(payload["provenance"]["status"], "not-configured")

    def test_managed_rollout_actions_review_sends_metadata_only_lifecycle_feedback(self):
        bundle = self._rollout_action_bundle(action_type="widen", raw_like=False)
        ManagedFeedbackFlushClient.calls = []

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_review_cli(
                        ["--config-dir", tmp, "--db", db_path, "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                        stderr=io.StringIO(),
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "reviewed")
        self.assertEqual(sent_payload["policy_sections"], ["crunch"])
        self.assertTrue(str(sent_payload["bundle_hash"]).startswith("sha256:"))
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["action_type_counts"]["widen"], 1)
        self.assertEqual(metadata["policy_section_counts"]["crunch"], 1)
        self.assertFalse(metadata["privacy"]["file_paths_included"])
        rendered_payload = json.dumps(sent_payload)
        self.assertNotIn("crunch_rules.yaml", rendered_payload)
        self.assertNotIn("raw_request", rendered_payload)
        output = json.loads(stdout.getvalue())
        self.assertEqual(output["managed_lifecycle_feedback"]["status"], "sent")
        self.assertFalse(output["managed_lifecycle_feedback"]["payload_included"])

    def test_managed_rollout_actions_apply_dry_run_reports_without_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            after = path.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["applied_sections"], ["crunch"])
        self.assertTrue(payload["files"][0]["changed"])
        self.assertIsNone(payload["files"][0]["backup_path"])
        self.assertEqual(before, after)
        self.assertEqual(payload["actions"][0]["proposed_edit"]["rollout"]["canary_fraction"], 0.25)

    def test_managed_rollout_actions_accept_family_specific_safe_actions(self):
        specs = [
            ("crunch", "tool_results", "b", {"type": "shorten", "head_chars": 900, "tail_chars": 700, "max_replacement_chars": 2200, "preserve_tool_protocol": True}),
            ("crunch", "diffs", "c", {"type": "shorten", "head_chars": 1200, "tail_chars": 800, "max_replacement_chars": 2600, "marker": "[AgentFlow: diff headers and hunks preserved]", "preserve_diff_headers": True, "preserve_hunk_boundaries": True}),
            ("crunch", "generated_artifacts", "d", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: generated artifact marker preserved]", "exactness_preserving_marker": True}),
            ("crunch", "tabular_data", "e", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: tabular sample preserved]"}),
            ("crunch", "terminal_logs", "f", {"type": "shorten", "head_chars": 800, "tail_chars": 800, "max_replacement_chars": 2200, "marker": "[AgentFlow: terminal log errors preserved]"}),
            ("cache", "cacheability", "1", {"type": "exact_cache", "allow_tool_calls": False, "safe_invalidation": False}),
        ]
        actions = []
        crunch_rules = []
        cache_rules = []
        for section, family, hash_char, local_action in specs:
            pattern_hash = self._pattern_hash_char(hash_char)
            rule_id = f"managed-{section}-{family}"
            candidate_id = f"candidate-{section}-{family}"
            actions.append(
                self._rollout_action_for_rule(
                    section=section,
                    rule_id=rule_id,
                    candidate_id=candidate_id,
                    pattern_hash=pattern_hash,
                    module_family=family,
                    recommended_fraction=0.4,
                )
            )
            rule = self._family_pattern_rule(
                section=section,
                module_family=family,
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
                action=local_action,
                replayability_levels=["static_information"] if section == "cache" else None,
            )
            if section == "cache":
                cache_rules.append(rule)
            else:
                crunch_rules.append(rule)
        bundle = self._rollout_bundle_for_actions(actions)

        with TemporaryDirectory() as tmp:
            paths = self._write_rollout_pattern_files(tmp, crunch_rules=crunch_rules, cache_rules=cache_rules)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            crunch_written = yaml.safe_load(paths["crunch"].read_text(encoding="utf-8"))
            cache_written = yaml.safe_load(paths["cache"].read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["review"]["planned_action_count"], 6)
        self.assertEqual(payload["applied_sections"], ["cache", "crunch"])
        self.assertEqual({action["family_validation"]["status"] for action in payload["actions"]}, {"accepted"})
        self.assertEqual({action["family_validation"]["family"] for action in payload["actions"]}, {
            "cacheability",
            "diffs",
            "generated_artifacts",
            "tabular_data",
            "terminal_logs",
            "tool_results",
        })
        generated_rule = next(rule for rule in crunch_written["pattern_rules"] if rule["module_family"] == "generated_artifacts")
        self.assertEqual(generated_rule["rollout"]["canary_fraction"], 0.4)
        self.assertEqual(generated_rule["rollout"]["canary_unit"], "request_fingerprint")
        self.assertEqual(generated_rule["rollout_action"]["family_validation"]["family"], "generated_artifacts")
        cache_rule = cache_written["pattern_rules"][0]
        self.assertEqual(cache_rule["rollout"]["canary_fraction"], 0.4)
        self.assertEqual(cache_rule["rollout_action"]["family_validation"]["family"], "cacheability")

    def test_managed_rollout_actions_reject_family_specific_unsafe_actions_before_writing(self):
        ManagedFeedbackFlushClient.calls = []
        unsafe_specs = [
            (
                "cache",
                "tool_results",
                "2",
                {"type": "semantic_cache", "allow_tool_calls": True, "safe_invalidation": False},
                "semantic",
                ["current_state"],
            ),
            (
                "crunch",
                "diffs",
                "3",
                {"type": "omit", "head_chars": 0, "tail_chars": 0, "max_replacement_chars": 200},
                "lossy",
                None,
            ),
            (
                "crunch",
                "generated_artifacts",
                "4",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500},
                "conservative",
                None,
            ),
            (
                "crunch",
                "tabular_data",
                "5",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500, "marker": "[AgentFlow: table sample]"},
                "aggressive",
                None,
            ),
            (
                "crunch",
                "cacheability",
                "6",
                {"type": "shorten", "head_chars": 100, "tail_chars": 100, "max_replacement_chars": 500, "marker": "[AgentFlow: cacheability]"},
                "conservative",
                None,
            ),
        ]
        actions = []
        crunch_rules = []
        cache_rules = []
        for section, family, hash_char, local_action, profile, replayability in unsafe_specs:
            pattern_hash = self._pattern_hash_char(hash_char)
            rule_id = f"managed-unsafe-{section}-{family}"
            candidate_id = f"candidate-unsafe-{section}-{family}"
            actions.append(
                self._rollout_action_for_rule(
                    section=section,
                    rule_id=rule_id,
                    candidate_id=candidate_id,
                    pattern_hash=pattern_hash,
                    module_family=family,
                    policy_profile=profile,
                )
            )
            rule = self._family_pattern_rule(
                section=section,
                module_family=family,
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hash=pattern_hash,
                action=local_action,
                policy_profile=profile,
                replayability_levels=replayability,
            )
            if section == "cache":
                cache_rules.append(rule)
            else:
                crunch_rules.append(rule)
        bundle = self._rollout_bundle_for_actions(actions)

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            paths = self._write_rollout_pattern_files(tmp, crunch_rules=crunch_rules, cache_rules=cache_rules)
            before = {section: path.read_text(encoding="utf-8") for section, path in paths.items()}
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
            after = {section: path.read_text(encoding="utf-8") for section, path in paths.items()}

        self.assertEqual(code, 1)
        self.assertEqual(after, before)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertEqual({action["status"] for action in payload["actions"]}, {"rejected"})
        self.assertEqual({action["reason"] for action in payload["actions"]}, {"family-specific-validation-failed"})
        messages = {error["message"] for error in payload["review"]["errors"]}
        self.assertIn("semantic cache profiles are not accepted for managed pattern YAML rollout", messages)
        self.assertIn("diff crunching may not omit matched content", messages)
        self.assertIn("generated artifact crunching requires an exactness-preserving generated-artifact marker", messages)
        self.assertIn("cacheability pattern actions can only write cache rules", messages)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/policy-events")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["event_type"], "rejected")
        metadata = sent_payload["metadata"]
        self.assertEqual(metadata["local_status_counts"]["rejected"], 5)
        self.assertEqual(metadata["rejection_reason_counts"]["family-specific-validation-failed"], 5)
        self.assertEqual(metadata["family_validation_status_counts"]["rejected"], 5)
        self.assertFalse(metadata["privacy"]["file_paths_included"])
        rendered = json.dumps(sent_payload)
        self.assertNotIn("crunch_rules.yaml", rendered)
        self.assertNotIn("cache_rules.yaml", rendered)
        self.assertNotIn("raw_request", rendered)

    def test_managed_rollout_actions_apply_dry_run_queues_lifecycle_feedback_when_server_unavailable(self):
        bundle = self._rollout_action_bundle(action_type="widen")
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_RETRY_DELAY_SECONDS": "0",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_rollout_actions_apply_cli(
                        ["--config-dir", tmp, "--db", db_path, "--dry-run", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
            status_stdout = io.StringIO()
            cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "rollout_action_lifecycle"],
                stdout=status_stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "retryable-error")
        self.assertEqual(payload["managed_lifecycle_feedback"]["endpoint"], "/v1/policy-events")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        status_payload = json.loads(status_stdout.getvalue())
        self.assertEqual(status_payload["summary"]["retryable_error"], 1)
        self.assertEqual(status_payload["oldest_pending"]["source_surface"], "rollout_action_lifecycle")
        self.assertIsNone(status_payload["oldest_pending"]["optimization_unit_id"])
        rendered_status = status_stdout.getvalue()
        self.assertNotIn("crunch_rules.yaml", rendered_status)
        self.assertNotIn("raw_request", rendered_status)
        self.assertFalse(status_payload["privacy"]["payload_json_included"])

    def test_signed_managed_rollout_actions_apply_disables_rule_and_creates_backup(self):
        from agentflow_proxy.rollout_actions import attach_rollout_action_provenance

        secret = "rollout-secret"
        bundle = attach_rollout_action_provenance(
            self._rollout_action_bundle(action_type="rollback"),
            secret=secret,
            issuer="agentflow-server",
            server_id="local-dev",
            key_id="rollout-key",
        )

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRETS": json.dumps({"rollout-key": secret})}, clear=False):
                code = cli.managed_rollout_actions_apply_cli(
                    ["--config-dir", tmp, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
            written = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["provenance"]["status"], "verified")
            self.assertEqual(payload["applied_sections"], ["crunch"])
            self.assertTrue(payload["files"][0]["changed"])
            self.assertIsNotNone(payload["files"][0]["backup_path"])
            self.assertEqual(len(list(Path(tmp).glob("crunch_rules.yaml.bak-*"))), 1)
            rule = written["pattern_rules"][0]
            self.assertFalse(rule["enabled"])
            self.assertEqual(rule["rollout"]["canary_fraction"], 0.0)
            self.assertFalse(rule["rollout"]["canary_enabled"])
            self.assertEqual(rule["rollout_action"]["action_type"], "rollback")
            self.assertEqual(rule["rollout_action"]["pattern_hash"], self._pattern_hash())

        from agentflow_proxy.policy_events import recent_policy_events

        event = recent_policy_events(limit=1)["events"][0]
        self.assertEqual(event["action"], "rollout-actions-apply")
        self.assertTrue(event["ok"])

    def test_managed_rollout_actions_reject_raw_like_payload_before_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen", raw_like=True)

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            messages = [error["message"] for error in payload["validation"]["errors"]]
            self.assertIn("raw or prompt-like rollout action payloads are not accepted", messages)

    def test_managed_rollout_actions_reject_unknown_rule_before_writing(self):
        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            path = self._write_rollout_crunch_rule(tmp)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["pattern_rules"][0]["candidate_id"] = "different-candidate"
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            before = path.read_text(encoding="utf-8")
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmp).glob("crunch_rules.yaml.bak-*")), [])
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["review"]["actions"][0]["reason"], "unknown-rule")

    def _log_rollout_pattern_call(
        self,
        store,
        *,
        pattern_hash: str,
        cohort: str,
        status_code: int = 200,
        created_at: str = "2026-06-09T03:20:00+00:00",
        id_suffix: str = "",
        error: str | None = None,
    ):
        from agentflow_proxy.store import stable_json

        applied = cohort == "canary_applied"
        rule = {
            "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
            "candidate_id": "candidate-crunch-rollout",
            "policy_source": "managed-recommended",
            "applied_count": 1 if applied else 0,
            "saved_chars": 800 if applied else 0,
            "matched_hashes": [pattern_hash],
            "canary": {
                "enabled": True,
                "status": "holdout" if cohort == "canary_holdout" else "applied",
                "cohort": cohort,
                "fraction": 0.1,
            },
        }
        if cohort == "bypassed":
            rule["skip_reasons"] = [{"reason": "local-canary-safety-stop", "count": 1, "pattern_hash": pattern_hash}]
            rule["canary"]["status"] = "applied"

        store.log_call(
            id=f"call-{cohort}-{status_code}{id_suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=1234 if status_code < 400 else 12000,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({
                "changed": applied,
                "policy_source": "managed-recommended",
                "pattern_rules": {
                    "configured_count": 1,
                    "policy_source": "managed-recommended",
                    "category": "tool-result",
                    "rules": [rule],
                },
            }),
            routing_json=stable_json({"category": "tool-result"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=error,
            request_json=None,
            response_json=None,
            session_id="session-hidden",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )

    def test_managed_rollout_actions_dry_run_reports_recent_traffic_impact(self):
        from agentflow_proxy.store import Store, stable_json

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_applied")
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_holdout")
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="bypassed")
                store.log_codex_app_event(
                    id="codex-start-rollout",
                    created_at="2026-06-09T03:21:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-rollout",
                    thread_id="thread-rollout",
                    message_chars=100,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=100,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-session-hidden",
                    routing_json=stable_json({"category": "codex_turn"}),
                    crunch_json=stable_json({
                        "changed": True,
                        "policy_source": "managed-recommended",
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "category": "codex_turn",
                            "rules": [
                                {
                                    "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
                                    "candidate_id": "candidate-crunch-rollout",
                                    "policy_source": "managed-recommended",
                                    "applied_count": 1,
                                    "saved_chars": 400,
                                    "matched_hashes": [self._pattern_hash()],
                                    "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied", "fraction": 0.1},
                                }
                            ],
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled"}),
                )
                store.log_codex_app_event(
                    id="codex-end-rollout",
                    created_at="2026-06-09T03:21:01+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-rollout",
                    thread_id="thread-rollout",
                    message_chars=20,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=20,
                    error_code=None,
                    error_message=None,
                    latency_ms=100,
                    session_id="codex-session-hidden",
                )
            finally:
                store.conn.close()

            before_rule_text = (Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8")
            before_db_size = Path(db_path).stat().st_size
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                stdin=io.StringIO(json.dumps(bundle)),
                stdout=stdout,
            )
            after_rule_text = (Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8")
            after_db_size = Path(db_path).stat().st_size

        self.assertEqual(code, 0)
        self.assertEqual(before_rule_text, after_rule_text)
        self.assertEqual(before_db_size, after_db_size)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.pattern_rollout_actions_dry_run.v1")
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["wrote_policy_files"])
        self.assertFalse(payload["wrote_store"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertEqual(payload["summary"]["sampled_provider_calls"], 3)
        self.assertEqual(payload["summary"]["sampled_codex_turns"], 1)
        self.assertEqual(payload["summary"]["affected_metadata_row_count"], 4)
        action = payload["actions"][0]
        self.assertEqual(action["affected_provider_call_count"], 3)
        self.assertEqual(action["affected_codex_turn_count"], 1)
        self.assertEqual(action["current_canary_applied_count"], 2)
        self.assertEqual(action["current_canary_holdout_count"], 1)
        self.assertEqual(action["current_bypassed_or_disabled_count"], 1)
        self.assertEqual(action["current_fraction"], 0.1)
        self.assertEqual(action["projected_fraction"], 0.25)
        self.assertEqual(action["projected_canary_applied_count"], 2)
        self.assertGreater(action["historical_tokens_saved_est"], 0)
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        rendered = json.dumps(payload)
        self.assertNotIn("session-hidden", rendered)
        self.assertNotIn("codex-session-hidden", rendered)

    def test_managed_rollout_actions_impact_reports_post_apply_projection_deltas(self):
        from agentflow_proxy.policy_events import recent_policy_events
        from agentflow_proxy.store import Store, stable_json

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T03:20:00+00:00",
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_holdout",
                    created_at="2026-06-09T03:20:01+00:00",
                )
                dry_stdout = io.StringIO()
                self.assertEqual(
                    cli.managed_rollout_actions_dry_run_cli(
                        ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=dry_stdout,
                    ),
                    0,
                )
                dry_run_report = json.loads(dry_stdout.getvalue())
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T04:00:01+00:00",
                    id_suffix="-post",
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_holdout",
                    status_code=500,
                    created_at="2026-06-09T04:00:02+00:00",
                    id_suffix="-post",
                    error='{"error":{"type":"overloaded_error"}}',
                )
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="bypassed",
                    created_at="2026-06-09T04:00:03+00:00",
                    id_suffix="-post",
                )
                store.log_codex_app_event(
                    id="codex-start-rollout-post",
                    created_at="2026-06-09T04:00:04+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="req-rollout-post",
                    thread_id="thread-rollout-post",
                    message_chars=100,
                    params_chars=10,
                    input_items=1,
                    input_text_chars=100,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-session-hidden-post",
                    routing_json=stable_json({"category": "codex_turn"}),
                    crunch_json=stable_json({
                        "changed": True,
                        "policy_source": "managed-recommended",
                        "pattern_rules": {
                            "configured_count": 1,
                            "policy_source": "managed-recommended",
                            "category": "codex_turn",
                            "rules": [
                                {
                                    "rule_id": "managed-crunch-pattern-candidate-crunch-rollout",
                                    "candidate_id": "candidate-crunch-rollout",
                                    "policy_source": "managed-recommended",
                                    "applied_count": 1,
                                    "saved_chars": 400,
                                    "matched_hashes": [self._pattern_hash()],
                                    "canary": {"enabled": True, "status": "applied", "cohort": "canary_applied", "fraction": 0.25},
                                }
                            ],
                        },
                    }),
                    cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled"}),
                )
                store.log_codex_app_event(
                    id="codex-end-rollout-post",
                    created_at="2026-06-09T04:00:05+00:00",
                    direction="server_to_client",
                    method="turn/completed",
                    request_id="req-rollout-post",
                    thread_id="thread-rollout-post",
                    message_chars=20,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=20,
                    error_code=None,
                    error_message=None,
                    latency_ms=100,
                    session_id="codex-session-hidden-post",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "20", "--since", "2026-06-09T04:00:00+00:00", "-"],
                stdin=io.StringIO(json.dumps(dry_run_report)),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.pattern_rollout_actions_impact.v1")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "matched")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["wrote_policy_files"])
        self.assertFalse(payload["wrote_store"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertEqual(payload["summary"]["projected_affected_metadata_row_count"], 2)
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 4)
        self.assertEqual(payload["summary"]["actual_matched_provider_call_count"], 3)
        self.assertEqual(payload["summary"]["actual_matched_codex_turn_count"], 1)
        action = payload["actions"][0]
        self.assertTrue(action["action_id"].startswith("rollout-action:"))
        self.assertEqual(action["projection"]["affected_metadata_row_count"], 2)
        self.assertEqual(action["actual"]["matched_metadata_row_count"], 4)
        self.assertEqual(action["actual"]["actual_canary_applied_count"], 2)
        self.assertEqual(action["actual"]["actual_canary_holdout_count"], 1)
        self.assertEqual(action["actual"]["actual_bypassed_or_disabled_count"], 1)
        self.assertEqual(action["delta"]["matched_vs_projected_affected_delta"], 2)
        self.assertIn("overloaded_error", {row["value"] for row in action["actual"]["error_buckets"]})
        self.assertIn("gte_10s", {row["value"] for row in action["actual"]["latency_buckets"]})
        self.assertIn("changed", {row["value"] for row in action["actual"]["crunch_decision_status_buckets"]})
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        rendered = json.dumps(payload)
        self.assertNotIn("session-hidden", rendered)
        self.assertNotIn("req-rollout-post", rendered)
        self.assertNotIn("raw_request", rendered)
        self.assertEqual(recent_policy_events(limit=1)["events"][0]["action"], "rollout-actions-impact")

    def test_managed_rollout_actions_impact_is_useful_before_post_apply_matches(self):
        from agentflow_proxy.store import Store

        bundle = self._rollout_action_bundle(action_type="widen")

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(
                    store,
                    pattern_hash=self._pattern_hash(),
                    cohort="canary_applied",
                    created_at="2026-06-09T03:20:00+00:00",
                )
                dry_stdout = io.StringIO()
                self.assertEqual(
                    cli.managed_rollout_actions_dry_run_cli(
                        ["--config-dir", tmp, "--db", db_path, "--limit", "20", "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=dry_stdout,
                    ),
                    0,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_rollout_actions_impact_cli(
                ["--db", db_path, "--limit", "20", "--since", "2026-06-09T05:00:00+00:00", "-"],
                stdin=io.StringIO(dry_stdout.getvalue()),
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "no-post-apply-matches")
        self.assertEqual(payload["summary"]["actual_matched_metadata_row_count"], 0)
        self.assertEqual(payload["summary"]["actions_without_post_apply_matches"], 1)
        self.assertEqual(payload["actions"][0]["status"], "no-post-apply-matches")
        self.assertEqual(payload["actions"][0]["projection"]["affected_metadata_row_count"], 1)
        self.assertEqual(payload["actions"][0]["actual"]["matched_metadata_row_count"], 0)
        self.assertEqual(payload["actions"][0]["actual"]["status_risk_buckets"], [])

    def test_managed_rollout_actions_dry_run_covers_hold_rollback_unknown_and_raw_rejection(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._write_rollout_crunch_rule(tmp)
                self._log_rollout_pattern_call(store, pattern_hash=self._pattern_hash(), cohort="canary_applied")
            finally:
                store.conn.close()

            hold_bundle = self._rollout_action_bundle(action_type="hold")
            hold_bundle["actions"][0]["recommended_fraction"] = 0.1
            rollback_bundle = self._rollout_action_bundle(action_type="rollback")
            for bundle, expected_fraction, expected_disabled in (
                (hold_bundle, 0.1, 0),
                (rollback_bundle, 0.0, 1),
            ):
                stdout = io.StringIO()
                code = cli.managed_rollout_actions_dry_run_cli(
                    ["--config-dir", tmp, "--db", db_path, "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
                self.assertEqual(code, 0)
                action = json.loads(stdout.getvalue())["actions"][0]
                self.assertEqual(action["projected_fraction"], expected_fraction)
                self.assertEqual(action["projected_local_bypass_or_disable_count"], expected_disabled)

            unknown = self._rollout_action_bundle(action_type="widen")
            unknown["actions"][0]["target_rule_id"] = "missing-rule"
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "-"],
                stdin=io.StringIO(json.dumps(unknown)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["actions"][0]["reason"], "unknown-rule")
            self.assertEqual(payload["error"]["type"], "review_failed")

            raw_like = self._rollout_action_bundle(action_type="widen", raw_like=True)
            stdout = io.StringIO()
            code = cli.managed_rollout_actions_dry_run_cli(
                ["--config-dir", tmp, "--db", db_path, "-"],
                stdin=io.StringIO(json.dumps(raw_like)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            messages = [error["message"] for error in payload["validation"]["errors"]]
            self.assertIn("raw or prompt-like rollout action payloads are not accepted", messages)

    def test_policy_fetch_review_cli_without_config_skips_network(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(os.environ, {cli.POLICY_BUNDLE_RECOMMENDATION_URL_ENV: "", cli.MANAGED_POLICY_API_KEY_ENV: ""}, clear=False):
            with patch("agentflow_proxy.cli.httpx.get") as get:
                code = cli.policy_fetch_review_cli([], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "missing_url")
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_files"])
        get.assert_not_called()

    def test_policy_fetch_review_cli_fetches_reviews_and_does_not_write_rules(self):
        bundle = self._managed_policy_bundle()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_CONFIG_DIR": tmp, cli.MANAGED_POLICY_API_KEY_ENV: ""}, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--min-samples",
                            "3",
                            "--limit",
                            "7",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation"]["ok"])
        self.assertTrue(payload["review"]["ok"])
        self.assertEqual(payload["provenance"]["status"], "not-configured")
        self.assertEqual(payload["review"]["provenance"]["status"], "not-configured")
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_files"])
        self.assertEqual(payload["recommendation"]["candidate_ids"], ["candidate-route-chat"])
        self.assertEqual(payload["recommendation"]["candidates"][0]["confidence"], 0.82)
        self.assertEqual(payload["recommendation"]["codex_app_candidate_ids"], ["candidate-codex-summary"])
        self.assertEqual(payload["recommendation"]["codex_app_application_status"], "not-applied")
        self.assertTrue(payload["recommendation"]["codex_app_review_only"])
        codex_review = payload["review"]["section_reviews"]["codex_app"]
        self.assertEqual(codex_review["status"], "review-only")
        self.assertEqual(codex_review["application"]["status"], "not-applied")
        self.assertFalse(codex_review["application"]["writes_local_policy_files"])
        self.assertEqual(payload["bundle"]["schema"], "agentflow.policy_bundle.v1")
        self.assertIn("agentflow-policy-apply", payload["next_manual_command"])
        call = get.call_args
        self.assertEqual(call.kwargs["headers"], {})
        self.assertEqual(call.kwargs["params"]["min_samples"], 3)
        self.assertEqual(call.kwargs["params"]["limit"], 7)

    def test_policy_fetch_review_cli_fetches_signed_openai_review_bundle_with_capabilities(self):
        from agentflow_proxy import __version__
        from agentflow_proxy.policy_bundle import attach_policy_bundle_provenance

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        bundle = json.loads(exported.getvalue())
        supported = ["cache", "old_context_summarization", "routing"]
        selected = {
            "schema": "agentflow.openai_optimization_review_action.v1",
            "action_id": "openai-review:routing",
            "target_candidate_id": "openai-routing-candidate",
            "action_family": "routing",
            "candidate_family": "provider-routing-rule",
            "policy_section": "routing",
            "policy_source": "managed-recommended",
            "decision": "selected",
            "review_only": True,
            "required_local_review": True,
            "managed_enforced": False,
            "feature_only": True,
            "locally_executed": True,
            "provider_forwarding": False,
            "local_policy_surface": {
                "policy_file": "config/routing_rules.yaml",
                "policy_section": "routing",
                "writes_local_policy_files": False,
                "requires_local_apply": True,
            },
            "local_executor_compatibility": {
                "compatible": True,
                "supported_local_action_families": supported,
                "reason_codes": [],
            },
            "expected_impact": {"net_savings_usd": 0.05, "score": 0.91},
            "evidence_freshness": {"latest_event_at": "2026-06-10T12:00:00+00:00", "stale": False},
            "reason_codes": ["preferred-highest-ranked-openai-optimization"],
            "conflict_key": "openai_responses:gpt-5:routing",
        }
        suppressed = {
            **selected,
            "action_id": "openai-review:summary",
            "target_candidate_id": "openai-summary-candidate",
            "action_family": "old_context_summarization",
            "candidate_family": "old-context-summary-policy-rule",
            "policy_section": "crunch",
            "decision": "suppressed",
            "reason_codes": ["suppressed-by-higher-ranked-openai-optimization"],
            "suppressed_by": {"target_candidate_id": "openai-routing-candidate"},
            "local_policy_surface": {
                "policy_file": "config/crunch_rules.yaml",
                "policy_section": "crunch",
                "writes_local_policy_files": False,
                "requires_local_apply": True,
            },
        }
        omitted = {
            **selected,
            "action_id": "openai-review:cache",
            "target_candidate_id": "openai-cache-candidate",
            "action_family": "cache",
            "candidate_family": "cache-replay-policy-rule",
            "policy_section": "cache",
            "decision": "omitted",
            "reason_codes": ["local-executor-cache-unsupported"],
            "local_executor_compatibility": {
                "compatible": False,
                "supported_local_action_families": ["routing"],
                "reason_codes": ["local-executor-cache-unsupported"],
            },
            "local_policy_surface": {
                "policy_file": "config/cache_rules.yaml",
                "policy_section": "cache",
                "writes_local_policy_files": False,
                "requires_local_apply": True,
            },
        }
        bundle["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "recommendation_mode": "review-only",
            "note": "Review-only managed OpenAI recommendation.",
        }
        bundle["local_executor_compatibility"] = {
            "minimum_local_client_version": "0.1.0",
            "compatible": True,
            "supported_local_action_families": supported,
            "writes_local_policy_files": False,
            "provider_forwarding": False,
        }
        bundle["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "openai_optimization_schema": "agentflow.openai_optimization_review_bundle.v1",
            "created_at": bundle["generated_at"],
            "policy_source": "managed-recommended",
            "recommendation_mode": "review-only-openai-optimization-bundle",
            "required_local_review": True,
            "selected_action_count": 1,
            "suppressed_action_count": 1,
            "omitted_action_count": 1,
            "candidate_count": 3,
            "candidate_ids": ["openai-routing-candidate"],
            "policy_sections": ["routing"],
            "supported_local_action_families": supported,
            "expected_impact": {"net_savings_usd": 0.05, "source": "ranked-openai-optimization-lifecycle-feedback"},
            "conflict_summary": {
                "conflict_bucket_count": 1,
                "selected_action_count": 1,
                "suppressed_conflicting_action_count": 1,
                "suppressed_reason_counts": {"suppressed-by-higher-ranked-openai-optimization": 1},
                "omitted_reason_counts": {"local-executor-cache-unsupported": 1},
            },
        }
        bundle["policies"]["routing"]["policy_source"] = "managed-recommended"
        bundle["policies"]["routing"]["review_actions"] = [selected]
        bundle["policies"]["routing"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "selected_action_count": 1,
            "suppressed_action_count": 0,
            "omitted_action_count": 0,
            "selected_actions": [selected],
            "suppressed_actions": [],
            "omitted_actions": [],
        }
        bundle["openai_optimization"] = {
            "schema": "agentflow.openai_optimization_review_bundle.v1",
            "ranking_schema": "agentflow.openai_optimization_rollout_ranking.v1",
            "ranking_summary": {"candidate_count": 3, "conflict_bucket_count": 1},
            "selected_actions": [selected],
            "suppressed_actions": [suppressed],
            "omitted_actions": [omitted],
            "filters": {"source_surface": "openai_responses", "provider_endpoint": "responses"},
            "thresholds": {"max_retry_rate": 0.02},
        }
        bundle["privacy_summary"] = {
            "telemetry_profile": "metadata-only",
            "metadata_only": True,
            "feature_only": True,
            "raw_payloads_returned": False,
            "raw_prompts_returned": False,
            "raw_responses_returned": False,
            "provider_bodies_returned": False,
            "request_ids_returned": False,
            "tenant_ids_returned": False,
            "cache_keys_returned": False,
            "file_paths_returned": False,
            "provider_forwarding": False,
        }
        signed = attach_policy_bundle_provenance(
            bundle,
            secret="openai-review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="openai-review-key",
            generated_at="2026-06-10T12:00:00+00:00",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            env = {
                "AGENTFLOW_POLICY_CONFIG_DIR": tmp,
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                cli.MANAGED_POLICY_API_KEY_ENV: "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=signed)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/openai-optimization-review-bundle",
                            "--allow-unauthenticated",
                            "--source-surface",
                            "openai_responses",
                            "--provider-endpoint",
                            "responses",
                            "--requested-model-family",
                            "gpt-5",
                            "--max-retry-rate",
                            "0.02",
                            "--max-latency-regression-ms",
                            "250",
                            "--max-invalidation-rate",
                            "0.01",
                            "--supported-local-action-families",
                            "routing",
                            "--supported-local-action-families",
                            "old_context_summarization",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

        self.assertEqual(code, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["fetch"]["status"], "received")
        self.assertEqual(payload["provenance"]["status"], "verified")
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_files"])
        review = payload["openai_optimization_review"]
        self.assertEqual(review["status"], "present")
        self.assertEqual(review["selected_action_count"], 1)
        self.assertEqual(review["suppressed_action_count"], 1)
        self.assertEqual(review["omitted_action_count"], 1)
        self.assertEqual(review["counts_by_family"]["routing"]["selected"], 1)
        self.assertEqual(review["counts_by_family"]["old_context_summarization"]["suppressed"], 1)
        self.assertEqual(review["counts_by_family"]["cache"]["omitted"], 1)
        self.assertEqual(review["conflict_summary"]["conflict_bucket_count"], 1)
        self.assertEqual(review["local_capability_gaps"][0]["action_family"], "cache")
        self.assertIn("agentflow-policy-draft-stage", payload["next_manual_commands"][0])
        call = get.call_args
        self.assertEqual(call.kwargs["headers"]["x-agentflow-local-version"], __version__)
        self.assertEqual(call.kwargs["headers"]["x-agentflow-supported-local-action-families"], "old_context_summarization,routing")
        self.assertEqual(call.kwargs["params"]["provider_endpoint"], "responses")
        self.assertEqual(call.kwargs["params"]["requested_model_family"], "gpt-5")
        self.assertEqual(call.kwargs["params"]["max_retry_rate"], 0.02)
        self.assertEqual(call.kwargs["params"]["max_latency_regression_ms"], 250.0)
        self.assertEqual(call.kwargs["params"]["max_invalidation_rate"], 0.01)
        self.assertEqual(call.kwargs["params"]["supported_local_action_families"], ["old_context_summarization", "routing"])
        rendered_payload = json.dumps(payload, sort_keys=True)
        for forbidden in (
            '"raw_prompt"',
            '"raw_response"',
            '"provider_body"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, rendered_payload)

    def _openai_review_bundle_for_draft(self, *, mutate=None, expires_at: str | None = None, signed: bool = True):
        from agentflow_proxy.policy_bundle import attach_policy_bundle_provenance

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        bundle = json.loads(exported.getvalue())
        supported = ["cache", "old_context_summarization", "routing"]
        selected = {
            "schema": "agentflow.openai_optimization_review_action.v1",
            "action_id": "openai-review:routing",
            "target_candidate_id": "openai-routing-candidate",
            "action_family": "routing",
            "candidate_family": "provider-routing-rule",
            "policy_section": "routing",
            "policy_source": "managed-recommended",
            "decision": "selected",
            "review_only": True,
            "required_local_review": True,
            "managed_enforced": False,
            "feature_only": True,
            "locally_executed": True,
            "provider_forwarding": False,
            "local_policy_surface": {
                "policy_file": "config/routing_rules.yaml",
                "policy_section": "routing",
                "writes_local_policy_files": False,
                "requires_local_apply": True,
            },
            "local_executor_compatibility": {
                "compatible": True,
                "supported_local_action_families": supported,
                "reason_codes": [],
            },
            "local_policy_update": {
                "policy_source": "managed-recommended",
                "managed_enforced": False,
                "required_local_review": True,
                "requested_model_family": "gpt-5",
                "candidate_target_model": "gpt-5-mini",
                "openai_canary": {
                    "policy_id": "managed-openai-routing-candidate",
                    "model_pattern": "gpt-5",
                    "target_model": "gpt-5-mini",
                    "eligible_categories": ["chat", "short-completion"],
                    "excluded_categories": ["tool-heavy", "tool-result"],
                    "allow_tools": False,
                    "allow_stream": False,
                    "min_text_chars": 0,
                    "max_text_chars": 8000,
                    "safety_stop": {
                        "enabled": True,
                        "window_hours": 24,
                        "min_samples": 10,
                        "min_holdout_samples": 5,
                        "max_error_rate": 0.03,
                        "max_retry_rate": 0.1,
                        "max_fallback_rate": 0.1,
                        "max_latency_regression_ratio": 1.5,
                        "limit": 500,
                    },
                },
            },
            "expected_impact": {"net_savings_usd": 0.05, "score": 0.91},
            "reason_codes": ["preferred-highest-ranked-openai-optimization"],
            "conflict_key": "openai_responses:gpt-5:summary-cache-route",
        }
        suppressed_summary = {
            **selected,
            "action_id": "openai-review:summary",
            "target_candidate_id": "openai-summary-candidate",
            "action_family": "old_context_summarization",
            "candidate_family": "old-context-summary-policy-rule",
            "policy_section": "crunch",
            "decision": "suppressed",
            "reason_codes": ["suppressed-by-higher-ranked-openai-optimization"],
            "suppressed_by": {"target_candidate_id": "openai-routing-candidate"},
            "local_policy_update": {
                "old_context_summarization": {
                    "rule_id": "managed-openai-summary-candidate",
                    "candidate_id": "openai-summary-candidate",
                    "model": "gpt-5-mini",
                    "min_request_chars": 32000,
                }
            },
        }
        omitted_cache = {
            **selected,
            "action_id": "openai-review:cache",
            "target_candidate_id": "openai-cache-candidate",
            "action_family": "cache",
            "candidate_family": "cache-replay-policy-rule",
            "policy_section": "cache",
            "decision": "omitted",
            "reason_codes": ["local-executor-cache-unsupported"],
            "local_executor_compatibility": {
                "compatible": False,
                "supported_local_action_families": ["routing"],
                "reason_codes": ["local-executor-cache-unsupported"],
            },
            "local_policy_update": {
                "conditions": {"pattern_hash": "managed:openai-cache-candidate"},
                "action": {"type": "exact_cache_pattern", "allow_tool_calls": False},
            },
        }
        bundle["managed_optimizer"] = {
            "enabled": False,
            "policy_source": "managed-recommended",
            "recommendation_mode": "review-only",
        }
        bundle["local_executor_compatibility"] = {
            "minimum_local_client_version": "0.1.0",
            "compatible": True,
            "supported_local_action_families": supported,
            "writes_local_policy_files": False,
            "provider_forwarding": False,
        }
        bundle["recommendation"] = {
            "schema": "agentflow.policy_bundle_recommendation.v1",
            "openai_optimization_schema": "agentflow.openai_optimization_review_bundle.v1",
            "created_at": bundle["generated_at"],
            "expires_at": expires_at or "2099-06-11T12:00:00+00:00",
            "policy_source": "managed-recommended",
            "recommendation_mode": "review-only-openai-optimization-bundle",
            "required_local_review": True,
            "selected_action_count": 1,
            "suppressed_action_count": 1,
            "omitted_action_count": 1,
            "candidate_count": 3,
            "candidate_ids": ["openai-routing-candidate"],
            "policy_sections": ["routing"],
            "supported_local_action_families": supported,
            "conflict_summary": {
                "conflict_bucket_count": 1,
                "selected_action_count": 1,
                "suppressed_conflicting_action_count": 1,
                "suppressed_reason_counts": {"suppressed-by-higher-ranked-openai-optimization": 1},
                "omitted_reason_counts": {"local-executor-cache-unsupported": 1},
            },
        }
        bundle["openai_optimization"] = {
            "schema": "agentflow.openai_optimization_review_bundle.v1",
            "selected_actions": [selected],
            "suppressed_actions": [suppressed_summary],
            "omitted_actions": [omitted_cache],
            "filters": {"source_surface": "openai_responses", "provider_endpoint": "responses"},
            "thresholds": {"max_retry_rate": 0.02},
        }
        bundle["privacy_summary"] = {
            "telemetry_profile": "metadata-only",
            "metadata_only": True,
            "feature_only": True,
            "raw_payloads_returned": False,
            "raw_prompts_returned": False,
            "raw_responses_returned": False,
            "provider_bodies_returned": False,
            "request_ids_returned": False,
            "tenant_ids_returned": False,
            "cache_keys_returned": False,
            "file_paths_returned": False,
            "provider_forwarding": False,
        }
        if mutate is not None:
            mutate(bundle)
        if not signed:
            return bundle
        return attach_policy_bundle_provenance(
            bundle,
            secret="openai-review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="openai-review-key",
            generated_at="2026-06-10T12:00:00+00:00",
        )

    def test_policy_draft_stage_cli_stages_selected_openai_review_action_only(self):
        bundle = self._openai_review_bundle_for_draft()

        with TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            active_routing = config_dir / "routing_rules.yaml"
            active_routing.write_text("rules: []\n", encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_POLICY_CONFIG_DIR": str(config_dir),
                    "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                },
                clear=False,
            ):
                code = cli.policy_draft_stage_cli(
                    ["--draft-id", "openai-review-stage", "--workspace", str(workspace), "-"],
                    stdin=io.StringIO(json.dumps({"schema": "agentflow.policy_bundle_fetch_review.v1", "ok": True, "bundle": bundle})),
                    stdout=stdout,
                )

            self.assertEqual(code, 0, stdout.getvalue())
            self.assertEqual(active_routing.read_text(encoding="utf-8"), "rules: []\n")
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["wrote_active_policy_files"])
            self.assertEqual(payload["diff"]["changed_sections"], ["routing"])
            metadata = payload["draft"]["metadata"]["openai_optimization_review"]
            self.assertEqual(metadata["selected_action_count"], 1)
            self.assertEqual(metadata["suppressed_action_count"], 1)
            self.assertEqual(metadata["omitted_action_count"], 1)
            self.assertEqual(metadata["suppressed_actions"][0]["target_candidate_id"], "openai-summary-candidate")
            self.assertEqual(metadata["conflict_summary"]["conflict_bucket_count"], 1)

            staged = json.loads(Path(payload["bundle_path"]).read_text(encoding="utf-8"))
            canary = staged["policies"]["routing"]["openai"]["canary"]
            self.assertFalse(canary["enabled"])
            self.assertFalse(canary["review_only"])
            self.assertEqual(canary["policy_source"], "managed-recommended")
            self.assertEqual(canary["target_candidate_id"], "openai-routing-candidate")
            self.assertEqual(canary["target_model"], "gpt-5-mini")
            self.assertNotIn("openai-summary-candidate", json.dumps(staged["policies"], sort_keys=True))
            self.assertNotIn("openai-cache-candidate", json.dumps(staged["policies"], sort_keys=True))

            draft_yaml = yaml.safe_load((workspace / "openai-review-stage" / "sections" / "routing_rules.yaml").read_text(encoding="utf-8"))
            self.assertEqual(draft_yaml["openai_canary"]["target_candidate_id"], "openai-routing-candidate")
            self.assertFalse(draft_yaml["openai_canary"]["enabled"])

    def test_policy_draft_stage_cli_rejects_unsafe_openai_review_bundles(self):
        cases = [
            (
                self._openai_review_bundle_for_draft(expires_at="2000-01-01T00:00:00+00:00"),
                "expired",
            ),
            (
                self._openai_review_bundle_for_draft(signed=False),
                "provenance",
            ),
            (
                self._openai_review_bundle_for_draft(
                    mutate=lambda bundle: bundle["openai_optimization"]["selected_actions"][0].update({"action_family": "provider_body_rewrite"})
                ),
                "action_family",
            ),
            (
                self._openai_review_bundle_for_draft(
                    mutate=lambda bundle: bundle["openai_optimization"]["selected_actions"][0].update({"raw_prompt": "secret prompt"})
                ),
                "raw_prompt",
            ),
            (
                self._openai_review_bundle_for_draft(
                    mutate=lambda bundle: bundle["openai_optimization"]["selected_actions"][0].update({"provider_forwarding": True})
                ),
                "provider_forwarding",
            ),
            (
                self._openai_review_bundle_for_draft(
                    mutate=lambda bundle: bundle["openai_optimization"]["selected_actions"][0].pop("local_executor_compatibility", None)
                ),
                "local_executor_compatibility",
            ),
        ]

        for bundle, expected_path_fragment in cases:
            with TemporaryDirectory() as tmp:
                stdout = io.StringIO()
                with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret"}, clear=False):
                    code = cli.policy_draft_stage_cli(
                        ["--workspace", str(Path(tmp) / "drafts"), "-"],
                        stdin=io.StringIO(json.dumps(bundle)),
                        stdout=stdout,
                    )
                payload = json.loads(stdout.getvalue())
                self.assertEqual(code, 1, expected_path_fragment)
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["type"], "openai_optimization_review_rejected")
                paths = {error["path"] for error in payload["error"]["errors"]}
                messages = {error["message"] for error in payload["error"]["errors"]}
                self.assertTrue(
                    any(expected_path_fragment in path for path in paths)
                    or any(expected_path_fragment in message for message in messages),
                    (expected_path_fragment, paths, messages),
                )
                self.assertFalse((Path(tmp) / "drafts").exists())

    def test_openai_optimization_draft_dry_run_projects_governor_conflicts_without_writes(self):
        from agentflow_proxy.store import Store, stable_json

        def select_all_actions(bundle):
            selected = bundle["openai_optimization"]["selected_actions"][0]
            summary = bundle["openai_optimization"]["suppressed_actions"][0]
            cache = bundle["openai_optimization"]["omitted_actions"][0]
            for action in (selected, summary, cache):
                action["decision"] = "selected"
                action["local_executor_compatibility"] = {
                    "compatible": True,
                    "supported_local_action_families": ["cache", "old_context_summarization", "routing"],
                    "reason_codes": [],
                }
            bundle["openai_optimization"]["selected_actions"] = [selected, summary, cache]
            bundle["openai_optimization"]["suppressed_actions"] = []
            bundle["openai_optimization"]["omitted_actions"] = []
            bundle["recommendation"]["selected_action_count"] = 3
            bundle["recommendation"]["suppressed_action_count"] = 0
            bundle["recommendation"]["omitted_action_count"] = 0
            bundle["recommendation"]["candidate_ids"] = [
                "openai-routing-candidate",
                "openai-summary-candidate",
                "openai-cache-candidate",
            ]
            bundle["recommendation"]["policy_sections"] = ["routing", "crunch", "cache"]

        bundle = self._openai_review_bundle_for_draft(mutate=select_all_actions)

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            active_routing = config_dir / "routing_rules.yaml"
            active_crunch = config_dir / "crunch_rules.yaml"
            active_cache = config_dir / "cache_rules.yaml"
            active_routing.write_text("rules: []\n", encoding="utf-8")
            active_crunch.write_text("enabled: true\n", encoding="utf-8")
            active_cache.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_POLICY_CONFIG_DIR": str(config_dir),
                    "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                },
                clear=False,
            ):
                stage_stdout = io.StringIO()
                stage_code = cli.policy_draft_stage_cli(
                    ["--draft-id", "openai-review-all", "--workspace", str(workspace), "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stage_stdout,
                )
            self.assertEqual(stage_code, 0, stage_stdout.getvalue())

            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="openai-routing-cache-match",
                    created_at="2026-06-11T10:00:00+00:00",
                    path="/v1/responses",
                    requested_model="gpt-5",
                    routed_model="gpt-5",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.004,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    crunch_json=stable_json({}),
                    cache_json=stable_json({"status": "miss", "pattern_hash": "managed:openai-cache-candidate"}),
                    retry_count=0,
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    requested_model_family="gpt-5",
                    session_id="raw-session-id-must-not-leak",
                )
                store.log_call(
                    id="openai-summary-cache-match",
                    created_at="2026-06-11T10:01:00+00:00",
                    path="/v1/responses",
                    requested_model="gpt-5",
                    routed_model="gpt-5",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1200,
                    input_tokens_est=12000,
                    output_tokens_est=250,
                    actual_input_tokens=12000,
                    actual_output_tokens=250,
                    cost_est_usd=0.04,
                    cost_baseline_usd=0.04,
                    routing_json=stable_json({"category": "chat", "text_chars": 48000, "has_tools": False}),
                    crunch_json=stable_json({}),
                    cache_json=stable_json({"status": "miss", "pattern_hash": "managed:openai-cache-candidate"}),
                    retry_count=0,
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    requested_model_family="gpt-5",
                    session_id="raw-session-id-must-not-leak-2",
                )
            finally:
                store.conn.close()

            dry_stdout = io.StringIO()
            dry_code = cli.openai_optimization_draft_dry_run_cli(
                [
                    "openai-review-all",
                    "--workspace",
                    str(workspace),
                    "--db",
                    db_path,
                    "--queue-feedback",
                    "--pretty",
                ],
                stdout=dry_stdout,
            )

            self.assertEqual(active_routing.read_text(encoding="utf-8"), "rules: []\n")
            self.assertEqual(active_crunch.read_text(encoding="utf-8"), "enabled: true\n")
            self.assertEqual(active_cache.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")

            payload = json.loads(dry_stdout.getvalue())
            self.assertEqual(dry_code, 0, dry_stdout.getvalue())
            self.assertEqual(payload["schema"], "agentflow.openai_optimization_draft_dry_run.v1")
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["summary"]["openai_rows_considered"], 2)
            self.assertGreaterEqual(payload["families"]["routing"]["applied_if_enabled"], 1)
            self.assertGreaterEqual(payload["families"]["old_context_summary"]["applied_if_enabled"], 1)
            self.assertGreaterEqual(payload["families"]["cache_replay"]["applied_if_enabled"], 2)
            self.assertGreaterEqual(payload["families"]["cache_replay"]["suppressed"], 2)
            self.assertGreaterEqual(payload["families"]["cache_replay"]["conflict"], 2)
            self.assertGreater(payload["summary"]["expected_net_savings_usd"], 0)
            self.assertFalse(payload["privacy"]["provider_calls_made"])
            self.assertFalse(payload["privacy"]["managed_server_calls_made"])
            self.assertFalse(payload["privacy"]["active_policy_files_written"])
            self.assertIsNotNone(payload["feedback"]["queue_id"])
            rendered = dry_stdout.getvalue()
            self.assertNotIn("raw-session-id-must-not-leak", rendered)
            self.assertNotIn('"request_json"', rendered)
            self.assertNotIn('"response_json"', rendered)
            self.assertNotIn('"cache_key"', rendered)

            store = Store(db_path)
            try:
                queued = store.conn.execute(
                    "select source_surface, payload_json from managed_outcome_feedback_queue"
                ).fetchall()
            finally:
                store.conn.close()
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0]["source_surface"], "openai_optimization_lifecycle")
            self.assertNotIn("raw-session-id-must-not-leak", queued[0]["payload_json"])

            holdout_stdout = io.StringIO()
            holdout_code = cli.openai_optimization_draft_dry_run_cli(
                [
                    "openai-review-all",
                    "--workspace",
                    str(workspace),
                    "--db",
                    db_path,
                    "--canary-fraction",
                    "0",
                    "--holdout-fraction",
                    "1",
                ],
                stdout=holdout_stdout,
            )
            holdout = json.loads(holdout_stdout.getvalue())
            self.assertEqual(holdout_code, 0)
            self.assertGreaterEqual(holdout["summary"]["holdout_total"], 1)
            self.assertEqual(holdout["summary"]["applied_if_enabled_total"], 0)

    def test_openai_optimization_draft_dry_run_rejects_raw_like_staged_payload(self):
        from agentflow_proxy.store import Store

        bundle = self._openai_review_bundle_for_draft()

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret"}, clear=False):
                stage_stdout = io.StringIO()
                stage_code = cli.policy_draft_stage_cli(
                    ["--draft-id", "openai-review-raw-edited", "--workspace", str(workspace), "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stage_stdout,
                )
            self.assertEqual(stage_code, 0, stage_stdout.getvalue())
            staged_path = workspace / "openai-review-raw-edited" / "policy_bundle.json"
            staged = json.loads(staged_path.read_text(encoding="utf-8"))
            staged["policies"]["routing"]["openai"]["canary"]["managed_recommendation"]["raw_prompt"] = "raw prompt must not leak"
            staged_path.write_text(json.dumps(staged), encoding="utf-8")

            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            store.conn.close()
            stdout = io.StringIO()
            code = cli.openai_optimization_draft_dry_run_cli(
                ["openai-review-raw-edited", "--workspace", str(workspace), "--db", db_path],
                stdout=stdout,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertIn("raw_prompt", {error["path"].split(".")[-1] for error in payload["error"]["errors"]})
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_openai_optimization_draft_apply_dry_run_write_and_rollback(self):
        import hashlib

        from agentflow_proxy.policy_workbench import rollback_policy_apply
        from agentflow_proxy.store import Store

        bundle = self._openai_review_bundle_for_draft()

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "drafts"
            config_dir = root / "config"
            config_dir.mkdir()
            events_log = root / "policy_events.jsonl"
            routing_path = config_dir / "routing_rules.yaml"
            crunch_path = config_dir / "crunch_rules.yaml"
            cache_path = config_dir / "cache_rules.yaml"
            before_routing = "rules: []\n"
            before_crunch = "enabled: true\n"
            before_cache = "exact_cache:\n  enabled: true\n"
            routing_path.write_text(before_routing, encoding="utf-8")
            crunch_path.write_text(before_crunch, encoding="utf-8")
            cache_path.write_text(before_cache, encoding="utf-8")

            env = {
                "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                "AGENTFLOW_POLICY_EVENTS_LOG": str(events_log),
            }
            with patch.dict(os.environ, env, clear=False):
                stage_stdout = io.StringIO()
                stage_code = cli.policy_draft_stage_cli(
                    ["--draft-id", "openai-review-apply", "--workspace", str(workspace), "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stage_stdout,
                )
            self.assertEqual(stage_code, 0, stage_stdout.getvalue())

            db_path = str(root / "agentflow.sqlite3")
            store = Store(db_path)
            store.conn.close()

            dry_stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_EVENTS_LOG": str(events_log)}, clear=False):
                dry_code = cli.openai_optimization_draft_apply_cli(
                    [
                        "openai-review-apply",
                        "--workspace",
                        str(workspace),
                        "--config-dir",
                        str(config_dir),
                        "--db",
                        db_path,
                        "--canary-fraction",
                        "0.25",
                        "--holdout-fraction",
                        "0.15",
                    ],
                    stdout=dry_stdout,
                )

            dry_payload = json.loads(dry_stdout.getvalue())
            self.assertEqual(dry_code, 0, dry_stdout.getvalue())
            self.assertEqual(dry_payload["schema"], "agentflow.openai_optimization_draft_apply.v1")
            self.assertTrue(dry_payload["dry_run"])
            self.assertFalse(dry_payload["privacy"]["active_policy_files_written"])
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)

            apply_stdout = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_EVENTS_LOG": str(events_log)}, clear=False):
                apply_code = cli.openai_optimization_draft_apply_cli(
                    [
                        "openai-review-apply",
                        "--workspace",
                        str(workspace),
                        "--config-dir",
                        str(config_dir),
                        "--db",
                        db_path,
                        "--write",
                        "--canary-fraction",
                        "0.25",
                        "--holdout-fraction",
                        "0.15",
                        "--queue-feedback",
                    ],
                    stdout=apply_stdout,
                )

            payload = json.loads(apply_stdout.getvalue())
            self.assertEqual(apply_code, 0, apply_stdout.getvalue())
            self.assertEqual(payload["status"], "applied")
            self.assertTrue(payload["privacy"]["active_policy_files_written"])
            self.assertEqual(payload["changed_sections"], ["routing"])
            self.assertIn("agentflow-policy-rollback", payload["rollback_command"])
            self.assertTrue(Path(payload["backups"][0]["path"]).exists())
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), before_crunch)
            self.assertEqual(cache_path.read_text(encoding="utf-8"), before_cache)

            written = yaml.safe_load(routing_path.read_text(encoding="utf-8"))
            canary = written["openai_canary"]
            self.assertTrue(canary["enabled"])
            self.assertEqual(canary["target_candidate_id"], "openai-routing-candidate")
            self.assertEqual(canary["policy_source"], "managed-recommended")
            self.assertEqual(canary["target_model"], "gpt-5-mini")
            self.assertEqual(canary["canary_fraction"], 0.25)
            self.assertEqual(canary["holdout_fraction"], 0.15)
            self.assertEqual(canary["safety_stop"]["max_error_rate"], 0.03)

            restored_sha = hashlib.sha256(before_routing.encode("utf-8")).hexdigest()

            async def reload_state():
                return {
                    "ok": True,
                    "policies": {
                        "routing": {
                            "file": {
                                "loaded": {"sha256": restored_sha},
                                "current": {"sha256": restored_sha},
                                "reload_required": False,
                            }
                        }
                    },
                }

            with patch.dict(os.environ, {"AGENTFLOW_POLICY_EVENTS_LOG": str(events_log)}, clear=False):
                rollback = asyncio.run(
                    rollback_policy_apply(
                        payload["apply_id"],
                        config_dir=config_dir,
                        sections=["routing"],
                        reload_policy_state=reload_state,
                        event_source="test",
                    )
                )

            self.assertTrue(rollback["ok"], rollback)
            self.assertEqual(rollback["status"], "rolled-back")
            self.assertEqual(routing_path.read_text(encoding="utf-8"), before_routing)

    def test_openai_optimization_draft_apply_blocks_unsafe_staged_edits(self):
        from agentflow_proxy.store import Store

        cases = [
            (
                lambda staged, manifest: staged["policies"]["routing"]["openai"]["canary"]["managed_recommendation"].update({"raw_prompt": "secret"}),
                "raw_prompt",
                False,
            ),
            (
                lambda staged, manifest: staged["policies"]["routing"]["openai"]["canary"]["managed_recommendation"].update({"expires_at": "2000-01-01T00:00:00+00:00"}),
                "expired",
                False,
            ),
            (
                lambda staged, manifest: staged["policies"]["routing"]["openai"]["canary"]["safety_stop"].update({"tripped": True}),
                "safety_stop",
                False,
            ),
            (
                lambda staged, manifest: manifest["metadata"]["openai_optimization_review"]["selected_actions"][0].update({"action_family": "provider_body_rewrite"}),
                "action_family",
                False,
            ),
            (
                lambda staged, manifest: None,
                "provenance",
                True,
            ),
        ]

        for mutate, expected, require_verified in cases:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                workspace = root / "drafts"
                config_dir = root / "config"
                config_dir.mkdir()
                events_log = root / "policy_events.jsonl"
                routing_path = config_dir / "routing_rules.yaml"
                routing_path.write_text("rules: []\n", encoding="utf-8")
                db_path = str(root / "agentflow.sqlite3")
                store = Store(db_path)
                store.conn.close()

                with patch.dict(
                    os.environ,
                    {
                        "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                        "AGENTFLOW_POLICY_EVENTS_LOG": str(events_log),
                    },
                    clear=False,
                ):
                    stage_stdout = io.StringIO()
                    stage_code = cli.policy_draft_stage_cli(
                        ["--draft-id", "unsafe-apply", "--workspace", str(workspace), "-"],
                        stdin=io.StringIO(json.dumps(self._openai_review_bundle_for_draft())),
                        stdout=stage_stdout,
                    )
                self.assertEqual(stage_code, 0, stage_stdout.getvalue())

                staged_path = workspace / "unsafe-apply" / "policy_bundle.json"
                manifest_path = workspace / "unsafe-apply" / "draft.json"
                staged = json.loads(staged_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(staged, manifest)
                staged_path.write_text(json.dumps(staged), encoding="utf-8")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                stdout = io.StringIO()
                stderr = io.StringIO()
                env = {"AGENTFLOW_POLICY_EVENTS_LOG": str(events_log)}
                if not require_verified:
                    env["AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET"] = "openai-review-secret"
                with patch.dict(os.environ, env, clear=True):
                    code = cli.openai_optimization_draft_apply_cli(
                        [
                            "unsafe-apply",
                            "--workspace",
                            str(workspace),
                            "--config-dir",
                            str(config_dir),
                            "--db",
                            db_path,
                            "--write",
                        ]
                        + (["--require-verified-provenance"] if require_verified else []),
                        stdout=stdout,
                        stderr=stderr,
                    )

                rendered = stdout.getvalue() + stderr.getvalue()
                payload = json.loads(rendered)
                self.assertEqual(code, 1, expected)
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["type"], "validation_failed")
                self.assertIn(expected, rendered)
                self.assertEqual(routing_path.read_text(encoding="utf-8"), "rules: []\n")
                self.assertNotIn("secret", rendered)

    def test_openai_optimization_draft_apply_blocks_governor_conflicts(self):
        from agentflow_proxy.store import Store, stable_json

        def select_all_actions(bundle):
            selected = bundle["openai_optimization"]["selected_actions"][0]
            summary = bundle["openai_optimization"]["suppressed_actions"][0]
            cache = bundle["openai_optimization"]["omitted_actions"][0]
            for action in (selected, summary, cache):
                action["decision"] = "selected"
                action["local_executor_compatibility"] = {
                    "compatible": True,
                    "supported_local_action_families": ["cache", "old_context_summarization", "routing"],
                    "reason_codes": [],
                }
            bundle["openai_optimization"]["selected_actions"] = [selected, summary, cache]
            bundle["openai_optimization"]["suppressed_actions"] = []
            bundle["openai_optimization"]["omitted_actions"] = []
            bundle["recommendation"]["candidate_ids"] = [
                "openai-routing-candidate",
                "openai-summary-candidate",
                "openai-cache-candidate",
            ]
            bundle["recommendation"]["policy_sections"] = ["routing", "crunch", "cache"]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "drafts"
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "routing_rules.yaml").write_text("rules: []\n", encoding="utf-8")
            events_log = root / "policy_events.jsonl"

            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "openai-review-secret",
                    "AGENTFLOW_POLICY_EVENTS_LOG": str(events_log),
                },
                clear=False,
            ):
                stage_stdout = io.StringIO()
                stage_code = cli.policy_draft_stage_cli(
                    ["--draft-id", "conflict-apply", "--workspace", str(workspace), "-"],
                    stdin=io.StringIO(json.dumps(self._openai_review_bundle_for_draft(mutate=select_all_actions))),
                    stdout=stage_stdout,
                )
            self.assertEqual(stage_code, 0, stage_stdout.getvalue())

            db_path = str(root / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="openai-conflict-row",
                    created_at="2026-06-11T10:00:00+00:00",
                    path="/v1/responses",
                    requested_model="gpt-5",
                    routed_model="gpt-5",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.004,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    crunch_json=stable_json({}),
                    cache_json=stable_json({"status": "miss", "pattern_hash": "managed:openai-cache-candidate"}),
                    retry_count=0,
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    requested_model_family="gpt-5",
                    session_id="raw-session-id-must-not-leak",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_EVENTS_LOG": str(events_log)}, clear=False):
                code = cli.openai_optimization_draft_apply_cli(
                    [
                        "conflict-apply",
                        "--workspace",
                        str(workspace),
                        "--config-dir",
                        str(config_dir),
                        "--db",
                        db_path,
                        "--write",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )

            rendered = stdout.getvalue() + stderr.getvalue()
            payload = json.loads(rendered)
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            self.assertIn("local governor projection", rendered)
            self.assertNotIn("raw-session-id-must-not-leak", rendered)
            self.assertEqual((config_dir / "routing_rules.yaml").read_text(encoding="utf-8"), "rules: []\n")

    def test_codex_app_policy_dry_run_projects_synthetic_fixture_and_recent_rows_without_mutation(self):
        from agentflow_proxy.store import Store, stable_json, utc_now

        bundle = self._managed_policy_bundle()
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            policy_path = Path(tmp) / "managed-codex-app-bundle.json"
            fixture_path = Path(tmp) / "codex-fixture.json"
            policy_path.write_text(json.dumps(bundle), encoding="utf-8")
            fixture_path.write_text(
                json.dumps({
                    "rows": [
                        {
                            "workflow_phase": "summary",
                            "model_field_state": "derived_present",
                            "input_size_bucket": "small",
                            "cache_eligible": False,
                            "cache_status": "skipped",
                            "input_text_chars": 700,
                            "result_chars": 80,
                            "requested_model": "gpt-5.4",
                            "raw_prompt": "fixture raw prompt must not leak",
                        }
                    ]
                }),
                encoding="utf-8",
            )
            store = Store(db_path)
            try:
                store.set_cache("codex-dry-run-existing-cache-key", "gpt-5.4", 10, {"result": "cached"})
                store.log_codex_app_event(
                    id="codex-dry-run-event",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    request_id="raw-request-id-must-not-leak",
                    thread_id="raw-thread-id-must-not-leak",
                    session_id="raw-session-id-must-not-leak",
                    message_chars=1200,
                    params_chars=900,
                    input_items=1,
                    input_text_chars=600,
                    result_chars=120,
                    routing_json=stable_json({
                        "requested_model": "gpt-5.4",
                        "workflow_phase": "summary",
                    }),
                    crunch_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
                    cache_json=stable_json({
                        "status": "skipped",
                        "eligible": False,
                        "replayability_level": "turn-metadata-only",
                        "policy_source": "local-default",
                    }),
                    event_window_json=stable_json({
                        "schema": "agentflow.codex_app_event_window.v1",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                        "input_text_chars": 600,
                        "result_chars": 120,
                        "session_id": "raw-session-id-must-not-leak",
                        "request_id": "raw-request-id-must-not-leak",
                    }),
                    metadata_json=stable_json({"kind": "turn_window"}),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_app_policy_dry_run_cli(
                [str(policy_path), "--db", db_path, "--fixture", str(fixture_path), "--recent-limit", "10"],
                stdout=stdout,
                stderr=io.StringIO(),
            )

            with sqlite3.connect(db_path) as conn:
                cache_rows = conn.execute("select count(*) from cache").fetchone()[0]

        self.assertEqual(code, 0)
        self.assertEqual(cache_rows, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.codex_app_policy_dry_run.v1")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["applied"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertFalse(payload["cache_table_mutated"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertEqual(payload["managed_lifecycle_feedback"]["event_phase"], "dry_run")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        self.assertIn("candidate-codex-summary", payload["managed_lifecycle_feedback"]["results"])
        self.assertEqual(payload["summary"]["synthetic_rows"], 1)
        self.assertEqual(payload["summary"]["fixture_rows"], 1)
        self.assertEqual(payload["summary"]["recent_rows"], 1)
        self.assertEqual(payload["summary"]["projected_applied_count"], 3)
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "candidate-codex-summary")
        self.assertEqual(candidate["projected_applied_count"], 3)
        self.assertIn("model_hint", candidate["action_keys"])
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("fixture raw prompt must not leak", encoded)
        self.assertNotIn("raw-request-id-must-not-leak", encoded)
        self.assertNotIn("raw-session-id-must-not-leak", encoded)
        self.assertNotIn("codex-dry-run-existing-cache-key", encoded)
        self.assertFalse(payload["privacy"]["raw_payloads_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])

    def test_codex_app_policy_dry_run_fetches_bundle_without_writing_rules(self):
        bundle = self._managed_policy_bundle()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            config_dir = Path(tmp) / "config"
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_CONFIG_DIR": str(config_dir), cli.MANAGED_POLICY_API_KEY_ENV: ""}, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.codex_app_policy_dry_run_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--db",
                            db_path,
                            "--recent-limit",
                            "0",
                            "--no-synthetic",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((config_dir / "codex_app_rules.yaml").exists())

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation"]["ok"])
        self.assertTrue(payload["managed_server_calls_made"])
        self.assertFalse(payload["wrote_local_policy_files"])
        self.assertEqual(payload["fetch"]["status"], "received")
        self.assertEqual(payload["summary"]["candidate_count"], 1)
        call = get.call_args
        self.assertEqual(call.kwargs["params"]["limit"], 50)

    def test_policy_fetch_review_cli_surfaces_pattern_candidates_without_raw_leakage(self):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["candidate_ids"].extend([
            "pattern-crunch-representable",
            "pattern-cache-health-changed",
            "pattern-cache-omitted",
            "pattern-cache-unchanged",
        ])
        bundle["recommendation"]["candidate_count"] = 5
        bundle["policies"]["crunch"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-crunch-representable",
                    "candidate_family": "crunch-policy-rule",
                    "confidence": 0.81,
                    "sample_count": 64,
                    "estimated_savings_usd": 2.5,
                    "action": {
                        "crunch_profile": "repeated-section-dedupe",
                        "command": "raw command must not print",
                    },
                    "local_action_requirements": {
                        "expected_policy_section": "crunch",
                        "actionability_status": "review-only-local-action",
                    },
                    "confidence_inputs": {
                        "score_family": "crunch-policy-rule",
                        "privacy_profile_counts": {"metadata-only": 64},
                    },
                    "review_evidence": {
                        "crunch": {"saved_tokens_est": 4200},
                        "raw_policy_yaml": "raw yaml must not print",
                    },
                }
            ],
        }
        bundle["policies"]["cache"]["recommendation"] = {
            "policy_source": "managed-recommended",
            "candidate_count": 2,
            "omitted_candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": "pattern-cache-health-changed",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.66,
                    "sample_count": 18,
                    "delta": {"status": "changed-health"},
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {"api_key": "secret must not print"},
                },
                {
                    "candidate_id": "pattern-cache-unchanged",
                    "candidate_family": "cache-policy-rule",
                    "confidence": 0.58,
                    "sample_count": 14,
                    "delta": {"status": "unchanged"},
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                },
            ],
            "omitted_candidates": [
                {
                    "candidate_id": "pattern-cache-omitted",
                    "candidate_family": "cache-policy-rule",
                    "reason": "cache-policy-rule-not-representable-in-local-bundle-schema-yet",
                    "sample_count": 7,
                    "local_action_requirements": {
                        "expected_policy_section": "cache",
                        "actionability_status": "review-only-local-action",
                    },
                    "evidence": {"raw_response": "raw provider body must not print"},
                }
            ],
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            env = {
                "AGENTFLOW_POLICY_CONFIG_DIR": tmp,
                "AGENTFLOW_POLICY_EVENTS_LOG": str(Path(tmp) / "policy_events.jsonl"),
                cli.MANAGED_POLICY_API_KEY_ENV: "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--pretty",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

        payload = json.loads(stdout.getvalue())
        rendered = stdout.getvalue() + stderr.getvalue()

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["recommendation"]["pattern_candidate_count"], 4)
        self.assertEqual(payload["recommendation"]["crunch_pattern_candidate_count"], 1)
        self.assertEqual(payload["recommendation"]["cache_pattern_candidate_count"], 3)
        self.assertEqual(payload["review"]["section_reviews"]["crunch"]["candidate_count"], 1)
        self.assertEqual(payload["review"]["section_reviews"]["cache"]["changed_health_candidate_count"], 1)
        self.assertEqual(payload["review"]["section_reviews"]["cache"]["unchanged_candidate_count"], 1)
        self.assertIn("crunch pattern candidates: 1 total", " ".join(payload["review"]["human_summary"]))
        self.assertNotIn("raw command must not print", rendered)
        self.assertNotIn("raw yaml must not print", rendered)
        self.assertNotIn("secret must not print", rendered)
        self.assertNotIn("raw provider body must not print", rendered)

    def test_policy_fetch_review_cli_surfaces_managed_health_without_raw_leakage(self):
        bundle = self._managed_policy_bundle()
        bundle["recommendation"]["health"] = {
            "generated_at": "2026-06-08T12:00:00+00:00",
            "privacy_summary": {
                "telemetry_profile": "metadata-only",
                "raw_body_storage": False,
                "raw_prompts_included": False,
            },
            "stale_evidence": [
                {
                    "candidate_id": "candidate-route-chat",
                    "last_seen_at": "2026-06-01T12:00:00+00:00",
                    "raw_prompt": "raw prompt secret must not print",
                }
            ],
            "insufficient_samples": [
                {
                    "candidate_id": "candidate-route-chat",
                    "sample_count": 2,
                    "min_samples": 10,
                    "body": "raw request body must not print",
                }
            ],
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            env = {
                "AGENTFLOW_POLICY_CONFIG_DIR": tmp,
                "AGENTFLOW_POLICY_EVENTS_LOG": str(Path(tmp) / "policy_events.jsonl"),
                cli.MANAGED_POLICY_API_KEY_ENV: "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=bundle)
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                            "--min-samples",
                            "10",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )

                from agentflow_proxy.policy_events import recent_policy_events

                events = recent_policy_events(limit=5)["events"]

        rendered = stdout.getvalue() + stderr.getvalue()
        payload = json.loads(stdout.getvalue())
        warning_codes = {warning["code"] for warning in payload["review"]["safety_warnings"]}

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["validation"]["ok"])
        self.assertTrue(payload["review"]["ok"])
        self.assertEqual(payload["recommendation"]["health"]["status"], "warning")
        self.assertEqual(payload["recommendation"]["health"]["counts"]["stale_evidence"], 1)
        self.assertEqual(payload["recommendation"]["health"]["counts"]["insufficient_samples"], 1)
        self.assertIn("managed-recommendation-stale-evidence", warning_codes)
        self.assertIn("managed-recommendation-insufficient-samples", warning_codes)
        self.assertNotIn("raw prompt secret", rendered)
        self.assertNotIn("raw request body", rendered)
        self.assertNotIn('"body"', rendered)
        self.assertEqual(events[0]["details"]["recommendation_health"]["warning_count"], 2)

    def test_policy_fetch_review_cli_rejects_invalid_bundle(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTFLOW_POLICY_CONFIG_DIR": tmp}, clear=False):
                with patch("agentflow_proxy.cli.httpx.get") as get:
                    get.return_value = httpx.Response(200, json=self._managed_policy_bundle(invalid=True))
                    code = cli.policy_fetch_review_cli(
                        [
                            "--url",
                            "http://managed.test/v1/policy-bundle-recommendation",
                            "--allow-unauthenticated",
                        ],
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.assertEqual(list(Path(tmp).glob("*.yaml")), [])

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertIn("$.schema", {error["path"] for error in payload["validation"]["errors"]})
        self.assertFalse(payload["wrote_local_files"])

    def test_policy_fetch_review_cli_sends_auth_without_secret_leakage(self):
        secret = "super-secret-managed-key"
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.dict(os.environ, {cli.MANAGED_POLICY_API_KEY_ENV: secret}, clear=False):
            with patch("agentflow_proxy.cli.httpx.get") as get:
                get.return_value = httpx.Response(200, json=self._managed_policy_bundle())
                code = cli.policy_fetch_review_cli(
                    [
                        "--url",
                        "http://managed.test/v1/policy-bundle-recommendation",
                        "--tenant",
                        "tenant-a",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                )

        self.assertEqual(code, 0)
        self.assertEqual(get.call_args.kwargs["headers"]["authorization"], f"Bearer {secret}")
        self.assertEqual(get.call_args.kwargs["headers"]["x-agentflow-tenant"], "tenant-a")
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret, rendered)

        from agentflow_proxy.policy_events import recent_policy_events

        event_text = json.dumps(recent_policy_events(limit=5)["events"])
        self.assertNotIn(secret, event_text)
        self.assertIn("env:AGENTFLOW_MANAGED_API_KEY", event_text)

    def test_policy_apply_cli_dry_run_reports_files_without_writing(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--dry-run", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(set(payload["applied_sections"]), {"routing", "crunch", "cache", "routing_experiments", "codex_app"})
            self.assertFalse(payload["skipped_sections"])
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())
            self.assertTrue(any(file["section"] == "cache" and file["changed"] for file in payload["files"]))
            self.assertTrue(any(file["section"] == "codex_app" for file in payload["files"]))

    def test_policy_draft_stage_cli_stages_section_diff_without_active_writes(self):
        with TemporaryDirectory() as tmp:
            active_path = Path(tmp) / "cache_rules.yaml"
            active_text = "exact_cache:\n  enabled: true\nsemantic_cache:\n  enabled: false\n  threshold: 0.95\n"
            active_path.write_text(active_text, encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            stdout = io.StringIO()

            from agentflow_proxy import cache as cache_module
            from agentflow_proxy.policy_files import policy_file_snapshot, utc_now

            with (
                patch.object(cache_module, "CACHE_RULES_PATH", str(active_path)),
                patch.object(cache_module, "CACHE_POLICY_SOURCE", "local-manual"),
                patch.object(cache_module, "CACHE_RULES_LOADED_AT", utc_now()),
                patch.object(cache_module, "CACHE_RULES_LOADED_FILE", policy_file_snapshot(active_path)),
                patch.object(cache_module, "CACHE_ENABLED", True),
                patch.object(cache_module, "SEMANTIC_CACHE_ENABLED", False),
                patch.object(cache_module, "SEMANTIC_CACHE_THRESHOLD", 0.95),
            ):
                code = cli.policy_draft_stage_cli(
                    [
                        "--section",
                        "cache",
                        "--draft-id",
                        "cli-cache-draft",
                        "--workspace",
                        str(workspace),
                        "-",
                    ],
                    stdin=io.StringIO("semantic_cache:\n  enabled: true\n  threshold: 0.91\n"),
                    stdout=stdout,
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["draft_id"], "cli-cache-draft")
            self.assertEqual(payload["diff"]["changed_sections"], ["cache"])
            self.assertFalse(payload["wrote_active_policy_files"])
            self.assertFalse(payload["provider_calls_made"])
            self.assertEqual(active_path.read_text(encoding="utf-8"), active_text)
            self.assertTrue((workspace / "cli-cache-draft" / "draft.json").exists())
            cache_section = {section["section"]: section for section in payload["sections"]}["cache"]
            self.assertEqual(cache_section["target_file"], str(active_path))
            self.assertIn("$.policies.cache.semantic_cache.threshold", {change["path"] for change in cache_section["changes"]})

    def test_policy_draft_stage_cli_rejects_raw_provider_payloads(self):
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            code = cli.policy_draft_stage_cli(
                ["--section", "cache", "--workspace", str(Path(tmp) / "drafts"), "-"],
                stdin=io.StringIO("raw_response:\n  content: secret\n"),
                stdout=stdout,
            )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "raw_payload_rejected")
            self.assertFalse((Path(tmp) / "drafts").exists())

    def _routing_promotion_report(self, *, extra_candidate=None):
        promoted = {
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "stream": True,
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-haiku-4-5-20251001",
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "mode": "shadow_candidate_pass_through",
            "samples": 24,
            "compared_samples": 24,
            "compared_coverage": 1.0,
            "avg_similarity": 0.94,
            "pass_rate": 1.0,
            "primary_error_rate": 0.0,
            "shadow_error_rate": 0.0,
            "fallback_or_retry_count": 0,
            "cost_delta_usd": 0.42,
            "avg_latency_delta_ms": -12.5,
            "last_sample_at": "2026-06-10T22:00:00+00:00",
            "last_sample_age_hours": 1.0,
            "promotion": {
                "schema": "agentflow.routing_experiment_promotion_verdict.v1",
                "verdict": "promote",
                "promotion_ready": True,
                "reason_codes": ["promotion-thresholds-met"],
                "evidence_kind": "shadow_pass_through",
                "promotion_scope": "stage_local_canary_from_shadow",
                "shadow_only": True,
                "canary_evidence": False,
                "thresholds": {
                    "min_samples": 20,
                    "min_compared_coverage": 0.8,
                    "min_similarity_pass_rate": 0.9,
                    "max_shadow_error_rate": 0.05,
                    "max_primary_error_rate": 0.05,
                    "freshness_max_age_hours": 168,
                },
                "coverage": {"samples": 24, "compared_samples": 24, "compared_coverage": 1.0},
                "budget": {"daily_budget_usd": 10.0, "today_shadow_spend_usd": 0.1, "daily_budget_exhausted": False},
            },
            "promotion_verdict": "promote",
            "promotion_reason_codes": ["promotion-thresholds-met"],
        }
        held = {
            **promoted,
            "category": "chat",
            "workflow_phase": "summary",
            "samples": 4,
            "compared_samples": 4,
            "promotion": {
                **promoted["promotion"],
                "verdict": "hold",
                "promotion_ready": False,
                "reason_codes": ["stale-evidence"],
            },
            "promotion_verdict": "hold",
            "promotion_reason_codes": ["stale-evidence"],
        }
        candidates = [promoted, held]
        if extra_candidate is not None:
            candidates.append(extra_candidate)
        return {
            "schema": "agentflow.routing_experiment_report.v1",
            "generated_at": "2026-06-10T22:30:00+00:00",
            "policy": {
                "policy_source": "local-default",
                "min_text_chars": 0,
                "max_text_chars": 8000,
                "daily_budget_usd": 10.0,
            },
            "candidates": candidates,
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_responses_included_by_default": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "file_paths_included": False,
            },
        }

    def test_routing_promotion_draft_stage_cli_stages_only_promoted_shadow_candidate(self):
        with TemporaryDirectory() as tmp:
            active_path = Path(tmp) / "routing_rules.yaml"
            active_text = "rules: []\nphase_canary:\n  enabled: false\n"
            active_path.write_text(active_text, encoding="utf-8")
            report_path = Path(tmp) / "routing_report.json"
            report_path.write_text(json.dumps(self._routing_promotion_report()), encoding="utf-8")
            workspace = Path(tmp) / "drafts"
            stdout = io.StringIO()

            from agentflow_proxy import router
            from agentflow_proxy.policy_files import policy_file_snapshot, utc_now

            with (
                patch.object(router, "ROUTING_RULES_PATH", str(active_path)),
                patch.object(router, "ROUTING_RULES_SOURCE", "local-manual"),
                patch.object(router, "ROUTING_RULES_LOADED_AT", utc_now()),
                patch.object(router, "ROUTING_RULES_LOADED_FILE", policy_file_snapshot(active_path)),
            ):
                code = cli.routing_promotion_draft_stage_cli(
                    [
                        str(report_path),
                        "--workspace",
                        str(workspace),
                        "--draft-id",
                        "shadow-routing-promotion",
                        "--initial-canary-fraction",
                        "0.12",
                        "--holdout-fraction",
                        "0.08",
                    ],
                    stdout=stdout,
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["summary"]["candidate_count"], 2)
            self.assertEqual(payload["summary"]["staged_count"], 1)
            self.assertEqual(payload["summary"]["omitted_count"], 1)
            self.assertEqual(payload["omitted"][0]["reason"], "not-promoted")
            self.assertFalse(payload["wrote_active_policy_files"])
            self.assertFalse(payload["provider_calls_made"])
            self.assertEqual(active_path.read_text(encoding="utf-8"), active_text)

            staged = payload["staged_drafts"][0]
            self.assertEqual(staged["section"], "routing")
            self.assertEqual(staged["target_local_policy"], "phase_canary")
            self.assertEqual(staged["canary_fraction"], 0.12)
            self.assertEqual(staged["holdout_fraction"], 0.08)
            bundle = json.loads(Path(staged["bundle_path"]).read_text(encoding="utf-8"))
            canary = bundle["policies"]["routing"]["phase_canary"]
            self.assertTrue(canary["enabled"])
            self.assertEqual(canary["target_model"], "claude-haiku-4-5-20251001")
            self.assertEqual(canary["provider"], "anthropic")
            self.assertEqual(canary["source_surface"], "anthropic_messages")
            self.assertTrue(canary["stream"])
            self.assertEqual(canary["requested_model"], "claude-sonnet-4-6")
            self.assertEqual(canary["routed_model"], "claude-haiku-4-5-20251001")
            self.assertEqual(canary["eligible_categories"], ["tool-result"])
            self.assertEqual(canary["eligible_workflow_phases"], ["tool-execution"])
            self.assertEqual(canary["min_text_chars"], 0)
            self.assertEqual(canary["max_text_chars"], 8000)
            self.assertTrue(canary["safety_gates"]["block_thinking_history"])
            self.assertTrue(canary["safety_gates"]["strip_model_incompatible_params"])
            self.assertTrue(canary["safety_gates"]["fallback_to_requested_on_rate_limit"])
            self.assertEqual(canary["promotion"]["evidence_summary"]["samples"], 24)
            self.assertTrue(canary["promotion"]["evidence_summary"]["stream"])
            self.assertEqual(canary["promotion"]["rollback_metadata"]["rollback_action_type"], "disable_canary")
            rendered = json.dumps(canary["promotion"], sort_keys=True)
            self.assertIn("tool-result", rendered)
            self.assertNotIn('"category": "chat"', rendered)
            for secret in ("raw prompt secret", "req_123", "session-abc", "/tmp/private.py", "cache-key-secret"):
                self.assertNotIn(secret, rendered)

    def test_routing_promotion_draft_stage_cli_rejects_raw_and_stale_candidates(self):
        raw_report = self._routing_promotion_report()
        raw_report["candidates"][0]["raw_prompt"] = "raw prompt secret"
        raw_report["candidates"][0]["evidence"] = {
            "messages": [{"content": "raw nested message secret"}],
            "provider_body": {"prompt": "raw nested provider body secret"},
            "tool_payload": {"arguments": "raw nested tool payload secret"},
            "file_path": "/tmp/private-shadow-routing.py",
            "cache_key": "cache-key-secret",
            "request_id": "req_123",
            "session_id": "session-abc",
            "tenant_id": "tenant-secret",
            "account_id": "account-secret",
            "authorization": "Bearer auth-secret",
            "api_key": "sk-shadow-secret",
        }
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.routing_promotion_draft_stage_cli(
                ["-", "--workspace", str(Path(tmp) / "drafts")],
                stdin=io.StringIO(json.dumps(raw_report)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "raw_payload_rejected")
            self.assertFalse((Path(tmp) / "drafts").exists())
            self.assertNotIn("raw prompt secret", stdout.getvalue())
            self.assertNotIn("raw nested message secret", stdout.getvalue())
            self.assertNotIn("raw nested provider body secret", stdout.getvalue())
            self.assertNotIn("raw nested tool payload secret", stdout.getvalue())
            self.assertNotIn("/tmp/private-shadow-routing.py", stdout.getvalue())
            self.assertNotIn("cache-key-secret", stdout.getvalue())
            self.assertNotIn("req_123", stdout.getvalue())
            self.assertNotIn("session-abc", stdout.getvalue())
            self.assertNotIn("tenant-secret", stdout.getvalue())
            self.assertNotIn("account-secret", stdout.getvalue())
            self.assertNotIn("auth-secret", stdout.getvalue())
            self.assertNotIn("sk-shadow-secret", stdout.getvalue())

        stale_report = self._routing_promotion_report()
        stale_report["candidates"][0]["last_sample_age_hours"] = 999
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.routing_promotion_draft_stage_cli(
                ["-", "--workspace", str(Path(tmp) / "drafts")],
                stdin=io.StringIO(json.dumps(stale_report)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["summary"]["staged_count"], 0)
            self.assertEqual(payload["omitted"][0]["reason"], "stale-evidence")
            self.assertFalse((Path(tmp) / "drafts").exists())

    def test_routing_promotion_draft_stage_cli_fails_closed_for_malformed_shadow_promotion_inputs(self):
        server_content_report = self._routing_promotion_report()
        server_content_report["candidates"][0]["replacement_prompt"] = "raw replacement prompt must not leak"
        server_content_report["candidates"][0]["provider_body_rewrite"] = {"enabled": True}
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.routing_promotion_draft_stage_cli(
                ["-", "--workspace", str(Path(tmp) / "drafts")],
                stdin=io.StringIO(json.dumps(server_content_report)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["summary"]["staged_count"], 0)
            self.assertEqual(payload["omitted"][0]["reason"], "requires-server-content-processing")
            self.assertFalse(payload["provider_calls_made"])
            self.assertFalse(payload["managed_server_calls_made"])
            self.assertFalse((Path(tmp) / "drafts").exists())
            self.assertNotIn("raw replacement prompt", stdout.getvalue())

        malformed_report = self._routing_promotion_report(extra_candidate="not-a-candidate")
        malformed_report["candidates"][0]["workflow_phase"] = ""
        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.routing_promotion_draft_stage_cli(
                ["-", "--workspace", str(Path(tmp) / "drafts")],
                stdin=io.StringIO(json.dumps(malformed_report)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["summary"]["staged_count"], 0)
            reasons = {item["reason"] for item in payload["omitted"]}
            self.assertIn("missing-evidence", reasons)
            self.assertIn("invalid-candidate", reasons)
            self.assertFalse(payload["provider_calls_made"])
            self.assertFalse(payload["managed_server_calls_made"])
            self.assertFalse((Path(tmp) / "drafts").exists())

    def test_policy_draft_validate_cli_combines_validation_dry_run_and_section_impacts(self):
        from agentflow_proxy.store import Store, stable_json

        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"] = [{
            "conditions": {"model_pattern": "sonnet", "category": "chat", "has_tools": False},
            "action": {"route_to": "haiku", "reason": "workbench test chat downgrade"},
        }]
        proposed["policies"]["crunch"]["threshold_chars"] = 12000
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91
        proposed["policies"]["codex_app"] = {
            **proposed["policies"]["codex_app"],
            "enabled": True,
            "policy_source": "local-manual",
            "review_only": False,
            "rules": [
                {
                    "id": "local-codex-summary",
                    "conditions": {
                        "app_family": "codex",
                        "granularity": "agent_turn",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                        "input_size_bucket": "small",
                        "cache_eligible": False,
                        "has_action_like_params": False,
                        "supported_action_family": "routing",
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "reason": "local workbench dry-run projection",
                    },
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            config_dir = Path(tmp) / "config"
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="workbench-routing-match",
                    created_at="2026-06-08T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1000,
                    input_tokens_est=1000,
                    output_tokens_est=100,
                    actual_input_tokens=1000,
                    actual_output_tokens=100,
                    cost_est_usd=0.0045,
                    routing_json=stable_json({"category": "chat", "text_chars": 4000, "has_tools": False}),
                    cache_json=stable_json({"status": "miss"}),
                    retry_count=0,
                    provider="anthropic",
                )
                store.log_codex_app_event(
                    id="workbench-codex-event",
                    created_at="2026-06-08T10:01:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    request_id="raw-request-id-must-not-leak",
                    thread_id="raw-thread-id-must-not-leak",
                    session_id="raw-session-id-must-not-leak",
                    message_chars=900,
                    params_chars=100,
                    input_items=1,
                    input_text_chars=700,
                    result_chars=140,
                    latency_ms=100,
                    routing_json=stable_json({"requested_model": "gpt-5.4", "workflow_phase": "summary"}),
                    crunch_json=stable_json({}),
                    cache_json=stable_json({"status": "skipped", "eligible": False, "replayability_level": "turn-metadata-only"}),
                    event_window_json=stable_json({
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                        "input_text_chars": 700,
                        "result_chars": 140,
                    }),
                    metadata_json=stable_json({"kind": "turn_window"}),
                )
            finally:
                store.conn.close()

            stage_stdout = io.StringIO()
            stage_code = cli.policy_draft_stage_cli(
                ["--draft-id", "workbench-all", "--workspace", str(workspace), "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stage_stdout,
            )
            self.assertEqual(stage_code, 0)

            validate_stdout = io.StringIO()
            validate_code = cli.policy_draft_validate_cli(
                [
                    "workbench-all",
                    "--workspace",
                    str(workspace),
                    "--config-dir",
                    str(config_dir),
                    "--db",
                    db_path,
                ],
                stdout=validate_stdout,
            )

        self.assertEqual(validate_code, 0)
        payload = json.loads(validate_stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_draft_validate.v1")
        self.assertEqual(payload["status"], "pass")
        self.assertTrue(payload["can_apply"])
        self.assertFalse(payload["apply_blocked"])
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        self.assertFalse(payload["privacy"]["managed_server_calls_made"])
        sections = {section["section"]: section for section in payload["sections"]}
        for section in ("routing", "crunch", "cache", "codex_app"):
            self.assertTrue(sections[section]["changed"])
            self.assertEqual(sections[section]["verdict"], "pass")
            self.assertTrue(sections[section]["reload_required_after_apply"])
        self.assertGreaterEqual(sections["routing"]["projected_impact"]["projected_applied_count"], 1)
        self.assertGreaterEqual(sections["codex_app"]["projected_impact"]["projected_applied_count"], 2)
        self.assertIn("codex_app_dry_run", payload)
        rendered = validate_stdout.getvalue()
        self.assertNotIn("raw-request-id-must-not-leak", rendered)
        self.assertNotIn("raw-session-id-must-not-leak", rendered)

    def test_policy_draft_validate_cli_blocks_risky_draft_before_apply(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "drafts"
            stage_code = cli.policy_draft_stage_cli(
                ["--draft-id", "risky-cache", "--workspace", str(workspace), "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=io.StringIO(),
            )
            self.assertEqual(stage_code, 0)

            validate_stdout = io.StringIO()
            validate_code = cli.policy_draft_validate_cli(
                [
                    str(workspace / "risky-cache"),
                    "--workspace",
                    str(workspace),
                    "--config-dir",
                    str(Path(tmp) / "config"),
                    "--db",
                    str(Path(tmp) / "missing.sqlite3"),
                ],
                stdout=validate_stdout,
            )

        self.assertEqual(validate_code, 1)
        payload = json.loads(validate_stdout.getvalue())
        self.assertEqual(payload["status"], "warn")
        self.assertFalse(payload["can_apply"])
        self.assertTrue(payload["apply_blocked"])
        self.assertIn("tool-call-cache-enabled", payload["apply_prerequisites"]["blocker_reason_codes"])
        cache_section = {section["section"]: section for section in payload["sections"]}["cache"]
        self.assertEqual(cache_section["verdict"], "warn")
        self.assertIn("tool-call-cache-enabled", cache_section["blocker_reason_codes"])

    def test_policy_apply_cli_dry_run_shows_old_context_summary_canary_yaml_diff(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            crunch_path = Path(tmp) / "crunch_rules.yaml"
            crunch_path.write_text("enabled: true\nthreshold_chars: 24000\n", encoding="utf-8")
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "--dry-run", "--pretty", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["applied_sections"], ["crunch"])
            self.assertEqual(payload["old_context_summarization"]["status"], "dry-run")
            self.assertEqual(payload["old_context_summarization"]["selected_candidate_id"], "candidate-old-context-summary")
            self.assertEqual(payload["old_context_summarization"]["canary_fraction"], 0.25)
            self.assertEqual(crunch_path.read_text(encoding="utf-8"), "enabled: true\nthreshold_chars: 24000\n")
            self.assertEqual(len(payload["files"]), 1)
            self.assertEqual(payload["files"][0]["section"], "crunch")
            diff = payload["files"][0]["diff"]
            self.assertIn("+  candidate_id: candidate-old-context-summary", diff)
            self.assertIn("+    fraction: 0.25", diff)
            self.assertIn("+  policy_source: managed-recommended", diff)
            self.assertNotIn("recommendation:", diff)
            self.assertNotIn("raw-secret-old-context", stdout.getvalue())

    def test_policy_apply_cli_writes_managed_old_context_summary_canary_and_rollback_finds_backup(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            crunch_path = Path(tmp) / "crunch_rules.yaml"
            previous = "enabled: true\nthreshold_chars: 24000\n"
            crunch_path.write_text(previous, encoding="utf-8")
            apply_stdout = io.StringIO()
            apply_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=apply_stdout,
            )
            apply_payload = json.loads(apply_stdout.getvalue())

            self.assertEqual(apply_code, 0)
            self.assertEqual(apply_payload["old_context_summarization"]["status"], "applied")
            applied = yaml.safe_load(crunch_path.read_text(encoding="utf-8"))
            rule = applied["old_context_summarization"]
            self.assertTrue(rule["enabled"])
            self.assertEqual(rule["candidate_id"], "candidate-old-context-summary")
            self.assertEqual(rule["policy_source"], "managed-recommended")
            self.assertEqual(rule["model"], "claude-haiku-4-5-20251001")
            self.assertEqual(rule["source_model"], "claude-sonnet-4-6")
            self.assertEqual(rule["min_summarized_chars"], 10)
            self.assertEqual(rule["max_turns"], 3)
            self.assertTrue(rule["canary"]["enabled"])
            self.assertEqual(rule["canary"]["fraction"], 0.25)
            self.assertEqual(rule["canary"]["salt"], "summary-canary-test")
            self.assertEqual(rule["canary"]["widening_threshold"], 0.8)
            self.assertEqual(rule["canary"]["rollback_threshold"], 0.2)
            self.assertTrue(rule["safety_stop"]["enabled"])
            self.assertEqual(rule["safety_stop"]["max_error_rate"], 0.05)
            self.assertEqual(rule["safety_stop"]["max_summary_failure_rate"], 0.02)
            backups = list(Path(tmp).glob("crunch_rules.yaml.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), previous)

            rollback_stdout = io.StringIO()
            rollback_code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "crunch", "--dry-run"],
                stdout=rollback_stdout,
            )
            rollback_payload = json.loads(rollback_stdout.getvalue())

            self.assertEqual(rollback_code, 0)
            self.assertEqual(rollback_payload["old_context_summarization"]["status"], "rollback-dry-run")
            self.assertEqual(rollback_payload["restored_sections"], ["crunch"])
            self.assertEqual(rollback_payload["files"][0]["restored_from"], str(backups[0]))
            self.assertNotEqual(crunch_path.read_text(encoding="utf-8"), previous)

    def _managed_codex_apply_bundle(self):
        bundle = self._managed_policy_bundle()
        rule = bundle["policies"]["codex_app"]["rules"][0]
        rule["conditions"] = {
            "app_family": "codex",
            "workflow_phase": "summary",
            "granularity": "agent_turn",
            "model_field_state": "present",
            "input_size_bucket": "small",
            "cache_eligible": True,
            "replayability_level": "local-exact-response",
            "has_action_like_params": False,
            "stale_risk": False,
            "supported_action_family": "routing",
        }
        rule["action"] = {
            "summary_model_hint": {
                "enabled": True,
                "recommended_model": "gpt-5-mini",
                "workflow_phase": "summary",
            },
            "exact_cache": {
                "enabled": True,
                "profile": "exact",
                "replayability_level": "agent_turn",
            },
            "cache_eligible": True,
            "crunch_profile": "codex-repeated-scaffolding",
            "cache_eligibility_reason": "managed exact-cache canary evidence passed",
            "canary": {
                "enabled": True,
                "fraction": 1.0,
                "holdout_fraction": 0.0,
                "salt": "managed-codex-app-test",
                "unit": "source_hash",
            },
            "safety_stop": {
                "max_error_rate": 0.05,
                "max_retry_rate": 0.05,
                "rollback_on_quality_regression": True,
            },
            "reason": "managed Codex summary rule",
        }
        rule["managed_recommendation"]["candidate_family"] = "codex-agent-turn-policy"
        rule["managed_recommendation"]["source_surface"] = "codex_turn"
        rule["managed_recommendation"]["codex_policy_sections"] = ["summary_model_hint", "exact_cache"]
        bundle["recommendation"]["candidate_ids"] = ["candidate-codex-summary"]
        bundle["recommendation"]["candidate_count"] = 1
        return bundle

    def test_policy_apply_cli_writes_reviewed_managed_codex_app_rules_and_rollback_finds_backup(self):
        proposed = self._managed_codex_apply_bundle()

        with TemporaryDirectory() as tmp:
            codex_path = Path(tmp) / "codex_app_rules.yaml"
            previous = "enabled: true\nrules: []\n"
            codex_path.write_text(previous, encoding="utf-8")

            dry_stdout = io.StringIO()
            dry_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "codex_app", "--dry-run", "--pretty", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=dry_stdout,
            )
            dry_payload = json.loads(dry_stdout.getvalue())

            self.assertEqual(dry_code, 0)
            self.assertEqual(dry_payload["applied_sections"], ["codex_app"])
            self.assertEqual(dry_payload["codex_app"]["status"], "dry-run")
            self.assertEqual(dry_payload["codex_app"]["selected_candidate_ids"], ["candidate-codex-summary"])
            self.assertEqual(codex_path.read_text(encoding="utf-8"), previous)
            diff = dry_payload["files"][0]["diff"]
            self.assertIn("+  candidate_id: candidate-codex-summary", diff)
            self.assertIn("+  policy_source: managed-recommended", diff)
            self.assertIn("+    model_hint: gpt-5-mini", diff)

            apply_stdout = io.StringIO()
            apply_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "codex_app", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=apply_stdout,
            )
            apply_payload = json.loads(apply_stdout.getvalue())

            self.assertEqual(apply_code, 0)
            self.assertEqual(apply_payload["codex_app"]["status"], "applied")
            applied = yaml.safe_load(codex_path.read_text(encoding="utf-8"))
            self.assertTrue(applied["enabled"])
            self.assertEqual(applied["policy_source"], "managed-recommended")
            self.assertEqual(len(applied["rules"]), 1)
            rule = applied["rules"][0]
            self.assertEqual(rule["candidate_id"], "candidate-codex-summary")
            self.assertEqual(rule["policy_source"], "managed-recommended")
            self.assertEqual(rule["action"]["model_hint"], "gpt-5-mini")
            self.assertTrue(rule["action"]["cache_eligible"])
            self.assertEqual(rule["action"]["crunch_profile"], "codex-repeated-scaffolding")
            self.assertEqual(rule["canary"]["salt"], "managed-codex-app-test")
            self.assertTrue(rule["safety_stop"]["rollback_on_quality_regression"])
            backups = list(Path(tmp).glob("codex_app_rules.yaml.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), previous)

            rollback_stdout = io.StringIO()
            rollback_code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "codex_app", "--dry-run"],
                stdout=rollback_stdout,
            )
            rollback_payload = json.loads(rollback_stdout.getvalue())

            self.assertEqual(rollback_code, 0)
            self.assertEqual(rollback_payload["codex_app"]["status"], "rollback-dry-run")
            self.assertEqual(rollback_payload["restored_sections"], ["codex_app"])
            self.assertEqual(rollback_payload["files"][0]["restored_from"], str(backups[0]))

    def test_policy_apply_cli_rejects_unsafe_managed_codex_app_bundles(self):
        cases = []
        unsupported_action = self._managed_codex_apply_bundle()
        unsupported_action["policies"]["codex_app"]["rules"][0]["action"]["provider_body_rewrite"] = True
        cases.append((unsupported_action, "$.policies.codex_app.rules[0].action.provider_body_rewrite"))

        raw_payload = self._managed_codex_apply_bundle()
        raw_payload["policies"]["codex_app"]["rules"][0]["managed_recommendation"]["raw_request"] = {
            "prompt": "raw managed Codex prompt must not be accepted"
        }
        cases.append((raw_payload, "$.policies.codex_app.rules[0].managed_recommendation.raw_request"))

        managed_enforced = self._managed_codex_apply_bundle()
        managed_enforced["policies"]["codex_app"]["rules"][0]["managed_recommendation"]["policy_source"] = "managed-enforced"
        cases.append((managed_enforced, "$.policies.codex_app.rules[0].managed_recommendation.policy_source"))

        unsupported_family = self._managed_codex_apply_bundle()
        unsupported_family["policies"]["codex_app"]["rules"][0]["local_action"] = "provider_body_rewrite"
        cases.append((unsupported_family, "$.policies.codex_app.rules[0]"))

        for bundle, expected_path in cases:
            with TemporaryDirectory() as tmp:
                stdout = io.StringIO()
                code = cli.policy_apply_cli(
                    ["--config-dir", tmp, "--section", "codex_app", "-"],
                    stdin=io.StringIO(json.dumps(bundle)),
                    stdout=stdout,
                )
                payload = json.loads(stdout.getvalue())
                self.assertEqual(code, 1)
                self.assertEqual(payload["error"]["type"], "validation_failed")
                self.assertIn(expected_path, {error["path"] for error in payload["validation"]["errors"]})
                self.assertFalse((Path(tmp) / "codex_app_rules.yaml").exists())
                self.assertNotIn("raw managed Codex prompt", stdout.getvalue())

    def test_policy_apply_cli_rejects_unsigned_managed_codex_bundle_when_signature_required(self):
        proposed = self._managed_codex_apply_bundle()

        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "codex-secret"}, clear=False):
                stdout = io.StringIO()
                code = cli.policy_apply_cli(
                    ["--config-dir", tmp, "--section", "codex_app", "-"],
                    stdin=io.StringIO(json.dumps(proposed)),
                    stdout=stdout,
                )
                payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 1)
        self.assertEqual(payload["error"]["type"], "validation_failed")
        self.assertIn("$.provenance", {error["path"] for error in payload["validation"]["errors"]})

    def test_policy_apply_cli_normalizes_managed_enforced_old_context_summary_to_recommended_canary(self):
        proposed = self._managed_policy_bundle()
        proposed["policies"]["crunch"]["policy_source"] = "managed-recommended"
        proposed["policies"]["crunch"]["old_context_summarization"] = {
            "enabled": True,
            "rule_id": "managed-enforced-summary-rule",
            "candidate_id": "candidate-old-context-enforced",
            "policy_source": "managed-enforced",
            "model": "claude-haiku-4-5-20251001",
            "min_request_chars": 100,
            "min_summarized_chars": 50,
            "canary": {
                "enabled": True,
                "fraction": 0.1,
                "salt": "direct-canary",
                "unit": "source_hash",
            },
            "safety_stop": {
                "enabled": True,
                "max_error_rate": 0.05,
            },
        }

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "--allow-risky", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            applied = yaml.safe_load((Path(tmp) / "crunch_rules.yaml").read_text(encoding="utf-8"))
            rule = applied["old_context_summarization"]
            self.assertEqual(rule["candidate_id"], "candidate-old-context-enforced")
            self.assertEqual(rule["policy_source"], "managed-recommended")

    def test_policy_apply_cli_records_old_context_summary_events_without_raw_payloads(self):
        proposed = self._managed_old_context_summary_bundle()

        with TemporaryDirectory() as tmp:
            (Path(tmp) / "crunch_rules.yaml").write_text("enabled: true\n", encoding="utf-8")
            cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=io.StringIO(),
            )
            cli.policy_rollback_cli(["--config-dir", tmp, "--section", "crunch"], stdout=io.StringIO())

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        event_text = json.dumps(events)
        self.assertEqual(events[0]["action"], "rollback")
        self.assertEqual(events[0]["details"]["old_context_summarization"]["status"], "rolled-back")
        self.assertEqual(events[1]["action"], "apply")
        self.assertEqual(events[1]["details"]["old_context_summarization"]["status"], "applied")
        self.assertEqual(events[1]["details"]["old_context_summarization"]["selected_candidate_id"], "candidate-old-context-summary")
        self.assertNotIn("raw-secret-old-context", event_text)
        for forbidden in ("prompt", "summary_text", "messages", "transcript", "provider_body", "cache_key"):
            self.assertNotIn(forbidden, event_text)

    def test_policy_apply_cli_rejected_old_context_summary_event_omits_raw_payloads(self):
        proposed = self._managed_old_context_summary_bundle(raw_like=True)

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "crunch", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["old_context_summarization"]["status"], "rejected")

        from agentflow_proxy.policy_events import recent_policy_events

        event_text = json.dumps(recent_policy_events(limit=5)["events"])
        self.assertIn("candidate-old-context-summary", event_text)
        self.assertNotIn("raw-secret-old-context", event_text)

    def test_policy_apply_cli_writes_selected_section_and_creates_backup(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 0.91

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: false\n", encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["applied_sections"], ["cache"])
            self.assertEqual(payload["skipped_sections"][0]["reason"], "not-requested")
            applied = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(applied["semantic_cache"]["threshold"], 0.91)
            backups = list(Path(tmp).glob("cache_rules.yaml.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("enabled: false", backups[0].read_text(encoding="utf-8"))

            second_stdout = io.StringIO()
            second_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=second_stdout,
            )
            second_payload = json.loads(second_stdout.getvalue())
            self.assertEqual(second_code, 0)
            self.assertFalse(second_payload["files"][0]["changed"])
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_apply_cli_refuses_risky_bundle_unless_allowed(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "risky_policy")
            self.assertEqual(payload["safety_warning_count"], 1)
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

            allowed_stdout = io.StringIO()
            allowed_code = cli.policy_apply_cli(
                ["--config-dir", tmp, "--section", "cache", "--allow-risky", "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=allowed_stdout,
            )
            allowed_payload = json.loads(allowed_stdout.getvalue())
            self.assertEqual(allowed_code, 0)
            self.assertTrue(allowed_payload["ok"])
            self.assertTrue(yaml.safe_load((Path(tmp) / "cache_rules.yaml").read_text(encoding="utf-8"))["exact_cache"]["cache_tool_calls"])

    def test_policy_apply_cli_rejects_malformed_section_schema_before_writing(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["rules"][0]["conditions"]["text_chars_lt"] = "small"
        proposed["policies"]["cache"]["semantic_cache"]["threshold"] = 2

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            code = cli.policy_apply_cli(
                ["--config-dir", tmp, "-"],
                stdin=io.StringIO(json.dumps(proposed)),
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "validation_failed")
            paths = {error["path"] for error in payload["validation"]["errors"]}
            self.assertIn("$.policies.routing.rules[0].conditions.text_chars_lt", paths)
            self.assertIn("$.policies.cache.semantic_cache.threshold", paths)
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())

    def test_policy_rollback_cli_dry_run_reports_latest_backup_without_writing(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "cache", "--dry-run"],
                stdout=stdout,
            )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["restored_sections"], ["cache"])
            self.assertTrue(payload["files"][0]["changed"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_rollback_cli_apply_id_dry_run_reports_exact_backup(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            apply_id = "20260610T220000Z-cache-apply"
            backup = Path(tmp) / f"cache_rules.yaml.bak-{apply_id}"
            backup.write_text("exact_cache:\n  enabled: false\n", encoding="utf-8")
            event_log = Path(tmp) / "policy_events.jsonl"
            event_log.write_text(
                json.dumps({
                    "schema": "agentflow.policy_event.v1",
                    "id": "event-apply",
                    "created_at": "2026-06-10T22:00:00+00:00",
                    "action": "draft-apply",
                    "ok": True,
                    "details": {
                        "apply_id": apply_id,
                        "backup_id": apply_id,
                        "status": "applied",
                        "changed_sections": ["cache"],
                        "changed_files": [str(cache_path)],
                        "backup_paths": [str(backup)],
                    },
                })
                + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with patch.dict(os.environ, {"AGENTFLOW_POLICY_EVENTS_LOG": str(event_log)}, clear=False):
                code = cli.policy_rollback_cli(
                    ["--config-dir", tmp, "--apply-id", apply_id, "--section", "cache", "--dry-run"],
                    stdout=stdout,
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "agentflow.policy_draft_rollback.v1")
            self.assertEqual(payload["status"], "dry-run")
            self.assertEqual(payload["apply_id"], apply_id)
            self.assertEqual(payload["restored_sections"], ["cache"])
            self.assertEqual(payload["files"][0]["restored_from"], str(backup))
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")

    def test_policy_rollback_cli_restores_selected_section_and_backs_up_current_file(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            latest = Path(tmp) / "cache_rules.yaml.bak-20260102T000000000000Z"
            latest.write_text("exact_cache:\n  enabled: newest\n", encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(["--config-dir", tmp, "--section", "cache"], stdout=stdout)

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "agentflow.policy_bundle_rollback.v1")
            self.assertEqual(payload["restored_sections"], ["cache"])
            self.assertEqual(payload["files"][0]["restored_from"], str(latest))
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: newest\n")
            current_backups = [
                path
                for path in Path(tmp).glob("cache_rules.yaml.bak-*")
                if path.name != "cache_rules.yaml.bak-20260101T000000000000Z"
                and path.name != "cache_rules.yaml.bak-20260102T000000000000Z"
            ]
            self.assertEqual(len(current_backups), 1)
            self.assertEqual(current_backups[0].read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")

    def test_policy_rollback_cli_missing_backup_fails_without_partial_writes(self):
        with TemporaryDirectory() as tmp:
            routing_path = Path(tmp) / "routing_rules.yaml"
            cache_path = Path(tmp) / "cache_rules.yaml"
            routing_path.write_text("rules: []\n", encoding="utf-8")
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            code = cli.policy_rollback_cli(
                ["--config-dir", tmp, "--section", "routing", "--section", "cache"],
                stdout=stdout,
            )

            self.assertEqual(code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"]["type"], "missing_backups")
            self.assertEqual(payload["error"]["sections"], ["routing"])
            self.assertEqual(cache_path.read_text(encoding="utf-8"), "exact_cache:\n  enabled: true\n")
            self.assertEqual(len(list(Path(tmp).glob("cache_rules.yaml.bak-*"))), 1)

    def test_policy_rollback_cli_records_compact_event(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache_rules.yaml"
            cache_path.write_text("exact_cache:\n  enabled: true\n", encoding="utf-8")
            (Path(tmp) / "cache_rules.yaml.bak-20260101T000000000000Z").write_text(
                "exact_cache:\n  enabled: false\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            cli.policy_rollback_cli(["--config-dir", tmp, "--section", "cache"], stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "rollback")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["restored_sections"], ["cache"])
        self.assertEqual(events[0]["details"]["exit_code"], 0)

    def test_policy_cli_records_compact_local_events(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "diff")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["routing"])
        self.assertEqual(events[0]["details"]["change_count"], 1)
        self.assertEqual(events[1]["action"], "export")
        self.assertIn("routing", events[1]["details"]["policies"])


class ManagedFeedbackCliTests(unittest.TestCase):
    def setUp(self):
        ManagedFeedbackFlushClient.calls = []
        ManagedFeedbackFlushClient.status_code = 200
        ManagedFeedbackFlushClient.text = '{"ok":true}'

    def _enqueue_feedback(self, store, *, status="queued", attempts=0):
        from agentflow_proxy.store import stable_json

        store.enqueue_managed_outcome_feedback(
            id=f"queue-{status}-{attempts}",
            created_at="2026-06-08T10:00:00+00:00",
            updated_at="2026-06-08T10:00:00+00:00",
            source_surface="codex_turn",
            endpoint="/v1/optimization-units/77/outcome",
            optimization_unit_id=77,
            payload_json=stable_json({
                "status": "success",
                "quality_signals": {"status": "success"},
                "pattern_decisions": [
                    {
                        "schema": "agentflow.pattern_decision_summary.v1",
                        "decision_type": "routing",
                        "status": "applied",
                        "policy_source": "managed-recommended",
                    }
                ],
                "pattern_policy_evidence": [
                    {
                        "schema": "agentflow.managed_pattern_policy_evidence.v1",
                        "source_surface": "codex_turn",
                        "app_family": "codex",
                        "action_family": "routing",
                        "pattern_family": "routing",
                        "pattern_hash": "sha256:" + "4" * 64,
                        "pattern_hashes": ["sha256:" + "4" * 64],
                        "candidate_id": "candidate-routing",
                        "rule_id": "rule-routing",
                        "policy_source": "managed-recommended",
                        "cohort": "canary_applied",
                        "outcome": "applied",
                        "status_code_bucket": "2xx",
                        "retry_bucket": "none",
                        "latency_bucket": "lt_500ms",
                        "savings_bucket": "lt_0_001_usd",
                        "raw_pattern_strings_included": False,
                        "raw_payload_included": False,
                    }
                ],
            }),
            status=status,
            attempts=attempts,
            next_attempt_at="2026-06-08T10:00:00+00:00",
        )

    def _log_post_promotion_feedback(
        self,
        store,
        *,
        row_id,
        action_family,
        status,
        recommendation,
        applied,
        holdout,
        safety_stopped=0,
        observed=0.0,
        projected=0.0,
        feedback_extra=None,
    ):
        from agentflow_proxy.store import stable_json

        entry = {
            "schema": "agentflow.promotion_outcome_feedback_entry.v1",
            "id": row_id,
            "created_at": "2026-06-15T07:00:00+00:00",
            "policy_id": f"local-policy-{action_family}",
            "action_family": action_family,
            "policy_section": action_family,
            "rule_source": "managed-recommended",
            "source_evidence_schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "status": status,
            "recommendation": recommendation,
            "rollback_needed": recommendation == "rollback",
            "reason_codes": ["post-promotion-fixture"],
            "observed_savings_usd": observed,
            "projected_savings_usd": projected,
            "applied_count": applied,
            "holdout_count": holdout,
            "skipped_count": 0,
            "bypassed_count": 0,
            "safety_stop_count": safety_stopped,
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
            },
        }
        if feedback_extra:
            entry.update(feedback_extra)
        store.log_promotion_outcome_feedback(
            id=row_id,
            created_at="2026-06-15T07:00:00+00:00",
            impact_generated_at="2026-06-15T06:55:00+00:00",
            policy_id=f"local-policy-{action_family}",
            action_family=action_family,
            policy_section=action_family,
            rule_source="managed-recommended",
            rule_id=f"rule-{action_family}",
            candidate_id=f"candidate-{action_family}",
            action_id=f"action-{action_family}",
            source_evidence_schema="agentflow.optimization_promotion_rollout_actions.v1",
            status=status,
            recommendation=recommendation,
            rollback_needed=1 if recommendation == "rollback" else 0,
            observed_savings_usd=observed,
            projected_savings_usd=projected,
            projection_realization_ratio=None,
            applied_count=applied,
            holdout_count=holdout,
            skipped_count=0,
            bypassed_count=0,
            safety_stop_count=safety_stopped,
            error_rate_delta=0.0,
            retry_rate_delta=0.0,
            latency_delta_ms=0.0,
            feedback_json=stable_json(entry),
        )

    def test_managed_feedback_status_cli_reports_metadata_only_queue_counts(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.managed_feedback_status_cli(
                ["--db", db_path, "--source-surface", "codex_turn"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.managed_feedback_status.v1")
        self.assertEqual(payload["summary"]["queued"], 1)
        self.assertEqual(payload["summary"]["due"], 1)
        self.assertEqual(payload["pattern_evidence"]["queue_rows"], 1)
        self.assertEqual(payload["pattern_evidence"]["evidence_items"], 1)
        self.assertEqual(
            payload["pattern_evidence"]["endpoint_status_breakdown"][0],
            {
                "endpoint": "/v1/optimization-units/77/outcome",
                "status": "queued",
                "queue_rows": 1,
                "evidence_items": 1,
            },
        )
        self.assertFalse(payload["pattern_evidence"]["payload_json_included"])
        self.assertFalse(payload["due_samples"][0]["payload_included"])
        rendered = stdout.getvalue()
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw codex response secret", rendered)

    def test_managed_feedback_flush_dry_run_does_not_claim_or_send(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                code = cli.managed_feedback_flush_cli(
                    ["--db", db_path, "--source-surface", "codex_turn", "--dry-run"],
                    stdout=stdout,
                )

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls, [])
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["flush"]["would_attempt"], 1)
        self.assertEqual(payload["results"][0]["status"], "would-send")
        self.assertFalse(payload["results"][0]["payload_included"])

    def test_managed_feedback_flush_sends_sanitized_payload_and_updates_queue(self):
        from agentflow_proxy import stats as stats_views
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(
                        ["--db", db_path, "--source-surface", "codex_turn", "--limit", "1"],
                        stdout=stdout,
                    )

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
                codex_stats = asyncio.run(stats_views.stats_codex_effectiveness(store, limit=10))
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(row["status"], "sent")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(ManagedFeedbackFlushClient.calls[0]["url"], "http://managed.test/v1/optimization-units/77/outcome")
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["status"], "success")
        self.assertNotIn("raw_request", sent_payload)
        self.assertNotIn("raw_response", sent_payload)
        rendered = stdout.getvalue()
        self.assertNotIn("must stay local", rendered)
        self.assertNotIn("raw codex response secret", rendered)
        payload = json.loads(rendered)
        self.assertEqual(payload["flush"]["sent"], 1)
        self.assertEqual(payload["after"]["sent"], 1)
        self.assertEqual(codex_stats["summary"]["managed_feedback_queue_sent"], payload["after"]["sent"])

    def test_managed_feedback_flush_records_retryable_error(self):
        from agentflow_proxy.store import Store

        ManagedFeedbackFlushClient.status_code = 503
        ManagedFeedbackFlushClient.text = "managed unavailable"

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._enqueue_feedback(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_OUTCOME_FEEDBACK_QUEUE_MAX_ATTEMPTS": "3",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(["--db", db_path], stdout=stdout)

            store = Store(db_path)
            try:
                row = store.get_managed_outcome_feedback("queue-queued-0")
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(row["status"], "retryable-error")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["last_status_code"], 503)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["flush"]["retryable_error"], 1)

    def test_managed_feedback_flush_posts_post_promotion_action_outcome_rollups(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_post_promotion_feedback(
                    store,
                    row_id="post-promotion-routing",
                    action_family="routing",
                    status="positive",
                    recommendation="widen",
                    applied=3,
                    holdout=2,
                    observed=0.009,
                    projected=0.006,
                )
                self._log_post_promotion_feedback(
                    store,
                    row_id="post-promotion-cache",
                    action_family="cache",
                    status="rollback-needed",
                    recommendation="rollback",
                    applied=1,
                    holdout=1,
                    safety_stopped=1,
                    observed=-0.001,
                    projected=0.004,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(
                        ["--db", db_path, "--post-promotion-action-outcomes", "--limit", "5"],
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(len(ManagedFeedbackFlushClient.calls), 1)
        self.assertEqual(
            ManagedFeedbackFlushClient.calls[0]["url"],
            "http://managed.test/v1/promotion-blocker-action-outcome-rollups",
        )
        sent_payload = ManagedFeedbackFlushClient.calls[0]["json"]
        self.assertEqual(sent_payload["schema"], "agentflow.promotion_blocker_action_outcome_rollup_ingest.v1")
        self.assertEqual(len(sent_payload["rollups"]), 2)
        rollups = {row["local_action_family"]: row for row in sent_payload["rollups"]}
        self.assertEqual(rollups["routing"]["applied_count"], 3)
        self.assertEqual(rollups["routing"]["metadata"]["holdout_count"], 2)
        self.assertEqual(rollups["routing"]["next_action"], "widen-local-policy")
        self.assertEqual(rollups["cache"]["safety_stopped_count"], 1)
        self.assertEqual(rollups["cache"]["next_action"], "rollback-local-policy")
        rendered = json.dumps(sent_payload, sort_keys=True)
        for forbidden in (
            "raw request secret",
            "raw response secret",
            "candidate-routing",
            "candidate-cache",
            "rule-routing",
            "rule-cache",
            "action-routing",
            "action-cache",
        ):
            self.assertNotIn(forbidden, rendered)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["flush"]["sent"], 1)
        self.assertEqual(result["post_promotion_action_outcome_rollups"]["status"], "flushed")
        self.assertEqual(result["post_promotion_action_outcome_rollups"]["rollup_count"], 2)

    def test_post_promotion_action_outcome_rollups_reject_raw_stored_feedback_before_send(self):
        from agentflow_proxy.store import Store

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_post_promotion_feedback(
                    store,
                    row_id="post-promotion-unsafe",
                    action_family="routing",
                    status="positive",
                    recommendation="widen",
                    applied=1,
                    holdout=1,
                    feedback_extra={"raw_request": "raw post promotion request must stay local"},
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                },
                clear=False,
            ):
                with patch("agentflow_proxy.recommendations.httpx.AsyncClient", ManagedFeedbackFlushClient):
                    code = cli.managed_feedback_flush_cli(
                        ["--db", db_path, "--post-promotion-action-outcomes", "--limit", "5"],
                        stdout=stdout,
                    )

        self.assertEqual(code, 0)
        self.assertEqual(ManagedFeedbackFlushClient.calls, [])
        rendered = stdout.getvalue()
        self.assertNotIn("raw post promotion request must stay local", rendered)
        result = json.loads(rendered)
        self.assertEqual(result["post_promotion_action_outcome_rollups"]["status"], "rejected")
        self.assertEqual(result["post_promotion_action_outcome_rollups"]["reason"], "privacy-blocked")

    def test_sqlite_maintenance_cli_reports_dry_run(self):
        from agentflow_proxy.store import Store, stable_json

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                store.log_call(
                    id="old-cli-call",
                    created_at="2026-06-01T00:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1,
                    input_tokens_est=1,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "test"}),
                    cache_json=stable_json({"status": "miss"}),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.sqlite_maintenance_cli(
                [
                    "--db",
                    db_path,
                    "--retention-days",
                    "7",
                    "--dry-run",
                    "--no-analyze",
                    "--no-optimize",
                ],
                stdout=stdout,
            )

            payload = json.loads(stdout.getvalue())
            store = Store(db_path)
            try:
                retained = store.conn.execute("select count(*) as count from calls").fetchone()["count"]
            finally:
                store.conn.close()

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.sqlite_maintenance_run.v1")
        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["retention_days"], 7)
        self.assertEqual(payload["deleted_rows"]["calls"], 1)
        self.assertEqual(retained, 1)


if __name__ == "__main__":
    unittest.main()
