from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class TokenClawEnvironmentTests(unittest.TestCase):
    def _run_server_import(self, env_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "AGENTFLOW_CONFIG_DIR",
            "AGENTFLOW_DB",
            "AGENTFLOW_DATABASE_URL",
            "AGENTFLOW_POLICY_CONFIG_DIR",
            "AGENTFLOW_PORT",
            "TOKENCLAW_CONFIG_DIR",
            "TOKENCLAW_DB",
            "TOKENCLAW_DATABASE_URL",
            "TOKENCLAW_POLICY_CONFIG_DIR",
            "TOKENCLAW_PORT",
        ):
            env.pop(key, None)
        env.update(env_overrides)
        return subprocess.run(
            [
                sys.executable,
                "-W",
                "default",
                "-c",
                "import tokenclaw.server as s; print(s.DEFAULT_PORT)",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_tokenclaw_port_controls_server_default_port(self):
        with TemporaryDirectory() as tmp:
            result = self._run_server_import(
                {
                    "HOME": tmp,
                    "TOKENCLAW_DB": str(Path(tmp) / "tokenclaw.sqlite3"),
                    "TOKENCLAW_PORT": "4001",
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "4001")
        self.assertNotIn("AGENTFLOW_PORT is deprecated", result.stderr)

    def test_legacy_agentflow_port_still_works_with_deprecation_warning(self):
        with TemporaryDirectory() as tmp:
            result = self._run_server_import(
                {
                    "AGENTFLOW_DB": str(Path(tmp) / "agentflow.sqlite3"),
                    "AGENTFLOW_PORT": "4001",
                    "HOME": tmp,
                }
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "4001")
        self.assertIn("AGENTFLOW_PORT is deprecated; use TOKENCLAW_PORT", result.stderr)

    def test_run_args_use_tokenclaw_port_over_activated_profile_port(self):
        from tokenclaw.activation import proxy_args_for_target

        config = {
            "targets": {
                "claude": {
                    "id": "claude",
                    "configured": True,
                    "provider": "anthropic",
                    "host": "127.0.0.1",
                    "port": 4000,
                    "local_base_url": "http://127.0.0.1:4000",
                    "health_url": "http://127.0.0.1:4000/health",
                    "upstream_base_url": "https://api.anthropic.com",
                }
            }
        }

        with patch.dict(os.environ, {"TOKENCLAW_PORT": "4001"}, clear=False):
            args = proxy_args_for_target(config, "claude")

        self.assertEqual(args[args.index("--port") + 1], "4001")

    def test_run_args_accept_legacy_agentflow_port_alias(self):
        from tokenclaw.activation import proxy_args_for_target

        config = {
            "targets": {
                "claude": {
                    "id": "claude",
                    "configured": True,
                    "provider": "anthropic",
                    "host": "127.0.0.1",
                    "port": 4000,
                    "local_base_url": "http://127.0.0.1:4000",
                    "health_url": "http://127.0.0.1:4000/health",
                    "upstream_base_url": "https://api.anthropic.com",
                }
            }
        }

        with patch.dict(os.environ, {"AGENTFLOW_PORT": "4001"}, clear=False):
            args = proxy_args_for_target(config, "claude")

        self.assertEqual(args[args.index("--port") + 1], "4001")

    def test_default_config_dir_copies_legacy_agentflow_dir_to_tokenclaw(self):
        from tokenclaw.paths import default_config_dir, default_db_path

        cleared_keys = (
            "AGENTFLOW_CONFIG_DIR",
            "AGENTFLOW_DB",
            "AGENTFLOW_POLICY_CONFIG_DIR",
            "TOKENCLAW_CONFIG_DIR",
            "TOKENCLAW_DB",
            "TOKENCLAW_POLICY_CONFIG_DIR",
        )
        with TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy_dir = home / ".agentflow"
            legacy_dir.mkdir()
            (legacy_dir / "activation.json").write_text('{"schema":"agentflow.activation_config.v1"}\n', encoding="utf-8")
            old_env = {key: os.environ.get(key) for key in (*cleared_keys, "HOME")}
            try:
                for key in cleared_keys:
                    os.environ.pop(key, None)
                os.environ["HOME"] = str(home)

                self.assertEqual(default_config_dir(), home / ".tokenclaw")
                self.assertTrue((home / ".tokenclaw" / "activation.json").exists())
                self.assertTrue((home / ".agentflow" / "activation.json").exists())
                self.assertEqual(default_db_path(), home / ".tokenclaw" / "tokenclaw.sqlite3")
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
