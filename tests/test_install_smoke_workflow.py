from pathlib import Path
import builtins
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest

import yaml


class InstallSmokeWorkflowTests(unittest.TestCase):
    def test_default_install_metadata_keeps_advanced_dependencies_optional(self):
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        optional = project["project"]["optional-dependencies"]

        for package in ("psycopg", "zstandard", "websockets"):
            with self.subTest(package=package):
                self.assertFalse(any(req.lower().startswith(package) for req in dependencies))

        self.assertIn("psycopg[binary,pool]>=3.2", optional["managed"])
        self.assertIn("zstandard>=0.23", optional["compression"])
        self.assertIn("websockets>=15.0", optional["openai-realtime"])

    def test_clean_install_smoke_workflow_covers_issue_acceptance(self):
        workflow_path = Path(".github/workflows/install-smoke.yml")
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual(workflow["name"], "Clean install smoke")
        triggers = workflow["on"]
        self.assertEqual(triggers["schedule"], [{"cron": "0 3 * * *"}])
        self.assertEqual(triggers["push"]["tags"], ["v*"])
        self.assertIn("pyproject.toml", triggers["pull_request"]["paths"])
        self.assertIn("tokenclaw/activation.py", triggers["pull_request"]["paths"])
        self.assertIn("tokenclaw/cli*.py", triggers["pull_request"]["paths"])
        self.assertIn("tokenclaw/server.py", triggers["pull_request"]["paths"])
        self.assertIn("tokenclaw/store.py", triggers["pull_request"]["paths"])

        job = workflow["jobs"]["wheel-smoke"]
        matrix = job["strategy"]["matrix"]["include"]
        self.assertIn({"os": "ubuntu-latest", "python-version": "3.11", "proxy_smoke": True}, matrix)
        self.assertIn({"os": "ubuntu-latest", "python-version": "3.12", "proxy_smoke": True}, matrix)
        self.assertIn({"os": "macos-latest", "python-version": "3.12", "proxy_smoke": True}, matrix)
        self.assertIn({"os": "windows-latest", "python-version": "3.12", "proxy_smoke": False}, matrix)

        step_text = "\n".join(str(step) for step in job["steps"])
        for expected in (
            "python -m build --wheel",
            "python -m venv .venv-smoke",
            # Two-phase install since the library split: base wheel first
            # (server-free), then the [server] extra for the CLI/proxy smoke.
            '"$venv_python" -m pip install "$wheel"',
            '"$venv_python" -m pip install "${wheel}[server]"',
            "base wheel unexpectedly requires optional dependency",
            "tokenclaw version",
            "tokenclaw activate claude --dry-run",
            "tokenclaw activate openai --dry-run",
            "tokenclaw activate codex",
            "tokenclaw activate claude >/dev/null",
            "tokenclaw activate openai >/dev/null",
            "tokenclaw activate claude-desktop",
            "tokenclaw doctor",
            "tokenclaw stats --json",
            "tokenclaw db adopt-legacy --dry-run",
            "tokenclaw run claude",
            "http://127.0.0.1:4000/health",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, step_text)

    def test_base_server_import_does_not_require_openai_realtime_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["TOKENCLAW_DB"] = str(Path(tmp) / "tokenclaw.sqlite3")
            env.pop("TOKENCLAW_DATABASE_URL", None)
            code = """
import builtins
import importlib
import sys

sys.modules.pop("websockets", None)
original_import = builtins.__import__

def import_without_websockets(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "websockets" or name.startswith("websockets.")):
        raise ImportError("blocked optional websockets dependency")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_websockets
server = importlib.import_module("tokenclaw.server")
assert server.PROVIDER == "anthropic", server.PROVIDER
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

    def test_postgres_store_missing_extra_mentions_managed_extra(self):
        from tokenclaw.store import PostgresStore

        original_import = builtins.__import__

        def import_without_psycopg_pool(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "psycopg_pool" or name.startswith("psycopg_pool."):
                raise ImportError("blocked optional psycopg dependency")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_psycopg_pool
        try:
            with self.assertRaisesRegex(RuntimeError, r"tokenclaw\[managed\]"):
                PostgresStore("postgresql://example.invalid/tokenclaw")
        finally:
            builtins.__import__ = original_import

    def test_openai_realtime_missing_extra_mentions_openai_realtime_extra(self):
        from tokenclaw import openai_proxy

        original_import = builtins.__import__

        def import_without_websockets(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "websockets" or name.startswith("websockets.")):
                raise ImportError("blocked optional websockets dependency")
            return original_import(name, globals, locals, fromlist, level)

        old_module = sys.modules.pop("websockets", None)
        builtins.__import__ = import_without_websockets
        try:
            with self.assertRaisesRegex(RuntimeError, r"tokenclaw\[openai-realtime\]"):
                openai_proxy._import_websockets()
        finally:
            builtins.__import__ = original_import
            if old_module is not None:
                sys.modules["websockets"] = old_module

    def test_zstd_missing_extra_mentions_compression_extra(self):
        from tokenclaw import headers

        class Request:
            headers = {"content-encoding": "zstd"}

            async def body(self):
                return b"{}"

        old_zstd = headers.zstd
        headers.zstd = None
        try:
            with self.assertRaisesRegex(headers.ClientJsonRequestError, r"tokenclaw\[compression\]"):
                import asyncio

                asyncio.run(headers.read_json_object_body(Request(), allow_compressed=True))
        finally:
            headers.zstd = old_zstd


if __name__ == "__main__":
    unittest.main()
