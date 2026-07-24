from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_dev.sh"


class StartDevScriptTests(unittest.TestCase):
    def test_start_dev_script_targets_tokenclaw_dev_proxy_contract(self):
        raw = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("python", raw)
        self.assertIn("-m tokenclaw.server", raw)
        self.assertIn("--provider anthropic", raw)
        self.assertIn('--host "${TOKENCLAW_HOST}"', raw)
        self.assertIn('--port "${TOKENCLAW_PORT}"', raw)
        self.assertIn('export TOKENCLAW_HOST="127.0.0.1"', raw)
        self.assertIn('export TOKENCLAW_PORT="4001"', raw)
        self.assertIn('export TOKENCLAW_DB="${TOKENCLAW_DEV_DB}"', raw)
        self.assertIn("unset TOKENCLAW_DATABASE_URL", raw)
        self.assertIn("/.tokenclaw/dev.sqlite3", raw)

        self.assertNotIn("agentflow_proxy", raw)
        self.assertNotIn("AGENTFLOW_DB", raw)
        self.assertNotIn("AGENTFLOW_PORT", raw)
        self.assertNotIn("pkill", raw)
        self.assertNotIn("kill ", raw)

    @unittest.skipUnless(os.name == "posix", "POSIX exec bit and bash only present on POSIX")
    def test_start_dev_script_is_executable_and_shell_syntax_valid(self):
        mode = SCRIPT.stat().st_mode

        self.assertTrue(mode & stat.S_IXUSR, oct(mode))
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_start_dev_script_does_not_reference_prod_database_default(self):
        raw = SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn(".tokenclaw/tokenclaw.sqlite3", raw)
        self.assertEqual(raw.count("4001"), 1)


class ServerBrandingTests(unittest.TestCase):
    def test_server_title_and_cli_description_use_tokenclaw_name(self):
        with TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["TOKENCLAW_DB"] = str(Path(tmp) / "tokenclaw.sqlite3")
            env.pop("TOKENCLAW_DATABASE_URL", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import tokenclaw.server as s; "
                        "print(s.app.title); "
                        "s.configure_provider('openai'); "
                        "print(s.app.title); "
                        "s.main(['--help'])"
                    ),
                ],
                cwd=ROOT,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TokenClaw Anthropic Proxy", result.stdout)
        self.assertIn("TokenClaw Openai Proxy", result.stdout)
        self.assertIn("TokenClaw provider-specific local proxy", result.stdout)
        self.assertNotIn("AgentFlow", result.stdout)
