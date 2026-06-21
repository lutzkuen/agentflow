from pathlib import Path
import unittest

import yaml


class InstallSmokeWorkflowTests(unittest.TestCase):
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
            '"$venv_python" -m pip install dist/*.whl',
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


if __name__ == "__main__":
    unittest.main()
