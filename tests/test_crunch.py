import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import agentflow_proxy.crunch as crunch_module


class CrunchRulesTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CRUNCH",
        "AGENTFLOW_CRUNCH_THRESHOLD_CHARS",
        "AGENTFLOW_PROMPT_CACHE",
        "AGENTFLOW_PROMPT_CACHE_MIN_CHARS",
        "AGENTFLOW_CRUNCH_RULES",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(crunch_module)

    def test_default_crunch_policy_reports_bundled_local_default_source(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CRUNCH": "1",
                "AGENTFLOW_CRUNCH_THRESHOLD_CHARS": "24000",
                "AGENTFLOW_PROMPT_CACHE": "1",
                "AGENTFLOW_PROMPT_CACHE_MIN_CHARS": "4096",
                "AGENTFLOW_CRUNCH_RULES": "",
            },
        ):
            manual = importlib.reload(crunch_module)

            _, meta = manual.crunch_body({"model": "claude-sonnet-4-6", "messages": []})

            self.assertTrue(meta["enabled"])
            self.assertEqual(meta["policy_source"], "local-default")
            self.assertTrue(meta["rule_path"].endswith("agentflow_proxy/crunch_rules.yaml"))

    def test_config_crunch_rules_can_disable_crunch_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text("enabled: false\n", encoding="utf-8")
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            body = {"messages": [{"role": "user", "content": "hello"}]}
            crunched, meta = manual.crunch_body(body)

            self.assertIs(crunched, body)
            self.assertFalse(meta["enabled"])
            self.assertFalse(meta["changed"])
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["rule_path"], str(config / "crunch_rules.yaml"))

    def test_config_crunch_rules_can_change_shortening_threshold_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "crunch_rules.yaml").write_text(
                """
enabled: true
threshold_chars: 10
prompt_cache:
  enabled: true
  min_chars: 4096
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(crunch_module)

            long_text = "alpha " * 1600
            body = {
                "messages": [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": "one"},
                    {"role": "user", "content": "two"},
                    {"role": "assistant", "content": "three"},
                    {"role": "user", "content": "four"},
                ]
            }
            crunched, meta = manual.crunch_body(body)

            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["threshold_chars"], 10)
            self.assertEqual(meta["long_blocks_shortened"], 1)
            self.assertIn("middle of long older text block omitted", crunched["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
