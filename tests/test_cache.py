import unittest
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import agentflow_proxy.cache as cache_module


class CacheDecisionMetaTest(unittest.TestCase):
    ENV_KEYS = (
        "AGENTFLOW_CACHE",
        "AGENTFLOW_CACHE_TOOL_CALLS",
        "AGENTFLOW_SEMANTIC_CACHE",
        "AGENTFLOW_SEMANTIC_THRESHOLD",
        "AGENTFLOW_CACHE_RULES",
        "HOME",
    )

    def setUp(self):
        self.old_cwd = Path.cwd()
        self.home = TemporaryDirectory()
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["HOME"] = self.home.name
        importlib.reload(cache_module)

    def tearDown(self):
        os.chdir(self.old_cwd)
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.home.cleanup()
        importlib.reload(cache_module)

    def test_cache_hit_metadata_has_explicit_status_and_source(self):
        meta = cache_module.cache_decision_meta(
            "hit",
            "exact-match",
            hit_type="exact",
            exact_enabled=True,
            semantic_enabled=False,
        )

        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["status"], "hit")
        self.assertEqual(meta["reason"], "exact-match")
        self.assertEqual(meta["hit_type"], "exact")
        self.assertEqual(meta["policy_source"], "local-default")

    def test_tool_requests_are_skipped_when_tool_cache_disabled(self):
        can_exact, can_semantic, meta = cache_module.cache_lookup_meta(has_tool_blocks=True)

        self.assertFalse(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "tools-disabled")
        self.assertTrue(meta["enabled"])
        self.assertFalse(meta["tool_cache_enabled"])

    def test_non_tool_requests_report_exact_miss_by_default(self):
        can_exact, can_semantic, meta = cache_module.cache_lookup_meta(has_tool_blocks=False)

        self.assertTrue(can_exact)
        self.assertFalse(can_semantic)
        self.assertEqual(meta["status"], "miss")
        self.assertEqual(meta["reason"], "exact-miss")

    def test_config_cache_rules_can_change_cache_behavior_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                """
exact_cache:
  enabled: false
  cache_tool_calls: false
semantic_cache:
  enabled: true
  threshold: 0.82
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)

            can_exact, can_semantic, meta = manual.cache_lookup_meta(has_tool_blocks=False)

            self.assertFalse(can_exact)
            self.assertTrue(can_semantic)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "semantic-miss")
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertEqual(meta["rule_path"], str(config / "cache_rules.yaml"))
            self.assertEqual(meta["semantic_threshold"], 0.82)

    def test_config_cache_rules_can_enable_exact_tool_cache_without_env_flags(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "config"
            config.mkdir()
            (config / "cache_rules.yaml").write_text(
                """
exact_cache:
  enabled: true
  cache_tool_calls: true
semantic_cache:
  enabled: false
  threshold: 0.95
""",
                encoding="utf-8",
            )
            os.chdir(tmp_path)
            manual = importlib.reload(cache_module)

            can_exact, can_semantic, meta = manual.cache_lookup_meta(has_tool_blocks=True)

            self.assertTrue(can_exact)
            self.assertFalse(can_semantic)
            self.assertEqual(meta["status"], "miss")
            self.assertEqual(meta["reason"], "exact-miss")
            self.assertEqual(meta["policy_source"], "local-manual")
            self.assertTrue(meta["tool_cache_enabled"])


if __name__ == "__main__":
    unittest.main()
