import asyncio
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.router as router_module
from agentflow_proxy.admin import reload_policy_modules
from agentflow_proxy import stats
from agentflow_proxy.policy_files import policy_file_snapshot, policy_file_status, utc_now


class PolicyFileStatusTest(unittest.TestCase):
    def test_policy_file_status_marks_reload_required_after_file_change(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: initial
""",
                encoding="utf-8",
            )
            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                importlib.reload(router_module)
                first = asyncio.run(stats.stats_policies())
                self.assertEqual(first["routing"]["policy_source"], "local-manual")
                self.assertEqual(first["routing"]["rule_path"], str(rules_path))
                self.assertFalse(first["routing"]["file"]["reload_required"])
                self.assertTrue(first["routing"]["file"]["loaded"]["exists"])
                self.assertIsNotNone(first["routing"]["file"]["loaded"]["size"])
                self.assertIsNotNone(first["routing"]["file"]["loaded"]["mtime_ns"])

                rules_path.write_text(
                    """
rules:
  - conditions:
      model_pattern: sonnet
      category: tool-result
    action:
      route_to: haiku
      reason: changed
""",
                    encoding="utf-8",
                )

                changed = asyncio.run(stats.stats_policies())
                self.assertTrue(changed["routing"]["file"]["reload_required"])
                self.assertNotEqual(
                    changed["routing"]["file"]["loaded"]["sha256"],
                    changed["routing"]["file"]["current"]["sha256"],
                )
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                importlib.reload(router_module)

    def test_policy_file_status_handles_missing_file_without_reload(self):
        missing = Path("/tmp/agentflow-policy-file-status-missing.yaml")
        loaded = policy_file_snapshot(missing)

        status = policy_file_status(missing, loaded_at=utc_now(), loaded_snapshot=loaded)

        self.assertFalse(status["loaded"]["exists"])
        self.assertFalse(status["current"]["exists"])
        self.assertFalse(status["reload_required"])

    def test_reload_policy_modules_applies_changed_routing_file(self):
        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "routing_rules.yaml"
            rules_path.write_text(
                """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
    action:
      route_to: haiku
      reason: initial reload test
""",
                encoding="utf-8",
            )

            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                first = asyncio.run(reload_policy_modules())
                self.assertEqual(first["schema"], "agentflow.policy_reload.v1")
                self.assertFalse(first["policies"]["routing"]["file"]["reload_required"])

                body = {
                    "model": router_module.SONNET_DEFAULT,
                    "messages": [{"role": "user", "content": "Say ok."}],
                }
                routed, meta = router_module.route_model(body)
                self.assertEqual(routed, router_module.HAIKU_DEFAULT)
                self.assertEqual(meta["reason"], "initial reload test")

                rules_path.write_text(
                    """
rules:
  - conditions:
      model_pattern: sonnet
      has_tools: false
    action:
      route_to: haiku
      reason: changed reload test
""",
                    encoding="utf-8",
                )
                stale = asyncio.run(stats.stats_policies())
                self.assertTrue(stale["routing"]["file"]["reload_required"])

                changed = asyncio.run(reload_policy_modules())
                self.assertFalse(changed["policies"]["routing"]["file"]["reload_required"])
                _routed, changed_meta = router_module.route_model(body)
                self.assertEqual(changed_meta["reason"], "changed reload test")
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                asyncio.run(reload_policy_modules())


if __name__ == "__main__":
    unittest.main()
