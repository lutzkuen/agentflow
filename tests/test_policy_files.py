import asyncio
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.router as router_module
from agentflow_proxy.admin import reload_policy_modules
from agentflow_proxy import stats
from agentflow_proxy.policy_files import policy_file_snapshot, policy_file_status, utc_now
from agentflow_proxy.policy_bundle import build_policy_bundle, compare_policy_bundles, validate_policy_bundle


class PolicyFileStatusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        self.tmp.cleanup()

    def test_policy_bundle_exports_effective_default_policy_state(self):
        bundle = asyncio.run(build_policy_bundle())

        self.assertEqual(bundle["schema"], "agentflow.policy_bundle.v1")
        self.assertEqual(bundle["generator"]["name"], "agentflow-proxy")
        self.assertEqual(bundle["generator"]["mode"], "local-offline")
        self.assertFalse(bundle["managed_optimizer"]["enabled"])
        self.assertEqual(bundle["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertIn("routing", bundle["policies"])
        self.assertIn("crunch", bundle["policies"])
        self.assertIn("cache", bundle["policies"])
        self.assertIn("routing_experiments", bundle["policies"])

    def test_policy_bundle_validation_accepts_exported_bundle(self):
        bundle = asyncio.run(build_policy_bundle())

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_rejects_missing_policy_section(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"].pop("cache")

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        self.assertIn("$.policies.cache", {error["path"] for error in result["errors"]})

    def test_policy_bundle_compare_reports_no_changes_for_identical_bundles(self):
        bundle = asyncio.run(build_policy_bundle())

        result = compare_policy_bundles(bundle, json.loads(json.dumps(bundle)))

        self.assertTrue(result["ok"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["change_count"], 0)
        self.assertEqual(result["changed_sections"], [])

    def test_policy_bundle_compare_reports_policy_section_changes(self):
        before = asyncio.run(build_policy_bundle())
        after = json.loads(json.dumps(before))
        after["policies"]["routing"]["rules"][0]["action"]["reason"] = "changed in diff test"
        after["policies"]["cache"]["semantic_cache"]["threshold"] = 0.99

        result = compare_policy_bundles(before, after)

        self.assertTrue(result["ok"])
        self.assertTrue(result["changed"])
        self.assertEqual(result["change_count"], 2)
        self.assertEqual(result["changed_sections"], ["cache", "routing"])
        changes_by_path = {change["path"]: change for change in result["changes"]}
        self.assertEqual(
            changes_by_path["$.policies.routing.rules[0].action.reason"]["new"],
            "changed in diff test",
        )
        self.assertEqual(changes_by_path["$.policies.cache.semantic_cache.threshold"]["new"], 0.99)

    def test_policy_bundle_compare_returns_validation_errors_for_bad_input(self):
        bundle = asyncio.run(build_policy_bundle())

        result = compare_policy_bundles({"schema": "wrong"}, bundle)

        self.assertFalse(result["ok"])
        self.assertFalse(result["changed"])
        self.assertIn("$.schema", {error["path"] for error in result["before_validation"]["errors"]})

    def test_policy_bundle_exports_manual_policy_source_and_file_status(self):
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
      reason: manual bundle export test
""",
                encoding="utf-8",
            )
            old_env = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(rules_path)
            try:
                asyncio.run(reload_policy_modules())
                bundle = asyncio.run(build_policy_bundle())

                routing = bundle["policies"]["routing"]
                self.assertEqual(routing["policy_source"], "local-manual")
                self.assertEqual(routing["rule_path"], str(rules_path))
                self.assertFalse(routing["file"]["reload_required"])
                self.assertEqual(routing["file"]["loaded"]["path"], str(rules_path))
                self.assertEqual(routing["rules"][0]["action"]["reason"], "manual bundle export test")
            finally:
                if old_env is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_env
                asyncio.run(reload_policy_modules())

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
