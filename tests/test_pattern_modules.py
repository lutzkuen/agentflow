import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import agentflow_proxy.crunch as crunch_module
from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.pattern_modules import (
    LocalPatternModule,
    PatternCrunchResult,
    PatternDetection,
    PatternModuleContext,
    PatternModuleRegistry,
    PromptRolePatternModule,
    TerminalLogPatternModule,
    evaluate_pattern_modules,
    registered_pattern_modules,
)


class TestLocalCrunchModule(LocalPatternModule):
    family = "test_safe_crunch"
    version = "test.1"
    feature_schema = "agentflow.test_safe_crunch_features.v1"
    supports_local_crunch = True

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        detected = "SECRET_REPEAT" in context.text
        return PatternDetection(detected=detected, reason="test-pattern-detected" if detected else "missing")

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict:
        return {
            "schema": self.feature_schema,
            "module_family": self.family,
            "module_version": self.version,
            "detected": detection.detected,
            "repeated_marker_bucket": "gte_11",
            "privacy": {"metadata_only": True, "raw_content_included": False},
        }

    def apply_local_crunch(
        self,
        body: dict,
        context: PatternModuleContext,
        detection: PatternDetection,
    ) -> PatternCrunchResult:
        new_body = copy.deepcopy(body)
        before = json.dumps(new_body, sort_keys=True)

        def visit(value):
            if isinstance(value, str):
                return value.replace("SECRET_REPEAT " * 20, "[AgentFlow: test repeated block omitted]")
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {key: visit(child) for key, child in value.items()}
            return value

        new_body = visit(new_body)
        after = json.dumps(new_body, sort_keys=True)
        return PatternCrunchResult(
            body=new_body,
            changed=before != after,
            saved_chars=max(0, len(before) - len(after)),
            reason="test-local-crunch-applied",
        )


class UnsafeFeatureModule(LocalPatternModule):
    family = "unsafe_fixture"
    version = "test.1"
    feature_schema = "agentflow.unsafe_fixture_features.v1"

    def detect(self, context: PatternModuleContext) -> PatternDetection:
        return PatternDetection(detected=True, reason="unsafe-fixture")

    def features(self, context: PatternModuleContext, detection: PatternDetection) -> dict:
        return {
            "schema": self.feature_schema,
            "prompt": "SECRET raw prompt text",
        }


class PatternModuleTests(unittest.TestCase):
    def test_default_registry_exposes_two_independent_families(self):
        modules = registered_pattern_modules()

        families = {module["family"] for module in modules}
        self.assertIn("terminal_logs", families)
        self.assertIn("prompt_role", families)
        self.assertTrue(all(module["feature_schema"] for module in modules))

    def test_registered_modules_emit_privacy_safe_bucketed_features(self):
        body = {
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": "\n".join([
                    "Please inspect this secret failure.",
                    "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
                    "Find current outstanding voucher 12345.",
                ]),
            }],
        }

        crunched, meta = evaluate_pattern_modules(body, category="chat")

        rendered = json.dumps(meta["server_features"], sort_keys=True)
        self.assertEqual(crunched, body)
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])
        self.assertEqual(meta["features_emitted_count"], 2)
        self.assertEqual({item["family"] for item in meta["server_features"]["features"]}, {"prompt_role", "terminal_logs"})
        self.assertNotIn("secret-app", rendered)
        self.assertNotIn("voucher 12345", rendered)

    def test_modules_can_be_disabled_independently(self):
        body = {
            "messages": [{
                "role": "user",
                "content": "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
            }],
        }

        _, meta = evaluate_pattern_modules(
            body,
            module_settings={"terminal_logs": {"enabled": False}, "prompt_role": {"enabled": True}},
            category="chat",
        )

        terminal = next(item for item in meta["modules"] if item["family"] == "terminal_logs")
        prompt = next(item for item in meta["modules"] if item["family"] == "prompt_role")
        self.assertEqual(terminal["status"], "skipped")
        self.assertEqual(terminal["reason"], "disabled")
        self.assertEqual(prompt["status"], "skipped")
        self.assertTrue(prompt["features_emitted"])
        self.assertEqual({item["family"] for item in meta["server_features"]["features"]}, {"prompt_role"})

    def test_local_crunch_application_reports_outcome_metadata(self):
        body = {"messages": [{"role": "user", "content": "SECRET_REPEAT " * 20 + "keep"}]}
        registry = PatternModuleRegistry([TestLocalCrunchModule()])

        crunched, meta = evaluate_pattern_modules(
            body,
            registry=registry,
            module_settings={"test_safe_crunch": {"enabled": True, "local_crunch_enabled": True}},
            category="chat",
        )

        module_meta = meta["modules"][0]
        self.assertNotEqual(crunched, body)
        self.assertIn("test repeated block omitted", crunched["messages"][0]["content"])
        self.assertEqual(module_meta["status"], "applied")
        self.assertEqual(module_meta["reason"], "test-local-crunch-applied")
        self.assertGreater(module_meta["saved_chars"], 0)
        self.assertEqual(meta["applied_count"], 1)
        self.assertEqual(managed_egress_violations(meta["server_features"]), [])

    def test_privacy_guard_rejects_raw_like_feature_keys(self):
        body = {"messages": [{"role": "user", "content": "SECRET raw prompt text"}]}
        registry = PatternModuleRegistry([UnsafeFeatureModule()])

        crunched, meta = evaluate_pattern_modules(body, registry=registry, category="chat")

        module_meta = meta["modules"][0]
        self.assertEqual(crunched, body)
        self.assertEqual(module_meta["status"], "bypass")
        self.assertEqual(module_meta["reason"], "privacy-guard-rejected")
        self.assertFalse(module_meta["features_emitted"])
        self.assertEqual(module_meta["privacy_guard"]["blocked_keys"], ["prompt"])
        self.assertEqual(meta["features_emitted_count"], 0)

    def test_crunch_body_reports_pattern_module_features_and_honors_config_disable(self):
        saved_env = os.environ.get("AGENTFLOW_CRUNCH_RULES")
        try:
            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                rules_path = tmp_path / "crunch_rules.yaml"
                rules_path.write_text(
                    """
enabled: true
pattern_modules:
  terminal_logs:
    enabled: false
  prompt_role:
    enabled: true
""",
                    encoding="utf-8",
                )
                os.environ["AGENTFLOW_CRUNCH_RULES"] = str(rules_path)
                manual = importlib.reload(crunch_module)
                body = {
                    "messages": [{
                        "role": "user",
                        "content": "2026-06-09T20:00:00Z ERROR pid=1234 secret-app failed",
                    }],
                }

                _, meta = manual.crunch_body(body)

                modules = meta["pattern_modules"]["modules"]
                terminal = next(item for item in modules if item["family"] == "terminal_logs")
                self.assertEqual(terminal["reason"], "disabled")
                self.assertEqual(meta["pattern_modules"]["features_emitted_count"], 1)
                self.assertEqual(
                    {item["family"] for item in meta["pattern_modules"]["server_features"]["features"]},
                    {"prompt_role"},
                )
                self.assertEqual(managed_egress_violations(meta["pattern_modules"]["server_features"]), [])
        finally:
            if saved_env is None:
                os.environ.pop("AGENTFLOW_CRUNCH_RULES", None)
            else:
                os.environ["AGENTFLOW_CRUNCH_RULES"] = saved_env
            importlib.reload(crunch_module)


if __name__ == "__main__":
    unittest.main()
