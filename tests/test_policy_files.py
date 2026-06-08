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
from agentflow_proxy.policy_bundle import (
    apply_policy_bundle,
    build_policy_bundle,
    compare_policy_bundles,
    review_policy_bundle,
    validate_policy_bundle,
)


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
        self.assertEqual(bundle["policies"]["summary"]["policy_count"], 5)
        self.assertFalse(bundle["policies"]["summary"]["reload_required"])
        self.assertEqual(bundle["policies"]["summary"]["reload_required_sections"], [])
        self.assertIn("routing", bundle["policies"])
        self.assertIn("crunch", bundle["policies"])
        self.assertIn("cache", bundle["policies"])
        self.assertIn("routing_experiments", bundle["policies"])
        self.assertIn("codex_app", bundle["policies"])
        self.assertTrue(bundle["policies"]["codex_app"]["review_only"])

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

    def test_policy_bundle_validation_rejects_malformed_section_shapes(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["routing"]["rules"][0]["conditions"]["text_chars_lt"] = "many"
        bundle["policies"]["routing"]["rules"][0]["conditions"]["unsupported"] = True
        bundle["policies"]["routing"]["rules"][0]["action"]["route_to"] = ""
        bundle["policies"]["crunch"]["old_context_summarization"]["placement"] = "message"
        bundle["policies"]["crunch"]["thinking_deduplication"]["similarity_threshold"] = 1.5
        bundle["policies"]["cache"]["semantic_cache"]["threshold"] = -0.1
        bundle["policies"]["cache"]["file_watch"]["max_paths"] = "lots"

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        paths = {error["path"] for error in result["errors"]}
        self.assertIn("$.policies.routing.rules[0].conditions.text_chars_lt", paths)
        self.assertIn("$.policies.routing.rules[0].conditions.unsupported", paths)
        self.assertIn("$.policies.routing.rules[0].action.route_to", paths)
        self.assertIn("$.policies.crunch.old_context_summarization.placement", paths)
        self.assertIn("$.policies.crunch.thinking_deduplication.similarity_threshold", paths)
        self.assertIn("$.policies.cache.semantic_cache.threshold", paths)
        self.assertIn("$.policies.cache.file_watch.max_paths", paths)

    def test_policy_bundle_validation_accepts_reviewable_codex_app_policy(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"] = {
            "enabled": True,
            "policy_source": "managed-recommended",
            "surface": "codex_app_turn",
            "review_only": True,
            "rules": [
                {
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "missing",
                        "input_size_bucket": "medium",
                        "cache_eligible": False,
                        "replayability_level": "turn-metadata-only",
                        "has_action_like_params": False,
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "crunch_profile": "codex-summary",
                        "cache_eligible": False,
                        "cache_eligibility_reason": "metadata-only recommendation requires local replayability review",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                }
            ],
        }

        result = validate_policy_bundle(bundle)

        self.assertTrue(result["ok"])
        self.assertEqual(result["errors"], [])

    def test_policy_bundle_validation_rejects_codex_app_managed_enforced(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"]["policy_source"] = "managed-enforced"

        result = validate_policy_bundle(bundle)

        self.assertFalse(result["ok"])
        self.assertIn("$.policies.codex_app.policy_source", {error["path"] for error in result["errors"]})

    def test_policy_review_reports_codex_app_section_without_provider_routing_apply(self):
        current = asyncio.run(build_policy_bundle())
        proposed = json.loads(json.dumps(current))
        proposed["policies"]["codex_app"]["policy_source"] = "managed-recommended"
        proposed["policies"]["codex_app"]["rules"] = [
            {
                "candidate_id": "codex-tool-execution-small",
                "conditions": {
                    "app_family": "codex",
                    "workflow_phase": "tool_execution",
                    "model_field_state": "present",
                    "input_size_bucket": "small",
                    "cache_eligible": True,
                    "replayability_level": "local-exact-response",
                },
                "action": {
                    "recommended_model": "gpt-5-mini",
                    "crunch_profile": "pass-through",
                    "cache_eligibility_reason": "local exact replay only",
                    "reason": "managed Codex app turn policy for local review",
                },
            }
        ]

        result = review_policy_bundle(current, proposed, include_impact=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["changed_sections"], ["codex_app"])
        self.assertTrue(any(change["path"].startswith("$.policies.codex_app") for change in result["diff"]["changes"]))
        codex_review = result["section_reviews"]["codex_app"]
        self.assertEqual(codex_review["status"], "review-only")
        self.assertTrue(codex_review["review_only"])
        self.assertEqual(codex_review["candidate_ids"], ["codex-tool-execution-small"])
        self.assertEqual(codex_review["application"]["status"], "not-applied")
        self.assertFalse(codex_review["application"]["writes_local_policy_files"])
        self.assertIn("workflow_phase", codex_review["condition_keys_present"])
        self.assertIn("recommended_model", codex_review["action_keys_present"])
        self.assertFalse(codex_review["privacy"]["raw_prompts_included"])

    def test_policy_apply_reports_codex_app_review_only_and_writes_no_codex_yaml(self):
        bundle = asyncio.run(build_policy_bundle())
        bundle["policies"]["codex_app"] = {
            **bundle["policies"]["codex_app"],
            "policy_source": "managed-recommended",
            "rules": [
                {
                    "candidate_id": "codex-summary-pass-through",
                    "conditions": {
                        "app_family": "codex",
                        "workflow_phase": "summary",
                        "model_field_state": "derived_present",
                    },
                    "action": {
                        "model_hint": "gpt-5-mini",
                        "pass_through_reason": "review-only Codex app recommendation",
                    },
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            result = apply_policy_bundle(bundle, config_dir=tmp, dry_run=True)

            self.assertTrue(result["ok"])
            skipped = {item["section"]: item for item in result["skipped_sections"]}
            self.assertEqual(skipped["codex_app"]["reason"], "review-only-not-applied")
            self.assertFalse((Path(tmp) / "codex_app_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "crunch_rules.yaml").exists())
            self.assertFalse((Path(tmp) / "cache_rules.yaml").exists())
            self.assertFalse(any("codex-summary-pass-through" in json.dumps(file) for file in result["files"]))

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
                self.assertFalse(first["summary"]["reload_required"])
                self.assertEqual(first["summary"]["reload_required_sections"], [])
                self.assertEqual(first["summary"]["manual_policy_count"], 1)
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
                self.assertTrue(changed["summary"]["reload_required"])
                self.assertEqual(changed["summary"]["reload_required_sections"], ["routing"])
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
