import io
import json
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from agentflow_proxy import cli
from agentflow_proxy.local_activation_executor import build_local_activation_executor_plan


NOW = datetime(2026, 6, 19, 5, 30, tzinfo=timezone.utc)


def _executor_fixture_plan():
    ledger = {
        "schema": "agentflow.evidence_to_activation_next_action_ledger.v1",
        "status": "tracked",
        "entries": [
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 1,
                "fingerprint": "activation:crunch",
                "lever": "crunch",
                "local_action_family": "crunch",
                "evidence_schema": "agentflow.crunch_savings_signal.v1",
                "current_status": "full-rollout",
                "state": "full-rollout-active",
                "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
                "blocker_codes": ["repeated-context-crunch-full-rollout-active"],
                "sample_count": 2093,
                "applied_count": 107,
                "holdout_count": 40,
                "fallback_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "projected_saved_usd": 25.818387,
                "crunch_savings_usd": 25.818613,
                "target_local_policy_section": "crunch.rules",
                "target_local_rule_file": "crunch_rules.yaml",
                "duplicate_suppression": {
                    "reason": "repeated-context-crunch-full-rollout-active",
                    "suppresses_new_activation_issue": True,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 2,
                "fingerprint": "activation:openai-routing",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.openai_routing_promotion_decision_report.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "next_action": "review-openai-routing-canary-blockers",
                "blocker_codes": ["semantic-quality-regression-observed"],
                "sample_count": 365,
                "applied_count": 25,
                "holdout_count": 21,
                "savings_per_1000_calls_usd": 4.375,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "request_id": "req-routing-secret",
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 3,
                "fingerprint": "activation:anthropic-safety-stop",
                "lever": "activation-feedback",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.activation_safety_stop_burndown.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                "blocker_codes": ["anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked"],
                "sample_count": 492,
                "safety_stop_count": 492,
                "missing_applied_coverage": True,
                "missing_holdout_coverage": True,
                "source_surface": "anthropic_messages",
                "endpoint": "/v1/messages",
                "category": "tool-result",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "required_local_executor": "anthropic-routing-rules",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "session_id": "session-anthropic-secret",
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 4,
                "fingerprint": "activation:tool-cache",
                "lever": "cache",
                "local_action_family": "cache",
                "evidence_schema": "agentflow.request_shape_tool_cache_replay_evidence.v1",
                "current_status": "blocked",
                "state": "missing-evidence",
                "next_action": "collect-file-invalidation-evidence",
                "blocker_codes": [
                    "invalidation-evidence-missing",
                    "tools-present",
                    "unsafe-tool-calls-without-invalidation",
                ],
                "sample_count": 249,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "target_local_policy_section": "cache.pattern_rules",
                "target_local_rule_file": "cache_rules.yaml",
                "emits_cache_apply_action": False,
                "policy_files_written": False,
                "cache_key": "cache-tool-secret",
                "file_path": "/tmp/private-tool-cache.py",
                "raw_prompt": "raw prompt must not leak",
            },
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }
    return {
        "schema": "agentflow.orchestrator_research_plan.v1",
        "generated_at": NOW.isoformat(),
        "evidence": {"stats_summary": {"evidence_to_activation_next_action_ledger": ledger}},
    }


class LocalActivationExecutorTest(unittest.TestCase):
    def test_executor_plan_selects_one_safe_review_action(self):
        result = build_local_activation_executor_plan(_executor_fixture_plan(), now=NOW)

        self.assertEqual(result["schema"], "agentflow.local_activation_executor_plan.v1")
        self.assertEqual(result["status"], "ranked")
        self.assertEqual(result["summary"]["selected_action_count"], 1)
        self.assertEqual(result["summary"]["selected_executor_action_class"], "review-only")
        self.assertEqual(result["summary"]["selected_next_action"], "review-openai-routing-canary-blockers")
        self.assertEqual(result["summary"]["selected_local_action_family"], "routing")
        self.assertEqual(result["summary"]["selected_target_local_rule_file"], "routing_rules.yaml")
        self.assertFalse(result["summary"]["policy_files_written"])
        self.assertFalse(result["summary"]["provider_calls_made"])
        self.assertFalse(result["summary"]["managed_server_calls_made"])

        entries = result["entries"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(len({entry["fingerprint"] for entry in entries}), 4)
        selected = result["selected_action"]
        self.assertTrue(selected["selected"])
        self.assertTrue(selected["fingerprint"].startswith("executor:"))
        self.assertTrue(selected["source_successor_fingerprint"].startswith("successor:"))
        self.assertEqual(selected["source_fingerprint"], "activation:openai-routing")
        self.assertEqual(selected["executor_action_class"], "review-only")
        self.assertIn("semantic-quality-regression-observed", selected["reason_codes"])

        by_next_action = {entry["executor_next_action"]: entry for entry in entries}
        self.assertEqual(
            by_next_action["keep-current-rule-only"]["executor_action_class"],
            "keep-current-rule",
        )
        self.assertEqual(
            by_next_action["review-openai-routing-canary-blockers"]["executor_action_class"],
            "review-only",
        )
        self.assertEqual(
            by_next_action["keep-anthropic-routing-blocked-until-safety-stop-burndown"]["executor_action_class"],
            "keep-blocked",
        )
        self.assertEqual(
            by_next_action["collect-file-invalidation-evidence"]["executor_action_class"],
            "keep-blocked",
        )
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertTrue(result["privacy"]["aggregate_only"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["provider_bodies_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["policy_file_contents_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])
        self.assertFalse(result["privacy"]["managed_server_calls_made"])
        self.assertFalse(result["privacy"]["policy_files_written"])

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("req-routing-secret", rendered)
        self.assertNotIn("session-anthropic-secret", rendered)
        self.assertNotIn("cache-tool-secret", rendered)
        self.assertNotIn("/tmp/private-tool-cache.py", rendered)
        self.assertNotIn("raw prompt must not leak", rendered)

    def test_local_activation_executor_cli_reads_plan_json(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.local_activation_executor_cli(["--plan-json", str(plan_path), "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.local_activation_executor_plan.v1")
        self.assertEqual(result["summary"]["selected_action_count"], 1)
        self.assertEqual(result["selected_action"]["executor_next_action"], "review-openai-routing-canary-blockers")


if __name__ == "__main__":
    unittest.main()
