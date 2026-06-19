import io
import json
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from agentflow_proxy import cli
from agentflow_proxy.local_activation_executor import (
    build_managed_activation_preview_request,
    build_managed_activation_preview_result,
    build_local_activation_executor_managed_handoff,
    build_local_activation_executor_plan,
)
from agentflow_proxy.managed_egress import managed_egress_violations


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


def _full_rollout_gate(state="keep-active", next_action="keep-active", reason_codes=None):
    return {
        "schema": "agentflow.full_rollout_crunch_keep_active_regression_gate.v1",
        "state": state,
        "gate_passed": state == "keep-active",
        "deterministic_next_action": next_action,
        "next_action": next_action,
        "reason_codes": reason_codes or [],
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "regression_counters": {
            "schema": "agentflow.full_rollout_crunch_keep_active_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": 107,
            "holdout_count": 40,
            "skipped_count": 280,
            "fallback_count": 0,
            "retry_count": 0,
            "rollback_count": 0,
            "safety_stop_count": 0,
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "fallback_rate_delta": 0.0,
            "decision_age_hours": 0.0,
            "stale_evidence": {
                "metadata_only": True,
                "aggregate_only": True,
                "stale": state == "review-stale-evidence",
                "status": "stale" if state == "review-stale-evidence" else "fresh-or-active",
                "reason": "stale-full-rollout-evidence" if state == "review-stale-evidence" else "full-rollout-local-policy-active",
            },
        },
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _full_rollout_queue_entry(*, gate=None, outcome="keep-active"):
    gate = gate or _full_rollout_gate()
    return {
        "schema": "agentflow.local_activation_next_action_queue_entry.v1",
        "rank": 1,
        "ledger_rank": 1,
        "fingerprint": "activation:f5f6eae5f0a0081a",
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
        "skipped_count": 280,
        "fallback_count": 0,
        "retry_count": 0,
        "rollback_count": gate["regression_counters"]["rollback_count"],
        "safety_stop_count": 0,
        "projected_savings_usd": 25.818387,
        "realized_savings_usd": 25.818613,
        "observed_saved_tokens": 8606129,
        "projected_saved_tokens": 8606129,
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "duplicate_suppression_status": "suppressed",
        "duplicate_suppression_reason": "repeated-context-crunch-full-rollout-active",
        "measured_full_rollout_activation": True,
        "durable_outcome_ledger_entry": True,
        "full_rollout_outcome": outcome,
        "full_rollout_outcome_next_action": gate["next_action"],
        "full_rollout_successor_decision": "no-op" if outcome == "keep-active" else outcome,
        "full_rollout_successor_next_action": "keep-current-rule-only" if outcome == "keep-active" else gate["next_action"],
        "full_rollout_successor_no_op_reason": "no-unsuppressed-post-full-rollout-crunch-cohort",
        "full_rollout_activation_outcome": {
            "schema": "agentflow.full_rollout_crunch_activation_outcome.v1",
            "durable_outcome_ledger_entry": True,
            "ledger_fingerprint": "activation:f5f6eae5f0a0081a",
            "ledger_rank": 1,
            "lever": "crunch",
            "local_action_family": "crunch",
            "outcome": outcome,
            "next_action": gate["next_action"],
            "outcome_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
            "applied_count": 107,
            "holdout_count": 40,
            "skipped_count": 280,
            "fallback_count": 0,
            "retry_count": 0,
            "rollback_count": gate["regression_counters"]["rollback_count"],
            "safety_stop_count": 0,
            "observed_saved_tokens": 8606129,
            "observed_savings_usd": 25.818387,
            "projected_saved_tokens": 8606129,
            "projected_savings_usd": 25.818387,
            "successor_decision": "no-op" if outcome == "keep-active" else outcome,
            "successor_next_action": "keep-current-rule-only" if outcome == "keep-active" else gate["next_action"],
            "successor_no_op_reason": "no-unsuppressed-post-full-rollout-crunch-cohort",
            "keep_active_regression_gate": gate,
            "privacy": {"metadata_only": True, "aggregate_only": True},
        },
        "keep_active_regression_gate": gate,
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _full_rollout_queue_plan(entry):
    return {
        "schema": "agentflow.local_activation_next_action_queue.v1",
        "status": "ranked",
        "entries": [entry],
        "successor_actions": [
            {
                "schema": "agentflow.local_activation_successor_action.v1",
                "rank": 1,
                "fingerprint": "successor:stale-precomputed-row",
                "source_fingerprint": entry["fingerprint"],
                "lever": "crunch",
                "local_action_family": "crunch",
                "successor_status": "review",
                "current_status": "full-rollout",
                "state": "full-rollout-active",
                "recommended_next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            }
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _executor_handoff_fixture_plan():
    plan = _full_rollout_queue_plan(_full_rollout_queue_entry())
    plan["entries"].extend(
        [
            {
                "schema": "agentflow.local_activation_next_action_queue_entry.v1",
                "rank": 2,
                "ledger_rank": 2,
                "fingerprint": "activation:routing-blocked",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.activation_safety_stop_burndown.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "successor_status": "keep-blocked",
                "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                "blocker_codes": ["anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked"],
                "sample_count": 492,
                "safety_stop_count": 492,
                "source_surface": "anthropic_messages",
                "endpoint": "/v1/messages",
                "category": "tool-result",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "required_local_executor": "anthropic-routing-rules",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "session_id": "session-routing-secret",
            },
            {
                "schema": "agentflow.local_activation_next_action_queue_entry.v1",
                "rank": 3,
                "ledger_rank": 3,
                "fingerprint": "activation:cache-review",
                "lever": "cache",
                "local_action_family": "cache",
                "evidence_schema": "agentflow.request_shape_cache_replay_policy_decision.v1",
                "current_status": "review",
                "state": "review",
                "successor_status": "review-only",
                "next_action": "review-cache-replay-canary-promotion-readiness",
                "blocker_codes": [],
                "sample_count": 102,
                "applied_count": 30,
                "holdout_count": 72,
                "projected_savings_usd": 0.075373,
                "realized_savings_usd": 0.0,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "target_local_policy_section": "cache.pattern_rules",
                "target_local_rule_file": "cache_rules.yaml",
                "cache_key": "cache-review-secret",
                "file_path": "/tmp/private-cache-review.py",
                "raw_prompt": "raw cache review prompt must not leak",
            },
        ]
    )
    return plan


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

    def test_executor_managed_handoff_exports_feature_only_outcome_rows(self):
        result = build_local_activation_executor_managed_handoff(
            _executor_handoff_fixture_plan(),
            now=NOW,
        )

        self.assertEqual(result["schema"], "agentflow.local_activation_managed_handoff.v1")
        self.assertEqual(result["status"], "exported")
        self.assertEqual(result["summary"]["handoff_row_count"], 3)
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["policy_files_written"])
        self.assertFalse(result["server_ingestion_required"])
        self.assertTrue(result["feature_only"])
        self.assertTrue(result["locally_executed"])
        self.assertFalse(result["provider_forwarding"])
        self.assertFalse(result["server_content_processing"])
        self.assertFalse(result["managed_enforced"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(result["egress_guard"]["violation_count"], 0)
        self.assertEqual(managed_egress_violations(result), [])

        privacy = result["privacy"]
        self.assertTrue(privacy["feature_only"])
        self.assertTrue(privacy["metadata_only"])
        self.assertTrue(privacy["aggregate_only"])
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["provider_bodies_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["cache_keys_included"])
        self.assertFalse(privacy["absolute_paths_included"])
        self.assertFalse(privacy["policy_file_contents_included"])

        by_family = {row["local_action_family"]: row for row in result["rows"]}
        self.assertEqual(set(by_family), {"crunch", "routing", "cache"})
        self.assertEqual(by_family["crunch"]["executor_action_class"], "keep-current-rule")
        self.assertEqual(by_family["crunch"]["activation_outcome"], "keep-active")
        self.assertEqual(by_family["crunch"]["coverage"]["applied_count"], 107)
        self.assertEqual(by_family["crunch"]["coverage"]["holdout_count"], 40)
        self.assertEqual(by_family["crunch"]["coverage"]["safety_stop_count"], 0)
        self.assertEqual(by_family["crunch"]["coverage"]["rollback_count"], 0)
        self.assertEqual(by_family["routing"]["executor_action_class"], "keep-blocked")
        self.assertEqual(by_family["routing"]["executor_next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertEqual(by_family["routing"]["coverage"]["safety_stop_count"], 492)
        self.assertEqual(by_family["cache"]["executor_action_class"], "review-only")
        self.assertEqual(by_family["cache"]["executor_next_action"], "review-cache-replay-canary-promotion-readiness")
        self.assertEqual(by_family["cache"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertTrue(by_family["cache"]["handoff_ref"].startswith("handoff:"))
        self.assertNotIn("fingerprint", by_family["cache"])
        self.assertNotIn("request_id", by_family["cache"])
        self.assertNotIn("session_id", by_family["routing"])
        self.assertNotIn("cache_key", by_family["cache"])

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("session-routing-secret", rendered)
        self.assertNotIn("cache-review-secret", rendered)
        self.assertNotIn("/tmp/private-cache-review.py", rendered)
        self.assertNotIn("raw cache review prompt must not leak", rendered)

    def test_local_activation_executor_cli_emits_managed_handoff(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.local_activation_executor_cli(
                ["--plan-json", str(plan_path), "--managed-handoff", "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.local_activation_managed_handoff.v1")
        self.assertEqual(result["summary"]["handoff_row_count"], 3)
        self.assertEqual(result["egress_guard"]["status"], "passed")

    def test_managed_activation_preview_request_is_feature_only(self):
        result = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)

        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_request.v1")
        self.assertEqual(result["summary"]["handoff_row_count"], 3)
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["policy_files_written"])
        self.assertTrue(result["feature_only"])
        self.assertTrue(result["locally_executed"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])

    def test_managed_activation_preview_result_summarizes_review_only_decisions(self):
        request_payload = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)
        first_ref = request_payload["rows"][0]["handoff_ref"]
        second_ref = request_payload["rows"][1]["handoff_ref"]
        response_payload = {
            "schema": "agentflow.managed_activation_preview_response.v1",
            "decisions": [
                {
                    "handoff_ref": first_ref,
                    "decision": "no-op",
                    "no_op_reason": "full-rollout-policy-active",
                    "review_only": True,
                    "raw_prompt": "raw server value must not leak",
                },
                {
                    "handoff_ref": second_ref,
                    "decision": "keep-blocked",
                    "omitted_reason": "safety-stop-observed",
                    "reason_codes": ["safety-stop-observed"],
                    "review_only": True,
                },
            ],
        }

        result = build_managed_activation_preview_result(
            request_payload,
            response_payload=response_payload,
            fetch={"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        )

        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_result.v1")
        self.assertEqual(result["status"], "previewed")
        self.assertEqual(result["coverage"]["handoff_row_count"], 3)
        self.assertEqual(result["coverage"]["preview_decision_count"], 2)
        self.assertEqual(result["coverage"]["matched_handoff_ref_count"], 2)
        self.assertEqual(result["coverage"]["missing_preview_decision_count"], 1)
        self.assertEqual(result["coverage"]["no_op_count"], 2)
        self.assertEqual(result["coverage"]["omitted_count"], 1)
        self.assertEqual(result["coverage"]["review_only_count"], 2)
        self.assertEqual(result["coverage"]["active_policy_write_count"], 0)
        self.assertTrue(result["coverage"]["managed_server_calls_made"])
        self.assertFalse(result["coverage"]["provider_calls_made"])
        self.assertFalse(result["coverage"]["policy_files_written"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn('"raw_prompt":', rendered)
        self.assertNotIn("raw server value must not leak", rendered)

    def test_managed_activation_preview_cli_skips_without_url(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.managed_activation_preview_cli(["--plan-json", str(plan_path), "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_result.v1")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["fetch"]["reason"], "managed-preview-url-not-configured")
        self.assertFalse(result["coverage"]["managed_server_calls_made"])
        self.assertEqual(result["coverage"]["handoff_row_count"], 3)
        self.assertEqual(result["egress_guard"]["status"], "passed")

    def test_managed_activation_preview_cli_posts_opt_in_payload(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()
            response = Mock()
            response.status_code = 200
            response.text = "{}"

            def response_json():
                posted = post_mock.call_args.kwargs["json"]
                return {
                    "schema": "agentflow.managed_activation_preview_response.v1",
                    "decisions": [
                        {
                            "handoff_ref": posted["rows"][0]["handoff_ref"],
                            "decision": "no-op",
                            "no_op_reason": "full-rollout-policy-active",
                            "review_only": True,
                        }
                    ],
                }

            response.json.side_effect = response_json

            with patch("agentflow_proxy.cli.httpx.post", return_value=response) as post_mock:
                code = cli.managed_activation_preview_cli(
                    [
                        "--plan-json",
                        str(plan_path),
                        "--managed-preview-url",
                        "https://managed.example.test/v1/activation-preview",
                        "--pretty",
                    ],
                    stdout=stdout,
                )

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(post_mock.call_count, 1)
        posted = post_mock.call_args.kwargs["json"]
        self.assertEqual(posted["schema"], "agentflow.managed_activation_preview_request.v1")
        self.assertEqual(posted["summary"]["handoff_row_count"], 3)
        self.assertEqual(managed_egress_violations(posted), [])
        self.assertEqual(result["status"], "previewed")
        self.assertTrue(result["coverage"]["managed_server_calls_made"])
        self.assertFalse(result["coverage"]["provider_calls_made"])
        self.assertFalse(result["coverage"]["policy_files_written"])
        self.assertEqual(result["coverage"]["preview_decision_count"], 1)
        self.assertEqual(result["coverage"]["matched_handoff_ref_count"], 1)
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])

    def test_full_rollout_crunch_collapses_to_durable_keep_current_rule_outcome(self):
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry()),
            now=NOW,
        )

        self.assertEqual(result["summary"]["executor_entry_count"], 1)
        self.assertEqual(result["summary"]["selected_action_count"], 0)
        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "keep-current-rule")
        self.assertEqual(entry["executor_status"], "blocked-or-terminal")
        self.assertEqual(entry["executor_next_action"], "keep-current-rule-only")
        self.assertTrue(entry["durable_outcome_ledger_entry"])
        self.assertTrue(entry["measured_full_rollout_activation"])
        self.assertEqual(entry["full_rollout_outcome"], "keep-active")
        self.assertEqual(entry["full_rollout_successor_decision"], "no-op")
        self.assertEqual(entry["full_rollout_successor_no_op_reason"], "no-unsuppressed-post-full-rollout-crunch-cohort")
        self.assertFalse(entry["new_activation_issue_recommended"])
        self.assertEqual(entry["applied_count"], 107)
        self.assertEqual(entry["holdout_count"], 40)
        self.assertEqual(entry["safety_stop_count"], 0)
        self.assertEqual(entry["rollback_count"], 0)
        self.assertEqual(entry["observed_saved_tokens"], 8606129)
        self.assertAlmostEqual(entry["observed_savings_usd"], 25.818613)
        self.assertEqual(entry["duplicate_suppression_status"], "suppressed")
        self.assertEqual(entry["duplicate_suppression_reason"], "repeated-context-crunch-full-rollout-active")
        self.assertEqual(entry["keep_active_regression_gate"]["state"], "keep-active")
        self.assertTrue(entry["keep_active_regression_gate"]["gate_passed"])
        self.assertTrue(entry["privacy"]["metadata_only"])
        self.assertTrue(entry["privacy"]["aggregate_only"])

    def test_full_rollout_crunch_duplicate_successors_collapse_to_single_executor_outcome(self):
        entry = _full_rollout_queue_entry()
        duplicate = dict(entry)
        duplicate.pop("full_rollout_activation_outcome")
        duplicate.pop("keep_active_regression_gate")
        duplicate["fingerprint"] = "activation:duplicate-rollup"
        duplicate["lever"] = "request-shape-rollups"
        duplicate["evidence_schema"] = "agentflow.request_shape_follow_up_candidates.v1"
        plan = _full_rollout_queue_plan(entry)
        plan["entries"].append(duplicate)

        result = build_local_activation_executor_plan(plan, now=NOW)

        self.assertEqual(result["summary"]["executor_entry_count"], 1)
        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "keep-current-rule")
        self.assertEqual(entry["executor_next_action"], "keep-current-rule-only")
        self.assertEqual(entry["source_fingerprint"], "activation:f5f6eae5f0a0081a")
        self.assertEqual(entry["full_rollout_outcome"], "keep-active")
        self.assertEqual(entry["keep_active_regression_gate"]["state"], "keep-active")

    def test_full_rollout_crunch_stale_evidence_emits_review_next_action(self):
        gate = _full_rollout_gate(
            state="review-stale-evidence",
            next_action="refresh-full-rollout-repeated-context-crunch-evidence",
            reason_codes=["stale-evidence"],
        )
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry(gate=gate, outcome="review-stale-evidence")),
            now=NOW,
        )

        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "review-only")
        self.assertEqual(entry["executor_status"], "selected")
        self.assertEqual(entry["executor_next_action"], "refresh-full-rollout-repeated-context-crunch-evidence")
        self.assertEqual(entry["full_rollout_outcome"], "review-stale-evidence")
        self.assertIn("stale-evidence", entry["reason_codes"])
        self.assertFalse(entry["keep_active_regression_gate"]["gate_passed"])

    def test_full_rollout_crunch_regression_emits_rollback_next_action(self):
        gate = _full_rollout_gate(
            state="rollback-required",
            next_action="rollback-full-rollout-repeated-context-crunch-rule",
            reason_codes=["rollback-observed", "fallback-rate-regression"],
        )
        gate["regression_counters"]["rollback_count"] = 1
        gate["regression_counters"]["fallback_count"] = 1
        gate["regression_counters"]["fallback_rate_delta"] = 0.02
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry(gate=gate, outcome="rollback-required")),
            now=NOW,
        )

        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "rollback-required")
        self.assertEqual(entry["executor_status"], "blocked-or-terminal")
        self.assertEqual(entry["executor_next_action"], "rollback-full-rollout-repeated-context-crunch-rule")
        self.assertEqual(entry["full_rollout_outcome"], "rollback-required")
        self.assertEqual(entry["rollback_count"], 1)
        self.assertIn("rollback-observed", entry["reason_codes"])
        self.assertIn("fallback-rate-regression", entry["reason_codes"])
        self.assertEqual(entry["keep_active_regression_gate"]["regression_counters"]["rollback_count"], 1)


if __name__ == "__main__":
    unittest.main()
