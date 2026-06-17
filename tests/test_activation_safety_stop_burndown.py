import io
import json
from pathlib import Path
import tempfile
import unittest

from agentflow_proxy import cli
from agentflow_proxy.activation_lifecycle_feedback import (
    LIFECYCLE_SOURCE_SURFACE,
    activation_safety_stop_burndown_report,
    build_activation_safety_stop_burndown,
    build_activation_staged_lifecycle_feedback,
)
from agentflow_proxy.store import Store, stable_json, utc_now


class ActivationSafetyStopBurndownTests(unittest.TestCase):
    def _anthropic_routing_safety_stop_plan(self) -> dict:
        return {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {
                "stats_summary": {
                    "pass_through_routing_report": {
                        "schema": "agentflow.pass_through_routing_activation_candidates.v1",
                        "buckets": [
                            {
                                "rank": 1,
                                "provider": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "/v1/messages",
                                "requested_model": "claude-sonnet-4-6",
                                "routed_model": "claude-sonnet-4-6",
                                "candidate_target_model": "claude-haiku-4-5-20251001",
                                "category": "tool-result",
                                "workflow_phase": "unknown",
                                "sample_count": 1250,
                                "actionability": "actionable",
                                "required_local_executor": "anthropic-routing-rules",
                                "estimated_savings_per_1000_calls_usd": 4.5,
                                "anthropic_canary_lifecycle_evidence": {
                                    "schema": "agentflow.anthropic_routing_canary_lifecycle_evidence.v1",
                                    "status": "matched",
                                    "matched_count": 1250,
                                    "observed_count": 390,
                                    "cohort_counts": {
                                        "canary_applied": 0,
                                        "canary_holdout": 0,
                                        "safety_stopped": 390,
                                        "skipped": 0,
                                        "bypassed_or_disabled": 0,
                                        "unknown": 0,
                                    },
                                    "coverage": {
                                        "matched_count": 1250,
                                        "observed_rate": 0.312,
                                        "applied_rate": 0.0,
                                        "holdout_rate": 0.0,
                                    },
                                    "error_count": 0,
                                    "retry_count": 0,
                                    "fallback_count": 0,
                                    "latest_observed_at": "2026-06-15T20:00:15.598631+00:00",
                                    "stale_evidence": {"stale": False, "age_hours": 4.0, "max_age_hours": 72.0},
                                    "blocker_codes": [
                                        "missing-applied-coverage",
                                        "missing-holdout-coverage",
                                        "safety-stop-observed",
                                    ],
                                    "blocker_reason_breakdown": [
                                        {"value": "missing-applied-coverage", "count": 1250},
                                        {"value": "missing-holdout-coverage", "count": 1250},
                                        {"value": "safety-stop-observed", "count": 390},
                                    ],
                                    "safety_stop_breakdown": [
                                        {
                                            "category": "tool-result",
                                            "count": 390,
                                            "durable_blocked_reason": "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
                                            "endpoint": "/v1/messages",
                                            "executor_compatible": True,
                                            "expected_local_executor": "anthropic-routing-rules",
                                            "missing_applied_coverage": True,
                                            "missing_holdout_coverage": True,
                                            "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                                            "reason_code": "local-canary-safety-stop",
                                            "source_surface": "anthropic_messages",
                                            "workflow_phase": "unknown",
                                        }
                                    ],
                                    "durable_blocked_reason": "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
                                    "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                                },
                            }
                        ],
                        "privacy": {
                            "metadata_only": True,
                            "aggregate_only": True,
                            "raw_prompts_included": False,
                            "provider_bodies_included": False,
                            "request_ids_included": False,
                            "session_ids_included": False,
                        },
                    }
                }
            },
        }

    def test_anthropic_routing_safety_stop_plan_keeps_activation_blocked(self):
        report = build_activation_safety_stop_burndown(
            research_plan=self._anthropic_routing_safety_stop_plan()
        )

        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["status"], "ranked")
        self.assertEqual(report["summary"]["anthropic_routing_safety_stop_count"], 390)
        self.assertEqual(report["summary"]["top_next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertEqual(report["summary"]["top_next_state"], "keep-blocked")

        group = report["groups"][0]
        self.assertEqual(group["source"], "pass_through_routing_report")
        self.assertEqual(group["action_family"], "routing")
        self.assertEqual(group["status"], "blocked")
        self.assertEqual(group["provider"], "anthropic")
        self.assertEqual(group["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(group["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(group["source_surface"], "anthropic_messages")
        self.assertEqual(group["endpoint"], "/v1/messages")
        self.assertEqual(group["category"], "tool-result")
        self.assertTrue(group["executor_compatible"])
        self.assertEqual(group["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(group["target_local_policy_section"], "routing.rules")
        representation = group["local_file_backed_representation"]
        self.assertTrue(representation["exists"])
        self.assertEqual(representation["policy_section"], "routing")
        self.assertEqual(representation["rule_file"], "routing_rules.yaml")
        self.assertTrue(representation["metadata_only"])
        self.assertTrue(representation["aggregate_only"])
        self.assertEqual(group["safety_stop_count"], 390)
        self.assertEqual(group["matched_count"], 1250)
        self.assertEqual(group["applied_count"], 0)
        self.assertEqual(group["holdout_count"], 0)
        self.assertEqual(group["coverage"]["applied_rate"], 0.0)
        self.assertEqual(group["coverage"]["holdout_rate"], 0.0)
        self.assertFalse(group["stale_evidence"]["stale"])
        self.assertEqual(
            group["durable_blocked_reason"],
            "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
        )
        self.assertEqual(group["blocker_code"], "local-canary-safety-stop")
        self.assertEqual(group["next_action_class"], "continue-blocked")
        self.assertIn("applied_coverage", group["needed_resolution"])
        self.assertIn("holdout_coverage", group["needed_resolution"])
        self.assertIn("rollback_proof", group["needed_resolution"])
        self.assertFalse(group["promotion_allowed"])
        self.assertFalse(group["stage_allowed"])
        self.assertFalse(group["active_policy_changed"])
        self.assertFalse(group["wrote_active_policy_files"])
        self.assertEqual(group["burndown_status"], "safety-stop-active")
        self.assertEqual(group["safety_stop_breakdown"][0]["count"], 390)
        self.assertTrue(group["safety_stop_breakdown"][0]["missing_applied_coverage"])
        self.assertTrue(group["safety_stop_breakdown"][0]["missing_holdout_coverage"])
        unblock = group["unblock_criteria"]
        self.assertEqual(unblock["schema"], "agentflow.anthropic_routing_safety_stop_unblock_criteria.v1")
        self.assertEqual(unblock["status"], "blocked")
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertFalse(unblock["applied_coverage_present"])
        self.assertFalse(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["safer_threshold_or_executor_guard_present"])
        self.assertFalse(unblock["rollback_proof_present"])
        self.assertFalse(unblock["promotion_allowed"])
        self.assertFalse(unblock["stage_allowed"])
        self.assertIn("safety_stop_reason_review", unblock["needed_resolution"])
        self.assertIn("safer_threshold_or_executor_guard", unblock["needed_resolution"])
        self.assertIn("rollback_proof", unblock["needed_resolution"])
        self.assertIn("applied_coverage", unblock["needed_resolution"])
        self.assertIn("holdout_coverage", unblock["needed_resolution"])
        self.assertEqual(
            unblock["suppresses_ready_issue_until"],
            "safety_stop_count_zero_and_applied_holdout_coverage_present",
        )
        self.assertTrue(unblock["metadata_only"])
        self.assertTrue(unblock["aggregate_only"])
        duplicate_suppression = group["duplicate_suppression"]
        self.assertEqual(
            duplicate_suppression["schema"],
            "agentflow.anthropic_routing_activation_issue_duplicate_suppression.v1",
        )
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertEqual(duplicate_suppression["safety_stop_count"], 390)
        self.assertTrue(duplicate_suppression["missing_applied_coverage"])
        self.assertTrue(duplicate_suppression["missing_holdout_coverage"])
        self.assertEqual(
            duplicate_suppression["reason"],
            "anthropic-routing-safety-stop-burndown-not-cleared",
        )
        self.assertEqual(
            duplicate_suppression["suppresses_ready_issue_until"],
            "safety_stop_count_zero_and_applied_holdout_coverage_present",
        )
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])

    def test_lifecycle_safety_stop_groups_have_specific_next_action(self):
        result = {
            "schema": "fixture.activation.apply.v1",
            "ok": False,
            "summary": {"projected_savings_usd": 0.04},
            "actions": [
                {
                    "action_family": "routing",
                    "policy_section": "routing",
                    "status": "safety_stopped",
                    "target_candidate_id": "raw-candidate-secret",
                    "target_rule_id": "raw-rule-secret",
                    "projected_savings_usd": 0.04,
                    "reason_codes": ["local-canary-safety-stop", "error-rate-regression"],
                }
            ],
        }
        payload = build_activation_staged_lifecycle_feedback(
            result,
            event_phase="apply",
            command="activation-apply-fixture",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["family_events"][0]["cohort"], "safety_stopped")

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="activation-safety-stop",
                    created_at=now,
                    updated_at=now,
                    source_surface=LIFECYCLE_SOURCE_SURFACE,
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                report = activation_safety_stop_burndown_report(store, limit=50)
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        group = report["groups"][0]
        self.assertEqual(group["action_family"], "routing")
        self.assertEqual(group["blocker_code"], "error-rate-regression")
        self.assertEqual(group["keep_blocked_reason"], "routing-safety-stop-needs-rollback-proof")
        self.assertEqual(group["next_state"], "keep-blocked")
        self.assertEqual(group["next_state_reason"], "safety-stop-requires-safer-threshold-or-rollback-proof")
        self.assertIn("rollback_proof", group["needed_resolution"])
        self.assertEqual(group["next_action"], "record-routing-rollback-proof-before-reactivation")
        self.assertEqual(report["summary"]["top_next_state"], "keep-blocked")
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-candidate-secret", rendered)
        self.assertNotIn("raw-rule-secret", rendered)

    def test_repeated_research_diagnostic_resolves_to_action_without_raw_examples(self):
        plan = {
            "schema": "agentflow.orchestrator_research_plan.v1",
            "evidence": {
                "repeated_diagnostics": [
                    {
                        "reason": "safety-stop",
                        "diagnostic_class": "safety-stop",
                        "count": 8,
                        "example": "routing blocker=safety-stop request_id=req-secret path=/tmp/raw.py session_id=session-secret",
                    }
                ]
            },
        }

        report = build_activation_safety_stop_burndown(research_plan=plan)

        self.assertEqual(report["status"], "ranked")
        self.assertEqual(report["summary"]["top_next_action"], "review-activation-feedback-safety-stop-and-record-keep-blocked-reason")
        self.assertEqual(
            report["summary"]["top_keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertEqual(report["groups"][0]["repeated_noop_status"], "repeated")
        self.assertEqual(report["groups"][0]["next_state"], "keep-blocked")
        self.assertEqual(
            report["groups"][0]["next_state_reason"],
            "safety-stop-requires-safer-threshold-or-rollback-proof",
        )
        self.assertEqual(
            report["groups"][0]["keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertIn("human_review", report["groups"][0]["needed_resolution"])
        unblock = report["groups"][0]["unblock_criteria"]
        self.assertEqual(unblock["schema"], "agentflow.activation_feedback_safety_stop_unblock_criteria.v1")
        self.assertEqual(unblock["status"], "blocked")
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertFalse(unblock["applied_coverage_present"])
        self.assertFalse(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["safer_threshold_or_executor_guard_present"])
        self.assertFalse(unblock["rollback_proof_present"])
        self.assertIn("human_review", unblock["needed_resolution"])
        self.assertIn("safer_threshold", unblock["needed_resolution"])
        self.assertIn("rollback_proof", unblock["needed_resolution"])
        self.assertEqual(
            unblock["suppresses_ready_issue_until"],
            "safety_stop_count_zero_and_applied_holdout_coverage_present",
        )
        self.assertTrue(unblock["metadata_only"])
        self.assertTrue(unblock["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("req-secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("/tmp/raw.py", rendered)

    def test_lifecycle_rows_classify_retry_later_and_superseded_next_states(self):
        report = build_activation_safety_stop_burndown(
            {
                "schema": "agentflow.activation_staged_lifecycle_feedback_summary.v1",
                "cohort_lifecycle_metadata": [
                    {
                        "policy_ref": "policy:stale",
                        "cohort_label": "safety_stopped",
                        "action_family": "cache_replay",
                        "event_count": 1,
                        "safety_stop_count": 1,
                        "applied_count": 1,
                        "holdout_count": 0,
                        "reason_codes": ["stale-lifecycle-evidence"],
                    },
                    {
                        "policy_ref": "policy:superseded",
                        "cohort_label": "applied",
                        "action_family": "crunch",
                        "event_count": 3,
                        "safety_stop_count": 0,
                        "applied_count": 2,
                        "holdout_count": 1,
                        "reason_codes": ["safety-stop-observed"],
                    },
                ],
            }
        )

        by_policy = {group["policy_ref"]: group for group in report["groups"]}
        self.assertEqual(by_policy["policy:stale"]["next_state"], "retry-later")
        self.assertEqual(
            by_policy["policy:stale"]["next_state_reason"],
            "safety-stop-awaits-fresh-lifecycle-or-holdout-evidence",
        )
        self.assertEqual(by_policy["policy:superseded"]["next_state"], "superseded")
        self.assertEqual(
            by_policy["policy:superseded"]["next_state_reason"],
            "safety-stop-no-longer-dominates-current-lifecycle-evidence",
        )

    def test_cli_reads_plan_json_and_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "agentflow.sqlite3"
            store = Store(str(db_path))
            store.conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema": "agentflow.orchestrator_research_plan.v1",
                        "evidence": {
                            "repeated_diagnostics": [
                                {"reason": "safety-stop", "diagnostic_class": "safety-stop", "count": 3}
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            code = cli.activation_safety_stop_burndown_cli(
                ["--db", str(db_path), "--plan-json", str(plan_path), "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["summary"]["safety_stop_count"], 3)
        self.assertEqual(report["summary"]["top_next_action"], "review-activation-feedback-safety-stop-and-record-keep-blocked-reason")

    def test_cli_reads_anthropic_routing_safety_stop_from_plan_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "agentflow.sqlite3"
            store = Store(str(db_path))
            store.conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(self._anthropic_routing_safety_stop_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.activation_safety_stop_burndown_cli(
                ["--db", str(db_path), "--plan-json", str(plan_path), "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(report["summary"]["anthropic_routing_safety_stop_count"], 390)
        self.assertEqual(report["summary"]["top_next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertFalse(report["groups"][0]["promotion_allowed"])
        self.assertFalse(report["groups"][0]["stage_allowed"])


if __name__ == "__main__":
    unittest.main()
