import io
import json
from pathlib import Path
import tempfile
import unittest

from tokenclaw import cli
from tokenclaw.anthropic_routing_unblock_drill import (
    build_anthropic_routing_safety_stop_unblock_drill,
)
from tokenclaw.store import Store


def _anthropic_routing_plan() -> dict:
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
                            "workflow_phase": "tool-execution",
                            "sample_count": 492,
                            "actionability": "actionable",
                            "required_local_executor": "anthropic-routing-rules",
                            "estimated_savings_per_1000_calls_usd": 4.5,
                            "anthropic_canary_lifecycle_evidence": {
                                "schema": "agentflow.anthropic_routing_canary_lifecycle_evidence.v1",
                                "status": "matched",
                                "matched_count": 492,
                                "observed_count": 492,
                                "cohort_counts": {
                                    "canary_applied": 0,
                                    "canary_holdout": 0,
                                    "safety_stopped": 492,
                                    "skipped": 0,
                                    "bypassed_or_disabled": 0,
                                    "unknown": 0,
                                },
                                "coverage": {
                                    "matched_count": 492,
                                    "observed_rate": 1.0,
                                    "applied_rate": 0.0,
                                    "holdout_rate": 0.0,
                                },
                                "error_count": 0,
                                "retry_count": 0,
                                "fallback_count": 0,
                                "stale_evidence": {"stale": False, "age_hours": 1.0, "max_age_hours": 72.0},
                                "blocker_codes": [
                                    "missing-applied-coverage",
                                    "missing-holdout-coverage",
                                    "safety-stop-observed",
                                ],
                                "blocker_reason_breakdown": [
                                    {"value": "missing-applied-coverage", "count": 492},
                                    {"value": "missing-holdout-coverage", "count": 492},
                                    {"value": "safety-stop-observed", "count": 492},
                                ],
                                "safety_stop_breakdown": [
                                    {
                                        "reason_code": "local-canary-safety-stop",
                                        "count": 492,
                                        "source_surface": "anthropic_messages",
                                        "endpoint": "/v1/messages",
                                        "category": "tool-result",
                                        "workflow_phase": "tool-execution",
                                        "expected_local_executor": "anthropic-routing-rules",
                                        "executor_compatible": True,
                                        "missing_applied_coverage": True,
                                        "missing_holdout_coverage": True,
                                        "durable_blocked_reason": "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
                                        "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                                    }
                                ],
                                "durable_blocked_reason": "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
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


class AnthropicRoutingUnblockDrillTests(unittest.TestCase):
    def test_blocked_safety_stop_drill_records_rollback_proof_without_activation(self) -> None:
        result = build_anthropic_routing_safety_stop_unblock_drill(_anthropic_routing_plan())

        self.assertEqual(result["schema"], "agentflow.anthropic_routing_safety_stop_unblock_drill.v1")
        self.assertEqual(result["status"], "ranked")
        self.assertEqual(result["summary"]["drill_entry_count"], 1)
        self.assertEqual(result["summary"]["blocked_count"], 1)
        self.assertEqual(result["summary"]["stage_ready_count"], 0)
        self.assertEqual(result["summary"]["promotion_allowed_count"], 0)
        self.assertEqual(result["summary"]["safety_stop_count"], 492)
        self.assertEqual(result["acceptance"]["status"], "met")
        self.assertTrue(result["acceptance"]["promotion_never_allowed_by_drill"])
        self.assertTrue(result["acceptance"]["no_active_policy_write"])

        entry = result["entries"][0]
        self.assertEqual(entry["status"], "keep-blocked")
        self.assertFalse(entry["stage_allowed"])
        self.assertFalse(entry["promotion_allowed"])
        self.assertEqual(entry["safety_stop_count"], 492)
        self.assertEqual(entry["applied_count"], 0)
        self.assertEqual(entry["holdout_count"], 0)
        for field in (
            "safety_stop_reason_review",
            "safer_threshold_or_executor_guard",
            "rollback_proof",
            "applied_coverage",
            "holdout_coverage",
        ):
            self.assertFalse(entry["criteria_passed"][field])
            self.assertEqual(entry["criterion_results"][field]["status"], "failed")
        rollback = entry["rollback_metadata"]
        self.assertEqual(rollback["rollback_action_type"], "keep_anthropic_routing_policy_disabled")
        self.assertEqual(rollback["disabled_policy_state"], "anthropic-routing-canary-disabled")
        self.assertEqual(rollback["target_local_policy_section"], "routing.rules")
        self.assertEqual(rollback["target_local_rule_file"], "routing_rules.yaml")
        self.assertFalse(rollback["active_policy_changed"])
        self.assertFalse(rollback["wrote_active_policy_files"])
        self.assertFalse(rollback["policy_file_contents_included"])
        self.assertFalse(entry["active_policy_changed"])
        self.assertFalse(entry["wrote_active_policy_files"])

    def test_partial_unblock_stays_blocked_until_safety_stop_and_guard_clear(self) -> None:
        plan = _anthropic_routing_plan()
        lifecycle = plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0][
            "anthropic_canary_lifecycle_evidence"
        ]
        lifecycle["cohort_counts"]["canary_applied"] = 9
        lifecycle["cohort_counts"]["canary_holdout"] = 6
        lifecycle["coverage"]["applied_rate"] = 0.018293
        lifecycle["coverage"]["holdout_rate"] = 0.012195
        lifecycle["blocker_codes"] = ["safety-stop-observed"]
        for row in lifecycle["safety_stop_breakdown"]:
            row["missing_applied_coverage"] = False
            row["missing_holdout_coverage"] = False

        result = build_anthropic_routing_safety_stop_unblock_drill(plan)

        entry = result["entries"][0]
        self.assertEqual(entry["status"], "keep-blocked")
        self.assertFalse(entry["stage_allowed"])
        self.assertFalse(entry["promotion_allowed"])
        self.assertEqual(entry["safety_stop_count"], 492)
        self.assertTrue(entry["criteria_passed"]["applied_coverage"])
        self.assertTrue(entry["criteria_passed"]["holdout_coverage"])
        self.assertFalse(entry["criteria_passed"]["safety_stop_reason_review"])
        self.assertFalse(entry["criteria_passed"]["safer_threshold_or_executor_guard"])
        self.assertFalse(entry["criteria_passed"]["rollback_proof"])
        self.assertEqual(result["summary"]["stage_ready_count"], 0)

    def test_cleared_fixture_is_recovery_ready_but_still_review_only_and_no_promotion(self) -> None:
        plan = _anthropic_routing_plan()
        lifecycle = plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0][
            "anthropic_canary_lifecycle_evidence"
        ]
        lifecycle["observed_count"] = 15
        lifecycle["cohort_counts"]["canary_applied"] = 8
        lifecycle["cohort_counts"]["canary_holdout"] = 7
        lifecycle["cohort_counts"]["safety_stopped"] = 0
        lifecycle["coverage"]["observed_rate"] = 1.0
        lifecycle["coverage"]["applied_rate"] = 0.533333
        lifecycle["coverage"]["holdout_rate"] = 0.466667
        lifecycle["blocker_codes"] = []
        lifecycle["blocker_reason_breakdown"] = []
        lifecycle["safety_stop_breakdown"] = []
        lifecycle["durable_blocked_reason"] = None

        result = build_anthropic_routing_safety_stop_unblock_drill(plan)

        entry = result["entries"][0]
        self.assertEqual(entry["status"], "recovery-ready")
        self.assertTrue(entry["stage_allowed"])
        self.assertFalse(entry["promotion_allowed"])
        self.assertTrue(entry["review_only"])
        self.assertTrue(entry["dry_run_only"])
        self.assertEqual(entry["safety_stop_count"], 0)
        self.assertTrue(all(entry["criteria_passed"].values()))
        self.assertEqual(result["summary"]["stage_ready_count"], 1)
        self.assertEqual(result["summary"]["promotion_allowed_count"], 0)
        self.assertTrue(result["acceptance"]["stage_ready_rows_require_all_criteria"])
        self.assertTrue(result["acceptance"]["promotion_never_allowed_by_drill"])
        self.assertFalse(entry["active_policy_changed"])
        self.assertFalse(entry["wrote_active_policy_files"])
        self.assertFalse(entry["provider_calls_made"])
        self.assertFalse(entry["managed_server_calls_made"])

    def test_cli_reads_plan_json_and_emits_drill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "agentflow.sqlite3"
            store = Store(str(db_path))
            store.conn.close()
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(_anthropic_routing_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.anthropic_routing_safety_stop_unblock_drill_cli(
                ["--db", str(db_path), "--plan-json", str(plan_path), "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.anthropic_routing_safety_stop_unblock_drill.v1")
        self.assertEqual(result["summary"]["safety_stop_count"], 492)
        self.assertFalse(result["entries"][0]["stage_allowed"])
        self.assertFalse(result["entries"][0]["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
