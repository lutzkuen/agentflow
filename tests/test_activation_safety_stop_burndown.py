import io
import json
from pathlib import Path
import tempfile
import unittest

from tokenclaw import cli
from tokenclaw.activation_lifecycle_feedback import (
    LIFECYCLE_SOURCE_SURFACE,
    activation_safety_stop_burndown_report,
    build_activation_safety_stop_burndown,
    build_activation_staged_lifecycle_feedback,
)
from tokenclaw.orchestrator_research import build_local_activation_next_action_queue
from tokenclaw.store import Store, stable_json, utc_now


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
                                "source_surface": "unknown",
                                "endpoint": "unknown",
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
        self.assertEqual(report["summary"]["anthropic_routing_refresh_proof_count"], 1)
        self.assertTrue(report["summary"]["anthropic_routing_refresh_proof_fields_recorded"])
        self.assertFalse(report["summary"]["anthropic_routing_policy_files_written"])
        self.assertFalse(report["summary"]["anthropic_routing_active_policy_changed"])
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
        self.assertEqual(group["evidence_freshness_status"], "fresh")
        self.assertEqual(group["evidence_freshness"]["status"], "fresh")
        self.assertFalse(group["evidence_freshness"]["stale"])
        self.assertEqual(
            group["durable_blocked_reason"],
            "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
        )
        self.assertEqual(
            group["keep_blocked_reason"],
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
        refresh_proof = group["burndown_refresh_proof"]
        self.assertEqual(
            refresh_proof["schema"],
            "agentflow.anthropic_routing_safety_stop_burndown_refresh_proof.v1",
        )
        self.assertEqual(refresh_proof["status"], "blocked")
        self.assertTrue(refresh_proof["all_required_fields_recorded"])
        self.assertTrue(all(refresh_proof["required_field_results"].values()))
        self.assertEqual(refresh_proof["safety_stop_count"], 390)
        self.assertEqual(refresh_proof["applied_count"], 0)
        self.assertEqual(refresh_proof["holdout_count"], 0)
        self.assertEqual(refresh_proof["evidence_freshness_status"], "fresh")
        self.assertEqual(refresh_proof["evidence_age_hours"], 4.0)
        self.assertFalse(refresh_proof["promotion_allowed"])
        self.assertFalse(refresh_proof["stage_allowed"])
        self.assertTrue(refresh_proof["keeps_policy_blocked"])
        self.assertFalse(refresh_proof["active_policy_changed"])
        self.assertFalse(refresh_proof["wrote_active_policy_files"])
        self.assertTrue(refresh_proof["metadata_only"])
        self.assertTrue(refresh_proof["aggregate_only"])
        self.assertEqual(group["burndown_status"], "safety-stop-active")
        self.assertEqual(group["safety_stop_breakdown"][0]["count"], 390)
        self.assertTrue(group["safety_stop_breakdown"][0]["missing_applied_coverage"])
        self.assertTrue(group["safety_stop_breakdown"][0]["missing_holdout_coverage"])
        self.assertEqual(group["safety_stop_reason_review"]["status"], "missing")
        self.assertFalse(group["safety_stop_reason_review"]["present"])
        self.assertFalse(group["safety_stop_reason_review"]["passed"])
        self.assertEqual(group["safety_stop_reason_review"]["safety_stop_count"], 390)
        self.assertEqual(group["safer_threshold_or_executor_guard"]["status"], "missing")
        self.assertFalse(group["safer_threshold_or_executor_guard"]["present"])
        self.assertFalse(group["safer_threshold_or_executor_guard"]["passed"])
        self.assertTrue(group["safer_threshold_or_executor_guard"]["executor_compatible"])
        self.assertEqual(
            group["safer_threshold_or_executor_guard"]["required_local_executor"],
            "anthropic-routing-rules",
        )
        self.assertEqual(group["rollback_proof"]["status"], "missing")
        self.assertFalse(group["rollback_proof"]["passed"])
        self.assertEqual(group["rollback_proof"]["rollback_action_type"], "keep_anthropic_routing_policy_disabled")
        self.assertEqual(group["rollback_proof"]["target_local_policy_section"], "routing.rules")
        self.assertEqual(group["rollback_proof"]["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(group["rollback_proof"]["disabled_policy_state"], "anthropic-routing-canary-disabled")
        self.assertEqual(group["rollback_proof"]["keep_disabled_action"], "do-not-stage-or-widen-until-unblock-criteria-pass")
        self.assertFalse(group["rollback_proof"]["active_policy_changed"])
        self.assertFalse(group["rollback_proof"]["wrote_active_policy_files"])
        rollback = group["rollback_metadata"]
        self.assertEqual(rollback["schema"], "agentflow.anthropic_routing_safety_stop_rollback_metadata.v1")
        self.assertEqual(rollback["rollback_action_type"], "keep_anthropic_routing_policy_disabled")
        self.assertEqual(rollback["target_local_policy_section"], "routing.rules")
        self.assertEqual(rollback["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(rollback["disabled_policy_state"], "anthropic-routing-canary-disabled")
        self.assertEqual(rollback["keep_disabled_action"], "do-not-stage-or-widen-until-unblock-criteria-pass")
        self.assertFalse(rollback["active_policy_changed"])
        self.assertFalse(rollback["wrote_active_policy_files"])
        self.assertFalse(rollback["promotion_allowed"])
        self.assertFalse(rollback["stage_allowed"])
        self.assertFalse(rollback["policy_file_contents_included"])
        self.assertTrue(rollback["metadata_only"])
        self.assertTrue(rollback["aggregate_only"])
        self.assertEqual(group["applied_coverage"]["status"], "missing")
        self.assertFalse(group["applied_coverage"]["passed"])
        self.assertEqual(group["applied_coverage"]["applied_count"], 0)
        self.assertEqual(group["holdout_coverage"]["status"], "missing")
        self.assertFalse(group["holdout_coverage"]["passed"])
        self.assertEqual(group["holdout_coverage"]["holdout_count"], 0)
        unblock = group["unblock_criteria"]
        self.assertEqual(unblock["schema"], "agentflow.anthropic_routing_safety_stop_unblock_criteria.v1")
        self.assertEqual(unblock["status"], "blocked")
        self.assertEqual(
            unblock["required_resolution_fields"],
            [
                "safety_stop_reason_review",
                "safer_threshold_or_executor_guard",
                "rollback_proof",
                "applied_coverage",
                "holdout_coverage",
            ],
        )
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertFalse(unblock["applied_coverage_present"])
        self.assertFalse(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["safer_threshold_or_executor_guard_present"])
        self.assertFalse(unblock["rollback_proof_present"])
        self.assertFalse(unblock["promotion_allowed"])
        self.assertFalse(unblock["stage_allowed"])
        for field in unblock["required_resolution_fields"]:
            self.assertIn(field, unblock["criterion_results"])
            self.assertFalse(unblock["criterion_results"][field]["passed"])
            self.assertEqual(unblock["criterion_results"][field]["status"], "failed")
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

        queue = build_local_activation_next_action_queue({"activation_safety_stop_burndown": report})
        self.assertIsNotNone(queue)
        queue_row = queue["entries"][0]
        self.assertEqual(queue_row["evidence_schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(queue_row["local_action_family"], "routing")
        self.assertEqual(queue_row["current_status"], "keep-blocked")
        self.assertEqual(queue_row["next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertEqual(queue_row["safety_stop_count"], 390)
        self.assertEqual(queue_row["applied_count"], 0)
        self.assertEqual(queue_row["holdout_count"], 0)
        self.assertFalse(queue_row["promotion_allowed"])
        self.assertFalse(queue_row["stage_allowed"])
        self.assertFalse(queue_row["active_policy_changed"])
        self.assertFalse(queue_row["wrote_active_policy_files"])
        self.assertEqual(queue_row["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(queue_row["target_local_policy_section"], "routing.rules")
        self.assertEqual(queue_row["required_local_executor"], "anthropic-routing-rules")
        self.assertEqual(queue_row["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(queue_row["unblock_criteria"]["status"], "blocked")
        for field in queue_row["unblock_criteria"]["required_resolution_fields"]:
            self.assertFalse(queue_row["unblock_criteria"]["criterion_results"][field]["passed"])
        self.assertEqual(queue_row["rollback_metadata"]["keep_disabled_action"], "do-not-stage-or-widen-until-unblock-criteria-pass")
        self.assertFalse(queue_row["rollback_metadata"]["policy_file_contents_included"])
        self.assertEqual(queue_row["duplicate_suppression_status"], "suppressed")
        self.assertTrue(queue_row["duplicate_suppression"]["suppresses_new_activation_issue"])

    def test_anthropic_routing_cleared_safety_stop_is_queue_stage_eligible_only_after_all_criteria(self):
        plan = self._anthropic_routing_safety_stop_plan()
        bucket = plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0]
        lifecycle = bucket["anthropic_canary_lifecycle_evidence"]
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
        lifecycle["next_action"] = None

        report = build_activation_safety_stop_burndown(research_plan=plan)

        group = report["groups"][0]
        self.assertEqual(group["next_state"], "recovery-ready")
        self.assertEqual(group["status"], "recovery-ready")
        self.assertEqual(group["burndown_status"], "recovery-ready")
        self.assertEqual(group["safety_stop_count"], 0)
        self.assertEqual(group["applied_count"], 8)
        self.assertEqual(group["holdout_count"], 7)
        self.assertEqual(report["summary"]["anthropic_routing_refresh_proof_count"], 1)
        self.assertTrue(report["summary"]["anthropic_routing_refresh_proof_fields_recorded"])
        self.assertFalse(report["summary"]["anthropic_routing_policy_files_written"])
        self.assertFalse(report["summary"]["anthropic_routing_active_policy_changed"])
        self.assertTrue(group["executor_compatible"])
        self.assertTrue(group["promotion_allowed"])
        self.assertTrue(group["stage_allowed"])
        refresh_proof = group["burndown_refresh_proof"]
        self.assertEqual(refresh_proof["status"], "recovery-ready")
        self.assertTrue(refresh_proof["all_required_fields_recorded"])
        self.assertTrue(all(refresh_proof["required_field_results"].values()))
        self.assertEqual(refresh_proof["safety_stop_count"], 0)
        self.assertEqual(refresh_proof["applied_count"], 8)
        self.assertEqual(refresh_proof["holdout_count"], 7)
        self.assertEqual(refresh_proof["evidence_freshness_status"], "fresh")
        self.assertFalse(refresh_proof["keeps_policy_blocked"])
        self.assertFalse(refresh_proof["active_policy_changed"])
        self.assertFalse(refresh_proof["wrote_active_policy_files"])
        self.assertTrue(group["safety_stop_reason_review"]["passed"])
        self.assertTrue(group["safer_threshold_or_executor_guard"]["passed"])
        self.assertTrue(group["rollback_proof"]["passed"])
        self.assertTrue(group["applied_coverage"]["passed"])
        self.assertTrue(group["holdout_coverage"]["passed"])
        unblock = group["unblock_criteria"]
        self.assertEqual(unblock["status"], "recovery-ready")
        self.assertTrue(unblock["safety_stop_count_zero"])
        self.assertTrue(unblock["applied_coverage_present"])
        self.assertTrue(unblock["holdout_coverage_present"])
        self.assertTrue(unblock["safer_threshold_or_executor_guard_present"])
        self.assertTrue(unblock["rollback_proof_present"])
        self.assertTrue(unblock["promotion_allowed"])
        self.assertTrue(unblock["stage_allowed"])

        queue = build_local_activation_next_action_queue(report)
        self.assertIsNotNone(queue)
        queue_row = queue["entries"][0]
        self.assertEqual(queue_row["local_action_family"], "routing")
        self.assertEqual(queue_row["state"], "recovery-ready")
        self.assertEqual(queue_row["current_status"], "staged")
        self.assertEqual(queue_row["next_action"], "mark-anthropic-routing-recovery-ready")
        self.assertEqual(queue_row["safety_stop_count"], 0)
        self.assertEqual(queue_row["applied_count"], 8)
        self.assertEqual(queue_row["holdout_count"], 7)
        self.assertTrue(queue_row["promotion_allowed"])
        self.assertTrue(queue_row["stage_allowed"])
        self.assertEqual(queue_row["unblock_criteria"]["status"], "recovery-ready")
        self.assertTrue(queue_row["safer_threshold_or_executor_guard"]["passed"])
        self.assertTrue(queue_row["rollback_proof"]["passed"])
        self.assertFalse(queue_row["duplicate_suppression"]["suppresses_new_activation_issue"])

    def test_anthropic_routing_safety_stop_stays_blocked_even_with_applied_holdout_coverage(self):
        plan = self._anthropic_routing_safety_stop_plan()
        lifecycle = plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0][
            "anthropic_canary_lifecycle_evidence"
        ]
        lifecycle["cohort_counts"]["canary_applied"] = 8
        lifecycle["cohort_counts"]["canary_holdout"] = 7
        lifecycle["coverage"]["applied_rate"] = 0.0064
        lifecycle["coverage"]["holdout_rate"] = 0.0056
        lifecycle["blocker_codes"] = ["safety-stop-observed"]
        lifecycle["blocker_reason_breakdown"] = [{"value": "safety-stop-observed", "count": 390}]
        for row in lifecycle["safety_stop_breakdown"]:
            row["missing_applied_coverage"] = False
            row["missing_holdout_coverage"] = False

        report = build_activation_safety_stop_burndown(research_plan=plan)

        group = report["groups"][0]
        self.assertEqual(group["next_state"], "keep-blocked")
        self.assertEqual(group["burndown_status"], "safety-stop-active")
        self.assertFalse(group["promotion_allowed"])
        self.assertFalse(group["stage_allowed"])
        self.assertFalse(group["active_policy_changed"])
        self.assertFalse(group["wrote_active_policy_files"])
        self.assertEqual(group["applied_coverage"]["status"], "present")
        self.assertTrue(group["applied_coverage"]["passed"])
        self.assertEqual(group["holdout_coverage"]["status"], "present")
        self.assertTrue(group["holdout_coverage"]["passed"])
        unblock = group["unblock_criteria"]
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertTrue(unblock["applied_coverage_present"])
        self.assertTrue(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["promotion_allowed"])
        self.assertFalse(unblock["stage_allowed"])
        self.assertFalse(unblock["criterion_results"]["safety_stop_reason_review"]["passed"])
        self.assertTrue(unblock["criterion_results"]["applied_coverage"]["passed"])
        self.assertTrue(unblock["criterion_results"]["holdout_coverage"]["passed"])
        rollback = group["rollback_metadata"]
        self.assertEqual(rollback["keep_disabled_action"], "do-not-stage-or-widen-until-unblock-criteria-pass")
        self.assertFalse(rollback["active_policy_changed"])
        self.assertFalse(rollback["wrote_active_policy_files"])

    def test_anthropic_routing_stale_or_missing_coverage_refreshes_before_recovery_ready(self):
        stale_plan = self._anthropic_routing_safety_stop_plan()
        stale_lifecycle = stale_plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0][
            "anthropic_canary_lifecycle_evidence"
        ]
        stale_lifecycle["cohort_counts"]["safety_stopped"] = 0
        stale_lifecycle["cohort_counts"]["canary_applied"] = 8
        stale_lifecycle["cohort_counts"]["canary_holdout"] = 7
        stale_lifecycle["coverage"]["applied_rate"] = 0.533333
        stale_lifecycle["coverage"]["holdout_rate"] = 0.466667
        stale_lifecycle["blocker_codes"] = []
        stale_lifecycle["blocker_reason_breakdown"] = []
        stale_lifecycle["safety_stop_breakdown"] = []
        stale_lifecycle["stale_evidence"] = {"stale": True, "age_hours": 73.25, "max_age_hours": 72.0}

        stale_report = build_activation_safety_stop_burndown(research_plan=stale_plan)
        stale_group = stale_report["groups"][0]
        self.assertEqual(stale_group["next_state"], "keep-blocked")
        self.assertEqual(stale_group["burndown_status"], "stale-evidence")
        self.assertEqual(stale_group["next_action"], "refresh-anthropic-routing-safety-stop-burndown")
        self.assertEqual(stale_group["evidence_freshness_status"], "stale")
        self.assertEqual(stale_group["evidence_age_hours"], 73.25)
        self.assertEqual(stale_report["summary"]["anthropic_routing_refresh_proof_count"], 1)
        self.assertTrue(stale_report["summary"]["anthropic_routing_refresh_proof_fields_recorded"])
        self.assertFalse(stale_report["summary"]["anthropic_routing_policy_files_written"])
        self.assertFalse(stale_report["summary"]["anthropic_routing_active_policy_changed"])
        stale_proof = stale_group["burndown_refresh_proof"]
        self.assertEqual(stale_proof["status"], "blocked")
        self.assertTrue(stale_proof["all_required_fields_recorded"])
        self.assertEqual(stale_proof["evidence_freshness_status"], "stale")
        self.assertEqual(stale_proof["evidence_age_hours"], 73.25)
        self.assertFalse(stale_proof["promotion_allowed"])
        self.assertFalse(stale_proof["stage_allowed"])
        self.assertTrue(stale_proof["keeps_policy_blocked"])
        self.assertFalse(stale_proof["active_policy_changed"])
        self.assertFalse(stale_proof["wrote_active_policy_files"])
        self.assertFalse(stale_group["promotion_allowed"])
        self.assertFalse(stale_group["stage_allowed"])
        self.assertTrue(stale_group["duplicate_suppression"]["suppresses_new_activation_issue"])

        stale_queue = build_local_activation_next_action_queue(stale_report)
        self.assertIsNotNone(stale_queue)
        stale_queue_row = stale_queue["entries"][0]
        self.assertEqual(stale_queue_row["current_status"], "keep-blocked")
        self.assertEqual(stale_queue_row["evidence_freshness_status"], "stale")
        self.assertTrue(stale_queue_row["duplicate_suppression"]["suppresses_new_activation_issue"])
        self.assertFalse(stale_queue_row["active_policy_changed"])
        self.assertFalse(stale_queue_row["wrote_active_policy_files"])

        missing_plan = self._anthropic_routing_safety_stop_plan()
        missing_lifecycle = missing_plan["evidence"]["stats_summary"]["pass_through_routing_report"]["buckets"][0][
            "anthropic_canary_lifecycle_evidence"
        ]
        missing_lifecycle["cohort_counts"]["safety_stopped"] = 0
        missing_lifecycle["blocker_codes"] = ["missing-applied-coverage", "missing-holdout-coverage"]
        missing_lifecycle["safety_stop_breakdown"] = []

        missing_report = build_activation_safety_stop_burndown(research_plan=missing_plan)
        missing_group = missing_report["groups"][0]
        self.assertEqual(missing_group["next_state"], "keep-blocked")
        self.assertEqual(missing_group["burndown_status"], "missing-coverage")
        self.assertEqual(missing_group["unblock_criteria"]["status"], "blocked")
        self.assertFalse(missing_group["promotion_allowed"])
        self.assertFalse(missing_group["stage_allowed"])
        self.assertTrue(missing_group["duplicate_suppression"]["suppresses_new_activation_issue"])

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
