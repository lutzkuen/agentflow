import io
import json
import unittest

from tokenclaw import cli
from tokenclaw.anthropic_routing_canary_stage import build_anthropic_routing_canary_stage_report


def _pass_through_report() -> dict:
    return {
        "schema": "agentflow.pass_through_routing_activation_candidates.v1",
        "generated_at": "2026-06-14T19:30:00+00:00",
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
                "sample_count": 1102,
                "actionability": "actionable",
                "required_local_executor": "anthropic-routing-rules",
                "estimated_savings_per_1000_calls_usd": 4.5,
                "candidate_reason": "phase/category metadata matches existing local Haiku executor shapes",
            },
            {
                "rank": 2,
                "provider": "anthropic",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "category": "tool-result",
                "sample_count": 32,
                "actionability": "needs-lifecycle-evidence",
                "no_op_reason": "Anthropic aggregate bucket needs phase or thinking/tool safety evidence before downgrade",
                "required_local_executor": "anthropic-routing-rules",
                "estimated_savings_per_1000_calls_usd": 4.5,
            },
            {
                "rank": 3,
                "provider": "anthropic",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": None,
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "category": "tool-result",
                "sample_count": 38,
                "actionability": "unsupported-provider-action",
                "no_op_reason": "routed model metadata is missing for this bucket",
                "required_local_executor": None,
                "estimated_savings_per_1000_calls_usd": 0.0,
            },
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


def _pass_through_report_with_blocked_lifecycle() -> dict:
    report = _pass_through_report()
    bucket = report["buckets"][0]
    bucket["sample_count"] = 1250
    bucket["anthropic_canary_lifecycle_evidence"] = {
        "schema": "agentflow.anthropic_routing_canary_lifecycle_evidence.v1",
        "status": "matched",
        "matched_count": 1250,
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
            "matched_count": 1250,
            "observed_rate": 0.0408,
            "applied_rate": 0.0,
            "holdout_rate": 0.0,
        },
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "latest_observed_at": "2026-06-15T10:01:52.308969+00:00",
        "stale_evidence": {"stale": False, "age_hours": 3.222, "max_age_hours": 72.0},
        "blocker_codes": ["missing-applied-coverage", "missing-holdout-coverage", "safety-stop-observed"],
        "blocker_reason_breakdown": [
            {"value": "missing-applied-coverage", "count": 1250},
            {"value": "missing-holdout-coverage", "count": 1250},
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
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }
    report["buckets"] = [bucket]
    return report


def _pass_through_report_with_guard_ready_lifecycle() -> dict:
    report = _pass_through_report()
    bucket = report["buckets"][0]
    bucket["sample_count"] = 120
    bucket["anthropic_canary_lifecycle_evidence"] = {
        "schema": "agentflow.anthropic_routing_canary_lifecycle_evidence.v1",
        "status": "matched",
        "matched_count": 120,
        "observed_count": 54,
        "cohort_counts": {
            "canary_applied": 24,
            "canary_holdout": 30,
            "safety_stopped": 0,
            "skipped": 0,
            "bypassed_or_disabled": 0,
            "unknown": 0,
        },
        "coverage": {
            "matched_count": 120,
            "observed_rate": 0.45,
            "applied_rate": 0.2,
            "holdout_rate": 0.25,
        },
        "error_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "latest_observed_at": "2026-06-17T20:30:00+00:00",
        "stale_evidence": {"stale": False, "age_hours": 0.25, "max_age_hours": 72.0},
        "blocker_codes": [],
        "blocker_reason_breakdown": [],
        "safety_stop_breakdown": [],
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "raw_prompts_included": False,
            "provider_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
        },
    }
    report["buckets"] = [bucket]
    return report


def _assert_privacy_clean(testcase: unittest.TestCase, payload: dict) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "raw prompt secret",
        "raw response secret",
        "request-id-secret",
        "session-id-secret",
        "cache-key-secret",
        "/tmp/private",
    ):
        testcase.assertNotIn(forbidden, rendered)
    testcase.assertFalse(payload["privacy"]["provider_calls_made"])
    testcase.assertFalse(payload["privacy"]["managed_server_calls_made"])
    testcase.assertFalse(payload["privacy"]["raw_prompts_included"])
    testcase.assertFalse(payload["privacy"]["provider_bodies_included"])


class AnthropicRoutingCanaryStageTests(unittest.TestCase):
    def test_stages_tool_result_sonnet_to_haiku_canary_with_projected_lifecycle(self) -> None:
        result = build_anthropic_routing_canary_stage_report(
            _pass_through_report(),
            canary_fraction=0.05,
            holdout_fraction=0.10,
            min_samples=5,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema"], "agentflow.anthropic_routing_canary_stage.v1")
        self.assertEqual(result["summary"]["candidate_count"], 3)
        self.assertEqual(result["summary"]["eligible_candidate_count"], 1)
        self.assertEqual(result["summary"]["staged_count"], 1)
        self.assertEqual(result["summary"]["omitted_count"], 2)
        self.assertEqual(result["summary"]["projected_canary_applied_count"], 56)
        self.assertEqual(result["summary"]["projected_canary_holdout_count"], 111)
        self.assertEqual(result["summary"]["projected_safety_stopped_count"], 32)
        self.assertTrue(result["summary"]["acceptance_met"])

        acceptance = result["acceptance"]
        self.assertEqual(acceptance["schema"], "agentflow.anthropic_routing_canary_stage_acceptance.v1")
        self.assertEqual(acceptance["status"], "met")
        self.assertTrue(acceptance["tool_result_sonnet_to_haiku_candidate_reported"])
        self.assertTrue(acceptance["holdout_coverage_projected"])
        self.assertTrue(acceptance["lifecycle_counts_include_applied_holdout_skipped_safety_error_retry_fallback"])
        self.assertTrue(acceptance["thinking_and_tool_safety_gates_present"])
        self.assertTrue(acceptance["metadata_only_privacy_proof"])

        draft = result["staged_drafts"][0]
        self.assertFalse(draft["active_policy_changed"])
        self.assertFalse(draft["wrote_active_policy_files"])
        canary = draft["policies"]["routing"]["phase_canary"]
        self.assertFalse(canary["enabled"])
        self.assertTrue(canary["review_only"])
        self.assertEqual(canary["provider"], "anthropic")
        self.assertEqual(canary["source_surface"], "anthropic_messages")
        self.assertEqual(canary["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(canary["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(canary["eligible_categories"], ["tool-result"])
        self.assertEqual(canary["eligible_workflow_phases"], ["tool-execution"])
        self.assertEqual(canary["cohort_unit"], "session")
        self.assertTrue(canary["stream"])
        self.assertEqual(canary["max_text_chars"], 0)
        self.assertTrue(canary["safety_gates"]["block_thinking_history"])
        self.assertTrue(canary["safety_gates"]["block_unsafe_tool_call_context"])

        lifecycle = canary["promotion"]["projected_anthropic_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 56)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 111)
        self.assertEqual(lifecycle["cohort_counts"]["safety_stopped"], 0)
        self.assertEqual(lifecycle["estimated_savings_per_1000_calls_usd"], 4.5)

        omitted_by_reason = {item["reason"]: item for item in result["omitted"]}
        thinking_omission = next(
            item
            for item in result["omitted"]
            if isinstance(item.get("projected_lifecycle_evidence"), dict)
            and "thinking-routing-guard" in item["projected_lifecycle_evidence"].get("blocker_codes", [])
        )
        self.assertEqual(thinking_omission["status"], "safety_stopped")
        self.assertIn("thinking-routing-guard", thinking_omission["projected_lifecycle_evidence"]["blocker_codes"])
        self.assertEqual(omitted_by_reason["missing-routed-model"]["status"], "omitted")
        self.assertEqual(omitted_by_reason["missing-routed-model"]["target_model"], "claude-haiku-4-5-20251001")
        _assert_privacy_clean(self, result)

    def test_blocks_current_lifecycle_evidence_without_promotion_or_stage(self) -> None:
        result = build_anthropic_routing_canary_stage_report(
            _pass_through_report_with_blocked_lifecycle(),
            canary_fraction=0.05,
            holdout_fraction=0.10,
            min_samples=5,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["eligible_candidate_count"], 0)
        self.assertEqual(result["summary"]["staged_count"], 0)
        self.assertEqual(result["summary"]["omitted_count"], 1)
        self.assertEqual(result["summary"]["blocked_review_count"], 1)
        self.assertEqual(result["summary"]["projected_canary_applied_count"], 0)
        self.assertEqual(result["summary"]["projected_canary_holdout_count"], 0)
        self.assertEqual(result["summary"]["projected_safety_stopped_count"], 492)
        self.assertEqual(result["summary"]["executor_guard_dry_run_count"], 1)
        self.assertEqual(result["summary"]["executor_guard_keep_blocked_count"], 1)
        self.assertEqual(result["summary"]["top_executor_guard_status"], "keep-blocked")
        self.assertEqual(result["summary"]["top_required_local_executor"], "anthropic-routing-rules")
        self.assertTrue(result["summary"]["acceptance_met"])
        self.assertTrue(result["acceptance"]["blocked_review_recorded"])
        self.assertTrue(result["acceptance"]["blocked_review_has_local_rule_file_representation"])
        self.assertTrue(result["acceptance"]["no_automatic_promotion_while_blocked"])
        self.assertTrue(result["acceptance"]["blocked_review_has_keep_blocked_reason"])
        self.assertTrue(result["acceptance"]["executor_guard_dry_run_reports_all_criteria"])
        self.assertTrue(result["acceptance"]["executor_guard_blocks_safety_stopped_cohorts"])
        self.assertEqual(result["summary"]["top_next_state"], "keep-blocked")
        self.assertEqual(
            result["summary"]["top_keep_blocked_reason"],
            "anthropic-routing-safety-stop-needs-narrower-cohort-applied-holdout-coverage-safer-threshold-or-executor-guard-rollback-proof",
        )
        self.assertIn("rollback_proof", result["summary"]["top_needed_resolution"])

        blocked = result["blocked_reviews"][0]
        self.assertEqual(blocked["status"], "blocked-review")
        self.assertEqual(blocked["next_state"], "keep-blocked")
        self.assertEqual(
            blocked["next_state_reason"],
            "safety-stop-requires-safer-threshold-or-executor-guard-and-rollback-proof",
        )
        self.assertEqual(
            blocked["keep_blocked_reason"],
            "anthropic-routing-safety-stop-needs-narrower-cohort-applied-holdout-coverage-safer-threshold-or-executor-guard-rollback-proof",
        )
        self.assertEqual(
            blocked["needed_resolution"],
            [
                "applied_coverage",
                "holdout_coverage",
                "narrower_cohort",
                "rollback_proof",
                "safer_threshold_or_executor_guard",
                "safety_stop_reason_review",
            ],
        )
        self.assertEqual(blocked["matched_count"], 1250)
        self.assertEqual(blocked["observed_count"], 492)
        self.assertEqual(blocked["cohort_counts"]["canary_applied"], 0)
        self.assertEqual(blocked["cohort_counts"]["canary_holdout"], 0)
        self.assertEqual(blocked["cohort_counts"]["safety_stopped"], 492)
        self.assertEqual(blocked["coverage"]["applied_rate"], 0.0)
        self.assertEqual(blocked["coverage"]["holdout_rate"], 0.0)
        self.assertIn("missing-applied-coverage", blocked["blocker_codes"])
        self.assertIn("missing-holdout-coverage", blocked["blocker_codes"])
        self.assertIn("safety-stop-observed", blocked["blocker_codes"])
        self.assertEqual(blocked["safety_stop_breakdown"][0]["reason_code"], "local-canary-safety-stop")
        self.assertTrue(blocked["safety_stop_breakdown"][0]["executor_compatible"])
        self.assertEqual(blocked["local_file_backed_representation"]["rule_file"], "routing_rules.yaml")
        self.assertEqual(blocked["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(blocked["next_action"], "review-anthropic-routing-safety-stop-before-canary")
        self.assertEqual(blocked["durable_next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertFalse(blocked["promotion_allowed"])
        self.assertFalse(blocked["stage_allowed"])
        self.assertFalse(blocked["active_policy_changed"])
        self.assertFalse(blocked["wrote_active_policy_files"])
        guard = blocked["executor_guard_dry_run"]
        self.assertEqual(guard["schema"], "agentflow.anthropic_routing_executor_guard_dry_run.v1")
        self.assertEqual(guard["status"], "keep-blocked")
        self.assertFalse(guard["stage_allowed"])
        self.assertFalse(guard["promotion_allowed"])
        self.assertFalse(guard["active_policy_changed"])
        self.assertFalse(guard["wrote_active_policy_files"])
        self.assertEqual(guard["safety_stop_count"], 492)
        self.assertEqual(guard["required_local_executor"], "anthropic-routing-rules")
        self.assertTrue(guard["executor_compatible"])
        for field, reason in {
            "safety_stop_reason_review": "safety-stop-observed",
            "safer_threshold_or_executor_guard": "safer-threshold-or-executor-guard-missing",
            "rollback_proof": "rollback-proof-missing",
            "applied_coverage": "missing-applied-coverage",
            "holdout_coverage": "missing-holdout-coverage",
        }.items():
            self.assertFalse(guard["criterion_results"][field]["passed"])
            self.assertEqual(guard["criterion_results"][field]["status"], "failed")
            self.assertEqual(guard["criterion_results"][field]["reason_codes"], [reason])
            self.assertIn(field, guard["needed_resolution"])
            self.assertIn(reason, guard["reason_codes"])

        omitted = result["omitted"][0]
        self.assertEqual(omitted["status"], "blocked-review")
        self.assertEqual(omitted["projected_lifecycle_evidence"]["cohort_counts"]["safety_stopped"], 492)
        self.assertEqual(
            omitted["projected_lifecycle_evidence"]["safety_stop_breakdown"][0]["durable_blocked_reason"],
            "anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked",
        )
        self.assertEqual(omitted["blocked_review"]["target_local_rule_file"], "routing_rules.yaml")
        _assert_privacy_clean(self, result)

    def test_executor_guard_ready_dry_run_does_not_enable_traffic(self) -> None:
        result = build_anthropic_routing_canary_stage_report(
            _pass_through_report_with_guard_ready_lifecycle(),
            canary_fraction=0.05,
            holdout_fraction=0.10,
            min_samples=5,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["eligible_candidate_count"], 1)
        self.assertEqual(result["summary"]["staged_count"], 1)
        self.assertEqual(result["summary"]["executor_guard_dry_run_count"], 1)
        self.assertEqual(result["summary"]["executor_guard_ready_count"], 1)
        self.assertEqual(result["summary"]["top_executor_guard_status"], "guard-ready")
        self.assertTrue(result["acceptance"]["executor_guard_ready_without_activation"])

        guard = result["executor_guard_dry_runs"][0]
        self.assertEqual(guard["schema"], "agentflow.anthropic_routing_executor_guard_dry_run.v1")
        self.assertEqual(guard["status"], "guard-ready")
        self.assertTrue(guard["guard_ready"])
        self.assertTrue(guard["stage_allowed"])
        self.assertFalse(guard["promotion_allowed"])
        self.assertEqual(guard["safety_stop_count"], 0)
        self.assertEqual(guard["applied_count"], 24)
        self.assertEqual(guard["holdout_count"], 30)
        self.assertEqual(guard["needed_resolution"], [])
        self.assertEqual(guard["reason_codes"], [])
        self.assertEqual(guard["next_action"], "operator-review-anthropic-routing-guard-ready")
        for field in (
            "safety_stop_reason_review",
            "safer_threshold_or_executor_guard",
            "rollback_proof",
            "applied_coverage",
            "holdout_coverage",
        ):
            self.assertTrue(guard["criterion_results"][field]["passed"])
            self.assertEqual(guard["criterion_results"][field]["status"], "passed")
        self.assertFalse(guard["active_policy_changed"])
        self.assertFalse(guard["wrote_active_policy_files"])
        self.assertFalse(guard["provider_calls_made"])
        self.assertFalse(guard["managed_server_calls_made"])

        draft = result["staged_drafts"][0]
        self.assertFalse(draft["active_policy_changed"])
        self.assertFalse(draft["wrote_active_policy_files"])
        canary = draft["policies"]["routing"]["phase_canary"]
        self.assertFalse(canary["enabled"])
        self.assertTrue(canary["review_only"])
        self.assertEqual(canary["promotion"]["executor_guard_dry_run"]["status"], "guard-ready")
        _assert_privacy_clean(self, result)

    def test_cli_extracts_nested_research_plan_report(self) -> None:
        plan = {"schema": "agentflow.orchestrator_research_plan.v1", "evidence": {"pass_through_routing_report": _pass_through_report()}}
        stdout = io.StringIO()

        code = cli.anthropic_routing_canary_stage_cli(["-", "--draft-id", "anthropic-stage-test"], stdin=io.StringIO(json.dumps(plan)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.anthropic_routing_canary_stage.v1")
        self.assertEqual(payload["staged_drafts"][0]["draft_id"], "anthropic-stage-test")
        self.assertEqual(payload["summary"]["projected_canary_applied_count"], 56)
        self.assertTrue(payload["acceptance"]["acceptance_met"])
        _assert_privacy_clean(self, payload)

    def test_rejects_raw_payload_keys(self) -> None:
        report = _pass_through_report()
        report["raw_request"] = {"prompt": "raw prompt secret", "request_id": "request-id-secret"}

        result = build_anthropic_routing_canary_stage_report(report)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "raw_payload_rejected")
        self.assertFalse(result["acceptance"]["acceptance_met"])
        _assert_privacy_clean(self, result)


if __name__ == "__main__":
    unittest.main()
