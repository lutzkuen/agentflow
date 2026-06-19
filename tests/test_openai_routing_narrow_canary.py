from __future__ import annotations

import io
import json
import unittest

from agentflow_proxy import cli
from agentflow_proxy.openai_routing_narrow_canary import build_openai_routing_narrow_canary_review
from agentflow_proxy.store import utc_now


class OpenAIRoutingNarrowCanaryReviewTests(unittest.TestCase):
    def _cohort(
        self,
        *,
        category: str,
        applied_count: int,
        holdout_count: int,
        reason_codes: list[str] | None = None,
        savings_per_1000: float = 4.375,
    ) -> dict[str, object]:
        return {
            "provider": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "requested_model": "gpt-5.4",
            "target_model": "gpt-5.4-mini",
            "category": category,
            "matched_count": applied_count + holdout_count,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "estimated_savings_per_1000_calls_usd": savings_per_1000,
            "projected_savings_usd": round((savings_per_1000 * (applied_count + holdout_count)) / 1000.0, 6),
            "reason_codes": reason_codes or [],
            "semantic_quality": {
                "gate_passed": "semantic-quality-regression-observed" not in (reason_codes or []),
                "reason_codes": reason_codes or [],
            },
        }

    def test_mixed_regressed_and_clean_cohorts_emit_one_review_only_narrower_canary(self) -> None:
        report = {
            "schema": "agentflow.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                self._cohort(
                    category="tool-light",
                    applied_count=25,
                    holdout_count=21,
                    reason_codes=["semantic-quality-regression-observed"],
                    savings_per_1000=4.375,
                ),
                self._cohort(
                    category="chat",
                    applied_count=12,
                    holdout_count=14,
                    savings_per_1000=2.5,
                ),
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

        result = build_openai_routing_narrow_canary_review(report, canary_fraction=0.07, holdout_fraction=0.13)

        self.assertEqual(result["schema"], "agentflow.openai_routing_narrow_canary_review.v1")
        self.assertEqual(result["decision"], "draft-narrower-canary")
        self.assertEqual(result["status"], "review-only")
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["regressed_cohort_count"], 1)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])

        draft = result["drafts"][0]
        self.assertEqual(draft["schema"], "agentflow.openai_routing_narrow_canary_draft.v1")
        self.assertTrue(draft["review_only"])
        self.assertFalse(draft["active_policy_changed"])
        self.assertFalse(draft["policy_files_written"])
        self.assertEqual(draft["category"], "chat")
        self.assertEqual(draft["requested_model"], "gpt-5.4")
        self.assertEqual(draft["target_model"], "gpt-5.4-mini")
        self.assertEqual(draft["canary_fraction"], 0.07)
        self.assertEqual(draft["holdout_fraction"], 0.13)
        self.assertEqual(draft["proposed_rule_conditions"]["category"], "chat")
        self.assertEqual(draft["rollback_condition"]["rollback_action_type"], "disable_openai_routing_narrow_canary")
        self.assertFalse(draft["privacy"]["provider_calls_made"])
        self.assertFalse(draft["privacy"]["managed_server_calls_made"])
        self.assertFalse(draft["privacy"]["raw_prompts_included"])
        self.assertEqual(draft["recovery_plan"]["selected_option"], "restage-review-only")
        self.assertEqual(draft["recovery_plan"]["blocker_status"], "cleared")
        self.assertEqual(draft["recovery_plan"]["coverage"]["applied_count"], 12)
        self.assertEqual(draft["recovery_plan"]["coverage"]["holdout_count"], 14)
        self.assertFalse(draft["recovery_plan"]["rollback_no_write"]["policy_files_written"])
        self.assertFalse(draft["recovery_plan"]["rollback_no_write"]["active_policy_changed"])
        self.assertEqual(result["recovery_plan"]["selected_option"], "restage-review-only")
        self.assertEqual(result["summary"]["recovery_selected_option"], "restage-review-only")

        regressed = result["regressed_cohorts"][0]
        self.assertEqual(regressed["reason"], "semantic-quality-regression-observed")
        self.assertEqual(regressed["category"], "tool-light")
        self.assertIn("semantic-quality-regression-observed", regressed["reason_codes"])

    def test_only_regressed_cohorts_keep_blocked_without_policy_writes(self) -> None:
        report = {
            "schema": "agentflow.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                self._cohort(
                    category="tool-light",
                    applied_count=25,
                    holdout_count=21,
                    reason_codes=["semantic-quality-regression-observed"],
                )
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        result = build_openai_routing_narrow_canary_review(report)

        self.assertEqual(result["decision"], "keep-blocked")
        self.assertEqual(result["status"], "keep-blocked")
        self.assertEqual(result["reason"], "semantic-quality-regression-observed")
        self.assertEqual(result["drafts"], [])
        self.assertEqual(result["summary"]["draft_count"], 0)
        self.assertEqual(result["summary"]["regressed_cohort_count"], 1)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertEqual(result["recovery_plan"]["selected_option"], "keep-blocked")
        self.assertEqual(result["recovery_plan"]["blocker_status"], "active")
        self.assertEqual(result["recovery_plan"]["blocker_reason"], "semantic-quality-regression-observed")
        self.assertEqual(result["recovery_plan"]["coverage"]["applied_count"], 25)
        self.assertEqual(result["recovery_plan"]["coverage"]["holdout_count"], 21)
        self.assertEqual(result["recovery_plan"]["target_local_policy_section"], "routing.rules")
        self.assertEqual(result["recovery_plan"]["target_local_rule_file"], "routing_rules.yaml")
        self.assertFalse(result["recovery_plan"]["rollback_no_write"]["active_policy_changed"])
        self.assertFalse(result["recovery_plan"]["rollback_no_write"]["policy_files_written"])
        options = {item["option"]: item for item in result["recovery_plan"]["options"]}
        self.assertTrue(options["keep-blocked"]["selected"])
        self.assertTrue(options["retire-disabled-rule"]["allowed"])
        self.assertFalse(options["restage-review-only"]["allowed"])

    def test_cleared_semantic_blocker_with_fresh_coverage_emits_review_only_recovery_plan(self) -> None:
        report = {
            "schema": "agentflow.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                {
                    **self._cohort(category="tool-light", applied_count=25, holdout_count=21),
                    "openai_canary_lifecycle_evidence": {
                        "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
                        "status": "matched",
                        "latest_observed_at": "2026-06-18T17:37:11.818295+00:00",
                        "applied_count": 25,
                        "holdout_count": 21,
                        "safety_stop_count": 0,
                        "error_count": 0,
                        "fallback_count": 0,
                        "retry_count": 0,
                        "stale_evidence": {
                            "stale": False,
                            "age_hours": 9.634,
                            "max_age_hours": 72.0,
                        },
                    },
                }
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        result = build_openai_routing_narrow_canary_review(report)

        self.assertEqual(result["decision"], "draft-narrower-canary")
        self.assertEqual(result["status"], "review-only")
        self.assertEqual(result["recovery_plan"]["selected_option"], "restage-review-only")
        self.assertEqual(result["recovery_plan"]["blocker_status"], "cleared")
        self.assertEqual(result["recovery_plan"]["stale_evidence"]["status"], "fresh")
        self.assertEqual(result["recovery_plan"]["coverage"]["applied_count"], 25)
        self.assertEqual(result["recovery_plan"]["coverage"]["holdout_count"], 21)
        self.assertTrue(result["recovery_plan"]["coverage"]["has_no_safety_stops"])
        self.assertFalse(result["recovery_plan"]["rollback_no_write"]["policy_files_written"])
        self.assertFalse(result["recovery_plan"]["rollback_no_write"]["provider_calls_made"])
        options = {item["option"]: item for item in result["recovery_plan"]["options"]}
        self.assertTrue(options["restage-review-only"]["selected"])
        self.assertTrue(options["restage-review-only"]["allowed"])
        self.assertTrue(options["narrow-threshold"]["allowed"])
        self.assertFalse(options["retire-disabled-rule"]["allowed"])

    def test_cli_reads_fixture_from_stdin(self) -> None:
        report = {
            "schema": "agentflow.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                self._cohort(
                    category="tool-light",
                    applied_count=25,
                    holdout_count=21,
                    reason_codes=["semantic-quality-regression-observed"],
                ),
                self._cohort(category="chat", applied_count=8, holdout_count=9, savings_per_1000=3.0),
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }
        stdout = io.StringIO()

        code = cli.openai_routing_narrow_canary_review_cli(
            ["-", "--canary-fraction", "0.05", "--holdout-fraction", "0.1"],
            stdin=io.StringIO(json.dumps(report)),
            stdout=stdout,
        )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["decision"], "draft-narrower-canary")
        self.assertEqual(payload["summary"]["draft_count"], 1)
        self.assertFalse(payload["summary"]["policy_files_written"])
        self.assertFalse(payload["summary"]["provider_calls_made"])
        self.assertFalse(payload["summary"]["managed_server_calls_made"])


if __name__ == "__main__":
    unittest.main()
