from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.openai_routing_narrow_canary import build_openai_routing_narrow_canary_review
from tokenclaw.store import utc_now


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

    def _managed_outcomes(
        self,
        *,
        next_action: str = "draft-openai-routing-recovery-canary",
        classification: str = "review-only",
        stale: bool = False,
        missing: bool = False,
        failed_closed: bool = False,
        disagreement: bool = False,
        candidate_fingerprint: str | None = None,
    ) -> dict[str, object]:
        outcome = {
            "schema": "tokenclaw.managed_activation_preview_outcome.v1",
            "handoff_ref": "managed-preview-handoff:routing",
            "preview_ref": "managed-preview:openai-routing",
            "local_action_family": "routing",
            "evidence_schema": "tokenclaw.openai_routing_promotion_decision_report.v1",
            "decision": "review-only-recommendation",
            "classification": classification,
            "next_action": next_action,
            "preview_age_hours": 2.0,
            "stale": stale,
            "missing_preview_decision": missing,
            "failed_closed": failed_closed,
            "disagrees_with_local_evidence": disagreement,
            "review_only": True,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }
        if candidate_fingerprint is not None:
            outcome["candidate_fingerprint"] = candidate_fingerprint
        return {
            "schema": "tokenclaw.managed_activation_preview_outcomes.v1",
            "status": "tracked",
            "outcomes": [outcome],
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }

    def _managed_health(self, *, stale: bool = False) -> dict[str, object]:
        return {
            "schema": "tokenclaw.managed_activation_preview_health.v1",
            "status": "tracked",
            "accepted_batch_count": 1,
            "previewed_row_count": 1,
            "latest_preview_age_hours": 80.0 if stale else 2.0,
            "stale_after_hours": 72.0,
            "stale": stale,
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }

    def _pathway_outcome(
        self,
        *,
        candidate_fingerprint: str,
        source_surface: str,
        app_family: str,
        requested_model: str,
        target_model: str,
        category: str,
        applied_count: int,
        holdout_count: int,
        savings_per_1000: float,
    ) -> dict[str, object]:
        return {
            "schema": "tokenclaw.local_routing_pathway_outcome_feedback_row.v1",
            "status": "ready",
            "provider": "openai",
            "source_surface": source_surface,
            "app_family": app_family,
            "endpoint": "responses",
            "requested_model": requested_model,
            "target_model": target_model,
            "category": category,
            "workflow_phase": "stateless_text",
            "matched_count": applied_count + holdout_count,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "safety_stop_count": 0,
            "error_count": 0,
            "fallback_count": 0,
            "retry_count": 0,
            "blocker_status": "applied-and-holdout-coverage-present",
            "recommended_next_action": "stage-narrow-routing-canary",
            "candidate_fingerprint": candidate_fingerprint,
            "savings_per_1000_calls_usd": savings_per_1000,
            "projected_savings_usd": round((savings_per_1000 * (applied_count + holdout_count)) / 1000.0, 6),
            "semantic_quality": {
                "gate_passed": True,
                "reason_codes": [],
            },
        }

    def _managed_pathway_outcomes(self, *candidate_fingerprints: str) -> dict[str, object]:
        return {
            "schema": "tokenclaw.managed_activation_preview_outcomes.v1",
            "status": "tracked",
            "outcomes": [
                {
                    "schema": "tokenclaw.managed_activation_preview_outcome.v1",
                    "handoff_ref": f"managed-preview-handoff:{candidate_fingerprint}",
                    "preview_ref": f"managed-preview:{candidate_fingerprint}",
                    "local_action_family": "routing",
                    "evidence_schema": "tokenclaw.local_routing_pathway_outcome_feedback_row.v1",
                    "decision": "review-only-recommendation",
                    "classification": "review-only",
                    "next_action": "stage-narrow-routing-canary",
                    "candidate_fingerprint": candidate_fingerprint,
                    "preview_age_hours": 2.0,
                    "stale": False,
                    "missing_preview_decision": False,
                    "failed_closed": False,
                    "disagrees_with_local_evidence": False,
                    "review_only": True,
                    "policy_files_written": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "provider_bodies_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                    },
                }
                for candidate_fingerprint in candidate_fingerprints
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

    def _semantic_recovery_action(self) -> dict[str, object]:
        return {
            "schema": "tokenclaw.openai_routing_semantic_regression_action.v1",
            "observed": True,
            "status": "classified",
            "action_classification": "narrow-canary-shape",
            "deterministic_next_action": "draft-narrow-openai-routing-canary-shape",
            "reason_codes": ["semantic-quality-regression-observed"],
        }

    def test_mixed_regressed_and_clean_cohorts_emit_one_review_only_narrower_canary(self) -> None:
        report = {
            "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                {
                    **self._cohort(
                        category="tool-light",
                        applied_count=25,
                        holdout_count=21,
                        reason_codes=["semantic-quality-regression-observed"],
                        savings_per_1000=4.375,
                    ),
                    "semantic_regression_action": self._semantic_recovery_action(),
                },
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

        result = build_openai_routing_narrow_canary_review(
            report,
            managed_preview_outcomes=self._managed_outcomes(),
            managed_preview_health=self._managed_health(),
            canary_fraction=0.07,
            holdout_fraction=0.13,
        )

        self.assertEqual(result["schema"], "tokenclaw.openai_routing_narrow_canary_review.v1")
        self.assertEqual(result["decision"], "draft-narrower-canary")
        self.assertEqual(result["status"], "review-only")
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["regressed_cohort_count"], 1)
        self.assertEqual(result["summary"]["managed_preview_agreement_count"], 1)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])

        draft = result["drafts"][0]
        self.assertEqual(draft["schema"], "tokenclaw.openai_routing_narrow_canary_draft.v1")
        self.assertTrue(draft["review_only"])
        self.assertFalse(draft["active_policy_changed"])
        self.assertFalse(draft["policy_files_written"])
        self.assertEqual(draft["category"], "tool-light")
        self.assertEqual(draft["requested_model"], "gpt-5.4")
        self.assertEqual(draft["target_model"], "gpt-5.4-mini")
        self.assertEqual(draft["canary_fraction"], 0.07)
        self.assertEqual(draft["holdout_fraction"], 0.13)
        self.assertEqual(draft["proposed_rule_conditions"]["category"], "tool-light")
        self.assertEqual(draft["rollback_condition"]["rollback_action_type"], "disable_openai_routing_narrow_canary")
        self.assertFalse(draft["privacy"]["provider_calls_made"])
        self.assertFalse(draft["privacy"]["managed_server_calls_made"])
        self.assertFalse(draft["privacy"]["raw_prompts_included"])
        self.assertEqual(draft["recovery_plan"]["selected_option"], "restage-review-only")
        self.assertEqual(draft["recovery_plan"]["blocker_status"], "cleared")
        self.assertTrue(draft["managed_preview_agreement"]["agreed"])
        self.assertEqual(draft["managed_preview_agreement"]["reason"], "local-managed-preview-agree")
        self.assertEqual(draft["managed_preview_agreement"]["health_gate"]["status"], "fresh-preview-health")
        self.assertEqual(draft["recovery_plan"]["coverage"]["applied_count"], 25)
        self.assertEqual(draft["recovery_plan"]["coverage"]["holdout_count"], 21)
        self.assertFalse(draft["recovery_plan"]["rollback_no_write"]["policy_files_written"])
        self.assertFalse(draft["recovery_plan"]["rollback_no_write"]["active_policy_changed"])
        self.assertEqual(result["recovery_plan"]["selected_option"], "restage-review-only")
        self.assertEqual(result["summary"]["recovery_selected_option"], "restage-review-only")

        self.assertEqual(result["summary"]["regressed_cohort_count"], 1)
        clean_omission = next(item for item in result["omitted"] if item["category"] == "chat")
        self.assertEqual(clean_omission["reason"], "not-semantic-regression-row")

    def test_only_regressed_cohorts_keep_blocked_without_policy_writes(self) -> None:
        report = {
            "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
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

    def test_research_plan_successor_keeps_blocked_with_preview_health_and_no_write(self) -> None:
        health_gate = {
            "schema": "tokenclaw.managed_activation_preview_health_gate.v1",
            "status": "no-data-preview-health",
            "reason": "managed-preview-health-no-data",
            "next_action": "refresh-managed-activation-preview",
            "passed": False,
            "accepted_batch_count": 0,
            "previewed_row_count": 0,
            "policy_files_written": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }
        report = {
            "schema": "tokenclaw.orchestrator_research_plan.v1",
            "evidence": {
                "stats_summary": {
                    "openai_routing_promotion_decision": {
                        "schema": "tokenclaw.openai_routing_promotion_decision_report.v1",
                        "promotion_decision": {
                            "schema": "tokenclaw.openai_routing_promotion_decision.v1",
                            "decision": "keep-blocked",
                            "matched_count": 345,
                            "projected_savings_usd": 1.509375,
                            "savings_per_1000_calls_usd": 4.375,
                            "reason": "semantic-quality-regression-observed",
                            "reason_codes": ["semantic-quality-regression-observed"],
                            "target": {
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "provider": "openai",
                                "requested_model": "gpt-5.4",
                                "target_model": "gpt-5.4-mini",
                                "category": "tool-light",
                                "target_local_policy_section": "routing.rules",
                                "target_local_rule_file": "routing_rules.yaml",
                            },
                            "lifecycle": {
                                "schema": "tokenclaw.openai_routing_canary_lifecycle_evidence.v1",
                                "status": "matched",
                                "applied_count": 27,
                                "holdout_count": 23,
                                "safety_stop_count": 0,
                                "error_count": 0,
                                "fallback_count": 0,
                                "retry_count": 0,
                                "skipped_count": 230,
                                "unknown_count": 6,
                            },
                        },
                    },
                    "local_activation_next_action_queue": {
                        "schema": "tokenclaw.local_activation_next_action_queue.v1",
                        "successor_actions": [
                            {
                                "schema": "tokenclaw.local_activation_successor_action.v1",
                                "fingerprint": "successor:a9729de3a6d5873b",
                                "source_fingerprint": "activation:9ddae7127b2ccbaf",
                                "evidence_schema": "tokenclaw.openai_routing_promotion_decision_report.v1",
                                "local_action_family": "routing",
                                "current_status": "keep-blocked",
                                "successor_status": "keep-blocked",
                                "blocker_codes": ["semantic-quality-regression-observed"],
                                "applied_count": 27,
                                "holdout_count": 23,
                                "sample_count": 345,
                                "projected_savings_usd": 1.509375,
                                "savings_per_1000_calls_usd": 4.375,
                                "preview_verified": False,
                                "preview_verification_status": "no-data-preview-health",
                                "preview_verification_decision": "keep-blocked",
                                "recommended_next_action": "refresh-managed-activation-preview",
                                "target_local_policy_section": "routing.rules",
                                "target_local_rule_file": "routing_rules.yaml",
                                "managed_preview_gate": {
                                    "schema": "tokenclaw.preview_verified_activation_successor_gate.v1",
                                    "status": "no-data-preview-health",
                                    "reason": "managed-preview-health-no-data",
                                    "verified": False,
                                    "required": True,
                                    "policy_files_written": False,
                                    "provider_calls_made": False,
                                    "managed_server_calls_made": False,
                                    "health_gate": health_gate,
                                },
                                "privacy": {
                                    "metadata_only": True,
                                    "aggregate_only": True,
                                    "raw_prompts_included": False,
                                    "provider_bodies_included": False,
                                    "request_ids_included": False,
                                    "session_ids_included": False,
                                },
                            }
                        ],
                    },
                },
            },
            "backlog_changes": {
                "create_issues": [
                    {
                        "title": "Keep OpenAI routing recovery blocked",
                        "body": "Generated GitHub issue prose is not provider request content.",
                    }
                ]
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        result = build_openai_routing_narrow_canary_review(report)

        self.assertEqual(result["decision"], "keep-blocked")
        self.assertEqual(result["reason"], "semantic-quality-regression-observed")
        self.assertEqual(result["summary"]["managed_preview_health_status"], "no-data-preview-health")
        self.assertEqual(result["summary"]["managed_preview_health_reason"], "managed-preview-health-no-data")
        self.assertEqual(result["summary"]["draft_count"], 0)
        self.assertFalse(result["summary"]["policy_files_written"])
        self.assertFalse(result["provider_calls_made"])
        omitted = result["omitted"][0]
        self.assertEqual(omitted["reason"], "semantic-quality-regression-observed")
        self.assertEqual(omitted["coverage"]["applied_count"], 27)
        self.assertEqual(omitted["coverage"]["holdout_count"], 23)
        self.assertEqual(omitted["coverage"]["safety_stop_count"], 0)
        self.assertEqual(omitted["managed_preview_agreement"]["health_gate"]["status"], "no-data-preview-health")
        self.assertEqual(omitted["managed_preview_agreement"]["health_gate"]["reason"], "managed-preview-health-no-data")
        self.assertEqual(omitted["recovery_sizing"]["status"], "not-available")
        self.assertEqual(omitted["recovery_sizing"]["reason"], "managed-preview-health-no-data")
        self.assertFalse(omitted["rollback_no_write"]["policy_files_written"])
        self.assertFalse(result["recovery_plan"]["rollback_no_write"]["policy_files_written"])
        self.assertEqual(result["recovery_plan"]["target_local_rule_file"], "routing_rules.yaml")

    def test_semantic_regression_recovery_drafts_only_with_managed_preview_agreement(self) -> None:
        report = {
            "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                {
                    **self._cohort(
                        category="tool-light",
                        applied_count=25,
                        holdout_count=21,
                        reason_codes=["semantic-quality-regression-observed"],
                    ),
                    "semantic_regression_action": {
                        "schema": "tokenclaw.openai_routing_semantic_regression_action.v1",
                        "observed": True,
                        "status": "classified",
                        "action_classification": "narrow-canary-shape",
                        "deterministic_next_action": "draft-narrow-openai-routing-canary-shape",
                        "reason_codes": ["semantic-quality-regression-observed"],
                    },
                }
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        result = build_openai_routing_narrow_canary_review(
            report,
            managed_preview_outcomes=self._managed_outcomes(next_action="draft-openai-routing-recovery-canary"),
            managed_preview_health=self._managed_health(),
        )

        self.assertEqual(result["decision"], "draft-narrower-canary")
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["managed_preview_agreement_count"], 1)
        draft = result["drafts"][0]
        self.assertEqual(draft["category"], "tool-light")
        self.assertEqual(draft["requested_model"], "gpt-5.4")
        self.assertEqual(draft["target_model"], "gpt-5.4-mini")
        self.assertTrue(draft["managed_preview_agreement"]["agreed"])
        self.assertEqual(draft["managed_preview_agreement"]["normalized_local_action"], "draft-recovery-canary")
        self.assertEqual(draft["managed_preview_agreement"]["normalized_managed_action"], "draft-recovery-canary")
        self.assertEqual(draft["semantic_regression_recovery"]["classification"], "narrower-canary")
        self.assertEqual(draft["coverage"]["applied_count"], 25)
        self.assertEqual(draft["coverage"]["holdout_count"], 21)
        self.assertFalse(draft["policy_files_written"])
        self.assertFalse(draft["rollback_no_write"]["policy_files_written"])
        self.assertEqual(result["recovery_plan"]["selected_option"], "restage-review-only")

    def test_pathway_feedback_drafts_preview_agreed_openai_and_codex_canaries_without_policy_writes(self) -> None:
        report = {
            "schema": "tokenclaw.local_routing_pathway_outcome_feedback.v1",
            "generated_at": utc_now(),
            "outcomes": [
                self._pathway_outcome(
                    candidate_fingerprint="routing-pathway-candidate:openai-tool-light",
                    source_surface="openai_responses",
                    app_family="generic_openai",
                    requested_model="gpt-5.4",
                    target_model="gpt-5.4-mini",
                    category="tool-light",
                    applied_count=29,
                    holdout_count=26,
                    savings_per_1000=4.375,
                ),
                self._pathway_outcome(
                    candidate_fingerprint="routing-pathway-candidate:codex-stateless",
                    source_surface="codex_turn",
                    app_family="codex",
                    requested_model="gpt-5-codex",
                    target_model="gpt-5.4-mini",
                    category="stateless_text",
                    applied_count=17,
                    holdout_count=15,
                    savings_per_1000=3.25,
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
        managed_health = {
            **self._managed_health(),
            "previewed_row_count": 2,
        }

        result = build_openai_routing_narrow_canary_review(
            report,
            managed_preview_outcomes=self._managed_pathway_outcomes(
                "routing-pathway-candidate:openai-tool-light",
                "routing-pathway-candidate:codex-stateless",
            ),
            managed_preview_health=managed_health,
            top_candidates=2,
        )

        self.assertEqual(result["decision"], "draft-narrower-canary")
        self.assertEqual(result["status"], "review-only")
        self.assertEqual(result["summary"]["draft_count"], 2)
        self.assertEqual(result["summary"]["managed_preview_agreement_count"], 2)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])

        drafts_by_surface = {draft["source_surface"]: draft for draft in result["drafts"]}
        openai_draft = drafts_by_surface["openai_responses"]
        self.assertEqual(openai_draft["target_local_policy_section"], "routing.rules")
        self.assertEqual(openai_draft["target_local_rule_file"], "routing_rules.yaml")
        self.assertIn("proposed_openai_canary", openai_draft)
        self.assertEqual(openai_draft["coverage"]["applied_count"], 29)
        self.assertEqual(openai_draft["coverage"]["holdout_count"], 26)
        self.assertTrue(openai_draft["managed_preview_agreement"]["agreed"])

        codex_draft = drafts_by_surface["codex_turn"]
        self.assertEqual(codex_draft["app_family"], "codex")
        self.assertEqual(codex_draft["target_local_policy_section"], "codex_app.summary_model_hint")
        self.assertEqual(codex_draft["target_local_rule_file"], "codex_app_rules.yaml")
        self.assertEqual(codex_draft["rollback_condition"]["rollback_action_type"], "disable_codex_app_routing_narrow_canary")
        self.assertEqual(codex_draft["rollback_no_write"]["target_local_rule_file"], "codex_app_rules.yaml")
        self.assertEqual(codex_draft["recovery_plan"]["target_model"], "gpt-5.4-mini")
        self.assertEqual(codex_draft["recovery_plan"]["target_local_policy_section"], "codex_app.summary_model_hint")
        self.assertIn("proposed_codex_app_canary", codex_draft)
        self.assertEqual(codex_draft["proposed_codex_app_canary"]["section"], "summary_model_hint")
        self.assertEqual(codex_draft["proposed_codex_app_canary"]["cohort_unit"], "turn")
        self.assertEqual(codex_draft["coverage"]["applied_count"], 17)
        self.assertEqual(codex_draft["coverage"]["holdout_count"], 15)
        self.assertTrue(codex_draft["managed_preview_agreement"]["agreed"])
        self.assertFalse(codex_draft["privacy"]["raw_prompts_included"])
        self.assertFalse(codex_draft["privacy"]["provider_calls_made"])
        self.assertFalse(codex_draft["privacy"]["managed_server_calls_made"])

    def test_semantic_recovery_noops_when_preview_disagrees_stale_or_coverage_missing(self) -> None:
        base_cohort = {
            **self._cohort(
                category="tool-light",
                applied_count=25,
                holdout_count=21,
                reason_codes=["semantic-quality-regression-observed"],
            ),
            "candidate_fingerprint": "candidate:semantic-row",
            "semantic_regression_action": {
                "schema": "tokenclaw.openai_routing_semantic_regression_action.v1",
                "observed": True,
                "status": "classified",
                "action_classification": "narrow-canary-shape",
                "deterministic_next_action": "draft-narrow-openai-routing-canary-shape",
                "reason_codes": ["semantic-quality-regression-observed"],
            },
        }
        cases = [
            (
                "disagreement",
                base_cohort,
                self._managed_outcomes(next_action="keep-openai-routing-blocked"),
                "managed-preview-action-disagreement",
            ),
            (
                "stale-managed-preview",
                base_cohort,
                self._managed_outcomes(stale=True, classification="stale-preview"),
                "managed-preview-health-stale",
            ),
            (
                "missing-holdout",
                {**base_cohort, "holdout_count": 0, "matched_count": 25},
                self._managed_outcomes(),
                "missing-holdout-coverage",
            ),
            (
                "candidate-fingerprint-mismatch",
                base_cohort,
                self._managed_outcomes(candidate_fingerprint="candidate:other-row"),
                "missing-managed-preview-outcome",
            ),
            (
                "active-semantic-regression",
                self._cohort(
                    category="tool-light",
                    applied_count=25,
                    holdout_count=21,
                    reason_codes=["semantic-quality-regression-observed"],
                ),
                self._managed_outcomes(),
                "semantic-quality-regression-observed",
            ),
        ]

        for _, cohort, managed_outcomes, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = build_openai_routing_narrow_canary_review(
                    {
                        "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
                        "generated_at": utc_now(),
                        "cohorts": [cohort],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    managed_preview_outcomes=managed_outcomes,
                    managed_preview_health=self._managed_health(stale=expected_reason == "managed-preview-health-stale"),
                )

                self.assertEqual(result["decision"], "keep-blocked")
                self.assertEqual(result["drafts"], [])
                self.assertEqual(result["summary"]["draft_count"], 0)
                self.assertEqual(result["omitted"][0]["reason"], expected_reason)
                self.assertFalse(result["omitted"][0]["rollback_no_write"]["policy_files_written"])

    def test_cleared_semantic_blocker_with_fresh_coverage_emits_review_only_recovery_plan(self) -> None:
        report = {
            "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                {
                    **self._cohort(
                        category="tool-light",
                        applied_count=25,
                        holdout_count=21,
                        reason_codes=["semantic-quality-regression-observed"],
                    ),
                    "semantic_regression_action": self._semantic_recovery_action(),
                    "openai_canary_lifecycle_evidence": {
                        "schema": "tokenclaw.openai_routing_canary_lifecycle_evidence.v1",
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

        result = build_openai_routing_narrow_canary_review(
            report,
            managed_preview_outcomes=self._managed_outcomes(),
            managed_preview_health=self._managed_health(),
        )

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
            "schema": "tokenclaw.openai_routing_semantic_regression_fixture.v1",
            "generated_at": utc_now(),
            "cohorts": [
                {
                    **self._cohort(
                        category="tool-light",
                        applied_count=25,
                        holdout_count=21,
                        reason_codes=["semantic-quality-regression-observed"],
                    ),
                    "semantic_regression_action": self._semantic_recovery_action(),
                },
                self._cohort(category="chat", applied_count=8, holdout_count=9, savings_per_1000=3.0),
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }
        stdout = io.StringIO()

        with TemporaryDirectory() as tmpdir:
            managed_path = Path(tmpdir) / "managed.json"
            managed_path.write_text(json.dumps(self._managed_outcomes()), encoding="utf-8")
            code = cli.openai_routing_narrow_canary_review_cli(
                [
                    "-",
                    "--managed-preview-outcomes",
                    str(managed_path),
                    "--managed-preview-health",
                    str(managed_path),
                    "--canary-fraction",
                    "0.05",
                    "--holdout-fraction",
                    "0.1",
                ],
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
