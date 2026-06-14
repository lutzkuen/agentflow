from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml

from agentflow_proxy import cli
from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.openai_optimization_governor import LIFECYCLE_SOURCE_SURFACE
from agentflow_proxy.openai_routing_canary_stage import (
    apply_openai_routing_canary_draft,
    stage_openai_routing_canary_drafts,
)
from agentflow_proxy.openai_routing_report import build_openai_routing_report
from agentflow_proxy.optimization import feedback
from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
from agentflow_proxy.promotion_blocker_review import build_promotion_blocker_recommendation_review
from agentflow_proxy.store import SQLiteStore, stable_json


class OpenAIRoutingCanaryStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_openai_call(
        self,
        *,
        path: str = "/v1/responses",
        source_surface: str = "openai_responses",
        endpoint: str = "responses",
        requested_model: str = "gpt-5.4",
        category: str = "chat",
        text_chars: int = 1200,
        status_code: int = 200,
        retry_count: int = 0,
        has_tools: bool = False,
        stream: int = 0,
        cost_baseline_usd: float = 0.004,
    ) -> None:
        actual_input_tokens = max(1, text_chars // 4)
        actual_output_tokens = 40
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=requested_model,
            stream=stream,
            cache_hit=0,
            status_code=status_code,
            latency_ms=120,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=0.001,
            cost_baseline_usd=cost_baseline_usd,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "enabled": False,
                "provider": "openai",
                "requested_model": requested_model,
                "routed_model": requested_model,
                "reason": "openai routing disabled",
                "text_chars": text_chars,
                "has_tools": has_tools,
                "category": category,
            }),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error=None,
            request_json='{"input":"raw prompt must not appear"}',
            response_json=None,
            session_id="secret-openai-session-id",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )

    def test_stages_responses_and_chat_large_to_mini_candidates_without_applying_policy(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200)
            self._log_openai_call(
                path="/v1/chat/completions",
                source_surface="openai_chat",
                endpoint="chat_completions",
                category="summary",
                text_chars=1400,
            )

        active_path = Path(self.tmpdir.name) / "routing_rules.yaml"
        active_text = "rules: []\nopenai_canary:\n  enabled: false\n"
        active_path.write_text(active_text, encoding="utf-8")
        workspace = Path(self.tmpdir.name) / "drafts"
        report = build_openai_routing_report(self.store, limit=50)

        from agentflow_proxy import router

        with (
            patch.object(router, "ROUTING_RULES_PATH", str(active_path)),
            patch.object(router, "ROUTING_RULES_SOURCE", "local-manual"),
            patch.object(router, "ROUTING_RULES_LOADED_AT", utc_now()),
            patch.object(router, "ROUTING_RULES_LOADED_FILE", policy_file_snapshot(active_path)),
        ):
            result = asyncio.run(stage_openai_routing_canary_drafts(
                report,
                draft_id="openai-large-mini",
                workspace=str(workspace),
                canary_fraction=0.07,
                holdout_fraction=0.13,
                top_candidates=10,
            ))

        self.assertTrue(result["ok"], json.dumps(result, indent=2, sort_keys=True))
        self.assertEqual(result["summary"]["staged_count"], 2)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertEqual(active_path.read_text(encoding="utf-8"), active_text)

        endpoints = {item["endpoint"] for item in result["staged_drafts"]}
        self.assertEqual(endpoints, {"responses", "chat_completions"})
        for staged in result["staged_drafts"]:
            self.assertEqual(staged["target_model"], "gpt-5.4-mini")
            self.assertEqual(staged["canary_fraction"], 0.07)
            self.assertEqual(staged["holdout_fraction"], 0.13)
            self.assertGreater(staged["estimated_savings_per_1000_calls_usd"], 0)
            self.assertEqual(staged["projected_cohort_counts"]["matched"], 6)
            self.assertIn("eligible-openai-large-to-mini", staged["reason_codes"])
            self.assertEqual(staged["rollback_metadata"]["rollback_action_type"], "disable_openai_canary")
            bundle = json.loads(Path(staged["bundle_path"]).read_text(encoding="utf-8"))
            canary = bundle["policies"]["routing"]["openai"]["canary"]
            self.assertFalse(canary["enabled"])
            self.assertTrue(canary["review_only"])
            self.assertEqual(canary["target_model"], "gpt-5.4-mini")
            self.assertEqual(canary["policy_source"], "local-manual")
            self.assertEqual(canary["fallback"]["fallback_model"], "gpt-5.4")
            self.assertGreater(canary["promotion"]["estimated_savings_per_1000_calls_usd"], 0)
            self.assertEqual(canary["promotion"]["endpoint"], staged["endpoint"])

            draft_yaml = yaml.safe_load((Path(staged["workspace"]) / "sections" / "routing_rules.yaml").read_text(encoding="utf-8"))
            self.assertFalse(draft_yaml["openai_canary"]["enabled"])
            self.assertEqual(draft_yaml["openai_canary"]["target_model"], "gpt-5.4-mini")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw prompt must not appear", rendered)
        self.assertNotIn("secret-openai-session-id", rendered)

    def test_cli_stages_only_top_ranked_gpt54_pass_through_candidate_by_default(self) -> None:
        for _ in range(8):
            self._log_openai_call(
                requested_model="gpt-5.4",
                category="chat",
                text_chars=1200,
                cost_baseline_usd=0.006,
            )
        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                path="/v1/chat/completions",
                source_surface="openai_chat",
                endpoint="chat_completions",
                category="summary",
                text_chars=900,
                cost_baseline_usd=0.003,
            )

        stdout = io.StringIO()
        code = cli.openai_routing_canary_stage_cli(
            ["--db", self.db_path, "--workspace", str(Path(self.tmpdir.name) / "drafts")],
            stdout=stdout,
        )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(payload["summary"]["eligible_candidate_count"], 2)
        self.assertEqual(payload["summary"]["staged_count"], 1)
        self.assertEqual(payload["summary"]["omission_reason_counts"], [{"value": "lower-ranked-candidate-not-staged", "count": 1}])
        staged = payload["staged_drafts"][0]
        self.assertEqual(staged["requested_model"], "gpt-5.4")
        self.assertEqual(staged["target_model"], "gpt-5.4-mini")
        self.assertEqual(staged["matched_count"], 8)
        self.assertEqual(staged["endpoint"], "responses")
        self.assertGreater(staged["estimated_savings_per_1000_calls_usd"], 0)
        self.assertEqual(staged["projected_cohort_counts"]["matched"], 8)
        self.assertEqual(staged["review_intent"]["routing_change_mode"], "draft-only")
        self.assertFalse(staged["review_intent"]["active_policy_changed"])
        self.assertEqual(staged["review_intent"]["matched_count"], 8)
        self.assertGreater(staged["review_intent"]["projected_canary_applied_count"], 0)
        self.assertGreater(staged["review_intent"]["projected_canary_holdout_count"], 0)
        self.assertTrue(staged["review_intent"]["safety_stop_enabled"])
        self.assertTrue(staged["review_intent"]["fallback_enabled"])
        self.assertTrue(staged["review_intent"]["privacy_proof"]["metadata_only"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw prompt must not appear", rendered)
        self.assertNotIn("secret-openai-session-id", rendered)

    def test_stages_ranked_gpt54_pass_through_bucket_with_projected_cohort_coverage(self) -> None:
        report = {
            "schema": "agentflow.pass_through_routing_activation_candidates.v1",
            "generated_at": utc_now(),
            "summary": {
                "pass_through_rows": 223,
                "top_actionability": "actionable",
                "top_requested_model": "gpt-5.4",
                "top_candidate_target_model": "gpt-5.4-mini",
            },
            "buckets": [{
                "rank": 1,
                "provider": "openai",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "unknown",
                "workflow_phase": "unknown",
                "sample_count": 223,
                "actionability": "actionable",
                "candidate_target_model": "gpt-5.4-mini",
                "required_local_executor": "openai-routing-canary",
                "candidate_reason": "gpt-5.4 canary can evaluate gpt-5.4-mini on local metadata cohorts",
                "estimated_savings_per_1000_calls_usd": 4.375,
                "openai_canary_lifecycle_evidence": {
                    "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
                    "status": "no-openai-canary-metadata",
                    "observed_count": 0,
                    "cohort_counts": {
                        "canary_applied": 0,
                        "canary_holdout": 0,
                        "safety_stopped": 0,
                    },
                    "blocker_codes": [
                        "missing-applied-coverage",
                        "missing-canary-lifecycle-evidence",
                        "missing-holdout-coverage",
                    ],
                },
            }],
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }

        result = asyncio.run(stage_openai_routing_canary_drafts(
            report,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
            canary_fraction=0.05,
            holdout_fraction=0.10,
        ))

        self.assertTrue(result["ok"], json.dumps(result, indent=2, sort_keys=True))
        self.assertEqual(result["summary"]["staged_count"], 1)
        self.assertEqual(result["summary"]["projected_canary_applied_count"], 12)
        self.assertEqual(result["summary"]["projected_canary_holdout_count"], 23)
        staged = result["staged_drafts"][0]
        self.assertEqual(staged["source_surface"], "openai_provider_request")
        self.assertEqual(staged["endpoint"], "responses")
        self.assertEqual(staged["requested_model"], "gpt-5.4")
        self.assertEqual(staged["target_model"], "gpt-5.4-mini")
        self.assertEqual(staged["matched_count"], 223)
        self.assertEqual(staged["projected_cohort_counts"]["matched"], 223)
        self.assertEqual(staged["projected_cohort_counts"]["canary_applied"], 12)
        self.assertEqual(staged["projected_cohort_counts"]["canary_holdout"], 23)
        self.assertEqual(staged["review_intent"]["status"], "ready-for-operator-review")
        self.assertEqual(staged["review_intent"]["routing_change_mode"], "draft-only")
        self.assertFalse(staged["review_intent"]["active_policy_changed"])
        self.assertEqual(staged["review_intent"]["matched_count"], 223)
        self.assertEqual(staged["review_intent"]["projected_canary_applied_count"], 12)
        self.assertEqual(staged["review_intent"]["projected_canary_holdout_count"], 23)
        self.assertTrue(staged["review_intent"]["safety_stop_enabled"])
        self.assertTrue(staged["review_intent"]["fallback_enabled"])
        self.assertEqual(staged["fallback_metadata"]["fallback_model"], "gpt-5.4")
        self.assertTrue(staged["review_intent"]["privacy_proof"]["metadata_only"])
        self.assertFalse(staged["review_intent"]["privacy_proof"]["raw_prompts_included"])
        lifecycle = staged["projected_openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.openai_routing_canary_projected_lifecycle_coverage.v1")
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 12)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 23)
        self.assertGreater(lifecycle["coverage"]["applied_rate"], 0)
        self.assertGreater(lifecycle["coverage"]["holdout_rate"], 0)
        inference = staged["aggregate_inference"]
        self.assertTrue(inference["source_surface_inferred"])
        self.assertTrue(inference["endpoint_inferred"])
        self.assertTrue(inference["category_inferred"])
        self.assertEqual(inference["inference_reason"], "aggregate-openai-canary-bucket")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw prompt", rendered)
        self.assertNotIn("secret-openai-session-id", rendered)

    def test_pass_through_bucket_with_safety_stop_is_omitted_with_counts(self) -> None:
        report = {
            "schema": "agentflow.pass_through_routing_activation_candidates.v1",
            "generated_at": utc_now(),
            "buckets": [{
                "rank": 1,
                "provider": "openai",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "unknown",
                "sample_count": 20,
                "actionability": "actionable",
                "candidate_target_model": "gpt-5.4-mini",
                "required_local_executor": "openai-routing-canary",
                "estimated_savings_per_1000_calls_usd": 4.375,
                "openai_canary_lifecycle_evidence": {
                    "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
                    "status": "matched",
                    "observed_count": 6,
                    "cohort_counts": {
                        "canary_applied": 4,
                        "canary_holdout": 1,
                        "safety_stopped": 1,
                    },
                    "error_count": 1,
                    "retry_count": 0,
                    "fallback_count": 0,
                    "blocker_codes": ["safety-stop-observed", "error-observed"],
                },
            }],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        result = asyncio.run(stage_openai_routing_canary_drafts(
            report,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
        ))

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"]["staged_count"], 0)
        self.assertEqual(result["summary"]["omission_reason_counts"], [{"value": "safety-stop-observed", "count": 1}])
        self.assertEqual(result["omitted"][0]["reason"], "safety-stop-observed")
        self.assertIn("safety-stop-observed", result["omitted"][0]["blocker_codes"])
        self.assertIn("error-observed", result["omitted"][0]["blocker_codes"])
        self.assertFalse(result["provider_calls_made"])

    def test_applies_reviewed_openai_routing_canary_draft_to_local_yaml_with_holdout(self) -> None:
        report = {
            "schema": "agentflow.pass_through_routing_activation_candidates.v1",
            "generated_at": utc_now(),
            "buckets": [{
                "rank": 1,
                "provider": "openai",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "unknown",
                "sample_count": 242,
                "actionability": "actionable",
                "candidate_target_model": "gpt-5.4-mini",
                "required_local_executor": "openai-routing-canary",
                "estimated_savings_per_1000_calls_usd": 4.375,
                "openai_canary_lifecycle_evidence": {
                    "schema": "agentflow.openai_routing_canary_lifecycle_evidence.v1",
                    "status": "no-openai-canary-metadata",
                    "cohort_counts": {
                        "canary_applied": 0,
                        "canary_holdout": 0,
                        "safety_stopped": 0,
                    },
                    "blocker_codes": [
                        "missing-applied-coverage",
                        "missing-canary-lifecycle-evidence",
                        "missing-holdout-coverage",
                    ],
                },
            }],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }
        staged = asyncio.run(stage_openai_routing_canary_drafts(
            report,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
            canary_fraction=0.05,
            holdout_fraction=0.10,
        ))
        bundle_path = Path(staged["staged_drafts"][0]["bundle_path"])
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        config_dir = Path(self.tmpdir.name) / "config"

        applied = apply_openai_routing_canary_draft(bundle, config_dir=config_dir, dry_run=False)

        self.assertTrue(applied["ok"], json.dumps(applied, indent=2, sort_keys=True))
        self.assertTrue(applied["wrote_policy_files"])
        self.assertFalse(applied["provider_calls_made"])
        self.assertFalse(applied["managed_server_calls_made"])
        self.assertEqual(applied["summary"]["projected_canary_applied_count"], 13)
        self.assertEqual(applied["summary"]["projected_canary_holdout_count"], 25)
        self.assertEqual(applied["summary"]["estimated_savings_per_1000_calls_usd"], 4.375)
        self.assertEqual(applied["summary"]["error_count"], 0)
        self.assertEqual(applied["summary"]["retry_count"], 0)
        self.assertEqual(applied["summary"]["fallback_count"], 0)
        self.assertEqual(applied["summary"]["stale_evidence_count"], 0)
        self.assertEqual(applied["summary"]["safety_stopped_count"], 0)

        data = yaml.safe_load((config_dir / "routing_rules.yaml").read_text(encoding="utf-8"))
        canary = data["openai_canary"]
        self.assertTrue(canary["enabled"])
        self.assertFalse(canary["review_only"])
        self.assertEqual(canary["model_pattern"], "gpt-5.4")
        self.assertEqual(canary["target_model"], "gpt-5.4-mini")
        self.assertEqual(canary["canary_fraction"], 0.05)
        self.assertEqual(canary["holdout_fraction"], 0.10)
        lifecycle = canary["promotion"]["projected_openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 13)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 25)
        self.assertEqual(lifecycle["estimated_savings_per_1000_calls_usd"], 4.375)
        self.assertEqual(lifecycle["error_count"], 0)
        self.assertEqual(lifecycle["retry_count"], 0)
        self.assertEqual(lifecycle["fallback_count"], 0)
        self.assertFalse(lifecycle["stale_evidence"]["stale"])

        rendered = json.dumps(applied, sort_keys=True) + json.dumps(data, sort_keys=True)
        self.assertNotIn("raw prompt must not appear", rendered)
        self.assertNotIn("secret-openai-session-id", rendered)

    def test_apply_refuses_safety_stopped_openai_routing_canary_draft_without_writing(self) -> None:
        report = {
            "schema": "agentflow.pass_through_routing_activation_candidates.v1",
            "generated_at": utc_now(),
            "buckets": [{
                "rank": 1,
                "provider": "openai",
                "source_surface": "unknown",
                "endpoint": "unknown",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "category": "unknown",
                "sample_count": 80,
                "actionability": "actionable",
                "candidate_target_model": "gpt-5.4-mini",
                "required_local_executor": "openai-routing-canary",
                "estimated_savings_per_1000_calls_usd": 4.375,
            }],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }
        staged = asyncio.run(stage_openai_routing_canary_drafts(
            report,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
        ))
        bundle = json.loads(Path(staged["staged_drafts"][0]["bundle_path"]).read_text(encoding="utf-8"))
        canary = bundle["policies"]["routing"]["openai"]["canary"]
        lifecycle = canary["promotion"]["projected_openai_canary_lifecycle_evidence"]
        lifecycle["cohort_counts"]["safety_stopped"] = 1
        lifecycle["error_count"] = 1
        lifecycle["blocker_codes"] = ["safety-stop-observed", "error-observed"]
        config_dir = Path(self.tmpdir.name) / "config"

        applied = apply_openai_routing_canary_draft(bundle, config_dir=config_dir, dry_run=False)

        self.assertFalse(applied["ok"])
        self.assertFalse(applied["wrote_policy_files"])
        self.assertFalse((config_dir / "routing_rules.yaml").exists())
        self.assertEqual(applied["omitted"][0]["reason"], "safety-stop-observed")
        self.assertEqual(applied["summary"]["safety_stopped_count"], 1)
        self.assertEqual(applied["summary"]["error_count"], 1)
        self.assertEqual(applied["summary"]["estimated_savings_per_1000_calls_usd"], 4.375)

    def test_stages_promotion_blocker_routing_recommendation_as_canary_lifecycle_plan(self) -> None:
        recommendations = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [{
                "recommendation_id": "promotion-blocker-next-action:openai:routing:canary-gpt54-mini",
                "rank": 1,
                "status": "recommended",
                "local_action_family": "routing",
                "candidate_family": "provider-routing-rule",
                "source_surface": "openai_provider_request",
                "provider_family": "openai",
                "provider_endpoint": "responses",
                "blocker_family": "canary-missing",
                "blocker_reason_codes": [
                    "missing-canary-lifecycle-evidence",
                    "missing-applied-coverage",
                    "missing-holdout-coverage",
                ],
                "blocker_count": 238,
                "recommendation_type": "collect-canary-lifecycle-evidence",
                "next_action": "collect-local-canary-evidence",
                "expected_local_executor": "openai-routing-canary",
                "file_backed_policy_representation": {
                    "exists": True,
                    "policy_section": "routing",
                    "policy_source": "local-manual",
                    "rule_file": "routing_rules.yaml",
                },
                "local_executor_compatibility": {
                    "status": "compatible",
                    "local_action_family": "routing",
                },
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "category": "unknown",
                "text_bucket": "unknown",
                "token_bucket": "unknown",
                "confidence": 0.93,
                "projected_savings_usd": 1.04125,
                "evidence_summary": {
                    "record_count": 238,
                    "candidate_count": 238,
                    "promotion_status": "blocked",
                    "capability_checked": "canary_holdout",
                    "capability_status": "observe-only",
                },
                "prompt": "raw promotion blocker prompt secret",
                "provider_body": {"input": "raw promotion blocker provider body secret"},
                "request_id": "promotion-blocker-request-id-secret",
                "session_id": "promotion-blocker-session-id-secret",
                "cache_key": "promotion-blocker-cache-key-secret",
            }],
        }
        review = build_promotion_blocker_recommendation_review(recommendations, limit=10)

        result = asyncio.run(stage_openai_routing_canary_drafts(
            review,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
            canary_fraction=0.05,
            holdout_fraction=0.10,
        ))

        self.assertTrue(result["ok"], json.dumps(result, indent=2, sort_keys=True))
        self.assertEqual(result["source_report_schema"], "agentflow.promotion_blocker_recommendation_review.v1")
        self.assertEqual(result["summary"]["staged_count"], 1)
        self.assertEqual(result["summary"]["projected_canary_applied_count"], 12)
        self.assertEqual(result["summary"]["projected_canary_holdout_count"], 24)
        staged = result["staged_drafts"][0]
        self.assertEqual(staged["requested_model"], "gpt-5.4")
        self.assertEqual(staged["target_model"], "gpt-5.4-mini")
        self.assertEqual(staged["expected_local_executor"], "openai-routing-canary")
        self.assertEqual(staged["projected_cohort_counts"]["matched"], 238)
        self.assertEqual(staged["projected_cohort_counts"]["canary_applied"], 12)
        self.assertEqual(staged["projected_cohort_counts"]["canary_holdout"], 24)
        self.assertTrue(staged["safety_stop_metadata"]["enabled"])
        self.assertEqual(staged["safety_stop_metadata"]["max_error_rate"], 0.03)
        lifecycle = staged["projected_openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 12)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 24)
        self.assertGreater(lifecycle["coverage"]["applied_rate"], 0)
        self.assertGreater(lifecycle["coverage"]["holdout_rate"], 0)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("raw promotion blocker prompt secret", rendered)
        self.assertNotIn("raw promotion blocker provider body secret", rendered)
        self.assertNotIn("promotion-blocker-request-id-secret", rendered)
        self.assertNotIn("promotion-blocker-session-id-secret", rendered)
        self.assertNotIn("promotion-blocker-cache-key-secret", rendered)

    def test_promotion_blocker_recommendation_without_file_backed_policy_is_noop(self) -> None:
        review = {
            "schema": "agentflow.promotion_blocker_recommendation_review.v1",
            "generated_at": utc_now(),
            "candidates": [{
                "schema": "agentflow.promotion_blocker_review_candidate.v1",
                "recommendation_id": "promotion-blocker-next-action:openai:routing:missing-policy",
                "rank": 1,
                "status": "recommended",
                "recommendation_type": "collect-canary-lifecycle-evidence",
                "local_action_family": "routing",
                "candidate_family": "provider-routing-rule",
                "source_surface": "openai_provider_request",
                "provider_family": "openai",
                "provider_endpoint": "responses",
                "blocker_family": "canary-missing",
                "blocker_reason_codes": ["missing-canary-lifecycle-evidence"],
                "blocker_count": 20,
                "next_action": "collect-local-canary-evidence",
                "expected_local_executor": "openai-routing-canary",
                "file_backed_policy_representation": {
                    "exists": False,
                    "reason": "missing-file-backed-local-policy",
                },
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "evidence_summary": {"record_count": 20},
                "projected_savings_usd": 0.1,
                "privacy": {"metadata_only": True},
            }],
            "privacy": {"metadata_only": True},
        }

        result = asyncio.run(stage_openai_routing_canary_drafts(
            review,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
        ))

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"]["staged_count"], 0)
        self.assertEqual(result["summary"]["omission_reason_counts"], [{"value": "missing-file-backed-local-policy", "count": 1}])
        self.assertEqual(result["omitted"][0]["reason"], "missing-file-backed-local-policy")
        self.assertFalse((Path(self.tmpdir.name) / "drafts").exists())
        self.assertFalse(result["provider_calls_made"])

    def test_unsafe_candidates_are_omitted_with_machine_readable_reasons(self) -> None:
        base = {
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "requested_model": "gpt-5.4",
            "target_model": "gpt-5.4-mini",
            "category": "chat",
            "matched_count": 6,
            "blocked_count": 0,
            "projected_savings_usd": 0.01,
            "estimated_baseline_cost_usd": 0.02,
            "text_bucket": "lt-1_5k",
            "token_bucket": "lt-1k",
            "input_tokens": 1800,
        }
        candidates = [
            {**base, "candidate_id": "high-error", "error_rate": 0.2},
            {**base, "candidate_id": "high-retry", "retry_rate": 0.5},
            {**base, "candidate_id": "unknown-pricing", "requested_model": "unknown-openai-model"},
            {**base, "candidate_id": "missing-baseline", "projected_savings_usd": 0.0},
            {**base, "candidate_id": "unsupported-target", "target_model": "unsupported-target-model"},
            {**base, "candidate_id": "tool-blocker", "has_tools": True},
            {**base, "candidate_id": "already-routed", "current_routed_count": 1},
        ]
        report = {
            "schema": "agentflow.openai_routing_opportunity.v1",
            "generated_at": utc_now(),
            "candidates": candidates,
            "privacy": {"metadata_only": True, "raw_prompts_included": False},
        }
        result = asyncio.run(stage_openai_routing_canary_drafts(
            report,
            workspace=str(Path(self.tmpdir.name) / "drafts"),
        ))

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"]["staged_count"], 0)
        reasons = {item["reason"] for item in result["omitted"]}
        self.assertEqual(
            reasons,
            {
                "high-error-rate",
                "high-retry-rate",
                "unknown-pricing",
                "missing-baseline-cost",
                "unsupported-target-model",
                "tool-safety-blocker",
                "already-routed",
            },
        )
        self.assertFalse((Path(self.tmpdir.name) / "drafts").exists())
        self.assertFalse(result["provider_calls_made"])

    def test_cli_accepts_report_stdin_and_rejects_raw_payloads_without_leaking_values(self) -> None:
        raw_report = {
            "schema": "agentflow.openai_routing_opportunity.v1",
            "candidates": [{
                "candidate_id": "raw-candidate",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "raw_prompt": "secret raw prompt",
                "request_id": "req_secret",
            }],
        }
        stdout = io.StringIO()
        code = cli.openai_routing_canary_stage_cli(
            ["-", "--workspace", str(Path(self.tmpdir.name) / "drafts")],
            stdin=io.StringIO(json.dumps(raw_report)),
            stdout=stdout,
        )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "raw_payload_rejected")
        self.assertFalse((Path(self.tmpdir.name) / "drafts").exists())
        self.assertNotIn("secret raw prompt", stdout.getvalue())
        self.assertNotIn("req_secret", stdout.getvalue())

    def test_cli_stages_from_promotion_blocker_review_stdin(self) -> None:
        recommendations = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [{
                "recommendation_id": "promotion-blocker-next-action:openai:routing:cli-canary",
                "rank": 1,
                "status": "recommended",
                "local_action_family": "routing",
                "candidate_family": "provider-routing-rule",
                "source_surface": "openai_provider_request",
                "provider_family": "openai",
                "provider_endpoint": "responses",
                "blocker_family": "canary-missing",
                "blocker_reason_codes": ["missing-canary-lifecycle-evidence"],
                "blocker_count": 30,
                "recommendation_type": "collect-canary-lifecycle-evidence",
                "next_action": "collect-local-canary-evidence",
                "expected_local_executor": "openai-routing-canary",
                "file_backed_policy_representation": {
                    "exists": True,
                    "policy_section": "routing",
                    "policy_source": "local-manual",
                    "rule_file": "routing_rules.yaml",
                },
                "local_executor_compatibility": {"status": "compatible", "local_action_family": "routing"},
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "evidence_summary": {"record_count": 30},
                "projected_savings_usd": 0.13,
                "prompt": "raw cli canary prompt secret",
            }],
        }

        stdout = io.StringIO()
        code = cli.openai_routing_canary_stage_cli(
            [
                "--promotion-blocker-review",
                "-",
                "--workspace",
                str(Path(self.tmpdir.name) / "drafts"),
                "--top-candidates",
                "1",
            ],
            stdin=io.StringIO(json.dumps(recommendations)),
            stdout=stdout,
        )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(payload["summary"]["staged_count"], 1)
        staged = payload["staged_drafts"][0]
        self.assertEqual(staged["requested_model"], "gpt-5.4")
        self.assertEqual(staged["target_model"], "gpt-5.4-mini")
        self.assertEqual(staged["expected_local_executor"], "openai-routing-canary")
        self.assertNotIn("raw cli canary prompt secret", stdout.getvalue())

    def test_cli_queue_feedback_emits_metadata_only_activation_lifecycle_row(self) -> None:
        report = {
            "schema": "agentflow.openai_routing_opportunity.v1",
            "generated_at": utc_now(),
            "candidates": [{
                "candidate_id": "activation-public-candidate",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model": "gpt-5.4",
                "target_model": "gpt-5.4-mini",
                "category": "chat",
                "matched_count": 6,
                "blocked_count": 0,
                "projected_savings_usd": 0.01,
                "estimated_baseline_cost_usd": 0.02,
                "text_bucket": "lt-1_5k",
                "token_bucket": "lt-1k",
                "input_tokens": 1800,
            }],
            "privacy": {"metadata_only": True, "raw_prompts_included": False},
        }
        stdout = io.StringIO()
        with patch.dict(
            "os.environ",
            {
                "AGENTFLOW_RECOMMENDATION_ENABLED": "0",
                "AGENTFLOW_RECOMMENDATION_SERVER_URL": "",
            },
            clear=False,
        ):
            code = cli.openai_routing_canary_stage_cli(
                [
                    "-",
                    "--workspace",
                    str(Path(self.tmpdir.name) / "drafts"),
                    "--db",
                    self.db_path,
                    "--queue-feedback",
                ],
                stdin=io.StringIO(json.dumps(report)),
                stdout=stdout,
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(code, 0, stdout.getvalue())
        self.assertEqual(payload["lifecycle_feedback"]["status"], "queued")
        self.assertFalse(payload["lifecycle_feedback"]["payload_included"])
        rows = self.store.managed_outcome_feedback_payload_rows(source_surface=LIFECYCLE_SOURCE_SURFACE, limit=10)
        self.assertEqual(len(rows), 1)
        event = json.loads(rows[0]["payload_json"])
        self.assertEqual(event["schema"], "agentflow.openai_optimization_lifecycle_feedback.v1")
        self.assertEqual(event["event_type"], "activation_staged_optimization_lifecycle")
        self.assertEqual(event["event_phase"], "stage")
        self.assertEqual(event["lifecycle_state"], "healthy_canary")
        self.assertEqual(event["family_events"][0]["action_family"], "routing")
        self.assertEqual(event["family_events"][0]["cohort"], "staged")
        self.assertEqual(event["family_events"][0]["candidate_id"], "activation-public-candidate")
        self.assertFalse(event["privacy"]["raw_prompts_included"])
        self.assertFalse(event["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(event["privacy"]["raw_tool_payloads_included"])
        self.assertFalse(event["privacy"]["request_ids_included"])
        self.assertFalse(event["privacy"]["raw_session_ids_included"])
        self.assertFalse(event["privacy"]["cache_keys_included"])
        self.assertFalse(event["privacy"]["tenant_ids_included"])
        self.assertFalse(event["privacy"]["file_paths_included"])
        assert_managed_egress_safe(event)

        status = feedback.managed_feedback_status_result(
            self.store,
            source_surface=LIFECYCLE_SOURCE_SURFACE,
            sample_limit=5,
        )
        lifecycle = status["openai_optimization_lifecycle"]
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertIn({"value": "staged", "count": 1}, lifecycle["cohort_breakdown"])
        rendered = json.dumps(status, sort_keys=True) + stdout.getvalue()
        self.assertNotIn("raw prompt must not appear", rendered)
        self.assertNotIn("secret-openai-session-id", rendered)


if __name__ == "__main__":
    unittest.main()
