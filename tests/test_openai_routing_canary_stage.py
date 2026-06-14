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
from agentflow_proxy.openai_routing_canary_stage import stage_openai_routing_canary_drafts
from agentflow_proxy.openai_routing_report import build_openai_routing_report
from agentflow_proxy.optimization import feedback
from agentflow_proxy.policy_files import policy_file_snapshot, utc_now
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
        self.assertEqual(staged["endpoint"], "responses")
        self.assertGreater(staged["estimated_savings_per_1000_calls_usd"], 0)
        self.assertEqual(staged["projected_cohort_counts"]["matched"], 8)
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
        self.assertEqual(staged["projected_cohort_counts"]["matched"], 223)
        self.assertEqual(staged["projected_cohort_counts"]["canary_applied"], 12)
        self.assertEqual(staged["projected_cohort_counts"]["canary_holdout"], 23)
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
