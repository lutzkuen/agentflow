from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from agentflow_proxy import cli
from agentflow_proxy.managed_routing_pathway_outcomes import build_local_routing_pathway_outcome_feedback
from agentflow_proxy.store import Store, stable_json


class ManagedRoutingPathwayOutcomeFeedbackTests(unittest.TestCase):
    def _source(self) -> dict[str, object]:
        generated_at = "2026-06-20T01:00:00+00:00"
        return {
            "schema": "agentflow.policy_decision.v1",
            "generated_at": generated_at,
            "routing_pathway_matrix": {
                "schema": "agentflow.routing_pathway_matrix.v1",
                "generated_at": generated_at,
                "status": "recommended",
                "pathways": [
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 1,
                        "pathway_id": "pathway-openai-tool-light",
                        "pathway_type": "adjacent_downroute",
                        "source_surface": "openai_responses",
                        "app_family": "generic_openai",
                        "category": "tool-light",
                        "workflow_phase": "tool-execution",
                        "requested_model": "gpt-5.4",
                        "requested_model_family": "gpt-5",
                        "target_model": "gpt-5.4-mini",
                        "target_model_family": "gpt-5-mini",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "2k_8k_tokens",
                        "suggested_next_action": "shadow",
                        "activation_recommendation": True,
                    },
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 2,
                        "pathway_id": "pathway-codex-summary",
                        "pathway_type": "adjacent_downroute",
                        "source_surface": "codex_turn",
                        "app_family": "codex",
                        "category": "codex-turn",
                        "workflow_phase": "summary",
                        "requested_model": "gpt-5.5",
                        "requested_model_family": "gpt-5",
                        "target_model": "gpt-5.3-codex",
                        "target_model_family": "gpt-5-codex",
                        "text_bucket": "lt_2k_chars",
                        "token_bucket": "lt_500_tokens",
                        "suggested_next_action": "canary",
                        "activation_recommendation": True,
                    },
                ],
            },
        }

    def _log_openai_canary_call(self, store: object, *, created_at: str, cohort: str, routed_model: str) -> None:
        status = "applied" if cohort == "canary_applied" else "holdout"
        store.log_call(
            id=str(uuid4()),
            created_at=created_at,
            path="/v1/responses",
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model=routed_model,
            requested_model_family="gpt-5",
            routed_model_family="gpt-5-mini",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=120,
            input_tokens_est=1500,
            output_tokens_est=200,
            actual_input_tokens=1500,
            actual_output_tokens=200,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            category="tool-light",
            retry_count=0,
            routing_json=stable_json(
                {
                    "requested_model": "gpt-5.4",
                    "routed_model": routed_model,
                    "source_surface": "openai_responses",
                    "category": "tool-light",
                    "workflow_phase": "tool-execution",
                    "openai_canary": {
                        "status": status,
                        "cohort": cohort,
                        "reason": cohort,
                        "requested_model": "gpt-5.4",
                        "target_model": "gpt-5.4-mini",
                        "category": "tool-light",
                        "source_surface": "openai_responses",
                    },
                }
            ),
            cache_json=stable_json({"status": "miss"}),
        )

    def _log_failed_semantic_comparisons(self, store: object) -> None:
        for index in range(20):
            store.log_routing_experiment(
                id=str(uuid4()),
                call_id=f"call-{index}",
                created_at=f"2026-06-20T01:{index:02d}:00+00:00",
                provider="openai",
                source_surface="openai_responses",
                stream=0,
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                primary_model="gpt-5.4",
                shadow_model="gpt-5.4-mini",
                category="tool-light",
                routing_reason="pathway-matrix-shadow",
                input_tokens_est=1000,
                primary_status_code=200,
                shadow_status_code=200,
                output_similarity=0.5,
                passed_threshold=0,
            )

    def test_report_records_openai_and_codex_pathway_outcomes_without_raw_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self._log_openai_canary_call(
                    store,
                    created_at="2026-06-20T01:01:00+00:00",
                    cohort="canary_applied",
                    routed_model="gpt-5.4-mini",
                )
                self._log_openai_canary_call(
                    store,
                    created_at="2026-06-20T01:02:00+00:00",
                    cohort="canary_holdout",
                    routed_model="gpt-5.4",
                )
                self._log_failed_semantic_comparisons(store)
                store.log_codex_app_event(
                    id=str(uuid4()),
                    created_at="2026-06-20T01:03:00+00:00",
                    direction="client_to_server",
                    method="turn/start",
                    input_text_chars=640,
                    result_chars=160,
                    error_code=0,
                    routing_json=stable_json({"requested_model": "gpt-5.5", "workflow_phase": "summary"}),
                    event_window_json=stable_json(
                        {
                            "workflow_phase": "summary",
                            "model_state": {"normalized_model": "gpt-5.5"},
                        }
                    ),
                )

                report = build_local_routing_pathway_outcome_feedback(
                    store,
                    self._source(),
                    limit=100,
                    stale_after_hours=72,
                )
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.local_routing_pathway_outcome_feedback.v1")
        self.assertEqual(report["status"], "tracked")
        self.assertEqual(report["summary"]["outcome_count"], 2)
        self.assertEqual(report["summary"]["applied_count"], 1)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])

        openai = next(row for row in report["outcomes"] if row["source_surface"] == "openai_responses")
        self.assertTrue(openai["candidate_fingerprint"].startswith("routing-pathway-candidate:"))
        self.assertEqual(openai["applied_count"], 1)
        self.assertEqual(openai["holdout_count"], 1)
        self.assertEqual(openai["status"], "regressed")
        self.assertEqual(openai["blocker_status"], "semantic-quality-regression-observed")
        self.assertEqual(openai["recommended_next_action"], "review-openai-routing-canary-blockers")

        codex = next(row for row in report["outcomes"] if row["source_surface"] == "codex_turn")
        self.assertEqual(codex["matched_count"], 1)
        self.assertEqual(codex["applied_count"], 0)
        self.assertEqual(codex["holdout_count"], 0)
        self.assertEqual(codex["status"], "observed")
        self.assertEqual(codex["blocker_status"], "local-routing-pathway-observed-missing-coverage")

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in ('"raw_prompt"', '"provider_body"', '"request_id"', '"session_id"', '"cache_key"', '"file_path"'):
            self.assertNotIn(forbidden, rendered)

    def test_cli_reads_matrix_json_and_local_db(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "agentflow.sqlite3"
            source_path = Path(tmp) / "decision.json"
            source_path.write_text(json.dumps(self._source()), encoding="utf-8")
            store = Store(str(db_path))
            try:
                self._log_openai_canary_call(
                    store,
                    created_at="2026-06-20T01:01:00+00:00",
                    cohort="canary_applied",
                    routed_model="gpt-5.4-mini",
                )
            finally:
                store.conn.close()
            output = io.StringIO()

            code = cli.managed_routing_pathway_outcomes_cli(
                ["--decision-json", str(source_path), "--db", str(db_path), "--limit", "100"],
                stdout=output,
            )

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["outcome_count"], 2)
        self.assertEqual(payload["egress_guard"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
