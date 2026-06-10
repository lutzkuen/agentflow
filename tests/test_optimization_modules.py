import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.optimization import feedback, openai_outcomes
from agentflow_proxy.optimization_eval_plan import _add_common
from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report
from agentflow_proxy.optimization_shadow_eval import run_optimization_shadow_eval
from agentflow_proxy.store import Store, stable_json


class FakeFeedbackStore:
    def managed_outcome_feedback_summary(self, *, source_surface=None):
        self.source_surface = source_surface
        return [{"status": "queued", "count": 1}]

    def managed_outcome_feedback_rows(self, *, source_surface=None, limit=10000):
        self.row_limit = limit
        return [
            {
                "id": "queue-1",
                "created_at": "2026-06-08T10:00:00+00:00",
                "updated_at": "2026-06-08T10:00:01+00:00",
                "source_surface": "openai_responses",
                "endpoint": "/v1/optimization-units/77/outcome",
                "optimization_unit_id": 77,
                "status": "queued",
                "attempts": 0,
                "next_attempt_at": "2026-06-08T10:00:02+00:00",
                "last_status_code": None,
                "sent_at": None,
            }
        ]

    def due_managed_outcome_feedback(self, *, limit, source_surface=None):
        self.due_limit = limit
        return self.managed_outcome_feedback_rows(source_surface=source_surface, limit=limit)


class FakeOutcomeStore:
    def __init__(self):
        self.updated = []

    def update_call_routing_json(self, call_id, routing_json):
        self.updated.append((call_id, json.loads(routing_json)))


class OptimizationModuleTests(unittest.TestCase):
    def _assert_privacy_clean(self, payload):
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw eval prompt secret",
            "raw eval message secret",
            "raw eval content secret",
            "raw eval request secret",
            "raw eval response secret",
            "raw eval output secret",
            "raw eval tool secret",
            "/tmp/raw-eval-secret.py",
            "eval-cache-key-secret",
            "eval-request-id-secret",
            "eval-session-id-secret",
            "sk-eval-secret",
        ):
            self.assertNotIn(forbidden, rendered)
        for forbidden_key in (
            '"api_key"',
            '"cache_key"',
            '"content"',
            '"file_path"',
            '"messages"',
            '"output"',
            '"prompt"',
            '"raw_request"',
            '"raw_response"',
            '"request_id"',
            '"session_id"',
            '"tool_payload"',
        ):
            self.assertNotIn(forbidden_key, rendered)

    def _dangerous_metadata(self):
        return {
            "safe_count": 3,
            "safe_score": 0.98,
            "prompt": "raw eval prompt secret",
            "messages": [{"role": "user", "content": "raw eval message secret"}],
            "content": "raw eval content secret",
            "raw_request": {"body": "raw eval request secret"},
            "raw_response": {"output": "raw eval response secret"},
            "tool_payload": {"command": "raw eval tool secret"},
            "file_path": "/tmp/raw-eval-secret.py",
            "cache_key": "eval-cache-key-secret",
            "request_id": "eval-request-id-secret",
            "session_id": "eval-session-id-secret",
            "api_key": "sk-eval-secret",
            "nested": {
                "safe_reason": "offline-fixture-passed",
                "prompt": "raw eval prompt secret",
                "content": "raw eval content secret",
            },
        }

    def test_eval_plan_rows_scrub_raw_like_nested_evidence(self):
        rows = []
        _add_common(
            rows,
            candidate_id="eval-privacy-candidate",
            optimization_family="phase_routing",
            action_family="routing",
            source_surface="anthropic_messages",
            app_family="claude_code",
            workflow_phase="tool_execution",
            category="tool-result",
            candidate_target_model="claude-haiku-4-5-20251001",
            projected_savings_usd=0.0123,
            sample_count=5,
            current_canary_count=2,
            holdout_count=1,
            blocker_reason_codes=["local-replay-input-unavailable"],
            replayability_level="features_only",
            evidence=self._dangerous_metadata(),
        )

        row = rows[0]
        self.assertEqual(row["candidate_id"], "eval-privacy-candidate")
        self.assertEqual(row["sample_count"], 5)
        self.assertEqual(row["blocker_reason_codes"], ["local-replay-input-unavailable"])
        self.assertEqual(row["evidence"]["safe_count"], 3)
        self.assertEqual(row["evidence"]["nested"]["safe_reason"], "offline-fixture-passed")
        self.assertTrue(row["privacy"]["metadata_only"])
        self._assert_privacy_clean(row)

    def test_shadow_eval_output_and_stored_result_scrub_raw_like_fixture_fields(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-privacy-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "projected_savings_usd": 0.02,
                    "shadow_eval_fixture": {
                        **self._dangerous_metadata(),
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.98,
                        "quality_score": 0.97,
                        "baseline_cost_usd": 0.03,
                        "candidate_cost_usd": 0.01,
                    },
                    "request_json": {"messages": [{"content": "raw eval message secret"}]},
                    "session_id": "eval-session-id-secret",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                result = run_optimization_shadow_eval(plan, store=store)
                stored = store.conn.execute(
                    "select result_json, score_json, cost_json from optimization_eval_results"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(result["results"][0]["candidate_id"], "shadow-privacy-candidate")
        self.assertEqual(result["results"][0]["status_class"], "pass")
        self.assertIn("offline-fixture-passed", result["results"][0]["reason_codes"])
        self.assertEqual(result["results"][0]["score_summary"]["output_similarity"], 0.98)
        self.assertEqual(json.loads(stored["score_json"])["quality_score"], 0.97)
        self.assertEqual(json.loads(stored["cost_json"])["observed_savings_usd"], 0.02)
        self._assert_privacy_clean(result)
        self._assert_privacy_clean(json.loads(stored["result_json"]))

    def test_promotion_report_omits_raw_like_plan_and_eval_result_fields(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "promotion-privacy-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "current_canary_count": 2,
                    "holdout_count": 1,
                    "sample_count": 3,
                    "projected_savings_usd": 0.02,
                    "evidence": {
                        **self._dangerous_metadata(),
                        "canary_evidence": {
                            "applied": {
                                "count": 2,
                                "error_rate": 0.0,
                                "retry_rate": 0.0,
                                "latency_avg_ms": 100,
                                "net_savings_usd": 0.02,
                                "tool_payload": {"command": "raw eval tool secret"},
                            },
                            "holdout": {
                                "count": 1,
                                "error_rate": 0.0,
                                "retry_rate": 0.0,
                                "latency_avg_ms": 120,
                                "prompt": "raw eval prompt secret",
                            },
                        },
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                store.log_optimization_eval_result(
                    id="promotion-privacy-result",
                    run_id="promotion-privacy-run",
                    created_at="2026-06-10T03:30:00+00:00",
                    candidate_id="promotion-privacy-candidate",
                    source_surface="anthropic_messages",
                    optimization_family="phase_routing",
                    action_family="routing",
                    status_class="pass",
                    reason_codes_json=stable_json(["offline-fixture-passed"]),
                    score_json=stable_json({"output_similarity": 0.98, "content": "raw eval content secret"}),
                    cost_json=stable_json({"projected_savings_usd": 0.02, "cache_key": "eval-cache-key-secret"}),
                    result_json=stable_json(self._dangerous_metadata()),
                )
                report = build_optimization_promotion_report(store, plan=plan)
            finally:
                store.conn.close()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "promotion-privacy-candidate")
        self.assertEqual(candidate["verdict"], "widen")
        self.assertIn("promotion-thresholds-met", candidate["reason_codes"])
        self.assertEqual(candidate["eval_evidence"]["pass_count"], 1)
        self.assertEqual(candidate["cohort_counts"]["canary_applied"], 2)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self._assert_privacy_clean(report)

    def test_feedback_status_result_is_metadata_only(self):
        result = feedback.managed_feedback_status_result(
            FakeFeedbackStore(),
            source_surface="openai_responses",
            sample_limit=5,
        )

        self.assertEqual(result["schema"], "agentflow.managed_feedback_status.v1")
        self.assertEqual(result["summary"]["queued"], 1)
        self.assertEqual(result["summary"]["due"], 1)
        self.assertFalse(result["due_samples"][0]["payload_included"])
        self.assertTrue(result["privacy"]["metadata_only"])

    def test_openai_outcome_summary_is_feature_only(self):
        routing_meta = {"reason": "test"}
        openai_outcomes.attach_openai_outcome_summary(
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            status_code=200,
            latency_ms=12,
            retry_count=0,
            input_tokens_est=20,
            output_tokens_est=4,
            actual_input_tokens=21,
            actual_output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            cache_meta={"status": "miss", "reason": "exact-miss"},
            crunch_meta={"changed": False},
            routing_meta=routing_meta,
            category="chat",
            session_id="raw session secret",
        )

        summary = routing_meta["openai_outcome_unit"]
        self.assertEqual(summary["schema"], "agentflow.openai_outcome_summary.v1")
        self.assertEqual(summary["source_surface"], "openai_responses")
        self.assertFalse(summary["raw_payload_included"])
        self.assertEqual(managed_egress_violations(summary), [])
        self.assertNotIn("raw session secret", json.dumps(summary, sort_keys=True))

    def test_openai_managed_outcome_feedback_records_queue_metadata(self):
        store = FakeOutcomeStore()
        routing_meta = {
            "reason": "test",
            "managed_recommendation": {
                "enabled": True,
                "optimization_unit_id": 77,
                "policy_id": "policy-1",
            },
        }
        queued = {
            "queued": True,
            "queue_id": "queue-1",
            "source_surface": "openai_responses",
        }

        async def fake_queue(store_obj, managed, outcome, *, source_surface):
            self.assertIs(store_obj, store)
            self.assertEqual(source_surface, "openai_responses")
            self.assertEqual(outcome["provider"], "openai")
            self.assertEqual(managed_egress_violations(outcome), [])
            return queued

        with patch("agentflow_proxy.optimization.openai_outcomes.queue_outcome_feedback", fake_queue):
            asyncio.run(
                openai_outcomes.record_managed_outcome_feedback(
                    store=store,
                    call_id="call-1",
                    path="/v1/responses",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-mini",
                    status_code=200,
                    latency_ms=12,
                    retry_count=0,
                    input_tokens_est=20,
                    output_tokens_est=4,
                    actual_input_tokens=21,
                    actual_output_tokens=5,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=None,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    cache_meta={"status": "miss", "reason": "exact-miss"},
                    crunch_meta={"changed": False},
                    routing_meta=routing_meta,
                    category="chat",
                    session_id="raw session secret",
                )
            )

        self.assertEqual(routing_meta["managed_recommendation"]["outcome_feedback"], queued)
        self.assertEqual(store.updated[0][0], "call-1")
        self.assertEqual(store.updated[0][1]["managed_recommendation"]["outcome_feedback"], queued)


if __name__ == "__main__":
    unittest.main()
