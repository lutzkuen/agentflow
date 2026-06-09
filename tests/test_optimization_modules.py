import asyncio
import json
import unittest
from unittest.mock import patch

from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.optimization import feedback, openai_outcomes


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
