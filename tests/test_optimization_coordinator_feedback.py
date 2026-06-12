from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from agentflow_proxy.managed_egress import assert_managed_egress_safe
from agentflow_proxy.optimization import feedback
from agentflow_proxy.optimization_coordinator_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE,
    build_optimization_coordinator_lifecycle_feedback,
    queue_optimization_coordinator_lifecycle_feedback,
)
from agentflow_proxy.store import Store, stable_json, utc_now


def _decision(*, selected_family: str = "routing", holdout: bool = False) -> dict[str, object]:
    if holdout:
        selected_family = "none"
    suppressed_reason = "coordinator-holdout" if holdout else "conflicts-with-selected-family"
    return {
        "schema": "agentflow.optimization_coordinator.v1",
        "selected_family": selected_family,
        "selected_action_family": selected_family,
        "selected_candidate": None if selected_family == "none" else {
            "candidate_id": "routing candidate from /tmp/private-session",
            "rule_id": "routing-rule",
            "status": "eligible",
            "policy_source": "managed-recommended",
        },
        "suppressed_families": [
            {
                "family": "cache_replay",
                "status": "eligible",
                "candidate_id": "cache candidate with session id 123",
                "rule_id": "cache-rule",
                "reason_codes": [suppressed_reason],
            }
        ],
        "candidate_count": 2,
        "entry_count": 2,
        "reason_codes": ["coordinator-holdout"] if holdout else [],
        "source_surface": "openai_responses",
        "provider_family": "openai",
        "endpoint": "responses",
        "category": "tool-result",
        "phase": "tool-execution",
        "text_bucket": "8k_30k",
        "input_token_bucket": "2k_8k",
        "decision_hash": "sha256:" + "a" * 64,
        "privacy": {"metadata_only": True},
    }


class OptimizationCoordinatorFeedbackTests(unittest.TestCase):
    def test_builds_metadata_only_selected_suppressed_rollback_and_safety_stop_event(self) -> None:
        payload = build_optimization_coordinator_lifecycle_feedback(
            _decision(),
            status_code=429,
            retry_count=3,
            cost_est_usd=0.01,
            cost_baseline_usd=0.03,
            extra_family_events=[
                {
                    "family": "old_context_summary",
                    "lifecycle_status": "rollback",
                    "candidate_id": "summary-candidate",
                    "reason_codes": ["rollback-required"],
                },
                {
                    "family": "terminal_output_compaction",
                    "lifecycle_status": "safety_stop",
                    "candidate_id": "terminal-candidate",
                    "reason_codes": ["safety-stop-tripped"],
                },
            ],
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        assert_managed_egress_safe(payload)
        metadata = payload["metadata"]
        self.assertEqual(metadata["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(metadata["selected_family"], "routing")
        self.assertEqual(metadata["status_bucket"], "4xx")
        self.assertEqual(metadata["retry_bucket"], "gte_3")
        self.assertEqual(metadata["savings_bucket"], "0_01_0_10")
        statuses = {item["lifecycle_status"] for item in metadata["family_events"]}
        self.assertIn("selected", statuses)
        self.assertIn("suppressed", statuses)
        self.assertIn("rollback", statuses)
        self.assertIn("safety_stop", statuses)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("/tmp/private-session", rendered)
        self.assertNotIn("session id 123", rendered)
        self.assertFalse(metadata["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(metadata["privacy"]["cache_keys_included"])

    def test_queues_selected_holdout_suppressed_rollback_and_safety_stop_events(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                selected_meta = asyncio.run(
                    queue_optimization_coordinator_lifecycle_feedback(
                        store,
                        _decision(),
                        status_code=200,
                        retry_count=0,
                        cost_est_usd=0.01,
                        cost_baseline_usd=0.02,
                    )
                )
                holdout_meta = asyncio.run(
                    queue_optimization_coordinator_lifecycle_feedback(
                        store,
                        _decision(holdout=True),
                        status_code=200,
                        retry_count=1,
                        extra_family_events=[
                            {
                                "family": "old_context_summary",
                                "lifecycle_status": "rollback",
                                "candidate_id": "summary-candidate",
                                "reason_codes": ["rollback-required"],
                            },
                            {
                                "family": "terminal_output_compaction",
                                "lifecycle_status": "safety_stop",
                                "candidate_id": "terminal-candidate",
                                "reason_codes": ["safety-stop-tripped"],
                            },
                        ],
                    )
                )
                rows = store.managed_outcome_feedback_payload_rows(source_surface=SOURCE_SURFACE)
            finally:
                store.conn.close()

        self.assertEqual(selected_meta["status"], "queued")
        self.assertEqual(holdout_meta["status"], "queued")
        self.assertEqual(len(rows), 2)
        payloads = [json.loads(row["payload_json"]) for row in rows]
        statuses = {
            event["lifecycle_status"]
            for payload in payloads
            for event in payload["metadata"]["family_events"]
        }
        for expected in {"selected", "holdout", "suppressed", "rollback", "safety_stop"}:
            self.assertIn(expected, statuses)
        for payload in payloads:
            assert_managed_egress_safe(payload)
            self.assertFalse(payload["metadata"]["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["metadata"]["privacy"]["raw_tool_payloads_included"])

    def test_status_reports_coordinator_breakdowns_failures_and_privacy_drops(self) -> None:
        payload = build_optimization_coordinator_lifecycle_feedback(
            _decision(),
            status_code=500,
            retry_count=2,
        )
        self.assertIsNotNone(payload)
        assert payload is not None

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="coordinator-queued",
                    created_at=now,
                    updated_at=now,
                    source_surface=SOURCE_SURFACE,
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                store.enqueue_managed_outcome_feedback(
                    id="coordinator-retryable",
                    created_at=now,
                    updated_at=now,
                    source_surface=SOURCE_SURFACE,
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="retryable-error",
                    attempts=1,
                    next_attempt_at=now,
                    last_error="request failed",
                    last_status_code=503,
                )
                store.enqueue_managed_outcome_feedback(
                    id="coordinator-privacy-drop",
                    created_at=now,
                    updated_at=now,
                    source_surface=SOURCE_SURFACE,
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="dropped-after-limit",
                    attempts=3,
                    next_attempt_at=now,
                    last_error="unsafe managed egress payload blocked",
                )
                result = feedback.managed_feedback_status_result(
                    store,
                    source_surface=SOURCE_SURFACE,
                    sample_limit=10,
                )
            finally:
                store.conn.close()

        lifecycle = result["optimization_coordinator_lifecycle"]
        self.assertEqual(lifecycle["schema"], "agentflow.optimization_coordinator_lifecycle_queue_status.v1")
        self.assertEqual(lifecycle["queue_rows"], 3)
        self.assertEqual(lifecycle["retryable_failures"], 1)
        self.assertEqual(lifecycle["dropped_privacy_violations"], 1)
        reasons = {item["value"]: item["count"] for item in lifecycle["reason_code_breakdown"]}
        self.assertGreaterEqual(reasons["coordinator-selected"], 3)
        self.assertGreaterEqual(reasons["conflicts-with-selected-family"], 3)
        queue_states = {item["value"]: item["count"] for item in lifecycle["queue_state_breakdown"]}
        self.assertEqual(queue_states["pending"], 2)
        self.assertEqual(queue_states["error"], 1)
        self.assertFalse(lifecycle["payload_json_included"])
        self.assertTrue(result["privacy"]["metadata_only"])


if __name__ == "__main__":
    unittest.main()
