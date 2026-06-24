from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json, utc_now
from tokenclaw.streaming_cohort_feedback import (
    STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA,
    STREAMING_AGENTIC_COHORT_SOURCE_SURFACE,
    build_streaming_agentic_cohort_rollup_feedback,
    record_streaming_agentic_cohort_rollup_feedback,
)


class StreamingCohortFeedbackTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _log_call(
        self,
        store: Store,
        *,
        call_id: str,
        category: str,
        stream: bool,
        status_code: int,
        cache_reason: str,
        local_result: str,
        raw_marker: str | None = None,
    ) -> None:
        request_json = stable_json({
            "messages": [{"content": raw_marker or "local raw prompt must stay local"}],
            "file_path": "/home/lutz/private/project/secret.py",
            "cache_key": "secret-cache-key",
        })
        response_json = stable_json({"content": raw_marker or "raw response must stay local"})
        store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-5",
            routed_model="claude-haiku-4-5",
            stream=1 if stream else 0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=1200,
            input_tokens_est=1800,
            output_tokens_est=120,
            actual_input_tokens=1700,
            actual_output_tokens=100,
            cost_est_usd=0.04 if stream else 9.99,
            cost_baseline_usd=0.12 if stream else 19.99,
            crunch_json=stable_json({
                "status": "changed",
                "changed": True,
                "saved_chars": 4000,
                "tokens_saved_est": 1000,
            }),
            routing_json=stable_json({
                "status": local_result,
                "reason": "fixture-routing-outcome",
                "managed_recommendation": {
                    "enabled": True,
                    "local_result": local_result,
                    "status": local_result,
                    "policy_id": "streaming-policy",
                },
                "old_context_summary_feedback": {"status": "queued"},
            }),
            cache_json=stable_json({"status": "skipped", "reason": cache_reason}),
            error="upstream error body must stay local" if status_code >= 400 else None,
            request_json=request_json,
            response_json=response_json,
            session_id="local-session-id-must-stay-local",
            category=category,
            cache_creation_input_tokens=250,
            cache_read_input_tokens=500,
            retry_count=1 if status_code >= 400 else 0,
            thinking_output_tokens=50,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="/v1/messages",
            requested_model_family="claude-sonnet",
            routed_model_family="claude-haiku",
            routing_outcome_label=local_result,
        )

    def test_build_rollup_aggregates_streaming_agentic_rows_and_excludes_raw_content(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                self._log_call(
                    store,
                    call_id="stream-tool-result",
                    category="tool-result",
                    stream=True,
                    status_code=200,
                    cache_reason="streaming-tools-disabled",
                    local_result="applied",
                    raw_marker="RAW_STREAM_TOOL_RESULT_SECRET",
                )
                self._log_call(
                    store,
                    call_id="stream-tool-heavy",
                    category="tool-heavy",
                    stream=True,
                    status_code=429,
                    cache_reason="tools-disabled",
                    local_result="vetoed",
                    raw_marker="RAW_STREAM_TOOL_HEAVY_SECRET",
                )
                self._log_call(
                    store,
                    call_id="stream-tool-result-holdout",
                    category="tool-result",
                    stream=True,
                    status_code=200,
                    cache_reason="streaming-tools-disabled",
                    local_result="heldout",
                    raw_marker="RAW_STREAM_HOLDOUT_SECRET",
                )
                self._log_call(
                    store,
                    call_id="non-streaming-excluded",
                    category="tool-result",
                    stream=False,
                    status_code=200,
                    cache_reason="miss",
                    local_result="applied",
                    raw_marker="RAW_NON_STREAMING_SECRET",
                )
                self._log_call(
                    store,
                    call_id="chat-stream-excluded",
                    category="chat",
                    stream=True,
                    status_code=200,
                    cache_reason="miss",
                    local_result="noop",
                    raw_marker="RAW_CHAT_STREAM_SECRET",
                )

                rows = store.streaming_agentic_cohort_feedback_rows(window_hours=24, limit=100)
                payload = build_streaming_agentic_cohort_rollup_feedback(rows, window_hours=24)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA)
        self.assertEqual(payload["rows_considered"], 3)
        self.assertEqual(payload["privacy_summary"]["metadata_only"], True)
        self.assertFalse(managed_egress_violations(payload))

        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "RAW_STREAM_TOOL_RESULT_SECRET",
            "RAW_STREAM_TOOL_HEAVY_SECRET",
            "RAW_STREAM_HOLDOUT_SECRET",
            "RAW_NON_STREAMING_SECRET",
            "RAW_CHAT_STREAM_SECRET",
            "local-session-id-must-stay-local",
            "secret-cache-key",
            "/home/lutz/private/project/secret.py",
            "upstream error body must stay local",
        ):
            self.assertNotIn(forbidden, rendered)

        cache_reasons = {
            item["value"]
            for rollup in payload["rollups"]
            for item in rollup["cache_reason_breakdown"]
        }
        self.assertIn("streaming-tools-disabled", cache_reasons)
        self.assertIn("tools-disabled", cache_reasons)

        local_actions = {
            item["value"]
            for rollup in payload["rollups"]
            for item in rollup["local_action_breakdown"]
        }
        self.assertIn("applied", local_actions)
        self.assertIn("vetoed", local_actions)
        self.assertIn("holdout", local_actions)

    def test_record_queues_rollup_when_managed_is_disabled(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                self._log_call(
                    store,
                    call_id="stream-tool-result",
                    category="tool-result",
                    stream=True,
                    status_code=200,
                    cache_reason="streaming-tools-disabled",
                    local_result="applied",
                )
                meta = asyncio.run(
                    record_streaming_agentic_cohort_rollup_feedback(
                        store,
                        window_hours=24,
                        max_rows=100,
                        max_cohorts=10,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self.assertEqual(meta["rows_considered"], 1)
        self.assertEqual(row["source_surface"], STREAMING_AGENTIC_COHORT_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertIn(STREAMING_AGENTIC_COHORT_ROLLUP_SCHEMA, row["payload_json"])


if __name__ == "__main__":
    unittest.main()
