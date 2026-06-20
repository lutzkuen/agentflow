from datetime import datetime, timedelta, timezone
import unittest

from tokenclaw.quality import derive_codex_turn_quality_signals, derive_provider_quality_signals


class QualitySignalTest(unittest.TestCase):
    def test_provider_success_retry_failure_and_local_throttled_signals(self):
        success = derive_provider_quality_signals(
            source_surface="anthropic_messages",
            status_code=200,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            routing_meta={"applied": True},
        )
        self.assertEqual(success["status"], "success")
        self.assertIn("success", success["signal_codes"])
        self.assertIn("optimized-success", success["signal_codes"])

        retry = derive_provider_quality_signals(
            source_surface="openai_responses",
            status_code=200,
            retry_count=1,
        )
        self.assertIn("retry-after-error", retry["signal_codes"])

        failure = derive_provider_quality_signals(
            source_surface="anthropic_messages",
            status_code=400,
            routing_meta={"applied": True},
        )
        self.assertEqual(failure["status"], "failure")
        self.assertIn("optimized-failure", failure["signal_codes"])

        throttled = derive_provider_quality_signals(
            source_surface="anthropic_messages",
            status_code=429,
            error="temporarily limiting requests for tier sonnet",
        )
        self.assertEqual(throttled["status"], "local_throttled")
        self.assertIn("local-throttled", throttled["signal_codes"])

    def test_provider_adoption_windows_add_risk_without_raw_identifiers(self):
        quality = derive_provider_quality_signals(
            source_surface="anthropic_messages",
            status_code=200,
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            routing_meta={
                "applied": True,
                "phase_canary": {
                    "status": "applied",
                    "cohort": "canary_applied",
                    "policy_id": "phase-tool-result-haiku",
                },
            },
            provider_adoption_windows=[
                {
                    "status": "abandoned",
                    "reason": "ttl-expired-without-tool-result",
                    "age_bucket": "1_6h",
                    "relationship": "emitted_tool_use",
                    "tool_use_count": 1,
                    "tool_result_count": 0,
                    "correlation_digest": "sha256:secret-digest",
                    "session_id": "raw-session",
                    "tool_id": "raw-tool",
                }
            ],
        )

        rendered = str(quality)
        self.assertEqual(quality["status"], "success")
        self.assertEqual(quality["risk_level"], "warning")
        self.assertIn("tool-use-abandoned", quality["signal_codes"])
        self.assertIn("optimized-adoption-risk", quality["signal_codes"])
        self.assertEqual(quality["provider_adoption"]["status_counts"]["abandoned"], 1)
        self.assertEqual(quality["provider_adoption"]["risk_window_count"], 1)
        self.assertEqual(quality["optimization_cohorts"][0]["cohort"], "canary_applied")
        self.assertNotIn("secret-digest", rendered)
        self.assertNotIn("raw-session", rendered)
        self.assertNotIn("raw-tool", rendered)

    def test_codex_pending_and_abandoned_turns_use_metadata_age(self):
        now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
        pending = derive_codex_turn_quality_signals(
            created_at=(now - timedelta(seconds=30)).isoformat(),
            now=now,
            abandoned_after_seconds=60,
        )
        self.assertEqual(pending["status"], "pending")
        self.assertIn("pending", pending["signal_codes"])

        abandoned = derive_codex_turn_quality_signals(
            created_at=(now - timedelta(seconds=120)).isoformat(),
            now=now,
            abandoned_after_seconds=60,
        )
        self.assertEqual(abandoned["status"], "abandoned")
        self.assertIn("abandoned", abandoned["signal_codes"])

        failure = derive_codex_turn_quality_signals(
            response_event_id="response-1",
            error_code=-32000,
            error_message="tool failed",
            routing_meta={"applied": True},
        )
        self.assertEqual(failure["status"], "failure")
        self.assertIn("jsonrpc-error", failure["signal_codes"])
        self.assertIn("optimized-failure", failure["signal_codes"])


if __name__ == "__main__":
    unittest.main()
