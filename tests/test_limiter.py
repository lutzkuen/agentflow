import asyncio
import time
import unittest

from tokenclaw.limiter import TierBackoffActive, TierLimiter, model_tier, tier_backoff_headers, tier_backoff_payload


class TierLimiterTests(unittest.TestCase):
    def test_model_tier_classification(self):
        self.assertEqual(model_tier("claude-haiku-4-5-20251001"), "haiku")
        self.assertEqual(model_tier("claude-opus-4-5"), "opus")
        self.assertEqual(model_tier("claude-sonnet-4-6"), "sonnet")
        self.assertEqual(model_tier("gpt-5-codex"), "sonnet")

    def test_status_reports_active_cooldown_and_concurrency(self):
        limiter = TierLimiter(max_tier_backoff_wait=30, max_concurrent_per_tier=2)
        now = time.time()
        limiter.backoff_until["haiku"] = now + 5

        tiers = {row["tier"]: row for row in limiter.status(now)}

        self.assertTrue(tiers["haiku"]["active"])
        self.assertEqual(tiers["haiku"]["seconds_remaining"], 5.0)
        self.assertEqual(tiers["haiku"]["max_concurrent"], 2)
        self.assertEqual(tiers["haiku"]["available_slots"], 2)
        self.assertFalse(tiers["sonnet"]["active"])

    def test_long_backoff_raises_local_rate_limit_error(self):
        limiter = TierLimiter(max_tier_backoff_wait=0.01)
        limiter.backoff_until["sonnet"] = time.time() + 2

        with self.assertRaises(TierBackoffActive) as ctx:
            asyncio.run(limiter.await_backoff("claude-sonnet-4-6"))

        self.assertEqual(ctx.exception.tier, "sonnet")
        self.assertGreaterEqual(ctx.exception.retry_after, 1)
        self.assertEqual(tier_backoff_headers(ctx.exception, "claude-sonnet-4-6")["x-agentflow-routed-model"], "claude-sonnet-4-6")
        self.assertEqual(tier_backoff_payload(ctx.exception)["error"]["type"], "rate_limit_error")

    def test_record_backoff_keeps_longer_existing_window(self):
        limiter = TierLimiter()
        existing_until = time.time() + 120
        limiter.backoff_until["haiku"] = existing_until

        asyncio.run(limiter.record_backoff("claude-haiku-4-5-20251001", {"retry-after": "1"}))

        self.assertEqual(limiter.backoff_until["haiku"], existing_until)


if __name__ == "__main__":
    unittest.main()
