import asyncio
import time
import unittest

from tokenclaw.limiter import TierBackoffActive, TierLimiter, TierSlot, model_tier, tier_backoff_headers, tier_backoff_payload


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
        self.assertEqual(tier_backoff_headers(ctx.exception, "claude-sonnet-4-6")["x-tokenclaw-routed-model"], "claude-sonnet-4-6")
        self.assertEqual(tier_backoff_payload(ctx.exception)["error"]["type"], "rate_limit_error")

    def test_record_backoff_keeps_longer_existing_window(self):
        limiter = TierLimiter()
        existing_until = time.time() + 120
        limiter.backoff_until["haiku"] = existing_until

        asyncio.run(limiter.record_backoff("claude-haiku-4-5-20251001", {"retry-after": "1"}))

        self.assertEqual(limiter.backoff_until["haiku"], existing_until)

    def test_slot_early_release_frees_capacity_for_next_request(self):
        # The slot guards request initiation, not the whole stream. Releasing it early
        # (at first token) must return capacity so a queued parallel request proceeds
        # without waiting out the first request's generation.
        async def scenario():
            limiter = TierLimiter(max_concurrent_per_tier=1)
            slot = limiter.slot("claude-sonnet-4-6")
            await slot.__aenter__()
            second = limiter.slot("claude-sonnet-4-6")
            waiter = asyncio.ensure_future(second.__aenter__())
            await asyncio.sleep(0)
            self.assertFalse(waiter.done())  # capped at 1 -> second is blocked
            slot.release()                    # first request got its first token
            await asyncio.wait_for(waiter, timeout=1)
            self.assertTrue(waiter.done())    # second acquired the freed slot
            await second.__aexit__(None, None, None)

        asyncio.run(scenario())

    def test_slot_release_is_idempotent(self):
        # __aexit__ must not double-release after an explicit early release.
        async def scenario():
            limiter = TierLimiter(max_concurrent_per_tier=1)
            async with limiter.slot("claude-sonnet-4-6") as slot:
                slot.release()
                slot.release()  # extra explicit release is a no-op
            # exit also released; capacity is exactly 1, not inflated
            self.assertEqual(limiter.semaphores["sonnet"]._value, 1)
            slot2 = limiter.slot("claude-sonnet-4-6")
            await slot2.__aenter__()
            self.assertEqual(limiter.semaphores["sonnet"]._value, 0)
            await slot2.__aexit__(None, None, None)
            self.assertEqual(limiter.semaphores["sonnet"]._value, 1)

        asyncio.run(scenario())

    def test_record_backoff_skips_long_usage_limit_retry_after(self):
        # A long retry-after is the user's account/usage limit, not transient
        # server overload. Recording it would make later requests short-circuit with
        # a synthetic "temporarily limiting requests" error that masks the real
        # usage limit. It must not be stored, so the next request reaches the
        # upstream and the genuine usage-limit 429 passes through.
        limiter = TierLimiter(max_recorded_backoff_seconds=300)

        asyncio.run(limiter.record_backoff("claude-opus-4-8", {"retry-after": "9834"}))

        self.assertNotIn("opus", limiter.backoff_until)
        # await_backoff does not raise; the request proceeds to the upstream.
        asyncio.run(limiter.await_backoff("claude-opus-4-8"))

    def test_record_backoff_still_records_short_transient_backoff(self):
        limiter = TierLimiter(max_recorded_backoff_seconds=300)

        asyncio.run(limiter.record_backoff("claude-opus-4-8", {"retry-after": "45"}))

        self.assertIn("opus", limiter.backoff_until)
        self.assertGreater(limiter.backoff_until["opus"], time.time())


if __name__ == "__main__":
    unittest.main()
