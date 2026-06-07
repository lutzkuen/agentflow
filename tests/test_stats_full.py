import asyncio
import importlib.util
import json
import tempfile
import time
import unittest
import uuid

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from agentflow_proxy import server
    from agentflow_proxy.store import Store, stable_json, utc_now


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class StatsFullTest(unittest.TestCase):
    def setUp(self):
        self.old_store = server.store
        self.old_tier_backoff_until = dict(server._tier_backoff_until)
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store
        server._tier_backoff_until.clear()
        server._tier_backoff_until.update(self.old_tier_backoff_until)

    def test_crunch_savings_uses_cache_blended_input_rate(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2026-06-07T01:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=1_000,
            output_tokens_est=0,
            actual_input_tokens=1_000,
            actual_output_tokens=0,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 1_000}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-a",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=9_000,
            retry_count=0,
            provider="anthropic",
        )

        result = asyncio.run(server.stats_full())
        summary = result["summary"]

        self.assertAlmostEqual(summary["crunch_savings_usd"], 0.00057, places=6)
        self.assertAlmostEqual(summary["today_crunch_savings_usd"], 0.00057, places=6)

    def test_cache_decision_breakdown_groups_status_reason_and_hit_type(self):
        rows = [
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            {"status": "hit", "reason": "exact-match", "hit_type": "exact", "policy_source": "local-default"},
        ]
        for cache_json in rows:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1 if cache_json["reason"] == "streaming" else 0,
                cache_hit=1 if cache_json["status"] == "hit" else 0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=0.0,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({"changed": False}),
                routing_json=None,
                cache_json=stable_json(cache_json),
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-cache",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(server.stats_full())
        breakdown = {
            (row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in result["cache_decision_breakdown"]
        }

        self.assertEqual(breakdown[("skipped", "streaming", "")], 2)
        self.assertEqual(breakdown[("miss", "exact-miss", "")], 1)
        self.assertEqual(breakdown[("hit", "exact-match", "exact")], 1)
        json.dumps(result["cache_decision_breakdown"])

    def test_sessions_include_thinking_token_breakdown(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2026-06-07T01:00:01+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.02,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-thinking",
            category="tool-heavy",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=1_000,
            provider="anthropic",
        )

        result = asyncio.run(server.stats_sessions())
        [session] = result["sessions"]

        self.assertEqual(session["session_id"], "session-thinking")
        self.assertEqual(session["thinking_tokens"], 1_000)
        self.assertAlmostEqual(session["thinking_cost_usd"], 0.015, places=6)
        json.dumps(result)

    def test_sessions_include_prompt_cache_warmup_breakdown(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.02,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-warmup",
            category="tool-heavy",
            cache_creation_input_tokens=1_000,
            cache_read_input_tokens=10_000,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )

        result = asyncio.run(server.stats_sessions())
        [session] = result["sessions"]

        self.assertEqual(session["session_id"], "session-cache-warmup")
        self.assertEqual(session["cache_creation_tokens"], 1_000)
        self.assertEqual(session["cache_read_tokens"], 10_000)
        self.assertEqual(session["cache_write_read_token_ratio"], 0.1)
        self.assertAlmostEqual(session["cache_creation_cost_usd"], 0.00375, places=6)
        self.assertAlmostEqual(session["cache_read_savings_usd"], 0.027, places=6)
        self.assertEqual(session["cache_warmup_payback_ratio"], 0.139)
        json.dumps(result)

    def test_recent_session_spending_summary_breaks_down_cost_drivers(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=1_000,
            output_tokens_est=100,
            actual_input_tokens=1_000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"reason": "test route"}),
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-spending",
            category="tool-result",
            cache_creation_input_tokens=500,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=300,
            provider="anthropic",
        )

        [summary] = server._recent_session_spending_summary()

        self.assertEqual(summary["session_id"], "session-spending")
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["cache_creation_tokens"], 500)
        self.assertEqual(summary["cache_read_tokens"], 2_000)
        self.assertEqual(summary["thinking_tokens"], 300)
        self.assertAlmostEqual(summary["cost_usd"], 0.001, places=6)
        self.assertAlmostEqual(summary["baseline_savings_usd"], 0.009, places=6)
        self.assertAlmostEqual(summary["routing_savings_usd"], 0.003, places=6)
        self.assertAlmostEqual(summary["prompt_cache_savings_usd"], 0.0018, places=6)
        self.assertAlmostEqual(summary["thinking_cost_usd"], 0.0015, places=6)
        json.dumps(summary)

    def test_sessions_identify_context_plateaus(self):
        text_sizes = [10_000, 10_200, 10_150, 15_000]
        for idx, text_chars in enumerate(text_sizes):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=text_chars // 4,
                output_tokens_est=1,
                actual_input_tokens=text_chars // 4,
                actual_output_tokens=1,
                cost_est_usd=0.01,
                cost_baseline_usd=0.01,
                crunch_json=stable_json({"changed": True, "saved_chars": 100 + idx}),
                routing_json=stable_json({"text_chars": text_chars}),
                cache_json=None,
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-plateau",
                category="tool-heavy",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=1_000 if idx == 0 else 0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        result = asyncio.run(server.stats_sessions())
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]

        self.assertEqual(session["session_id"], "session-plateau")
        self.assertEqual(session["plateau_pairs"], 2)
        self.assertEqual(session["median_text_chars"], 10_175)
        self.assertEqual(session["p90_text_chars"], 15_000)
        self.assertEqual(plateau["session_id"], "session-plateau")
        self.assertEqual(plateau["calls"], 4)
        self.assertEqual(plateau["plateau_pairs"], 2)
        self.assertEqual(plateau["crunch_saved_chars"], 406)
        self.assertAlmostEqual(plateau["cache_read_savings_usd"], 0.0027, places=6)
        self.assertFalse(plateau["flagged"])
        self.assertEqual(result["context_plateau_policy"]["min_text_chars"], 8_000)
        json.dumps(result)

    def test_limiter_stats_include_active_cooldown_and_recent_rate_limits(self):
        server._tier_backoff_until.clear()
        server._tier_backoff_until["haiku"] = time.time() + 90
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=429,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error="temporarily limiting requests for haiku tier; retry after 90s",
            request_json=None,
            response_json=None,
            session_id="session-limiter",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=429,
            latency_ms=10,
            input_tokens_est=10,
            output_tokens_est=1,
            actual_input_tokens=10,
            actual_output_tokens=1,
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error="upstream_error: status=429",
            request_json=None,
            response_json=None,
            session_id="session-limiter",
            category="tool-heavy",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=3,
            provider="anthropic",
        )

        result = asyncio.run(server.stats_limiter())
        tiers = {row["tier"]: row for row in result["tiers"]}

        self.assertTrue(tiers["haiku"]["active"])
        self.assertGreater(tiers["haiku"]["seconds_remaining"], 0)
        self.assertIsNotNone(tiers["haiku"]["cooldown_until"])
        self.assertEqual(tiers["haiku"]["max_concurrent"], server.MAX_CONCURRENT_PER_TIER)
        self.assertIsNotNone(tiers["sonnet"]["last_upstream_429_at"])
        self.assertEqual(result["summary"]["active_cooldowns"], 1)
        self.assertEqual(result["summary"]["local_throttled_recent"], 1)
        self.assertEqual(result["summary"]["upstream_limited_recent"], 1)
        self.assertEqual(result["recent_rate_limits"][0]["tier"], "sonnet")
        self.assertEqual(result["recent_rate_limits"][1]["tier"], "haiku")
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
