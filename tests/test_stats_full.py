import asyncio
import importlib.util
import json
import tempfile
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
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        server.store = Store(self.tmp.name)

    def tearDown(self):
        server.store.conn.close()
        self.tmp.close()
        server.store = self.old_store

    def test_crunch_savings_uses_cache_blended_input_rate(self):
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


if __name__ == "__main__":
    unittest.main()
