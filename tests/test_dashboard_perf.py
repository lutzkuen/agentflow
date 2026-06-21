import asyncio
import importlib.util
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from tokenclaw import stats as stats_views
    from tokenclaw.store import Store, stable_json


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class DashboardStatsPerfTest(unittest.TestCase):
    def test_dashboard_stats_cold_response_under_1s(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        now = datetime.now(timezone.utc)
        routing = stable_json({"category": "chat"})
        crunch_changed = stable_json({
            "changed": True,
            "saved_chars": 800,
            "tokens_saved_est": 200,
            "crunch_ratio": 0.9,
        })
        crunch_unchanged = stable_json({"changed": False, "saved_chars": 0, "tokens_saved_est": 0})
        cache = stable_json({"status": "miss", "reason": "no-entry"})
        rows = []
        for index in range(500):
            provider = "openai" if index % 3 == 0 else "anthropic"
            requested = "gpt-5" if provider == "openai" else "claude-sonnet-4-6"
            routed = "gpt-5-mini" if provider == "openai" and index % 4 == 0 else requested
            baseline = 0.006 if routed != requested else 0.004
            cost = 0.003 if routed != requested else 0.004
            rows.append((
                f"perf-{index}",
                (now - timedelta(seconds=index)).isoformat(),
                "/v1/responses" if provider == "openai" else "/v1/messages",
                requested,
                routed,
                index % 5 == 0,
                1 if index % 17 == 0 else 0,
                200 if index % 19 else 429,
                90 + index % 50,
                800 + index,
                120 + index % 20,
                780 + index,
                110 + index % 20,
                cost,
                baseline,
                crunch_changed if index % 2 == 0 else crunch_unchanged,
                routing,
                cache,
                None,
                None,
                None,
                f"session-{index % 12}",
                "chat",
                0,
                200 if index % 7 == 0 else 0,
                1 if index % 23 == 0 else 0,
                0,
                provider,
                "openai_responses" if provider == "openai" else "anthropic_messages",
                "responses" if provider == "openai" else "messages",
                "gpt-5" if provider == "openai" else "sonnet",
                "gpt-5-mini" if routed.endswith("mini") else ("gpt-5" if provider == "openai" else "sonnet"),
                "unknown",
                None,
            ))
        try:
            sql = """
            insert into calls(
                    id, created_at, path, requested_model, routed_model, stream, cache_hit,
                    status_code, latency_ms, input_tokens_est, output_tokens_est,
                    actual_input_tokens, actual_output_tokens, cost_est_usd, cost_baseline_usd,
                    crunch_json, routing_json, cache_json, error, request_json, response_json,
                    session_id, category, cache_creation_input_tokens, cache_read_input_tokens,
                    retry_count, thinking_output_tokens, provider, source_surface, endpoint,
                    requested_model_family, routed_model_family, routing_outcome_label,
                    managed_routing_json
            ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """
            for row in rows:
                store.conn.execute(sql, row)
            store.conn.commit()

            elapsed = []
            for _ in range(10):
                start = time.perf_counter()
                payload = asyncio.run(stats_views.stats(store, tmp.name))
                elapsed.append(time.perf_counter() - start)
                self.assertEqual(payload["schema"], "tokenclaw.lightweight_dashboard_stats.v1")
                self.assertEqual(payload["today_calls"], 500)
                self.assertEqual(len(payload["recent"]), 50)

            mean_elapsed = sum(elapsed) / len(elapsed)
            self.assertLess(mean_elapsed, 1.0)
            self.assertLess(max(elapsed), 2.0)
        finally:
            store.conn.close()
            tmp.close()
