import asyncio
import importlib.util
import json
import os
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import server, stats as stats_views
    from agentflow_proxy.dashboard_app import create_dashboard_app
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

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]

        self.assertAlmostEqual(summary["crunch_savings_usd"], 0.00057, places=6)
        self.assertAlmostEqual(summary["today_crunch_savings_usd"], 0.00057, places=6)

    def test_policy_state_exposes_codex_app_surface_cache_disabled(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "1",
                "AGENTFLOW_CODEX_APP_CACHE": "0",
                "AGENTFLOW_CODEX_APP_UPSTREAM": "ws://127.0.0.1:4999",
            },
            clear=False,
        ):
            result = asyncio.run(stats_views.stats_policies())

        self.assertIn("routing", result)
        self.assertIn("crunch", result)
        self.assertIn("cache", result)
        surface = result["source_surfaces"]["codex_turn"]
        self.assertTrue(surface["optimization"]["enabled"])
        self.assertFalse(surface["cache"]["enabled"])
        self.assertFalse(surface["cache"]["exact_cache"]["enabled"])
        self.assertEqual(surface["cache"]["disabled_reason"], "AGENTFLOW_CODEX_APP_CACHE is not 1")
        self.assertEqual(surface["cache"]["exact_cache"]["upstream"], "ws://127.0.0.1:4999")
        self.assertIn("input", surface["safe_turn_params"]["allowed_keys"])
        self.assertEqual(surface["action_like_skip_behavior"]["reason"], "action-like-params")
        self.assertEqual(surface["routing"]["policy_source"], result["routing"]["policy_source"])
        self.assertEqual(surface["crunch"]["rule_path"], result["crunch"]["rule_path"])
        self.assertEqual(surface["cache"]["rule_path"], result["cache"]["rule_path"])
        self.assertEqual(surface["managed_optimizer_required"], False)
        self.assertIn("codex_app", result)
        self.assertTrue(result["codex_app"]["review_only"])
        self.assertEqual(result["summary"]["policy_count"], 5)
        self.assertEqual(result["summary"]["source_surface_policy_count"], 1)

    def test_policy_state_exposes_codex_app_surface_cache_enabled(self):
        with patch.dict(
            os.environ,
            {
                "AGENTFLOW_CODEX_APP_OPTIMIZE": "0",
                "AGENTFLOW_CODEX_APP_CACHE": "1",
                "AGENTFLOW_CODEX_APP_CACHE_NAMESPACE": "codex-test",
            },
            clear=False,
        ):
            result = asyncio.run(stats_views.stats_policies())

        surface = result["source_surfaces"]["codex_turn"]
        self.assertFalse(surface["optimization"]["enabled"])
        self.assertTrue(surface["cache"]["enabled"])
        self.assertTrue(surface["cache"]["exact_cache"]["enabled"])
        self.assertEqual(surface["cache"]["exact_cache"]["namespace"], "codex-test")
        self.assertEqual(surface["cache"]["exact_cache"]["provider"], "codex-app")
        self.assertEqual(surface["cache"]["exact_cache"]["cache_url"], "codex-app://turn/start")
        self.assertEqual(surface["cache"]["exact_cache"]["replayability_level"], "local-exact-response")

    def test_full_stats_include_executive_summary_for_top_dashboard_tiles(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=1,
            status_code=200,
            latency_ms=10,
            input_tokens_est=1_200,
            output_tokens_est=120,
            actual_input_tokens=1_000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 500}),
            routing_json=stable_json({"reason": "test route"}),
            cache_json=stable_json({"status": "hit", "reason": "exact-match", "hit_type": "exact"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-exec",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2_000,
            retry_count=0,
            thinking_output_tokens=25,
            provider="anthropic",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-exec",
            thread_id="thread-exec",
            message_chars=200,
            params_chars=50,
            input_items=1,
            input_text_chars=123,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-exec",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-exec",
            thread_id="thread-exec",
            message_chars=120,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=2000,
            session_id="codex-exec",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        executive = result["executive_summary"]

        self.assertEqual(executive["schema"], "agentflow.executive_summary.v1")
        self.assertEqual(executive["tokens_today"]["provider_input_tokens"], 3_000)
        self.assertEqual(executive["tokens_today"]["provider_output_tokens"], 100)
        self.assertEqual(executive["tokens_today"]["provider_total_tokens"], 3_100)
        self.assertEqual(executive["tokens_today"]["codex_app_turns"], 1)
        self.assertEqual(executive["tokens_today"]["codex_app_input_text_chars"], 123)
        self.assertEqual(executive["tokens_today"]["codex_app_input_tokens_est"], 30)
        self.assertEqual(executive["tokens_today"]["codex_app_output_tokens_est"], 20)
        self.assertEqual(executive["tokens_today"]["codex_app_total_tokens_est"], 50)
        self.assertEqual(executive["tokens_today"]["total_tokens"], 3_150)
        self.assertTrue(executive["tokens_today"]["codex_app_cost_known"])
        self.assertTrue(executive["tokens_today"]["codex_app_cost_estimated"])
        self.assertEqual(executive["tokens_today"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(executive["spend"]["today_provider_spend_usd"], 0.001, places=6)
        self.assertEqual(
            executive["tokens_today"]["codex_app_pricing_basis"]["model"],
            "gpt-5.3-codex",
        )
        self.assertAlmostEqual(executive["spend"]["today_codex_app_estimated_spend_usd"], 0.000333, places=6)
        self.assertAlmostEqual(executive["spend"]["today_calculated_spend_usd"], 0.001332, places=6)
        self.assertEqual(executive["spend"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertGreater(executive["spend"]["thinking_cost_today_usd"], 0)
        buckets = executive["savings"]["today_buckets"]
        self.assertIn("routing_usd", buckets)
        self.assertIn("crunching_usd", buckets)
        self.assertAlmostEqual(buckets["exact_local_cache_usd"], 0.003, places=6)
        self.assertIn("provider_prompt_cache_discount_usd", buckets)
        self.assertFalse(executive["hard_floor"]["excludes_unknown_codex_app_cost"])
        self.assertTrue(executive["hard_floor"]["codex_app_cost_estimated"])
        self.assertLessEqual(
            executive["hard_floor"]["today_unavoidable_provider_spend_usd"],
            executive["spend"]["today_baseline_calculated_cost_usd"],
        )
        self.assertIn("accounting_today", executive)
        self.assertIn("source_surfaces", executive["accounting_today"])
        json.dumps(executive)

    def test_weekly_stats_include_codex_turns_and_zero_daily_rows(self):
        days = stats_views._utc_day_window(7)
        provider_day = days[-3]
        empty_day = days[-2]
        today = days[-1]

        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=f"{provider_day}T12:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="provider-session",
            category="chat",
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=f"{today}T09:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="weekly-codex",
            thread_id="thread-weekly",
            message_chars=500,
            params_chars=20,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-weekly",
            cache_json=stable_json({"status": "miss", "reason": "codex-app-cache-disabled"}),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=f"{today}T09:00:02+00:00",
            direction="server_to_client",
            method="turn/completed",
            request_id="weekly-codex",
            thread_id="thread-weekly",
            message_chars=80,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=2000,
            session_id="codex-weekly",
        )

        result = asyncio.run(stats_views.stats_weekly(server.store))

        self.assertEqual(result["schema"], "agentflow.weekly_activity.v1")
        self.assertIn("generated_at", result)
        self.assertEqual([row["day"] for row in result["days"]], days)
        self.assertEqual(len(result["days"]), 7)

        by_day = {row["day"]: row for row in result["days"]}
        provider_row = by_day[provider_day]
        self.assertEqual(provider_row["provider_calls"], 1)
        self.assertEqual(provider_row["codex_turns"], 0)
        self.assertEqual(provider_row["total_calls"], 1)
        self.assertEqual(provider_row["provider_tokens"], 160)
        self.assertAlmostEqual(provider_row["cost_est_usd"], 0.001, places=6)
        self.assertAlmostEqual(provider_row["savings_usd"], 0.002, places=6)

        zero_row = by_day[empty_day]
        self.assertEqual(zero_row["total_units"], 0)
        self.assertEqual(zero_row["provider_calls"], 0)
        self.assertEqual(zero_row["codex_turns"], 0)
        self.assertEqual(zero_row["total_tokens"], 0)
        self.assertEqual(zero_row["cost_est_usd"], 0.0)

        today_row = by_day[today]
        expected_codex = stats_views._codex_estimates_with_cache(400, 80, {"status": "miss"})
        self.assertEqual(today_row["provider_calls"], 0)
        self.assertEqual(today_row["codex_turns"], 1)
        self.assertEqual(today_row["total_calls"], 1)
        self.assertEqual(today_row["successful_calls"], 1)
        self.assertEqual(today_row["codex_tokens_est"], expected_codex["total_tokens_est"])
        self.assertEqual(today_row["total_tokens"], expected_codex["total_tokens_est"])
        self.assertAlmostEqual(today_row["codex_cost_est_usd"], expected_codex["cost_est_usd"], places=6)
        self.assertAlmostEqual(today_row["cost_est_usd"], expected_codex["cost_est_usd"], places=6)
        self.assertEqual(today_row["avg_latency_ms"], 2000)
        self.assertEqual(today_row["cost_basis"], "provider-reported + codex-estimated-from-chars")

        totals = result["totals"]
        self.assertEqual(totals["provider_calls"], 1)
        self.assertEqual(totals["codex_turns"], 1)
        self.assertEqual(totals["total_units"], 2)
        self.assertEqual(totals["total_calls"], 2)
        self.assertEqual(totals["provider_tokens"], 160)
        self.assertEqual(totals["codex_tokens_est"], expected_codex["total_tokens_est"])
        self.assertEqual(totals["total_tokens"], 160 + expected_codex["total_tokens_est"])

    def test_dashboard_weekly_table_exposes_provider_and_codex_columns(self):
        html = stats_views.dashboard_html()

        self.assertIn("<h2>7-day activity statistics</h2>", html)
        self.assertIn('<th data-sort-type="number">Provider calls</th>', html)
        self.assertIn('<th data-sort-type="number">Codex turns</th>', html)
        self.assertIn('<th data-sort-type="number">Tokens</th>', html)
        self.assertIn('<th data-sort-type="text">Cost basis</th>', html)
        self.assertIn("row.codex_turns", html)
        self.assertIn("row.codex_tokens_est", html)

    def test_managed_recommendation_stats_cover_recent_statuses_and_feedback(self):
        saved_enabled = os.environ.get("AGENTFLOW_RECOMMENDATION_ENABLED")
        saved_url = os.environ.get("AGENTFLOW_RECOMMENDATION_SERVER_URL")
        os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.local"

        def log_call(created_at, routing_json):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=created_at,
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=10,
                input_tokens_est=100,
                output_tokens_est=10,
                actual_input_tokens=100,
                actual_output_tokens=10,
                cost_est_usd=0.001,
                cost_baseline_usd=0.003,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing_json) if routing_json is not None else None,
                cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="managed-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=0,
                provider="anthropic",
            )

        try:
            log_call(
                "2026-06-08T10:00:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": False,
                        "server_url": "http://managed.local",
                        "status": "skipped",
                        "reason": "disabled",
                        "fallback": "local-policy",
                        "applied": False,
                        "outcome_feedback": {
                            "enabled": False,
                            "status": "skipped",
                            "reason": "disabled",
                        },
                    }
                },
            )
            log_call(
                "2026-06-08T10:01:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": True,
                        "server_url": "http://managed.local",
                        "status": "received",
                        "reason": "candidate matched",
                        "policy_source": "managed-recommended",
                        "policy_id": "candidate-route-chat",
                        "target_model": "claude-haiku-4-5-20251001",
                        "applied": True,
                        "changed_model": True,
                        "latency_ms": 120,
                        "outcome_feedback": {
                            "enabled": True,
                            "status": "sent",
                            "reason": "accepted",
                            "latency_ms": 30,
                            "optimization_unit_id": 42,
                        },
                    }
                },
            )
            log_call(
                "2026-06-08T10:02:00+00:00",
                {
                    "managed_recommendation": {
                        "enabled": True,
                        "server_url": "http://managed.local",
                        "status": "error",
                        "reason": "server-error",
                        "fallback": "local-policy",
                        "applied": False,
                        "latency_ms": 50,
                        "error": '{"error":{"type":"managed_server_error","message":"down"}}',
                        "outcome_feedback": {
                            "enabled": True,
                            "status": "error",
                            "reason": "request-failed",
                            "latency_ms": 20,
                            "error": "RuntimeError('feedback unavailable')",
                        },
                    }
                },
            )
            log_call("2026-06-08T10:03:00+00:00", None)

            result = asyncio.run(stats_views.stats_managed_recommendations(server.store, limit=20))
        finally:
            if saved_enabled is None:
                os.environ.pop("AGENTFLOW_RECOMMENDATION_ENABLED", None)
            else:
                os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = saved_enabled
            if saved_url is None:
                os.environ.pop("AGENTFLOW_RECOMMENDATION_SERVER_URL", None)
            else:
                os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = saved_url

        summary = result["summary"]
        self.assertEqual(result["schema"], "agentflow.managed_recommendations.v1")
        self.assertFalse(result["current_config"]["enabled"])
        self.assertIn("local policy remains authoritative", result["current_config"]["offline_state"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertEqual(summary["window_calls"], 4)
        self.assertEqual(summary["metadata_rows"], 3)
        self.assertEqual(summary["historical_null_rows"], 1)
        self.assertEqual(summary["disabled_count"], 1)
        self.assertEqual(summary["enabled_count"], 2)
        self.assertEqual(summary["received_count"], 1)
        self.assertEqual(summary["applied_count"], 1)
        self.assertEqual(summary["changed_model_count"], 1)
        self.assertEqual(summary["server_error_count"], 1)
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(summary["feedback_sent_count"], 1)
        self.assertEqual(summary["feedback_skipped_count"], 1)
        self.assertEqual(summary["feedback_failed_count"], 1)
        self.assertEqual(summary["feedback_sanitized_count"], 3)
        self.assertEqual(summary["avg_recommendation_latency_ms"], 85)
        self.assertEqual(summary["avg_feedback_latency_ms"], 25)
        self.assertEqual(summary["last_recommendation_error_class"], "server-error")
        self.assertEqual(summary["last_feedback_error_class"], "request-failed")
        self.assertEqual({row["value"]: row["count"] for row in result["policy_ids"]}, {"candidate-route-chat": 1})
        self.assertEqual(result["recent"][0]["recommendation_status"], "missing")
        self.assertEqual(result["recent"][0]["recommendation_reason"], "historical-null")
        json.dumps(result)

    def test_managed_recommendation_dashboard_endpoint_and_panel_render_without_server(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AGENTFLOW_POLICY_EVENTS_LOG": os.path.join(tmp, "policy_events.jsonl")},
            clear=False,
        ):
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "fetch-review",
                ok=True,
                details={
                    "source": "cli",
                    "recommendation_health": {
                        "schema": "agentflow.recommendation_health.v1",
                        "status": "warning",
                        "warning_count": 1,
                        "rows": [
                            {
                                "kind": "stale_evidence",
                                "code": "stale-evidence",
                                "candidate_id": "candidate-route-chat",
                                "details": {"sample_count": 24, "last_seen_at": "2026-06-01T12:00:00+00:00"},
                            }
                        ],
                        "privacy": {"metadata_only": True, "raw_prompts_included": False},
                    },
                },
            )

            app = create_dashboard_app(
                store_obj=server.store,
                default_db=self.tmp.name,
                upstream="https://api.anthropic.com",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)

            stats_response = client.get("/agentflow/stats/managed-recommendations")
            self.assertEqual(stats_response.status_code, 200)
            payload = stats_response.json()
            self.assertEqual(payload["schema"], "agentflow.managed_recommendations.v1")
            self.assertEqual(
                payload["recommendation_health"]["latest_fetch_review"]["rows"][0]["candidate_id"],
                "candidate-route-chat",
            )

            html = client.get("/agentflow/dashboard")
            self.assertEqual(html.status_code, 200)
            self.assertIn("Managed recommendation status", html.text)
            self.assertIn("Managed recommendation health", html.text)
            self.assertIn("/agentflow/stats/managed-recommendations", html.text)
            self.assertIn("managed-summary-tbody", html.text)
            self.assertIn("managed-health-tbody", html.text)

    def test_full_stats_unifies_source_surface_accounting_for_mixed_traffic(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=10,
            input_tokens_est=110,
            output_tokens_est=11,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=0.001,
            cost_baseline_usd=0.003,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 40, "policy_source": "local-default"}),
            routing_json=stable_json({"policy_source": "local-manual"}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="mixed-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=20,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5",
            routed_model="gpt-5-mini",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=20,
            input_tokens_est=210,
            output_tokens_est=21,
            actual_input_tokens=200,
            actual_output_tokens=20,
            cost_est_usd=0.002,
            cost_baseline_usd=0.004,
            crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
            routing_json=stable_json({"policy_source": "local-manual", "reason": "test route"}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="mixed-session",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
        )
        start_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-mixed",
            thread_id="thread-mixed",
            message_chars=80,
            params_chars=10,
            input_items=1,
            input_text_chars=40,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="mixed-session",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-mixed",
            thread_id="thread-mixed",
            message_chars=20,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=20,
            error_code=None,
            error_message=None,
            latency_ms=200,
            session_id="mixed-session",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        accounting = result["executive_summary"]["accounting_today"]
        by_surface = {row["source_surface"]: row for row in accounting["source_surfaces"]}
        savings = {
            (row["source_surface"], row["optimization_type"]): row["savings_usd"]
            for row in result["today_savings_by_source_surface"]
        }

        self.assertEqual(set(by_surface), {"anthropic_messages", "openai_responses", "codex_turn"})
        self.assertEqual(by_surface["anthropic_messages"]["input_tokens"], 120)
        self.assertEqual(by_surface["openai_responses"]["input_tokens"], 200)
        self.assertEqual(by_surface["codex_turn"]["input_tokens"], 10)
        self.assertEqual(by_surface["anthropic_messages"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["openai_responses"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["codex_turn"]["token_basis"], "estimated-from-chars")
        self.assertEqual(accounting["input_tokens"], 330)
        self.assertEqual(accounting["output_tokens"], 35)
        self.assertEqual(accounting["total_tokens"], 365)
        self.assertEqual(result["executive_summary"]["tokens_today"]["total_tokens"], 365)
        self.assertGreater(savings[("anthropic_messages", "crunching")], 0)
        self.assertGreater(savings[("anthropic_messages", "cache")], 0)
        self.assertGreater(savings[("openai_responses", "routing")], 0)
        self.assertNotIn(("codex_turn", "routing"), savings)
        json.dumps(result)

    def test_codex_effectiveness_report_summarizes_live_like_metadata_without_raw_text(self):
        secret = "secret prompt text"

        def log_turn(
            request_id,
            *,
            routing,
            crunch,
            cache,
            input_text_chars=0,
            params_chars=100,
            response_error_code=None,
            response_error_message=None,
            response_latency_ms=100,
        ):
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=f"thread-{request_id}",
                message_chars=200,
                params_chars=params_chars,
                input_items=1 if input_text_chars else 0,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-effectiveness",
                routing_json=stable_json(routing),
                crunch_json=stable_json(crunch),
                cache_json=stable_json(cache),
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=f"thread-{request_id}",
                message_chars=160,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=80,
                error_code=response_error_code,
                error_message=response_error_message,
                latency_ms=response_latency_ms,
                session_id="codex-effectiveness",
            )

        log_turn(
            "model-absent",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": False,
                    "status": "skipped",
                    "reason": "disabled",
                    "outcome_feedback": {"enabled": False, "status": "skipped", "reason": "disabled"},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True, "policy_source": "local-default"},
            input_text_chars=120,
        )
        log_turn(
            "model-routed",
            routing={
                "status": "applied",
                "reason": "small non-tool Sonnet request routed to Haiku",
                "applied": True,
                "model_field": "model",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": True,
                    "status": "received",
                    "policy_id": "codex-policy-1",
                    "target_model": "claude-haiku-4-5-20251001",
                    "applied": False,
                    "apply_reason": "codex-app-managed-recommendation-observed-only",
                    "outcome_feedback": {"enabled": True, "status": "sent", "reason": "accepted", "optimization_unit_id": 77},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": True, "policy_source": "local-default"},
            input_text_chars=80,
            response_error_code=-32000,
            response_error_message="upstream rejected routed model",
            response_latency_ms=250,
        )
        log_turn(
            "action-like",
            routing={"status": "not-applied", "reason": "action-like-params", "applied": False, "policy_source": "local-default"},
            crunch={"status": "not-applied", "reason": "action-like-params", "applied": False},
            cache={"status": "skipped", "reason": "action-like-params", "eligible": False, "policy_source": "local-default"},
        )
        log_turn(
            "crunch-applied",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
            },
            crunch={
                "status": "applied",
                "reason": "codex-turn-start-crunched",
                "applied": True,
                "changed": True,
                "saved_chars": 1600,
                "tokens_saved_est": 400,
                "codex_repeated_scaffolding": {
                    "status": "applied",
                    "reason": "codex-repeated-scaffolding-crunched",
                    "saved_chars": 1200,
                    "patterns": [
                        {
                            "type": "repeated_input_section",
                            "count": 2,
                            "saved_chars_est": 700,
                            "hashes": ["abcdef123456"],
                        },
                        {
                            "type": "older_input_head_tail",
                            "count": 1,
                            "saved_chars_est": 500,
                            "hashes": ["123456abcdef"],
                        },
                    ],
                },
                "codex_patterns": [
                    {"type": "repeated_input_section", "count": 2, "saved_chars_est": 700},
                    {"type": "older_input_head_tail", "count": 1, "saved_chars_est": 500},
                ],
                "note": secret,
            },
            cache={"status": "skipped", "reason": "unknown-param-shape", "eligible": False, "policy_source": "local-default"},
            input_text_chars=3000,
        )
        log_turn(
            "cache-hit",
            routing={
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
                "managed_recommendation": {
                    "enabled": True,
                    "status": "error",
                    "reason": "server-error",
                    "outcome_feedback": {"enabled": True, "status": "error", "reason": "request-failed"},
                },
            },
            crunch={"status": "skipped", "reason": "no-change", "applied": False, "changed": False},
            cache={"status": "hit", "reason": "exact-match", "eligible": True, "hit_type": "exact", "policy_source": "local-default"},
            input_text_chars=200,
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=20))
        summary = result["summary"]

        self.assertEqual(result["schema"], "agentflow.codex_app_effectiveness.v1")
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_params_included"])
        self.assertFalse(result["privacy"]["raw_responses_included"])
        self.assertEqual(summary["turn_start_rows"], 5)
        self.assertEqual(summary["model_field_present"], 1)
        self.assertEqual(summary["model_field_absent"], 3)
        self.assertEqual(summary["routing_applied"], 1)
        self.assertEqual(summary["crunch_applied"], 1)
        self.assertEqual(summary["cache_hits"], 1)
        self.assertEqual(summary["cache_eligible"], 3)
        self.assertEqual(summary["action_like_skips"], 1)
        self.assertEqual(summary["unknown_param_skips"], 1)
        self.assertEqual(summary["total_saved_chars"], 1600)
        self.assertEqual(summary["total_saved_tokens_est"], 400)
        self.assertEqual(summary["codex_repeated_scaffolding_saved_chars"], 1200)
        self.assertEqual(summary["error_rows"], 1)
        self.assertEqual(summary["optimized_rows"], 3)
        self.assertEqual(summary["pass_through_rows"], 2)
        self.assertGreater(summary["optimized_error_rate"], 0)
        self.assertEqual(summary["managed_recommendation_rows"], 3)
        self.assertEqual(summary["managed_recommendation_enabled"], 2)
        self.assertEqual(summary["managed_recommendation_disabled"], 1)
        self.assertEqual(summary["managed_feedback_sent"], 1)
        self.assertEqual(summary["managed_feedback_skipped"], 1)
        self.assertEqual(summary["managed_feedback_error"], 1)

        model_fields = {row["value"]: row["count"] for row in result["model_field_breakdown"]}
        self.assertEqual(model_fields["present"], 1)
        self.assertEqual(model_fields["absent"], 3)
        shapes = {row["value"]: row["count"] for row in result["param_shape_breakdown"]}
        self.assertEqual(shapes["action-like-params"], 1)
        self.assertEqual(shapes["unknown-param-shape"], 1)
        routing_statuses = {row["status"] for row in result["routing_breakdown"]}
        self.assertIn("applied", routing_statuses)
        self.assertIn("not-applicable", routing_statuses)
        patterns = {row["type"]: row for row in result["crunch_pattern_breakdown"]}
        self.assertEqual(patterns["repeated_input_section"]["count"], 2)
        self.assertEqual(patterns["older_input_head_tail"]["saved_chars_est"], 500)
        feedback_statuses = {row["value"]: row["count"] for row in result["managed_feedback_breakdown"]}
        self.assertEqual(feedback_statuses, {"sent": 1, "skipped": 1, "error": 1})
        sample = next(row for row in result["recent_samples"] if row["saved_chars"] == 1600)
        self.assertEqual(set(sample["codex_pattern_types"]), {"repeated_input_section", "older_input_head_tail"})
        self.assertNotIn(secret, json.dumps(result))

    def test_codex_effectiveness_reports_quota_token_usage_without_raw_payloads(self):
        raw_prompt = "seeded raw prompt must not appear"
        raw_command = "seeded raw command must not appear"
        raw_transcript = "seeded raw transcript must not appear"
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="quota-turn",
            thread_id="thread-quota",
            message_chars=200,
            params_chars=100,
            input_items=1,
            input_text_chars=4000,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            routing_json=stable_json({"status": "not-applicable", "reason": "codex-turn-start-model-field-absent"}),
            crunch_json=stable_json({"status": "skipped", "changed": False}),
            cache_json=stable_json({"status": "skipped", "eligible": False}),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="quota-turn",
            thread_id="thread-quota",
            message_chars=80,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=400,
            error_code=None,
            error_message=None,
            latency_ms=50,
            session_id="codex-quota",
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="thread/tokenUsage/updated",
            request_id=None,
            thread_id="thread-quota",
            message_chars=120,
            params_chars=120,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            metadata_json=stable_json({
                "schema": "agentflow.codex_app_metadata.v1",
                "kind": "token_usage",
                "method": "thread/tokenUsage/updated",
                "token_usage": {
                    "input_tokens": 1200,
                    "cached_input_tokens": 200,
                    "output_tokens": 125,
                    "reasoning_output_tokens": 25,
                    "total_tokens": 1550,
                    "total_tokens_bucket": "1k_10k",
                },
                "debug": raw_prompt,
            }),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method="account/rateLimits/updated",
            request_id=None,
            thread_id=None,
            message_chars=120,
            params_chars=120,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-quota",
            metadata_json=stable_json({
                "schema": "agentflow.codex_app_metadata.v1",
                "kind": "rate_limits",
                "method": "account/rateLimits/updated",
                "rate_limits": {
                    "plan_type": "pro",
                    "pressure": "high",
                    "scopes": [
                        {
                            "name": "primary",
                            "used_percent": 92.5,
                            "used_percent_bucket": "90_99",
                            "remaining": 42,
                            "remaining_bucket": "10_99",
                            "reset_bucket": "1m_1h",
                        }
                    ],
                },
                "debug": raw_command,
                "transcript": raw_transcript,
            }),
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        quota = result["quota_and_token_usage"]

        self.assertEqual(result["summary"]["rate_limit_update_rows"], 1)
        self.assertEqual(result["summary"]["token_usage_update_rows"], 1)
        self.assertEqual(quota["latest_rate_limits"]["pressure"], "high")
        self.assertEqual(quota["latest_rate_limits"]["scopes"][0]["remaining_bucket"], "10_99")
        self.assertEqual(quota["token_usage_totals"]["total_tokens"], 1550)
        self.assertEqual(quota["agentflow_estimated_totals"]["total_tokens_est"], 1100)
        self.assertEqual(quota["reconciliation"]["total_drift_tokens"], 450)
        self.assertEqual(quota["reconciliation"]["total_drift_bucket"], "100_999")
        self.assertFalse(quota["privacy"]["raw_prompts_included"])
        self.assertFalse(quota["privacy"]["raw_commands_included"])
        rendered = json.dumps(result)
        self.assertNotIn(raw_prompt, rendered)
        self.assertNotIn(raw_command, rendered)
        self.assertNotIn(raw_transcript, rendered)

    def test_codex_effectiveness_normalizes_historical_missing_decision_metadata(self):
        fixtures = [
            (
                "complete",
                stable_json({"status": "not-applied", "reason": "fixture-route", "applied": False, "policy_source": "local-default"}),
                stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
                None,
            ),
            ("historical", None, None, None, None),
            (
                "partial",
                None,
                stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
                None,
            ),
            (
                "current-missing",
                None,
                None,
                None,
                stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 1,
                    "method_counts": {"turn/start": 1},
                    "direction_counts": {"client_to_server": 1},
                    "model_field_state": "derived_absent",
                }),
            ),
        ]
        for index, (suffix, routing_json, crunch_json, cache_json, event_window_json) in enumerate(fixtures):
            server.store.log_codex_app_event(
                id=f"start-decision-{suffix}",
                created_at=f"2026-06-08T11:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-decision-{suffix}",
                thread_id=f"thread-decision-{suffix}",
                message_chars=160,
                params_chars=90,
                input_items=1,
                input_text_chars=72,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-decision-metadata",
                routing_json=routing_json,
                crunch_json=crunch_json,
                cache_json=cache_json,
                event_window_json=event_window_json,
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        summary = result["summary"]
        metadata = {row["value"]: row["count"] for row in result["decision_metadata_breakdown"]}
        routing = {row["status"]: row["count"] for row in result["routing_breakdown"]}
        current_missing = {row["value"]: row["count"] for row in result["current_missing_decision_breakdown"]}
        not_instrumented = {row["value"]: row["count"] for row in result["not_instrumented_decision_breakdown"]}
        historical = {row["value"]: row["count"] for row in result["historical_unavailable_decision_breakdown"]}
        sample_states = {row["decision_metadata_state"] for row in result["recent_samples"]}

        self.assertEqual(summary["turn_start_rows"], 4)
        self.assertEqual(summary["decision_metadata_complete_rows"], 1)
        self.assertEqual(summary["decision_metadata_historical_unavailable_rows"], 1)
        self.assertEqual(summary["decision_metadata_not_instrumented_rows"], 1)
        self.assertEqual(summary["decision_metadata_current_missing_rows"], 1)
        self.assertEqual(summary["current_missing_decisions"], 3)
        self.assertEqual(summary["not_instrumented_decisions"], 1)
        self.assertEqual(summary["historical_unavailable_decisions"], 3)
        self.assertEqual(metadata, {
            "complete": 1,
            "historical-unavailable": 1,
            "not-instrumented": 1,
            "current-missing": 1,
        })
        self.assertEqual(routing["not-applied"], 1)
        self.assertEqual(routing["historical-unavailable"], 1)
        self.assertEqual(routing["not-instrumented"], 1)
        self.assertEqual(routing["missing"], 1)
        self.assertEqual(current_missing, {"routing": 1, "crunch": 1, "cache": 1})
        self.assertEqual(not_instrumented, {"routing": 1})
        self.assertEqual(historical, {"routing": 1, "crunch": 1, "cache": 1})
        self.assertIn("historical-unavailable", sample_states)
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_codex_effectiveness_classifies_workflow_phases_from_event_sequences(self):
        def log_turn(
            name,
            *,
            start_at,
            signal_method=None,
            phase_thread=None,
            routing=None,
            crunch=None,
            cache=None,
            input_text_chars=120,
            result_chars=80,
            response_error_code=None,
            latency_ms=100,
        ):
            thread_id = phase_thread or f"thread-{name}"
            request_id = f"req-{name}"
            server.store.log_codex_app_event(
                id=f"start-{name}",
                created_at=f"2026-06-08T10:{start_at:02d}:00+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=200,
                params_chars=100,
                input_items=1 if input_text_chars else 0,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-phase-session",
                routing_json=stable_json(routing or {
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                    "policy_source": "local-default",
                }),
                crunch_json=stable_json(crunch or {"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
                cache_json=stable_json(cache or {"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False, "policy_source": "local-default"}),
            )
            if signal_method:
                server.store.log_codex_app_event(
                    id=f"signal-{name}",
                    created_at=f"2026-06-08T10:{start_at:02d}:01+00:00",
                    direction="server_to_client",
                    method=signal_method,
                    request_id=None,
                    thread_id=thread_id,
                    message_chars=90,
                    params_chars=None,
                    input_items=None,
                    input_text_chars=None,
                    result_chars=None,
                    error_code=None,
                    error_message=None,
                    latency_ms=None,
                    session_id="codex-phase-session",
                )
            server.store.log_codex_app_event(
                id=f"end-{name}",
                created_at=f"2026-06-08T10:{start_at:02d}:02+00:00",
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=thread_id,
                message_chars=120,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=response_error_code,
                error_message="phase fixture error" if response_error_code is not None else None,
                latency_ms=latency_ms,
                session_id="codex-phase-session",
            )

        log_turn(
            "planning",
            start_at=0,
            signal_method="turn/plan/updated",
            routing={"status": "applied", "reason": "fixture-route", "applied": True, "policy_source": "local-default"},
            input_text_chars=400,
        )
        log_turn(
            "tool",
            start_at=1,
            signal_method="item/commandExecution/outputDelta",
            response_error_code=-32000,
            latency_ms=300,
            input_text_chars=240,
        )
        log_turn(
            "verification",
            start_at=2,
            signal_method="turn/diff/updated",
            crunch={"status": "applied", "reason": "fixture-crunch", "applied": True, "changed": True, "saved_chars": 40, "tokens_saved_est": 10},
            input_text_chars=160,
        )
        log_turn(
            "summary",
            start_at=3,
            signal_method="item/agentMessage/delta",
            cache={"status": "hit", "reason": "exact-match", "eligible": True, "hit_type": "exact", "policy_source": "local-default"},
            input_text_chars=80,
        )
        log_turn("idle", start_at=4, input_text_chars=0, result_chars=10)
        log_turn("unknown", start_at=5, input_text_chars=140)

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=20))
        phases = {row["phase"]: row for row in result["workflow_phase_breakdown"]}

        self.assertEqual(result["summary"]["turn_start_rows"], 6)
        self.assertEqual(result["summary"]["workflow_phase_known"], 5)
        self.assertEqual(result["summary"]["workflow_phase_unknown"], 1)
        self.assertEqual(phases["planning"]["routing_applied"], 1)
        self.assertEqual(phases["tool_execution"]["errors"], 1)
        self.assertEqual(phases["tool_execution"]["avg_latency_ms"], 300)
        self.assertEqual(phases["verification"]["crunch_applied"], 1)
        self.assertEqual(phases["verification"]["saved_chars"], 40)
        self.assertEqual(phases["summary"]["cache_hits"], 1)
        self.assertEqual(phases["idle_control"]["turns"], 1)
        self.assertEqual(phases["unknown"]["phase_reasons"][0]["value"], "insufficient-metadata")
        self.assertGreater(phases["planning"]["input_tokens_est"], 0)
        self.assertGreaterEqual(phases["planning"]["cost_est_usd"], 0)
        sample_phases = {row["workflow_phase"] for row in result["recent_samples"]}
        self.assertIn("planning", sample_phases)
        self.assertFalse(result["privacy"]["raw_params_included"])

        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        with TestClient(app) as client:
            response = client.get("/agentflow/stats/codex-effectiveness?limit=20")
        self.assertEqual(response.status_code, 200)
        endpoint_payload = response.json()
        endpoint_phases = {row["phase"] for row in endpoint_payload["workflow_phase_breakdown"]}
        self.assertEqual(endpoint_phases, set(phases))

    def test_codex_effectiveness_reports_repeated_context_plateau_candidates(self):
        raw_prompt_text = "raw prompt must not appear in plateau report"
        sizes = [10_000, 10_100, 9_950, 10_150]
        saved = [10, 20, 0, 0]
        for index, input_text_chars in enumerate(sizes):
            request_id = f"req-plateau-{index}"
            server.store.log_codex_app_event(
                id=f"start-plateau-{index}",
                created_at=f"2026-06-08T12:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id="thread-plateau-candidate",
                message_chars=input_text_chars + 100,
                params_chars=input_text_chars + 50,
                input_items=2,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-plateau-candidate",
                routing_json=stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                    "policy_source": "local-default",
                }),
                crunch_json=stable_json({
                    "status": "applied" if saved[index] else "skipped",
                    "reason": "codex-repeated-scaffolding-crunched" if saved[index] else "no-change",
                    "applied": bool(saved[index]),
                    "changed": bool(saved[index]),
                    "saved_chars": saved[index],
                    "tokens_saved_est": saved[index] // 4,
                }),
                cache_json=stable_json({
                    "status": "miss",
                    "reason": "exact-miss",
                    "eligible": True,
                    "policy_source": "local-default",
                }),
                event_window_json=stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 3,
                    "method_counts": {"turn/start": 1, "item/agentMessage/delta": 2},
                    "direction_counts": {"client_to_server": 1, "server_to_client": 2},
                    "input_text_chars": input_text_chars,
                    "debug_prompt": raw_prompt_text,
                }),
            )
            server.store.log_codex_app_event(
                id=f"end-plateau-{index}",
                created_at=f"2026-06-08T12:00:1{index}+00:00",
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id="thread-plateau-candidate",
                message_chars=180,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=160,
                error_code=None,
                error_message=None,
                latency_ms=100 + index,
                session_id="session-plateau-candidate",
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        report = result["repeated_context_plateau_candidates"]
        [candidate] = report["candidates"]
        encoded = json.dumps(result)

        self.assertEqual(result["summary"]["repeated_context_plateau_candidate_count"], 1)
        self.assertEqual(candidate["scope_id"], "thread-plateau-candidate")
        self.assertEqual(candidate["scope_basis"], "thread_id")
        self.assertEqual(candidate["turns"], 4)
        self.assertEqual(candidate["plateau_count"], 3)
        self.assertEqual(candidate["candidate_pairs"], 3)
        self.assertEqual(candidate["median_input_chars"], 10_050)
        self.assertEqual(candidate["p90_input_chars"], 10_150)
        self.assertEqual(candidate["current_saved_chars"], 30)
        self.assertGreater(candidate["estimated_opportunity_saved_chars"], 0)
        self.assertGreater(candidate["estimated_opportunity_tokens"], 0)
        self.assertEqual(report["policy"]["min_input_chars"], 8_000)
        self.assertFalse(result["privacy"]["raw_params_included"])
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("debug_prompt", encoded)

    def test_codex_effectiveness_uses_persisted_metadata_only_event_window(self):
        secret = "raw prompt must not appear"
        server.store.log_codex_app_event(
            id="start-window",
            created_at="2026-06-08T10:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="req-window",
            thread_id="thread-window",
            message_chars=240,
            params_chars=180,
            input_items=1,
            input_text_chars=96,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="session-window",
            routing_json=stable_json({
                "status": "not-applicable",
                "reason": "codex-turn-start-model-field-absent",
                "applied": False,
                "policy_source": "local-default",
            }),
            crunch_json=stable_json({"status": "skipped", "reason": "no-change", "applied": False, "changed": False}),
            cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False}),
            event_window_json=stable_json({
                "schema": "agentflow.codex_app_event_window.v1",
                "start_event_id": "start-window",
                "created_at": "2026-06-08T10:00:00+00:00",
                "session_id": "session-window",
                "request_id": "req-window",
                "thread_id": "thread-window",
                "event_count": 4,
                "method_counts": {
                    "turn/start": 1,
                    "turn/diff/updated": 2,
                    "turn/completed": 1,
                },
                "direction_counts": {"client_to_server": 1, "server_to_client": 3},
                "first_event_delta_ms": 0,
                "last_event_delta_ms": 1200,
                "input_items": 1,
                "input_text_chars": 96,
                "start_message_chars": 240,
                "start_params_chars": 180,
                "result_chars": 52,
                "server_message_chars": 300,
                "error_count": 0,
                "model_field_state": "absent",
                "debug_prompt": secret,
            }),
        )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=5))

        self.assertEqual(result["summary"]["workflow_phase_known"], 1)
        self.assertEqual(result["workflow_phase_breakdown"][0]["phase"], "verification")
        self.assertEqual(result["workflow_phase_source_breakdown"][0]["value"], "event_window")
        sample = result["recent_samples"][0]
        self.assertEqual(sample["workflow_phase_source"], "event_window")
        self.assertEqual(sample["workflow_phase_reason"], "event-window-signal:verification")
        self.assertEqual(sample["event_window"]["event_count"], 4)
        self.assertEqual(sample["event_window"]["model_field_state"], "absent")
        self.assertTrue(sample["event_window"]["request_id_present"])
        self.assertNotIn("debug_prompt", json.dumps(sample["event_window"]))
        self.assertNotIn(secret, json.dumps(result))
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_old_context_summary_stats_are_attributed_separately(self):
        for cache_hit, cost in ((False, 0.0002), (True, 0.0)):
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
                cost_est_usd=cost,
                cost_baseline_usd=0.0,
                crunch_json=stable_json({
                    "changed": False,
                    "old_context_summarization": {
                        "status": "applied",
                        "reason": "summary-cache-hit" if cache_hit else "summary-created",
                        "summary_cache_hit": cache_hit,
                        "summary_cost_est_usd": cost,
                        "saved_chars": 2_000,
                        "tokens_saved_est": 500,
                    },
                }),
                routing_json=None,
                cache_json=None,
                error=None,
                request_json=None,
                response_json=None,
                session_id="session-summary",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        summary = result["summary"]

        self.assertEqual(summary["old_context_summary_applied_count"], 2)
        self.assertEqual(summary["old_context_summary_created_count"], 1)
        self.assertEqual(summary["old_context_summary_cache_hits"], 1)
        self.assertAlmostEqual(summary["old_context_summary_cache_hit_rate"], 0.5, places=6)
        self.assertEqual(summary["old_context_summary_tokens_saved"], 1_000)
        self.assertAlmostEqual(summary["old_context_summary_cost_usd"], 0.0002, places=6)
        self.assertAlmostEqual(summary["old_context_summary_savings_usd"], 0.003, places=6)
        self.assertAlmostEqual(summary["today_old_context_summary_net_usd"], 0.0028, places=6)

    def test_cache_decision_breakdown_groups_status_reason_and_hit_type(self):
        rows = [
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "skipped", "reason": "streaming", "policy_source": "local-default"},
            {"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            {"status": "miss", "reason": "file-dependency-changed", "policy_source": "local-default"},
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

        result = asyncio.run(stats_views.stats_full(server.store))
        breakdown = {
            (row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in result["cache_decision_breakdown"]
        }

        self.assertEqual(breakdown[("skipped", "streaming", "")], 2)
        self.assertEqual(breakdown[("miss", "exact-miss", "")], 1)
        self.assertEqual(breakdown[("miss", "file-dependency-changed", "")], 1)
        self.assertEqual(breakdown[("hit", "exact-match", "exact")], 1)
        json.dumps(result["cache_decision_breakdown"])

    def test_cache_decision_breakdown_infers_legacy_null_cache_rows(self):
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:00:00+00:00",
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
            cost_est_usd=0.0,
            cost_baseline_usd=0.0,
            crunch_json=stable_json({"changed": False}),
            routing_json=None,
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:01:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=1,
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
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-hit-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:02:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=0,
            cache_hit=0,
            status_code=400,
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
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-error-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at="2020-01-01T00:03:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
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
            cache_json=None,
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-unknown-old",
            category="chat",
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
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-today",
            category="chat",
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
            cache_json=stable_json({"hit_type": "skip-streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="session-cache-partial",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        all_time = {
            (row["status"], row["reason"], row["policy_source"]): row["count"]
            for row in result["cache_decision_breakdown"]
        }
        today = {
            (row["status"], row["reason"]): row["count"]
            for row in result["today_cache_decision_breakdown"]
        }

        self.assertEqual(all_time[("skipped", "legacy-streaming", "legacy-inferred")], 1)
        self.assertEqual(all_time[("skipped", "legacy-streaming", "local-default")], 1)
        self.assertEqual(all_time[("hit", "legacy-cache-hit", "legacy-inferred")], 1)
        self.assertEqual(all_time[("skipped", "legacy-upstream-error", "legacy-inferred")], 1)
        self.assertEqual(all_time[("missing", "legacy-unknown", "legacy-inferred")], 1)
        self.assertEqual(today[("skipped", "streaming")], 1)
        self.assertEqual(today[("skipped", "legacy-streaming")], 1)
        json.dumps(result["today_cache_decision_breakdown"])

    def test_codex_cache_breakdown_merges_legacy_and_canonical_source_surfaces(self):
        for surface in ("codex_app_turn", "codex_turn"):
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-{surface}",
                thread_id="thread-codex-surface",
                message_chars=100,
                params_chars=50,
                input_items=1,
                input_text_chars=40,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="codex-surface-session",
                cache_json=stable_json({
                    "status": "hit",
                    "reason": "exact-match",
                    "hit_type": "exact",
                    "policy_source": "local-default",
                    "surface": surface,
                }),
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        cache_rows = {
            (row["source_surface"], row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in result["today_cache_decision_breakdown"]
        }

        self.assertEqual(cache_rows[("codex_turn", "hit", "exact-match", "exact")], 2)
        self.assertNotIn(("codex_app_turn", "hit", "exact-match", "exact"), cache_rows)

    def test_cache_replayability_report_groups_repeated_skipped_shapes_and_blockers(self):
        def log_provider_call(
            *,
            cache_json,
            routing_json,
            session_id,
            stream,
            category,
            cost,
            request_json=None,
        ):
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=stream,
                cache_hit=0,
                status_code=200,
                latency_ms=1,
                input_tokens_est=10,
                output_tokens_est=1,
                actual_input_tokens=10,
                actual_output_tokens=1,
                cost_est_usd=cost,
                cost_baseline_usd=cost,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json(routing_json),
                cache_json=stable_json(cache_json),
                error=None,
                request_json=request_json,
                response_json=None,
                session_id=session_id,
                category=category,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=0,
                provider="anthropic",
            )

        streaming_cache = {
            "status": "skipped",
            "reason": "streaming",
            "policy_source": "local-default",
            "semantic_enabled": False,
            "tool_cache_enabled": False,
            "file_watch_enabled": False,
        }
        streaming_routing = {"text_chars": 12_000, "category": "chat", "has_tools": False}
        log_provider_call(
            cache_json=streaming_cache,
            routing_json=streaming_routing,
            session_id="session-stream-a",
            stream=1,
            category="chat",
            cost=0.01,
            request_json=stable_json({"messages": [{"content": "raw-secret-should-not-leak"}]}),
        )
        log_provider_call(
            cache_json=streaming_cache,
            routing_json=streaming_routing,
            session_id="session-stream-a",
            stream=1,
            category="chat",
            cost=0.02,
        )

        tool_cache = {
            "status": "skipped",
            "reason": "tools-disabled",
            "policy_source": "local-default",
            "semantic_enabled": False,
            "tool_cache_enabled": False,
            "file_watch_enabled": False,
        }
        tool_routing = {"text_chars": 24_000, "category": "tool-result", "has_tools": True}
        log_provider_call(
            cache_json=tool_cache,
            routing_json=tool_routing,
            session_id="session-tool-a",
            stream=0,
            category="tool-result",
            cost=0.03,
        )
        log_provider_call(
            cache_json=tool_cache,
            routing_json=tool_routing,
            session_id="session-tool-b",
            stream=0,
            category="tool-result",
            cost=0.04,
        )

        log_provider_call(
            cache_json={"status": "miss", "reason": "exact-miss", "policy_source": "local-default"},
            routing_json={"text_chars": 400, "category": "chat", "has_tools": False},
            session_id="session-one-off",
            stream=0,
            category="chat",
            cost=0.005,
        )

        result = asyncio.run(stats_views.stats_cache_replayability(server.store, limit=10))
        groups = {(row["cache_reason"], row["category"]): row for row in result["groups"]}

        self.assertEqual(result["schema"], "agentflow.cache_replayability.v1")
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertEqual(result["summary"]["repeated_shape_groups"], 2)
        self.assertTrue(result["summary"]["repeated_shape_exists_but_cache_is_unsafe"])
        self.assertEqual(groups[("streaming", "chat")]["count"], 2)
        self.assertIn("streaming", groups[("streaming", "chat")]["replayability_blockers"])
        self.assertEqual(groups[("tools-disabled", "tool-result")]["count"], 2)
        self.assertEqual(groups[("tools-disabled", "tool-result")]["sessions"], 2)
        self.assertIn("tool-call-disabled", groups[("tools-disabled", "tool-result")]["replayability_blockers"])
        self.assertIn("file-dependency-unknown", groups[("tools-disabled", "tool-result")]["replayability_blockers"])
        self.assertIn("session-context-changed", groups[("tools-disabled", "tool-result")]["replayability_blockers"])
        self.assertIn("true-one-off-miss", groups[("exact-miss", "chat")]["replayability_blockers"])
        self.assertNotIn("raw-secret-should-not-leak", json.dumps(result))

        by_blocker = {row["blocker"]: row["calls"] for row in result["blocker_breakdown"]}
        self.assertEqual(by_blocker["streaming"], 2)
        self.assertEqual(by_blocker["tool-call-disabled"], 2)
        self.assertEqual(by_blocker["true-one-off-miss"], 1)

    def test_cache_replayability_endpoint_and_dashboard_are_read_only_metadata(self):
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
            cost_est_usd=0.01,
            cost_baseline_usd=0.01,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"text_chars": 12_000, "category": "chat", "has_tools": False}),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=stable_json({"messages": [{"content": "private request body"}]}),
            response_json=None,
            session_id="session-api",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        app = create_dashboard_app(
            store_obj=server.store,
            default_db=self.tmp.name,
            upstream="https://api.anthropic.com",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )

        with TestClient(app) as client:
            response = client.get("/agentflow/stats/cache-replayability?limit=5")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["schema"], "agentflow.cache_replayability.v1")
            self.assertFalse(data["privacy"]["raw_prompts_included"])
            self.assertNotIn("private request body", json.dumps(data))
            html = client.get("/agentflow/dashboard").text
            self.assertIn("Skipped cache replayability", html)
            self.assertIn("cache-replayability-tbody", html)

    def test_codex_effectiveness_counts_direct_derived_absent_and_unknown_model_state(self):
        rows = [
            (
                "direct-model",
                stable_json({
                    "status": "skipped",
                    "reason": "keep requested model",
                    "model_field": "model",
                    "applied": False,
                }),
                None,
            ),
            (
                "derived-model",
                stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                }),
                stable_json({
                    "schema": "agentflow.codex_app_event_window.v1",
                    "event_count": 2,
                    "method_counts": {"turn/start": 1, "initialize": 1},
                    "direction_counts": {"client_to_server": 2},
                    "model_field_state": "derived_present",
                    "model_field": "model",
                    "model_state": {
                        "state": "derived_present",
                        "field": "model",
                        "normalized_model": "gpt-5-codex",
                        "source_method": "initialize",
                        "confidence": "high",
                        "reason": "metadata-model-field",
                    },
                }),
            ),
            (
                "absent-model",
                stable_json({
                    "status": "not-applicable",
                    "reason": "codex-turn-start-model-field-absent",
                    "applied": False,
                }),
                None,
            ),
            ("legacy-unknown", None, None),
        ]
        for index, (suffix, routing_json, event_window_json) in enumerate(rows):
            server.store.log_codex_app_event(
                id=f"start-{suffix}",
                created_at=f"2026-06-08T10:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=f"req-{suffix}",
                thread_id=f"thread-{suffix}",
                message_chars=120,
                params_chars=80,
                input_items=1,
                input_text_chars=64,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="session-model-state",
                routing_json=routing_json,
                crunch_json=stable_json({"status": "skipped", "reason": "no-change", "applied": False}),
                cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled", "eligible": False}),
                event_window_json=event_window_json,
            )

        result = asyncio.run(stats_views.stats_codex_effectiveness(server.store, limit=10))
        summary = result["summary"]
        breakdown = {row["value"]: row["count"] for row in result["model_field_breakdown"]}
        names = {row["value"]: row["count"] for row in result["model_field_names"]}
        derived_sample = next(
            sample for sample in result["recent_samples"]
            if sample["event_window"].get("model_field_state") == "derived_present"
        )

        self.assertEqual(summary["turn_start_rows"], 4)
        self.assertEqual(summary["model_field_present"], 2)
        self.assertEqual(summary["model_field_derived"], 1)
        self.assertEqual(summary["model_field_absent"], 1)
        self.assertEqual(summary["model_field_unknown"], 1)
        self.assertEqual(breakdown["present"], 1)
        self.assertEqual(breakdown["derived_present"], 1)
        self.assertEqual(breakdown["absent"], 1)
        self.assertEqual(breakdown["unknown"], 1)
        self.assertEqual(names["model"], 2)
        self.assertEqual(derived_sample["model_field"], "derived_present")
        self.assertEqual(derived_sample["event_window"]["model_state"]["normalized_model"], "gpt-5-codex")
        self.assertFalse(result["privacy"]["raw_params_included"])

    def test_codex_app_cache_hit_counts_decision_and_saved_cost(self):
        cache_meta = {
            "enabled": True,
            "status": "hit",
            "reason": "exact-match",
            "hit_type": "exact",
            "eligible": True,
            "policy_source": "local-default",
            "surface": "codex_app_turn",
            "replayability_level": "local-exact-response",
        }
        start_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-codex-cache",
            thread_id="thread-cache",
            message_chars=600,
            params_chars=500,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-cache-session",
            cache_json=stable_json(cache_meta),
        )
        server.store.log_codex_app_event(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            direction="server_to_client",
            method=None,
            request_id="req-codex-cache",
            thread_id="thread-cache",
            message_chars=100,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=80,
            error_code=None,
            error_message=None,
            latency_ms=5,
            session_id="codex-cache-session",
        )

        full = asyncio.run(stats_views.stats_full(server.store))
        activity = asyncio.run(stats_views.stats_activity(server.store))
        usage = asyncio.run(stats_views.stats_usage_by_owner(server.store))

        self.assertEqual(full["summary"]["today_codex_app_cost_est_usd"], 0.0)
        self.assertGreater(full["summary"]["today_codex_app_cache_savings_usd"], 0.0)
        self.assertAlmostEqual(
            full["executive_summary"]["savings"]["today_buckets"]["codex_app_exact_local_cache_usd"],
            full["summary"]["today_codex_app_cache_savings_usd"],
            places=6,
        )
        cache_rows = {
            (row["source_surface"], row["status"], row["reason"], row["hit_type"]): row["count"]
            for row in full["today_cache_decision_breakdown"]
        }
        self.assertEqual(cache_rows[("codex_turn", "hit", "exact-match", "exact")], 1)

        codex = {unit["unit_id"]: unit for unit in activity["units"]}[f"codex_turn:{start_id}"]
        self.assertEqual(codex["replayability_level"], "local-exact-response")
        self.assertEqual(codex["optimization_features"]["cache"]["eligible"], True)
        self.assertEqual(codex["outcome_features"]["cost_est_usd"], 0.0)
        self.assertGreater(codex["outcome_features"]["cache_savings_usd"], 0.0)

        [bucket] = usage["buckets"]
        self.assertEqual(bucket["local_cache_hits"], 1)
        self.assertGreater(bucket["codex_exact_cache_savings_usd"], 0.0)
        self.assertAlmostEqual(bucket["spend_usd"], 0.0, places=6)
        self.assertGreater(bucket["captured_savings_usd"], 0.0)
        json.dumps(full)
        json.dumps(activity)
        json.dumps(usage)

    def test_error_breakdown_groups_sanitized_error_families(self):
        legacy_id = str(uuid.uuid4())
        server.store.log_call(
            id=legacy_id,
            created_at="2020-01-01T00:00:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=400,
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
            error=stable_json({
                "error": {
                    "message": "This model does not support the effort parameter.",
                    "type": "invalid_request_error",
                },
            }),
            request_json=None,
            response_json=None,
            session_id="session-error-old",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="anthropic",
        )
        server.store.conn.execute("update calls set provider = null where id = ?", (legacy_id,))
        server.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-codex",
            stream=0,
            cache_hit=0,
            status_code=401,
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
            error=stable_json({
                "error": {
                    "code": "invalid_api_key",
                    "message": "Incorrect API key provided: intentionally_invalid.",
                },
            }),
            request_json=None,
            response_json=None,
            session_id="session-error-today",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            provider="openai",
        )

        result = asyncio.run(stats_views.stats_full(server.store))
        all_time = {
            (row["provider"], row["status_code"], row["tier"], row["error_type"]): row
            for row in result["error_breakdown"]
        }
        today = {
            (row["provider"], row["status_code"], row["error_type"]): row
            for row in result["today_error_breakdown"]
        }

        legacy = all_time[("anthropic", 400, "haiku", "model_incompatible_param")]
        self.assertEqual(legacy["count"], 1)
        self.assertEqual(legacy["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(legacy["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(legacy["error_sample"], "This model does not support the effort parameter.")
        self.assertIn(("openai", 401, "auth_error"), today)
        self.assertNotIn(("anthropic", 400, "model_incompatible_param"), today)
        json.dumps(result["error_breakdown"])
        json.dumps(result["today_error_breakdown"])

    def test_routing_experiment_stats_produce_confidence_scores(self):
        for similarity, passed in ((0.9, 1), (0.7, 0)):
            server.store.log_routing_experiment(
                id=str(uuid.uuid4()),
                call_id=str(uuid.uuid4()),
                created_at=utc_now(),
                requested_model="claude-sonnet-4-6",
                routed_model="claude-haiku-4-5-20251001",
                primary_model="claude-haiku-4-5-20251001",
                shadow_model="claude-sonnet-4-6",
                category="tool-result",
                routing_reason="tool-result processing turn routed to Haiku",
                input_tokens_est=100,
                primary_status_code=200,
                shadow_status_code=200,
                primary_latency_ms=50,
                shadow_latency_ms=75,
                primary_output_chars=12,
                shadow_output_chars=14,
                primary_output_sha256="primary",
                shadow_output_sha256="shadow",
                output_similarity=similarity,
                passed_threshold=passed,
                primary_cost_est_usd=0.001,
                shadow_cost_est_usd=0.003,
                error=None,
                routing_json=stable_json({"reason": "tool-result processing turn routed to Haiku"}),
                experiment_json=stable_json({"sampled": True}),
                primary_response_json=None,
                shadow_response_json=None,
            )

        result = asyncio.run(stats_views.stats_full(server.store))
        [row] = result["routing_experiment_summary"]

        self.assertEqual(result["summary"]["routing_experiment_samples"], 2)
        self.assertEqual(result["summary"]["routing_experiment_compared_samples"], 2)
        self.assertAlmostEqual(result["summary"]["routing_experiment_avg_similarity"], 0.8, places=6)
        self.assertEqual(result["summary"]["routing_experiment_feedback_status_counts"], {"not-exported": 2})
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["compared_samples"], 2)
        self.assertAlmostEqual(row["avg_similarity"], 0.8, places=6)
        self.assertAlmostEqual(row["pass_rate"], 0.5, places=6)
        self.assertAlmostEqual(row["confidence_score"], 0.08, places=6)
        self.assertEqual(row["min_samples_for_confidence"], 20)
        json.dumps(result["routing_experiment_summary"])

    def test_sessions_include_thinking_token_breakdown(self):
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
            session_id="session-thinking",
            category="tool-heavy",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=1_000,
            provider="anthropic",
        )

        result = asyncio.run(stats_views.stats_sessions(server.store))
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

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]

        self.assertEqual(session["session_id"], "session-cache-warmup")
        self.assertEqual(session["cache_creation_tokens"], 1_000)
        self.assertEqual(session["cache_read_tokens"], 10_000)
        self.assertEqual(session["cache_write_read_token_ratio"], 0.1)
        self.assertAlmostEqual(session["cache_creation_cost_usd"], 0.00375, places=6)
        self.assertAlmostEqual(session["cache_read_savings_usd"], 0.027, places=6)
        self.assertEqual(session["cache_warmup_payback_ratio"], 0.139)
        json.dumps(result)

    def test_sessions_include_codex_app_turns_without_raw_payloads(self):
        raw_prompt_text = "secret raw prompt must not appear"

        def log_codex_turn(
            request_id: str,
            *,
            input_text_chars: int,
            result_chars: int | None,
            routing: dict | None = None,
            crunch: dict | None = None,
            cache: dict | None = None,
            error_code: int | None = None,
            latency_ms: int = 100,
        ) -> None:
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id="thread-codex-sessions",
                message_chars=input_text_chars + 50,
                params_chars=input_text_chars + 20,
                input_items=2,
                input_text_chars=input_text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=None,
                routing_json=stable_json(routing) if routing is not None else None,
                crunch_json=stable_json(crunch) if crunch is not None else None,
                cache_json=stable_json(cache) if cache is not None else None,
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id="thread-codex-sessions",
                message_chars=(result_chars or 0) + 20,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=result_chars,
                error_code=error_code,
                error_message="metadata-only error" if error_code is not None else None,
                latency_ms=latency_ms,
                session_id=None,
            )

        log_codex_turn(
            "codex-s1",
            input_text_chars=10_000,
            result_chars=400,
            routing={
                "applied": True,
                "requested_model": "gpt-5.3-codex",
                "routed_model": "gpt-5.1-codex",
                "policy_source": "local-manual",
            },
            crunch={
                "changed": True,
                "applied": True,
                "saved_chars": 1_200,
                "policy_source": "local-default",
            },
            cache={
                "status": "miss",
                "reason": "exact-miss",
                "policy_source": "local-default",
            },
        )
        log_codex_turn(
            "codex-s2",
            input_text_chars=10_200,
            result_chars=200,
            cache={
                "status": "hit",
                "reason": "exact-match",
                "hit_type": "exact",
                "policy_source": "local-default",
            },
        )
        log_codex_turn(
            "codex-s3",
            input_text_chars=14_000,
            result_chars=None,
            error_code=-32000,
        )

        result = asyncio.run(stats_views.stats_sessions(server.store))
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]
        encoded = json.dumps(result)

        self.assertTrue(session["session_id"].startswith("codex-workflow:"))
        self.assertEqual(session["session_key_basis"], "workflow_thread_id")
        self.assertEqual(
            session["codex_workflow_grouping"]["original_key_basis_counts"],
            {"thread_id": 3},
        )
        self.assertEqual(session["codex_workflow_grouping"]["original_key_count"], 1)
        self.assertFalse(session["codex_workflow_grouping"]["raw_keys_included"])
        self.assertEqual(session["source_surface"], "codex_turn")
        self.assertEqual(session["app_family"], "codex")
        self.assertEqual(session["calls"], 3)
        self.assertEqual(session["provider_calls"], 0)
        self.assertEqual(session["codex_turns"], 3)
        self.assertEqual(session["codex_input_text_chars"], 34_200)
        self.assertEqual(session["codex_input_tokens_est"], 8_550)
        self.assertEqual(session["codex_output_tokens_est"], 150)
        self.assertEqual(session["codex_total_tokens_est"], 8_700)
        self.assertGreater(session["codex_cost_est_usd"], 0.0)
        self.assertGreater(session["codex_baseline_cost_est_usd"], session["codex_cost_est_usd"])
        self.assertGreater(session["codex_exact_cache_savings_usd"], 0.0)
        self.assertEqual(session["codex_routed_turns"], 1)
        self.assertEqual(session["codex_crunched_turns"], 1)
        self.assertEqual(session["codex_cache_hits"], 1)
        self.assertEqual(session["codex_optimized_turns"], 2)
        self.assertEqual(session["codex_errors"], 1)
        self.assertEqual(session["codex_method_counts"], [{"method": "turn/start", "turns": 3}])
        self.assertEqual(plateau["session_id"], session["session_id"])
        self.assertEqual(plateau["session_key_basis"], "workflow_thread_id")
        self.assertEqual(plateau["source_surface"], "codex_turn")
        self.assertEqual(plateau["calls"], 3)
        self.assertEqual(plateau["plateau_pairs"], 1)
        self.assertEqual(plateau["median_text_chars"], 10_200)
        self.assertEqual(plateau["p90_text_chars"], 14_000)
        self.assertEqual(plateau["crunch_saved_chars"], 1_200)
        self.assertGreater(plateau["cache_read_savings_usd"], 0.0)
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("raw_prompt", encoded)
        self.assertNotIn("thread-codex-sessions", encoded)

    def test_sessions_group_fragmented_codex_turns_into_workflow_window(self):
        raw_prompt_text = "secret raw codex prompt must not appear"
        created_base = utc_now()
        text_sizes = [10_000, 10_100, 10_050, 10_080, 10_060]
        phases = ["planning", "planning", "tool_execution", "summary", "summary"]
        for idx, text_chars in enumerate(text_sizes):
            request_id = f"request-fragment-{idx}"
            session_id = f"codex-fragment-session-{idx}"
            server.store.log_codex_app_event(
                id=f"fragment-start-{idx}",
                created_at=created_base,
                direction="client_to_server",
                method="turn/start",
                request_id=request_id,
                thread_id=None,
                message_chars=text_chars + 25,
                params_chars=text_chars + 10,
                input_items=2,
                input_text_chars=text_chars,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=session_id,
                routing_json=stable_json({
                    "status": "skipped",
                    "applied": False,
                    "workflow_phase": phases[idx],
                    "reason": "test-metadata-only",
                }),
                crunch_json=stable_json({
                    "changed": idx == 2,
                    "saved_chars": 400 if idx == 2 else 0,
                    "workflow_phase": phases[idx],
                }),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            )
            server.store.log_codex_app_event(
                id=f"fragment-response-{idx}",
                created_at=created_base,
                direction="server_to_client",
                method="turn/completed",
                request_id=request_id,
                thread_id=None,
                message_chars=200,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=200,
                error_code=None,
                error_message=None,
                latency_ms=100 + idx,
                session_id=session_id,
            )

        app = create_dashboard_app(
            store_obj=lambda: server.store,
            default_db=self.tmp.name,
            upstream="https://anthropic.test",
            limiter_status=lambda: [],
            limiter_config={},
            full_stats_ttl_s=0,
        )
        response = TestClient(app).get("/agentflow/stats/sessions")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        [session] = result["sessions"]
        [plateau] = result["context_plateaus"]
        encoded = json.dumps(result)

        self.assertTrue(session["session_id"].startswith("codex-workflow:"))
        self.assertEqual(session["session_key_basis"], "workflow_window")
        self.assertEqual(session["source_surface"], "codex_turn")
        self.assertEqual(session["app_family"], "codex")
        self.assertEqual(session["calls"], 5)
        self.assertEqual(session["codex_turns"], 5)
        self.assertEqual(session["provider_calls"], 0)
        self.assertEqual(session["codex_input_text_chars"], sum(text_sizes))
        self.assertGreater(session["codex_cost_est_usd"], 0.0)
        self.assertEqual(session["codex_crunched_turns"], 1)
        self.assertEqual(
            session["codex_workflow_grouping"]["original_key_basis_counts"],
            {"session_id": 5},
        )
        self.assertEqual(session["codex_workflow_grouping"]["original_key_count"], 5)
        self.assertFalse(session["codex_workflow_grouping"]["raw_keys_included"])
        self.assertEqual(
            {row["phase"]: row["turns"] for row in session["codex_workflow_phase_counts"]},
            {"planning": 2, "summary": 2, "tool_execution": 1},
        )
        self.assertEqual(plateau["session_id"], session["session_id"])
        self.assertEqual(plateau["session_key_basis"], "workflow_window")
        self.assertEqual(plateau["calls"], 5)
        self.assertEqual(plateau["plateau_pairs"], 4)
        self.assertNotIn(raw_prompt_text, encoded)
        self.assertNotIn("raw_prompt", encoded)
        self.assertNotIn("raw_response", encoded)
        self.assertNotIn("codex-fragment-session-", encoded)
        self.assertNotIn("request-fragment-", encoded)

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

    def test_activity_stats_normalize_provider_calls_and_codex_turns(self):
        provider_id = str(uuid.uuid4())
        server.store.log_call(
            id=provider_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001",
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=12,
            input_tokens_est=100,
            output_tokens_est=20,
            actual_input_tokens=90,
            actual_output_tokens=18,
            cost_est_usd=0.001,
            cost_baseline_usd=0.004,
            crunch_json=stable_json({"changed": True, "tokens_saved_est": 10, "policy_source": "local-default"}),
            routing_json=stable_json({
                "reason": "tool result routed to Haiku",
                "text_chars": 360,
                "has_tools": True,
                "category": "tool-result",
                "policy_source": "local-manual",
            }),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id="provider-session",
            category="tool-result",
            cache_creation_input_tokens=5,
            cache_read_input_tokens=50,
            retry_count=1,
            thinking_output_tokens=0,
            provider="anthropic",
        )
        start_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        raw_prompt_text = "secret raw prompt must not appear"
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-1",
            thread_id="thread-1",
            message_chars=500,
            params_chars=450,
            input_items=2,
            input_text_chars=len(raw_prompt_text),
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-session",
        )
        server.store.log_codex_app_event(
            id=response_id,
            created_at=utc_now(),
            direction="server_to_client",
            method="turn/completed",
            request_id="req-1",
            thread_id="thread-1",
            message_chars=300,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=200,
            error_code=None,
            error_message=None,
            latency_ms=3000,
            session_id="codex-session",
        )

        result = asyncio.run(stats_views.stats_activity(server.store))
        units = {unit["unit_id"]: unit for unit in result["units"]}
        provider = units[f"provider_call:{provider_id}"]
        codex = units[f"codex_turn:{start_id}"]

        self.assertEqual(result["schema"], "agentflow.optimization_activity.v1")
        self.assertEqual(provider["source_surface"], "anthropic_messages")
        self.assertEqual(provider["granularity"], "provider_request")
        self.assertEqual(provider["app_family"], "claude_code")
        self.assertEqual(provider["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(provider["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(provider["tool_features"]["has_tools"], True)
        self.assertEqual(provider["input_features"]["text_chars"], 360)
        self.assertEqual(provider["input_features"]["input_tokens"], 90)
        self.assertEqual(provider["optimization_features"]["cache"]["status"], "skipped")
        self.assertEqual(provider["optimization_features"]["crunch"]["changed"], True)
        self.assertEqual(provider["optimization_features"]["policy_sources"], ["local-default", "local-manual"])
        self.assertEqual(provider["outcome_features"]["status_code"], 200)
        self.assertEqual(provider["outcome_features"]["cost_est_usd"], 0.001)
        self.assertEqual(provider["replayability_level"], "features_only")
        self.assertEqual(provider["local_ids"]["calls_id"], provider_id)

        self.assertEqual(codex["schema"], "agentflow.optimization_unit.v1")
        self.assertEqual(codex["source_surface"], "codex_turn")
        self.assertEqual(codex["granularity"], "agent_turn")
        self.assertEqual(codex["app_family"], "codex")
        self.assertEqual(codex["requested_model"], stats_views.CODEX_APP_MODEL)
        self.assertEqual(codex["target_model"], stats_views.CODEX_APP_MODEL)
        self.assertEqual(codex["model_basis"], "estimated")
        self.assertEqual(codex["input_features"]["category"], "codex-app-turn")
        self.assertEqual(codex["input_features"]["input_text_chars"], len(raw_prompt_text))
        self.assertEqual(codex["input_features"]["input_tokens_est"], 8)
        self.assertEqual(codex["input_features"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertEqual(codex["tool_features"]["mutation_safe"], False)
        self.assertEqual(codex["tool_features"]["mutation_safe_reason"], "codex-app-telemetry-only")
        self.assertEqual(codex["tool_features"]["tool_or_approval_hints"]["captured"], False)
        self.assertEqual(codex["risk_features"]["params_shape"]["has_params"], True)
        self.assertEqual(codex["risk_features"]["params_shape"]["input_items"], 2)
        self.assertEqual(codex["risk_features"]["raw_prompt_stored"], False)
        self.assertEqual(codex["mutation_safe"], False)
        self.assertEqual(codex["optimization_features"]["routing"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["routing"]["reason"], "codex-app-telemetry-only")
        self.assertEqual(codex["optimization_features"]["crunch"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["cache"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["policy_sources"], ["local-default"])
        self.assertEqual(codex["outcome_features"]["status"], "success")
        self.assertEqual(codex["outcome_features"]["latency_ms"], 3000)
        self.assertEqual(codex["outcome_features"]["output_tokens_est"], 50)
        self.assertEqual(codex["outcome_features"]["total_tokens_est"], 58)
        self.assertEqual(codex["outcome_features"]["pricing_basis"]["model"], "gpt-5.3-codex")
        self.assertAlmostEqual(codex["outcome_features"]["cost_est_usd"], 0.000714, places=6)
        self.assertAlmostEqual(codex["outcome_features"]["cost_baseline_usd"], 0.000714, places=6)
        self.assertAlmostEqual(codex["outcome_features"]["hard_floor_usd"], 0.000714, places=6)
        self.assertEqual(codex["outcome_features"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertEqual(codex["replayability_level"], "features_only")
        self.assertEqual(codex["local_ids"]["codex_app_response_event_id"], response_id)
        self.assertNotIn(raw_prompt_text, json.dumps(codex))
        self.assertEqual(result["summary"]["provider_request_units"], 1)
        self.assertEqual(result["summary"]["codex_turn_units"], 1)
        self.assertEqual(result["summary"]["codex_app_turn_units"], 1)
        self.assertEqual(result["summary"]["by_source_surface"]["anthropic_messages"], 1)
        self.assertEqual(result["summary"]["by_source_surface"]["codex_turn"], 1)
        json.dumps(result)

    def test_activity_stats_use_codex_app_policy_metadata_when_recorded(self):
        start_id = str(uuid.uuid4())
        routing_meta = {
            "enabled": True,
            "status": "applied",
            "applied": True,
            "requested_model": "claude-sonnet-4-6",
            "routed_model": "claude-haiku-4-5-20251001",
            "reason": "small non-tool Sonnet request routed to Haiku",
            "policy_source": "local-manual",
            "surface": "codex_app_turn",
        }
        crunch_meta = {
            "enabled": True,
            "status": "applied",
            "changed": True,
            "saved_chars": 400,
            "tokens_before_est": 200,
            "tokens_saved_est": 100,
            "policy_source": "local-default",
            "surface": "codex_app_turn",
        }
        cache_meta = {
            "enabled": True,
            "status": "not-applied",
            "reason": "codex-app-cache-not-implemented",
            "policy_source": "local-default",
            "surface": "codex_app_turn",
        }
        server.store.log_codex_app_event(
            id=start_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="req-policy",
            thread_id="thread-policy",
            message_chars=300,
            params_chars=250,
            input_items=1,
            input_text_chars=400,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            session_id="codex-policy",
            routing_json=stable_json(routing_meta),
            crunch_json=stable_json(crunch_meta),
            cache_json=stable_json(cache_meta),
        )

        result = asyncio.run(stats_views.stats_activity(server.store))
        codex = {unit["unit_id"]: unit for unit in result["units"]}[f"codex_turn:{start_id}"]

        self.assertEqual(codex["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(codex["target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(codex["routed_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(codex["optimization_features"]["routing"]["status"], "applied")
        self.assertEqual(codex["optimization_features"]["crunch"]["saved_chars"], 400)
        self.assertEqual(codex["optimization_features"]["cache"]["status"], "not-applied")
        self.assertEqual(codex["optimization_features"]["policy_sources"], ["local-default", "local-manual"])
        self.assertEqual(codex["replayability_level"], "features_only")
        json.dumps(result)

    def test_quality_signal_summary_uses_metadata_only_provider_and_codex_rows(self):
        def log_provider(status_code, *, retry_count=0, error=None, routing=None):
            call_id = str(uuid.uuid4())
            server.store.log_call(
                id=call_id,
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model=(routing or {}).get("routed_model") or "claude-sonnet-4-6",
                stream=0,
                cache_hit=0,
                status_code=status_code,
                latency_ms=100,
                input_tokens_est=10,
                output_tokens_est=2,
                actual_input_tokens=10,
                actual_output_tokens=2,
                cost_est_usd=0.001,
                cost_baseline_usd=0.001,
                crunch_json=stable_json({"changed": False, "policy_source": "local-default"}),
                routing_json=stable_json(routing or {"policy_source": "local-default"}),
                cache_json=stable_json({"status": "miss", "policy_source": "local-default"}),
                error=error,
                request_json=None,
                response_json=None,
                session_id="quality-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                retry_count=retry_count,
                provider="anthropic",
            )
            return call_id

        log_provider(
            200,
            retry_count=1,
            routing={
                "applied": True,
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-haiku-4-5-20251001",
                "policy_source": "local-default",
            },
        )
        log_provider(400)
        log_provider(429, error="temporarily limiting requests for tier sonnet")

        abandoned_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=abandoned_id,
            created_at="2000-01-01T00:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="quality-abandoned",
            thread_id="quality-thread",
            message_chars=100,
            params_chars=100,
            input_items=1,
            input_text_chars=100,
            session_id="quality-codex",
            routing_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            crunch_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
        )
        pending_id = str(uuid.uuid4())
        server.store.log_codex_app_event(
            id=pending_id,
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="quality-pending",
            thread_id="quality-thread",
            message_chars=100,
            params_chars=100,
            input_items=1,
            input_text_chars=100,
            session_id="quality-codex",
            routing_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            crunch_json=stable_json({"status": "not-applied", "policy_source": "local-default"}),
            cache_json=stable_json({"status": "skipped", "policy_source": "local-default"}),
        )

        activity = asyncio.run(stats_views.stats_activity(server.store, limit=20))
        quality = asyncio.run(stats_views.stats_quality_signals(server.store, limit=20))

        summary = activity["summary"]["quality_signal_summary"]
        by_status = {row["status"]: row["count"] for row in summary["by_status"]}
        by_signal = {row["signal"]: row["count"] for row in summary["by_signal"]}
        self.assertEqual(quality["schema"], "agentflow.quality_signal_report.v1")
        self.assertFalse(quality["privacy"]["raw_prompts_included"])
        self.assertEqual(quality["summary"], summary)
        self.assertGreaterEqual(by_status["success"], 1)
        self.assertGreaterEqual(by_status["failure"], 1)
        self.assertGreaterEqual(by_status["local_throttled"], 1)
        self.assertGreaterEqual(by_status["abandoned"], 1)
        self.assertGreaterEqual(by_status["pending"], 1)
        self.assertGreaterEqual(by_signal["retry-after-error"], 1)
        self.assertGreaterEqual(by_signal["local-throttled"], 1)
        self.assertGreaterEqual(by_signal["abandoned"], 1)
        self.assertNotIn("request_json", json.dumps(quality))

    def test_usage_by_owner_groups_provider_calls_and_codex_turns(self):
        old_engineer = os.environ.get("AGENTFLOW_ENGINEER")
        old_app = os.environ.get("AGENTFLOW_APP")
        os.environ["AGENTFLOW_ENGINEER"] = "ada"
        os.environ["AGENTFLOW_APP"] = "code-workbench"
        try:
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/messages",
                requested_model="claude-sonnet-4-6",
                routed_model="claude-sonnet-4-6",
                stream=1,
                cache_hit=0,
                status_code=200,
                latency_ms=20,
                input_tokens_est=1_000,
                output_tokens_est=100,
                actual_input_tokens=1_000,
                actual_output_tokens=100,
                cost_est_usd=0.02,
                cost_baseline_usd=0.025,
                crunch_json=stable_json({"changed": False}),
                routing_json=stable_json({"text_chars": 10_000, "category": "tool-result"}),
                cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="shared-session",
                category="tool-result",
                cache_creation_input_tokens=1_000,
                cache_read_input_tokens=0,
                retry_count=0,
                thinking_output_tokens=200,
                provider="anthropic",
            )
            server.store.log_call(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                path="/v1/responses",
                requested_model="gpt-5-codex",
                routed_model="gpt-5-codex",
                stream=0,
                cache_hit=0,
                status_code=200,
                latency_ms=30,
                input_tokens_est=2_000,
                output_tokens_est=300,
                actual_input_tokens=2_000,
                actual_output_tokens=300,
                cost_est_usd=0.03,
                cost_baseline_usd=0.05,
                crunch_json=stable_json({"changed": True}),
                routing_json=stable_json({"text_chars": 8_200, "category": "chat"}),
                cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                error=None,
                request_json=None,
                response_json=None,
                session_id="shared-session",
                category="chat",
                cache_creation_input_tokens=0,
                cache_read_input_tokens=500,
                retry_count=0,
                thinking_output_tokens=0,
                provider="openai",
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="client_to_server",
                method="turn/start",
                request_id="req-usage",
                thread_id="thread-usage",
                message_chars=500,
                params_chars=50,
                input_items=2,
                input_text_chars=321,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id="shared-session",
            )
            server.store.log_codex_app_event(
                id=str(uuid.uuid4()),
                created_at=utc_now(),
                direction="server_to_client",
                method="turn/completed",
                request_id="req-usage",
                thread_id="thread-usage",
                message_chars=100,
                params_chars=None,
                input_items=None,
                input_text_chars=None,
                result_chars=123,
                error_code=None,
                error_message=None,
                latency_ms=1200,
                session_id="shared-session",
            )

            result = asyncio.run(stats_views.stats_usage_by_owner(server.store))
        finally:
            if old_engineer is None:
                os.environ.pop("AGENTFLOW_ENGINEER", None)
            else:
                os.environ["AGENTFLOW_ENGINEER"] = old_engineer
            if old_app is None:
                os.environ.pop("AGENTFLOW_APP", None)
            else:
                os.environ["AGENTFLOW_APP"] = old_app

        self.assertEqual(result["schema"], "agentflow.usage_by_owner.v1")
        self.assertEqual(result["summary"]["buckets"], 1)
        self.assertFalse(result["summary"]["codex_cost_unknown"])
        self.assertEqual(result["summary"]["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(result["summary"]["provider_reported_spend_usd"], 0.05, places=6)
        self.assertAlmostEqual(result["summary"]["codex_estimated_spend_usd"], 0.00056, places=6)
        self.assertAlmostEqual(result["summary"]["calculated_spend_usd"], 0.05056, places=6)
        [bucket] = result["buckets"]

        self.assertEqual(bucket["bucket_kind"], "engineer_app")
        self.assertEqual(bucket["bucket_label"], "ada / code-workbench")
        self.assertEqual(bucket["provider_calls"], 2)
        self.assertEqual(bucket["codex_turns"], 1)
        self.assertEqual(bucket["turns"], 3)
        self.assertEqual(bucket["provider_input_tokens"], 4_500)
        self.assertEqual(bucket["provider_output_tokens"], 400)
        self.assertEqual(bucket["provider_total_tokens"], 4_900)
        self.assertEqual(bucket["codex_input_text_chars"], 321)
        self.assertEqual(bucket["codex_result_chars"], 123)
        self.assertEqual(bucket["codex_input_tokens_est"], 80)
        self.assertEqual(bucket["codex_output_tokens_est"], 30)
        self.assertEqual(bucket["codex_total_tokens_est"], 110)
        self.assertEqual(bucket["input_tokens"], 4_580)
        self.assertEqual(bucket["output_tokens"], 430)
        self.assertEqual(bucket["total_tokens"], 5_010)
        self.assertEqual(bucket["token_basis"], "mixed")
        self.assertTrue(bucket["provider_cost_known"])
        self.assertTrue(bucket["codex_cost_known"])
        self.assertTrue(bucket["codex_cost_estimated"])
        self.assertFalse(bucket["excludes_unknown_codex_app_cost"])
        self.assertEqual(bucket["codex_mutation_safe_turns"], 0)
        self.assertEqual(bucket["codex_telemetry_only_turns"], 1)
        self.assertEqual(bucket["cost_basis"], "provider-reported + codex-estimated-from-chars")
        self.assertAlmostEqual(bucket["codex_cost_est_usd"], 0.00056, places=6)
        self.assertAlmostEqual(bucket["spend_usd"], 0.05056, places=6)
        self.assertAlmostEqual(bucket["baseline_cost_usd"], 0.07556, places=6)
        self.assertAlmostEqual(bucket["cache_savings_usd"], 0.000563, places=6)
        self.assertAlmostEqual(bucket["captured_savings_usd"], 0.025, places=6)
        self.assertAlmostEqual(bucket["hard_floor_usd"], 0.05056, places=6)
        self.assertEqual(
            {row["source_surface"]: row["units"] for row in bucket["source_surfaces"]},
            {"anthropic_messages": 1, "codex_turn": 1, "openai_responses": 1},
        )
        self.assertEqual(bucket["thinking_tokens"], 200)
        self.assertEqual(bucket["large_tool_result_calls"], 1)
        self.assertGreater(bucket["potential_hint_count"], 0)
        hint_codes = {hint["code"] for hint in bucket["remaining_saving_potential_hints"]}
        self.assertIn("thinking_output", hint_codes)
        self.assertIn("cache_warmup", hint_codes)
        self.assertIn("large_tool_result_context", hint_codes)
        self.assertFalse(result["grouping"]["raw_prompt_logging"])
        json.dumps(result)

    def test_dashboard_exposes_unified_recent_calls_table(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Recent calls</button>", html)
        self.assertIn("<h2>Recent calls</h2>", html)
        self.assertIn("id=\"activity-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/activity?limit=100')", html)
        self.assertIn('<th data-sort-type="text">Surface</th>', html)
        self.assertIn('<th data-sort-type="text">Granularity</th>', html)
        self.assertIn('<th data-sort-type="text">App family</th>', html)
        self.assertIn("not provider-replayable", html)
        self.assertIn("Codex estimated from chars", html)
        self.assertNotIn(">Activity</button>", html)
        self.assertNotIn(">Provider calls</button>", html)
        self.assertNotIn(">Codex debug</button>", html)
        self.assertNotIn("id=\"provider-tbody\"", html)
        self.assertNotIn("id=\"codex-tbody\"", html)
        self.assertIn("const tabs=['safety','activity','usage','codex','weekly','categories','cache','errors','limiter','policies','managed','sessions']", html)

    def test_dashboard_exposes_codex_quota_token_usage_panel(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Codex quota</button>", html)
        self.assertIn("<h2>Codex quota and token usage</h2>", html)
        self.assertIn("id=\"codex-quota-tbody\"", html)
        self.assertIn("id=\"codex-rate-scopes-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/codex-effectiveness?limit=500')", html)
        self.assertIn("quota_and_token_usage", html)
        self.assertIn("raw commands omitted", html)
        self.assertIn("raw transcripts omitted", html)

    def test_dashboard_policy_panel_renders_codex_app_surface_state(self):
        html = stats_views.dashboard_html()

        self.assertIn("Codex app-server", html)
        self.assertIn("Codex exact cache off", html)
        self.assertIn("safe keys", html)
        self.assertIn("action-like skip on", html)

    def test_dashboard_exposes_usage_by_app_engineer_table(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Usage by app / engineer</button>", html)
        self.assertIn("<h2>Usage by app / engineer</h2>", html)
        self.assertIn("id=\"usage-tbody\"", html)
        self.assertIn("fetch('/agentflow/stats/usage')", html)
        self.assertIn('<th data-sort-type="text">Bucket</th>', html)
        self.assertIn('<th data-sort-type="number">Turns</th>', html)
        self.assertIn('<th data-sort-type="number">Provider calls</th>', html)
        self.assertIn('<th data-sort-type="number">Codex turns</th>', html)
        self.assertIn("Remaining saving potential", html)
        self.assertIn("Codex estimated", html)
        self.assertNotIn("Codex cost unknown", html)

    def test_dashboard_exposes_executive_summary_cards(self):
        html = stats_views.dashboard_html()

        self.assertEqual(html.count("class=\"card\""), 2)
        self.assertEqual(html.count("class=\"card green\""), 1)
        self.assertEqual(html.count("class=\"card yellow\""), 1)
        self.assertEqual(html.count("class=\"card blue\""), 1)
        self.assertIn("Tokens today", html)
        self.assertIn("Calculated spend", html)
        self.assertIn("Hard floor", html)
        self.assertIn("Ops health", html)
        self.assertIn("executive_summary", html)
        self.assertIn("today_buckets", html)
        self.assertIn("Codex estimated", html)
        self.assertNotIn("Calls today", html)
        self.assertNotIn("Saved by routing", html)
        self.assertNotIn("Provider cache discount", html)
        self.assertNotIn("Old-context summaries", html)
        self.assertNotIn("Thinking cost today", html)
        self.assertIn("Codex app-server", html)

    def test_dashboard_exposes_error_breakdown_tables(self):
        html = stats_views.dashboard_html()

        self.assertIn(">Errors</button>", html)
        self.assertIn("<h2>Errors today</h2>", html)
        self.assertIn("<h2>Errors all time</h2>", html)
        self.assertIn("id=\"errors-today-tbody\"", html)
        self.assertIn("id=\"errors-tbody\"", html)
        self.assertIn("today_error_breakdown", html)
        self.assertIn("error_breakdown", html)
        self.assertIn("refreshErrors", html)
        self.assertIn('<th data-sort-type="text">Type</th>', html)
        self.assertIn('<th data-sort-type="number">Status</th>', html)
        self.assertIn('<th data-sort-type="text">Provider</th>', html)
        self.assertIn('<th data-sort-type="text">Tier</th>', html)

    def test_dashboard_tables_are_sortable_and_filterable(self):
        html = stats_views.dashboard_html()

        self.assertIn("function initDataTables", html)
        self.assertIn("function applyDataTableState", html)
        self.assertIn("function applyAllDataTables", html)
        self.assertIn("const tableState={}", html)
        self.assertIn("className='table-filter'", html)
        self.assertIn("setTableSort(table,index)", html)
        self.assertIn("data-sort-type=\"money\"", html)
        self.assertIn("data-sort-type=\"percent\"", html)
        self.assertIn("data-sort-type=\"latency\"", html)
        self.assertIn("data-sort-type=\"time\"", html)
        self.assertIn("<th data-sort-type=\"text\">Surface</th><th data-sort-type=\"text\">App</th><th data-sort-type=\"text\">Session</th>", html)
        self.assertIn("<th data-sort-type=\"number\">Codex turns</th>", html)
        self.assertIn("<th data-sort-type=\"number\">Codex input</th>", html)
        self.assertIn("row.codex_routed_turns", html)
        self.assertIn("No matching rows", html)
        self.assertIn("applyAllDataTables();", html)

        for table_id in (
            "activity",
            "usage",
            "cache-today",
            "cache-all",
            "errors-today",
            "errors-all",
            "codex-quota",
            "codex-rate-scopes",
            "sessions",
        ):
            self.assertIn(f'data-table-id="{table_id}"', html)

    def test_dashboard_coalesces_full_stats_loading(self):
        html = stats_views.dashboard_html()

        self.assertEqual(html.count("fetch('/agentflow/stats/full')"), 1)
        self.assertIn("const FULL_STATS_TTL_MS=5000", html)
        self.assertIn("let fullStatsInFlight=null", html)
        self.assertIn("if(fullStatsInFlight)return fullStatsInFlight", html)
        self.assertEqual(html.count("const d=await loadFullStats();"), 4)
        self.assertIn("async function refresh()", html)
        self.assertIn("async function refreshCategories()", html)
        self.assertIn("async function refreshCache()", html)
        self.assertIn("async function refreshErrors()", html)

    def test_proxy_dashboard_router_uses_current_store(self):
        response = TestClient(server.app).get("/agentflow/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["calls"], 0)

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

        result = asyncio.run(stats_views.stats_sessions(server.store))
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

        result = asyncio.run(stats_views.stats_limiter(server.store, server._tier_backoff_status, server._dashboard_limiter_config()))
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
