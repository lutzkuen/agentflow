import asyncio
import importlib.util
import json
import os
import tempfile
import time
import unittest
import uuid

HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("dotenv", "fastapi", "httpx", "websockets")
)

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy import server, stats as stats_views
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

        self.assertEqual(set(by_surface), {"anthropic_messages", "openai_responses", "codex_app_turn"})
        self.assertEqual(by_surface["anthropic_messages"]["input_tokens"], 120)
        self.assertEqual(by_surface["openai_responses"]["input_tokens"], 200)
        self.assertEqual(by_surface["codex_app_turn"]["input_tokens"], 10)
        self.assertEqual(by_surface["anthropic_messages"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["openai_responses"]["token_basis"], "provider-reported")
        self.assertEqual(by_surface["codex_app_turn"]["token_basis"], "estimated-from-chars")
        self.assertEqual(accounting["input_tokens"], 330)
        self.assertEqual(accounting["output_tokens"], 35)
        self.assertEqual(accounting["total_tokens"], 365)
        self.assertEqual(result["executive_summary"]["tokens_today"]["total_tokens"], 365)
        self.assertGreater(savings[("anthropic_messages", "crunching")], 0)
        self.assertGreater(savings[("anthropic_messages", "cache")], 0)
        self.assertGreater(savings[("openai_responses", "routing")], 0)
        self.assertNotIn(("codex_app_turn", "routing"), savings)
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
        self.assertEqual(summary["error_rows"], 1)
        self.assertEqual(summary["optimized_rows"], 3)
        self.assertEqual(summary["pass_through_rows"], 2)
        self.assertGreater(summary["optimized_error_rate"], 0)

        model_fields = {row["value"]: row["count"] for row in result["model_field_breakdown"]}
        self.assertEqual(model_fields["present"], 1)
        self.assertEqual(model_fields["absent"], 3)
        shapes = {row["value"]: row["count"] for row in result["param_shape_breakdown"]}
        self.assertEqual(shapes["action-like-params"], 1)
        self.assertEqual(shapes["unknown-param-shape"], 1)
        routing_statuses = {row["status"] for row in result["routing_breakdown"]}
        self.assertIn("applied", routing_statuses)
        self.assertIn("not-applicable", routing_statuses)
        self.assertNotIn(secret, json.dumps(result))

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
        self.assertEqual(cache_rows[("codex_app_turn", "hit", "exact-match", "exact")], 1)

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
        self.assertEqual(codex["source_surface"], "codex_app_turn")
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
        self.assertEqual(result["summary"]["by_source_surface"]["codex_app_turn"], 1)
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
            {"anthropic_messages": 1, "codex_app_turn": 1, "openai_responses": 1},
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
        self.assertIn("const tabs=['activity','usage','weekly','categories','cache','errors','limiter','policies','sessions']", html)

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
        self.assertNotIn("Codex app-server", html)

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
        self.assertIn("No matching rows", html)
        self.assertIn("applyAllDataTables();", html)

        for table_id in (
            "activity",
            "usage",
            "cache-today",
            "cache-all",
            "errors-today",
            "errors-all",
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
