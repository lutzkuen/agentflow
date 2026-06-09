import sys
import asyncio
import importlib.util
import os
import json
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_RUNTIME_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("fastapi", "httpx")
)

if HAS_RUNTIME_DEPS:
    import httpx
    from fastapi.testclient import TestClient

    import agentflow_proxy.dashboard_app as dashboard_app
    from agentflow_proxy.dashboard_app import create_dashboard_app
    from agentflow_proxy.store import Store


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class DashboardImportTests(unittest.TestCase):
    def test_dashboard_import_does_not_import_provider_server(self):
        old_dashboard = sys.modules.pop("agentflow_proxy.dashboard", None)
        old_dashboard_app = sys.modules.pop("agentflow_proxy.dashboard_app", None)
        old_server = sys.modules.pop("agentflow_proxy.server", None)
        old_provider_handlers = sys.modules.pop("agentflow_proxy.provider_handlers", None)
        try:
            import agentflow_proxy.dashboard  # noqa: F401

            self.assertNotIn("agentflow_proxy.server", sys.modules)
            self.assertNotIn("agentflow_proxy.provider_handlers", sys.modules)
        finally:
            sys.modules.pop("agentflow_proxy.dashboard", None)
            sys.modules.pop("agentflow_proxy.dashboard_app", None)
            if old_dashboard is not None:
                sys.modules["agentflow_proxy.dashboard"] = old_dashboard
            if old_dashboard_app is not None:
                sys.modules["agentflow_proxy.dashboard_app"] = old_dashboard_app
            if old_server is not None:
                sys.modules["agentflow_proxy.server"] = old_server
            if old_provider_handlers is not None:
                sys.modules["agentflow_proxy.provider_handlers"] = old_provider_handlers

    def test_dashboard_app_uses_injected_store_and_preserves_routes(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        store = Store(tmp.name)
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event("validate", ok=False, details={"source": "test", "error_count": 1})
            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
            )
            client = TestClient(app)

            health = client.get("/health")
            stats = client.get("/agentflow/stats")
            policies = client.get("/agentflow/stats/policies")
            policy_events = client.get("/agentflow/stats/policy-events")
            codex_effectiveness = client.get("/agentflow/stats/codex-effectiveness")
            rollout_readiness = client.get("/agentflow/stats/rollout-actions/readiness")
            phase_routing = client.get("/agentflow/stats/phase-routing")
            safety = client.get("/agentflow/stats/safety")
            admin_reload = client.post("/agentflow/admin/reload-policies")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["mode"], "dashboard-read-only")
            self.assertEqual(stats.status_code, 200)
            self.assertEqual(stats.json()["db"], tmp.name)
            self.assertEqual(stats.json()["calls"], 0)
            self.assertEqual(policies.status_code, 200)
            self.assertEqual(policy_events.status_code, 200)
            self.assertEqual(policy_events.json()["schema"], "agentflow.policy_events.v1")
            self.assertEqual(policy_events.json()["events"][0]["action"], "validate")
            self.assertEqual(codex_effectiveness.status_code, 200)
            self.assertEqual(codex_effectiveness.json()["schema"], "agentflow.codex_app_effectiveness.v1")
            self.assertFalse(codex_effectiveness.json()["privacy"]["raw_prompts_included"])
            self.assertEqual(rollout_readiness.status_code, 200)
            self.assertEqual(rollout_readiness.json()["schema"], "agentflow.rollout_actions_readiness.v1")
            self.assertFalse(rollout_readiness.json()["privacy"]["raw_action_payloads_included"])
            self.assertEqual(phase_routing.status_code, 200)
            self.assertEqual(phase_routing.json()["schema"], "agentflow.phase_routing_dashboard.v1")
            self.assertFalse(phase_routing.json()["privacy"]["raw_prompts_included"])
            self.assertEqual(safety.status_code, 200)
            self.assertEqual(safety.json()["schema"], "agentflow.safety_privacy.v1")
            self.assertFalse(safety.json()["privacy"]["raw_prompts_included"])
            policy_json = policies.json()
            self.assertEqual(policy_json["schema"], "agentflow.policy_state.v1")
            self.assertIn("summary", policy_json)
            self.assertIn("reload_required", policy_json["summary"])
            self.assertIn("reload_required_sections", policy_json["summary"])
            self.assertEqual(policy_json["summary"]["policy_count"], 5)
            self.assertIn("routing", policy_json)
            self.assertIn("crunch", policy_json)
            self.assertIn("cache", policy_json)
            self.assertIn("routing_experiments", policy_json)
            self.assertIn("codex_app", policy_json)
            self.assertIn("policy_source", policy_json["routing"])
            self.assertIn("rule_path", policy_json["routing"])
            self.assertIn("file", policy_json["routing"])
            self.assertIn("reload_required", policy_json["routing"]["file"])
            self.assertIn("policy_source", policy_json["crunch"])
            self.assertIn("rule_path", policy_json["crunch"])
            self.assertIn("file", policy_json["crunch"])
            self.assertIn("reload_required", policy_json["crunch"]["file"])
            self.assertIn("policy_source", policy_json["cache"])
            self.assertIn("rule_path", policy_json["cache"])
            self.assertIn("file", policy_json["cache"])
            self.assertIn("reload_required", policy_json["cache"]["file"])
            self.assertIn("policy_source", policy_json["routing_experiments"])
            self.assertIn("rule_path", policy_json["routing_experiments"])
            self.assertIn("file", policy_json["routing_experiments"])
            self.assertIn("reload_required", policy_json["routing_experiments"]["file"])
            self.assertEqual(policy_json["codex_app"]["surface"], "codex_turn")
            self.assertFalse(policy_json["codex_app"]["review_only"])
            self.assertIn("policy_source", policy_json["codex_app"])
            self.assertIn("rule_path", policy_json["codex_app"])
            self.assertIn("file", policy_json["codex_app"])
            self.assertIn("reload_required", policy_json["codex_app"]["file"])
            self.assertEqual(admin_reload.status_code, 404)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("AgentFlow", dashboard.text)
            self.assertIn("Policies", dashboard.text)
            self.assertIn("/agentflow/stats/policies", dashboard.text)
            self.assertIn("/agentflow/stats/policy-events", dashboard.text)
            self.assertIn("Policy reload summary", dashboard.text)
            self.assertIn("policy-summary-tbody", dashboard.text)
            self.assertIn("Codex rules", dashboard.text)
            self.assertIn("Recent policy events", dashboard.text)
            self.assertIn("Safety / privacy status", dashboard.text)
            self.assertIn("/agentflow/stats/safety", dashboard.text)
            self.assertIn("safety-warnings-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/rollout-actions/readiness", dashboard.text)
            self.assertIn("Rollout-action readiness", dashboard.text)
            self.assertIn("rollout-readiness-tbody", dashboard.text)
            self.assertIn("/agentflow/stats/phase-routing", dashboard.text)
            self.assertIn("Phase-routing rollout health", dashboard.text)
            self.assertIn("phase-routing-health-tbody", dashboard.text)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()

    def test_rollout_action_readiness_endpoint_summarizes_metadata_only(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        event_tmp = tempfile.TemporaryDirectory()
        old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(event_tmp.name) / "policy_events.jsonl")
        store = Store(tmp.name)
        try:
            from agentflow_proxy.policy_events import log_policy_event

            log_policy_event(
                "rollout-actions-review",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/raw-action-payload.json",
                    "config_dir": "/tmp/local-yaml-config",
                    "action_count": 2,
                    "planned_action_count": 2,
                    "changed_action_count": 1,
                    "provenance_status": "verified",
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "rollout-actions-dry-run",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/raw-action-payload.json",
                    "config_dir": "/tmp/local-yaml-config",
                    "db_path": tmp.name,
                    "dry_run": True,
                    "action_count": 2,
                    "affected_metadata_row_count": 9,
                    "exit_code": 0,
                },
            )
            log_policy_event(
                "pattern-canary-safety-stop",
                ok=True,
                details={
                    "reason": "local-canary-safety-stop",
                    "policy_section": "crunch",
                    "rule_id": "rule-id-not-rendered",
                    "pattern_hash": "sha256:" + ("b" * 64),
                    "sample_count": 6,
                    "raw_payload_included": False,
                },
            )
            log_policy_event(
                "rollout-actions-impact",
                ok=True,
                details={
                    "source": "cli",
                    "path": "/tmp/dry-run-report.json",
                    "db_path": tmp.name,
                    "action_count": 2,
                    "projected_affected_metadata_row_count": 9,
                    "actual_matched_metadata_row_count": 4,
                    "actual_matched_provider_call_count": 3,
                    "actual_matched_codex_turn_count": 1,
                    "actual_canary_applied_count": 2,
                    "actual_canary_holdout_count": 1,
                    "actual_bypassed_or_disabled_count": 1,
                    "actual_tokens_saved_est": 700,
                    "actual_estimated_cost_savings_usd": 0.007,
                    "actions_without_post_apply_matches": 0,
                    "exit_code": 0,
                },
            )
            store.enqueue_managed_outcome_feedback(
                id="rollout-feedback-queued",
                created_at="2026-06-09T03:40:00+00:00",
                updated_at="2026-06-09T03:40:00+00:00",
                source_surface="rollout_action_lifecycle",
                endpoint="/v1/policy-events",
                optimization_unit_id=0,
                payload_json=json.dumps({
                    "event_type": "dry-run",
                    "occurred_at": "2026-06-09T03:40:00+00:00",
                    "bundle_hash": "sha256:" + ("c" * 64),
                    "action_ids": ["must-not-render-action-id"],
                    "metadata": {
                        "schema": "agentflow.rollout_action_lifecycle_metadata.v1",
                        "command": "rollout-actions-dry-run",
                        "local_result_status": "ok",
                        "dry_run": True,
                        "read_only": True,
                        "action_count": 2,
                        "planned_action_count": 2,
                        "changed_action_count": 1,
                        "action_type_counts": {"widen": 1, "rollback": 1},
                        "policy_section_counts": {"crunch": 2},
                        "local_status_counts": {"planned": 2},
                        "affected_metadata_row_count": 9,
                        "affected_provider_call_count": 5,
                        "affected_codex_turn_count": 4,
                        "projected_additional_applied_count": 3,
                        "projected_local_bypass_or_disable_count": 2,
                        "historical_tokens_saved_est": 1200,
                        "historical_estimated_cost_savings_usd": 0.0123,
                        "safety_stop_reason_counts": {"local-canary-safety-stop": 1},
                        "candidate_ids": ["candidate-id-not-rendered"],
                        "rule_ids": ["rule-id-not-rendered"],
                        "pattern_hashes": ["sha256:" + ("b" * 64)],
                        "raw_prompt": "raw prompt must stay hidden",
                        "yaml_contents": "local YAML must stay hidden",
                        "privacy": {
                            "metadata_only": True,
                            "raw_prompts_included": False,
                            "raw_messages_included": False,
                            "raw_responses_included": False,
                            "raw_transcripts_included": False,
                            "raw_params_included": False,
                            "tool_payloads_included": False,
                            "request_ids_included": False,
                            "local_session_ids_included": False,
                            "file_paths_included": False,
                            "yaml_contents_included": False,
                        },
                    },
                    "raw_request": "raw request body must stay hidden",
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T09:00:00+00:00",
            )

            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={},
                full_stats_ttl_s=0,
            )
            client = TestClient(app)
            response = client.get("/agentflow/stats/rollout-actions/readiness")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.rollout_actions_readiness.v1")
            self.assertEqual(payload["summary"]["pending_lifecycle_feedback_count"], 1)
            self.assertEqual(payload["summary"]["due_lifecycle_feedback_count"], 1)
            self.assertEqual(payload["summary"]["affected_metadata_row_count"], 9)
            self.assertEqual(payload["dry_run_impact"]["projected_additional_applied_count"], 3)
            self.assertEqual(payload["dry_run_impact"]["projected_local_bypass_or_disable_count"], 2)
            self.assertEqual(payload["latest_impact"]["stage"], "impact")
            self.assertEqual(payload["post_apply_impact"]["actual_matched_metadata_row_count"], 4)
            self.assertEqual(payload["post_apply_impact"]["actual_canary_applied_count"], 2)
            self.assertEqual({row["value"]: row["count"] for row in payload["action_type_counts"]}, {"widen": 1, "rollback": 1})
            self.assertTrue(payload["safety_stop"]["active"])
            self.assertFalse(payload["privacy"]["raw_action_payloads_included"])
            self.assertFalse(payload["privacy"]["yaml_contents_included"])
            self.assertIn("rollout-readiness-tbody", dashboard.text)
            self.assertIn("rollout-action-counts-tbody", dashboard.text)

            rendered = json.dumps(payload) + dashboard.text
            self.assertNotIn("raw prompt must stay hidden", rendered)
            self.assertNotIn("raw request body must stay hidden", rendered)
            self.assertNotIn("local YAML must stay hidden", rendered)
            self.assertNotIn("must-not-render-action-id", rendered)
            self.assertNotIn("candidate-id-not-rendered", rendered)
            self.assertNotIn("rule-id-not-rendered", rendered)
            self.assertNotIn("/tmp/raw-action-payload.json", rendered)
            self.assertNotIn("/tmp/dry-run-report.json", rendered)
            self.assertNotIn("/tmp/local-yaml-config", rendered)
        finally:
            if old_event_log is None:
                os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
            else:
                os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_event_log
            store.conn.close()
            tmp.close()
            event_tmp.cleanup()

    def test_safety_stats_warn_and_redact_unsafe_configuration(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        env = {
            "AGENTFLOW_LOG_BODIES": "1",
            "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
            "AGENTFLOW_RECOMMENDATION_SERVER_URL": "https://user:supersecret@managed.test/v1/recommendation?api_key=managedsecret&mode=dev",
            "AGENTFLOW_POLICY_BUNDLE_RECOMMENDATION_URL": "https://managed.test/v1/policy-bundle-recommendation?token=policysecret",
            "AGENTFLOW_MANAGED_API_KEY": "",
            "AGENTFLOW_POLICY_EVENTS": "0",
        }
        try:
            with patch.dict(os.environ, env, clear=False):
                app = create_dashboard_app(
                    store_obj=lambda: store,
                    default_db=tmp.name,
                    upstream="https://anthropic.test",
                    limiter_status=lambda: [],
                    limiter_config={},
                    proxy_host="0.0.0.0",
                    dashboard_host="0.0.0.0",
                    full_stats_ttl_s=0,
                )
                client = TestClient(app)
                response = client.get("/agentflow/stats/safety")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            warning_codes = {row["code"] for row in payload["warnings"]}
            self.assertIn("proxy-bind-non-loopback", warning_codes)
            self.assertIn("body-logging-enabled", warning_codes)
            self.assertIn("managed-recommendation-unauthenticated", warning_codes)
            self.assertIn("managed-policy-fetch-unauthenticated", warning_codes)
            self.assertIn("policy-events-disabled", warning_codes)
            self.assertFalse(payload["privacy"]["raw_prompts_included"])
            self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
            self.assertFalse(payload["checks"]["managed"]["api_key_value_included"])
            redacted = str(payload)
            self.assertIn("[redacted]", redacted)
            self.assertNotIn("supersecret", redacted)
            self.assertNotIn("managedsecret", redacted)
            self.assertNotIn("policysecret", redacted)
            self.assertNotIn("user:supersecret", redacted)
        finally:
            store.conn.close()
            tmp.close()

    def test_safety_stats_warn_on_stuck_managed_feedback_queue_without_payloads(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        try:
            store.enqueue_managed_outcome_feedback(
                id="queue-due",
                created_at="2000-01-01T09:00:00+00:00",
                updated_at="2000-01-01T09:00:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/77/outcome",
                optimization_unit_id=77,
                payload_json=json.dumps({
                    "prompt": "must not appear in dashboard",
                    "raw_response": "provider body must stay local",
                    "params": {"secret": "raw params must stay local"},
                }),
                status="queued",
                attempts=0,
                next_attempt_at="2000-01-01T09:00:00+00:00",
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-retry",
                created_at="2000-01-01T09:05:00+00:00",
                updated_at="2000-01-01T09:05:00+00:00",
                source_surface="anthropic_messages",
                endpoint="/v1/optimization-units/88/outcome",
                optimization_unit_id=88,
                payload_json=json.dumps({"messages": ["raw prompt text"]}),
                status="retryable-error",
                attempts=2,
                next_attempt_at="2000-01-01T09:05:00+00:00",
                last_error="ConnectError: managed feedback down",
                last_status_code=503,
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-dropped",
                created_at="2000-01-01T09:10:00+00:00",
                updated_at="2000-01-01T09:10:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/99/outcome",
                optimization_unit_id=99,
                payload_json=json.dumps({"raw_request": "dropped raw request"}),
                status="dropped-after-limit",
                attempts=3,
                next_attempt_at="2000-01-01T09:10:00+00:00",
                last_error="HTTP 500: raw failure body",
                last_status_code=500,
            )
            store.enqueue_managed_outcome_feedback(
                id="queue-sent",
                created_at="2000-01-01T08:55:00+00:00",
                updated_at="2000-01-01T09:15:00+00:00",
                source_surface="codex_turn",
                endpoint="/v1/optimization-units/66/outcome",
                optimization_unit_id=66,
                payload_json=json.dumps({"content": "sent raw content"}),
                status="sent",
                attempts=1,
                next_attempt_at="2000-01-01T08:55:00+00:00",
                sent_at="2000-01-01T09:15:00+00:00",
                last_status_code=200,
            )

            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_RECOMMENDATION_ENABLED": "1",
                    "AGENTFLOW_RECOMMENDATION_SERVER_URL": "http://managed.test",
                    "AGENTFLOW_MANAGED_API_KEY": "test-key",
                },
                clear=False,
            ):
                app = create_dashboard_app(
                    store_obj=lambda: store,
                    default_db=tmp.name,
                    upstream="https://anthropic.test",
                    limiter_status=lambda: [],
                    limiter_config={},
                    full_stats_ttl_s=0,
                )
                client = TestClient(app)
                response = client.get("/agentflow/stats/safety")
                dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            queue = payload["checks"]["managed"]["feedback_queue"]
            warning_codes = {row["code"] for row in payload["warnings"]}
            self.assertIn("managed-feedback-due-queue", warning_codes)
            self.assertIn("managed-feedback-retryable-errors", warning_codes)
            self.assertIn("managed-feedback-dropped-after-limit", warning_codes)
            self.assertEqual(queue["summary"]["queued"], 1)
            self.assertEqual(queue["summary"]["due"], 2)
            self.assertEqual(queue["summary"]["retryable_error"], 1)
            self.assertEqual(queue["summary"]["dropped_after_limit"], 1)
            self.assertEqual(queue["last_successful_flush"]["optimization_unit_id"], 66)
            self.assertFalse(queue["due_samples"][0]["payload_included"])
            self.assertFalse(queue["privacy"]["payload_json_included"])
            self.assertFalse(payload["privacy"]["managed_feedback_payload_json_included"])
            self.assertIn("safety-managed-feedback-tbody", dashboard.text)
            self.assertIn("managed-feedback-queue-tbody", dashboard.text)
            rendered = json.dumps(payload) + dashboard.text
            self.assertNotIn("must not appear in dashboard", rendered)
            self.assertNotIn("provider body must stay local", rendered)
            self.assertNotIn("raw params must stay local", rendered)
            self.assertNotIn("raw prompt text", rendered)
            self.assertNotIn("dropped raw request", rendered)
            self.assertNotIn("sent raw content", rendered)
        finally:
            store.conn.close()
            tmp.close()

    def test_full_stats_endpoint_coalesces_concurrent_requests(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3")
        store = Store(tmp.name)
        old_stats_full = dashboard_app.stats_views.stats_full
        call_count = {"value": 0}

        async def fake_stats_full(store_obj):
            call_count["value"] += 1
            await asyncio.sleep(0.05)
            return {
                "summary": {"total_calls": call_count["value"]},
                "generated_by": call_count["value"],
            }

        dashboard_app.stats_views.stats_full = fake_stats_full
        try:
            app = create_dashboard_app(
                store_obj=lambda: store,
                default_db=tmp.name,
                upstream="https://anthropic.test",
                limiter_status=lambda: [],
                limiter_config={
                    "min_request_interval_ms": 0,
                    "max_tier_backoff_wait_s": 30,
                    "max_concurrent_per_tier": 2,
                },
                full_stats_ttl_s=60,
            )

            async def exercise():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    responses = await asyncio.gather(
                        client.get("/agentflow/stats/full"),
                        client.get("/agentflow/stats/full"),
                        client.get("/agentflow/stats/full"),
                    )
                    warm_start = time.perf_counter()
                    cached = await client.get("/agentflow/stats/full")
                    warm_seconds = time.perf_counter() - warm_start
                    return responses, cached, warm_seconds

            responses, cached, warm_seconds = asyncio.run(exercise())

            self.assertEqual([response.status_code for response in responses], [200, 200, 200])
            self.assertEqual(cached.status_code, 200)
            self.assertEqual(call_count["value"], 1)
            self.assertEqual([response.json()["generated_by"] for response in responses], [1, 1, 1])
            self.assertEqual(cached.json()["generated_by"], 1)
            self.assertLess(warm_seconds, 0.2)
        finally:
            dashboard_app.stats_views.stats_full = old_stats_full
            store.conn.close()
            tmp.close()


if __name__ == "__main__":
    unittest.main()
