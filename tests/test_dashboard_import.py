import sys
import asyncio
import importlib.util
import os
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
            self.assertEqual(safety.status_code, 200)
            self.assertEqual(safety.json()["schema"], "agentflow.safety_privacy.v1")
            self.assertFalse(safety.json()["privacy"]["raw_prompts_included"])
            policy_json = policies.json()
            self.assertEqual(policy_json["schema"], "agentflow.policy_state.v1")
            self.assertIn("summary", policy_json)
            self.assertIn("reload_required", policy_json["summary"])
            self.assertIn("reload_required_sections", policy_json["summary"])
            self.assertEqual(policy_json["summary"]["policy_count"], 4)
            self.assertIn("routing", policy_json)
            self.assertIn("crunch", policy_json)
            self.assertIn("cache", policy_json)
            self.assertIn("routing_experiments", policy_json)
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
            self.assertEqual(admin_reload.status_code, 404)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("AgentFlow", dashboard.text)
            self.assertIn("Policies", dashboard.text)
            self.assertIn("/agentflow/stats/policies", dashboard.text)
            self.assertIn("/agentflow/stats/policy-events", dashboard.text)
            self.assertIn("Policy reload summary", dashboard.text)
            self.assertIn("policy-summary-tbody", dashboard.text)
            self.assertIn("Recent policy events", dashboard.text)
            self.assertIn("Safety / privacy status", dashboard.text)
            self.assertIn("/agentflow/stats/safety", dashboard.text)
            self.assertIn("safety-warnings-tbody", dashboard.text)
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
