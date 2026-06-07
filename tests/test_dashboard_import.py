import sys
import importlib.util
import tempfile
import unittest


HAS_RUNTIME_DEPS = importlib.util.find_spec("fastapi") is not None

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

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
        store = Store(tmp.name)
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
            )
            client = TestClient(app)

            health = client.get("/health")
            stats = client.get("/agentflow/stats")
            dashboard = client.get("/agentflow/dashboard")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["mode"], "dashboard-read-only")
            self.assertEqual(stats.status_code, 200)
            self.assertEqual(stats.json()["db"], tmp.name)
            self.assertEqual(stats.json()["calls"], 0)
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("AgentFlow", dashboard.text)
        finally:
            store.conn.close()
            tmp.close()


if __name__ == "__main__":
    unittest.main()
