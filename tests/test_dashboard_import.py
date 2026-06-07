import sys
import importlib.util
import unittest


HAS_RUNTIME_DEPS = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class DashboardImportTests(unittest.TestCase):
    def test_dashboard_import_does_not_import_provider_server(self):
        old_dashboard = sys.modules.pop("agentflow_proxy.dashboard", None)
        old_server = sys.modules.pop("agentflow_proxy.server", None)
        try:
            import agentflow_proxy.dashboard  # noqa: F401

            self.assertNotIn("agentflow_proxy.server", sys.modules)
        finally:
            sys.modules.pop("agentflow_proxy.dashboard", None)
            if old_dashboard is not None:
                sys.modules["agentflow_proxy.dashboard"] = old_dashboard
            if old_server is not None:
                sys.modules["agentflow_proxy.server"] = old_server


if __name__ == "__main__":
    unittest.main()
