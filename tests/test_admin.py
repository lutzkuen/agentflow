import importlib.util
import unittest


HAS_RUNTIME_DEPS = importlib.util.find_spec("fastapi") is not None

if HAS_RUNTIME_DEPS:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentflow_proxy.admin import create_admin_router


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class AdminRouterTests(unittest.TestCase):
    def test_reload_route_is_loopback_only(self):
        app = FastAPI()
        app.include_router(create_admin_router())

        remote = TestClient(app, client=("192.168.1.50", 50000))
        blocked = remote.post("/agentflow/admin/reload-policies")

        self.assertEqual(blocked.status_code, 403)
        self.assertFalse(blocked.json()["ok"])

    def test_reload_route_returns_policy_state_for_loopback(self):
        callbacks = []

        def after_reload():
            callbacks.append("called")

        app = FastAPI()
        app.include_router(create_admin_router(after_reload=after_reload))

        local = TestClient(app, client=("127.0.0.1", 50000))
        response = local.post("/agentflow/admin/reload-policies")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_reload.v1")
        self.assertEqual(callbacks, ["called"])
        self.assertIn("agentflow_proxy.router", payload["reloaded_modules"])
        self.assertEqual(payload["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertIn("routing", payload["policies"])
        self.assertIn("crunch", payload["policies"])
        self.assertIn("cache", payload["policies"])


if __name__ == "__main__":
    unittest.main()
