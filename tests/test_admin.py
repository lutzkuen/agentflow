import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


HAS_RUNTIME_DEPS = importlib.util.find_spec("fastapi") is not None

if HAS_RUNTIME_DEPS:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tokenclaw.admin import create_admin_router


@unittest.skipUnless(HAS_RUNTIME_DEPS, "runtime web dependencies are not installed")
class AdminRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.old_event_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
        os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(Path(self.tmp.name) / "policy_events.jsonl")

    def tearDown(self):
        if self.old_event_log is None:
            os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
        else:
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = self.old_event_log
        self.tmp.cleanup()

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
        self.assertIn("tokenclaw.router", payload["reloaded_modules"])
        self.assertEqual(payload["policies"]["schema"], "agentflow.policy_state.v1")
        self.assertIn("routing", payload["policies"])
        self.assertIn("crunch", payload["policies"])
        self.assertIn("cache", payload["policies"])

        from tokenclaw.policy_events import recent_policy_events

        events = recent_policy_events(limit=1)["events"]
        self.assertEqual(events[0]["action"], "reload")
        self.assertEqual(events[0]["details"]["source"], "admin_api")

    def test_policy_draft_stage_route_is_loopback_only(self):
        app = FastAPI()
        app.include_router(create_admin_router())

        remote = TestClient(app, client=("192.168.1.50", 50000))
        blocked = remote.post(
            "/agentflow/admin/policy-drafts/stage",
            json={"section": "cache", "policy": {"semantic_cache": {"threshold": 0.91}}},
        )

        self.assertEqual(blocked.status_code, 403)
        payload = blocked.json()
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["wrote_active_policy_files"])
        self.assertFalse(payload["provider_calls_made"])

    def test_policy_workbench_mutating_routes_are_loopback_only(self):
        app = FastAPI()
        app.include_router(create_admin_router())

        remote = TestClient(app, client=("192.168.1.50", 50000))
        cases = [
            ("/agentflow/admin/reload-policies", {}),
            ("/agentflow/admin/policy-drafts/stage", {"section": "cache", "policy": {"semantic_cache": {"threshold": 0.91}}}),
            ("/agentflow/admin/policy-drafts/validate", {"draft": "draft-one"}),
            ("/agentflow/admin/policy-drafts/apply", {"draft": "draft-one"}),
            ("/agentflow/admin/policy-drafts/rollback", {"apply_id": "apply-one"}),
        ]

        for path, payload in cases:
            response = remote.post(path, json=payload)
            with self.subTest(path=path):
                self.assertEqual(response.status_code, 403)
                data = response.json()
                self.assertFalse(data["ok"])
                self.assertEqual(data["error"]["type"], "forbidden")
                if path != "/agentflow/admin/reload-policies":
                    self.assertFalse(data["wrote_active_policy_files"])
                    self.assertFalse(data["provider_calls_made"])
                    self.assertFalse(data["managed_server_calls_made"])

    def test_policy_draft_stage_route_returns_structured_diff_for_loopback(self):
        app = FastAPI()
        app.include_router(create_admin_router())

        local = TestClient(app, client=("127.0.0.1", 50000))
        workspace = str(Path(self.tmp.name) / "drafts")
        response = local.post(
            "/agentflow/admin/policy-drafts/stage",
            json={
                "section": "cache",
                "policy": {"semantic_cache": {"enabled": True, "threshold": 0.91}},
                "draft_id": "admin-cache-draft",
                "workspace": workspace,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_draft_stage.v1")
        self.assertEqual(payload["draft_id"], "admin-cache-draft")
        self.assertFalse(payload["wrote_active_policy_files"])
        self.assertFalse(payload["reloaded_modules"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertEqual(payload["diff"]["changed_sections"], ["cache"])
        cache_section = {section["section"]: section for section in payload["sections"]}["cache"]
        self.assertTrue(cache_section["changed"])
        self.assertTrue(cache_section["reload_required_after_apply"])
        self.assertTrue((Path(workspace) / "admin-cache-draft" / "draft.json").exists())

        from tokenclaw.policy_events import recent_policy_events

        event_text = json.dumps(recent_policy_events(limit=1)["events"])
        self.assertIn("draft-stage", event_text)
        self.assertNotIn("semantic_cache", event_text)

    def test_policy_draft_stage_route_rejects_raw_payloads(self):
        app = FastAPI()
        app.include_router(create_admin_router())

        local = TestClient(app, client=("127.0.0.1", 50000))
        response = local.post(
            "/agentflow/admin/policy-drafts/stage",
            json={
                "section": "cache",
                "policy": {"raw_request": {"prompt": "do not stage"}},
                "draft_id": "unsafe-admin-draft",
                "workspace": str(Path(self.tmp.name) / "drafts"),
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "raw_payload_rejected")
        self.assertFalse((Path(self.tmp.name) / "drafts" / "unsafe-admin-draft").exists())


if __name__ == "__main__":
    unittest.main()
