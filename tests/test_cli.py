import io
import json
import os
import unittest
from unittest.mock import patch

import httpx

from agentflow_proxy import cli


class PolicyReloadCliTests(unittest.TestCase):
    def test_default_policy_reload_url_uses_agentflow_port(self):
        with patch.dict(os.environ, {"AGENTFLOW_PORT": "4001"}, clear=False):
            self.assertEqual(
                cli._default_policy_reload_url(),
                "http://127.0.0.1:4001/agentflow/admin/reload-policies",
            )

    def test_loopback_url_validation(self):
        self.assertTrue(cli._is_loopback_url("http://127.0.0.1:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://localhost:4000/agentflow/admin/reload-policies"))
        self.assertTrue(cli._is_loopback_url("http://[::1]:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("http://192.168.1.20:4000/agentflow/admin/reload-policies"))
        self.assertFalse(cli._is_loopback_url("file:///tmp/reload"))

    def test_policy_reload_cli_prints_reload_json_on_success(self):
        payload = {
            "ok": True,
            "schema": "agentflow.policy_reload.v1",
            "policies": {"schema": "agentflow.policy_state.v1"},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            post.return_value = httpx.Response(200, json=payload)
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4001/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), payload)
        post.assert_called_once_with("http://127.0.0.1:4001/agentflow/admin/reload-policies", timeout=10.0)

    def test_policy_reload_cli_rejects_non_loopback_url(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            code = cli.policy_reload_cli(["--url", "http://192.168.1.20:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(json.loads(stderr.getvalue())["error"]["type"], "unsafe_url")
        post.assert_not_called()

    def test_policy_reload_cli_reports_non_success_response(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("agentflow_proxy.cli.httpx.post") as post:
            post.return_value = httpx.Response(403, json={"ok": False, "error": {"type": "forbidden"}})
            code = cli.policy_reload_cli(["--url", "http://127.0.0.1:4000/agentflow/admin/reload-policies"], stdout=stdout, stderr=stderr)

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status_code"], 403)

    def test_policy_export_cli_prints_policy_bundle_json(self):
        stdout = io.StringIO()

        code = cli.policy_export_cli([], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.policy_bundle.v1")
        self.assertEqual(payload["generator"]["mode"], "local-offline")
        self.assertFalse(payload["managed_optimizer"]["enabled"])
        self.assertEqual(payload["policies"]["schema"], "agentflow.policy_state.v1")


if __name__ == "__main__":
    unittest.main()
