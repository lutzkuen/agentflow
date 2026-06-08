import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from agentflow_proxy import cli


class PolicyReloadCliTests(unittest.TestCase):
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

    def test_policy_validate_cli_accepts_exported_bundle_from_stdin(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO(exported.getvalue()), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertEqual(payload["errors"], [])

    def test_policy_validate_cli_rejects_invalid_json_with_structured_errors(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(["-"], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_validation.v1")
        self.assertIn("invalid JSON", payload["errors"][0]["message"])

    def test_policy_validate_cli_rejects_malformed_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_validate_cli(
            ["-"],
            stdin=io.StringIO(json.dumps({"schema": "wrong"})),
            stdout=stdout,
        )

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("$.schema", {error["path"] for error in payload["errors"]})

    def test_policy_diff_cli_reports_file_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["changes"][0]["path"], "$.policies.routing.enabled")

    def test_policy_diff_cli_accepts_one_stdin_input(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO(json.dumps(before)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["changed"])

    def test_policy_diff_cli_rejects_invalid_json_with_structured_errors(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)

        with TemporaryDirectory() as tmp:
            after_path = Path(tmp) / "after.json"
            after_path.write_text(exported.getvalue(), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.policy_diff_cli(["-", str(after_path)], stdin=io.StringIO("{"), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_diff.v1")
        self.assertIn("invalid JSON", payload["before_validation"]["errors"][0]["message"])

    def test_policy_review_cli_reports_current_to_proposed_changes(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["routing"]["enabled"] = not proposed["policies"]["routing"]["enabled"]
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["changed"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertEqual(payload["changed_sections"], ["routing"])
        self.assertEqual(payload["change_count"], 1)
        self.assertEqual(payload["safety_warning_count"], 0)

    def test_policy_review_cli_rejects_invalid_bundle(self):
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps({"schema": "wrong"})), stdout=stdout)

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["schema"], "agentflow.policy_bundle_review.v1")
        self.assertIn("$.schema", {error["path"] for error in payload["proposed_validation"]["errors"]})
        self.assertFalse(payload["changed"])

    def test_policy_review_cli_surfaces_risky_policy_warnings(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        proposed["policies"]["cache"]["semantic_cache"]["enabled"] = True
        proposed["policies"]["crunch"]["old_context_summarization"]["enabled"] = True
        proposed["policies"]["routing"]["policy_source"] = "managed-enforced"
        stdout = io.StringIO()

        code = cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        warning_codes = {warning["code"] for warning in payload["safety_warnings"]}
        self.assertIn("tool-call-cache-enabled", warning_codes)
        self.assertIn("semantic-cache-enabled", warning_codes)
        self.assertIn("old-context-summarization-enabled", warning_codes)
        self.assertIn("managed-enforced-policy-source", warning_codes)
        self.assertEqual(payload["safety_warning_count"], 4)

    def test_policy_review_cli_records_compact_event(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        proposed = json.loads(exported.getvalue())
        proposed["policies"]["cache"]["exact_cache"]["cache_tool_calls"] = True
        stdout = io.StringIO()

        cli.policy_review_cli(["-"], stdin=io.StringIO(json.dumps(proposed)), stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "review")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["cache"])
        self.assertEqual(events[0]["details"]["safety_warning_count"], 1)

    def test_policy_cli_records_compact_local_events(self):
        exported = io.StringIO()
        cli.policy_export_cli([], stdout=exported)
        before = json.loads(exported.getvalue())
        after = json.loads(exported.getvalue())
        after["policies"]["routing"]["enabled"] = not before["policies"]["routing"]["enabled"]

        with TemporaryDirectory() as tmp:
            before_path = Path(tmp) / "before.json"
            after_path = Path(tmp) / "after.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            stdout = io.StringIO()

            cli.policy_diff_cli([str(before_path), str(after_path)], stdout=stdout)

        from agentflow_proxy.policy_events import recent_policy_events

        events = recent_policy_events(limit=5)["events"]
        self.assertEqual(events[0]["action"], "diff")
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["details"]["changed_sections"], ["routing"])
        self.assertEqual(events[0]["details"]["change_count"], 1)
        self.assertEqual(events[1]["action"], "export")
        self.assertIn("routing", events[1]["details"]["policies"])


if __name__ == "__main__":
    unittest.main()
