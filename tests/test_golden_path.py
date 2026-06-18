from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.golden_path import build_golden_path_summary
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


SECRET_PROMPT = "secret golden path prompt body"
SECRET_RESPONSE = "secret golden path response body"
SECRET_SESSION = "secret-golden-path-session"
SECRET_REQUEST_ID = "req-secret-golden-path"


class TestGoldenPathSummary(unittest.TestCase):
    def test_fixture_summary_proves_local_savings_without_provider_or_managed_server(self) -> None:
        result = build_golden_path_summary()

        self.assertEqual(result["schema"], "agentflow.golden_path_summary.v1")
        self.assertTrue(result["ok"])
        self.assertEqual(result["surface"], "openai_responses")
        self.assertEqual(result["local_action_family"], "crunch")
        self.assertIn(result["decision_status"], {"demo_applied", "active"})
        self.assertGreater(result["estimated_agentflow_savings_usd"], 0)
        self.assertFalse(result["managed_server_required"])
        self.assertFalse(result["provider_calls_made"])

        fixture = result["fixture"]
        self.assertTrue(fixture["mocked_provider_response"])
        self.assertTrue(fixture["outcome_evidence_written"])
        self.assertEqual(fixture["outcome_evidence_store"], "ephemeral-fixture")
        self.assertTrue(fixture["local_outcome"]["crunch_changed"])
        self.assertGreater(fixture["local_outcome"]["tokens_saved_est"], 0)

    def test_summary_is_metadata_only(self) -> None:
        result = build_golden_path_summary()
        rendered = json.dumps(result, sort_keys=True)

        self.assertNotIn(SECRET_PROMPT, rendered)
        self.assertNotIn(SECRET_RESPONSE, rendered)
        self.assertNotIn(SECRET_SESSION, rendered)
        self.assertNotIn(SECRET_REQUEST_ID, rendered)
        self.assertNotIn("AgentFlow fixture repeated context", rendered)

        privacy = result["privacy"]
        self.assertTrue(privacy["metadata_only"])
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["raw_request_bodies_included"])
        self.assertFalse(privacy["raw_response_bodies_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["managed_server_calls_made"])

    def test_live_evidence_reports_active_openai_routing_without_raw_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "agentflow.sqlite3")
            store = SQLiteStore(db_path)
            try:
                store.log_call(
                    id=str(uuid.uuid4()),
                    created_at=utc_now(),
                    path="/v1/responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4-mini",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=25,
                    input_tokens_est=1000,
                    output_tokens_est=30,
                    actual_input_tokens=800,
                    actual_output_tokens=30,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.004,
                    crunch_json=stable_json({"changed": True, "tokens_saved_est": 200}),
                    routing_json=stable_json(
                        {
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "requested_model": "gpt-5.4",
                            "routed_model": "gpt-5.4-mini",
                            "reason": "fixture live routing evidence",
                        }
                    ),
                    cache_json=stable_json({"status": "skipped", "reason": "fixture"}),
                    error=None,
                    request_json=json.dumps({"input": SECRET_PROMPT, "request_id": SECRET_REQUEST_ID}),
                    response_json=json.dumps({"output_text": SECRET_RESPONSE}),
                    session_id=SECRET_SESSION,
                    category="tool-light",
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    retry_count=0,
                    thinking_output_tokens=0,
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    requested_model_family="gpt-5",
                    routed_model_family="gpt-5",
                )

                result = build_golden_path_summary(store=store, limit=50)
            finally:
                store.conn.close()

        self.assertEqual(result["decision_status"], "active")
        self.assertEqual(result["live_evidence"]["status"], "active")
        self.assertEqual(result["live_evidence"]["routing_applied_count"], 1)
        self.assertEqual(result["live_evidence"]["crunch_changed_count"], 1)
        self.assertFalse(result["live_evidence"]["managed_server_required"])
        self.assertEqual(result["routing_coverage"]["status"], "openai_api_only")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(SECRET_PROMPT, rendered)
        self.assertNotIn(SECRET_RESPONSE, rendered)
        self.assertNotIn(SECRET_SESSION, rendered)
        self.assertNotIn(SECRET_REQUEST_ID, rendered)


class TestGoldenPathCLI(unittest.TestCase):
    def test_agentflow_demo_golden_path_json(self) -> None:
        stdout = io.StringIO()
        code = cli.agentflow_cli(["demo", "golden-path", "--json"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.golden_path_summary.v1")
        self.assertEqual(result["surface"], "openai_responses")
        self.assertIn("local_action_family", result)
        self.assertIn("decision_status", result)
        self.assertIn("estimated_agentflow_savings_usd", result)
        self.assertFalse(result["managed_server_required"])

    def test_agentflow_demo_golden_path_human_summary(self) -> None:
        stdout = io.StringIO()
        code = cli.agentflow_cli(["demo", "golden-path"], stdout=stdout)

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("AgentFlow golden path:", output)
        self.assertIn("managed_server_required=false", output)


if __name__ == "__main__":
    unittest.main()
