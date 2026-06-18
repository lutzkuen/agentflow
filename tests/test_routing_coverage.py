from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.routing_coverage import build_routing_coverage_report
from agentflow_proxy.stats import stats_routing_coverage_report
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class RoutingCoverageReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        provider: str,
        source_surface: str,
        endpoint: str,
        requested_model: str,
        routed_model: str | None = None,
        category: str = "chat",
        routing_json: dict[str, object] | None = None,
        session_id: str = "secret-session-id",
    ) -> None:
        routed_model = routed_model or requested_model
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=f"/v1/{endpoint}",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=1000,
            output_tokens_est=100,
            actual_input_tokens=1000,
            actual_output_tokens=100,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json(routing_json or {}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error=None,
            request_json=None,
            response_json=None,
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family=requested_model.split("-")[0],
            routed_model_family=routed_model.split("-")[0],
        )

    def _row(self, report: dict[str, object], surface: str) -> dict[str, object]:
        rows = report["rows"]
        self.assertIsInstance(rows, list)
        for row in rows:
            self.assertIsInstance(row, dict)
            if row.get("surface") == surface:
                return row
        self.fail(f"missing routing coverage row for {surface}")

    def test_report_marks_openai_api_as_active_only_routing_surface(self) -> None:
        self._log_call(
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            category="tool-light",
            routing_json={
                "provider": "openai",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "openai_canary": {"status": "applied"},
            },
        )
        self._log_call(
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            category="tool-light",
            routing_json={
                "provider": "openai",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
                "openai_canary": {"status": "holdout"},
            },
        )
        self._log_call(
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="/v1/messages",
            requested_model="claude-sonnet-4-6",
            category="tool-result",
            routing_json={"phase_canary": {"status": "safety-stopped", "reason": "safety-stop-observed"}},
        )

        report = build_routing_coverage_report(self.store, limit=20)

        self.assertEqual(report["schema"], "agentflow.routing_coverage_report.v1")
        self.assertEqual(report["summary"]["routing_currently_active_only_for"], ["openai_api"])
        openai = self._row(report, "openai_api")
        self.assertTrue(openai["traffic_seen"])
        self.assertTrue(openai["routing_supported"])
        self.assertTrue(openai["routing_active"])
        self.assertTrue(openai["local_mutation_possible"])
        self.assertTrue(openai["holdout_available"])
        self.assertTrue(openai["outcome_feedback_available"])
        self.assertEqual(openai["top_blocker_reason"], "none-routing-active")
        self.assertEqual(openai["next_action"], "measure-openai-routing-rule-outcomes")

    def test_report_marks_anthropic_surface_eligible_but_held(self) -> None:
        self._log_call(
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="/v1/messages",
            requested_model="claude-sonnet-4-6",
            category="tool-result",
            routing_json={"phase_canary": {"status": "safety-stopped", "reason": "safety-stop-observed"}},
        )

        report = build_routing_coverage_report(self.store, limit=20)
        anthropic = self._row(report, "anthropic_api")

        self.assertTrue(anthropic["traffic_seen"])
        self.assertTrue(anthropic["routing_supported"])
        self.assertFalse(anthropic["routing_active"])
        self.assertTrue(anthropic["local_mutation_possible"])
        self.assertTrue(anthropic["outcome_feedback_available"])
        self.assertEqual(anthropic["top_blocker_reason"], "anthropic-routing-safety-stop-active")
        self.assertEqual(anthropic["next_action"], "collect-anthropic-applied-holdout-coverage-before-routing")

    def test_report_marks_codex_app_server_as_telemetry_only(self) -> None:
        self.store.log_codex_app_event(
            id="codex-event-1",
            created_at=utc_now(),
            direction="client_to_server",
            method="turn/start",
            request_id="secret-codex-request",
            thread_id="secret-thread",
            message_chars=1000,
            params_chars=200,
            input_items=1,
            input_text_chars=800,
            result_chars=0,
            error_code=None,
            error_message=None,
            latency_ms=20,
            session_id="secret-codex-session",
            routing_json=stable_json({"status": "not-applied", "reason": "codex-app-telemetry-only"}),
            crunch_json=None,
            cache_json=None,
            event_window_json=None,
            metadata_json=None,
        )

        report = build_routing_coverage_report(self.store, limit=20)
        codex = self._row(report, "codex_app_server_telemetry")

        self.assertTrue(codex["traffic_seen"])
        self.assertEqual(codex["sample_count"], 1)
        self.assertTrue(codex["telemetry_only"])
        self.assertFalse(codex["routing_supported"])
        self.assertFalse(codex["routing_active"])
        self.assertFalse(codex["local_mutation_possible"])
        self.assertEqual(codex["top_blocker_reason"], "telemetry-only-no-provider-request-mutation")
        self.assertIn("codex_app_server_telemetry", report["summary"]["telemetry_only_surfaces_with_traffic"])

    def test_stats_wrapper_and_cli_emit_metadata_only_report(self) -> None:
        self._log_call(
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            routing_json={"openai_canary": {"status": "applied"}},
        )

        result = asyncio.run(stats_routing_coverage_report(self.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.routing_coverage_report.v1")
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["provider_bodies_included"])
        self.assertEqual(result["summary"]["next_expansion_surface"]["surface"], "codex_vscode_or_cli")

        output = io.StringIO()
        exit_code = cli.routing_coverage_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.routing_coverage_report.v1")
        self.assertNotIn("secret-session-id", output.getvalue())


if __name__ == "__main__":
    unittest.main()
