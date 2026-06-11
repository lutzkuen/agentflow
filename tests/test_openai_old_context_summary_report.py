from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from agentflow_proxy import cli
from agentflow_proxy.openai_old_context_summary_report import build_openai_old_context_summary_report
from agentflow_proxy.stats import stats_openai_old_context_summary_report
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class OpenAIOldContextSummaryReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _feature(
        self,
        *,
        endpoint: str = "responses",
        source_surface: str = "openai_responses",
        has_tools: bool = False,
        category: str = "chat",
        workflow_phase: str = "tool-execution",
        older_bucket: str = "32k_128k_chars",
    ) -> dict[str, object]:
        return {
            "schema": "agentflow.openai_feature_summary.v1",
            "provider": "openai",
            "source_surface": source_surface,
            "endpoint": endpoint,
            "requested_model_family": "gpt-5",
            "routed_model_family": "gpt-5",
            "stream": False,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_bucket": "32k_128k_chars",
            "input_token_bucket": "16k_64k_tokens",
            "has_tools": has_tools,
            "declared_tool_count": 1 if has_tools else 0,
            "chat_tool_call_count": 1 if has_tools and endpoint == "chat_completions" else 0,
            "chat_tool_result_count": 1 if has_tools and endpoint == "chat_completions" else 0,
            "old_context": {
                "shape": "responses_input_items" if endpoint == "responses" else "chat_messages",
                "conversation_item_count": 12,
                "older_context_item_count": 8,
                "older_context_text_bucket": older_bucket,
                "older_context_token_bucket": "16k_64k_tokens",
                "raw_payload_included": False,
            },
            "raw_payload_included": False,
        }

    def _log_openai_call(
        self,
        *,
        path: str = "/v1/responses",
        endpoint: str = "responses",
        source_surface: str = "openai_responses",
        feature: dict[str, object] | None = None,
        has_tools: bool = False,
        category: str = "chat",
        text_chars: int = 64_000,
        stream: int = 0,
        cache_status: str = "miss",
        session_id: str = "secret-openai-session",
        request_json: str | None = '{"input":"secret raw openai prompt","request_id":"req_secret"}',
        omit_feature: bool = False,
    ) -> None:
        actual_input_tokens = max(1, text_chars // 4)
        actual_output_tokens = 80
        if not omit_feature and feature is None:
            feature = self._feature(
                endpoint=endpoint,
                source_surface=source_surface,
                has_tools=has_tools,
                category=category,
            )
        routing: dict[str, object] = {
            "enabled": False,
            "provider": "openai",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "reason": "openai routing disabled",
            "text_chars": text_chars,
            "has_tools": has_tools,
            "category": category,
            "workflow_phase": "tool-execution",
            "context_plateau_status": "plateau-adjacent",
            "policy_source": "local-default",
        }
        if feature is not None:
            routing["openai_feature_unit"] = feature
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            stream=stream,
            cache_hit=1 if cache_status == "hit" else 0,
            status_code=200,
            latency_ms=125,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=0.01,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": cache_status, "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=request_json,
            response_json=None,
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="openai",
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )

    def test_report_measures_openai_responses_and_chat_without_raw_fields(self) -> None:
        self._log_openai_call()
        self._log_openai_call(
            path="/v1/chat/completions",
            endpoint="chat_completions",
            source_surface="openai_chat",
            category="summary",
            feature=self._feature(endpoint="chat_completions", source_surface="openai_chat", category="summary"),
        )
        self._log_openai_call(omit_feature=True)
        self._log_openai_call(has_tools=True, feature=self._feature(has_tools=True), category="tool-light")

        result = build_openai_old_context_summary_report(
            self.store,
            limit=20,
            summary_provider_configured=True,
            summary_model="gpt-5-mini",
        )

        self.assertEqual(result["schema"], "agentflow.openai_old_context_summary_opportunity.v1")
        self.assertEqual(result["summary"]["openai_call_count"], 4)
        self.assertEqual(result["summary"]["feature_row_count"], 3)
        self.assertEqual(result["summary"]["eligible_count"], 2)
        self.assertEqual(result["summary"]["blocked_count"], 2)
        self.assertGreater(result["summary"]["projected_summarized_chars"], 0)
        self.assertGreater(result["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(result["summary"]["estimated_summary_cost_usd"], 0)
        self.assertGreater(result["summary"]["projected_gross_savings_usd"], 0)
        self.assertIn("projected_net_savings_usd", result["summary"])

        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertEqual(blockers["blocked_missing_body_or_feature"], 1)
        self.assertEqual(blockers["tool_protocol_risk"], 1)
        endpoints = {row["value"]: row["count"] for row in result["endpoint_breakdown"]}
        self.assertEqual(endpoints["responses"], 3)
        self.assertEqual(endpoints["chat_completions"], 1)

        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "secret-openai-session",
            "secret raw openai prompt",
            "req_secret",
            "request_json",
            "response_json",
            "raw chat message",
            "raw function args",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(result["privacy"]["tool_payloads_included"])
        self.assertFalse(result["privacy"]["function_arguments_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

    def test_provider_configuration_blocks_rows_until_local_summary_provider_exists(self) -> None:
        self._log_openai_call()

        result = build_openai_old_context_summary_report(
            self.store,
            limit=10,
            summary_provider_configured=False,
        )

        self.assertEqual(result["summary"]["eligible_count"], 0)
        self.assertEqual(result["summary"]["blocked_count"], 1)
        self.assertEqual(
            result["blocker_reason_breakdown"],
            [{"value": "summary_provider_not_configured", "count": 1}],
        )
        self.assertFalse(result["measurement_policy"]["summary_provider_configured"])

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        self._log_openai_call()

        with mock.patch.dict(os.environ, {"AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED": "1"}):
            result = asyncio.run(stats_openai_old_context_summary_report(self.store, limit=10))
            self.assertEqual(result["schema"], "agentflow.openai_old_context_summary_opportunity.v1")

            output = io.StringIO()
            exit_code = cli.openai_old_context_summary_report_cli(
                ["--db", self.db_path, "--limit", "10"],
                stdout=output,
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_old_context_summary_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 1)
        self.assertNotIn("secret-openai-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
