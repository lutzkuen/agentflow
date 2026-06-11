from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from agentflow_proxy import cli
from agentflow_proxy.openai_old_context_summary_dry_run import build_openai_old_context_summary_dry_run
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class OpenAIOldContextSummaryDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _feature(self, *, endpoint: str, source_surface: str, category: str = "chat", has_tools: bool = False) -> dict[str, object]:
        return {
            "schema": "agentflow.openai_feature_summary.v1",
            "provider": "openai",
            "source_surface": source_surface,
            "endpoint": endpoint,
            "requested_model_family": "gpt-5",
            "routed_model_family": "gpt-5",
            "stream": False,
            "category": category,
            "workflow_phase": "tool-execution",
            "text_bucket": "32k_128k_chars",
            "input_token_bucket": "16k_64k_tokens",
            "has_tools": has_tools,
            "declared_tool_count": 1 if has_tools else 0,
            "chat_tool_call_count": 0,
            "chat_tool_result_count": 0,
            "response_tool_item_types": ["function_call"] if has_tools and endpoint == "responses" else [],
            "old_context": {
                "shape": "responses_input_items" if endpoint == "responses" else "chat_messages",
                "conversation_item_count": 10,
                "older_context_item_count": 6,
                "older_context_text_bucket": "32k_128k_chars",
                "older_context_token_bucket": "16k_64k_tokens",
                "raw_payload_included": False,
            },
            "raw_payload_included": False,
        }

    def _long_text(self, label: str, repeats: int = 900) -> str:
        return (f"{label} old context secret-openai-prompt should not leak. " * repeats).strip()

    def _responses_body(self) -> dict[str, object]:
        return {
            "model": "gpt-5.4",
            "instructions": "developer instruction with secret-response-request-id",
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "kept_response_format"}},
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": self._long_text("responses old 1")}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": self._long_text("responses old 2")}]},
                {"role": "user", "content": [{"type": "input_text", "text": self._long_text("responses old 3")}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": self._long_text("responses old 4")}]},
                {"role": "user", "content": [{"type": "input_text", "text": "recent item stays raw locally only"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "recent item stays raw locally only"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "recent item stays raw locally only"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "recent item stays raw locally only"}]},
            ],
        }

    def _chat_body(self) -> dict[str, object]:
        return {
            "model": "gpt-5.4",
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "system instruction with secret chat policy"},
                {"role": "developer", "content": "developer instruction with request id req_secret_chat"},
                {"role": "user", "content": self._long_text("chat old 1")},
                {"role": "assistant", "content": self._long_text("chat old 2")},
                {"role": "user", "content": self._long_text("chat old 3")},
                {"role": "assistant", "content": self._long_text("chat old 4")},
                {"role": "user", "content": "recent user message"},
                {"role": "assistant", "content": "recent assistant message"},
                {"role": "user", "content": "recent user message"},
                {"role": "assistant", "content": "recent assistant message"},
            ],
        }

    def _log_openai_call(
        self,
        *,
        body: dict[str, object] | None,
        path: str = "/v1/responses",
        endpoint: str = "responses",
        source_surface: str = "openai_responses",
        category: str = "chat",
        has_tools: bool = False,
        text_chars: int = 64_000,
    ) -> None:
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
            "policy_source": "local-default",
            "openai_feature_unit": self._feature(
                endpoint=endpoint,
                source_surface=source_surface,
                category=category,
                has_tools=has_tools,
            ),
        }
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            stream=1 if body and body.get("stream") else 0,
            cache_hit=0,
            status_code=200,
            latency_ms=125,
            input_tokens_est=max(1, text_chars // 4),
            output_tokens_est=80,
            actual_input_tokens=max(1, text_chars // 4),
            actual_output_tokens=80,
            cost_est_usd=0.01,
            cost_baseline_usd=0.02,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "miss", "policy_source": "local-default"}),
            error=None,
            request_json=stable_json(body) if body is not None else None,
            response_json=None,
            session_id="secret-openai-session",
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

    def test_dry_run_plans_responses_and_chat_without_raw_fields(self) -> None:
        self._log_openai_call(body=self._responses_body())
        self._log_openai_call(
            body=self._chat_body(),
            path="/v1/chat/completions",
            endpoint="chat_completions",
            source_surface="openai_chat",
        )

        result = build_openai_old_context_summary_dry_run(
            self.store,
            limit=10,
            summary_provider_configured=True,
            summary_model="gpt-5-mini",
        )

        self.assertEqual(result["schema"], "agentflow.openai_old_context_summary_dry_run.v1")
        self.assertEqual(result["summary"]["eligible_count"], 2)
        self.assertEqual(result["summary"]["blocked_count"], 0)
        self.assertGreater(result["summary"]["expected_tokens_saved"], 0)
        self.assertEqual({plan["endpoint"] for plan in result["plans"]}, {"responses", "chat_completions"})
        for plan in result["plans"]:
            self.assertTrue(plan["candidate_id"].startswith("openai-old-context-summary-"))
            self.assertTrue(plan["canary_eligible"])
            self.assertEqual(plan["reason_codes"], ["eligible"])
            self.assertGreater(plan["expected_chars_saved"], 0)
            self.assertGreater(plan["expected_tokens_saved"], 0)
            self.assertEqual(plan["summary_request_shape"]["provider"], "openai")
            self.assertFalse(plan["summary_request_shape"]["raw_source_included"])
            self.assertTrue(plan["preservation_checks"]["system_developer_instructions_preserved"])
            self.assertTrue(plan["preservation_checks"]["tool_function_protocol_preserved"])
            self.assertTrue(plan["preservation_checks"]["streaming_compatible"])

        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "secret-openai-session",
            "secret-openai-prompt",
            "secret-response-request-id",
            "req_secret_chat",
            "recent item stays raw",
            "request_json",
            "tool arguments",
            "/tmp/secret-file.txt",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(result["privacy"]["metadata_only_output"])
        self.assertTrue(result["privacy"]["raw_bodies_read_locally"])
        self.assertFalse(result["privacy"]["raw_request_bodies_included"])
        self.assertFalse(result["privacy"]["tool_payloads_included"])
        self.assertFalse(result["privacy"]["function_arguments_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

    def test_dry_run_fails_closed_for_required_blockers(self) -> None:
        tool_body = self._responses_body()
        tool_body["tools"] = [{"type": "function", "name": "secret_tool", "parameters": {"secret": "tool arguments"}}]
        tool_body["input"] = [
            {"type": "function_call", "call_id": "call_secret", "arguments": "{\"secret\":\"tool arguments\"}"},
            *tool_body["input"],
        ]
        unsupported_body = {"model": "gpt-5.4", "input": "single raw prompt secret unsupported"}
        file_body = self._responses_body()
        file_body["input"] = [
            {"role": "user", "content": [{"type": "input_file", "file_id": "file_secret", "filename": "/tmp/secret-file.txt"}]},
            *file_body["input"],
        ]
        self._log_openai_call(body=tool_body, has_tools=True)
        self._log_openai_call(body=unsupported_body)
        self._log_openai_call(body=None)
        self._log_openai_call(body=file_body)

        with mock.patch.dict(os.environ, {"AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_MAX_SUMMARY_COST_USD": "0.00000001"}):
            result = build_openai_old_context_summary_dry_run(
                self.store,
                limit=10,
                summary_provider_configured=False,
            )

        self.assertEqual(result["summary"]["eligible_count"], 0)
        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertGreaterEqual(blockers["summary_provider_not_configured"], 4)
        self.assertGreaterEqual(blockers["tool_function_protocol_ambiguous"], 1)
        self.assertGreaterEqual(blockers["unsupported_request_shape"], 1)
        self.assertGreaterEqual(blockers["raw_body_unavailable"], 1)
        self.assertGreaterEqual(blockers["summary_cost_over_budget"], 1)
        self.assertGreaterEqual(blockers["file_reference_in_source_window"], 1)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("tool arguments", rendered)
        self.assertNotIn("/tmp/secret-file.txt", rendered)
        self.assertNotIn("single raw prompt secret", rendered)

    def test_cli_emits_dry_run(self) -> None:
        self._log_openai_call(body=self._responses_body())

        with mock.patch.dict(os.environ, {"AGENTFLOW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED": "1"}):
            output = io.StringIO()
            exit_code = cli.openai_old_context_summary_dry_run_cli(
                ["--db", self.db_path, "--limit", "10"],
                stdout=output,
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_old_context_summary_dry_run.v1")
        self.assertEqual(payload["summary"]["eligible_count"], 1)
        self.assertNotIn("secret-openai-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
