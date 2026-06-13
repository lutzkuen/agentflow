from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.openai_routing_report import build_openai_routing_report
from agentflow_proxy.stats import stats_openai_routing_report
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class OpenAIRoutingReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_openai_call(
        self,
        *,
        requested_model: str = "gpt-5.4-mini",
        routed_model: str | None = None,
        requested_model_family: str | None = None,
        category: str = "chat",
        text_chars: int = 1200,
        stream: int = 0,
        has_tools: bool = False,
        status_code: int = 200,
        retry_count: int = 0,
        session_id: str = "secret-openai-session",
        request_json: str | None = None,
    ) -> None:
        routed_model = routed_model or requested_model
        actual_input_tokens = max(1, text_chars // 4)
        actual_output_tokens = 40
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=0,
            status_code=status_code,
            latency_ms=125,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(
                {
                    "enabled": False,
                    "provider": "openai",
                    "requested_model": requested_model,
                    "routed_model": routed_model,
                    "reason": "openai routing disabled",
                    "text_chars": text_chars,
                    "has_tools": has_tools,
                    "category": category,
                    "policy_source": "local-default",
                }
            ),
            cache_json=stable_json(
                {
                    "status": "skipped" if has_tools else "miss",
                    "reason": "tools-disabled" if has_tools else "exact-miss",
                    "policy_source": "local-default",
                }
            ),
            error=None,
            request_json=request_json,
            response_json=None,
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family=requested_model_family or ("other" if requested_model.startswith("unknown") else "gpt-5"),
            routed_model_family="gpt-5",
        )

    def test_report_surfaces_disabled_openai_routing_candidates_without_raw_fields(self) -> None:
        for _ in range(6):
            self._log_openai_call(category="chat", text_chars=1200, request_json='{"input":"secret raw prompt"}')
        for _ in range(5):
            self._log_openai_call(category="short-completion", text_chars=700)
        self._log_openai_call(category="tool-light", text_chars=1100, has_tools=True)
        self._log_openai_call(category="chat", text_chars=900, stream=1)
        self._log_openai_call(requested_model="unknown-openai-model", category="chat", text_chars=900)

        result = build_openai_routing_report(self.store, limit=50)

        self.assertEqual(result["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(result["summary"]["openai_call_count"], 14)
        self.assertGreaterEqual(result["summary"]["candidate_count"], 2)
        self.assertEqual(result["summary"]["current_routed_count"], 0)
        self.assertGreater(result["summary"]["matched_count"], 0)
        self.assertGreater(result["summary"]["projected_savings_usd"], 0)
        self.assertEqual(result["summary"]["suggested_canary_fraction"], 0.05)

        chat_candidate = next(
            row for row in result["candidates"] if row["category"] == "chat" and row["blocked_count"] == 0
        )
        short_candidate = next(row for row in result["candidates"] if row["category"] == "short-completion")
        self.assertEqual(chat_candidate["matched_count"], 6)
        self.assertEqual(chat_candidate["target_model"], "gpt-5-mini")
        self.assertEqual(short_candidate["target_model"], "gpt-5-nano")

        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertIn("tools-disabled", blockers)
        self.assertIn("unknown-model-family", blockers)
        self.assertIn("stream-only-evidence", blockers)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-openai-session", rendered)
        self.assertNotIn("secret raw prompt", rendered)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_messages_included"])
        self.assertFalse(result["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(result["privacy"]["tool_payloads_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["secrets_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

    def test_report_preserves_gpt54_large_to_gpt54_mini_pass_through_candidate(self) -> None:
        for _ in range(6):
            self._log_openai_call(
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                category="chat",
                text_chars=1200,
            )

        result = build_openai_routing_report(self.store, limit=20)

        candidate = result["candidates"][0]
        self.assertEqual(candidate["requested_model"], "gpt-5.4")
        self.assertEqual(candidate["target_model"], "gpt-5.4-mini")
        self.assertEqual(candidate["current_routed_count"], 0)
        self.assertEqual(candidate["blocked_count"], 0)
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        for _ in range(5):
            self._log_openai_call(category="chat", text_chars=1200)

        result = asyncio.run(stats_openai_routing_report(self.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.openai_routing_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.openai_routing_report_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.openai_routing_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 5)
        self.assertNotIn("secret-openai-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
