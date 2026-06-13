from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.anthropic_thinking_compaction_report import (
    build_anthropic_thinking_compaction_opportunity_report,
)
from agentflow_proxy.store import Store, stable_json


def _thinking_tool_result_body(secret: str) -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": (f"private chain of thought {secret} /workspace/private/plan.py\n" * 500),
                    },
                    {
                        "type": "tool_use",
                        "id": "raw-tool-use-id-must-not-leak",
                        "name": "Read",
                        "input": {"file_path": "/workspace/private/plan.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "raw-tool-use-id-must-not-leak",
                        "content": [{"type": "text", "text": f"tool payload secret {secret}"}],
                    }
                ],
            },
        ]
    }


def _active_unverified_thinking_body(secret: str) -> dict:
    return {
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": (f"active private chain {secret} /workspace/private/unsafe.py\n" * 500),
                    },
                    {"type": "text", "text": "assistant continuity"},
                ],
            },
            {"role": "user", "content": "raw prompt continuation must not leak"},
        ],
    }


def _log_call(
    store: Store,
    call_id: str,
    *,
    created_at: str,
    category: str = "tool-result",
    text_chars: int = 120_000,
    status_code: int = 200,
    request_json: dict | None = None,
    session_id: str = "raw-thinking-session-id-must-not-leak",
    provider: str = "anthropic",
    reason: str = "keep requested model for thinking request",
) -> None:
    routing = {
        "category": category,
        "workflow_phase": "tool-execution",
        "reason": reason,
        "text_chars": text_chars,
        "has_tools": category.startswith("tool"),
    }
    requested_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5.4"
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages" if provider == "anthropic" else "/v1/responses",
        requested_model=requested_model,
        routed_model=requested_model,
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1000,
        input_tokens_est=text_chars // 4,
        output_tokens_est=250,
        actual_input_tokens=200,
        actual_output_tokens=250,
        cost_est_usd=0.05,
        cost_baseline_usd=0.50,
        crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
        routing_json=stable_json(routing),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
        error=None,
        request_json=stable_json(request_json) if request_json is not None else None,
        response_json=stable_json({"text": "raw response must not leak"}),
        session_id=session_id,
        category=category,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=max(0, (text_chars // 4) - 300),
        retry_count=0,
        thinking_output_tokens=120,
        provider=provider,
        source_surface="anthropic_messages" if provider == "anthropic" else "openai_responses",
        endpoint="messages" if provider == "anthropic" else "responses",
        requested_model_family="sonnet" if provider == "anthropic" else "gpt",
        routed_model_family="sonnet" if provider == "anthropic" else "gpt",
    )


class AnthropicThinkingCompactionReportTests(unittest.TestCase):
    def test_report_measures_metadata_only_thinking_candidates_privately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(store, "thinking-1", created_at="2026-06-13T00:00:00+00:00")
                _log_call(store, "thinking-2", created_at="2026-06-13T00:01:00+00:00", text_chars=121_000)
                _log_call(
                    store,
                    "tool-heavy-thinking",
                    created_at="2026-06-13T00:02:00+00:00",
                    category="tool-heavy",
                    text_chars=80_000,
                    session_id="raw-tool-heavy-session-must-not-leak",
                )
                payload = build_anthropic_thinking_compaction_opportunity_report(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_opportunity.v1")
        self.assertEqual(payload["summary"]["metadata_candidate_count"], 2)
        self.assertEqual(payload["summary"]["body_verified_candidate_count"], 0)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["projected_saved_usd"], 0)
        self.assertGreaterEqual(payload["summary"]["plateau_pair_count"], 1)
        privacy_modes = {item["mode"]: item["count"] for item in payload["privacy_mode_breakdown"]}
        self.assertEqual(privacy_modes["metadata-only"], 3)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("request-body-unavailable", blockers)
        self.assertIn("non-tool-result-category", blockers)
        self.assertFalse(payload["privacy"]["raw_thinking_text_included"])
        self.assertFalse(payload["privacy"]["session_ids_included"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-thinking-session-id-must-not-leak", rendered)
        self.assertNotIn("raw-tool-heavy-session-must-not-leak", rendered)

    def test_report_uses_local_body_counts_without_emitting_thinking_text(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "body-thinking-1",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=_thinking_tool_result_body("raw-thinking-secret-one"),
                )
                _log_call(
                    store,
                    "body-thinking-2",
                    created_at="2026-06-13T00:01:00+00:00",
                    text_chars=120_500,
                    request_json=_thinking_tool_result_body("raw-thinking-secret-two"),
                )
                payload = build_anthropic_thinking_compaction_opportunity_report(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["metadata_candidate_count"], 2)
        self.assertEqual(payload["summary"]["body_verified_candidate_count"], 2)
        self.assertEqual(payload["summary"]["thinking_history_block_count"], 2)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        top = payload["candidates"][0]
        self.assertEqual(top["privacy_mode"], "local-body-derived-metadata")
        self.assertGreater(top["unique_thinking_fingerprint_count"], 0)
        self.assertFalse(top["privacy"]["thinking_block_fingerprints_included"])
        self.assertIn("ready-for-thinking-compaction-review", top["blockers"])
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw-thinking-secret-one",
            "raw-thinking-secret-two",
            "raw-tool-use-id-must-not-leak",
            "tool payload secret",
            "private chain of thought",
            "/workspace/private",
            "raw response must not leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_report_blocks_active_and_unverified_body_contexts_privately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "active-unverified-body",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=_active_unverified_thinking_body("raw-active-unverified-secret"),
                )
                payload = build_anthropic_thinking_compaction_opportunity_report(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["metadata_candidate_count"], 1)
        self.assertEqual(payload["summary"]["body_verified_candidate_count"], 0)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("active-top-level-thinking-request", blockers)
        self.assertIn("tool-result-thinking-continuation-unverified", blockers)
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw-active-unverified-secret",
            "raw prompt continuation",
            "active private chain",
            "/workspace/private/unsafe.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cli_emits_anthropic_thinking_opportunity_report(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "cli-thinking-1",
                    created_at="2026-06-13T00:00:00+00:00",
                    request_json=_thinking_tool_result_body("raw-cli-thinking-secret"),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.anthropic_thinking_compaction_opportunity_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_opportunity.v1")
        self.assertEqual(payload["summary"]["metadata_candidate_count"], 1)
        self.assertNotIn("raw-cli-thinking-secret", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
