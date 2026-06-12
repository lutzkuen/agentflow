import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.store import Store, stable_json
from agentflow_proxy.terminal_compaction_report import build_terminal_output_compaction_opportunity_report


def _tool_result_body(secret: str) -> dict:
    terminal_text = "\n".join(
        [
            "$ pytest tests/test_secret_output.py",
            "============================= FAILURES =============================",
            f"FAILED tests/test_secret_output.py::test_hidden - AssertionError: {secret}",
            "Traceback (most recent call last):",
            '  File "/workspace/private/tests/test_secret_output.py", line 42, in test_hidden',
            f"2026-06-12T10:00:00Z ERROR pid=1234 build failed with {secret}",
        ]
        * 80
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "raw-tool-use-id-must-not-leak",
                        "content": [{"type": "text", "text": terminal_text}],
                    }
                ],
            }
        ]
    }


def _log_call(
    store: Store,
    call_id: str,
    *,
    created_at: str,
    provider: str = "anthropic",
    path: str = "/v1/messages",
    category: str = "tool-result",
    text_chars: int = 48_000,
    session_id: str = "raw-session-id-must-not-leak",
    request_json: dict | None = None,
    terminal_bucket: str = "gte_75pct",
) -> None:
    routing = {
        "category": category,
        "workflow_phase": "tool-execution",
        "text_chars": text_chars,
        "has_tools": category.startswith("tool"),
        "terminal_log_features": {
            "schema": "agentflow.terminal_log_features.v1",
            "terminal_output_char_fraction_bucket": terminal_bucket,
            "privacy": {"metadata_only": True, "raw_terminal_text_included": False},
        },
    }
    requested_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5.4-mini"
    store.log_call(
        id=call_id,
        created_at=created_at,
        path=path,
        requested_model=requested_model,
        routed_model=requested_model,
        stream=1,
        cache_hit=0,
        status_code=200,
        latency_ms=100,
        input_tokens_est=text_chars // 4,
        output_tokens_est=100,
        actual_input_tokens=text_chars // 4,
        actual_output_tokens=100,
        cost_est_usd=0.05,
        cost_baseline_usd=0.05,
        crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
        routing_json=stable_json(routing),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
        error=None,
        request_json=stable_json(request_json) if request_json is not None else None,
        response_json=stable_json({"text": "raw response must not leak"}),
        session_id=session_id,
        category=category,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        retry_count=0,
        thinking_output_tokens=0,
        provider=provider,
        source_surface="anthropic_messages" if provider == "anthropic" else "openai_responses",
        endpoint="messages" if provider == "anthropic" else "responses",
        requested_model_family="sonnet" if provider == "anthropic" else "gpt",
        routed_model_family="sonnet" if provider == "anthropic" else "gpt",
    )


class TerminalOutputCompactionReportTests(unittest.TestCase):
    def test_report_ranks_plateaued_anthropic_tool_result_cohorts_privately(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "anthropic-1",
                    created_at="2026-06-12T10:00:00+00:00",
                    request_json=_tool_result_body("raw-terminal-secret-one"),
                )
                _log_call(
                    store,
                    "anthropic-2",
                    created_at="2026-06-12T10:01:00+00:00",
                    text_chars=48_400,
                    request_json=_tool_result_body("raw-terminal-secret-two"),
                )
                _log_call(
                    store,
                    "openai-tool-light",
                    provider="openai",
                    path="/v1/responses",
                    category="tool-light",
                    created_at="2026-06-12T10:02:00+00:00",
                    text_chars=12_000,
                    session_id="raw-openai-session-must-not-leak",
                    request_json=None,
                )

                payload = build_terminal_output_compaction_opportunity_report(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_opportunity.v1")
        self.assertGreaterEqual(payload["summary"]["candidate_count"], 2)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(payload["summary"]["plateau_pair_count"], 0)
        provider_counts = {item["value"]: item["count"] for item in payload["provider_breakdown"]}
        category_counts = {item["value"]: item["count"] for item in payload["category_breakdown"]}
        self.assertEqual(provider_counts["anthropic"], 2)
        self.assertEqual(provider_counts["openai"], 1)
        self.assertEqual(category_counts["tool-result"], 2)
        self.assertEqual(category_counts["tool-light"], 1)
        top = payload["candidates"][0]
        self.assertEqual(top["provider"], "anthropic")
        self.assertEqual(top["category"], "tool-result")
        self.assertGreater(top["projected_saved_tokens"], 0)
        self.assertIn("ready-for-dry-run-review", top["blockers"])
        openai = [candidate for candidate in payload["candidates"] if candidate["provider"] == "openai"][0]
        self.assertIn("non-anthropic-provider", openai["blockers"])
        self.assertIn("non-tool-result-category", openai["blockers"])
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["session_ids_included"])
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw-terminal-secret-one",
            "raw-terminal-secret-two",
            "raw-tool-use-id-must-not-leak",
            "raw-session-id-must-not-leak",
            "raw-openai-session-must-not-leak",
            "tests/test_secret_output.py",
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cli_emits_terminal_output_compaction_report(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "cli-anthropic-1",
                    created_at="2026-06-12T10:00:00+00:00",
                    request_json=_tool_result_body("raw-cli-terminal-secret-one"),
                )
                _log_call(
                    store,
                    "cli-anthropic-2",
                    created_at="2026-06-12T10:01:00+00:00",
                    request_json=_tool_result_body("raw-cli-terminal-secret-two"),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.terminal_output_compaction_opportunity_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_opportunity.v1")
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-cli-terminal-secret", rendered)
