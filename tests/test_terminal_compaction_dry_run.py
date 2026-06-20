import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.store import Store, stable_json
from tokenclaw.terminal_compaction_dry_run import (
    apply_terminal_output_compaction_plan,
    build_terminal_output_compaction_dry_run,
    plan_terminal_output_compaction,
)


def _terminal_text(secret: str) -> str:
    important = [
        "$ pytest tests/test_secret_terminal.py",
        "============================= FAILURES =============================",
        f"FAILED tests/test_secret_terminal.py::test_hidden - AssertionError: {secret}",
        "Traceback (most recent call last):",
        '  File "/workspace/private/tests/test_secret_terminal.py", line 42, in test_hidden',
        "AssertionError: expected ok",
        "exit code: 1",
        "modified tokenclaw/terminal_compaction_dry_run.py",
    ]
    noisy = [
        f"2026-06-12T10:00:{second:02d}Z INFO pid=1234 compiling shard={second} secret={secret}"
        for second in range(60)
    ]
    return "\n".join((important + noisy) * 8)


def _plateau_tool_result_body(secret: str, *, recent_secret: str = "RECENT_TERMINAL_OUTPUT_MUST_STAY") -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_preserve_me", "name": "Bash", "input": {"command": "pytest"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_preserve_me",
                        "content": [{"type": "text", "text": _terminal_text(secret)}],
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Recent assistant turn must stay."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_recent",
                        "content": [{"type": "text", "text": recent_secret}],
                    }
                ],
            },
        ],
    }


def _log_call(
    store: Store,
    call_id: str,
    *,
    created_at: str,
    request_json: dict | None,
    provider: str = "anthropic",
    category: str = "tool-result",
    text_chars: int = 48_000,
    session_id: str = "raw-session-id-must-not-leak",
    status_code: int = 200,
) -> None:
    routing = {
        "category": category,
        "workflow_phase": "tool-execution",
        "text_chars": text_chars,
        "has_tools": category.startswith("tool"),
    }
    requested_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-5.4-mini"
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages" if provider == "anthropic" else "/v1/responses",
        requested_model=requested_model,
        routed_model=requested_model,
        stream=1,
        cache_hit=0,
        status_code=status_code,
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
        error=None if status_code < 400 else "provider error bucket",
        request_json=stable_json(request_json) if request_json is not None else None,
        response_json=None,
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


class TerminalOutputCompactionDryRunTests(unittest.TestCase):
    def test_planner_compacts_old_terminal_tool_result_and_preserves_protocol_evidence_and_recent_turns(self):
        body = _plateau_tool_result_body("RAW_TERMINAL_SECRET_A")

        plan, meta = plan_terminal_output_compaction(body, keep_recent_turns=2, min_block_chars=500)
        self.assertIsNotNone(plan)
        self.assertEqual(meta["status"], "planned")
        self.assertGreater(plan["saved_chars"], 0)
        self.assertTrue(plan["preservation_flags"]["tool_protocol_ids_preserved"])
        self.assertTrue(plan["preservation_flags"]["recent_turns_preserved"])
        self.assertTrue(plan["preservation_flags"]["command_summaries_preserved"])
        self.assertTrue(plan["preservation_flags"]["failure_lines_preserved"])
        self.assertTrue(plan["preservation_flags"]["stack_traces_preserved"])
        self.assertTrue(plan["preservation_flags"]["exit_status_preserved"])
        self.assertTrue(plan["preservation_flags"]["file_change_hints_preserved"])

        planned_body = apply_terminal_output_compaction_plan(body, plan)
        rendered = stable_json(planned_body)
        planned_text = planned_body["messages"][1]["content"][0]["content"][0]["text"]
        self.assertIn("toolu_preserve_me", rendered)
        self.assertIn("toolu_recent", rendered)
        self.assertIn("RECENT_TERMINAL_OUTPUT_MUST_STAY", rendered)
        self.assertIn("$ pytest tests/test_secret_terminal.py", planned_text)
        self.assertIn("FAILED tests/test_secret_terminal.py::test_hidden", planned_text)
        self.assertIn("Traceback (most recent call last):", planned_text)
        self.assertIn('File "/workspace/private/tests/test_secret_terminal.py", line 42', planned_text)
        self.assertIn("exit code: 1", planned_text)
        self.assertIn("modified tokenclaw/terminal_compaction_dry_run.py", planned_text)
        self.assertLess(len(rendered), len(stable_json(body)))

    def test_dry_run_report_is_metadata_only_and_ranks_plateaued_anthropic_plan(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "anthropic-1",
                    created_at="2026-06-12T10:00:00+00:00",
                    request_json=_plateau_tool_result_body("RAW_TERMINAL_SECRET_ONE"),
                    text_chars=48_000,
                )
                _log_call(
                    store,
                    "anthropic-2",
                    created_at="2026-06-12T10:01:00+00:00",
                    request_json=_plateau_tool_result_body("RAW_TERMINAL_SECRET_TWO"),
                    text_chars=48_100,
                )
                _log_call(
                    store,
                    "openai-tool-light",
                    created_at="2026-06-12T10:02:00+00:00",
                    request_json=None,
                    provider="openai",
                    category="tool-light",
                    text_chars=12_000,
                    session_id="raw-openai-session-must-not-leak",
                )
                payload = build_terminal_output_compaction_dry_run(store, limit=10, min_block_chars=500)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_dry_run.v1")
        self.assertTrue(payload["dry_run"])
        self.assertGreaterEqual(payload["summary"]["planned_call_count"], 2)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        planned = [item for item in payload["plans"] if item["status"] == "planned"]
        self.assertGreaterEqual(len(planned), 2)
        self.assertTrue(planned[0]["preservation_flags"]["tool_protocol_ids_preserved"])
        blocked = [item for item in payload["plans"] if "unsupported-source-surface" in item["blockers"]]
        self.assertEqual(len(blocked), 1)
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "RAW_TERMINAL_SECRET_ONE",
            "RAW_TERMINAL_SECRET_TWO",
            "raw-session-id-must-not-leak",
            "raw-openai-session-must-not-leak",
            "toolu_preserve_me",
            "tests/test_secret_terminal.py",
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["raw_terminal_text_included"])

    def test_unsafe_cases_return_blockers_instead_of_plans(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                _log_call(
                    store,
                    "missing-body-1",
                    created_at="2026-06-12T10:00:00+00:00",
                    request_json=None,
                    text_chars=48_000,
                )
                _log_call(
                    store,
                    "missing-body-2",
                    created_at="2026-06-12T10:01:00+00:00",
                    request_json=None,
                    text_chars=48_200,
                )
                payload = build_terminal_output_compaction_dry_run(store, limit=10, min_block_chars=500)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["planned_call_count"], 0)
        self.assertGreater(payload["summary"]["blocked_call_count"], 0)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("request-body-unavailable", blockers)

    def test_cli_emits_terminal_output_compaction_dry_run(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    "cli-1",
                    created_at="2026-06-12T10:00:00+00:00",
                    request_json=_plateau_tool_result_body("RAW_CLI_SECRET_ONE"),
                    text_chars=48_000,
                )
                _log_call(
                    store,
                    "cli-2",
                    created_at="2026-06-12T10:01:00+00:00",
                    request_json=_plateau_tool_result_body("RAW_CLI_SECRET_TWO"),
                    text_chars=48_100,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.terminal_output_compaction_dry_run_cli(
                ["--db", db_path, "--limit", "10", "--min-block-chars", "500"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.terminal_output_compaction_dry_run.v1")
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)
        self.assertNotIn("RAW_CLI_SECRET", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
