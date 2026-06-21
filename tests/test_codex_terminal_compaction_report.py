from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenclaw import cli
from tokenclaw.codex_terminal_compaction_report import build_codex_terminal_transcript_opportunity_report
from tokenclaw.store import Store, stable_json


class CodexTerminalTranscriptOpportunityTests(unittest.TestCase):
    def _log_high_terminal_turn(self, store: Store) -> None:
        store.log_codex_app_event(
            id="codex-terminal-start",
            created_at="2026-06-13T00:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="raw-terminal-request-id-must-not-leak",
            thread_id="raw-terminal-thread-id-must-not-leak",
            session_id="raw-terminal-session-id-must-not-leak",
            message_chars=22_000,
            params_chars=21_000,
            input_items=1,
            input_text_chars=20_000,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            routing_json=stable_json({
                "requested_model": "gpt-5.4",
                "workflow_phase": "tool_execution",
                "terminal_log_features": {
                    "schema": "tokenclaw.terminal_log_features.v1",
                    "terminal_output_char_fraction_bucket": "gte_75pct",
                    "log_line_fraction_bucket": "gte_75pct",
                    "stack_trace_present": False,
                    "test_output_present": False,
                    "error_line_count_bucket": "zero",
                    "class_count_buckets": {
                        "command_transcript": "gte_11",
                        "stdio_stream": "gte_11",
                        "log_line": "gte_11",
                        "stack_trace": "zero",
                        "test_output": "zero",
                        "build_output": "zero",
                    },
                    "privacy": {
                        "metadata_only": True,
                        "raw_terminal_text_included": False,
                    },
                },
            }),
            crunch_json=stable_json({
                "status": "skipped",
                "reason": "no-codex-repeated-scaffolding",
                "policy_source": "local-default",
            }),
            cache_json=stable_json({
                "status": "skipped",
                "eligible": False,
                "reason": "codex-app-cache-disabled",
                "policy_source": "local-default",
            }),
            event_window_json=stable_json({
                "schema": "tokenclaw.codex_app_event_window.v1",
                "workflow_phase": "tool_execution",
                "input_text_chars": 20_000,
                "method_counts": {
                    "turn/start": 1,
                    "item/commandExecution/outputDelta": 25,
                    "turn/completed": 1,
                },
                "session_id": "raw-terminal-session-id-must-not-leak",
                "request_id": "raw-terminal-request-id-must-not-leak",
                "thread_id": "raw-terminal-thread-id-must-not-leak",
                "diagnostic_path": "/workspace/private/raw-terminal-path-must-not-leak.log",
            }),
        )
        store.log_codex_app_event(
            id="codex-terminal-output",
            created_at="2026-06-13T00:00:01+00:00",
            direction="server_to_client",
            method="item/commandExecution/outputDelta",
            request_id="raw-terminal-request-id-must-not-leak",
            thread_id="raw-terminal-thread-id-must-not-leak",
            session_id="raw-terminal-session-id-must-not-leak",
            message_chars=16_000,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message="raw terminal output secret must not leak",
            latency_ms=None,
        )

    def _log_blocked_turn(self, store: Store) -> None:
        store.log_codex_app_event(
            id="codex-terminal-blocked",
            created_at="2026-06-13T00:00:02+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id="blocked-request-id-must-not-leak",
            thread_id="blocked-thread-id-must-not-leak",
            session_id="blocked-session-id-must-not-leak",
            message_chars=500,
            params_chars=400,
            input_items=1,
            input_text_chars=0,
            result_chars=None,
            error_code=None,
            error_message=None,
            latency_ms=None,
            routing_json=stable_json({"reason": "non-text-input"}),
            crunch_json=stable_json({}),
            cache_json=stable_json({"status": "skipped", "reason": "action-like-params"}),
            event_window_json=stable_json({
                "workflow_phase": "unknown",
                "input_text_chars": 0,
                "file_path": "/workspace/private/blocked-path-must-not-leak",
            }),
        )

    def test_report_promotes_high_terminal_turn_without_raw_content(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                self._log_high_terminal_turn(store)
                self._log_blocked_turn(store)
                payload = build_codex_terminal_transcript_opportunity_report(
                    store,
                    limit=10,
                    min_input_chars=100,
                    min_terminal_chars=100,
                )
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "tokenclaw.codex_terminal_transcript_opportunity.v1")
        self.assertEqual(payload["summary"]["scanned_turn_count"], 2)
        self.assertEqual(payload["summary"]["candidate_turn_count"], 1)
        self.assertEqual(payload["summary"]["candidate_count"], 1)
        self.assertGreater(payload["summary"]["projected_saved_chars"], 0)
        self.assertGreater(payload["summary"]["terminal_event_message_chars"], 0)
        self.assertEqual(payload["workflow_phase_breakdown"][0]["phase"], "tool_execution")
        self.assertFalse(payload["privacy"]["raw_terminal_text_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["file_paths_included"])
        blocker_values = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("non-text-input", blocker_values)
        self.assertIn("action-like-params", blocker_values)

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-terminal-request-id-must-not-leak", rendered)
        self.assertNotIn("raw-terminal-thread-id-must-not-leak", rendered)
        self.assertNotIn("raw-terminal-session-id-must-not-leak", rendered)
        self.assertNotIn("raw terminal output secret must not leak", rendered)
        self.assertNotIn("/workspace/private", rendered)

    def test_cli_emits_privacy_safe_summary(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                self._log_high_terminal_turn(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_terminal_transcript_opportunity_cli(
                ["--db", db_path, "--limit", "10", "--min-input-chars", "100", "--min-terminal-chars", "100"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.codex_terminal_transcript_opportunity.v1")
        self.assertEqual(payload["summary"]["candidate_turn_count"], 1)
        self.assertFalse(payload["privacy"]["raw_terminal_lines_included"])
        self.assertNotIn("raw-terminal-request-id-must-not-leak", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
