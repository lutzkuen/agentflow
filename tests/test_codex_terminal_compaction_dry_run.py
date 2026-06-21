from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenclaw import cli
from tokenclaw.codex_terminal_compaction_dry_run import build_codex_terminal_transcript_compaction_dry_run
from tokenclaw.store import Store, stable_json


RAW_REQUEST_ID = "raw-terminal-request-id-must-not-leak"
RAW_THREAD_ID = "raw-terminal-thread-id-must-not-leak"
RAW_SESSION_ID = "raw-terminal-session-id-must-not-leak"
RAW_PATH = "/workspace/private/raw-terminal-path-must-not-leak.log"
RAW_TERMINAL = "secret shell command and terminal output must not leak"
RAW_CACHE_KEY = "raw-cache-key-must-not-leak"


class CodexTerminalTranscriptDryRunTests(unittest.TestCase):
    def _policy(self) -> dict:
        return {
            "terminal_transcript_compaction": {
                "enabled": False,
                "review_only": True,
                "policy_source": "local-default",
                "rule_id": "local-codex-terminal-transcript-test",
                "candidate_id": "/workspace/private/raw-candidate-id-must-not-leak",
                "conditions": {
                    "source_surface": "codex_turn",
                    "app_family": "codex",
                    "granularity": "agent_turn",
                    "workflow_phase": "tool_execution",
                    "text_bucket": ["8k_32k_chars", "32k_128k_chars"],
                    "terminal_fraction_bucket": ["50_75pct", "gte_75pct"],
                    "terminal_event_count_bucket": ["6_20", "21_100"],
                    "terminal_signal_source": [
                        "input-terminal-features",
                        "event-window-terminal-events",
                        "input-terminal-features+event-window",
                    ],
                    "cache_status": "skipped",
                    "already_crunched_repeated_scaffold": False,
                    "safety_preserve_diagnostics": True,
                    "min_input_chars": 100,
                    "min_terminal_chars": 100,
                    "min_projected_saved_chars": 100,
                },
                "action": {
                    "type": "compact_terminal_transcript",
                    "keep_recent_turns": 2,
                    "min_block_chars": 100,
                    "head_lines": 4,
                    "tail_lines": 4,
                    "max_evidence_lines": 20,
                    "min_saved_chars": 100,
                    "preserve_diagnostics": True,
                    "preserve_tool_protocol": True,
                    "preserve_recent_turns": True,
                    "preserve_error_lines": True,
                },
                "canary": {
                    "enabled": True,
                    "fraction": 0.0,
                    "holdout_fraction": 1.0,
                    "salt": "secret-canary-salt-must-not-leak",
                    "unit": "source_hash",
                },
            }
        }

    def _log_candidate_turn(self, store: Store) -> None:
        store.log_codex_app_event(
            id="codex-terminal-start",
            created_at="2026-06-13T00:00:00+00:00",
            direction="client_to_server",
            method="turn/start",
            request_id=RAW_REQUEST_ID,
            thread_id=RAW_THREAD_ID,
            session_id=RAW_SESSION_ID,
            message_chars=42_000,
            params_chars=41_000,
            input_items=1,
            input_text_chars=40_000,
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
                    "stack_trace_present": True,
                    "test_output_present": False,
                    "error_line_count_bucket": "2_5",
                    "class_count_buckets": {
                        "command_transcript": "gte_11",
                        "stdio_stream": "gte_11",
                        "log_line": "gte_11",
                        "stack_trace": "2_5",
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
                "cache_key": RAW_CACHE_KEY,
                "policy_source": "local-default",
            }),
            event_window_json=stable_json({
                "schema": "tokenclaw.codex_app_event_window.v1",
                "workflow_phase": "tool_execution",
                "input_text_chars": 40_000,
                "method_counts": {
                    "turn/start": 1,
                    "item/commandExecution/outputDelta": 25,
                    "turn/completed": 1,
                },
                "session_id": RAW_SESSION_ID,
                "request_id": RAW_REQUEST_ID,
                "thread_id": RAW_THREAD_ID,
                "diagnostic_path": RAW_PATH,
                "provider_body": {"raw": "provider body must not leak"},
            }),
        )
        store.log_codex_app_event(
            id="codex-terminal-output",
            created_at="2026-06-13T00:00:01+00:00",
            direction="server_to_client",
            method="item/commandExecution/outputDelta",
            request_id=RAW_REQUEST_ID,
            thread_id=RAW_THREAD_ID,
            session_id=RAW_SESSION_ID,
            message_chars=36_000,
            params_chars=None,
            input_items=None,
            input_text_chars=None,
            result_chars=None,
            error_code=None,
            error_message=RAW_TERMINAL,
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
            crunch_json=stable_json({"status": "skipped", "reason": "file-dependency-missing"}),
            cache_json=stable_json({"status": "skipped", "reason": "action-like-params"}),
            event_window_json=stable_json({
                "workflow_phase": "unknown",
                "input_text_chars": 0,
                "file_path": "/workspace/private/blocked-path-must-not-leak",
            }),
        )

    def test_dry_run_returns_holdout_plan_blockers_and_sanitized_metadata(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                self._log_candidate_turn(store)
                self._log_blocked_turn(store)
                payload = build_codex_terminal_transcript_compaction_dry_run(
                    store,
                    limit=10,
                    policy=self._policy(),
                )
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "tokenclaw.codex_terminal_transcript_compaction_dry_run.v1")
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["policy"]["runtime_mutation_enabled"])
        self.assertEqual(payload["summary"]["scanned_turn_count"], 2)
        self.assertEqual(payload["summary"]["planned_candidate_count"], 1)
        self.assertEqual(payload["summary"]["holdout_candidate_count"], 1)
        self.assertEqual(payload["summary"]["blocked_candidate_count"], 1)
        self.assertGreater(payload["summary"]["projected_saved_chars"], 0)
        self.assertGreater(payload["summary"]["projected_saved_tokens"], 0)

        holdout = next(item for item in payload["plans"] if item["status"] == "holdout")
        self.assertEqual(holdout["canary"]["cohort"], "holdout")
        self.assertEqual(holdout["target_count"], 1)
        self.assertGreater(holdout["target_summaries"][0]["preserved_head_line_count"], 0)
        self.assertGreater(holdout["target_summaries"][0]["preserved_tail_line_count"], 0)
        self.assertGreater(holdout["target_summaries"][0]["preserved_diagnostic_line_count"], 0)
        self.assertGreater(holdout["target_summaries"][0]["omitted_line_count_estimate"], 0)

        blocker_values = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("non-text-input", blocker_values)
        self.assertIn("action-like-params", blocker_values)
        self.assertIn("stale-risk-blockers", blocker_values)
        self.assertFalse(payload["privacy"]["raw_terminal_lines_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["file_paths_included"])
        self.assertFalse(payload["privacy"]["cache_keys_included"])

        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            RAW_REQUEST_ID,
            RAW_THREAD_ID,
            RAW_SESSION_ID,
            RAW_PATH,
            RAW_TERMINAL,
            RAW_CACHE_KEY,
            "secret-canary-salt-must-not-leak",
            "raw-candidate-id-must-not-leak",
            "provider body must not leak",
            "/workspace/private",
            "blocked-request-id-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cli_emits_codex_terminal_transcript_dry_run(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                self._log_candidate_turn(store)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.codex_terminal_transcript_dry_run_cli(
                ["--db", db_path, "--limit", "10"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.codex_terminal_transcript_compaction_dry_run.v1")
        self.assertEqual(payload["summary"]["planned_candidate_count"], 1)
        self.assertEqual(payload["summary"]["holdout_candidate_count"], 1)
        self.assertNotIn(RAW_REQUEST_ID, stdout.getvalue())
        self.assertNotIn(RAW_TERMINAL, stdout.getvalue())

    def test_raw_policy_metadata_and_unsupported_conditions_are_publicized(self):
        raw_rule_id = "/workspace/private/raw-terminal-rule-id-must-not-leak"
        raw_candidate_id = "/workspace/private/raw-terminal-candidate-id-must-not-leak"
        raw_action_id = "/workspace/private/raw-terminal-action-id-must-not-leak"
        raw_condition_key = "/workspace/private/raw-terminal-condition-key-must-not-leak"
        raw_condition_value = "raw terminal condition value must not leak"
        policy = json.loads(json.dumps(self._policy()))
        terminal_policy = policy["terminal_transcript_compaction"]
        terminal_policy["rule_id"] = raw_rule_id
        terminal_policy["candidate_id"] = raw_candidate_id
        terminal_policy["action_id"] = raw_action_id
        terminal_policy["policy_source"] = "managed-recommended"
        terminal_policy["conditions"][raw_condition_key] = raw_condition_value

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                self._log_candidate_turn(store)
                payload = build_codex_terminal_transcript_compaction_dry_run(
                    store,
                    limit=10,
                    policy=policy,
                )
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["blocked_candidate_count"], 1)
        blockers = {item["value"] for item in payload["blocker_reason_breakdown"]}
        self.assertIn("unsupported-condition:unknown", blockers)
        plan = payload["plans"][0]
        self.assertTrue(str(plan["candidate_id"]).startswith("codex-terminal-transcript-dry-run:"))
        self.assertTrue(str(plan["policy_candidate_id"]).startswith("codex-terminal-transcript-candidate:"))
        self.assertTrue(str(plan["rule_id"]).startswith("codex-terminal-transcript-rule:"))
        self.assertTrue(str(plan["action_id"]).startswith("codex-terminal-transcript-action:"))

        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            raw_rule_id,
            raw_candidate_id,
            raw_action_id,
            raw_condition_key,
            raw_condition_value,
            RAW_REQUEST_ID,
            RAW_THREAD_ID,
            RAW_SESSION_ID,
            RAW_PATH,
            RAW_TERMINAL,
            RAW_CACHE_KEY,
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
