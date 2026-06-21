from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tokenclaw import cli
from tokenclaw.codex_terminal_compaction_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE,
    build_codex_terminal_transcript_lifecycle_feedback,
    queue_codex_terminal_transcript_lifecycle_feedback,
)
from tokenclaw.codex_terminal_compaction_impact import build_codex_terminal_transcript_compaction_impact_report
from tokenclaw.codex_terminal_transcript_compaction import FAMILY, SCHEMA as CANARY_SCHEMA
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.optimization import feedback
from tokenclaw.store import Store, stable_json


RAW_REQUEST_ID = "raw-codex-terminal-request-id-must-not-leak"
RAW_THREAD_ID = "raw-codex-terminal-thread-id-must-not-leak"
RAW_SESSION_ID = "raw-codex-terminal-session-id-must-not-leak"
RAW_CACHE_KEY = "raw-codex-terminal-cache-key-must-not-leak"
RAW_TERMINAL = "RAW_CODEX_TERMINAL_TRANSCRIPT_SECRET"
RAW_PATH = "/workspace/private/codex-terminal-secret.log"


def _terminal_meta(
    *,
    cohort: str,
    saved_chars: int = 4000,
    planned_saved_chars: int = 4000,
    reason: str | None = None,
    candidate_id: str = "codex-terminal-transcript-candidate-safe",
    rule_id: str = "local-codex-terminal-transcript-rule",
) -> dict:
    applied = cohort == "canary_applied"
    return {
        "schema": CANARY_SCHEMA,
        "optimization_family": FAMILY,
        "status": "applied" if applied else "holdout",
        "reason": reason or ("terminal-transcript-compacted" if applied else "terminal-compaction-holdout"),
        "applied": applied,
        "changed": applied,
        "before_chars": 40_000,
        "after_chars": 40_000 - saved_chars if applied else 40_000,
        "saved_chars": saved_chars if applied else 0,
        "planned_saved_chars": planned_saved_chars,
        "policy_source": "local-manual",
        "rule_id": rule_id,
        "candidate_id": candidate_id,
        "canary": {
            "enabled": True,
            "cohort": cohort,
            "status": "applied" if applied else "holdout",
            "candidate_id": candidate_id,
            "raw_basis_included": False,
        },
        "raw_text_included": False,
        "raw_commands_included": False,
        "coordinator": {
            "selected_families": [FAMILY] if applied else [],
            "suppressed_families": [],
            "suppressed_by": None,
        },
    }


def _log_codex_turn(
    store: Store,
    suffix: str,
    *,
    cohort: str,
    response_error_code: int | None = None,
    response_latency_ms: int = 1000,
    saved_chars: int = 4000,
    planned_saved_chars: int = 4000,
    candidate_id: str = "codex-terminal-transcript-candidate-safe",
    rule_id: str = "local-codex-terminal-transcript-rule",
) -> None:
    request_id = f"{RAW_REQUEST_ID}-{suffix}"
    meta = _terminal_meta(
        cohort=cohort,
        saved_chars=saved_chars,
        planned_saved_chars=planned_saved_chars,
        candidate_id=candidate_id,
        rule_id=rule_id,
    )
    store.log_codex_app_event(
        id=f"start-{suffix}",
        created_at=f"2026-06-13T00:00:0{len(suffix) % 9}+00:00",
        direction="client_to_server",
        method="turn/start",
        request_id=request_id,
        thread_id=RAW_THREAD_ID,
        session_id=RAW_SESSION_ID,
        message_chars=41_000,
        params_chars=40_500,
        input_items=1,
        input_text_chars=40_000,
        result_chars=None,
        error_code=None,
        error_message=None,
        latency_ms=None,
        routing_json=stable_json({
            "requested_model": "gpt-5-codex",
            "routed_model": "gpt-5-codex",
            "workflow_phase": "tool_execution",
            "category": "tool-execution",
        }),
        crunch_json=stable_json({
            "status": "applied" if cohort == "canary_applied" else "holdout",
            FAMILY: meta,
        }),
        cache_json=stable_json({
            "status": "skipped",
            "reason": "codex-app-cache-disabled",
            "cache_key": RAW_CACHE_KEY,
        }),
        event_window_json=stable_json({
            "workflow_phase": "tool_execution",
            "input_text_chars": 40_000,
            "request_id": request_id,
            "thread_id": RAW_THREAD_ID,
            "session_id": RAW_SESSION_ID,
            "file_path": RAW_PATH,
            "provider_body": {"raw": "provider body must not leak"},
        }),
    )
    store.log_codex_app_event(
        id=f"response-{suffix}",
        created_at=f"2026-06-13T00:00:1{len(suffix) % 9}+00:00",
        direction="server_to_client",
        method="turn/completed",
        request_id=request_id,
        thread_id=RAW_THREAD_ID,
        session_id=RAW_SESSION_ID,
        message_chars=2000,
        params_chars=None,
        input_items=None,
        input_text_chars=None,
        result_chars=2000,
        error_code=response_error_code,
        error_message=RAW_TERMINAL if response_error_code is not None else None,
        latency_ms=response_latency_ms,
    )


class CodexTerminalTranscriptImpactTests(unittest.TestCase):
    def test_impact_report_promotes_positive_applied_vs_holdout_metadata(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_codex_turn(store, "a1", cohort="canary_applied", response_latency_ms=900)
                _log_codex_turn(store, "a2", cohort="canary_applied", response_latency_ms=950)
                _log_codex_turn(store, "h1", cohort="canary_holdout", response_latency_ms=1200)
                report = build_codex_terminal_transcript_compaction_impact_report(store, limit=10)
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "tokenclaw.codex_terminal_transcript_compaction_impact.v1")
        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertGreater(report["summary"]["net_savings_usd"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "promote")
        self.assertEqual(candidate["deltas"]["error_rate_delta"], 0.0)
        self.assertLess(candidate["deltas"]["latency_avg_ms_delta"], 0)

    def test_impact_report_holds_on_negative_savings(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_codex_turn(store, "a1", cohort="canary_applied", saved_chars=-400)
                _log_codex_turn(store, "a2", cohort="canary_applied", saved_chars=-400)
                _log_codex_turn(store, "h1", cohort="canary_holdout")
                report = build_codex_terminal_transcript_compaction_impact_report(store, limit=10)
            finally:
                store.conn.close()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "hold")
        self.assertIn("hold-negative-savings", candidate["reason_codes"])

    def test_impact_report_rolls_back_on_error_regression(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_codex_turn(store, "a1", cohort="canary_applied", response_error_code=-32603)
                _log_codex_turn(store, "a2", cohort="canary_applied", response_error_code=-32603)
                _log_codex_turn(store, "h1", cohort="canary_holdout")
                report = build_codex_terminal_transcript_compaction_impact_report(store, limit=10)
            finally:
                store.conn.close()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "rollback")
        self.assertIn("rollback-absolute-error-rate", candidate["reason_codes"])
        self.assertEqual(report["summary"]["rollback_action_count"], 1)
        action = report["rollback_actions"][0]
        self.assertTrue(action["review_only"])
        self.assertEqual(
            action["recommended_local_policy_patch"]["terminal_transcript_compaction"]["canary"]["holdout_fraction"],
            1.0,
        )

    def test_impact_report_marks_insufficient_evidence(self):
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_codex_turn(store, "a1", cohort="canary_applied")
                report = build_codex_terminal_transcript_compaction_impact_report(store, limit=10)
            finally:
                store.conn.close()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "insufficient-evidence")
        self.assertIn("insufficient-applied-samples", candidate["reason_codes"])
        self.assertIn("insufficient-holdout-samples", candidate["reason_codes"])

    def test_report_cli_and_lifecycle_payload_are_content_free(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            store = Store(db_path)
            try:
                _log_codex_turn(
                    store,
                    "a1",
                    cohort="canary_applied",
                    candidate_id="/workspace/private/raw-candidate-id-must-not-leak",
                    rule_id="raw-rule-id-must-not-leak",
                )
                _log_codex_turn(
                    store,
                    "a2",
                    cohort="canary_applied",
                    candidate_id="/workspace/private/raw-candidate-id-must-not-leak",
                    rule_id="raw-rule-id-must-not-leak",
                )
                _log_codex_turn(
                    store,
                    "h1",
                    cohort="canary_holdout",
                    candidate_id="/workspace/private/raw-candidate-id-must-not-leak",
                    rule_id="raw-rule-id-must-not-leak",
                )
                report = build_codex_terminal_transcript_compaction_impact_report(store, limit=10)
                payload = build_codex_terminal_transcript_lifecycle_feedback(report)
            finally:
                store.conn.close()

            stdout = io.StringIO()
            with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}, clear=False):
                code = cli.codex_terminal_transcript_impact_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(code, 0)
        cli_payload = json.loads(stdout.getvalue())
        self.assertEqual(cli_payload["schema"], "tokenclaw.codex_terminal_transcript_compaction_impact.v1")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["metadata"]["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(payload["metadata"]["lifecycle_kind"], "codex_terminal_transcript_compaction")
        self.assertEqual(managed_egress_violations(payload), [])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertFalse(report["privacy"]["raw_terminal_text_included"])
        rendered = json.dumps({"report": report, "feedback": payload, "cli": cli_payload}, sort_keys=True)
        for forbidden in (
            RAW_REQUEST_ID,
            RAW_THREAD_ID,
            RAW_SESSION_ID,
            RAW_CACHE_KEY,
            RAW_TERMINAL,
            RAW_PATH,
            "provider body must not leak",
            "raw-candidate-id-must-not-leak",
            "raw-rule-id-must-not-leak",
            "/workspace/private",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_lifecycle_feedback_queues_offline_and_status_is_payload_free(self):
        result = {
            "schema": "tokenclaw.codex_terminal_transcript_compaction_impact.v1",
            "ok": True,
            "read_only": True,
            "summary": {"applied_count": 2, "holdout_count": 1, "candidate_group_count": 1},
            "candidates": [
                {
                    "candidate_id": "codex-terminal-transcript-candidate-safe",
                    "rule_id": "local-codex-terminal-transcript-rule",
                    "policy_source": "local-manual",
                    "source_surface": "codex_turn",
                    "app_family": "codex",
                    "granularity": "agent_turn",
                    "workflow_phase": "tool_execution",
                    "requested_model": "gpt-5-codex",
                    "routed_model": "gpt-5-codex",
                    "cohorts": {
                        "applied": {"count": 2, "error_rate": 0.0},
                        "holdout": {"count": 1, "error_rate": 0.0},
                    },
                    "deltas": {"error_rate_delta": 0.0, "latency_avg_ms_delta": -100},
                    "verdict": "promote",
                    "reason_codes": ["impact-positive"],
                    "net_savings_usd": 0.01,
                }
            ],
        }

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}, clear=False):
                    meta = asyncio.run(
                        queue_codex_terminal_transcript_lifecycle_feedback(
                            store,
                            result,
                            flush_immediately=False,
                        )
                    )
                status = feedback.managed_feedback_status_result(
                    store,
                    source_surface=SOURCE_SURFACE,
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "queued")
        lifecycle = status["codex_terminal_transcript_lifecycle"]
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertFalse(lifecycle["payload_json_included"])
        rendered = json.dumps(status, sort_keys=True)
        for forbidden in (RAW_REQUEST_ID, RAW_TERMINAL, RAW_PATH, RAW_CACHE_KEY):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
