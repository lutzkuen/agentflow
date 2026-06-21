from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.optimization_action_ledger import (
    build_optimization_action_ledger,
    build_optimization_action_ledger_report,
)
from tokenclaw.store import SQLiteStore, stable_json, utc_now


FORBIDDEN_VALUES = (
    "raw-ledger-prompt-secret",
    "raw-ledger-response-secret",
    "raw-ledger-session-secret",
    "req-ledger-secret",
    "cache-key-ledger-secret",
    "raw-ledger-pattern-secret",
    "/home/lutz/private/ledger_secret.py",
    "terminal output raw line",
    "tool payload secret",
)

FORBIDDEN_KEYS = (
    '"cache_key"',
    '"content"',
    '"file_path"',
    '"messages":',
    '"prompt"',
    '"raw_request"',
    '"request_id"',
    '"response_json"',
    '"session_id"',
    '"tool_payload"',
)


class OptimizationActionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _assert_private(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for value in FORBIDDEN_VALUES:
            self.assertNotIn(value, rendered)
        for key in FORBIDDEN_KEYS:
            self.assertNotIn(key, rendered)

    def _log_call(
        self,
        *,
        provider: str = "openai",
        source_surface: str = "openai_responses",
        endpoint: str = "responses",
        category: str = "tool-result",
        requested_model: str = "gpt-5.4",
        routed_model: str = "gpt-5.4-mini",
        routing_meta: dict[str, object] | None = None,
        crunch_meta: dict[str, object] | None = None,
        cache_meta: dict[str, object] | None = None,
        cache_hit: int = 0,
        stream: int = 0,
    ) -> None:
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=f"/v1/{endpoint}",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=cache_hit,
            status_code=200,
            latency_ms=125,
            input_tokens_est=3200,
            output_tokens_est=120,
            actual_input_tokens=3000,
            actual_output_tokens=100,
            cost_est_usd=0.01,
            cost_baseline_usd=0.05,
            crunch_json=stable_json(crunch_meta or {"changed": False}),
            routing_json=stable_json(
                routing_meta
                or {
                    "provider": provider,
                    "source_surface": source_surface,
                    "endpoint": endpoint,
                    "category": category,
                    "workflow_phase": "tool-execution",
                    "text_chars": 12_000,
                    "enabled": True,
                    "requested_model": requested_model,
                    "routed_model": routed_model,
                    "reason": "selected-canary",
                }
            ),
            cache_json=stable_json(cache_meta or {"status": "miss", "reason": "exact-miss"}),
            error=None,
            request_json=stable_json(
                {
                    "request_id": "req-ledger-secret",
                    "messages": [{"content": "raw-ledger-prompt-secret"}],
                    "cache_key": "cache-key-ledger-secret",
                    "file_path": "/home/lutz/private/ledger_secret.py",
                }
            ),
            response_json=stable_json({"content": "raw-ledger-response-secret"}),
            session_id="raw-ledger-session-secret",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family="gpt-5" if provider == "openai" else "sonnet",
            routed_model_family="gpt-5" if provider == "openai" else "haiku",
        )

    def test_ledger_builds_cross_family_entries_from_representative_metadata(self) -> None:
        routing_meta = {
            "provider": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "chat",
            "workflow_phase": "tool-execution",
            "text_chars": 2400,
            "managed_pattern_features": {
                "local_pattern_module_families": ["terminal_logs"],
                "pattern_hashes": ["sha256:raw-ledger-pattern-secret"],
            },
            "openai_optimization_governor": {
                "schema": "tokenclaw.openai_optimization_governor.v1",
                "selected_action_families": ["routing"],
                "family_status": {
                    "routing": {"eligible": True, "selected": True, "policy_source": "local-manual"},
                    "old_context_summary": {"eligible": True, "selected": False, "candidate_id": "raw-ledger-session-secret"},
                    "cache_replay": {"eligible": True, "selected": False, "candidate_id": "cache-key-ledger-secret"},
                },
                "suppressed_families": [
                    {"family": "old_context_summary", "reason_codes": ["conflicts-with-selected-family"]},
                    {"family": "cache_replay", "reason_codes": ["cache_key must not leak"]},
                ],
            },
        }
        crunch_meta = {
            "old_context_summarization": {
                "status": "applied",
                "candidate_id": "summary-candidate",
                "raw_request": {"messages": [{"content": "raw-ledger-prompt-secret"}]},
            },
            "codex_repeated_scaffolding": {
                "status": "applied",
                "candidate_id": "scaffold-candidate",
                "saved_chars": 1200,
            },
            "terminal_output_compaction": {
                "status": "holdout",
                "reason": "canary-holdout",
                "action_id": "terminal-action",
                "terminal_line": "terminal output raw line",
            },
        }
        cache_meta = {
            "status": "bypassed",
            "reason": "file-dependency-missing",
            "cache_key": "cache-key-ledger-secret",
            "pattern_rule": {
                "rule_id": "cache-replay-rule",
                "candidate_id": "cache-replay-candidate",
                "policy_source": "local-manual",
            },
            "cache_replay_canary": {
                "status": "bypassed",
                "reason": "file-dependency-missing",
                "tool_payload": "tool payload secret",
            },
        }

        ledger = build_optimization_action_ledger(
            row={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "actual_input_tokens": 600,
                "cost_est_usd": 0.003,
                "cost_baseline_usd": 0.012,
            },
            routing_meta=routing_meta,
            crunch_meta=crunch_meta,
            cache_meta=cache_meta,
        )

        self.assertEqual(ledger["schema"], "tokenclaw.optimization_action_ledger.v1")
        by_family = {entry["family"]: entry for entry in ledger["entries"]}
        self.assertEqual(by_family["routing"]["status"], "applied")
        self.assertEqual(by_family["old_context_summary"]["status"], "suppressed")
        self.assertEqual(by_family["cache_replay"]["status"], "suppressed")
        self.assertEqual(by_family["repeated_scaffold_crunch"]["status"], "applied")
        self.assertEqual(by_family["terminal_output_compaction"]["status"], "holdout")
        self.assertEqual(by_family["pattern_crunch:terminal_logs"]["status"], "eligible")
        self.assertEqual(by_family["routing"]["text_bucket"], "1_5k_8k")
        self.assertIn("reason:", by_family["cache_replay"]["reason_codes"][0])
        self.assertFalse(ledger["privacy"]["provider_bodies_included"])
        self._assert_private(ledger)

    def test_report_summarizes_recent_rows_by_family_status_reason_without_raw_payloads(self) -> None:
        governor = {
            "schema": "tokenclaw.openai_optimization_governor.v1",
            "selected_action_families": ["routing"],
            "family_status": {
                "routing": {"eligible": True, "selected": True, "policy_source": "local-manual"},
                "old_context_summary": {"eligible": True, "selected": False},
                "cache_replay": {"eligible": True, "selected": False},
            },
            "suppressed_families": [
                {"family": "old_context_summary", "reason_codes": ["conflicts-with-selected-family"]},
                {"family": "cache_replay", "reason_codes": ["conflicts-with-selected-family"]},
            ],
        }
        self._log_call(
            routing_meta={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "planning",
                "text_chars": 2400,
                "openai_optimization_governor": governor,
            },
            crunch_meta={"openai_optimization_governor": governor},
            cache_meta={"openai_optimization_governor": governor},
        )
        self._log_call(
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            crunch_meta={
                "changed": True,
                "codex_repeated_scaffolding": {"status": "applied", "saved_chars": 1000},
                "terminal_output_compaction": {"status": "bypass", "reason": "active-thinking-blocked"},
            },
            cache_meta={"status": "skipped", "reason": "streaming"},
            stream=1,
        )
        self._log_call(
            routed_model="gpt-5.4",
            cache_hit=1,
            cache_meta={
                "status": "hit",
                "reason": "dependency-stable",
                "cache_replay_canary": {
                    "status": "applied",
                    "reason": "dependency-stable",
                    "candidate_id": "cache-replay-candidate",
                },
            },
        )

        report = build_optimization_action_ledger_report(self.store, limit=20)

        self.assertEqual(report["schema"], "tokenclaw.optimization_action_ledger_report.v1")
        self.assertEqual(report["sampled_call_count"], 3)
        self.assertGreaterEqual(report["entry_count"], 6)
        status_counts = {
            (row["family"], row["status"]): row["count"]
            for row in report["family_status_counts"]
        }
        self.assertEqual(status_counts[("routing", "applied")], 1)
        self.assertEqual(status_counts[("old_context_summary", "suppressed")], 1)
        self.assertEqual(status_counts[("cache_replay", "applied")], 1)
        self.assertEqual(status_counts[("repeated_scaffold_crunch", "applied")], 1)
        self.assertEqual(status_counts[("terminal_output_compaction", "suppressed")], 1)
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self._assert_private(report)

    def test_cli_emits_optimization_action_ledger_report(self) -> None:
        self._log_call(
            routing_meta={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "summary",
                "text_chars": 1800,
                "enabled": True,
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "reason": "small-request",
            },
            cache_meta={"status": "miss", "reason": "exact-miss"},
        )

        output = io.StringIO()
        code = cli.optimization_action_ledger_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "tokenclaw.optimization_action_ledger_report.v1")
        self.assertEqual(payload["sampled_call_count"], 1)
        self.assertEqual(payload["family_status_counts"][0]["family"], "routing")
        self._assert_private(payload)
