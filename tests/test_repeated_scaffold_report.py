from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.repeated_scaffold_report import build_repeated_scaffold_opportunity_report
from tokenclaw.stats import stats_repeated_scaffold_opportunity
from tokenclaw.store import SQLiteStore, stable_json, utc_now


class RepeatedScaffoldReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        provider: str = "anthropic",
        path: str = "/v1/messages",
        category: str = "long-context",
        workflow_phase: str = "tool-execution",
        request_json: dict[str, object] | None = None,
        routing_extra: dict[str, object] | None = None,
        cache_extra: dict[str, object] | None = None,
        requested_model: str = "claude-sonnet-4-6",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        requested_model_family: str = "sonnet",
        stream: int = 1,
        text_chars: int = 24_000,
        cost: float = 0.024,
    ) -> None:
        routing = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_chars": text_chars,
            "has_tools": category.startswith("tool"),
        }
        if routing_extra:
            routing.update(routing_extra)
        cache = {"status": "skipped", "reason": "streaming", "policy_source": "local-default"}
        if cache_extra:
            cache.update(cache_extra)
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=requested_model,
            stream=stream,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=text_chars // 4,
            output_tokens_est=50,
            actual_input_tokens=text_chars // 4,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=cost,
            crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
            routing_json=stable_json(routing),
            cache_json=stable_json(cache),
            error=None,
            request_json=stable_json(request_json) if request_json is not None else None,
            response_json=stable_json({"text": "raw response must not leak"}),
            session_id="raw-session-must-not-leak",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family=requested_model_family,
            routed_model_family=requested_model_family,
        )

    def test_body_logging_on_measures_repeated_provider_message_scaffold_without_content(self) -> None:
        repeated = (
            "Agent scaffold banner with invariant rules and workflow framing that repeats across provider messages. "
            "Keep this instruction stable and never expose the raw prompt text."
        )
        raw_body = {
            "model": "claude-sonnet-4-6",
            "system": repeated,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"{repeated}\nunique first request secret"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "short acknowledgement"}]},
                {"role": "user", "content": [{"type": "text", "text": f"{repeated}\nunique second request secret"}]},
            ],
            "cache_key": "raw-cache-key-must-not-leak",
        }
        self._log_call(request_json=raw_body)
        self._log_call(request_json=raw_body)

        report = build_repeated_scaffold_opportunity_report(self.store, limit=20)

        self.assertEqual(report["schema"], "tokenclaw.repeated_scaffold_opportunity.v1")
        self.assertEqual(report["summary"]["body_rows"], 2)
        self.assertGreater(report["summary"]["normalized_scaffold_fingerprint_rows"], 0)
        self.assertGreater(report["summary"]["projected_saved_chars"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["body_rows"], 2)
        self.assertGreater(candidate["repeated_fingerprint_rows"], 0)
        self.assertTrue(candidate["normalized_scaffold_fingerprint"]["present"])
        self.assertFalse(candidate["normalized_scaffold_fingerprint"]["included"])
        self.assertGreater(candidate["projected_saved_tokens"], 0)

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("Agent scaffold banner", rendered)
        self.assertNotIn("unique first request secret", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)
        self.assertNotIn("raw-session-must-not-leak", rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["normalized_scaffold_fingerprints_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])

    def test_body_logging_off_uses_metadata_hashes_and_reports_blockers_without_hash_leakage(self) -> None:
        routing_extra = {
            "managed_pattern_features": {
                "pattern_hashes": ["sha256:raw-pattern-hash-must-not-leak"],
                "local_pattern_module_families": ["terminal_logs"],
                "pattern_hash_count": 1,
            }
        }
        for _ in range(3):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                category="chat",
                workflow_phase="execution",
                stream=0,
                text_chars=20_000,
                routing_extra=routing_extra,
                request_json=None,
            )

        report = build_repeated_scaffold_opportunity_report(self.store, limit=20)

        self.assertEqual(report["summary"]["body_logging_off_rows"], 3)
        self.assertEqual(report["summary"]["metadata_pattern_hash_rows"], 3)
        self.assertGreater(report["summary"]["projected_saved_tokens"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["metadata_only_rows"], 3)
        self.assertEqual(candidate["fingerprint_source"], "metadata_pattern_hash")
        self.assertIn("request-body-unavailable", candidate["blockers"])
        self.assertTrue(candidate["normalized_scaffold_fingerprint"]["present"])
        self.assertFalse(candidate["normalized_scaffold_fingerprint"]["included"])

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-pattern-hash-must-not-leak", rendered)
        self.assertFalse(report["privacy"]["pattern_hashes_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        self._log_call(
            request_json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Repeated scaffold line long enough to fingerprint and repeat across the request.",
                    },
                    {
                        "role": "assistant",
                        "content": "Repeated scaffold line long enough to fingerprint and repeat across the request.",
                    },
                ]
            }
        )
        self._log_call(
            request_json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Repeated scaffold line long enough to fingerprint and repeat across the request.",
                    },
                    {
                        "role": "assistant",
                        "content": "Repeated scaffold line long enough to fingerprint and repeat across the request.",
                    },
                ]
            }
        )

        result = asyncio.run(stats_repeated_scaffold_opportunity(self.store, limit=10))
        self.assertEqual(result["schema"], "tokenclaw.repeated_scaffold_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.repeated_scaffold_opportunity_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.repeated_scaffold_opportunity.v1")
        self.assertEqual(payload["summary"]["provider_call_count"], 2)
