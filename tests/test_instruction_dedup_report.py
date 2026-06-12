from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.instruction_dedup_report import build_instruction_dedup_opportunity_report
from agentflow_proxy.stats import stats_instruction_dedup_opportunity
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class InstructionDedupReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        provider: str = "anthropic",
        path: str = "/v1/messages",
        requested_model: str = "claude-sonnet-4-6",
        requested_model_family: str = "sonnet",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        category: str = "tool-result",
        workflow_phase: str = "tool-execution",
        request_json: dict[str, object] | None = None,
        routing_extra: dict[str, object] | None = None,
        crunch_extra: dict[str, object] | None = None,
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
        crunch = {"changed": False, "tokens_saved_est": 0}
        if crunch_extra:
            crunch.update(crunch_extra)
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=requested_model,
            stream=1,
            cache_hit=0,
            status_code=200,
            latency_ms=100,
            input_tokens_est=text_chars // 4,
            output_tokens_est=50,
            actual_input_tokens=text_chars // 4,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=cost,
            crunch_json=stable_json(crunch),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "skipped", "reason": "streaming", "policy_source": "local-default"}),
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

    def test_anthropic_body_on_measures_system_instruction_repeats_without_content(self) -> None:
        instruction = (
            "System instruction section with stable agent workflow rules and privacy boundaries. "
            "This exact private instruction repeats across provider calls and must never be emitted."
        )
        for idx in range(2):
            self._log_call(
                request_json={
                    "model": "claude-sonnet-4-6",
                    "system": instruction,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": f"user secret {idx} must not leak"}]},
                        {"role": "user", "content": [{"type": "tool_result", "content": "tool payload must not leak"}]},
                    ],
                    "cache_key": "raw-cache-key-must-not-leak",
                }
            )

        report = build_instruction_dedup_opportunity_report(self.store, limit=20)

        self.assertEqual(report["schema"], "agentflow.instruction_dedup_opportunity.v1")
        self.assertEqual(report["summary"]["body_rows"], 2)
        self.assertGreater(report["summary"]["instruction_fingerprint_rows"], 0)
        self.assertGreater(report["summary"]["projected_saved_chars"], 0)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["source_surface"], "anthropic_messages")
        self.assertGreater(candidate["repeated_fingerprint_rows"], 0)
        self.assertTrue(candidate["instruction_section_fingerprint"]["present"])
        self.assertFalse(candidate["instruction_section_fingerprint"]["included"])

        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("System instruction section", rendered)
        self.assertNotIn("user secret", rendered)
        self.assertNotIn("tool payload", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-session-must-not-leak", rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)
        self.assertFalse(report["privacy"]["raw_instruction_text_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["instruction_section_fingerprints_included"])
        self.assertFalse(report["privacy"]["tool_payloads_included"])

    def test_openai_body_on_measures_instructions_and_developer_messages(self) -> None:
        instruction = (
            "Developer instruction block with repeated coding-agent operating constraints. "
            "Keep this local and only emit aggregate savings measurements."
        )
        for _ in range(2):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                requested_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                source_surface="openai_responses",
                endpoint="responses",
                category="chat",
                workflow_phase="planning",
                request_json={
                    "model": "gpt-5.4-mini",
                    "instructions": instruction,
                    "input": [
                        {"role": "developer", "content": [{"type": "input_text", "text": instruction}]},
                        {"role": "user", "content": [{"type": "input_text", "text": "private user request"}]},
                    ],
                },
            )

        report = build_instruction_dedup_opportunity_report(self.store, limit=20)

        self.assertGreater(report["summary"]["projected_saved_tokens"], 0)
        self.assertEqual(report["source_surface_breakdown"][0]["value"], "openai_responses")
        candidate = report["candidates"][0]
        self.assertIn("openai.instructions", [item["value"] for item in candidate["source_field_breakdown"]])
        self.assertIn("openai.input.system_or_developer", [item["value"] for item in candidate["source_field_breakdown"]])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("Developer instruction block", rendered)
        self.assertNotIn("private user request", rendered)

    def test_codex_provider_body_on_is_reported_as_codex_surface(self) -> None:
        instruction = (
            "Codex startup instruction section with repeatable workspace protocol and local privacy rules. "
            "The report may count it but must not include the raw instruction."
        )
        for _ in range(2):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                requested_model="gpt-5-codex",
                requested_model_family="gpt-5-codex",
                source_surface="openai_responses",
                endpoint="responses",
                category="tool-execution",
                workflow_phase="tool-execution",
                request_json={"model": "gpt-5-codex", "instructions": instruction, "input": "raw task text"},
            )

        report = build_instruction_dedup_opportunity_report(self.store, limit=20)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["app_family"], "codex")
        self.assertGreater(candidate["projected_saved_chars"], 0)
        self.assertNotIn("Codex startup instruction", json.dumps(report))

    def test_body_off_uses_metadata_hashes_with_blockers_without_hash_leakage(self) -> None:
        routing_extra = {
            "managed_pattern_features": {
                "local_pattern_module_families": ["prompt_role"],
                "instruction_section_hashes": ["sha256:raw-instruction-hash-must-not-leak"],
            }
        }
        for _ in range(3):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                requested_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                source_surface="openai_responses",
                endpoint="responses",
                category="chat",
                workflow_phase="planning",
                request_json=None,
                routing_extra=routing_extra,
            )

        report = build_instruction_dedup_opportunity_report(self.store, limit=20)

        self.assertEqual(report["summary"]["body_logging_off_rows"], 3)
        self.assertEqual(report["summary"]["metadata_pattern_hash_rows"], 3)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["metadata_only_rows"], 3)
        self.assertEqual(candidate["fingerprint_source"], "metadata_instruction_or_pattern_hash")
        self.assertIn("request-body-unavailable", candidate["blockers"])
        self.assertGreater(candidate["projected_saved_chars"], 0)
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-instruction-hash-must-not-leak", rendered)
        self.assertFalse(report["privacy"]["pattern_hashes_included"])

    def test_codex_app_metadata_only_rows_are_included_without_raw_ids(self) -> None:
        for idx in range(2):
            self.store.log_codex_app_event(
                id=f"codex-event-{idx}",
                created_at=utc_now(),
                direction="client_to_server",
                method="initialize",
                request_id=f"raw-request-{idx}-must-not-leak",
                thread_id=f"raw-thread-{idx}-must-not-leak",
                message_chars=1000,
                params_chars=900,
                input_items=1,
                input_text_chars=600,
                session_id="raw-codex-session-must-not-leak",
                routing_json=stable_json({"workflow_phase": "startup"}),
                crunch_json=stable_json({
                    "instruction_section_fingerprints": ["sha256:codex-instruction-hash-must-not-leak"],
                }),
                metadata_json=stable_json({"instructions_present": True}),
            )

        report = build_instruction_dedup_opportunity_report(self.store, limit=20)

        self.assertEqual(report["summary"]["scanned_codex_event_count"], 2)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["source_surface"], "codex_app_server")
        self.assertEqual(candidate["app_family"], "codex")
        self.assertIn("request-body-unavailable", candidate["blockers"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("codex-instruction-hash-must-not-leak", rendered)
        self.assertNotIn("raw-request", rendered)
        self.assertNotIn("raw-thread", rendered)
        self.assertNotIn("raw-codex-session", rendered)
        self.assertFalse(report["privacy"]["thread_ids_included"])

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        instruction = (
            "Repeated instruction section long enough to fingerprint across body-on rows. "
            "No raw instruction text should be emitted by the CLI report."
        )
        for _ in range(2):
            self._log_call(request_json={"system": instruction, "messages": []})

        result = asyncio.run(stats_instruction_dedup_opportunity(self.store, limit=10))
        self.assertEqual(result["schema"], "agentflow.instruction_dedup_opportunity.v1")

        output = io.StringIO()
        exit_code = cli.instruction_dedup_opportunity_cli(["--db", self.db_path, "--limit", "10"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.instruction_dedup_opportunity.v1")
        self.assertEqual(payload["summary"]["scanned_provider_call_count"], 2)
        self.assertNotIn("Repeated instruction section", output.getvalue())


if __name__ == "__main__":
    unittest.main()
