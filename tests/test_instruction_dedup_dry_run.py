from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.instruction_dedup_dry_run import build_instruction_dedup_dry_run
from tokenclaw.store import SQLiteStore, stable_json, utc_now


FORBIDDEN_VALUES = (
    "private instruction dry run secret",
    "private user dry run secret",
    "tool payload dry run secret",
    "thinking dry run secret",
    "raw-response-dry-run-secret",
    "raw-session-dry-run-secret",
    "raw-request-dry-run-secret",
    "local-salt-dry-run-secret",
    "sha256:raw-instruction-hash-must-not-leak",
    "raw category dry run secret",
    "raw-policy-rule-secret",
    "raw-candidate-dry-run-secret",
    "raw coordinator dry run secret",
    "private replacement notice dry run secret",
)


class InstructionDedupDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _policy(self, **overrides: object) -> dict[str, object]:
        policy: dict[str, object] = {
            "enabled": True,
            "policy_source": "local-manual",
            "rules": [],
            "source_surfaces": [
                "anthropic_messages",
                "openai_responses",
                "openai_chat_completions",
                "codex_turn",
            ],
            "categories": [],
            "workflow_phases": [],
            "min_section_chars": 80,
            "min_repeated_count": 2,
            "keep_recent_sections": 1,
            "replacement_notice": "[repeated instruction section omitted by AgentFlow]",
            "max_replacements": 4,
            "block_tool_protocol": True,
            "block_tool_payloads": True,
            "block_responses": True,
            "block_thinking": True,
            "canary": {
                "enabled": True,
                "fraction": 1.0,
                "holdout_fraction": 0.0,
                "salt": "test-policy-salt",
                "unit": "instruction_section_fingerprint",
            },
            "safety_stop": {
                "enabled": True,
                "min_outcome_samples": 5,
                "window": 500,
                "max_error_rate": 0.1,
                "max_retry_rate": 0.25,
                "max_negative_savings_rate": 0.25,
                "max_error_rate_delta": 0.05,
            },
        }
        policy.update(overrides)
        return policy

    def _log_call(
        self,
        *,
        provider: str = "anthropic",
        path: str = "/v1/messages",
        requested_model: str = "claude-sonnet-4-6",
        requested_model_family: str = "sonnet",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        category: str = "chat",
        workflow_phase: str = "planning",
        request_json: dict[str, object] | None = None,
        routing_extra: dict[str, object] | None = None,
        status_code: int = 200,
        text_chars: int = 16_000,
    ) -> None:
        routing = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_chars": text_chars,
        }
        if routing_extra:
            routing.update(routing_extra)
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=requested_model,
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=100,
            input_tokens_est=text_chars // 4,
            output_tokens_est=50,
            actual_input_tokens=text_chars // 4,
            actual_output_tokens=50,
            cost_est_usd=0.012,
            cost_baseline_usd=0.012,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error=None if status_code < 400 else "upstream error secret must not leak",
            request_json=stable_json(request_json) if request_json is not None else None,
            response_json=stable_json({"text": "raw-response-dry-run-secret"}),
            session_id="raw-session-dry-run-secret",
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

    def _assert_private(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for value in FORBIDDEN_VALUES:
            self.assertNotIn(value, rendered)

    def test_body_on_candidates_plan_across_anthropic_openai_and_codex_surfaces(self) -> None:
        instruction = (
            "Repeated private instruction dry run secret with stable coding-agent operating rules. "
            "The dry run may plan omission but must never emit the source instruction text."
        )
        for _ in range(2):
            self._log_call(
                request_json={
                    "model": "claude-sonnet-4-6",
                    "system": instruction,
                    "messages": [{"role": "user", "content": "private user dry run secret"}],
                }
            )
            self._log_call(
                provider="openai",
                path="/v1/responses",
                requested_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                source_surface="openai_responses",
                endpoint="responses",
                request_json={
                    "model": "gpt-5.4-mini",
                    "instructions": instruction,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": "private user dry run secret"}]}],
                },
            )
            self._log_call(
                provider="openai",
                path="/v1/responses",
                requested_model="gpt-5-codex",
                requested_model_family="gpt-5-codex",
                source_surface="codex_turn",
                endpoint="responses",
                category="tool-execution",
                workflow_phase="tool-execution",
                request_json={"model": "gpt-5-codex", "instructions": instruction, "input": "private user dry run secret"},
            )

        report = build_instruction_dedup_dry_run(
            self.store,
            limit=20,
            examples=20,
            policy=self._policy(),
            policy_source="local-manual",
            rule_path=None,
            local_salt="local-salt-dry-run-secret",
        )

        self.assertEqual(report["schema"], "agentflow.instruction_dedup_dry_run.v1")
        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["summary"]["actionable_plan_count"], 6)
        surfaces = {plan["source_surface"] for plan in report["plans"]}
        self.assertIn("anthropic_messages", surfaces)
        self.assertIn("openai_responses", surfaces)
        self.assertIn("codex_turn", surfaces)
        for plan in report["plans"]:
            if plan["status"] != "actionable":
                continue
            self.assertEqual(plan["canary"]["cohort"], "canary")
            self.assertEqual(plan["selected_rule_id"], "instruction-section-dedup-policy")
            self.assertGreater(plan["counts"]["saved_tokens_est"], 0)
            self.assertFalse(plan["instruction_section"]["fingerprint_included"])
            self.assertTrue(plan["replacement_preview"]["redacted"])
            self.assertFalse(plan["mutation"]["request_body_changed"])
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["managed_server_calls_made"])
        self.assertFalse(report["privacy"]["instruction_section_fingerprints_included"])
        self._assert_private(report)

    def test_body_off_rows_are_blocked_without_hash_or_identifier_leakage(self) -> None:
        self._log_call(
            provider="openai",
            path="/v1/responses",
            requested_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            source_surface="openai_responses",
            endpoint="responses",
            request_json=None,
            routing_extra={
                "instruction_section_hashes": ["sha256:raw-instruction-hash-must-not-leak"],
            },
        )
        self.store.log_codex_app_event(
            id="raw-request-dry-run-secret",
            created_at=utc_now(),
            direction="client_to_server",
            method="initialize",
            request_id="raw-request-dry-run-secret",
            thread_id="raw-thread-dry-run-secret",
            message_chars=1000,
            params_chars=800,
            input_items=1,
            input_text_chars=600,
            session_id="raw-session-dry-run-secret",
            routing_json=stable_json({"category": "startup", "workflow_phase": "startup"}),
            crunch_json=stable_json({"instruction_section_hashes": ["sha256:raw-instruction-hash-must-not-leak"]}),
            metadata_json=stable_json({"instructions_present": True}),
        )

        report = build_instruction_dedup_dry_run(self.store, limit=20, examples=20, policy=self._policy(), rule_path=None)

        bodyless = [plan for plan in report["plans"] if "request-body-unavailable" in plan["blockers"]]
        self.assertGreaterEqual(len(bodyless), 2)
        self.assertTrue(all(plan["status"] == "blocked" for plan in bodyless))
        self.assertIn("request-body-unavailable", {row["reason"] for row in report["blocker_reason_breakdown"]})
        self._assert_private(report)

    def test_tool_and_thinking_content_block_instruction_plan_without_leaking_payloads(self) -> None:
        instruction = (
            "Repeated private instruction dry run secret for a risky tool and thinking request. "
            "The plan must be blocked while preserving protocol-bearing content."
        )
        for _ in range(2):
            self._log_call(
                category="tool-result",
                workflow_phase="tool-execution",
                request_json={
                    "model": "claude-sonnet-4-6",
                    "system": instruction,
                    "messages": [
                        {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking dry run secret"}]},
                        {"role": "user", "content": [{"type": "tool_result", "content": "tool payload dry run secret"}]},
                    ],
                },
            )

        report = build_instruction_dedup_dry_run(self.store, limit=20, examples=20, policy=self._policy(), rule_path=None)

        plan = report["plans"][0]
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("tool-protocol-risk", plan["blockers"])
        self.assertIn("thinking-content-risk", plan["blockers"])
        self.assertFalse(plan["privacy"]["tool_payloads_included"])
        self._assert_private(report)

    def test_coordinator_conflicts_and_holdouts_are_surfaced_as_blockers(self) -> None:
        instruction = (
            "Repeated private instruction dry run secret with a coordinator-selected routing action. "
            "Instruction dedup must surface the conflict before runtime mutation."
        )
        for _ in range(2):
            self._log_call(
                request_json={"model": "claude-sonnet-4-6", "system": instruction, "messages": []},
                routing_extra={
                    "optimization_coordinator": {
                        "schema": "agentflow.optimization_coordinator.v1",
                        "selected_family": "routing",
                        "reason_codes": [],
                    }
                },
            )

        report = build_instruction_dedup_dry_run(
            self.store,
            limit=20,
            examples=20,
            policy=self._policy(canary={"enabled": True, "fraction": 0.0, "holdout_fraction": 1.0}),
            rule_path=None,
        )

        plan = report["plans"][0]
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("coordinator-conflict", plan["blockers"])
        self.assertIn("instruction-dedup-holdout", plan["blockers"])
        self.assertEqual(plan["coordinator_compatibility"]["status"], "conflict")
        self.assertEqual(plan["canary"]["cohort"], "holdout")
        self._assert_private(report)

    def test_invalid_canary_and_raw_policy_metadata_fail_closed_without_leaking(self) -> None:
        instruction = (
            "Repeated private instruction dry run secret with unsafe policy metadata. "
            "Invalid rollout math must block instead of silently selecting a cohort."
        )
        for _ in range(2):
            self._log_call(
                category="raw category dry run secret",
                workflow_phase="/home/lutz/private/raw-phase-secret.py",
                request_json={"model": "claude-sonnet-4-6", "system": instruction, "messages": []},
                routing_extra={
                    "optimization_coordinator": {
                        "selected_family": "cache_replay",
                        "reason_codes": ["raw coordinator dry run secret"],
                    }
                },
            )

        report = build_instruction_dedup_dry_run(
            self.store,
            limit=20,
            examples=20,
            policy=self._policy(
                replacement_notice="private replacement notice dry run secret",
                rules=[
                    {
                        "id": "raw-policy-rule-secret",
                        "candidate_id": "raw-candidate-dry-run-secret",
                        "enabled": True,
                        "policy_source": "managed-recommended",
                        "max_replacements": 2,
                        "min_section_chars": 80,
                        "min_repeated_count": 2,
                        "canary": {
                            "enabled": True,
                            "fraction": 0.8,
                            "holdout_fraction": 0.4,
                            "salt": "local-salt-dry-run-secret",
                        },
                    }
                ],
            ),
            rule_path="/home/lutz/private/crunch_rules.yaml",
            local_salt="local-salt-dry-run-secret",
        )

        plan = report["plans"][0]
        self.assertEqual(plan["status"], "blocked")
        self.assertIn("invalid-canary-configuration", plan["blockers"])
        self.assertIn("coordinator-conflict", plan["blockers"])
        self.assertFalse(plan["canary"]["valid"])
        self.assertIn("invalid-canary-fraction-sum", plan["canary"]["validation_errors"])
        self.assertEqual(plan["selected_rule_id"], "instruction-section-dedup-policy")
        self.assertTrue(str(plan["candidate_id"]).startswith("instruction-dedup-candidate:"))
        self.assertEqual(plan["category"], "unknown")
        self.assertEqual(plan["workflow_phase"], "unknown")
        self.assertIn("sanitized-reason", plan["coordinator_compatibility"]["reason_codes"])
        self.assertFalse(report["policy"]["file"]["path_included"])
        self._assert_private(report)

    def test_default_cli_emits_dry_run_schema_without_raw_text(self) -> None:
        instruction = (
            "Repeated private instruction dry run secret for CLI output privacy. "
            "No raw instruction content should be emitted by the dry-run command."
        )
        for _ in range(2):
            self._log_call(request_json={"model": "claude-sonnet-4-6", "system": instruction, "messages": []})

        output = io.StringIO()
        code = cli.instruction_dedup_dry_run_cli(
            ["--db", self.db_path, "--limit", "10", "--examples", "10", "--local-salt", "local-salt-dry-run-secret"],
            stdout=output,
        )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.instruction_dedup_dry_run.v1")
        self.assertGreaterEqual(payload["summary"]["plan_count"], 1)
        self.assertFalse(payload["summary"]["provider_calls_made"])
        self._assert_private(payload)


if __name__ == "__main__":
    unittest.main()
