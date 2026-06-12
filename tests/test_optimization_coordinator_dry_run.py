from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.optimization_coordinator_dry_run import build_optimization_coordinator_dry_run
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


FORBIDDEN_VALUES = (
    "raw-dry-run-prompt-secret",
    "raw-dry-run-response-secret",
    "raw-dry-run-session-secret",
    "req-dry-run-secret",
    "cache-key-dry-run-secret",
    "terminal raw output secret",
    "tool payload dry-run secret",
    "/home/lutz/private/dry_run_secret.py",
    "local-salt-dry-run-secret",
    "rollout-action-secret",
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


class OptimizationCoordinatorDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
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
        requested_model: str = "gpt-5.4",
        routed_model: str = "gpt-5.4-mini",
        routing_meta: dict[str, object] | None = None,
        crunch_meta: dict[str, object] | None = None,
        cache_meta: dict[str, object] | None = None,
        status_code: int = 200,
        retry_count: int = 0,
    ) -> None:
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path=f"/v1/{endpoint}",
            requested_model=requested_model,
            routed_model=routed_model,
            stream=0,
            cache_hit=0,
            status_code=status_code,
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
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "text_chars": 12000,
                    "enabled": True,
                    "requested_model": requested_model,
                    "routed_model": routed_model,
                    "reason": "selected-canary",
                }
            ),
            cache_json=stable_json(cache_meta or {"status": "miss", "reason": "exact-miss"}),
            error="raw-dry-run-response-secret" if status_code >= 400 else None,
            request_json=stable_json(
                {
                    "request_id": "req-dry-run-secret",
                    "messages": [{"content": "raw-dry-run-prompt-secret"}],
                    "cache_key": "cache-key-dry-run-secret",
                    "file_path": "/home/lutz/private/dry_run_secret.py",
                }
            ),
            response_json=stable_json({"content": "raw-dry-run-response-secret"}),
            session_id="raw-dry-run-session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family="gpt-5" if provider == "openai" else "sonnet",
            routed_model_family="gpt-5-mini" if provider == "openai" else "sonnet",
        )

    def _rollout_bundle(self) -> dict[str, object]:
        return {
            "schema": "agentflow.optimization_rollout_actions.v1",
            "local_executor_compatibility": {"compatible": True},
            "privacy_summary": {
                "metadata_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
            },
            "actions": [
                {
                    "schema": "agentflow.optimization_rollout_action.v1",
                    "action_id": "rollout-action-secret",
                    "action_type": "widen",
                    "target_candidate_id": "routing-candidate",
                    "action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "policy_section": "routing",
                    "source_surface": "openai_responses",
                    "provider_endpoint": "responses",
                    "evidence_summary": {"local_eval_verdict": {"reason_codes": ["local-eval-widen"]}},
                    "action": {
                        "target_rule_id": "routing-rule",
                        "proposed_edit": {"prompt": "raw-dry-run-prompt-secret"},
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "raw_payloads_returned": False,
                        "raw_prompts_returned": False,
                        "raw_responses_returned": False,
                        "provider_bodies_returned": False,
                        "request_ids_returned": False,
                        "tenant_ids_returned": False,
                        "cache_keys_returned": False,
                        "file_paths_returned": False,
                    },
                }
            ],
            "omitted_actions": [],
        }

    def test_dry_run_reports_coordinator_counts_conflicts_and_projected_savings(self) -> None:
        self._log_call(
            crunch_meta={
                "terminal_output_compaction": {
                    "status": "eligible",
                    "candidate_id": "terminal-candidate",
                    "terminal_line": "terminal raw output secret",
                },
            },
            cache_meta={"status": "miss", "reason": "exact-miss"},
            status_code=429,
            retry_count=3,
        )

        report = build_optimization_coordinator_dry_run(
            self.store,
            rollout_actions=self._rollout_bundle(),
            limit=10,
            local_salt="local-salt-dry-run-secret",
            examples=5,
        )

        self.assertEqual(report["schema"], "agentflow.optimization_coordinator_dry_run.v1")
        self.assertTrue(report["ok"])
        self.assertEqual(report["sampled_call_count"], 1)
        self.assertEqual(report["rows_with_rollout_action_candidates"], 1)
        selected = {row["family"]: row["count"] for row in report["selected_family_counts"]}
        self.assertEqual(selected["routing"], 1)
        suppressed = {row["family"]: row["count"] for row in report["suppressed_family_counts"]}
        self.assertEqual(suppressed["terminal_output_compaction"], 1)
        self.assertEqual(report["conflict_buckets"][0]["reason"], "conflicts-with-selected-family")
        self.assertEqual(report["projected_savings_usd_est"], 0.04)
        self.assertEqual(report["status_counts"][0]["status_code"], 429)
        self.assertEqual(report["rows_with_errors"], 1)
        self.assertEqual(report["retry_count_buckets"][0]["retry_count_bucket"], "gte_3")
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["policy_files_changed"])
        self._assert_private(report)

    def test_filters_provider_and_source_surface(self) -> None:
        self._log_call()
        self._log_call(
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            routing_meta={
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "text_chars": 12000,
            },
            crunch_meta={"terminal_output_compaction": {"status": "eligible", "candidate_id": "terminal-candidate"}},
        )

        report = build_optimization_coordinator_dry_run(
            self.store,
            limit=10,
            provider="anthropic",
            source_surface="anthropic_messages",
            local_salt="coordinator-test",
        )

        self.assertEqual(report["sampled_call_count"], 1)
        selected = {row["family"]: row["count"] for row in report["selected_family_counts"]}
        self.assertEqual(selected["terminal_output_compaction"], 1)
        self.assertEqual(report["filters"]["provider"], "anthropic")
        self._assert_private(report)

    def test_holdout_and_noop_counts_are_reported(self) -> None:
        self._log_call(
            routed_model="gpt-5.4",
            routing_meta={
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4",
            },
        )
        self._log_call()

        report = build_optimization_coordinator_dry_run(
            self.store,
            limit=10,
            local_salt="coordinator-test",
            holdout_fraction=1.0,
            canary_fraction=0.0,
        )

        self.assertEqual(report["sampled_call_count"], 2)
        selected = {row["family"]: row["count"] for row in report["selected_family_counts"]}
        self.assertEqual(selected["none"], 2)
        self.assertGreaterEqual(report["holdout_count"], 1)
        self.assertGreaterEqual(report["noop_count"], 1)
        self._assert_private(report)

    def test_cli_emits_dry_run_report_with_optional_rollout_bundle(self) -> None:
        self._log_call()
        output = io.StringIO()
        code = cli.optimization_coordinator_dry_run_cli(
            ["--db", self.db_path, "--limit", "10", "--local-salt", "local-salt-dry-run-secret", "-"],
            stdin=io.StringIO(json.dumps(self._rollout_bundle())),
            stdout=output,
        )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.optimization_coordinator_dry_run.v1")
        self.assertEqual(payload["managed_rollout_actions"]["action_count"], 1)
        self._assert_private(payload)


if __name__ == "__main__":
    unittest.main()
