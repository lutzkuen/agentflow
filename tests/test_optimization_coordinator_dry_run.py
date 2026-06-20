from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.optimization_coordinator_dry_run import build_optimization_coordinator_dry_run
from tokenclaw.store import SQLiteStore, stable_json, utc_now


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

    def _unsafe_rollout_bundle(self) -> dict[str, object]:
        bundle = self._rollout_bundle()
        bundle["privacy_summary"] = {
            "metadata_only": False,
            "raw_prompts_returned": True,
            "raw_responses_returned": True,
            "provider_bodies_returned": True,
            "request_ids_returned": True,
            "tenant_ids_returned": True,
            "cache_keys_returned": True,
            "file_paths_returned": True,
        }
        actions = bundle["actions"]
        assert isinstance(actions, list)
        action = actions[0]
        assert isinstance(action, dict)
        action["reason_codes"] = [
            "prompt leak raw-dry-run-prompt-secret",
            "messages carried raw-dry-run-response-secret",
        ]
        return bundle

    def _cache_rollout_bundle_without_holdout_or_freshness(self) -> dict[str, object]:
        bundle = self._rollout_bundle()
        actions = bundle["actions"]
        assert isinstance(actions, list)
        action = actions[0]
        assert isinstance(action, dict)
        action.update(
            {
                "action_family": "cache",
                "candidate_family": "cache-replay-rule",
                "target_candidate_id": "cache-candidate",
                "policy_section": "cache",
                "reason_codes": ["request body missing holdout evidence"],
                "evidence_summary": {
                    "local_eval_verdict": {
                        "reason_codes": ["missing-holdout", "stale-evidence"],
                    }
                },
            }
        )
        action.pop("safe_invalidation_evidence", None)
        return bundle

    def test_dry_run_reports_coordinator_counts_conflicts_and_projected_savings(self) -> None:
        self._log_call(
            crunch_meta={
                "terminal_output_compaction": {
                    "status": "eligible",
                    "candidate_id": "terminal-candidate",
                    "projected_saved_usd": 0.025,
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
        suppression_bucket = report["suppression_opportunity_buckets"][0]
        self.assertEqual(suppression_bucket["selected_family"], "routing")
        self.assertEqual(suppression_bucket["suppressed_family"], "terminal_output_compaction")
        self.assertEqual(suppression_bucket["projected_savings_lost_usd"], 0.025)
        self.assertEqual(suppression_bucket["actionability"], "actionable")
        self.assertEqual(suppression_bucket["next_action"], "run-suppressed-crunch-eval")
        self.assertEqual(report["top_suppression_next_action"], "run-suppressed-crunch-eval")
        self.assertEqual(report["projected_savings_usd_est"], 0.04)
        self.assertEqual(report["status_counts"][0]["status_code"], 429)
        self.assertEqual(report["rows_with_errors"], 1)
        self.assertEqual(report["retry_count_buckets"][0]["retry_count_bucket"], "gte_3")
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["policy_files_changed"])
        self._assert_private(report)

    def test_suppression_buckets_rank_positive_zero_and_unknown_savings(self) -> None:
        self._log_call(
            crunch_meta={
                "terminal_output_compaction": {
                    "status": "eligible",
                    "candidate_id": "terminal-positive-candidate",
                    "projected_saved_usd": 0.03,
                },
            }
        )
        self._log_call(
            crunch_meta={
                "codex_repeated_scaffolding": {
                    "status": "eligible",
                    "candidate_id": "scaffold-zero-candidate",
                    "projected_saved_usd": 0.0,
                },
            }
        )
        self._log_call(
            routing_meta={
                "provider": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "text_chars": 12000,
                "enabled": True,
                "requested_model": "gpt-5.4",
                "routed_model": "gpt-5.4-mini",
                "reason": "selected-canary",
                "managed_pattern_features": {
                    "local_pattern_module_families": ["cacheability"],
                    "reason_codes": ["pattern-family-observed"],
                },
            },
        )

        report = build_optimization_coordinator_dry_run(
            self.store,
            limit=10,
            local_salt="coordinator-test",
        )

        buckets = {
            row["suppressed_family"]: row
            for row in report["suppression_opportunity_buckets"]
        }
        self.assertEqual(report["suppression_opportunity_buckets"][0]["suppressed_family"], "terminal_output_compaction")
        self.assertEqual(buckets["terminal_output_compaction"]["projected_savings_lost_usd"], 0.03)
        self.assertEqual(buckets["terminal_output_compaction"]["actionability"], "actionable")
        self.assertEqual(buckets["repeated_scaffold_crunch"]["projected_savings_lost_usd"], 0.0)
        self.assertEqual(buckets["repeated_scaffold_crunch"]["actionability"], "no-op")
        self.assertEqual(buckets["repeated_scaffold_crunch"]["no_op_reason"], "no-positive-suppressed-savings")
        self.assertEqual(buckets["pattern_crunch:cacheability"]["savings_status"], "unknown")
        self.assertEqual(buckets["pattern_crunch:cacheability"]["actionability"], "no-op")
        self.assertEqual(buckets["pattern_crunch:cacheability"]["no_op_reason"], "unknown-suppressed-savings")
        self.assertEqual(buckets["pattern_crunch:cacheability"]["next_action"], "measure-pattern-crunch-savings")
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

    def test_unsafe_rollout_payload_is_summarized_but_not_used(self) -> None:
        self._log_call()

        report = build_optimization_coordinator_dry_run(
            self.store,
            rollout_actions=self._unsafe_rollout_bundle(),
            limit=10,
            local_salt="local-salt-dry-run-secret",
        )

        self.assertEqual(report["managed_rollout_actions"]["unsafe_privacy_flag_count"], 7)
        self.assertFalse(report["managed_rollout_actions"]["actions_included"])
        self.assertFalse(report["managed_rollout_actions"]["raw_rollout_payload_included"])
        self.assertEqual(report["rows_with_rollout_action_candidates"], 0)
        self._assert_private(report)

    def test_cache_rollout_with_stale_or_missing_evidence_is_suppressed(self) -> None:
        self._log_call()

        report = build_optimization_coordinator_dry_run(
            self.store,
            rollout_actions=self._cache_rollout_bundle_without_holdout_or_freshness(),
            limit=10,
            local_salt="coordinator-test",
        )

        self.assertEqual(report["rows_with_rollout_action_candidates"], 1)
        reasons = {(row["family"], row["reason"]): row["count"] for row in report["top_suppression_reason_codes"]}
        self.assertEqual(reasons[("cache_replay", "stale-evidence")], 1)
        self.assertNotIn("request body missing holdout evidence", json.dumps(report, sort_keys=True))
        self._assert_private(report)

    def test_cli_rejects_corrupt_rollout_json_without_raw_payload(self) -> None:
        output = io.StringIO()
        code = cli.optimization_coordinator_dry_run_cli(
            ["--db", self.db_path, "-"],
            stdin=io.StringIO('{"prompt":"raw-dry-run-prompt-secret",'),
            stdout=output,
        )
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["type"], "read_failed")
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        self.assertNotIn("raw-dry-run-prompt-secret", json.dumps(payload, sort_keys=True))

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
