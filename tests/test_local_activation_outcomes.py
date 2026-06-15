import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import uuid

from agentflow_proxy.cli_commands import optimization_reports as optimization_reports_cli
from agentflow_proxy.local_activation_outcomes import build_local_activation_outcome_summary
from agentflow_proxy.store import Store, stable_json, utc_now


def _log_call(store: Store, **overrides):
    base = {
        "id": str(uuid.uuid4()),
        "created_at": utc_now(),
        "path": "/v1/messages",
        "provider": "anthropic",
        "source_surface": "anthropic_messages",
        "endpoint": "messages",
        "requested_model": "claude-sonnet-4-6",
        "routed_model": "claude-sonnet-4-6",
        "stream": 0,
        "cache_hit": 0,
        "status_code": 200,
        "latency_ms": 100,
        "input_tokens_est": 1000,
        "output_tokens_est": 100,
        "actual_input_tokens": 1000,
        "actual_output_tokens": 100,
        "cost_est_usd": 0.002,
        "cost_baseline_usd": 0.002,
        "retry_count": 0,
        "category": "chat",
        "routing_json": stable_json({"status": "pass-through", "reason": "keep-requested-model"}),
        "crunch_json": stable_json({"changed": False, "reason": "below-threshold"}),
        "cache_json": stable_json({"status": "miss", "reason": "exact-cache-miss"}),
    }
    base.update(overrides)
    store.log_call(**base)


class LocalActivationOutcomeSummaryTests(unittest.TestCase):
    def test_summary_exports_feature_only_routing_crunch_and_cache_outcomes(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    path="/v1/responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4-mini",
                    cost_est_usd=0.002,
                    cost_baseline_usd=0.01,
                    retry_count=1,
                    routing_json=stable_json(
                        {
                            "status": "applied",
                            "reason": "openai-routing-canary-applied",
                            "openai_routing_canary": {"status": "applied"},
                        }
                    ),
                )
                _log_call(
                    store,
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.006,
                    crunch_json=stable_json(
                        {
                            "changed": True,
                            "reason": "repeated-context-crunch-canary",
                            "saved_chars": 8000,
                            "tokens_saved_est": 2000,
                            "projected_savings_usd": 0.003,
                        }
                    ),
                )
                _log_call(
                    store,
                    status_code=500,
                    cache_json=stable_json(
                        {
                            "status": "holdout",
                            "reason": "canary_holdout",
                            "cache_replay_canary": {"status": "holdout"},
                        }
                    ),
                )

                report = build_local_activation_outcome_summary(store, limit=20, config_dir=tmp)
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(report["status"], "tracked")
        self.assertFalse(report["provider_calls_made"])
        self.assertFalse(report["managed_server_calls_made"])
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertTrue(report["privacy"]["feature_only"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["tenant_ids_included"])

        by_family = {row["local_action_family"]: row for row in report["outcome_summaries"]}
        self.assertEqual(set(by_family), {"routing", "crunch", "cache"})
        self.assertEqual(by_family["routing"]["applied_count"], 1)
        self.assertEqual(by_family["routing"]["retry_count"], 1)
        self.assertGreater(by_family["routing"]["observed_savings_usd"], 0)
        self.assertEqual(by_family["crunch"]["applied_count"], 1)
        self.assertEqual(by_family["crunch"]["projected_saved_tokens"], 2000)
        self.assertEqual(by_family["crunch"]["projected_saved_chars"], 8000)
        self.assertEqual(by_family["cache"]["holdout_count"], 1)
        self.assertEqual(by_family["cache"]["error_count"], 1)
        for row in by_family.values():
            self.assertEqual(row["local_file_backed_representation"]["rule_file"], f"{row['policy_section']}_rules.yaml")
            self.assertFalse(row["local_file_backed_representation"]["path_included"])
            self.assertFalse(row["local_file_backed_representation"]["policy_file_contents_included"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in ("raw prompt", "req-secret", "session-secret", "cache-key-secret", str(Path(tmp).resolve())):
            self.assertNotIn(forbidden, rendered)

    def test_cli_emits_local_activation_outcome_summary(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                _log_call(
                    store,
                    cache_hit=1,
                    cache_json=stable_json({"status": "hit", "reason": "exact-cache-hit"}),
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = optimization_reports_cli.local_activation_outcome_summary_cli(
                ["--db", db_path, "--limit", "5", "--config-dir", tmp],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(payload["egress_guard"]["status"], "passed")
        self.assertEqual(payload["summary"]["local_action_family_count"], 3)

