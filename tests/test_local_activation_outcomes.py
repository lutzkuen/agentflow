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

    def test_summary_merges_cache_and_crunch_policy_decision_reports(self):
        crunch_decision = {
            "schema": "agentflow.request_shape_crunch_policy_decision.v1",
            "status": "decided",
            "decision": "widen",
            "graduation_decision": "widen",
            "decision_id": "request-shape-crunch-policy-decision:fixture",
            "top_decision": {
                "decision_id": "request-shape-crunch-policy-decision:fixture",
                "decision": "widen",
                "graduation_decision": "widen",
                "safety_stop_state": "none",
                "source_recommended_next_action": "widen-repeated-context-crunch-canary",
                "metrics": {
                    "applied_count": 7,
                    "holdout_count": 8,
                    "observed_saved_tokens": 10738,
                    "observed_saved_usd": 0.032214,
                    "applied_retry_count": 1,
                    "holdout_retry_count": 2,
                    "fallback_count": 0,
                    "safety_stop_count": 0,
                },
                "coverage": {
                    "schema": "agentflow.request_shape_crunch_canary_coverage.v1",
                    "observed_count": 74,
                    "applied_count": 7,
                    "holdout_count": 8,
                    "skipped_count": 59,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
                "reason_codes": ["promotion-ready"],
            },
            "summary": {
                "decision": "widen",
                "graduation_decision": "widen",
                "decision_id": "request-shape-crunch-policy-decision:fixture",
                "applied_count": 7,
                "holdout_count": 8,
                "observed_saved_tokens": 10738,
                "observed_saved_usd": 0.032214,
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }
        cache_decision = {
            "schema": "agentflow.request_shape_cache_replay_policy_decision.v1",
            "status": "decided",
            "decision": "widen",
            "top_decision": {
                "decision_id": "request-shape-cache-replay-policy-decision:fixture",
                "decision": "widen",
                "recommended_next_action": "widen-openai-exact-cache-replay-policy",
                "metrics": {
                    "observed_row_count": 71,
                    "applied_count": 6,
                    "holdout_count": 5,
                    "exact_hit_count": 4,
                    "observed_hits": 4,
                    "projected_hits": 35,
                    "observed_savings_usd": 0.011,
                    "projected_savings_usd": 0.075373,
                    "retry_count": 1,
                    "fallback_count": 0,
                    "error_count": 0,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
                "coverage": {
                    "schema": "agentflow.request_shape_cache_replay_policy_decision_coverage.v1",
                    "has_applied_coverage": True,
                    "has_holdout_coverage": True,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
                "reason_codes": [],
            },
            "summary": {
                "decision": "widen",
                "applied_count": 6,
                "holdout_count": 5,
                "observed_hits": 4,
                "projected_hits": 35,
                "observed_savings_usd": 0.011,
                "projected_savings_usd": 0.075373,
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                report = build_local_activation_outcome_summary(
                    store,
                    limit=20,
                    config_dir=tmp,
                    activation_reports=[crunch_decision, cache_decision],
                )
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertEqual(report["summary"]["policy_decision_report_count"], 2)
        self.assertEqual(report["summary"]["policy_decision_families"], ["cache", "crunch"])

        by_family = {row["local_action_family"]: row for row in report["outcome_summaries"]}
        self.assertEqual(by_family["crunch"]["source_evidence_schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(by_family["crunch"]["source_decision"], "widen")
        self.assertEqual(by_family["crunch"]["graduation_decision"], "widen")
        self.assertEqual(by_family["crunch"]["applied_count"], 7)
        self.assertEqual(by_family["crunch"]["holdout_count"], 8)
        self.assertEqual(by_family["crunch"]["observed_saved_tokens"], 10738)
        self.assertEqual(by_family["crunch"]["retry_count"], 3)
        self.assertEqual(by_family["crunch"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(by_family["crunch"]["coverage"]["metadata_only"])
        self.assertTrue(by_family["crunch"]["coverage"]["aggregate_only"])

        self.assertEqual(by_family["cache"]["source_evidence_schema"], "agentflow.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(by_family["cache"]["source_decision"], "widen")
        self.assertEqual(by_family["cache"]["applied_count"], 6)
        self.assertEqual(by_family["cache"]["holdout_count"], 5)
        self.assertEqual(by_family["cache"]["observed_hits"], 4)
        self.assertEqual(by_family["cache"]["projected_hits"], 35)
        self.assertEqual(by_family["cache"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertTrue(by_family["cache"]["coverage"]["metadata_only"])
        self.assertTrue(by_family["cache"]["coverage"]["aggregate_only"])

    def test_summary_emits_keep_active_outcome_for_crunch_activation_evidence(self):
        activation_evidence = {
            "schema": "agentflow.request_shape_crunch_activation_evidence.v1",
            "status": "active-rule-evidence-observed",
            "decision": "widen",
            "graduation_decision": "widen",
            "decision_id": "request-shape-crunch-policy-decision:fixture",
            "next_action": "keep-active",
            "summary": {
                "active_rule_count": 1,
                "matching_active_rule_count": 1,
                "widened_rule_count": 1,
                "matching_widened_rule_count": 1,
                "decision": "widen",
                "graduation_decision": "widen",
                "decision_id": "request-shape-crunch-policy-decision:fixture",
                "applied_count": 107,
                "holdout_count": 40,
                "skipped_count": 280,
                "fallback_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "error_rate_delta": 0.0,
                "retry_rate_delta": 0.0,
                "fallback_rate_delta": 0.0,
                "safety_stop_state": "none",
                "observed_saved_tokens": 8606129,
                "observed_saved_usd": 25.818387,
                "policy_source": "local-manual",
                "canary_fraction": 0.3,
                "max_rollout_fraction": 0.3,
                "post_widening_status": "post-widening-active-at-max-rollout",
                "post_widening_next_action": "keep-active",
                "post_widening_reason_codes": [],
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
            },
            "rules": [
                {
                    "rank": 1,
                    "rule_ref": "rule:public",
                    "policy_source": "local-manual",
                    "decision": "widen",
                    "decision_id": "request-shape-crunch-policy-decision:fixture",
                    "source_evidence_schema": "agentflow.request_shape_crunch_policy_decision.v1",
                    "metadata_only": True,
                    "aggregate_only": True,
                }
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                report = build_local_activation_outcome_summary(
                    store,
                    limit=20,
                    config_dir=tmp,
                    activation_reports=[activation_evidence],
                )
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(report["status"], "tracked")
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertEqual(report["summary"]["policy_decision_report_count"], 1)
        self.assertEqual(report["summary"]["policy_decision_families"], ["crunch"])
        by_family = {row["local_action_family"]: row for row in report["outcome_summaries"]}
        crunch = by_family["crunch"]
        self.assertEqual(crunch["source_evidence_schema"], "agentflow.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(crunch["source_decision"], "widen")
        self.assertEqual(crunch["outcome"], "keep-active")
        self.assertEqual(crunch["next_action"], "keep-active")
        self.assertEqual(crunch["applied_count"], 107)
        self.assertEqual(crunch["holdout_count"], 40)
        self.assertEqual(crunch["skipped_count"], 280)
        self.assertEqual(crunch["observed_saved_tokens"], 8606129)
        self.assertAlmostEqual(crunch["observed_savings_usd"], 25.818387)
        self.assertEqual(crunch["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(crunch["target_local_policy_section"], "crunch.rules")
        self.assertEqual(crunch["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(crunch["post_widening_next_action"], "keep-active")
        self.assertEqual(crunch["coverage"]["applied_count"], 107)
        self.assertEqual(crunch["coverage"]["holdout_count"], 40)
        self.assertEqual(crunch["coverage"]["safety_stop_count"], 0)
        self.assertEqual(crunch["coverage"]["fallback_count"], 0)
        duplicate_suppression = crunch["duplicate_suppression"]
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertTrue(duplicate_suppression["suppresses_generic_crunch_activation_issue"])
        self.assertEqual(duplicate_suppression["matching_local_policy"], "crunch_rules")
        self.assertTrue(str(duplicate_suppression["activation_ref"]).startswith("activation:"))
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(Path(tmp).resolve()), rendered)

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
