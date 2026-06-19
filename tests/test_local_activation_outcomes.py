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
            "next_action": "promote-full-repeated-context-crunch-rule",
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
                "post_max_rollout_status": "post-max-rollout-full-rollout-ready",
                "post_max_rollout_decision": "promote-full",
                "post_max_rollout_next_action": "promote-full-repeated-context-crunch-rule",
                "post_max_rollout_reason_codes": ["max-rollout-cap-only"],
                "post_max_rollout_promotion_allowed": True,
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
        self.assertEqual(crunch["outcome"], "promote-full")
        self.assertEqual(crunch["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(crunch["applied_count"], 107)
        self.assertEqual(crunch["holdout_count"], 40)
        self.assertEqual(crunch["skipped_count"], 280)
        self.assertEqual(crunch["observed_saved_tokens"], 8606129)
        self.assertAlmostEqual(crunch["observed_savings_usd"], 25.818387)
        self.assertEqual(crunch["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(crunch["target_local_policy_section"], "crunch.rules")
        self.assertEqual(crunch["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(crunch["post_widening_next_action"], "keep-active")
        self.assertEqual(crunch["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(crunch["post_max_rollout_decision"], "promote-full")
        self.assertEqual(crunch["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(crunch["post_max_rollout_reason_codes"], ["max-rollout-cap-only"])
        self.assertTrue(crunch["post_max_rollout_promotion_allowed"])
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

    def test_summary_keeps_post_widening_crunch_active_at_max_rollout_with_coverage(self):
        activation_evidence = {
            "schema": "agentflow.request_shape_crunch_activation_evidence.v1",
            "status": "active-rule-evidence-observed",
            "decision": "widen",
            "graduation_decision": "widen",
            "decision_id": "request-shape-crunch-policy-decision:9c21d88a7c434d6f",
            "next_action": "promote-full-repeated-context-crunch-rule",
            "summary": {
                "active_rule_count": 2,
                "matching_active_rule_count": 1,
                "widened_rule_count": 1,
                "matching_widened_rule_count": 1,
                "decision": "widen",
                "graduation_decision": "widen",
                "decision_id": "request-shape-crunch-policy-decision:9c21d88a7c434d6f",
                "applied_count": 1,
                "holdout_count": 1,
                "skipped_count": 6,
                "fallback_count": 0,
                "retry_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "error_rate_delta": 0.0,
                "retry_rate_delta": 0.0,
                "fallback_rate_delta": 0.0,
                "safety_stop_state": "none",
                "observed_saved_tokens": 34356,
                "observed_saved_usd": 0.103068,
                "policy_source": "local-manual",
                "canary_fraction": 0.2,
                "holdout_fraction": 0.1,
                "max_rollout_fraction": 0.2,
                "post_widening_status": "post-widening-active-at-max-rollout",
                "post_widening_next_action": "keep-active",
                "post_widening_reason_codes": [],
                "post_max_rollout_status": "post-max-rollout-full-rollout-ready",
                "post_max_rollout_decision": "promote-full",
                "post_max_rollout_next_action": "promote-full-repeated-context-crunch-rule",
                "post_max_rollout_reason_codes": ["max-rollout-cap-only"],
                "post_max_rollout_promotion_allowed": True,
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
            },
            "rules": [
                {
                    "rank": 1,
                    "rule_ref": "local-repeated-context-crunch-canary-f274203cb090",
                    "policy_source": "local-manual",
                    "decision": "widen",
                    "decision_id": "request-shape-crunch-policy-decision:9c21d88a7c434d6f",
                    "source_evidence_schema": "agentflow.request_shape_crunch_policy_decision.v1",
                    "metadata_only": True,
                    "aggregate_only": True,
                }
            ],
            "duplicate_suppression": {
                "schema": "agentflow.request_shape_crunch_keep_active_duplicate_suppression.v1",
                "fingerprint": "activation:post-widening-test",
                "matching_local_policy": "crunch_rules",
                "reason": "repeated-context-crunch-active-at-max-rollout",
                "suppresses_generic_crunch_activation_issue": True,
                "suppresses_new_activation_issue": True,
                "target_local_policy_section": "crunch.rules",
                "target_local_rule_file": "crunch_rules.yaml",
                "metadata_only": True,
                "aggregate_only": True,
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                for _ in range(3):
                    _log_call(
                        store,
                        cost_est_usd=0.004,
                        cost_baseline_usd=0.014,
                        crunch_json=stable_json(
                            {
                                "changed": True,
                                "reason": "unrelated-live-crunch-row",
                                "saved_chars": 9000,
                                "tokens_saved_est": 2250,
                            }
                        ),
                    )
                report = build_local_activation_outcome_summary(
                    store,
                    limit=20,
                    config_dir=tmp,
                    activation_reports=[activation_evidence],
                )
            finally:
                store.conn.close()

        crunch = {row["local_action_family"]: row for row in report["outcome_summaries"]}["crunch"]
        self.assertEqual(crunch["post_widening_outcome"], "keep-active")
        self.assertEqual(crunch["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(crunch["post_widening_next_action"], "keep-active")
        self.assertEqual(crunch["coverage"]["applied_count"], 1)
        self.assertEqual(crunch["coverage"]["holdout_count"], 1)
        self.assertEqual(crunch["coverage"]["skipped_count"], 6)
        self.assertEqual(crunch["coverage"]["safety_stop_count"], 0)
        self.assertEqual(crunch["coverage"]["rollback_count"], 0)
        gate = crunch["keep_active_regression_gate"]
        self.assertEqual(gate["state"], "keep-active")
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["deterministic_next_action"], "keep-active")
        self.assertEqual(gate["reason_codes"], [])
        self.assertEqual(gate["regression_counters"]["applied_count"], 1)
        self.assertEqual(gate["regression_counters"]["holdout_count"], 1)
        self.assertEqual(gate["regression_counters"]["skipped_count"], 6)
        self.assertEqual(gate["regression_counters"]["safety_stop_count"], 0)
        self.assertEqual(gate["regression_counters"]["rollback_count"], 0)
        duplicate_suppression = crunch["duplicate_suppression"]
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertEqual(duplicate_suppression["reason"], "repeated-context-crunch-active-at-max-rollout")

        ledger_entry = report["outcome_ledger_entries"][0]
        self.assertEqual(ledger_entry["post_widening_outcome"], "keep-active")
        self.assertEqual(ledger_entry["applied_count"], 1)
        self.assertEqual(ledger_entry["holdout_count"], 1)
        self.assertEqual(ledger_entry["skipped_count"], 6)
        self.assertEqual(ledger_entry["safety_stop_count"], 0)
        self.assertEqual(ledger_entry["rollback_count"], 0)
        self.assertEqual(ledger_entry["keep_active_regression_gate"]["state"], "keep-active")
        self.assertTrue(ledger_entry["duplicate_suppression"]["suppresses_new_activation_issue"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(Path(tmp).resolve()), rendered)

    def test_summary_exports_cache_replay_warmup_lifecycle_outcome(self):
        cache_evidence = {
            "schema": "agentflow.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "reason": "cache-replay-canary-evidence-observed",
            "next_action": "review-cache-replay-canary-promotion-readiness",
            "staged_canary_count": 1,
            "summary": {
                "observed_row_count": 40,
                "applied_count": 24,
                "holdout_count": 16,
                "exact_hit_count": 0,
                "miss_count": 24,
                "bypass_count": 0,
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
                "projected_hits": 35,
                "observed_hits": 0,
                "projected_savings_usd": 0.075373,
                "observed_savings_usd": 0.0,
                "hit_observation_rate": 0.0,
                "top_blocker": None,
                "top_applied_miss_blocker": "cache-warmup-miss",
            },
            "lifecycle_counts": {
                "canary_applied_count": 24,
                "canary_holdout_count": 16,
                "exact_hit_count": 0,
                "miss_count": 24,
                "bypass_count": 0,
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "miss_reason_breakdown": [{"value": "cache-warmup-miss", "count": 24}],
            "stale_evidence": {"stale": False, "age_hours": 1.5},
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
                    activation_reports=[cache_evidence],
                )
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertEqual(report["summary"]["policy_decision_families"], ["cache"])
        by_family = {row["local_action_family"]: row for row in report["outcome_summaries"]}
        cache = by_family["cache"]
        self.assertEqual(cache["source_evidence_schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache["local_action_family"], "cache")
        self.assertEqual(cache["applied_count"], 24)
        self.assertEqual(cache["holdout_count"], 16)
        self.assertEqual(cache["miss_count"], 24)
        self.assertEqual(cache["observed_hits"], 0)
        self.assertEqual(cache["exact_hit_count"], 0)
        self.assertEqual(cache["top_miss_reason"], "cache-warmup-miss")
        self.assertEqual(cache["miss_reason_breakdown"], [{"value": "cache-warmup-miss", "count": 24}])
        self.assertEqual(cache["next_action"], "review-cache-replay-canary-promotion-readiness")
        self.assertEqual(cache["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(cache["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(cache["managed_dependency"], "optional")
        self.assertTrue(cache["coverage"]["metadata_only"])
        self.assertTrue(cache["coverage"]["aggregate_only"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

    def test_summary_exports_cache_replay_hit_recovery_lifecycle_outcome(self):
        cache_evidence = {
            "schema": "agentflow.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "reason": "cache-replay-canary-evidence-observed",
            "next_action": "review-cache-replay-canary-promotion-readiness",
            "staged_canary_count": 1,
            "summary": {
                "observed_row_count": 44,
                "applied_count": 28,
                "holdout_count": 16,
                "exact_hit_count": 4,
                "miss_count": 24,
                "projected_hits": 35,
                "observed_hits": 4,
                "projected_savings_usd": 0.075373,
                "observed_savings_usd": 0.011,
                "hit_observation_rate": 0.142857,
                "top_applied_miss_blocker": "cache-warmup-miss",
            },
            "lifecycle_counts": {
                "canary_applied_count": 28,
                "canary_holdout_count": 16,
                "exact_hit_count": 4,
                "miss_count": 24,
                "bypass_count": 0,
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "miss_reason_breakdown": [{"value": "cache-warmup-miss", "count": 24}],
            "stale_evidence": {"stale": False, "age_hours": 2.0},
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
                    activation_reports=[cache_evidence],
                )
            finally:
                store.conn.close()

        cache = {row["local_action_family"]: row for row in report["outcome_summaries"]}["cache"]
        self.assertEqual(report["egress_guard"]["status"], "passed")
        self.assertEqual(cache["source_evidence_schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache["applied_count"], 28)
        self.assertEqual(cache["holdout_count"], 16)
        self.assertEqual(cache["miss_count"], 24)
        self.assertEqual(cache["observed_hits"], 4)
        self.assertEqual(cache["exact_hit_count"], 4)
        self.assertEqual(cache["projected_hits"], 35)
        self.assertAlmostEqual(cache["observed_savings_usd"], 0.011)
        self.assertAlmostEqual(cache["projected_savings_usd"], 0.075373)
        self.assertEqual(cache["next_action"], "review-cache-replay-canary-promotion-readiness")
        self.assertEqual(cache["outcome"], "cache-replay-lifecycle-outcome-recorded")

    def test_summary_exports_full_rollout_crunch_activation_outcome(self):
        crunch_evidence = {
            "schema": "agentflow.request_shape_crunch_activation_evidence.v1",
            "status": "active-rule-evidence-observed",
            "decision": "widen",
            "graduation_decision": "widen",
            "decision_id": "request-shape-crunch-policy-decision:test",
            "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            "summary": {
                "applied_count": 107,
                "holdout_count": 40,
                "skipped_count": 280,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "fallback_count": 0,
                "observed_saved_tokens": 8_606_129,
                "observed_saved_usd": 25.818387,
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
                "active_rule_count": 1,
                "widened_rule_count": 0,
                "full_rollout_rule_count": 1,
                "full_rollout_active": True,
                "full_rollout_fraction": 1.0,
                "canary_fraction": 1.0,
                "holdout_fraction": 0.0,
                "max_rollout_fraction": 1.0,
                "post_widening_status": "post-widening-active-at-max-rollout",
                "post_widening_next_action": "keep-active",
                "post_max_rollout_status": "post-max-rollout-full-rollout-applied",
                "post_max_rollout_decision": "full-rollout-applied",
                "post_max_rollout_next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
                "post_max_rollout_promotion_allowed": False,
                "post_max_rollout_full_rollout_allowed": True,
            },
            "rules": [
                {
                    "rule_ref": "local-repeated-context-crunch-canary-test",
                    "policy_source": "local-manual",
                    "decision_id": "request-shape-crunch-policy-decision:test",
                    "source_evidence_schema": "agentflow.request_shape_crunch_activation_evidence.v1",
                }
            ],
            "duplicate_suppression": {
                "schema": "agentflow.request_shape_crunch_keep_active_duplicate_suppression.v1",
                "fingerprint": "activation:full-rollout-test",
                "matching_local_policy": "crunch_rules",
                "reason": "repeated-context-crunch-full-rollout-active",
                "suppresses_generic_crunch_activation_issue": True,
                "suppresses_new_activation_issue": True,
                "target_local_policy_section": "crunch.rules",
                "target_local_rule_file": "crunch_rules.yaml",
                "metadata_only": True,
                "aggregate_only": True,
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
                    activation_reports=[crunch_evidence],
                )
            finally:
                store.conn.close()

        crunch = {row["local_action_family"]: row for row in report["outcome_summaries"]}["crunch"]
        self.assertEqual(crunch["source_evidence_schema"], "agentflow.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(crunch["outcome"], "keep-active")
        self.assertEqual(crunch["next_action"], "measure-full-rollout-repeated-context-crunch-outcomes")
        self.assertTrue(crunch["full_rollout_active"])
        self.assertEqual(crunch["full_rollout_fraction"], 1.0)
        self.assertEqual(crunch["coverage"]["full_rollout_fraction"], 1.0)
        self.assertFalse(crunch["post_max_rollout_promotion_allowed"])
        self.assertTrue(crunch["post_max_rollout_full_rollout_allowed"])
        self.assertEqual(crunch["observed_saved_tokens"], 8_606_129)
        self.assertAlmostEqual(crunch["observed_savings_usd"], 25.818387)
        gate = crunch["keep_active_regression_gate"]
        self.assertEqual(gate["schema"], "agentflow.full_rollout_crunch_keep_active_regression_gate.v1")
        self.assertEqual(gate["state"], "keep-active")
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["deterministic_next_action"], "keep-active")
        self.assertEqual(gate["reason_codes"], [])
        self.assertEqual(gate["regression_counters"]["applied_count"], 107)
        self.assertEqual(gate["regression_counters"]["holdout_count"], 40)
        self.assertEqual(gate["regression_counters"]["rollback_count"], 0)
        self.assertEqual(gate["target_local_policy_section"], "crunch.rules")
        self.assertEqual(gate["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(report["summary"]["outcome_ledger_entry_count"], 1)
        self.assertEqual(len(report["outcome_ledger_entries"]), 1)
        ledger_entry = report["outcome_ledger_entries"][0]
        self.assertEqual(ledger_entry["schema"], "agentflow.local_activation_outcome_ledger_entry.v1")
        self.assertTrue(ledger_entry["durable_outcome_ledger_entry"])
        self.assertTrue(str(ledger_entry["ledger_ref"]).startswith("activation:"))
        self.assertEqual(ledger_entry["local_action_family"], "crunch")
        self.assertEqual(ledger_entry["source_evidence_schema"], "agentflow.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(ledger_entry["source_decision_id"], "request-shape-crunch-policy-decision:test")
        self.assertEqual(ledger_entry["outcome"], "keep-active")
        self.assertEqual(ledger_entry["next_action"], "measure-full-rollout-repeated-context-crunch-outcomes")
        self.assertEqual(ledger_entry["applied_count"], 107)
        self.assertEqual(ledger_entry["holdout_count"], 40)
        self.assertEqual(ledger_entry["skipped_count"], 280)
        self.assertEqual(ledger_entry["observed_saved_tokens"], 8_606_129)
        self.assertAlmostEqual(ledger_entry["observed_savings_usd"], 25.818387)
        self.assertEqual(ledger_entry["safety_stop_state"], "none")
        self.assertEqual(ledger_entry["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(ledger_entry["target_local_policy_section"], "crunch.rules")
        self.assertEqual(ledger_entry["active_rule_ref"], "local-repeated-context-crunch-canary-test")
        self.assertEqual(ledger_entry["active_rule_decision_id"], "request-shape-crunch-policy-decision:test")
        self.assertEqual(ledger_entry["coverage"]["applied_count"], 107)
        self.assertEqual(ledger_entry["coverage"]["holdout_count"], 40)
        self.assertEqual(ledger_entry["coverage"]["safety_stop_count"], 0)
        self.assertEqual(ledger_entry["keep_active_regression_gate"]["state"], "keep-active")
        self.assertTrue(ledger_entry["keep_active_regression_gate"]["gate_passed"])
        ledger_suppression = ledger_entry["duplicate_suppression"]
        self.assertEqual(ledger_suppression["reason"], "repeated-context-crunch-full-rollout-active")
        self.assertTrue(ledger_suppression["suppresses_new_activation_issue"])
        self.assertTrue(ledger_suppression["suppresses_generic_crunch_activation_issue"])
        self.assertTrue(ledger_entry["privacy"]["metadata_only"])
        self.assertTrue(ledger_entry["privacy"]["aggregate_only"])
        self.assertFalse(ledger_entry["privacy"]["raw_prompts_included"])
        self.assertFalse(ledger_entry["privacy"]["provider_bodies_included"])

    def test_summary_exports_full_rollout_crunch_rollback_gate(self):
        crunch_evidence = {
            "schema": "agentflow.request_shape_crunch_activation_evidence.v1",
            "status": "active-rule-evidence-observed",
            "decision": "widen",
            "graduation_decision": "widen",
            "decision_id": "request-shape-crunch-policy-decision:rollback-test",
            "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            "summary": {
                "applied_count": 107,
                "holdout_count": 40,
                "skipped_count": 280,
                "safety_stop_count": 0,
                "rollback_count": 1,
                "fallback_count": 1,
                "retry_count": 0,
                "fallback_rate_delta": 0.01,
                "observed_saved_tokens": 8_606_129,
                "observed_saved_usd": 25.818387,
                "target_local_rule_file": "crunch_rules.yaml",
                "target_local_policy_section": "crunch.rules",
                "active_rule_count": 1,
                "full_rollout_active": True,
                "full_rollout_fraction": 1.0,
                "post_widening_status": "post-widening-active-at-max-rollout",
                "post_widening_next_action": "keep-active",
                "post_max_rollout_status": "post-max-rollout-full-rollout-applied",
                "post_max_rollout_decision": "full-rollout-applied",
                "post_max_rollout_next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
                "stale_evidence": {
                    "stale": False,
                    "age_hours": 2.5,
                    "status": "fresh",
                    "reason": "recent-local-outcome-window",
                },
            },
            "rules": [
                {
                    "rule_ref": "local-repeated-context-crunch-canary-rollback-test",
                    "policy_source": "local-manual",
                    "decision_id": "request-shape-crunch-policy-decision:rollback-test",
                    "source_evidence_schema": "agentflow.request_shape_crunch_activation_evidence.v1",
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
                    activation_reports=[crunch_evidence],
                )
            finally:
                store.conn.close()

        crunch = {row["local_action_family"]: row for row in report["outcome_summaries"]}["crunch"]
        gate = crunch["keep_active_regression_gate"]
        self.assertEqual(gate["state"], "rollback-required")
        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["deterministic_next_action"], "rollback-full-rollout-repeated-context-crunch-rule")
        self.assertIn("rollback-observed", gate["reason_codes"])
        self.assertIn("fallback-observed", gate["reason_codes"])
        self.assertIn("fallback-rate-regression", gate["reason_codes"])
        self.assertEqual(gate["regression_counters"]["decision_age_hours"], 2.5)
        self.assertFalse(gate["regression_counters"]["stale_evidence"]["stale"])
        ledger_entry = report["outcome_ledger_entries"][0]
        self.assertEqual(ledger_entry["keep_active_regression_gate"]["state"], "rollback-required")
        self.assertEqual(ledger_entry["keep_active_regression_gate"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertFalse(ledger_entry["privacy"]["file_paths_included"])
        self.assertFalse(ledger_entry["privacy"]["request_ids_included"])
        self.assertFalse(ledger_entry["privacy"]["session_ids_included"])
        self.assertFalse(ledger_entry["privacy"]["cache_keys_included"])
        self.assertFalse(ledger_entry["privacy"]["individual_candidate_ids_included"])
        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt",
            "provider body",
            str(Path(tmp).resolve()),
            "request-secret",
            "session-secret",
            "cache-key-secret",
            "candidate-secret",
        ):
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
