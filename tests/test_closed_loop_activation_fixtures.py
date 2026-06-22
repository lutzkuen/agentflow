import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import uuid

import yaml

from tokenclaw.local_activation_executor import (
    apply_local_activation_executor_bundle,
    build_local_activation_executor_bundle,
)
from tokenclaw.local_activation_outcomes import build_local_activation_outcome_summary
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json, utc_now


FORBIDDEN_SENTINELS = (
    "raw prompt must not leak",
    "provider body must not leak",
    "req-closed-loop-secret",
    "session-closed-loop-secret",
    "cache-key-closed-loop-secret",
    "/tmp/private-closed-loop",
)

FALSE_PRIVACY_FLAGS = (
    "raw_prompts_included",
    "raw_messages_included",
    "provider_bodies_included",
    "raw_request_bodies_included",
    "raw_response_bodies_included",
    "raw_provider_bodies_included",
    "raw_responses_included",
    "tool_payloads_included",
    "cache_keys_included",
    "request_ids_included",
    "session_ids_included",
    "tenant_ids_included",
    "file_paths_included",
    "absolute_paths_included",
    "policy_file_contents_included",
    "provider_calls_made",
)


def _assert_metadata_only(testcase: unittest.TestCase, payload: object, *extra_forbidden: str) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_SENTINELS + tuple(extra_forbidden):
        testcase.assertNotIn(forbidden, rendered)
    testcase.assertNotIn('"managed_enforced": true', rendered)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "metadata_only" in value:
                testcase.assertTrue(value["metadata_only"], value)
            if "aggregate_only" in value:
                testcase.assertTrue(value["aggregate_only"], value)
            for flag in FALSE_PRIVACY_FLAGS:
                if flag in value:
                    testcase.assertFalse(value[flag], (flag, value))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)


def _log_call(store: Store, **overrides: object) -> None:
    base = {
        "id": str(uuid.uuid4()),
        "created_at": utc_now(),
        "path": "/v1/responses",
        "provider": "openai",
        "source_surface": "openai_responses",
        "endpoint": "responses",
        "requested_model": "gpt-5.4",
        "routed_model": "gpt-5.4",
        "stream": 0,
        "cache_hit": 0,
        "status_code": 200,
        "latency_ms": 100,
        "input_tokens_est": 1000,
        "output_tokens_est": 100,
        "actual_input_tokens": 1000,
        "actual_output_tokens": 100,
        "cost_est_usd": 0.002,
        "cost_baseline_usd": 0.004,
        "retry_count": 0,
        "category": "chat",
        "routing_json": stable_json({"status": "pass-through", "reason": "keep-requested-model"}),
        "crunch_json": stable_json({"changed": False, "reason": "below-threshold"}),
        "cache_json": stable_json({"status": "miss", "reason": "exact-cache-miss"}),
    }
    base.update(overrides)
    store.log_call(**base)


def _empty_store_summary(*, activation_reports: list[dict[str, object]] | None = None) -> dict[str, object]:
    with TemporaryDirectory() as tmp:
        store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
        try:
            return build_local_activation_outcome_summary(
                store,
                limit=20,
                config_dir=tmp,
                activation_reports=activation_reports or [],
            )
        finally:
            store.conn.close()


def _cache_policy_decision(decision: str, next_action: str) -> dict[str, object]:
    return {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision.v1",
        "status": "decided",
        "decision": decision,
        "top_decision": {
            "decision_id": f"cache-replay-policy-decision:{decision}",
            "decision": decision,
            "recommended_next_action": next_action,
            "metrics": {
                "observed_row_count": 36,
                "applied_count": 12,
                "holdout_count": 8,
                "exact_hit_count": 4,
                "observed_hits": 4,
                "miss_count": 8,
                "projected_hits": 35,
                "observed_savings_usd": 0.011,
                "projected_savings_usd": 0.075373,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "coverage": {
                "schema": "tokenclaw.request_shape_cache_replay_policy_decision_coverage.v1",
                "has_applied_coverage": True,
                "has_holdout_coverage": True,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "local_policy_patch": {
                "schema": "tokenclaw.request_shape_cache_replay_policy_decision_local_patch.v1",
                "target_local_rule_file": "cache_rules.yaml",
                "rules_path_included": False,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        },
        "summary": {
            "decision": decision,
            "applied_count": 12,
            "holdout_count": 8,
            "observed_hits": 4,
            "projected_hits": 35,
            "observed_savings_usd": 0.011,
            "projected_savings_usd": 0.075373,
            "target_local_rule_file": "cache_rules.yaml",
            "target_local_policy_section": "cache.pattern_rules",
        },
        "privacy": {"metadata_only": True, "aggregate_only": True},
        "raw_prompt": "raw prompt must not leak",
        "request_id": "req-closed-loop-secret",
        "session_id": "session-closed-loop-secret",
        "cache_key": "cache-key-closed-loop-secret",
        "file_path": "/tmp/private-closed-loop/cache-rules.yaml",
    }


def _crunch_rollback_evidence() -> dict[str, object]:
    return {
        "schema": "tokenclaw.request_shape_crunch_activation_evidence.v1",
        "status": "active-rule-evidence-observed",
        "decision": "widen",
        "graduation_decision": "widen",
        "decision_id": "request-shape-crunch-policy-decision:closed-loop",
        "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
        "summary": {
            "applied_count": 107,
            "holdout_count": 40,
            "skipped_count": 280,
            "safety_stop_count": 1,
            "rollback_count": 1,
            "fallback_count": 0,
            "retry_count": 0,
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "fallback_rate_delta": 0.0,
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
            "stale_evidence": {"stale": False, "age_hours": 1.5, "reason": "recent-local-outcome-window"},
        },
        "rules": [
            {
                "rule_ref": "local-repeated-context-crunch-canary-closed-loop",
                "policy_source": "local-manual",
                "decision_id": "request-shape-crunch-policy-decision:closed-loop",
                "source_evidence_schema": "tokenclaw.request_shape_crunch_activation_evidence.v1",
                "metadata_only": True,
                "aggregate_only": True,
            }
        ],
        "duplicate_suppression": {
            "schema": "tokenclaw.request_shape_crunch_keep_active_duplicate_suppression.v1",
            "fingerprint": "activation:closed-loop-crunch",
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
        "provider_body": "provider body must not leak",
        "request_id": "req-closed-loop-secret",
    }


def _cache_rollback_successor_source(*, verified: bool) -> dict[str, object]:
    rule_id = "local-openai-cache-replay-promoted:closedloop"
    gate = {
        "schema": "tokenclaw.preview_verified_activation_successor_gate.v1",
        "required": True,
        "status": "preview-verified" if verified else "no-data-preview-health",
        "verified": verified,
        "decision": "keep-blocked",
        "next_action": "rollback-cache-replay-rule",
        "reason": "rollback-required",
        "local_executor_gate": {
            "schema": "tokenclaw.preview_verified_successor_local_executor_gate.v1",
            "passed": verified,
            "policy_write_candidate": True,
            "policy_files_written": False,
            "provider_calls_made": False,
            "reason": "local-executor-gate-passed" if verified else "local-executor-gate-not-passed",
        },
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": {"metadata_only": True, "aggregate_only": True, "review_only": True},
    }
    patch = {
        "schema": "tokenclaw.request_shape_cache_replay_policy_decision_local_patch.v1",
        "patch_type": "rollback_openai_exact_cache_replay_policy",
        "target_local_rule_file": "cache_rules.yaml",
        "pattern_rules": [{"id": rule_id, "enabled": False, "disabled_reason": "stale-cache-replay-evidence"}],
        "metadata_only": True,
        "aggregate_only": True,
        "rules_path_included": False,
    }
    return {
        "schema": "tokenclaw.local_activation_next_action_queue.v1",
        "status": "ranked",
        "entries": [
            {
                "schema": "tokenclaw.local_activation_next_action_queue_entry.v1",
                "rank": 1,
                "fingerprint": "activation:cache-rollback-closed-loop",
                "lever": "cache",
                "local_action_family": "cache",
                "current_status": "blocked",
                "state": "blocked",
                "successor_status": "keep-blocked",
                "next_action": "rollback-cache-replay-rule",
                "recommended_next_action": "rollback-cache-replay-rule",
                "promotion_readiness": "rollback-required",
                "rollback_required": True,
                "target_local_policy_section": "cache.pattern_rules",
                "target_local_rule_file": "cache_rules.yaml",
                "managed_preview_required": True,
                "managed_preview_gate": gate,
                "preview_verified": verified,
                "preview_verification_status": gate["status"],
                "preview_verification_decision": "keep-blocked",
                "local_policy_patch": patch,
                "cache_key": "cache-key-closed-loop-secret",
                "request_id": "req-closed-loop-secret",
                "session_id": "session-closed-loop-secret",
                "raw_prompt": "raw prompt must not leak",
                "file_path": "/tmp/private-closed-loop/cache-rules.yaml",
                "privacy": {"metadata_only": True, "aggregate_only": True},
            }
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


class ClosedLoopActivationRegressionFixtureTests(unittest.TestCase):
    def test_observe_apply_and_safety_stop_rows_cover_routing_crunch_and_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4-mini",
                    routing_json=stable_json(
                        {
                            "openai_routing_canary": {"status": "applied", "cohort": "canary_applied"},
                            "request_id": "req-closed-loop-secret",
                            "session_id": "session-closed-loop-secret",
                        }
                    ),
                )
                _log_call(
                    store,
                    routing_json=stable_json({"openai_routing_canary": {"status": "holdout", "cohort": "canary_holdout"}}),
                )
                _log_call(
                    store,
                    routing_json=stable_json(
                        {
                            "openai_routing_canary": {
                                "status": "safety_stopped",
                                "cohort": "safety_stop",
                                "reason": "local-canary-safety-stop",
                            }
                        }
                    ),
                )
                _log_call(
                    store,
                    crunch_json=stable_json(
                        {
                            "changed": True,
                            "reason": "repeated-context-crunch-canary",
                            "tokens_saved_est": 1200,
                            "saved_chars": 4800,
                            "repeated_context_crunch_canary": {"status": "applied", "cohort": "canary_applied"},
                        }
                    ),
                )
                _log_call(
                    store,
                    crunch_json=stable_json(
                        {"repeated_context_crunch_canary": {"status": "holdout", "cohort": "canary_holdout"}}
                    ),
                )
                _log_call(
                    store,
                    cache_hit=1,
                    cache_json=stable_json({"exact_cache_replay": {"status": "applied", "cohort": "canary_applied"}}),
                )
                _log_call(
                    store,
                    cache_json=stable_json({"cache_replay_canary": {"status": "holdout", "cohort": "canary_holdout"}}),
                )

                report = build_local_activation_outcome_summary(store, limit=20, config_dir=tmp)
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "tokenclaw.local_activation_outcome_summary.v1")
        self.assertFalse(report["provider_calls_made"])
        self.assertFalse(report["managed_server_calls_made"])
        by_family = {row["local_action_family"]: row for row in report["outcome_summaries"]}
        self.assertGreaterEqual(by_family["routing"]["applied_count"], 1)
        self.assertGreaterEqual(by_family["routing"]["holdout_count"], 1)
        self.assertEqual(by_family["routing"]["safety_stopped_count"], 1)
        self.assertGreaterEqual(by_family["crunch"]["applied_count"], 1)
        self.assertGreaterEqual(by_family["crunch"]["holdout_count"], 1)
        self.assertGreaterEqual(by_family["cache"]["applied_count"], 1)
        self.assertGreaterEqual(by_family["cache"]["holdout_count"], 1)
        for row in by_family.values():
            self.assertFalse(row["local_file_backed_representation"]["path_included"])
            self.assertFalse(row["local_file_backed_representation"]["policy_file_contents_included"])
        _assert_metadata_only(self, report, str(Path(tmp).resolve()))

    def test_policy_reports_cover_cache_rollback_retire_and_crunch_duplicate_suppression(self) -> None:
        cases = [
            ("rollback", _cache_policy_decision("rollback", "rollback-cache-replay-rule")),
            ("retire-staged-no-repeat", _cache_policy_decision("retire-staged-no-repeat", "retire-cache-replay-canary-no-repeat")),
        ]
        for decision, report in cases:
            with self.subTest(cache_decision=decision):
                summary = _empty_store_summary(activation_reports=[report])
                cache = {row["local_action_family"]: row for row in summary["outcome_summaries"]}["cache"]
                self.assertEqual(cache["source_evidence_schema"], "tokenclaw.request_shape_cache_replay_policy_decision.v1")
                self.assertEqual(cache["source_decision"], decision)
                self.assertEqual(cache["next_action"], report["top_decision"]["recommended_next_action"])
                self.assertEqual(cache["applied_count"], 12)
                self.assertEqual(cache["holdout_count"], 8)
                ledger = summary["outcome_ledger_entries"][0]
                self.assertEqual(ledger["local_action_family"], "cache")
                self.assertEqual(ledger["source_decision"], decision)
                _assert_metadata_only(self, summary)

        summary = _empty_store_summary(activation_reports=[_crunch_rollback_evidence()])
        crunch = {row["local_action_family"]: row for row in summary["outcome_summaries"]}["crunch"]
        self.assertEqual(crunch["source_evidence_schema"], "tokenclaw.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(crunch["keep_active_regression_gate"]["state"], "rollback-required")
        self.assertIn("rollback-observed", crunch["keep_active_regression_gate"]["reason_codes"])
        self.assertIn("safety-stop-observed", crunch["keep_active_regression_gate"]["reason_codes"])
        self.assertTrue(crunch["duplicate_suppression"]["suppresses_new_activation_issue"])
        self.assertTrue(crunch["duplicate_suppression"]["suppresses_generic_crunch_activation_issue"])
        ledger = summary["outcome_ledger_entries"][0]
        self.assertEqual(ledger["local_action_family"], "crunch")
        self.assertEqual(ledger["rollback_count"], 1)
        self.assertEqual(ledger["safety_stop_count"], 1)
        self.assertTrue(ledger["duplicate_suppression"]["suppresses_new_activation_issue"])
        _assert_metadata_only(self, summary)

    def test_file_backed_cache_rollback_writes_only_after_preview_verified_local_executor_gate(self) -> None:
        blocked_bundle = build_local_activation_executor_bundle(
            _cache_rollback_successor_source(verified=False),
            now=None,
        )
        self.assertNotEqual(blocked_bundle["executor_action_class"], "apply-cache-rollback")
        self.assertFalse(blocked_bundle["policy_files_written"])
        self.assertFalse(blocked_bundle["provider_calls_made"])
        self.assertFalse(blocked_bundle["managed_server_calls_made"])
        _assert_metadata_only(self, blocked_bundle)

        verified_bundle = build_local_activation_executor_bundle(
            _cache_rollback_successor_source(verified=True),
            now=None,
        )
        self.assertEqual(verified_bundle["executor_action_class"], "apply-cache-rollback")
        self.assertEqual(verified_bundle["local_gate"]["preview_verification_status"], "preview-verified")
        self.assertTrue(verified_bundle["local_gate"]["preview_verified"])
        self.assertFalse(verified_bundle["policy_files_written"])
        _assert_metadata_only(self, verified_bundle)

        with TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "cache_rules.yaml"
            rules_path.write_text(
                "\n".join(
                    [
                        "schema: tokenclaw.cache_rules.v1",
                        "pattern_rules:",
                        "  - id: local-openai-cache-replay-promoted:closedloop",
                        "    enabled: true",
                        "    policy_source: local-manual",
                        "    conditions:",
                        "      provider_family: openai",
                        "    action:",
                        "      type: exact_cache_pattern",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            applied = apply_local_activation_executor_bundle(verified_bundle, config_dir=tmp, now=None)
            policy = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

            self.assertEqual(applied["schema"], "tokenclaw.local_activation_executor_apply.v1")
            self.assertEqual(applied["status"], "applied")
            self.assertTrue(applied["summary"]["policy_files_written"])
            self.assertEqual(applied["summary"]["rollback_count"], 1)
            self.assertFalse(applied["summary"]["provider_calls_made"])
            self.assertFalse(applied["summary"]["managed_server_calls_made"])
            self.assertFalse(policy["pattern_rules"][0]["enabled"])
            self.assertEqual(policy["pattern_rules"][0]["disabled_reason"], "stale-cache-replay-evidence")
            self.assertEqual(applied["egress_guard"]["status"], "passed")
            self.assertEqual(managed_egress_violations(applied), [])
            _assert_metadata_only(self, applied, str(Path(tmp).resolve()), str(rules_path))


if __name__ == "__main__":
    unittest.main()
