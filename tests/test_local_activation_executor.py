import io
import json
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from tokenclaw import cli
from tokenclaw.local_activation_executor import (
    build_managed_activation_preview_request,
    build_managed_activation_preview_result,
    build_local_activation_executor_managed_handoff,
    build_local_activation_executor_plan,
)
from tokenclaw.managed_activation_preview_outcomes import (
    build_managed_activation_preview_outcomes_report,
    persist_managed_activation_preview_outcomes,
    persist_unavailable_managed_activation_preview_outcomes,
)
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json


NOW = datetime(2026, 6, 19, 5, 30, tzinfo=timezone.utc)


def _routing_pathway_source():
    return {
        "schema": "agentflow.policy_decision.v1",
        "generated_at": "2026-06-20T08:00:00+00:00",
        "routing_pathway_matrix": {
            "schema": "agentflow.routing_pathway_matrix.v1",
            "generated_at": "2026-06-20T08:00:00+00:00",
            "pathways": [
                {
                    "schema": "agentflow.routing_pathway_matrix_entry.v1",
                    "rank": 1,
                    "pathway_id": "pathway-openai-tool-light",
                    "source_surface": "openai_responses",
                    "app_family": "generic_openai",
                    "category": "tool-light",
                    "workflow_phase": "tool-execution",
                    "requested_model": "gpt-5.4",
                    "requested_model_family": "gpt-5",
                    "target_model": "gpt-5.4-mini",
                    "target_model_family": "gpt-5-mini",
                    "text_bucket": "2k_8k_chars",
                    "token_bucket": "2k_8k_tokens",
                    "suggested_next_action": "canary",
                    "activation_recommendation": True,
                }
            ],
        },
    }


def _log_openai_pathway_canary_call(store, *, cohort: str, created_at: str) -> None:
    status = "applied" if cohort == "canary_applied" else "holdout"
    routed_model = "gpt-5.4-mini" if cohort == "canary_applied" else "gpt-5.4"
    store.log_call(
        id=str(uuid4()),
        created_at=created_at,
        path="/v1/responses",
        provider="openai",
        source_surface="openai_responses",
        endpoint="responses",
        requested_model="gpt-5.4",
        routed_model=routed_model,
        requested_model_family="gpt-5",
        routed_model_family="gpt-5-mini",
        stream=0,
        cache_hit=0,
        status_code=200,
        latency_ms=100,
        input_tokens_est=1200,
        output_tokens_est=200,
        actual_input_tokens=1200,
        actual_output_tokens=200,
        cost_est_usd=0.001,
        cost_baseline_usd=0.002,
        category="tool-light",
        retry_count=0,
        routing_json=stable_json(
            {
                "requested_model": "gpt-5.4",
                "routed_model": routed_model,
                "source_surface": "openai_responses",
                "category": "tool-light",
                "workflow_phase": "tool-execution",
                "openai_canary": {
                    "status": status,
                    "cohort": cohort,
                    "reason": cohort,
                    "requested_model": "gpt-5.4",
                    "target_model": "gpt-5.4-mini",
                    "category": "tool-light",
                    "source_surface": "openai_responses",
                },
            }
        ),
        cache_json=stable_json({"status": "miss"}),
    )


def _executor_fixture_plan():
    ledger = {
        "schema": "agentflow.evidence_to_activation_next_action_ledger.v1",
        "status": "tracked",
        "entries": [
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 1,
                "fingerprint": "activation:crunch",
                "lever": "crunch",
                "local_action_family": "crunch",
                "evidence_schema": "agentflow.crunch_savings_signal.v1",
                "current_status": "full-rollout",
                "state": "full-rollout-active",
                "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
                "blocker_codes": ["repeated-context-crunch-full-rollout-active"],
                "sample_count": 2093,
                "applied_count": 107,
                "holdout_count": 40,
                "fallback_count": 0,
                "safety_stop_count": 0,
                "rollback_count": 0,
                "projected_saved_usd": 25.818387,
                "crunch_savings_usd": 25.818613,
                "target_local_policy_section": "crunch.rules",
                "target_local_rule_file": "crunch_rules.yaml",
                "duplicate_suppression": {
                    "reason": "repeated-context-crunch-full-rollout-active",
                    "suppresses_new_activation_issue": True,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 2,
                "fingerprint": "activation:openai-routing",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.openai_routing_promotion_decision_report.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "next_action": "review-openai-routing-canary-blockers",
                "blocker_codes": ["semantic-quality-regression-observed"],
                "sample_count": 365,
                "applied_count": 25,
                "holdout_count": 21,
                "savings_per_1000_calls_usd": 4.375,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "request_id": "req-routing-secret",
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 3,
                "fingerprint": "activation:anthropic-safety-stop",
                "lever": "activation-feedback",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.activation_safety_stop_burndown.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                "blocker_codes": ["anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked"],
                "sample_count": 492,
                "safety_stop_count": 492,
                "missing_applied_coverage": True,
                "missing_holdout_coverage": True,
                "source_surface": "anthropic_messages",
                "endpoint": "/v1/messages",
                "category": "tool-result",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "required_local_executor": "anthropic-routing-rules",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "session_id": "session-anthropic-secret",
            },
            {
                "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                "rank": 4,
                "fingerprint": "activation:tool-cache",
                "lever": "cache",
                "local_action_family": "cache",
                "evidence_schema": "agentflow.request_shape_tool_cache_replay_evidence.v1",
                "current_status": "blocked",
                "state": "missing-evidence",
                "next_action": "collect-file-invalidation-evidence",
                "blocker_codes": [
                    "invalidation-evidence-missing",
                    "tools-present",
                    "unsafe-tool-calls-without-invalidation",
                ],
                "sample_count": 249,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "target_local_policy_section": "cache.pattern_rules",
                "target_local_rule_file": "cache_rules.yaml",
                "emits_cache_apply_action": False,
                "policy_files_written": False,
                "cache_key": "cache-tool-secret",
                "file_path": "/tmp/private-tool-cache.py",
                "raw_prompt": "raw prompt must not leak",
            },
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }
    return {
        "schema": "agentflow.orchestrator_research_plan.v1",
        "generated_at": NOW.isoformat(),
        "evidence": {"stats_summary": {"evidence_to_activation_next_action_ledger": ledger}},
    }


def _full_rollout_gate(state="keep-active", next_action="keep-active", reason_codes=None):
    return {
        "schema": "agentflow.full_rollout_crunch_keep_active_regression_gate.v1",
        "state": state,
        "gate_passed": state == "keep-active",
        "deterministic_next_action": next_action,
        "next_action": next_action,
        "reason_codes": reason_codes or [],
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "regression_counters": {
            "schema": "agentflow.full_rollout_crunch_keep_active_regression_counters.v1",
            "metadata_only": True,
            "aggregate_only": True,
            "applied_count": 107,
            "holdout_count": 40,
            "skipped_count": 280,
            "fallback_count": 0,
            "retry_count": 0,
            "rollback_count": 0,
            "safety_stop_count": 0,
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "fallback_rate_delta": 0.0,
            "decision_age_hours": 0.0,
            "stale_evidence": {
                "metadata_only": True,
                "aggregate_only": True,
                "stale": state == "review-stale-evidence",
                "status": "stale" if state == "review-stale-evidence" else "fresh-or-active",
                "reason": "stale-full-rollout-evidence" if state == "review-stale-evidence" else "full-rollout-local-policy-active",
            },
        },
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _full_rollout_queue_entry(*, gate=None, outcome="keep-active"):
    gate = gate or _full_rollout_gate()
    return {
        "schema": "agentflow.local_activation_next_action_queue_entry.v1",
        "rank": 1,
        "ledger_rank": 1,
        "fingerprint": "activation:f5f6eae5f0a0081a",
        "lever": "crunch",
        "local_action_family": "crunch",
        "evidence_schema": "agentflow.crunch_savings_signal.v1",
        "current_status": "full-rollout",
        "state": "full-rollout-active",
        "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
        "blocker_codes": ["repeated-context-crunch-full-rollout-active"],
        "sample_count": 2093,
        "applied_count": 107,
        "holdout_count": 40,
        "skipped_count": 280,
        "fallback_count": 0,
        "retry_count": 0,
        "rollback_count": gate["regression_counters"]["rollback_count"],
        "safety_stop_count": 0,
        "projected_savings_usd": 25.818387,
        "realized_savings_usd": 25.818613,
        "observed_saved_tokens": 8606129,
        "projected_saved_tokens": 8606129,
        "target_local_policy_section": "crunch.rules",
        "target_local_rule_file": "crunch_rules.yaml",
        "duplicate_suppression_status": "suppressed",
        "duplicate_suppression_reason": "repeated-context-crunch-full-rollout-active",
        "measured_full_rollout_activation": True,
        "durable_outcome_ledger_entry": True,
        "full_rollout_outcome": outcome,
        "full_rollout_outcome_next_action": gate["next_action"],
        "full_rollout_successor_decision": "no-op" if outcome == "keep-active" else outcome,
        "full_rollout_successor_next_action": "keep-current-rule-only" if outcome == "keep-active" else gate["next_action"],
        "full_rollout_successor_no_op_reason": "no-unsuppressed-post-full-rollout-crunch-cohort",
        "full_rollout_activation_outcome": {
            "schema": "agentflow.full_rollout_crunch_activation_outcome.v1",
            "durable_outcome_ledger_entry": True,
            "ledger_fingerprint": "activation:f5f6eae5f0a0081a",
            "ledger_rank": 1,
            "lever": "crunch",
            "local_action_family": "crunch",
            "outcome": outcome,
            "next_action": gate["next_action"],
            "outcome_options": ["keep-active", "review-stale-evidence", "rollback-required", "keep-blocked"],
            "applied_count": 107,
            "holdout_count": 40,
            "skipped_count": 280,
            "fallback_count": 0,
            "retry_count": 0,
            "rollback_count": gate["regression_counters"]["rollback_count"],
            "safety_stop_count": 0,
            "observed_saved_tokens": 8606129,
            "observed_savings_usd": 25.818387,
            "projected_saved_tokens": 8606129,
            "projected_savings_usd": 25.818387,
            "successor_decision": "no-op" if outcome == "keep-active" else outcome,
            "successor_next_action": "keep-current-rule-only" if outcome == "keep-active" else gate["next_action"],
            "successor_no_op_reason": "no-unsuppressed-post-full-rollout-crunch-cohort",
            "keep_active_regression_gate": gate,
            "privacy": {"metadata_only": True, "aggregate_only": True},
        },
        "keep_active_regression_gate": gate,
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _full_rollout_queue_plan(entry):
    return {
        "schema": "agentflow.local_activation_next_action_queue.v1",
        "status": "ranked",
        "entries": [entry],
        "successor_actions": [
            {
                "schema": "agentflow.local_activation_successor_action.v1",
                "rank": 1,
                "fingerprint": "successor:stale-precomputed-row",
                "source_fingerprint": entry["fingerprint"],
                "lever": "crunch",
                "local_action_family": "crunch",
                "successor_status": "review",
                "current_status": "full-rollout",
                "state": "full-rollout-active",
                "recommended_next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            }
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


def _executor_handoff_fixture_plan():
    plan = _full_rollout_queue_plan(_full_rollout_queue_entry())
    plan["entries"].extend(
        [
            {
                "schema": "agentflow.local_activation_next_action_queue_entry.v1",
                "rank": 2,
                "ledger_rank": 2,
                "fingerprint": "activation:openai-routing-blocked",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.pass_through_routing_activation_candidates.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "successor_status": "keep-blocked",
                "next_action": "keep-openai-routing-blocked-until-canary-coverage",
                "blocker_codes": ["missing-applied-coverage", "missing-holdout-coverage"],
                "sample_count": 2,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-heavy",
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
                "required_local_executor": "openai-routing-canary",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "request_id": "req-openai-routing-secret",
            },
            {
                "schema": "agentflow.local_activation_next_action_queue_entry.v1",
                "rank": 3,
                "ledger_rank": 3,
                "fingerprint": "activation:routing-blocked",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.activation_safety_stop_burndown.v1",
                "current_status": "keep-blocked",
                "state": "keep-blocked",
                "successor_status": "keep-blocked",
                "next_action": "keep-anthropic-routing-blocked-until-safety-stop-burndown",
                "blocker_codes": ["anthropic-routing-safety-stop-local-canary-safety-stop-keep-blocked"],
                "sample_count": 492,
                "safety_stop_count": 492,
                "source_surface": "anthropic_messages",
                "endpoint": "/v1/messages",
                "category": "tool-result",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "required_local_executor": "anthropic-routing-rules",
                "target_local_policy_section": "routing.rules",
                "target_local_rule_file": "routing_rules.yaml",
                "session_id": "session-routing-secret",
            },
            {
                "schema": "agentflow.local_activation_next_action_queue_entry.v1",
                "rank": 4,
                "ledger_rank": 4,
                "fingerprint": "activation:cache-review",
                "lever": "cache",
                "local_action_family": "cache",
                "evidence_schema": "agentflow.request_shape_cache_replay_policy_decision.v1",
                "current_status": "review",
                "state": "review",
                "successor_status": "review-only",
                "next_action": "review-cache-replay-canary-promotion-readiness",
                "blocker_codes": [],
                "sample_count": 102,
                "applied_count": 30,
                "holdout_count": 72,
                "projected_savings_usd": 0.075373,
                "realized_savings_usd": 0.0,
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "target_local_policy_section": "cache.pattern_rules",
                "target_local_rule_file": "cache_rules.yaml",
                "cache_key": "cache-review-secret",
                "file_path": "/tmp/private-cache-review.py",
                "raw_prompt": "raw cache review prompt must not leak",
            },
        ]
    )
    return plan


def _preview_required_successor_queue_plan():
    gate = {
        "schema": "agentflow.preview_verified_activation_successor_gate.v1",
        "required": True,
        "status": "no-data-preview-health",
        "verified": False,
        "decision": "keep-blocked",
        "next_action": "refresh-managed-activation-preview",
        "reason": "managed-preview-health-no-data",
        "policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "privacy": {"metadata_only": True, "aggregate_only": True, "review_only": True},
    }
    common = {
        "schema": "agentflow.local_activation_next_action_queue_entry.v1",
        "current_status": "keep-blocked",
        "state": "keep-blocked",
        "successor_status": "keep-blocked",
        "next_action": "refresh-managed-activation-preview",
        "sample_count": 10,
        "managed_preview_required": True,
        "managed_preview_gate": gate,
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }
    return {
        "schema": "agentflow.local_activation_next_action_queue.v1",
        "status": "ranked",
        "entries": [
            {
                **common,
                "rank": 1,
                "fingerprint": "activation:preview-routing",
                "lever": "routing",
                "local_action_family": "routing",
                "evidence_schema": "agentflow.openai_routing_promotion_decision_report.v1",
                "blocker_codes": ["semantic-quality-regression-observed"],
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "requested_model": "gpt-5.4",
                "candidate_target_model": "gpt-5.4-mini",
            },
            {
                **common,
                "rank": 2,
                "fingerprint": "activation:preview-cache",
                "lever": "cache",
                "local_action_family": "cache",
                "evidence_schema": "agentflow.request_shape_tool_cache_replay_evidence.v1",
                "blocker_codes": ["invalidation-evidence-missing"],
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
            },
            {
                **common,
                "rank": 3,
                "fingerprint": "activation:preview-feedback",
                "lever": "activation-feedback",
                "local_action_family": "activation-feedback",
                "evidence_schema": "agentflow.orchestrator_research_log_diagnostics.v1",
                "blocker_codes": ["safety-stop"],
                "diagnostic_class": "safety-stop",
                "diagnostic_reason": "safety-stop",
            },
        ],
        "privacy": {"metadata_only": True, "aggregate_only": True},
    }


class LocalActivationExecutorTest(unittest.TestCase):
    def test_executor_plan_selects_one_safe_review_action(self):
        result = build_local_activation_executor_plan(_executor_fixture_plan(), now=NOW)

        self.assertEqual(result["schema"], "agentflow.local_activation_executor_plan.v1")
        self.assertEqual(result["status"], "ranked")
        self.assertEqual(result["summary"]["selected_action_count"], 1)
        self.assertEqual(result["summary"]["selected_executor_action_class"], "review-only")
        self.assertEqual(result["summary"]["selected_next_action"], "review-openai-routing-canary-blockers")
        self.assertEqual(result["summary"]["selected_local_action_family"], "routing")
        self.assertEqual(result["summary"]["selected_target_local_rule_file"], "routing_rules.yaml")
        self.assertFalse(result["summary"]["policy_files_written"])
        self.assertFalse(result["summary"]["provider_calls_made"])
        self.assertFalse(result["summary"]["managed_server_calls_made"])

        entries = result["entries"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(len({entry["fingerprint"] for entry in entries}), 4)
        selected = result["selected_action"]
        self.assertTrue(selected["selected"])
        self.assertTrue(selected["fingerprint"].startswith("executor:"))
        self.assertTrue(selected["source_successor_fingerprint"].startswith("successor:"))
        self.assertEqual(selected["source_fingerprint"], "activation:openai-routing")
        self.assertEqual(selected["executor_action_class"], "review-only")
        self.assertIn("semantic-quality-regression-observed", selected["reason_codes"])

        by_next_action = {entry["executor_next_action"]: entry for entry in entries}
        self.assertEqual(
            by_next_action["keep-current-rule-only"]["executor_action_class"],
            "keep-current-rule",
        )
        self.assertEqual(
            by_next_action["review-openai-routing-canary-blockers"]["executor_action_class"],
            "review-only",
        )
        self.assertEqual(
            by_next_action["keep-anthropic-routing-blocked-until-safety-stop-burndown"]["executor_action_class"],
            "keep-blocked",
        )
        self.assertEqual(
            by_next_action["collect-file-invalidation-evidence"]["executor_action_class"],
            "keep-blocked",
        )
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertTrue(result["privacy"]["aggregate_only"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["provider_bodies_included"])
        self.assertFalse(result["privacy"]["cache_keys_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["file_paths_included"])
        self.assertFalse(result["privacy"]["policy_file_contents_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])
        self.assertFalse(result["privacy"]["managed_server_calls_made"])
        self.assertFalse(result["privacy"]["policy_files_written"])

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("req-routing-secret", rendered)
        self.assertNotIn("session-anthropic-secret", rendered)
        self.assertNotIn("cache-tool-secret", rendered)
        self.assertNotIn("/tmp/private-tool-cache.py", rendered)
        self.assertNotIn("raw prompt must not leak", rendered)

    def test_local_activation_executor_cli_reads_plan_json(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.local_activation_executor_cli(["--plan-json", str(plan_path), "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.local_activation_executor_plan.v1")
        self.assertEqual(result["summary"]["selected_action_count"], 1)
        self.assertEqual(result["selected_action"]["executor_next_action"], "review-openai-routing-canary-blockers")

    def test_executor_managed_handoff_exports_feature_only_outcome_rows(self):
        result = build_local_activation_executor_managed_handoff(
            _executor_handoff_fixture_plan(),
            now=NOW,
        )

        self.assertEqual(result["schema"], "agentflow.local_activation_managed_handoff.v1")
        self.assertEqual(result["status"], "exported")
        self.assertEqual(result["summary"]["handoff_row_count"], 4)
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["policy_files_written"])
        self.assertFalse(result["server_ingestion_required"])
        self.assertTrue(result["feature_only"])
        self.assertTrue(result["locally_executed"])
        self.assertFalse(result["provider_forwarding"])
        self.assertFalse(result["server_content_processing"])
        self.assertFalse(result["managed_enforced"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(result["egress_guard"]["violation_count"], 0)
        self.assertEqual(managed_egress_violations(result), [])

        privacy = result["privacy"]
        self.assertTrue(privacy["feature_only"])
        self.assertTrue(privacy["metadata_only"])
        self.assertTrue(privacy["aggregate_only"])
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["provider_bodies_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["cache_keys_included"])
        self.assertFalse(privacy["absolute_paths_included"])
        self.assertFalse(privacy["policy_file_contents_included"])

        by_next_action = {row["executor_next_action"]: row for row in result["rows"]}
        crunch = by_next_action["keep-current-rule-only"]
        openai_routing = by_next_action["keep-openai-routing-blocked-until-canary-coverage"]
        anthropic_routing = by_next_action["keep-anthropic-routing-blocked-until-safety-stop-burndown"]
        cache = by_next_action["review-cache-replay-canary-promotion-readiness"]
        self.assertEqual(crunch["local_action_family"], "crunch")
        self.assertEqual(crunch["executor_action_class"], "keep-current-rule")
        self.assertEqual(crunch["activation_outcome"], "keep-active")
        self.assertEqual(crunch["coverage"]["applied_count"], 107)
        self.assertEqual(crunch["coverage"]["holdout_count"], 40)
        self.assertEqual(crunch["coverage"]["safety_stop_count"], 0)
        self.assertEqual(crunch["coverage"]["rollback_count"], 0)
        self.assertEqual(openai_routing["local_action_family"], "routing")
        self.assertEqual(openai_routing["source_surface"], "openai_responses")
        self.assertEqual(openai_routing["executor_action_class"], "keep-blocked")
        self.assertIn("missing-applied-coverage", openai_routing["blocker_codes"])
        self.assertEqual(anthropic_routing["local_action_family"], "routing")
        self.assertEqual(anthropic_routing["source_surface"], "anthropic_messages")
        self.assertEqual(anthropic_routing["executor_action_class"], "keep-blocked")
        self.assertEqual(anthropic_routing["coverage"]["safety_stop_count"], 492)
        self.assertEqual(cache["local_action_family"], "cache")
        self.assertEqual(cache["executor_action_class"], "review-only")
        self.assertEqual(cache["target_local_rule_file"], "cache_rules.yaml")
        self.assertTrue(cache["handoff_ref"].startswith("handoff:"))
        self.assertNotIn("fingerprint", cache)
        self.assertNotIn("request_id", openai_routing)
        self.assertNotIn("session_id", anthropic_routing)
        self.assertNotIn("cache_key", cache)

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("req-openai-routing-secret", rendered)
        self.assertNotIn("session-routing-secret", rendered)
        self.assertNotIn("cache-review-secret", rendered)
        self.assertNotIn("/tmp/private-cache-review.py", rendered)
        self.assertNotIn("raw cache review prompt must not leak", rendered)

    def test_local_activation_executor_cli_emits_managed_handoff(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.local_activation_executor_cli(
                ["--plan-json", str(plan_path), "--managed-handoff", "--pretty"],
                stdout=stdout,
            )

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.local_activation_managed_handoff.v1")
        self.assertEqual(result["summary"]["handoff_row_count"], 4)
        self.assertEqual(result["egress_guard"]["status"], "passed")

    def test_managed_activation_preview_request_is_feature_only(self):
        result = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)

        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_request.v1")
        self.assertEqual(result["summary"]["handoff_row_count"], 4)
        self.assertIn("activation-feedback", result["supported_local_action_families"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["policy_files_written"])
        self.assertTrue(result["feature_only"])
        self.assertTrue(result["locally_executed"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])

    def test_managed_activation_preview_request_marks_preview_required_successors(self):
        result = build_managed_activation_preview_request(_preview_required_successor_queue_plan(), now=NOW)

        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_request.v1")
        self.assertEqual(result["summary"]["handoff_row_count"], 3)
        self.assertEqual(result["summary"]["preview_required_row_count"], 3)
        required_counts = {
            item["value"]: item["count"]
            for item in result["summary"]["preview_required_local_action_family_counts"]
        }
        self.assertEqual(required_counts["routing"], 1)
        self.assertEqual(required_counts["cache"], 1)
        self.assertEqual(required_counts["activation-feedback"], 1)
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["policy_files_written"])
        for row in result["rows"]:
            self.assertTrue(row["managed_preview_required"])
            self.assertFalse(row["preview_verified"])
            self.assertEqual(row["preview_verification_status"], "no-data-preview-health")
            self.assertEqual(row["preview_verification_decision"], "keep-blocked")
            self.assertEqual(row["executor_next_action"], "refresh-managed-activation-preview")
            self.assertTrue(row["handoff_ref"].startswith("handoff:"))
            self.assertTrue(str(row["source_activation_ref"]).startswith(("activation:", "activation-ref:")))
            self.assertTrue(str(row["source_successor_ref"]).startswith(("successor:", "successor-ref:")))
            self.assertTrue(row["privacy"]["metadata_only"])
            self.assertFalse(row["privacy"]["provider_calls_made"])
            self.assertFalse(row["privacy"]["managed_server_calls_made"])
            self.assertFalse(row["privacy"]["policy_files_written"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])

    def test_managed_activation_preview_result_summarizes_review_only_decisions(self):
        request_payload = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)
        first_ref = request_payload["rows"][0]["handoff_ref"]
        second_ref = request_payload["rows"][1]["handoff_ref"]
        response_payload = {
            "schema": "agentflow.managed_activation_preview_response.v1",
            "decisions": [
                {
                    "handoff_ref": first_ref,
                    "decision": "no-op",
                    "no_op_reason": "full-rollout-policy-active",
                    "review_only": True,
                    "raw_prompt": "raw server value must not leak",
                },
                {
                    "handoff_ref": second_ref,
                    "decision": "keep-blocked",
                    "omitted_reason": "safety-stop-observed",
                    "reason_codes": ["safety-stop-observed"],
                    "review_only": True,
                },
            ],
        }

        result = build_managed_activation_preview_result(
            request_payload,
            response_payload=response_payload,
            fetch={"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        )

        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_result.v1")
        self.assertEqual(result["status"], "previewed")
        self.assertEqual(result["coverage"]["handoff_row_count"], 4)
        self.assertEqual(result["coverage"]["preview_decision_count"], 2)
        self.assertEqual(result["coverage"]["matched_handoff_ref_count"], 2)
        self.assertEqual(result["coverage"]["missing_preview_decision_count"], 2)
        self.assertEqual(result["coverage"]["no_op_count"], 2)
        self.assertEqual(result["coverage"]["omitted_count"], 1)
        self.assertEqual(result["coverage"]["review_only_count"], 2)
        self.assertEqual(result["coverage"]["active_policy_write_count"], 0)
        self.assertEqual(result["summary"]["submitted_row_count"], 4)
        self.assertEqual(result["summary"]["preview_row_count"], 2)
        self.assertEqual(result["summary"]["omission_count"], 1)
        self.assertTrue(result["coverage"]["managed_server_calls_made"])
        self.assertFalse(result["coverage"]["provider_calls_made"])
        self.assertFalse(result["coverage"]["policy_files_written"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn('"raw_prompt":', rendered)
        self.assertNotIn("raw server value must not leak", rendered)

    def test_managed_activation_preview_outcomes_persist_review_only_rows_idempotently(self):
        request_payload = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)
        rows = request_payload["rows"]
        response_payload = {
            "schema": "agentflow.managed_activation_preview_response.v1",
            "decisions": [
                {
                    "handoff_ref": rows[0]["handoff_ref"],
                    "decision": "no-op",
                    "no_op_reason": "full-rollout-policy-active",
                    "review_only": True,
                    "raw_prompt": "raw preview value must not leak",
                },
                {
                    "handoff_ref": rows[1]["handoff_ref"],
                    "decision": "review-only-recommendation",
                    "recommended_next_action": "draft-openai-routing-recovery-canary",
                    "reason_codes": ["managed-preview-would-draft-recovery"],
                    "review_only": True,
                },
                {
                    "handoff_ref": rows[2]["handoff_ref"],
                    "decision": "keep-blocked",
                    "recommended_next_action": "review-anthropic-safety-stop",
                    "provider_calls_made": True,
                    "review_only": True,
                },
            ],
        }
        preview_result = build_managed_activation_preview_result(
            request_payload,
            response_payload=response_payload,
            fetch={"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        )

        with TemporaryDirectory() as tmpdir:
            store = Store(str(Path(tmpdir) / "agentflow.sqlite3"))
            try:
                first = persist_managed_activation_preview_outcomes(store, preview_result, now=NOW)
                second = persist_managed_activation_preview_outcomes(store, preview_result, now=NOW + timedelta(minutes=5))
                stored_count = store.conn.execute("select count(*) as c from managed_activation_preview_outcomes").fetchone()["c"]
                stale = build_managed_activation_preview_outcomes_report(
                    store,
                    now=NOW + timedelta(hours=80),
                    stale_after_hours=72,
                )
            finally:
                store.conn.close()

        self.assertEqual(first["schema"], "agentflow.managed_activation_preview_outcomes.v1")
        self.assertEqual(first["import"]["imported_count"], 4)
        self.assertEqual(first["import"]["created_count"], 4)
        self.assertEqual(first["summary"]["stored_preview_outcome_count"], 4)
        self.assertEqual(first["summary"]["missing_preview_decision_count"], 1)
        self.assertEqual(first["summary"]["failed_closed_count"], 1)
        self.assertEqual(first["summary"]["disagreement_count"], 1)
        self.assertEqual(second["import"]["created_count"], 0)
        self.assertEqual(second["import"]["updated_count"], 4)
        self.assertEqual(stored_count, 4)
        fingerprints = [row["outcome_fingerprint"] for row in first["outcomes"]]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        classifications = {row["handoff_ref"]: row["classification"] for row in first["outcomes"]}
        self.assertEqual(classifications[rows[1]["handoff_ref"]], "managed-local-disagreement")
        self.assertEqual(classifications[rows[2]["handoff_ref"]], "failed-closed")
        self.assertEqual(classifications[rows[3]["handoff_ref"]], "missing-preview-decision")
        fresh = {row["handoff_ref"]: row for row in first["outcomes"]}
        self.assertFalse(fresh[rows[0]["handoff_ref"]]["policy_files_written"])
        self.assertFalse(fresh[rows[0]["handoff_ref"]]["provider_calls_made"])
        stale_by_ref = {row["handoff_ref"]: row for row in stale["outcomes"]}
        self.assertTrue(stale_by_ref[rows[0]["handoff_ref"]]["stale"])
        self.assertEqual(stale_by_ref[rows[0]["handoff_ref"]]["classification"], "stale-preview")
        self.assertEqual(stale_by_ref[rows[0]["handoff_ref"]]["next_action"], "refresh-managed-activation-preview")
        rendered = json.dumps(first, sort_keys=True)
        self.assertNotIn("raw preview value must not leak", rendered)
        self.assertNotIn("req-routing-secret", rendered)
        self.assertNotIn("session-anthropic-secret", rendered)
        self.assertNotIn("cache-tool-secret", rendered)
        self.assertNotIn("/tmp/private-tool-cache.py", rendered)
        self.assertEqual(managed_egress_violations(first), [])

    def test_managed_activation_preview_outcomes_normalize_server_classifications(self):
        request_payload = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)
        rows = request_payload["rows"]
        response_payload = {
            "schema": "agentflow.managed_activation_preview_response.v1",
            "status": "previewed",
            "decisions": [
                {
                    "handoff_ref": rows[0]["handoff_ref"],
                    "fingerprint": "managed-preview:cache-accepted",
                    "classification": "accepted",
                    "decision": "accepted",
                    "status": "accepted",
                    "recommended_next_action": rows[0]["executor_next_action"],
                    "agreement_status": "agreed",
                    "agrees_with_local_next_action": True,
                    "reason_codes": ["managed-preview-accepted"],
                    "review_only": True,
                },
                {
                    "handoff_ref": rows[1]["handoff_ref"],
                    "fingerprint": "managed-preview:routing-needs-local-evidence",
                    "classification": "needs-local-evidence",
                    "decision": "keep-blocked",
                    "status": "needs-local-evidence",
                    "recommended_next_action": "collect-local-evidence-before-activation",
                    "agreement_status": "needs-local-evidence",
                    "omitted_reason": "local-evidence-required",
                    "reason_codes": ["missing-applied-coverage"],
                    "review_only": True,
                },
                {
                    "handoff_ref": rows[3]["handoff_ref"],
                    "fingerprint": "managed-preview:crunch-omitted",
                    "classification": "omitted",
                    "decision": "no-op",
                    "status": "omitted",
                    "recommended_next_action": "keep-current-local-decision",
                    "agreement_status": "omitted",
                    "no_op_reason": "full-rollout-policy-active",
                    "reason_codes": ["current-rule-active"],
                    "review_only": True,
                },
            ],
        }
        preview_result = build_managed_activation_preview_result(
            request_payload,
            response_payload=response_payload,
            fetch={"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        )

        with TemporaryDirectory() as tmpdir:
            store = Store(str(Path(tmpdir) / "agentflow.sqlite3"))
            try:
                report = persist_managed_activation_preview_outcomes(store, preview_result, now=NOW)
            finally:
                store.conn.close()

        by_ref = {row["handoff_ref"]: row for row in report["outcomes"]}
        self.assertEqual(by_ref[rows[0]["handoff_ref"]]["managed_preview_classification"], "accepted")
        self.assertEqual(by_ref[rows[0]["handoff_ref"]]["classification"], "review-only")
        self.assertEqual(by_ref[rows[1]["handoff_ref"]]["managed_preview_classification"], "needs-local-evidence")
        self.assertEqual(by_ref[rows[1]["handoff_ref"]]["classification"], "preview-omitted")
        self.assertEqual(by_ref[rows[3]["handoff_ref"]]["managed_preview_classification"], "omitted")
        self.assertEqual(by_ref[rows[3]["handoff_ref"]]["classification"], "preview-omitted")
        self.assertEqual(report["summary"]["no_data_preview_health_count"], 0)
        self.assertEqual(report["summary"]["failed_closed_count"], 0)
        self.assertFalse(report["summary"]["policy_files_written"])
        self.assertFalse(report["summary"]["provider_calls_made"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn('"policy_files_written": true', rendered.lower())
        self.assertEqual(managed_egress_violations(report), [])

    def test_managed_activation_preview_outcomes_preserve_cache_rollback_guidance(self):
        preview_result = {
            "schema": "agentflow.managed_activation_preview_result.v1",
            "generated_at": NOW.isoformat(),
            "preview_request": {
                "schema": "agentflow.managed_activation_preview_request.v1",
                "generated_at": NOW.isoformat(),
                "rows": [
                    {
                        "handoff_ref": "handoff:cache-rollback",
                        "source_activation_ref": "activation-ref:cache-rollback",
                        "source_successor_ref": "successor-ref:cache-rollback",
                        "local_action_family": "cache",
                        "evidence_schema": "agentflow.request_shape_cache_replay_evidence.v1",
                        "current_status": "blocked",
                        "executor_action_class": "keep-blocked",
                        "executor_next_action": "rollback-cache-replay-rule",
                        "blocker_codes": ["evidence-older-than-max-age"],
                    }
                ],
            },
            "preview": {
                "schema": "agentflow.managed_activation_preview_response.v1",
                "decisions": [
                    {
                        "handoff_ref": "handoff:cache-rollback",
                        "classification": "accepted",
                        "decision": "accepted",
                        "recommended_next_action": "rollback-cache-replay-rule",
                        "rollback_required": True,
                        "promotion_readiness": "rollback-required",
                        "rollback_metadata": {
                            "rollback_action_type": "disable_openai_exact_cache_replay_policy",
                            "target_local_rule_file": "cache_rules.yaml",
                            "target_local_policy_section": "cache.pattern_rules",
                            "disabled_reason": "stale-cache-replay-evidence",
                        },
                        "review_only": True,
                        "policy_files_written": False,
                        "provider_calls_made": False,
                        "raw_prompt": "raw rollback response prompt must not leak",
                    }
                ],
            },
            "fetch": {"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        }

        with TemporaryDirectory() as tmpdir:
            store = Store(str(Path(tmpdir) / "agentflow.sqlite3"))
            try:
                report = persist_managed_activation_preview_outcomes(store, preview_result, now=NOW)
            finally:
                store.conn.close()

        outcome = report["outcomes"][0]
        guidance = outcome["cache_rollback_guidance"]
        self.assertEqual(outcome["classification"], "review-only")
        self.assertTrue(outcome["rollback_required"])
        self.assertEqual(outcome["promotion_readiness"], "rollback-required")
        self.assertEqual(outcome["next_action"], "rollback-cache-replay-rule")
        self.assertEqual(guidance["rollback_action_type"], "disable_openai_exact_cache_replay_policy")
        self.assertEqual(guidance["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(guidance["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(guidance["disabled_reason"], "stale-cache-replay-evidence")
        self.assertEqual(outcome["cache_apply_action_count"], 0)
        self.assertEqual(outcome["cache_entries_written"], 0)
        self.assertFalse(outcome["emits_cache_apply_action"])
        self.assertFalse(outcome["policy_files_written"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw rollback response prompt must not leak", rendered)
        self.assertNotIn('"policy_files_written": true', rendered.lower())
        self.assertEqual(managed_egress_violations(report), [])

    def test_managed_activation_preview_outcomes_preserve_crunch_preview_fields(self):
        preview_result = {
            "schema": "agentflow.managed_activation_preview_result.v1",
            "generated_at": NOW.isoformat(),
            "preview_request": {
                "schema": "agentflow.managed_activation_preview_request.v1",
                "generated_at": NOW.isoformat(),
                "rows": [
                    {
                        "handoff_ref": "handoff:crunch-preview",
                        "source_activation_ref": "activation-ref:crunch-preview",
                        "source_successor_ref": "successor-ref:crunch-preview",
                        "local_action_family": "crunch",
                        "evidence_schema": "agentflow.crunch_savings_signal.v1",
                        "current_status": "ready",
                        "executor_action_class": "review-only",
                        "executor_next_action": "rank-repeated-context-crunch-dry-run",
                        "cohort_class": "repeated-context-crunch",
                        "source_queue_rank": 7,
                        "source_ledger_rank": 2,
                        "source_successor_fingerprint": "successor:crunch-preview",
                    }
                ],
            },
            "preview": {
                "schema": "agentflow.managed_activation_preview_response.v1",
                "decisions": [
                    {
                        "handoff_ref": "handoff:crunch-preview",
                        "classification": "accepted",
                        "decision": "review-ready",
                        "crunch_preview_decision": "review-ready",
                        "crunch_preview_confidence": 0.88,
                        "recommended_next_action": "rank-repeated-context-crunch-dry-run",
                        "reason_codes": [
                            "crunch-preview:review-ready",
                            "repeated-context-crunch-review-ready",
                        ],
                        "quality_risk_reason_codes": [],
                        "projected_saved_tokens": 18400,
                        "projected_saved_usd": 0.184,
                        "projected_savings_usd": 0.184,
                        "observed_saved_tokens": 4200,
                        "observed_saved_usd": 0.042,
                        "observed_crunch_ratio": 0.31,
                        "sample_count": 72,
                        "applied_count": 28,
                        "holdout_count": 24,
                        "rollback_count": 0,
                        "safety_stop_count": 0,
                        "source_queue_rank": 7,
                        "source_ledger_rank": 2,
                        "source_successor_fingerprint": "successor:crunch-preview",
                        "successor_action_fingerprint": "successor-action:crunch-preview",
                        "successor_decision_fingerprint": "successor-decision:crunch-preview",
                        "target_local_policy_section": "crunch.rules",
                        "target_local_rule_file": "crunch_rules.yaml",
                        "review_only": True,
                        "policy_files_written": False,
                        "provider_calls_made": False,
                        "raw_prompt": "raw crunch preview response must not leak",
                    }
                ],
            },
            "fetch": {"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        }

        with TemporaryDirectory() as tmpdir:
            store = Store(str(Path(tmpdir) / "agentflow.sqlite3"))
            try:
                report = persist_managed_activation_preview_outcomes(store, preview_result, now=NOW)
            finally:
                store.conn.close()

        outcome = report["outcomes"][0]
        self.assertEqual(outcome["classification"], "review-only")
        self.assertEqual(outcome["crunch_preview_decision"], "review-ready")
        self.assertEqual(outcome["crunch_preview_confidence"], 0.88)
        self.assertEqual(outcome["projected_saved_tokens"], 18400)
        self.assertEqual(outcome["projected_saved_usd"], 0.184)
        self.assertEqual(outcome["observed_saved_tokens"], 4200)
        self.assertEqual(outcome["observed_saved_usd"], 0.042)
        self.assertEqual(outcome["observed_crunch_ratio"], 0.31)
        self.assertEqual(outcome["successor_decision_fingerprint"], "successor-decision:crunch-preview")
        self.assertEqual(outcome["target_local_rule_file"], "crunch_rules.yaml")
        self.assertFalse(outcome["policy_files_written"])
        self.assertFalse(outcome["provider_calls_made"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw crunch preview response must not leak", rendered)
        self.assertNotIn('"policy_files_written": true', rendered.lower())
        self.assertEqual(managed_egress_violations(report), [])

    def test_managed_activation_preview_outcomes_cli_reports_stored_outcomes(self):
        request_payload = build_managed_activation_preview_request(_executor_handoff_fixture_plan(), now=NOW)
        preview_result = build_managed_activation_preview_result(
            request_payload,
            response_payload={
                "schema": "agentflow.managed_activation_preview_response.v1",
                "decisions": [
                    {
                        "handoff_ref": request_payload["rows"][0]["handoff_ref"],
                        "decision": "no-op",
                        "no_op_reason": "full-rollout-policy-active",
                        "review_only": True,
                    }
                ],
            },
            fetch={"status": "ok", "status_code": 200, "managed_server_calls_made": True},
        )

        with TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                persist_managed_activation_preview_outcomes(store, preview_result, now=NOW)
            finally:
                store.conn.close()
            stdout = io.StringIO()

            code = cli.managed_activation_preview_outcomes_cli(["--db", db_path, "--pretty"], stdout=stdout)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema"], "agentflow.managed_activation_preview_outcomes.v1")
        self.assertEqual(payload["summary"]["stored_preview_outcome_count"], 4)
        self.assertEqual(payload["summary"]["missing_preview_decision_count"], 3)
        self.assertFalse(payload["summary"]["policy_files_written"])
        self.assertFalse(payload["summary"]["provider_calls_made"])
        self.assertEqual(payload["egress_guard"]["status"], "passed")

    def test_managed_activation_preview_cli_skips_without_url(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.managed_activation_preview_cli(["--plan-json", str(plan_path), "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["schema"], "agentflow.managed_activation_preview_result.v1")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["fetch"]["reason"], "managed-preview-url-not-configured")
        self.assertFalse(result["coverage"]["managed_server_calls_made"])
        self.assertEqual(result["coverage"]["handoff_row_count"], 4)
        self.assertEqual(result["summary"]["submitted_row_count"], 4)
        self.assertEqual(result["summary"]["preview_row_count"], 0)
        self.assertEqual(result["summary"]["omission_count"], 0)
        self.assertEqual(result["routing_pathway_outcome_batch"]["status"], "skipped")
        self.assertEqual(
            result["routing_pathway_outcome_batch"]["reason"],
            "routing-pathway-outcome-batch-input-not-configured",
        )
        self.assertEqual(result["summary"]["routing_pathway_outcome_batch_outcome_count"], 0)
        self.assertEqual(result["egress_guard"]["status"], "passed")

    def test_managed_activation_preview_cli_persists_no_data_health_without_url(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            db_path = Path(tmpdir) / "agentflow.sqlite3"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()

            code = cli.managed_activation_preview_cli(
                [
                    "--plan-json",
                    str(plan_path),
                    "--persist-outcomes",
                    "--db",
                    str(db_path),
                    "--pretty",
                ],
                stdout=stdout,
            )

        result = json.loads(stdout.getvalue())
        stored = result["stored_preview_outcomes"]
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(stored["schema"], "agentflow.managed_activation_preview_outcomes.v1")
        self.assertEqual(stored["import"]["imported_count"], 4)
        self.assertEqual(stored["summary"]["stored_preview_outcome_count"], 4)
        self.assertEqual(stored["summary"]["no_data_preview_health_count"], 4)
        self.assertEqual(stored["summary"]["missing_preview_decision_count"], 0)
        self.assertFalse(stored["summary"]["policy_files_written"])
        self.assertFalse(stored["summary"]["provider_calls_made"])
        self.assertFalse(stored["managed_server_calls_made"])
        fingerprints = [row["outcome_fingerprint"] for row in stored["outcomes"]]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        for row in stored["outcomes"]:
            self.assertEqual(row["classification"], "no-data-preview-health")
            self.assertEqual(row["preview_status"], "no-data-preview-health")
            self.assertEqual(row["preview_reason"], "managed-preview-url-not-configured")
            self.assertEqual(row["fetch_status"], "skipped")
            self.assertEqual(row["next_action"], "refresh-managed-activation-preview")
            self.assertTrue(row["local_action_family"])
            self.assertTrue(row["handoff_ref"].startswith("handoff:"))
            self.assertTrue(row["privacy"]["metadata_only"])
            self.assertTrue(row["privacy"]["review_only"])
            self.assertFalse(row["privacy"]["provider_calls_made"])
            self.assertFalse(row["privacy"]["managed_server_calls_made"])
            self.assertFalse(row["privacy"]["policy_files_written"])
        self.assertEqual(
            result["routing_pathway_outcome_batch"]["reason"],
            "routing-pathway-outcome-batch-input-not-configured",
        )
        self.assertEqual(managed_egress_violations(stored), [])

    def test_unavailable_preview_refresh_persists_successor_queue_health_rows(self):
        with TemporaryDirectory() as tmpdir:
            store = Store(str(Path(tmpdir) / "agentflow.sqlite3"))
            try:
                first = persist_unavailable_managed_activation_preview_outcomes(
                    store,
                    _preview_required_successor_queue_plan(),
                    now=NOW,
                    reason="managed-preview-refresh-not-configured",
                )
                second = persist_unavailable_managed_activation_preview_outcomes(
                    store,
                    _preview_required_successor_queue_plan(),
                    now=NOW + timedelta(minutes=10),
                    reason="managed-preview-refresh-not-configured",
                )
            finally:
                store.conn.close()

        self.assertEqual(first["schema"], "agentflow.managed_activation_preview_outcomes.v1")
        self.assertEqual(first["refresh"]["schema"], "agentflow.managed_activation_preview_refresh_status.v1")
        self.assertEqual(first["refresh"]["status"], "unavailable")
        self.assertEqual(first["refresh"]["reason"], "managed-preview-refresh-not-configured")
        self.assertEqual(first["refresh"]["submitted_row_count"], 3)
        self.assertEqual(first["summary"]["stored_preview_outcome_count"], 3)
        self.assertEqual(first["summary"]["no_data_preview_health_count"], 3)
        self.assertEqual(first["import"]["created_count"], 3)
        self.assertEqual(second["import"]["created_count"], 0)
        self.assertEqual(second["import"]["updated_count"], 3)
        fingerprints = [row["outcome_fingerprint"] for row in first["outcomes"]]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        families = {row["local_action_family"] for row in first["outcomes"]}
        self.assertEqual(families, {"activation-feedback", "cache", "routing"})
        for row in first["outcomes"]:
            self.assertEqual(row["classification"], "no-data-preview-health")
            self.assertEqual(row["preview_status"], "no-data-preview-health")
            self.assertEqual(row["preview_reason"], "managed-preview-refresh-not-configured")
            self.assertEqual(row["next_action"], "refresh-managed-activation-preview")
            self.assertTrue(row["privacy"]["metadata_only"])
            self.assertTrue(row["privacy"]["aggregate_only"])
            self.assertTrue(row["privacy"]["review_only"])
            self.assertFalse(row["privacy"]["managed_server_calls_made"])
            self.assertFalse(row["privacy"]["provider_calls_made"])
            self.assertFalse(row["privacy"]["policy_files_written"])
        self.assertFalse(first["managed_server_calls_made"])
        self.assertEqual(managed_egress_violations(first), [])

    def test_managed_activation_preview_cli_posts_opt_in_payload(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            db_path = Path(tmpdir) / "agentflow.sqlite3"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            stdout = io.StringIO()
            response = Mock()
            response.status_code = 200
            response.text = "{}"

            def response_json():
                posted = post_mock.call_args.kwargs["json"]
                return {
                    "schema": "agentflow.managed_activation_preview_response.v1",
                    "decisions": [
                        {
                            "handoff_ref": posted["rows"][0]["handoff_ref"],
                            "decision": "no-op",
                            "no_op_reason": "full-rollout-policy-active",
                            "review_only": True,
                        }
                    ],
                }

            response.json.side_effect = response_json

            with patch("tokenclaw.cli.httpx.post", return_value=response) as post_mock:
                code = cli.managed_activation_preview_cli(
                    [
                        "--plan-json",
                        str(plan_path),
                        "--managed-preview-url",
                        "https://managed.example.test/v1/activation-preview",
                        "--persist-outcomes",
                        "--db",
                        str(db_path),
                        "--pretty",
                    ],
                    stdout=stdout,
                )

        self.assertEqual(code, 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(post_mock.call_count, 1)
        posted = post_mock.call_args.kwargs["json"]
        self.assertEqual(posted["schema"], "agentflow.managed_activation_preview_request.v1")
        self.assertEqual(posted["summary"]["handoff_row_count"], 4)
        self.assertEqual(managed_egress_violations(posted), [])
        self.assertEqual(result["status"], "previewed")
        self.assertTrue(result["coverage"]["managed_server_calls_made"])
        self.assertFalse(result["coverage"]["provider_calls_made"])
        self.assertFalse(result["coverage"]["policy_files_written"])
        self.assertEqual(result["coverage"]["preview_decision_count"], 1)
        self.assertEqual(result["coverage"]["matched_handoff_ref_count"], 1)
        self.assertEqual(result["summary"]["submitted_row_count"], 4)
        self.assertEqual(result["summary"]["preview_row_count"], 1)
        self.assertEqual(result["summary"]["omission_count"], 0)
        self.assertEqual(result["stored_preview_outcomes"]["import"]["imported_count"], 4)
        self.assertEqual(result["stored_preview_outcomes"]["summary"]["missing_preview_decision_count"], 3)
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertEqual(managed_egress_violations(result), [])

    def test_managed_activation_preview_cli_submits_pathway_outcomes_before_preview(self):
        with TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            pathway_path = Path(tmpdir) / "pathway.json"
            db_path = Path(tmpdir) / "agentflow.sqlite3"
            plan_path.write_text(json.dumps(_executor_handoff_fixture_plan()), encoding="utf-8")
            pathway_path.write_text(json.dumps(_routing_pathway_source()), encoding="utf-8")
            store = Store(str(db_path))
            try:
                _log_openai_pathway_canary_call(
                    store,
                    cohort="canary_applied",
                    created_at="2026-06-20T08:01:00+00:00",
                )
                _log_openai_pathway_canary_call(
                    store,
                    cohort="canary_holdout",
                    created_at="2026-06-20T08:02:00+00:00",
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()

            def fake_post(url, *, json, headers, timeout):
                response = Mock()
                response.status_code = 200
                response.text = "{}"
                if url.endswith("/v1/managed-routing-pathway-outcomes"):
                    response.json.return_value = {
                        "schema": "agentflow.managed_routing_pathway_outcomes_stored.v1",
                        "accepted": True,
                        "accepted_count": len(json["outcomes"]),
                    }
                    return response
                response.json.return_value = {
                    "schema": "agentflow.managed_activation_preview_response.v1",
                    "decisions": [
                        {
                            "handoff_ref": json["rows"][0]["handoff_ref"],
                            "decision": "no-op",
                            "no_op_reason": "full-rollout-policy-active",
                            "review_only": True,
                        }
                    ],
                }
                return response

            with patch("tokenclaw.cli.httpx.post", side_effect=fake_post) as post_mock:
                code = cli.managed_activation_preview_cli(
                    [
                        "--plan-json",
                        str(plan_path),
                        "--managed-preview-url",
                        "https://managed.example.test/v1/local-activation-outcome-policy-decision-previews",
                        "--routing-pathway-outcomes-json",
                        str(pathway_path),
                        "--persist-outcomes",
                        "--db",
                        str(db_path),
                        "--pretty",
                    ],
                    stdout=stdout,
                )

        self.assertEqual(code, 0)
        self.assertEqual(post_mock.call_count, 2)
        outcome_call = post_mock.call_args_list[0]
        preview_call = post_mock.call_args_list[1]
        self.assertEqual(
            outcome_call.args[0],
            "https://managed.example.test/v1/managed-routing-pathway-outcomes",
        )
        self.assertEqual(
            preview_call.args[0],
            "https://managed.example.test/v1/local-activation-outcome-policy-decision-previews",
        )
        outcome_payload = outcome_call.kwargs["json"]
        self.assertEqual(outcome_payload["schema"], "agentflow.managed_routing_pathway_outcomes.v1")
        self.assertEqual(outcome_payload["summary"]["outcome_count"], 1)
        self.assertEqual(outcome_payload["summary"]["applied_count"], 1)
        self.assertEqual(outcome_payload["summary"]["holdout_count"], 1)
        self.assertFalse(outcome_payload["provider_calls_made"])
        self.assertFalse(outcome_payload["policy_files_written"])
        self.assertEqual(managed_egress_violations(outcome_payload), [])

        result = json.loads(stdout.getvalue())
        batch = result["routing_pathway_outcome_batch"]
        self.assertEqual(batch["status"], "submitted")
        self.assertEqual(batch["reason"], "managed-routing-pathway-outcome-batch-submitted")
        self.assertEqual(batch["outcome_count"], 1)
        self.assertEqual(batch["submitted_outcome_count"], 1)
        self.assertTrue(batch["managed_server_calls_made"])
        self.assertEqual(result["summary"]["routing_pathway_outcome_batch_status"], "submitted")
        self.assertEqual(result["stored_preview_outcomes"]["summary"]["stored_preview_outcome_count"], 4)
        self.assertEqual(managed_egress_violations(result), [])

    def test_full_rollout_crunch_collapses_to_durable_keep_current_rule_outcome(self):
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry()),
            now=NOW,
        )

        self.assertEqual(result["summary"]["executor_entry_count"], 1)
        self.assertEqual(result["summary"]["selected_action_count"], 0)
        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "keep-current-rule")
        self.assertEqual(entry["executor_status"], "blocked-or-terminal")
        self.assertEqual(entry["executor_next_action"], "keep-current-rule-only")
        self.assertTrue(entry["durable_outcome_ledger_entry"])
        self.assertTrue(entry["measured_full_rollout_activation"])
        self.assertEqual(entry["full_rollout_outcome"], "keep-active")
        self.assertEqual(entry["full_rollout_successor_decision"], "no-op")
        self.assertEqual(entry["full_rollout_successor_no_op_reason"], "no-unsuppressed-post-full-rollout-crunch-cohort")
        self.assertFalse(entry["new_activation_issue_recommended"])
        self.assertEqual(entry["applied_count"], 107)
        self.assertEqual(entry["holdout_count"], 40)
        self.assertEqual(entry["safety_stop_count"], 0)
        self.assertEqual(entry["rollback_count"], 0)
        self.assertEqual(entry["observed_saved_tokens"], 8606129)
        self.assertAlmostEqual(entry["observed_savings_usd"], 25.818613)
        self.assertEqual(entry["duplicate_suppression_status"], "suppressed")
        self.assertEqual(entry["duplicate_suppression_reason"], "repeated-context-crunch-full-rollout-active")
        self.assertEqual(entry["keep_active_regression_gate"]["state"], "keep-active")
        self.assertTrue(entry["keep_active_regression_gate"]["gate_passed"])
        self.assertTrue(entry["privacy"]["metadata_only"])
        self.assertTrue(entry["privacy"]["aggregate_only"])

    def test_full_rollout_crunch_duplicate_successors_collapse_to_single_executor_outcome(self):
        entry = _full_rollout_queue_entry()
        duplicate = dict(entry)
        duplicate.pop("full_rollout_activation_outcome")
        duplicate.pop("keep_active_regression_gate")
        duplicate["fingerprint"] = "activation:duplicate-rollup"
        duplicate["lever"] = "request-shape-rollups"
        duplicate["evidence_schema"] = "agentflow.request_shape_follow_up_candidates.v1"
        plan = _full_rollout_queue_plan(entry)
        plan["entries"].append(duplicate)

        result = build_local_activation_executor_plan(plan, now=NOW)

        self.assertEqual(result["summary"]["executor_entry_count"], 1)
        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "keep-current-rule")
        self.assertEqual(entry["executor_next_action"], "keep-current-rule-only")
        self.assertEqual(entry["source_fingerprint"], "activation:f5f6eae5f0a0081a")
        self.assertEqual(entry["full_rollout_outcome"], "keep-active")
        self.assertEqual(entry["keep_active_regression_gate"]["state"], "keep-active")

    def test_full_rollout_crunch_stale_evidence_emits_review_next_action(self):
        gate = _full_rollout_gate(
            state="review-stale-evidence",
            next_action="refresh-full-rollout-repeated-context-crunch-evidence",
            reason_codes=["stale-evidence"],
        )
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry(gate=gate, outcome="review-stale-evidence")),
            now=NOW,
        )

        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "review-only")
        self.assertEqual(entry["executor_status"], "selected")
        self.assertEqual(entry["executor_next_action"], "refresh-full-rollout-repeated-context-crunch-evidence")
        self.assertEqual(entry["full_rollout_outcome"], "review-stale-evidence")
        self.assertIn("stale-evidence", entry["reason_codes"])
        self.assertFalse(entry["keep_active_regression_gate"]["gate_passed"])

    def test_full_rollout_crunch_regression_emits_rollback_next_action(self):
        gate = _full_rollout_gate(
            state="rollback-required",
            next_action="rollback-full-rollout-repeated-context-crunch-rule",
            reason_codes=["rollback-observed", "fallback-rate-regression"],
        )
        gate["regression_counters"]["rollback_count"] = 1
        gate["regression_counters"]["fallback_count"] = 1
        gate["regression_counters"]["fallback_rate_delta"] = 0.02
        result = build_local_activation_executor_plan(
            _full_rollout_queue_plan(_full_rollout_queue_entry(gate=gate, outcome="rollback-required")),
            now=NOW,
        )

        entry = result["entries"][0]
        self.assertEqual(entry["executor_action_class"], "rollback-required")
        self.assertEqual(entry["executor_status"], "blocked-or-terminal")
        self.assertEqual(entry["executor_next_action"], "rollback-full-rollout-repeated-context-crunch-rule")
        self.assertEqual(entry["full_rollout_outcome"], "rollback-required")
        self.assertEqual(entry["rollback_count"], 1)
        self.assertIn("rollback-observed", entry["reason_codes"])
        self.assertIn("fallback-rate-regression", entry["reason_codes"])
        self.assertEqual(entry["keep_active_regression_gate"]["regression_counters"]["rollback_count"], 1)


if __name__ == "__main__":
    unittest.main()
