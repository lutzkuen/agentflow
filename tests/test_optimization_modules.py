import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import yaml

from agentflow_proxy.activation_lifecycle_feedback import LIFECYCLE_SOURCE_SURFACE, build_activation_staged_lifecycle_feedback
from agentflow_proxy import recommendations
from agentflow_proxy.managed_egress import managed_egress_violations
from agentflow_proxy.optimization import feedback, openai_outcomes
from agentflow_proxy.optimization_eval_plan import _add_common
from agentflow_proxy.optimization_eval_queue import (
    backfill_promotion_eval_tasks,
    queue_promotion_recommendation_eval_tasks,
    run_optimization_eval_queue,
)
from agentflow_proxy.openai_canary_impact import build_openai_canary_impact_report
from agentflow_proxy.optimization_promotion_canary import (
    apply_optimization_promotion_canaries,
    evaluate_promotion_canary_safety_stop,
    promotion_canary_decision,
)
from agentflow_proxy.optimization_promotion_actions import build_optimization_promotion_actions
from agentflow_proxy.optimization_promotion_impact import measure_optimization_promotion_impact
from agentflow_proxy.optimization_promotion_report import build_optimization_promotion_report
from agentflow_proxy.promotion_outcome_feedback import (
    promotion_outcome_feedback_summary,
    record_promotion_outcome_feedback,
)
from agentflow_proxy.promotion_blocker_review import build_promotion_blocker_recommendation_review
from agentflow_proxy.optimization_rollout_review import (
    attach_optimization_rollout_provenance,
    review_optimization_rollout_actions,
)
from agentflow_proxy.cli import optimization_rollout_actions_apply_cli
from agentflow_proxy.optimization_shadow_eval import run_optimization_shadow_eval
from agentflow_proxy.stats import stats_optimization_promotion_funnel
from agentflow_proxy.store import Store, stable_json, utc_now


class FakeFeedbackStore:
    def managed_outcome_feedback_summary(self, *, source_surface=None):
        self.source_surface = source_surface
        return [{"status": "queued", "count": 1}]

    def managed_outcome_feedback_rows(self, *, source_surface=None, limit=10000):
        self.row_limit = limit
        return [
            {
                "id": "queue-1",
                "created_at": "2026-06-08T10:00:00+00:00",
                "updated_at": "2026-06-08T10:00:01+00:00",
                "source_surface": "openai_responses",
                "endpoint": "/v1/optimization-units/77/outcome",
                "optimization_unit_id": 77,
                "status": "queued",
                "attempts": 0,
                "next_attempt_at": "2026-06-08T10:00:02+00:00",
                "last_status_code": None,
                "sent_at": None,
            }
        ]

    def due_managed_outcome_feedback(self, *, limit, source_surface=None):
        self.due_limit = limit
        return self.managed_outcome_feedback_rows(source_surface=source_surface, limit=limit)

    def managed_outcome_feedback_payload_rows(self, *, source_surface=None, limit=10000):
        return []


class FakeOutcomeStore:
    def __init__(self):
        self.updated = []

    def update_call_routing_json(self, call_id, routing_json):
        self.updated.append((call_id, json.loads(routing_json)))


class OptimizationModuleTests(unittest.TestCase):
    def _assert_privacy_clean(self, payload):
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "raw eval prompt secret",
            "raw eval message secret",
            "raw eval content secret",
            "raw eval request secret",
            "raw eval response secret",
            "raw eval output secret",
            "raw eval tool secret",
            "/tmp/raw-eval-secret.py",
            "eval-cache-key-secret",
            "eval-request-id-secret",
            "eval-session-id-secret",
            "sk-eval-secret",
        ):
            self.assertNotIn(forbidden, rendered)
        for forbidden_key in (
            '"api_key"',
            '"cache_key"',
            '"content"',
            '"file_path"',
            '"messages"',
            '"output"',
            '"prompt"',
            '"raw_request"',
            '"raw_response"',
            '"request_id"',
            '"session_id"',
            '"tool_payload"',
        ):
            self.assertNotIn(forbidden_key, rendered)

    def _dangerous_metadata(self):
        return {
            "safe_count": 3,
            "safe_score": 0.98,
            "prompt": "raw eval prompt secret",
            "messages": [{"role": "user", "content": "raw eval message secret"}],
            "content": "raw eval content secret",
            "raw_request": {"body": "raw eval request secret"},
            "raw_response": {"output": "raw eval response secret"},
            "tool_payload": {"command": "raw eval tool secret"},
            "file_path": "/tmp/raw-eval-secret.py",
            "cache_key": "eval-cache-key-secret",
            "request_id": "eval-request-id-secret",
            "session_id": "eval-session-id-secret",
            "api_key": "sk-eval-secret",
            "nested": {
                "safe_reason": "offline-fixture-passed",
                "prompt": "raw eval prompt secret",
                "content": "raw eval content secret",
            },
        }

    def test_eval_plan_rows_scrub_raw_like_nested_evidence(self):
        rows = []
        _add_common(
            rows,
            candidate_id="eval-privacy-candidate",
            optimization_family="phase_routing",
            action_family="routing",
            source_surface="anthropic_messages",
            app_family="claude_code",
            workflow_phase="tool_execution",
            category="tool-result",
            candidate_target_model="claude-haiku-4-5-20251001",
            projected_savings_usd=0.0123,
            sample_count=5,
            current_canary_count=2,
            holdout_count=1,
            blocker_reason_codes=["local-replay-input-unavailable"],
            replayability_level="features_only",
            evidence=self._dangerous_metadata(),
        )

        row = rows[0]
        self.assertEqual(row["candidate_id"], "eval-privacy-candidate")
        self.assertEqual(row["sample_count"], 5)
        self.assertEqual(row["blocker_reason_codes"], ["local-replay-input-unavailable"])
        self.assertEqual(row["evidence"]["safe_count"], 3)
        self.assertEqual(row["evidence"]["nested"]["safe_reason"], "offline-fixture-passed")
        self.assertTrue(row["privacy"]["metadata_only"])
        self._assert_privacy_clean(row)

    def test_shadow_eval_output_and_stored_result_scrub_raw_like_fixture_fields(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "candidate_id": "shadow-privacy-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "projected_savings_usd": 0.02,
                    "shadow_eval_fixture": {
                        **self._dangerous_metadata(),
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.98,
                        "quality_score": 0.97,
                        "baseline_cost_usd": 0.03,
                        "candidate_cost_usd": 0.01,
                    },
                    "request_json": {"messages": [{"content": "raw eval message secret"}]},
                    "session_id": "eval-session-id-secret",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                result = run_optimization_shadow_eval(plan, store=store)
                stored = store.conn.execute(
                    "select result_json, score_json, cost_json from optimization_eval_results"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(result["results"][0]["candidate_id"], "shadow-privacy-candidate")
        self.assertEqual(result["results"][0]["status_class"], "pass")
        self.assertIn("offline-fixture-passed", result["results"][0]["reason_codes"])
        self.assertEqual(result["results"][0]["score_summary"]["output_similarity"], 0.98)
        self.assertEqual(json.loads(stored["score_json"])["quality_score"], 0.97)
        self.assertEqual(json.loads(stored["cost_json"])["observed_savings_usd"], 0.02)
        self._assert_privacy_clean(result)
        self._assert_privacy_clean(json.loads(stored["result_json"]))

    def test_eval_queue_runs_bounded_mixed_batch_without_provider_calls(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "generated_at": "2026-06-10T02:00:00+00:00",
            "plans": [
                {
                    "candidate_id": "queue-pass",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "candidate_created_at": "2026-06-10T01:00:00+00:00",
                    "projected_savings_usd": 0.09,
                    "shadow_eval_fixture": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.97,
                        "baseline_cost_usd": 0.10,
                        "candidate_cost_usd": 0.01,
                        "prompt": "raw queue pass prompt secret",
                    },
                },
                {
                    "candidate_id": "queue-blocked",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "features_only",
                    "candidate_created_at": "2026-06-10T01:30:00+00:00",
                    "projected_savings_usd": 0.05,
                    "blocker_reason_codes": ["tool-call-disabled"],
                },
                {
                    "candidate_id": "queue-stale",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "candidate_created_at": "2026-06-08T00:00:00+00:00",
                    "projected_savings_usd": 0.04,
                    "shadow_eval_fixture": {
                        "baseline_status_code": 200,
                        "candidate_status_code": 200,
                        "output_similarity": 0.99,
                    },
                },
                {
                    "candidate_id": "queue-other-family",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "candidate_created_at": "2026-06-10T01:45:00+00:00",
                    "projected_savings_usd": 1.0,
                    "shadow_eval_fixture": {"baseline_status_code": 200, "candidate_status_code": 200},
                },
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                result = asyncio.run(
                    run_optimization_eval_queue(
                        store,
                        plan=plan,
                        family="phase_routing",
                        limit=3,
                        max_candidate_age_hours=24,
                        now=datetime(2026, 6, 10, 2, 0, tzinfo=timezone.utc),
                    )
                )
                stored = store.conn.execute(
                    "select candidate_id, status_class, reason_codes_json, result_json "
                    "from optimization_eval_results order by candidate_id"
                ).fetchall()
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "agentflow.optimization_eval_queue_run.v1")
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertTrue(result["wrote_result_records"])
        self.assertEqual(result["summary"]["input_candidate_count"], 4)
        self.assertEqual(result["summary"]["family_filtered_count"], 1)
        self.assertEqual(result["summary"]["selected_candidate_count"], 3)
        by_candidate = {row["candidate_id"]: row for row in result["results"]}
        self.assertEqual(by_candidate["queue-pass"]["status_class"], "pass")
        self.assertEqual(by_candidate["queue-blocked"]["status_class"], "blocked")
        self.assertEqual(by_candidate["queue-stale"]["status_class"], "blocked")
        self.assertIn("tool-call-disabled", by_candidate["queue-blocked"]["reason_codes"])
        self.assertIn("candidate-stale", by_candidate["queue-stale"]["reason_codes"])
        self.assertEqual({row["candidate_id"] for row in stored}, {"queue-pass", "queue-blocked", "queue-stale"})
        stored_stale = next(row for row in stored if row["candidate_id"] == "queue-stale")
        self.assertEqual(stored_stale["status_class"], "blocked")
        self.assertIn("candidate-stale", json.loads(stored_stale["reason_codes_json"]))
        self._assert_privacy_clean(result)
        self._assert_privacy_clean(json.loads(stored_stale["result_json"]))

    def test_promotion_eval_backfill_queues_needs_eval_candidates_and_updates_report_status(self):
        promotion_report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "promotion-eval-high",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.25,
                    "sample_count": 12,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing", "insufficient-eval-pass-results"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "thresholds": {"min_eval_pass_count": 1},
                    "prompt": "raw eval prompt secret",
                    "request_id": "eval-request-id-secret",
                    "session_id": "eval-session-id-secret",
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "promotion-eval-low",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.05,
                    "sample_count": 3,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "promotion-already-widen",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "projected_savings_usd": 1.0,
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
            ],
        }
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "promotion-eval-high",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "current_canary_count": 2,
                    "holdout_count": 1,
                    "sample_count": 3,
                    "projected_savings_usd": 0.25,
                    "evidence": {
                        "canary_evidence": {
                            "applied": {"count": 2, "error_rate": 0.0, "retry_rate": 0.0, "latency_avg_ms": 100, "net_savings_usd": 0.25},
                            "holdout": {"count": 1, "error_rate": 0.0, "retry_rate": 0.0, "latency_avg_ms": 100},
                        }
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                dry_run = backfill_promotion_eval_tasks(store, promotion_report, limit=1, apply=False)
                dry_count = store.conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
                applied = backfill_promotion_eval_tasks(store, promotion_report, limit=1, apply=True)
                stored = store.conn.execute(
                    "select candidate_id, status_class, reason_codes_json, result_json from optimization_eval_results"
                ).fetchall()
                updated_report = build_optimization_promotion_report(
                    store,
                    plan=plan,
                    min_canary_applied_samples=2,
                    min_canary_holdout_samples=1,
                )
            finally:
                store.conn.close()

        self.assertEqual(dry_run["schema"], "agentflow.optimization_promotion_eval_backfill.v1")
        self.assertTrue(dry_run["dry_run"])
        self.assertFalse(dry_run["wrote_eval_queue_rows"])
        self.assertEqual(dry_count, 0)
        self.assertEqual([task["candidate_id"] for task in dry_run["tasks"]], ["promotion-eval-high"])
        self.assertTrue(any(row["candidate_id"] == "promotion-eval-low" and row["reason"] == "limit-exceeded" for row in dry_run["skipped"]))
        self.assertFalse(applied["dry_run"])
        self.assertTrue(applied["wrote_eval_queue_rows"])
        self.assertEqual(applied["summary"]["written_task_count"], 1)
        self.assertEqual(stored[0]["candidate_id"], "promotion-eval-high")
        self.assertEqual(stored[0]["status_class"], "queued")
        self.assertIn("eval-queued", json.loads(stored[0]["reason_codes_json"]))
        candidate = updated_report["candidates"][0]
        self.assertEqual(candidate["eval_evidence"]["queued_count"], 1)
        self.assertEqual(candidate["verdict"], "needs_eval")
        self.assertIn("eval-queued", candidate["reason_codes"])
        self.assertIn("insufficient-eval-pass-results", candidate["reason_codes"])
        self.assertNotIn("eval-results-missing", candidate["reason_codes"])
        self._assert_privacy_clean(dry_run)
        self._assert_privacy_clean(applied)
        self._assert_privacy_clean(json.loads(stored[0]["result_json"]))

    def test_promotion_report_omits_raw_like_plan_and_eval_result_fields(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "promotion-privacy-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "current_canary_count": 2,
                    "holdout_count": 1,
                    "sample_count": 3,
                    "projected_savings_usd": 0.02,
                    "evidence": {
                        **self._dangerous_metadata(),
                        "canary_evidence": {
                            "applied": {
                                "count": 2,
                                "error_rate": 0.0,
                                "retry_rate": 0.0,
                                "latency_avg_ms": 100,
                                "net_savings_usd": 0.02,
                                "tool_payload": {"command": "raw eval tool secret"},
                            },
                            "holdout": {
                                "count": 1,
                                "error_rate": 0.0,
                                "retry_rate": 0.0,
                                "latency_avg_ms": 120,
                                "prompt": "raw eval prompt secret",
                            },
                        },
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                store.log_optimization_eval_result(
                    id="promotion-privacy-result",
                    run_id="promotion-privacy-run",
                    created_at="2026-06-10T03:30:00+00:00",
                    candidate_id="promotion-privacy-candidate",
                    source_surface="anthropic_messages",
                    optimization_family="phase_routing",
                    action_family="routing",
                    status_class="pass",
                    reason_codes_json=stable_json(["offline-fixture-passed"]),
                    score_json=stable_json({"output_similarity": 0.98, "content": "raw eval content secret"}),
                    cost_json=stable_json({"projected_savings_usd": 0.02, "cache_key": "eval-cache-key-secret"}),
                    result_json=stable_json(self._dangerous_metadata()),
                )
                report = build_optimization_promotion_report(store, plan=plan)
            finally:
                store.conn.close()

        candidate = report["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "promotion-privacy-candidate")
        self.assertEqual(candidate["verdict"], "widen")
        self.assertIn("promotion-thresholds-met", candidate["reason_codes"])
        self.assertEqual(candidate["eval_evidence"]["pass_count"], 1)
        self.assertEqual(candidate["cohort_counts"]["canary_applied"], 2)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self._assert_privacy_clean(report)

    def test_promotion_report_uses_activation_lifecycle_evidence_to_clear_canary_blockers(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "activation-routing-candidate",
                    "optimization_family": "openai_local_routing",
                    "action_family": "routing",
                    "source_surface": "openai_provider_request",
                    "app_family": "generic_openai",
                    "granularity": "provider_request",
                    "replayability_level": "features_only",
                    "candidate_target_model": "gpt-5-mini",
                    "current_canary_count": 0,
                    "holdout_count": 0,
                    "sample_count": 3,
                    "projected_savings_usd": 0.03,
                    "blocker_reason_codes": [
                        "insufficient-canary-applied-samples",
                        "insufficient-canary-holdout-samples",
                    ],
                    "evidence": self._dangerous_metadata(),
                },
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "activation-cache-candidate",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "granularity": "provider_request",
                    "replayability_level": "local-exact-response",
                    "current_canary_count": 0,
                    "holdout_count": 0,
                    "sample_count": 3,
                    "projected_savings_usd": 0.02,
                    "blocker_reason_codes": [
                        "insufficient-canary-applied-samples",
                        "insufficient-canary-holdout-samples",
                    ],
                    "evidence": self._dangerous_metadata(),
                },
            ],
        }

        def lifecycle_payload(candidate_id: str, family: str, section: str, savings: float) -> dict:
            result = {
                "schema": "agentflow.optimization_promotion_rollout_actions.v1",
                "ok": True,
                "generated_at": "2026-06-10T03:00:00+00:00",
                "summary": {"projected_savings_usd": savings},
                "actions": [
                    {
                        "action_family": family,
                        "policy_section": section,
                        "status": "applied",
                        "target_candidate_id": candidate_id,
                        "target_rule_id": f"{candidate_id}-rule",
                        "projected_savings_usd": savings,
                        "reason_codes": ["activation-fixture-applied"],
                    },
                    {
                        "action_family": family,
                        "policy_section": section,
                        "status": "applied",
                        "target_candidate_id": candidate_id,
                        "target_rule_id": f"{candidate_id}-rule",
                        "projected_savings_usd": savings,
                        "reason_codes": ["activation-fixture-applied"],
                    },
                    {
                        "action_family": family,
                        "policy_section": section,
                        "status": "holdout",
                        "target_candidate_id": candidate_id,
                        "target_rule_id": f"{candidate_id}-rule",
                        "projected_savings_usd": 0,
                        "reason_codes": ["activation-fixture-holdout"],
                    },
                ],
            }
            payload = build_activation_staged_lifecycle_feedback(
                result,
                event_phase="apply",
                command="fixture-activation-apply",
            )
            assert payload is not None
            return payload

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                for candidate_id, family, section, savings in (
                    ("activation-routing-candidate", "routing", "routing", 0.03),
                    ("activation-cache-candidate", "cache", "cache", 0.02),
                ):
                    store.enqueue_managed_outcome_feedback(
                        id=f"activation-lifecycle-{candidate_id}",
                        source_surface=LIFECYCLE_SOURCE_SURFACE,
                        endpoint="/v1/policy-feedback",
                        optimization_unit_id=0,
                        payload_json=stable_json(lifecycle_payload(candidate_id, family, section, savings)),
                        status="queued",
                    )
                    store.log_optimization_eval_result(
                        id=f"activation-eval-{candidate_id}",
                        run_id="activation-eval-run",
                        created_at="2026-06-10T03:30:00+00:00",
                        candidate_id=candidate_id,
                        source_surface="openai_provider_request" if family == "routing" else "anthropic_messages",
                        optimization_family="openai_local_routing" if family == "routing" else "cache_replayability",
                        action_family=family,
                        status_class="pass",
                        reason_codes_json=stable_json(["offline-fixture-passed"]),
                        score_json=stable_json({"output_similarity": 0.98, "quality_score": 0.97}),
                        cost_json=stable_json({"projected_savings_usd": savings}),
                        result_json=stable_json(self._dangerous_metadata()),
                    )
                report = build_optimization_promotion_report(store, plan=plan)
            finally:
                store.conn.close()

        by_candidate = {row["candidate_id"]: row for row in report["candidates"]}
        for candidate_id in ("activation-routing-candidate", "activation-cache-candidate"):
            candidate = by_candidate[candidate_id]
            self.assertEqual(candidate["verdict"], "widen")
            self.assertEqual(candidate["cohort_counts"]["canary_applied"], 2)
            self.assertEqual(candidate["cohort_counts"]["canary_holdout"], 1)
            self.assertNotIn("insufficient-canary-applied-samples", candidate["reason_codes"])
            self.assertNotIn("insufficient-canary-holdout-samples", candidate["reason_codes"])
            self.assertEqual(candidate["evidence_sources"][0]["source"], "activation_lifecycle_feedback")

        source_counts = {row["value"]: row["count"] for row in report["summary"]["evidence_source_counts"]}
        self.assertEqual(source_counts["activation_lifecycle_feedback"], 2)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self._assert_privacy_clean(report)

    def _log_openai_canary_call(
        self,
        store,
        *,
        candidate_id: str = "openai-canary-candidate",
        cohort: str,
        status_code: int = 200,
        retry_count: int = 0,
        latency_ms: int = 1000,
        cost_est: float = 0.001,
        cost_baseline: float = 0.003,
        projected_savings: float = 0.002,
        created_at: str = "2026-06-10T04:00:00+00:00",
        suffix: str = "",
        fallback_reason: str | None = None,
        action_id: str = "test-openai-canary-action",
        rule_id: str = "test-openai-canary-policy",
    ):
        status = "applied" if cohort == "canary_applied" else "holdout" if cohort == "canary_holdout" else "safety_stopped"
        canary = {
            "enabled": True,
            "policy_id": rule_id,
            "rule_id": rule_id,
            "promotion_action_id": action_id,
            "target_candidate_id": candidate_id,
            "candidate_id": candidate_id,
            "status": status,
            "cohort": cohort,
            "reason": "selected-canary" if status == "applied" else "selected-holdout" if status == "holdout" else "safety-stop-tripped",
            "original_model": "gpt-5-codex",
            "requested_model": "gpt-5-codex",
            "target_model": "gpt-5-mini",
            "actual_forwarded_model": "gpt-5-mini" if status == "applied" and not fallback_reason else "gpt-5-codex",
            "source_surface": "openai_provider_request",
            "app_family": "generic_openai",
            "category": "chat",
            "text_bucket": "lt_2k",
            "token_bucket": "lt_2k",
            "projected_input_savings_usd": projected_savings,
            "canary_fraction": 0.5,
            "holdout_fraction": 0.25,
            "policy_source": "local-manual",
            "cohort_key_hash": f"sha256:test-{cohort}-{suffix}",
        }
        if fallback_reason:
            canary["fallback_reason"] = fallback_reason
            canary["fallback_model"] = "gpt-5-codex"
        if status == "safety_stopped":
            canary["safety_stop"] = {"tripped": True, "reason_codes": ["error-rate"]}
        store.log_call(
            id=f"openai-canary-impact-{cohort}-{suffix}",
            created_at=created_at,
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model=canary["actual_forwarded_model"],
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=500,
            output_tokens_est=100,
            actual_input_tokens=500,
            actual_output_tokens=100,
            cost_est_usd=cost_est,
            cost_baseline_usd=cost_baseline,
            crunch_json=stable_json({"changed": suffix.endswith("crunch")}),
            routing_json=stable_json({"openai_canary": canary}),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
            error='{"error":{"message":"raw provider response secret"}}' if status_code >= 400 else None,
            request_json=None,
            response_json=None,
            session_id="raw-openai-session-secret",
            category="chat",
            retry_count=retry_count,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )

    def test_openai_canary_impact_reports_verdicts_and_feeds_promotion_report(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [
                {
                    "schema": "agentflow.optimization_eval_plan_row.v1",
                    "candidate_id": "openai-canary-candidate",
                    "optimization_family": "openai_local_routing",
                    "action_family": "routing",
                    "source_surface": "openai_provider_request",
                    "app_family": "generic_openai",
                    "granularity": "provider_request",
                    "replayability_level": "features_only",
                    "candidate_target_model": "gpt-5-mini",
                    "sample_count": 3,
                    "projected_savings_usd": 0.006,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self._log_openai_canary_call(store, cohort="canary_applied", suffix="a1")
                self._log_openai_canary_call(store, cohort="canary_applied", suffix="a2-crunch", latency_ms=900)
                self._log_openai_canary_call(store, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                store.log_optimization_eval_result(
                    id="openai-canary-eval-pass",
                    run_id="openai-canary-run",
                    created_at="2026-06-10T04:05:00+00:00",
                    candidate_id="openai-canary-candidate",
                    source_surface="openai_provider_request",
                    optimization_family="openai_local_routing",
                    action_family="routing",
                    status_class="pass",
                    reason_codes_json=stable_json(["offline-fixture-passed"]),
                    score_json=stable_json({"output_similarity": 0.98, "quality_score": 0.96}),
                    cost_json=stable_json({"projected_savings_usd": 0.006}),
                    result_json=stable_json({"metadata_only": True}),
                )
                impact = build_openai_canary_impact_report(
                    store,
                    limit=10,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                )
                promotion = build_optimization_promotion_report(store, plan=plan, evidence_reports=[impact])
            finally:
                store.conn.close()

        self.assertEqual(impact["schema"], "agentflow.openai_canary_impact.v1")
        self.assertTrue(impact["read_only"])
        self.assertFalse(impact["provider_calls_made"])
        self.assertEqual(impact["summary"]["observed_openai_canary_metadata_row_count"], 3)
        candidate = impact["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "openai-canary-candidate")
        self.assertEqual(candidate["verdict"], "widen")
        self.assertIn("target-savings-met", candidate["reason_codes"])
        self.assertEqual(candidate["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(candidate["cohort_counts"]["canary_holdout"], 1)
        self.assertEqual(candidate["crunch_interaction_counts"][0]["value"], "unchanged")
        self.assertFalse(impact["privacy"]["raw_prompts_included"])
        self.assertFalse(impact["privacy"]["request_ids_included"])
        self._assert_privacy_clean(impact)

        promoted = promotion["candidates"][0]
        self.assertEqual(promoted["candidate_id"], "openai-canary-candidate")
        self.assertEqual(promoted["verdict"], "widen")
        self.assertEqual(promoted["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(promoted["cohort_counts"]["canary_holdout"], 1)
        self.assertIn("target-savings-met", promoted["reason_codes"])
        self._assert_privacy_clean(promotion)

    def test_openai_canary_impact_verdicts_cover_regression_stale_and_insufficient(self):
        scenarios = (
            ("insufficient", "needs_eval", "insufficient-holdout-samples"),
            ("error", "hold", "error-rate-regression"),
            ("retry", "hold", "retry-rate-regression"),
            ("latency", "hold", "latency-regression"),
            ("stale", "hold", "stale-evidence"),
        )
        for scenario, expected_verdict, expected_reason in scenarios:
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as tmp:
                    store = Store(str(Path(tmp) / "agentflow.sqlite3"))
                    try:
                        if scenario == "insufficient":
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", suffix="a1")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", suffix="a2")
                        elif scenario == "error":
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", status_code=500, cost_est=0.001, cost_baseline=0.003, suffix="a1")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", suffix="a2")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        elif scenario == "retry":
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", retry_count=1, suffix="a1")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", suffix="a2")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        elif scenario == "latency":
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", latency_ms=5000, suffix="a1")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", latency_ms=6000, suffix="a2")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_holdout", latency_ms=1000, cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        else:
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", created_at="2026-06-01T00:00:00+00:00", suffix="a1")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_applied", created_at="2026-06-01T00:00:01+00:00", suffix="a2")
                            self._log_openai_canary_call(store, candidate_id=f"candidate-{scenario}", cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, created_at="2026-06-01T00:00:02+00:00", suffix="h1")
                        report = build_openai_canary_impact_report(
                            store,
                            limit=10,
                            since="2026-06-01T00:00:00+00:00" if scenario == "stale" else None,
                            min_applied_samples=2,
                            min_holdout_samples=1,
                            max_evidence_age_hours=1,
                            max_error_rate_delta=0.10,
                            max_retry_rate_delta=0.10,
                            max_latency_regression_ms=2000,
                            rollback_error_rate=1.0,
                            now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                        )
                    finally:
                        store.conn.close()
                candidate = report["candidates"][0]
                self.assertEqual(candidate["verdict"], expected_verdict)
                self.assertIn(expected_reason, candidate["reason_codes"])
                self._assert_privacy_clean(report)

    def test_promotion_actions_emit_local_rollout_actions_and_explicit_omissions(self):
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "routing-action-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "candidate_target_model": "claude-haiku-4-5-20251001",
                    "projected_savings_usd": 0.05,
                    "sample_count": 3,
                    "cohort_counts": {"canary_applied": 2, "canary_holdout": 1, "bypassed_or_disabled": 0},
                    "eval_evidence": {
                        "result_count": 1,
                        "pass_count": 1,
                        "fail_count": 0,
                        "blocked_count": 0,
                        "latest_result_at": "2026-06-10T03:30:00+00:00",
                        "score_summary": {"avg_output_similarity": 0.98, "avg_quality_score": 0.97},
                    },
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"raw_prompts_included": False},
                },
                {
                    "candidate_id": "cache-action-candidate",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "candidate_profile": "replay-safe-exact-candidate",
                    "projected_savings_usd": 0.03,
                    "sample_count": 4,
                    "cohort_counts": {"canary_applied": 2, "canary_holdout": 2, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0, "blocked_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"raw_prompts_included": False},
                },
                {
                    "candidate_id": "blocked-action-candidate",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.01,
                    "sample_count": 0,
                    "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 0, "pass_count": 0, "fail_count": 0, "blocked_count": 0},
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing", "insufficient-eval-pass-results"],
                    "privacy": {"raw_prompts_included": False},
                },
            ],
        }

        result = build_optimization_promotion_actions(report, widen_step=0.25, holdout_fraction=0.1)

        self.assertEqual(result["schema"], "agentflow.optimization_promotion_rollout_actions.v1")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertEqual(result["summary"]["action_count"], 2)
        self.assertEqual(result["summary"]["omitted_count"], 1)
        by_candidate = {row["target_candidate_id"]: row for row in result["actions"]}
        routing = by_candidate["routing-action-candidate"]
        cache = by_candidate["cache-action-candidate"]
        self.assertEqual(routing["policy_section"], "routing")
        self.assertEqual(routing["target_local_policy_section"], "routing.rules")
        self.assertEqual(routing["action_type"], "widen")
        self.assertEqual(routing["canary_fraction"], 0.916667)
        self.assertEqual(routing["holdout_fraction"], 0.1)
        self.assertEqual(routing["evidence_summary"]["eval_pass_count"], 1)
        self.assertEqual(cache["policy_section"], "cache")
        self.assertEqual(cache["local_policy_update"]["candidate_profile"], "replay-safe-exact-candidate")
        self.assertEqual(result["omitted"][0]["target_candidate_id"], "blocked-action-candidate")
        self.assertEqual(result["omitted"][0]["reason"], "insufficient-eval-evidence")
        self._assert_privacy_clean(result)

    def test_promotion_actions_rank_aggregate_omission_buckets_without_candidate_ids(self):
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "blocked-cache-high",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.12,
                    "sample_count": 12,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing", "insufficient-eval-pass-results"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "blocked-cache-low",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.03,
                    "sample_count": 4,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "blocked-routing-holdout",
                    "optimization_family": "openai_local_routing",
                    "action_family": "routing",
                    "source_surface": "openai_provider_request",
                    "app_family": "generic_openai",
                    "projected_savings_usd": 0.08,
                    "sample_count": 8,
                    "verdict": "needs_eval",
                    "reason_codes": ["insufficient-canary-holdout-samples"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "blocked-cache-stale",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.01,
                    "sample_count": 2,
                    "verdict": "needs_eval",
                    "reason_codes": ["stale-evidence", "dependency-freshness-missing"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
            ],
        }

        result = build_optimization_promotion_actions(report)

        self.assertEqual(result["summary"]["action_count"], 0)
        self.assertEqual(result["summary"]["omitted_count"], 4)
        self.assertEqual(result["summary"]["omission_bucket_count"], 3)
        buckets = result["omission_buckets"]
        top = buckets[0]
        self.assertEqual(top["action_family"], "cache")
        self.assertEqual(top["optimization_family"], "cache-replayability")
        self.assertEqual(top["source_surface"], "anthropic-messages")
        self.assertEqual(top["candidate_count"], 2)
        self.assertEqual(top["sample_count"], 16)
        self.assertEqual(top["projected_savings_usd"], 0.15)
        self.assertEqual(top["next_action"], "run-local-shadow-eval")
        self.assertEqual(top["top_reason_codes"][:2], ["eval-results-missing", "insufficient-eval-pass-results"])
        next_actions = {bucket["next_action"] for bucket in buckets}
        self.assertIn("collect-canary-holdout", next_actions)
        self.assertIn("fix-dependency-freshness", next_actions)
        encoded_buckets = json.dumps(buckets, sort_keys=True)
        for candidate_id in (
            "blocked-cache-high",
            "blocked-cache-low",
            "blocked-routing-holdout",
            "blocked-cache-stale",
        ):
            self.assertNotIn(candidate_id, encoded_buckets)
        self.assertTrue(all(bucket["privacy"]["metadata_only"] for bucket in buckets))
        self._assert_privacy_clean(result)

    def test_cache_promotion_actions_apply_dependency_gated_replay_canary(self):
        pattern_hash = "sha256:" + ("e" * 64)
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "stable-cache-replay-candidate",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "openai_responses",
                    "app_family": "generic_openai",
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "text_bucket": "8k_32k_chars",
                    "token_bucket": "1k_4k_tokens",
                    "has_tools": True,
                    "stream": False,
                    "pattern_hash": pattern_hash,
                    "replayability_level": "local-exact-response",
                    "file_dependency_status": "stable",
                    "safe_invalidation_evidence": True,
                    "file_dependency_evidence_available": True,
                    "matched_count": 10,
                    "projected_hit_count": 4,
                    "projected_hit_rate": 0.4,
                    "projected_savings_usd": 0.042,
                    "sample_count": 10,
                    "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0, "blocked_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                }
            ],
        }

        result = build_optimization_promotion_actions(report, initial_canary_fraction=0.1, holdout_fraction=0.2)

        self.assertEqual(result["summary"]["action_count"], 1)
        self.assertEqual(result["summary"]["omitted_count"], 0)
        action = result["actions"][0]
        local_update = action["local_policy_update"]
        self.assertEqual(action["policy_section"], "cache")
        self.assertEqual(action["canary_fraction"], 0.1)
        self.assertEqual(action["holdout_fraction"], 0.2)
        self.assertTrue(local_update["action"]["allow_tool_calls"])
        self.assertTrue(local_update["action"]["safe_invalidation_evidence"])
        self.assertEqual(local_update["action"]["projected_hit_count"], 4)
        self.assertEqual(local_update["action"]["projected_hit_rate"], 0.4)
        gate = local_update["cache_replay_canary"]["dependency_gate"]
        self.assertEqual(gate["status"], "ready")
        self.assertEqual(gate["reason"], "dependency-stable")
        self.assertTrue(gate["safe_invalidation_evidence"])
        self.assertIn("cache-hit", local_update["cache_replay_canary"]["feedback_outcomes"])
        self.assertIn("canary-holdout", local_update["cache_replay_canary"]["feedback_outcomes"])
        self.assertIn("stale-risk", local_update["cache_replay_canary"]["feedback_outcomes"])
        self.assertFalse(local_update["cache_replay_canary"]["dependency_path_values_included"])
        self.assertFalse(action["evidence_summary"]["cache_replay_dependency_gate"]["dependency_path_values_included"])

        with tempfile.TemporaryDirectory() as tmp:
            applied = apply_optimization_promotion_canaries(result, config_dir=tmp, dry_run=False)
            self.assertTrue(applied["ok"])
            written = yaml.safe_load((Path(tmp) / "cache_rules.yaml").read_text(encoding="utf-8"))

        rule = written["pattern_rules"][0]
        self.assertEqual(rule["conditions"]["pattern_hashes"], [pattern_hash])
        self.assertTrue(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertTrue(rule["action"]["allow_tool_calls"])
        self.assertTrue(rule["action"]["safe_invalidation_evidence"])
        self.assertEqual(rule["action"]["estimated_saved_cost_usd"], 0.042)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.1)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.2)
        self._assert_privacy_clean(result)

    def test_cache_promotion_actions_omit_stale_or_missing_dependency_replay(self):
        base = {
            "optimization_family": "cache_replayability",
            "action_family": "cache",
            "source_surface": "openai_responses",
            "app_family": "generic_openai",
            "category": "tool-result",
            "has_tools": True,
            "pattern_hash": "sha256:" + ("f" * 64),
            "matched_count": 6,
            "projected_hit_count": 3,
            "projected_savings_usd": 0.03,
            "sample_count": 6,
            "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
            "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0},
            "verdict": "widen",
            "privacy": {"metadata_only": True, "raw_prompts_included": False},
        }
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    **base,
                    "candidate_id": "stale-cache-replay-candidate",
                    "file_dependency_status": "invalidated",
                    "reason_codes": ["stale-risk-blockers"],
                },
                {
                    **base,
                    "candidate_id": "missing-cache-replay-candidate",
                    "file_dependency_status": "missing",
                    "reason_codes": ["file-dependency-missing"],
                },
            ],
        }

        result = build_optimization_promotion_actions(report)

        self.assertEqual(result["summary"]["action_count"], 0)
        self.assertEqual(result["summary"]["omitted_count"], 2)
        reasons = {row["target_candidate_id"]: row["reason"] for row in result["omitted"]}
        self.assertEqual(reasons["stale-cache-replay-candidate"], "cache-replay-stale-dependency-risk")
        self.assertEqual(reasons["missing-cache-replay-candidate"], "cache-replay-missing-invalidation-evidence")
        summaries = {row["target_candidate_id"]: row["evidence_summary"] for row in result["omitted"]}
        self.assertEqual(summaries["stale-cache-replay-candidate"]["cache_replay_dependency_gate"]["status"], "blocked")
        self.assertEqual(summaries["missing-cache-replay-candidate"]["cache_replay_dependency_gate"]["status"], "blocked")
        self.assertFalse(summaries["stale-cache-replay-candidate"]["cache_replay_dependency_gate"]["dependency_path_values_included"])
        next_actions = {bucket["next_action"] for bucket in result["omission_buckets"]}
        self.assertIn("fix-dependency-freshness", next_actions)
        self._assert_privacy_clean(result)

    def test_crunch_promotion_actions_require_positive_lifecycle_before_widening(self):
        pattern_hash = "sha256:" + ("a" * 64)
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "crunch-positive-lifecycle",
                    "optimization_family": "managed_pattern_candidate",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "pattern_hash": pattern_hash,
                    "projected_savings_usd": 0.06,
                    "sample_count": 12,
                    "cohort_counts": {"canary_applied": 7, "canary_holdout": 5, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 2, "pass_count": 2, "fail_count": 0, "blocked_count": 0},
                    "crunch_lifecycle": {
                        "tokens_before_est": 48000,
                        "tokens_after_est": 36000,
                        "tokens_saved_est": 12000,
                        "observed_savings_usd": 0.018,
                        "safety_status": "ok",
                    },
                    "conditions": {
                        "pattern_hashes": [pattern_hash],
                        "category": "tool-result",
                        "min_repeated_count": 2,
                    },
                    "action": {"type": "shorten", "head_chars": 800, "tail_chars": 600},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                    "prompt": "raw eval prompt secret",
                    "request_id": "eval-request-id-secret",
                    "session_id": "eval-session-id-secret",
                    "file_path": "/tmp/raw-eval-secret.py",
                },
                {
                    "candidate_id": "crunch-missing-measurement",
                    "optimization_family": "managed_pattern_candidate",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.04,
                    "sample_count": 6,
                    "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["missing-canary-lifecycle-evidence"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "crunch-safety-stop",
                    "optimization_family": "old_context_summarization",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.05,
                    "sample_count": 8,
                    "cohort_counts": {"canary_applied": 4, "canary_holdout": 4, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 2, "pass_count": 1, "fail_count": 1},
                    "crunch_lifecycle": {
                        "tokens_before_est": 32000,
                        "tokens_after_est": 26000,
                        "tokens_saved_est": 6000,
                        "safety_status": "safety-stopped",
                    },
                    "verdict": "widen",
                    "reason_codes": ["safety-stop-observed", "summary-failure-rate"],
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
            ],
        }

        result = build_optimization_promotion_actions(report, initial_canary_fraction=0.1, holdout_fraction=0.2)

        self.assertEqual(result["summary"]["action_count"], 1)
        self.assertEqual(result["summary"]["omitted_count"], 2)
        action = result["actions"][0]
        self.assertEqual(action["target_candidate_id"], "crunch-positive-lifecycle")
        self.assertEqual(action["policy_section"], "crunch")
        self.assertEqual(action["action_type"], "widen")
        self.assertEqual(action["holdout_fraction"], 0.2)
        gate = action["evidence_summary"]["crunch_lifecycle_gate"]
        self.assertEqual(gate["status"], "ready")
        self.assertTrue(gate["positive_savings"])
        self.assertTrue(gate["safety_clean"])
        self.assertEqual(gate["before_after"]["tokens_before_est"], 48000)
        self.assertEqual(gate["before_after"]["tokens_after_est"], 36000)
        self.assertEqual(gate["before_after"]["tokens_saved_est"], 12000)
        self.assertEqual(gate["holdout_metadata"]["canary_applied_count"], 7)
        self.assertEqual(gate["holdout_metadata"]["canary_holdout_count"], 5)
        self.assertFalse(gate["holdout_metadata"]["individual_candidate_ids_included"])
        self.assertFalse(result["wrote_local_policy_files"])

        omitted_reasons = {row["target_candidate_id"]: row["reason"] for row in result["omitted"]}
        self.assertEqual(
            omitted_reasons,
            {
                "crunch-missing-measurement": "crunch-canary-missing-lifecycle-measurement",
                "crunch-safety-stop": "crunch-canary-safety-stop",
            },
        )
        omitted = {row["target_candidate_id"]: row for row in result["omitted"]}
        self.assertEqual(
            omitted["crunch-missing-measurement"]["evidence_summary"]["crunch_lifecycle_gate"]["status"],
            "blocked",
        )
        self.assertEqual(
            omitted["crunch-safety-stop"]["evidence_summary"]["crunch_lifecycle_gate"]["reason"],
            "crunch-canary-safety-stop",
        )
        next_actions = {bucket["next_action"] for bucket in result["omission_buckets"]}
        self.assertIn("collect-canary-applied", next_actions)
        self.assertIn("review-crunch-quality-gate", next_actions)
        self._assert_privacy_clean(result)

    def test_promotion_actions_attach_family_specific_safety_stop_reasons(self):
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "routing-missing-holdout",
                    "optimization_family": "openai_local_routing",
                    "action_family": "routing",
                    "source_surface": "openai_provider_request",
                    "app_family": "generic_openai",
                    "projected_savings_usd": 0.11,
                    "sample_count": 11,
                    "verdict": "needs_eval",
                    "reason_codes": ["insufficient-canary-holdout-samples"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "cache-stale-dependency",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "openai_responses",
                    "app_family": "generic_openai",
                    "projected_savings_usd": 0.09,
                    "sample_count": 9,
                    "verdict": "widen",
                    "file_dependency_status": "invalidated",
                    "reason_codes": ["stale-risk-blockers"],
                    "eval_evidence": {"result_count": 1, "pass_count": 1},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
                {
                    "candidate_id": "crunch-quality-gate",
                    "optimization_family": "old_context_summarization",
                    "action_family": "crunch",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.07,
                    "sample_count": 7,
                    "verdict": "needs_eval",
                    "reason_codes": ["safety-stop-observed", "summary-failure-rate"],
                    "eval_evidence": {"result_count": 1, "fail_count": 1},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                },
            ],
        }

        result = build_optimization_promotion_actions(report)

        by_candidate = {row["target_candidate_id"]: row for row in result["omitted"]}
        self.assertEqual(by_candidate["routing-missing-holdout"]["safety_stop_reason_code"], "routing-missing-holdout-coverage")
        self.assertEqual(by_candidate["routing-missing-holdout"]["recommended_unblock_action"], "collect-canary-holdout")
        self.assertEqual(by_candidate["routing-missing-holdout"]["recommended_blocker_state"], "unblockable")
        self.assertEqual(by_candidate["cache-stale-dependency"]["safety_stop_reason_code"], "cache-dependency-instability")
        self.assertEqual(by_candidate["cache-stale-dependency"]["recommended_unblock_action"], "refresh-cache-dependency-evidence")
        self.assertEqual(by_candidate["crunch-quality-gate"]["safety_stop_reason_code"], "crunch-quality-gate-failed")
        self.assertEqual(by_candidate["crunch-quality-gate"]["recommended_blocker_state"], "keep-blocked")
        bucket_codes = {row["reason_code"] for row in result["safety_stop_reason_buckets"]}
        self.assertEqual(
            bucket_codes,
            {
                "routing-missing-holdout-coverage",
                "cache-dependency-instability",
                "crunch-quality-gate-failed",
            },
        )
        next_actions = {row["next_action"] for row in result["omission_buckets"]}
        self.assertIn("collect-canary-holdout", next_actions)
        self.assertIn("fix-dependency-freshness", next_actions)
        self.assertIn("review-crunch-quality-gate", next_actions)
        self._assert_privacy_clean(result)

    def test_promotion_blocker_recommendation_review_groups_sanitized_local_candidates(self):
        payload = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [
                {
                    "schema": "agentflow.promotion_blocker_next_action_recommendation.v1",
                    "recommendation_id": "promotion-blocker-next-action:openai:routing:eval-missing",
                    "rank": 1,
                    "status": "recommended",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "openai_provider_request",
                    "provider_family": "openai",
                    "provider_endpoint": "responses",
                    "blocker_family": "eval-missing",
                    "blocker_reason_codes": ["missing-eval-evidence", "eval-results-missing"],
                    "blocker_count": 120,
                    "recommendation_type": "collect-eval-evidence",
                    "next_action": "backfill-local-eval-evidence",
                    "expected_local_executor": "optimization-shadow-eval",
                    "file_backed_policy_representation": {
                        "exists": True,
                        "policy_section": "routing",
                        "policy_source": "local-manual",
                        "rule_file": "routing_rules.yaml",
                        "reason": "known-file-backed-local-policy",
                    },
                    "local_executor_compatibility": {
                        "status": "compatible",
                        "local_action_family": "routing",
                        "file_path": "/tmp/promotion-blocker-local-secret.py",
                    },
                    "confidence": 0.97,
                    "projected_savings_usd": 72.25,
                    "evidence_summary": {
                        "record_count": 120,
                        "candidate_count": 120,
                        "promotion_status": "needs-eval",
                        "rank_score": 0.97,
                        "raw_request": {"prompt": "raw promotion blocker prompt must stay local"},
                        "request_id": "promotion-blocker-request-secret",
                        "session_id": "promotion-blocker-session-secret",
                        "cache_key": "promotion-blocker-cache-secret",
                    },
                    "prompt": "raw promotion blocker prompt must stay local",
                    "messages": [{"content": "raw promotion blocker provider body must stay local"}],
                    "provider_body": {"input": "raw promotion blocker provider body must stay local"},
                    "request_id": "promotion-blocker-request-secret",
                    "session_id": "promotion-blocker-session-secret",
                    "cache_key": "promotion-blocker-cache-secret",
                    "file_path": "/tmp/promotion-blocker-local-secret.py",
                },
                {
                    "recommendation_id": "promotion-blocker-next-action:codex:routing:canary-missing",
                    "rank": 2,
                    "status": "noop",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "codex_turn",
                    "provider_family": "codex",
                    "blocker_family": "canary-missing",
                    "blocker_reason_codes": ["missing-canary-evidence"],
                    "blocker_count": 20,
                    "recommendation_type": "noop",
                    "next_action": "keep-blocked",
                    "confidence": 0.5,
                    "projected_savings_usd": 50.0,
                    "no_op_reasons": ["provider-capability-canary_holdout-unavailable"],
                    "file_backed_policy_representation": {"exists": False, "rule_file": "/private/routing_rules.yaml"},
                },
            ],
        }

        result = build_promotion_blocker_recommendation_review(payload, limit=10)

        self.assertEqual(result["schema"], "agentflow.promotion_blocker_recommendation_review.v1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        self.assertEqual(result["summary"]["review_candidate_count"], 2)
        self.assertEqual(result["summary"]["recommended_count"], 1)
        self.assertEqual(result["summary"]["noop_count"], 1)
        self.assertEqual(result["summary"]["group_count"], 1)

        group = result["groups"][0]
        self.assertEqual(group["local_action_family"], "routing")
        candidate = group["recommendations"][0]
        self.assertEqual(candidate["recommendation_type"], "collect-eval-evidence")
        self.assertEqual(candidate["expected_local_executor"], "optimization-shadow-eval")
        self.assertEqual(candidate["blocker_reason_codes"], ["eval-results-missing", "missing-eval-evidence"])
        self.assertEqual(candidate["file_backed_policy_representation"]["rule_file"], "routing_rules.yaml")
        self.assertEqual(candidate["confidence"], 0.97)

        noop = result["omitted_actions"][0]
        self.assertEqual(noop["next_action"], "keep-blocked")
        self.assertEqual(noop["no_op_reasons"], ["provider-capability-canary-holdout-unavailable"])

        encoded = json.dumps(result, sort_keys=True)
        for forbidden in (
            "raw promotion blocker prompt",
            "raw promotion blocker provider body",
            "promotion-blocker-request-secret",
            "promotion-blocker-session-secret",
            "promotion-blocker-cache-secret",
            "/tmp/promotion-blocker-local-secret.py",
            "/private/routing_rules.yaml",
            '"prompt"',
            '"messages"',
            '"provider_body"',
            '"raw_request"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_promotion_blocker_recommendations_queue_shadow_eval_rows_and_noops_idempotently(self):
        payload = {
            "schema": "agentflow.promotion_blocker_next_action_recommendations.v1",
            "recommendations": [
                {
                    "recommendation_id": "promotion-blocker-next-action:openai:routing:eval-missing",
                    "rank": 1,
                    "status": "recommended",
                    "local_action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "source_surface": "openai_provider_request",
                    "provider_family": "openai",
                    "provider_endpoint": "responses",
                    "blocker_family": "eval-missing",
                    "blocker_reason_codes": ["missing-eval-evidence", "eval-results-missing"],
                    "blocker_count": 120,
                    "recommendation_type": "collect-eval-evidence",
                    "next_action": "backfill-local-eval-evidence",
                    "expected_local_executor": "optimization-shadow-eval",
                    "file_backed_policy_representation": {
                        "exists": True,
                        "policy_section": "routing",
                        "policy_source": "local-manual",
                        "rule_file": "routing_rules.yaml",
                    },
                    "local_executor_compatibility": {"status": "compatible", "local_action_family": "routing"},
                    "confidence": 0.97,
                    "projected_savings_usd": 72.25,
                    "evidence_summary": {
                        "record_count": 120,
                        "promotion_status": "needs-eval",
                        "raw_request": {"prompt": "raw eval prompt secret"},
                        "request_id": "eval-request-id-secret",
                        "session_id": "eval-session-id-secret",
                        "cache_key": "eval-cache-key-secret",
                    },
                    "messages": [{"content": "raw eval message secret"}],
                    "provider_body": {"input": "raw eval content secret"},
                    "file_path": "/tmp/raw-eval-secret.py",
                },
                {
                    "recommendation_id": "promotion-blocker-next-action:openai:unknown:eval-missing",
                    "rank": 2,
                    "status": "recommended",
                    "local_action_family": "unknown-family",
                    "candidate_family": "unknown-policy",
                    "source_surface": "openai_provider_request",
                    "blocker_family": "eval-missing",
                    "blocker_reason_codes": ["eval-results-missing"],
                    "recommendation_type": "collect-eval-evidence",
                    "next_action": "backfill-local-eval-evidence",
                    "expected_local_executor": "optimization-shadow-eval",
                    "projected_savings_usd": 10.0,
                    "evidence_summary": {"record_count": 12, "promotion_status": "needs-eval"},
                },
            ],
        }
        review = build_promotion_blocker_recommendation_review(payload, limit=10)

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                dry_run = queue_promotion_recommendation_eval_tasks(store, review, apply=False)
                dry_count = store.conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
                applied = queue_promotion_recommendation_eval_tasks(store, review, apply=True, now="2026-06-14T09:30:00+00:00")
                second = queue_promotion_recommendation_eval_tasks(store, review, apply=True, now="2026-06-14T09:31:00+00:00")
                stored = store.conn.execute(
                    "select id, candidate_id, status_class, reason_codes_json, result_json "
                    "from optimization_eval_results order by status_class desc"
                ).fetchall()
            finally:
                store.conn.close()

        self.assertEqual(dry_run["schema"], "agentflow.promotion_recommendation_eval_queue.v1")
        self.assertTrue(dry_run["dry_run"])
        self.assertEqual(dry_count, 0)
        self.assertEqual(dry_run["summary"]["selected_task_count"], 1)
        self.assertEqual(dry_run["summary"]["noop_count"], 1)
        self.assertFalse(applied["dry_run"])
        self.assertTrue(applied["wrote_eval_queue_rows"])
        self.assertTrue(applied["wrote_noop_records"])
        self.assertEqual(applied["summary"]["written_task_count"], 1)
        self.assertEqual(applied["summary"]["written_noop_count"], 1)
        self.assertEqual(second["summary"]["written_task_count"], 0)
        self.assertEqual(second["summary"]["written_noop_count"], 0)
        self.assertEqual(second["summary"]["already_recorded_count"], 2)
        self.assertEqual(len(stored), 2)
        by_status = {row["status_class"]: row for row in stored}
        self.assertIn("eval-queued", json.loads(by_status["queued"]["reason_codes_json"]))
        self.assertIn("unsupported-local-action-family", json.loads(by_status["noop"]["reason_codes_json"]))
        queued_result = json.loads(by_status["queued"]["result_json"])
        self.assertEqual(queued_result["task"]["recommendation_id"], "promotion-blocker-next-action:openai:routing:eval-missing")
        self.assertEqual(queued_result["task"]["recommended_eval_mode"], "local-shadow-eval")
        self.assertFalse(applied["provider_calls_made"])
        self.assertFalse(applied["managed_server_calls_made"])
        self._assert_privacy_clean(dry_run)
        self._assert_privacy_clean(applied)
        self._assert_privacy_clean(second)
        for row in stored:
            self._assert_privacy_clean(json.loads(row["result_json"]))

    def test_promotion_funnel_stats_include_aggregate_omission_buckets(self):
        plan = {
            "schema": "agentflow.optimization_eval_plan.v1",
            "plans": [],
        }
        promotion_report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "stats-omitted-cache-candidate",
                    "optimization_family": "cache_replayability",
                    "action_family": "cache",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "projected_savings_usd": 0.04,
                    "sample_count": 7,
                    "verdict": "needs_eval",
                    "reason_codes": ["eval-results-missing"],
                    "eval_evidence": {"result_count": 0, "pass_count": 0},
                    "privacy": {"metadata_only": True, "raw_prompts_included": False},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch(
                    "agentflow_proxy.optimization_eval_plan.build_optimization_eval_plan",
                    new=AsyncMock(return_value=plan),
                ), patch(
                    "agentflow_proxy.optimization_promotion_report.build_optimization_promotion_report",
                    return_value=promotion_report,
                ), patch(
                    "agentflow_proxy.policy_events.recent_policy_events",
                    return_value={"events": []},
                ):
                    result = asyncio.run(stats_optimization_promotion_funnel(store, limit=10))
            finally:
                store.conn.close()

        self.assertEqual(result["schema"], "agentflow.optimization_promotion_funnel.v1")
        self.assertEqual(result["summary"]["promotion_action_count"], 0)
        self.assertEqual(result["summary"]["promotion_omitted_count"], 1)
        self.assertEqual(result["summary"]["promotion_omission_bucket_count"], 1)
        self.assertEqual(result["summary"]["top_promotion_omission_next_action"], "run-local-shadow-eval")
        self.assertEqual(result["source_reports"]["promotion_actions_schema"], "agentflow.optimization_promotion_rollout_actions.v1")
        self.assertEqual(result["omission_buckets"][0]["next_action"], "run-local-shadow-eval")
        self.assertNotIn("stats-omitted-cache-candidate", json.dumps(result["omission_buckets"], sort_keys=True))
        self._assert_privacy_clean(result)

    def test_promotion_canary_apply_records_deterministic_holdout_and_safety_stop(self):
        report = {
            "schema": "agentflow.optimization_promotion_report.v1",
            "candidates": [
                {
                    "candidate_id": "routing-canary-candidate",
                    "optimization_family": "phase_routing",
                    "action_family": "routing",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "candidate_target_model": "claude-haiku-4-5-20251001",
                    "projected_savings_usd": 0.05,
                    "sample_count": 2,
                    "cohort_counts": {"canary_applied": 0, "canary_holdout": 0, "bypassed_or_disabled": 0},
                    "eval_evidence": {"result_count": 1, "pass_count": 1, "fail_count": 0, "blocked_count": 0},
                    "verdict": "widen",
                    "reason_codes": ["promotion-thresholds-met"],
                    "privacy": {"raw_prompts_included": False},
                }
            ],
        }
        bundle = build_optimization_promotion_actions(report, initial_canary_fraction=0.10, holdout_fraction=0.10)
        action = bundle["actions"][0]

        decisions = {}
        for index in range(300):
            metadata = {
                "request_fingerprint": f"fixture-request-{index}",
                "session_id_hash": f"sha256:session-{index % 11}",
                "source_surface": "anthropic_messages",
                "app_family": "claude_code",
                "workflow_phase": "tool-execution",
                "category": "tool-result",
                "text_bucket": "2k-8k",
                "requested_model": "claude-sonnet-4-6",
                "candidate_target_model": "claude-haiku-4-5-20251001",
                "has_tools": True,
            }
            first = promotion_canary_decision(action, metadata)
            second = promotion_canary_decision(action, metadata)
            self.assertEqual(first, second)
            decisions.setdefault(first["cohort"], first)
            if {"canary_applied", "canary_holdout", "skipped"}.issubset(decisions):
                break

        self.assertEqual(decisions["canary_applied"]["status"], "applied")
        self.assertTrue(decisions["canary_applied"]["selected"])
        self.assertEqual(decisions["canary_holdout"]["status"], "holdout")
        self.assertFalse(decisions["canary_holdout"]["selected"])
        self.assertEqual(decisions["skipped"]["reason"], "outside-canary-and-holdout")
        rendered_decision = stable_json(decisions)
        self.assertNotIn("raw prompt", rendered_decision)
        self.assertNotIn("messages", rendered_decision)

        with tempfile.TemporaryDirectory() as tmp:
            apply_result = apply_optimization_promotion_canaries(bundle, config_dir=tmp, dry_run=False)
            self.assertTrue(apply_result["ok"])
            self.assertTrue(apply_result["wrote_policy_files"])
            routing_yaml = Path(tmp) / "routing_rules.yaml"
            data = yaml.safe_load(routing_yaml.read_text(encoding="utf-8"))
            phase_canary = data["phase_canary"]
            self.assertTrue(phase_canary["enabled"])
            self.assertEqual(phase_canary["promotion_action_id"], action["action_id"])
            self.assertEqual(phase_canary["target_candidate_id"], "routing-canary-candidate")
            self.assertEqual(phase_canary["canary_fraction"], 0.1)
            self.assertEqual(phase_canary["holdout_fraction"], 0.1)
            self.assertEqual(phase_canary["policy_source"], "managed-recommended")

        safety = evaluate_promotion_canary_safety_stop(
            action,
            [
                {"cohort": "canary_applied", "status_code": 500, "error_bucket": "http_5xx"},
                {"cohort": "canary_applied", "status_code": 400, "error_bucket": "unsupported_model"},
                {"cohort": "canary_holdout", "status_code": 200},
            ],
            thresholds={"min_samples": 2, "max_error_rate": 0.5, "max_5xx_rate": 0.0, "max_unsupported_model_errors": 0},
        )
        self.assertTrue(safety["tripped"])
        self.assertIn("error-rate", safety["reason_codes"])
        self.assertIn("provider-5xx-rate", safety["reason_codes"])
        self.assertIn("unsupported-model-errors", safety["reason_codes"])
        stopped = promotion_canary_decision(action, {"request_fingerprint": "fixture-request-stop"}, safety_stop=safety)
        self.assertEqual(stopped["status"], "safety_stopped")
        self.assertEqual(stopped["cohort"], "bypassed_or_disabled")
        self.assertEqual(stopped["reason"], "local-canary-safety-stop")

    def _promotion_impact_bundle(self, *actions):
        return {
            "schema": "agentflow.optimization_promotion_rollout_actions.v1",
            "generated_at": "2026-06-10T00:00:00+00:00",
            "ok": True,
            "actions": list(actions),
            "privacy": {"metadata_only": True},
        }

    def _promotion_impact_action(self, section: str, candidate_id: str, *, projected_savings: float = 0.004):
        return {
            "schema": "agentflow.optimization_promotion_rollout_action.v1",
            "action_id": f"promotion-action-{candidate_id}",
            "action_type": "widen",
            "policy_section": section,
            "target_candidate_id": candidate_id,
            "target_rule_id": f"promotion-{section}-{candidate_id}",
            "canary_fraction": 0.25,
            "holdout_fraction": 0.10,
            "evidence_summary": {
                "projected_savings_usd": projected_savings,
                "sample_count": 3,
                "cohort_counts": {"canary_applied": 1, "canary_holdout": 1, "bypassed_or_disabled": 0},
            },
            "local_policy_update": {"policy_source": "managed-recommended"},
            "privacy": {"metadata_only": True},
        }

    def _log_promotion_impact_call(
        self,
        store,
        action: dict,
        *,
        cohort: str,
        status_code: int = 200,
        cost_est: float = 0.001,
        cost_baseline: float = 0.003,
        created_at: str = "2026-06-10T00:10:00+00:00",
        suffix: str = "",
    ):
        section = action["policy_section"]
        applied = cohort == "canary_applied"
        status = "applied" if applied else "holdout" if cohort == "canary_holdout" else "safety_stopped"
        reason = "selected-canary" if applied else "selected-holdout" if cohort == "canary_holdout" else "local-canary-safety-stop"
        base_meta = {
            "promotion_action_id": action["action_id"],
            "target_candidate_id": action["target_candidate_id"],
            "target_rule_id": action["target_rule_id"],
            "policy_section": section,
            "policy_source": "managed-recommended",
            "status": status,
            "cohort": cohort,
            "reason": reason,
        }
        if cohort == "safety_stopped":
            base_meta["safety_stop"] = {"tripped": True, "reason_codes": ["error-rate"]}
        routing_json = {"category": "tool-result"}
        crunch_json = {"changed": False}
        cache_json = {"status": "miss", "reason": "exact-miss"}
        if section == "routing":
            routing_json["phase_canary"] = base_meta
        else:
            rule = {
                "rule_id": action["target_rule_id"],
                "candidate_id": action["target_candidate_id"],
                "promotion_action_id": action["action_id"],
                "policy_source": "managed-recommended",
                "canary": {"status": status, "cohort": cohort},
                "reason": reason,
                "estimated_cost_savings_usd": max(0.0, cost_baseline - cost_est),
            }
            if cohort == "safety_stopped":
                rule["safety_stop"] = {"tripped": True, "reason_codes": ["error-rate"]}
            if section == "crunch":
                crunch_json = {
                    "changed": applied,
                    "pattern_rules": {"configured_count": 1, "rules": [rule], "skip_reasons": []},
                }
            else:
                cache_json = {
                    "status": "hit" if applied else "miss",
                    "pattern_rules": {"configured_count": 1, "rules": [rule], "skip_reasons": []},
                }
        store.log_call(
            id=f"promotion-impact-{section}-{cohort}-{suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-haiku-4-5-20251001" if section == "routing" and applied else "claude-sonnet-4-6",
            stream=0,
            cache_hit=1 if section == "cache" and applied else 0,
            status_code=status_code,
            latency_ms=1000 if status_code < 400 else 12000,
            input_tokens_est=100,
            output_tokens_est=10,
            actual_input_tokens=100,
            actual_output_tokens=10,
            cost_est_usd=cost_est,
            cost_baseline_usd=cost_baseline,
            crunch_json=stable_json(crunch_json),
            routing_json=stable_json(routing_json),
            cache_json=stable_json(cache_json),
            error='{"error":{"type":"overloaded_error"}}' if status_code >= 400 else None,
            request_json=None,
            response_json=None,
            session_id="raw-session-secret",
            category="tool-result",
            retry_count=1 if status_code >= 400 else 0,
            provider="anthropic",
            source_surface="anthropic_messages",
        )

    def test_optimization_promotion_impact_reports_routing_crunch_cache_metadata_only(self):
        routing = self._promotion_impact_action("routing", "routing-impact")
        crunch = self._promotion_impact_action("crunch", "crunch-impact")
        cache = self._promotion_impact_action("cache", "cache-impact")
        eval_only = self._promotion_impact_action("evaluation", "eval-impact")
        eval_only["action_family"] = "evaluation-only"
        bundle = self._promotion_impact_bundle(routing, crunch, cache, eval_only)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                for action in (routing, crunch, cache):
                    self._log_promotion_impact_call(store, action, cohort="canary_applied", suffix="a")
                    self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h")
                report = measure_optimization_promotion_impact(
                    bundle,
                    store_obj=store,
                    limit=20,
                    now=datetime(2026, 6, 10, 1, tzinfo=timezone.utc),
                    max_evidence_age_hours=24,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                )
            finally:
                store.conn.close()

        self.assertEqual(report["schema"], "agentflow.optimization_promotion_impact.v1")
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "matched")
        self.assertTrue(report["read_only"])
        self.assertFalse(report["wrote_policy_files"])
        self.assertFalse(report["wrote_store"])
        self.assertFalse(report["provider_calls_made"])
        self.assertFalse(report["managed_server_calls_made"])
        self.assertEqual(report["summary"]["actual_canary_applied_count"], 3)
        self.assertEqual(report["summary"]["actual_canary_holdout_count"], 3)
        self.assertEqual({row["next_step"]["verdict"] for row in report["actions"] if row["policy_section"] != "evaluation"}, {"widen"})
        self.assertGreater(report["summary"]["observed_savings_usd"], 0)
        by_section = {row["policy_section"]: row for row in report["actions"]}
        self.assertEqual(by_section["routing"]["actual"]["actual_canary_applied_count"], 1)
        self.assertEqual(by_section["crunch"]["actual"]["actual_canary_holdout_count"], 1)
        self.assertEqual(by_section["cache"]["next_step"]["projected_vs_observed_savings_ratio"], 0.5)
        self.assertEqual(by_section["routing"]["next_step"]["recommendation"], "promote")
        by_family = {row["action_family"]: row for row in report["family_impacts"]}
        self.assertEqual(set(by_family), {"routing", "crunch", "cache", "evaluation-only"})
        self.assertEqual(by_family["routing"]["applied_count"], 1)
        self.assertEqual(by_family["routing"]["holdout_count"], 1)
        self.assertEqual(by_family["routing"]["holdout_or_bypass_count"], 1)
        self.assertEqual(by_family["routing"]["blocker_delta"], 0)
        self.assertGreater(by_family["routing"]["savings_delta_usd"], 0)
        self.assertEqual(by_family["routing"]["missing_measurements"], [])
        self.assertEqual(by_family["routing"]["cohort_metrics"]["canary_applied"]["count"], 1)
        self.assertEqual(by_family["routing"]["cohort_metrics"]["canary_holdout"]["count"], 1)
        self.assertGreater(by_family["routing"]["applied_vs_holdout_deltas"]["applied_minus_holdout_observed_savings_usd"], 0)
        self.assertGreater(
            by_family["routing"]["applied_vs_holdout_deltas"]["applied_minus_holdout_or_bypass_observed_savings_usd"],
            0,
        )
        self.assertEqual(by_family["routing"]["recommendation"], "promote")
        self.assertEqual(by_family["crunch"]["recommendation"], "promote")
        self.assertEqual(by_family["cache"]["recommendation"], "promote")
        self.assertEqual(by_family["evaluation-only"]["recommendation"], "needs-more-evidence")
        self.assertEqual(by_family["evaluation-only"]["top_blocker"], "insufficient-canary-applied-samples")
        self.assertEqual(by_family["evaluation-only"]["applied_count"], 0)
        self.assertIn("missing-applied-promotion-metadata", by_family["evaluation-only"]["missing_measurements"])
        self.assertIn("missing-holdout-or-bypass-promotion-metadata", by_family["evaluation-only"]["missing_measurements"])
        self.assertIn("missing-observed-savings-delta", by_family["evaluation-only"]["missing_measurements"])
        self.assertEqual(report["summary"]["recommendation_counts"], [
            {"value": "promote", "count": 3},
            {"value": "needs-more-evidence", "count": 1},
        ])
        rendered = stable_json(report)
        self.assertNotIn("raw-session-secret", rendered)
        self.assertNotIn("request_json", rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self._assert_privacy_clean(report)

    def test_optimization_promotion_impact_verdicts_cover_safety_negative_stale_insufficient_and_privacy(self):
        scenarios = (
            ("insufficient", "needs_more_samples", "insufficient-canary-applied-samples"),
            ("safety", "rollback", "safety-stop-observed"),
            ("negative", "rollback", "negative-observed-savings"),
            ("stale", "hold", "stale-evidence"),
        )
        for scenario, expected_verdict, expected_reason in scenarios:
            action = self._promotion_impact_action("routing", f"routing-{scenario}", projected_savings=0.001)
            bundle = self._promotion_impact_bundle(action)
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as tmp:
                    store = Store(str(Path(tmp) / "agentflow.sqlite3"))
                    try:
                        if scenario == "insufficient":
                            self._log_promotion_impact_call(store, action, cohort="canary_applied", suffix="one")
                        elif scenario == "safety":
                            self._log_promotion_impact_call(store, action, cohort="canary_applied", suffix="a1")
                            self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h")
                            self._log_promotion_impact_call(store, action, cohort="safety_stopped", status_code=500, suffix="stop")
                        elif scenario == "negative":
                            self._log_promotion_impact_call(store, action, cohort="canary_applied", cost_est=0.004, cost_baseline=0.001, suffix="a1")
                            self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h")
                        else:
                            self._log_promotion_impact_call(store, action, cohort="canary_applied", created_at="2026-06-01T00:00:00+00:00", suffix="a1")
                            self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, created_at="2026-06-01T00:00:01+00:00", suffix="h")
                        report = measure_optimization_promotion_impact(
                            bundle,
                            store_obj=store,
                            limit=10,
                            since="2026-06-01T00:00:00+00:00" if scenario == "stale" else "2026-06-10T00:00:00+00:00",
                            min_applied_samples=2 if scenario == "insufficient" else 1,
                            min_holdout_samples=1,
                            max_evidence_age_hours=1,
                            now=datetime(2026, 6, 10, 1, tzinfo=timezone.utc),
                        )
                    finally:
                        store.conn.close()
                step = report["actions"][0]["next_step"]
                self.assertEqual(step["verdict"], expected_verdict)
                self.assertEqual(step["recommendation"], "needs-more-evidence" if expected_verdict == "needs_more_samples" else "keep-canary" if expected_verdict == "hold" else expected_verdict)
                self.assertIn(expected_reason, step["reason_codes"])
                self.assertEqual(report["family_impacts"][0]["recommendation"], step["recommendation"])
                if scenario == "stale":
                    self.assertTrue(report["actions"][0]["stale_evidence"]["stale"])

        blocked = self._promotion_impact_bundle(self._promotion_impact_action("routing", "privacy"))
        blocked["actions"][0]["raw_prompt"] = "raw promotion impact secret"
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                blocked_report = measure_optimization_promotion_impact(blocked, store_obj=store)
            finally:
                store.conn.close()
        self.assertFalse(blocked_report["ok"])
        self.assertEqual(blocked_report["status"], "privacy-blocked")
        self.assertFalse(blocked_report["privacy"]["raw_prompts_included"])
        self.assertNotIn("raw promotion impact secret", stable_json(blocked_report))

    def test_promotion_outcome_feedback_records_append_only_metadata_and_rollback_status(self):
        action = self._promotion_impact_action("routing", "routing-feedback-ledger", projected_savings=0.004)
        bundle = self._promotion_impact_bundle(action)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self._log_promotion_impact_call(store, action, cohort="canary_applied", status_code=500, suffix="a")
                self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h")
                impact = measure_optimization_promotion_impact(
                    bundle,
                    store_obj=store,
                    limit=10,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    max_evidence_age_hours=999999,
                    now=datetime(2026, 6, 10, 1, tzinfo=timezone.utc),
                )
                first = record_promotion_outcome_feedback(impact, store_obj=store, recorded_at="2026-06-10T01:00:00+00:00")
                second = record_promotion_outcome_feedback(impact, store_obj=store, recorded_at="2026-06-10T01:00:01+00:00")
                rows = store.promotion_outcome_feedback_rows(limit=10)
                summary = promotion_outcome_feedback_summary(store, limit=10)
                promotion_report = build_optimization_promotion_report(
                    store,
                    plan={
                        "schema": "agentflow.optimization_eval_plan.v1",
                        "plans": [{
                            "candidate_id": action["target_candidate_id"],
                            "optimization_family": "phase_routing",
                            "action_family": "routing",
                            "source_surface": "anthropic_messages",
                            "app_family": "claude_code",
                            "projected_savings_usd": 0.004,
                            "sample_count": 2,
                            "evidence": {"verdict": "widen"},
                        }],
                    },
                    limit=10,
                    min_eval_pass_count=0,
                    min_canary_applied_samples=1,
                    min_canary_holdout_samples=1,
                )
            finally:
                store.conn.close()

        self.assertTrue(first["ok"])
        self.assertTrue(first["wrote_store"])
        self.assertEqual(first["summary"]["rows_written"], 1)
        self.assertEqual(second["summary"]["rows_written"], 1)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])
        entry = first["entries"][0]
        self.assertEqual(entry["schema"], "agentflow.promotion_outcome_feedback_entry.v1")
        self.assertEqual(entry["policy_id"], action["target_rule_id"])
        self.assertEqual(entry["action_family"], "routing")
        self.assertEqual(entry["rule_source"], "managed-recommended")
        self.assertEqual(entry["source_evidence_schema"], "agentflow.optimization_promotion_rollout_actions.v1")
        self.assertEqual(entry["status"], "rollback-needed")
        self.assertTrue(entry["rollback_needed"])
        self.assertEqual(entry["applied_count"], 1)
        self.assertEqual(entry["holdout_count"], 1)
        self.assertGreaterEqual(entry["error_rate_delta"], 1.0)
        self.assertEqual(summary["summary"]["rollback_needed_count"], 2)
        self.assertEqual(summary["candidates"][0]["verdict"], "rollback")
        self.assertEqual(promotion_report["promotion_outcome_feedback"]["entry_count"], 2)
        self.assertEqual(
            promotion_report["candidates"][0]["evidence_sources"][0]["source"],
            "promotion_outcome_feedback",
        )
        self.assertEqual(promotion_report["candidates"][0]["cohort_counts"]["canary_applied"], 1)
        rendered = stable_json(summary)
        self.assertNotIn("raw-session-secret", rendered)
        self.assertNotIn("request_json", rendered)
        self.assertFalse(summary["privacy"]["request_ids_included"])
        self.assertFalse(summary["privacy"]["raw_provider_bodies_included"])

    def test_optimization_promotion_impact_blocks_provider_adoption_regression(self):
        action = self._promotion_impact_action("routing", "routing-adoption-regression", projected_savings=0.001)
        bundle = self._promotion_impact_bundle(action)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self._log_promotion_impact_call(store, action, cohort="canary_applied", suffix="a1")
                self._log_promotion_impact_call(store, action, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                store.log_provider_tool_adoption_window(
                    id="provider-adoption-applied-risk",
                    created_at="2026-06-10T00:10:05+00:00",
                    updated_at="2026-06-10T01:10:05+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    app_family="claude",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    workflow_phase="tool-execution",
                    policy_source="managed-recommended",
                    policy_ids_json=stable_json(["promotion-routing-adoption-regression"]),
                    call_id="promotion-impact-routing-canary_applied-a1",
                    fulfilled_call_id=None,
                    session_digest="sha256:session-secret",
                    correlation_digest="sha256:tool-secret-applied",
                    status="abandoned",
                    reason="ttl-expired-without-tool-result",
                    age_bucket="1_6h",
                    tool_use_count=1,
                    tool_result_count=0,
                    metadata_json=stable_json({"metadata_only": True}),
                )
                store.log_provider_tool_adoption_window(
                    id="provider-adoption-holdout-fulfilled",
                    created_at="2026-06-10T00:10:06+00:00",
                    updated_at="2026-06-10T00:10:20+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    app_family="claude",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    workflow_phase="tool-execution",
                    policy_source="managed-recommended",
                    policy_ids_json=stable_json(["promotion-routing-adoption-regression"]),
                    call_id="promotion-impact-routing-canary_holdout-h1",
                    fulfilled_call_id="promotion-impact-routing-canary_holdout-h1",
                    session_digest="sha256:session-secret",
                    correlation_digest="sha256:tool-secret-holdout",
                    status="fulfilled",
                    reason="matched-subsequent-tool-result",
                    age_bucket="0_1m",
                    tool_use_count=1,
                    tool_result_count=1,
                    metadata_json=stable_json({"metadata_only": True}),
                )
                report = measure_optimization_promotion_impact(
                    bundle,
                    store_obj=store,
                    limit=10,
                    since="2026-06-10T00:00:00+00:00",
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    now=datetime(2026, 6, 10, 1, tzinfo=timezone.utc),
                )
            finally:
                store.conn.close()

        step = report["actions"][0]["next_step"]
        gate = report["actions"][0]["provider_adoption_gate"]
        self.assertEqual(step["verdict"], "hold")
        self.assertIn("provider-adoption-regression", step["reason_codes"])
        self.assertTrue(gate["blocking"])
        self.assertEqual(gate["cohorts"]["applied"]["abandoned_count"], 1)
        self.assertEqual(gate["cohorts"]["holdout"]["fulfilled_count"], 1)
        rendered = stable_json(report)
        self.assertNotIn("tool-secret", rendered)
        self.assertNotIn("session-secret", rendered)

    def _optimization_rollout_bundle(self):
        return {
            "schema": "agentflow.optimization_rollout_actions.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "expires_at": "2099-06-11T05:00:00+00:00",
            "tenant_scope": "current-authenticated-tenant",
            "summary": {
                "candidate_count": 1,
                "action_count": 1,
                "omitted_count": 0,
                "managed_enforced": False,
                "required_local_review": True,
                "provider_forwarding": False,
                "server_content_processing": False,
            },
            "recommendation": {
                "policy_source": "managed-recommended",
                "required_local_review": True,
                "managed_enforced": False,
                "provider_forwarding": False,
            },
            "local_executor_compatibility": {
                "minimum_local_client_version": "0.1.0",
                "compatible": True,
                "supported_local_action_families": ["routing", "crunch", "cache", "old_context_summarization"],
                "local_review_required": True,
            },
            "actions": [
                {
                    "schema": "agentflow.optimization_rollout_action.v1",
                    "action_id": "optimization-rollout-action:routing",
                    "action_type": "widen",
                    "target_candidate_id": "routing-policy-candidate",
                    "action_family": "routing",
                    "candidate_family": "provider-routing-rule",
                    "policy_section": "routing",
                    "source_surface": "openai_responses",
                    "provider_endpoint": "responses",
                    "confidence": 0.91,
                    "generated_at": "2026-06-10T05:00:00+00:00",
                    "expires_at": "2099-06-11T05:00:00+00:00",
                    "required_local_review": True,
                    "managed_enforced": False,
                    "local_executor_compatibility": {
                        "minimum_local_client_version": "0.1.0",
                        "compatible": True,
                        "supported_local_action_families": ["routing", "crunch", "cache", "old_context_summarization"],
                        "local_review_required": True,
                    },
                    "evidence_summary": {
                        "local_eval_verdict": {
                            "verdict": "widen",
                            "latest_eval_at": "2026-06-10T04:30:00+00:00",
                            "pass_count": 3,
                            "fail_count": 0,
                            "reason_codes": ["local-eval-widen"],
                        },
                        "provider_capability": {"capabilities": {"rollout_actions": "supported"}},
                        "sample_count": 3,
                    },
                    "action": {
                        "schema": "agentflow.openai_rollout_action.v1",
                        "target_rule_id": "openai-routing-rule",
                        "proposed_edit": {
                            "rule_id": "openai-routing-rule",
                            "changed": True,
                            "action": {"route_to": "gpt-5-mini"},
                        },
                    },
                    "privacy_summary": {
                        "metadata_only": True,
                        "feature_only": True,
                        "raw_payloads_returned": False,
                        "raw_prompts_returned": False,
                        "raw_responses_returned": False,
                        "provider_bodies_returned": False,
                        "request_ids_returned": False,
                        "tenant_ids_returned": False,
                        "cache_keys_returned": False,
                        "file_paths_returned": False,
                        "locally_executed": False,
                        "provider_forwarding": False,
                        "managed_enforced": False,
                    },
                }
            ],
            "omitted_actions": [],
            "privacy_summary": {
                "metadata_only": True,
                "feature_only": True,
                "raw_payloads_returned": False,
                "raw_prompts_returned": False,
                "raw_responses_returned": False,
                "provider_bodies_returned": False,
                "request_ids_returned": False,
                "tenant_ids_returned": False,
                "cache_keys_returned": False,
                "file_paths_returned": False,
                "provider_forwarding": False,
                "managed_enforced": False,
            },
        }

    def test_optimization_rollout_review_accepts_signed_fixture_read_only(self):
        bundle = self._optimization_rollout_bundle()
        signed = attach_optimization_rollout_provenance(
            bundle,
            secret="review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="review-key",
            generated_at="2026-06-10T05:00:00+00:00",
        )

        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            result = review_optimization_rollout_actions(
                signed,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["schema"], "agentflow.optimization_rollout_actions_review.v1")
        self.assertEqual(result["provenance"]["status"], "verified")
        self.assertEqual(result["summary"]["accepted_action_count"], 1)
        self.assertTrue(result["read_only"])
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        action = result["actions"][0]
        self.assertEqual(action["target_candidate_id"], "routing-policy-candidate")
        self.assertEqual(action["target_rule_id"], "openai-routing-rule")
        self.assertTrue(action["local_apply_hint"]["review_only"])

    def test_optimization_rollout_review_fails_closed_for_unsigned_expired_incompatible_and_raw_like(self):
        unsigned = self._optimization_rollout_bundle()
        unsigned["expires_at"] = "2026-06-09T05:00:00+00:00"
        unsigned["thresholds"] = {"max_evidence_age_seconds": 60}
        unsigned["local_executor_compatibility"]["compatible"] = False
        unsigned["actions"][0]["evidence_summary"]["local_eval_verdict"]["verdict"] = "rollback"
        unsigned["actions"][0]["privacy_summary"]["raw_prompts_returned"] = True
        unsigned["actions"][0]["raw_request"] = {"prompt": "raw managed prompt must not be accepted"}

        with patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            result = review_optimization_rollout_actions(
                unsigned,
                now=datetime(2026, 6, 10, 6, 0, tzinfo=timezone.utc),
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"]["accepted_action_count"], 0)
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        messages = {error["message"] for error in result["errors"]}
        self.assertIn("managed optimization rollout bundle is missing provenance required by configured verification", messages)
        self.assertIn("optimization rollout bundle is expired", messages)
        self.assertIn("managed bundle reports local executor incompatibility", messages)
        self.assertIn("local eval verdict must be widen", messages)
        self.assertIn("local eval evidence is stale", messages)
        self.assertIn("privacy summary reports raw payloads or local identifiers", messages)
        self.assertIn("raw or local-identifier rollout payloads are not accepted", messages)
        self.assertEqual(result["provenance"]["status"], "missing")

    def _signed_optimization_rollout_bundle(self, bundle=None):
        return attach_optimization_rollout_provenance(
            bundle or self._optimization_rollout_bundle(),
            secret="review-secret",
            issuer="agentflow-server",
            server_id="managed-test",
            key_id="review-key",
            generated_at="2026-06-10T05:00:00+00:00",
        )

    def test_optimization_rollout_apply_writes_openai_canary_and_rolls_back(self):
        signed = self._signed_optimization_rollout_bundle()

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            dry_run = apply_optimization_promotion_canaries(signed, config_dir=tmp, dry_run=True)
            self.assertTrue(dry_run["ok"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertEqual(dry_run["actions"][0]["target_local_policy"], "openai_canary")
            self.assertEqual(dry_run["actions"][0]["canary_fraction"], 0.1)
            self.assertEqual(dry_run["actions"][0]["holdout_fraction"], 0.1)
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())

            applied = apply_optimization_promotion_canaries(signed, config_dir=tmp, dry_run=False)
            self.assertTrue(applied["ok"])
            self.assertTrue(applied["wrote_policy_files"])
            data = yaml.safe_load((Path(tmp) / "routing_rules.yaml").read_text(encoding="utf-8"))
            canary = data["openai_canary"]
            self.assertTrue(canary["enabled"])
            self.assertEqual(canary["policy_id"], "openai-routing-rule")
            self.assertEqual(canary["promotion_action_id"], "optimization-rollout-action:routing")
            self.assertEqual(canary["target_candidate_id"], "routing-policy-candidate")
            self.assertEqual(canary["target_model"], "gpt-5-mini")
            self.assertEqual(canary["policy_source"], "managed-recommended")
            self.assertEqual(canary["canary_fraction"], 0.1)
            self.assertEqual(canary["holdout_fraction"], 0.1)

            rollback_bundle = self._optimization_rollout_bundle()
            rollback_bundle["actions"][0]["action_type"] = "rollback"
            rollback_bundle["actions"][0]["evidence_summary"]["local_eval_verdict"]["verdict"] = "rollback"
            rollback = self._signed_optimization_rollout_bundle(rollback_bundle)
            rollback_result = apply_optimization_promotion_canaries(rollback, config_dir=tmp, dry_run=False)
            self.assertTrue(rollback_result["ok"])
            rolled_back = yaml.safe_load((Path(tmp) / "routing_rules.yaml").read_text(encoding="utf-8"))["openai_canary"]
            self.assertFalse(rolled_back["enabled"])
            self.assertEqual(rolled_back["canary_fraction"], 0.0)
            self.assertEqual(rolled_back["holdout_fraction"], 0.0)

    def test_optimization_rollout_apply_rejects_raw_payloads_and_enabled_manual_policy(self):
        raw_bundle = self._optimization_rollout_bundle()
        raw_bundle["actions"][0]["raw_request"] = {"prompt": "raw managed prompt must not be written"}
        signed_raw = self._signed_optimization_rollout_bundle(raw_bundle)

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            rejected = apply_optimization_promotion_canaries(signed_raw, config_dir=tmp, dry_run=False)
            self.assertFalse(rejected["ok"])
            self.assertFalse(rejected["wrote_policy_files"])
            self.assertFalse((Path(tmp) / "routing_rules.yaml").exists())
            rendered = stable_json(rejected)
            self.assertNotIn("raw managed prompt", rendered)

            routing_yaml = Path(tmp) / "routing_rules.yaml"
            routing_yaml.write_text(
                yaml.safe_dump({
                    "rules": [],
                    "openai_canary": {
                        "enabled": True,
                        "policy_id": "manual-openai-canary",
                        "policy_source": "local-manual",
                        "target_model": "gpt-5-mini",
                        "canary_fraction": 0.2,
                    },
                }),
                encoding="utf-8",
            )
            blocked = apply_optimization_promotion_canaries(self._signed_optimization_rollout_bundle(), config_dir=tmp, dry_run=False)
            self.assertFalse(blocked["ok"])
            self.assertFalse(blocked["wrote_policy_files"])
            self.assertIn("unsafe-policy-source", stable_json(blocked))
            preserved = yaml.safe_load(routing_yaml.read_text(encoding="utf-8"))["openai_canary"]
            self.assertEqual(preserved["policy_id"], "manual-openai-canary")

    def test_optimization_rollout_cli_and_impact_are_metadata_only(self):
        signed = self._signed_optimization_rollout_bundle()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            bundle_path = Path(tmp) / "rollout.json"
            bundle_path.write_text(json.dumps(signed), encoding="utf-8")
            out = io.StringIO()
            code = optimization_rollout_actions_apply_cli(
                [str(bundle_path), "--config-dir", tmp, "--pretty"],
                stdout=out,
            )
            self.assertEqual(code, 0)
            cli_result = json.loads(out.getvalue())
            self.assertTrue(cli_result["ok"])
            self.assertTrue(cli_result["dry_run"])
            self.assertFalse(cli_result["wrote_policy_files"])
            self.assertEqual(cli_result["actions"][0]["target_local_policy"], "openai_canary")

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"AGENTFLOW_MANAGED_POLICY_VERIFICATION_SECRET": "review-secret"}):
            db_path = Path(tmp) / "agentflow.sqlite3"
            store = Store(str(db_path))
            try:
                self._log_openai_canary_call(
                    store,
                    candidate_id="routing-policy-candidate",
                    cohort="canary_applied",
                    action_id="optimization-rollout-action:routing",
                    rule_id="openai-routing-rule",
                    created_at="2026-06-10T06:00:00+00:00",
                    suffix="apply",
                )
                self._log_openai_canary_call(
                    store,
                    candidate_id="routing-policy-candidate",
                    cohort="canary_holdout",
                    action_id="optimization-rollout-action:routing",
                    rule_id="openai-routing-rule",
                    cost_est=0.003,
                    cost_baseline=0.003,
                    created_at="2026-06-10T06:01:00+00:00",
                    suffix="holdout",
                )
                impact = measure_optimization_promotion_impact(signed, store_obj=store, limit=10, min_applied_samples=1, min_holdout_samples=1)
            finally:
                store.conn.close()

        self.assertTrue(impact["ok"])
        self.assertEqual(impact["summary"]["actual_matched_metadata_row_count"], 2)
        self.assertEqual(impact["summary"]["actual_canary_applied_count"], 1)
        self.assertEqual(impact["summary"]["actual_canary_holdout_count"], 1)
        self.assertFalse(impact["privacy"]["raw_prompts_included"])
        self.assertFalse(impact["privacy"]["request_ids_included"])
        self._assert_privacy_clean(impact)

    def test_feedback_status_result_is_metadata_only(self):
        result = feedback.managed_feedback_status_result(
            FakeFeedbackStore(),
            source_surface="openai_responses",
            sample_limit=5,
        )

        self.assertEqual(result["schema"], "agentflow.managed_feedback_status.v1")
        self.assertEqual(result["summary"]["queued"], 1)
        self.assertEqual(result["summary"]["due"], 1)
        self.assertFalse(result["due_samples"][0]["payload_included"])
        self.assertTrue(result["privacy"]["metadata_only"])

    def test_feedback_status_counts_openai_optimization_lifecycle_cohorts(self):
        payload = {
            "schema": "agentflow.openai_optimization_lifecycle_feedback.v1",
            "event_type": "openai_optimization_lifecycle",
            "provider": "openai",
            "source_surface": "openai_optimization_lifecycle",
            "endpoint": "responses",
            "status_bucket": "2xx",
            "retry_bucket": "none",
            "family_events": [
                {
                    "action_family": "routing",
                    "cohort": "applied",
                    "selected": True,
                    "eligible": True,
                    "status": "applied",
                    "candidate_id": "routing-candidate",
                    "rule_id": "routing-rule",
                    "reason_codes": ["selected-canary"],
                },
                {
                    "action_family": "old_context_summary",
                    "cohort": "holdout",
                    "selected": False,
                    "eligible": True,
                    "status": "holdout",
                    "candidate_id": "summary-candidate",
                    "rule_id": "summary-rule",
                    "reason_codes": ["missing-holdout"],
                },
                {
                    "action_family": "cache_replay",
                    "cohort": "invalidated",
                    "selected": False,
                    "eligible": True,
                    "status": "invalidated",
                    "candidate_id": "cache-candidate",
                    "rule_id": "cache-rule",
                    "reason_codes": ["cache-replay-invalidation-missing"],
                },
                {
                    "action_family": "routing",
                    "cohort": "fallback",
                    "selected": False,
                    "eligible": True,
                    "status": "applied",
                    "candidate_id": "routing-candidate",
                    "rule_id": "routing-rule",
                    "reason_codes": ["rate-limited"],
                },
                {
                    "action_family": "old_context_summary",
                    "cohort": "safety_stop",
                    "selected": False,
                    "eligible": True,
                    "status": "safety-stopped",
                    "candidate_id": "summary-candidate",
                    "rule_id": "summary-rule",
                    "reason_codes": ["stale-evidence"],
                },
                {
                    "action_family": "cache_replay",
                    "cohort": "suppressed",
                    "selected": False,
                    "eligible": True,
                    "status": "hit",
                    "candidate_id": "cache-candidate",
                    "rule_id": "cache-rule",
                    "reason_codes": ["conflicts-with-selected-family"],
                },
            ],
            "privacy": {"metadata_only": True, "raw_payload_included": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="openai-lifecycle-queue-1",
                    created_at=now,
                    updated_at=now,
                    source_surface="openai_optimization_lifecycle",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(payload),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                result = feedback.managed_feedback_status_result(
                    store,
                    source_surface="openai_optimization_lifecycle",
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        lifecycle = result["openai_optimization_lifecycle"]
        self.assertEqual(lifecycle["schema"], "agentflow.openai_optimization_lifecycle_queue_status.v1")
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertEqual(lifecycle["family_event_count"], 6)
        cohorts = {item["value"]: item["count"] for item in lifecycle["cohort_breakdown"]}
        for cohort in ("applied", "holdout", "suppressed", "invalidated", "safety_stop", "fallback"):
            self.assertEqual(cohorts[cohort], 1)
        family_cohorts = {item["value"]: item["count"] for item in lifecycle["family_cohort_breakdown"]}
        self.assertEqual(family_cohorts["routing:applied"], 1)
        self.assertEqual(family_cohorts["old_context_summary:safety_stop"], 1)
        self.assertEqual(family_cohorts["cache_replay:suppressed"], 1)
        self.assertFalse(lifecycle["payload_json_included"])

    def test_feedback_status_includes_routing_experiment_queue_summary(self):
        event = {
            "schema": "agentflow.routing_experiment_outcome_event.v1",
            "event_type": "routing_experiment_outcome",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "candidate": {
                "candidate_bucket": "tool-result:sonnet->haiku",
                "requested_model_family": "sonnet",
                "routed_model_family": "haiku",
                "shadow_model_family": "sonnet",
            },
            "outcome": {
                "status": "compared",
                "primary_status_class": "2xx",
                "shadow_status_class": "2xx",
                "output_similarity": 0.91,
            },
            "reason_codes": ["passed"],
            "privacy": {"metadata_only": True, "raw_prompts_included": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="routing-exp-queue-1",
                    created_at=now,
                    updated_at=now,
                    source_surface="routing_experiment_outcome",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(event),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                result = feedback.managed_feedback_status_result(
                    store,
                    source_surface="routing_experiment_outcome",
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        self.assertEqual(result["routing_experiments"]["schema"], "agentflow.routing_experiment_feedback_queue_status.v1")
        self.assertEqual(result["routing_experiments"]["queue_rows"], 1)
        self.assertEqual(result["routing_experiments"]["status_breakdown"], [{"value": "queued", "count": 1}])
        self.assertEqual(result["routing_experiments"]["outcome_status_breakdown"], [{"value": "compared", "count": 1}])
        self.assertEqual(result["routing_experiments"]["reason_code_breakdown"], [{"value": "passed", "count": 1}])
        self.assertFalse(result["routing_experiments"]["payload_json_included"])

    def test_shadow_routing_policy_event_egress_blocks_raw_payload_before_queue(self):
        raw_event = {
            "schema": "agentflow.routing_experiment_outcome_event.v1",
            "event_type": "routing_experiment_outcome",
            "source_surface": "codex_turn",
            "candidate": {
                "candidate_bucket": "summary:gpt-5-codex->gpt-5-mini",
                "requested_model_family": "gpt-5",
                "routed_model_family": "gpt-5-mini",
                "prompt": "raw prompt must not leave",
                "messages": [{"content": "raw message must not leave"}],
                "provider_body": {"input": "raw provider body must not leave"},
                "tool_payload": {"arguments": "raw tool payload must not leave"},
                "file_path": "/tmp/shadow-routing-secret.py",
                "cache_key": "cache-key-secret",
                "request_id": "request-id-secret",
                "session_id": "session-id-secret",
                "tenant_id": "tenant-id-secret",
                "account_id": "account-id-secret",
                "authorization": "Bearer auth-secret",
                "api_key": "sk-shadow-secret",
            },
            "outcome": {"status": "compared", "primary_status_class": "2xx", "shadow_status_class": "2xx"},
            "reason_codes": ["passed"],
            "privacy": {"metadata_only": True, "raw_prompts_included": False},
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                with patch.dict(
                    "os.environ",
                    {
                        "AGENTFLOW_RECOMMENDATION_ENABLED": "0",
                        "AGENTFLOW_RECOMMENDATION_SERVER_URL": "",
                    },
                    clear=False,
                ):
                    meta = asyncio.run(
                        recommendations.queue_policy_event_feedback(
                            store,
                            raw_event,
                            source_surface="routing_experiment_outcome",
                            queue_when_disabled=True,
                            flush_immediately=False,
                        )
                    )
                rows = store.managed_outcome_feedback_rows(source_surface="routing_experiment_outcome", limit=10)
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "skipped")
        self.assertEqual(meta["reason"], "unsafe-egress-payload")
        self.assertEqual(meta["fallback"], "local-policy")
        self.assertTrue(meta["egress_guard"]["blocked"])
        self.assertFalse(meta["egress_guard"]["raw_values_logged"])
        self.assertEqual(rows, [])
        rendered = json.dumps(meta, sort_keys=True)
        for secret in (
            "raw prompt must not leave",
            "raw message must not leave",
            "raw provider body must not leave",
            "raw tool payload must not leave",
            "/tmp/shadow-routing-secret.py",
            "cache-key-secret",
            "request-id-secret",
            "session-id-secret",
            "tenant-id-secret",
            "account-id-secret",
            "auth-secret",
            "sk-shadow-secret",
        ):
            self.assertNotIn(secret, rendered)
        self.assertIn("prompt", meta["egress_guard"]["blocked_keys"])
        self.assertIn("messages", meta["egress_guard"]["blocked_keys"])
        self.assertIn("provider_body", meta["egress_guard"]["blocked_keys"])

    def test_feedback_status_includes_codex_canary_lifecycle_queue_summary(self):
        base_event = {
            "schema": "agentflow.codex_app_canary_lifecycle_feedback.v1",
            "event_type": "codex_app_canary_lifecycle",
            "source_surface": "codex_turn",
            "app_family": "codex",
            "lifecycle_kind": "codex_app_canary",
            "action_family": "routing",
            "policy_id": "local-codex-app-summary-model-hint-canary",
            "workflow_phase": "summary",
            "canary_cohort": "canary_applied",
            "outcome": {"status": "applied", "status_class": "success"},
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "request_ids_included": False,
                "cache_keys_included": False,
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="codex-canary-queue-1",
                    created_at=now,
                    updated_at=now,
                    source_surface="codex_app_canary_lifecycle",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(base_event),
                    status="queued",
                    attempts=0,
                    next_attempt_at=now,
                )
                cache_event = dict(base_event)
                cache_event["action_family"] = "cache"
                cache_event["policy_id"] = "local-codex-app-exact-cache-canary"
                cache_event["canary_cohort"] = "canary_holdout"
                cache_event["outcome"] = {"status": "holdout", "status_class": "success"}
                store.enqueue_managed_outcome_feedback(
                    id="codex-canary-sent-1",
                    created_at=now,
                    updated_at=now,
                    source_surface="codex_app_canary_lifecycle",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(cache_event),
                    status="sent",
                    attempts=1,
                    next_attempt_at=now,
                    sent_at=now,
                )
                result = feedback.managed_feedback_status_result(
                    store,
                    source_surface="codex_app_canary_lifecycle",
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        codex_status = result["codex_app_canaries"]
        self.assertEqual(codex_status["schema"], "agentflow.codex_app_canary_lifecycle_queue_status.v1")
        self.assertEqual(codex_status["queue_rows"], 2)
        self.assertEqual(codex_status["queue_state_breakdown"], [
            {"value": "pending", "count": 1},
            {"value": "sent", "count": 1},
        ])
        self.assertEqual(codex_status["action_family_breakdown"], [
            {"value": "cache", "count": 1},
            {"value": "routing", "count": 1},
        ])
        self.assertEqual(codex_status["rule_id_breakdown"], [
            {"value": "local-codex-app-exact-cache-canary", "count": 1},
            {"value": "local-codex-app-summary-model-hint-canary", "count": 1},
        ])
        self.assertEqual(codex_status["candidate_id_breakdown"], [
            {"value": "local-codex-app-exact-cache-canary", "count": 1},
            {"value": "local-codex-app-summary-model-hint-canary", "count": 1},
        ])
        self.assertEqual(len(codex_status["rule_candidate_breakdown"]), 2)
        self.assertFalse(codex_status["rule_candidate_breakdown"][0].get("payload_json_included", False))
        self.assertFalse(codex_status["payload_json_included"])

    def test_feedback_status_includes_routing_promotion_lifecycle_summary(self):
        base_event = {
            "event_type": "impact",
            "metadata": {
                "schema": "agentflow.optimization_promotion_lifecycle_feedback.v1",
                "lifecycle_kind": "optimization_promotion_canary",
                "command": "optimization-promotion-impact",
                "local_result_status": "ok",
                "action_snapshots": [
                    {
                        "action_id": "promotion-rollout-action:routing-status",
                        "action_type": "widen",
                        "status": "matched",
                        "action_family": "routing",
                        "optimization_family": "phase_routing",
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "policy_section": "routing",
                        "policy_source": "managed-recommended",
                        "target_candidate_id": "routing-status-candidate",
                        "target_rule_id": "promotion-routing-status",
                        "requested_model_family": "sonnet",
                        "routed_model_family": "haiku",
                        "model_family_pair": "sonnet->haiku",
                        "actual_cohort_counts": {
                            "canary_applied": 2,
                            "canary_holdout": 1,
                            "safety_stopped": 1,
                        },
                        "observed_savings_usd": 0.012,
                        "error_rate_delta": 0.25,
                        "retry_rate_delta": 0.5,
                        "latency_avg_delta_ms": 1200,
                        "error_buckets": [{"value": "rate_limited", "count": 1}],
                        "reason_buckets": [{"value": "local-canary-safety-stop", "count": 1}],
                        "next_step_verdict": "rollback",
                        "next_step_reason_codes": ["safety-stop-observed", "rollback-error-rate"],
                    }
                ],
                "privacy": {
                    "metadata_only": True,
                    "raw_prompts_included": False,
                    "raw_responses_included": False,
                    "request_ids_included": False,
                    "local_session_ids_included": False,
                    "file_paths_included": False,
                    "cache_keys_included": False,
                },
            },
        }

        disable_event = {
            "event_type": "apply",
            "metadata": {
                "schema": "agentflow.optimization_promotion_lifecycle_feedback.v1",
                "lifecycle_kind": "optimization_promotion_canary",
                "command": "optimization-promotion-apply",
                "local_result_status": "ok",
                "action_snapshots": [
                    {
                        "action_id": "promotion-rollout-action:routing-disable",
                        "action_type": "disable",
                        "status": "planned",
                        "action_family": "routing",
                        "source_surface": "openai_responses",
                        "policy_section": "routing",
                        "policy_source": "managed-recommended",
                        "target_candidate_id": "routing-disable-candidate",
                        "target_rule_id": "promotion-routing-disable",
                        "requested_model_family": "gpt-5",
                        "routed_model_family": "gpt-5-mini",
                        "model_family_pair": "gpt-5->gpt-5-mini",
                        "projected_cohort_counts": {"canary_applied": 3, "canary_holdout": 2},
                    }
                ],
                "privacy": {"metadata_only": True},
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                now = utc_now()
                store.enqueue_managed_outcome_feedback(
                    id="routing-promotion-queued",
                    created_at=now,
                    updated_at=now,
                    source_surface="optimization_promotion_lifecycle",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(base_event),
                    status="retryable-error",
                    attempts=1,
                    next_attempt_at=now,
                )
                store.enqueue_managed_outcome_feedback(
                    id="routing-promotion-sent",
                    created_at=now,
                    updated_at=now,
                    source_surface="optimization_promotion_lifecycle",
                    endpoint="/v1/policy-events",
                    optimization_unit_id=0,
                    payload_json=stable_json(disable_event),
                    status="sent",
                    attempts=1,
                    next_attempt_at=now,
                    sent_at=now,
                )
                result = feedback.managed_feedback_status_result(
                    store,
                    source_surface="optimization_promotion_lifecycle",
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        lifecycle = result["routing_promotion_lifecycle"]
        self.assertEqual(lifecycle["schema"], "agentflow.routing_promotion_lifecycle_queue_status.v1")
        self.assertEqual(lifecycle["queue_rows"], 2)
        self.assertEqual(lifecycle["action_count"], 2)
        self.assertEqual(lifecycle["queue_state_breakdown"], [
            {"value": "pending", "count": 1},
            {"value": "sent", "count": 1},
        ])
        self.assertIn({"value": "rollback", "count": 1}, lifecycle["outcome_status_breakdown"])
        self.assertIn({"value": "disable", "count": 1}, lifecycle["action_type_breakdown"])
        self.assertIn({"value": "safety_stopped", "count": 1}, lifecycle["cohort_count_breakdown"])
        self.assertIn({"value": "rate_limited", "count": 1}, lifecycle["error_bucket_breakdown"])
        self.assertIn({"value": "sonnet->haiku", "count": 1}, lifecycle["model_family_pair_breakdown"])
        by_candidate = {item["candidate_id"]: item for item in lifecycle["candidate_breakdown"]}
        self.assertEqual(by_candidate["routing-status-candidate"]["safety_stopped_count"], 1)
        self.assertEqual(by_candidate["routing-status-candidate"]["applied_minus_holdout_error_rate"], 0.25)
        self.assertFalse(lifecycle["payload_json_included"])
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("session_id", rendered)
        self.assertNotIn("raw prompt", rendered)

    def test_openai_outcome_summary_is_feature_only(self):
        routing_meta = {"reason": "test"}
        openai_outcomes.attach_openai_outcome_summary(
            path="/v1/responses",
            requested_model="gpt-5-codex",
            routed_model="gpt-5-mini",
            status_code=200,
            latency_ms=12,
            retry_count=0,
            input_tokens_est=20,
            output_tokens_est=4,
            actual_input_tokens=21,
            actual_output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            thinking_output_tokens=None,
            cost_est_usd=0.001,
            cost_baseline_usd=0.002,
            cache_meta={"status": "miss", "reason": "exact-miss"},
            crunch_meta={"changed": False},
            routing_meta=routing_meta,
            category="chat",
            session_id="raw session secret",
        )

        summary = routing_meta["openai_outcome_unit"]
        self.assertEqual(summary["schema"], "agentflow.openai_outcome_summary.v1")
        self.assertEqual(summary["source_surface"], "openai_responses")
        self.assertFalse(summary["raw_payload_included"])
        self.assertEqual(managed_egress_violations(summary), [])
        self.assertNotIn("raw session secret", json.dumps(summary, sort_keys=True))

    def test_openai_managed_outcome_feedback_records_queue_metadata(self):
        store = FakeOutcomeStore()
        routing_meta = {
            "reason": "test",
            "managed_recommendation": {
                "enabled": True,
                "optimization_unit_id": 77,
                "policy_id": "policy-1",
            },
        }
        queued = {
            "queued": True,
            "queue_id": "queue-1",
            "source_surface": "openai_responses",
        }

        async def fake_queue(store_obj, managed, outcome, *, source_surface):
            self.assertIs(store_obj, store)
            self.assertEqual(source_surface, "openai_responses")
            self.assertEqual(outcome["provider"], "openai")
            self.assertEqual(managed_egress_violations(outcome), [])
            return queued

        with patch("agentflow_proxy.optimization.openai_outcomes.queue_outcome_feedback", fake_queue):
            asyncio.run(
                openai_outcomes.record_managed_outcome_feedback(
                    store=store,
                    call_id="call-1",
                    path="/v1/responses",
                    requested_model="gpt-5-codex",
                    routed_model="gpt-5-mini",
                    status_code=200,
                    latency_ms=12,
                    retry_count=0,
                    input_tokens_est=20,
                    output_tokens_est=4,
                    actual_input_tokens=21,
                    actual_output_tokens=5,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    thinking_output_tokens=None,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    cache_meta={"status": "miss", "reason": "exact-miss"},
                    crunch_meta={"changed": False},
                    routing_meta=routing_meta,
                    category="chat",
                    session_id="raw session secret",
                )
            )

        self.assertEqual(routing_meta["managed_recommendation"]["outcome_feedback"], queued)
        self.assertEqual(store.updated[0][0], "call-1")
        self.assertEqual(store.updated[0][1]["managed_recommendation"]["outcome_feedback"], queued)


class TestPostPromotionPriorityDeltaReview(unittest.TestCase):
    def _fixture(self) -> dict:
        return {
            "schema": "agentflow.post_promotion_policy_priority_deltas.v1",
            "deltas": [
                {
                    "delta_id": "post-promo-delta:routing:widen:crunch-savings",
                    "rank": 1,
                    "status": "recommended",
                    "next_action": "widen-local-policy",
                    "action_family": "routing",
                    "source_surface": "crunch_proxy_request",
                    "recommendation_type": "widen-crunch-threshold",
                    "savings_delta_usd": 55.50,
                    "confidence": 0.92,
                    "policy_section": "crunch",
                    "local_executor_compatibility": {
                        "status": "compatible",
                        "local_action_family": "routing",
                    },
                    "evidence_summary": {
                        "record_count": 200,
                        "promotion_status": "promoted",
                        "rank_score": 0.92,
                        "savings_delta_usd": 55.50,
                    },
                    "prompt": "raw prompt must not appear",
                    "messages": [{"content": "raw message must not appear"}],
                    "request_id": "secret-request-id",
                    "session_id": "secret-session-id",
                    "cache_key": "secret-cache-key",
                },
                {
                    "delta_id": "post-promo-delta:crunch:rollback:below-threshold",
                    "rank": 2,
                    "status": "recommended",
                    "next_action": "rollback-local-policy",
                    "action_family": "crunch",
                    "source_surface": "crunch_proxy_request",
                    "recommendation_type": "rollback-crunch-rule",
                    "savings_delta_usd": -12.00,
                    "confidence": 0.85,
                    "evidence_summary": {"record_count": 80, "promotion_status": "demoted"},
                },
                {
                    "delta_id": "post-promo-delta:cache:keep:low-confidence",
                    "rank": 3,
                    "status": "noop",
                    "next_action": "keep-blocked",
                    "action_family": "cache",
                    "source_surface": "cache_proxy_request",
                    "recommendation_type": "noop",
                    "savings_delta_usd": 0.0,
                    "confidence": 0.30,
                    "no_op_reasons": ["low-confidence", "insufficient-evidence"],
                    "evidence_summary": {"record_count": 5},
                },
            ],
        }

    def test_ranks_widen_before_rollback_before_keep_blocked(self):
        from agentflow_proxy.post_promotion_priority_delta_review import build_post_promotion_priority_delta_review

        result = build_post_promotion_priority_delta_review(self._fixture(), limit=10)

        self.assertEqual(result["schema"], "agentflow.post_promotion_priority_delta_review.v1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["wrote_local_policy_files"])
        self.assertFalse(result["provider_calls_made"])

        summary = result["summary"]
        self.assertEqual(summary["source_delta_count"], 3)
        self.assertEqual(summary["review_candidate_count"], 3)
        self.assertEqual(summary["recommended_count"], 2)
        self.assertEqual(summary["noop_count"], 1)

        candidates = result["candidates"]
        self.assertEqual(len(candidates), 3)
        # widen must appear before rollback in the routing group
        routing_group = next(g for g in result["groups"] if g["action_family"] == "routing")
        self.assertEqual(routing_group["rank"], 1)
        self.assertEqual(routing_group["top_next_action"], "widen-local-policy")

        # noop item appears in omitted_actions with explicit no_op_reasons
        omitted = result["omitted_actions"]
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]["next_action"], "keep-blocked")
        self.assertIn("low-confidence", omitted[0]["no_op_reasons"])
        self.assertIn("insufficient-evidence", omitted[0]["no_op_reasons"])

    def test_strips_raw_identifiers_from_output(self):
        from agentflow_proxy.post_promotion_priority_delta_review import build_post_promotion_priority_delta_review

        result = build_post_promotion_priority_delta_review(self._fixture(), limit=10)
        encoded = json.dumps(result, sort_keys=True)

        for forbidden in (
            "raw prompt must not appear",
            "raw message must not appear",
            "secret-request-id",
            "secret-session-id",
            "secret-cache-key",
            '"prompt"',
            '"messages"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
        ):
            self.assertNotIn(forbidden, encoded, msg=f"forbidden token found in output: {forbidden!r}")

    def test_empty_payload_returns_ok_no_candidates(self):
        from agentflow_proxy.post_promotion_priority_delta_review import build_post_promotion_priority_delta_review

        result = build_post_promotion_priority_delta_review({}, limit=10)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["review_candidate_count"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["groups"], [])
        self.assertFalse(result["privacy"]["provider_calls_made"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])

    def test_schema_mismatch_adds_validation_warning(self):
        from agentflow_proxy.post_promotion_priority_delta_review import build_post_promotion_priority_delta_review

        result = build_post_promotion_priority_delta_review({"schema": "unknown.schema.v1", "deltas": []})

        self.assertFalse(result["validation"]["ok"])
        self.assertTrue(len(result["validation"]["warnings"]) > 0)

    def test_cli_reads_fixture_from_stdin_and_emits_report(self):
        from agentflow_proxy.cli_commands.optimization_reports import post_promotion_priority_delta_review_cli
        import io

        stdin = io.StringIO(json.dumps(self._fixture()))
        stdout = io.StringIO()

        rc = post_promotion_priority_delta_review_cli(["-"], stdin=stdin, stdout=stdout)

        self.assertEqual(rc, 0)
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["schema"], "agentflow.post_promotion_priority_delta_review.v1")
        self.assertTrue(out["ok"])
        self.assertEqual(out["summary"]["review_candidate_count"], 3)
        self.assertFalse(out["provider_calls_made"])
        self.assertFalse(out["managed_server_calls_made"])

    def test_cli_no_managed_url_returns_empty_report(self):
        from agentflow_proxy.cli_commands.optimization_reports import post_promotion_priority_delta_review_cli
        import io

        stdout = io.StringIO()
        rc = post_promotion_priority_delta_review_cli([], stdout=stdout)

        self.assertEqual(rc, 0)
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["schema"], "agentflow.post_promotion_priority_delta_review.v1")
        self.assertEqual(out["summary"]["review_candidate_count"], 0)
        self.assertEqual(out["fetch"]["status"], "skipped")
        self.assertEqual(out["fetch"]["reason"], "no-managed-url-configured")


if __name__ == "__main__":
    unittest.main()
