import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import uuid

from agentflow_proxy import cli
from agentflow_proxy.orchestrator_research import (
    build_evidence_to_activation_burndown,
    build_evidence_to_activation_next_action_ledger,
    build_research_plan,
)
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


NOW = datetime(2026, 6, 11, 8, 40, tzinfo=timezone.utc)


def issue(number, title, labels, *, repo="lutzkuen/agentflow", author="lutzkuen", state="OPEN", updated="2026-06-11T08:00:00Z"):
    return {
        "repo": repo,
        "number": number,
        "title": title,
        "state": state,
        "url": f"https://github.com/{repo}/issues/{number}",
        "author": {"login": author},
        "labels": [{"name": name} for name in labels],
        "updatedAt": updated,
    }


class OrchestratorResearchPlanTests(unittest.TestCase):
    def test_no_ready_issues_enters_research_and_creates_actionable_issue(self):
        plan = build_research_plan(
            issues=[
                issue(10, "Blocked old milestone", ["status:blocked", "priority:p1"], updated="2026-05-01T08:00:00Z"),
                issue(11, "External idea", ["status:ready", "priority:p1"], author="external"),
            ],
            stats={"calls": 5266, "cache_hit_rate": 0.0, "today_cost_usd": 12.34},
            threshold=3,
            now=NOW,
        )

        self.assertTrue(plan["research_trigger"]["should_run"])
        self.assertEqual(plan["research_trigger"]["reason"], "ready-actionable-count-below-threshold")
        self.assertEqual(plan["research_trigger"]["actionable_ready_count"], 0)
        created = plan["backlog_changes"]["create_issues"]
        self.assertGreaterEqual(len(created), 1)
        self.assertIn("Acceptance Criteria", created[0]["body"])
        self.assertIn("Implementation Approach", created[0]["body"])
        self.assertIn("## Labels", created[0]["body"])
        self.assertIn("- status:ready", created[0]["body"])
        self.assertIn("status:ready", created[0]["labels"])
        self.assertIn("Top ranked optimization candidate:", created[0]["body"])
        self.assertIn("next_backlog_milestone", created[0]["body"])
        self.assertNotIn("Recent metadata summary:", created[0]["body"])

        milestone = plan["evidence"]["next_backlog_milestone"]
        self.assertEqual(milestone["schema"], "agentflow.next_backlog_milestone.v1")
        self.assertEqual(milestone["status"], "ready")
        self.assertEqual(milestone["summary"]["proposal_count"], len(created))
        self.assertGreaterEqual(milestone["summary"]["ranked_candidate_count"], 1)
        self.assertEqual(milestone["issues"][0]["title"], created[0]["title"])
        self.assertIn("priority", milestone["issues"][0])
        self.assertTrue(milestone["privacy"]["metadata_only"])
        self.assertTrue(milestone["privacy"]["aggregate_only"])
        self.assertFalse(milestone["privacy"]["raw_prompts_included"])
        self.assertFalse(milestone["privacy"]["provider_bodies_included"])
        self.assertFalse(milestone["privacy"]["request_ids_included"])
        self.assertFalse(milestone["privacy"]["session_ids_included"])

    def test_low_backlog_emits_ranked_metadata_only_optimization_candidates(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "cache rollout omitted reason=aggregate-only candidate_id=cache-candidate-secret",
                        "routing safety blocker=safety-stop request_id=req-candidate-secret",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                stats={
                    "calls": 2778,
                    "cache_hits": 0,
                    "cache_hit_rate": 0.0,
                    "today_crunch_savings_usd": 1.25,
                    "routing": [
                        {
                            "provider": "anthropic",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-sonnet-4-6",
                            "c": 1248,
                        },
                        {
                            "provider": "openai",
                            "requested_model": "gpt-5.4",
                            "routed_model": "gpt-5.4",
                            "c": 198,
                        },
                    ],
                    "cache_decision_breakdown": [
                        {
                            "source_surface": "openai_responses",
                            "status": "skipped",
                            "reason": "streaming",
                            "count": 700,
                        }
                    ],
                    "streaming_cache_hit_recovery": {
                        "schema": "agentflow.streaming_cache_hit_recovery.v1",
                        "summary": {
                            "recovery_verdict": "store-missing",
                            "eligible_calls": 8,
                            "replay_attempts": 8,
                            "successful_hits": 0,
                        },
                        "verdict_breakdown": [{"value": "store-missing", "count": 1}],
                        "cohorts": [
                            {
                                "provider": "anthropic",
                                "source_surface": "anthropic_messages",
                                "category": "summary",
                                "workflow_phase": "summary",
                                "policy_id": "policy-id:public",
                                "rule_id": "rule-id:public",
                                "candidate_id": "candidate-id:cache-candidate-secret",
                                "recovery_verdict": "store-missing",
                                "cache_key": "cache-key-secret",
                                "session_id": "session-secret",
                                "file_path": "/tmp/secret.py",
                            }
                        ],
                        "privacy": {
                            "metadata_only": True,
                            "aggregate_only": True,
                            "cache_keys_included": False,
                            "request_ids_included": False,
                            "session_ids_included": False,
                            "file_paths_included": False,
                        },
                    },
                },
                log_sources=[log_path],
                threshold=3,
                now=NOW,
            )

        candidates = plan["evidence"]["optimization_candidates"]
        recovery = plan["evidence"]["stats_summary"]["streaming_cache_hit_recovery"]
        self.assertEqual(recovery["summary"]["recovery_verdict"], "store-missing")
        rendered_plan = json.dumps(plan, sort_keys=True)
        self.assertNotIn("cache-key-secret", rendered_plan)
        self.assertNotIn("session-secret", rendered_plan)
        self.assertNotIn("/tmp/secret.py", rendered_plan)
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual([item["rank"] for item in candidates], list(range(1, len(candidates) + 1)))
        for candidate in candidates:
            for field in (
                "lever",
                "blocker",
                "estimated_savings_path",
                "confidence",
                "sequencing",
                "repo",
                "privacy",
                "projected_savings_signal",
            ):
                self.assertIn(field, candidate)
            self.assertTrue(candidate["privacy"]["metadata_only"])
            self.assertTrue(candidate["privacy"]["aggregate_only"])
            self.assertFalse(candidate["privacy"]["raw_prompts_included"])
            self.assertFalse(candidate["privacy"]["provider_bodies_included"])
            self.assertFalse(candidate["privacy"]["request_ids_included"])
            self.assertFalse(candidate["privacy"]["session_ids_included"])
            self.assertFalse(candidate["privacy"]["individual_candidate_ids_included"])

        levers = {candidate["lever"] for candidate in candidates}
        self.assertIn("cache", levers)
        self.assertIn("routing", levers)
        self.assertIn("crunch", levers)
        rendered = json.dumps(plan)
        self.assertNotIn("cache-candidate-secret", rendered)
        self.assertNotIn("req-candidate-secret", rendered)

    def test_thin_low_backlog_plan_expands_into_implementation_ready_milestone(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "cache replay omitted reason=aggregate-only candidate_id=private-candidate-secret",
                        "routing gate blocked blocker=aggregate-only request_id=req-private-secret",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[
                    issue(
                        90,
                        "Stage routing evidence for gpt-5.4 to gpt-5.4-mini",
                        ["backlog", "status:closed", "priority:p1", "core-feature"],
                        state="CLOSED",
                    )
                ],
                stats={
                    "calls": 2487,
                    "cache_hits": 0,
                    "cache_hit_rate": 0.0,
                    "today_crunch_savings_usd": 0.0,
                    "routing": [
                        {
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "requested_model": "gpt-5.4",
                            "routed_model": "gpt-5.4",
                            "category": "chat",
                            "c": 210,
                        },
                        {
                            "provider": "anthropic",
                            "source_surface": "anthropic_messages",
                            "endpoint": "messages",
                            "requested_model": "claude-sonnet-4-6",
                            "routed_model": "claude-sonnet-4-6",
                            "category": "tool-result",
                            "reason": "thinking-context-blocked",
                            "c": 956,
                        },
                    ],
                },
                log_sources=[log_path],
                threshold=3,
                now=NOW,
            )

        created = plan["backlog_changes"]["create_issues"]
        titles = [item["title"] for item in created]
        self.assertGreaterEqual(len(created), 6)
        self.assertLessEqual(len(created), 10)
        self.assertNotIn("Stage routing evidence for gpt-5.4 to gpt-5.4-mini", titles)
        self.assertEqual(len(titles), len(set(titles)))

        required_labels = {"backlog", "status:ready", "correctness"}
        core_feature_count = 0
        for proposal in created:
            labels = set(proposal["labels"])
            self.assertTrue(required_labels.issubset(labels))
            self.assertTrue(any(label.startswith("priority:") for label in labels))
            if "core-feature" in labels:
                core_feature_count += 1
            body = proposal["body"]
            self.assertIn("## Rationale", body)
            self.assertIn("## Evidence", body)
            self.assertIn("## Implementation Approach", body)
            self.assertIn("## Acceptance Criteria", body)
            self.assertIn("## Expected Savings Path Or Bottleneck Removed", body)
            self.assertIn("## Labels", body)
            for label in proposal["labels"]:
                self.assertIn(f"- {label}", body)
            self.assertIn("## Sequencing Notes", body)
        self.assertGreaterEqual(core_feature_count, len(created) // 2)

        rendered = json.dumps(plan)
        self.assertNotIn("private-candidate-secret", rendered)
        self.assertNotIn("req-private-secret", rendered)
        self.assertFalse(plan["privacy"]["raw_prompts_included"])
        self.assertFalse(plan["privacy"]["provider_bodies_included"])
        self.assertFalse(plan["privacy"]["request_ids_included"])
        self.assertFalse(plan["privacy"]["session_ids_included"])

    def test_promotion_blocker_status_suppresses_closed_titles_and_creates_successor(self):
        plan = build_research_plan(
            issues=[
                issue(
                    90,
                    "Stage routing evidence for gpt-5.4 to gpt-5.4-mini",
                    ["backlog", "status:closed", "priority:p1", "core-feature"],
                    state="CLOSED",
                )
            ],
            stats={
                "calls": 2487,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "chat",
                        "c": 210,
                    }
                ],
                "promotion_blocker_next_actions": {
                    "schema": "agentflow.promotion_blocker_next_actions_dashboard.v1",
                    "status": "available",
                    "summary": {
                        "review_candidate_count": 2,
                        "recommended_count": 1,
                        "noop_count": 1,
                        "projected_savings_usd": 18.75,
                        "top_local_action_family": "routing",
                        "top_blocker_reason": "ready-to-widen",
                        "top_safety_stop_reason": None,
                        "top_next_action": "widen_local_openai_canary",
                        "top_expected_local_executor": "openai-routing-canary",
                    },
                    "next_actions": [
                        {"value": "widen_local_openai_canary", "count": 1},
                        {"value": "keep-blocked", "count": 1},
                    ],
                    "groups": [
                        {
                            "local_action_family": "routing",
                            "candidate_count": 1,
                            "projected_savings_usd": 18.75,
                            "top_next_action": "widen_local_openai_canary",
                            "top_blocker_reason": "ready-to-widen",
                            "sample_recommendations": [
                                {
                                    "candidate_id": "promotion-blocker-candidate-secret",
                                    "request_id": "promotion-blocker-request-secret",
                                    "session_id": "promotion-blocker-session-secret",
                                    "cache_key": "promotion-blocker-cache-secret",
                                    "file_path": "/home/lutz/private/promotion_blocker_secret.py",
                                }
                            ],
                        }
                    ],
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "provider_bodies_included": False,
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertIn("Widen promotion blocker routing canary from next-action status", titles)
        self.assertNotIn("Stage routing evidence for gpt-5.4 to gpt-5.4-mini", titles)

        status = plan["evidence"]["stats_summary"]["promotion_blocker_next_action_status"]
        self.assertEqual(status["schema"], "agentflow.promotion_blocker_next_action_research_status.v1")
        self.assertEqual(status["summary"]["top_next_action"], "widen_local_openai_canary")
        self.assertEqual(status["summary"]["top_local_action_family"], "routing")
        self.assertTrue(status["privacy"]["metadata_only"])
        self.assertFalse(status["privacy"]["raw_prompts_included"])
        self.assertFalse(status["privacy"]["provider_bodies_included"])
        self.assertFalse(status["privacy"]["request_ids_included"])
        self.assertFalse(status["privacy"]["session_ids_included"])
        self.assertFalse(status["privacy"]["cache_keys_included"])
        self.assertFalse(status["privacy"]["individual_candidate_ids_included"])

        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertGreaterEqual(suppression["closed_prior_issue_count"], 1)
        suppressed_titles = [row["title"] for row in suppression["suppressed"]]
        self.assertIn("Stage routing evidence for gpt-5.4 to gpt-5.4-mini", suppressed_titles)
        closed = next(row for row in suppression["suppressed"] if row["title"] == "Stage routing evidence for gpt-5.4 to gpt-5.4-mini")
        self.assertEqual(closed["suppression_kind"], "closed-prior-issue")
        self.assertEqual(closed["existing_issue"]["number"], 90)

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("promotion-blocker-candidate-secret", rendered)
        self.assertNotIn("promotion-blocker-request-secret", rendered)
        self.assertNotIn("promotion-blocker-session-secret", rendered)
        self.assertNotIn("promotion-blocker-cache-secret", rendered)
        self.assertNotIn("/home/lutz/private/promotion_blocker_secret.py", rendered)
        self.assertFalse(plan["privacy"]["raw_prompts_included"])
        self.assertFalse(plan["privacy"]["provider_bodies_included"])
        self.assertFalse(plan["privacy"]["request_ids_included"])
        self.assertFalse(plan["privacy"]["session_ids_included"])

    def test_current_pass_through_routing_summary_is_ranked_into_activation_candidates(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2490,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4-mini",
                        "routed_model": "gpt-5.4-mini",
                        "category": "unknown",
                        "c": 1232,
                        "request_id": "req-secret-should-not-leak",
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "reason": "thinking-context-blocked",
                        "c": 956,
                        "session_id": "session-secret-should-not-leak",
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "chat",
                        "c": 212,
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-haiku-4-5-20251001",
                        "routed_model": "claude-haiku-4-5-20251001",
                        "category": "tool-result",
                        "c": 51,
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": None,
                        "category": "unknown",
                        "c": 38,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        self.assertEqual(report["schema"], "agentflow.pass_through_routing_activation_candidates.v1")
        self.assertEqual(report["summary"]["pass_through_rows"], 2489)
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])

        buckets = {
            (bucket["provider"], bucket["requested_model"], bucket["routed_model"]): bucket
            for bucket in report["buckets"]
        }
        gpt54 = buckets[("openai", "gpt-5.4", "gpt-5.4")]
        gpt54_mini = buckets[("openai", "gpt-5.4-mini", "gpt-5.4-mini")]
        self.assertEqual(gpt54["actionability"], "actionable")
        self.assertEqual(gpt54["candidate_target_model"], "gpt-5.4-mini")
        self.assertEqual(gpt54["required_local_executor"], "openai-routing-canary")
        self.assertGreater(gpt54["estimated_savings_per_1000_calls_usd"], 0)
        lifecycle = gpt54["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.openai_routing_canary_lifecycle_evidence.v1")
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 0)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 0)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 212)
        self.assertIn("missing-canary-lifecycle-evidence", lifecycle["blocker_codes"])
        self.assertIn("missing-applied-coverage", lifecycle["blocker_codes"])
        self.assertIn("missing-holdout-coverage", lifecycle["blocker_codes"])
        self.assertFalse(lifecycle["privacy"]["raw_prompts_included"])
        self.assertFalse(lifecycle["privacy"]["request_ids_included"])
        self.assertFalse(lifecycle["privacy"]["session_ids_included"])
        self.assertEqual(gpt54_mini["actionability"], "already-cheapest")
        self.assertIsNotNone(gpt54_mini["no_op_reason"])
        self.assertIsNone(gpt54_mini["openai_canary_lifecycle_evidence"])
        self.assertEqual(report["summary"]["openai_canary_applied_count"], 0)
        self.assertEqual(report["summary"]["openai_canary_holdout_count"], 0)

        classes = {row["class"] for row in report["actionability_breakdown"]}
        self.assertIn("actionable", classes)
        self.assertIn("already-cheapest", classes)
        self.assertIn("safety-blocked", classes)
        self.assertIn("unsupported-provider-action", classes)

        routing_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "routing")
        signal = routing_candidate["projected_savings_signal"]
        self.assertEqual(signal["actionability"], "actionable")
        self.assertEqual(signal["requested_model"], "gpt-5.4")
        self.assertEqual(signal["candidate_target_model"], "gpt-5.4-mini")

        rendered = json.dumps(plan)
        self.assertNotIn("req-secret-should-not-leak", rendered)
        self.assertNotIn("session-secret-should-not-leak", rendered)

    def test_pass_through_routing_report_merges_openai_canary_lifecycle_counts(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 40,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "chat",
                        "c": 20,
                        "openai_canary_holdout_count": 3,
                        "openai_canary_latest_observed_at": observed_at,
                        "session_id": "secret-holdout-session-id",
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4-mini",
                        "category": "chat",
                        "c": 2,
                        "openai_canary_applied_count": 2,
                        "openai_canary_error_count": 1,
                        "openai_canary_retry_count": 1,
                        "openai_canary_fallback_count": 1,
                        "openai_canary_latest_observed_at": observed_at,
                        "request_id": "secret-applied-request-id",
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        self.assertEqual(report["summary"]["pass_through_rows"], 20)
        self.assertEqual(report["summary"]["routed_down_rows"], 2)
        self.assertEqual(report["summary"]["openai_canary_applied_count"], 2)
        self.assertEqual(report["summary"]["openai_canary_holdout_count"], 3)
        self.assertEqual(report["summary"]["openai_canary_error_count"], 1)
        self.assertEqual(report["summary"]["openai_canary_retry_count"], 1)
        self.assertEqual(report["summary"]["openai_canary_fallback_count"], 1)

        candidate = report["buckets"][0]
        self.assertEqual(candidate["requested_model"], "gpt-5.4")
        self.assertEqual(candidate["candidate_target_model"], "gpt-5.4-mini")
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)
        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 3)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 20)
        self.assertGreater(lifecycle["coverage"]["applied_rate"], 0)
        self.assertGreater(lifecycle["coverage"]["holdout_rate"], 0)
        self.assertEqual(lifecycle["error_count"], 1)
        self.assertEqual(lifecycle["retry_count"], 1)
        self.assertEqual(lifecycle["fallback_count"], 1)
        self.assertFalse(lifecycle["stale_evidence"]["stale"])
        self.assertIn("error-observed", lifecycle["blocker_codes"])
        self.assertIn("retry-observed", lifecycle["blocker_codes"])
        self.assertIn("fallback-observed", lifecycle["blocker_codes"])
        self.assertNotIn("missing-applied-coverage", lifecycle["blocker_codes"])
        self.assertNotIn("missing-holdout-coverage", lifecycle["blocker_codes"])
        self.assertFalse(lifecycle["privacy"]["raw_prompts_included"])
        self.assertFalse(lifecycle["privacy"]["request_ids_included"])
        self.assertFalse(lifecycle["privacy"]["session_ids_included"])

        rendered = json.dumps(plan)
        self.assertNotIn("secret-holdout-session-id", rendered)
        self.assertNotIn("secret-applied-request-id", rendered)

    def test_pass_through_routing_report_names_anthropic_lifecycle_blockers(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 100,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "c": 100,
                        "session_id": "secret-anthropic-session-id",
                        "request_id": "secret-anthropic-request-id",
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        candidate = report["buckets"][0]
        self.assertEqual(candidate["provider"], "anthropic")
        self.assertEqual(candidate["actionability"], "actionable")
        self.assertEqual(candidate["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(candidate["required_local_executor"], "anthropic-routing-rules")
        self.assertGreater(candidate["estimated_savings_per_1000_calls_usd"], 0)

        lifecycle = candidate["anthropic_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.anthropic_routing_canary_lifecycle_evidence.v1")
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 0)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 0)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 100)
        self.assertEqual(lifecycle["error_count"], 0)
        self.assertEqual(lifecycle["retry_count"], 0)
        self.assertEqual(lifecycle["fallback_count"], 0)
        self.assertIn("missing-anthropic-canary-lifecycle-evidence", lifecycle["blocker_codes"])
        self.assertIn("missing-applied-coverage", lifecycle["blocker_codes"])
        self.assertIn("missing-holdout-coverage", lifecycle["blocker_codes"])
        self.assertFalse(lifecycle["privacy"]["raw_prompts_included"])
        self.assertFalse(lifecycle["privacy"]["request_ids_included"])
        self.assertFalse(lifecycle["privacy"]["session_ids_included"])
        self.assertEqual(report["summary"]["anthropic_canary_applied_count"], 0)
        self.assertEqual(report["summary"]["anthropic_canary_holdout_count"], 0)

        routing_stage = next(row for row in plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]["levers"] if row["lever"] == "routing")
        self.assertEqual(routing_stage["state"], "missing-evidence")
        self.assertIn("missing-anthropic-canary-lifecycle-evidence", routing_stage["blocker_codes"])
        self.assertEqual(routing_stage["next_action"], "activate-anthropic-routing-canary-cohorts")

        rendered = json.dumps(plan)
        self.assertNotIn("secret-anthropic-session-id", rendered)
        self.assertNotIn("secret-anthropic-request-id", rendered)

    def test_pass_through_routing_report_merges_anthropic_canary_lifecycle_counts(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 42,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "c": 40,
                        "anthropic_canary_holdout_count": 3,
                        "anthropic_canary_latest_observed_at": observed_at,
                        "session_id": "secret-anthropic-holdout-session-id",
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "c": 2,
                        "anthropic_canary_applied_count": 2,
                        "anthropic_canary_error_count": 1,
                        "anthropic_canary_retry_count": 1,
                        "anthropic_canary_fallback_count": 1,
                        "anthropic_canary_latest_observed_at": observed_at,
                        "request_id": "secret-anthropic-applied-request-id",
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        self.assertEqual(report["summary"]["pass_through_rows"], 40)
        self.assertEqual(report["summary"]["routed_down_rows"], 2)
        self.assertEqual(report["summary"]["anthropic_canary_applied_count"], 2)
        self.assertEqual(report["summary"]["anthropic_canary_holdout_count"], 3)
        self.assertEqual(report["summary"]["anthropic_canary_error_count"], 1)
        self.assertEqual(report["summary"]["anthropic_canary_retry_count"], 1)
        self.assertEqual(report["summary"]["anthropic_canary_fallback_count"], 1)

        candidate = report["buckets"][0]
        self.assertEqual(candidate["requested_model"], "claude-sonnet-4-6")
        self.assertEqual(candidate["candidate_target_model"], "claude-haiku-4-5-20251001")
        lifecycle = candidate["anthropic_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 3)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 40)
        self.assertGreater(lifecycle["coverage"]["applied_rate"], 0)
        self.assertGreater(lifecycle["coverage"]["holdout_rate"], 0)
        self.assertEqual(lifecycle["error_count"], 1)
        self.assertEqual(lifecycle["retry_count"], 1)
        self.assertEqual(lifecycle["fallback_count"], 1)
        self.assertFalse(lifecycle["stale_evidence"]["stale"])
        self.assertIn("error-observed", lifecycle["blocker_codes"])
        self.assertIn("retry-observed", lifecycle["blocker_codes"])
        self.assertIn("fallback-observed", lifecycle["blocker_codes"])
        self.assertNotIn("missing-applied-coverage", lifecycle["blocker_codes"])
        self.assertNotIn("missing-holdout-coverage", lifecycle["blocker_codes"])

        rendered = json.dumps(plan)
        self.assertNotIn("secret-anthropic-holdout-session-id", rendered)
        self.assertNotIn("secret-anthropic-applied-request-id", rendered)

    def test_evidence_to_activation_loop_tracks_missing_local_cohort_evidence(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "unknown",
                        "c": 223,
                        "request_id": "req-secret-loop",
                    }
                ],
                "crunch_savings_usd": 0.0,
                "today_crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
            },
            threshold=3,
            now=NOW,
        )

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        self.assertEqual(loop["schema"], "agentflow.evidence_to_activation_savings_loop.v1")
        self.assertEqual(loop["status"], "missing-evidence")
        self.assertEqual(loop["summary"]["top_lever"], "routing")
        self.assertEqual(loop["summary"]["top_next_action"], "activate-openai-routing-canary-cohorts")
        routing = next(row for row in loop["levers"] if row["lever"] == "routing")
        self.assertEqual(routing["state"], "missing-evidence")
        self.assertEqual(routing["requested_model"], "gpt-5.4")
        self.assertEqual(routing["candidate_target_model"], "gpt-5.4-mini")
        self.assertIn("missing-applied-coverage", routing["blocker_codes"])
        self.assertIn("missing-holdout-coverage", routing["blocker_codes"])
        self.assertTrue(loop["privacy"]["metadata_only"])
        self.assertTrue(loop["privacy"]["aggregate_only"])
        self.assertFalse(loop["privacy"]["raw_prompts_included"])
        self.assertFalse(loop["privacy"]["provider_bodies_included"])
        self.assertFalse(loop["privacy"]["request_ids_included"])
        self.assertFalse(loop["privacy"]["session_ids_included"])
        self.assertFalse(loop["privacy"]["cache_keys_included"])
        rendered = json.dumps(plan)
        self.assertNotIn("req-secret-loop", rendered)

    def test_evidence_to_activation_burndown_ranks_current_blocker_families(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing blocker=safety-stop request_id=req-secret-burndown path=/tmp/secret.py",
                        "routing blocker=safety-stop session_id=session-secret-burndown",
                    ]
                ),
                encoding="utf-8",
            )
            plan = build_research_plan(
                issues=[],
                stats={
                    "calls": 2626,
                    "cache_hits": 0,
                    "cache_hit_rate": 0.0,
                    "routing": [
                        {
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "requested_model": "gpt-5.4",
                            "routed_model": "gpt-5.4",
                            "category": "unknown",
                            "c": 244,
                        }
                    ],
                    "crunch_savings_usd": 0.0,
                    "today_crunch_savings_usd": 0.0,
                    "crunch_tokens_saved": 0,
                    "crunch_chars_saved": 0,
                },
                log_sources=[log_path],
                threshold=3,
                now=NOW,
            )

        report = build_evidence_to_activation_burndown(plan, now=NOW)

        self.assertEqual(report["schema"], "agentflow.evidence_to_activation_burndown.v1")
        self.assertEqual(report["summary"]["top_lever"], "routing")
        self.assertEqual(report["summary"]["top_next_action"], "activate-openai-routing-canary-cohorts")
        families = set(report["summary"]["represented_blocker_families"])
        self.assertIn("routing", families)
        self.assertIn("cache", families)
        self.assertIn("crunch", families)
        self.assertIn("request-shape-rollups", families)
        self.assertIn("managed-recommendation", families)
        self.assertGreaterEqual(report["summary"]["blocker_family_count"], 4)
        routing = next(row for row in report["blockers"] if row["lever"] == "routing")
        self.assertIn("missing-applied-coverage", routing["blocker_codes"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["absolute_paths_included"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("req-secret-burndown", rendered)
        self.assertNotIn("session-secret-burndown", rendered)
        self.assertNotIn("/tmp/secret.py", rendered)

    def test_evidence_to_activation_burndown_uses_promotion_feedback_over_stale_projected_rows(self):
        plan = build_research_plan(
            issues=[
                issue(
                    90,
                    "Stage routing evidence for gpt-5.4 to gpt-5.4-mini",
                    ["backlog", "priority:p1", "core-feature"],
                    state="CLOSED",
                    updated="2026-06-10T08:00:00Z",
                )
            ],
            stats={
                "calls": 100,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "tool-light",
                        "c": 100,
                    }
                ],
                "promotion_outcome_feedback": {
                    "schema": "agentflow.promotion_outcome_feedback_summary.v1",
                    "entry_count": 1,
                    "entries": [
                        {
                            "schema": "agentflow.promotion_outcome_feedback_entry.v1",
                            "created_at": "2026-06-11T08:30:00+00:00",
                            "policy_id": "policy-secret-burndown",
                            "candidate_id": "candidate-secret-burndown",
                            "request_id": "req-secret-promotion-feedback",
                            "session_id": "session-secret-promotion-feedback",
                            "action_family": "routing",
                            "policy_section": "routing",
                            "status": "positive",
                            "recommendation": "widen",
                            "reason_codes": ["promotion-ready"],
                            "observed_savings_usd": 0.25,
                            "projected_savings_usd": 0.4,
                            "applied_count": 8,
                            "holdout_count": 7,
                        }
                    ],
                    "summary": {
                        "entry_count": 1,
                        "status_counts": [{"value": "positive", "count": 1}],
                        "action_family_counts": [{"value": "routing", "count": 1}],
                        "observed_savings_usd": 0.25,
                        "projected_savings_usd": 0.4,
                    },
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "raw_provider_bodies_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                        "cache_keys_included": False,
                        "file_paths_included": False,
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        self.assertIn("promotion_outcome_feedback", plan["evidence"]["inspected_sources"])
        report = build_evidence_to_activation_burndown(plan, now=NOW)

        routing_rows = [row for row in report["blockers"] if row["lever"] == "routing"]
        self.assertEqual(len(routing_rows), 1)
        routing = routing_rows[0]
        self.assertEqual(routing["source"], "promotion-outcome-feedback")
        self.assertEqual(routing["state"], "measured-savings")
        self.assertEqual(routing["next_action"], "widen-local-promotion-from-outcome-feedback")
        self.assertEqual(routing["sample_count"], 15)
        self.assertEqual(routing["applied_count"], 8)
        self.assertEqual(routing["holdout_count"], 7)
        self.assertEqual(routing["observed_savings_usd"], 0.25)
        self.assertNotIn("activate-openai-routing-canary-cohorts", [row["next_action"] for row in routing_rows])
        self.assertIn("routing", report["summary"]["represented_blocker_families"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["individual_candidate_ids_included"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("policy-secret-burndown", rendered)
        self.assertNotIn("candidate-secret-burndown", rendered)
        self.assertNotIn("req-secret-promotion-feedback", rendered)
        self.assertNotIn("session-secret-promotion-feedback", rendered)
        self.assertNotIn("Stage routing evidence for gpt-5.4 to gpt-5.4-mini", rendered)

    def test_evidence_to_activation_burndown_cli_reads_plan_json(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 10,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "c": 10,
                    }
                ],
            },
            threshold=3,
            now=NOW,
        )
        with TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            stdout = io.StringIO()
            code = cli.evidence_to_activation_burndown_cli(["--plan-json", str(plan_path), "--pretty"], stdout=stdout)

        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["schema"], "agentflow.evidence_to_activation_burndown.v1")
        self.assertEqual(report["summary"]["top_lever"], "routing")
        self.assertGreaterEqual(report["summary"]["ranked_blocker_count"], 4)

    def test_evidence_to_activation_ledger_tracks_metadata_only_next_actions(self):
        summary = {
            "evidence_to_activation_loop": {
                "schema": "agentflow.evidence_to_activation_savings_loop.v1",
                "status": "activation-ready",
                "levers": [
                    {
                        "lever": "routing",
                        "state": "activation-ready",
                        "evidence_source": "agentflow.openai_canary_impact.v1",
                        "local_action_family": "routing",
                        "next_action": "widen_local_openai_canary",
                        "sample_count": 8,
                        "applied_count": 4,
                        "holdout_count": 4,
                        "savings_per_1000_calls_usd": 3.25,
                        "requested_model": "gpt-5.4",
                        "candidate_target_model": "gpt-5.4-mini",
                        "policy_id": "raw-ledger-policy-secret",
                    },
                    {
                        "lever": "cache",
                        "state": "missing-evidence",
                        "evidence_source": "agentflow.cache_replayability.v1",
                        "local_action_family": "cache",
                        "next_action": "resolve-cache-replayability-blocker",
                        "blocker_codes": ["invalidation-evidence-missing request_id=req-ledger-secret"],
                        "sample_count": 3,
                    },
                ],
            }
        }

        ledger = build_evidence_to_activation_next_action_ledger(summary)

        self.assertEqual(ledger["schema"], "agentflow.evidence_to_activation_next_action_ledger.v1")
        self.assertEqual(ledger["summary"]["top_next_action"], "widen_local_openai_canary")
        self.assertEqual(ledger["summary"]["top_current_status"], "holdout")
        self.assertEqual(ledger["summary"]["top_expected_savings_path"], "Move routing from local lifecycle evidence into the next canary, widening, or blocked-review step.")
        routing = ledger["entries"][0]
        self.assertEqual(routing["cohort_bucket"], "gpt-5.4->gpt-5.4-mini")
        self.assertEqual(routing["current_status"], "holdout")
        self.assertTrue(ledger["privacy"]["metadata_only"])
        self.assertTrue(ledger["privacy"]["aggregate_only"])
        self.assertFalse(ledger["privacy"]["raw_prompts_included"])
        self.assertFalse(ledger["privacy"]["provider_bodies_included"])
        self.assertFalse(ledger["privacy"]["request_ids_included"])
        self.assertFalse(ledger["privacy"]["session_ids_included"])
        self.assertFalse(ledger["privacy"]["cache_keys_included"])
        rendered = json.dumps(ledger, sort_keys=True)
        self.assertNotIn("raw-ledger-policy-secret", rendered)
        self.assertNotIn("req-ledger-secret", rendered)

    def test_closed_prior_issue_with_advanced_ledger_status_generates_next_stage_issue(self):
        stale_title = "Stage cache replay canary for activation-ready on openai/openai_responses/responses"
        plan = build_research_plan(
            issues=[issue(44, stale_title, ["backlog", "status:ready", "cache"], state="CLOSED")],
            stats={
                "calls": 120,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {"candidate_rows": 1, "activation_ready_count": 1, "projected_ready_hits": 30},
                    "cohorts": [
                        {
                            "readiness": "activation-ready",
                            "count": 31,
                            "projected_hits": 30,
                            "projected_saved_cost_usd": 0.42,
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "cohort_id": "raw-cache-cohort-secret",
                            "file_path": "/tmp/private-cache-ledger.py",
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        ledger_titles = [
            title for title in titles
            if title.startswith("Stage cache replay canary from evidence-to-activation ledger")
        ]
        self.assertEqual(len(ledger_titles), 1)
        self.assertIn("(evidence ", ledger_titles[0])
        self.assertNotIn(stale_title, titles)
        ledger_issue = next(item for item in plan["backlog_changes"]["create_issues"] if item["title"] == ledger_titles[0])
        self.assertIn("Fingerprint: activation:", ledger_issue["body"])
        self.assertIn("Continues closed predecessor: #44", ledger_issue["body"])
        self.assertIn(stale_title, ledger_issue["body"])
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        self.assertEqual(ledger["summary"]["top_current_status"], "staged")
        self.assertEqual(ledger["summary"]["closed_issue_seen_count"], 1)
        self.assertEqual(ledger["entries"][0]["issue_status"], "closed-issue-seen")
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["closed_prior_issue_count"], 1)
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-cache-cohort-secret", rendered)
        self.assertNotIn("/tmp/private-cache-ledger.py", rendered)

    def test_open_legacy_ledger_issue_suppresses_duplicate_proposal(self):
        stale_title = "Stage cache replay canary for activation-ready on openai/openai_responses/responses"
        plan = build_research_plan(
            issues=[issue(44, stale_title, ["backlog", "status:ready", "cache"], state="OPEN")],
            stats={
                "calls": 120,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {"candidate_rows": 1, "activation_ready_count": 1, "projected_ready_hits": 30},
                    "cohorts": [
                        {
                            "readiness": "activation-ready",
                            "count": 31,
                            "projected_hits": 30,
                            "projected_saved_cost_usd": 0.42,
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(
            any(title.startswith("Stage cache replay canary from evidence-to-activation ledger") for title in titles)
        )
        self.assertNotIn(stale_title, titles)
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertGreaterEqual(suppression["open_existing_issue_count"], 1)

    def test_evidence_to_activation_loop_reports_activation_progress(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "openai_canary_impact": {
                    "schema": "agentflow.openai_canary_impact.v1",
                    "summary": {
                        "candidate_count": 1,
                        "canary_applied_count": 2,
                        "canary_holdout_count": 1,
                    },
                    "candidates": [
                        {
                            "candidate_id": "raw-loop-routing-candidate",
                            "source_surface": "openai_responses",
                            "original_model": "gpt-5.4",
                            "candidate_target_model": "gpt-5.4-mini",
                            "verdict": "widen",
                            "next_action": "widen_local_openai_canary",
                            "reason_codes": ["target-savings-met"],
                            "sample_count": 3,
                            "cohort_counts": {
                                "canary_applied": 2,
                                "canary_holdout": 1,
                                "safety_stopped": 0,
                            },
                            "observed_savings_usd": 0.02,
                            "projected_savings_usd": 0.03,
                            "stale_evidence": {"stale": False},
                        }
                    ],
                    "activation_lifecycle_feedback": {
                        "queue_rows": 1,
                        "state_breakdown": [{"value": "healthy_canary", "count": 1}],
                        "cohort_lifecycle_metadata": [
                            {
                                "policy_ref": "policy:public",
                                "cohort_label": "canary_applied",
                                "action_family": "routing",
                                "event_count": 2,
                                "applied_count": 2,
                                "policy_id": "raw-loop-policy-secret",
                            },
                            {
                                "policy_ref": "policy:public",
                                "cohort_label": "canary_holdout",
                                "action_family": "routing",
                                "event_count": 1,
                                "holdout_count": 1,
                                "policy_id": "raw-loop-policy-secret",
                            },
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                },
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {"candidate_rows": 1, "activation_ready_count": 1, "projected_ready_hits": 2},
                    "cohorts": [
                        {
                            "readiness": "activation-ready",
                            "count": 3,
                            "projected_hits": 2,
                            "projected_saved_cost_usd": 0.04,
                            "cohort_id": "raw-loop-cache-cohort",
                        }
                    ],
                },
                "old_context_summary_opportunity": {
                    "schema": "agentflow.old_context_summary_opportunity.v1",
                    "summary": {
                        "candidate_count": 1,
                        "projected_saved_tokens": 5000,
                        "projected_saved_usd": 0.25,
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        self.assertEqual(loop["status"], "activation-ready")
        self.assertGreaterEqual(loop["summary"]["progressed_count"], 3)
        self.assertEqual(loop["summary"]["activation_ready_count"], 2)
        self.assertEqual(loop["summary"]["top_lever"], "routing")
        routing = next(row for row in loop["levers"] if row["lever"] == "routing")
        cache = next(row for row in loop["levers"] if row["lever"] == "cache")
        crunch = next(row for row in loop["levers"] if row["lever"] == "crunch")
        self.assertEqual(routing["state"], "activation-ready")
        self.assertEqual(routing["applied_count"], 2)
        self.assertEqual(routing["holdout_count"], 1)
        self.assertEqual(cache["state"], "replay-ready")
        self.assertEqual(cache["projected_hits"], 2)
        self.assertEqual(crunch["state"], "projected-savings")
        self.assertEqual(crunch["projected_saved_usd"], 0.25)
        rendered = json.dumps(plan)
        self.assertNotIn("raw-loop-routing-candidate", rendered)
        self.assertNotIn("raw-loop-policy-secret", rendered)
        self.assertNotIn("raw-loop-cache-cohort", rendered)

    def test_cache_replay_ledger_prefers_observed_local_replay_evidence(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {"candidate_rows": 1, "activation_ready_count": 1, "projected_ready_hits": 108},
                    "cohorts": [
                        {
                            "readiness": "activation-ready",
                            "count": 116,
                            "projected_hits": 108,
                            "projected_saved_cost_usd": 0.235509,
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "category": "chat",
                            "cohort_id": "raw-projected-cache-cohort",
                        }
                    ],
                },
                "request_shape_rollup_candidates": {
                    "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "summary": {
                            "replay_ready_cohort_count": 1,
                            "projected_hits": 108,
                            "projected_savings_usd": 0.235509,
                        },
                        "cohorts": [
                            {
                                "readiness": "replay-ready",
                                "row_count": 116,
                                "projected_hits": 108,
                                "projected_savings_usd": 0.235509,
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "candidate_id": "raw-request-shape-cache-candidate",
                            }
                        ],
                    },
                },
                "openai_cache_replay_impact": {
                    "schema": "agentflow.openai_cache_replay_impact.v1",
                    "status": "matched",
                    "summary": {
                        "observed_openai_cache_replay_metadata_row_count": 4,
                        "applied_count": 2,
                        "holdout_count": 1,
                        "projected_hits": 3,
                        "projected_saved_usd": 0.09,
                        "actual_hits": 1,
                        "actual_saved_cost_usd": 0.03,
                        "miss_count": 1,
                        "bypass_skipped_count": 1,
                    },
                    "candidates": [
                        {
                            "candidate_id": "raw-observed-cache-candidate",
                            "rule_id": "raw-observed-cache-rule",
                            "readiness": "replay-ready",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "category": "chat",
                            "verdict": "hold",
                            "next_action": "collect_more_applied_and_holdout_cache_replay_evidence",
                            "sample_count": 4,
                            "applied_count": 2,
                            "holdout_count": 1,
                            "actual_hits": 1,
                            "actual_saved_cost_usd": 0.03,
                            "miss_count": 1,
                            "bypass_skipped_count": 1,
                            "projected_hits": 3,
                            "projected_saved_usd": 0.09,
                            "reason_codes": ["target-savings-met"],
                            "canary_hit_measurement": {
                                "schema": "agentflow.openai_cache_replay_canary_hit_measurement.v1",
                                "observed_hits": 1,
                                "holdout_count": 1,
                                "raw_request_id": "req-observed-cache-secret",
                            },
                        }
                    ],
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                        "cache_keys_included": False,
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        cache = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache["state"], "measured-savings")
        self.assertEqual(cache["evidence_source"], "agentflow.openai_cache_replay_impact.v1")
        self.assertEqual(cache["applied_count"], 2)
        self.assertEqual(cache["holdout_count"], 1)
        self.assertEqual(cache["actual_hits"], 1)
        self.assertAlmostEqual(cache["actual_saved_cost_usd"], 0.03)
        self.assertEqual(cache["projected_hits"], 3)

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        cache_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "cache")
        self.assertEqual(cache_entry["evidence_schema"], "agentflow.openai_cache_replay_impact.v1")
        self.assertEqual(cache_entry["current_status"], "holdout")
        self.assertEqual(cache_entry["next_action"], "collect_more_applied_and_holdout_cache_replay_evidence")
        self.assertEqual(cache_entry["actual_hits"], 1)
        self.assertAlmostEqual(cache_entry["actual_saved_cost_usd"], 0.03)
        self.assertEqual(cache_entry["miss_count"], 1)
        self.assertEqual(cache_entry["bypass_skipped_count"], 1)
        self.assertEqual(cache_entry["projected_hits"], 3)
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-observed-cache-candidate", rendered)
        self.assertNotIn("raw-observed-cache-rule", rendered)
        self.assertNotIn("req-observed-cache-secret", rendered)
        self.assertNotIn("raw-projected-cache-cohort", rendered)
        self.assertNotIn("raw-request-shape-cache-candidate", rendered)

    def test_crunch_candidate_ranks_projected_savings_report(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "old_context_summary_opportunity": {
                    "schema": "agentflow.old_context_summary_opportunity.v1",
                    "summary": {
                        "scanned_call_count": 500,
                        "candidate_count": 4,
                        "projected_saved_chars": 24000,
                        "projected_saved_tokens": 6000,
                        "projected_saved_usd": 0.42,
                    },
                    "blocker_reason_breakdown": [
                        {"value": "needs-dry-run candidate_id=crunch-secret-candidate", "count": 4}
                    ],
                    "candidates": [
                        {
                            "candidate_id": "raw-crunch-candidate-secret",
                            "session_id": "raw-crunch-session-secret",
                            "file_path": "/home/lutz/private/crunch_secret.py",
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["schema"], "agentflow.crunch_savings_signal.v1")
        self.assertEqual(signal["status"], "projected-savings-ranked")
        self.assertEqual(signal["top_report"]["report_key"], "old_context_summary_opportunity")
        self.assertEqual(signal["top_report"]["projected_saved_usd"], 0.42)
        self.assertEqual(signal["top_report"]["candidate_count"], 4)
        self.assertTrue(signal["privacy"]["metadata_only"])
        self.assertTrue(signal["privacy"]["aggregate_only"])

        crunch_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "crunch")
        self.assertEqual(crunch_candidate["blocker"], "crunch-projected-savings-ranked")
        self.assertEqual(crunch_candidate["provider_surface_bucket"], "old_context_summary_opportunity")
        self.assertEqual(crunch_candidate["projected_savings_signal"]["status"], "projected-savings-ranked")

        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertIn("Rank crunch savings follow-up for crunch-projected-savings-ranked", created_titles)
        rendered = json.dumps(plan)
        self.assertNotIn("raw-crunch-candidate-secret", rendered)
        self.assertNotIn("raw-crunch-session-secret", rendered)
        self.assertNotIn("/home/lutz/private/crunch_secret.py", rendered)
        self.assertNotIn("crunch-secret-candidate", rendered)

    def test_crunch_candidate_ranks_aggregate_measurement_when_no_savings_signal_exists(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["status"], "non-positive-projection")
        self.assertNotIn("crunch-opportunity-report", signal["missing_measurements"])
        self.assertIn("positive-observed-or-projected-savings", signal["missing_measurements"])
        self.assertEqual(signal["top_report"]["report_key"], "aggregate_crunch_measurement")
        self.assertEqual(signal["top_report"]["rows_considered"], 2483)
        self.assertEqual(signal["top_report"]["applied_count"], 0)
        self.assertEqual(signal["top_report"]["skipped_count"], 2483)
        self.assertEqual(signal["top_report"]["next_action"], "inspect-crunch-coverage-and-projection")
        self.assertEqual(signal["top_report"]["no_op_reason"], "no-observed-or-projected-crunch-savings")
        self.assertTrue(signal["privacy"]["metadata_only"])
        self.assertTrue(signal["privacy"]["aggregate_only"])

        crunch_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "crunch")
        self.assertEqual(crunch_candidate["blocker"], "crunch-non-positive-projection")
        self.assertEqual(crunch_candidate["projected_savings_signal"]["status"], "non-positive-projection")

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        crunch_stage = next(stage for stage in loop["levers"] if stage["lever"] == "crunch")
        self.assertEqual(crunch_stage["state"], "no-op")
        self.assertEqual(crunch_stage["next_action"], "inspect-crunch-coverage-and-projection")

    def test_crunch_candidate_uses_request_shape_crunch_opportunity_dry_run(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 30, "rollup_count": 1},
                    "crunch_opportunity_dry_run": {
                        "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                        "status": "ranked",
                        "summary": {
                            "candidate_count": 1,
                            "matched_count": 24,
                            "recommended_action_count": 1,
                            "projected_saved_chars": 48000,
                            "projected_saved_tokens": 12000,
                            "projected_saved_usd": 0.036,
                            "activation_state": "activation-ready",
                            "top_next_action": "stage-repeated-context-crunch-canary",
                        },
                        "activation_follow_up": {
                            "schema": "agentflow.request_shape_crunch_activation_follow_up.v1",
                            "activation_state": "activation-ready",
                            "next_action": "stage-repeated-context-crunch-canary",
                            "missing_measurements": [],
                            "privacy": {"metadata_only": True, "aggregate_only": True},
                        },
                        "blocker_reason_breakdown": [
                            {"value": "session_id=raw-session-id-secret should not leak", "count": 1}
                        ],
                        "cohorts": [
                            {
                                "candidate_id": "raw-crunch-shape-secret",
                                "session_id": "raw-session-id-secret",
                                "cache_key": "cache-secret",
                                "file_path": "/home/lutz/private/shape_secret.py",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["status"], "projected-savings-ranked")
        self.assertEqual(signal["top_report"]["report_key"], "request_shape_crunch_opportunity")
        self.assertEqual(signal["top_report"]["schema"], "agentflow.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(signal["top_report"]["projected_saved_tokens"], 12000)
        self.assertEqual(signal["top_report"]["projected_saved_usd"], 0.036)
        self.assertEqual(signal["top_report"]["recommended_action_count"], 1)
        self.assertEqual(signal["top_report"]["activation_state"], "activation-ready")
        self.assertEqual(signal["top_report"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(signal["missing_measurements"], [])

        crunch_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "crunch")
        self.assertEqual(crunch_candidate["blocker"], "crunch-projected-savings-ranked")
        self.assertEqual(crunch_candidate["provider_surface_bucket"], "request_shape_crunch_opportunity")
        self.assertEqual(crunch_candidate["projected_savings_signal"]["top_report"]["matched_count"], 24)
        rendered = json.dumps(plan)
        self.assertNotIn("raw-crunch-shape-secret", rendered)
        self.assertNotIn("raw-session-id-secret", rendered)
        self.assertNotIn("cache-secret", rendered)
        self.assertNotIn("/home/lutz/private/shape_secret.py", rendered)

    def test_crunch_candidate_preserves_activation_missing_measurement(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 30, "rollup_count": 1},
                    "crunch_opportunity_dry_run": {
                        "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                        "status": "canary-staged",
                        "summary": {
                            "candidate_count": 1,
                            "matched_count": 24,
                            "projected_saved_chars": 48000,
                            "projected_saved_tokens": 12000,
                            "projected_saved_usd": 0.036,
                            "activation_state": "measurement-required",
                            "top_next_action": "measure-repeated-context-crunch-canary-impact",
                        },
                        "activation_follow_up": {
                            "schema": "agentflow.request_shape_crunch_activation_follow_up.v1",
                            "activation_state": "measurement-required",
                            "next_action": "measure-repeated-context-crunch-canary-impact",
                            "missing_measurements": ["missing-crunch-canary-impact-measurement"],
                            "privacy": {"metadata_only": True, "aggregate_only": True},
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["status"], "projected-savings-ranked")
        self.assertEqual(signal["top_report"]["activation_state"], "measurement-required")
        self.assertEqual(signal["top_report"]["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(signal["missing_measurements"], ["missing-crunch-canary-impact-measurement"])

    def test_crunch_candidate_prefers_request_shape_crunch_canary_impact(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 30, "rollup_count": 1},
                    "crunch_canary_impact": {
                        "schema": "agentflow.request_shape_crunch_canary_impact.v1",
                        "status": "widen-ready",
                        "summary": {
                            "candidate_count": 1,
                            "observed_canary_metadata_row_count": 12,
                            "applied_count": 6,
                            "holdout_count": 6,
                            "saved_chars": 24000,
                            "saved_tokens": 6000,
                            "saved_usd": 0.024,
                            "top_blocker_code": None,
                            "next_action": "widen-repeated-context-crunch-canary",
                        },
                        "candidates": [
                            {
                                "policy_id": "raw-policy-secret must not leak",
                                "session_id": "raw-session-id-secret",
                                "cache_key": "cache-secret",
                                "file_path": "/home/lutz/private/shape_secret.py",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "crunch_opportunity_dry_run": {
                        "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                        "status": "ranked",
                        "summary": {
                            "candidate_count": 1,
                            "matched_count": 24,
                            "projected_saved_chars": 48000,
                            "projected_saved_tokens": 12000,
                            "projected_saved_usd": 0.036,
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["top_report"]["report_key"], "request_shape_crunch_canary_impact")
        self.assertEqual(signal["top_report"]["schema"], "agentflow.request_shape_crunch_canary_impact.v1")
        self.assertEqual(signal["top_report"]["applied_count"], 6)
        self.assertEqual(signal["top_report"]["matched_count"], 12)
        self.assertEqual(signal["top_report"]["projected_saved_tokens"], 6000)
        self.assertEqual(signal["top_report"]["next_action"], "widen-repeated-context-crunch-canary")
        rendered = json.dumps(plan)
        self.assertNotIn("raw-policy-secret", rendered)
        self.assertNotIn("raw-session-id-secret", rendered)
        self.assertNotIn("cache-secret", rendered)
        self.assertNotIn("/home/lutz/private/shape_secret.py", rendered)

    def test_managed_recommendation_health_ranks_omissions_and_local_representation(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "managed_recommendations": {
                    "schema": "agentflow.managed_recommendations.v1",
                    "summary": {
                        "window_calls": 400,
                        "metadata_rows": 40,
                        "received_count": 10,
                        "applied_count": 0,
                        "observed_savings_usd": 0.0,
                    },
                    "reason_breakdown": [
                        {
                            "value": "provider-capability-mismatch",
                            "count": 12,
                            "local_action": "routing",
                        },
                        {
                            "value": "prompt-replacement-omitted",
                            "count": 3,
                            "local_action": "prompt_replacement",
                            "candidate_id": "managed-candidate-secret",
                        },
                    ],
                    "recommendation_health": {
                        "latest_fetch_review": {
                            "rows": [
                                {
                                    "kind": "omitted_candidate",
                                    "code": "no-local-representation",
                                    "candidate_id": "health-candidate-secret",
                                    "details": {
                                        "local_action": "prompt_replacement",
                                        "reason": "server-content-processing",
                                    },
                                }
                            ]
                        }
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["schema"], "agentflow.managed_recommendation_handoff_health.v1")
        self.assertEqual(signal["status"], "omission-reasons-ranked")
        self.assertEqual(signal["summary"]["local_file_backed_count"], 1)
        self.assertGreaterEqual(signal["summary"]["no_local_representation_count"], 1)
        top = signal["top_omission"]
        self.assertEqual(top["omitted_reason"], "prompt-replacement-omitted")
        self.assertEqual(top["local_action_family"], "prompt-replacement")
        self.assertFalse(top["local_file_backed_representation"]["exists"])
        self.assertEqual(
            top["local_file_backed_representation"]["reason"],
            "server-content-processing-not-local-policy",
        )
        routing_omission = next(row for row in signal["omissions"] if row["omitted_reason"] == "provider-capability-mismatch")
        self.assertTrue(routing_omission["local_file_backed_representation"]["exists"])
        self.assertEqual(routing_omission["local_file_backed_representation"]["rule_file"], "routing_rules.yaml")

        managed_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "managed-recommendation")
        self.assertEqual(managed_candidate["blocker"], "managed-recommendation-no-local-representation")
        self.assertEqual(managed_candidate["projected_savings_signal"]["top_omission"]["omitted_reason"], "prompt-replacement-omitted")
        self.assertTrue(managed_candidate["privacy"]["metadata_only"])

        rendered = json.dumps(plan)
        self.assertNotIn("managed-candidate-secret", rendered)
        self.assertNotIn("health-candidate-secret", rendered)
        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertFalse(signal["privacy"]["request_ids_included"])
        self.assertFalse(signal["privacy"]["session_ids_included"])

    def test_managed_recommendation_candidate_records_missing_health_report(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "c": 244,
                    },
                    {
                        "provider": "openai",
                        "requested_model": "gpt-5.4-mini",
                        "routed_model": "gpt-5.4-mini",
                        "c": 1255,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["status"], "missing-managed-recommendation-health-report")
        self.assertTrue(signal["top_omission"]["omitted_reason"].startswith("managed-recommendation-health-report-missing"))
        self.assertEqual(signal["top_omission"]["local_action_family"], "routing")
        self.assertEqual(signal["top_omission"]["local_file_backed_representation"]["rule_file"], "routing_rules.yaml")
        self.assertEqual(signal["top_omission"]["follow_up_owner"], "local-policy")
        self.assertEqual(signal["top_omission"]["next_action"], "activate-openai-routing-canary-cohorts")
        self.assertGreaterEqual(signal["summary"]["ranked_omission_count"], 2)
        self.assertGreaterEqual(signal["summary"]["local_file_backed_count"], 2)
        self.assertGreater(signal["summary"]["omitted_count"], 0)
        self.assertIn("managed_recommendations_report", signal["missing_measurements"])
        self.assertTrue(signal["top_omission"]["local_file_backed_representation"]["exists"])
        self.assertIn("cache", {row["local_action_family"] for row in signal["omissions"]})

        managed_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "managed-recommendation")
        self.assertTrue(managed_candidate["blocker"].startswith("managed-recommendation-health-report-missing"))
        self.assertEqual(managed_candidate["safety_status"], "review-required")
        self.assertEqual(managed_candidate["projected_savings_signal"]["status"], "missing-managed-recommendation-health-report")

    def test_managed_recommendation_handoff_reports_omitted_local_action_reason(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2714,
                "managed_recommendations": {
                    "schema": "agentflow.managed_recommendations.v1",
                    "summary": {
                        "window_calls": 2714,
                        "metadata_rows": 50,
                        "received_count": 0,
                        "applied_count": 0,
                        "observed_savings_usd": 0.0,
                    },
                    "reason_breakdown": [
                        {
                            "value": "repeated-context-crunch-opportunity",
                            "count": 2714,
                            "local_action": "crunch",
                        },
                    ],
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["schema"], "agentflow.managed_recommendation_handoff_health.v1")
        self.assertEqual(signal["status"], "omission-reasons-ranked")

        self.assertIn("omitted_local_action_reason", signal)
        self.assertEqual(signal["omitted_local_action_reason"], "repeated-context-crunch-opportunity")

        self.assertIn("top_local_file_backed_exists", signal)
        self.assertTrue(signal["top_local_file_backed_exists"])

        self.assertNotIn("omitted_local_action_reason", signal.get("missing_measurements") or [])

        top = signal["top_omission"]
        self.assertEqual(top["omitted_reason"], "repeated-context-crunch-opportunity")
        self.assertTrue(top["local_file_backed_representation"]["exists"])
        self.assertEqual(top["local_file_backed_representation"]["rule_file"], "crunch_rules.yaml")

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        managed_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "managed-recommendation")
        self.assertEqual(managed_entry["current_status"], "projected")
        self.assertEqual(managed_entry["omitted_reason"], "repeated-context-crunch-opportunity")
        self.assertEqual(managed_entry["local_action_family"], "crunch")
        self.assertEqual(managed_entry["managed_dependency"], "optional")
        self.assertTrue(managed_entry["local_file_backed_representation"]["exists"])
        self.assertEqual(managed_entry["local_file_backed_representation"]["policy_section"], "crunch")

        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertTrue(signal["privacy"]["metadata_only"])

    def test_missing_managed_report_ledger_points_to_projected_local_policy_handoff(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2767,
                "request_shape_crunch_opportunity": {
                    "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                    "status": "projected-savings-ranked",
                    "summary": {
                        "rows_considered": 2767,
                        "matched_count": 724,
                        "projected_saved_usd": 3.865624,
                        "projected_saved_tokens": 1301438,
                        "activation_state": "activation-ready",
                        "next_action": "stage-repeated-context-crunch-canary",
                        "top_blocker": "repeated-context-crunch-opportunity",
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["status"], "missing-managed-recommendation-health-report")
        self.assertEqual(signal["summary"]["managed_dependency"], "optional")
        self.assertEqual(signal["top_omission"]["local_action_family"], "crunch")
        self.assertEqual(signal["top_omission"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(signal["top_omission"]["follow_up_owner"], "local-policy")
        self.assertTrue(signal["top_omission"]["local_file_backed_representation"]["exists"])
        self.assertEqual(signal["top_omission"]["local_file_backed_representation"]["rule_file"], "crunch_rules.yaml")

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        managed_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "managed-recommendation")
        self.assertEqual(managed_entry["current_status"], "projected")
        self.assertEqual(managed_entry["follow_up_owner"], "local-policy")
        self.assertEqual(managed_entry["managed_dependency"], "optional")
        self.assertEqual(managed_entry["local_action_family"], "crunch")
        self.assertEqual(managed_entry["next_action"], "stage-repeated-context-crunch-canary")
        self.assertTrue(managed_entry["omitted_reason"].startswith("managed-recommendation-health-report-missing"))
        self.assertTrue(managed_entry["local_file_backed_representation"]["exists"])
        self.assertEqual(managed_entry["local_file_backed_representation"]["policy_section"], "crunch")

    def test_missing_managed_report_ranks_local_policy_handoff_omissions(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2873,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "request_shape_crunch_opportunity": {
                    "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                    "status": "projected-savings-ranked",
                    "summary": {
                        "rows_considered": 749,
                        "matched_count": 749,
                        "candidate_count": 11,
                        "projected_saved_usd": 4.086506,
                        "projected_saved_tokens": 1375441,
                        "activation_state": "activation-ready",
                        "next_action": "stage-repeated-context-crunch-canary",
                        "top_blocker": "repeated-context-crunch-opportunity",
                    },
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "provider_bodies_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                    },
                },
                "routing": [
                    {
                        "provider": "anthropic",
                        "source_surface": "unknown",
                        "endpoint": "unknown",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "c": 1197,
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4-mini",
                        "routed_model": "gpt-5.4-mini",
                        "category": "chat",
                        "c": 640,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["status"], "missing-managed-recommendation-health-report")
        self.assertEqual(signal["managed_dependency"], "optional")
        self.assertEqual(signal["summary"]["managed_dependency"], "optional")
        self.assertEqual(signal["summary"]["no_local_representation_count"], 0)
        self.assertGreaterEqual(signal["summary"]["ranked_omission_count"], 3)

        by_family = {row["local_action_family"]: row for row in signal["omissions"]}
        self.assertIn("crunch", by_family)
        self.assertIn("routing", by_family)
        self.assertIn("cache", by_family)
        self.assertEqual(signal["top_omission"]["local_action_family"], "crunch")
        self.assertTrue(
            signal["top_omission"]["omitted_reason"].startswith(
                "managed-recommendation-health-report-missing:repeated-context-crunch-opportunity"
            )
        )

        for family, rule_file in {
            "crunch": "crunch_rules.yaml",
            "routing": "routing_rules.yaml",
            "cache": "cache_rules.yaml",
        }.items():
            row = by_family[family]
            self.assertTrue(row["local_file_backed_representation"]["exists"])
            self.assertEqual(row["local_file_backed_representation"]["rule_file"], rule_file)
            self.assertEqual(row["managed_dependency"], "optional")
            self.assertEqual(row["follow_up_owner"], "local-policy")
            self.assertIn(row["next_action"], row["local_handoff_reason"])
            self.assertIn(rule_file, row["local_handoff_reason"])
            self.assertTrue(row["local_handoff_reason"].startswith("local-file-backed-policy-handoff:"))

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        managed_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "managed-recommendation")
        self.assertEqual(managed_entry["managed_dependency"], "optional")
        self.assertEqual(managed_entry["local_action_family"], "crunch")
        self.assertIn("crunch_rules.yaml", managed_entry["local_handoff_reason"])
        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertFalse(signal["privacy"]["request_ids_included"])
        self.assertFalse(signal["privacy"]["session_ids_included"])

    def test_request_shape_rollup_report_ranks_repeated_context_candidate(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {
                        "rows_considered": 40,
                        "rollup_count": 2,
                    },
                    "rollups": [
                        {
                            "candidate_id": "request-shape:secret-candidate-id",
                            "provider_family": "anthropic",
                            "source_surface": "anthropic_messages",
                            "endpoint": "messages",
                            "requested_model_family": "claude-sonnet",
                            "routed_model_family": "claude-sonnet",
                            "category": "tool-result",
                            "workflow_phase": "tool-execution",
                            "stream": True,
                            "has_tools": True,
                            "text_bucket": "32k_128k_chars",
                            "token_bucket": "8k_32k_tokens",
                            "cache_status": "skipped",
                            "routing_status": "passthrough",
                            "row_count": 24,
                            "error_count": 0,
                            "retry_count": 1,
                            "cost_est_usd": 0.96,
                            "observed_savings_usd": 0.0,
                            "candidate_work_classes": ["repeated_context", "replayability", "crunch"],
                            "candidate_families": ["cache_replay", "cache_blocker"],
                            "blocker_codes": ["unsupported-streaming-shape"],
                            "metadata": {
                                "cache_key": "cache-key-secret",
                                "request_id": "request-secret",
                                "session_id": "session-secret",
                                "file_path": "/tmp/shape-secret.py",
                            },
                        },
                        {
                            "candidate_id": "request-shape:smaller-secret",
                            "provider_family": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "category": "chat",
                            "workflow_phase": "chat",
                            "stream": False,
                            "has_tools": False,
                            "text_bucket": "lt_2k_chars",
                            "token_bucket": "lt_500_tokens",
                            "cache_status": "miss",
                            "routing_status": "passthrough",
                            "row_count": 2,
                            "cost_est_usd": 0.01,
                            "candidate_families": ["routing_candidate"],
                            "blocker_codes": ["exact-cache-miss"],
                        },
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["request_shape_rollup_candidates"]
        self.assertEqual(signal["schema"], "agentflow.request_shape_rollup_candidate_signal.v1")
        self.assertEqual(signal["status"], "candidates-ranked")
        self.assertEqual(signal["summary"]["ranked_candidate_count"], 2)
        self.assertEqual(signal["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        top = signal["top_candidate"]
        self.assertEqual(top["provider_surface_bucket"], "anthropic/anthropic_messages/messages")
        self.assertEqual(top["row_count"], 24)
        self.assertIn("repeated_context", top["candidate_work_classes"])
        self.assertIn("replayability", top["candidate_work_classes"])
        self.assertIn("unsupported-streaming-shape", top["blocker_codes"])

        shape_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "request-shape-rollups")
        self.assertEqual(shape_candidate["blocker"], "request-shape-stage-repeated-context-crunch-canary")
        self.assertEqual(shape_candidate["provider_surface_bucket"], "anthropic/anthropic_messages/messages")
        self.assertEqual(shape_candidate["projected_savings_signal"]["top_candidate"]["row_count"], 24)
        self.assertTrue(shape_candidate["privacy"]["metadata_only"])

        created = plan["backlog_changes"]["create_issues"]
        shape_issues = [item for item in created if item["title"] == "Stage request-shape repeated-context crunch canary"]
        self.assertEqual(len(shape_issues), 1)
        self.assertIn("stage-repeated-context-crunch-canary", shape_issues[0]["body"])
        rendered = json.dumps(plan)
        self.assertNotIn("secret-candidate-id", rendered)
        self.assertNotIn("smaller-secret", rendered)
        self.assertNotIn("cache-key-secret", rendered)
        self.assertNotIn("request-secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("/tmp/shape-secret.py", rendered)
        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertFalse(signal["privacy"]["request_ids_included"])
        self.assertFalse(signal["privacy"]["session_ids_included"])
        self.assertFalse(signal["privacy"]["cache_keys_included"])

    def test_zero_hit_cache_ladder_generates_cache_replay_issue(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2778,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_zero_hit_blocker_ladder": {
                    "schema": "agentflow.cache_zero_hit_blocker_ladder.v1",
                    "summary": {
                        "scanned_rows": 1000,
                        "cache_hits": 0,
                        "zero_hit_window": True,
                        "top_blocker_code": "skipped-streaming",
                        "top_next_action_family": "stage-replay-policy",
                    },
                    "ladder": [
                        {
                            "blocker_code": "skipped-streaming",
                            "provider": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "stream_mode": "stream",
                            "tool_presence": "no-tools",
                            "replayability_level": "features_only",
                            "cache_policy_source": "local-default",
                            "cache_status": "skipped",
                            "cache_reason": "streaming",
                            "next_action_family": "stage-replay-policy",
                            "next_action_label": "stage streaming-safe cache replay policy",
                            "count": 700,
                            "cache_key": "cache-secret-should-redact",
                            "session_id": "session-secret-should-redact",
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        created = plan["backlog_changes"]["create_issues"]
        cache_issues = [item for item in created if "cache replay" in item["title"].lower()]
        self.assertGreaterEqual(len(cache_issues), 1)
        body = cache_issues[0]["body"]
        self.assertIn("Top blocker cohort: skipped-streaming on openai/openai_responses/responses", body)
        self.assertIn("Local action needed: stage streaming-safe cache replay policy", body)
        self.assertIn("Activation mode: research-only", body)
        self.assertIn("hit recovery", body)
        self.assertIn("safe bypass/blocker reduction", body)
        self.assertIn("cache", cache_issues[0]["labels"])
        rendered = json.dumps(plan)
        self.assertNotIn("cache-secret-should-redact", rendered)
        self.assertNotIn("session-secret-should-redact", rendered)

    def test_request_shape_replayability_dry_run_names_cache_blocker_before_zero_hit(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 100,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {"candidate_rows": 0},
                    "cohorts": [],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "cache_zero_hit_blocker_ladder": {
                    "schema": "agentflow.cache_zero_hit_blocker_ladder.v1",
                    "summary": {"top_blocker_code": "zero-cache-hits", "scanned_rows": 100},
                    "ladder": [{"blocker_code": "zero-cache-hits", "count": 100}],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 12, "rollup_count": 1},
                    "rollups": [
                        {
                            "provider_family": "anthropic",
                            "source_surface": "anthropic_messages",
                            "endpoint": "messages",
                            "requested_model_family": "claude-sonnet",
                            "routed_model_family": "claude-sonnet",
                            "category": "tool-result",
                            "workflow_phase": "tool-execution",
                            "stream": True,
                            "has_tools": True,
                            "text_bucket": "32k_128k_chars",
                            "token_bucket": "8k_32k_tokens",
                            "cache_status": "skipped",
                            "routing_status": "passthrough",
                            "row_count": 12,
                            "cost_est_usd": 0.5,
                            "observed_savings_usd": 0.0,
                            "candidate_work_classes": ["repeated_context", "replayability"],
                            "candidate_families": ["cache_replay", "cache_blocker"],
                            "blocker_codes": ["unsupported-streaming-shape"],
                        }
                    ],
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "status": "ranked",
                        "summary": {
                            "cohort_count": 1,
                            "rows_considered": 12,
                            "replay_ready_cohort_count": 0,
                            "skipped_cohort_count": 1,
                            "top_blocker_code": "invalidation-evidence-missing",
                            "projected_hits": 0,
                        },
                        "cohorts": [
                            {
                                "readiness": "skipped",
                                "reason": "invalidation-evidence-missing",
                                "blockers": ["tools-present", "invalidation-evidence-missing"],
                                "row_count": 12,
                                "projected_hits": 0,
                                "raw_session_id": "raw-session-secret",
                                "cache_key": "cache-secret",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        stats_summary = plan["evidence"]["stats_summary"]
        loop = stats_summary["evidence_to_activation_loop"]
        cache_stage = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache_stage["evidence_source"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(cache_stage["next_action"], "resolve-cache-replayability-blocker")
        self.assertIn("invalidation-evidence-missing", cache_stage["blocker_codes"])
        self.assertNotIn("zero-cache-hits", cache_stage["blocker_codes"])
        rendered = json.dumps(plan)
        self.assertNotIn("raw-session-secret", rendered)
        self.assertNotIn("cache-secret", rendered)

    def test_request_shape_replay_ready_cache_evidence_supersedes_zero_hit_candidate(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 100,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_zero_hit_blocker_ladder": {
                    "schema": "agentflow.cache_zero_hit_blocker_ladder.v1",
                    "summary": {
                        "top_blocker_code": "zero-cache-hits",
                        "scanned_rows": 100,
                        "zero_hit_window": True,
                    },
                    "ladder": [{"blocker_code": "zero-cache-hits", "count": 100}],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 56, "rollup_count": 1},
                    "rollups": [],
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "stage-cache-replay-canary",
                            "top_local_action_family": "cache",
                        },
                        "candidates": [
                            {
                                "provider_surface_bucket": "openai/openai_responses/responses",
                                "provider_family": "openai",
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "cache_status": "miss",
                                "routing_status": "disabled",
                                "next_action": "stage-cache-replay-canary",
                                "local_action_family": "cache",
                                "candidate_work_classes": ["replayability"],
                                "candidate_families": ["cache_replay"],
                                "blocker_codes": [],
                                "row_count": 56,
                                "cost_est_usd": 0.12,
                                "observed_savings_usd": 0.0,
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "status": "ranked",
                        "summary": {
                            "cohort_count": 1,
                            "rows_considered": 56,
                            "replay_ready_cohort_count": 1,
                            "replay_ready_rows": 56,
                            "skipped_cohort_count": 0,
                            "projected_hits": 55,
                            "projected_savings_usd": 0.121981,
                        },
                        "cohorts": [
                            {
                                "readiness": "replay-ready",
                                "reason": "replay-ready-exact-non-tool-shape",
                                "blockers": [],
                                "provider_family": "openai",
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "stream": False,
                                "has_tools": False,
                                "cache_status": "miss",
                                "routing_status": "disabled",
                                "row_count": 56,
                                "projected_hits": 55,
                                "projected_savings_usd": 0.121981,
                                "request_id": "request-secret-should-redact",
                                "session_id": "session-secret-should-redact",
                                "cache_key": "cache-secret-should-redact",
                                "file_path": "/tmp/private-cache-replay.py",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        stats_summary = plan["evidence"]["stats_summary"]
        loop = stats_summary["evidence_to_activation_loop"]
        cache_stage = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache_stage["state"], "replay-ready")
        self.assertEqual(cache_stage["evidence_source"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(cache_stage["next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_stage["projected_hits"], 55)
        self.assertNotIn("zero-cache-hits", cache_stage["blocker_codes"])

        created = plan["backlog_changes"]["create_issues"]
        titles = [item["title"] for item in created]
        self.assertIn("Stage cache replay canary for replay-ready on openai/openai_responses/responses", titles)
        self.assertNotIn("Turn zero-cache-hits cache candidate into local replay evidence", titles)
        cache_issue = next(item for item in created if item["title"].startswith("Stage cache replay canary"))
        self.assertIn("Source metadata: request_shape_cache_replayability_dry_run", cache_issue["body"])
        self.assertIn("projected_hits: 55", cache_issue["body"])
        self.assertIn("Activation mode: activation-candidate", cache_issue["body"])

        candidates = plan["evidence"]["optimization_candidates"]
        cache_candidate = next(row for row in candidates if row["lever"] == "cache")
        self.assertEqual(cache_candidate["blocker"], "replay-ready")
        self.assertEqual(cache_candidate["projected_savings_signal"]["readiness"], "replay-ready")
        self.assertEqual(cache_candidate["projected_savings_signal"]["projected_hits"], 55)

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("request-secret-should-redact", rendered)
        self.assertNotIn("session-secret-should-redact", rendered)
        self.assertNotIn("cache-secret-should-redact", rendered)
        self.assertNotIn("/tmp/private-cache-replay.py", rendered)

    def test_cache_replay_cohort_ranking_prefers_activation_ready_issue(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 10,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "cache_replay_cohort_ranking": {
                    "schema": "agentflow.cache_replay_plateau_cohort_ranking.v1",
                    "summary": {
                        "candidate_rows": 3,
                        "activation_ready_count": 1,
                        "projected_ready_hits": 2,
                    },
                    "cohorts": [
                        {
                            "readiness": "activation-ready",
                            "source_surface": "anthropic_messages",
                            "category": "tool-result",
                            "workflow_phase": "tool-execution",
                            "stream": True,
                            "has_tools": True,
                            "replayability_level": "local-exact-response",
                            "dependency_state": "stable",
                            "provider_adoption_state": "ready",
                            "count": 3,
                            "projected_hits": 2,
                            "projected_saved_cost_usd": 0.04,
                            "cohort_id": "cache-replay-cohort:secretid",
                            "raw_session_id": "raw-session-secret",
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        cache_issue = [item for item in plan["backlog_changes"]["create_issues"] if "cache replay" in item["title"].lower()][0]
        self.assertIn("Stage cache replay canary", cache_issue["title"])
        self.assertIn("activation-candidate", cache_issue["body"])
        self.assertIn("projected_hits: 2", cache_issue["body"])
        rendered = json.dumps(plan)
        self.assertNotIn("raw-session-secret", rendered)
        self.assertNotIn("secretid", rendered)

    def test_openai_canary_lifecycle_feedback_generates_routing_activation_issue(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "openai_canary_impact": {
                    "schema": "agentflow.openai_canary_impact.v1",
                    "status": "matched",
                    "summary": {
                        "candidate_count": 1,
                        "canary_applied_count": 2,
                        "canary_holdout_count": 1,
                    },
                    "candidates": [
                        {
                            "candidate_id": "openai-canary-secret-id",
                            "policy_id": "local-openai-routing-secret",
                            "source_surface": "openai_responses",
                            "original_model": "gpt-5.4",
                            "candidate_target_model": "gpt-5.4-mini",
                            "verdict": "widen",
                            "next_action": "widen_local_openai_canary",
                            "reason_codes": ["target-savings-met"],
                            "warning_codes": [],
                            "sample_count": 3,
                            "cohort_counts": {
                                "canary_applied": 2,
                                "canary_holdout": 1,
                                "safety_stopped": 0,
                            },
                            "applied_vs_holdout_deltas": {
                                "applied_minus_holdout_error_rate": 0.0,
                                "applied_minus_holdout_retry_rate": 0.0,
                                "applied_minus_holdout_fallback_rate": 0.0,
                                "applied_minus_holdout_latency_avg_ms": -120,
                            },
                            "observed_savings_usd": 0.02,
                            "projected_savings_usd": 0.03,
                            "stale_evidence": {"stale": False, "age_hours": 2.0},
                            "privacy": {"metadata_only": True, "aggregate_only": True},
                        }
                    ],
                    "activation_lifecycle_feedback": {
                        "queue_rows": 1,
                        "state_breakdown": [{"value": "healthy_canary", "count": 1}],
                        "cohort_breakdown": [
                            {"value": "canary_applied", "count": 2},
                            {"value": "canary_holdout", "count": 1},
                        ],
                        "payload_json_included": False,
                        "privacy": {
                            "metadata_only": True,
                            "aggregate_only": False,
                            "raw_prompts_included": False,
                            "request_ids_included": False,
                            "raw_session_ids_included": False,
                        },
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        created = plan["backlog_changes"]["create_issues"]
        routing_issues = [item for item in created if "OpenAI routing canary" in item["title"]]
        self.assertEqual(len(routing_issues), 1)
        issue = routing_issues[0]
        self.assertIn("status:ready", issue["labels"])
        self.assertIn("routing", issue["labels"])
        self.assertIn("Widen OpenAI routing canary", issue["title"])
        self.assertIn("Activation mode: activation-candidate", issue["body"])
        self.assertIn("Next action: widen_local_openai_canary", issue["body"])
        self.assertIn("Savings per 1000 calls estimate: 10.0", issue["body"])
        self.assertIn("healthy_canary", issue["body"])
        self.assertIn("applied/holdout coverage", issue["body"])
        rendered = json.dumps(plan)
        self.assertNotIn("openai-canary-secret-id", rendered)
        self.assertNotIn("local-openai-routing-secret", rendered)

    def test_openai_canary_cohort_lifecycle_metadata_clears_aggregate_only_gate(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "openai_canary_impact": {
                    "schema": "agentflow.openai_canary_impact.v1",
                    "status": "matched",
                    "summary": {
                        "candidate_count": 1,
                        "canary_applied_count": 2,
                        "canary_holdout_count": 1,
                    },
                    "candidates": [
                        {
                            "candidate_id": "raw-openai-canary-candidate-secret",
                            "source_surface": "openai_responses",
                            "original_model": "gpt-5.4",
                            "candidate_target_model": "gpt-5.4-mini",
                            "verdict": "widen",
                            "next_action": "widen_local_openai_canary",
                            "aggregate_only_feedback": True,
                            "reason_codes": ["target-savings-met"],
                            "warning_codes": [],
                            "sample_count": 3,
                            "cohort_counts": {
                                "canary_applied": 2,
                                "canary_holdout": 1,
                                "safety_stopped": 0,
                            },
                            "applied_vs_holdout_deltas": {
                                "applied_minus_holdout_error_rate": 0.0,
                                "applied_minus_holdout_retry_rate": 0.0,
                                "applied_minus_holdout_fallback_rate": 0.0,
                            },
                            "observed_savings_usd": 0.02,
                            "projected_savings_usd": 0.03,
                            "stale_evidence": {"stale": False, "age_hours": 1.0},
                        }
                    ],
                    "activation_lifecycle_feedback": {
                        "queue_rows": 1,
                        "state_breakdown": [{"value": "healthy_canary", "count": 1}],
                        "cohort_breakdown": [
                            {"value": "canary_applied", "count": 2},
                            {"value": "canary_holdout", "count": 1},
                        ],
                        "cohort_lifecycle_metadata": [
                            {
                                "policy_ref": "policy:public-cohort-ref",
                                "policy_id": "raw-policy-secret-should-redact",
                                "cohort_label": "canary_applied",
                                "action_family": "routing",
                                "event_count": 2,
                                "applied_count": 2,
                                "holdout_count": 0,
                                "fallback_count": 0,
                                "error_rate": 0.0,
                                "savings_estimate_usd": 0.02,
                            },
                            {
                                "policy_ref": "policy:public-cohort-ref",
                                "policy_id": "raw-policy-secret-should-redact",
                                "cohort_label": "canary_holdout",
                                "action_family": "routing",
                                "event_count": 1,
                                "applied_count": 0,
                                "holdout_count": 1,
                                "fallback_count": 0,
                                "error_rate": 0.0,
                                "savings_estimate_usd": 0.0,
                            },
                        ],
                        "payload_json_included": False,
                        "privacy": {
                            "metadata_only": True,
                            "aggregate_only": True,
                            "raw_prompts_included": False,
                            "request_ids_included": False,
                            "raw_session_ids_included": False,
                        },
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        routing_issues = [
            item
            for item in plan["backlog_changes"]["create_issues"]
            if "OpenAI routing canary" in item["title"]
        ]
        self.assertEqual(len(routing_issues), 1)
        issue = routing_issues[0]
        self.assertIn("status:ready", issue["labels"])
        self.assertIn("Activation mode: activation-candidate", issue["body"])
        self.assertIn("Cohort lifecycle metadata", issue["body"])
        self.assertNotIn("aggregate-only-feedback", issue["body"])

        routing_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "routing")
        signal = routing_candidate["projected_savings_signal"]
        self.assertEqual(signal["omission_reason"], "none")
        self.assertEqual(signal["cohort_lifecycle_metadata"][0]["applied_count"], 2)
        self.assertEqual(signal["cohort_lifecycle_metadata"][1]["holdout_count"], 1)
        rendered = json.dumps(plan)
        self.assertNotIn("raw-openai-canary-candidate-secret", rendered)
        self.assertNotIn("raw-policy-secret-should-redact", rendered)

    def test_openai_canary_missing_lifecycle_feedback_generates_blocked_update(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "openai_canary_impact": {
                    "schema": "agentflow.openai_canary_impact.v1",
                    "status": "matched",
                    "summary": {"candidate_count": 1},
                    "candidates": [
                        {
                            "candidate_id": "blocked-openai-canary-secret",
                            "source_surface": "openai_responses",
                            "original_model": "gpt-5.4",
                            "candidate_target_model": "gpt-5.4-mini",
                            "verdict": "widen",
                            "next_action": "widen_local_openai_canary",
                            "reason_codes": ["target-savings-met"],
                            "sample_count": 3,
                            "cohort_counts": {
                                "canary_applied": 2,
                                "canary_holdout": 1,
                                "safety_stopped": 0,
                            },
                            "observed_savings_usd": 0.02,
                            "projected_savings_usd": 0.03,
                            "stale_evidence": {"stale": False, "age_hours": 2.0},
                        }
                    ],
                    "activation_lifecycle_feedback": {
                        "queue_rows": 0,
                        "state_breakdown": [],
                        "payload_json_included": False,
                        "privacy": {"metadata_only": True, "aggregate_only": False},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        created = plan["backlog_changes"]["create_issues"]
        blocked = [item for item in created if item["title"].startswith("Blocked: Resolve OpenAI routing canary evidence")]
        self.assertEqual(len(blocked), 1)
        issue = blocked[0]
        self.assertIn("status:blocked", issue["labels"])
        self.assertNotIn("status:ready", issue["labels"])
        self.assertIn("Activation mode: blocked", issue["body"])
        self.assertIn("Omission reason: missing-lifecycle-feedback", issue["body"])
        self.assertIn("Activation lifecycle feedback reports a healthy canary state", issue["body"])
        rendered = json.dumps(plan)
        self.assertNotIn("blocked-openai-canary-secret", rendered)

    def test_enough_ready_issues_is_noop(self):
        plan = build_research_plan(
            issues=[
                issue(1, "Ready one", ["status:ready", "priority:p1"]),
                issue(2, "Ready two", ["status:ready", "priority:p2"]),
                issue(3, "Ready three", ["status:ready", "priority:p3"]),
                issue(4, "Blocked", ["status:blocked", "priority:p1"], updated="2026-05-01T08:00:00Z"),
            ],
            threshold=3,
            now=NOW,
        )

        self.assertFalse(plan["research_trigger"]["should_run"])
        self.assertEqual(plan["backlog_changes"]["create_issues"], [])
        self.assertEqual(plan["backlog_changes"]["comment_issues"], [])
        self.assertEqual(plan["evidence"]["optimization_candidates"], [])
        self.assertIn("should not run", plan["run_log_summary"])

    def test_stale_blocked_issues_get_current_evidence_comment(self):
        plan = build_research_plan(
            issues=[
                issue(
                    220,
                    "Milestone: Local workflow phase memory",
                    ["status:blocked", "priority:p1", "core-feature"],
                    updated="2026-05-20T00:00:00Z",
                )
            ],
            stats={"calls": 20, "today_errors": 2},
            threshold=2,
            stale_days=14,
            now=NOW,
        )

        comments = plan["backlog_changes"]["comment_issues"]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]["number"], 220)
        self.assertIn("Blocked issue has been stale", comments[0]["body"])
        self.assertIn("Acceptance Criteria", comments[0]["body"])

    def test_repeated_skip_diagnostics_become_a_targeted_proposal(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "rollout skipped skip_reason=missing-dependency-evidence request_id=req-secret-12345",
                        "candidate omitted reason=missing-dependency-evidence session_id=session-secret-67890",
                        "quality gate blocked blocker=need-more-samples",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        self.assertEqual(diagnostics[0]["reason"], "missing-dependency-evidence")
        self.assertGreaterEqual(diagnostics[0]["count"], 2)
        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertTrue(any("missing dependency evidence" in title for title in created_titles))
        rendered = json.dumps(plan)
        self.assertNotIn("req-secret-12345", rendered)
        self.assertNotIn("session-secret-67890", rendered)

    def test_repeated_pass_diagnostics_are_ignored_for_blocker_issue_generation(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "Test verdict: PASS.",
                        "Test verdict: PASS.",
                        "cache replay omitted reason=aggregate-only candidate_id=cache-candidate-secret",
                        "routing candidate skipped blocker=aggregate-only request_id=req-secret-12345",
                        "managed recommendation omitted reason=provider-capability-mismatch session_id=session-secret-67890",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                stats={
                    "calls": 50,
                    "routing": [
                        {
                            "provider": "openai",
                            "requested_model": "gpt-5.4",
                            "routed_model": "gpt-5.4",
                            "c": 12,
                        }
                    ],
                },
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        self.assertEqual(diagnostics[0]["reason"], "pass")
        self.assertGreaterEqual(diagnostics[0]["count"], 2)

        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertTrue(any("missing lifecycle feedback diagnostics" in title for title in created_titles))
        self.assertFalse(any("aggregate only diagnostics" in title for title in created_titles))
        self.assertFalse(any("pass diagnostics" in title for title in created_titles))

        repeated_issue = [item for item in plan["backlog_changes"]["create_issues"] if "missing lifecycle feedback diagnostics" in item["title"].lower()][0]
        self.assertIn("Source lever:", repeated_issue["body"])
        self.assertIn("Lifecycle action family: routing", repeated_issue["body"])
        self.assertIn("Lifecycle blocker code: missing-applied-coverage", repeated_issue["body"])
        self.assertIn("Lifecycle sample count bucket: 10_99", repeated_issue["body"])
        self.assertIn("Expected unblock path:", repeated_issue["body"])
        self.assertIn("metadata-only", repeated_issue["body"])
        self.assertIn("no raw prompts", repeated_issue["body"])

        candidates = plan["evidence"]["optimization_candidates"]
        diagnostic_candidates = [item for item in candidates if item["lever"] == "activation-feedback"]
        self.assertTrue(diagnostic_candidates)
        self.assertEqual(diagnostic_candidates[0]["blocker"], "repeated-missing-lifecycle-feedback")
        lifecycle_context = diagnostic_candidates[0]["projected_savings_signal"]["lifecycle_context"]
        self.assertEqual(lifecycle_context["action_family"], "routing")
        self.assertEqual(lifecycle_context["blocker_code"], "missing-applied-coverage")
        self.assertNotEqual(diagnostic_candidates[0]["projected_savings_signal"]["diagnostic_reason"], "pass")

        rendered = json.dumps(plan)
        self.assertNotIn("cache-candidate-secret", rendered)
        self.assertNotIn("req-secret-12345", rendered)
        self.assertNotIn("session-secret-67890", rendered)

    def test_healthy_lifecycle_metadata_suppresses_repeated_aggregate_only_diagnostic(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "activation feedback omitted reason=aggregate-only candidate_id=secret-candidate-one",
                        "routing candidate skipped blocker=aggregate-only request_id=req-secret-12345",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                stats={
                    "calls": 50,
                    "openai_canary_impact": {
                        "schema": "agentflow.openai_canary_impact.v1",
                        "status": "matched",
                        "candidates": [
                            {
                                "candidate_id": "raw-openai-canary-candidate-secret",
                                "source_surface": "openai_responses",
                                "original_model": "gpt-5.4",
                                "candidate_target_model": "gpt-5.4-mini",
                                "verdict": "widen",
                                "next_action": "widen_local_openai_canary",
                                "aggregate_only_feedback": True,
                                "sample_count": 3,
                                "cohort_counts": {
                                    "canary_applied": 2,
                                    "canary_holdout": 1,
                                    "safety_stopped": 0,
                                },
                                "observed_savings_usd": 0.02,
                                "projected_savings_usd": 0.03,
                            }
                        ],
                        "activation_lifecycle_feedback": {
                            "schema": "agentflow.activation_staged_lifecycle_feedback_summary.v1",
                            "queue_rows": 1,
                            "family_event_count": 3,
                            "state_breakdown": [{"value": "healthy_canary", "count": 1}],
                            "cohort_breakdown": [
                                {"value": "canary_applied", "count": 2},
                                {"value": "canary_holdout", "count": 1},
                            ],
                            "cohort_lifecycle_metadata": [
                                {
                                    "policy_ref": "policy:public-cohort-ref",
                                    "policy_id": "raw-policy-secret-should-redact",
                                    "cohort_label": "canary_applied",
                                    "action_family": "routing",
                                    "event_count": 2,
                                    "applied_count": 2,
                                    "holdout_count": 0,
                                    "fallback_count": 0,
                                },
                                {
                                    "policy_ref": "policy:public-cohort-ref",
                                    "policy_id": "raw-policy-secret-should-redact",
                                    "cohort_label": "canary_holdout",
                                    "action_family": "routing",
                                    "event_count": 1,
                                    "applied_count": 0,
                                    "holdout_count": 1,
                                    "fallback_count": 0,
                                },
                            ],
                            "payload_json_included": False,
                            "privacy": {
                                "metadata_only": True,
                                "aggregate_only": True,
                                "raw_prompts_included": False,
                                "request_ids_included": False,
                                "raw_session_ids_included": False,
                            },
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                },
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostic_reasons = [item["reason"] for item in plan["evidence"]["repeated_diagnostics"]]
        self.assertNotIn("aggregate-only", diagnostic_reasons)
        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(any("aggregate only diagnostics" in title for title in created_titles))
        self.assertFalse(any("missing lifecycle feedback diagnostics" in title for title in created_titles))
        rendered = json.dumps(plan)
        self.assertNotIn("secret-candidate-one", rendered)
        self.assertNotIn("req-secret-12345", rendered)
        self.assertNotIn("raw-openai-canary-candidate-secret", rendered)
        self.assertNotIn("raw-policy-secret-should-redact", rendered)

    def test_unclassified_success_verdict_line_is_ignored(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "VERDICT: PASS quality-gate-passed skipped remaining items",
                        "VERDICT: PASS quality-gate-passed skipped remaining items",
                        "eval-pass threshold-met skipped 3 observations",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        reasons = [d["reason"] for d in diagnostics]
        self.assertNotIn("unclassified-skip-or-blocker", reasons)
        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(any("unclassified" in t for t in created_titles))

    def test_unclassified_managed_omission_line_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "action was blocked by server-content-processing limit session_id=sec-secret-1",
                        "action was blocked by server-content-processing limit session_id=sec-secret-2",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        reasons = [d["reason"] for d in diagnostics]
        self.assertNotIn("unclassified-skip-or-blocker", reasons)
        self.assertIn("unsupported-provider-action", reasons)
        rendered = json.dumps(plan)
        self.assertNotIn("sec-secret-1", rendered)
        self.assertNotIn("sec-secret-2", rendered)

    def test_unclassified_missing_measurement_line_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "activation skipped: missing-crunch-measurement no positive projection",
                        "activation skipped: missing-crunch-measurement no positive projection",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        reasons = [d["reason"] for d in diagnostics]
        self.assertNotIn("unclassified-skip-or-blocker", reasons)
        self.assertIn("missing-dependency-evidence", reasons)

    def test_unclassified_true_unknown_blocker_gets_needs_human_review(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "activation skipped due to xyz-unknown-blocker-type",
                        "activation skipped due to xyz-unknown-blocker-type",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        actionable = [d for d in diagnostics if d.get("reason") == "unclassified-skip-or-blocker"]
        self.assertTrue(actionable)
        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(any("unclassified skip or blocker" in t for t in created_titles))

    def test_privacy_redacts_raw_fields_paths_and_ids(self):
        plan = build_research_plan(
            issues=[
                {
                    "repo": "lutzkuen/agentflow",
                    "number": 9,
                    "title": "Raw prompt must not leak",
                    "state": "OPEN",
                    "author": {"login": "lutzkuen"},
                    "labels": [{"name": "status:blocked"}],
                    "updatedAt": "2026-05-01T00:00:00Z",
                    "request_json": {"messages": [{"content": "private prompt text"}]},
                    "session_id": "session-raw-secret",
                }
            ],
            stats={
                "calls": 1,
                "request_json": {"messages": [{"content": "private stats prompt"}]},
                "routing": [{"requested_model": "gpt-5", "path": "/home/lutz/private/project/file.py"}],
            },
            log_sources=[
                "skip_reason=privacy-blocked request_id=req-raw-secret /home/lutz/private/project/file.py sk-testsecret123456"
            ],
            threshold=2,
            now=NOW,
        )

        rendered = json.dumps(plan)
        self.assertNotIn("private prompt text", rendered)
        self.assertNotIn("private stats prompt", rendered)
        self.assertNotIn("/home/lutz/private/project/file.py", rendered)
        self.assertNotIn("req-raw-secret", rendered)
        self.assertNotIn("session-raw-secret", rendered)
        self.assertNotIn("sk-testsecret123456", rendered)
        self.assertIn("[REDACTED", rendered)
        for proposal in plan["backlog_changes"]["create_issues"]:
            body = proposal["body"]
            self.assertIn("## Labels", body)
            self.assertNotIn("private prompt text", body)
            self.assertNotIn("private stats prompt", body)
            self.assertNotIn("/home/lutz/private/project/file.py", body)
            self.assertNotIn("req-raw-secret", body)
            self.assertNotIn("session-raw-secret", body)
        self.assertFalse(plan["privacy"]["raw_prompts_included"])
        self.assertFalse(plan["privacy"]["absolute_paths_included"])


class OrchestratorResearchCliTests(unittest.TestCase):
    def test_cli_reads_json_files_and_emits_plan(self):
        with TemporaryDirectory() as tmp:
            issues_path = Path(tmp) / "issues.json"
            stats_path = Path(tmp) / "stats.json"
            issues_path.write_text(json.dumps([issue(1, "Blocked", ["status:blocked"], updated="2026-05-01T00:00:00Z")]), encoding="utf-8")
            stats_path.write_text(json.dumps({"calls": 5, "cache_hit_rate": 0.0}), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            code = cli.orchestrator_research_cli(
                ["--issues-json", str(issues_path), "--stats-json", str(stats_path), "--threshold", "2"],
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "agentflow.orchestrator_research_plan.v1")
        self.assertTrue(payload["research_trigger"]["should_run"])

    def test_cli_builds_request_shape_rollups_from_stats_db(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = SQLiteStore(db_path)
            try:
                for cost in (0.02, 0.03):
                    store.log_call(
                        id=str(uuid.uuid4()),
                        created_at=utc_now(),
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=1,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=125,
                        input_tokens_est=12_000,
                        output_tokens_est=100,
                        actual_input_tokens=12_000,
                        actual_output_tokens=100,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
                        routing_json=stable_json(
                            {
                                "category": "tool-result",
                                "workflow_phase": "tool-execution",
                                "text_chars": 48_000,
                                "has_tools": True,
                                "reason": "keep requested model",
                            }
                        ),
                        cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
                        error=None,
                        request_json=stable_json({"prompt": "raw prompt must not leak"}),
                        response_json=stable_json({"content": "raw response must not leak"}),
                        session_id="raw-session-id-must-not-leak",
                        category="tool-result",
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        retry_count=0,
                        thinking_output_tokens=0,
                        provider="anthropic",
                        source_surface="anthropic_messages",
                        endpoint="messages",
                        requested_model_family="claude-sonnet",
                        routed_model_family="claude-sonnet",
                    )
            finally:
                store.conn.close()

            issues_path = Path(tmp) / "issues.json"
            stats_path = Path(tmp) / "stats.json"
            issues_path.write_text(json.dumps([]), encoding="utf-8")
            stats_path.write_text(json.dumps({"calls": 2, "cache_hit_rate": 0.0, "db": db_path}), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            code = cli.orchestrator_research_cli(
                ["--issues-json", str(issues_path), "--stats-json", str(stats_path), "--threshold", "3"],
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        signal = payload["evidence"]["stats_summary"]["request_shape_rollup_candidates"]
        self.assertEqual(signal["schema"], "agentflow.request_shape_rollup_candidate_signal.v1")
        self.assertEqual(signal["status"], "candidates-ranked")
        self.assertEqual(signal["summary"]["rows_considered"], 2)
        self.assertGreaterEqual(signal["summary"]["ranked_candidate_count"], 1)
        self.assertEqual(signal["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(signal["top_candidate"]["row_count"], 2)
        self.assertIn("request_shape_crunch_opportunity", payload["evidence"]["stats_summary"]["crunch_savings_signal"]["top_report"]["report_key"])
        rendered = json.dumps(payload)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-session-id-must-not-leak", rendered)


class RepeatedSafetyStopDiagnosticTests(unittest.TestCase):
    def test_unclassified_safety_stop_example_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "activation skipped: safety-stopped by canary gate session_id=sec-secret-a",
                        "activation skipped: safety-stopped by canary gate session_id=sec-secret-b",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        reasons = [d["reason"] for d in diagnostics]
        self.assertNotIn("unclassified-skip-or-blocker", reasons)
        self.assertIn("safety-stop", reasons)
        safety_burndown = plan["evidence"]["activation_safety_stop_burndown"]
        self.assertEqual(safety_burndown["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(safety_burndown["summary"]["top_blocker_code"], "safety-stop")
        self.assertEqual(
            safety_burndown["summary"]["top_keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertEqual(safety_burndown["summary"]["top_next_state"], "keep-blocked")
        self.assertEqual(
            safety_burndown["groups"][0]["next_state_reason"],
            "safety-stop-requires-safer-threshold-or-rollback-proof",
        )
        self.assertIn("activation_safety_stop_burndown", plan["evidence"]["inspected_sources"])
        rendered = json.dumps(plan)
        self.assertNotIn("sec-secret-a", rendered)
        self.assertNotIn("sec-secret-b", rendered)

    def test_successful_field_list_does_not_become_safety_stop_diagnostic(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "- Captured aggregate cache/crunch/routing outcome fields: action family, evidence source, bucket, "
                        "policy/rule reference, applied/holdout/safety-stop/error/retry/fallback counts.",
                        "- Captured aggregate cache/crunch/routing outcome fields: action family, evidence source, bucket, "
                        "policy/rule reference, applied/holdout/safety-stop/error/retry/fallback counts.",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        self.assertNotIn("safety-stop", [d["reason"] for d in diagnostics])
        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(any("safety stop diagnostics" in title for title in created_titles))

    def test_evidence_to_activation_burndown_uses_safety_stop_keep_blocked_reason(self):
        plan = build_research_plan(
            issues=[],
            log_sources=[
                "routing blocker=safety-stop request_id=req-secret-stop-a",
                "routing blocker=safety-stop request_id=req-secret-stop-b",
            ],
            threshold=1,
            now=NOW,
        )

        report = build_evidence_to_activation_burndown(plan, now=NOW)

        safety_rows = [
            row
            for row in report["blockers"]
            if row.get("evidence_source") == "agentflow.activation_safety_stop_burndown.v1"
        ]
        self.assertTrue(safety_rows)
        self.assertEqual(
            safety_rows[0]["blocker_codes"],
            ["activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof"],
        )
        self.assertEqual(safety_rows[0]["state"], "keep-blocked")
        self.assertEqual(safety_rows[0]["next_state"], "keep-blocked")
        self.assertEqual(
            safety_rows[0]["next_state_reason"],
            "safety-stop-requires-safer-threshold-or-rollback-proof",
        )
        self.assertEqual(
            safety_rows[0]["keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertIn("rollback_proof", safety_rows[0]["needed_resolution"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("req-secret-stop-a", rendered)
        self.assertNotIn("req-secret-stop-b", rendered)

    def test_repeated_safety_stop_creates_issue_when_no_existing_match(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing blocker=safety-stop request_id=req-secret-new",
                        "routing blocker=safety-stop request_id=req-secret-new2",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[],
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        repeated_diag_proposals = [
            item for item in plan["backlog_changes"]["create_issues"]
            if "repeated" in item["title"].lower() and "safety stop" in item["title"].lower()
        ]
        self.assertTrue(repeated_diag_proposals, "expected a repeated-diagnostic create proposal for safety-stop")
        comment_issues = [
            c for c in plan["backlog_changes"]["comment_issues"]
            if "safety" in (c.get("body") or "").lower() and c.get("action") == "comment"
        ]
        self.assertFalse(any(c.get("number") for c in comment_issues), "should not comment when no existing issue")

        body = repeated_diag_proposals[0]["body"]
        self.assertIn("agentflow.repeated-diagnostic.safety-stop.v1", body)
        self.assertIn("Evidence count:", body)
        self.assertIn("Proposed owner:", body)
        self.assertIn("Action: create", body)

    def test_repeated_safety_stop_comments_on_existing_open_issue_not_duplicate_create(self):
        existing_issue = issue(
            444,
            "Turn repeated safety stop diagnostics into an actionable optimization issue",
            ["status:ready", "priority:p2", "backlog", "core-feature", "correctness"],
        )
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing blocker=safety-stop request_id=req-secret-dup1",
                        "routing blocker=safety-stop request_id=req-secret-dup2",
                    ]
                ),
                encoding="utf-8",
            )

            plan = build_research_plan(
                issues=[existing_issue],
                log_sources=[log_path],
                threshold=2,
                now=NOW,
            )

        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(
            any("repeated" in t and "safety stop" in t for t in created_titles),
            "should not create a new issue when fingerprint matches open issue #444",
        )

        comment_issues = plan["backlog_changes"]["comment_issues"]
        safety_stop_comments = [c for c in comment_issues if c.get("number") == 444]
        self.assertTrue(safety_stop_comments, "expected a comment action on issue #444")
        comment_body = safety_stop_comments[0]["body"]
        self.assertIn("agentflow.repeated-diagnostic.safety-stop.v1", comment_body)
        self.assertIn("Action: update", comment_body)
        self.assertIn("Duplicate of open issue: #444", comment_body)
        rendered = json.dumps(plan)
        self.assertNotIn("req-secret-dup1", rendered)
        self.assertNotIn("req-secret-dup2", rendered)

    def test_diagnostic_fingerprint_is_stable(self):
        from agentflow_proxy.orchestrator_research import _diagnostic_fingerprint
        self.assertEqual(
            _diagnostic_fingerprint("safety-stop"),
            _diagnostic_fingerprint("safety-stop"),
        )
        self.assertEqual(
            _diagnostic_fingerprint("safety-stop"),
            "agentflow.repeated-diagnostic.safety-stop.v1",
        )
        self.assertNotEqual(
            _diagnostic_fingerprint("safety-stop"),
            _diagnostic_fingerprint("missing-dependency-evidence"),
        )

    def test_repeated_diagnostic_proposal_includes_required_fields(self):
        plan = build_research_plan(
            issues=[],
            log_sources=[
                "routing blocker=safety-stop x",
                "routing blocker=safety-stop y",
            ],
            threshold=1,
            now=NOW,
        )
        created = plan["backlog_changes"]["create_issues"]
        repeated_proposals = [
            p for p in created
            if "repeated" in p.get("title", "").lower() and "safety stop" in p.get("title", "").lower()
        ]
        self.assertTrue(repeated_proposals, "expected a repeated-diagnostic proposal with safety-stop")
        body = repeated_proposals[0]["body"]
        self.assertIn("Evidence count:", body)
        self.assertIn("Example excerpt:", body)
        self.assertIn("Proposed owner:", body)
        self.assertIn("Fingerprint:", body)
        self.assertIn("Action:", body)


if __name__ == "__main__":
    unittest.main()
