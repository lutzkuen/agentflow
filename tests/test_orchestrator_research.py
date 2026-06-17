import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import uuid

from agentflow_proxy import cli
from agentflow_proxy.orchestrator_research import (
    build_evidence_to_activation_burndown,
    build_evidence_to_activation_next_action_ledger,
    build_local_activation_next_action_queue,
    build_research_plan,
    _dedupe_create_issue_proposals_with_metadata,
)
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


NOW = datetime(2026, 6, 11, 8, 40, tzinfo=timezone.utc)


def cache_replayability_stats():
    return {
        "calls": 100,
        "cache_hits": 0,
        "cache_hit_rate": 0.0,
        "request_shape_rollup_report": {
            "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
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
                    }
                ],
                "privacy": {"metadata_only": True, "aggregate_only": True},
            },
            "privacy": {"metadata_only": True, "aggregate_only": True},
        },
    }


def issue(
    number,
    title,
    labels,
    *,
    repo="lutzkuen/agentflow",
    author="lutzkuen",
    state="OPEN",
    updated="2026-06-11T08:00:00Z",
    closed=None,
    body=None,
):
    payload = {
        "repo": repo,
        "number": number,
        "title": title,
        "state": state,
        "url": f"https://github.com/{repo}/issues/{number}",
        "author": {"login": author},
        "labels": [{"name": name} for name in labels],
        "updatedAt": updated,
    }
    if closed is not None:
        payload["closedAt"] = closed
    if body is not None:
        payload["body"] = body
    return payload


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
        self.assertIn("recommended_next_issue", milestone["summary"])
        self.assertNotEqual(milestone["summary"]["recommended_next_issue"]["lever"], "milestone-planning")
        self.assertEqual(milestone["summary"]["recommended_next_issue"]["implementation_rank"], 1)
        self.assertEqual(
            [item["implementation_rank"] for item in milestone["implementation_order"]],
            list(range(1, len(created) + 1)),
        )
        self.assertIn("priority", milestone["issues"][0])
        self.assertTrue(milestone["privacy"]["metadata_only"])
        self.assertTrue(milestone["privacy"]["aggregate_only"])
        self.assertFalse(milestone["privacy"]["raw_prompts_included"])
        self.assertFalse(milestone["privacy"]["provider_bodies_included"])
        self.assertFalse(milestone["privacy"]["request_ids_included"])
        self.assertFalse(milestone["privacy"]["session_ids_included"])

    def test_issue_533_low_backlog_milestone_is_targeted_ranked_and_private(self):
        plan = build_research_plan(
            issues=[
                issue(
                    406,
                    "Generate next backlog milestone from local telemetry evidence",
                    ["backlog", "status:ready", "priority:p1", "core-feature", "correctness"],
                    state="CLOSED",
                    closed="2026-06-10T08:00:00Z",
                ),
                issue(
                    451,
                    "Generate next backlog milestone from local telemetry evidence",
                    ["backlog", "status:ready", "priority:p1", "core-feature", "correctness"],
                    state="CLOSED",
                    closed="2026-06-10T09:00:00Z",
                ),
            ],
            stats={
                "calls": 3130,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "request_id": "req-issue-533-secret",
                "session_id": "session-issue-533-secret",
                "raw_prompt": "raw prompt must not leak",
                "file_path": "/home/lutz/private/issue_533_secret.py",
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 1000, "rollup_count": 32},
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "rows_considered": 1000,
                            "ranked_candidate_count": 10,
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "category": "tool-result",
                                "workflow_phase": "thinking",
                                "stream": True,
                                "has_tools": True,
                                "cache_status": "skipped",
                                "routing_status": "passthrough",
                                "row_count": 388,
                                "sample_count": 388,
                                "cost_est_usd": 20.565063,
                                "observed_savings_usd": 137.523788,
                                "projected_saved_tokens": 843452,
                                "projected_savings_usd": 2.530359,
                                "candidate_work_classes": [
                                    "crunch",
                                    "repeated_context",
                                    "replayability",
                                    "routing",
                                    "routing_evidence",
                                ],
                                "candidate_families": [
                                    "cache_blocker",
                                    "cache_replay",
                                    "routing_candidate",
                                    "routing_evidence",
                                ],
                                "blocker_codes": [
                                    "thinking-routing-guard",
                                    "tool-call-cache-disabled",
                                    "unsupported-streaming-shape",
                                ],
                                "readiness_state": "activation-ready",
                                "local_action_family": "crunch",
                                "next_action": "stage-repeated-context-crunch-canary",
                                "candidate_id": "raw-request-shape-candidate-secret",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "crunch_savings_signal": {
                    "schema": "agentflow.crunch_savings_signal.v1",
                    "status": "projected-savings-ranked",
                    "top_report": {
                        "matched_count": 41,
                        "projected_saved_usd": 0.006894,
                        "next_action": "widen",
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "pass_through_routing_report": {
                    "schema": "agentflow.pass_through_routing_activation_candidates.v1",
                    "summary": {
                        "top_actionability": "actionable",
                        "top_requested_model": "claude-sonnet-4-6",
                        "top_candidate_target_model": "claude-haiku-4-5-20251001",
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        created = plan["backlog_changes"]["create_issues"]
        titles = [item["title"] for item in created]
        self.assertIn("Rank next savings milestone from local telemetry evidence gaps", titles)
        self.assertNotIn("Generate next backlog milestone from local telemetry evidence", titles)

        milestone_issue = next(
            item for item in created
            if item["title"] == "Rank next savings milestone from local telemetry evidence gaps"
        )
        self.assertIn("## Evidence", milestone_issue["body"])
        self.assertIn("## Implementation Approach", milestone_issue["body"])
        self.assertIn("## Acceptance Criteria", milestone_issue["body"])
        self.assertIn("## Labels", milestone_issue["body"])
        self.assertIn("## Sequencing Notes", milestone_issue["body"])
        self.assertIn("stage-repeated-context-crunch-canary", milestone_issue["body"])

        milestone = plan["evidence"]["next_backlog_milestone"]
        self.assertEqual(milestone["schema"], "agentflow.next_backlog_milestone.v1")
        self.assertEqual(milestone["status"], "ready")
        self.assertEqual(milestone["summary"]["proposal_count"], len(created))
        self.assertEqual(
            milestone["summary"]["top_next_action"],
            "stage-repeated-context-crunch-canary",
        )
        self.assertEqual(milestone["summary"]["top_issue"]["rank"], 1)
        self.assertEqual(milestone["summary"]["top_issue"]["title"], created[0]["title"])
        self.assertEqual(milestone["summary"]["recommended_next_issue"]["implementation_rank"], 1)
        self.assertNotEqual(milestone["summary"]["recommended_next_issue"]["lever"], "milestone-planning")
        self.assertEqual([item["rank"] for item in milestone["issues"]], list(range(1, len(created) + 1)))
        self.assertEqual(
            [item["implementation_rank"] for item in milestone["implementation_order"]],
            list(range(1, len(created) + 1)),
        )

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("req-issue-533-secret", rendered)
        self.assertNotIn("session-issue-533-secret", rendered)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("/home/lutz/private/issue_533_secret.py", rendered)
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

    def test_post_promotion_priority_deltas_generate_successor_instead_of_stale_stage_titles(self):
        stale_titles = [
            "Stage cache replay canary for replay-ready on openai/openai_responses/responses",
            "Rank crunch savings follow-up for crunch-projected-savings-ranked",
            "Stage request-shape repeated-context crunch canary",
            "Rank managed recommendation omission reasons for local policy handoff",
            "Stage routing evidence for claude-sonnet-4-6 to claude-haiku-4-5-20251001",
            "Turn replay-ready cache candidate into local replay evidence",
            "Resolve repeated-safety-stop activation feedback blocker",
        ]
        plan = build_research_plan(
            issues=[
                issue(
                    470 + index,
                    title,
                    ["backlog", "status:ready", "priority:p1", "core-feature"],
                    state="CLOSED",
                )
                for index, title in enumerate(stale_titles)
            ],
            stats={
                "calls": 2876,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "anthropic",
                        "source_surface": "unknown",
                        "endpoint": "unknown",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "c": 1197,
                    }
                ],
                "post_promotion_priority_review": {
                    "schema": "agentflow.post_promotion_priority_delta_review.v1",
                    "status": "ranked",
                    "summary": {
                        "review_candidate_count": 3,
                        "recommended_count": 2,
                        "noop_count": 1,
                        "top_next_action": "widen-local-policy",
                        "widen_count": 1,
                        "rollback_count": 1,
                        "keep_blocked_count": 1,
                    },
                    "groups": [
                        {
                            "rank": 1,
                            "action_family": "routing",
                            "candidate_count": 1,
                            "top_next_action": "widen-local-policy",
                            "savings_delta_usd": 4.5,
                        }
                    ],
                    "candidates": [
                        {
                            "rank": 1,
                            "status": "recommended",
                            "next_action": "widen-local-policy",
                            "action_family": "routing",
                            "recommendation_type": "widen-routing-canary",
                            "policy_section": "routing",
                            "savings_delta_usd": 4.5,
                            "confidence": 0.91,
                            "prompt": "raw post promotion prompt must not leak",
                            "request_id": "post-promotion-request-secret",
                            "session_id": "post-promotion-session-secret",
                            "cache_key": "post-promotion-cache-secret",
                            "file_path": "/home/lutz/private/post_promotion_secret.py",
                        },
                        {
                            "rank": 2,
                            "status": "recommended",
                            "next_action": "rollback-local-policy",
                            "action_family": "cache",
                            "recommendation_type": "rollback-cache-canary",
                            "policy_section": "cache",
                            "savings_delta_usd": -1.0,
                        },
                        {
                            "rank": 3,
                            "status": "noop",
                            "next_action": "keep-blocked",
                            "action_family": "crunch",
                            "recommendation_type": "noop",
                            "policy_section": "crunch",
                            "no_op_reasons": ["low-confidence", "stale-evidence"],
                        },
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
        self.assertIn("Widen post-promotion routing policy from priority deltas", titles)
        for stale_title in stale_titles:
            self.assertNotIn(stale_title, titles)

        status = plan["evidence"]["stats_summary"]["post_promotion_priority_delta_status"]
        self.assertEqual(status["schema"], "agentflow.post_promotion_priority_delta_research_status.v1")
        self.assertEqual(status["summary"]["top_next_action"], "widen-local-policy")
        self.assertEqual(status["summary"]["top_local_action_family"], "routing")
        self.assertTrue(status["privacy"]["metadata_only"])
        self.assertFalse(status["privacy"]["raw_prompts_included"])
        self.assertFalse(status["privacy"]["provider_bodies_included"])
        self.assertFalse(status["privacy"]["request_ids_included"])
        self.assertFalse(status["privacy"]["session_ids_included"])
        self.assertFalse(status["privacy"]["cache_keys_included"])
        self.assertFalse(status["privacy"]["individual_candidate_ids_included"])
        self.assertIn("post_promotion_priority_delta_status", plan["evidence"]["inspected_sources"])

        successor = next(item for item in plan["backlog_changes"]["create_issues"] if item["title"] == "Widen post-promotion routing policy from priority deltas")
        self.assertIn("next_action=widen-local-policy", successor["body"])
        self.assertIn("post-promotion priority-delta", successor["body"])
        self.assertIn("routing", successor["labels"])
        self.assertIn("privacy", successor["labels"])

        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertGreaterEqual(suppression["closed_prior_issue_count"], 1)
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw post promotion prompt must not leak", rendered)
        self.assertNotIn("post-promotion-request-secret", rendered)
        self.assertNotIn("post-promotion-session-secret", rendered)
        self.assertNotIn("post-promotion-cache-secret", rendered)
        self.assertNotIn("/home/lutz/private/post_promotion_secret.py", rendered)
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

    def test_openai_tool_light_promotion_decision_is_carried_from_pass_through_lifecycle(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 67,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "tool-light",
                        "c": 40,
                        "openai_canary_holdout_count": 15,
                        "openai_canary_skipped_count": 10,
                        "openai_canary_latest_observed_at": observed_at,
                        "request_id": "secret-promotion-holdout-request",
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4-mini",
                        "category": "tool-light",
                        "c": 12,
                        "openai_canary_applied_count": 12,
                        "openai_canary_latest_observed_at": observed_at,
                        "session_id": "secret-promotion-applied-session",
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4-mini",
                        "routed_model": "gpt-5.4-mini",
                        "category": "chat",
                        "c": 15,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        summary = plan["evidence"]["stats_summary"]
        decision_report = summary["openai_routing_promotion_decision"]
        self.assertEqual(decision_report["schema"], "agentflow.openai_routing_promotion_decision_report.v1")
        self.assertEqual(decision_report["decision"], "active-local-policy")
        self.assertFalse(decision_report["promotion_ready"])
        self.assertEqual(decision_report["summary"]["applied_count"], 12)
        self.assertEqual(decision_report["summary"]["holdout_count"], 15)
        self.assertEqual(decision_report["summary"]["safety_stop_count"], 0)
        self.assertEqual(decision_report["summary"]["error_count"], 0)
        self.assertEqual(decision_report["summary"]["fallback_count"], 0)
        self.assertEqual(decision_report["summary"]["retry_count"], 0)
        self.assertEqual(decision_report["summary"]["next_action"], "measure-openai-routing-rule-outcomes")
        self.assertEqual(decision_report["summary"]["target_local_rule_file"], "routing_rules.yaml")

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        routing_lever = next(row for row in loop["levers"] if row["lever"] == "routing")
        self.assertEqual(routing_lever["evidence_source"], "agentflow.openai_routing_promotion_decision_report.v1")
        self.assertEqual(routing_lever["state"], "active-local-policy")
        self.assertEqual(routing_lever["next_action"], "measure-openai-routing-rule-outcomes")
        self.assertEqual(routing_lever["applied_count"], 12)
        self.assertEqual(routing_lever["holdout_count"], 15)

        routing_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "routing")
        self.assertEqual(routing_candidate["blocker"], "openai-routing-promotion-active-local-policy")
        self.assertEqual(routing_candidate["projected_savings_signal"]["decision"], "active-local-policy")
        self.assertEqual(routing_candidate["projected_savings_signal"]["target_local_rule_file"], "routing_rules.yaml")

        rendered = json.dumps(plan)
        self.assertNotIn("secret-promotion-holdout-request", rendered)
        self.assertNotIn("secret-promotion-applied-session", rendered)

    def test_pass_through_routing_report_keeps_openai_lifecycle_shape_specific_for_low_volume_holdout(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 17,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4",
                        "category": "tool-heavy",
                        "workflow_phase": "unknown",
                        "c": 2,
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4-mini",
                        "category": "tool-heavy",
                        "workflow_phase": "unknown",
                        "c": 2,
                        "openai_canary_applied_count": 2,
                        "openai_canary_latest_observed_at": observed_at,
                    },
                    {
                        "provider": "openai",
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.4",
                        "routed_model": "gpt-5.4-mini",
                        "category": "tool-light",
                        "workflow_phase": "unknown",
                        "c": 13,
                        "openai_canary_applied_count": 13,
                        "openai_canary_latest_observed_at": observed_at,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        candidate = next(
            bucket
            for bucket in report["buckets"]
            if bucket["provider"] == "openai"
            and bucket["requested_model"] == "gpt-5.4"
            and bucket["category"] == "tool-heavy"
        )
        lifecycle = candidate["openai_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 0)
        self.assertEqual(lifecycle["coverage"]["matched_count"], 2)
        self.assertLessEqual(lifecycle["coverage"]["observed_rate"], 1.0)
        self.assertLessEqual(lifecycle["coverage"]["applied_rate"], 1.0)
        self.assertIn("missing-holdout-coverage", lifecycle["blocker_codes"])
        self.assertIn("insufficient-volume-for-holdout", lifecycle["blocker_codes"])
        self.assertNotIn("lifecycle-observed-count-exceeds-matched-count", lifecycle["integrity_warning_codes"])
        self.assertEqual(report["summary"]["openai_canary_applied_count"], 2)
        self.assertEqual(report["summary"]["openai_canary_holdout_count"], 0)

        rendered = json.dumps(plan)
        self.assertNotIn("secret", rendered)

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

    def test_pass_through_routing_report_merges_anthropic_safety_stop_lifecycle_across_tool_result_cohorts(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 156,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "routing": [
                    {
                        "provider": "anthropic",
                        "source_surface": "unknown",
                        "endpoint": "unknown",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "workflow_phase": "unknown",
                        "c": 100,
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-sonnet-4-6",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "c": 51,
                        "anthropic_canary_safety_stopped_count": 51,
                        "anthropic_canary_latest_observed_at": observed_at,
                        "session_id": "secret-anthropic-safety-session-id",
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        report = plan["evidence"]["stats_summary"]["pass_through_routing_report"]
        self.assertEqual(report["summary"]["anthropic_canary_safety_stopped_count"], 51)
        top_candidate = report["buckets"][0]
        self.assertEqual(top_candidate["provider"], "anthropic")
        self.assertEqual(top_candidate["source_surface"], "unknown")
        self.assertEqual(top_candidate["candidate_target_model"], "claude-haiku-4-5-20251001")
        self.assertTrue(top_candidate["anthropic_canary_lifecycle_related_only"])
        lifecycle = top_candidate["anthropic_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.anthropic_routing_canary_lifecycle_evidence.v1")
        self.assertEqual(lifecycle["cohort_counts"]["safety_stopped"], 51)
        self.assertEqual(lifecycle["latest_observed_at"], observed_at)
        self.assertIn("safety-stop-observed", lifecycle["blocker_codes"])
        self.assertIn("missing-applied-coverage", lifecycle["blocker_codes"])
        self.assertIn("missing-holdout-coverage", lifecycle["blocker_codes"])
        safety_breakdown = lifecycle["safety_stop_breakdown"]
        self.assertEqual(safety_breakdown[0]["reason_code"], "thinking-routing-guard")
        self.assertEqual(safety_breakdown[0]["count"], 51)
        self.assertEqual(safety_breakdown[0]["category"], "tool-result")
        self.assertEqual(safety_breakdown[0]["source_surface"], "anthropic_messages")
        self.assertFalse(safety_breakdown[0]["executor_compatible"])
        self.assertTrue(safety_breakdown[0]["missing_applied_coverage"])
        self.assertTrue(safety_breakdown[0]["missing_holdout_coverage"])
        self.assertEqual(
            safety_breakdown[0]["durable_blocked_reason"],
            "anthropic-routing-safety-stop-thinking-routing-guard-keep-blocked",
        )
        safety_bucket = next(bucket for bucket in report["buckets"] if bucket["source_surface"] == "anthropic_messages")
        self.assertEqual(
            safety_bucket["anthropic_canary_lifecycle_evidence"]["cohort_counts"]["safety_stopped"],
            51,
        )
        routing_stage = next(row for row in plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]["levers"] if row["lever"] == "routing")
        self.assertEqual(routing_stage["state"], "keep-blocked")
        self.assertEqual(routing_stage["next_action"], "keep-anthropic-routing-blocked-until-safety-stop-burndown")
        self.assertEqual(
            routing_stage["keep_blocked_reason"],
            "anthropic-routing-safety-stop-thinking-routing-guard-keep-blocked",
        )
        self.assertEqual(routing_stage["safety_stop_count"], 51)
        self.assertIn("safety_stop_reason_review", routing_stage["needed_resolution"])
        self.assertIn("applied_coverage", routing_stage["needed_resolution"])
        self.assertNotEqual(routing_stage["state"], "missing-evidence")

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        routing_entry = next(row for row in ledger["entries"] if row["lever"] == "routing")
        self.assertEqual(routing_entry["current_status"], "keep-blocked")
        self.assertEqual(routing_entry["issue_worthy_status"], "blocked")
        self.assertEqual(
            routing_entry["keep_blocked_reason"],
            "anthropic-routing-safety-stop-thinking-routing-guard-keep-blocked",
        )
        self.assertEqual(routing_entry["safety_stop_breakdown"][0]["reason_code"], "thinking-routing-guard")
        duplicate_suppression = routing_entry["duplicate_suppression"]
        self.assertEqual(
            duplicate_suppression["schema"],
            "agentflow.anthropic_routing_activation_issue_duplicate_suppression.v1",
        )
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertEqual(duplicate_suppression["safety_stop_count"], 51)
        self.assertTrue(duplicate_suppression["missing_applied_coverage"])
        self.assertTrue(duplicate_suppression["missing_holdout_coverage"])
        self.assertTrue(str(duplicate_suppression["fingerprint"]).startswith("activation:"))

        burndown = plan["evidence"]["activation_safety_stop_burndown"]
        self.assertEqual(burndown["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(burndown["summary"]["anthropic_routing_safety_stop_count"], 51)
        self.assertIn("activation_safety_stop_burndown", plan["evidence"]["inspected_sources"])
        burndown_group = next(row for row in burndown["groups"] if row["source"] == "pass_through_routing_report")
        self.assertEqual(burndown_group["burndown_status"], "safety-stop-active")
        self.assertEqual(burndown_group["status"], "blocked")
        self.assertEqual(burndown_group["safety_stop_count"], 51)
        self.assertEqual(burndown_group["source_surface"], "unknown")
        self.assertEqual(burndown_group["endpoint"], "unknown")
        self.assertTrue(burndown_group["missing_applied_coverage"])
        self.assertTrue(burndown_group["missing_holdout_coverage"])
        self.assertIn("rollback_proof", burndown_group["needed_resolution"])
        self.assertEqual(burndown_group["safety_stop_reason_review"]["status"], "missing")
        self.assertEqual(burndown_group["safer_threshold_or_executor_guard"]["status"], "missing")
        self.assertFalse(burndown_group["safer_threshold_or_executor_guard"]["executor_compatible"])
        self.assertEqual(burndown_group["rollback_proof"]["status"], "missing")
        self.assertEqual(burndown_group["applied_coverage"]["status"], "missing")
        self.assertEqual(burndown_group["holdout_coverage"]["status"], "missing")
        self.assertFalse(burndown_group["promotion_allowed"])
        self.assertFalse(burndown_group["stage_allowed"])
        unblock = burndown_group["unblock_criteria"]
        self.assertEqual(unblock["schema"], "agentflow.anthropic_routing_safety_stop_unblock_criteria.v1")
        self.assertEqual(unblock["status"], "blocked")
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertFalse(unblock["applied_coverage_present"])
        self.assertFalse(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["safer_threshold_or_executor_guard_present"])
        self.assertFalse(unblock["rollback_proof_present"])
        self.assertTrue(burndown_group["duplicate_suppression"]["suppresses_new_activation_issue"])

        burndown_entry = next(
            row
            for row in ledger["entries"]
            if row.get("evidence_schema") == "agentflow.activation_safety_stop_burndown.v1"
            and row.get("local_action_family") == "routing"
        )
        self.assertEqual(burndown_entry["current_status"], "keep-blocked")
        self.assertEqual(burndown_entry["status"], "blocked")
        self.assertEqual(burndown_entry["safety_stop_count"], 51)
        self.assertEqual(burndown_entry["source_surface"], "unknown")
        self.assertEqual(burndown_entry["endpoint"], "unknown")
        self.assertTrue(burndown_entry["missing_applied_coverage"])
        self.assertTrue(burndown_entry["missing_holdout_coverage"])
        self.assertEqual(burndown_entry["safety_stop_reason_review"]["status"], "missing")
        self.assertEqual(burndown_entry["safer_threshold_or_executor_guard"]["status"], "missing")
        self.assertFalse(burndown_entry["safer_threshold_or_executor_guard"]["executor_compatible"])
        self.assertEqual(burndown_entry["rollback_proof"]["status"], "missing")
        self.assertEqual(burndown_entry["applied_coverage"]["status"], "missing")
        self.assertEqual(burndown_entry["holdout_coverage"]["status"], "missing")
        self.assertEqual(burndown_entry["safety_stop_breakdown"][0]["reason_code"], "thinking-routing-guard")
        self.assertTrue(burndown_entry["duplicate_suppression"]["suppresses_new_activation_issue"])
        self.assertFalse(burndown_entry["promotion_allowed"])
        self.assertFalse(burndown_entry["stage_allowed"])
        self.assertFalse(burndown_entry["active_policy_changed"])
        self.assertFalse(burndown_entry["wrote_active_policy_files"])
        self.assertTrue(burndown_entry["durable_action_ledger_entry"])
        self.assertFalse(burndown_entry["executor_compatible"])
        self.assertEqual(burndown_entry["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(burndown_entry["target_local_policy_section"], "routing.rules")
        representation = burndown_entry["local_file_backed_representation"]
        self.assertTrue(representation["exists"])
        self.assertEqual(representation["policy_section"], "routing")
        self.assertEqual(representation["rule_file"], "routing_rules.yaml")
        self.assertTrue(str(burndown_entry["fingerprint"]).startswith("activation:"))
        self.assertNotEqual(
            burndown_entry["fingerprint"],
            routing_entry["fingerprint"],
        )
        self.assertEqual(burndown_entry["unblock_criteria"]["status"], "blocked")
        self.assertFalse(burndown_entry["unblock_criteria"]["safety_stop_count_zero"])
        self.assertFalse(burndown_entry["unblock_criteria"]["applied_coverage_present"])
        self.assertFalse(burndown_entry["unblock_criteria"]["holdout_coverage_present"])
        self.assertFalse(burndown_entry["unblock_criteria"]["promotion_allowed"])
        self.assertFalse(burndown_entry["unblock_criteria"]["stage_allowed"])

        routing_candidate = next(
            row for row in plan["evidence"]["optimization_candidates"]
            if row["lever"] == "routing"
        )
        self.assertEqual(
            routing_candidate["issue_generation_status"],
            "suppressed-anthropic-routing-safety-stop-burndown",
        )
        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn(
            "Stage routing evidence for claude-sonnet-4-6 to claude-haiku-4-5-20251001",
            created_titles,
        )

        rendered = json.dumps(plan)
        self.assertNotIn("secret-anthropic-safety-session-id", rendered)

    def test_pass_through_routing_report_treats_clean_anthropic_lifecycle_as_activation_ready(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 44,
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
                        "anthropic_canary_holdout_count": 4,
                        "anthropic_canary_latest_observed_at": observed_at,
                    },
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "requested_model": "claude-sonnet-4-6",
                        "routed_model": "claude-haiku-4-5-20251001",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "c": 4,
                        "anthropic_canary_applied_count": 4,
                        "anthropic_canary_latest_observed_at": observed_at,
                    },
                ],
            },
            threshold=3,
            now=NOW,
        )

        routing_stage = next(
            row for row in plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]["levers"]
            if row["lever"] == "routing"
        )
        self.assertEqual(routing_stage["state"], "ranked-evidence")
        self.assertEqual(routing_stage["next_action"], "stage-anthropic-routing-canary")
        self.assertEqual(routing_stage["safety_stop_count"], 0)
        self.assertNotIn("duplicate_suppression", routing_stage)

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        routing_entry = next(row for row in ledger["entries"] if row["lever"] == "routing")
        self.assertEqual(routing_entry["current_status"], "projected")
        self.assertEqual(routing_entry["issue_worthy_status"], "ready")
        burndown = plan["evidence"]["activation_safety_stop_burndown"]
        self.assertEqual(burndown["schema"], "agentflow.activation_safety_stop_burndown.v1")
        self.assertEqual(burndown["status"], "no-safety-stop-evidence")
        self.assertEqual(burndown["groups"], [])
        self.assertEqual(burndown["summary"]["anthropic_routing_safety_stop_count"], 0)
        self.assertNotIn("activation_safety_stop_burndown", plan["evidence"]["inspected_sources"])
        self.assertFalse(
            [
                row
                for row in ledger["entries"]
                if row.get("evidence_schema") == "agentflow.activation_safety_stop_burndown.v1"
                and row.get("keep_blocked_reason")
            ]
        )

        routing_candidate = next(
            row for row in plan["evidence"]["optimization_candidates"]
            if row["lever"] == "routing"
        )
        self.assertNotIn("issue_generation_status", routing_candidate)
        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertIn(
            "Stage routing evidence for claude-sonnet-4-6 to claude-haiku-4-5-20251001",
            created_titles,
        )

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

    def test_local_activation_next_action_queue_ranks_savings_and_unblock_reasons(self):
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
                    "state": "measured-active",
                    "current_status": "applied",
                    "next_action": "keep-active",
                    "blocker_codes": ["repeated-context-crunch-active-at-max-rollout"],
                    "sample_count": 2657,
                    "applied_count": 107,
                    "holdout_count": 40,
                    "crunch_savings_usd": 25.818387,
                    "projected_saved_usd": 25.818387,
                    "target_local_rule_file": "crunch_rules.yaml",
                    "target_local_policy_section": "crunch.rules",
                    "duplicate_suppression": {
                        "reason": "repeated-context-crunch-active-at-max-rollout",
                        "suppresses_new_activation_issue": True,
                        "metadata_only": True,
                        "aggregate_only": True,
                    },
                },
                {
                    "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                    "rank": 2,
                    "fingerprint": "activation:routing",
                    "lever": "routing",
                    "local_action_family": "routing",
                    "state": "keep-staged",
                    "current_status": "holdout",
                    "next_action": "collect-openai-routing-canary-evidence",
                    "blocker_codes": ["unknown-canary-lifecycle-rows"],
                    "sample_count": 414,
                    "applied_count": 21,
                    "holdout_count": 18,
                    "savings_per_1000_calls_usd": 4.375,
                    "target_local_rule_file": "routing_rules.yaml",
                    "target_local_policy_section": "routing.rules",
                },
                {
                    "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                    "rank": 3,
                    "fingerprint": "activation:tool-cache",
                    "lever": "cache",
                    "local_action_family": "cache",
                    "state": "ranked-evidence",
                    "current_status": "projected",
                    "next_action": "collect-file-invalidation-evidence",
                    "blocker_codes": [
                        "invalidation-evidence-missing",
                        "tools-present",
                        "unsafe-tool-calls-without-invalidation",
                    ],
                    "sample_count": 183,
                    "target_local_rule_file": "cache_rules.yaml",
                    "target_local_policy_section": "cache.pattern_rules",
                    "request_id": "req-queue-secret",
                    "session_id": "session-queue-secret",
                    "cache_key": "cache-queue-secret",
                    "file_path": "/tmp/private-queue.py",
                },
                {
                    "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                    "rank": 4,
                    "fingerprint": "activation:retired-cache",
                    "lever": "cache",
                    "local_action_family": "cache",
                    "state": "retired-no-repeat",
                    "current_status": "superseded",
                    "next_action": "retire-cache-replay-canary-no-repeat",
                    "blocker_codes": [
                        "retire-staged-no-repeat",
                        "repeat-window-elapsed-no-live-repeat",
                    ],
                    "sample_count": 75,
                    "applied_count": 28,
                    "holdout_count": 47,
                    "projected_saved_usd": 0.075373,
                    "target_local_rule_file": "cache_rules.yaml",
                    "target_local_policy_section": "cache.pattern_rules",
                    "duplicate_suppression": {
                        "reason": "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
                        "suppresses_new_cache_replay_stage_issue": True,
                        "metadata_only": True,
                        "aggregate_only": True,
                    },
                },
            ],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        queue = build_local_activation_next_action_queue(
            {"evidence_to_activation_next_action_ledger": ledger}
        )

        self.assertEqual(queue["schema"], "agentflow.local_activation_next_action_queue.v1")
        self.assertEqual(queue["status"], "ranked")
        self.assertEqual(
            [(entry["lever"], entry["next_action"]) for entry in queue["entries"]],
            [
                ("crunch", "keep-active"),
                ("routing", "collect-openai-routing-canary-evidence"),
                ("cache", "collect-file-invalidation-evidence"),
                ("cache", "retire-cache-replay-canary-no-repeat"),
            ],
        )
        self.assertEqual(queue["entries"][0]["realized_savings_usd"], 25.818387)
        self.assertEqual(queue["entries"][0]["unblock_reason"], "repeated-context-crunch-active-at-max-rollout")
        self.assertEqual(queue["entries"][0]["duplicate_suppression_status"], "suppressed")
        self.assertEqual(queue["entries"][1]["projected_savings_usd"], 1.81125)
        self.assertEqual(queue["entries"][1]["unblock_reason"], "unknown-canary-lifecycle-rows")
        self.assertEqual(queue["entries"][2]["unblock_reason"], "invalidation-evidence-missing")
        self.assertEqual(queue["entries"][3]["current_status"], "superseded")
        self.assertEqual(queue["entries"][3]["duplicate_suppression_status"], "suppressed")
        self.assertEqual(queue["summary"]["top_lever"], "crunch")
        self.assertEqual(queue["summary"]["top_unblock_reason"], "repeated-context-crunch-active-at-max-rollout")
        self.assertTrue(queue["privacy"]["metadata_only"])
        self.assertTrue(queue["privacy"]["aggregate_only"])
        self.assertFalse(queue["privacy"]["raw_prompts_included"])
        self.assertFalse(queue["privacy"]["provider_bodies_included"])
        self.assertFalse(queue["privacy"]["cache_keys_included"])
        self.assertFalse(queue["privacy"]["request_ids_included"])
        self.assertFalse(queue["privacy"]["session_ids_included"])
        self.assertFalse(queue["privacy"]["tenant_ids_included"])
        self.assertFalse(queue["privacy"]["tool_payloads_included"])
        self.assertFalse(queue["privacy"]["file_paths_included"])
        self.assertFalse(queue["privacy"]["absolute_paths_included"])

        report = build_evidence_to_activation_burndown(
            {
                "schema": "agentflow.orchestrator_research_plan.v1",
                "evidence": {"stats_summary": {"evidence_to_activation_next_action_ledger": ledger}},
            },
            now=NOW,
        )
        self.assertEqual(
            report["next_action_queue"]["entries"][0]["next_action"],
            "keep-active",
        )
        rendered = json.dumps({"queue": queue, "report": report}, sort_keys=True)
        self.assertNotIn("req-queue-secret", rendered)
        self.assertNotIn("session-queue-secret", rendered)
        self.assertNotIn("cache-queue-secret", rendered)
        self.assertNotIn("/tmp/private-queue.py", rendered)

    def test_repeated_activation_feedback_blockers_become_durable_ledger_entries(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing skipped blocker=missing-anthropic-canary-lifecycle-evidence request_id=req-ledger-secret",
                        "routing skipped blocker=missing-applied-coverage session_id=session-ledger-secret",
                        "routing skipped blocker=missing-holdout-coverage candidate_id=candidate-ledger-secret",
                        "crunch omitted reason=repeated-context-crunch-opportunity request_id=req-crunch-secret",
                        "cache skipped blocker=invalidation-evidence-missing cache_key=cache-ledger-secret",
                        "cache skipped blocker=unsupported-streaming-shape request_id=req-stream-secret",
                        "activation feedback omitted by unknown gate with no machine reason request_id=req-unclassified-secret",
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

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        self.assertEqual(ledger["schema"], "agentflow.evidence_to_activation_next_action_ledger.v1")
        self.assertTrue(ledger["privacy"]["metadata_only"])
        self.assertTrue(ledger["privacy"]["aggregate_only"])
        self.assertFalse(ledger["privacy"]["raw_prompts_included"])
        self.assertFalse(ledger["privacy"]["provider_bodies_included"])
        self.assertFalse(ledger["privacy"]["request_ids_included"])
        self.assertFalse(ledger["privacy"]["session_ids_included"])
        self.assertFalse(ledger["privacy"]["cache_keys_included"])

        by_blocker = {
            blocker: entry
            for entry in ledger["entries"]
            for blocker in entry.get("blocker_codes") or []
        }
        self.assertEqual(
            by_blocker["missing-anthropic-canary-lifecycle-evidence"]["next_action"],
            "activate-anthropic-routing-canary-cohorts",
        )
        self.assertEqual(by_blocker["missing-anthropic-canary-lifecycle-evidence"]["lever"], "routing")
        self.assertEqual(by_blocker["missing-applied-coverage"]["local_action_family"], "routing")
        self.assertEqual(by_blocker["missing-holdout-coverage"]["current_status"], "blocked")
        self.assertEqual(by_blocker["repeated-context-crunch-opportunity"]["lever"], "crunch")
        self.assertEqual(
            by_blocker["repeated-context-crunch-opportunity"]["next_action"],
            "stage-repeated-context-crunch-canary",
        )
        self.assertEqual(by_blocker["invalidation-evidence-missing"]["lever"], "cache")
        self.assertEqual(
            by_blocker["invalidation-evidence-missing"]["next_action"],
            "collect-cache-invalidation-evidence",
        )
        self.assertEqual(by_blocker["unsupported-streaming-shape"]["lever"], "cache")
        self.assertEqual(
            by_blocker["unsupported-streaming-shape"]["next_action"],
            "add-streaming-cache-replay-support-or-route-to-crunch-canary",
        )
        self.assertNotIn("unclassified-skip-or-blocker", by_blocker)
        self.assertEqual(by_blocker["activation-feedback-blocker-review"]["lever"], "activation-feedback")
        self.assertEqual(by_blocker["activation-feedback-blocker-review"]["local_action_family"], "activation-feedback")
        self.assertEqual(
            by_blocker["activation-feedback-blocker-review"]["next_action"],
            "keep-activation-feedback-blocker-review-blocked-until-new-sanitized-local-evidence",
        )
        self.assertEqual(by_blocker["activation-feedback-blocker-review"]["current_status"], "keep-blocked")
        self.assertEqual(by_blocker["activation-feedback-blocker-review"]["issue_worthy_status"], "blocked")
        self.assertEqual(
            by_blocker["activation-feedback-blocker-review"]["keep_blocked_reason"],
            "activation-feedback-blocker-review-already-resolved-to-bounded-local-action-ledger",
        )
        self.assertEqual(
            by_blocker["activation-feedback-blocker-review"]["diagnostic_fingerprint"],
            "agentflow.repeated-diagnostic.activation-feedback-blocker-review.v1",
        )
        self.assertIn(
            "durable local action issue",
            by_blocker["activation-feedback-blocker-review"]["expected_savings_path"],
        )
        self.assertTrue(all(entry.get("issue_worthy_status") for entry in ledger["entries"]))
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("req-ledger-secret", rendered)
        self.assertNotIn("session-ledger-secret", rendered)
        self.assertNotIn("candidate-ledger-secret", rendered)
        self.assertNotIn("cache-ledger-secret", rendered)
        self.assertNotIn("req-unclassified-secret", rendered)

    def test_no_local_representation_diagnostic_gets_review_only_local_action_ledger_entry(self):
        log_lines = [
            "activation feedback blocked reason=no-local-representation request_id=req-no-local-secret",
            "activation feedback blocked reason=no-local-representation session_id=session-no-local-secret",
        ]

        plan = build_research_plan(
            issues=[],
            log_sources=log_lines,
            threshold=1,
            now=NOW,
        )

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        entries = [
            entry
            for entry in ledger["entries"]
            if entry.get("diagnostic_class") == "no-local-representation"
        ]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["schema"], "agentflow.evidence_to_activation_next_action_ledger_entry.v1")
        self.assertEqual(entry["lever"], "activation-feedback")
        self.assertEqual(entry["local_action_family"], "activation-feedback")
        self.assertEqual(entry["current_status"], "keep-blocked")
        self.assertEqual(entry["state"], "keep-blocked")
        self.assertEqual(entry["issue_worthy_status"], "blocked")
        self.assertEqual(entry["sample_count"], 2)
        self.assertEqual(
            entry["next_action"],
            "record-review-only-local-representation-and-wait-for-supported-local-action",
        )
        self.assertEqual(
            entry["keep_blocked_reason"],
            "activation-feedback-no-local-representation-resolved-to-review-only-local-artifact",
        )
        self.assertEqual(
            entry["diagnostic_fingerprint"],
            "agentflow.repeated-diagnostic.no-local-representation.v1",
        )
        self.assertTrue(entry["durable_action_ledger_entry"])
        self.assertEqual(entry["review_status"], "resolved-to-review-only-no-op")
        self.assertIn("supported_file_backed_local_action", entry["needed_resolution"])
        representation = entry["local_action_representation"]
        self.assertEqual(
            representation["schema"],
            "agentflow.activation_feedback_local_action_representation.v1",
        )
        self.assertEqual(representation["representation_kind"], "review-only-no-op")
        self.assertEqual(representation["review_artifact"], "evidence-to-activation-next-action-ledger")
        self.assertFalse(representation["local_rule_available"])
        self.assertFalse(representation["file_backed_policy_available"])
        self.assertFalse(representation["dry_run_evidence_available"])
        self.assertFalse(representation["canary_evidence_available"])
        self.assertFalse(representation["managed_enforcement_required"])
        self.assertFalse(representation["managed_enforced"])
        self.assertFalse(representation["provider_body_rewrite_required"])
        self.assertFalse(representation["provider_body_rewrite"])
        self.assertFalse(representation["policy_files_written"])
        self.assertTrue(representation["privacy"]["metadata_only"])
        self.assertTrue(entry["privacy"]["metadata_only"])
        self.assertTrue(entry["privacy"]["aggregate_only"])
        self.assertFalse(entry["privacy"]["raw_prompts_included"])
        self.assertFalse(entry["privacy"]["provider_bodies_included"])
        self.assertFalse(entry["privacy"]["request_ids_included"])
        self.assertFalse(entry["privacy"]["session_ids_included"])
        self.assertFalse(entry["privacy"]["cache_keys_included"])

        created_titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn("Resolve repeated-no-local-representation activation feedback blocker", created_titles)
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["activation_feedback_keep_blocked_suppressed_count"], 1)
        self.assertEqual(
            suppression["suppressed"][-1]["keep_blocked_reason"],
            "activation-feedback-no-local-representation-resolved-to-review-only-local-artifact",
        )

        repeated_plan = build_research_plan(
            issues=[],
            log_sources=log_lines,
            threshold=1,
            now=NOW,
        )
        repeated_entry = [
            item
            for item in repeated_plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]["entries"]
            if item.get("diagnostic_class") == "no-local-representation"
        ][0]
        self.assertEqual(repeated_entry["fingerprint"], entry["fingerprint"])
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("req-no-local-secret", rendered)
        self.assertNotIn("session-no-local-secret", rendered)

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

    def test_issue_534_request_shape_crunch_ledger_advances_to_measurement(self):
        open_ledger_issue = issue(
            534,
            "Advance repeated-context cohort from evidence-to-activation ledger (evidence 3a012e702da0)",
            ["backlog", "status:ready", "priority:p2", "privacy"],
            body="Existing implementation issue.\n\nFingerprint: activation:3a012e702da0a8a8\n",
        )
        plan = build_research_plan(
            issues=[open_ledger_issue],
            stats={
                "calls": 1000,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "summary": {
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                            "ranked_candidate_count": 1,
                            "rows_considered": 388,
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "local_action_family": "crunch",
                                "next_action": "stage-repeated-context-crunch-canary",
                                "readiness_state": "activation-ready",
                                "sample_count": 388,
                                "row_count": 388,
                                "projected_savings_usd": 2.530359,
                                "projected_saved_tokens": 843452,
                                "blocker_codes": [
                                    "thinking-routing-guard",
                                    "tool-call-cache-disabled",
                                    "unsupported-streaming-shape",
                                ],
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "category": "tool-result",
                                "workflow_phase": "thinking",
                                "stream": True,
                                "has_tools": True,
                                "text_bucket": "gte_128k_chars",
                                "token_bucket": "lt_500_tokens",
                                "cache_status": "skipped",
                                "routing_status": "passthrough",
                                "request_id": "raw-request-shape-issue-534-secret",
                                "session_id": "raw-session-issue-534-secret",
                                "cache_key": "raw-cache-issue-534-secret",
                                "file_path": "/tmp/private-issue-534.py",
                            }
                        ],
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "request_shape_crunch_opportunity": {
                    "schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                    "status": "projected-savings-ranked",
                    "summary": {
                        "rows_considered": 738,
                        "matched_count": 738,
                        "candidate_count": 15,
                        "projected_saved_usd": 3.997114,
                        "projected_saved_tokens": 1344975,
                        "recommended_action_count": 1,
                    },
                    "activation_follow_up": {
                        "status": "canary-staged",
                        "activation_state": "measurement-required",
                        "activation_mode": "staged-canary-measurement",
                        "next_action": "measure-repeated-context-crunch-canary-impact",
                        "missing_measurements": ["missing-crunch-canary-impact-measurement"],
                        "canary_already_staged": True,
                        "canary_already_applied": True,
                        "no_op_reason": "matching-repeated-context-crunch-canary-already-staged",
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        entry = next(item for item in ledger["entries"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(entry["fingerprint"], "activation:3a012e702da0a8a8")
        self.assertEqual(entry["fingerprint_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(entry["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(entry["current_status"], "staged")
        self.assertEqual(entry["state"], "measurement-required")
        self.assertEqual(entry["blocker_codes"], ["missing-crunch-canary-impact-measurement"])
        self.assertEqual(entry["evidence_schema"], "agentflow.request_shape_follow_up_candidates.v1")
        self.assertEqual(
            entry["activation_follow_up_evidence_schema"],
            "agentflow.request_shape_crunch_opportunity_dry_run.v1",
        )
        self.assertTrue(entry["canary_already_staged"])
        self.assertTrue(entry["canary_already_applied"])

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn("Stage request-shape repeated-context crunch canary", titles)
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["fingerprint_match_count"], 1)
        self.assertEqual(suppression["open_existing_issue_count"], 1)
        self.assertEqual(suppression["suppressed"][0]["evidence_fingerprint"], "activation:3a012e702da0a8a8")
        self.assertEqual(suppression["suppressed"][0]["existing_issue"]["number"], 534)

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-request-shape-issue-534-secret", rendered)
        self.assertNotIn("raw-session-issue-534-secret", rendered)
        self.assertNotIn("raw-cache-issue-534-secret", rendered)
        self.assertNotIn("/tmp/private-issue-534.py", rendered)

    def test_issue_544_request_shape_widening_proposal_is_action_specific(self):
        plan = build_research_plan(
            issues=[
                issue(
                    544,
                    "Rank request-shape blockers into local action cohorts",
                    ["backlog", "status:ready", "priority:p1", "privacy"],
                    state="CLOSED",
                    closed="2026-06-16T01:00:00Z",
                )
            ],
            stats={
                "calls": 3130,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 1000, "rollup_count": 32},
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "category": "tool-result",
                                "workflow_phase": "thinking",
                                "stream": True,
                                "has_tools": True,
                                "cache_status": "skipped",
                                "routing_status": "passthrough",
                                "row_count": 388,
                                "sample_count": 388,
                                "projected_saved_tokens": 843452,
                                "projected_savings_usd": 2.530359,
                                "candidate_work_classes": ["crunch", "repeated_context", "replayability"],
                                "blocker_codes": [
                                    "thinking-routing-guard",
                                    "tool-call-cache-disabled",
                                    "unsupported-streaming-shape",
                                ],
                                "request_id": "raw-request-shape-issue-544-secret",
                                "session_id": "raw-session-issue-544-secret",
                            }
                        ],
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "crunch_policy_decision": {
                        "schema": "agentflow.request_shape_crunch_policy_decision.v1",
                        "status": "decided",
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "summary": {
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "applied_count": 2,
                            "holdout_count": 7,
                            "observed_saved_tokens": 2298,
                            "observed_saved_usd": 0.006894,
                            "safety_stop_state": "none",
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        stats_summary = plan["evidence"]["stats_summary"]
        shape_signal = stats_summary["request_shape_rollup_candidates"]
        top_cohort = shape_signal["local_action_cohorts"][0]
        self.assertEqual(top_cohort["rank"], 1)
        self.assertEqual(top_cohort["local_action_family"], "crunch")
        self.assertEqual(top_cohort["readiness_state"], "activation-ready")
        self.assertEqual(top_cohort["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(top_cohort["sample_count"], 388)
        self.assertEqual(top_cohort["projected_saved_tokens"], 843452)
        self.assertAlmostEqual(top_cohort["projected_savings_usd"], 2.530359)
        self.assertEqual(top_cohort["blocker_reason"], "thinking-routing-guard")
        self.assertTrue(shape_signal["privacy"]["metadata_only"])
        self.assertTrue(shape_signal["privacy"]["aggregate_only"])

        ledger = stats_summary["evidence_to_activation_next_action_ledger"]
        request_shape_entry = next(item for item in ledger["entries"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_entry["local_action_family"], "crunch")
        self.assertEqual(request_shape_entry["next_action"], "widen")
        self.assertEqual(request_shape_entry["legacy_issue_title"], "Apply measured request-shape crunch widening to local rules")

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn("Rank request-shape blockers into local action cohorts", titles)
        self.assertTrue(
            any(title.startswith("Apply measured request-shape crunch widening") for title in titles),
            titles,
        )
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-request-shape-issue-544-secret", rendered)
        self.assertNotIn("raw-session-issue-544-secret", rendered)

    def test_recent_trusted_closed_issue_suppresses_same_stage_proposal(self):
        stale_title = "Stage cache replay canary for replay-ready on openai/openai_responses/responses"
        plan = build_research_plan(
            issues=[
                issue(
                    515,
                    stale_title,
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-06-11T08:20:00Z",
                )
            ],
            stats=cache_replayability_stats(),
            threshold=3,
            now=NOW,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn(stale_title, titles)
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertGreaterEqual(suppression["closed_prior_issue_count"], 1)
        closed = next(row for row in suppression["suppressed"] if row["title"] == stale_title)
        self.assertEqual(closed["suppression_kind"], "closed-prior-issue")
        self.assertEqual(closed["existing_issue"]["number"], 515)

    def test_only_recent_trusted_closed_issues_suppress_stage_proposals(self):
        stale_title = "Stage cache replay canary for replay-ready on openai/openai_responses/responses"
        plan = build_research_plan(
            issues=[
                issue(
                    40,
                    stale_title,
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-05-01T00:00:00Z",
                ),
                issue(
                    41,
                    stale_title,
                    ["backlog", "status:ready", "cache"],
                    author="external",
                    state="CLOSED",
                    closed="2026-06-11T08:20:00Z",
                ),
            ],
            stats=cache_replayability_stats(),
            threshold=3,
            now=NOW,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertIn(stale_title, titles)
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["closed_prior_issue_count"], 0)

    def test_recent_closed_issue_with_matching_evidence_fingerprint_suppresses_proposal(self):
        proposal = {
            "repo": "lutzkuen/agentflow",
            "title": "Follow up cache replay cohort with renamed title",
            "labels": ["backlog", "status:ready"],
            "body": "## Evidence\n\n- Fingerprint: activation:abc123456789\n",
        }

        deduped, suppression = _dedupe_create_issue_proposals_with_metadata(
            [proposal],
            existing_issues=[
                issue(
                    616,
                    "Completed cache replay predecessor with different title",
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-06-11T08:20:00Z",
                    body="Resolved predecessor.\n\nFingerprint: activation:abc123456789\n",
                )
            ],
            trusted_author="lutzkuen",
            now=NOW,
        )

        self.assertEqual(deduped, [])
        self.assertEqual(suppression["fingerprint_match_count"], 1)
        self.assertEqual(suppression["closed_prior_issue_count"], 1)
        suppressed = suppression["suppressed"][0]
        self.assertEqual(suppressed["reason"], "evidence-fingerprint-already-exists")
        self.assertEqual(suppressed["suppression_kind"], "closed-prior-issue")
        self.assertEqual(suppressed["existing_issue"]["number"], 616)

    def test_recent_closed_issue_with_same_fingerprint_allows_advanced_next_action(self):
        proposal = {
            "repo": "lutzkuen/agentflow",
            "title": "Advance cache next action from evidence-to-activation ledger (evidence abc123456789)",
            "labels": ["backlog", "status:ready", "cache"],
            "body": "## Evidence\n\n- Fingerprint: activation:abc123456789\n- Top next action: review-cache-replay-canary-promotion-readiness\n",
        }

        deduped, suppression = _dedupe_create_issue_proposals_with_metadata(
            [proposal],
            existing_issues=[
                issue(
                    537,
                    "Stage cache replay canary from evidence-to-activation ledger (evidence abc123456789)",
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-06-11T08:20:00Z",
                    body="Resolved predecessor.\n\nFingerprint: activation:abc123456789\nNext action: `stage-cache-replay-canary`\n",
                )
            ],
            trusted_author="lutzkuen",
            now=NOW,
        )

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["closed_lifecycle_predecessor"]["number"], 537)
        self.assertEqual(deduped[0]["closed_lifecycle_predecessor_reason"], "same-fingerprint-next-action-progressed")
        self.assertEqual(suppression["fingerprint_match_count"], 0)
        self.assertEqual(suppression["closed_prior_issue_count"], 0)

    def test_saved_research_plan_summary_ranks_advanced_lifecycle_titles(self):
        now = datetime(2026, 6, 17, 4, 30, tzinfo=timezone.utc)
        closed_issues = [
            issue(
                533,
                "Rank next savings milestone from local telemetry evidence gaps",
                ["backlog", "status:ready", "priority:p1"],
                state="CLOSED",
                closed="2026-06-15T21:24:02Z",
            ),
            issue(
                537,
                "Stage cache replay canary from evidence-to-activation ledger (evidence 243c92b5d91f)",
                ["backlog", "status:ready", "cache"],
                state="CLOSED",
                closed="2026-06-15T23:14:38Z",
                body="Fingerprint: activation:243c92b5d91f9149\nNext action: `stage-cache-replay-canary`\n",
            ),
            issue(
                593,
                "Stage Anthropic messages repeated-context crunch cohort with holdout",
                ["backlog", "status:ready", "crunch"],
                state="CLOSED",
                closed="2026-06-17T01:13:35Z",
            ),
            issue(
                595,
                "Rank remaining replay-ready cache cohorts after current canary decision",
                ["backlog", "status:ready", "cache"],
                state="CLOSED",
                closed="2026-06-17T02:13:23Z",
            ),
        ]
        stats_summary = {
            "calls": 3685,
            "cache_hits": 1,
            "cache_hit_rate": 0.00027,
            "evidence_to_activation_next_action_ledger": {
                "schema": "agentflow.evidence_to_activation_next_action_ledger.v1",
                "status": "tracked",
                "summary": {"tracked_entry_count": 1, "closed_issue_seen_count": 1},
                "entries": [
                    {
                        "schema": "agentflow.evidence_to_activation_next_action_ledger_entry.v1",
                        "fingerprint": "activation:243c92b5d91f9149",
                        "lever": "cache",
                        "local_action_family": "cache",
                        "evidence_schema": "agentflow.request_shape_cache_replay_evidence.v1",
                        "cohort_bucket": "openai_responses/responses/chat",
                        "current_status": "holdout",
                        "state": "replay-ready",
                        "next_action": "review-cache-replay-canary-promotion-readiness",
                        "issue_status": "closed-issue-seen",
                        "prior_issue": {
                            "repo": "lutzkuen/agentflow",
                            "number": 595,
                            "title": "Rank remaining replay-ready cache cohorts after current canary decision",
                            "url": "https://github.com/lutzkuen/agentflow/issues/595",
                        },
                        "expected_savings_path": "Move cache replay evidence toward the next local replay decision.",
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    }
                ],
                "privacy": {"metadata_only": True, "aggregate_only": True},
            },
            "request_shape_rollup_candidates": {
                "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
                "status": "candidates-ranked",
                "summary": {
                    "calls": 3685,
                    "ranked_candidate_count": 1,
                    "top_next_action": "keep-active",
                    "top_local_action_family": "crunch",
                },
                "top_candidate": {
                    "local_action_family": "crunch",
                    "next_action": "keep-active",
                    "readiness_state": "measured-active",
                    "candidate_work_classes": ["crunch", "repeated_context"],
                    "provider_surface_bucket": "anthropic/anthropic_messages/messages",
                    "row_count": 155,
                    "sample_count": 155,
                    "projected_savings_usd": 1.027594,
                    "blocker_codes": ["repeated-context-crunch-active-at-max-rollout"],
                },
                "cache_replayability_dry_run": {
                    "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                    "status": "ranked",
                    "summary": {
                        "replay_ready_cohort_count": 7,
                        "remaining_replay_ready_cohort_count": 5,
                        "remaining_replay_ready_rows": 39,
                        "remaining_projected_hits": 34,
                        "remaining_projected_savings_usd": 0.077742,
                    },
                    "cohorts": [
                        {
                            "readiness": "replay-ready",
                            "remaining_replay_ready": True,
                            "provider_family": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "category": "chat",
                            "workflow_phase": "chat",
                            "stream": False,
                            "has_tools": False,
                            "cache_status": "miss",
                            "row_count": 10,
                            "projected_hits": 9,
                            "projected_savings_usd": 0.031711,
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "privacy": {"metadata_only": True, "aggregate_only": True},
            },
            "crunch_savings_signal": {
                "schema": "agentflow.crunch_savings_signal.v1",
                "status": "observed-savings-ranked",
                "calls": 3685,
                "observed": {"crunch_savings_usd": 25.818387, "crunch_tokens_saved": 8606129},
                "top_report": {
                    "report_key": "request_shape_crunch_activation_evidence",
                    "next_action": "keep-active",
                    "duplicate_suppression": {
                        "schema": "agentflow.request_shape_crunch_keep_active_duplicate_suppression.v1",
                        "suppresses_new_activation_issue": True,
                        "suppresses_generic_crunch_activation_issue": True,
                        "reason": "repeated-context-crunch-active-at-max-rollout",
                        "matching_local_policy": "crunch_rules",
                        "target_local_rule_file": "crunch_rules.yaml",
                        "metadata_only": True,
                        "aggregate_only": True,
                    },
                },
                "privacy": {"metadata_only": True, "aggregate_only": True},
            },
        }

        plan = build_research_plan(
            issues=closed_issues,
            stats=stats_summary,
            threshold=3,
            now=now,
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertTrue(
            any(title.startswith("Advance cache next action from evidence-to-activation ledger") for title in titles),
            titles,
        )
        self.assertIn("Stage remaining replay-ready cache cohort for replay-ready on openai/openai_responses/responses", titles)
        self.assertIn("Advance remaining replay-ready cache cohort into local replay evidence", titles)
        self.assertIn("Record request-shape repeated-context crunch keep-active outcome", titles)
        for stale_title in (
            "Rank next savings milestone from local telemetry evidence gaps",
            "Stage cache replay canary from evidence-to-activation ledger (evidence 243c92b5d91f)",
            "Stage Anthropic messages repeated-context crunch cohort with holdout",
            "Rank remaining replay-ready cache cohorts after current canary decision",
            "Stage cache replay canary for replay-ready on openai/openai_responses/responses",
            "Turn replay-ready cache candidate into local replay evidence",
            "Stage request-shape repeated-context crunch canary",
            "Rank crunch savings follow-up for crunch-observed-savings-ranked",
        ):
            self.assertNotIn(stale_title, titles)

        crunch_candidate = next(
            item for item in plan["evidence"]["optimization_candidates"]
            if item.get("lever") == "crunch"
        )
        self.assertEqual(crunch_candidate["issue_generation_status"], "suppressed-active-crunch-keep-active")
        self.assertEqual(
            crunch_candidate["issue_generation_suppression_reason"],
            "repeated-context-crunch-active-at-max-rollout",
        )

        ledger_issue = next(
            item for item in plan["backlog_changes"]["create_issues"]
            if item["title"].startswith("Advance cache next action from evidence-to-activation ledger")
        )
        self.assertEqual(ledger_issue["closed_lifecycle_predecessor"]["number"], 537)
        self.assertIn("Continues closed predecessor: #595", ledger_issue["body"])
        self.assertIn("Top next action: review-cache-replay-canary-promotion-readiness", ledger_issue["body"])

        milestone = plan["evidence"]["next_backlog_milestone"]
        self.assertEqual(milestone["summary"]["recommended_next_issue"]["title"], ledger_issue["title"])
        self.assertEqual(milestone["summary"]["recommended_next_issue"]["implementation_rank"], 1)
        self.assertEqual(milestone["implementation_order"][0]["title"], ledger_issue["title"])

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        self.assertGreaterEqual(ledger["summary"]["closed_issue_seen_count"], 1)
        cache_entry = next(
            entry for entry in ledger["entries"]
            if entry.get("fingerprint") == "activation:243c92b5d91f9149"
        )
        self.assertEqual(cache_entry["prior_issue"]["number"], 595)

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

    def test_cache_replay_ledger_reports_staged_request_shape_canary_progress(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "cache_hits": 0,
                "request_shape_rollup_candidates": {
                    "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "summary": {
                            "replay_ready_cohort_count": 1,
                            "projected_hits": 35,
                            "projected_savings_usd": 0.075373,
                        },
                        "cohorts": [
                            {
                                "readiness": "replay-ready",
                                "row_count": 36,
                                "projected_hits": 35,
                                "projected_savings_usd": 0.075373,
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "candidate_id": "raw-staged-dry-run-cache-candidate",
                            }
                        ],
                    },
                },
                "request_shape_cache_replay_evidence": {
                    "schema": "agentflow.request_shape_cache_replay_evidence.v1",
                    "status": "staged-no-traffic",
                    "reason": "missing-observed-cache-replay-traffic",
                    "next_action": "collect-cache-replay-canary-traffic",
                    "staged_canary_count": 1,
                    "staged_canaries": [
                        {
                            "rank": 1,
                            "policy_id": "local-openai-cache-replay-canary-secret",
                            "rule_id": "raw-cache-replay-rule-secret",
                            "candidate_id": "request-shape-cache-replay-secret",
                            "shape": {
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "text_bucket": "2k_8k_chars",
                                "token_bucket": "500_2k_tokens",
                                "stream": False,
                                "has_tools": False,
                            },
                            "sample_count": 36,
                            "projected_hits": 35,
                            "projected_savings_usd": 0.075373,
                            "canary_fraction": 0.1,
                            "holdout_fraction": 0.1,
                        }
                    ],
                    "summary": {
                        "observed_row_count": 0,
                        "applied_count": 0,
                        "holdout_count": 0,
                        "exact_hit_count": 0,
                        "miss_count": 0,
                        "bypass_count": 0,
                        "unsupported_shape_count": 0,
                        "projected_hits": 35,
                        "observed_hits": 0,
                        "projected_savings_usd": 0.075373,
                        "observed_savings_usd": 0.0,
                    },
                    "stale_evidence": {
                        "stale": False,
                        "reason": "fresh-or-not-yet-observed",
                    },
                    "privacy": {
                        "metadata_only": True,
                        "aggregate_only": True,
                        "raw_prompts_included": False,
                        "request_ids_included": False,
                        "session_ids_included": False,
                        "cache_keys_included": False,
                        "policy_ids_included": False,
                        "rule_ids_included": False,
                        "cohort_ids_included": False,
                    },
                },
            },
            threshold=3,
            now=NOW,
        )

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        cache = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache["state"], "canary-staged")
        self.assertEqual(cache["evidence_source"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache["next_action"], "collect-cache-replay-canary-traffic")
        self.assertEqual(cache["fingerprint_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache["projected_hits"], 35)
        self.assertAlmostEqual(cache["projected_saved_usd"], 0.075373)

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        cache_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "cache")
        self.assertEqual(cache_entry["evidence_schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache_entry["current_status"], "staged")
        self.assertEqual(cache_entry["next_action"], "collect-cache-replay-canary-traffic")
        self.assertEqual(cache_entry["fingerprint_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_entry["fingerprint_evidence_schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(cache_entry["fingerprint_cohort_bucket"], "cache:10_99")
        self.assertEqual(cache_entry["lifecycle_progressed_from_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_entry["projected_hits"], 35)
        self.assertEqual(cache_entry["projected_saved_usd"], 0.075373)

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-staged-dry-run-cache-candidate", rendered)
        self.assertNotIn("local-openai-cache-replay-canary-secret", rendered)
        self.assertNotIn("raw-cache-replay-rule-secret", rendered)
        self.assertNotIn("request-shape-cache-replay-secret", rendered)

    def test_cache_replay_ledger_prefers_request_shape_canary_hit_evidence_over_dry_run(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 50,
                "request_shape_rollup_candidates": {
                    "schema": "agentflow.request_shape_rollup_candidate_signal.v1",
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "summary": {"replay_ready_cohort_count": 1, "projected_hits": 35},
                        "cohorts": [
                            {
                                "readiness": "replay-ready",
                                "row_count": 36,
                                "projected_hits": 35,
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                            }
                        ],
                    },
                },
                "request_shape_cache_replay_evidence": {
                    "schema": "agentflow.request_shape_cache_replay_evidence.v1",
                    "status": "observed",
                    "reason": "cache-replay-canary-evidence-observed",
                    "next_action": "review-cache-replay-canary-promotion-readiness",
                    "staged_canary_count": 1,
                    "staged_canaries": [
                        {
                            "shape": {
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "chat",
                                "workflow_phase": "chat",
                                "stream": False,
                                "has_tools": False,
                            },
                            "sample_count": 36,
                            "projected_hits": 35,
                            "projected_savings_usd": 0.075373,
                        }
                    ],
                    "summary": {
                        "observed_row_count": 4,
                        "applied_count": 2,
                        "holdout_count": 1,
                        "exact_hit_count": 1,
                        "miss_count": 1,
                        "bypass_count": 1,
                        "unsupported_shape_count": 0,
                        "projected_hits": 35,
                        "observed_hits": 1,
                        "projected_savings_usd": 0.075373,
                        "observed_savings_usd": 0.012345,
                    },
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
        self.assertEqual(cache["evidence_source"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache["next_action"], "review-cache-replay-canary-promotion-readiness")
        self.assertEqual(cache["applied_count"], 2)
        self.assertEqual(cache["holdout_count"], 1)
        self.assertEqual(cache["actual_hits"], 1)
        self.assertAlmostEqual(cache["actual_saved_cost_usd"], 0.012345)

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        cache_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "cache")
        self.assertEqual(cache_entry["current_status"], "holdout")
        self.assertEqual(cache_entry["next_action"], "review-cache-replay-canary-promotion-readiness")
        self.assertEqual(cache_entry["fingerprint_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_entry["fingerprint_evidence_schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(cache_entry["fingerprint_cohort_bucket"], "cache:10_99")
        self.assertEqual(cache_entry["actual_hits"], 1)
        self.assertAlmostEqual(cache_entry["actual_saved_cost_usd"], 0.012345)

    def test_cache_replay_ledger_advances_review_action_from_policy_decision(self):
        evidence = {
            "schema": "agentflow.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "reason": "cache-replay-canary-evidence-observed",
            "next_action": "review-cache-replay-canary-promotion-readiness",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "rank": 1,
                    "rule_id": "raw-cache-replay-rule-secret",
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "sample_count": 36,
                    "projected_hits": 35,
                    "projected_savings_usd": 0.075373,
                }
            ],
            "summary": {
                "observed_row_count": 43,
                "applied_count": 25,
                "holdout_count": 18,
                "exact_hit_count": 0,
                "miss_count": 25,
                "bypass_count": 0,
                "unsupported_shape_count": 0,
                "projected_hits": 35,
                "observed_hits": 0,
                "projected_savings_usd": 0.075373,
                "observed_savings_usd": 0.0,
                "top_applied_miss_blocker": "first-seen-cache-warmup",
            },
            "applied_miss_blocker_breakdown": [{"value": "first-seen-cache-warmup", "count": 25}],
            "stale_evidence": {"stale": False, "age_hours": 0.2},
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
                "individual_candidate_ids_included": False,
            },
        }
        policy_decision = {
            "schema": "agentflow.request_shape_cache_replay_policy_decision.v1",
            "status": "decided",
            "decision": "keep-staged",
            "promotion_decision": "keep-staged-warmup",
            "promotion_readiness": "keep-staged-warmup",
            "reason": "first-seen-cache-warmup",
            "promotion_blocker": "first-seen-cache-warmup",
            "observed_hit_blocker": "first-seen-cache-warmup",
            "reason_codes": [
                "missing-observed-cache-hits",
                "missing-observed-cache-savings",
                "applied-cache-replay-miss-observed",
                "first-seen-cache-warmup",
                "applied-miss:first-seen-cache-warmup",
            ],
            "next_action": "keep-cache-replay-canary-staged",
            "summary": {
                "decision": "keep-staged",
                "promotion_decision": "keep-staged-warmup",
                "promotion_readiness": "keep-staged-warmup",
                "next_action": "keep-cache-replay-canary-staged",
                "promotion_allowed": False,
                "rollback_required": False,
                "keep_staged_warmup": True,
                "keep_staged": True,
                "keep_blocked": False,
                "staged_canary_count": 1,
                "observed_row_count": 43,
                "applied_count": 25,
                "holdout_count": 18,
                "exact_hit_count": 0,
                "miss_count": 25,
                "bypass_count": 0,
                "projected_hits": 35,
                "observed_hits": 0,
                "projected_savings_usd": 0.075373,
                "observed_savings_usd": 0.0,
                "top_applied_miss_blocker": "first-seen-cache-warmup",
                "promotion_blocker": "first-seen-cache-warmup",
                "observed_hit_blocker": "first-seen-cache-warmup",
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
            },
            "top_decision": {
                "decision_id": "cache-replay-policy-decision:public",
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
                "applied_miss_blocker_breakdown": [{"value": "first-seen-cache-warmup", "count": 25}],
                "local_policy_patch": None,
            },
            "source_evidence": {
                "schema": evidence["schema"],
                "status": evidence["status"],
                "summary": evidence["summary"],
                "applied_miss_blocker_breakdown": evidence["applied_miss_blocker_breakdown"],
                "privacy": evidence["privacy"],
            },
            "privacy": {
                "metadata_only": True,
                "aggregate_only": True,
                "raw_prompts_included": False,
                "provider_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "cache_keys_included": False,
                "individual_candidate_ids_included": False,
            },
        }
        plan = build_research_plan(
            issues=[
                issue(
                    537,
                    "Stage cache replay canary from evidence-to-activation ledger (evidence 243c92b5d91f)",
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-06-15T23:14:38Z",
                    body="Fingerprint: activation:243c92b5d91f9149\nNext action: `stage-cache-replay-canary`\n",
                ),
                issue(
                    622,
                    "Turn tools-present cache candidate into local replay evidence",
                    ["backlog", "status:ready", "cache"],
                    state="CLOSED",
                    closed="2026-06-17T12:00:00Z",
                    body="Advanced cache replay warmup for fingerprint activation:243c92b5d91f9149.\n",
                ),
            ],
            stats={
                "calls": 50,
                "request_shape_cache_replay_evidence": evidence,
                "request_shape_cache_replay_policy_decision": policy_decision,
            },
            threshold=3,
            now=NOW,
        )

        stats_summary = plan["evidence"]["stats_summary"]
        self.assertEqual(
            stats_summary["request_shape_cache_replay_policy_decision"]["schema"],
            "agentflow.request_shape_cache_replay_policy_decision.v1",
        )
        loop = stats_summary["evidence_to_activation_loop"]
        cache = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache["evidence_source"], "agentflow.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(cache["source_evidence_schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache["state"], "replay-ready")
        self.assertEqual(cache["next_action"], "keep-cache-replay-canary-staged")
        self.assertEqual(cache["policy_decision"], "keep-staged")
        self.assertEqual(cache["promotion_decision"], "keep-staged-warmup")
        self.assertEqual(cache["promotion_blocker"], "first-seen-cache-warmup")
        self.assertEqual(cache["observed_hit_blocker"], "first-seen-cache-warmup")
        self.assertIn("first-seen-cache-warmup", cache["blocker_codes"])
        self.assertEqual(cache["applied_count"], 25)
        self.assertEqual(cache["holdout_count"], 18)
        self.assertEqual(cache["actual_hits"], 0)
        self.assertEqual(cache["projected_hits"], 35)

        ledger = stats_summary["evidence_to_activation_next_action_ledger"]
        cache_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "cache")
        self.assertEqual(cache_entry["evidence_schema"], "agentflow.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(cache_entry["source_evidence_schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(cache_entry["current_status"], "holdout")
        self.assertEqual(cache_entry["next_action"], "keep-cache-replay-canary-staged")
        self.assertEqual(cache_entry["fingerprint_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_entry["fingerprint_evidence_schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(cache_entry["fingerprint_cohort_bucket"], "cache:10_99")
        self.assertEqual(cache_entry["fingerprint"], "activation:243c92b5d91f9149")
        self.assertEqual(cache_entry["policy_decision"], "keep-staged")
        self.assertEqual(cache_entry["promotion_readiness"], "keep-staged-warmup")
        self.assertEqual(cache_entry["issue_status"], "closed-issue-seen")
        self.assertIn("first-seen-cache-warmup", cache_entry["blocker_codes"])
        self.assertEqual(cache_entry["top_miss_reason"], "first-seen-cache-warmup")
        self.assertEqual(cache_entry["promotion_blocker"], "first-seen-cache-warmup")
        self.assertEqual(cache_entry["observed_hit_blocker"], "first-seen-cache-warmup")
        self.assertEqual(cache_entry["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(
            cache_entry["duplicate_suppression"]["schema"],
            "agentflow.request_shape_cache_replay_warmup_carry_forward_duplicate_suppression.v1",
        )
        self.assertTrue(cache_entry["duplicate_suppression"]["suppresses_new_cache_replay_stage_issue"])
        self.assertTrue(cache_entry["duplicate_suppression"]["suppresses_closed_stage_replay_predecessor_titles"])
        self.assertIn(
            "Stage cache replay canary from evidence-to-activation ledger",
            cache_entry["duplicate_suppression"]["suppressed_predecessor_title_families"],
        )

        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertIn(
            "Record cache replay canary warmup carry-forward in evidence ledger (evidence 243c92b5d91f)",
            titles,
        )
        self.assertNotIn(
            "Stage cache replay canary from evidence-to-activation ledger (evidence 243c92b5d91f)",
            titles,
        )
        ledger_issue = next(
            item for item in plan["backlog_changes"]["create_issues"]
            if item["title"].startswith("Record cache replay canary warmup carry-forward")
        )
        self.assertIn("Promotion readiness: keep-staged-warmup", ledger_issue["body"])
        self.assertIn("Warmup miss blocker breakdown:", ledger_issue["body"])
        self.assertIn("Duplicate suppression:", ledger_issue["body"])

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-cache-replay-rule-secret", rendered)
        self.assertNotIn("raw-request-secret", rendered)
        self.assertNotIn("raw-session-secret", rendered)

    def test_cache_replay_retirement_policy_decision_is_superseded_in_ledger(self):
        evidence = {
            "schema": "agentflow.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                    },
                    "sample_count": 36,
                    "projected_hits": 35,
                    "projected_savings_usd": 0.075373,
                }
            ],
            "summary": {
                "observed_row_count": 75,
                "applied_count": 28,
                "holdout_count": 47,
                "miss_count": 28,
                "observed_hits": 0,
                "projected_hits": 35,
                "observed_savings_usd": 0.0,
                "projected_savings_usd": 0.075373,
            },
            "applied_miss_blocker_breakdown": [{"value": "first-seen-cache-warmup", "count": 28}],
        }
        policy_decision = {
            "schema": "agentflow.request_shape_cache_replay_policy_decision.v1",
            "decision": "retire-staged-no-repeat",
            "promotion_decision": "retire-staged-no-repeat",
            "promotion_readiness": "retire-staged-no-repeat",
            "reason": "repeat-window-elapsed-no-live-repeat",
            "reason_codes": [
                "retire-staged-no-repeat",
                "repeat-window-elapsed-no-live-repeat",
                "first-seen-cache-warmup",
            ],
            "next_action": "retire-cache-replay-canary-no-repeat",
            "duplicate_suppression": {
                "schema": "agentflow.request_shape_cache_replay_policy_decision_duplicate_suppression.v1",
                "reason": "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
                "suppresses_generic_replay_ready_issue": True,
                "suppresses_new_cache_replay_stage_issue": True,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "summary": {
                "decision": "retire-staged-no-repeat",
                "promotion_decision": "retire-staged-no-repeat",
                "promotion_readiness": "retire-staged-no-repeat",
                "next_action": "retire-cache-replay-canary-no-repeat",
                "staged_canary_count": 1,
                "observed_row_count": 75,
                "applied_count": 28,
                "holdout_count": 47,
                "miss_count": 28,
                "observed_hits": 0,
                "projected_hits": 35,
                "observed_savings_usd": 0.0,
                "projected_savings_usd": 0.075373,
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
            },
            "top_decision": {
                "decision_id": "cache-replay-policy-decision:public",
                "target_local_rule_file": "cache_rules.yaml",
                "target_local_policy_section": "cache.pattern_rules",
            },
            "source_evidence": evidence,
        }

        ledger = build_evidence_to_activation_next_action_ledger(
            {
                "request_shape_cache_replay_evidence": evidence,
                "request_shape_cache_replay_policy_decision": policy_decision,
            }
        )

        cache_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "cache")
        self.assertEqual(cache_entry["state"], "retired-no-repeat")
        self.assertEqual(cache_entry["current_status"], "superseded")
        self.assertEqual(cache_entry["issue_worthy_status"], "review")
        self.assertEqual(cache_entry["next_action"], "retire-cache-replay-canary-no-repeat")
        self.assertEqual(
            cache_entry["duplicate_suppression"]["reason"],
            "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
        )
        self.assertTrue(cache_entry["duplicate_suppression"]["suppresses_new_cache_replay_stage_issue"])

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

    def test_crunch_candidate_records_missing_aggregate_measurement(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "cache_hit_rate": 0.0,
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["status"], "missing-crunch-measurement")
        self.assertEqual(signal["top_report"]["report_key"], "aggregate_crunch_measurement")
        self.assertEqual(signal["top_report"]["status"], "missing-measurement")
        self.assertEqual(signal["top_report"]["next_action"], "emit-crunch-aggregate-measurement")
        self.assertEqual(signal["top_report"]["no_op_reason"], "missing-crunch-aggregate-measurement")
        self.assertIn("crunched-count", signal["missing_measurements"])
        self.assertIn("crunch-token-or-char-savings", signal["missing_measurements"])
        self.assertIn("crunch-savings-usd", signal["missing_measurements"])
        self.assertIn("avg-crunch-ratio", signal["missing_measurements"])
        self.assertTrue(signal["privacy"]["metadata_only"])
        self.assertTrue(signal["privacy"]["aggregate_only"])
        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertFalse(signal["privacy"]["request_ids_included"])
        self.assertFalse(signal["privacy"]["session_ids_included"])

        crunch_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "crunch")
        self.assertEqual(crunch_candidate["blocker"], "missing-crunch-savings-signal")
        self.assertEqual(crunch_candidate["projected_savings_signal"]["status"], "missing-crunch-measurement")

        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        crunch_stage = next(stage for stage in loop["levers"] if stage["lever"] == "crunch")
        self.assertEqual(crunch_stage["state"], "missing-evidence")
        self.assertEqual(crunch_stage["next_action"], "emit-crunch-aggregate-measurement")
        self.assertIn("crunched-count", crunch_stage["blocker_codes"])

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
                            "report_key": "request_shape_crunch_opportunity",
                            "evidence_schema": "agentflow.request_shape_crunch_opportunity_dry_run.v1",
                            "projected_saved_chars": 48000,
                            "projected_saved_tokens": 12000,
                            "projected_saved_usd": 0.036,
                            "canary_already_staged": False,
                            "no_op_reason": None,
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
        self.assertEqual(signal["top_report"]["savings_status"], "projected-savings-ranked")
        self.assertFalse(signal["top_report"]["canary_already_staged"])
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
                            "no_op_reason": "matching-repeated-context-crunch-canary-already-staged",
                            "canary_already_staged": True,
                            "duplicate_suppression": {
                                "schema": "agentflow.request_shape_crunch_follow_up_duplicate_suppression.v1",
                                "suppresses_new_stage_action": True,
                                "reason": "matching-repeated-context-crunch-canary-already-staged",
                                "matching_local_policy": "crunch_rules",
                                "metadata_only": True,
                                "aggregate_only": True,
                            },
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
        self.assertEqual(signal["top_report"]["no_op_reason"], "matching-repeated-context-crunch-canary-already-staged")
        self.assertTrue(signal["top_report"]["canary_already_staged"])
        self.assertTrue(signal["top_report"]["duplicate_suppression"]["suppresses_new_stage_action"])
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
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "local_action_family": "crunch",
                                "next_action": "stage-repeated-context-crunch-canary",
                                "readiness_state": "activation-ready",
                                "candidate_work_classes": ["crunch", "repeated_context"],
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "category": "tool-result",
                                "workflow_phase": "thinking",
                                "stream": True,
                                "has_tools": True,
                                "text_bucket": "gte_128k_chars",
                                "token_bucket": "lt_500_tokens",
                                "sample_count": 12,
                                "row_count": 12,
                                "projected_saved_tokens": 6000,
                                "projected_savings_usd": 0.024,
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
        self.assertEqual(signal["missing_measurements"], [])
        loop = plan["evidence"]["stats_summary"]["evidence_to_activation_loop"]
        request_shape_stage = next(item for item in loop["levers"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_stage["state"], "measured-savings")
        self.assertEqual(request_shape_stage["next_action"], "widen-repeated-context-crunch-canary")
        self.assertEqual(request_shape_stage["blocker_codes"], [])
        self.assertEqual(
            request_shape_stage["activation_follow_up_evidence_schema"],
            "agentflow.request_shape_crunch_canary_impact.v1",
        )
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        request_shape_entry = next(item for item in ledger["entries"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_entry["current_status"], "holdout")
        self.assertEqual(request_shape_entry["applied_count"], 6)
        self.assertEqual(request_shape_entry["holdout_count"], 6)
        self.assertEqual(request_shape_entry.get("blocker_codes", []), [])
        rendered = json.dumps(plan)
        self.assertNotIn("raw-policy-secret", rendered)
        self.assertNotIn("raw-session-id-secret", rendered)
        self.assertNotIn("cache-secret", rendered)
        self.assertNotIn("/home/lutz/private/shape_secret.py", rendered)

    def test_crunch_candidate_prefers_request_shape_crunch_policy_decision(self):
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
                    "crunch_policy_decision": {
                        "schema": "agentflow.request_shape_crunch_policy_decision.v1",
                        "status": "decided",
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": "request-shape-crunch-policy-decision:public",
                        "summary": {
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "decision_count": 1,
                            "applied_count": 6,
                            "holdout_count": 6,
                            "observed_saved_tokens": 6000,
                            "observed_saved_usd": 0.024,
                            "error_rate_delta": 0.0,
                            "retry_rate_delta": 0.0,
                            "fallback_rate_delta": 0.0,
                            "safety_stop_state": "none",
                        },
                        "top_decision": {
                            "decision": "widen",
                            "policy_id": "raw-policy-secret must not leak",
                            "cohort_id": "cohort-public",
                            "metrics": {
                                "applied_count": 6,
                                "holdout_count": 6,
                                "observed_saved_tokens": 6000,
                                "observed_saved_usd": 0.024,
                            },
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "crunch_canary_impact": {
                        "schema": "agentflow.request_shape_crunch_canary_impact.v1",
                        "status": "widen-ready",
                        "summary": {
                            "candidate_count": 1,
                            "observed_canary_metadata_row_count": 12,
                            "applied_count": 6,
                            "holdout_count": 6,
                            "saved_tokens": 6000,
                            "saved_usd": 0.024,
                            "next_action": "widen",
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "local_action_family": "crunch",
                                "next_action": "stage-repeated-context-crunch-canary",
                                "readiness_state": "activation-ready",
                                "candidate_work_classes": ["crunch", "repeated_context"],
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "row_count": 12,
                                "projected_saved_tokens": 6000,
                                "projected_savings_usd": 0.024,
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
        signal = stats_summary["crunch_savings_signal"]
        self.assertEqual(signal["status"], "policy-decision-emitted")
        self.assertEqual(signal["top_report"]["report_key"], "request_shape_crunch_policy_decision")
        self.assertEqual(signal["top_report"]["schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(signal["top_report"]["decision"], "widen")
        self.assertEqual(signal["top_report"]["next_action"], "widen")
        self.assertEqual(signal["top_report"]["projected_saved_tokens"], 6000)
        self.assertEqual(signal["missing_measurements"], [])
        shape_signal = stats_summary["request_shape_rollup_candidates"]
        self.assertEqual(shape_signal["crunch_policy_decision"]["decision"], "widen")
        loop = stats_summary["evidence_to_activation_loop"]
        request_shape_stage = next(item for item in loop["levers"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_stage["state"], "measured-savings")
        self.assertEqual(request_shape_stage["next_action"], "widen")
        self.assertEqual(
            request_shape_stage["activation_follow_up_evidence_schema"],
            "agentflow.request_shape_crunch_policy_decision.v1",
        )
        rendered = json.dumps(plan)
        self.assertNotIn("raw-policy-secret", rendered)

    def test_issue_555_active_crunch_rule_coverage_populates_observed_savings_signal(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2483,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
                "crunched_count": 0,
                "active_crunch_rule_coverage": {
                    "schema": "agentflow.active_crunch_rule_coverage.v1",
                    "status": "observed",
                    "rule_file": "crunch_rules.yaml",
                    "target_local_policy": "crunch_rules",
                    "target_local_policy_section": "crunch.rules",
                    "summary": {
                        "active_rule_count": 1,
                        "widened_rule_count": 1,
                        "applied_count": 7,
                        "holdout_count": 8,
                        "skipped_count": 59,
                        "blocked_count": 0,
                        "observed_saved_chars": 42952,
                        "observed_saved_tokens": 10738,
                        "observed_saved_usd": 0.032214,
                        "policy_source": "local-manual",
                        "target_local_rule_file": "crunch_rules.yaml",
                        "target_local_policy_section": "crunch.rules",
                        "next_action": "rank-observed-crunch-family-follow-up",
                    },
                    "rules": [
                        {
                            "policy_source": "local-manual",
                            "decision": "widen",
                            "applied_count": 7,
                            "holdout_count": 8,
                            "observed_saved_tokens": 10738,
                            "observed_saved_usd": 0.032214,
                            "metadata_only": True,
                            "aggregate_only": True,
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
        self.assertEqual(signal["status"], "observed-savings-ranked")
        self.assertEqual(signal["observed"]["source"], "active_crunch_rule_coverage")
        self.assertEqual(signal["observed"]["crunched_count"], 7)
        self.assertEqual(signal["observed"]["crunch_tokens_saved"], 10738)
        self.assertEqual(signal["observed"]["crunch_chars_saved"], 42952)
        self.assertEqual(signal["observed"]["crunch_savings_usd"], 0.032214)
        self.assertEqual(signal["top_report"]["report_key"], "active_crunch_rule_coverage")
        self.assertEqual(signal["top_report"]["applied_count"], 7)
        self.assertEqual(signal["top_report"]["holdout_count"], 8)
        self.assertEqual(signal["top_report"]["skipped_count"], 59)
        self.assertEqual(signal["top_report"]["blocked_count"], 0)
        self.assertEqual(signal["top_report"]["savings_status"], "active-rule-coverage-observed")
        self.assertEqual(signal["missing_measurements"], [])
        self.assertTrue(signal["privacy"]["metadata_only"])
        self.assertTrue(signal["privacy"]["aggregate_only"])
        rendered = json.dumps(plan)
        self.assertNotIn("/home/lutz/private", rendered)
        self.assertNotIn("secret raw prompt", rendered.lower())

    def test_issue_555_zero_active_crunch_rule_coverage_has_explicit_no_applied_reason(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 20,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
                "crunched_count": 0,
                "active_crunch_rule_coverage": {
                    "schema": "agentflow.active_crunch_rule_coverage.v1",
                    "status": "no-applied-coverage",
                    "rule_file": "crunch_rules.yaml",
                    "summary": {
                        "active_rule_count": 1,
                        "widened_rule_count": 1,
                        "applied_count": 0,
                        "holdout_count": 4,
                        "skipped_count": 16,
                        "blocked_count": 0,
                        "observed_saved_chars": 0,
                        "observed_saved_tokens": 0,
                        "observed_saved_usd": 0.0,
                        "no_op_reason": "no-applied-coverage",
                        "next_action": "inspect-active-crunch-rule-coverage",
                    },
                    "missing_measurements": ["no-applied-coverage"],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["crunch_savings_signal"]
        self.assertEqual(signal["status"], "non-positive-projection")
        self.assertEqual(signal["observed"]["crunched_count"], 0)
        self.assertEqual(signal["observed"]["crunch_tokens_saved"], 0)
        self.assertEqual(signal["top_report"]["report_key"], "active_crunch_rule_coverage")
        self.assertEqual(signal["top_report"]["no_op_reason"], "no-applied-coverage")
        self.assertEqual(signal["top_report"]["next_action"], "inspect-active-crunch-rule-coverage")
        self.assertIn("no-applied-coverage", signal["missing_measurements"])
        self.assertTrue(signal["privacy"]["metadata_only"])
        self.assertTrue(signal["privacy"]["aggregate_only"])

    def test_issue_558_active_crunch_rule_coverage_advances_request_shape_ledger(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 3312,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
                "crunched_count": 0,
                "active_crunch_rule_coverage": {
                    "schema": "agentflow.active_crunch_rule_coverage.v1",
                    "status": "observed",
                    "rule_file": "crunch_rules.yaml",
                    "target_local_policy": "crunch_rules",
                    "target_local_policy_section": "crunch.rules",
                    "summary": {
                        "active_rule_count": 1,
                        "widened_rule_count": 1,
                        "applied_count": 26,
                        "holdout_count": 17,
                        "skipped_count": 136,
                        "blocked_count": 0,
                        "observed_saved_chars": 6590776,
                        "observed_saved_tokens": 1647683,
                        "observed_saved_usd": 4.943049,
                        "policy_source": "local-manual",
                        "target_local_rule_file": "crunch_rules.yaml",
                        "target_local_policy_section": "crunch.rules",
                        "next_action": "rank-observed-crunch-family-follow-up",
                    },
                    "rules": [
                        {
                            "rank": 1,
                            "rule_id": "request-shape-crunch-canary:public-rule",
                            "rule_ref": "request-shape-crunch-canary:public-rule",
                            "policy_source": "local-manual",
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "decision_id": "request-shape-crunch-policy-decision:public",
                            "source_evidence_schema": "agentflow.request_shape_crunch_policy_decision.v1",
                            "applied_count": 26,
                            "holdout_count": 17,
                            "observed_saved_chars": 6590776,
                            "observed_saved_tokens": 1647683,
                            "observed_saved_usd": 4.943049,
                            "metadata_only": True,
                            "aggregate_only": True,
                        }
                    ],
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                },
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 1000, "rollup_count": 40},
                    "crunch_policy_decision": {
                        "schema": "agentflow.request_shape_crunch_policy_decision.v1",
                        "status": "decided",
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": "request-shape-crunch-policy-decision:public",
                        "summary": {
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "decision_count": 1,
                            "applied_count": 26,
                            "holdout_count": 17,
                            "observed_saved_tokens": 1647683,
                            "observed_saved_usd": 4.943049,
                            "safety_stop_state": "none",
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "stage-repeated-context-crunch-canary",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "local_action_family": "crunch",
                                "next_action": "stage-repeated-context-crunch-canary",
                                "readiness_state": "activation-ready",
                                "candidate_work_classes": ["crunch", "repeated_context"],
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "messages",
                                "row_count": 329,
                                "sample_count": 329,
                                "projected_saved_tokens": 1647683,
                                "projected_savings_usd": 4.943049,
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
        self.assertEqual(signal["status"], "observed-savings-ranked")
        self.assertEqual(signal["top_report"]["report_key"], "active_crunch_rule_coverage")
        self.assertEqual(signal["top_report"]["active_rule_ref"], "request-shape-crunch-canary:public-rule")
        self.assertEqual(signal["top_report"]["active_rule_decision_id"], "request-shape-crunch-policy-decision:public")
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        entry = next(item for item in ledger["entries"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(entry["local_action_family"], "crunch")
        self.assertEqual(entry["state"], "measured-active")
        self.assertEqual(entry["current_status"], "applied")
        self.assertEqual(entry["next_action"], "rank-observed-crunch-family-follow-up")
        self.assertEqual(entry["activation_follow_up_evidence_schema"], "agentflow.active_crunch_rule_coverage.v1")
        self.assertEqual(entry["applied_count"], 26)
        self.assertEqual(entry["holdout_count"], 17)
        self.assertEqual(entry["active_rule_count"], 1)
        self.assertEqual(entry["widened_rule_count"], 1)
        self.assertEqual(entry["active_rule_ref"], "request-shape-crunch-canary:public-rule")
        self.assertEqual(entry["active_rule_source"], "local-manual")
        self.assertEqual(entry["active_rule_decision_id"], "request-shape-crunch-policy-decision:public")
        self.assertEqual(entry["active_rule_source_evidence_schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(entry["projected_saved_usd"], 4.943049)
        self.assertTrue(ledger["privacy"]["metadata_only"])
        self.assertTrue(ledger["privacy"]["aggregate_only"])
        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertFalse(any(title.startswith("Apply measured request-shape crunch widening") for title in titles), titles)
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-request-secret", rendered)

    def test_issue_564_crunch_activation_evidence_advances_current_request_shape_rules(self):
        decision_id = "request-shape-crunch-policy-decision:9db327d1abdec766"
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 3467,
                "today_crunch_savings_usd": 0.0,
                "crunch_savings_usd": 0.0,
                "crunch_tokens_saved": 0,
                "crunch_chars_saved": 0,
                "crunched_count": 0,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 1000, "rollup_count": 33},
                    "crunch_activation_evidence": {
                        "schema": "agentflow.request_shape_crunch_activation_evidence.v1",
                        "status": "active-rule-evidence-observed",
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "next_action": "promote-full-repeated-context-crunch-rule",
                        "summary": {
                            "active_rule_count": 1,
                            "matching_active_rule_count": 1,
                            "widened_rule_count": 1,
                            "matching_widened_rule_count": 1,
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "decision_id": decision_id,
                            "applied_count": 107,
                            "holdout_count": 40,
                            "skipped_count": 280,
                            "blocked_count": 0,
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
                            "next_action": "promote-full-repeated-context-crunch-rule",
                        },
                        "rules": [
                            {
                                "rank": 1,
                                "rule_ref": "request-shape-crunch-canary:public-rule",
                                "policy_source": "local-manual",
                                "decision": "widen",
                                "decision_id": decision_id,
                                "source_evidence_schema": "agentflow.request_shape_crunch_policy_decision.v1",
                                "metadata_only": True,
                                "aggregate_only": True,
                            }
                        ],
                        "duplicate_suppression": {
                            "schema": "agentflow.request_shape_crunch_keep_active_duplicate_suppression.v1",
                            "suppresses_new_activation_issue": True,
                            "suppresses_generic_crunch_activation_issue": True,
                            "reason": "repeated-context-crunch-active-at-max-rollout",
                            "fingerprint": "activation:public",
                            "matching_local_policy": "crunch_rules",
                            "target_local_rule_file": "crunch_rules.yaml",
                            "target_local_policy_section": "crunch.rules",
                            "metadata_only": True,
                            "aggregate_only": True,
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "crunch_policy_decision": {
                        "schema": "agentflow.request_shape_crunch_policy_decision.v1",
                        "status": "decided",
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "summary": {
                            "decision": "widen",
                            "graduation_decision": "widen",
                            "decision_id": decision_id,
                            "applied_count": 107,
                            "holdout_count": 40,
                            "observed_saved_tokens": 8606129,
                            "observed_saved_usd": 25.818387,
                            "safety_stop_state": "none",
                        },
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    "follow_up_candidates": {
                        "schema": "agentflow.request_shape_follow_up_candidates.v1",
                        "status": "candidates-ranked",
                        "summary": {
                            "ranked_candidate_count": 1,
                            "top_next_action": "measure-repeated-context-crunch-canary-impact",
                            "top_local_action_family": "crunch",
                        },
                        "blocker_cohorts": [
                            {
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
                                "local_action_family": "crunch",
                                "next_action": "measure-repeated-context-crunch-canary-impact",
                                "readiness_state": "measurement-required",
                                "candidate_work_classes": ["crunch", "repeated_context"],
                                "provider_family": "anthropic",
                                "source_surface": "anthropic_messages",
                                "endpoint": "unknown",
                                "row_count": 363,
                                "sample_count": 363,
                                "projected_saved_tokens": 1655076,
                                "projected_savings_usd": 4.965228,
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
        signal = stats_summary["crunch_savings_signal"]
        self.assertEqual(signal["status"], "observed-savings-ranked")
        self.assertEqual(signal["observed"]["source"], "request_shape_crunch_activation_evidence")
        self.assertEqual(signal["top_report"]["report_key"], "request_shape_crunch_activation_evidence")
        self.assertEqual(signal["top_report"]["decision_id"], decision_id)
        self.assertEqual(signal["top_report"]["applied_count"], 107)
        self.assertEqual(signal["top_report"]["holdout_count"], 40)
        self.assertEqual(signal["top_report"]["projected_saved_tokens"], 8606129)
        self.assertEqual(signal["top_report"]["active_rule_count"], 1)
        self.assertEqual(signal["top_report"]["widened_rule_count"], 1)
        self.assertEqual(signal["top_report"]["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(signal["top_report"]["post_widening_next_action"], "keep-active")
        self.assertEqual(signal["top_report"]["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(signal["top_report"]["post_max_rollout_decision"], "promote-full")
        self.assertEqual(signal["top_report"]["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(signal["top_report"]["post_max_rollout_promotion_allowed"])
        self.assertEqual(signal["top_report"]["missing_measurements"], [])
        self.assertTrue(signal["top_report"]["duplicate_suppression"]["suppresses_new_activation_issue"])
        local_outcome_summary = stats_summary["local_activation_outcome_summary"]
        self.assertEqual(local_outcome_summary["schema"], "agentflow.local_activation_outcome_summary.v1")
        self.assertEqual(local_outcome_summary["status"], "tracked")
        self.assertTrue(local_outcome_summary["read_only"])
        self.assertFalse(local_outcome_summary["provider_calls_made"])
        self.assertFalse(local_outcome_summary["managed_server_calls_made"])
        self.assertEqual(local_outcome_summary["summary"]["policy_decision_families"], ["crunch"])
        keep_active_outcome = local_outcome_summary["outcome_summaries"][0]
        self.assertEqual(keep_active_outcome["source_evidence_schema"], "agentflow.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(keep_active_outcome["source_decision_id"], decision_id)
        self.assertEqual(keep_active_outcome["active_rule_ref"], "request-shape-crunch-canary:public-rule")
        self.assertEqual(keep_active_outcome["active_rule_source"], "local-manual")
        self.assertEqual(keep_active_outcome["active_rule_source_evidence_schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertEqual(keep_active_outcome["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(keep_active_outcome["target_local_policy_section"], "crunch.rules")
        self.assertEqual(keep_active_outcome["outcome"], "promote-full")
        self.assertEqual(keep_active_outcome["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(keep_active_outcome["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(keep_active_outcome["post_widening_next_action"], "keep-active")
        self.assertEqual(keep_active_outcome["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(keep_active_outcome["post_max_rollout_decision"], "promote-full")
        self.assertEqual(keep_active_outcome["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(keep_active_outcome["post_max_rollout_promotion_allowed"])
        self.assertEqual(keep_active_outcome["applied_count"], 107)
        self.assertEqual(keep_active_outcome["holdout_count"], 40)
        self.assertEqual(keep_active_outcome["coverage"]["applied_count"], 107)
        self.assertEqual(keep_active_outcome["coverage"]["holdout_count"], 40)
        self.assertEqual(keep_active_outcome["coverage"]["safety_stop_count"], 0)
        self.assertTrue(keep_active_outcome["coverage"]["metadata_only"])
        self.assertTrue(keep_active_outcome["coverage"]["aggregate_only"])
        self.assertTrue(keep_active_outcome["duplicate_suppression"]["suppresses_new_activation_issue"])
        self.assertTrue(keep_active_outcome["duplicate_suppression"]["suppresses_generic_crunch_activation_issue"])
        self.assertEqual(
            keep_active_outcome["duplicate_suppression"]["reason"],
            "repeated-context-crunch-active-at-max-rollout",
        )
        self.assertFalse(local_outcome_summary["privacy"]["raw_prompts_included"])
        self.assertFalse(local_outcome_summary["privacy"]["provider_bodies_included"])
        self.assertFalse(local_outcome_summary["privacy"]["request_ids_included"])
        self.assertFalse(local_outcome_summary["privacy"]["session_ids_included"])
        self.assertFalse(local_outcome_summary["privacy"]["cache_keys_included"])
        self.assertFalse(local_outcome_summary["privacy"]["absolute_paths_included"])
        loop = stats_summary["evidence_to_activation_loop"]
        crunch_stage = next(item for item in loop["levers"] if item["lever"] == "crunch")
        self.assertEqual(crunch_stage["state"], "measured-active")
        self.assertEqual(crunch_stage["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(crunch_stage["activation_follow_up_evidence_schema"], "agentflow.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(crunch_stage["applied_count"], 107)
        self.assertEqual(crunch_stage["holdout_count"], 40)
        self.assertEqual(crunch_stage["safety_stop_count"], 0)
        self.assertEqual(crunch_stage["fallback_count"], 0)
        self.assertEqual(crunch_stage["error_rate_delta"], 0.0)
        self.assertEqual(crunch_stage["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(crunch_stage["post_widening_next_action"], "keep-active")
        self.assertEqual(crunch_stage["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(crunch_stage["post_max_rollout_decision"], "promote-full")
        self.assertEqual(crunch_stage["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(crunch_stage["duplicate_suppression"]["suppresses_generic_crunch_activation_issue"])
        request_shape_stage = next(item for item in loop["levers"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_stage["state"], "measured-active")
        self.assertEqual(request_shape_stage["active_rule_count"], 1)
        self.assertEqual(request_shape_stage["widened_rule_count"], 1)
        self.assertEqual(request_shape_stage["applied_count"], 107)
        self.assertEqual(request_shape_stage["holdout_count"], 40)
        self.assertEqual(request_shape_stage["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(request_shape_stage["post_widening_next_action"], "keep-active")
        self.assertEqual(request_shape_stage["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(request_shape_stage["post_max_rollout_decision"], "promote-full")
        self.assertEqual(request_shape_stage["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(request_shape_stage["duplicate_suppression"]["suppresses_new_activation_issue"])
        ledger = stats_summary["evidence_to_activation_next_action_ledger"]
        crunch_entry = next(item for item in ledger["entries"] if item["lever"] == "crunch")
        self.assertEqual(crunch_entry["state"], "measured-active")
        self.assertEqual(crunch_entry["current_status"], "applied")
        self.assertEqual(crunch_entry["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(crunch_entry["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(crunch_entry["post_widening_next_action"], "keep-active")
        self.assertEqual(crunch_entry["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(crunch_entry["post_max_rollout_decision"], "promote-full")
        self.assertEqual(crunch_entry["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(crunch_entry["applied_count"], 107)
        self.assertEqual(crunch_entry["holdout_count"], 40)
        self.assertEqual(crunch_entry["safety_stop_count"], 0)
        self.assertEqual(crunch_entry["projected_saved_usd"], 25.818387)
        self.assertTrue(crunch_entry["duplicate_suppression"]["suppresses_generic_crunch_activation_issue"])
        crunch_candidate = next(
            item for item in plan["evidence"]["optimization_candidates"]
            if item.get("lever") == "crunch"
        )
        self.assertEqual(crunch_candidate["issue_generation_status"], "suppressed-active-crunch-keep-active")
        titles = [item["title"] for item in plan["backlog_changes"]["create_issues"]]
        self.assertNotIn("Rank crunch savings follow-up for crunch-observed-savings-ranked", titles)
        entry = next(item for item in ledger["entries"] if item["lever"] == "request-shape-rollups")
        self.assertEqual(entry["state"], "measured-active")
        self.assertEqual(entry["current_status"], "applied")
        self.assertEqual(entry["active_rule_count"], 1)
        self.assertEqual(entry["widened_rule_count"], 1)
        self.assertEqual(entry["applied_count"], 107)
        self.assertEqual(entry["holdout_count"], 40)
        self.assertEqual(entry["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(entry["post_widening_next_action"], "keep-active")
        self.assertEqual(entry["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(entry["post_max_rollout_decision"], "promote-full")
        self.assertEqual(entry["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(entry["active_rule_source_evidence_schema"], "agentflow.request_shape_crunch_policy_decision.v1")
        self.assertTrue(entry["duplicate_suppression"]["suppresses_new_activation_issue"])
        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-request-secret", rendered)

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
        self.assertTrue(signal["duplicate_suppression"]["suppresses_generic_missing_health_issue"])
        self.assertEqual(
            signal["duplicate_suppression"]["reason"],
            "local-file-backed-handoff-outcome-recorded",
        )
        outcome_families = {row["local_action_family"] for row in signal["local_file_backed_handoff_outcomes"]}
        self.assertIn("routing", outcome_families)
        self.assertIn("cache", outcome_families)
        self.assertEqual(
            signal["top_local_file_backed_handoff_outcome"]["outcome"],
            "local-file-backed-handoff-recorded",
        )

        managed_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "managed-recommendation")
        self.assertTrue(managed_candidate["blocker"].startswith("managed-recommendation-health-report-missing"))
        self.assertEqual(managed_candidate["safety_status"], "review-required")
        self.assertEqual(managed_candidate["projected_savings_signal"]["status"], "missing-managed-recommendation-health-report")
        self.assertEqual(managed_candidate["issue_generation_status"], "suppressed-local-file-backed-handoff")
        self.assertFalse(
            any(
                item["title"] == "Rank managed recommendation omission reasons for local policy handoff"
                for item in plan["backlog_changes"]["create_issues"]
            )
        )

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
        outcome_by_family = {
            row["local_action_family"]: row
            for row in signal["local_file_backed_handoff_outcomes"]
        }
        self.assertEqual(set(outcome_by_family), {"cache", "crunch", "routing"})
        self.assertTrue(signal["duplicate_suppression"]["suppresses_generic_missing_health_issue"])
        self.assertEqual(
            signal["duplicate_suppression"]["covered_local_action_families"],
            ["cache", "crunch", "routing"],
        )
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
            outcome = outcome_by_family[family]
            self.assertEqual(outcome["outcome"], "local-file-backed-handoff-recorded")
            self.assertEqual(outcome["rule_file"], rule_file)
            self.assertEqual(outcome["follow_up_owner"], "local-policy")

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        managed_entry = next(entry for entry in ledger["entries"] if entry["lever"] == "managed-recommendation")
        self.assertEqual(managed_entry["managed_dependency"], "optional")
        self.assertEqual(managed_entry["local_action_family"], "crunch")
        self.assertIn("crunch_rules.yaml", managed_entry["local_handoff_reason"])
        self.assertFalse(
            any(
                item["title"] == "Rank managed recommendation omission reasons for local policy handoff"
                for item in plan["backlog_changes"]["create_issues"]
            )
        )
        self.assertFalse(signal["privacy"]["raw_prompts_included"])
        self.assertFalse(signal["privacy"]["provider_bodies_included"])
        self.assertFalse(signal["privacy"]["request_ids_included"])
        self.assertFalse(signal["privacy"]["session_ids_included"])

    def test_managed_report_without_omissions_falls_back_to_local_policy_handoffs(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 2873,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "managed_recommendations": {
                    "schema": "agentflow.managed_recommendations.v1",
                    "current_config": {"enabled": False},
                    "summary": {
                        "window_calls": 2873,
                        "metadata_rows": 0,
                        "received_count": 0,
                        "applied_count": 0,
                        "disabled_count": 2873,
                    },
                    "reason_breakdown": [],
                    "status_breakdown": [],
                    "recommendation_health": {"rows": []},
                },
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
                    "privacy": {"metadata_only": True, "aggregate_only": True},
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
                    }
                ],
            },
            threshold=3,
            now=NOW,
        )

        signal = plan["evidence"]["stats_summary"]["managed_recommendation_health"]
        self.assertEqual(signal["status"], "omission-reasons-ranked")
        self.assertEqual(signal["top_omission"]["local_action_family"], "crunch")
        self.assertEqual(signal["top_omission"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(signal["top_omission"]["follow_up_owner"], "local-policy")
        self.assertEqual(signal["top_omission"]["local_file_backed_representation"]["rule_file"], "crunch_rules.yaml")
        self.assertGreaterEqual(signal["summary"]["local_file_backed_count"], 3)
        self.assertEqual(signal["summary"]["no_local_representation_count"], 0)
        self.assertIn("routing", {row["local_action_family"] for row in signal["omissions"]})
        self.assertIn("cache", {row["local_action_family"] for row in signal["omissions"]})

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
        self.assertEqual(top["local_action_family"], "crunch")
        self.assertEqual(top["readiness_state"], "activation-ready")
        self.assertEqual(top["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(top["sample_count"], 24)
        self.assertEqual(top["blocker_reason"], "unsupported-streaming-shape")
        self.assertEqual(signal["local_action_cohorts"][0], top)
        self.assertEqual(
            signal["summary"]["local_action_family_breakdown"][0],
            {"value": "crunch", "count": 24},
        )
        self.assertEqual(
            signal["summary"]["readiness_breakdown"][0],
            {"value": "activation-ready", "count": 26},
        )
        self.assertEqual(
            signal["summary"]["next_action_breakdown"][0],
            {"value": "stage-repeated-context-crunch-canary", "count": 24},
        )

        shape_candidate = next(candidate for candidate in plan["evidence"]["optimization_candidates"] if candidate["lever"] == "request-shape-rollups")
        self.assertEqual(shape_candidate["blocker"], "request-shape-crunch-stage-repeated-context-crunch-canary")
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

    def test_request_shape_replayability_ranks_skipped_blocker_when_remaining_ready_zero(self):
        plan = build_research_plan(
            issues=[],
            stats={
                "calls": 3385,
                "cache_hits": 1,
                "cache_hit_rate": 0.000295,
                "request_shape_rollups": {
                    "schema": "agentflow.request_shape_rollups.v1",
                    "summary": {"rows_considered": 999, "rollup_count": 32},
                    "rollups": [],
                    "cache_replayability_dry_run": {
                        "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "status": "ranked",
                        "summary": {
                            "cohort_count": 32,
                            "rows_considered": 999,
                            "replay_ready_cohort_count": 7,
                            "replay_ready_rows": 108,
                            "remaining_replay_ready_cohort_count": 0,
                            "remaining_replay_ready_rows": 0,
                            "handled_replay_ready_cohort_count": 7,
                            "handled_replay_ready_rows": 108,
                            "skipped_cohort_count": 25,
                            "skipped_rows": 891,
                            "projected_hits": 101,
                            "projected_savings_usd": 0.214063,
                            "remaining_projected_hits": 0,
                            "remaining_projected_savings_usd": 0.0,
                            "handled_projected_hits": 101,
                            "handled_projected_savings_usd": 0.214063,
                            "top_blocker_code": "invalidation-evidence-missing",
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
                                "routing_status": "passthrough",
                                "row_count": 51,
                                "projected_hits": 50,
                                "projected_savings_usd": 0.106025,
                                "handled_by_local_policy": True,
                                "remaining_replay_ready": False,
                                "next_action": "already-handled-by-local-cache-policy",
                                "request_id": "handled-request-secret",
                            },
                            {
                                "readiness": "skipped",
                                "reason": "invalidation-evidence-missing",
                                "blockers": [
                                    "invalidation-evidence-missing",
                                    "tools-present",
                                    "unsafe-tool-calls-without-invalidation",
                                ],
                                "provider_family": "openai",
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                                "category": "tool-light",
                                "workflow_phase": "tool-light",
                                "stream": False,
                                "has_tools": True,
                                "cache_status": "skipped",
                                "routing_status": "passthrough",
                                "row_count": 30,
                                "projected_hits": 0,
                                "projected_savings_usd": 0.020955,
                                "cache_key": "skipped-cache-secret",
                            },
                        ],
                        "skipped_openai_blockers": {
                            "schema": "agentflow.request_shape_skipped_openai_cache_replay_blockers.v1",
                            "status": "ranked",
                            "next_action": "add-invalidation-evidence",
                            "summary": {
                                "skipped_openai_cohort_count": 3,
                                "replay_ready_count": 7,
                                "replay_ready_rows": 108,
                                "skipped_count": 25,
                                "skipped_rows": 891,
                                "sample_count": 891,
                                "affected_rows": 891,
                                "projected_hits": 0,
                                "projected_savings_usd": 0.020955,
                                "top_blocker_code": "invalidation-evidence-missing",
                                "top_blocker_count": 30,
                                "top_next_action": "add-invalidation-evidence",
                                "cache_apply_action_count": 0,
                                "cache_entries_written": 0,
                                "policy_files_written": False,
                            },
                            "blocker_breakdown": [
                                {"value": "invalidation-evidence-missing", "count": 30},
                                {"value": "tools-present", "count": 30},
                                {"value": "unsafe-tool-calls-without-invalidation", "count": 30},
                                {"value": "streaming-replay-not-supported", "count": 15},
                            ],
                            "next_action_breakdown": [
                                {"value": "add-invalidation-evidence", "count": 30},
                                {"value": "wait-for-streaming-replay-support", "count": 15},
                            ],
                            "cohorts": [
                                {
                                    "schema": "agentflow.request_shape_skipped_openai_cache_replay_blocker.v1",
                                    "rank": 1,
                                    "provider_family": "openai",
                                    "source_surface": "openai_responses",
                                    "endpoint": "responses",
                                    "category": "tool-light",
                                    "workflow_phase": "tool-light",
                                    "stream": False,
                                    "has_tools": True,
                                    "cache_status": "skipped",
                                    "routing_status": "passthrough",
                                    "sample_count": 30,
                                    "row_count": 30,
                                    "projected_hits": 0,
                                    "projected_savings_usd": 0.020955,
                                    "reason": "invalidation-evidence-missing",
                                    "blocker_codes": [
                                        "invalidation-evidence-missing",
                                        "tools-present",
                                        "unsafe-tool-calls-without-invalidation",
                                    ],
                                    "next_action": "add-invalidation-evidence",
                                    "tool_cache_replay_enabled": False,
                                    "streaming_replay_enabled": False,
                                    "emits_cache_apply_action": False,
                                    "request_id": "skipped-request-secret",
                                    "session_id": "skipped-session-secret",
                                    "cache_key": "skipped-cache-secret",
                                    "file_path": "/tmp/private-skipped-cache.py",
                                }
                            ],
                            "acceptance": {
                                "has_ranked_skipped_openai_cohorts": True,
                                "emits_no_cache_apply_actions": True,
                                "tool_and_streaming_replay_remain_disabled": True,
                                "metadata_only": True,
                                "aggregate_only": True,
                            },
                            "privacy": {"metadata_only": True, "aggregate_only": True},
                        },
                        "tool_replay_evidence": {
                            "schema": "agentflow.request_shape_tool_cache_replay_evidence.v1",
                            "status": "ranked",
                            "next_action": "collect-file-invalidation-evidence",
                            "summary": {
                                "tool_cache_replay_evidence_cohort_count": 1,
                                "sample_count": 30,
                                "affected_rows": 30,
                                "tools_present_rows": 30,
                                "tools_present_replay_evidence_rows": 30,
                                "generic_tools_present_blocker_reduced_rows": 30,
                                "unsafe_tool_call_blocker_rows": 30,
                                "missing_dependency_evidence_rows": 30,
                                "top_evidence_state": "blocked-missing-dependency-evidence",
                                "top_next_action": "collect-file-invalidation-evidence",
                                "top_blocker_code": "invalidation-evidence-missing",
                                "cache_apply_action_count": 0,
                                "cache_entries_written": 0,
                                "policy_files_written": False,
                            },
                            "evidence_state_breakdown": [
                                {"value": "blocked-missing-dependency-evidence", "count": 30},
                            ],
                            "dependency_evidence_decision_breakdown": [
                                {"value": "missing-dependency-evidence", "count": 30},
                            ],
                            "next_action_breakdown": [
                                {"value": "collect-file-invalidation-evidence", "count": 30},
                            ],
                            "blocker_breakdown": [
                                {"value": "invalidation-evidence-missing", "count": 30},
                                {"value": "tools-present", "count": 30},
                                {"value": "unsafe-tool-calls-without-invalidation", "count": 30},
                            ],
                            "cohorts": [
                                {
                                    "schema": "agentflow.request_shape_tool_cache_replay_evidence_row.v1",
                                    "rank": 1,
                                    "provider_family": "openai",
                                    "source_surface": "openai_responses",
                                    "endpoint": "responses",
                                    "category": "tool-light",
                                    "workflow_phase": "tool-light",
                                    "has_tools": True,
                                    "sample_count": 30,
                                    "row_count": 30,
                                    "evidence_state": "blocked-missing-dependency-evidence",
                                    "evidence_reason": "invalidation-evidence-missing",
                                    "blocker_codes": [
                                        "invalidation-evidence-missing",
                                        "tools-present",
                                        "unsafe-tool-calls-without-invalidation",
                                    ],
                                    "next_action": "collect-file-invalidation-evidence",
                                    "tools_present_replay_evidence": True,
                                    "generic_tools_present_blocker_reduced": True,
                                    "tool_cache_replay_enabled": False,
                                    "streaming_replay_enabled": False,
                                    "emits_cache_apply_action": False,
                                    "request_id": "tool-evidence-request-secret",
                                    "session_id": "tool-evidence-session-secret",
                                    "cache_key": "tool-evidence-cache-secret",
                                    "file_path": "/tmp/private-tool-evidence.py",
                                }
                            ],
                            "acceptance": {
                                "has_ranked_tool_cache_replay_evidence": True,
                                "reports_tools_present_replay_evidence": True,
                                "reduces_generic_tools_present_blocker": True,
                                "emits_no_cache_apply_actions": True,
                                "tool_and_streaming_replay_remain_disabled": True,
                                "metadata_only": True,
                                "aggregate_only": True,
                            },
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

        stats_summary = plan["evidence"]["stats_summary"]
        shape_signal = stats_summary["request_shape_rollup_candidates"]
        skipped = shape_signal["cache_replayability_dry_run"]["skipped_openai_blockers"]
        self.assertEqual(skipped["schema"], "agentflow.request_shape_skipped_openai_cache_replay_blockers.v1")
        self.assertEqual(skipped["summary"]["top_blocker_code"], "invalidation-evidence-missing")
        self.assertEqual(skipped["summary"]["affected_rows"], 891)
        self.assertTrue(skipped["privacy"]["metadata_only"])
        self.assertTrue(skipped["privacy"]["aggregate_only"])
        tool_replay = shape_signal["cache_replayability_dry_run"]["tool_replay_evidence"]
        self.assertEqual(tool_replay["schema"], "agentflow.request_shape_tool_cache_replay_evidence.v1")
        self.assertEqual(tool_replay["summary"]["tools_present_replay_evidence_rows"], 30)
        self.assertEqual(tool_replay["summary"]["generic_tools_present_blocker_reduced_rows"], 30)
        self.assertTrue(tool_replay["acceptance"]["reduces_generic_tools_present_blocker"])
        self.assertTrue(tool_replay["privacy"]["metadata_only"])
        self.assertTrue(tool_replay["privacy"]["aggregate_only"])

        loop = stats_summary["evidence_to_activation_loop"]
        cache_stage = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache_stage["state"], "missing-evidence")
        self.assertEqual(cache_stage["next_action"], "resolve-cache-replayability-blocker")
        self.assertIn("invalidation-evidence-missing", cache_stage["blocker_codes"])
        self.assertEqual(cache_stage["sample_count"], 30)

        candidates = plan["evidence"]["optimization_candidates"]
        cache_candidate = next(row for row in candidates if row["lever"] == "cache")
        self.assertEqual(cache_candidate["blocker"], "invalidation-evidence-missing")
        self.assertEqual(cache_candidate["projected_savings_signal"]["remaining_replay_ready_rows"], 0)
        self.assertEqual(cache_candidate["projected_savings_signal"]["skipped_openai_cohort_count"], 3)

        created = plan["backlog_changes"]["create_issues"]
        titles = [item["title"] for item in created]
        self.assertNotIn("Stage cache replay canary for replay-ready on openai/openai_responses/responses", titles)
        cache_issue = next(item for item in created if item["title"].startswith("Collect cache replay dependency evidence"))
        self.assertIn("invalidation-evidence-missing on openai/openai_responses/responses", cache_issue["title"])
        self.assertIn("Source metadata: request_shape_skipped_openai_cache_replay_blockers", cache_issue["body"])
        self.assertIn("count: 30", cache_issue["body"])
        self.assertIn("projected_hits: 0", cache_issue["body"])
        self.assertIn("projected_saved_cost_usd: 0.020955", cache_issue["body"])
        self.assertIn("cache", cache_issue["labels"])

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("handled-request-secret", rendered)
        self.assertNotIn("skipped-request-secret", rendered)
        self.assertNotIn("skipped-session-secret", rendered)
        self.assertNotIn("skipped-cache-secret", rendered)
        self.assertNotIn("/tmp/private-skipped-cache.py", rendered)
        self.assertNotIn("tool-evidence-request-secret", rendered)
        self.assertNotIn("tool-evidence-session-secret", rendered)
        self.assertNotIn("tool-evidence-cache-secret", rendered)
        self.assertNotIn("/tmp/private-tool-evidence.py", rendered)

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
                                "schema": "agentflow.request_shape_blocker_cohort.v1",
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
                                "readiness_state": "activation-ready",
                                "actionability_reason": "replay-ready-exact-non-tool-shape",
                                "candidate_work_classes": ["replayability"],
                                "candidate_families": ["cache_replay"],
                                "blocker_codes": [],
                                "row_count": 56,
                                "sample_count": 56,
                                "projected_hits": 55,
                                "projected_savings_usd": 0.121981,
                                "cost_est_usd": 0.12,
                                "observed_savings_usd": 0.0,
                                "candidate_id": "raw-request-shape-cohort-secret",
                                "request_id": "raw-request-shape-request-secret",
                                "session_id": "raw-request-shape-session-secret",
                                "cache_key": "raw-request-shape-cache-secret",
                                "file_path": "/tmp/private-request-shape.py",
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

        shape_signal = stats_summary["request_shape_rollup_candidates"]
        self.assertEqual(shape_signal["status"], "candidates-ranked")
        self.assertEqual(shape_signal["summary"]["top_next_action"], "stage-cache-replay-canary")
        self.assertEqual(shape_signal["summary"]["top_local_action_family"], "cache")
        top_shape = shape_signal["top_candidate"]
        self.assertEqual(top_shape["schema"], "agentflow.request_shape_blocker_cohort.v1")
        self.assertEqual(top_shape["rank"], 1)
        self.assertEqual(top_shape["local_action_family"], "cache")
        self.assertEqual(top_shape["readiness_state"], "activation-ready")
        self.assertEqual(top_shape["next_action"], "stage-cache-replay-canary")
        self.assertEqual(top_shape["sample_count"], 56)
        self.assertEqual(top_shape["projected_hits"], 55)
        self.assertAlmostEqual(top_shape["projected_savings_usd"], 0.121981)
        request_shape_stage = next(row for row in loop["levers"] if row["lever"] == "request-shape-rollups")
        self.assertEqual(request_shape_stage["state"], "activation-ready")
        self.assertEqual(request_shape_stage["local_action_family"], "cache")
        self.assertEqual(request_shape_stage["next_action"], "stage-cache-replay-canary")
        self.assertEqual(request_shape_stage["sample_count"], 56)
        self.assertAlmostEqual(request_shape_stage["projected_saved_usd"], 0.121981)

        rendered = json.dumps(plan, sort_keys=True)
        self.assertNotIn("raw-request-shape-cohort-secret", rendered)
        self.assertNotIn("raw-request-shape-request-secret", rendered)
        self.assertNotIn("raw-request-shape-session-secret", rendered)
        self.assertNotIn("raw-request-shape-cache-secret", rendered)
        self.assertNotIn("/tmp/private-request-shape.py", rendered)
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

    def test_repeated_skip_diagnostics_record_narrow_missing_dependency_blocker(self):
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
        self.assertFalse(any("missing dependency evidence" in title for title in created_titles))
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        entries = [
            entry for entry in ledger["entries"]
            if entry.get("diagnostic_class") == "missing-dependency-evidence"
        ]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["current_status"], "keep-blocked")
        self.assertEqual(entry["issue_worthy_status"], "blocked")
        self.assertEqual(
            entry["next_action"],
            "keep-missing-dependency-evidence-blocked-until-sanitized-source-report",
        )
        self.assertEqual(
            entry["keep_blocked_reason"],
            "activation-feedback-missing-dependency-evidence-needs-sanitized-source-report",
        )
        self.assertEqual(entry["dependency_evidence_status"], "missing-sanitized-source-report")
        self.assertTrue(entry["durable_action_ledger_entry"])
        self.assertEqual(
            entry["diagnostic_fingerprint"],
            "agentflow.repeated-diagnostic.missing-dependency-evidence.v1",
        )
        self.assertIn("sanitized_source_report", entry["needed_resolution"])
        self.assertEqual(
            entry["missing_dependency_evidence_review"]["schema"],
            "agentflow.activation_feedback_missing_dependency_evidence_review.v1",
        )
        self.assertTrue(entry["missing_dependency_evidence_review"]["privacy"]["metadata_only"])
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["activation_feedback_keep_blocked_suppressed_count"], 1)
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

    def test_unclassified_true_unknown_blocker_emits_one_bounded_human_review_proposal(self):
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
        self.assertNotIn("unclassified-skip-or-blocker", [d.get("reason") for d in diagnostics])
        actionable = [d for d in diagnostics if d.get("reason") == "activation-feedback-blocker-review"]
        self.assertTrue(actionable)
        self.assertEqual(actionable[0].get("diagnostic_class"), "activation-feedback-blocker-review")
        self.assertEqual(actionable[0].get("reclassification_source"), "no-match-bounded-human-review")
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        ledger_entries = [
            entry
            for entry in ledger["entries"]
            if entry.get("lever") == "activation-feedback"
        ]
        self.assertEqual(len(ledger_entries), 1)
        self.assertEqual(
            ledger_entries[0]["next_action"],
            "keep-activation-feedback-blocker-review-blocked-until-new-sanitized-local-evidence",
        )
        self.assertEqual(ledger_entries[0]["local_action_family"], "activation-feedback")
        self.assertEqual(ledger_entries[0]["sample_count"], 2)
        self.assertEqual(ledger_entries[0]["current_status"], "keep-blocked")
        self.assertEqual(ledger_entries[0]["issue_worthy_status"], "blocked")
        self.assertEqual(
            ledger_entries[0]["keep_blocked_reason"],
            "activation-feedback-blocker-review-already-resolved-to-bounded-local-action-ledger",
        )
        self.assertEqual(
            ledger_entries[0]["diagnostic_fingerprint"],
            "agentflow.repeated-diagnostic.activation-feedback-blocker-review.v1",
        )
        self.assertTrue(ledger_entries[0]["durable_action_ledger_entry"])
        self.assertEqual(ledger_entries[0]["review_status"], "resolved-to-keep-blocked")
        self.assertEqual(
            ledger_entries[0]["verification_check"],
            "The next research plan emits a durable activation-feedback ledger entry with a concrete next action, stable fingerprint, and metadata-only privacy flags.",
        )
        self.assertTrue(ledger_entries[0]["privacy"]["metadata_only"])
        self.assertTrue(ledger_entries[0]["privacy"]["aggregate_only"])
        self.assertFalse(ledger_entries[0]["privacy"]["raw_prompts_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["provider_bodies_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["absolute_paths_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["request_ids_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["session_ids_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["cache_keys_included"])
        self.assertFalse(ledger_entries[0]["privacy"]["individual_candidate_ids_included"])
        self.assertTrue(ledger["privacy"]["metadata_only"])
        self.assertTrue(ledger["privacy"]["aggregate_only"])
        optimization_candidates = plan["evidence"]["optimization_candidates"]
        self.assertFalse(
            [
                item for item in optimization_candidates
                if item["lever"] == "activation-feedback"
                and item["blocker"] == "repeated-activation-feedback-blocker-review"
            ],
            "resolved activation-feedback blocker-review diagnostics should not become optimization proposals",
        )
        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        bounded_proposals = [t for t in created_titles if "unclassified activation skip or blocker" in t]
        self.assertFalse(
            bounded_proposals,
            "durable keep-blocked activation-feedback blocker-review ledger should suppress duplicate ready proposals",
        )
        self.assertFalse(
            [
                item for item in plan["backlog_changes"]["create_issues"]
                if item["title"] == "Resolve repeated-activation-feedback-blocker-review activation feedback blocker"
            ],
            "durable keep-blocked activation-feedback blocker-review ledger should suppress candidate proposals",
        )
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["activation_feedback_keep_blocked_suppressed_count"], 1)
        self.assertEqual(
            suppression["suppressed"][-1]["keep_blocked_reason"],
            "activation-feedback-blocker-review-already-resolved-to-bounded-local-action-ledger",
        )
        repeated_plan = build_research_plan(
            issues=[],
            log_sources=[
                "activation skipped due to xyz-unknown-blocker-type",
                "activation skipped due to xyz-unknown-blocker-type",
            ],
            threshold=1,
            now=NOW,
        )
        repeated_entry = [
            entry
            for entry in repeated_plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]["entries"]
            if entry.get("lever") == "activation-feedback"
        ][0]
        self.assertEqual(repeated_entry["fingerprint"], ledger_entries[0]["fingerprint"])
        self.assertEqual(repeated_entry["diagnostic_fingerprint"], ledger_entries[0]["diagnostic_fingerprint"])

    def test_unclassified_canary_cohort_skip_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing action skipped: not-in-canary-cohort, session omitted",
                        "routing action skipped: not-in-canary-cohort, session omitted",
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
        self.assertIn("missing-lifecycle-feedback", reasons)
        classified = next(d for d in diagnostics if d["reason"] == "missing-lifecycle-feedback")
        self.assertEqual(classified.get("reclassification_source"), "canary-cohort-skip-pattern")

    def test_unclassified_below_threshold_skip_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "crunch action skipped: context-below-threshold token count insufficient",
                        "crunch action skipped: context-below-threshold token count insufficient",
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
        classified = next(d for d in diagnostics if d["reason"] == "missing-dependency-evidence")
        self.assertEqual(classified.get("reclassification_source"), "below-threshold-pattern")

    def test_unclassified_stale_lifecycle_skip_is_reclassified(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "cache replay action omitted: activation-data-not-fresh evidence window exceeded",
                        "cache replay action omitted: activation-data-not-fresh evidence window exceeded",
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
        self.assertIn("stale-quality-evidence", reasons)
        classified = next(d for d in diagnostics if d["reason"] == "stale-quality-evidence")
        self.assertEqual(classified.get("reclassification_source"), "stale-lifecycle-pattern")

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
                    "response_body": {"content": "private response text"},
                    "cache_key": "cache-issue-secret",
                    "session_id": "session-raw-secret",
                }
            ],
            stats={
                "calls": 1,
                "request_json": {"messages": [{"content": "private stats prompt"}]},
                "raw_response": "private stats response",
                "cache_key": "cache-stats-secret",
                "routing": [{"requested_model": "gpt-5", "path": "/home/lutz/private/project/file.py"}],
            },
            log_sources=[
                "skip_reason=privacy-blocked request_id=req-raw-secret cache_key=cache-log-secret /home/lutz/private/project/file.py sk-testsecret123456"
            ],
            threshold=2,
            now=NOW,
        )

        rendered = json.dumps(plan)
        self.assertNotIn("private prompt text", rendered)
        self.assertNotIn("private stats prompt", rendered)
        self.assertNotIn("private response text", rendered)
        self.assertNotIn("private stats response", rendered)
        self.assertNotIn("/home/lutz/private/project/file.py", rendered)
        self.assertNotIn("req-raw-secret", rendered)
        self.assertNotIn("session-raw-secret", rendered)
        self.assertNotIn("cache-issue-secret", rendered)
        self.assertNotIn("cache-stats-secret", rendered)
        self.assertNotIn("cache-log-secret", rendered)
        self.assertNotIn("sk-testsecret123456", rendered)
        self.assertIn("[REDACTED", rendered)
        for proposal in plan["backlog_changes"]["create_issues"]:
            body = proposal["body"]
            self.assertIn("## Labels", body)
            self.assertNotIn("private prompt text", body)
            self.assertNotIn("private stats prompt", body)
            self.assertNotIn("private response text", body)
            self.assertNotIn("private stats response", body)
            self.assertNotIn("/home/lutz/private/project/file.py", body)
            self.assertNotIn("req-raw-secret", body)
            self.assertNotIn("session-raw-secret", body)
            self.assertNotIn("cache-issue-secret", body)
            self.assertNotIn("cache-stats-secret", body)
            self.assertNotIn("cache-log-secret", body)
        self.assertFalse(plan["privacy"]["raw_prompts_included"])
        self.assertFalse(plan["privacy"]["provider_bodies_included"])
        self.assertFalse(plan["privacy"].get("cache_keys_included", False))
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

    def test_cli_enriches_staged_request_shape_cache_replay_evidence_from_local_policy(self):
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            config_dir = Path(tmp) / "config"
            store = SQLiteStore(db_path)
            try:
                for cost in (0.01, 0.03, 0.02):
                    store.log_call(
                        id=str(uuid.uuid4()),
                        created_at=utc_now(),
                        path="/v1/responses",
                        requested_model="gpt-5.4-mini",
                        routed_model="gpt-5.4-mini",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=125,
                        input_tokens_est=1500,
                        output_tokens_est=100,
                        actual_input_tokens=1500,
                        actual_output_tokens=100,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost,
                        crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
                        routing_json=stable_json(
                            {
                                "category": "chat",
                                "workflow_phase": "chat",
                                "text_chars": 6000,
                                "has_tools": False,
                                "source_surface": "openai_responses",
                                "endpoint": "responses",
                            }
                        ),
                        cache_json=stable_json({"status": "miss", "reason": "exact-miss"}),
                        error=None,
                        request_json=stable_json({"prompt": "raw prompt must not leak"}),
                        response_json=stable_json({"content": "raw response must not leak"}),
                        session_id="raw-session-id-must-not-leak",
                        category="chat",
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        retry_count=0,
                        thinking_output_tokens=0,
                        provider="openai",
                        source_surface="openai_responses",
                        endpoint="responses",
                        requested_model_family="gpt-5",
                        routed_model_family="gpt-5",
                    )

                from agentflow_proxy.request_shape_rollups import (
                    apply_request_shape_cache_replay_canary_action,
                    build_request_shape_cache_replay_canary_stage_report,
                )

                stage = build_request_shape_cache_replay_canary_stage_report(
                    store,
                    limit=20,
                    run_id="cli-cache-replay-stage-enrichment",
                    rollout_fraction=0.10,
                    holdout_fraction=0.10,
                    mark_handled_cache_replay_cohorts=False,
                )
                apply_request_shape_cache_replay_canary_action(
                    stage["top_stage_action"],
                    rules_path=config_dir / "cache_canary_policy.yaml",
                )
            finally:
                store.conn.close()

            issues_path = Path(tmp) / "issues.json"
            stats_path = Path(tmp) / "stats.json"
            issues_path.write_text(json.dumps([]), encoding="utf-8")
            stats_path.write_text(json.dumps({"calls": 3, "cache_hits": 0, "cache_hit_rate": 0.0, "db": db_path}), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.dict("os.environ", {"AGENTFLOW_CONFIG_DIR": str(config_dir)}, clear=False):
                code = cli.orchestrator_research_cli(
                    ["--issues-json", str(issues_path), "--stats-json", str(stats_path), "--threshold", "3"],
                    stdout=stdout,
                    stderr=stderr,
                )

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        stats_summary = payload["evidence"]["stats_summary"]
        evidence = stats_summary["request_shape_cache_replay_evidence"]
        self.assertEqual(evidence["schema"], "agentflow.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "staged-no-traffic")
        self.assertEqual(evidence["next_action"], "collect-cache-replay-canary-traffic")
        self.assertEqual(evidence["staged_canary_count"], 1)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)

        loop = stats_summary["evidence_to_activation_loop"]
        cache = next(row for row in loop["levers"] if row["lever"] == "cache")
        self.assertEqual(cache["state"], "canary-staged")
        self.assertEqual(cache["next_action"], "collect-cache-replay-canary-traffic")
        self.assertEqual(cache["fingerprint_next_action"], "stage-cache-replay-canary")

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-session-id-must-not-leak", rendered)
        self.assertNotIn(str(config_dir), rendered)


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

    def test_repeated_safety_stop_keep_blocked_ledger_suppresses_ready_issue(self):
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
        self.assertFalse(repeated_diag_proposals, "current keep-blocked safety-stop ledger should suppress duplicate ready issue")
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        keep_blocked = [
            entry for entry in ledger["entries"]
            if entry.get("evidence_schema") == "agentflow.activation_safety_stop_burndown.v1"
        ]
        self.assertTrue(keep_blocked)
        self.assertEqual(keep_blocked[0]["current_status"], "keep-blocked")
        self.assertEqual(keep_blocked[0]["issue_worthy_status"], "blocked")
        self.assertEqual(
            keep_blocked[0]["keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertEqual(
            keep_blocked[0]["next_state_reason"],
            "safety-stop-requires-safer-threshold-or-rollback-proof",
        )
        self.assertIn("human_review", keep_blocked[0]["needed_resolution"])
        self.assertIn("safer_threshold", keep_blocked[0]["needed_resolution"])
        self.assertIn("rollback_proof", keep_blocked[0]["needed_resolution"])
        unblock = keep_blocked[0]["unblock_criteria"]
        self.assertEqual(unblock["schema"], "agentflow.activation_feedback_safety_stop_unblock_criteria.v1")
        self.assertEqual(unblock["status"], "blocked")
        self.assertFalse(unblock["safety_stop_count_zero"])
        self.assertFalse(unblock["applied_coverage_present"])
        self.assertFalse(unblock["holdout_coverage_present"])
        self.assertFalse(unblock["safer_threshold_or_executor_guard_present"])
        self.assertFalse(unblock["rollback_proof_present"])
        self.assertIn("human_review", unblock["needed_resolution"])
        self.assertIn("safer_threshold", unblock["needed_resolution"])
        self.assertIn("rollback_proof", unblock["needed_resolution"])
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["keep_blocked_ledger_suppressed_count"], 1)
        self.assertEqual(suppression["suppressed"][-1]["suppression_kind"], "current-keep-blocked-ledger-record")
        self.assertEqual(
            suppression["suppressed"][-1]["unblock_criteria"]["suppresses_ready_issue_until"],
            "safety_stop_count_zero_and_applied_holdout_coverage_present",
        )

    def test_repeated_missing_dependency_diagnostic_suppresses_existing_duplicate_work(self):
        existing_issue = issue(
            444,
            "Turn repeated missing dependency evidence diagnostics into an actionable optimization issue",
            ["status:ready", "priority:p2", "backlog", "core-feature", "correctness"],
        )
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "routing blocker=missing-dependency-evidence request_id=req-secret-dup1",
                        "routing blocker=missing-dependency-evidence request_id=req-secret-dup2",
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
            any("repeated" in t and "missing dependency evidence" in t for t in created_titles),
            "should not create a new issue when fingerprint matches open issue #444",
        )

        comment_issues = plan["backlog_changes"]["comment_issues"]
        self.assertFalse([c for c in comment_issues if c.get("number") == 444])
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["activation_feedback_keep_blocked_suppressed_count"], 1)
        suppressed = suppression["suppressed"][-1]
        self.assertEqual(
            suppressed["fingerprint"],
            "agentflow.repeated-diagnostic.missing-dependency-evidence.v1",
        )
        self.assertEqual(suppressed["suppression_kind"], "durable-keep-blocked-ledger-record")
        self.assertEqual(
            suppressed["keep_blocked_reason"],
            "activation-feedback-missing-dependency-evidence-needs-sanitized-source-report",
        )
        self.assertIn("sanitized_source_report", suppressed["needed_resolution"])
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

    def test_repeated_missing_dependency_ledger_includes_required_fields(self):
        plan = build_research_plan(
            issues=[],
            log_sources=[
                "routing blocker=missing-dependency-evidence x",
                "routing blocker=missing-dependency-evidence y",
            ],
            threshold=1,
            now=NOW,
        )
        created = plan["backlog_changes"]["create_issues"]
        repeated_proposals = [
            p for p in created
            if "repeated" in p.get("title", "").lower() and "missing dependency evidence" in p.get("title", "").lower()
        ]
        self.assertFalse(repeated_proposals)
        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        entry = [
            item for item in ledger["entries"]
            if item.get("diagnostic_class") == "missing-dependency-evidence"
        ][0]
        self.assertEqual(entry["sample_count"], 2)
        self.assertEqual(entry["current_status"], "keep-blocked")
        self.assertEqual(entry["state"], "keep-blocked")
        self.assertEqual(entry["local_action_family"], "activation-feedback")
        self.assertEqual(entry["evidence_schema"], "agentflow.orchestrator_research_log_diagnostics.v1")
        self.assertEqual(entry["review_status"], "resolved-to-narrower-blocker")
        self.assertIn("dependency_evidence_summary", entry["needed_resolution"])
        self.assertTrue(entry["privacy"]["metadata_only"])
        self.assertFalse(entry["privacy"]["request_ids_included"])
        self.assertFalse(entry["privacy"]["session_ids_included"])

    def test_issue_532_combined_activation_feedback_and_safety_stop_fixture(self):
        # Issue #532 acceptance metric: repeated activation-feedback-blocker-review (6x) and
        # safety-stop (7x) diagnostics plus pass diagnostics (3x) must produce durable ledger
        # entries with stable fingerprints, suppress keep-blocked safety-stop proposals, and
        # create at most one bounded ready proposal for the unresolved diagnostic class.
        log_lines = (
            [f"routing blocker=safety-stop request_id=req-safety-stop-{n}" for n in range(7)]
            + [f"activation feedback omitted by unknown gate with no machine reason request_id=req-af-{n}" for n in range(6)]
            + [f"verdict: pass threshold=ok run_id=pass-{n}" for n in range(3)]
        )
        plan = build_research_plan(
            issues=[],
            log_sources=log_lines,
            threshold=1,
            now=NOW,
        )

        ledger = plan["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]
        self.assertEqual(ledger["schema"], "agentflow.evidence_to_activation_next_action_ledger.v1")
        self.assertTrue(ledger["privacy"]["metadata_only"])
        self.assertTrue(ledger["privacy"]["aggregate_only"])

        # Durable ledger entry for activation-feedback-blocker-review with stable fingerprint,
        # source schema, action family, next action, and current state.
        af_entries = [
            entry for entry in ledger["entries"]
            if entry.get("lever") == "activation-feedback"
            and entry.get("diagnostic_class") == "activation-feedback-blocker-review"
        ]
        self.assertEqual(len(af_entries), 1, "expected exactly one ledger entry for activation-feedback-blocker-review")
        af_entry = af_entries[0]
        self.assertEqual(
            af_entry["diagnostic_fingerprint"],
            "agentflow.repeated-diagnostic.activation-feedback-blocker-review.v1",
        )
        self.assertEqual(
            af_entry["next_action"],
            "keep-activation-feedback-blocker-review-blocked-until-new-sanitized-local-evidence",
        )
        self.assertEqual(af_entry["local_action_family"], "activation-feedback")
        self.assertEqual(af_entry["evidence_schema"], "agentflow.orchestrator_research_log_diagnostics.v1")
        self.assertTrue(af_entry.get("fingerprint"), "durable stable fingerprint must be non-empty")
        self.assertEqual(af_entry.get("state"), "keep-blocked")
        self.assertEqual(af_entry.get("current_status"), "keep-blocked")
        self.assertEqual(af_entry.get("issue_worthy_status"), "blocked")
        self.assertEqual(
            af_entry.get("keep_blocked_reason"),
            "activation-feedback-blocker-review-already-resolved-to-bounded-local-action-ledger",
        )

        # Keep-blocked safety-stop ledger entry with keep_blocked_reason.
        safety_entries = [
            entry for entry in ledger["entries"]
            if entry.get("evidence_schema") == "agentflow.activation_safety_stop_burndown.v1"
        ]
        self.assertTrue(safety_entries, "expected a safety-stop ledger entry")
        self.assertEqual(safety_entries[0]["current_status"], "keep-blocked")
        self.assertEqual(
            safety_entries[0]["keep_blocked_reason"],
            "activation-feedback-safety-stop-needs-human-review-safer-threshold-rollback-proof",
        )
        self.assertEqual(safety_entries[0]["unblock_criteria"]["status"], "blocked")
        self.assertFalse(safety_entries[0]["unblock_criteria"]["safety_stop_count_zero"])
        self.assertFalse(safety_entries[0]["unblock_criteria"]["applied_coverage_present"])
        self.assertFalse(safety_entries[0]["unblock_criteria"]["holdout_coverage_present"])
        self.assertFalse(safety_entries[0]["unblock_criteria"]["safer_threshold_or_executor_guard_present"])
        self.assertFalse(safety_entries[0]["unblock_criteria"]["rollback_proof_present"])

        # Keep-blocked suppresses the repeated safety-stop ready issue proposal.
        repeated_safety_proposals = [
            item for item in plan["backlog_changes"]["create_issues"]
            if "repeated" in item["title"].lower() and "safety stop" in item["title"].lower()
        ]
        self.assertFalse(repeated_safety_proposals, "keep-blocked safety-stop must suppress its ready issue proposal")

        # The resolved activation-feedback blocker-review ledger suppresses bounded ready proposals.
        af_proposals = [
            item for item in plan["backlog_changes"]["create_issues"]
            if "unclassified activation skip or blocker" in item["title"].lower()
        ]
        self.assertFalse(af_proposals, "resolved activation-feedback blocker-review must not generate another ready proposal")

        # Pass diagnostics produce no proposals.
        pass_proposals = [
            item for item in plan["backlog_changes"]["create_issues"]
            if "pass" in item["title"].lower() and "threshold" in item["title"].lower()
        ]
        self.assertFalse(pass_proposals, "pass diagnostics must not generate proposals")

        # Suppression metadata records the keep-blocked count.
        suppression = plan["evidence"]["issue_proposal_suppression"]
        self.assertEqual(suppression["keep_blocked_ledger_suppressed_count"], 1)

        # Privacy: no raw request IDs in rendered output.
        rendered = json.dumps(plan, sort_keys=True)
        for n in range(7):
            self.assertNotIn(f"req-safety-stop-{n}", rendered)
        for n in range(6):
            self.assertNotIn(f"req-af-{n}", rendered)

        # Fingerprint stability: two identical runs produce the same fingerprints.
        plan2 = build_research_plan(
            issues=[],
            log_sources=log_lines,
            threshold=1,
            now=NOW,
        )
        af_entries2 = [
            entry for entry in plan2["evidence"]["stats_summary"]["evidence_to_activation_next_action_ledger"]["entries"]
            if entry.get("lever") == "activation-feedback"
            and entry.get("diagnostic_class") == "activation-feedback-blocker-review"
        ]
        self.assertTrue(af_entries2, "expected repeated run to also produce an af-blocker-review ledger entry")
        self.assertEqual(af_entries2[0]["fingerprint"], af_entry["fingerprint"])
        self.assertEqual(af_entries2[0]["diagnostic_fingerprint"], af_entry["diagnostic_fingerprint"])


if __name__ == "__main__":
    unittest.main()
