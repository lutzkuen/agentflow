import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentflow_proxy import cli
from agentflow_proxy.orchestrator_research import build_research_plan


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
        self.assertIn("status:ready", created[0]["labels"])

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
            self.assertIn("## Sequencing Notes", body)
        self.assertGreaterEqual(core_feature_count, len(created) // 2)

        rendered = json.dumps(plan)
        self.assertNotIn("private-candidate-secret", rendered)
        self.assertNotIn("req-private-secret", rendered)
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
        self.assertEqual(gpt54_mini["actionability"], "already-cheapest")
        self.assertIsNotNone(gpt54_mini["no_op_reason"])

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
                log_sources=[log_path],
                threshold=1,
                now=NOW,
            )

        diagnostics = plan["evidence"]["repeated_diagnostics"]
        self.assertEqual(diagnostics[0]["reason"], "pass")
        self.assertGreaterEqual(diagnostics[0]["count"], 2)

        created_titles = [item["title"].lower() for item in plan["backlog_changes"]["create_issues"]]
        self.assertTrue(any("aggregate only diagnostics" in title for title in created_titles))
        self.assertFalse(any("pass diagnostics" in title for title in created_titles))

        repeated_issue = [item for item in plan["backlog_changes"]["create_issues"] if "aggregate only diagnostics" in item["title"].lower()][0]
        self.assertIn("Source lever:", repeated_issue["body"])
        self.assertIn("Expected unblock path:", repeated_issue["body"])
        self.assertIn("metadata-only privacy", repeated_issue["body"])

        candidates = plan["evidence"]["optimization_candidates"]
        diagnostic_candidates = [item for item in candidates if item["lever"] == "activation-feedback"]
        self.assertTrue(diagnostic_candidates)
        self.assertEqual(diagnostic_candidates[0]["blocker"], "repeated-aggregate-only")
        self.assertNotEqual(diagnostic_candidates[0]["projected_savings_signal"]["diagnostic_reason"], "pass")

        rendered = json.dumps(plan)
        self.assertNotIn("cache-candidate-secret", rendered)
        self.assertNotIn("req-secret-12345", rendered)
        self.assertNotIn("session-secret-67890", rendered)

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


if __name__ == "__main__":
    unittest.main()
