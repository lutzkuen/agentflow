from __future__ import annotations

import io
import json
import unittest

from tokenclaw import cli
from tokenclaw.cache_promotion_drafts import SCHEMA, dry_run_cache_promotion_drafts
from tokenclaw.local_promotion_candidates import build_local_promotion_candidates_from_reports


RAW_SECRET = "raw-cache-promotion-secret"


def _promotion_report(*, ready: bool = True, blocker: str | None = None, has_tools: bool = False, stream: bool = False) -> dict:
    report = build_local_promotion_candidates_from_reports(
        {
            "cache_impact": {
                "schema": "agentflow.openai_cache_replay_impact.v1",
                "status": "matched",
                "candidates": [
                    {
                        "verdict": "widen" if ready else "hold",
                        "reason_codes": ["target-savings-met"] if ready and blocker is None else [blocker or "insufficient-holdout-samples"],
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "stream": stream,
                        "has_tools": has_tools,
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "replay_source_schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                        "readiness": "replay-ready",
                        "replay_ready": True,
                        "sample_count": 8,
                        "applied_count": 5 if ready else 0,
                        "holdout_count": 3 if ready else 0,
                        "safety_stop_count": 0,
                        "projected_hits": 12,
                        "actual_hits": 4,
                        "actual_saved_cost_usd": 0.08,
                        "observed_savings_usd": 0.08,
                        "projected_saved_usd": 0.24,
                        "dry_run_projected_savings_usd": 0.24,
                        "miss_count": 1,
                        "invalidated_count": 0,
                        "last_observed_at": "2999-01-01T00:00:00+00:00",
                        "canary_hit_measurement": {
                            "hit_realization_rate": 0.333333,
                            "savings_realization_rate": 0.333333,
                            "raw_prompt": f"must not leak {RAW_SECRET}",
                        },
                        "candidate_id": f"source-candidate-{RAW_SECRET}",
                        "request_fingerprint": f"source-fingerprint-{RAW_SECRET}",
                        "cache_key": f"source-cache-key-{RAW_SECRET}",
                    }
                ],
            },
            "request_shape_rollups": {},
            "anthropic_thinking_compaction_impact": {},
            "claude_routing_impact": {},
            "openai_routing_report": {},
        }
    )
    report["candidates"] = [candidate for candidate in report["candidates"] if candidate["action_family"] == "cache"]
    return report


def _projected_replay_ready_report(*, blocker: str | None = None) -> dict:
    report = build_local_promotion_candidates_from_reports(
        {
            "cache_impact": {},
            "request_shape_rollups": {
                "schema": "agentflow.request_shape_rollups.v1",
                "cache_replayability_dry_run": {
                    "schema": "agentflow.request_shape_cache_replayability_dry_run.v1",
                    "status": "ranked",
                    "cohorts": [
                        {
                            "provider_family": "openai",
                            "source_surface": "openai_responses",
                            "endpoint": "responses",
                            "category": "chat",
                            "workflow_phase": "chat",
                            "readiness": "replay-ready",
                            "blockers": [blocker] if blocker else [],
                            "reason": "replay-ready-exact-non-tool-shape",
                            "cache_status": "miss",
                            "routing_status": "disabled",
                            "has_tools": False,
                            "stream": False,
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "500_2k_tokens",
                            "row_count": 56,
                            "projected_hits": 55,
                            "projected_savings_usd": 0.121981,
                        }
                    ],
                },
            },
            "anthropic_thinking_compaction_impact": {},
            "claude_routing_impact": {},
            "openai_routing_report": {},
        }
    )
    report["candidates"] = [candidate for candidate in report["candidates"] if candidate["action_family"] == "cache"]
    return report


class CachePromotionDraftTests(unittest.TestCase):
    def test_drafts_bounded_exact_cache_rule_from_promotion_candidate(self) -> None:
        result = dry_run_cache_promotion_drafts(
            _promotion_report(),
            initial_canary_fraction=0.2,
            holdout_fraction=0.15,
            max_evidence_age_hours=10_000_000,
            ttl_seconds=3600,
        )

        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["projected_hits"], 12)
        self.assertAlmostEqual(result["summary"]["projected_savings_usd"], 0.24)
        self.assertEqual(result["summary"]["max_drafts"], 10)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        draft = result["drafts"][0]
        self.assertEqual(draft["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(draft["target_local_policy_section"], "cache.pattern_rules")
        self.assertEqual(draft["activation_fraction"], 0.2)
        self.assertEqual(draft["holdout_fraction"], 0.15)
        self.assertEqual(draft["ttl_seconds"], 3600)
        rule = draft["proposed_rule"]
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:*"])
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertEqual(rule["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(rule["action"]["type"], "exact_cache_pattern")
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["safe_invalidation"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        assumptions = rule["action"]["invalidation_assumptions"]
        self.assertFalse(assumptions["tool_call_caching_enabled"])
        self.assertFalse(assumptions["streaming_replay_enabled"])
        self.assertTrue(assumptions["session_scoped_keys_required"])
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.2)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.15)
        self.assertEqual(rule["graduation"]["projected_hits"], 12)
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["actual_hits"], 4)
        self.assertEqual(rule["promotion"]["rollback_metadata"]["rollback_action_type"], "disable_rule")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(RAW_SECRET, rendered)
        self.assertNotIn('"cache_key"', rendered)
        self.assertNotIn('"request_fingerprint":', rendered)
        self.assertFalse(result["privacy"]["cache_keys_included"])

    def test_no_ops_for_blocked_tool_streaming_or_unready_cache_candidates(self) -> None:
        for kwargs, expected in (
            ({"ready": False}, "insufficient-holdout-samples"),
            ({"blocker": "invalidation-evidence-missing"}, "invalidation-evidence-missing"),
            ({"has_tools": True}, "tools-present"),
            ({"stream": True}, "streaming-replay-not-supported"),
        ):
            result = dry_run_cache_promotion_drafts(
                _promotion_report(**kwargs),
                max_evidence_age_hours=10_000_000,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "no-op")
            self.assertEqual(result["summary"]["draft_count"], 0)
            self.assertEqual(result["omitted"][0]["reason"], expected)

    def test_drafts_projected_replay_ready_request_shape_without_measured_hits(self) -> None:
        result = dry_run_cache_promotion_drafts(
            _projected_replay_ready_report(),
            initial_canary_fraction=0.1,
            holdout_fraction=0.2,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["promotion_ready_count"], 0)
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["projected_hits"], 55)
        draft = result["drafts"][0]
        self.assertEqual(draft["evidence_summary"]["source_evidence_schema"], "agentflow.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(draft["evidence_summary"]["sample_count"], 56)
        self.assertEqual(draft["evidence_summary"]["applied_count"], 0)
        self.assertEqual(draft["holdout_fraction"], 0.2)
        rule = draft["proposed_rule"]
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertEqual(rule["graduation"]["cohort_bucket"], "openai_responses/responses/chat/chat/2k_8k_chars/500_2k_tokens")

    def test_projected_replay_ready_refuses_explicit_blockers(self) -> None:
        result = dry_run_cache_promotion_drafts(_projected_replay_ready_report(blocker="invalidation-evidence-missing"))

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"]["draft_count"], 0)
        self.assertEqual(result["omitted"][0]["reason"], "invalidation-evidence-missing")

    def test_rejects_raw_payload_and_cli_emits_noop_for_valid_unready_report(self) -> None:
        raw_report = _promotion_report()
        raw_report["candidates"][0]["raw_prompt"] = f"private prompt {RAW_SECRET}"
        raw_report["candidates"][0]["request_fingerprint"] = f"private fingerprint {RAW_SECRET}"
        result = dry_run_cache_promotion_drafts(raw_report, max_evidence_age_hours=10_000_000)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "raw_payload_rejected")
        self.assertNotIn(RAW_SECRET, json.dumps(result, sort_keys=True))

        stdout = io.StringIO()
        code = cli.cache_promotion_draft_dry_run_cli(
            ["-", "--max-evidence-age-hours", "10000000"],
            stdin=io.StringIO(json.dumps(_promotion_report(ready=False))),
            stdout=stdout,
        )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "no-op")
        self.assertEqual(payload["summary"]["draft_count"], 0)


if __name__ == "__main__":
    unittest.main()
