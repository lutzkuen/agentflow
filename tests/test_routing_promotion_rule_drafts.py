from __future__ import annotations

import io
import json
import unittest

from tokenclaw import cli
from tokenclaw.local_promotion_candidates import build_local_promotion_candidates_from_reports
from tokenclaw.routing_promotion_rule_drafts import SCHEMA, dry_run_routing_promotion_drafts


RAW_SECRET = "raw-routing-promotion-secret"


def _promotion_report(
    *,
    ready: bool = True,
    target_model: str | None = "claude-haiku-4-5-20251001",
    workflow_phase: str = "tool-execution",
    safety_stop: int = 0,
    stale: bool = False,
    reason_codes: list[str] | None = None,
    error_count: int = 0,
) -> dict:
    if reason_codes is None:
        reason_codes = ["target-savings-met", "canary-full-coverage"] if ready else ["insufficient-holdout-samples"]
    applied = 6 if ready else 2
    holdout = 3 if ready else 0
    report = build_local_promotion_candidates_from_reports(
        {
            "claude_routing_impact": {
                "schema": "tokenclaw.claude_canary_impact.v1",
                "status": "matched",
                "candidates": [
                    {
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": workflow_phase,
                        "workflow_phase_confidence": "medium",
                        "stream": True,
                        "original_model": "claude-sonnet-4-6",
                        "candidate_target_model": target_model,
                        "policy_id": "local-anthropic-routing-canary",
                        "canary_fraction": 0.5,
                        "holdout_fraction": 0.5,
                        "sample_count": applied + holdout + safety_stop,
                        "cohort_counts": {
                            "canary_applied": applied,
                            "canary_holdout": holdout,
                            "safety_stopped": safety_stop,
                        },
                        "cohort_metrics": {
                            "canary_applied": {
                                "count": applied,
                                "error_count": error_count,
                                "retry_count": 0,
                                "fallback_count": 0,
                            },
                            "canary_holdout": {
                                "count": holdout,
                                "error_count": 0,
                                "retry_count": 0,
                                "fallback_count": 0,
                            },
                            "safety_stopped": {
                                "count": safety_stop,
                                "error_count": 0,
                                "retry_count": 0,
                                "fallback_count": 0,
                            },
                        },
                        "applied_vs_holdout_deltas": {
                            "applied_minus_holdout_error_rate": 0.0,
                            "applied_minus_holdout_retry_rate": 0.0,
                            "applied_minus_holdout_fallback_rate": 0.0,
                            "applied_minus_holdout_latency_avg_ms": -10,
                        },
                        "observed_savings_usd": 0.42 if ready else 0.0,
                        "projected_savings_usd": 0.9 if ready else 0.0,
                        "oldest_observed_at": "2026-06-14T00:00:00+00:00",
                        "latest_observed_at": "2026-06-14T01:00:00+00:00",
                        "stale_evidence": {"stale": stale},
                        "verdict": "promote" if ready else "needs_more_samples",
                        "reason_codes": reason_codes,
                        "stripped_param_counts": [{"value": "thinking", "count": 2}],
                        "safety_skip_counts": [],
                        "raw_request": f"must not leak {RAW_SECRET}",
                    }
                ],
            },
            "request_shape_rollups": {},
            "anthropic_thinking_compaction_impact": {},
            "cache_impact": {},
            "openai_routing_report": {},
        }
    )
    report["candidates"] = [candidate for candidate in report["candidates"] if candidate["action_family"] == "routing"]
    return report


class RoutingPromotionRuleDraftTests(unittest.TestCase):
    def test_drafts_guarded_anthropic_sonnet_to_haiku_rule_from_lifecycle_candidate(self) -> None:
        result = dry_run_routing_promotion_drafts(
            _promotion_report(),
            initial_canary_fraction=0.2,
            holdout_fraction=0.15,
            max_evidence_age_hours=10_000_000,
        )

        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["routing_candidate_count"], 1)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        draft = result["drafts"][0]
        self.assertEqual(draft["target_local_rule_file"], "routing_rules.yaml")
        self.assertEqual(draft["target_local_policy_section"], "routing.rules")
        self.assertEqual(draft["activation_fraction"], 0.2)
        self.assertEqual(draft["holdout_fraction"], 0.15)
        rule = draft["proposed_rule"]
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "local-promoted")
        self.assertEqual(rule["conditions"]["model_pattern"], "sonnet")
        self.assertEqual(rule["conditions"]["category"], "tool-result")
        self.assertEqual(rule["conditions"]["workflow_phase"], "tool-execution")
        self.assertEqual(rule["conditions"]["workflow_phase_confidence_gte"], "medium")
        self.assertTrue(rule["conditions"]["has_tools"])
        self.assertTrue(rule["conditions"]["stream"])
        self.assertEqual(rule["action"]["route_to"], "claude-haiku-4-5-20251001")
        self.assertTrue(rule["metadata"]["promoted_from_canary"])
        self.assertTrue(rule["metadata"]["safety_gates"]["block_thinking_history"])
        self.assertTrue(rule["metadata"]["safety_gates"]["strip_model_incompatible_params"])
        self.assertTrue(rule["metadata"]["safety_gates"]["fallback_to_requested_on_rate_limit"])
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["applied_count"], 6)
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["holdout_count"], 3)
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["observed_savings_usd"], 0.42)
        self.assertEqual(rule["promotion"]["rollback_metadata"]["rollback_action_type"], "disable_rule")

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(RAW_SECRET, rendered)
        self.assertNotIn('"raw_request"', rendered)
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])

    def test_no_ops_for_unsafe_missing_stale_or_regressed_routing_candidates(self) -> None:
        cases = (
            ({"workflow_phase": "thinking", "reason_codes": ["target-savings-met"]}, "thinking-routing-guard"),
            ({"target_model": None}, "missing-routed-model-metadata"),
            ({"stale": True}, "stale-evidence"),
            ({"reason_codes": ["error-rate-regression"]}, "error-rate-regression"),
            ({"ready": False}, "insufficient-holdout-samples"),
            ({"safety_stop": 1, "reason_codes": ["safety-stop-observed"]}, "safety-stop-observed"),
        )
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                result = dry_run_routing_promotion_drafts(
                    _promotion_report(**kwargs),
                    max_evidence_age_hours=10_000_000,
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "no-op")
                self.assertEqual(result["summary"]["draft_count"], 0)
                self.assertEqual(result["omitted"][0]["reason"], expected)

    def test_rejects_raw_payload_and_cli_emits_noop_for_valid_unready_report(self) -> None:
        raw_report = _promotion_report()
        raw_report["candidates"][0]["raw_prompt"] = f"private prompt {RAW_SECRET}"
        raw_report["candidates"][0]["session_id"] = f"private session {RAW_SECRET}"
        result = dry_run_routing_promotion_drafts(raw_report, max_evidence_age_hours=10_000_000)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "raw_payload_rejected")
        self.assertNotIn(RAW_SECRET, json.dumps(result, sort_keys=True))

        stdout = io.StringIO()
        code = cli.routing_promotion_draft_dry_run_cli(
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
