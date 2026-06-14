from __future__ import annotations

import io
import json
import unittest

from agentflow_proxy import cli
from agentflow_proxy.crunch_promotion_drafts import SCHEMA, dry_run_crunch_promotion_drafts
from agentflow_proxy.local_promotion_candidates import build_local_promotion_candidates_from_reports


RAW_SECRET = "raw-crunch-promotion-secret"


def _promotion_report(*, ready: bool = True) -> dict:
    report = build_local_promotion_candidates_from_reports(
        {
            "anthropic_thinking_compaction_impact": {
                "schema": "agentflow.anthropic_thinking_compaction_impact.v1",
                "status": "matched",
                "candidates": [
                    {
                        "canary_impact_decision": "widen" if ready else "remain-staged",
                        "verdict": "widen" if ready else "keep-holdout",
                        "reason_codes": ["impact-positive"] if ready else ["insufficient-holdout-samples"],
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "requested_model_family": "sonnet",
                        "routed_model_family": "sonnet",
                        "stream": True,
                        "first_observed_at": "2026-06-14T00:00:00+00:00",
                        "last_observed_at": "2999-01-01T00:00:00+00:00",
                        "cohorts": {
                            "applied": {
                                "count": 6 if ready else 0,
                                "tokens_saved_est": 12000,
                                "saved_chars": 48000,
                                "gross_savings_usd": 0.25,
                            },
                            "holdout": {
                                "count": 3 if ready else 0,
                                "planned_saved_tokens": 6000,
                                "planned_saved_chars": 24000,
                                "projected_savings_usd": 0.18,
                            },
                            "safety_stop": {"count": 0},
                            "skipped": {"count": 2},
                        },
                        "observed_saved_tokens": 12000,
                        "observed_saved_usd": 0.25,
                        "projected_saved_tokens": 18000,
                        "projected_saved_usd": 0.43,
                        "avg_crunch_ratio": 0.22,
                        "candidate_id": f"source-candidate-{RAW_SECRET}",
                        "session_id": f"source-session-{RAW_SECRET}",
                        "request_id": f"source-request-{RAW_SECRET}",
                    }
                ],
            },
            "request_shape_rollups": {},
            "cache_impact": {},
            "claude_routing_impact": {},
            "openai_routing_report": {},
        }
    )
    report["candidates"] = [candidate for candidate in report["candidates"] if candidate["action_family"] == "crunch"]
    return report


class CrunchPromotionDraftTests(unittest.TestCase):
    def test_drafts_guarded_repeated_context_crunch_rule_from_promotion_candidate(self) -> None:
        result = dry_run_crunch_promotion_drafts(
            _promotion_report(),
            initial_canary_fraction=0.12,
            holdout_fraction=0.08,
            max_evidence_age_hours=10_000_000,
        )

        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["draft_count"], 1)
        self.assertEqual(result["summary"]["omitted_count"], 0)
        self.assertFalse(result["wrote_active_policy_files"])
        self.assertFalse(result["provider_calls_made"])
        draft = result["drafts"][0]
        self.assertEqual(draft["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(draft["target_local_policy_section"], "anthropic_thinking_history_compaction.rules")
        self.assertEqual(draft["activation_fraction"], 0.12)
        self.assertEqual(draft["holdout_fraction"], 0.08)
        rule = draft["proposed_rule"]
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["conditions"]["source_surface"], "anthropic_messages")
        self.assertEqual(rule["conditions"]["category"], "tool-result")
        self.assertEqual(rule["conditions"]["phase"], "thinking")
        self.assertEqual(rule["conditions"]["model_pattern"], "sonnet")
        self.assertEqual(rule["action"]["type"], "compact_thinking_history_block")
        self.assertTrue(rule["action"]["preserve_tool_protocol"])
        self.assertEqual(rule["canary"]["canary_fraction"], 0.12)
        self.assertEqual(rule["canary"]["holdout_fraction"], 0.08)
        self.assertTrue(rule["safety_stop"]["enabled"])
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["projected_saved_tokens"], 18000)
        self.assertEqual(rule["promotion"]["dry_run_impact_estimate"]["observed_savings_usd"], 0.25)
        self.assertEqual(rule["promotion"]["rollback_metadata"]["rollback_action_type"], "disable_rule")
        self.assertFalse(rule["promotion"]["privacy"]["raw_prompts_included"])

        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn(RAW_SECRET, rendered)
        self.assertNotIn('"session_id"', rendered)
        self.assertNotIn('"request_id"', rendered)

    def test_no_ops_when_evidence_not_ready_or_stale(self) -> None:
        not_ready = dry_run_crunch_promotion_drafts(
            _promotion_report(ready=False),
            max_evidence_age_hours=10_000_000,
        )
        self.assertFalse(not_ready["ok"])
        self.assertEqual(not_ready["status"], "no-op")
        self.assertEqual(not_ready["summary"]["draft_count"], 0)
        self.assertEqual(not_ready["omitted"][0]["reason"], "insufficient-holdout-samples")

        stale_report = _promotion_report()
        stale_report["candidates"][0]["last_observed_at"] = "2020-01-01T00:00:00+00:00"
        stale = dry_run_crunch_promotion_drafts(stale_report, max_evidence_age_hours=1)
        self.assertFalse(stale["ok"])
        self.assertEqual(stale["omitted"][0]["reason"], "stale-evidence")

    def test_rejects_raw_payload_and_cli_emits_noop_for_valid_unready_report(self) -> None:
        raw_report = _promotion_report()
        raw_report["candidates"][0]["raw_prompt"] = f"private prompt {RAW_SECRET}"
        raw_report["candidates"][0]["messages"] = [{"content": f"private message {RAW_SECRET}"}]
        result = dry_run_crunch_promotion_drafts(raw_report, max_evidence_age_hours=10_000_000)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "raw_payload_rejected")
        self.assertNotIn(RAW_SECRET, json.dumps(result, sort_keys=True))

        stdout = io.StringIO()
        code = cli.crunch_promotion_draft_dry_run_cli(
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
