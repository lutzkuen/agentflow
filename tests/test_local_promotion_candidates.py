from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from tokenclaw import cli
from tokenclaw.local_promotion_candidates import (
    SCHEMA,
    build_local_promotion_candidates_from_reports,
)
from tokenclaw.store import SQLiteStore


def _source_reports() -> dict:
    return {
        "cache_impact": {
            "schema": "agentflow.openai_cache_replay_impact.v1",
            "status": "matched",
            "candidates": [
                {
                    "verdict": "widen",
                    "reason_codes": ["target-savings-met"],
                    "source_surface": "openai_responses",
                    "endpoint": "responses",
                    "category": "chat",
                    "sample_count": 4,
                    "applied_count": 3,
                    "holdout_count": 1,
                    "safety_stop_count": 0,
                    "projected_hits": 12,
                    "actual_hits": 2,
                    "actual_saved_cost_usd": 0.12,
                    "projected_saved_usd": 0.3,
                    "candidate_id": "cache-candidate-id-must-not-leak",
                    "request_id": "cache-request-id-must-not-leak",
                    "cache_key": "cache-key-must-not-leak",
                }
            ],
        },
        "request_shape_rollups": {
            "schema": "agentflow.request_shape_rollups.v1",
            "crunch_canary_impact": {
                "schema": "agentflow.request_shape_crunch_canary_impact.v1",
                "status": "widen-ready",
                "candidates": [
                    {
                        "verdict": "widen-ready",
                        "reason_codes": ["target-savings-met"],
                        "provider": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "observed_count": 3,
                        "applied_count": 2,
                        "holdout_count": 1,
                        "safety_stop_count": 0,
                        "saved_tokens": 900,
                        "saved_usd": 0.2,
                        "projected_saved_usd": 0.2,
                        "candidate_id": "crunch-candidate-id-must-not-leak",
                        "session_id": "crunch-session-id-must-not-leak",
                    }
                ],
            },
        },
        "claude_routing_impact": {
            "schema": "agentflow.claude_canary_impact.v1",
            "status": "matched",
            "candidates": [
                {
                    "verdict": "widen",
                    "reason_codes": ["target-savings-met"],
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "requested_model": "claude-sonnet-4-6",
                    "target_model": "claude-haiku-4-5-20251001",
                    "sample_count": 6,
                    "cohort_counts": {"canary_applied": 4, "canary_holdout": 2, "safety_stopped": 0},
                    "observed_savings_usd": 0.18,
                    "projected_savings_usd": 0.5,
                    "candidate_id": "routing-candidate-id-must-not-leak",
                    "messages": [{"content": "raw prompt must not leak"}],
                }
            ],
        },
        "openai_routing_report": {
            "schema": "agentflow.openai_routing_opportunity.v1",
            "status": "matched",
            "candidates": [],
        },
    }


class LocalPromotionCandidatesTests(unittest.TestCase):
    def test_ranks_cache_crunch_and_routing_candidates_without_raw_identifiers(self) -> None:
        report = build_local_promotion_candidates_from_reports(_source_reports())

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["summary"]["candidate_count"], 3)
        self.assertEqual(report["summary"]["promotion_ready_count"], 3)
        families = {candidate["action_family"] for candidate in report["candidates"]}
        self.assertEqual(families, {"cache", "crunch", "routing"})
        targets = {candidate["action_family"]: candidate["target_local_rule_file"] for candidate in report["candidates"]}
        self.assertEqual(targets["cache"], "cache_rules.yaml")
        self.assertEqual(targets["crunch"], "crunch_rules.yaml")
        self.assertEqual(targets["routing"], "routing_rules.yaml")
        self.assertTrue(all(candidate["next_action"].startswith("promote-") for candidate in report["candidates"]))

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "cache-candidate-id-must-not-leak",
            "crunch-candidate-id-must-not-leak",
            "routing-candidate-id-must-not-leak",
            "cache-request-id-must-not-leak",
            "cache-key-must-not-leak",
            "crunch-session-id-must-not-leak",
            "raw prompt must not leak",
            '"candidate_id"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(report["privacy"]["individual_candidate_ids_included"])
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["managed_server_calls_made"])

    def test_missing_holdout_and_safety_stop_are_not_promotion_ready(self) -> None:
        reports = _source_reports()
        reports["cache_impact"]["candidates"][0]["holdout_count"] = 0
        reports["request_shape_rollups"]["crunch_canary_impact"]["candidates"][0]["safety_stop_count"] = 1

        report = build_local_promotion_candidates_from_reports(reports)
        by_family = {candidate["action_family"]: candidate for candidate in report["candidates"]}

        self.assertFalse(by_family["cache"]["promotion_ready"])
        self.assertEqual(by_family["cache"]["readiness_state"], "blocked")
        self.assertIn("missing-holdout-evidence", by_family["cache"]["blocker_codes"])
        self.assertEqual(by_family["cache"]["next_action"], "collect-cache-holdout-evidence")

        self.assertFalse(by_family["crunch"]["promotion_ready"])
        self.assertEqual(by_family["crunch"]["readiness_state"], "blocked")
        self.assertIn("safety-stop-observed", by_family["crunch"]["blocker_codes"])
        self.assertEqual(by_family["crunch"]["next_action"], "review-crunch-safety-stop")

    def test_cli_emits_schema_for_empty_local_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "agentflow.sqlite3")
            store = SQLiteStore(db_path)
            store.conn.close()
            stdout = io.StringIO()

            rc = cli.local_promotion_candidates_cli(["--db", db_path, "--limit", "10"], stdout=stdout)

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], SCHEMA)
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["provider_calls_made"])


if __name__ == "__main__":
    unittest.main()
