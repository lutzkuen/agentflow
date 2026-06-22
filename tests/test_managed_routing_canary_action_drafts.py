from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.managed_routing_canary_action_drafts import build_managed_routing_canary_action_drafts


class ManagedRoutingCanaryActionDraftTests(unittest.TestCase):
    def _rollup(
        self,
        *,
        recommendation_id: str,
        decision: str,
        reason_codes: list[str] | None = None,
        applied_count: int = 4,
        holdout_count: int = 4,
        observed_savings_usd: float = 0.12,
        projected_savings_usd: float = 0.18,
    ) -> dict[str, object]:
        reason_codes = reason_codes or []
        return {
            "schema": "agentflow.managed_history_rollup.v1",
            "recommendation_id": recommendation_id,
            "rollup_kind": "managed_routing_pathway_outcomes",
            "source_surface": "openai_responses",
            "app_family": "generic_openai",
            "requested_model": "gpt-5.5",
            "candidate_target_model": "gpt-5.4",
            "phase": "summary",
            "category": "summary",
            "text_bucket": "lt_2k_chars",
            "token_bucket": "lt_1k_tokens",
            "record_count": applied_count + holdout_count,
            "metadata": {
                "schema": "agentflow.managed_routing_pathway_outcome_rollup_metadata.v1",
                "pathway_outcome": {
                    "schema": "tokenclaw.routing_experiment_lifecycle_outcome.v1",
                    "candidate_fingerprint": recommendation_id,
                    "pathway_id": "generic-openai-summary",
                    "source_surface": "openai_responses",
                    "app_family": "generic_openai",
                    "provider_family": "openai",
                    "endpoint": "responses",
                    "category": "summary",
                    "workflow_phase": "summary",
                    "requested_model": "gpt-5.5",
                    "target_model": "gpt-5.4",
                    "text_bucket": "lt_2k_chars",
                    "token_bucket": "lt_1k_tokens",
                    "applied_count": applied_count,
                    "holdout_count": holdout_count,
                    "safety_stop_count": 0,
                    "rollback_count": 0,
                    "error_count": 0,
                    "fallback_count": 0,
                    "retry_count": 0,
                    "observed_savings_usd": observed_savings_usd,
                    "projected_savings_usd": projected_savings_usd,
                    "reason_codes": reason_codes,
                    "local_executor_compatible": True,
                    "metadata_only": True,
                    "feature_only": True,
                },
                "pathway_decision": {
                    "schema": "agentflow.routing_pathway_lifecycle_decision.v1",
                    "decision": decision,
                    "next_action": {
                        "canary": "stage-routing-pathway-canary",
                        "widen": "widen-routing-pathway-canary",
                        "hold": "collect-routing-pathway-applied-holdout-coverage",
                        "rollback": "rollback-routing-pathway-canary",
                    }[decision],
                    "confidence": 0.7,
                    "reason_codes": reason_codes,
                    "inputs": {
                        "applied_count": applied_count,
                        "holdout_count": holdout_count,
                        "observed_savings_usd": observed_savings_usd,
                        "projected_savings_usd": projected_savings_usd,
                    },
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
        }

    def _source(self) -> dict[str, object]:
        return {
            "schema": "agentflow.managed_history_rollups.v1",
            "rollups": [
                self._rollup(recommendation_id="routing-candidate:canary-fixture", decision="canary"),
                self._rollup(
                    recommendation_id="routing-candidate:stale-fixture",
                    decision="hold",
                    reason_codes=["routing-lifecycle-stale-evidence"],
                    applied_count=4,
                    holdout_count=4,
                ),
                self._rollup(
                    recommendation_id="routing-candidate:unsafe-fixture",
                    decision="rollback",
                    reason_codes=["routing-lifecycle-safety-stop", "routing-lifecycle-retry-rate-regression"],
                    applied_count=6,
                    holdout_count=6,
                ),
            ],
        }

    def test_managed_scores_emit_file_backed_canary_draft_and_blocked_actions(self) -> None:
        result = build_managed_routing_canary_action_drafts(self._source())

        self.assertEqual(result["schema"], "tokenclaw.managed_routing_canary_action_drafts.v1")
        self.assertEqual(result["status"], "drafted")
        self.assertEqual(result["summary"]["scored_row_count"], 3)
        self.assertEqual(result["summary"]["action_draft_count"], 1)
        self.assertEqual(result["summary"]["blocked_action_count"], 2)
        self.assertEqual(result["summary"]["routing_apply_action_count"], 0)
        self.assertFalse(result["policy_files_written"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertEqual(result["egress_guard"]["status"], "passed")

        draft = result["actions"][0]
        self.assertEqual(draft["schema"], "tokenclaw.managed_routing_canary_action_draft.v1")
        self.assertEqual(draft["target_local_rule_file"], "routing_canary_policy.yaml")
        self.assertEqual(draft["target_local_policy_section"], "routing.canaries")
        self.assertEqual(draft["source_policy"], "managed-recommended")
        self.assertEqual(draft["canary_fraction"], 0.1)
        self.assertEqual(draft["holdout_fraction"], 0.1)
        self.assertFalse(draft["active_policy_write"])
        self.assertEqual(draft["routing_apply_action_count"], 0)
        self.assertTrue(draft["draft_fingerprint"].startswith("routing-canary-draft:"))
        self.assertEqual(draft["routing_canary_policy_patch"]["target_model"], "gpt-5.4")
        self.assertFalse(draft["routing_canary_policy_patch"]["enabled"])
        self.assertEqual(
            draft["rollback_metadata"]["rollback_action_type"],
            "disable-routing-canary-draft",
        )

        reasons = {row["reason"] for row in result["blocked_actions"]}
        self.assertEqual(reasons, {"stale-managed-routing-score", "unsafe-managed-routing-score"})
        self.assertTrue(all(row["routing_apply_action_count"] == 0 for row in result["blocked_actions"]))

        repeated = build_managed_routing_canary_action_drafts(self._source())
        self.assertEqual(draft["draft_fingerprint"], repeated["actions"][0]["draft_fingerprint"])

        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            '"raw_prompt"',
            '"provider_body"',
            '"request_id"',
            '"session_id"',
            '"cache_key"',
            '"file_path"',
            '"policy_file_contents"',
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cli_reads_managed_score_json(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.json"
            path.write_text(json.dumps(self._source()), encoding="utf-8")
            output = io.StringIO()

            code = cli.managed_routing_canary_action_drafts_cli(
                ["--scores-json", str(path)],
                stdout=output,
            )

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["action_draft_count"], 1)
        self.assertEqual(payload["summary"]["blocked_action_count"], 2)
        self.assertEqual(payload["summary"]["routing_apply_action_count"], 0)
        self.assertEqual(payload["egress_guard"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
