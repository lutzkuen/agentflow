from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenclaw import cli
from tokenclaw.managed_routing_pathway_candidates import (
    build_managed_routing_pathway_shadow_candidates,
)


class ManagedRoutingPathwayCandidateTests(unittest.TestCase):
    def _source(self, *, generated_at: str = "2026-06-20T01:00:00+00:00") -> dict[str, object]:
        return {
            "schema": "agentflow.policy_decision.v1",
            "generated_at": generated_at,
            "routing_pathway_matrix": {
                "schema": "agentflow.routing_pathway_matrix.v1",
                "generated_at": generated_at,
                "status": "recommended",
                "pathways": [
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 1,
                        "pathway_id": "pathway-openai-tool-light",
                        "pathway_type": "adjacent_downroute",
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
                        "sample_count": 322,
                        "compared_count": 51,
                        "pass_rate": 0.94,
                        "estimated_savings_usd": 1.40875,
                        "route_down_probability": 0.91,
                        "suggested_next_action": "shadow",
                        "activation_recommendation": True,
                        "reason_codes": ["routing-pathway-shadow"],
                    },
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 2,
                        "pathway_id": "pathway-codex-summary",
                        "pathway_type": "adjacent_downroute",
                        "source_surface": "codex_turn",
                        "app_family": "codex",
                        "category": "codex-turn",
                        "workflow_phase": "summary",
                        "requested_model": "gpt-5.5",
                        "requested_model_family": "gpt-5",
                        "target_model": "gpt-5.3-codex",
                        "target_model_family": "gpt-5-codex",
                        "text_bucket": "lt_2k_chars",
                        "token_bucket": "lt_500_tokens",
                        "sample_count": 80,
                        "compared_count": 20,
                        "pass_rate": 0.95,
                        "estimated_savings_usd": 0.25,
                        "route_down_probability": 0.88,
                        "suggested_next_action": "canary",
                        "activation_recommendation": True,
                        "reason_codes": ["routing-pathway-canary"],
                    },
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 3,
                        "pathway_id": "pathway-openai-hold",
                        "pathway_type": "aggressive_exploratory",
                        "source_surface": "openai_responses",
                        "app_family": "generic_openai",
                        "category": "chat",
                        "workflow_phase": "unknown",
                        "requested_model": "gpt-5.4",
                        "requested_model_family": "gpt-5",
                        "target_model": "gpt-5-mini",
                        "target_model_family": "gpt-5-mini",
                        "text_bucket": "lt_2k_chars",
                        "token_bucket": "lt_500_tokens",
                        "suggested_next_action": "hold",
                        "activation_recommendation": False,
                        "reason_codes": ["adjacent-path-evidence-missing"],
                    },
                    {
                        "schema": "agentflow.routing_pathway_matrix_entry.v1",
                        "rank": 4,
                        "pathway_id": "pathway-anthropic-unsupported",
                        "pathway_type": "adjacent_downroute",
                        "source_surface": "anthropic_messages",
                        "app_family": "claude_code",
                        "category": "chat",
                        "workflow_phase": "summary",
                        "requested_model": "claude-sonnet-4-6",
                        "requested_model_family": "claude-sonnet",
                        "target_model": "claude-haiku-4-5-20251001",
                        "target_model_family": "claude-haiku",
                        "text_bucket": "lt_2k_chars",
                        "token_bucket": "lt_500_tokens",
                        "suggested_next_action": "shadow",
                        "activation_recommendation": True,
                    },
                ],
            },
        }

    def test_policy_decision_matrix_becomes_review_only_openai_and_codex_shadow_candidates(self) -> None:
        now = datetime(2026, 6, 20, 2, 0, tzinfo=timezone.utc)
        result = build_managed_routing_pathway_shadow_candidates(self._source(), now=now)

        self.assertEqual(result["schema"], "agentflow.managed_routing_pathway_shadow_candidates.v1")
        self.assertEqual(result["status"], "review-only")
        self.assertEqual(result["summary"]["matrix_row_count"], 4)
        self.assertEqual(result["summary"]["accepted_count"], 2)
        self.assertEqual(result["summary"]["blocked_count"], 1)
        self.assertEqual(result["summary"]["omitted_count"], 1)
        self.assertEqual(result["summary"]["codex_candidate_count"], 1)
        self.assertEqual(result["summary"]["generic_openai_candidate_count"], 2)
        self.assertFalse(result["policy_files_written"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertEqual(result["egress_guard"]["status"], "passed")
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["provider_bodies_included"])

        openai = next(row for row in result["accepted"] if row["source_surface"] == "openai_responses")
        codex = next(row for row in result["accepted"] if row["source_surface"] == "codex_turn")
        self.assertEqual(openai["local_executor_compatibility"]["local_executor"], "openai-routing-shadow-candidate")
        self.assertEqual(codex["local_executor_compatibility"]["local_executor"], "codex-routing-shadow-candidate")
        self.assertNotEqual(openai["group_ref"], codex["group_ref"])
        self.assertEqual(codex["group_key"]["app_family"], "codex")
        self.assertTrue(openai["candidate_fingerprint"].startswith("routing-pathway-candidate:"))

        repeated = build_managed_routing_pathway_shadow_candidates(self._source(), now=now)
        self.assertEqual(
            result["accepted"][0]["candidate_fingerprint"],
            repeated["accepted"][0]["candidate_fingerprint"],
        )

        blocked = result["blocked"][0]
        self.assertEqual(blocked["reason"], "routing-pathway-hold")
        omitted = result["omitted"][0]
        self.assertEqual(omitted["reason"], "unsupported-local-routing-executor")

    def test_stale_matrix_rows_are_classified_without_policy_writes(self) -> None:
        result = build_managed_routing_pathway_shadow_candidates(
            self._source(generated_at="2026-06-15T00:00:00+00:00"),
            now=datetime(2026, 6, 20, 2, 0, tzinfo=timezone.utc),
            stale_after_hours=72,
        )

        self.assertEqual(result["summary"]["stale_count"], 4)
        self.assertEqual(result["summary"]["accepted_count"], 0)
        self.assertFalse(result["policy_files_written"])
        self.assertTrue(all(row["reason"] == "stale-routing-pathway-matrix" for row in result["stale"]))

    def test_cli_reads_decision_json_and_emits_metadata_only_report(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision.json"
            path.write_text(json.dumps(self._source()), encoding="utf-8")
            output = io.StringIO()

            code = cli.managed_routing_pathway_candidates_cli(
                ["--decision-json", str(path), "--preview-stale-after-hours", "72"],
                stdout=output,
            )

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["accepted_count"], 2)
        self.assertEqual(payload["egress_guard"]["status"], "passed")
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in ('"raw_prompt"', '"provider_body"', '"request_id"', '"session_id"', '"cache_key"', '"file_path"'):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
