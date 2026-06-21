from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tokenclaw import cli
from tokenclaw.crunch_blocker_outcomes import build_crunch_blocker_outcomes_report
from tokenclaw.store import SQLiteStore, stable_json, utc_now


_BLOCKER_REVIEW_SCHEMA = "tokenclaw.promotion_blocker_recommendation_review.v1"
_CANDIDATE_SCHEMA = "tokenclaw.promotion_blocker_review_candidate.v1"


def _privacy_clean() -> dict:
    return {
        "metadata_only": True,
        "feature_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "absolute_paths_included": False,
        "provider_calls_made": False,
        "wrote_local_policy_files": False,
        "managed_enforced": False,
    }


def _blocker_candidate(
    *,
    candidate_family: str,
    recommendation_type: str,
    next_action: str,
    status: str = "recommended",
    blocker_reason_codes: list[str] | None = None,
    projected_savings_usd: float = 0.0,
    local_action_family: str = "crunch",
) -> dict:
    return {
        "schema": _CANDIDATE_SCHEMA,
        "recommendation_id": f"rec-{candidate_family}-{uuid.uuid4().hex[:8]}",
        "rank": 1,
        "status": status,
        "recommendation_type": recommendation_type,
        "local_action_family": local_action_family,
        "candidate_family": candidate_family,
        "source_surface": "anthropic_proxy",
        "blocker_family": "crunch-blocker",
        "blocker_reason_codes": blocker_reason_codes or [],
        "blocker_count": len(blocker_reason_codes or []),
        "next_action": next_action,
        "confidence": 0.75,
        "projected_savings_usd": projected_savings_usd,
        "no_op_reasons": [],
        "required_local_review": True,
        "read_only": True,
        "feature_only": True,
        "locally_executed": True,
        "provider_forwarding": False,
        "server_content_processing": False,
        "managed_enforced": False,
        "privacy": _privacy_clean(),
    }


def _fixture_promotion_blocker_review() -> dict:
    return {
        "schema": _BLOCKER_REVIEW_SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "read_only": True,
        "candidates": [
            _blocker_candidate(
                candidate_family="repeated-context-crunch",
                recommendation_type="dry-run",
                next_action="stage-local-dry-run",
                projected_savings_usd=0.05,
            ),
            _blocker_candidate(
                candidate_family="anthropic-thinking-history-compaction",
                recommendation_type="canary-activate",
                next_action="stage-local-canary",
                projected_savings_usd=0.03,
            ),
            _blocker_candidate(
                candidate_family="instruction-dedup",
                recommendation_type="noop",
                next_action="keep-blocked",
                status="noop",
                blocker_reason_codes=["insufficient-repeat-evidence", "non-positive-projection"],
                projected_savings_usd=0.0,
            ),
            _blocker_candidate(
                candidate_family="custom-crunch-variant",
                recommendation_type="noop",
                next_action="keep-blocked",
                status="noop",
                blocker_reason_codes=["unsupported-crunch-family"],
                projected_savings_usd=0.0,
            ),
        ],
        "groups": [],
        "omitted_actions": [],
        "privacy": _privacy_clean(),
    }


class CrunchBlockerOutcomesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        text_chars: int = 10_000,
        input_tokens: int = 2500,
        crunch_tokens_saved: int = 0,
        cost: float = 0.02,
    ) -> str:
        call_id = str(uuid.uuid4())
        self.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=300,
            input_tokens_est=input_tokens,
            output_tokens_est=100,
            actual_input_tokens=input_tokens,
            actual_output_tokens=100,
            cost_est_usd=cost,
            cost_baseline_usd=cost,
            crunch_json=stable_json({
                "changed": crunch_tokens_saved > 0,
                "tokens_saved_est": crunch_tokens_saved,
                "savings_chars": crunch_tokens_saved * 4,
            }),
            routing_json=stable_json({
                "enabled": False,
                "provider": "anthropic",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "text_chars": text_chars,
                "has_tools": False,
                "category": "chat",
            }),
            cache_json=stable_json({"status": "miss", "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=stable_json({"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "raw prompt must not leak in crunch blocker test"}]}),
            response_json=stable_json({"content": "raw response must not leak in crunch blocker test"}),
            session_id="raw-crunch-blocker-session-id-must-not-leak",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
            source_surface="anthropic_proxy",
            endpoint="messages",
            requested_model_family="claude-sonnet",
            routed_model_family="claude-sonnet",
        )
        return call_id

    def test_crunch_blocker_outcomes_four_family_fixture_produces_lifecycle_counts(self) -> None:
        for _ in range(3):
            self._log_call(text_chars=12_000, input_tokens=3_000)

        review = _fixture_promotion_blocker_review()
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )

        self.assertEqual(report["schema"], "tokenclaw.crunch_blocker_outcomes.v1")
        summary = report["summary"]
        self.assertGreaterEqual(summary["dry_run_count"], 1)
        self.assertGreaterEqual(summary["canary_count"], 1)
        self.assertGreaterEqual(summary["no_op_count"], 1)
        self.assertGreater(summary["outcome_count"], 0)

        lifecycle_map = {row["lifecycle"]: row["count"] for row in report["lifecycle_breakdown"]}
        self.assertIn("dry-run", lifecycle_map)
        self.assertIn("canary", lifecycle_map)
        self.assertIn("no-op", lifecycle_map)

        family_map = {row["crunch_family"]: row["count"] for row in report["family_breakdown"]}
        self.assertIn("repeated-context", family_map)
        self.assertIn("thinking-compaction", family_map)
        self.assertIn("instruction-dedup", family_map)
        self.assertIn("unsupported", family_map)

    def test_crunch_blocker_outcomes_projected_savings_from_recommendations(self) -> None:
        review = _fixture_promotion_blocker_review()
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )
        self.assertGreater(report["summary"]["projected_savings_usd"], 0.0)

    def test_crunch_blocker_outcomes_blocker_reason_codes_present(self) -> None:
        review = _fixture_promotion_blocker_review()
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )
        reason_values = {row["value"] for row in report["reason_breakdown"]}
        self.assertIn("insufficient-repeat-evidence", reason_values)
        self.assertIn("non-positive-projection", reason_values)
        self.assertIn("unsupported-crunch-family", reason_values)

    def test_crunch_blocker_outcomes_metadata_only_no_raw_fields(self) -> None:
        for _ in range(3):
            self._log_call(text_chars=12_000)
        review = _fixture_promotion_blocker_review()
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak in crunch blocker test",
            "raw response must not leak in crunch blocker test",
            "raw-crunch-blocker-session-id-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)

        privacy = report["privacy"]
        self.assertFalse(privacy["raw_prompts_included"])
        self.assertFalse(privacy["raw_provider_bodies_included"])
        self.assertFalse(privacy["cache_keys_included"])
        self.assertFalse(privacy["request_ids_included"])
        self.assertFalse(privacy["session_ids_included"])
        self.assertFalse(privacy["individual_candidate_ids_included"])
        self.assertFalse(report["provider_calls_made"])
        self.assertFalse(report["managed_server_calls_made"])
        self.assertFalse(report["source_reports"]["individual_candidate_ids_included"])
        self.assertFalse(report["source_reports"]["raw_source_reports_included"])

    def test_crunch_blocker_outcomes_empty_store_returns_no_crunch_outcomes(self) -> None:
        report = build_crunch_blocker_outcomes_report(self.store, rollup_limit=20)
        self.assertEqual(report["schema"], "tokenclaw.crunch_blocker_outcomes.v1")
        self.assertEqual(report["status"], "no-crunch-outcomes")
        self.assertEqual(report["summary"]["outcome_count"], 0)

    def test_crunch_blocker_outcomes_safety_stop_lifecycle_detected(self) -> None:
        review = {
            "schema": _BLOCKER_REVIEW_SCHEMA,
            "generated_at": utc_now(),
            "ok": True,
            "read_only": True,
            "candidates": [
                _blocker_candidate(
                    candidate_family="repeated-context-crunch",
                    recommendation_type="safety-stop",
                    next_action="safety-stop",
                    blocker_reason_codes=["safety-stop-observed"],
                    projected_savings_usd=0.0,
                ),
            ],
            "groups": [],
            "omitted_actions": [],
            "privacy": _privacy_clean(),
        }
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        self.assertEqual(report["top_next_action"], "investigate-crunch-safety-stop")

    def test_crunch_blocker_outcomes_opportunity_dry_run_cohorts_contribute_lifecycle(self) -> None:
        for _ in range(3):
            self._log_call(text_chars=12_000, input_tokens=3_500)

        report = build_crunch_blocker_outcomes_report(self.store, rollup_limit=20)

        self.assertGreater(report["summary"]["crunch_opportunity_cohort_count"], 0)
        lifecycle_map = {row["lifecycle"]: row["count"] for row in report["lifecycle_breakdown"]}
        self.assertIn("dry-run", lifecycle_map)

    def test_crunch_blocker_outcomes_cli_emits_report(self) -> None:
        output = io.StringIO()
        exit_code = cli.crunch_blocker_outcomes_cli(
            ["--db", self.db_path, "--rollup-limit", "20"],
            stdout=output,
        )
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema"], "tokenclaw.crunch_blocker_outcomes.v1")

    def test_crunch_blocker_outcomes_candidate_without_savings_does_not_contribute_projected(self) -> None:
        review = {
            "schema": _BLOCKER_REVIEW_SCHEMA,
            "generated_at": utc_now(),
            "ok": True,
            "read_only": True,
            "candidates": [
                _blocker_candidate(
                    candidate_family="instruction-dedup",
                    recommendation_type="noop",
                    next_action="keep-blocked",
                    status="noop",
                    blocker_reason_codes=["non-positive-projection"],
                    projected_savings_usd=0.0,
                ),
            ],
            "groups": [],
            "omitted_actions": [],
            "privacy": _privacy_clean(),
        }
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )
        self.assertEqual(report["summary"]["projected_savings_usd"], 0.0)

    def test_crunch_blocker_outcomes_non_crunch_candidates_are_ignored(self) -> None:
        review = {
            "schema": _BLOCKER_REVIEW_SCHEMA,
            "generated_at": utc_now(),
            "ok": True,
            "read_only": True,
            "candidates": [
                _blocker_candidate(
                    candidate_family="model-routing",
                    recommendation_type="canary-activate",
                    next_action="stage-local-canary",
                    local_action_family="routing",
                    projected_savings_usd=0.10,
                ),
            ],
            "groups": [],
            "omitted_actions": [],
            "privacy": _privacy_clean(),
        }
        report = build_crunch_blocker_outcomes_report(
            self.store,
            rollup_limit=20,
            promotion_blocker_review=review,
        )
        self.assertEqual(report["summary"]["promotion_blocker_crunch_candidate_count"], 0)
        self.assertEqual(report["summary"]["canary_count"], 0)
