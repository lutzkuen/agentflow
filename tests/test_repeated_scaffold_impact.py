from __future__ import annotations

import io
import json
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.repeated_scaffold_impact import build_repeated_scaffold_impact_report
from agentflow_proxy.store import SQLiteStore, stable_json


class RepeatedScaffoldImpactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _log_call(
        self,
        *,
        cohort: str,
        created_at: str = "2026-06-12T00:00:00+00:00",
        status_code: int = 200,
        retry_count: int = 0,
        latency_ms: int = 100,
        tokens_saved: int = 1000,
        saved_chars: int | None = None,
        provider: str = "anthropic",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        requested_model: str = "claude-sonnet-4-6",
        model_family: str = "sonnet",
        category: str = "tool-result",
        workflow_phase: str = "tool-execution",
        rule_id: str = "reviewed-provider-scaffold",
        candidate_id: str = "candidate-provider-123",
        safety_stop: bool = False,
    ) -> None:
        saved_chars = tokens_saved * 4 if saved_chars is None else saved_chars
        applied = cohort == "canary_applied"
        holdout = cohort == "canary_holdout"
        safety = cohort == "safety_stop" or safety_stop
        canary = {
            "enabled": True,
            "selected": applied,
            "cohort": cohort,
            "fraction": 0.10,
            "unit": "request_fingerprint",
            "reason": "canary_holdout" if holdout else None,
        }
        rule = {
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "enabled": True,
            "policy_source": "managed-recommended",
            "matched_pattern_count": 1,
            "applied_count": 1 if applied else 0,
            "holdout_count": 1 if holdout else 0,
            "saved_chars": saved_chars if applied else 0,
            "canary": canary,
            "skip_reasons": [],
        }
        if holdout:
            rule["skip_reasons"] = [{"reason": "canary_holdout", "count": 1, "canary": canary}]
        if safety:
            rule["safety_stop"] = {"tripped": True, "reason_codes": ["rollback-error-rate"]}
        crunch = {
            "changed": applied,
            "repeated_provider_scaffolding": {
                "schema": "agentflow.repeated_provider_scaffolding.v1",
                "enabled": True,
                "status": "applied" if applied else "skipped",
                "reason": "repeated-provider-scaffolding-crunched" if applied else "canary_holdout",
                "policy_source": "managed-recommended",
                "category": category,
                "saved_chars": saved_chars if applied else 0,
                "tokens_saved_est": tokens_saved if applied else 0,
                "rules": [rule],
                "raw_text_included": False,
                "raw_hashes_included": False,
            },
        }
        routing = {
            "provider": provider,
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_chars": 48_000,
            "has_tools": True,
        }
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=created_at,
            path="/v1/messages" if provider == "anthropic" else "/v1/responses",
            requested_model=requested_model,
            routed_model=requested_model,
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=12_000,
            output_tokens_est=100,
            actual_input_tokens=12_000,
            actual_output_tokens=100,
            cost_est_usd=0.04,
            cost_baseline_usd=0.04,
            crunch_json=stable_json(crunch),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error="raw error text must not leak" if status_code >= 400 else None,
            request_json=stable_json({"raw": "prompt must not leak", "path": "/tmp/secret.py"}),
            response_json=stable_json({"text": "raw response must not leak"}),
            session_id="raw-session-must-not-leak",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family=model_family,
            routed_model_family=model_family,
        )

    def test_promote_verdict_and_cli_are_metadata_only(self) -> None:
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1200)
        self._log_call(cohort="canary_holdout", tokens_saved=0)

        report = build_repeated_scaffold_impact_report(self.store, limit=20)

        self.assertEqual(report["schema"], "agentflow.repeated_scaffold_impact.v1")
        self.assertEqual(report["summary"]["observed_repeated_scaffold_metadata_row_count"], 3)
        self.assertEqual(report["candidates"][0]["verdict"], "promote")
        self.assertEqual(report["candidates"][0]["next_action"], "widen_repeated_scaffold_crunch_canary")
        self.assertIn("target-savings-met", report["candidates"][0]["reason_codes"])
        self.assertEqual(report["candidates"][0]["provider"], "anthropic")
        self.assertEqual(report["candidates"][0]["source_surface"], "anthropic_messages")
        self.assertEqual(report["candidates"][0]["model_tier"], "sonnet")

        output = io.StringIO()
        exit_code = cli.repeated_scaffold_impact_cli(["--db", self.db_path, "--limit", "20"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "agentflow.repeated_scaffold_impact.v1")

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("prompt must not leak", rendered)
        self.assertNotIn("raw response must not leak", rendered)
        self.assertNotIn("raw-session-must-not-leak", rendered)
        self.assertNotIn("/tmp/secret.py", rendered)
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["pattern_hashes_included"])
        self.assertFalse(payload["privacy"]["request_fingerprints_included"])

    def test_need_more_samples_when_holdout_is_missing(self) -> None:
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1000)

        report = build_repeated_scaffold_impact_report(self.store, limit=20)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "need-more-samples")
        self.assertIn("insufficient-holdout-samples", candidate["reason_codes"])

    def test_hold_for_non_positive_savings(self) -> None:
        self._log_call(cohort="canary_applied", tokens_saved=0)
        self._log_call(cohort="canary_applied", tokens_saved=0)
        self._log_call(cohort="canary_holdout", tokens_saved=0)

        report = build_repeated_scaffold_impact_report(self.store, limit=20)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "hold")
        self.assertIn("non-positive-estimated-savings", candidate["reason_codes"])
        self.assertGreater(candidate["non_positive_savings_rate"], 0)

    def test_rollback_for_error_and_retry_regression(self) -> None:
        self._log_call(cohort="canary_applied", status_code=500, retry_count=1, tokens_saved=1000)
        self._log_call(cohort="canary_applied", status_code=200, retry_count=1, tokens_saved=1000)
        self._log_call(cohort="canary_holdout", status_code=200, retry_count=0, tokens_saved=0)

        report = build_repeated_scaffold_impact_report(self.store, limit=20)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "rollback")
        self.assertIn("error-rate-regression", candidate["reason_codes"])
        self.assertIn("retry-rate-regression", candidate["reason_codes"])

    def test_stale_evidence_holds_even_when_savings_are_positive(self) -> None:
        old = "2026-06-01T00:00:00+00:00"
        self._log_call(cohort="canary_applied", created_at=old, tokens_saved=1000)
        self._log_call(cohort="canary_applied", created_at=old, tokens_saved=1000)
        self._log_call(cohort="canary_holdout", created_at=old, tokens_saved=0)

        report = build_repeated_scaffold_impact_report(
            self.store,
            limit=20,
            max_evidence_age_hours=24,
            now=datetime(2026, 6, 12, tzinfo=timezone.utc),
        )

        candidate = report["candidates"][0]
        self.assertEqual(candidate["verdict"], "hold")
        self.assertIn("stale-evidence", candidate["reason_codes"])
        self.assertTrue(candidate["stale_evidence"]["stale"])

    def test_openai_rows_are_grouped_by_surface_phase_and_model_tier(self) -> None:
        self._log_call(
            cohort="canary_applied",
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            model_family="gpt-5-mini",
            category="chat",
            workflow_phase="execution",
            rule_id="reviewed-openai-provider-scaffold",
            candidate_id="candidate-openai",
        )
        self._log_call(
            cohort="canary_applied",
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            model_family="gpt-5-mini",
            category="chat",
            workflow_phase="execution",
            rule_id="reviewed-openai-provider-scaffold",
            candidate_id="candidate-openai",
        )
        self._log_call(
            cohort="canary_holdout",
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            model_family="gpt-5-mini",
            category="chat",
            workflow_phase="execution",
            rule_id="reviewed-openai-provider-scaffold",
            candidate_id="candidate-openai",
        )

        report = build_repeated_scaffold_impact_report(self.store, limit=20)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["provider"], "openai")
        self.assertEqual(candidate["source_surface"], "openai_responses")
        self.assertEqual(candidate["workflow_phase"], "execution")
        self.assertEqual(candidate["model_tier"], "gpt-5-mini")
