from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agentflow_proxy import cli
from agentflow_proxy.managed_egress import ManagedEgressBlocked, assert_managed_egress_safe
from agentflow_proxy.repeated_scaffold_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE as REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
    build_repeated_scaffold_lifecycle_feedback,
)
from agentflow_proxy.repeated_scaffold_impact import build_repeated_scaffold_impact_report
from agentflow_proxy.store import SQLiteStore, stable_json


class RepeatedScaffoldImpactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_env = {
            key: os.environ.get(key)
            for key in (
                "AGENTFLOW_RECOMMENDATION_ENABLED",
                "AGENTFLOW_RECOMMENDATION_SERVER_URL",
                "AGENTFLOW_MANAGED_API_KEY",
            )
        }
        for key in self.saved_env:
            os.environ.pop(key, None)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
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
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "disabled")
        self.assertEqual(
            self.store.conn.execute("select count(*) from managed_outcome_feedback_queue").fetchone()[0],
            0,
        )

    def test_lifecycle_feedback_payload_is_metadata_only_and_egress_safe(self) -> None:
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1200)
        self._log_call(cohort="canary_holdout", tokens_saved=0)

        report = build_repeated_scaffold_impact_report(self.store, limit=20)
        report["candidates"][0]["raw_prompt"] = "raw prompt must be redacted"
        report["candidates"][0]["messages"] = [{"content": "raw messages must be redacted"}]
        report["candidates"][0]["request_id"] = "req_secret"
        report["candidates"][0]["session_id"] = "session-secret"
        report["candidates"][0]["file_path"] = "/tmp/secret.py"
        report["candidates"][0]["request_fingerprint"] = "sha256:" + "1" * 64

        event = build_repeated_scaffold_lifecycle_feedback(report)

        self.assertIsNotNone(event)
        assert_managed_egress_safe(event)
        metadata = event["metadata"]
        self.assertEqual(metadata["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(metadata["lifecycle_kind"], "repeated_scaffold_crunch")
        self.assertEqual(metadata["candidate_feedback"][0]["source_surface"], "anthropic_messages")
        self.assertEqual(metadata["candidate_feedback"][0]["category"], "tool-result")
        self.assertEqual(metadata["candidate_feedback"][0]["workflow_phase"], "tool-execution")
        self.assertEqual(metadata["candidate_feedback"][0]["model_tier"], "sonnet")
        self.assertEqual(metadata["candidate_feedback"][0]["saved_tokens_bucket"], "1k_4k_tokens")
        self.assertEqual(metadata["candidate_feedback"][0]["cost_savings_bucket"], "0_001_0_01_usd")
        self.assertEqual(metadata["candidate_feedback"][0]["canary_cohort_counts"]["applied"], 2)
        self.assertTrue(metadata["privacy"]["metadata_only"])

        rendered = json.dumps(event, sort_keys=True)
        self.assertNotIn("raw prompt must be redacted", rendered)
        self.assertNotIn("raw messages must be redacted", rendered)
        self.assertNotIn("req_secret", rendered)
        self.assertNotIn("session-secret", rendered)
        self.assertNotIn("/tmp/secret.py", rendered)
        self.assertNotIn("sha256:" + "1" * 64, rendered)

    def test_lifecycle_feedback_rejects_raw_fingerprint_keys(self) -> None:
        with self.assertRaises(ManagedEgressBlocked):
            assert_managed_egress_safe({
                "schema": FEEDBACK_SCHEMA,
                "command": "repeated-scaffold-impact",
                "request_fingerprint": "sha256:" + "1" * 64,
            })

    def test_enabled_cli_queues_lifecycle_feedback_without_payload_leakage(self) -> None:
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1200)
        self._log_call(cohort="canary_holdout", tokens_saved=0)

        output = io.StringIO()
        exit_code = cli.repeated_scaffold_impact_cli(["--db", self.db_path, "--limit", "20"], stdout=output)

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        feedback = payload["managed_lifecycle_feedback"]
        self.assertEqual(feedback["status"], "queued")
        self.assertEqual(feedback["endpoint"], "/v1/policy-events")
        self.assertFalse(feedback["payload_included"])

        row = self.store.conn.execute(
            "select source_surface, endpoint, status, attempts, payload_json "
            "from managed_outcome_feedback_queue"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["source_surface"], REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        queued = json.loads(row["payload_json"])
        self.assertEqual(queued["metadata"]["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(queued["policy_sections"], ["crunch"])
        queued_text = json.dumps(queued, sort_keys=True)
        self.assertIn("reviewed-provider-scaffold", queued_text)
        self.assertNotIn("raw error text must not leak", queued_text)
        self.assertNotIn("prompt must not leak", queued_text)
        self.assertNotIn("raw response must not leak", queued_text)
        self.assertNotIn("raw-session-must-not-leak", queued_text)

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
