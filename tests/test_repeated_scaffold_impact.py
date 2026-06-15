from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from agentflow_proxy import cli
from agentflow_proxy.managed_egress import ManagedEgressBlocked, assert_managed_egress_safe
from agentflow_proxy.repeated_scaffold_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE as REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
    build_repeated_scaffold_lifecycle_feedback,
    build_repeated_scaffold_lifecycle_feedback_status,
)
from agentflow_proxy.repeated_scaffold_activation import build_repeated_scaffold_activation_report
from agentflow_proxy.repeated_scaffold_impact import build_repeated_scaffold_impact_report
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


class RepeatedScaffoldFeedbackClient:
    calls: list[dict] = []
    status_code = 200
    text = '{"ok":true}'

    def __init__(self, *, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)

    async def patch(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": dict(headers or {})})
        return httpx.Response(self.status_code, text=self.text)


class RepeatedScaffoldImpactTests(unittest.TestCase):
    REPEATED_SCAFFOLD_PRIVACY_DECOYS = (
        "raw activation request prompt must not leak",
        "raw activation response must not leak",
        "raw activation message must not leak",
        "raw activation provider body must not leak",
        "raw activation feedback payload must not leak",
        "raw activation request id must not leak",
        "raw activation session id must not leak",
        "raw activation cache key must not leak",
        "raw activation pattern hash must not leak",
        "/tmp/raw-activation-secret.py",
    )

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
        RepeatedScaffoldFeedbackClient.calls = []
        RepeatedScaffoldFeedbackClient.status_code = 200
        RepeatedScaffoldFeedbackClient.text = '{"ok":true}'
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
        created_at: str | None = None,
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
        created_at = created_at or utc_now()
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

    def _log_activation_call(
        self,
        *,
        managed: dict | None,
        repeated_status: str | None = None,
        repeated_reason: str | None = None,
        cohort: str | None = None,
        tokens_saved: int = 0,
        created_at: str = "2026-06-12T01:00:00+00:00",
        status_code: int = 200,
        category: str = "tool-result",
    ) -> None:
        routing = {
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "endpoint": "messages",
            "category": category,
            "workflow_phase": "tool-execution",
            "text_chars": 48_000,
            "has_tools": True,
        }
        if managed is not None:
            routing["managed_recommendation"] = managed
        provider_meta = None
        if repeated_status is not None:
            canary = {
                "enabled": True,
                "selected": cohort == "canary_applied",
                "cohort": cohort or "none",
                "fraction": 0.25,
                "unit": "request_fingerprint",
            }
            rule = {
                "rule_id": "activation-rule-must-not-leak",
                "candidate_id": "activation-candidate-must-not-leak",
                "enabled": True,
                "policy_source": "managed-recommended",
                "applied_count": 1 if repeated_status == "applied" else 0,
                "holdout_count": 1 if cohort == "canary_holdout" else 0,
                "saved_chars": tokens_saved * 4 if repeated_status == "applied" else 0,
                "canary": canary,
                "skip_reasons": [],
            }
            if repeated_status == "safety_stop":
                rule["skip_reasons"] = [{"reason": "safety_stop_error_rate", "count": 1}]
            provider_meta = {
                "schema": "agentflow.repeated_provider_scaffolding.v1",
                "enabled": True,
                "status": repeated_status,
                "reason": repeated_reason or repeated_status,
                "policy_source": "managed-recommended",
                "category": category,
                "saved_chars": tokens_saved * 4 if repeated_status == "applied" else 0,
                "tokens_saved_est": tokens_saved if repeated_status == "applied" else 0,
                "rules": [rule],
                "raw_text_included": False,
                "raw_hashes_included": False,
            }
        crunch = {"changed": repeated_status == "applied", "tokens_saved_est": tokens_saved}
        if provider_meta is not None:
            crunch["repeated_provider_scaffolding"] = provider_meta
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=100,
            input_tokens_est=12_000,
            output_tokens_est=100,
            actual_input_tokens=12_000,
            actual_output_tokens=100,
            cost_est_usd=0.04,
            cost_baseline_usd=0.04,
            crunch_json=stable_json(crunch),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error="raw activation error must not leak" if status_code >= 400 else None,
            request_json=stable_json({
                "messages": [{"role": "user", "content": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[0]}],
                "request_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[5],
                "session_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[6],
                "cache_key": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[7],
                "file_path": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[9],
            }),
            response_json=stable_json({"text": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[1]}),
            session_id=self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[6],
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model_family="sonnet",
            routed_model_family="sonnet",
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

    def test_policy_decision_activation_report_covers_repeated_scaffold_states(self) -> None:
        repeated_crunch = {
            "profile": "managed",
            "repeated_provider_scaffolding": {
                "enabled": True,
                "rules": [{"id": "managed-rule-secret", "candidate_id": "managed-candidate-secret"}],
            },
        }
        raw_like_managed_decoys = {
            "raw_prompt": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[0],
            "messages": [{"content": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[2]}],
            "provider_body": {"content": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[3]},
            "request_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[5],
            "session_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[6],
            "cache_key": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[7],
            "pattern_hashes": [self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[8]],
            "file_path": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[9],
        }
        self._log_activation_call(
            managed={
                "enabled": False,
                "status": "skipped",
                "reason": "disabled",
                "applied": False,
                **raw_like_managed_decoys,
            },
            created_at="2026-06-12T01:00:00+00:00",
        )
        self._log_activation_call(
            managed={
                "enabled": True,
                "status": "received",
                "reason": "No learned policy is active yet.",
                "policy_id": "baseline-pass-through",
                "optimization_unit_id": 101,
                "applied": False,
                "apply_reason": "missing-target-model",
                "crunch": {"profile": "baseline"},
                **raw_like_managed_decoys,
            },
            created_at="2026-06-12T01:01:00+00:00",
        )
        self._log_activation_call(
            managed={
                "enabled": True,
                "status": "received",
                "reason": "repeated scaffold canary",
                "policy_id": "managed-scaffold",
                "optimization_unit_id": 102,
                "applied": False,
                "apply_reason": "missing-target-model",
                "crunch": repeated_crunch,
                **raw_like_managed_decoys,
            },
            repeated_status="skipped",
            repeated_reason="canary_holdout",
            cohort="canary_holdout",
            created_at="2026-06-12T01:02:00+00:00",
        )
        self._log_activation_call(
            managed={
                "enabled": True,
                "status": "received",
                "reason": "repeated scaffold canary",
                "policy_id": "managed-scaffold",
                "optimization_unit_id": 103,
                "applied": False,
                "apply_reason": "missing-target-model",
                "crunch": repeated_crunch,
                "outcome_feedback": {
                    "enabled": True,
                    "status": "sent",
                    "reason": "accepted",
                    "optimization_unit_id": 103,
                    "payload_json": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[4],
                },
                **raw_like_managed_decoys,
            },
            repeated_status="applied",
            repeated_reason="repeated-provider-scaffolding-crunched",
            cohort="canary_applied",
            tokens_saved=900,
            created_at="2026-06-12T01:03:00+00:00",
        )
        self._log_activation_call(
            managed={
                "enabled": True,
                "status": "received",
                "reason": "repeated scaffold canary",
                "policy_id": "managed-scaffold",
                "optimization_unit_id": 104,
                "applied": False,
                "apply_reason": "missing-target-model",
                "crunch": repeated_crunch,
                **raw_like_managed_decoys,
            },
            repeated_status="safety_stop",
            repeated_reason="safety_stop_error_rate",
            cohort="safety_stop",
            created_at="2026-06-12T01:04:00+00:00",
        )
        self._log_activation_call(
            managed={
                "enabled": True,
                "status": "error",
                "reason": "server-error",
                "fallback": "local-policy",
                "applied": False,
                "error": "raw managed error must not leak",
                **raw_like_managed_decoys,
            },
            created_at="2026-06-12T01:05:00+00:00",
            status_code=502,
        )
        self.store.enqueue_managed_outcome_feedback(
            id="feedback-row-secret",
            source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
            endpoint="/v1/policy-feedback",
            optimization_unit_id=103,
            payload_json=stable_json({"raw_prompt": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[4]}),
            status="sent",
            sent_at="2026-06-12T01:06:00+00:00",
        )

        report = build_repeated_scaffold_activation_report(self.store, limit=20)

        self.assertEqual(report["schema"], "agentflow.repeated_scaffold_activation.v1")
        self.assertEqual(report["status"], "matched")
        summary = report["summary"]
        self.assertEqual(summary["sampled_call_count"], 6)
        self.assertEqual(summary["managed_recommendation_rows"], 6)
        self.assertEqual(summary["preflight_disabled_count"], 1)
        self.assertEqual(summary["baseline_count"], 1)
        self.assertEqual(summary["repeated_scaffold_recommended_count"], 3)
        self.assertEqual(summary["holdout_count"], 1)
        self.assertEqual(summary["applied_count"], 1)
        self.assertEqual(summary["safety_stop_count"], 1)
        self.assertEqual(summary["server_error_count"], 1)
        self.assertEqual(summary["optimization_unit_present_count"], 4)
        self.assertEqual(summary["feedback_sent_count"], 2)
        self.assertEqual(summary["estimated_saved_tokens"], 900)
        states = {row["value"]: row["count"] for row in report["activation_state_counts"]}
        self.assertEqual(states["preflight-disabled"], 1)
        self.assertEqual(states["baseline-no-repeated-scaffold-policy"], 1)
        self.assertEqual(states["recommended-holdout"], 1)
        self.assertEqual(states["applied-repeated-scaffold-profile"], 1)
        self.assertEqual(states["safety-stopped"], 1)
        self.assertEqual(states["server-error"], 1)
        queue = {row["value"]: row["count"] for row in report["feedback_queue_status_counts"]}
        self.assertEqual(queue["sent"], 1)
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["optimization_unit_ids_included"])
        self.assertFalse(report["privacy"]["feedback_payloads_included"])

        output = io.StringIO()
        exit_code = cli.repeated_scaffold_activation_cli(["--db", self.db_path, "--limit", "20"], stdout=output)
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["summary"]["applied_count"], 1)
        rendered = json.dumps(report, sort_keys=True) + json.dumps(payload, sort_keys=True)
        for forbidden in (
            *self.REPEATED_SCAFFOLD_PRIVACY_DECOYS,
            "activation-rule-must-not-leak",
            "activation-candidate-must-not-leak",
            "managed-rule-secret",
            "managed-candidate-secret",
            "raw managed error must not leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_repeated_scaffold_lifecycle_queue_public_view_omits_raw_payloads(self) -> None:
        self.store.enqueue_managed_outcome_feedback(
            id="queue-public-activation-privacy",
            created_at="2026-06-12T01:00:00+00:00",
            updated_at="2026-06-12T01:01:00+00:00",
            source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
            endpoint="/v1/policy-events",
            optimization_unit_id=99,
            payload_json=stable_json({
                "raw_prompt": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[0],
                "messages": [{"content": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[2]}],
                "provider_body": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[3],
                "request_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[5],
                "session_id": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[6],
                "cache_key": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[7],
                "file_path": self.REPEATED_SCAFFOLD_PRIVACY_DECOYS[9],
            }),
            status="retryable-error",
            attempts=1,
            next_attempt_at="2000-01-01T00:00:00+00:00",
            last_error="ConnectError: raw activation feedback payload must not leak",
        )

        status = build_repeated_scaffold_lifecycle_feedback_status(self.store, sample_limit=10)

        self.assertEqual(status["schema"], "agentflow.repeated_scaffold_lifecycle_feedback_queue_status.v1")
        self.assertEqual(status["summary"]["retryable_error"], 1)
        self.assertEqual(status["summary"]["due"], 1)
        self.assertEqual(status["due_samples"][0]["payload_included"], False)
        self.assertFalse(status["privacy"]["payload_json_included"])
        rendered = json.dumps(status, sort_keys=True)
        for forbidden in self.REPEATED_SCAFFOLD_PRIVACY_DECOYS:
            self.assertNotIn(forbidden, rendered)
        self.assertNotIn("ConnectError: raw activation feedback payload must not leak", rendered)

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

    def test_cli_flushes_repeated_scaffold_lifecycle_feedback_with_redacted_queue_audit(self) -> None:
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1200)
        self._log_call(cohort="canary_holdout", tokens_saved=0)
        self.store.enqueue_managed_outcome_feedback(
            id="sent-public-row",
            created_at="2026-06-12T01:00:00+00:00",
            updated_at="2026-06-12T01:01:00+00:00",
            source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
            endpoint="/v1/policy-events",
            optimization_unit_id=0,
            payload_json=stable_json({"raw_prompt": "sent payload must not leak", "file_path": "/tmp/sent-secret.py"}),
            status="sent",
            attempts=1,
            next_attempt_at="2026-06-12T01:00:00+00:00",
            sent_at="2026-06-12T01:02:00+00:00",
        )
        self.store.enqueue_managed_outcome_feedback(
            id="retry-public-row",
            created_at="2026-06-12T01:03:00+00:00",
            updated_at="2026-06-12T01:04:00+00:00",
            source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
            endpoint="/v1/policy-events",
            optimization_unit_id=0,
            payload_json=stable_json({"raw_messages": "retry payload must not leak", "path": "/tmp/retry-secret.py"}),
            status="retryable-error",
            attempts=1,
            next_attempt_at="2030-01-01T00:00:00+00:00",
            last_error="ConnectError: managed feedback down with retry secret /tmp/retry-secret.py",
        )
        self.store.enqueue_managed_outcome_feedback(
            id="dropped-public-row",
            created_at="2026-06-12T01:05:00+00:00",
            updated_at="2026-06-12T01:06:00+00:00",
            source_surface=REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,
            endpoint="/v1/policy-events",
            optimization_unit_id=0,
            payload_json=stable_json({"raw_response": "dropped payload must not leak", "path": "/tmp/drop-secret.py"}),
            status="dropped-after-limit",
            attempts=3,
            next_attempt_at="2026-06-12T01:05:00+00:00",
            last_error="HTTP 500: dropped payload body must not leak",
            last_status_code=500,
        )
        self.store.enqueue_managed_outcome_feedback(
            id="other-source-row",
            source_surface="codex_turn",
            endpoint="/v1/optimization-units/7/outcome",
            optimization_unit_id=7,
            payload_json=stable_json({"raw_prompt": "other source must not flush"}),
            status="queued",
            next_attempt_at="2026-06-12T01:00:00+00:00",
        )

        output = io.StringIO()
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", RepeatedScaffoldFeedbackClient):
            exit_code = cli.repeated_scaffold_impact_cli(
                ["--db", self.db_path, "--limit", "20", "--flush-feedback", "--feedback-limit", "10"],
                stdout=output,
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        flush = payload["managed_lifecycle_feedback_flush"]
        self.assertEqual(flush["schema"], "agentflow.repeated_scaffold_lifecycle_feedback_flush.v1")
        self.assertEqual(flush["flush"]["sent"], 1)
        self.assertEqual(flush["before"]["queued"], 1)
        self.assertEqual(flush["before"]["due"], 1)
        self.assertEqual(flush["before"]["sent"], 1)
        self.assertEqual(flush["before"]["retryable_error"], 1)
        self.assertEqual(flush["before"]["dropped_after_limit"], 1)
        self.assertEqual(flush["after"]["queued"], 0)
        self.assertEqual(flush["after"]["sent"], 2)
        self.assertEqual(flush["after"]["retryable_error"], 1)
        self.assertEqual(flush["after"]["dropped_after_limit"], 1)
        self.assertEqual(flush["after"]["last_error_class"], "http-5xx")
        queue = payload["managed_lifecycle_feedback_queue"]
        self.assertEqual(queue["summary"]["sent"], 2)
        self.assertEqual(queue["summary"]["retryable_error"], 1)
        self.assertEqual(queue["summary"]["dropped_after_limit"], 1)
        error_classes = {row["value"]: row["count"] for row in queue["last_error_class_breakdown"]}
        self.assertEqual(error_classes["ConnectError"], 1)
        self.assertEqual(error_classes["http-5xx"], 1)
        self.assertEqual(len(RepeatedScaffoldFeedbackClient.calls), 1)
        self.assertEqual(RepeatedScaffoldFeedbackClient.calls[0]["url"], "http://managed.test/v1/policy-events")

        rendered = output.getvalue()
        for forbidden in (
            "sent payload must not leak",
            "retry payload must not leak",
            "dropped payload must not leak",
            "other source must not flush",
            "managed feedback down with retry secret",
            "dropped payload body must not leak",
            "/tmp/sent-secret.py",
            "/tmp/retry-secret.py",
            "/tmp/drop-secret.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(queue["privacy"]["payload_json_included"])
        self.assertFalse(flush["privacy"]["payload_json_included"])

    def test_cli_feedback_dry_run_reports_due_repeated_scaffold_rows_without_claiming(self) -> None:
        os.environ["AGENTFLOW_RECOMMENDATION_ENABLED"] = "1"
        os.environ["AGENTFLOW_RECOMMENDATION_SERVER_URL"] = "http://managed.test"
        self._log_call(cohort="canary_applied", tokens_saved=1000)
        self._log_call(cohort="canary_applied", tokens_saved=1200)
        self._log_call(cohort="canary_holdout", tokens_saved=0)

        output = io.StringIO()
        with patch("agentflow_proxy.recommendations.httpx.AsyncClient", RepeatedScaffoldFeedbackClient):
            exit_code = cli.repeated_scaffold_impact_cli(
                ["--db", self.db_path, "--limit", "20", "--feedback-dry-run"],
                stdout=output,
            )
        row = self.store.conn.execute(
            "select status, attempts from managed_outcome_feedback_queue "
            "where source_surface = ?",
            (REPEATED_SCAFFOLD_LIFECYCLE_SOURCE_SURFACE,),
        ).fetchone()

        self.assertEqual(exit_code, 0)
        self.assertEqual(RepeatedScaffoldFeedbackClient.calls, [])
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "queued")
        self.assertEqual(row["attempts"], 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["managed_lifecycle_feedback_flush"]["dry_run"])
        self.assertEqual(payload["managed_lifecycle_feedback_flush"]["flush"]["would_attempt"], 1)
        self.assertEqual(payload["managed_lifecycle_feedback_queue"]["summary"]["queued"], 1)

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
