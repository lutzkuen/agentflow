from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from tokenclaw import cli
from tokenclaw.instruction_dedup_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE,
    build_instruction_dedup_lifecycle_feedback,
    queue_instruction_dedup_lifecycle_feedback,
)
from tokenclaw.instruction_dedup_impact import build_instruction_dedup_impact_report
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.optimization import feedback
from tokenclaw.stats import stats_instruction_dedup_impact
from tokenclaw.store import Store, stable_json, utc_now


FORBIDDEN_VALUES = (
    "private instruction impact secret",
    "private provider body secret",
    "private response impact secret",
    "private terminal output secret",
    "private tool payload secret",
    "raw-request-impact-secret",
    "raw-session-impact-secret",
    "raw-cache-key-impact-secret",
    "raw-tenant-impact-secret",
    "raw-impact-candidate-secret",
    "raw-impact-rule-secret",
    "raw impact category secret",
    "raw impact reason secret",
    "/workspace/private/impact.py",
)


class InstructionDedupImpactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _dedup_meta(
        self,
        *,
        status: str,
        reason: str,
        saved_chars: int = 800,
        projected_saved_usd: float = 0.0024,
        candidate_id: str = "instruction-dedup-candidate-impact",
        coordinator_status: str = "compatible",
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "schema": "tokenclaw.instruction_section_deduplication.v1",
            "enabled": True,
            "status": status,
            "reason": reason,
            "changed": status == "applied",
            "applied": status == "applied",
            "policy_source": "managed-recommended",
            "selected_rule_id": "managed-instruction-dedup-rule",
            "candidate_id": candidate_id,
            "source_surface": "anthropic_messages",
            "provider": "anthropic",
            "endpoint": "messages",
            "category": "chat",
            "workflow_phase": "planning",
            "applied_count": 1 if status == "applied" else 0,
            "holdout_count": 1 if status == "holdout" else 0,
            "saved_chars": saved_chars if status == "applied" else 0,
            "tokens_saved_est": saved_chars // 4 if status == "applied" else 0,
            "projected_saved_usd": projected_saved_usd if status == "applied" else 0.0,
            "reason_codes": [reason],
            "canary": {
                "cohort": "canary_applied" if status == "applied" else ("canary_holdout" if status == "holdout" else "not_selected"),
                "status": "applied" if status == "applied" else ("holdout" if status == "holdout" else "skipped"),
                "selected": status == "applied",
                "holdout": status == "holdout",
                "fingerprint_included": False,
                "salt_included": False,
            },
            "coordinator_compatibility": {
                "status": coordinator_status,
                "compatible": coordinator_status == "compatible",
                "selected_family": "none",
                "reason_codes": ["coordinator-private-secret-must-not-leak"] if coordinator_status == "conflict" else [],
            },
            "privacy": {
                "metadata_only_output": True,
                "raw_instruction_text_included": False,
                "instruction_section_fingerprint_included": False,
                "tool_payloads_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }
        if extra:
            meta.update(extra)
        return meta

    def _log_call(
        self,
        *,
        status: str = "applied",
        reason: str = "instruction-section-dedup-applied",
        status_code: int = 200,
        retry_count: int = 0,
        latency_ms: int = 100,
        cost: float = 0.01,
        baseline: float = 0.011,
        saved_chars: int = 800,
        projected_saved_usd: float = 0.0024,
        coordinator_status: str = "compatible",
        extra_meta: dict[str, object] | None = None,
    ) -> None:
        routing = {
            "source_surface": "anthropic_messages",
            "endpoint": "messages",
            "category": "chat",
            "workflow_phase": "planning",
            "text_chars": 12000,
            "managed_feedback": {"status": "queued", "reason": "private provider body secret"},
        }
        crunch = {
            "changed": status == "applied",
            "instruction_section_deduplication": self._dedup_meta(
                status=status,
                reason=reason,
                saved_chars=saved_chars,
                projected_saved_usd=projected_saved_usd,
                coordinator_status=coordinator_status,
                extra=extra_meta,
            ),
        }
        self.store.log_call(
            id=f"raw-request-impact-secret-{uuid.uuid4()}",
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model="claude-sonnet-4-6",
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=3000,
            output_tokens_est=50,
            actual_input_tokens=3000,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=baseline,
            crunch_json=stable_json(crunch),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "miss", "cache_key": "raw-cache-key-impact-secret"}),
            error=None if status_code < 400 else "private provider body secret",
            request_json=stable_json({
                "system": "private instruction impact secret",
                "messages": [{"role": "user", "content": "private tool payload secret"}],
                "tenant_id": "raw-tenant-impact-secret",
                "path": "/workspace/private/impact.py",
            }),
            response_json=stable_json({"text": "private response impact secret", "terminal": "private terminal output secret"}),
            session_id="raw-session-impact-secret",
            category="chat",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model_family="sonnet",
            routed_model_family="sonnet",
        )

    def _assert_private(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for value in FORBIDDEN_VALUES:
            self.assertNotIn(value, rendered)

    def test_impact_report_groups_cohorts_and_recommends_widen(self) -> None:
        self._log_call(status="holdout", reason="instruction-dedup-holdout", cost=0.012, baseline=0.012)
        self._log_call(status="applied", reason="instruction-section-dedup-applied", cost=0.009, baseline=0.011)
        self._log_call(status="applied", reason="instruction-section-dedup-applied", cost=0.009, baseline=0.011)

        report = build_instruction_dedup_impact_report(self.store, limit=20, min_applied_samples=2, min_holdout_samples=1)

        self.assertEqual(report["schema"], "tokenclaw.instruction_dedup_impact.v1")
        self.assertEqual(report["summary"]["applied_count"], 2)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertEqual(report["summary"]["saved_tokens_est"], 400)
        self.assertEqual(report["summary"]["rollback_action_count"], 0)
        self.assertEqual(report["candidates"][0]["next_action"], "widen")
        self.assertEqual(report["dashboard_rows"][0]["next_action"], "widen")
        self.assertFalse(report["privacy"]["raw_instruction_text_included"])
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["tenant_ids_included"])
        self._assert_private(report)

    def test_regression_gates_emit_rollback_action(self) -> None:
        self._log_call(status="holdout", reason="instruction-dedup-holdout", cost=0.010, baseline=0.010)
        self._log_call(status="applied", reason="instruction-section-dedup-applied", status_code=500, retry_count=2, cost=0.013, baseline=0.010)
        self._log_call(status="applied", reason="instruction-section-dedup-applied", status_code=500, retry_count=2, cost=0.013, baseline=0.010)

        report = build_instruction_dedup_impact_report(
            self.store,
            limit=20,
            min_applied_samples=2,
            min_holdout_samples=1,
            rollback_error_rate=0.5,
        )

        candidate = report["candidates"][0]
        self.assertEqual(candidate["next_action"], "rollback")
        self.assertGreater(report["summary"]["rollback_action_count"], 0)
        self.assertIn("rollback-absolute-error-rate", candidate["reason_codes"])
        self.assertEqual(report["rollback_actions"][0]["schema"], "tokenclaw.instruction_dedup_rollback_action.v1")
        self._assert_private(report)

    def test_stats_wrapper_and_cli_include_feedback_queue_status_without_payloads(self) -> None:
        self._log_call(status="holdout", reason="instruction-dedup-holdout", cost=0.012, baseline=0.012)
        self._log_call(status="applied", reason="instruction-section-dedup-applied")
        self._log_call(status="applied", reason="instruction-section-dedup-applied")

        result = asyncio.run(stats_instruction_dedup_impact(self.store, limit=20))
        self.assertEqual(result["schema"], "tokenclaw.instruction_dedup_impact.v1")
        self.assertIn("managed_lifecycle_feedback_queue", result)
        self.assertFalse(result["managed_lifecycle_feedback_queue"]["privacy"]["payload_json_included"])

        output = io.StringIO()
        with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}, clear=False):
            code = cli.instruction_dedup_impact_cli(["--db", self.db_path, "--limit", "20"], stdout=output)
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.instruction_dedup_impact.v1")
        self.assertEqual(payload["managed_lifecycle_feedback"]["status"], "queued")
        self.assertFalse(payload["managed_lifecycle_feedback"]["payload_included"])
        self.assertEqual(payload["managed_lifecycle_feedback_queue"]["summary"]["queued"], 1)
        self.assertFalse(payload["managed_lifecycle_feedback_queue"]["privacy"]["payload_json_included"])

        status = feedback.managed_feedback_status_result(self.store, source_surface=SOURCE_SURFACE, sample_limit=5)
        self.assertEqual(status["summary"]["queued"], 1)
        self.assertFalse(status["privacy"]["payload_json_included"])
        self._assert_private(payload)
        self._assert_private(status)

    def test_lifecycle_feedback_payload_is_metadata_only(self) -> None:
        self._log_call(status="holdout", reason="instruction-dedup-holdout", cost=0.012, baseline=0.012)
        self._log_call(status="applied", reason="instruction-section-dedup-applied")
        self._log_call(status="applied", reason="instruction-section-dedup-applied")
        report = build_instruction_dedup_impact_report(self.store, limit=20)

        payload = build_instruction_dedup_lifecycle_feedback(report)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["metadata"]["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(payload["event_type"], "runtime-selected")
        self.assertEqual(managed_egress_violations(payload), [])
        self.assertFalse(payload["metadata"]["privacy"]["raw_instruction_text_included"])
        self.assertFalse(payload["metadata"]["privacy"]["request_ids_included"])
        self._assert_private(payload)

    def test_impact_and_lifecycle_feedback_sanitize_raw_runtime_metadata(self) -> None:
        self._log_call(
            status="applied",
            reason="instruction-section-dedup-applied",
            extra_meta={
                "candidate_id": "raw-impact-candidate-secret",
                "selected_rule_id": "raw-impact-rule-secret",
                "category": "raw impact category secret",
                "workflow_phase": "/workspace/private/impact.py",
                "reason_codes": ["raw impact reason secret"],
            },
        )

        report = build_instruction_dedup_impact_report(
            self.store,
            limit=20,
            min_applied_samples=1,
            min_holdout_samples=0,
        )
        candidate = report["candidates"][0]
        self.assertTrue(candidate["candidate_id"].startswith("instruction-dedup-candidate:"))
        self.assertEqual(candidate["rule_id"], "unknown")
        self.assertEqual(candidate["category"], "unknown")
        self.assertEqual(candidate["workflow_phase"], "unknown")
        self.assertEqual(candidate["cohorts"]["applied"]["reason_breakdown"][0]["value"], "unknown")

        payload = build_instruction_dedup_lifecycle_feedback(report)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(managed_egress_violations(payload), [])
        self._assert_private(report)
        self._assert_private(payload)

    def test_feedback_queue_helper_queues_when_managed_disabled(self) -> None:
        self._log_call(status="applied", reason="instruction-section-dedup-applied")
        report = build_instruction_dedup_impact_report(self.store, limit=20, min_applied_samples=1, min_holdout_samples=0)

        with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}, clear=False):
            meta = asyncio.run(queue_instruction_dedup_lifecycle_feedback(self.store, report, flush_immediately=False))

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self._assert_private(meta)


if __name__ == "__main__":
    unittest.main()
