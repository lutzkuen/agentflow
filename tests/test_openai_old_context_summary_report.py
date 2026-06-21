from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tokenclaw import cli
from tokenclaw.openai_old_context_summary_report import build_openai_old_context_summary_report
from tokenclaw.stats import stats_openai_old_context_summary_report
from tokenclaw.store import SQLiteStore, stable_json, utc_now


class OpenAIOldContextSummaryReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _feature(
        self,
        *,
        endpoint: str = "responses",
        source_surface: str = "openai_responses",
        has_tools: bool = False,
        category: str = "chat",
        workflow_phase: str = "tool-execution",
        older_bucket: str = "32k_128k_chars",
    ) -> dict[str, object]:
        return {
            "schema": "tokenclaw.openai_feature_summary.v1",
            "provider": "openai",
            "source_surface": source_surface,
            "endpoint": endpoint,
            "requested_model_family": "gpt-5",
            "routed_model_family": "gpt-5",
            "stream": False,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_bucket": "32k_128k_chars",
            "input_token_bucket": "16k_64k_tokens",
            "has_tools": has_tools,
            "declared_tool_count": 1 if has_tools else 0,
            "chat_tool_call_count": 1 if has_tools and endpoint == "chat_completions" else 0,
            "chat_tool_result_count": 1 if has_tools and endpoint == "chat_completions" else 0,
            "old_context": {
                "shape": "responses_input_items" if endpoint == "responses" else "chat_messages",
                "conversation_item_count": 12,
                "older_context_item_count": 8,
                "older_context_text_bucket": older_bucket,
                "older_context_token_bucket": "16k_64k_tokens",
                "raw_payload_included": False,
            },
            "raw_payload_included": False,
        }

    def _log_openai_call(
        self,
        *,
        created_at: str | None = None,
        path: str = "/v1/responses",
        endpoint: str = "responses",
        source_surface: str = "openai_responses",
        feature: dict[str, object] | None = None,
        has_tools: bool = False,
        category: str = "chat",
        text_chars: int = 64_000,
        stream: int = 0,
        cache_status: str = "miss",
        status_code: int = 200,
        latency_ms: int = 125,
        retry_count: int = 0,
        cost_est_usd: float = 0.01,
        cost_baseline_usd: float = 0.02,
        summary_meta: dict[str, object] | None = None,
        session_id: str = "secret-openai-session",
        request_json: str | None = '{"input":"secret raw openai prompt","request_id":"req_secret"}',
        omit_feature: bool = False,
    ) -> None:
        actual_input_tokens = max(1, text_chars // 4)
        actual_output_tokens = 80
        if not omit_feature and feature is None:
            feature = self._feature(
                endpoint=endpoint,
                source_surface=source_surface,
                has_tools=has_tools,
                category=category,
            )
        routing: dict[str, object] = {
            "enabled": False,
            "provider": "openai",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "reason": "openai routing disabled",
            "text_chars": text_chars,
            "has_tools": has_tools,
            "category": category,
            "workflow_phase": "tool-execution",
            "context_plateau_status": "plateau-adjacent",
            "policy_source": "local-default",
        }
        if feature is not None:
            routing["openai_feature_unit"] = feature
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=created_at or utc_now(),
            path=path,
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            stream=stream,
            cache_hit=1 if cache_status == "hit" else 0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=actual_input_tokens,
            output_tokens_est=actual_output_tokens,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            crunch_json=stable_json({
                "changed": bool(summary_meta and summary_meta.get("applied")),
                "tokens_saved_est": int((summary_meta or {}).get("estimated_tokens_saved") or 0),
                **({"old_context_summarization": summary_meta} if summary_meta else {}),
            }),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": cache_status, "reason": "exact-miss", "policy_source": "local-default"}),
            error=None,
            request_json=request_json,
            response_json=None,
            session_id=session_id,
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )

    def _summary_meta(
        self,
        *,
        cohort: str = "canary_applied",
        status: str = "applied",
        enabled: bool = True,
        candidate_id: str = "secret-content-derived-candidate",
        rule_id: str = "local-openai-old-context-summary",
        net_savings: float = 0.004,
        gross_savings: float = 0.005,
        summary_cost: float = 0.001,
        tokens_saved: int = 1000,
        summary_status_code: int | None = 200,
        summary_error: str | None = None,
        reason_codes: list[str] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "schema": "tokenclaw.openai_old_context_summary.v1",
            "enabled": enabled,
            "status": status,
            "applied": status == "applied",
            "changed": status == "applied",
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "summary_model": "gpt-5-mini",
            "endpoint": "responses",
            "workflow_phase": "tool-execution",
            "canary": {
                "cohort": cohort,
                "canary_fraction": 0.5,
                "holdout_fraction": 0.5,
            },
            "estimated_tokens_saved": tokens_saved,
            "estimated_gross_savings_usd": gross_savings,
            "summary_cost_est_usd": summary_cost,
            "estimated_net_savings_usd": net_savings,
            "reason_codes": reason_codes or [status],
            "privacy": {
                "raw_source_included": False,
                "raw_summary_included": False,
                "raw_request_body_included": False,
                "summary_text_included": False,
                "session_id_included": False,
            },
        }
        if summary_status_code is not None:
            meta["summary_status_code"] = summary_status_code
        if summary_error:
            meta["summary_error"] = summary_error
        return meta

    def _add_gate_samples(
        self,
        *,
        applied: int = 2,
        holdout: int = 1,
        created_at: str | None = None,
        applied_status_codes: list[int] | None = None,
        applied_retries: list[int] | None = None,
        applied_latencies: list[int] | None = None,
        applied_metas: list[dict[str, object]] | None = None,
    ) -> None:
        applied_status_codes = applied_status_codes or [200] * applied
        applied_retries = applied_retries or [0] * applied
        applied_latencies = applied_latencies or [100] * applied
        applied_metas = applied_metas or [
            self._summary_meta(candidate_id=f"secret-applied-{index}")
            for index in range(applied)
        ]
        for index in range(applied):
            self._log_openai_call(
                created_at=created_at,
                status_code=applied_status_codes[index],
                retry_count=applied_retries[index],
                latency_ms=applied_latencies[index],
                summary_meta=applied_metas[index],
            )
        for index in range(holdout):
            self._log_openai_call(
                created_at=created_at,
                latency_ms=100,
                summary_meta=self._summary_meta(
                    cohort="holdout",
                    status="holdout",
                    candidate_id=f"secret-holdout-{index}",
                    net_savings=0.0,
                    gross_savings=0.0,
                    summary_cost=0.0,
                    tokens_saved=0,
                    reason_codes=["holdout"],
                ),
            )

    def test_report_measures_openai_responses_and_chat_without_raw_fields(self) -> None:
        self._log_openai_call()
        self._log_openai_call(
            path="/v1/chat/completions",
            endpoint="chat_completions",
            source_surface="openai_chat",
            category="summary",
            feature=self._feature(endpoint="chat_completions", source_surface="openai_chat", category="summary"),
        )
        self._log_openai_call(omit_feature=True)
        self._log_openai_call(has_tools=True, feature=self._feature(has_tools=True), category="tool-light")

        result = build_openai_old_context_summary_report(
            self.store,
            limit=20,
            summary_provider_configured=True,
            summary_model="gpt-5-mini",
        )

        self.assertEqual(result["schema"], "tokenclaw.openai_old_context_summary_opportunity.v1")
        self.assertEqual(result["summary"]["openai_call_count"], 4)
        self.assertEqual(result["summary"]["feature_row_count"], 3)
        self.assertEqual(result["summary"]["openai_old_context_summary_metadata_row_count"], 0)
        self.assertEqual(result["summary"]["eligible_count"], 2)
        self.assertEqual(result["summary"]["blocked_count"], 2)
        self.assertGreater(result["summary"]["projected_summarized_chars"], 0)
        self.assertGreater(result["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(result["summary"]["estimated_summary_cost_usd"], 0)
        self.assertGreater(result["summary"]["projected_gross_savings_usd"], 0)
        self.assertIn("projected_net_savings_usd", result["summary"])

        blockers = {row["value"]: row["count"] for row in result["blocker_reason_breakdown"]}
        self.assertEqual(blockers["blocked_missing_body_or_feature"], 1)
        self.assertEqual(blockers["tool_protocol_risk"], 1)
        endpoints = {row["value"]: row["count"] for row in result["endpoint_breakdown"]}
        self.assertEqual(endpoints["responses"], 3)
        self.assertEqual(endpoints["chat_completions"], 1)

        rendered = json.dumps(result, sort_keys=True)
        for forbidden in (
            "secret-openai-session",
            "secret raw openai prompt",
            "req_secret",
            "request_json",
            "response_json",
            "raw chat message",
            "raw function args",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertFalse(result["privacy"]["raw_prompts_included"])
        self.assertFalse(result["privacy"]["raw_provider_bodies_included"])
        self.assertFalse(result["privacy"]["tool_payloads_included"])
        self.assertFalse(result["privacy"]["function_arguments_included"])
        self.assertFalse(result["privacy"]["request_ids_included"])
        self.assertFalse(result["privacy"]["session_ids_included"])
        self.assertFalse(result["privacy"]["candidate_ids_included"])
        self.assertFalse(result["privacy"]["provider_calls_made"])

    def test_provider_configuration_blocks_rows_until_local_summary_provider_exists(self) -> None:
        self._log_openai_call()

        result = build_openai_old_context_summary_report(
            self.store,
            limit=10,
            summary_provider_configured=False,
        )

        self.assertEqual(result["summary"]["eligible_count"], 0)
        self.assertEqual(result["summary"]["blocked_count"], 1)
        self.assertEqual(
            result["blocker_reason_breakdown"],
            [{"value": "summary_provider_not_configured", "count": 1}],
        )
        self.assertFalse(result["measurement_policy"]["summary_provider_configured"])

    def test_stats_wrapper_and_cli_emit_report(self) -> None:
        self._log_openai_call()

        with mock.patch.dict(os.environ, {"TOKENCLAW_OPENAI_OLD_CONTEXT_SUMMARY_PROVIDER_CONFIGURED": "1"}):
            result = asyncio.run(stats_openai_old_context_summary_report(self.store, limit=10))
            self.assertEqual(result["schema"], "tokenclaw.openai_old_context_summary_opportunity.v1")

            output = io.StringIO()
            exit_code = cli.openai_old_context_summary_report_cli(
                ["--db", self.db_path, "--limit", "10"],
                stdout=output,
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.openai_old_context_summary_opportunity.v1")
        self.assertEqual(payload["summary"]["openai_call_count"], 1)
        self.assertNotIn("secret-openai-session", output.getvalue())

    def test_quality_gate_promotes_positive_openai_summary_evidence(self) -> None:
        self._add_gate_samples()

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "promote")
        self.assertEqual(gate["reason_codes"], ["quality-gate-passed"])
        self.assertEqual(gate["actual_matched_metadata_row_count"], 3)
        self.assertEqual(gate["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(gate["cohort_counts"]["canary_holdout"], 1)
        self.assertEqual(gate["totals"]["summary_failure_count"], 0)
        self.assertGreater(gate["totals"]["estimated_net_savings_usd"], 0)
        self.assertEqual(result["quality_gate_summary"]["canary_applied_count"], 2)
        rendered = json.dumps(result, sort_keys=True)
        self.assertNotIn("secret-applied", rendered)
        self.assertNotIn("secret-holdout", rendered)
        self.assertNotIn("secret-content-derived-candidate", rendered)
        self.assertFalse(gate["privacy"]["candidate_ids_included"])

    def test_quality_gate_needs_more_samples_for_insufficient_cohorts(self) -> None:
        self._add_gate_samples(applied=1, holdout=0)

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "needs_more_samples")
        self.assertIn("insufficient-applied-samples", gate["reason_codes"])
        self.assertIn("insufficient-holdout-samples", gate["reason_codes"])

    def test_quality_gate_rolls_back_summary_failure_rate(self) -> None:
        self._add_gate_samples(applied_metas=[
            self._summary_meta(candidate_id="secret-applied-ok"),
            self._summary_meta(
                candidate_id="secret-applied-failure",
                status="skipped",
                summary_status_code=500,
                summary_error="summary provider failed without raw content",
                net_savings=0.0,
                gross_savings=0.0,
                summary_cost=0.001,
                tokens_saved=0,
                reason_codes=["summary_empty_or_malformed"],
            ),
        ])

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "rollback")
        self.assertIn("summary-failure-rate", gate["reason_codes"])
        self.assertEqual(gate["totals"]["summary_failure_count"], 1)
        self.assertIn({"value": "5xx", "count": 1}, gate["summary_provider_status_buckets"])

    def test_quality_gate_rolls_back_negative_savings_rate(self) -> None:
        self._add_gate_samples(applied_metas=[
            self._summary_meta(candidate_id="secret-applied-ok"),
            self._summary_meta(
                candidate_id="secret-applied-negative",
                net_savings=-0.002,
                gross_savings=0.001,
                summary_cost=0.003,
            ),
        ])

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "rollback")
        self.assertIn("negative-net-savings-rate", gate["reason_codes"])
        self.assertEqual(gate["totals"]["negative_net_savings_count"], 1)

    def test_quality_gate_holds_on_error_regression(self) -> None:
        self._add_gate_samples(applied_status_codes=[500, 200])

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "hold")
        self.assertIn("error-rate-regression", gate["reason_codes"])
        self.assertEqual(gate["cohort_metrics"]["canary_applied"]["error_count"], 1)
        self.assertIn({"value": "5xx", "count": 1}, gate["status_code_buckets"])

    def test_quality_gate_holds_on_retry_regression(self) -> None:
        self._add_gate_samples(applied_retries=[1, 0])

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "hold")
        self.assertIn("retry-rate-regression", gate["reason_codes"])
        self.assertEqual(gate["cohort_metrics"]["canary_applied"]["retry_count"], 1)

    def test_quality_gate_holds_on_stale_evidence(self) -> None:
        old = datetime(2026, 6, 1, tzinfo=timezone.utc)
        self._add_gate_samples(created_at=old.isoformat())

        result = build_openai_old_context_summary_report(
            self.store,
            limit=10,
            summary_provider_configured=True,
            max_evidence_age_hours=1.0,
            now=old + timedelta(hours=2),
        )

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "hold")
        self.assertIn("stale-evidence", gate["reason_codes"])
        self.assertTrue(gate["freshness"]["stale"])

    def test_quality_gate_reports_disabled_policy_metadata(self) -> None:
        self._log_openai_call(summary_meta=self._summary_meta(
            enabled=False,
            status="disabled",
            cohort="not_selected",
            candidate_id="secret-disabled-candidate",
            net_savings=0.0,
            gross_savings=0.0,
            summary_cost=0.0,
            tokens_saved=0,
            reason_codes=["disabled"],
        ))

        result = build_openai_old_context_summary_report(self.store, limit=10, summary_provider_configured=True)

        gate = result["quality_gates"][0]
        self.assertEqual(gate["verdict"], "disabled")
        self.assertEqual(gate["reason_codes"], ["summary-policy-disabled"])
        self.assertEqual(gate["cohort_counts"]["disabled"], 1)
        self.assertNotIn("secret-disabled-candidate", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
