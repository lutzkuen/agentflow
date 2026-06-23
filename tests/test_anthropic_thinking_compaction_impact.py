from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from tokenclaw import cli
from tokenclaw.anthropic_thinking_compaction_impact import (
    MANAGED_FEEDBACK_SCHEMA,
    MANAGED_FEEDBACK_SOURCE_SURFACE,
    SERVER_REQUESTED_OUTCOME_FIELDS,
    build_anthropic_thinking_compaction_impact_report,
    build_anthropic_thinking_compaction_managed_feedback,
    queue_anthropic_thinking_compaction_managed_feedback,
)
from tokenclaw.dashboard_app import create_dashboard_app
from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json


RAW_SECRET = "raw-thinking-impact-secret"


def _compaction_meta(
    *,
    status: str,
    reason: str,
    applied: bool = False,
    cohort: str = "canary_applied",
    tokens_saved: int = 0,
    planned_tokens: int = 0,
    fallback: bool = False,
) -> dict:
    meta = {
        "schema": "tokenclaw.anthropic_thinking_history_compaction_decision.v1",
        "enabled": True,
        "status": status,
        "reason": reason,
        "changed": applied,
        "applied": applied,
        "policy_source": "local-manual",
        "rule_id": f"raw-rule-{RAW_SECRET}",
        "candidate_id": f"raw-candidate-{RAW_SECRET}",
        "category": "tool-result",
        "before_chars": 24000,
        "planned_saved_tokens": planned_tokens,
        "tokens_saved_est": tokens_saved,
        "saved_chars": tokens_saved * 4,
        "planned_saved_chars": planned_tokens * 4,
        "canary": {"cohort": cohort, "selected": applied, "reason": reason},
        "lifecycle_feedback": {
            "schema": "tokenclaw.anthropic_thinking_history_compaction_lifecycle_feedback.v1",
            "status": status if status != "bypass" else "safety_stop",
            "cohort": cohort,
            "candidate_id": f"raw-lifecycle-{RAW_SECRET}",
            "metadata_only": True,
            "raw_payload_included": False,
        },
    }
    if fallback:
        meta["fallback"] = True
        meta["fallback_reason"] = "rate-limit"
    if status == "bypass":
        meta["safety_stop_state"] = "stopped"
        meta["safety_stop"] = {
            "schema": "tokenclaw.anthropic_thinking_history_compaction_safety_stop.v1",
            "reason": "local-canary-safety-stop",
            "triggers": [{"metric": "error_rate", "value": 1.0, "threshold": 0.2}],
            "raw_payload_included": False,
        }
    return {"anthropic_thinking_history_compaction": meta}


def _log_call(
    store: Store,
    call_id: str,
    *,
    created_at: str,
    status: str,
    reason: str,
    applied: bool = False,
    cohort: str = "canary_applied",
    status_code: int = 200,
    cost_est: float = 0.02,
    cost_baseline: float = 0.05,
    tokens_saved: int = 0,
    planned_tokens: int = 0,
    retry_count: int = 0,
    thinking_tokens: int = 100,
    fallback: bool = False,
    session_id: str | None = None,
) -> None:
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=1000,
        input_tokens_est=6000,
        output_tokens_est=300,
        actual_input_tokens=6000,
        actual_output_tokens=300,
        cost_est_usd=cost_est,
        cost_baseline_usd=cost_baseline,
        crunch_json=stable_json(
            _compaction_meta(
                status=status,
                reason=reason,
                applied=applied,
                cohort=cohort,
                tokens_saved=tokens_saved,
                planned_tokens=planned_tokens,
                fallback=fallback,
            )
        ),
        routing_json=stable_json({"category": "tool-result", "workflow_phase": "tool-execution", "text_chars": 24000}),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "cache_key": f"raw-cache-key-{RAW_SECRET}"}),
        error=f"upstream failure {RAW_SECRET}" if status_code >= 400 else None,
        request_json=stable_json(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": f"private thinking {RAW_SECRET}"},
                            {
                                "type": "tool_use",
                                "id": f"raw-tool-id-{RAW_SECRET}",
                                "name": "Read",
                                "input": {"file_path": f"/private/{RAW_SECRET}.py"},
                            },
                        ],
                    }
                ],
                "request_id": f"raw-request-id-{RAW_SECRET}",
            }
        ),
        response_json=stable_json({"text": f"raw response {RAW_SECRET}"}),
        session_id=session_id or f"raw-session-id-{RAW_SECRET}",
        category="tool-result",
        cache_creation_input_tokens=100,
        cache_read_input_tokens=3000,
        retry_count=retry_count,
        thinking_output_tokens=thinking_tokens,
        provider="anthropic",
        source_surface="anthropic_messages",
        endpoint="messages",
        requested_model_family="sonnet",
        routed_model_family="sonnet",
    )


def _log_continuation_call(
    store: Store,
    call_id: str,
    *,
    created_at: str,
    session_id: str,
    status_code: int = 200,
    retry_count: int = 0,
    fallback: bool = False,
    category: str = "tool-result",
    workflow_phase: str = "tool-execution",
) -> None:
    routing = {"category": category, "workflow_phase": workflow_phase}
    if fallback:
        routing["fallback"] = True
        routing["fallback_reason"] = "rate-limit"
    store.log_call(
        id=call_id,
        created_at=created_at,
        path="/v1/messages",
        requested_model="claude-sonnet-4-6",
        routed_model="claude-sonnet-4-6",
        stream=1,
        cache_hit=0,
        status_code=status_code,
        latency_ms=750,
        input_tokens_est=5000,
        output_tokens_est=200,
        actual_input_tokens=5000,
        actual_output_tokens=200,
        cost_est_usd=0.015,
        cost_baseline_usd=0.015,
        crunch_json=stable_json({"status": "skipped", "reason": "not-compaction"}),
        routing_json=stable_json(routing),
        cache_json=stable_json({"status": "skipped", "reason": "streaming", "cache_key": f"raw-cache-key-{RAW_SECRET}"}),
        error=f"continuation failure {RAW_SECRET}" if status_code >= 400 else None,
        request_json=stable_json({"messages": [{"content": f"raw continuation prompt {RAW_SECRET}"}]}),
        response_json=stable_json({"text": f"raw continuation response {RAW_SECRET}"}),
        session_id=session_id,
        category=category,
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


class AnthropicThinkingCompactionImpactTests(unittest.TestCase):
    def test_report_summarizes_lifecycle_impact_and_budget_feedback_without_content(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-impact-applied-1",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                    cost_est=0.020,
                )
                _log_call(
                    store,
                    "thinking-impact-applied-2",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=900,
                    planned_tokens=900,
                    cost_est=0.022,
                    thinking_tokens=80,
                )
                _log_call(
                    store,
                    "thinking-impact-holdout",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1000,
                    cost_est=0.050,
                    thinking_tokens=200,
                )
                _log_call(
                    store,
                    "thinking-impact-skipped",
                    created_at="2026-06-13T00:03:00+00:00",
                    status="skipped",
                    reason="active-top-level-thinking-request",
                    cohort="skipped",
                    cost_est=0.040,
                )
                _log_call(
                    store,
                    "thinking-impact-safety",
                    created_at="2026-06-13T00:04:00+00:00",
                    status="bypass",
                    reason="local-canary-safety-stop",
                    cohort="safety_stop",
                    status_code=500,
                    cost_est=0.040,
                    retry_count=1,
                )
                payload = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], "tokenclaw.anthropic_thinking_compaction_impact.v1")
        self.assertEqual(payload["summary"]["applied_count"], 2)
        self.assertEqual(payload["summary"]["holdout_count"], 1)
        self.assertEqual(payload["summary"]["skipped_count"], 1)
        self.assertEqual(payload["summary"]["safety_stop_count"], 1)
        self.assertEqual(payload["summary"]["observed_saved_chars"], 7600)
        self.assertEqual(payload["summary"]["observed_saved_tokens"], 1900)
        self.assertGreater(payload["summary"]["observed_saved_usd"], 0)
        self.assertEqual(payload["summary"]["projected_saved_chars"], 11600)
        self.assertEqual(payload["summary"]["projected_saved_tokens"], 2900)
        self.assertGreater(payload["summary"]["projected_saved_usd"], 0)
        self.assertGreater(payload["summary"]["avg_crunch_ratio"], 0)
        self.assertEqual(payload["summary"]["applied_minus_holdout_error_rate"], 0.0)
        self.assertEqual(payload["summary"]["applied_minus_holdout_retry_rate"], 0.0)
        self.assertEqual(payload["summary"]["canary_impact_decision"], "stop")
        self.assertEqual(payload["canary_impact_decision"]["decision"], "stop")
        self.assertGreater(payload["summary"]["tokens_saved_est"], 0)
        self.assertGreater(payload["summary"]["projected_holdout_savings_usd"], 0)
        coverage = payload["summary"]["lifecycle_coverage"]
        self.assertEqual(coverage["schema"], "tokenclaw.anthropic_thinking_compaction_lifecycle_coverage.v1")
        self.assertEqual(coverage["observed_count"], 5)
        self.assertEqual(coverage["applied_count"], 2)
        self.assertEqual(coverage["holdout_count"], 1)
        self.assertEqual(coverage["safety_stop_count"], 1)
        self.assertEqual(coverage["applied_error_count"], 0)
        self.assertEqual(coverage["applied_retry_count"], 0)
        self.assertGreater(coverage["tokens_saved_est"], 0)
        self.assertEqual(coverage["observed_saved_chars"], 7600)
        self.assertEqual(coverage["observed_saved_tokens"], 1900)
        self.assertEqual(coverage["projected_saved_chars"], 11600)
        self.assertEqual(coverage["projected_saved_tokens"], 2900)
        self.assertGreater(coverage["projected_saved_usd"], 0)
        self.assertGreater(coverage["projected_holdout_savings_usd"], 0)
        self.assertTrue(coverage["metadata_only"])
        self.assertFalse(coverage["raw_payload_included"])
        self.assertEqual(payload["budget_governor_feedback"]["recommended_budget_action"], "suppress")
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["cohorts"]["applied"]["count"], 2)
        self.assertEqual(candidate["cohorts"]["holdout"]["count"], 1)
        self.assertEqual(candidate["canary_impact_decision"], "stop")
        self.assertEqual(candidate["observed_saved_chars"], 7600)
        self.assertEqual(candidate["observed_saved_tokens"], 1900)
        self.assertEqual(candidate["projected_saved_chars"], 11600)
        self.assertEqual(candidate["projected_saved_tokens"], 2900)
        self.assertGreater(candidate["projected_saved_usd"], 0)
        self.assertGreater(candidate["avg_crunch_ratio"], 0)
        self.assertIn("applied_minus_holdout_error_rate", candidate["deltas"])
        self.assertIn("applied_minus_holdout_retry_rate", candidate["deltas"])
        self.assertLess(candidate["deltas"]["cost_avg_usd_delta"], 0)
        self.assertIn("safety-stop-observed", candidate["reason_codes"])
        self.assertEqual(candidate["session_budget_impact"]["affected_session_count"], 1)
        blockers = {item["value"] for item in payload["summary"]["blocker_reason_breakdown"]}
        self.assertIn("active-top-level-thinking-request", blockers)
        self.assertFalse(payload["privacy"]["raw_thinking_text_included"])
        self.assertFalse(payload["privacy"]["raw_request_bodies_included"])
        self.assertFalse(payload["privacy"]["request_ids_included"])
        self.assertFalse(payload["privacy"]["session_ids_included"])
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            RAW_SECRET,
            "private thinking",
            "raw-tool-id",
            "raw-request-id",
            "raw-session-id",
            "raw-cache-key",
            "/private/",
            "raw response",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cli_and_dashboard_endpoint_are_content_free(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "tokenclaw.sqlite3"
            store = Store(str(db_path))
            try:
                _log_call(
                    store,
                    "thinking-impact-dashboard",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                )
                _log_call(
                    store,
                    "thinking-impact-dashboard-holdout",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1000,
                )
            finally:
                store.conn.close()

            stdout = io.StringIO()
            code = cli.anthropic_thinking_compaction_impact_cli(["--db", str(db_path), "--limit", "10"], stdout=stdout)
            self.assertEqual(code, 0)
            cli_payload = json.loads(stdout.getvalue())
            self.assertEqual(cli_payload["summary"]["applied_count"], 1)

            dashboard_store = Store(str(db_path))
            try:
                app = create_dashboard_app(
                    store_obj=lambda: dashboard_store,
                    default_db=str(db_path),
                    upstream="https://anthropic.test",
                    limiter_status=lambda: [],
                    limiter_config={},
                    full_stats_ttl_s=0,
                )
                with TestClient(app) as client:
                    response = client.get("/tokenclaw/stats/anthropic-thinking-compaction-impact?limit=10")
                    dashboard = client.get("/tokenclaw/dashboard")
            finally:
                dashboard_store.conn.close()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(dashboard.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "tokenclaw.anthropic_thinking_compaction_impact.v1")
            self.assertIn("TokenClaw", dashboard.text)
            self.assertIn("/tokenclaw/stats", dashboard.text)
            rendered = stdout.getvalue() + json.dumps(payload, sort_keys=True) + dashboard.text
            self.assertNotIn(RAW_SECRET, rendered)
            self.assertNotIn("raw-session-id", rendered)
            self.assertNotIn("raw-request-id", rendered)

    def test_report_recommends_widen_when_canary_has_positive_impact(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-impact-widen-applied-1",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                    cost_est=0.020,
                )
                _log_call(
                    store,
                    "thinking-impact-widen-applied-2",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=900,
                    planned_tokens=900,
                    cost_est=0.022,
                )
                _log_call(
                    store,
                    "thinking-impact-widen-holdout",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1000,
                    cost_est=0.050,
                )
                payload = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        self.assertEqual(payload["summary"]["canary_impact_decision"], "widen")
        self.assertEqual(payload["canary_impact_decision"]["decision"], "widen")
        self.assertEqual(payload["budget_governor_feedback"]["recommended_budget_action"], "widen")
        self.assertEqual(payload["candidates"][0]["canary_impact_decision"], "widen")
        self.assertGreater(payload["summary"]["observed_saved_chars"], 0)
        self.assertGreater(payload["summary"]["projected_saved_usd"], 0)

    def test_report_summarizes_downstream_continuation_quality_without_content(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                applied_session = f"raw-applied-session-{RAW_SECRET}"
                holdout_session = f"raw-holdout-session-{RAW_SECRET}"
                _log_call(
                    store,
                    "thinking-continuation-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                    session_id=applied_session,
                )
                _log_continuation_call(
                    store,
                    "thinking-continuation-applied-next",
                    created_at="2026-06-13T00:01:00+00:00",
                    session_id=applied_session,
                    status_code=200,
                    category="tool-result",
                    workflow_phase="tool-execution",
                )
                _log_call(
                    store,
                    "thinking-continuation-holdout",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1100,
                    session_id=holdout_session,
                )
                _log_continuation_call(
                    store,
                    "thinking-continuation-holdout-next",
                    created_at="2026-06-13T00:03:00+00:00",
                    session_id=holdout_session,
                    status_code=500,
                    retry_count=2,
                    fallback=True,
                    category="chat",
                    workflow_phase="chat",
                )
                payload = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        quality = payload["candidates"][0]["continuation_quality"]
        self.assertEqual(quality["schema"], "tokenclaw.thinking_tail_continuation_quality.v1")
        self.assertTrue(quality["window"]["same_session_scoped"])
        applied = quality["cohorts"]["applied"]
        holdout = quality["cohorts"]["holdout"]
        self.assertEqual(applied["source_count"], 1)
        self.assertEqual(applied["evaluated_count"], 1)
        self.assertEqual(applied["success_continuation_count"], 1)
        self.assertEqual(applied["tool_continuation_count"], 1)
        self.assertEqual(applied["downstream_issue_count"], 0)
        self.assertEqual(holdout["source_count"], 1)
        self.assertEqual(holdout["evaluated_count"], 1)
        self.assertEqual(holdout["success_continuation_count"], 0)
        self.assertEqual(holdout["downstream_issue_count"], 1)
        self.assertEqual(holdout["immediate_error_count"], 1)
        self.assertEqual(holdout["retry_attempts"], 2)
        self.assertEqual(holdout["fallback_count"], 1)
        self.assertEqual(quality["applied_minus_holdout"]["success_continuation_rate_delta"], 1.0)
        self.assertEqual(quality["applied_minus_holdout"]["downstream_issue_rate_delta"], -1.0)
        self.assertEqual(quality["applied_minus_holdout"]["retry_rate_delta"], -1.0)
        self.assertEqual(quality["applied_minus_holdout"]["fallback_rate_delta"], -1.0)

        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            RAW_SECRET,
            "raw-applied-session",
            "raw-holdout-session",
            "raw continuation prompt",
            "raw continuation response",
            "raw-cache-key",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_managed_feedback_event_contains_crunch_outcome_rollups_only(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-feedback-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                    cost_est=0.020,
                )
                _log_call(
                    store,
                    "thinking-feedback-holdout",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1100,
                    cost_est=0.050,
                )
                _log_call(
                    store,
                    "thinking-feedback-safety",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="bypass",
                    reason="local-canary-safety-stop",
                    cohort="safety_stop",
                    status_code=500,
                    retry_count=2,
                    cost_est=0.040,
                )
                report = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        event = build_anthropic_thinking_compaction_managed_feedback(report, now="2026-06-13T00:03:00+00:00")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["schema"], MANAGED_FEEDBACK_SCHEMA)
        self.assertEqual(event["event_type"], "crunch_outcome_rollup")
        self.assertEqual(event["source_surface"], MANAGED_FEEDBACK_SOURCE_SURFACE)
        self.assertEqual(event["summary"]["applied_count"], 1)
        self.assertEqual(event["summary"]["holdout_count"], 1)
        self.assertEqual(event["summary"]["safety_stop_count"], 1)
        self.assertGreater(event["summary"]["observed_saved_tokens"], 0)
        self.assertGreater(event["summary"]["projected_saved_usd"], 0)
        outcome = event["crunch_outcomes"][0]
        self.assertEqual(outcome["target_action_family"], "anthropic_thinking_history_compaction")
        self.assertEqual(outcome["cohort_counts"], {"applied": 1, "holdout": 1, "skipped": 0, "safety_stop": 1})
        self.assertEqual(outcome["outcomes"]["applied"]["estimated_saved_tokens"], 1000)
        self.assertEqual(outcome["outcomes"]["holdout"]["planned_saved_tokens"], 1100)
        self.assertEqual(outcome["outcomes"]["safety_stop"]["error_count"], 1)
        self.assertEqual(outcome["outcomes"]["safety_stop"]["retry_attempts"], 2)
        pricing = outcome["outcomes"]["applied"]["prompt_cache_blended_pricing_inputs"]
        self.assertEqual(pricing["actual_input_tokens"], 6000)
        self.assertEqual(pricing["prompt_cache_read_tokens"], 3000)
        self.assertEqual(managed_egress_violations(event), [])
        rendered = json.dumps(event, sort_keys=True)
        for forbidden in (
            RAW_SECRET,
            "private thinking",
            "raw-tool-id",
            "raw-request-id",
            "raw-session-id",
            "raw-cache-key",
            "/private/",
            "raw response",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_managed_feedback_populates_server_requested_outcome_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1200,
                    planned_tokens=1200,
                    cost_est=0.020,
                )
                _log_call(
                    store,
                    "thinking-holdout",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=1300,
                    cost_est=0.050,
                )
                report = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        event = build_anthropic_thinking_compaction_managed_feedback(report, now="2026-06-13T00:02:00+00:00")
        self.assertIsNotNone(event)
        assert event is not None

        # The rollup declares full coverage of the server-requested outcome fields.
        self.assertEqual(event["measured_outcome_fields"], list(SERVER_REQUESTED_OUTCOME_FIELDS))

        outcome = event["crunch_outcomes"][0]
        # Candidate-level treatment metadata is present (applied present, no stop).
        self.assertEqual(outcome["traffic_treatment"], "canary")
        self.assertEqual(outcome["action_state"], "applied")
        self.assertEqual(outcome["applied_action_families"], ["crunch"])
        self.assertEqual(outcome["vetoed_action_families"], [])
        self.assertIsNone(outcome["safety_stop_reason"])
        self.assertIsNone(outcome["rollback_reason"])

        # Every cohort carries every server-requested outcome field by name.
        for cohort_name in ("applied", "holdout", "skipped", "safety_stop"):
            fields = outcome["outcomes"][cohort_name]["server_outcome_fields"]
            present = set(fields) - {"schema"}
            self.assertEqual(
                present,
                set(SERVER_REQUESTED_OUTCOME_FIELDS),
                msg=f"cohort {cohort_name} missing fields: {set(SERVER_REQUESTED_OUTCOME_FIELDS) - present}",
            )
            self.assertEqual(fields["cohort"], cohort_name)
            self.assertEqual(fields["traffic_treatment"], "canary")

        applied_fields = outcome["outcomes"]["applied"]["server_outcome_fields"]
        self.assertEqual(applied_fields["action_state"], "applied")
        self.assertEqual(applied_fields["error_class"], "none")
        self.assertEqual(applied_fields["fallback_count"], 0)
        self.assertEqual(applied_fields["saved_tokens"], 1200)
        self.assertEqual(applied_fields["actual_input_tokens"], 6000)
        self.assertEqual(applied_fields["cache_read_input_tokens"], 3000)
        self.assertEqual(applied_fields["cache_creation_input_tokens"], 100)
        self.assertGreater(applied_fields["realized_savings_usd"], 0.0)
        self.assertEqual(outcome["outcomes"]["holdout"]["server_outcome_fields"]["action_state"], "heldout")

        # Privacy guarantees still hold.
        self.assertEqual(managed_egress_violations(event), [])
        rendered = json.dumps(event, sort_keys=True)
        for forbidden in (RAW_SECRET, "private thinking", "raw-tool-id", "raw-session-id", "raw-cache-key", "/private/"):
            self.assertNotIn(forbidden, rendered)

    def test_managed_feedback_includes_continuation_quality_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                applied_session = f"raw-managed-applied-session-{RAW_SECRET}"
                holdout_session = f"raw-managed-holdout-session-{RAW_SECRET}"
                _log_call(
                    store,
                    "thinking-managed-continuation-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                    session_id=applied_session,
                )
                _log_continuation_call(
                    store,
                    "thinking-managed-continuation-applied-next",
                    created_at="2026-06-13T00:01:00+00:00",
                    session_id=applied_session,
                )
                _log_call(
                    store,
                    "thinking-managed-continuation-holdout",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=900,
                    session_id=holdout_session,
                )
                _log_continuation_call(
                    store,
                    "thinking-managed-continuation-holdout-next",
                    created_at="2026-06-13T00:03:00+00:00",
                    session_id=holdout_session,
                    status_code=429,
                    retry_count=1,
                    fallback=True,
                    category="chat",
                    workflow_phase="chat",
                )
                report = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        event = build_anthropic_thinking_compaction_managed_feedback(report, now="2026-06-13T00:04:00+00:00")
        self.assertIsNotNone(event)
        assert event is not None
        outcome = event["crunch_outcomes"][0]
        quality = outcome["continuation_quality"]
        self.assertEqual(quality["schema"], "tokenclaw.thinking_tail_continuation_quality.v1")
        self.assertEqual(quality["cohorts"]["applied"]["success_continuation_count"], 1)
        self.assertEqual(quality["cohorts"]["holdout"]["downstream_issue_count"], 1)
        self.assertEqual(quality["cohorts"]["holdout"]["fallback_count"], 1)
        self.assertEqual(quality["applied_minus_holdout"]["success_continuation_rate_delta"], 1.0)
        self.assertEqual(managed_egress_violations(event), [])
        rendered = json.dumps(event, sort_keys=True)
        for forbidden in (
            RAW_SECRET,
            "raw-managed-applied-session",
            "raw-managed-holdout-session",
            "raw continuation prompt",
            "raw continuation response",
            "raw-cache-key",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_managed_feedback_reports_rollback_and_fallback_outcome_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=800,
                    planned_tokens=800,
                )
                _log_call(
                    store,
                    "thinking-holdout",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=900,
                )
                _log_call(
                    store,
                    "thinking-safety",
                    created_at="2026-06-13T00:02:00+00:00",
                    status="bypass",
                    reason="local-canary-safety-stop",
                    cohort="safety_stop",
                    status_code=500,
                    retry_count=2,
                    fallback=True,
                )
                report = build_anthropic_thinking_compaction_impact_report(store, limit=20)
            finally:
                store.conn.close()

        event = build_anthropic_thinking_compaction_managed_feedback(report, now="2026-06-13T00:03:00+00:00")
        assert event is not None
        outcome = event["crunch_outcomes"][0]
        # A safety stop drives the candidate to a rollback treatment with a reason.
        self.assertEqual(outcome["traffic_treatment"], "rollback")
        self.assertEqual(outcome["action_state"], "safety_stopped")
        self.assertIsNotNone(outcome["safety_stop_reason"])
        self.assertIsNotNone(outcome["rollback_reason"])

        safety_fields = outcome["outcomes"]["safety_stop"]["server_outcome_fields"]
        self.assertEqual(safety_fields["action_state"], "safety_stopped")
        self.assertEqual(safety_fields["error_class"], "server_error")
        self.assertEqual(safety_fields["fallback_count"], 1)
        self.assertEqual(safety_fields["retry_count"], 2)
        self.assertIsNotNone(safety_fields["safety_stop_reason"])
        self.assertEqual(managed_egress_violations(event), [])

    def test_managed_feedback_egress_guard_rejects_raw_like_fields(self) -> None:
        event = {
            "schema": MANAGED_FEEDBACK_SCHEMA,
            "event_type": "crunch_outcome_rollup",
            "crunch_outcomes": [
                {
                    "candidate_id": "safe-candidate",
                    "messages": [{"content": f"raw thinking {RAW_SECRET}"}],
                }
            ],
        }

        violations = managed_egress_violations(event)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["reason"], "raw-like-key")

    def test_managed_feedback_queue_helper_queues_when_managed_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                _log_call(
                    store,
                    "thinking-feedback-queue-applied",
                    created_at="2026-06-13T00:00:00+00:00",
                    status="applied",
                    reason="thinking-history-compaction-applied",
                    applied=True,
                    tokens_saved=1000,
                    planned_tokens=1000,
                )
                _log_call(
                    store,
                    "thinking-feedback-queue-holdout",
                    created_at="2026-06-13T00:01:00+00:00",
                    status="holdout",
                    reason="canary_holdout",
                    cohort="canary_holdout",
                    planned_tokens=900,
                )
                report = build_anthropic_thinking_compaction_impact_report(store, limit=20)
                with patch.dict(os.environ, {"TOKENCLAW_RECOMMENDATION_ENABLED": "0"}, clear=False):
                    meta = asyncio.run(
                        queue_anthropic_thinking_compaction_managed_feedback(
                            store,
                            report,
                            flush_immediately=False,
                        )
                    )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self.assertEqual(row["source_surface"], MANAGED_FEEDBACK_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertIn('"crunch_outcomes"', row["payload_json"])
        self.assertIn('"applied_count":1', row["payload_json"])
        self.assertNotIn(RAW_SECRET, row["payload_json"])


if __name__ == "__main__":
    unittest.main()
