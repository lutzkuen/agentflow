from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from agentflow_proxy import cli
from agentflow_proxy.anthropic_thinking_compaction_impact import build_anthropic_thinking_compaction_impact_report
from agentflow_proxy.dashboard_app import create_dashboard_app
from agentflow_proxy.store import Store, stable_json


RAW_SECRET = "raw-thinking-impact-secret"


def _compaction_meta(
    *,
    status: str,
    reason: str,
    applied: bool = False,
    cohort: str = "canary_applied",
    tokens_saved: int = 0,
    planned_tokens: int = 0,
) -> dict:
    meta = {
        "schema": "agentflow.anthropic_thinking_history_compaction_decision.v1",
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
            "schema": "agentflow.anthropic_thinking_history_compaction_lifecycle_feedback.v1",
            "status": status if status != "bypass" else "safety_stop",
            "cohort": cohort,
            "candidate_id": f"raw-lifecycle-{RAW_SECRET}",
            "metadata_only": True,
            "raw_payload_included": False,
        },
    }
    if status == "bypass":
        meta["safety_stop_state"] = "stopped"
        meta["safety_stop"] = {
            "schema": "agentflow.anthropic_thinking_history_compaction_safety_stop.v1",
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
        session_id=f"raw-session-id-{RAW_SECRET}",
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


class AnthropicThinkingCompactionImpactTests(unittest.TestCase):
    def test_report_summarizes_lifecycle_impact_and_budget_feedback_without_content(self) -> None:
        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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

        self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_impact.v1")
        self.assertEqual(payload["summary"]["applied_count"], 2)
        self.assertEqual(payload["summary"]["holdout_count"], 1)
        self.assertEqual(payload["summary"]["skipped_count"], 1)
        self.assertEqual(payload["summary"]["safety_stop_count"], 1)
        self.assertGreater(payload["summary"]["tokens_saved_est"], 0)
        self.assertGreater(payload["summary"]["projected_holdout_savings_usd"], 0)
        self.assertEqual(payload["budget_governor_feedback"]["recommended_budget_action"], "suppress")
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["cohorts"]["applied"]["count"], 2)
        self.assertEqual(candidate["cohorts"]["holdout"]["count"], 1)
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
            db_path = Path(tmp) / "agentflow.sqlite3"
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
                    response = client.get("/agentflow/stats/anthropic-thinking-compaction-impact?limit=10")
                    dashboard = client.get("/agentflow/dashboard")
            finally:
                dashboard_store.conn.close()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(dashboard.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema"], "agentflow.anthropic_thinking_compaction_impact.v1")
            self.assertIn("Thinking-compaction impact", dashboard.text)
            self.assertIn("/agentflow/stats/anthropic-thinking-compaction-impact?limit=500", dashboard.text)
            rendered = stdout.getvalue() + json.dumps(payload, sort_keys=True) + dashboard.text
            self.assertNotIn(RAW_SECRET, rendered)
            self.assertNotIn("raw-session-id", rendered)
            self.assertNotIn("raw-request-id", rendered)


if __name__ == "__main__":
    unittest.main()
