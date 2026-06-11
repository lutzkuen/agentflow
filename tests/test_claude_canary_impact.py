import asyncio
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from agentflow_proxy import cli
from agentflow_proxy.claude_canary_impact import build_claude_canary_impact_report
from agentflow_proxy.store import Store, stable_json


HAS_RUNTIME_DEPS = all(importlib.util.find_spec(module_name) is not None for module_name in ("fastapi", "httpx"))

if HAS_RUNTIME_DEPS:
    from fastapi.testclient import TestClient

    from agentflow_proxy.dashboard_app import create_dashboard_app
    import agentflow_proxy.stats as stats_views


def _assert_privacy_clean(testcase: unittest.TestCase, payload: dict) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "raw claude prompt secret",
        "raw claude response secret",
        "raw tool payload secret",
        "claude-cache-key-secret",
        "claude-request-id-secret",
        "claude-session-id-secret",
        "sk-ant-secret",
    ):
        testcase.assertNotIn(forbidden, rendered)
    for forbidden_key in (
        '"api_key"',
        '"cache_key"',
        '"content"',
        '"messages"',
        '"prompt"',
        '"raw_request"',
        '"raw_response"',
        '"request_id"',
        '"session_id"',
        '"tool_payload"',
    ):
        testcase.assertNotIn(forbidden_key, rendered)


class ClaudeCanaryImpactTests(unittest.TestCase):
    def _log_claude_canary_call(
        self,
        store: Store,
        *,
        candidate_id: str = "claude-canary-candidate",
        cohort: str,
        suffix: str,
        status_code: int = 200,
        retry_count: int = 0,
        latency_ms: int = 1000,
        cost_est: float = 0.001,
        cost_baseline: float = 0.003,
        created_at: str = "2026-06-10T04:00:00+00:00",
        fallback_reason: str | None = None,
        reason: str | None = None,
        workflow_phase: str = "tool-execution",
        stream: int = 1,
        stripped_params: list[str] | None = None,
        dangerous_meta: bool = False,
    ) -> None:
        status = "applied" if cohort == "canary_applied" else "holdout" if cohort == "canary_holdout" else "safety_stopped"
        canary = {
            "enabled": True,
            "policy_id": "test-claude-phase-canary",
            "rule_id": "test-claude-phase-canary",
            "promotion_action_id": "test-claude-action",
            "target_candidate_id": candidate_id,
            "candidate_id": candidate_id,
            "status": status,
            "cohort": cohort,
            "reason": reason or ("selected-canary" if status == "applied" else "selected-holdout" if status == "holdout" else "safety-stop-tripped"),
            "requested_model": "claude-sonnet-4-6",
            "target_model": "claude-haiku-4-5-20251001",
            "actual_forwarded_model": "claude-haiku-4-5-20251001" if status == "applied" and not fallback_reason else "claude-sonnet-4-6",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "category": "tool-result",
            "workflow_phase": workflow_phase,
            "workflow_phase_confidence": "high",
            "stream": bool(stream),
            "text_bucket": "8k-30k",
            "canary_fraction": 0.5,
            "holdout_fraction": 0.25,
            "policy_source": "local-manual",
            "cohort_key_hash": f"sha256:test-{cohort}-{suffix}",
        }
        if fallback_reason:
            canary["fallback_reason"] = fallback_reason
            canary["actual_forwarded_model"] = "claude-sonnet-4-6"
        if status == "safety_stopped":
            canary["safety_stop"] = {"tripped": True, "reason_codes": ["error-rate"]}
        if dangerous_meta:
            canary.update({
                "prompt": "raw claude prompt secret",
                "messages": [{"role": "user", "content": "raw claude prompt secret"}],
                "raw_response": {"body": "raw claude response secret"},
                "tool_payload": {"output": "raw tool payload secret"},
                "cache_key": "claude-cache-key-secret",
                "request_id": "claude-request-id-secret",
                "session_id": "claude-session-id-secret",
                "api_key": "sk-ant-secret",
            })
        routing = {"phase_canary": canary}
        if stripped_params:
            routing["stripped_params"] = stripped_params
        store.log_call(
            id=f"claude-canary-impact-{cohort}-{suffix}",
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-6",
            routed_model=canary["actual_forwarded_model"],
            stream=stream,
            cache_hit=0,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=500,
            output_tokens_est=100,
            actual_input_tokens=500,
            actual_output_tokens=100,
            cost_est_usd=cost_est,
            cost_baseline_usd=cost_baseline,
            crunch_json=stable_json({"changed": suffix.endswith("crunch")}),
            routing_json=stable_json(routing),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            error='{"error":{"message":"raw claude response secret"}}' if status_code >= 400 else None,
            request_json=None,
            response_json=None,
            session_id="claude-session-id-secret",
            category="tool-result",
            retry_count=retry_count,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model_family="sonnet",
            routed_model_family="haiku",
        )

    def test_claude_canary_impact_reports_widen_cli_and_dashboard_json_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_claude_canary_call(store, cohort="canary_applied", suffix="a1", stripped_params=["thinking", "effort"], dangerous_meta=True)
                self._log_claude_canary_call(store, cohort="canary_applied", suffix="a2-crunch")
                self._log_claude_canary_call(store, cohort="canary_holdout", suffix="h1", cost_est=0.003, cost_baseline=0.003)
                impact = build_claude_canary_impact_report(
                    store,
                    limit=10,
                    min_applied_samples=2,
                    min_holdout_samples=1,
                    now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                )
                stats_payload = asyncio.run(stats_views.stats_claude_canary_impact(store, limit=10)) if HAS_RUNTIME_DEPS else None
                if HAS_RUNTIME_DEPS:
                    app = create_dashboard_app(
                        store_obj=lambda: store,
                        default_db=db_path,
                        upstream="https://anthropic.test",
                        limiter_status=lambda: [],
                        limiter_config={
                            "min_request_interval_ms": 0,
                            "max_tier_backoff_wait_s": 30,
                            "max_concurrent_per_tier": 2,
                        },
                    )
                    endpoint_payload = TestClient(app).get("/agentflow/stats/claude-canary-impact?limit=10")
            finally:
                store.conn.close()

            cli_output = io.StringIO()
            exit_code = cli.claude_canary_impact_cli(["--db", db_path, "--limit", "10"], stdout=cli_output)

        self.assertEqual(impact["schema"], "agentflow.claude_canary_impact.v1")
        self.assertEqual(impact["summary"]["observed_claude_canary_metadata_row_count"], 3)
        self.assertEqual(impact["summary"]["canary_applied_count"], 2)
        self.assertEqual(impact["summary"]["canary_holdout_count"], 1)
        candidate = impact["candidates"][0]
        self.assertEqual(candidate["verdict"], "widen")
        self.assertEqual(candidate["provider"], "anthropic")
        self.assertEqual(candidate["source_surface"], "anthropic_messages")
        self.assertEqual(candidate["category"], "tool-result")
        self.assertEqual(candidate["workflow_phase"], "tool-execution")
        self.assertTrue(candidate["stream"])
        self.assertEqual(candidate["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(candidate["stripped_param_counts"][0]["value"], "effort")
        self.assertFalse(impact["privacy"]["raw_prompts_included"])
        _assert_privacy_clean(self, impact)

        self.assertEqual(exit_code, 0)
        cli_payload = json.loads(cli_output.getvalue())
        self.assertEqual(cli_payload["schema"], "agentflow.claude_canary_impact.v1")
        _assert_privacy_clean(self, cli_payload)

        if HAS_RUNTIME_DEPS:
            self.assertEqual(stats_payload["schema"], "agentflow.claude_canary_impact.v1")
            self.assertEqual(endpoint_payload.status_code, 200)
            self.assertEqual(endpoint_payload.json()["schema"], "agentflow.claude_canary_impact.v1")
            _assert_privacy_clean(self, endpoint_payload.json())

    def test_claude_canary_impact_verdicts_cover_gates(self) -> None:
        scenarios = (
            ("insufficient", "needs_more_samples", "insufficient-holdout-samples"),
            ("error", "hold", "error-rate-regression"),
            ("fallback", "hold", "fallback-rate-regression"),
            ("stale", "hold", "stale-evidence"),
            ("rollback", "rollback", "rollback-error-rate"),
        )
        for scenario, expected_verdict, expected_reason in scenarios:
            with self.subTest(scenario=scenario):
                with tempfile.TemporaryDirectory() as tmp:
                    store = Store(str(Path(tmp) / "agentflow.sqlite3"))
                    try:
                        candidate_id = f"candidate-{scenario}"
                        if scenario == "insufficient":
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", suffix="a1")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", suffix="a2")
                        elif scenario == "error":
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", status_code=500, suffix="a1")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", suffix="a2")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        elif scenario == "fallback":
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", fallback_reason="rate_limited", suffix="a1")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", suffix="a2")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        elif scenario == "stale":
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", created_at="2026-06-01T00:00:00+00:00", suffix="a1")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", created_at="2026-06-01T00:00:01+00:00", suffix="a2")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, created_at="2026-06-01T00:00:02+00:00", suffix="h1")
                        else:
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", status_code=500, suffix="a1")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_applied", status_code=500, suffix="a2")
                            self._log_claude_canary_call(store, candidate_id=candidate_id, cohort="canary_holdout", cost_est=0.003, cost_baseline=0.003, suffix="h1")
                        report = build_claude_canary_impact_report(
                            store,
                            limit=10,
                            since="2026-06-01T00:00:00+00:00" if scenario == "stale" else None,
                            min_applied_samples=2,
                            min_holdout_samples=1,
                            max_evidence_age_hours=1,
                            max_error_rate_delta=0.10,
                            max_fallback_rate_delta=0.10,
                            rollback_error_rate=0.90 if scenario != "rollback" else 0.20,
                            rollback_fallback_rate=1.0,
                            now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                        )
                    finally:
                        store.conn.close()
                candidate = report["candidates"][0]
                self.assertEqual(candidate["verdict"], expected_verdict)
                self.assertIn(expected_reason, candidate["reason_codes"])
                if scenario == "fallback":
                    self.assertEqual(candidate["cohort_metrics"]["canary_applied"]["rate_limit_fallback_count"], 1)
                    self.assertGreater(candidate["requested_model_fallback_cost_usd"], 0)
                _assert_privacy_clean(self, report)


if __name__ == "__main__":
    unittest.main()
