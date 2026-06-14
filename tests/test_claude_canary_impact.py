import asyncio
from datetime import datetime, timezone
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import yaml

from agentflow_proxy import cli
from agentflow_proxy.claude_canary_actions import apply_claude_canary_actions, build_claude_canary_actions
from agentflow_proxy.claude_canary_impact import build_anthropic_routing_canary_lifecycle_report, build_claude_canary_impact_report
from agentflow_proxy.routing_canary_promote import build_routing_canary_promotion_plan
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
        safety_reason_codes: list[str] | None = None,
        dangerous_meta: bool = False,
    ) -> None:
        status_by_cohort = {
            "canary_applied": "applied",
            "canary_holdout": "holdout",
            "safety_stopped": "safety_stopped",
            "skipped": "skipped",
            "bypassed_or_disabled": "disabled",
        }
        status = status_by_cohort.get(cohort, "unknown")
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
            canary["safety_stop"] = {"tripped": True, "reason_codes": safety_reason_codes or ["error-rate"]}
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

    def test_anthropic_routing_lifecycle_report_covers_holdout_thinking_guard_fallback_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "agentflow.sqlite3")
            store = Store(db_path)
            try:
                self._log_claude_canary_call(store, candidate_id="candidate-lifecycle", cohort="canary_applied", suffix="a1", dangerous_meta=True)
                self._log_claude_canary_call(
                    store,
                    candidate_id="candidate-lifecycle",
                    cohort="canary_applied",
                    suffix="a2",
                    fallback_reason="rate_limited",
                    retry_count=1,
                )
                self._log_claude_canary_call(store, candidate_id="candidate-lifecycle", cohort="canary_holdout", suffix="h1", cost_est=0.003, cost_baseline=0.003)
                self._log_claude_canary_call(
                    store,
                    candidate_id="candidate-lifecycle",
                    cohort="safety_stopped",
                    suffix="s1",
                    reason="thinking-safety-gate",
                    safety_reason_codes=["thinking-history-blocked"],
                )
                self._log_claude_canary_call(store, candidate_id="candidate-missing-holdout", cohort="canary_applied", suffix="m1")
                self._log_claude_canary_call(
                    store,
                    candidate_id="candidate-stale",
                    cohort="canary_applied",
                    suffix="old-a",
                    created_at="2026-06-01T00:00:00+00:00",
                )
                self._log_claude_canary_call(
                    store,
                    candidate_id="candidate-stale",
                    cohort="canary_holdout",
                    suffix="old-h",
                    created_at="2026-06-01T00:00:01+00:00",
                    cost_est=0.003,
                    cost_baseline=0.003,
                )

                report = build_anthropic_routing_canary_lifecycle_report(
                    store,
                    limit=20,
                    max_evidence_age_hours=1,
                    now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                )
            finally:
                store.conn.close()

            cli_output = io.StringIO()
            exit_code = cli.anthropic_routing_lifecycle_report_cli(
                ["--db", db_path, "--limit", "20", "--max-evidence-age-hours", "1"],
                stdout=cli_output,
            )

        self.assertEqual(report["schema"], "agentflow.anthropic_routing_canary_lifecycle_report.v1")
        self.assertEqual(report["summary"]["canary_applied_count"], 4)
        self.assertEqual(report["summary"]["canary_holdout_count"], 2)
        self.assertEqual(report["summary"]["safety_stopped_count"], 1)
        self.assertEqual(report["summary"]["fallback_count"], 1)
        self.assertEqual(report["summary"]["retry_count"], 1)

        by_candidate = {candidate["candidate_id"]: candidate for candidate in report["candidates"]}
        lifecycle = by_candidate["candidate-lifecycle"]["anthropic_canary_lifecycle_evidence"]
        self.assertEqual(lifecycle["schema"], "agentflow.anthropic_routing_canary_lifecycle_evidence.v1")
        self.assertEqual(lifecycle["cohort_counts"]["canary_applied"], 2)
        self.assertEqual(lifecycle["cohort_counts"]["canary_holdout"], 1)
        self.assertEqual(lifecycle["cohort_counts"]["safety_stopped"], 1)
        self.assertEqual(lifecycle["fallback_count"], 1)
        self.assertEqual(lifecycle["retry_count"], 1)
        self.assertIn("thinking-routing-guard", lifecycle["blocker_codes"])
        self.assertIn("thinking-history-blocked", lifecycle["blocker_codes"])

        missing_holdout = by_candidate["candidate-missing-holdout"]["anthropic_canary_lifecycle_evidence"]
        self.assertIn("missing-holdout-coverage", missing_holdout["blocker_codes"])
        stale = by_candidate["candidate-stale"]["anthropic_canary_lifecycle_evidence"]
        self.assertTrue(stale["stale_evidence"]["stale"])
        self.assertIn("stale-evidence", stale["blocker_codes"])

        self.assertEqual(exit_code, 0)
        cli_payload = json.loads(cli_output.getvalue())
        self.assertEqual(cli_payload["schema"], "agentflow.anthropic_routing_canary_lifecycle_report.v1")
        _assert_privacy_clean(self, report)
        _assert_privacy_clean(self, cli_payload)

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

    def test_claude_canary_impact_holds_on_provider_adoption_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self._log_claude_canary_call(store, cohort="canary_applied", suffix="a1")
                self._log_claude_canary_call(store, cohort="canary_holdout", suffix="h1", cost_est=0.003, cost_baseline=0.003)
                store.log_provider_tool_adoption_window(
                    id="claude-adoption-applied-risk",
                    created_at="2026-06-10T04:00:05+00:00",
                    updated_at="2026-06-10T05:00:05+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    app_family="claude_code",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    category="tool-result",
                    workflow_phase="tool-execution",
                    policy_source="local-manual",
                    policy_ids_json=stable_json(["test-claude-phase-canary"]),
                    call_id="claude-canary-impact-canary_applied-a1",
                    fulfilled_call_id=None,
                    session_digest="sha256:secret-session",
                    correlation_digest="sha256:secret-tool",
                    status="abandoned",
                    reason="ttl-expired-without-tool-result",
                    age_bucket="1_6h",
                    tool_use_count=1,
                    tool_result_count=0,
                    metadata_json=stable_json({"metadata_only": True}),
                )
                store.log_provider_tool_adoption_window(
                    id="claude-adoption-holdout-fulfilled",
                    created_at="2026-06-10T04:00:07+00:00",
                    updated_at="2026-06-10T04:00:17+00:00",
                    provider="anthropic",
                    source_surface="anthropic_messages",
                    endpoint="messages",
                    app_family="claude_code",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    category="tool-result",
                    workflow_phase="tool-execution",
                    policy_source="local-manual",
                    policy_ids_json=stable_json(["test-claude-phase-canary"]),
                    call_id="claude-canary-impact-canary_holdout-h1",
                    fulfilled_call_id="claude-canary-impact-canary_holdout-h1",
                    session_digest="sha256:secret-session",
                    correlation_digest="sha256:secret-tool-holdout",
                    status="fulfilled",
                    reason="matched-subsequent-tool-result",
                    age_bucket="0_1m",
                    tool_use_count=1,
                    tool_result_count=1,
                    metadata_json=stable_json({"metadata_only": True}),
                )
                impact = build_claude_canary_impact_report(
                    store,
                    limit=10,
                    min_applied_samples=1,
                    min_holdout_samples=1,
                    now=datetime(2026, 6, 10, 5, tzinfo=timezone.utc),
                )
            finally:
                store.conn.close()

        candidate = impact["candidates"][0]
        self.assertEqual(candidate["verdict"], "hold")
        self.assertIn("provider-adoption-regression", candidate["reason_codes"])
        self.assertTrue(candidate["provider_adoption_gate"]["blocking"])
        self.assertEqual(candidate["provider_adoption_gate"]["cohorts"]["applied"]["abandoned_count"], 1)
        rendered = stable_json(impact)
        self.assertNotIn("secret-tool", rendered)
        self.assertNotIn("secret-session", rendered)


class ClaudeCanaryActionTests(unittest.TestCase):
    def _candidate(self, *, verdict: str, candidate_id: str = "claude-action-candidate", canary_fraction: float = 0.5, holdout_fraction: float = 0.25, reason_codes: list[str] | None = None) -> dict:
        return {
            "schema": "agentflow.claude_canary_promotion_verdict.v1",
            "candidate_id": candidate_id,
            "rule_id": "test-claude-phase-canary",
            "policy_id": "test-claude-phase-canary",
            "promotion_action_id": "prior-action",
            "target_candidate_id": candidate_id,
            "policy_source": "local-manual",
            "optimization_family": "claude_phase_routing",
            "action_family": "routing",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "app_family": "claude_code",
            "original_model": "claude-sonnet-4-6",
            "candidate_target_model": "claude-haiku-4-5-20251001",
            "category": "tool-result",
            "workflow_phase": "tool-execution",
            "workflow_phase_confidence": "high",
            "stream": True,
            "canary_fraction": canary_fraction,
            "holdout_fraction": holdout_fraction,
            "sample_count": 12,
            "cohort_counts": {"canary_applied": 8, "canary_holdout": 4},
            "applied_vs_holdout_deltas": {
                "applied_minus_holdout_error_rate": 0.0,
                "applied_minus_holdout_retry_rate": 0.0,
                "applied_minus_holdout_fallback_rate": 0.0,
                "applied_minus_holdout_latency_avg_ms": -200,
            },
            "observed_savings_usd": 0.125,
            "requested_model_fallback_cost_usd": 0.0,
            "stale_evidence": {"stale": False, "age_hours": 1.0, "max_age_hours": 72.0},
            "verdict": verdict,
            "reason_codes": reason_codes or ["target-savings-met"],
            "warning_codes": [],
        }

    def _impact(self, candidates: list[dict]) -> dict:
        return {
            "schema": "agentflow.claude_canary_impact.v1",
            "generated_at": "2026-06-10T05:00:00+00:00",
            "read_only": True,
            "wrote_local_files": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "summary": {"candidate_group_count": len(candidates)},
            "candidates": candidates,
            "privacy": {"metadata_only": True, "content_free": True},
        }

    def test_claude_canary_actions_emit_widen_hold_rollback_and_more_samples(self) -> None:
        impact = self._impact([
            self._candidate(verdict="widen", candidate_id="candidate-widen", canary_fraction=0.7, holdout_fraction=0.1),
            self._candidate(verdict="hold", candidate_id="candidate-hold", canary_fraction=0.35, holdout_fraction=0.12, reason_codes=["latency-regression"]),
            self._candidate(verdict="rollback", candidate_id="candidate-rollback", reason_codes=["rollback-error-rate", "retry-rate-regression", "rate-limit-fallback-regression", "latency-regression", "negative-observed-savings", "safety-stop-observed"]),
            self._candidate(verdict="needs_more_samples", candidate_id="candidate-samples", reason_codes=["insufficient-holdout-samples"]),
        ])

        actions = build_claude_canary_actions(
            impact,
            widen_step=0.25,
            max_canary_fraction=0.95,
            preserved_holdout_fraction=0.20,
        )

        self.assertEqual(actions["schema"], "agentflow.claude_canary_rollout_actions.v1")
        by_id = {action["target_candidate_id"]: action for action in actions["actions"]}
        self.assertEqual(by_id["candidate-widen"]["action_type"], "widen")
        self.assertEqual(by_id["candidate-widen"]["canary_fraction"], 0.8)
        self.assertEqual(by_id["candidate-widen"]["holdout_fraction"], 0.2)
        self.assertEqual(by_id["candidate-hold"]["action_type"], "hold")
        self.assertEqual(by_id["candidate-hold"]["canary_fraction"], 0.35)
        self.assertEqual(by_id["candidate-hold"]["holdout_fraction"], 0.2)
        self.assertEqual(by_id["candidate-rollback"]["action_type"], "rollback")
        self.assertEqual(by_id["candidate-rollback"]["canary_fraction"], 0.0)
        self.assertEqual(by_id["candidate-rollback"]["holdout_fraction"], 0.0)
        self.assertIn("rollback-error", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertIn("rollback-retry", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertIn("rollback-fallback", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertIn("rollback-latency", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertIn("rollback-cost", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertIn("rollback-safety-stop", by_id["candidate-rollback"]["rollback_metadata"]["rollback_reason_codes"])
        self.assertEqual(by_id["candidate-samples"]["action_type"], "more-samples")
        self.assertEqual(by_id["candidate-samples"]["local_policy_update"]["target_local_policy"], "phase_canary")
        _assert_privacy_clean(self, actions)

    def test_claude_canary_actions_dry_run_and_write_only_routing_policy_with_event(self) -> None:
        action_bundle = build_claude_canary_actions(
            self._impact([self._candidate(verdict="widen", canary_fraction=0.1, holdout_fraction=0.1)]),
            widen_step=0.2,
            max_canary_fraction=0.5,
            preserved_holdout_fraction=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            event_log = tmp_path / "policy_events.jsonl"
            routing_file = tmp_path / "routing_rules.yaml"
            routing_file.write_text(
                yaml.safe_dump({
                    "phase_canary": {
                        "enabled": True,
                        "policy_id": "test-claude-phase-canary",
                        "policy_source": "local-manual",
                        "canary_fraction": 0.1,
                        "holdout_fraction": 0.1,
                    },
                    "openai_canary": {"enabled": False, "policy_id": "openai-untouched"},
                    "rules": [{"conditions": {"model_pattern": "sonnet"}, "action": {"route_to": "haiku", "reason": "fixture"}}],
                }, sort_keys=False),
                encoding="utf-8",
            )
            before = routing_file.read_text(encoding="utf-8")
            dry_run = apply_claude_canary_actions(action_bundle, config_dir=tmp_path, dry_run=True)

            self.assertTrue(dry_run["ok"])
            self.assertTrue(dry_run["dry_run"])
            self.assertFalse(dry_run["wrote_policy_files"])
            self.assertEqual(routing_file.read_text(encoding="utf-8"), before)
            self.assertEqual(dry_run["files"][0]["path"], str(routing_file))
            self.assertTrue(dry_run["files"][0]["changed"])

            old_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(event_log)
            try:
                output = io.StringIO()
                code = cli.claude_canary_actions_apply_cli(
                    ["-", "--config-dir", str(tmp_path), "--write"],
                    stdin=io.StringIO(json.dumps(action_bundle)),
                    stdout=output,
                )
            finally:
                if old_log is None:
                    os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
                else:
                    os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_log

            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["wrote_policy_files"])
            written = yaml.safe_load(routing_file.read_text(encoding="utf-8"))
            self.assertEqual(written["phase_canary"]["policy_id"], "test-claude-phase-canary")
            self.assertEqual(written["phase_canary"]["target_model"], "claude-haiku-4-5-20251001")
            self.assertEqual(written["phase_canary"]["canary_fraction"], 0.3)
            self.assertEqual(written["phase_canary"]["holdout_fraction"], 0.1)
            self.assertEqual(written["openai_canary"]["policy_id"], "openai-untouched")
            self.assertEqual(written["rules"][0]["action"]["reason"], "fixture")
            event = json.loads(event_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(event["action"], "claude-canary-actions-apply")
            self.assertTrue(event["ok"])
            self.assertFalse(event["details"]["dry_run"])
            _assert_privacy_clean(self, result)

    def test_routing_canary_promote_apply_writes_permanent_rule_and_bypasses_holdout(self) -> None:
        impact = self._impact([
            self._candidate(
                verdict="promote",
                candidate_id="candidate-promote",
                canary_fraction=0.9,
                holdout_fraction=0.1,
                reason_codes=["target-savings-met", "canary-full-coverage"],
            )
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            event_log = tmp_path / "policy_events.jsonl"
            routing_file = tmp_path / "routing_rules.yaml"
            routing_file.write_text(
                yaml.safe_dump({
                    "phase_canary": {
                        "enabled": True,
                        "policy_id": "test-claude-phase-canary",
                        "target_candidate_id": "candidate-promote",
                        "canary_fraction": 0.9,
                        "holdout_fraction": 0.1,
                    },
                    "rules": [],
                }, sort_keys=False),
                encoding="utf-8",
            )
            plan = build_routing_canary_promotion_plan(impact, config_dir=tmp_path)
            self.assertTrue(plan["ok"])
            self.assertEqual(plan["summary"]["promotion_action_count"], 1)
            self.assertEqual(plan["actions"][0]["permanent_rule"]["metadata"]["target_candidate_id"], "candidate-promote")

            old_log = os.environ.get("AGENTFLOW_POLICY_EVENTS_LOG")
            os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = str(event_log)
            try:
                output = io.StringIO()
                code = cli.routing_canary_promote_cli(
                    ["-", "--config-dir", str(tmp_path), "--apply"],
                    stdin=io.StringIO(json.dumps(impact)),
                    stdout=output,
                )
            finally:
                if old_log is None:
                    os.environ.pop("AGENTFLOW_POLICY_EVENTS_LOG", None)
                else:
                    os.environ["AGENTFLOW_POLICY_EVENTS_LOG"] = old_log

            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["wrote_policy_files"])
            written = yaml.safe_load(routing_file.read_text(encoding="utf-8"))
            self.assertFalse(written["phase_canary"]["enabled"])
            self.assertEqual(written["phase_canary"]["canary_fraction"], 0.0)
            self.assertEqual(written["rules"][0]["metadata"]["source"], "claude-canary-promote")
            self.assertTrue(written["rules"][0]["metadata"]["promoted_from_canary"])
            self.assertNotIn("canary", written["rules"][0])

            old_rules = os.environ.get("AGENTFLOW_ROUTING_RULES")
            os.environ["AGENTFLOW_ROUTING_RULES"] = str(routing_file)
            try:
                import agentflow_proxy.router as router_module

                manual_router = importlib.reload(router_module)
                body = {
                    "model": manual_router.SONNET_DEFAULT,
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                        }
                    ],
                }
                routed, meta = manual_router.route_model(body)
            finally:
                if old_rules is None:
                    os.environ.pop("AGENTFLOW_ROUTING_RULES", None)
                else:
                    os.environ["AGENTFLOW_ROUTING_RULES"] = old_rules
                import agentflow_proxy.router as router_module

                importlib.reload(router_module)

            self.assertEqual(routed, "claude-haiku-4-5-20251001")
            self.assertNotIn("phase_canary", meta)
            self.assertIsNone(meta["canary_cohort"])
            self.assertTrue(meta["routing_rule"]["promoted_from_canary"])
            event = json.loads(event_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(event["action"], "routing-canary-promote")
            self.assertTrue(event["ok"])
            self.assertFalse(event["details"]["dry_run"])
            _assert_privacy_clean(self, result)

    def test_claude_canary_actions_reject_raw_payloads_and_do_not_leak_from_impact(self) -> None:
        candidate = self._candidate(verdict="widen")
        candidate.update({
            "prompt": "raw claude prompt secret",
            "raw_response": "raw claude response secret",
            "provider_body": "raw fixture provider body",
            "session_id": "claude-session-id-secret",
            "file_path": "/tmp/fixture-secret.py",
            "api_key": "sk-ant-secret",
        })
        actions = build_claude_canary_actions(self._impact([candidate]))
        _assert_privacy_clean(self, actions)
        rendered = json.dumps(actions, sort_keys=True)
        self.assertNotIn("/tmp/fixture-secret.py", rendered)
        self.assertNotIn("raw fixture provider body", rendered)

        unsafe = json.loads(json.dumps(actions))
        unsafe["actions"][0]["local_policy_update"]["raw_prompt"] = "raw claude prompt secret"
        with tempfile.TemporaryDirectory() as tmp:
            result = apply_claude_canary_actions(unsafe, config_dir=tmp, dry_run=True)
        self.assertFalse(result["ok"])
        self.assertFalse(result["wrote_policy_files"])
        self.assertEqual(result["errors"][0]["path"], "$.actions[0].local_policy_update.raw_prompt")
        _assert_privacy_clean(self, result)


if __name__ == "__main__":
    unittest.main()
