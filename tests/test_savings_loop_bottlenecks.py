import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from tokenclaw import cli
from tokenclaw.cli_commands.onboarding import tokenclaw_cli
from tokenclaw.dashboard_app import create_dashboard_app
from tokenclaw.savings_loop_bottlenecks import build_savings_loop_bottlenecks_report
from tokenclaw.store import Store, stable_json


NOW = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


class SavingsLoopBottlenecksTest(unittest.TestCase):
    def _log_call(
        self,
        store: Store,
        *,
        call_id: str,
        created_at: str,
        category: str = "chat",
        workflow_phase: str = "chat",
        stream: int = 0,
        has_tools: bool = False,
        text_chars: int = 40,
        input_tokens: int = 10,
        cost_est_usd: float = 0.001,
        cost_baseline_usd: float = 0.001,
    ) -> None:
        store.log_call(
            id=call_id,
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-5",
            routed_model="claude-sonnet-4-5",
            stream=stream,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=input_tokens,
            output_tokens_est=2,
            actual_input_tokens=input_tokens,
            actual_output_tokens=2,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "reason": "test",
                "category": category,
                "workflow_phase": workflow_phase,
                "has_tools": has_tools,
                "text_chars": text_chars,
            }),
            cache_json=stable_json({"status": "skipped" if stream else "miss", "reason": "streaming" if stream else "exact-miss"}),
            category=category,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model_family="claude-sonnet",
            routed_model_family="claude-sonnet",
        )

    def _write_stale_cache_canary(self, config_dir: Path) -> None:
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "cache_canary_policy.yaml").write_text(
            """
pattern_rules:
  - id: raw-stale-cache-rule-secret
    enabled: true
    candidate_id: raw-stale-cache-candidate-secret
    policy_source: local-manual
    conditions:
      provider_family: openai
      source_surface: openai_responses
      endpoint: responses
      category: chat
      stream: false
      has_tools: false
    rollout:
      canary_enabled: true
      canary_fraction: 0.1
      holdout_fraction: 0.1
    graduation:
      source_schema: tokenclaw.request_shape_cache_replayability_dry_run.v1
      staged_at: "2026-06-01T00:00:00+00:00"
      projected_hits: 10
      projected_savings_usd: 0.25
""",
            encoding="utf-8",
        )

    def _log_phase_canary_call(
        self,
        store: Store,
        *,
        call_id: str,
        cohort: str,
        status: str,
        cost_est_usd: float,
        cost_baseline_usd: float,
        status_code: int = 200,
    ) -> None:
        routed_model = "claude-haiku-4-5-20251001" if cohort == "canary_applied" else "claude-sonnet-4-5"
        store.log_call(
            id=call_id,
            created_at="2026-06-21T11:45:00+00:00",
            path="/v1/messages",
            requested_model="claude-sonnet-4-5",
            routed_model=routed_model,
            stream=1,
            cache_hit=0,
            status_code=status_code,
            latency_ms=100,
            input_tokens_est=1000,
            output_tokens_est=100,
            actual_input_tokens=1000,
            actual_output_tokens=100,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "reason": "phase canary selected live route" if cohort == "canary_applied" else "phase canary holdout",
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "has_tools": True,
                "text_chars": 12000,
                "phase_canary": {
                    "enabled": True,
                    "policy_id": "routing-canary-policy-secret",
                    "rule_id": "routing-canary-rule-secret",
                    "candidate_id": "routing-canary-candidate-secret",
                    "target_candidate_id": "routing-canary-candidate-secret",
                    "promotion_action_id": "routing-canary-action-secret",
                    "status": status,
                    "cohort": cohort,
                    "reason": "selected-canary" if cohort == "canary_applied" else "selected-holdout",
                    "source_surface": "anthropic_messages",
                    "policy_source": "local-manual",
                    "target_model": "claude-haiku-4-5-20251001",
                    "promotion": {
                        "source_report_schema": "tokenclaw.anthropic_routing_canary_stage.v1",
                        "projected_savings_usd": 0.02,
                    },
                },
            }),
            cache_json=stable_json({"status": "skipped", "reason": "streaming"}),
            category="tool-result",
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="messages",
            requested_model_family="claude-sonnet",
            routed_model_family="claude-haiku" if cohort == "canary_applied" else "claude-sonnet",
        )

    def _log_cache_canary_call(
        self,
        store: Store,
        *,
        call_id: str,
        cohort: str,
        status: str,
        cache_status: str,
        cost_est_usd: float,
        cost_baseline_usd: float,
    ) -> None:
        store.log_call(
            id=call_id,
            created_at="2026-06-21T11:46:00+00:00",
            path="/v1/responses",
            requested_model="gpt-5-mini",
            routed_model="gpt-5-mini",
            stream=0,
            cache_hit=1 if cache_status == "hit" else 0,
            status_code=200,
            latency_ms=40,
            input_tokens_est=800,
            output_tokens_est=80,
            actual_input_tokens=800,
            actual_output_tokens=80,
            cost_est_usd=cost_est_usd,
            cost_baseline_usd=cost_baseline_usd,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({
                "enabled": False,
                "provider": "openai",
                "category": "chat",
                "workflow_phase": "chat",
                "has_tools": False,
            }),
            cache_json=stable_json({
                "status": cache_status,
                "reason": "canary_holdout" if cohort == "canary_holdout" else "exact-hit",
                "pattern_rule": {
                    "rule_id": "cache-canary-rule-secret",
                    "candidate_id": "cache-canary-candidate-secret",
                    "policy_source": "local-manual",
                    "graduation": {
                        "source_schema": "tokenclaw.request_shape_cache_replayability_dry_run.v1",
                        "projected_savings_usd": 0.02,
                    },
                },
                "cache_replay_canary": {
                    "schema": "tokenclaw.cache_replay_canary_decision.v1",
                    "rule_id": "cache-canary-rule-secret",
                    "candidate_id": "cache-canary-candidate-secret",
                    "policy_source": "local-manual",
                    "status": status,
                    "reason": "selected-canary" if cohort == "canary_applied" else "canary_holdout",
                    "canary_cohort": cohort,
                    "projected_input_savings_usd": 0.02,
                    "canary": {
                        "enabled": True,
                        "selected": cohort == "canary_applied",
                        "cohort": cohort,
                        "status": status,
                    },
                },
                "estimated_saved_cost_usd": 0.02,
            }),
            category="chat",
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
        )

    def test_reconciles_applied_canary_outcomes_and_captured_available_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tokenclaw.sqlite3"
            store = Store(str(db))
            try:
                self._log_phase_canary_call(
                    store,
                    call_id="phase-canary-applied",
                    cohort="canary_applied",
                    status="applied",
                    cost_est_usd=0.004,
                    cost_baseline_usd=0.010,
                )
                self._log_phase_canary_call(
                    store,
                    call_id="phase-canary-holdout",
                    cohort="canary_holdout",
                    status="holdout",
                    cost_est_usd=0.010,
                    cost_baseline_usd=0.010,
                )
                self._log_cache_canary_call(
                    store,
                    call_id="cache-canary-applied",
                    cohort="canary_applied",
                    status="applied",
                    cache_status="hit",
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.03,
                )
                self._log_cache_canary_call(
                    store,
                    call_id="cache-canary-holdout",
                    cohort="canary_holdout",
                    status="holdout",
                    cache_status="bypassed",
                    cost_est_usd=0.03,
                    cost_baseline_usd=0.03,
                )
                store.persist_request_shape_rollups(
                    run_id="blocked-rollups",
                    generated_at="2026-06-21T11:50:00+00:00",
                    rows=[
                        {
                            "id": "blocked-rollup",
                            "run_id": "blocked-rollups",
                            "generated_at": "2026-06-21T11:50:00+00:00",
                            "window_start": "2026-06-21T10:00:00+00:00",
                            "window_end": "2026-06-21T11:00:00+00:00",
                            "rollup_key": "surface:endpoint:tool-result",
                            "candidate_id": "blocked-candidate-secret",
                            "source_surface": "anthropic_messages",
                            "endpoint": "messages",
                            "provider_family": "anthropic",
                            "requested_model_family": "sonnet",
                            "routed_model_family": "sonnet",
                            "category": "tool-result",
                            "workflow_phase": "tool-execution",
                            "stream": 1,
                            "has_tools": 1,
                            "text_bucket": "8k_32k_chars",
                            "token_bucket": "2k_8k_tokens",
                            "cache_status": "skipped",
                            "routing_status": "kept",
                            "candidate_families_json": stable_json(["routing"]),
                            "blocker_codes_json": stable_json(["routing-rule-required"]),
                            "row_count": 40,
                            "error_count": 0,
                            "retry_count": 0,
                            "cache_hit_count": 0,
                            "cost_est_usd": 0.7,
                            "baseline_cost_usd": 1.2,
                            "observed_savings_usd": 0.5,
                            "input_tokens": 10000,
                            "output_tokens": 1000,
                            "metadata_json": stable_json({"metadata_only": True}),
                        }
                    ],
                )

                report = build_savings_loop_bottlenecks_report(
                    store,
                    db_path=db,
                    config_dir=root / "config",
                    activation_min_source_rows=1,
                    policy_scan_limit=50,
                    now=NOW,
                )
                again = build_savings_loop_bottlenecks_report(
                    store,
                    db_path=db,
                    config_dir=root / "config",
                    activation_min_source_rows=1,
                    policy_scan_limit=50,
                    now=NOW,
                )
                feedback_count = store.conn.execute("select count(*) from promotion_outcome_feedback").fetchone()[0]
                eval_count = store.conn.execute("select count(*) from optimization_eval_results").fetchone()[0]
            finally:
                store.conn.close()

        self.assertEqual(report["outcome_feedback_reconciliation"]["status"], "recorded")
        self.assertEqual(report["summary"]["promotion_outcome_rows_written"], 2)
        self.assertEqual(report["summary"]["optimization_eval_rows_written"], 2)
        self.assertEqual(again["outcome_feedback_reconciliation"]["status"], "up-to-date")
        self.assertEqual(feedback_count, 2)
        self.assertEqual(eval_count, 2)
        captured = report["captured_vs_available"]
        self.assertEqual(captured["schema"], "tokenclaw.savings_loop_captured_vs_available.v1")
        self.assertAlmostEqual(captured["captured_savings_usd"], 0.036, places=6)
        self.assertAlmostEqual(captured["available_blocked_savings_usd"], 0.5, places=6)
        self.assertAlmostEqual(captured["blocked_baseline_usd"], 1.2, places=6)
        self.assertEqual(captured["top_available_blocker"]["blocker_code"], "routing-rule-required")
        self.assertFalse(captured["privacy"]["raw_prompts_included"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("routing-canary-policy-secret", rendered)
        self.assertNotIn("cache-canary-rule-secret", rendered)
        self.assertNotIn("blocked-candidate-secret", rendered)

    def test_report_consolidates_source_legacy_rollup_and_stale_policy_stalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_db = root / "tokenclaw.sqlite3"
            legacy_db = root / "agentflow.sqlite3"
            config_dir = root / "config"
            canonical = Store(str(canonical_db))
            legacy = Store(str(legacy_db))
            try:
                self._log_call(canonical, call_id="canonical-call", created_at="2026-06-21T11:30:00+00:00")
                self._log_call(legacy, call_id="legacy-call-1", created_at="2026-06-21T11:31:00+00:00")
                self._log_call(legacy, call_id="legacy-call-2", created_at="2026-06-21T11:32:00+00:00")
                self._write_stale_cache_canary(config_dir)

                report = build_savings_loop_bottlenecks_report(
                    canonical,
                    db_path=canonical_db,
                    legacy_db=legacy_db,
                    config_dir=config_dir,
                    activation_min_source_rows=10,
                    policy_scan_limit=50,
                    now=NOW,
                )

                self.assertEqual(report["schema"], "tokenclaw.savings_loop_bottlenecks.v1")
                self.assertEqual(report["status"], "stalled")
                self.assertEqual(report["summary"]["source_traffic_rows"], 1)
                self.assertTrue(report["summary"]["below_activation_threshold"])
                self.assertEqual(report["summary"]["stranded_legacy_rows"], 1)
                self.assertEqual(report["summary"]["request_shape_rollup_count"], 0)
                self.assertTrue(report["summary"]["zero_row_crunch_dry_run"])
                self.assertEqual(report["summary"]["crunch_dry_run_rows_considered"], 0)
                self.assertGreaterEqual(report["summary"]["stale_policy_rule_count"], 1)
                blockers = {row["blocker_code"] for row in report["rows"] if row.get("blocker_code")}
                self.assertIn("source-traffic-below-activation-threshold", blockers)
                self.assertIn("stranded-legacy-agentflow-sqlite-evidence", blockers)
                self.assertIn("no-request-shape-rollups", blockers)
                self.assertIn("zero-row-crunch-dry-run", blockers)
                self.assertIn("stale-no-canary-traffic", blockers)
                commands = {row["command"] for row in report["rows"]}
                self.assertIn("tokenclaw db adopt-legacy", commands)
                self.assertIn("tokenclaw request-shape-rollups --dry-run", commands)
                self.assertIn("tokenclaw request-shape-cache-replay-policy-decision --apply", commands)
                rendered = json.dumps(report, sort_keys=True)
                self.assertNotIn(str(canonical_db), rendered)
                self.assertNotIn(str(legacy_db), rendered)
                self.assertNotIn("raw-stale-cache-rule-secret", rendered)
                self.assertNotIn("raw-stale-cache-candidate-secret", rendered)
                self.assertFalse(report["privacy"]["raw_prompts_included"])
                self.assertFalse(report["privacy"]["provider_bodies_included"])
                self.assertFalse(report["privacy"]["request_ids_included"])
                self.assertFalse(report["privacy"]["session_ids_included"])
                self.assertFalse(report["privacy"]["cache_keys_included"])
            finally:
                canonical.conn.close()
                legacy.conn.close()

    def test_legacy_adoption_preflight_clears_stranded_storage_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_db = root / "tokenclaw.sqlite3"
            legacy_db = root / "agentflow.sqlite3"
            canonical = Store(str(canonical_db))
            legacy = Store(str(legacy_db))
            try:
                self._log_call(canonical, call_id="canonical-call", created_at="2026-06-21T11:00:00+00:00")
                self._log_call(legacy, call_id="legacy-call-1", created_at="2026-06-21T11:10:00+00:00")
                self._log_call(legacy, call_id="legacy-call-2", created_at="2026-06-21T11:20:00+00:00")
                legacy.persist_request_shape_rollups(
                    run_id="legacy-rollups",
                    generated_at="2026-06-21T11:25:00+00:00",
                    rows=[
                        {
                            "id": "legacy-rollup-1",
                            "run_id": "legacy-rollups",
                            "generated_at": "2026-06-21T11:25:00+00:00",
                            "window_start": "2026-06-21T10:00:00+00:00",
                            "window_end": "2026-06-21T11:00:00+00:00",
                            "rollup_key": "surface:endpoint:chat",
                            "candidate_id": "legacy-candidate-redacted",
                            "source_surface": "anthropic_messages",
                            "endpoint": "messages",
                            "provider_family": "anthropic",
                            "requested_model_family": "sonnet",
                            "routed_model_family": "sonnet",
                            "category": "chat",
                            "workflow_phase": "chat",
                            "stream": 0,
                            "has_tools": 0,
                            "text_bucket": "2k_8k_chars",
                            "token_bucket": "500_2k_tokens",
                            "cache_status": "miss",
                            "routing_status": "kept",
                            "candidate_families_json": stable_json(["crunch"]),
                            "blocker_codes_json": stable_json([]),
                            "row_count": 2,
                            "error_count": 0,
                            "retry_count": 0,
                            "cache_hit_count": 0,
                            "cost_est_usd": 0.1,
                            "baseline_cost_usd": 0.2,
                            "observed_savings_usd": 0.1,
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "metadata_json": stable_json({"metadata_only": True}),
                        }
                    ],
                )
                legacy.persist_request_shape_rollup_snapshot(
                    {
                        "snapshot_id": "legacy-snapshot-1",
                        "run_id": "legacy-rollups",
                        "generated_at": "2026-06-21T11:25:00+00:00",
                        "window": {
                            "source": "fixture",
                            "start": "2026-06-21T10:00:00+00:00",
                            "end": "2026-06-21T11:00:00+00:00",
                        },
                        "summary": {
                            "rows_considered": 2,
                            "rollup_count": 1,
                            "ranked_candidate_count": 1,
                            "top_next_action": "rank-repeated-context-crunch-dry-run",
                            "top_local_action_family": "crunch",
                            "top_readiness_state": "ready",
                            "total_projected_savings_usd": 0.1,
                        },
                    }
                )

                report = build_savings_loop_bottlenecks_report(
                    canonical,
                    db_path=canonical_db,
                    legacy_db=legacy_db,
                    config_dir=root / "config",
                    activation_min_source_rows=1,
                    rollup_max_age_hours=72,
                    policy_scan_limit=50,
                    adopt_legacy_preflight=True,
                    now=NOW,
                )

                blockers = {row["blocker_code"] for row in report["rows"] if row.get("blocker_code")}
                self.assertNotIn("stranded-legacy-agentflow-sqlite-evidence", blockers)
                self.assertEqual(report["summary"]["stranded_legacy_rows"], 0)
                self.assertEqual(report["summary"]["source_traffic_rows"], 3)
                self.assertEqual(report["summary"]["request_shape_rollup_count"], 1)
                self.assertEqual(report["summary"]["request_shape_rollup_snapshot_count"], 1)
                self.assertEqual(report["summary"]["crunch_dry_run_rows_considered"], 1)
                self.assertFalse(report["summary"]["zero_row_crunch_dry_run"])
                self.assertEqual(report["legacy_adoption_preflight"]["status"], "adopted-gap-cleared")
                self.assertEqual(report["legacy_adoption_preflight"]["rows_inserted"], 4)
                self.assertTrue(report["legacy_adoption_preflight"]["gap_cleared"])
                self.assertTrue(report["legacy_adoption_preflight"]["privacy"]["metadata_only"])
                self.assertTrue(report["legacy_adoption_preflight"]["privacy"]["aggregate_only"])

                rendered = json.dumps(report, sort_keys=True)
                self.assertNotIn(str(canonical_db), rendered)
                self.assertNotIn(str(legacy_db), rendered)
                self.assertNotIn("legacy-candidate-redacted", rendered)
                self.assertFalse(report["privacy"]["raw_prompts_included"])
                self.assertFalse(report["privacy"]["provider_bodies_included"])
                self.assertFalse(report["privacy"]["request_ids_included"])
                self.assertFalse(report["privacy"]["session_ids_included"])
                self.assertFalse(report["privacy"]["cache_keys_included"])
            finally:
                canonical.conn.close()
                legacy.conn.close()

    def test_canonical_rollup_refresh_clears_missing_rollup_and_zero_crunch_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tokenclaw.sqlite3"
            store = Store(str(db))
            try:
                for index, cost in enumerate((0.02, 0.03, 0.04), start=1):
                    self._log_call(
                        store,
                        call_id=f"canonical-rollup-source-{index}",
                        created_at=f"2026-06-21T11:0{index}:00+00:00",
                        category="tool-result",
                        workflow_phase="tool-execution",
                        stream=1,
                        has_tools=True,
                        text_chars=24_000,
                        input_tokens=6_000,
                        cost_est_usd=cost,
                        cost_baseline_usd=cost + 0.01,
                    )

                dry_stdout = io.StringIO()
                dry_code = cli.request_shape_rollups_cli(
                    [
                        "--db",
                        str(db),
                        "--limit",
                        "20",
                        "--run-id",
                        "canonical-refresh-dry-run",
                        "--dry-run",
                    ],
                    stdout=dry_stdout,
                )
                self.assertEqual(dry_code, 0)
                dry_report = json.loads(dry_stdout.getvalue())
                self.assertGreaterEqual(dry_report["summary"]["rollup_count"], 1)
                self.assertGreaterEqual(dry_report["follow_up_candidates"]["summary"]["ranked_candidate_count"], 1)
                self.assertGreater(dry_report["crunch_opportunity_dry_run"]["summary"]["rows_considered"], 0)
                self.assertEqual(store.conn.execute("select count(*) from request_shape_rollups").fetchone()[0], 0)
                self.assertEqual(store.conn.execute("select count(*) from request_shape_rollup_snapshots").fetchone()[0], 0)

                report = build_savings_loop_bottlenecks_report(
                    store,
                    db_path=db,
                    config_dir=root / "config",
                    activation_min_source_rows=3,
                    policy_scan_limit=20,
                    now=NOW,
                )

                blockers = {row["blocker_code"] for row in report["rows"] if row.get("blocker_code")}
                self.assertNotIn("no-request-shape-rollups", blockers)
                self.assertNotIn("zero-row-crunch-dry-run", blockers)
                self.assertEqual(report["summary"]["request_shape_rollup_count"], 0)
                self.assertEqual(report["summary"]["request_shape_rollup_snapshot_count"], 0)
                self.assertEqual(report["summary"]["request_shape_rollup_refresh_status"], "refreshed")
                self.assertGreaterEqual(report["summary"]["request_shape_rollup_refresh_rollup_count"], 1)
                self.assertGreater(report["summary"]["request_shape_rollup_refresh_crunch_dry_run_rows_considered"], 0)
                self.assertFalse(report["summary"]["zero_row_crunch_dry_run"])
                self.assertEqual(report["request_shape_rollup_refresh"]["schema"], "tokenclaw.request_shape_rollup_refresh_preflight.v1")
                self.assertFalse(report["request_shape_rollup_refresh"]["persisted"])
                self.assertTrue(report["request_shape_rollup_refresh"]["privacy"]["metadata_only"])
                self.assertTrue(report["request_shape_rollup_refresh"]["privacy"]["aggregate_only"])
                rendered = json.dumps(report, sort_keys=True)
                self.assertNotIn(str(db), rendered)
                self.assertFalse(report["privacy"]["raw_prompts_included"])
                self.assertFalse(report["privacy"]["provider_bodies_included"])
                self.assertFalse(report["privacy"]["request_ids_included"])
                self.assertFalse(report["privacy"]["session_ids_included"])
                self.assertFalse(report["privacy"]["cache_keys_included"])
            finally:
                store.conn.close()

    def test_cli_and_dashboard_expose_same_metadata_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tokenclaw.sqlite3"
            config_dir = root / "config"
            store = Store(str(db))
            self._write_stale_cache_canary(config_dir)
            try:
                stdout = io.StringIO()
                code = cli.savings_loop_bottlenecks_cli(
                    [
                        "--db",
                        str(db),
                        "--config-dir",
                        str(config_dir),
                        "--activation-min-source-rows",
                        "1",
                    ],
                    stdout=stdout,
                )
                self.assertEqual(code, 0)
                cli_report = json.loads(stdout.getvalue())
                self.assertEqual(cli_report["schema"], "tokenclaw.savings_loop_bottlenecks.v1")
                self.assertFalse(cli_report["privacy"]["raw_prompts_included"])

                public_stdout = io.StringIO()
                public_code = tokenclaw_cli(
                    [
                        "--config-dir",
                        str(config_dir),
                        "savings",
                        "loop-bottlenecks",
                        "--db",
                        str(db),
                        "--activation-min-source-rows",
                        "1",
                        "--json",
                    ],
                    stdout=public_stdout,
                )
                self.assertEqual(public_code, 0)
                public_report = json.loads(public_stdout.getvalue())
                self.assertEqual(public_report["schema"], "tokenclaw.savings_loop_bottlenecks.v1")

                with patch.dict(os.environ, {"TOKENCLAW_CONFIG_DIR": str(config_dir)}, clear=False):
                    app = create_dashboard_app(
                        store_obj=lambda: store,
                        default_db=str(db),
                        upstream="https://anthropic.test",
                        limiter_status=lambda: [],
                        limiter_config={},
                        full_stats_ttl_s=0,
                    )
                    client = TestClient(app)
                    response = client.get("/tokenclaw/stats/savings-loop-bottlenecks?limit=50")
                    dashboard = client.get("/tokenclaw/dashboard")

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["schema"], "tokenclaw.savings_loop_bottlenecks.v1")
                rendered = json.dumps(payload, sort_keys=True) + dashboard.text
                self.assertNotIn(str(db), rendered)
                self.assertNotIn(str(config_dir), rendered)
                self.assertNotIn("raw-stale-cache-rule-secret", rendered)
            finally:
                store.conn.close()


if __name__ == "__main__":
    unittest.main()
