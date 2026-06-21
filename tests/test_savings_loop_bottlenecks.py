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
    def _log_call(self, store: Store, *, call_id: str, created_at: str) -> None:
        store.log_call(
            id=call_id,
            created_at=created_at,
            path="/v1/messages",
            requested_model="claude-sonnet-4-5",
            routed_model="claude-sonnet-4-5",
            stream=0,
            cache_hit=0,
            status_code=200,
            latency_ms=1,
            input_tokens_est=10,
            output_tokens_est=2,
            cost_est_usd=0.001,
            cost_baseline_usd=0.001,
            crunch_json=stable_json({"changed": False}),
            routing_json=stable_json({"reason": "test"}),
            cache_json=stable_json({"status": "miss"}),
            category="chat",
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
                self.assertIn("savings-loop-bottlenecks-tbody", dashboard.text)
                self.assertIn("c-savings-loop", dashboard.text)
                rendered = json.dumps(payload, sort_keys=True) + dashboard.text
                self.assertNotIn(str(db), rendered)
                self.assertNotIn(str(config_dir), rendered)
                self.assertNotIn("raw-stale-cache-rule-secret", rendered)
            finally:
                store.conn.close()


if __name__ == "__main__":
    unittest.main()
