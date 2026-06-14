from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from agentflow_proxy import stats as stats_views
from agentflow_proxy.store import SQLiteStore, stable_json, utc_now


FORBIDDEN_VALUES = (
    "raw-dashboard-prompt-secret",
    "raw-dashboard-response-secret",
    "raw-dashboard-session-secret",
    "req-dashboard-secret",
    "cache-key-dashboard-secret",
    "/home/lutz/private/dashboard_secret.py",
)


class OptimizationCoordinatorDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "agentflow.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def _report(self) -> dict[str, object]:
        return asyncio.run(stats_views.stats_optimization_coordinator_dashboard(self.store, limit=50))

    def _assert_private(self, payload: object) -> None:
        rendered = json.dumps(payload, sort_keys=True)
        for value in FORBIDDEN_VALUES:
            self.assertNotIn(value, rendered)
        for key in (
            '"cache_key"',
            '"content"',
            '"file_path"',
            '"messages"',
            '"prompt"',
            '"request_id"',
            '"response_json"',
            '"session_id"',
        ):
            self.assertNotIn(key, rendered)

    def _log_call(
        self,
        *,
        routing_meta: dict[str, object] | None = None,
        crunch_meta: dict[str, object] | None = None,
        cache_meta: dict[str, object] | None = None,
        status_code: int = 200,
        retry_count: int = 0,
    ) -> None:
        self.store.log_call(
            id=str(uuid.uuid4()),
            created_at=utc_now(),
            path="/v1/responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4-mini",
            stream=0,
            cache_hit=0,
            status_code=status_code,
            latency_ms=125,
            input_tokens_est=3200,
            output_tokens_est=120,
            actual_input_tokens=3000,
            actual_output_tokens=100,
            cost_est_usd=0.01,
            cost_baseline_usd=0.05,
            crunch_json=stable_json(crunch_meta or {"changed": False}),
            routing_json=stable_json(
                routing_meta
                or {
                    "provider": "openai",
                    "source_surface": "openai_responses",
                    "endpoint": "responses",
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "text_chars": 12000,
                    "enabled": True,
                    "requested_model": "gpt-5.4",
                    "routed_model": "gpt-5.4-mini",
                    "reason": "selected-canary",
                }
            ),
            cache_json=stable_json(cache_meta or {"status": "miss", "reason": "exact-miss"}),
            error="raw-dashboard-response-secret" if status_code >= 400 else None,
            request_json=stable_json(
                {
                    "request_id": "req-dashboard-secret",
                    "messages": [{"content": "raw-dashboard-prompt-secret"}],
                    "cache_key": "cache-key-dashboard-secret",
                    "file_path": "/home/lutz/private/dashboard_secret.py",
                }
            ),
            response_json=stable_json({"content": "raw-dashboard-response-secret"}),
            session_id="raw-dashboard-session-secret",
            category="tool-result",
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider="openai",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5-mini",
        )

    def _coordinator_meta(self, *, selected: str = "routing", suppressed: list[dict[str, object]] | None = None, reason_codes: list[str] | None = None) -> dict[str, object]:
        decision = {
            "schema": "agentflow.optimization_coordinator.v1",
            "selected_family": selected,
            "selected_action_family": selected,
            "suppressed_families": suppressed or [],
            "candidate_count": 1,
            "entry_count": 1,
            "reason_codes": reason_codes or [],
            "source_surface": "openai_responses",
            "provider_family": "openai",
            "endpoint": "responses",
            "category": "tool-result",
            "phase": "tool-execution",
            "canary": {"cohort": "coordinator_canary", "selected": True, "holdout": False},
            "privacy": {
                "metadata_only": True,
                "raw_prompts_included": False,
                "raw_responses_included": False,
                "provider_bodies_included": False,
                "file_paths_included": False,
                "cache_keys_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
            },
        }
        return {
            "optimization_coordinator": decision,
            "optimization_coordinator_enforcement": {
                "schema": "agentflow.optimization_coordinator_enforcement.v1",
                "enabled": True,
                "status": "applied",
                "selected_family": selected,
                "suppressed_families": [item["family"] for item in suppressed or []],
                "metadata_only": True,
                "provider_body_included": False,
            },
        }

    def test_disabled_state_without_coordinator_metadata(self) -> None:
        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "0"}):
            report = self._report()

        self.assertEqual(report["schema"], "agentflow.optimization_coordinator_dashboard.v1")
        self.assertEqual(report["state"], "disabled")
        self.assertFalse(report["capabilities"]["enforcement_enabled"])
        self.assertTrue(report["read_only"])
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self._assert_private(report)

    def test_dry_run_only_state_when_action_ledger_exists_without_runtime_metadata(self) -> None:
        self._log_call()

        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "0"}):
            report = self._report()

        self.assertEqual(report["state"], "dry-run-only")
        self.assertEqual(report["summary"]["runtime_decision_count"], 0)
        self.assertGreaterEqual(report["summary"]["rows_with_ledger_entries"], 1)
        self.assertTrue(report["capabilities"]["dry_run_only"])
        self._assert_private(report)

    def test_dry_run_summary_exposes_suppression_opportunity_buckets(self) -> None:
        self._log_call(
            crunch_meta={
                "terminal_output_compaction": {
                    "status": "eligible",
                    "candidate_id": "terminal-dashboard-candidate",
                    "projected_saved_usd": 0.02,
                },
            }
        )

        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "0"}):
            report = self._report()

        dry_run = report["dry_run_summary"]
        bucket = dry_run["suppression_opportunity_buckets"][0]
        self.assertEqual(bucket["selected_family"], "routing")
        self.assertEqual(bucket["suppressed_family"], "terminal_output_compaction")
        self.assertEqual(bucket["projected_savings_lost_usd"], 0.02)
        self.assertEqual(dry_run["top_suppression_next_action"], "run-suppressed-crunch-eval")
        self._assert_private(report)

    def test_active_selection_state_counts_runtime_selection(self) -> None:
        self._log_call(routing_meta=self._coordinator_meta(selected="routing"))

        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "1"}):
            report = self._report()

        self.assertEqual(report["state"], "active-selection")
        self.assertEqual(report["summary"]["selected_count"], 1)
        self.assertEqual(report["summary"]["observed_savings_usd_est"], 0.04)
        selected = {row["value"]: row["count"] for row in report["selected_family_counts"]}
        self.assertEqual(selected["routing"], 1)
        self._assert_private(report)

    def test_conflict_observed_state_counts_suppressed_families_and_dimensions(self) -> None:
        self._log_call(
            routing_meta=self._coordinator_meta(
                selected="routing",
                suppressed=[{"family": "cache_replay", "reason_codes": ["conflicts-with-selected-family"]}],
            ),
            status_code=429,
            retry_count=3,
        )

        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "1"}):
            report = self._report()

        self.assertEqual(report["state"], "conflict-observed")
        self.assertEqual(report["summary"]["conflict_count"], 1)
        self.assertEqual(report["summary"]["rows_with_errors"], 1)
        reasons = {(row["family"], row["reason"]): row["count"] for row in report["top_suppression_reason_codes"]}
        self.assertEqual(reasons[("cache_replay", "conflicts-with-selected-family")], 1)
        self.assertEqual(report["dimension_breakdown"][0]["provider"], "openai")
        self.assertEqual(report["dimension_breakdown"][0]["public_session_bucket"], "unknown")
        self._assert_private(report)

    def test_safety_stop_state_counts_safety_reasons(self) -> None:
        self._log_call(routing_meta=self._coordinator_meta(selected="routing", reason_codes=["safety-stop-priority"]))

        with patch.dict(os.environ, {"AGENTFLOW_OPTIMIZATION_COORDINATOR_ENFORCEMENT": "1"}):
            report = self._report()

        self.assertEqual(report["state"], "safety-stop")
        self.assertEqual(report["summary"]["safety_stop_count"], 1)
        self.assertFalse(report["privacy"]["policy_file_contents_included"])
        self._assert_private(report)


if __name__ == "__main__":
    unittest.main()
