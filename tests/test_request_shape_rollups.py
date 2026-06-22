from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml

from tokenclaw import cache as cache_module
from tokenclaw import cli
from tokenclaw import routing_experiments
from tokenclaw.request_shape_rollups import (
    apply_request_shape_cache_replay_canary_action,
    apply_request_shape_crunch_canary_action,
    apply_request_shape_crunch_canary_actions,
    apply_request_shape_crunch_policy_decision,
    build_request_shape_cache_replay_canary_stage_report,
    build_request_shape_cache_replay_evidence_report,
    build_request_shape_cache_replay_policy_decision_report,
    build_context_plateau_crunch_rollup_report,
    build_request_shape_crunch_canary_impact_report,
    build_request_shape_crunch_canary_impact_rows_report,
    build_request_shape_crunch_activation_evidence_report,
    build_request_shape_crunch_opportunity_dry_run,
    build_request_shape_crunch_remaining_measurement_report,
    build_request_shape_crunch_policy_decision_ledger,
    build_request_shape_crunch_policy_decision_report,
    build_request_shape_crunch_canary_stage_report,
    build_request_shape_follow_up_candidates,
    build_request_shape_routing_downgrade_drill_report,
    build_request_shape_rollups_report,
    build_request_shape_tool_cache_replay_evidence_report,
    latest_request_shape_rollup_snapshot_report,
    record_request_shape_crunch_policy_decision_ledger,
    request_shape_crunch_canary_lifecycle,
)
from tokenclaw.store import SQLiteStore, stable_json, utc_now


class RequestShapeRollupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "tokenclaw.sqlite3")
        self.store = SQLiteStore(self.db_path)

    def tearDown(self) -> None:
        self.store.conn.close()
        self.tmpdir.cleanup()

    def test_follow_up_candidates_reports_explicit_no_source_traffic_reason(self):
        report = build_request_shape_follow_up_candidates([], limit=10)

        self.assertEqual(report["schema"], "tokenclaw.request_shape_follow_up_candidates.v1")
        self.assertEqual(report["status"], "no-source-traffic")
        self.assertEqual(report["summary"]["rollup_count"], 0)
        self.assertEqual(report["summary"]["top_next_action"], "emit-request-shape-rollups")
        self.assertEqual(report["summary"]["top_local_action_family"], "cohort-ranking")
        self.assertEqual(report["summary"]["top_readiness_state"], "blocked")
        self.assertEqual(report["action_type"], "source-traffic-acquisition")
        self.assertEqual(report["source_traffic_acquisition_status"], "no-source-traffic")
        self.assertTrue(report["summary"]["source_traffic_acquisition_attempted"])
        acquisition = report["source_traffic_acquisition"]
        self.assertEqual(acquisition["schema"], "tokenclaw.source_traffic_acquisition_action.v1")
        self.assertEqual(acquisition["status"], "no-source-traffic")
        self.assertEqual(acquisition["action_type"], "source-traffic-acquisition")
        self.assertEqual(acquisition["recommended_command"], "tokenclaw-request-shape-rollups")
        self.assertEqual(acquisition["recommended_module"], "tokenclaw.request_shape_rollups")
        self.assertEqual(acquisition["blocker_code"], "no-source-traffic-for-request-shape-rollups")
        self.assertEqual(acquisition["rows_considered"], 0)
        self.assertEqual(acquisition["rollup_count"], 0)
        self.assertFalse(acquisition["provider_calls_made"])
        self.assertFalse(acquisition["managed_server_calls_made"])
        self.assertFalse(acquisition["policy_files_written"])
        self.assertTrue(acquisition["privacy"]["metadata_only"])
        self.assertTrue(acquisition["privacy"]["aggregate_only"])
        self.assertFalse(acquisition["privacy"]["raw_prompts_included"])
        self.assertFalse(acquisition["privacy"]["request_ids_included"])
        self.assertFalse(acquisition["privacy"]["session_ids_included"])
        self.assertFalse(acquisition["privacy"]["cache_keys_included"])
        self.assertFalse(acquisition["privacy"]["tool_payloads_included"])
        self.assertFalse(acquisition["privacy"]["file_paths_included"])
        self.assertEqual(
            report["summary"]["no_source_traffic_reason"],
            "no-source-traffic-for-request-shape-rollups",
        )
        self.assertEqual(report["missing_measurements"], ["no-source-traffic-for-request-shape-rollups"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])

    def test_follow_up_candidates_emit_ranked_local_activation_candidate_queue(self):
        rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "app_family": "generic_openai",
                "requested_model_family": "gpt-5",
                "routed_model_family": "gpt-5",
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "stream": True,
                "has_tools": True,
                "text_bucket": "32k_128k_chars",
                "token_bucket": "8k_32k_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "candidate_families": ["crunch_candidate"],
                "blocker_codes": [],
                "row_count": 42,
                "sample_count": 42,
                "successful_input_tokens": 420_000,
                "projected_crunch_tokens_saved": 21_000,
                "projected_crunch_savings_usd": 0.063,
                "request_id": "raw-request-id-must-not-leak",
                "session_id": "raw-session-id-must-not-leak",
                "file_path": "/tmp/private/rollup.py",
            },
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "app_family": "generic_openai",
                "requested_model_family": "gpt-5",
                "routed_model_family": "gpt-5",
                "category": "chat",
                "workflow_phase": "chat",
                "stream": False,
                "has_tools": False,
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "cache_status": "miss",
                "routing_status": "passthrough",
                "candidate_work_classes": ["replayability"],
                "candidate_families": ["cache_replay"],
                "blocker_codes": ["exact-cache-miss"],
                "row_count": 16,
                "sample_count": 16,
                "cost_est_usd": 0.032,
            },
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "provider_family": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "app_family": "claude_code",
                "requested_model_family": "claude-sonnet",
                "routed_model_family": "claude-sonnet",
                "category": "summary",
                "workflow_phase": "summary",
                "stream": True,
                "has_tools": False,
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "candidate_work_classes": ["routing"],
                "candidate_families": ["routing_candidate"],
                "blocker_codes": [],
                "row_count": 8,
                "sample_count": 8,
            },
        ]

        report = build_request_shape_follow_up_candidates(rollups, limit=10)
        repeat = build_request_shape_follow_up_candidates(rollups, limit=10)

        self.assertEqual(report["status"], "candidates-ranked")
        self.assertEqual(report["summary"]["rows_considered"], 66)
        self.assertEqual(report["summary"]["activation_candidate_count"], 3)
        queue = report["activation_candidate_queue"]
        self.assertEqual(queue["schema"], "tokenclaw.request_shape_local_activation_candidate_queue.v1")
        self.assertEqual(queue["status"], "ranked")
        self.assertEqual(queue["summary"]["queued_candidate_count"], 3)
        self.assertFalse(queue["summary"]["policy_files_written"])
        self.assertEqual(queue["summary"]["provider_calls_made"], 0)
        self.assertEqual(queue["summary"]["managed_server_calls_made"], 0)
        entries = queue["entries"]
        self.assertEqual([entry["rank"] for entry in entries], [1, 2, 3])
        self.assertEqual([entry["fingerprint"] for entry in entries], [entry["fingerprint"] for entry in repeat["activation_candidate_queue"]["entries"]])
        self.assertEqual(entries[0]["local_action_family"], "crunch")
        self.assertEqual(entries[0]["recommended_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(entries[0]["projected_saved_tokens"], 21_000)
        self.assertEqual(entries[0]["projected_savings_usd"], 0.063)
        self.assertEqual(entries[0]["freshness_state"], "fresh")
        self.assertEqual(entries[0]["preview_requirement"], "managed-preview-optional")
        self.assertFalse(entries[0]["managed_preview_required"])
        cache_entry = next(entry for entry in entries if entry["local_action_family"] == "cache")
        self.assertEqual(cache_entry["recommended_next_action"], "stage-cache-replay-canary")
        self.assertEqual(cache_entry["preview_requirement"], "managed-preview-required-before-policy-write")
        self.assertTrue(cache_entry["managed_preview_required"])
        for entry in entries:
            self.assertTrue(entry["fingerprint"].startswith("activation:"))
            self.assertTrue(entry["source_fingerprint"].startswith("request-shape-follow-up:"))
            self.assertIn("expected_savings_path", entry)
            self.assertFalse(entry["policy_files_written"])
            self.assertFalse(entry["provider_calls_made"])
            self.assertFalse(entry["managed_server_calls_made"])
            self.assertTrue(entry["privacy"]["metadata_only"])
            self.assertTrue(entry["privacy"]["aggregate_only"])
            self.assertFalse(entry["privacy"]["raw_prompts_included"])
            self.assertFalse(entry["privacy"]["provider_bodies_included"])
            self.assertFalse(entry["privacy"]["request_ids_included"])
            self.assertFalse(entry["privacy"]["session_ids_included"])
            self.assertFalse(entry["privacy"]["cache_keys_included"])
            self.assertFalse(entry["privacy"]["file_paths_included"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw-request-id-must-not-leak", rendered)
        self.assertNotIn("raw-session-id-must-not-leak", rendered)
        self.assertNotIn("/tmp/private/rollup.py", rendered)

    def test_rollups_backfill_from_recent_codex_metadata_windows_without_call_rows(self) -> None:
        raw_request_id = "raw-codex-request-id-must-not-leak"
        raw_session_id = "raw-codex-session-id-must-not-leak"
        raw_thread_id = "raw-codex-thread-id-must-not-leak"
        raw_prompt = "raw codex prompt must not leak"
        raw_path = "/tmp/private/codex/source.py"
        for index in range(2):
            self.store.log_codex_app_event(
                id=f"codex-start-{index}",
                created_at=f"2026-06-21T01:00:0{index}+00:00",
                direction="client_to_server",
                method="turn/start",
                request_id=raw_request_id,
                thread_id=raw_thread_id,
                message_chars=42_000,
                params_chars=200,
                input_items=3,
                input_text_chars=42_000,
                result_chars=None,
                error_code=None,
                error_message=None,
                latency_ms=None,
                session_id=raw_session_id,
                routing_json=stable_json({
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "has_tools": True,
                    "model": "gpt-5",
                }),
                crunch_json=stable_json({"changed": False, "tokens_saved_est": 0}),
                cache_json=stable_json({"status": "skipped", "reason": "codex-app-cache-disabled"}),
                event_window_json=stable_json({
                    "schema": "tokenclaw.codex_app_event_window.v1",
                    "workflow_phase": "tool-execution",
                    "input_text_chars": 42_000,
                    "method_counts": {"turn/start": 1, "item/commandExecution/outputDelta": 4},
                    "request_id": raw_request_id,
                    "thread_id": raw_thread_id,
                    "session_id": raw_session_id,
                    "prompt": raw_prompt,
                    "file_path": raw_path,
                }),
            )

        report = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="codex-backfill")
        follow_up = report["follow_up_candidates"]

        self.assertEqual(report["window"]["source"], "recent-local-metadata-window-backfill")
        self.assertEqual(report["summary"]["rows_considered"], 2)
        self.assertEqual(report["summary"]["metadata_window_backfill_rows"], 2)
        self.assertTrue(report["summary"]["source_traffic_acquisition_attempted"])
        self.assertEqual(report["summary"]["source_traffic_acquisition_status"], "completed")
        self.assertEqual(report["source_traffic_acquisition"]["status"], "completed")
        self.assertEqual(report["source_traffic_acquisition"]["rows_considered"], 2)
        self.assertGreaterEqual(report["source_traffic_acquisition"]["rollup_count"], 1)
        self.assertIsNone(report["source_traffic_acquisition"]["blocker_code"])
        self.assertGreaterEqual(report["summary"]["rollup_count"], 1)
        self.assertEqual(follow_up["schema"], "tokenclaw.request_shape_follow_up_candidates.v1")
        self.assertEqual(follow_up["status"], "candidates-ranked")
        self.assertEqual(follow_up["source_traffic_acquisition_status"], "completed")
        self.assertEqual(follow_up["summary"]["rows_considered"], 2)
        self.assertGreaterEqual(follow_up["summary"]["ranked_candidate_count"], 1)
        self.assertIsNotNone(follow_up["summary"]["top_next_action"])
        self.assertTrue(follow_up["privacy"]["metadata_only"])
        self.assertTrue(follow_up["privacy"]["aggregate_only"])
        self.assertFalse(follow_up["privacy"]["provider_calls_made"])
        self.assertFalse(follow_up["privacy"]["managed_server_calls_made"])
        self.assertFalse(follow_up["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["provider_calls_made"])
        self.assertFalse(report["privacy"]["managed_server_calls_made"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn(raw_request_id, rendered)
        self.assertNotIn(raw_session_id, rendered)
        self.assertNotIn(raw_thread_id, rendered)
        self.assertNotIn(raw_prompt, rendered)
        self.assertNotIn(raw_path, rendered)

    def test_rollups_preserve_no_source_traffic_when_calls_and_metadata_windows_empty(self) -> None:
        report = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="empty-backfill")

        self.assertEqual(report["summary"]["rows_considered"], 0)
        self.assertEqual(report["summary"]["rollup_count"], 0)
        self.assertFalse(report["summary"]["metadata_window_backfilled"])
        self.assertTrue(report["summary"]["source_traffic_acquisition_attempted"])
        self.assertEqual(report["source_traffic_acquisition"]["status"], "no-source-traffic")
        follow_up = report["follow_up_candidates"]
        self.assertEqual(follow_up["status"], "no-source-traffic")
        self.assertEqual(follow_up["source_traffic_acquisition_status"], "no-source-traffic")
        self.assertTrue(follow_up["source_traffic_acquisition"]["attempted"])
        self.assertEqual(follow_up["summary"]["top_next_action"], "emit-request-shape-rollups")
        self.assertEqual(follow_up["missing_measurements"], ["no-source-traffic-for-request-shape-rollups"])

    def test_rollups_backfill_from_dashboard_added_routing_candidates_without_call_rows(self) -> None:
        raw_candidate_id = "raw-dashboard-candidate-id-must-not-leak"
        policy = {
            "routing_candidates": [
                {
                    "candidate_id": raw_candidate_id,
                    "candidate_source": "dashboard-added",
                    "requested_model": "gpt-5-codex",
                    "routed_model": "gpt-5-mini",
                    "provider": "openai",
                    "source_surface": "codex_turn",
                    "app_family": "codex",
                    "category": "codex-turn",
                    "workflow_phase": "summary",
                    "stream": False,
                    "max_text_chars": 8000,
                    "sample_weight": 4,
                },
                {
                    "candidate_id": "dashboard-claude-tool-result",
                    "candidate_source": "dashboard-recent-call",
                    "requested_model": "claude-opus-4-8",
                    "routed_model": "claude-sonnet-4-6",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "app_family": "claude_code",
                    "category": "tool-result",
                    "workflow_phase": "tool-execution",
                    "stream": True,
                    "max_text_chars": 128000,
                },
            ],
            "model_pairs": [],
        }
        with patch.object(routing_experiments, "ROUTING_EXPERIMENT_POLICY", policy):
            report = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="dashboard-backfill")
            repeated = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="dashboard-backfill-2")

        follow_up = report["follow_up_candidates"]
        self.assertEqual(report["window"]["source"], "dashboard-routing-candidates")
        self.assertTrue(report["summary"]["dashboard_candidate_backfilled"])
        self.assertEqual(report["summary"]["dashboard_candidate_backfill_rows"], 5)
        self.assertEqual(report["summary"]["rows_considered"], 5)
        self.assertEqual(report["source_traffic_acquisition"]["source"], "dashboard-routing-candidates")
        self.assertEqual(report["source_traffic_acquisition"]["status"], "completed")
        self.assertEqual(follow_up["schema"], "tokenclaw.request_shape_follow_up_candidates.v1")
        self.assertEqual(follow_up["status"], "candidates-ranked")
        self.assertEqual(follow_up["source_traffic_acquisition_status"], "completed")
        self.assertEqual(follow_up["missing_measurements"], [])
        self.assertGreaterEqual(follow_up["summary"]["ranked_candidate_count"], 2)
        top = follow_up["top_candidate"]
        self.assertTrue(top["fingerprint"].startswith("request-shape-follow-up:"))
        self.assertEqual(top["local_action_family"], "routing")
        self.assertEqual(top["next_action"], "stage-routing-lifecycle-evidence")
        self.assertEqual(top["readiness_state"], "needs-lifecycle-evidence")
        self.assertIn("routing", top["candidate_work_classes"])
        self.assertIn("routing_candidate", top["candidate_families"])
        self.assertTrue(top["privacy"]["metadata_only"])
        self.assertTrue(top["privacy"]["aggregate_only"])
        self.assertFalse(top["privacy"]["provider_calls_made"])
        self.assertFalse(top["privacy"]["managed_server_calls_made"])
        self.assertFalse(top["privacy"]["individual_candidate_ids_included"])
        self.assertEqual(
            [item["fingerprint"] for item in follow_up["candidates"]],
            [item["fingerprint"] for item in repeated["follow_up_candidates"]["candidates"]],
        )
        rendered = json.dumps(report, sort_keys=True)
        rendered_follow_up = json.dumps(follow_up, sort_keys=True)
        self.assertNotIn(raw_candidate_id, rendered)
        self.assertNotIn('"candidate_id"', rendered_follow_up)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])

    def test_routing_downgrade_drills_rank_review_only_candidates_with_stable_fingerprints(self) -> None:
        rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "source_schema": "tokenclaw.request_shape_rollups.v1",
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "requested_model_family": "gpt-5",
                "routed_model_family": "gpt-5",
                "category": "chat",
                "workflow_phase": "chat",
                "stream": False,
                "has_tools": False,
                "text_bucket": "8k_32k_chars",
                "token_bucket": "500_2k_tokens",
                "cache_status": "miss",
                "routing_status": "passthrough",
                "candidate_families": ["routing_candidate"],
                "candidate_work_classes": ["routing"],
                "blocker_codes": [],
                "row_count": 20,
                "sample_count": 20,
                "error_count": 0,
                "retry_count": 0,
                "input_tokens": 20_000,
                "successful_input_tokens": 20_000,
                "output_tokens": 2_000,
            }
        ]

        report = build_request_shape_routing_downgrade_drill_report(rollups)
        repeated = build_request_shape_routing_downgrade_drill_report(rollups)

        self.assertEqual(report["schema"], "tokenclaw.request_shape_routing_downgrade_drills.v1")
        self.assertEqual(report["status"], "ranked")
        self.assertEqual(report["summary"]["candidate_count"], 1)
        self.assertEqual(report["summary"]["review_ready_count"], 1)
        self.assertEqual(report["summary"]["blocked_count"], 0)
        self.assertFalse(report["summary"]["policy_files_written"])
        candidate = report["candidates"][0]
        self.assertEqual(candidate["fingerprint"], repeated["candidates"][0]["fingerprint"])
        self.assertTrue(candidate["fingerprint"].startswith("routing-drill:"))
        self.assertEqual(candidate["status"], "review-ready")
        self.assertEqual(candidate["sample_count"], 20)
        self.assertEqual(candidate["candidate_target_model"], "gpt-5-mini")
        self.assertGreater(candidate["projected_savings_per_1000_calls_usd"], 0)
        self.assertEqual(candidate["recommended_canary_fraction"], 0.1)
        self.assertEqual(candidate["recommended_holdout_fraction"], 0.1)
        self.assertEqual(candidate["recommended_canary_sample_count"], 2)
        self.assertEqual(candidate["recommended_holdout_sample_count"], 2)
        self.assertIn("missing-quality-evidence", candidate["blocker_codes"])
        self.assertFalse(candidate["emits_routing_apply_action"])
        self.assertFalse(candidate["policy_files_written"])
        self.assertTrue(candidate["review_only"])
        self.assertTrue(candidate["privacy"]["metadata_only"])
        self.assertTrue(candidate["privacy"]["aggregate_only"])
        self.assertFalse(candidate["privacy"]["raw_prompts_included"])
        self.assertFalse(candidate["privacy"]["provider_calls_made"])
        self.assertFalse(candidate["privacy"]["managed_server_calls_made"])
        self.assertTrue(report["acceptance"]["has_stable_fingerprints"])
        self.assertTrue(report["acceptance"]["has_projected_savings_per_1000_calls"])
        self.assertTrue(report["acceptance"]["has_canary_and_holdout_sizing"])
        self.assertTrue(report["acceptance"]["emits_no_routing_apply_actions"])

    def test_routing_downgrade_drills_block_stale_too_small_and_unsafe_shapes(self) -> None:
        base = {
            "schema": "tokenclaw.request_shape_rollup_row.v1",
            "source_schema": "tokenclaw.request_shape_rollups.v1",
            "provider_family": "anthropic",
            "source_surface": "anthropic_messages",
            "endpoint": "messages",
            "requested_model_family": "claude-sonnet",
            "routed_model_family": "claude-sonnet",
            "workflow_phase": "tool-execution",
            "stream": True,
            "has_tools": False,
            "text_bucket": "8k_32k_chars",
            "token_bucket": "500_2k_tokens",
            "cache_status": "skipped",
            "routing_status": "passthrough",
            "candidate_families": ["routing_candidate"],
            "candidate_work_classes": ["routing"],
            "error_count": 0,
            "retry_count": 0,
            "input_tokens": 20_000,
            "successful_input_tokens": 20_000,
            "output_tokens": 2_000,
        }
        rollups = [
            {**base, "category": "chat", "row_count": 20, "sample_count": 20, "freshness_state": "stale"},
            {**base, "category": "chat", "row_count": 1, "sample_count": 1},
            {
                **base,
                "category": "thinking",
                "workflow_phase": "thinking",
                "row_count": 20,
                "sample_count": 20,
                "blocker_codes": ["thinking-routing-guard"],
            },
        ]

        report = build_request_shape_routing_downgrade_drill_report(rollups)

        self.assertEqual(report["status"], "ranked")
        self.assertEqual(report["summary"]["candidate_count"], 3)
        self.assertEqual(report["summary"]["review_ready_count"], 0)
        self.assertEqual(report["summary"]["blocked_count"], 3)
        blockers = {row["top_blocker_code"] for row in report["candidates"]}
        self.assertIn("stale-request-shape-rollup", blockers)
        self.assertIn("too-small-routing-drill-sample", blockers)
        self.assertIn("thinking-routing-guard", blockers)
        breakdown = {row["value"]: row["count"] for row in report["blocker_breakdown"]}
        self.assertGreater(breakdown["stale-request-shape-rollup"], 0)
        self.assertGreater(breakdown["too-small-routing-drill-sample"], 0)
        self.assertGreater(breakdown["thinking-routing-guard"], 0)
        for candidate in report["candidates"]:
            self.assertEqual(candidate["status"], "blocked")
            self.assertEqual(candidate["recommended_canary_fraction"], 0.0)
            self.assertEqual(candidate["recommended_holdout_fraction"], 0.0)
            self.assertEqual(candidate["recommended_canary_sample_count"], 0)
            self.assertEqual(candidate["recommended_holdout_sample_count"], 0)
            self.assertFalse(candidate["emits_routing_apply_action"])
            self.assertFalse(candidate["policy_files_written"])
            self.assertTrue(candidate["privacy"]["metadata_only"])

    def _cache_replay_hit_recovery_smoke(self) -> dict[str, object]:
        return {
            "schema": "tokenclaw.cache_replay_hit_recovery_smoke.v1",
            "status": "hit-recovered",
            "reason": "synthetic-repeat-exact-cache-hit",
            "target_rule_id": "local-openai-cache-replay-canary-ae8404ee817f89f4",
            "target_shape": {
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "chat",
                "workflow_phase": "chat",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "has_tools": False,
                "stream": False,
            },
            "summary": {
                "synthetic_requests": 2,
                "provider_calls_made": 0,
                "cache_entries_written": True,
                "exact_hit_count": 1,
                "observed_hits": 1,
                "hit_recovery_demonstrated": True,
            },
            "privacy": {
                "metadata_only": True,
                "synthetic_only": True,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "raw_request_bodies_included": False,
                "raw_responses_included": False,
                "cache_keys_included": False,
                "request_fingerprints_included": False,
                "session_ids_included": False,
            },
        }

    def _dependency_audit(
        self,
        *,
        reason: str | None = None,
        safe: bool = False,
        fingerprint_available: bool = False,
    ) -> dict[str, object]:
        return {
            "schema": "tokenclaw.cache_file_dependency_audit.v1",
            "file_watch_enabled": True,
            "snapshot_root_policy": "cwd-relative",
            "root_path_included": False,
            "snapshot_count": 1 if fingerprint_available else 0,
            "snapshot_count_bucket": "1" if fingerprint_available else "0",
            "candidate_path_count_bucket": "1" if fingerprint_available else "0",
            "raw_candidate_path_count_bucket": "1" if fingerprint_available else "0",
            "distinct_candidate_path_count_bucket": "1" if fingerprint_available else "0",
            "max_paths": 128,
            "cap_exceeded": False,
            "cap_trimmed": False,
            "dependency_capture_reason": "complete",
            "present_path_count": 1 if fingerprint_available else 0,
            "missing_path_count": 0,
            "changed_path_count": 1 if reason == "dependency-changed" else 0,
            "deleted_path_count": 0,
            "created_path_count": 0,
            "invalidation_reason": reason,
            "safe_invalidation_evidence": safe,
            "file_dependency_evidence_available": fingerprint_available,
            "paths_included": False,
            "path_hashes_included": False,
            "raw_stat_values_included": False,
        }

    def _managed_tool_cache_preview_outcomes(
        self,
        *,
        next_action: str = "review-tool-cache-replay-candidate",
        classification: str = "review-only",
        decision: str = "review-only-recommendation",
    ) -> dict[str, object]:
        return {
            "schema": "tokenclaw.managed_activation_preview_outcomes.v1",
            "review_only": True,
            "outcomes": [
                {
                    "schema": "tokenclaw.managed_activation_preview_outcome.v1",
                    "handoff_ref": "managed-preview-cache-ref",
                    "preview_ref": "managed-preview-cache-preview",
                    "local_action_family": "cache",
                    "evidence_schema": "tokenclaw.request_shape_tool_cache_replay_evidence.v1",
                    "classification": classification,
                    "decision": decision,
                    "next_action": next_action,
                    "review_only": True,
                    "failed_closed": False,
                    "stale": False,
                    "missing_preview_decision": False,
                    "disagrees_with_local_evidence": False,
                    "policy_files_written": False,
                    "provider_calls_made": False,
                    "managed_server_calls_made": False,
                    "privacy": {"metadata_only": True, "aggregate_only": True},
                }
            ],
        }

    def test_tool_cache_stable_dependency_evidence_emits_review_only_candidate(self) -> None:
        cohorts = [
            {
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "workflow_phase": "tool-light",
                "stream": False,
                "has_tools": True,
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "readiness": "skipped",
                "reason": "safe-invalidation-evidence-present",
                "blockers": ["safe-invalidation-evidence-present", "tools-present"],
                "row_count": 12,
                "projected_hits": 8,
                "projected_savings_usd": 0.02,
                "cache_hit_count": 1,
                "file_dependency_status": "stable",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(safe=True, fingerprint_available=True),
            }
        ]

        report = build_request_shape_tool_cache_replay_evidence_report(
            cohorts,
            limit=10,
            managed_preview_outcomes=self._managed_tool_cache_preview_outcomes(),
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_tool_cache_replay_evidence.v1")
        self.assertEqual(report["summary"]["stable_dependency_evidence_rows"], 12)
        self.assertEqual(report["summary"]["review_only_candidate_count"], 1)
        self.assertEqual(report["summary"]["review_ready_rows"], 12)
        self.assertEqual(report["summary"]["stageable_after_review_rows"], 12)
        self.assertTrue(report["acceptance"]["promotes_stable_dependency_evidence_to_review_only_candidates"])
        self.assertTrue(report["acceptance"]["promotes_stable_dependency_evidence_to_managed_local_replay_previews"])
        self.assertTrue(report["acceptance"]["stable_with_proof_emits_review_ready_replay_preview"])
        self.assertTrue(report["acceptance"]["review_only_candidates_do_not_allow_savings_floor_without_replay_proof"])
        self.assertTrue(report["acceptance"]["review_only_candidates_require_live_repeat_or_observed_hit_proof"])
        review = report["review_only_candidates"]
        self.assertEqual(review["schema"], "tokenclaw.request_shape_tool_cache_review_candidates.v1")
        self.assertEqual(review["summary"]["review_only_candidate_count"], 1)
        self.assertEqual(review["summary"]["cache_entries_written"], 0)
        self.assertFalse(review["summary"]["policy_files_written"])
        self.assertFalse(review["summary"]["tool_cache_replay_enabled"])
        self.assertFalse(review["summary"]["streaming_replay_enabled"])
        self.assertTrue(review["acceptance"]["stable_dependency_evidence_emits_review_only_candidate"])
        candidate = review["candidates"][0]
        self.assertTrue(candidate["review_only"])
        self.assertEqual(candidate["candidate_status"], "review-ready")
        self.assertEqual(candidate["candidate_decision"], "review-only-candidate")
        self.assertEqual(candidate["next_action"], "review-tool-cache-replay-candidate")
        self.assertTrue(candidate["stageable_after_review"])
        self.assertFalse(candidate["stage_allowed"])
        self.assertEqual(candidate["readiness_gate"]["gate_status"], "live-repeat-confirmed")
        self.assertTrue(candidate["readiness_gate"]["stage_allowed"])
        self.assertTrue(candidate["readiness_gate"]["tool_cache_replay_review_gate"]["proof_gate_passed"])
        self.assertTrue(candidate["replay_proof"]["proof_available"])
        self.assertTrue(candidate["replay_proof"]["live_repeat_confirmed"])
        self.assertFalse(candidate["replay_proof"]["observed_hit_proof"])
        self.assertEqual(candidate["projected_hits"], 8)
        self.assertEqual(candidate["projected_savings_usd"], 0.02)
        self.assertFalse(candidate["tool_cache_replay_enabled"])
        self.assertFalse(candidate["streaming_replay_enabled"])
        self.assertFalse(candidate["emits_cache_apply_action"])
        self.assertEqual(candidate["cache_entries_written"], 0)
        self.assertFalse(candidate["policy_files_written"])
        self.assertTrue(candidate["privacy"]["metadata_only"])
        self.assertFalse(candidate["privacy"]["cache_keys_included"])
        self.assertFalse(candidate["privacy"]["file_paths_included"])
        previews = report["managed_local_replay_previews"]
        self.assertEqual(previews["schema"], "tokenclaw.request_shape_tool_cache_managed_local_replay_previews.v1")
        self.assertEqual(previews["summary"]["preview_count"], 1)
        self.assertEqual(previews["summary"]["review_ready_preview_rows"], 12)
        self.assertEqual(previews["summary"]["managed_preview_required_rows"], 12)
        self.assertEqual(previews["summary"]["managed_preview_agreement_rows"], 12)
        self.assertEqual(previews["summary"]["cache_entries_written"], 0)
        self.assertFalse(previews["summary"]["policy_files_written"])
        self.assertFalse(previews["summary"]["tool_cache_replay_enabled"])
        preview = previews["previews"][0]
        self.assertEqual(preview["preview_status"], "review-ready")
        self.assertEqual(preview["preview_decision"], "review-only-replay-preview")
        self.assertTrue(preview["managed_preview_agreement"]["required"])
        self.assertTrue(preview["managed_preview_agreement"]["agreed"])
        self.assertEqual(preview["managed_preview_agreement"]["reason"], "local-managed-preview-agree")
        self.assertFalse(preview["tool_cache_replay_enabled"])
        self.assertFalse(preview["streaming_replay_enabled"])
        self.assertFalse(preview["emits_cache_apply_action"])
        self.assertEqual(preview["cache_entries_written"], 0)
        self.assertFalse(preview["policy_files_written"])
        self.assertTrue(previews["acceptance"]["managed_preview_agreement_is_review_only"])

    def test_tool_cache_stable_dependency_without_repeat_proof_is_noop_review_drill(self) -> None:
        cohorts = [
            {
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "workflow_phase": "tool-light",
                "stream": False,
                "has_tools": True,
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "readiness": "skipped",
                "reason": "safe-invalidation-evidence-present",
                "blockers": ["safe-invalidation-evidence-present", "tools-present"],
                "row_count": 20,
                "projected_hits": 12,
                "projected_savings_usd": 0.08,
                "file_dependency_status": "stable",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(safe=True, fingerprint_available=True),
            }
        ]

        report = build_request_shape_tool_cache_replay_evidence_report(cohorts, limit=10)

        review = report["review_only_candidates"]
        candidate = review["candidates"][0]
        self.assertEqual(candidate["candidate_status"], "review-only-gated")
        self.assertEqual(candidate["candidate_decision"], "no-op-missing-live-repeat-proof")
        self.assertEqual(candidate["next_action"], "wait-for-live-repeat-or-observed-hit-proof")
        self.assertEqual(candidate["no_op_reason"], "missing-live-repeat-or-observed-hit-proof")
        self.assertFalse(candidate["stageable_after_review"])
        self.assertFalse(candidate["stage_allowed"])
        self.assertEqual(candidate["readiness_gate"]["raw_stage_gate_status"], "savings-floor-met")
        self.assertEqual(candidate["readiness_gate"]["gate_status"], "missing-live-repeat-or-observed-hit-proof")
        self.assertFalse(candidate["readiness_gate"]["stage_allowed"])
        self.assertFalse(candidate["readiness_gate"]["tool_cache_replay_review_gate"]["proof_gate_passed"])
        self.assertFalse(candidate["allows_savings_floor_without_replay_proof"])
        self.assertFalse(candidate["review_only"])
        self.assertFalse(candidate["replay_proof"]["proof_available"])
        self.assertFalse(candidate["replay_proof"]["live_repeat_confirmed"])
        self.assertFalse(candidate["replay_proof"]["observed_hit_proof"])
        self.assertEqual(candidate["replay_proof"]["reason"], "missing-live-repeat-or-observed-hit-proof")
        self.assertFalse(candidate["tool_cache_replay_enabled"])
        self.assertFalse(candidate["streaming_replay_enabled"])
        self.assertFalse(candidate["emits_cache_apply_action"])
        self.assertEqual(candidate["cache_entries_written"], 0)
        self.assertFalse(candidate["policy_files_written"])
        self.assertTrue(review["acceptance"]["requires_live_repeat_or_observed_hit_proof"])
        self.assertTrue(review["acceptance"]["does_not_allow_savings_floor_without_replay_proof"])
        self.assertTrue(review["acceptance"]["review_only_candidates_have_stable_dependency_and_proof"])
        self.assertFalse(review["acceptance"]["stable_dependency_evidence_emits_review_only_candidate"])
        self.assertFalse(report["acceptance"]["promotes_stable_dependency_evidence_to_review_only_candidates"])
        self.assertTrue(review["acceptance"]["stable_without_live_repeat_or_observed_hit_is_noop"])
        self.assertTrue(report["acceptance"]["stable_without_live_repeat_or_observed_hit_is_noop"])
        preview = report["managed_local_replay_previews"]["previews"][0]
        self.assertEqual(preview["preview_status"], "review-only-noop")
        self.assertEqual(preview["preview_decision"], "no-op-missing-live-repeat-proof")
        self.assertFalse(preview["managed_preview_agreement"]["required"])
        self.assertFalse(preview["tool_cache_replay_enabled"])
        self.assertFalse(preview["streaming_replay_enabled"])
        self.assertFalse(preview["emits_cache_apply_action"])
        self.assertEqual(preview["cache_entries_written"], 0)
        self.assertTrue(report["acceptance"]["stable_without_live_repeat_or_observed_hit_preview_is_noop"])

    def test_tool_cache_stable_streaming_dependency_with_repeat_proof_stays_blocked(self) -> None:
        cohorts = [
            {
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "category": "tool-light",
                "workflow_phase": "tool-light",
                "stream": True,
                "has_tools": True,
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "readiness": "skipped",
                "reason": "safe-invalidation-evidence-present",
                "blockers": [
                    "safe-invalidation-evidence-present",
                    "streaming-replay-not-supported",
                    "tools-present",
                ],
                "row_count": 12,
                "projected_hits": 8,
                "projected_savings_usd": 0.02,
                "cache_hit_count": 1,
                "file_dependency_status": "stable",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(safe=True, fingerprint_available=True),
            }
        ]

        report = build_request_shape_tool_cache_replay_evidence_report(
            cohorts,
            limit=10,
            managed_preview_outcomes=self._managed_tool_cache_preview_outcomes(),
        )

        self.assertEqual(report["summary"]["stable_dependency_evidence_rows"], 12)
        self.assertEqual(report["summary"]["review_ready_rows"], 0)
        self.assertEqual(report["summary"]["review_only_candidate_count"], 0)
        self.assertEqual(report["summary"]["stageable_after_review_rows"], 0)
        self.assertEqual(report["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(report["summary"]["cache_entries_written"], 0)
        self.assertFalse(report["summary"]["policy_files_written"])
        self.assertTrue(report["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(report["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(report["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(report["acceptance"]["tool_and_streaming_replay_remain_disabled"])
        self.assertTrue(report["acceptance"]["streaming_candidates_do_not_become_review_ready"])
        self.assertTrue(report["acceptance"]["streaming_shapes_do_not_become_review_ready_previews"])
        self.assertTrue(report["acceptance"]["review_ready_previews_require_stable_dependency_and_proof"])

        review = report["review_only_candidates"]
        self.assertEqual(review["summary"]["review_ready_rows"], 0)
        self.assertEqual(review["summary"]["blocked_rows"], 12)
        self.assertTrue(review["acceptance"]["streaming_candidates_do_not_become_review_ready"])
        candidate = review["candidates"][0]
        self.assertEqual(candidate["candidate_status"], "blocked")
        self.assertEqual(candidate["candidate_decision"], "no-op-streaming-replay-not-supported")
        self.assertEqual(candidate["blocker_reason"], "streaming-replay-not-supported")
        self.assertEqual(candidate["next_action"], "stage-streaming-replay-buffer-fixture")
        self.assertEqual(candidate["readiness_gate"]["gate_status"], "streaming-replay-not-supported")
        self.assertTrue(candidate["replay_proof"]["proof_available"])
        self.assertFalse(candidate["review_only"])
        self.assertFalse(candidate["stageable_after_review"])
        self.assertFalse(candidate["stage_allowed"])
        self.assertFalse(candidate["tool_cache_replay_enabled"])
        self.assertFalse(candidate["streaming_replay_enabled"])
        self.assertFalse(candidate["emits_cache_apply_action"])
        self.assertEqual(candidate["cache_entries_written"], 0)
        self.assertFalse(candidate["policy_files_written"])

        previews = report["managed_local_replay_previews"]
        self.assertEqual(previews["summary"]["review_ready_preview_rows"], 0)
        self.assertEqual(previews["summary"]["blocked_preview_rows"], 12)
        self.assertTrue(previews["acceptance"]["streaming_shapes_do_not_become_review_ready_previews"])
        preview = previews["previews"][0]
        self.assertEqual(preview["preview_status"], "blocked")
        self.assertEqual(preview["preview_decision"], "keep-blocked")
        self.assertEqual(preview["blocker_reason"], "streaming-replay-not-supported")
        self.assertFalse(preview["tool_cache_replay_enabled"])
        self.assertFalse(preview["streaming_replay_enabled"])
        self.assertFalse(preview["emits_cache_apply_action"])
        self.assertEqual(preview["cache_entries_written"], 0)
        self.assertFalse(preview["policy_files_written"])

        coverage = report["dependency_fingerprint_coverage"]
        self.assertEqual(coverage["summary"]["stable_dependency_evidence_rows"], 12)
        self.assertEqual(coverage["summary"]["file_dependency_fingerprint_available_rows"], 12)
        self.assertEqual(coverage["summary"]["safe_invalidation_evidence_rows"], 12)
        self.assertTrue(coverage["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(coverage["acceptance"]["emits_no_cache_apply_actions"])

    def test_tool_cache_nonstable_dependency_evidence_emits_blocked_review_rows(self) -> None:
        base = {
            "provider_family": "openai",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "category": "tool-light",
            "workflow_phase": "tool-light",
            "stream": False,
            "has_tools": True,
            "cache_status": "skipped",
            "routing_status": "passthrough",
            "readiness": "skipped",
            "row_count": 3,
            "projected_hits": 2,
            "projected_savings_usd": 0.005,
        }
        cohorts = [
            {
                **base,
                "reason": "stale-dependency-evidence",
                "blockers": ["stale-dependency-evidence", "tools-present"],
                "file_dependency_status": "invalidated",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(
                    reason="dependency-changed",
                    safe=False,
                    fingerprint_available=True,
                ),
            },
            {
                **base,
                "reason": "unsafe-tool-calls-without-invalidation",
                "blockers": ["unsafe-tool-calls-without-invalidation", "tools-present"],
                "file_dependency_status": "unsafe",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(
                    reason="unsafe-tool-calls-without-invalidation",
                    safe=False,
                    fingerprint_available=True,
                ),
            },
            {
                **base,
                "reason": "invalidation-evidence-missing",
                "blockers": ["invalidation-evidence-missing", "tools-present"],
                "file_dependency_status": "missing",
                "file_dependency_fingerprint_available": False,
                "file_dependency_audit": self._dependency_audit(safe=False, fingerprint_available=False),
            },
            {
                **base,
                "reason": "dependency-evidence-unknown",
                "blockers": ["dependency-evidence-unknown", "tools-present"],
                "file_dependency_status": "unknown",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(safe=False, fingerprint_available=True),
            },
            {
                **base,
                "reason": "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
                "blockers": ["retire-staged-no-repeat", "repeat-window-elapsed-no-live-repeat", "tools-present"],
                "file_dependency_status": "stable",
                "file_dependency_fingerprint_available": True,
                "file_dependency_audit": self._dependency_audit(safe=True, fingerprint_available=True),
                "current_status": "retired-no-repeat",
                "promotion_decision": "retire-staged-no-repeat",
                "observed_hit_blocker": "repeat-window-elapsed-no-live-repeat",
                "reason_codes": [
                    "retire-staged-no-repeat",
                    "repeat-window-elapsed-no-live-repeat",
                ],
            },
        ]

        report = build_request_shape_tool_cache_replay_evidence_report(cohorts, limit=10)

        review = report["review_only_candidates"]
        self.assertEqual(review["summary"]["review_only_candidate_count"], 0)
        self.assertEqual(review["summary"]["blocked_rows"], 15)
        self.assertEqual(review["summary"]["review_ready_rows"], 0)
        self.assertEqual(review["summary"]["cache_entries_written"], 0)
        self.assertFalse(review["summary"]["policy_files_written"])
        self.assertTrue(review["acceptance"]["blocked_dependency_evidence_has_distinct_reason_codes"])
        self.assertTrue(review["acceptance"]["retired_no_repeat_emits_noop"])
        self.assertTrue(report["acceptance"]["retired_no_repeat_emits_noop"])
        previews = report["managed_local_replay_previews"]
        self.assertEqual(previews["summary"]["preview_count"], 5)
        self.assertEqual(previews["summary"]["cache_entries_written"], 0)
        self.assertFalse(previews["summary"]["policy_files_written"])
        self.assertFalse(previews["summary"]["tool_cache_replay_enabled"])
        self.assertTrue(previews["acceptance"]["unsafe_or_missing_dependency_emits_no_apply_actions"])
        by_reason = {row["blocker_reason"]: row for row in review["candidates"]}
        self.assertEqual(
            set(by_reason),
            {
                "stale-dependency-evidence",
                "unsafe-tool-calls-without-invalidation",
                "invalidation-evidence-missing",
                "dependency-evidence-unknown",
                "retire-staged-no-repeat",
            },
        )
        self.assertEqual(by_reason["stale-dependency-evidence"]["next_action"], "refresh-file-invalidation-evidence")
        self.assertEqual(
            by_reason["unsafe-tool-calls-without-invalidation"]["next_action"],
            "collect-file-invalidation-evidence",
        )
        self.assertEqual(
            by_reason["invalidation-evidence-missing"]["next_action"],
            "collect-file-invalidation-evidence",
        )
        self.assertEqual(
            by_reason["dependency-evidence-unknown"]["next_action"],
            "collect-file-invalidation-evidence",
        )
        self.assertEqual(
            by_reason["retire-staged-no-repeat"]["next_action"],
            "keep-tool-cache-replay-retired-no-repeat",
        )
        self.assertEqual(by_reason["retire-staged-no-repeat"]["candidate_decision"], "no-op-retired-no-repeat")
        self.assertEqual(by_reason["retire-staged-no-repeat"]["no_op_reason"], "retire-staged-no-repeat")
        self.assertTrue(by_reason["retire-staged-no-repeat"]["retired_no_repeat"])
        for candidate in review["candidates"]:
            self.assertFalse(candidate["review_only"])
            self.assertEqual(candidate["candidate_status"], "blocked")
            self.assertFalse(candidate["stageable_after_review"])
            self.assertFalse(candidate["stage_allowed"])
            self.assertFalse(candidate["tool_cache_replay_enabled"])
            self.assertFalse(candidate["streaming_replay_enabled"])
            self.assertFalse(candidate["emits_cache_apply_action"])
            self.assertEqual(candidate["cache_entries_written"], 0)
            self.assertFalse(candidate["policy_files_written"])
            self.assertTrue(candidate["privacy"]["metadata_only"])
            self.assertFalse(candidate["privacy"]["tool_payloads_included"])

    def _log_call(
        self,
        *,
        provider: str = "anthropic",
        path: str = "/v1/messages",
        source_surface: str = "anthropic_messages",
        endpoint: str = "messages",
        requested_model: str = "claude-sonnet-4-6",
        routed_model: str = "claude-sonnet-4-6",
        requested_model_family: str = "claude-sonnet",
        routed_model_family: str = "claude-sonnet",
        category: str = "tool-result",
        workflow_phase: str = "tool-execution",
        stream: int = 1,
        has_tools: bool = True,
        cache_status: str = "skipped",
        cache_reason: str = "streaming",
        cache_hit: int = 0,
        text_chars: int = 24_000,
        cost: float = 0.02,
        baseline: float = 0.03,
        status_code: int = 200,
        latency_ms: int = 125,
        retry_count: int = 0,
        routing_reason: str = "keep requested model",
        routing_extra: dict[str, object] | None = None,
        cache_extra: dict[str, object] | None = None,
        crunch_extra: dict[str, object] | None = None,
        actual_input_tokens: int | None = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        routing_json: dict[str, object] = {
            "provider": provider,
            "requested_model": requested_model,
            "routed_model": routed_model,
            "text_chars": text_chars,
            "has_tools": has_tools,
            "category": category,
            "workflow_phase": workflow_phase,
            "reason": routing_reason,
        }
        if routing_extra:
            routing_json.update(routing_extra)
        cache_json: dict[str, object] = {
            "status": cache_status,
            "reason": cache_reason,
            "policy_source": "local-default",
            "request_fingerprint": "raw-request-fingerprint-must-not-leak",
            "cache_key": "raw-cache-key-must-not-leak",
        }
        if cache_extra:
            cache_json.update(cache_extra)
        input_tokens = actual_input_tokens if actual_input_tokens is not None else max(1, text_chars // 4)
        crunch_json: dict[str, object] = {"changed": False, "tokens_saved_est": 0}
        if crunch_extra:
            crunch_json.update(crunch_extra)
        self.store.log_call(
            id=call_id,
            created_at=utc_now(),
            path=path,
            requested_model=requested_model,
            routed_model=routed_model,
            stream=stream,
            cache_hit=cache_hit,
            status_code=status_code,
            latency_ms=latency_ms,
            input_tokens_est=input_tokens,
            output_tokens_est=50,
            actual_input_tokens=input_tokens,
            actual_output_tokens=50,
            cost_est_usd=cost,
            cost_baseline_usd=baseline,
            crunch_json=stable_json(crunch_json),
            routing_json=stable_json(routing_json),
            cache_json=stable_json(cache_json),
            error="request-id-secret must not leak" if status_code >= 400 else None,
            request_json=stable_json(
                {
                    "prompt": "raw prompt must not leak",
                    "messages": [{"content": "provider body must not leak"}],
                    "path": "/tmp/private/source.py",
                }
            ),
            response_json=stable_json({"content": "raw response must not leak"}),
            session_id="raw-session-id-must-not-leak",
            category=category,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            retry_count=retry_count,
            thinking_output_tokens=0,
            provider=provider,
            source_surface=source_surface,
            endpoint=endpoint,
            requested_model_family=requested_model_family,
            routed_model_family=routed_model_family,
        )
        return call_id

    def test_repeated_shapes_collapse_and_persist_without_raw_fields(self) -> None:
        for cost in (0.02, 0.03, 0.04):
            self._log_call(cost=cost, baseline=cost + 0.01)
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.004,
            baseline=0.004,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=True, run_id="test-rollup")

        self.assertEqual(report["schema"], "tokenclaw.request_shape_rollups.v1")
        self.assertTrue(report["persisted"])
        self.assertEqual(report["persisted_count"], 2)
        self.assertEqual(report["summary"]["rows_considered"], 4)
        self.assertEqual(report["summary"]["rollup_count"], 2)
        self.assertEqual(report["summary"]["collapsed_rows"], 2)
        self.assertEqual(report["summary"]["follow_up_candidate_count"], 2)
        self.assertIn("routing_downgrade_drills", report)
        self.assertEqual(report["routing_downgrade_drills"]["schema"], "tokenclaw.request_shape_routing_downgrade_drills.v1")
        self.assertIn("routing_downgrade_drill_candidate_count", report["summary"])
        self.assertEqual(report["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        self.assertTrue(report["snapshot_persisted"])
        self.assertEqual(report["snapshot_persisted_count"], 1)
        snapshot = report["rollup_snapshot"]
        self.assertEqual(snapshot["schema"], "tokenclaw.request_shape_rollup_snapshot.v1")
        self.assertEqual(snapshot["summary"]["rollup_count"], 2)
        self.assertEqual(snapshot["summary"]["ranked_candidate_count"], 2)
        self.assertEqual(snapshot["summary"]["top_readiness_state"], "activation-ready")
        self.assertTrue(snapshot["privacy"]["metadata_only"])
        self.assertTrue(snapshot["privacy"]["aggregate_only"])
        self.assertGreaterEqual(len(snapshot["rollups"]), 1)
        self.assertEqual(snapshot["rollups"][0]["schema"], "tokenclaw.request_shape_rollup_row.v1")
        self.assertGreater(snapshot["rollups"][0]["projected_crunch_tokens_saved"], 0)
        rendered_snapshot = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered_snapshot)
        self.assertNotIn("raw-session-id-must-not-leak", rendered_snapshot)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered_snapshot)
        latest_snapshot_report = latest_request_shape_rollup_snapshot_report(self.store)
        self.assertIsNotNone(latest_snapshot_report)
        assert latest_snapshot_report is not None
        self.assertEqual(latest_snapshot_report["snapshot_freshness"]["status"], "fresh")
        self.assertEqual(latest_snapshot_report["rollup_snapshot"]["summary"]["rollup_count"], 2)
        self.assertEqual(latest_snapshot_report["summary"]["snapshot_rehydrated_rollup_count"], 2)
        self.assertEqual(latest_snapshot_report["follow_up_candidates"]["status"], "candidates-ranked")
        self.assertIn("routing_downgrade_drills", latest_snapshot_report)
        self.assertEqual(
            latest_snapshot_report["routing_downgrade_drills"]["schema"],
            "tokenclaw.request_shape_routing_downgrade_drills.v1",
        )
        self.assertEqual(latest_snapshot_report["crunch_opportunity_dry_run"]["status"], "ranked")
        self.assertGreater(
            latest_snapshot_report["crunch_opportunity_dry_run"]["summary"]["projected_saved_tokens"],
            0,
        )
        follow_up = report["follow_up_candidates"]
        self.assertEqual(follow_up["schema"], "tokenclaw.request_shape_follow_up_candidates.v1")
        self.assertEqual(follow_up["status"], "candidates-ranked")
        self.assertEqual(follow_up["summary"]["ranked_candidate_count"], 2)
        self.assertEqual(follow_up["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(follow_up["summary"]["top_readiness_state"], "activation-ready")
        self.assertEqual(follow_up["summary"]["activation_ready_count"], 1)
        self.assertEqual(follow_up["top_candidate"]["local_action_family"], "crunch")
        self.assertEqual(follow_up["top_candidate"]["readiness_state"], "activation-ready")
        self.assertEqual(follow_up["top_candidate"]["sample_count"], 3)
        self.assertEqual(follow_up["top_candidate"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertGreater(follow_up["top_candidate"]["projected_saved_tokens"], 0)
        self.assertGreater(follow_up["top_candidate"]["projected_savings_usd"], 0)
        self.assertEqual(follow_up["blocker_cohorts"][0]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertIn("repeated_context", follow_up["top_candidate"]["candidate_work_classes"])
        self.assertIn("replayability", follow_up["top_candidate"]["candidate_work_classes"])
        self.assertTrue(follow_up["privacy"]["metadata_only"])
        self.assertTrue(follow_up["privacy"]["aggregate_only"])
        self.assertFalse(follow_up["privacy"]["individual_candidate_ids_included"])
        rendered_follow_up = json.dumps(follow_up, sort_keys=True)
        self.assertNotIn('"candidate_id"', rendered_follow_up)
        self.assertNotIn('"cohort_id"', rendered_follow_up)
        self.assertNotIn('"policy_id"', rendered_follow_up)
        repeated = next(row for row in report["rollups"] if row["row_count"] == 3)
        self.assertEqual(repeated["provider_family"], "anthropic")
        self.assertEqual(repeated["text_bucket"], "8k_32k_chars")
        self.assertIn("cache_replay", repeated["candidate_families"])
        self.assertIn("cache_blocker", repeated["candidate_families"])
        self.assertIn("repeated_context", repeated["candidate_work_classes"])
        self.assertIn("replayability", repeated["candidate_work_classes"])
        self.assertIn("crunch", repeated["candidate_work_classes"])
        self.assertIn("unsupported-streaming-shape", repeated["blocker_codes"])
        class_breakdown = {item["value"]: item["count"] for item in repeated["metadata"]["candidate_class_breakdown"]}
        self.assertEqual(class_breakdown["repeated_context"], 3)
        self.assertEqual(
            {item["value"]: item["count"] for item in repeated["metadata"]["cost_bucket_breakdown"]},
            {"0_01_0_05_usd": 3},
        )

        rows = self.store.request_shape_rollup_rows(run_id="test-rollup")
        self.assertEqual(len(rows), 2)
        persisted = json.dumps(rows, sort_keys=True)
        rendered = json.dumps(report, sort_keys=True) + persisted
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "request-id-secret",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["request_fingerprints_included"])

    def test_follow_up_candidates_rank_concrete_blocker_cohorts(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                text_chars=132_000,
                cost=0.09,
                baseline=0.09,
            )
        for _ in range(2):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=4_000,
                cost=0.01,
                baseline=0.01,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            routing_reason="keep requested model",
            text_chars=3_000,
            cost=0.02,
            baseline=0.02,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="blocker-cohorts")
        follow_up = report["follow_up_candidates"]
        top = follow_up["top_blocker_cohort"]

        self.assertEqual(top["schema"], "tokenclaw.request_shape_blocker_cohort.v1")
        self.assertEqual(top["rank"], 1)
        self.assertEqual(top["readiness_state"], "activation-ready")
        self.assertEqual(top["local_action_family"], "crunch")
        self.assertEqual(top["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(top["sample_count"], 3)
        self.assertIn("thinking-routing-guard", top["blocker_codes"])
        self.assertIn("tool-call-cache-disabled", top["blocker_codes"])
        self.assertIn("unsupported-streaming-shape", top["blocker_codes"])
        self.assertGreater(top["projected_saved_tokens"], 0)
        self.assertGreater(top["projected_savings_usd"], 0)
        self.assertTrue(top["privacy"]["metadata_only"])
        self.assertTrue(top["privacy"]["aggregate_only"])

        replay_ready = next(item for item in follow_up["blocker_cohorts"] if item["next_action"] == "stage-cache-replay-canary")
        self.assertEqual(replay_ready["readiness_state"], "activation-ready")
        self.assertEqual(replay_ready["local_action_family"], "cache")
        self.assertEqual(replay_ready["projected_hits"], 1)

        tool_blocked = next(item for item in follow_up["blocker_cohorts"] if item["next_action"] == "collect-tool-call-cache-invalidation-evidence")
        self.assertEqual(tool_blocked["readiness_state"], "blocked")
        self.assertEqual(tool_blocked["local_action_family"], "cache")
        rendered = json.dumps(follow_up, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_prompt_like_labels_are_rejected_to_unknown(self) -> None:
        self._log_call(
            category="raw prompt must not leak /tmp/category-secret.py",
            workflow_phase="tenant-id-secret planning",
            cache_reason="cache-key-secret request-id-secret prompt payload",
            routing_reason="session-id-secret should not leak",
            routing_extra={"workflow_phase": "tenant-id-secret planning"},
        )

        report = build_request_shape_rollups_report(self.store, limit=10, persist=False, run_id="redaction")
        rendered = json.dumps(report, sort_keys=True)

        self.assertEqual(report["rollups"][0]["category"], "unknown")
        self.assertEqual(report["rollups"][0]["workflow_phase"], "unknown")
        self.assertIn('"cache_reason_breakdown": [{"count": 1, "value": "unknown"}]', rendered)
        for forbidden in (
            "raw prompt must not leak",
            "/tmp/category-secret.py",
            "tenant-id-secret",
            "cache-key-secret",
            "request-id-secret",
            "session-id-secret",
            "prompt payload",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replayability_dry_run_ranks_ready_and_skipped_blockers(self) -> None:
        for cost in (0.01, 0.03):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=1_200,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="summary",
            workflow_phase="summary",
            stream=1,
            has_tools=False,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=12_000,
            cost=0.02,
            baseline=0.02,
        )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            text_chars=24_000,
            cost=0.04,
            baseline=0.04,
        )
        self._log_call(
            provider="openai",
            path="/v1/unknown",
            source_surface="unknown_surface",
            endpoint="unknown_endpoint",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.01,
            baseline=0.01,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="replay-dry-run")
        dry_run = report["cache_replayability_dry_run"]
        classification = report["cache_replay_blocker_classification"]
        invalidation_evidence = dry_run["cache_invalidation_evidence"]
        skipped_openai = dry_run["skipped_openai_blockers"]
        tool_replay_evidence = dry_run["tool_replay_evidence"]

        self.assertEqual(dry_run["schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(dry_run["summary"]["replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["skipped_cohort_count"], 3)
        self.assertEqual(dry_run["summary"]["projected_hits"], 1)
        self.assertAlmostEqual(dry_run["summary"]["projected_savings_usd"], 0.02)
        ready = next(row for row in dry_run["cohorts"] if row["readiness"] == "replay-ready")
        self.assertEqual(ready["reason"], "replay-ready-exact-non-tool-shape")
        self.assertEqual(ready["projected_hits"], 1)
        self.assertEqual(ready["next_action"], "stage-cache-replay-canary")
        self.assertTrue(ready["remaining_replay_ready"])
        self.assertEqual(ready["readiness_gate"]["gate_status"], "savings-floor-met")
        reasons = {item["value"]: item["count"] for item in dry_run["skipped_reason_breakdown"]}
        self.assertEqual(reasons["streaming-replay-not-supported"], 1)
        self.assertEqual(reasons["invalidation-evidence-missing"], 1)
        self.assertEqual(reasons["unsupported-endpoint"], 1)
        blockers = {item["value"]: item["count"] for item in dry_run["blocker_breakdown"]}
        self.assertIn("streaming-replay-not-supported", blockers)
        self.assertIn("tools-present", blockers)
        self.assertIn("invalidation-evidence-missing", blockers)
        self.assertIn("unsafe-tool-calls-without-invalidation", blockers)
        self.assertIn("unsupported-endpoint", blockers)
        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(dry_run["privacy"]["raw_request_bodies_included"])
        self.assertFalse(dry_run["privacy"]["cache_keys_included"])
        self.assertFalse(dry_run["privacy"]["request_fingerprints_included"])
        self.assertFalse(dry_run["privacy"]["individual_candidate_ids_included"])
        self.assertTrue(dry_run["acceptance"]["emits_durable_invalidation_evidence"])
        self.assertTrue(dry_run["acceptance"]["emits_skipped_openai_blocker_ranking"])
        self.assertTrue(dry_run["acceptance"]["emits_tool_replay_evidence"])
        self.assertTrue(dry_run["acceptance"]["has_ranked_blocker_cohorts"])
        self.assertTrue(dry_run["acceptance"]["has_ranked_skipped_openai_cohorts"])
        self.assertTrue(dry_run["acceptance"]["has_ranked_tool_cache_replay_evidence"])
        self.assertTrue(dry_run["acceptance"]["reduces_generic_tools_present_blocker"])
        self.assertTrue(dry_run["acceptance"]["tool_and_streaming_replay_remain_disabled"])
        self.assertTrue(dry_run["acceptance"]["has_local_file_backed_policy_compatibility"])

        self.assertEqual(
            skipped_openai["schema"],
            "tokenclaw.request_shape_skipped_openai_cache_replay_blockers.v1",
        )
        self.assertEqual(skipped_openai["status"], "ranked")
        self.assertTrue(skipped_openai["read_only"])
        self.assertEqual(skipped_openai["summary"]["skipped_openai_cohort_count"], 3)
        self.assertEqual(skipped_openai["summary"]["replay_ready_count"], 1)
        self.assertEqual(skipped_openai["summary"]["skipped_count"], 3)
        self.assertEqual(skipped_openai["summary"]["affected_rows"], 3)
        self.assertEqual(skipped_openai["summary"]["top_blocker_code"], "invalidation-evidence-missing")
        self.assertEqual(skipped_openai["summary"]["top_blocker_count"], 1)
        self.assertGreaterEqual(skipped_openai["summary"]["blocker_code_count"], 4)
        self.assertEqual(skipped_openai["next_action"], skipped_openai["summary"]["top_next_action"])
        self.assertEqual(skipped_openai["summary"]["cache_entries_written"], 0)
        self.assertFalse(skipped_openai["summary"]["policy_files_written"])
        self.assertEqual(skipped_openai["summary"]["cache_apply_action_count"], 0)
        self.assertTrue(skipped_openai["acceptance"]["has_ranked_skipped_openai_cohorts"])
        self.assertTrue(skipped_openai["acceptance"]["has_rank"])
        self.assertTrue(skipped_openai["acceptance"]["has_blocker_codes"])
        self.assertTrue(skipped_openai["acceptance"]["has_sample_count"])
        self.assertTrue(skipped_openai["acceptance"]["has_deterministic_next_action"])
        self.assertTrue(skipped_openai["acceptance"]["has_acceptance_summary_fields"])
        self.assertTrue(skipped_openai["acceptance"]["covers_required_blockers"])
        self.assertTrue(skipped_openai["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(skipped_openai["acceptance"]["tool_and_streaming_replay_remain_disabled"])
        self.assertTrue(skipped_openai["privacy"]["metadata_only"])
        self.assertTrue(skipped_openai["privacy"]["aggregate_only"])
        self.assertFalse(skipped_openai["privacy"]["raw_request_bodies_included"])
        self.assertFalse(skipped_openai["privacy"]["cache_keys_included"])
        self.assertFalse(skipped_openai["privacy"]["provider_bodies_included"])
        self.assertFalse(skipped_openai["privacy"]["request_ids_included"])
        self.assertFalse(skipped_openai["privacy"]["session_ids_included"])
        self.assertFalse(skipped_openai["privacy"]["individual_candidate_ids_included"])
        skipped_actions = {item["value"] for item in skipped_openai["next_action_breakdown"]}
        self.assertIn("add-invalidation-evidence", skipped_actions)
        self.assertIn("wait-for-streaming-replay-support", skipped_actions)
        self.assertIn("unsupported-endpoint", skipped_actions)
        skipped_blockers = {item["value"] for item in skipped_openai["blocker_breakdown"]}
        self.assertIn("invalidation-evidence-missing", skipped_blockers)
        self.assertIn("tools-present", skipped_blockers)
        self.assertIn("unsafe-tool-calls-without-invalidation", skipped_blockers)
        self.assertIn("streaming-replay-not-supported", skipped_blockers)
        tool_blocker_row = next(row for row in skipped_openai["cohorts"] if row["has_tools"])
        self.assertEqual(tool_blocker_row["sample_count"], 1)
        self.assertEqual(tool_blocker_row["next_action"], "add-invalidation-evidence")
        per_blocker_actions = {
            item["blocker_code"]: item["next_action"]
            for item in tool_blocker_row["blocker_actions"]
        }
        self.assertEqual(per_blocker_actions["tools-present"], "keep-tool-cache-disabled")
        self.assertEqual(
            per_blocker_actions["unsafe-tool-calls-without-invalidation"],
            "add-invalidation-evidence",
        )
        rendered_skipped_openai = json.dumps(skipped_openai, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered_skipped_openai)

        self.assertEqual(
            tool_replay_evidence["schema"],
            "tokenclaw.request_shape_tool_cache_replay_evidence.v1",
        )
        self.assertEqual(tool_replay_evidence["status"], "ranked")
        self.assertTrue(tool_replay_evidence["read_only"])
        self.assertEqual(tool_replay_evidence["summary"]["tool_cache_replay_evidence_cohort_count"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["tools_present_rows"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["tools_present_replay_evidence_rows"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["generic_tools_present_blocker_reduced_rows"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["unsafe_tool_call_blocker_rows"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay_evidence["summary"]["cache_entries_written"], 0)
        self.assertFalse(tool_replay_evidence["summary"]["policy_files_written"])
        self.assertEqual(tool_replay_evidence["summary"]["cache_apply_action_count"], 0)
        self.assertTrue(tool_replay_evidence["acceptance"]["has_ranked_tool_cache_replay_evidence"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reports_tools_present_replay_evidence"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reduces_generic_tools_present_blocker"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reports_dependency_evidence_decisions"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reports_dependency_evidence_burndown"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(tool_replay_evidence["acceptance"]["reports_narrow_no_safe_invalidation_reason"])
        self.assertTrue(tool_replay_evidence["acceptance"]["distinguishes_missing_stable_and_stale_dependency_evidence"])
        self.assertTrue(tool_replay_evidence["acceptance"]["unsafe_or_missing_dependency_keeps_tool_replay_blocked"])
        self.assertTrue(tool_replay_evidence["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(tool_replay_evidence["acceptance"]["tool_and_streaming_replay_remain_disabled"])
        dependency_coverage = tool_replay_evidence["dependency_fingerprint_coverage"]
        self.assertEqual(
            dependency_coverage["schema"],
            "tokenclaw.request_shape_tool_cache_dependency_fingerprint_coverage.v1",
        )
        self.assertEqual(dependency_coverage["coverage_decision"], "missing-stable-coverage")
        self.assertEqual(dependency_coverage["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(dependency_coverage["summary"]["stable_dependency_evidence_rows"], 0)
        self.assertEqual(dependency_coverage["summary"]["file_dependency_fingerprint_missing_rows"], 1)
        self.assertEqual(dependency_coverage["summary"]["safe_invalidation_evidence_rows"], 0)
        self.assertEqual(dependency_coverage["next_action"], "collect-file-invalidation-evidence")
        self.assertEqual(dependency_coverage["no_apply_guarantee"]["cache_entries_written"], 0)
        self.assertFalse(dependency_coverage["no_apply_guarantee"]["tool_cache_replay_enabled"])
        self.assertFalse(dependency_coverage["no_apply_guarantee"]["streaming_replay_enabled"])
        self.assertTrue(dependency_coverage["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(dependency_coverage["acceptance"]["reports_narrow_no_safe_invalidation_reason"])
        self.assertTrue(dependency_coverage["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(dependency_coverage["privacy"]["metadata_only"])
        self.assertFalse(dependency_coverage["privacy"]["file_paths_included"])
        tool_classification = tool_replay_evidence["dependency_evidence_classification"]
        self.assertEqual(
            set(tool_classification["supported_evidence_classes"]),
            {
                "missing-dependency-evidence",
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
            },
        )
        self.assertTrue(tool_classification["supports_four_way_dependency_evidence_split"])
        self.assertTrue(tool_classification["supports_five_way_dependency_evidence_split"])
        self.assertEqual(tool_classification["observed_evidence_classes"], ["missing-dependency-evidence"])
        self.assertEqual(tool_classification["observed_evidence_class_count"], 1)
        self.assertEqual(
            set(tool_classification["missing_observed_evidence_classes"]),
            {
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
            },
        )
        self.assertTrue(tool_classification["all_rows_classified_into_supported_evidence_classes"])
        self.assertTrue(
            tool_replay_evidence["acceptance"][
                "distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence"
            ]
        )
        self.assertFalse(tool_classification["tool_cache_replay_enabled"])
        self.assertFalse(tool_classification["streaming_replay_enabled"])
        self.assertTrue(tool_replay_evidence["privacy"]["metadata_only"])
        self.assertTrue(tool_replay_evidence["privacy"]["aggregate_only"])
        tool_replay_row = tool_replay_evidence["cohorts"][0]
        self.assertTrue(tool_replay_row["tools_present_replay_evidence"])
        self.assertTrue(tool_replay_row["generic_tools_present_blocker_reduced"])
        self.assertEqual(tool_replay_row["evidence_state"], "blocked-missing-dependency-evidence")
        self.assertEqual(tool_replay_row["next_action"], "collect-file-invalidation-evidence")
        self.assertEqual(tool_replay_row["dependency_evidence_decision"]["decision"], "missing-dependency-evidence")
        self.assertEqual(tool_replay_row["dependency_evidence_decision"]["evidence_class"], "missing-dependency-evidence")
        self.assertFalse(tool_replay_row["tool_cache_replay_enabled"])
        tool_burndown = tool_replay_evidence["dependency_evidence_burndown"][0]
        self.assertEqual(tool_burndown["dependency_evidence_class"], "missing-dependency-evidence")
        self.assertEqual(tool_burndown["row_count"], 1)
        self.assertEqual(tool_burndown["next_action"], "collect-file-invalidation-evidence")
        self.assertEqual(tool_burndown["top_blocker_reason"], "invalidation-evidence-missing")
        self.assertEqual(tool_burndown["reason_breakdown"], [{"value": "invalidation-evidence-missing", "count": 1}])
        self.assertFalse(tool_burndown["tool_cache_replay_enabled"])
        self.assertFalse(tool_burndown["streaming_replay_enabled"])
        self.assertEqual(tool_burndown["cache_entries_written"], 0)
        self.assertFalse(tool_burndown["policy_files_written"])
        rendered_tool_replay = json.dumps(tool_replay_evidence, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered_tool_replay)

        self.assertEqual(invalidation_evidence["schema"], "tokenclaw.request_shape_cache_invalidation_evidence.v1")
        self.assertEqual(invalidation_evidence["status"], "ranked")
        self.assertTrue(invalidation_evidence["read_only"])
        self.assertEqual(invalidation_evidence["summary"]["ranked_blocker_cohort_count"], 3)
        self.assertEqual(invalidation_evidence["summary"]["policy_files_written"], False)
        self.assertEqual(invalidation_evidence["summary"]["cache_entries_written"], 0)
        self.assertEqual(invalidation_evidence["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertGreaterEqual(invalidation_evidence["summary"]["dependency_evidence_decision_count"], 2)
        self.assertTrue(invalidation_evidence["acceptance"]["has_ranked_blocker_cohorts"])
        self.assertTrue(invalidation_evidence["acceptance"]["has_next_action"])
        self.assertTrue(invalidation_evidence["acceptance"]["has_local_file_backed_policy_compatibility"])
        self.assertTrue(invalidation_evidence["acceptance"]["tool_cohorts_require_invalidation_evidence"])
        self.assertTrue(invalidation_evidence["acceptance"]["tool_and_streaming_replay_remain_disabled"])
        self.assertTrue(invalidation_evidence["acceptance"]["reports_dependency_evidence_decisions"])
        self.assertTrue(invalidation_evidence["acceptance"]["reports_dependency_evidence_burndown"])
        self.assertTrue(
            invalidation_evidence["acceptance"]["distinguishes_missing_stable_and_stale_dependency_evidence"]
        )
        self.assertTrue(invalidation_evidence["acceptance"]["stale_or_missing_dependency_evidence_keeps_replay_blocked"])
        self.assertTrue(invalidation_evidence["acceptance"]["no_cache_entries_written"])
        self.assertFalse(invalidation_evidence["acceptance"]["policy_files_written"])
        invalidation_classification = invalidation_evidence["dependency_evidence_classification"]
        self.assertEqual(
            set(invalidation_classification["supported_evidence_classes"]),
            {
                "missing-dependency-evidence",
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
            },
        )
        self.assertTrue(invalidation_classification["supports_four_way_dependency_evidence_split"])
        self.assertTrue(invalidation_classification["supports_five_way_dependency_evidence_split"])
        self.assertIn("missing-dependency-evidence", invalidation_classification["observed_evidence_classes"])
        self.assertTrue(invalidation_classification["all_rows_classified_into_supported_evidence_classes"])
        self.assertTrue(
            invalidation_evidence["acceptance"][
                "distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence"
            ]
        )
        self.assertFalse(invalidation_classification["tool_cache_replay_enabled"])
        self.assertFalse(invalidation_classification["streaming_replay_enabled"])
        self.assertTrue(invalidation_evidence["privacy"]["metadata_only"])
        self.assertTrue(invalidation_evidence["privacy"]["aggregate_only"])
        self.assertFalse(invalidation_evidence["privacy"]["tool_payloads_included"])
        self.assertFalse(invalidation_evidence["privacy"]["file_paths_included"])
        self.assertFalse(invalidation_evidence["privacy"]["cache_keys_included"])
        self.assertFalse(invalidation_evidence["privacy"]["request_fingerprints_included"])
        policy_compat = invalidation_evidence["local_file_backed_policy_compatibility"]
        self.assertTrue(policy_compat["compatible"])
        self.assertEqual(policy_compat["policy_source"], "local-file-backed")
        self.assertEqual(policy_compat["rule_file"], "cache_rules.yaml")
        self.assertEqual(policy_compat["policy_section"], "cache")
        self.assertFalse(policy_compat["tool_call_cache_enabled"])
        self.assertFalse(policy_compat["streaming_replay_enabled"])
        evidence_actions = {item["value"] for item in invalidation_evidence["next_action_breakdown"]}
        self.assertIn("collect-file-invalidation-evidence", evidence_actions)
        self.assertIn("stage-streaming-replay-buffer-fixture", evidence_actions)
        self.assertIn("exact-non-tool-only", evidence_actions)
        tool_evidence = next(row for row in invalidation_evidence["cohorts"] if row["has_tools"])
        self.assertEqual(tool_evidence["next_action"], "collect-file-invalidation-evidence")
        self.assertEqual(tool_evidence["dependency_evidence_status"], "missing")
        self.assertEqual(tool_evidence["dependency_evidence_decision"]["decision"], "missing-dependency-evidence")
        self.assertEqual(tool_evidence["dependency_evidence_decision"]["evidence_class"], "missing-dependency-evidence")
        self.assertEqual(tool_evidence["dependency_evidence_decision"]["reason"], "invalidation-evidence-missing")
        invalidation_burndown = {
            row["dependency_evidence_class"]: row
            for row in invalidation_evidence["dependency_evidence_burndown"]
        }
        self.assertIn("missing-dependency-evidence", invalidation_burndown)
        self.assertEqual(invalidation_burndown["missing-dependency-evidence"]["row_count"], 1)
        self.assertFalse(invalidation_burndown["missing-dependency-evidence"]["tool_cache_replay_enabled"])
        self.assertIn("keep-tool-cache-blocked", tool_evidence["secondary_next_actions"])
        self.assertTrue(tool_evidence["requires_explicit_invalidation_safety_evidence"])
        self.assertFalse(tool_evidence["safe_invalidation_evidence"])
        self.assertFalse(tool_evidence["tool_cache_replay_enabled"])
        self.assertFalse(tool_evidence["streaming_replay_enabled"])
        streaming_evidence = next(row for row in invalidation_evidence["cohorts"] if row["stream"])
        self.assertEqual(streaming_evidence["next_action"], "stage-streaming-replay-buffer-fixture")
        self.assertFalse(streaming_evidence["tool_cache_replay_enabled"])
        self.assertFalse(streaming_evidence["streaming_replay_enabled"])

        self.assertEqual(classification["schema"], "tokenclaw.request_shape_cache_replay_blocker_classification.v1")
        self.assertEqual(classification["status"], "classified")
        self.assertEqual(classification["summary"]["skipped_cohort_count"], 3)
        classes = {item["value"]: item["count"] for item in classification["class_breakdown"]}
        self.assertGreater(classes["collect-invalidation-evidence"], 0)
        self.assertGreater(classes["keep-tool-cache-disabled"], 0)
        self.assertGreater(classes["streaming-replay-support-needed"], 0)
        self.assertGreater(classes["insufficient-repeat-evidence"], 0)
        self.assertGreater(classes["unsupported-safety-shape"], 0)
        next_actions = {item["value"] for item in classification["next_action_breakdown"]}
        self.assertIn("collect-cache-invalidation-evidence", next_actions)
        self.assertIn("keep-tool-cache-disabled", next_actions)
        self.assertIn("design-streaming-cache-replay-support", next_actions)
        self.assertIn("collect-more-repeat-evidence", next_actions)
        self.assertIn("keep-cache-replay-noop", next_actions)
        self.assertEqual(classification["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(classification["summary"]["unsafe_cache_apply_action_count"], 0)
        self.assertTrue(classification["acceptance"]["has_tool_blocker_class"])
        self.assertTrue(classification["acceptance"]["has_invalidation_evidence_class"])
        self.assertTrue(classification["acceptance"]["has_streaming_support_class"])
        self.assertTrue(classification["acceptance"]["has_insufficient_repeat_class"])
        self.assertTrue(classification["acceptance"]["has_unsupported_safety_shape_class"])
        self.assertTrue(classification["acceptance"]["no_cache_apply_without_invalidation_safety_evidence"])
        self.assertTrue(classification["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(classification["privacy"]["metadata_only"])
        self.assertTrue(classification["privacy"]["aggregate_only"])
        self.assertFalse(classification["privacy"]["cache_keys_included"])
        self.assertFalse(classification["privacy"]["request_fingerprints_included"])
        self.assertFalse(classification["privacy"]["individual_candidate_ids_included"])
        rendered_classification = json.dumps(classification, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered_classification)
        rendered_invalidation = json.dumps(invalidation_evidence, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered_invalidation)

    def test_openai_tool_cache_blockers_record_sanitized_dependency_states_without_apply(self) -> None:
        stable_audit = self._dependency_audit(safe=True, fingerprint_available=True)
        stale_audit = self._dependency_audit(
            reason="dependency-changed",
            safe=False,
            fingerprint_available=True,
        )
        unsafe_audit = self._dependency_audit(
            reason="unsafe-tool-calls-without-invalidation",
            safe=False,
            fingerprint_available=True,
        )
        unknown_audit = self._dependency_audit(
            reason=None,
            safe=False,
            fingerprint_available=True,
        )
        shared = {
            "provider": "openai",
            "path": "/v1/responses",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "requested_model_family": "gpt-5",
            "routed_model_family": "gpt-5",
            "category": "tool-light",
            "workflow_phase": "tool-light",
            "stream": 0,
            "has_tools": True,
            "cache_status": "skipped",
            "cache_reason": "tools-disabled",
            "cost": 0.04,
            "baseline": 0.04,
        }
        self._log_call(
            **shared,
            text_chars=6_000,
            cache_extra={
                "file_dependency_audit": stable_audit,
                "file_dependency_fingerprint": {
                    "schema": "tokenclaw.cache_file_dependency_fingerprint.v1",
                    "fingerprint_available": True,
                    "fingerprint_sha256": "sha256:raw-stable-dependency-fingerprint-must-not-leak",
                    "paths_included": False,
                },
                "file_dependency_fingerprint_available": True,
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": True,
            },
        )
        self._log_call(
            **shared,
            text_chars=12_000,
            cache_extra={
                "file_dependency_audit": stale_audit,
                "file_dependency_fingerprint": {
                    "schema": "tokenclaw.cache_file_dependency_fingerprint.v1",
                    "fingerprint_available": True,
                    "fingerprint_sha256": "sha256:raw-stale-dependency-fingerprint-must-not-leak",
                    "paths_included": False,
                },
                "file_dependency_fingerprint_available": True,
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
            },
        )
        self._log_call(
            **shared,
            text_chars=18_000,
            cache_extra={
                "file_dependency_audit": unsafe_audit,
                "file_dependency_fingerprint": {
                    "schema": "tokenclaw.cache_file_dependency_fingerprint.v1",
                    "fingerprint_available": True,
                    "fingerprint_sha256": "sha256:raw-unsafe-dependency-fingerprint-must-not-leak",
                    "paths_included": False,
                },
                "file_dependency_fingerprint_available": True,
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
            },
        )
        self._log_call(
            **shared,
            text_chars=24_000,
            cache_extra={
                "file_dependency_audit": unknown_audit,
                "file_dependency_fingerprint": {
                    "schema": "tokenclaw.cache_file_dependency_fingerprint.v1",
                    "fingerprint_available": True,
                    "fingerprint_sha256": "sha256:raw-unknown-dependency-fingerprint-must-not-leak",
                    "paths_included": False,
                },
                "file_dependency_fingerprint_available": True,
                "file_dependency_evidence_available": True,
                "safe_invalidation_evidence": False,
            },
        )
        self._log_call(**shared, text_chars=30_000)

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="dependency-states")
        skipped = report["cache_replayability_dry_run"]["skipped_openai_blockers"]
        invalidation = report["cache_replayability_dry_run"]["cache_invalidation_evidence"]
        tool_replay = report["cache_replayability_dry_run"]["tool_replay_evidence"]

        self.assertEqual(skipped["summary"]["cache_apply_action_count"], 0)
        self.assertFalse(skipped["summary"]["policy_files_written"])
        self.assertTrue(skipped["acceptance"]["emits_no_cache_apply_actions"])
        by_status = {row["file_dependency_status"]: row for row in skipped["cohorts"]}
        self.assertIn("stable", by_status)
        self.assertIn("invalidated", by_status)
        self.assertIn("unsafe", by_status)
        self.assertIn("unknown", by_status)
        self.assertIn("missing", by_status)

        stable_row = by_status["stable"]
        self.assertEqual(stable_row["next_action"], "rank-safe-tool-cache-replay-readiness")
        self.assertTrue(stable_row["safe_invalidation_evidence"])
        self.assertTrue(stable_row["file_dependency_fingerprint_available"])
        self.assertIn("safe-invalidation-evidence-present", stable_row["blocker_codes"])
        self.assertNotIn("invalidation-evidence-missing", stable_row["blocker_codes"])
        self.assertFalse(stable_row["tool_cache_replay_enabled"])

        stale_row = by_status["invalidated"]
        self.assertEqual(stale_row["next_action"], "refresh-invalidation-evidence")
        self.assertIn("stale-dependency-evidence", stale_row["blocker_codes"])
        self.assertFalse(stale_row["safe_invalidation_evidence"])

        unsafe_row = by_status["unsafe"]
        self.assertEqual(unsafe_row["next_action"], "add-invalidation-evidence")
        self.assertIn("unsafe-dependency-evidence", unsafe_row["blocker_codes"])
        self.assertIn("unsafe-tool-calls-without-invalidation", unsafe_row["blocker_codes"])
        self.assertFalse(unsafe_row["safe_invalidation_evidence"])

        missing_row = by_status["missing"]
        self.assertEqual(missing_row["next_action"], "add-invalidation-evidence")
        self.assertIn("invalidation-evidence-missing", missing_row["blocker_codes"])

        unknown_row = by_status["unknown"]
        self.assertEqual(unknown_row["next_action"], "add-invalidation-evidence")
        self.assertIn("dependency-evidence-unknown", unknown_row["blocker_codes"])
        self.assertFalse(unknown_row["tool_cache_replay_enabled"])

        invalidation_by_status = {row["file_dependency_status"]: row for row in invalidation["cohorts"]}
        self.assertEqual(invalidation["summary"]["stable_dependency_evidence_rows"], 1)
        self.assertEqual(invalidation["summary"]["stale_dependency_evidence_rows"], 1)
        self.assertEqual(invalidation["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(invalidation["summary"]["unsafe_dependency_evidence_rows"], 1)
        self.assertEqual(invalidation["summary"]["unknown_dependency_evidence_rows"], 1)
        decision_breakdown = {
            row["value"]: row["count"]
            for row in invalidation["dependency_evidence_decision_breakdown"]
        }
        self.assertEqual(decision_breakdown["stable-dependency-evidence"], 1)
        self.assertEqual(decision_breakdown["stale-risk-blocker"], 1)
        self.assertEqual(decision_breakdown["missing-dependency-evidence"], 1)
        self.assertEqual(decision_breakdown["unsafe-dependency-evidence"], 1)
        self.assertEqual(decision_breakdown["unknown-dependency-evidence"], 1)
        self.assertTrue(invalidation["acceptance"]["reports_dependency_evidence_decisions"])
        self.assertTrue(invalidation["acceptance"]["reports_dependency_evidence_burndown"])
        self.assertTrue(invalidation["acceptance"]["distinguishes_missing_stable_and_stale_dependency_evidence"])
        self.assertTrue(invalidation["acceptance"]["distinguishes_missing_stable_stale_and_unsafe_dependency_evidence"])
        self.assertTrue(invalidation["acceptance"]["distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence"])
        self.assertTrue(invalidation["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(invalidation["acceptance"]["stale_or_missing_dependency_evidence_keeps_replay_blocked"])
        self.assertTrue(invalidation["acceptance"]["unsafe_dependency_evidence_keeps_replay_blocked"])
        invalidation_burndown = {
            row["dependency_evidence_class"]: row
            for row in invalidation["dependency_evidence_burndown"]
        }
        self.assertEqual(
            set(invalidation_burndown),
            {
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "missing-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
            },
        )
        self.assertTrue(all(row["row_count"] == 1 for row in invalidation_burndown.values()))
        self.assertFalse(any(row["tool_cache_replay_enabled"] for row in invalidation_burndown.values()))
        self.assertFalse(any(row["streaming_replay_enabled"] for row in invalidation_burndown.values()))
        self.assertTrue(all(row["cache_entries_written"] == 0 for row in invalidation_burndown.values()))
        self.assertFalse(any(row["policy_files_written"] for row in invalidation_burndown.values()))
        self.assertEqual(
            invalidation_by_status["stable"]["next_action"],
            "rank-safe-tool-cache-replay-readiness",
        )
        self.assertEqual(
            invalidation_by_status["stable"]["dependency_evidence_decision"]["decision"],
            "stable-dependency-evidence",
        )
        self.assertEqual(
            invalidation_by_status["invalidated"]["next_action"],
            "refresh-file-invalidation-evidence",
        )
        self.assertEqual(
            invalidation_by_status["invalidated"]["dependency_evidence_decision"]["decision"],
            "stale-risk-blocker",
        )
        self.assertEqual(
            invalidation_by_status["unsafe"]["dependency_evidence_decision"]["decision"],
            "unsafe-dependency-evidence",
        )
        self.assertEqual(
            invalidation_by_status["unknown"]["dependency_evidence_decision"]["decision"],
            "unknown-dependency-evidence",
        )
        self.assertEqual(
            invalidation_by_status["missing"]["dependency_evidence_decision"]["decision"],
            "missing-dependency-evidence",
        )
        self.assertTrue(invalidation_by_status["stable"]["local_dependency_fingerprint"]["fingerprint_available"])
        self.assertFalse(invalidation_by_status["stable"]["local_dependency_fingerprint"]["fingerprint_value_included"])
        self.assertTrue(invalidation_by_status["stable"]["local_dependency_fingerprint"]["stable_dependency_snapshot"])
        tool_replay_by_status = {row["file_dependency_status"]: row for row in tool_replay["cohorts"]}
        self.assertEqual(tool_replay["summary"]["stable_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["stale_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["unsafe_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["unknown_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["tools_present_replay_evidence_rows"], 5)
        self.assertTrue(tool_replay["acceptance"]["reports_tools_present_replay_evidence"])
        self.assertTrue(tool_replay["acceptance"]["reduces_generic_tools_present_blocker"])
        self.assertTrue(tool_replay["acceptance"]["reports_dependency_evidence_burndown"])
        self.assertTrue(tool_replay["acceptance"]["distinguishes_missing_stable_and_stale_dependency_evidence"])
        self.assertTrue(tool_replay["acceptance"]["distinguishes_missing_stable_stale_and_unsafe_dependency_evidence"])
        self.assertTrue(tool_replay["acceptance"]["distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence"])
        self.assertTrue(tool_replay["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(tool_replay["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(tool_replay["acceptance"]["reports_narrow_no_safe_invalidation_reason"])
        coverage = tool_replay["dependency_fingerprint_coverage"]
        self.assertEqual(coverage["schema"], "tokenclaw.request_shape_tool_cache_dependency_fingerprint_coverage.v1")
        self.assertEqual(coverage["coverage_decision"], "stable-coverage-observed")
        self.assertEqual(coverage["next_action"], "rank-safe-tool-cache-replay-readiness")
        self.assertEqual(coverage["summary"]["stable_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["stale_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["unsafe_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["unknown_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["file_dependency_fingerprint_available_rows"], 4)
        self.assertEqual(coverage["summary"]["file_dependency_fingerprint_missing_rows"], 1)
        self.assertEqual(coverage["summary"]["safe_invalidation_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["cache_apply_action_count"], 0)
        self.assertFalse(coverage["summary"]["tool_cache_replay_enabled"])
        self.assertFalse(coverage["summary"]["streaming_replay_enabled"])
        self.assertTrue(coverage["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(coverage["acceptance"]["emits_no_cache_apply_actions"])
        self.assertFalse(any(row["tool_cache_replay_enabled"] for row in coverage["cohorts"]))
        tool_burndown = {
            row["dependency_evidence_class"]: row
            for row in tool_replay["dependency_evidence_burndown"]
        }
        self.assertEqual(
            set(tool_burndown),
            {
                "stable-dependency-evidence",
                "stale-dependency-evidence",
                "missing-dependency-evidence",
                "unsafe-dependency-evidence",
                "unknown-dependency-evidence",
            },
        )
        self.assertTrue(all(row["row_count"] == 1 for row in tool_burndown.values()))
        self.assertFalse(any(row["tool_cache_replay_enabled"] for row in tool_burndown.values()))
        self.assertFalse(any(row["streaming_replay_enabled"] for row in tool_burndown.values()))
        self.assertTrue(all(row["cache_entries_written"] == 0 for row in tool_burndown.values()))
        self.assertFalse(any(row["policy_files_written"] for row in tool_burndown.values()))
        self.assertEqual(
            tool_replay_by_status["stable"]["evidence_state"],
            "dependency-gated-review-ready",
        )
        self.assertEqual(
            tool_replay_by_status["stable"]["next_action"],
            "rank-safe-tool-cache-replay-readiness",
        )
        self.assertEqual(
            tool_replay_by_status["invalidated"]["evidence_state"],
            "blocked-stale-dependency-evidence",
        )
        self.assertEqual(
            tool_replay_by_status["unsafe"]["evidence_state"],
            "blocked-unsafe-dependency-evidence",
        )
        self.assertEqual(
            tool_replay_by_status["unknown"]["evidence_state"],
            "blocked-unknown-dependency-evidence",
        )
        self.assertEqual(
            tool_replay_by_status["missing"]["evidence_state"],
            "blocked-missing-dependency-evidence",
        )
        self.assertTrue(tool_replay_by_status["stable"]["local_dependency_fingerprint"]["fingerprint_available"])
        self.assertFalse(tool_replay_by_status["stable"]["local_dependency_fingerprint"]["fingerprint_value_included"])
        self.assertFalse(any(row["tool_cache_replay_enabled"] for row in tool_replay["cohorts"]))

        rendered = json.dumps([skipped, invalidation, tool_replay], sort_keys=True)
        self.assertNotIn("raw-stable-dependency-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-stale-dependency-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-unsafe-dependency-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-unknown-dependency-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-request-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)
        self.assertFalse(skipped["privacy"]["cache_keys_included"])
        self.assertFalse(skipped["privacy"]["request_ids_included"])
        self.assertFalse(skipped["privacy"]["session_ids_included"])

    def test_openai_tool_cache_dependency_capture_fixtures_feed_replay_evidence_without_apply(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
            os.chdir(tmp_path)
            try:
                stable_body = {
                    "input": [
                        {
                            "role": "user",
                            "content": "Tool read pyproject.toml and returned deterministic local metadata.",
                        }
                    ]
                }
                cap_trimmed_body = {
                    "input": [
                        {
                            "role": "user",
                            "content": " ".join(f"pyproject.toml:{line}" for line in range(8)),
                        }
                    ]
                }
                no_path_body = {
                    "input": [
                        {
                            "role": "user",
                            "content": "Tool returned status with no local file references.",
                        }
                    ]
                }
                stable_meta = cache_module.attach_file_dependency_cache_meta(
                    {"status": "skipped", "reason": "tools-disabled"},
                    snapshots=cache_module.cache_file_dependency_snapshots(stable_body),
                    audit=cache_module.cache_file_dependency_audit(stable_body),
                    blocker_reasons=["tool-call-cache-disabled"],
                )
                cap_trimmed_audit = cache_module.cache_file_dependency_audit(cap_trimmed_body, max_paths=1)
                cap_trimmed_meta = cache_module.attach_file_dependency_cache_meta(
                    {"status": "skipped", "reason": "tools-disabled"},
                    snapshots=cache_module.cache_file_dependency_snapshots(cap_trimmed_body, max_paths=1),
                    audit=cap_trimmed_audit,
                    blocker_reasons=["tool-call-cache-disabled"],
                )
                no_path_meta = cache_module.attach_file_dependency_cache_meta(
                    {"status": "skipped", "reason": "tools-disabled"},
                    snapshots=cache_module.cache_file_dependency_snapshots(no_path_body),
                    audit=cache_module.cache_file_dependency_audit(no_path_body),
                    blocker_reasons=["tool-call-cache-disabled"],
                )
            finally:
                os.chdir(old_cwd)

        stale_meta = {
            "status": "skipped",
            "reason": "tools-disabled",
            "file_dependency_audit": self._dependency_audit(
                reason="dependency-changed",
                safe=False,
                fingerprint_available=True,
            ),
            "file_dependency_fingerprint_available": True,
            "file_dependency_evidence_available": True,
            "safe_invalidation_evidence": False,
        }
        unsafe_meta = {
            "status": "skipped",
            "reason": "tools-disabled",
            "file_dependency_audit": self._dependency_audit(
                reason="unsafe-tool-calls-without-invalidation",
                safe=False,
                fingerprint_available=True,
            ),
            "file_dependency_fingerprint_available": True,
            "file_dependency_evidence_available": True,
            "safe_invalidation_evidence": False,
        }
        unknown_meta = {
            "status": "skipped",
            "reason": "tools-disabled",
            "file_dependency_audit": self._dependency_audit(
                reason=None,
                safe=False,
                fingerprint_available=True,
            ),
            "file_dependency_fingerprint_available": True,
            "file_dependency_evidence_available": True,
            "safe_invalidation_evidence": False,
        }
        shared = {
            "provider": "openai",
            "path": "/v1/responses",
            "source_surface": "openai_responses",
            "endpoint": "responses",
            "requested_model": "gpt-5.4",
            "routed_model": "gpt-5.4",
            "requested_model_family": "gpt-5",
            "routed_model_family": "gpt-5",
            "category": "tool-light",
            "workflow_phase": "tool-light",
            "stream": 0,
            "has_tools": True,
            "cache_status": "skipped",
            "cache_reason": "tools-disabled",
            "cost": 0.04,
            "baseline": 0.04,
        }
        for text_chars, cache_extra in (
            (6_000, stable_meta),
            (12_000, cap_trimmed_meta),
            (18_000, stale_meta),
            (24_000, unsafe_meta),
            (30_000, unknown_meta),
            (36_000, no_path_meta),
        ):
            self._log_call(**shared, text_chars=text_chars, cache_extra=cache_extra)

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="captured-dependencies")
        tool_replay = report["cache_replayability_dry_run"]["tool_replay_evidence"]
        rows_by_status = {row["file_dependency_status"]: row for row in tool_replay["cohorts"]}

        self.assertEqual(tool_replay["schema"], "tokenclaw.request_shape_tool_cache_replay_evidence.v1")
        self.assertEqual(tool_replay["summary"]["stable_dependency_evidence_rows"], 2)
        self.assertEqual(tool_replay["summary"]["stale_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["unsafe_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["unknown_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(tool_replay["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(tool_replay["summary"]["cache_entries_written"], 0)
        self.assertFalse(tool_replay["summary"]["policy_files_written"])
        self.assertTrue(tool_replay["acceptance"]["distinguishes_stable_stale_unsafe_unknown_and_missing_dependency_evidence"])
        self.assertTrue(tool_replay["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(tool_replay["acceptance"]["stable_dependency_evidence_does_not_activate_replay"])
        self.assertTrue(tool_replay["acceptance"]["unsafe_or_missing_dependency_keeps_tool_replay_blocked"])
        self.assertFalse(any(row["tool_cache_replay_enabled"] for row in tool_replay["cohorts"]))
        self.assertFalse(any(row["streaming_replay_enabled"] for row in tool_replay["cohorts"]))
        self.assertTrue(rows_by_status["stable"]["local_dependency_fingerprint"]["fingerprint_available"])
        stable_audits = [
            row["local_dependency_fingerprint"]
            for row in tool_replay["cohorts"]
            if row["file_dependency_status"] == "stable"
        ]
        self.assertTrue(any(audit["candidate_path_count_bucket"] == "1" for audit in stable_audits))
        self.assertEqual(rows_by_status["missing"]["dependency_evidence_decision"]["decision"], "missing-dependency-evidence")

        rendered = json.dumps(tool_replay, sort_keys=True)
        self.assertNotIn("pyproject.toml", rendered)
        self.assertNotIn(str(tmp_path), rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)
        self.assertFalse(tool_replay["privacy"]["file_paths_included"])
        self.assertFalse(tool_replay["privacy"]["cache_keys_included"])

    def test_tool_cache_dependency_coverage_records_narrow_missing_reason_after_capture(self) -> None:
        audit = self._dependency_audit(safe=False, fingerprint_available=False)
        audit["raw_candidate_path_count_bucket"] = "6_20"
        audit["candidate_path_count_bucket"] = "0"
        audit["distinct_candidate_path_count_bucket"] = "0"
        audit["snapshot_count_bucket"] = "0"
        audit["dependency_capture_reason"] = "complete"
        audit["invalidation_reason"] = None

        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4",
            routed_model="gpt-5.4",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            text_chars=8_000,
            cost=0.02,
            baseline=0.02,
            cache_extra={
                "file_dependency_audit": audit,
                "file_dependency_fingerprint_available": False,
                "file_dependency_evidence_available": False,
                "safe_invalidation_evidence": False,
            },
        )

        report = build_request_shape_rollups_report(
            self.store,
            limit=20,
            persist=False,
            run_id="dependency-coverage-missing",
        )
        dry_run = report["cache_replayability_dry_run"]
        tool_replay = dry_run["tool_replay_evidence"]
        coverage = tool_replay["dependency_fingerprint_coverage"]

        self.assertTrue(dry_run["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(dry_run["acceptance"]["reports_narrow_no_safe_invalidation_reason"])
        self.assertEqual(coverage["coverage_decision"], "missing-stable-coverage")
        self.assertEqual(coverage["next_action"], "collect-file-invalidation-evidence")
        self.assertEqual(coverage["summary"]["affected_rows"], 1)
        self.assertEqual(coverage["summary"]["stable_dependency_evidence_rows"], 0)
        self.assertEqual(coverage["summary"]["missing_dependency_evidence_rows"], 1)
        self.assertEqual(coverage["summary"]["file_dependency_fingerprint_missing_rows"], 1)
        self.assertEqual(coverage["summary"]["safe_invalidation_evidence_rows"], 0)
        self.assertEqual(coverage["summary"]["top_missing_or_blocked_reason"], "no-stable-file-dependency-snapshots")
        reasons = {row["value"]: row["count"] for row in coverage["missing_or_blocked_reason_breakdown"]}
        self.assertEqual(reasons["no-stable-file-dependency-snapshots"], 1)
        self.assertEqual(coverage["summary"]["cache_apply_action_count"], 0)
        self.assertEqual(coverage["no_apply_guarantee"]["cache_entries_written"], 0)
        self.assertFalse(coverage["no_apply_guarantee"]["tool_cache_replay_enabled"])
        self.assertFalse(coverage["no_apply_guarantee"]["streaming_replay_enabled"])
        self.assertTrue(coverage["acceptance"]["reports_dependency_fingerprint_coverage_after_capture"])
        self.assertTrue(coverage["acceptance"]["reports_narrow_no_safe_invalidation_reason"])
        self.assertTrue(coverage["acceptance"]["emits_no_cache_apply_actions"])
        self.assertTrue(coverage["privacy"]["metadata_only"])
        rendered = json.dumps(coverage, sort_keys=True)
        self.assertNotIn("raw-request-fingerprint-must-not-leak", rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)
        self.assertNotIn("/tmp/private/source.py", rendered)

    def test_cache_replayability_ranks_remaining_replay_ready_after_handled_policy(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.20, 0.25, 0.30, 0.35):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=12_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.40, 0.45, 0.50, 0.55, 0.60):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=40_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_rollups_report(
            self.store,
            limit=20,
            persist=False,
            run_id="remaining-cache-replay-cohorts",
        )
        dry_run = report["cache_replayability_dry_run"]

        self.assertEqual(dry_run["schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(dry_run["summary"]["replay_ready_cohort_count"], 3)
        self.assertEqual(dry_run["summary"]["remaining_replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["handled_replay_ready_cohort_count"], 2)
        self.assertEqual(dry_run["summary"]["remaining_projected_hits"], 4)
        self.assertGreater(dry_run["summary"]["remaining_projected_savings_usd"], 0)
        self.assertTrue(dry_run["acceptance"]["ranks_remaining_replay_ready_cohorts"])
        self.assertTrue(dry_run["acceptance"]["marks_already_handled_replay_ready_cohorts"])
        self.assertTrue(dry_run["acceptance"]["reports_remaining_projected_hits_and_savings"])

        remaining = dry_run["remaining_replay_ready_cohorts"][0]
        self.assertEqual(remaining["rank"], 1)
        self.assertEqual(remaining["remaining_rank"], 1)
        self.assertEqual(remaining["text_bucket"], "32k_128k_chars")
        self.assertEqual(remaining["token_bucket"], "8k_32k_tokens")
        self.assertFalse(remaining["handled_by_local_policy"])
        self.assertEqual(remaining["next_action"], "stage-cache-replay-canary")

        handled_shapes = {
            (row["text_bucket"], row["token_bucket"]): row
            for row in dry_run["handled_replay_ready_cohorts"]
        }
        self.assertIn(("2k_8k_chars", "500_2k_tokens"), handled_shapes)
        self.assertIn(("8k_32k_chars", "2k_8k_tokens"), handled_shapes)
        expected_handled_states = {
            ("2k_8k_chars", "500_2k_tokens"): "blocked-local-policy",
            ("8k_32k_chars", "2k_8k_tokens"): "active-local-policy",
        }
        for bucket, handled in handled_shapes.items():
            self.assertTrue(handled["handled_by_local_policy"])
            self.assertEqual(handled["next_action"], "already-handled-by-local-cache-policy")
            self.assertEqual(handled["handled_local_policy"]["handled_state"], expected_handled_states[bucket])
            self.assertEqual(handled["handled_local_policy"]["source_policy_file"], "cache_rules.yaml")
            self.assertFalse(handled["handled_local_policy"]["rule_ids_included"])
            self.assertFalse(handled["handled_local_policy"]["cohort_ids_included"])
        self.assertFalse(dry_run["handled_policy_summary"]["policy_paths_included"])

        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "local-openai-cache-replay-canary",
            "request-shape-cache-replay:",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replayability_gates_tiny_remaining_cohort_without_live_repeat(self) -> None:
        for cost in (0.001, 0.002, 0.001459):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=1_200,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_rollups_report(
            self.store,
            limit=20,
            persist=False,
            run_id="tiny-cache-replay-cohort",
            mark_handled_cache_replay_cohorts=False,
        )
        dry_run = report["cache_replayability_dry_run"]

        self.assertEqual(dry_run["summary"]["replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["remaining_replay_ready_cohort_count"], 0)
        self.assertEqual(dry_run["summary"]["gated_too_small_replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["gated_too_small_replay_ready_rows"], 3)
        self.assertEqual(dry_run["summary"]["gated_too_small_projected_hits"], 2)
        self.assertAlmostEqual(dry_run["summary"]["gated_too_small_projected_savings_usd"], 0.002973)
        self.assertTrue(dry_run["acceptance"]["gates_tiny_replay_ready_cohorts_without_live_repeat"])
        self.assertTrue(dry_run["acceptance"]["reports_live_repeat_and_savings_floor_fields"])

        cohort = next(row for row in dry_run["cohorts"] if row["readiness"] == "replay-ready")
        self.assertFalse(cohort["remaining_replay_ready"])
        self.assertEqual(cohort["next_action"], "no-op-too-small-without-live-repeat")
        self.assertEqual(cohort["projected_hits"], 2)
        self.assertAlmostEqual(cohort["projected_savings_usd"], 0.002973)
        gate = cohort["readiness_gate"]
        self.assertEqual(gate["gate_status"], "replay-ready-but-too-small")
        self.assertEqual(gate["next_action"], "no-op-too-small-without-live-repeat")
        self.assertFalse(gate["live_repeat_confirmed"])
        self.assertEqual(gate["live_repeat_cache_hit_count"], 0)
        self.assertEqual(gate["minimum_projected_savings_usd"], 0.01)
        self.assertFalse(gate["savings_floor_met"])

        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="tiny-cache-replay-cohort",
            mark_handled_cache_replay_cohorts=False,
        )
        self.assertEqual(stage["status"], "no-stageable-cohort")
        self.assertEqual(stage["eligible_stageable_cohort_count"], 0)
        self.assertEqual(stage["staged_canary_count"], 0)
        self.assertEqual(stage["next_action"], "no-op-too-small-without-live-repeat")
        self.assertEqual(stage["reason"], "too-small-without-live-repeat")
        self.assertFalse(stage["ok"])
        self.assertFalse(stage["acceptance"]["has_replay_ready_openai_responses_cohort"])

        rendered = json.dumps(dry_run, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertNotIn("raw-cache-key-must-not-leak", rendered)

    def test_cache_replayability_live_repeat_overrides_tiny_savings_gate(self) -> None:
        for index, cost in enumerate((0.001, 0.002, 0.001459)):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                cache_hit=1 if index == 0 else 0,
                text_chars=1_200,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_rollups_report(
            self.store,
            limit=20,
            persist=False,
            run_id="tiny-cache-replay-live-repeat",
            mark_handled_cache_replay_cohorts=False,
        )
        dry_run = report["cache_replayability_dry_run"]
        cohort = dry_run["remaining_replay_ready_cohorts"][0]

        self.assertEqual(dry_run["summary"]["remaining_replay_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["gated_too_small_replay_ready_cohort_count"], 0)
        self.assertEqual(dry_run["summary"]["live_repeat_confirmed_replay_ready_cohort_count"], 1)
        self.assertTrue(cohort["remaining_replay_ready"])
        self.assertEqual(cohort["next_action"], "stage-cache-replay-canary")
        self.assertEqual(cohort["readiness_gate"]["gate_status"], "live-repeat-confirmed")
        self.assertTrue(cohort["readiness_gate"]["live_repeat_confirmed"])
        self.assertEqual(cohort["readiness_gate"]["live_repeat_cache_hit_count"], 1)

    def test_cache_replayability_marks_short_completion_remaining_cohorts_handled_by_local_policy(self) -> None:
        for cost in (0.004, 0.006, 0.005):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="short-completion",
                workflow_phase="short-completion",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=1_200,
                actual_input_tokens=300,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.007, 0.009):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="short-completion",
                workflow_phase="short-completion",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=1_800,
                actual_input_tokens=700,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_rollups_report(
            self.store,
            limit=20,
            persist=False,
            run_id="remaining-short-completion-cache-replay",
        )
        dry_run = report["cache_replayability_dry_run"]

        self.assertEqual(dry_run["summary"]["replay_ready_cohort_count"], 2)
        self.assertEqual(dry_run["summary"]["remaining_replay_ready_cohort_count"], 0)
        self.assertEqual(dry_run["summary"]["handled_replay_ready_cohort_count"], 2)
        self.assertEqual(dry_run["summary"]["remaining_projected_hits"], 0)
        self.assertEqual(dry_run["summary"]["remaining_projected_savings_usd"], 0.0)

        handled_shapes = {
            (row["category"], row["workflow_phase"], row["text_bucket"], row["token_bucket"]): row
            for row in dry_run["handled_replay_ready_cohorts"]
        }
        self.assertIn(("short-completion", "short-completion", "lt_2k_chars", "lt_500_tokens"), handled_shapes)
        self.assertIn(("short-completion", "short-completion", "lt_2k_chars", "500_2k_tokens"), handled_shapes)
        for handled in handled_shapes.values():
            self.assertTrue(handled["handled_by_local_policy"])
            self.assertEqual(handled["next_action"], "already-handled-by-local-cache-policy")
            self.assertEqual(handled["handled_local_policy"]["handled_state"], "active-local-policy")
            self.assertEqual(handled["handled_local_policy"]["source_policy_file"], "cache_rules.yaml")
            self.assertEqual(handled["handled_local_policy"]["target_local_rule_file"], "cache_rules.yaml")
            self.assertFalse(handled["handled_local_policy"]["rule_ids_included"])
            self.assertFalse(handled["handled_local_policy"]["cohort_ids_included"])

        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="remaining-short-completion-cache-replay",
        )
        self.assertEqual(stage["status"], "no-stageable-cohort")
        self.assertEqual(stage["eligible_stageable_cohort_count"], 0)
        self.assertEqual(stage["staged_canary_count"], 0)

        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_canary_stage_report_targets_openai_responses_replay_ready_cohort(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="tool-light",
            workflow_phase="tool-light",
            stream=0,
            has_tools=True,
            cache_status="skipped",
            cache_reason="tools-disabled",
            text_chars=6_000,
            cost=0.02,
            baseline=0.02,
        )
        for cost in (0.015, 0.025):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=1,
                has_tools=False,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.04, 0.03):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="short-completion",
                workflow_phase="summary",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.012, 0.018):
            self._log_call(
                provider="openai",
                path="/v1/embeddings",
                source_surface="openai_embeddings",
                endpoint="embeddings",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            mark_handled_cache_replay_cohorts=False,
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_cache_replay_canary_stage.v1")
        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["staged_canary_count"], 2)
        self.assertEqual(report["eligible_stageable_cohort_count"], 2)
        self.assertEqual(len(report["stage_actions"]), 2)
        self.assertTrue(report["acceptance"]["stages_top_ranked_cohort"])
        self.assertTrue(report["acceptance"]["stages_all_remaining_exact_safe_replay_ready_cohorts"])
        self.assertTrue(report["acceptance"]["has_replay_ready_openai_responses_cohort"])
        self.assertTrue(report["acceptance"]["stages_remaining_unhandled_replay_ready_cohort"])
        self.assertTrue(report["acceptance"]["excludes_already_handled_local_policy_cohorts"])
        self.assertTrue(report["acceptance"]["has_rank"])
        self.assertTrue(report["acceptance"]["has_shape_buckets"])
        self.assertTrue(report["acceptance"]["has_target_cache_policy_metadata"])
        self.assertTrue(report["acceptance"]["has_projected_hits"])
        self.assertTrue(report["acceptance"]["has_projected_savings"])
        self.assertTrue(report["acceptance"]["writes_no_provider_bodies"])
        self.assertTrue(report["acceptance"]["writes_no_cache_entries"])
        self.assertTrue(report["acceptance"]["has_holdout_metadata"])
        self.assertTrue(report["acceptance"]["has_lifecycle_metadata"])
        self.assertTrue(report["acceptance"]["has_applied_and_holdout_eligibility"])
        self.assertTrue(report["acceptance"]["records_hit_miss_bypass_invalidation_and_stale_risk"])
        self.assertTrue(report["acceptance"]["preserves_tool_and_streaming_guards"])
        self.assertTrue(report["acceptance"]["stages_only_openai_responses_exact_safe_categories"])
        self.assertTrue(report["acceptance"]["tool_streaming_and_invalidation_missing_cohorts_skipped"])
        self.assertEqual(
            {item["conditions"]["category"] for item in report["stage_actions"]},
            {"chat", "short-completion"},
        )

        action = report["top_stage_action"]
        self.assertEqual(action["schema"], "tokenclaw.request_shape_cache_replay_canary_action.v1")
        self.assertEqual(action["action_type"], "stage-local-openai-cache-replay-canary")
        self.assertEqual(action["target_local_policy"], "cache_canary_policy")
        self.assertEqual(action["target_local_rule_file"], "cache_canary_policy.yaml")
        self.assertEqual(action["target_cache_policy"]["schema"], "tokenclaw.request_shape_cache_replay_target_policy.v1")
        self.assertEqual(action["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(action["target_cache_policy"]["target_local_rule_file"], "cache_canary_policy.yaml")
        self.assertFalse(action["target_cache_policy"]["rules_path_included"])
        self.assertEqual(action["rank"], 1)
        self.assertEqual(action["cohort_rank"], 1)
        self.assertEqual(action["shape"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(action["shape"]["token_bucket"], "500_2k_tokens")
        self.assertEqual(action["conditions"]["provider_family"], "openai")
        self.assertEqual(action["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(action["conditions"]["endpoint"], "responses")
        self.assertEqual(action["conditions"]["category"], "chat")
        self.assertEqual(action["conditions"]["workflow_phase"], "chat")
        self.assertEqual(action["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertFalse(action["conditions"]["stream"])
        self.assertFalse(action["conditions"]["has_tools"])
        self.assertEqual(action["ttl_seconds"], 3600)
        self.assertEqual(action["invalidation"]["strategy"], "session-scoped-exact-non-tool")
        self.assertEqual(action["invalidation"]["scope"], "session")
        self.assertFalse(action["invalidation"]["safe_invalidation"])
        self.assertFalse(action["invalidation"]["tool_call_cache_enabled"])
        self.assertFalse(action["invalidation"]["streaming_replay_enabled"])
        self.assertIn("ttl-limited-replay-window", action["invalidation"]["assumptions"])
        self.assertIn("invalidation-evidence-missing", action["invalidation"]["bypass_reasons"])
        self.assertGreater(action["projected_hits"], 0)
        self.assertGreater(action["projected_savings_usd"], 0)
        self.assertEqual(action["rollout_fraction"], 0.05)
        self.assertEqual(action["holdout_fraction"], 0.2)
        self.assertTrue(action["canary_applied_eligible"])
        self.assertTrue(action["canary_holdout_eligible"])
        self.assertEqual(action["projected_lifecycle"]["schema"], "tokenclaw.request_shape_cache_replay_canary_projected_lifecycle.v1")
        self.assertGreater(action["projected_lifecycle"]["projected_canary_applied_count"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_canary_holdout_count"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_hits"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_holdout_hits"], 0)
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["exact_non_tool_only"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(action["safety_gates"]["streaming_replay_enabled"])
        self.assertTrue(action["cache_decision_metadata"]["records_applied"])
        self.assertTrue(action["cache_decision_metadata"]["records_holdout"])
        self.assertTrue(action["cache_decision_metadata"]["records_skipped"])
        self.assertTrue(action["cache_decision_metadata"]["records_bypass"])
        self.assertTrue(action["cache_decision_metadata"]["records_bypassed"])
        self.assertTrue(action["cache_decision_metadata"]["records_invalidated"])
        self.assertTrue(action["cache_decision_metadata"]["records_invalidation_blocked"])
        self.assertTrue(action["cache_decision_metadata"]["records_stale_risk"])
        self.assertTrue(action["cache_decision_metadata"]["records_cache_hit"])
        self.assertTrue(action["cache_decision_metadata"]["records_cache_miss"])
        self.assertTrue(action["lifecycle_metadata"]["emits_applied"])
        self.assertTrue(action["lifecycle_metadata"]["emits_holdout"])
        self.assertTrue(action["lifecycle_metadata"]["emits_skipped"])
        self.assertTrue(action["lifecycle_metadata"]["emits_bypass"])
        self.assertTrue(action["lifecycle_metadata"]["emits_bypassed"])
        self.assertTrue(action["lifecycle_metadata"]["emits_invalidated"])
        self.assertTrue(action["lifecycle_metadata"]["emits_invalidation_blocked"])
        self.assertTrue(action["lifecycle_metadata"]["emits_stale_risk"])
        self.assertTrue(action["lifecycle_metadata"]["canary_applied_eligible"])
        self.assertTrue(action["lifecycle_metadata"]["canary_holdout_eligible"])
        self.assertGreater(action["lifecycle_metadata"]["projected_canary_applied_count"], 0)
        self.assertGreater(action["lifecycle_metadata"]["projected_canary_holdout_count"], 0)
        self.assertEqual(action["lifecycle_metadata"]["impact_report"], "tokenclaw.openai_cache_replay_impact.v1")
        skipped_guards = report["skipped_cohort_guards"]
        self.assertEqual(skipped_guards["schema"], "tokenclaw.request_shape_cache_replay_canary_skipped_guards.v1")
        self.assertGreaterEqual(skipped_guards["tool_cohort_count"], 1)
        self.assertGreaterEqual(skipped_guards["streaming_cohort_count"], 1)
        self.assertGreaterEqual(skipped_guards["invalidation_missing_cohort_count"], 1)
        self.assertTrue(skipped_guards["tool_streaming_and_invalidation_missing_remain_skipped"])
        blocker_values = {item["value"] for item in skipped_guards["blocker_breakdown"]}
        self.assertIn("tools-present", blocker_values)
        self.assertIn("invalidation-evidence-missing", blocker_values)
        self.assertIn("streaming-replay-not-supported", blocker_values)
        self.assertIn("unsupported-endpoint", blocker_values)
        self.assertEqual(report["source_report"]["cache_replayability_summary"]["replay_ready_cohort_count"], 2)
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_packaged_cache_rules_stage_49_row_openai_replay_cohort(self) -> None:
        rules_path = Path(__file__).parents[1] / "tokenclaw" / "cache_rules.yaml"
        policy = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        rules = policy["pattern_rules"]
        rule = next(
            item
            for item in rules
            if item.get("candidate_id") == "request-shape-cache-replay:responses:chat:8e210a2f5680d16d"
        )

        self.assertEqual(rule["id"], "local-openai-cache-replay-canary-ae8404ee817f89f4")
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["target_cache_policy"]["schema"], "tokenclaw.request_shape_cache_replay_target_policy.v1")
        self.assertEqual(rule["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_policy"], "cache_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertFalse(rule["target_cache_policy"]["rules_path_included"])
        self.assertTrue(rule["target_cache_policy"]["metadata_only"])
        self.assertTrue(rule["target_cache_policy"]["aggregate_only"])
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertEqual(rule["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(rule["conditions"]["token_bucket"], "500_2k_tokens")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["canary_unit"], "request_fingerprint")

        graduation = rule["graduation"]
        self.assertEqual(graduation["schema"], "tokenclaw.request_shape_cache_replay_shape_activation.v1")
        self.assertEqual(graduation["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(graduation["source_reason"], "replay-ready-exact-non-tool-shape")
        self.assertEqual(graduation["sample_count"], 49)
        self.assertEqual(graduation["rank"], 1)
        self.assertEqual(graduation["cohort_rank"], 1)
        self.assertEqual(graduation["projected_hits"], 48)
        self.assertEqual(graduation["projected_savings_usd"], 0.102518)
        self.assertTrue(graduation["aggregate_only"])
        self.assertEqual(graduation["shape"]["readiness"], "replay-ready")
        self.assertEqual(graduation["shape"]["reason"], "replay-ready-exact-non-tool-shape")
        rendered = json.dumps(rule, sort_keys=True)
        for forbidden in (
            "raw prompt",
            "provider body",
            "cache_key",
            "request_id",
            "session_id",
            "/tmp/",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_packaged_cache_rules_stage_next_openai_replay_cohort(self) -> None:
        rules_path = Path(__file__).parents[1] / "tokenclaw" / "cache_rules.yaml"
        policy = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        rules = policy["pattern_rules"]
        rule = next(
            item
            for item in rules
            if item.get("candidate_id") == "request-shape-cache-replay:responses:chat:21ea87daf8319e93"
        )

        self.assertEqual(rule["id"], "local-openai-cache-replay-canary-204ae274924d75dd")
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["target_cache_policy"]["schema"], "tokenclaw.request_shape_cache_replay_target_policy.v1")
        self.assertEqual(rule["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_policy"], "cache_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertFalse(rule["target_cache_policy"]["rules_path_included"])
        self.assertTrue(rule["target_cache_policy"]["metadata_only"])
        self.assertTrue(rule["target_cache_policy"]["aggregate_only"])
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertEqual(rule["conditions"]["text_bucket"], "8k_32k_chars")
        self.assertEqual(rule["conditions"]["token_bucket"], "2k_8k_tokens")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["canary_unit"], "request_fingerprint")

        graduation = rule["graduation"]
        self.assertEqual(graduation["schema"], "tokenclaw.request_shape_cache_replay_shape_activation.v1")
        self.assertEqual(graduation["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(graduation["source_reason"], "replay-ready-exact-non-tool-shape")
        self.assertEqual(graduation["sample_count"], 10)
        self.assertEqual(graduation["rank"], 1)
        self.assertEqual(graduation["cohort_rank"], 1)
        self.assertEqual(graduation["projected_hits"], 9)
        self.assertEqual(graduation["projected_savings_usd"], 0.031711)
        self.assertTrue(graduation["aggregate_only"])
        self.assertEqual(graduation["shape"]["text_bucket"], "8k_32k_chars")
        self.assertEqual(graduation["shape"]["token_bucket"], "2k_8k_tokens")
        self.assertEqual(graduation["shape"]["readiness"], "replay-ready")
        self.assertEqual(graduation["shape"]["reason"], "replay-ready-exact-non-tool-shape")
        rendered = json.dumps(rule, sort_keys=True)
        for forbidden in (
            "raw prompt",
            "provider body",
            "cache_key",
            "request_id",
            "session_id",
            "/tmp/",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_packaged_cache_rules_stage_remaining_openai_replay_cohort(self) -> None:
        rules_path = Path(__file__).parents[1] / "tokenclaw" / "cache_rules.yaml"
        policy = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        rules = policy["pattern_rules"]
        rule = next(
            item
            for item in rules
            if item.get("candidate_id") == "request-shape-cache-replay:responses:chat:5f65035aa6826d9d"
        )

        self.assertEqual(rule["id"], "local-openai-cache-replay-canary-3acbfbd015741a58")
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["target_cache_policy"]["schema"], "tokenclaw.request_shape_cache_replay_target_policy.v1")
        self.assertEqual(rule["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_policy"], "cache_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertFalse(rule["target_cache_policy"]["rules_path_included"])
        self.assertTrue(rule["target_cache_policy"]["metadata_only"])
        self.assertTrue(rule["target_cache_policy"]["aggregate_only"])
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertEqual(rule["conditions"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(rule["conditions"]["token_bucket"], "2k_8k_tokens")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.10)
        self.assertEqual(rule["rollout"]["canary_unit"], "request_fingerprint")

        graduation = rule["graduation"]
        self.assertEqual(graduation["schema"], "tokenclaw.request_shape_cache_replay_shape_activation.v1")
        self.assertEqual(graduation["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(graduation["source_reason"], "replay-ready-exact-non-tool-shape")
        self.assertEqual(graduation["sample_count"], 8)
        self.assertEqual(graduation["rank"], 1)
        self.assertEqual(graduation["cohort_rank"], 1)
        self.assertEqual(graduation["projected_hits"], 7)
        self.assertEqual(graduation["projected_savings_usd"], 0.01869)
        self.assertTrue(graduation["aggregate_only"])
        self.assertEqual(graduation["shape"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(graduation["shape"]["token_bucket"], "2k_8k_tokens")
        self.assertEqual(graduation["shape"]["readiness"], "replay-ready")
        self.assertEqual(graduation["shape"]["reason"], "replay-ready-exact-non-tool-shape")
        rendered = json.dumps(rule, sort_keys=True)
        for forbidden in (
            "raw prompt",
            "provider body",
            "cache_key",
            "request_id",
            "session_id",
            "/tmp/",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_canary_stage_selects_only_top_ranked_replay_ready_cohort(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.20, 0.25, 0.30, 0.35):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=12_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.40, 0.45, 0.50, 0.55, 0.60):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4",
                routed_model="gpt-5.4",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=40_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-16-cache-replay-top-only",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
        )

        self.assertEqual(report["status"], "staged")
        self.assertEqual(report["eligible_stageable_cohort_count"], 1)
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertEqual(len(report["stage_actions"]), 1)
        self.assertTrue(report["acceptance"]["stages_single_top_ranked_cohort"])
        self.assertTrue(report["acceptance"]["stages_remaining_unhandled_replay_ready_cohort"])
        self.assertTrue(report["acceptance"]["excludes_already_handled_local_policy_cohorts"])
        self.assertEqual(report["source_report"]["cache_replayability_summary"]["replay_ready_cohort_count"], 3)
        self.assertEqual(report["source_report"]["cache_replayability_summary"]["remaining_replay_ready_cohort_count"], 1)
        self.assertEqual(report["source_report"]["cache_replayability_summary"]["handled_replay_ready_cohort_count"], 2)
        action = report["top_stage_action"]
        self.assertEqual(action["rank"], 1)
        self.assertEqual(action["cohort_rank"], 1)
        self.assertEqual(action["shape"]["text_bucket"], "32k_128k_chars")
        self.assertEqual(action["shape"]["token_bucket"], "8k_32k_tokens")
        self.assertEqual(action["conditions"]["text_bucket"], "32k_128k_chars")
        self.assertGreater(action["projected_hits"], 0)
        self.assertGreater(action["projected_savings_usd"], 0)
        self.assertEqual(action["target_cache_policy"]["target_local_policy"], "cache_canary_policy")
        self.assertEqual(action["target_cache_policy"]["target_local_rule_file"], "cache_canary_policy.yaml")
        self.assertEqual(report["top_cohort"]["rank"], 1)
        self.assertEqual(report["top_cohort"]["remaining_rank"], 1)
        self.assertEqual(report["top_cohort"]["text_bucket"], "32k_128k_chars")
        self.assertFalse(report["top_cohort"]["handled_by_local_policy"])

    def test_cache_replay_canary_stage_cli_emits_direct_stage_payload(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=40_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-cache-stage",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_cache_replay_canary_stage.v1")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertEqual(payload["top_stage_action"]["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(payload["top_stage_action"]["conditions"]["endpoint"], "responses")
        self.assertFalse(payload["top_stage_action"]["conditions"]["has_tools"])
        self.assertFalse(payload["top_stage_action"]["conditions"]["stream"])
        self.assertGreater(payload["top_stage_action"]["projected_hits"], 0)
        self.assertGreater(payload["top_stage_action"]["projected_savings_usd"], 0)
        self.assertEqual(payload["top_stage_action"]["holdout_fraction"], 0.2)
        self.assertTrue(payload["top_stage_action"]["lifecycle_metadata"]["emits_invalidation_blocked"])
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_cache_replay_canary_stage_apply_writes_local_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-stage-apply",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        result = apply_request_shape_cache_replay_canary_action(
            report["top_stage_action"],
            rules_path=policy_path,
        )

        self.assertEqual(result["schema"], "tokenclaw.request_shape_cache_replay_canary_apply.v1")
        self.assertTrue(result["ok"])
        self.assertFalse(result["dry_run"])
        self.assertTrue(result["wrote_policy_files"])
        self.assertFalse(result["cache_entries_written"])
        self.assertFalse(result["provider_calls_made"])
        self.assertFalse(result["managed_server_calls_made"])
        self.assertFalse(result["rules_path_included"])
        self.assertEqual(result["target_local_policy"], "cache_canary_policy")
        self.assertEqual(result["canary_fraction"], 0.05)
        self.assertEqual(result["holdout_fraction"], 0.2)
        self.assertEqual(result["ttl_seconds"], 3600)
        self.assertTrue(result["privacy"]["metadata_only"])
        self.assertTrue(result["privacy"]["aggregate_only"])

        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(written["schema"], "tokenclaw.openai_cache_replay_canary_policy.v1")
        self.assertEqual(written["policy_source"], "local-manual")
        self.assertEqual(len(written["pattern_rules"]), 1)
        rule = written["pattern_rules"][0]
        self.assertEqual(rule["id"], result["policy_id"])
        self.assertEqual(rule["policy_source"], "local-manual")
        self.assertEqual(rule["candidate_id"], result["cohort_id"])
        self.assertEqual(rule["target_cache_policy"]["schema"], "tokenclaw.request_shape_cache_replay_target_policy.v1")
        self.assertEqual(rule["target_cache_policy"]["policy_section"], "cache.pattern_rules")
        self.assertEqual(rule["target_cache_policy"]["target_local_rule_file"], "cache_canary_policy.yaml")
        self.assertFalse(rule["target_cache_policy"]["rules_path_included"])
        self.assertEqual(rule["conditions"]["pattern_hashes"], ["sha256:*"])
        self.assertEqual(rule["conditions"]["provider_family"], "openai")
        self.assertEqual(rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(rule["conditions"]["endpoint"], "responses")
        self.assertEqual(rule["conditions"]["category"], "chat")
        self.assertFalse(rule["conditions"]["has_tools"])
        self.assertFalse(rule["conditions"]["stream"])
        self.assertFalse(rule["action"]["allow_tool_calls"])
        self.assertFalse(rule["action"]["streaming"])
        self.assertEqual(rule["action"]["scope"], "session")
        self.assertEqual(rule["action"]["ttl_seconds"], 3600)
        self.assertEqual(rule["rollout"]["canary_fraction"], 0.05)
        self.assertEqual(rule["rollout"]["holdout_fraction"], 0.2)
        self.assertEqual(rule["graduation"]["source_schema"], "tokenclaw.request_shape_cache_replayability_dry_run.v1")
        self.assertEqual(rule["graduation"]["rank"], 1)
        self.assertEqual(rule["graduation"]["cohort_rank"], 1)
        self.assertEqual(rule["graduation"]["shape"]["text_bucket"], "2k_8k_chars")
        self.assertEqual(rule["graduation"]["shape"]["token_bucket"], "500_2k_tokens")
        self.assertEqual(rule["graduation"]["projected_hits"], 2)
        self.assertTrue(rule["graduation"]["aggregate_only"])
        self.assertEqual(rule["invalidation"]["strategy"], "session-scoped-exact-non-tool")

        rendered = json.dumps(written, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_canary_stage_cli_apply_writes_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=40_000,
                cost=cost,
                baseline=cost,
            )

        config_dir = Path(self.tmpdir.name) / "overlay"
        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-cache-stage-apply",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
                "--config-dir",
                str(config_dir),
                "--apply",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["dry_run"])
        self.assertFalse(payload["read_only"])
        self.assertTrue(payload["wrote_policy_files"])
        self.assertFalse(payload["cache_entries_written"])
        self.assertTrue(payload["apply_result"]["ok"])
        self.assertEqual(payload["apply_result"]["schema"], "tokenclaw.request_shape_cache_replay_canary_apply.v1")
        policy_path = config_dir / "cache_canary_policy.yaml"
        self.assertTrue(policy_path.exists())
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(written["pattern_rules"][0]["policy_source"], "local-manual")
        self.assertEqual(written["pattern_rules"][0]["conditions"]["source_surface"], "openai_responses")
        self.assertFalse(written["pattern_rules"][0]["conditions"]["has_tools"])
        self.assertFalse(written["pattern_rules"][0]["conditions"]["stream"])
        self.assertNotIn("raw prompt must not leak", policy_path.read_text(encoding="utf-8"))

    def test_cache_replay_evidence_reports_empty_no_canary_without_leaking_paths(self) -> None:
        policy_path = Path(self.tmpdir.name) / "missing" / "cache_canary_policy.yaml"

        report = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_cache_replay_evidence.v1")
        self.assertEqual(report["status"], "no-canary-policy")
        self.assertFalse(report["ok"])
        self.assertFalse(report["source"]["policy_file_present"])
        self.assertFalse(report["source"]["policy_path_included"])
        self.assertEqual(report["staged_canary_count"], 0)
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["holdout_count"], 0)
        self.assertEqual(report["summary"]["projected_hits"], 0)
        self.assertEqual(report["summary"]["observed_hits"], 0)
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["cache_canary_policy_path_included"])
        self.assertFalse(report["privacy"]["policy_ids_included"])
        self.assertFalse(report["privacy"]["rule_ids_included"])
        self.assertFalse(report["privacy"]["cohort_ids_included"])
        self.assertNotIn(str(policy_path), json.dumps(report, sort_keys=True))

    def test_cache_replay_evidence_reports_staged_but_no_traffic(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-evidence-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(report["top_stage_action"], rules_path=policy_path)
        empty_store = SQLiteStore(str(Path(self.tmpdir.name) / "empty.sqlite3"))
        try:
            evidence = build_request_shape_cache_replay_evidence_report(
                empty_store,
                rules_path=policy_path,
                limit=20,
            )
        finally:
            empty_store.conn.close()

        self.assertEqual(evidence["schema"], "tokenclaw.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "staged-no-traffic")
        self.assertTrue(evidence["ok"])
        self.assertEqual(evidence["staged_canary_count"], 1)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)
        self.assertEqual(evidence["summary"]["observed_hits"], 0)
        self.assertEqual(evidence["summary"]["observed_row_count"], 0)
        self.assertEqual(evidence["summary"]["applied_count"], 0)
        self.assertEqual(evidence["summary"]["holdout_count"], 0)
        self.assertTrue(evidence["acceptance"]["has_staged_canary_metadata"])
        self.assertTrue(evidence["acceptance"]["reports_projected_vs_observed_hits"])
        self.assertTrue(evidence["acceptance"]["reports_stale_evidence_metadata"])
        self.assertFalse(evidence["source"]["policy_path_included"])
        rendered = json.dumps(evidence, sort_keys=True)
        self.assertNotIn(str(policy_path), rendered)
        self.assertNotIn("local-openai-cache-replay-canary", rendered)
        self.assertNotIn("request-shape-cache-replay:", rendered)

    def test_cache_replay_evidence_reports_stale_no_traffic_as_durable_rollback(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-evidence-stale-no-traffic",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(report["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        written["pattern_rules"][0]["staged_at"] = "2000-01-01T00:00:00+00:00"
        written["pattern_rules"][0]["graduation"]["staged_at"] = "2000-01-01T00:00:00+00:00"
        policy_path.write_text(yaml.safe_dump(written), encoding="utf-8")
        empty_store = SQLiteStore(str(Path(self.tmpdir.name) / "stale-empty.sqlite3"))
        try:
            evidence = build_request_shape_cache_replay_evidence_report(
                empty_store,
                rules_path=policy_path,
                limit=20,
                max_age_hours=72.0,
            )
        finally:
            empty_store.conn.close()

        self.assertEqual(evidence["schema"], "tokenclaw.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "staged-stale-no-traffic")
        self.assertNotEqual(evidence["status"], "staged-no-traffic")
        self.assertEqual(evidence["reason"], "stale-no-canary-traffic")
        self.assertEqual(evidence["next_action"], "apply-cache-replay-rollback-before-reobserve")
        self.assertEqual(evidence["policy_next_action"], "rollback-cache-replay-rule")
        self.assertTrue(evidence["stale_evidence"]["stale"])
        self.assertEqual(evidence["stale_evidence"]["reason"], "stale-no-canary-traffic")
        self.assertEqual(evidence["stale_evidence"]["zero_traffic_rule_count"], 1)
        reobserve = evidence["reobserve_window"]
        self.assertEqual(reobserve["schema"], "tokenclaw.request_shape_cache_replay_bounded_reobserve_window.v1")
        self.assertEqual(reobserve["status"], "rollback-required-before-reobserve")
        self.assertEqual(reobserve["decision"], "rollback-required")
        self.assertEqual(reobserve["successor_resolution"], "rollback-required")
        self.assertEqual(reobserve["next_action"], "apply-cache-replay-rollback-before-reobserve")
        recorded = reobserve["recorded_evidence"]
        self.assertEqual(
            recorded["schema"],
            "tokenclaw.request_shape_cache_replay_reobserve_recorded_evidence.v1",
        )
        self.assertEqual(recorded["max_age_hours"], 72.0)
        self.assertEqual(recorded["applied_count"], 0)
        self.assertEqual(recorded["holdout_count"], 0)
        self.assertEqual(recorded["observed_hits"], 0)
        self.assertEqual(recorded["error_count"], 0)
        self.assertEqual(recorded["fallback_count"], 0)
        self.assertEqual(recorded["invalidation_skipped_count"], 0)
        self.assertTrue(recorded["rollback_required"])
        self.assertFalse(recorded["retirement_required"])
        self.assertIn("canary_fraction", recorded)
        self.assertIn("holdout_fraction", recorded)
        self.assertTrue(recorded["metadata_only"])
        self.assertTrue(recorded["aggregate_only"])
        self.assertEqual(
            evidence["summary"]["reobserve_window_successor_resolution"],
            "rollback-required",
        )
        self.assertTrue(evidence["acceptance"]["reports_reobserve_recorded_evidence"])
        self.assertTrue(evidence["acceptance"]["resolves_stale_successor_beyond_evidence_age"])
        self.assertEqual(reobserve["traffic_floor"]["minimum_observed_rows"], 10)
        self.assertEqual(reobserve["traffic_floor"]["minimum_applied_count"], 1)
        self.assertEqual(reobserve["traffic_floor"]["minimum_holdout_count"], 1)
        self.assertEqual(reobserve["expiry"]["max_age_hours"], 72.0)
        self.assertEqual(reobserve["cache_apply_action_count"], 0)
        self.assertEqual(reobserve["cache_entries_written"], 0)
        self.assertFalse(reobserve["policy_files_written"])
        self.assertEqual(evidence["summary"]["reobserve_window_decision"], "rollback-required")
        self.assertEqual(
            evidence["summary"]["reobserve_window_next_action"],
            "apply-cache-replay-rollback-before-reobserve",
        )
        self.assertTrue(evidence["privacy"]["rule_ids_included"])
        self.assertEqual(evidence["summary"]["observed_row_count"], 0)
        self.assertEqual(evidence["summary"]["applied_count"], 0)
        self.assertEqual(evidence["summary"]["holdout_count"], 0)
        durable = evidence["durable_outcome"]
        self.assertEqual(durable["schema"], "tokenclaw.request_shape_cache_replay_durable_outcome.v1")
        self.assertEqual(durable["decision"], "rollback")
        self.assertEqual(durable["reason"], "stale-no-canary-traffic")
        self.assertEqual(durable["next_action"], "rollback-cache-replay-rule")
        self.assertFalse(durable["policy_files_written"])
        self.assertFalse(durable["cache_entries_written"])
        self.assertTrue(durable["metadata_only"])
        self.assertTrue(durable["aggregate_only"])
        self.assertTrue(evidence["acceptance"]["reports_durable_rollback_or_retirement_reason"])
        self.assertTrue(evidence["acceptance"]["reports_bounded_reobserve_window"])
        self.assertTrue(evidence["acceptance"]["reports_reobserve_traffic_floor"])
        self.assertTrue(evidence["acceptance"]["reobserve_window_writes_no_cache_entries"])
        self.assertTrue(evidence["acceptance"]["reobserve_window_emits_no_cache_apply_actions"])
        self.assertTrue(evidence["privacy"]["metadata_only"])
        self.assertTrue(evidence["privacy"]["aggregate_only"])
        rendered = json.dumps(evidence, sort_keys=True)
        self.assertNotIn(str(policy_path), rendered)
        self.assertNotIn("request-shape-cache-replay:", rendered)

    def test_cache_replay_evidence_reports_observed_applied_holdout_and_blockers(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-evidence-observed",
            rollout_fraction=0.50,
            holdout_fraction=0.25,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_result = apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="hit",
            cache_reason="exact-pattern-hit",
            cache_hit=1,
            text_chars=6_000,
            cost=0.001,
            baseline=0.031,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
                "estimated_saved_cost_usd": 0.03,
            },
        )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-pattern-miss",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            retry_count=1,
            routing_extra={"fallback_reason": "rate_limited"},
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
            },
        )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )
        bypass_canary = {**applied_canary, "status": "bypassed", "reason": "session-scope-missing"}
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="skipped",
            cache_reason="session-scope-missing",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": bypass_canary,
            },
        )
        invalidated_canary = {**applied_canary, "status": "invalidated", "reason": "dependency-changed"}
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="skipped",
            cache_reason="dependency-changed",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": invalidated_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )

        self.assertTrue(apply_result["ok"])
        self.assertEqual(evidence["schema"], "tokenclaw.request_shape_cache_replay_evidence.v1")
        self.assertEqual(evidence["status"], "observed")
        self.assertEqual(evidence["summary"]["applied_count"], 2)
        self.assertEqual(evidence["summary"]["holdout_count"], 1)
        self.assertEqual(evidence["summary"]["exact_hit_count"], 1)
        self.assertEqual(evidence["summary"]["miss_count"], 1)
        self.assertEqual(evidence["summary"]["bypass_count"], 1)
        self.assertEqual(evidence["summary"]["invalidation_skipped_count"], 1)
        self.assertEqual(evidence["summary"]["retry_count"], 1)
        self.assertEqual(evidence["summary"]["fallback_count"], 1)
        self.assertEqual(evidence["summary"]["error_count"], 0)
        self.assertEqual(evidence["summary"]["observed_hits"], 1)
        self.assertEqual(evidence["summary"]["observed_savings_usd"], 0.03)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)
        self.assertFalse(evidence["stale_evidence"]["stale"])
        blockers = {item["value"]: item["count"] for item in evidence["blocker_breakdown"]}
        self.assertEqual(blockers["session-scope-missing"], 1)
        self.assertEqual(blockers["dependency-changed"], 1)
        self.assertEqual(blockers["fallback-observed"], 1)
        canary_statuses = {item["value"]: item["count"] for item in evidence["canary_status_breakdown"]}
        self.assertEqual(canary_statuses["applied"], 2)
        self.assertEqual(canary_statuses["holdout"], 1)
        self.assertTrue(evidence["acceptance"]["reports_applied_and_holdout_counts"])
        self.assertTrue(evidence["acceptance"]["reports_projected_vs_observed_hits"])
        self.assertTrue(evidence["acceptance"]["reports_observed_savings_estimate"])
        self.assertTrue(evidence["acceptance"]["reports_blocker_breakdown"])

        rendered = json.dumps(evidence, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_evidence_ranks_applied_miss_blockers_without_raw_metadata(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-16-cache-replay-applied-miss-blockers",
            rollout_fraction=0.75,
            holdout_fraction=0.25,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        for cache_reason, extra in (
            (
                "exact-pattern-miss",
                {
                    "cache_replay_store": {"status": "stored", "reason": "compatible-success-response"},
                    "cache_replay_blocker_reasons": ["file-dependency-missing", "replay-rule-required"],
                    "file_dependency_audit": {"safe_invalidation_evidence": False},
                },
            ),
            ("exact-pattern-miss", {"pattern_rules": {"skip_reasons": [{"reason": "pattern-hash-mismatch"}]}}),
            ("ttl-expired-without-tool-result", {}),
        ):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason=cache_reason,
                text_chars=6_000,
                cost=0.01,
                baseline=0.01,
                cache_extra={
                    "pattern_rule": public_pattern_rule,
                    "cache_replay_canary": applied_canary,
                    **extra,
                },
            )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(
            evidence,
            hit_recovery_report=self._cache_replay_hit_recovery_smoke(),
        )

        self.assertEqual(evidence["summary"]["applied_count"], 3)
        self.assertEqual(evidence["summary"]["holdout_count"], 1)
        self.assertEqual(evidence["summary"]["miss_count"], 3)
        self.assertEqual(evidence["summary"]["observed_hits"], 0)
        self.assertEqual(evidence["summary"]["projected_hits"], 2)
        applied_miss_blockers = {item["value"]: item["count"] for item in evidence["applied_miss_blocker_breakdown"]}
        self.assertEqual(applied_miss_blockers["cache-write-absence"], 2)
        self.assertEqual(applied_miss_blockers["first-seen-cache-warmup"], 1)
        self.assertEqual(applied_miss_blockers["fingerprint-drift"], 1)
        self.assertEqual(applied_miss_blockers["ttl-expiry"], 1)
        self.assertEqual(evidence["summary"]["top_applied_miss_blocker"], "cache-write-absence")
        self.assertTrue(evidence["acceptance"]["reports_applied_miss_blocker_breakdown"])
        self.assertEqual(decision["decision"], "keep-blocked")
        self.assertIn("applied-cache-replay-miss-observed", decision["reason_codes"])
        self.assertIn("applied-miss:cache-write-absence", decision["reason_codes"])
        self.assertTrue(decision["acceptance"]["reports_applied_miss_blocker_breakdown"])
        rendered = json.dumps({"evidence": evidence, "decision": decision}, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_evidence_splits_successful_write_warmup_from_generic_miss(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-17-cache-replay-hit-recovery-blockers",
            rollout_fraction=0.75,
            holdout_fraction=0.25,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        for _ in range(3):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-pattern-miss",
                text_chars=6_000,
                cost=0.01,
                baseline=0.01,
                cache_extra={
                    "pattern_rule": public_pattern_rule,
                    "cache_replay_canary": applied_canary,
                    "cache_replay_store": {
                        "status": "stored",
                        "reason": "compatible-success-response",
                        "cache_key_included": False,
                    },
                },
            )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(
            evidence,
            hit_recovery_report=self._cache_replay_hit_recovery_smoke(),
        )
        applied_miss_blockers = {item["value"]: item["count"] for item in evidence["applied_miss_blocker_breakdown"]}

        self.assertEqual(evidence["summary"]["applied_count"], 3)
        self.assertEqual(evidence["summary"]["holdout_count"], 1)
        self.assertEqual(evidence["summary"]["miss_count"], 3)
        self.assertEqual(evidence["summary"]["observed_hits"], 0)
        self.assertEqual(applied_miss_blockers, {"first-seen-cache-warmup": 3})
        self.assertEqual(evidence["summary"]["top_applied_miss_blocker"], "first-seen-cache-warmup")
        self.assertEqual(evidence["warmup_analysis"]["schema"], "tokenclaw.request_shape_cache_replay_warmup_analysis.v1")
        self.assertTrue(evidence["warmup_analysis"]["warmup_only_applied_misses"])
        self.assertEqual(evidence["warmup_analysis"]["warmup_miss_count"], 3)
        self.assertEqual(evidence["warmup_analysis"]["observed_hit_blocker"], "first-seen-cache-warmup")
        self.assertEqual(evidence["warmup_analysis"]["repeat_window"]["schema"], "tokenclaw.request_shape_cache_replay_repeat_window.v1")
        self.assertTrue(evidence["warmup_analysis"]["repeat_window"]["eligible"])
        self.assertTrue(evidence["warmup_analysis"]["repeat_window"]["later_exact_repeat_expected"])
        self.assertFalse(evidence["warmup_analysis"]["provider_calls_made"])
        self.assertFalse(evidence["warmup_analysis"]["cache_entries_written"])
        self.assertTrue(evidence["acceptance"]["reports_warmup_analysis"])
        self.assertTrue(evidence["acceptance"]["reports_repeat_window_metadata"])
        self.assertEqual(decision["decision"], "keep-staged")
        self.assertEqual(decision["promotion_decision"], "keep-staged-warmup")
        self.assertEqual(
            decision["top_decision"]["post_rollback_observation"]["successor_resolution"],
            "keep-staged-warmup",
        )
        self.assertEqual(
            decision["summary"]["post_rollback_observation_successor_resolution"],
            "keep-staged-warmup",
        )
        self.assertEqual(decision["reason"], "first-seen-cache-warmup")
        self.assertEqual(decision["warmup_analysis"]["schema"], "tokenclaw.request_shape_cache_replay_warmup_analysis.v1")
        self.assertEqual(decision["top_decision"]["warmup_analysis"]["status"], evidence["warmup_analysis"]["status"])
        self.assertTrue(decision["summary"]["later_exact_repeat_expected"])
        self.assertIn("first-seen-cache-warmup", decision["reason_codes"])
        self.assertIn("applied-miss:first-seen-cache-warmup", decision["reason_codes"])
        self.assertNotIn("applied-miss:cache-warmup-miss", decision["reason_codes"])
        self.assertTrue(evidence["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])
        rendered = json.dumps({"evidence": evidence, "decision": decision}, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_evidence_cli_reads_policy_overlay(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_evidence_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--rules-path",
                str(policy_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_cache_replay_evidence.v1")
        self.assertEqual(payload["status"], "staged-no-traffic")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn(str(policy_path), stdout.getvalue())

    def test_cache_replay_policy_decision_widens_observed_hit_with_holdout(self) -> None:
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )
        stage = build_request_shape_cache_replay_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-cache-replay-policy-decision",
            rollout_fraction=0.50,
            holdout_fraction=0.25,
            mark_handled_cache_replay_cohorts=False,
        )
        policy_path = Path(self.tmpdir.name) / "config" / "cache_canary_policy.yaml"
        apply_request_shape_cache_replay_canary_action(stage["top_stage_action"], rules_path=policy_path)
        written = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        rule = written["pattern_rules"][0]
        public_pattern_rule = {
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "allow_tool_calls": False,
            "safe_invalidation": False,
            "rollout": rule["rollout"],
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "graduation": rule["graduation"],
        }
        applied_canary = {
            "schema": "tokenclaw.cache_replay_canary_decision.v1",
            "rule_id": rule["id"],
            "candidate_id": rule["candidate_id"],
            "policy_source": "local-manual",
            "scope": "session",
            "canary": {"enabled": True, "selected": True, "cohort": "canary_applied"},
            "canary_cohort": "canary_applied",
            "status": "applied",
            "reason": "no-dependency-required",
            "projection": rule["graduation"],
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="hit",
            cache_reason="exact-pattern-hit",
            cache_hit=1,
            text_chars=6_000,
            cost=0.001,
            baseline=0.031,
            cache_extra={
                "pattern_rule": public_pattern_rule,
                "cache_replay_canary": applied_canary,
                "estimated_saved_cost_usd": 0.03,
            },
        )
        holdout_canary = {
            **applied_canary,
            "canary": {"enabled": True, "selected": False, "cohort": "canary_holdout"},
            "canary_cohort": "canary_holdout",
            "status": "holdout",
            "reason": "canary_holdout",
        }
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="canary_holdout",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
            cache_extra={
                "pattern_rule": {**public_pattern_rule, "canary": holdout_canary["canary"]},
                "cache_replay_canary": holdout_canary,
            },
        )

        evidence = build_request_shape_cache_replay_evidence_report(
            self.store,
            rules_path=policy_path,
            limit=20,
        )
        decision = build_request_shape_cache_replay_policy_decision_report(evidence)
        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_policy_decision_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--rules-path",
                str(policy_path),
            ],
            stdout=stdout,
        )
        cli_decision = json.loads(stdout.getvalue())

        self.assertEqual(decision["schema"], "tokenclaw.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(decision["decision"], "widen")
        self.assertEqual(decision["promotion_readiness"], "promotion-ready")
        self.assertEqual(decision["impact_recommendation"], "promotion-ready")
        self.assertEqual(decision["promotion_recommendation"], "promotion-ready")
        self.assertEqual(
            decision["top_decision"]["post_rollback_observation"]["successor_resolution"],
            "fresh-applied-holdout-evidence",
        )
        self.assertEqual(
            decision["summary"]["post_rollback_observation_successor_resolution"],
            "fresh-applied-holdout-evidence",
        )
        self.assertEqual(code, 0)
        self.assertEqual(cli_decision["schema"], "tokenclaw.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(cli_decision["decision"], "widen")
        self.assertEqual(cli_decision["summary"]["promotion_readiness"], "promotion-ready")
        self.assertEqual(cli_decision["summary"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertTrue(cli_decision["privacy"]["metadata_only"])
        self.assertTrue(cli_decision["privacy"]["aggregate_only"])
        self.assertNotIn(str(policy_path), stdout.getvalue())
        self.assertTrue(decision["summary"]["promotion_allowed"])
        self.assertTrue(decision["summary"]["promotion_ready"])
        self.assertFalse(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["keep_blocked"])
        self.assertIn("promotion-ready", decision["reason_codes"])
        self.assertFalse(decision["summary"]["policy_files_written"])
        self.assertFalse(decision["summary"]["cache_entries_written"])
        self.assertEqual(decision["summary"]["applied_count"], 1)
        self.assertEqual(decision["summary"]["holdout_count"], 1)
        self.assertEqual(decision["summary"]["observed_hits"], 1)
        self.assertEqual(decision["summary"]["observed_savings_usd"], 0.03)
        top = decision["top_decision"]
        self.assertEqual(top["decision_options"], ["widen", "rollback", "retire-staged-no-repeat", "keep-staged", "keep-blocked"])
        self.assertEqual(top["promotion_readiness"], "promotion-ready")
        self.assertEqual(
            top["promotion_readiness_options"],
            [
                "promotion-ready",
                "retire-staged-no-repeat",
                "keep-staged-warmup",
                "keep-staged",
                "keep-blocked",
                "rollback-required",
            ],
        )
        self.assertTrue(top["promotion_ready"])
        self.assertEqual(top["local_policy_patch"]["patch_type"], "widen_openai_exact_cache_replay_canary")
        self.assertEqual(top["local_policy_patch"]["target_local_rule_file"], "cache_rules.yaml")
        promoted_rule = top["local_policy_patch"]["pattern_rules"][0]
        self.assertEqual(promoted_rule["conditions"]["source_surface"], "openai_responses")
        self.assertEqual(promoted_rule["conditions"]["endpoint"], "responses")
        self.assertEqual(promoted_rule["conditions"]["category"], "chat")
        self.assertFalse(promoted_rule["conditions"]["has_tools"])
        self.assertFalse(promoted_rule["conditions"]["stream"])
        self.assertFalse(promoted_rule["action"]["allow_tool_calls"])
        self.assertFalse(promoted_rule["action"]["streaming"])
        self.assertEqual(promoted_rule["rollout"]["recommendation_mode"], "active")
        self.assertEqual(top["rollback_metadata"]["rollback_action_type"], "disable_openai_exact_cache_replay_policy")
        self.assertTrue(top["coverage"]["has_applied_coverage"])
        self.assertTrue(top["coverage"]["has_holdout_coverage"])
        self.assertTrue(top["coverage"]["has_observed_hits"])
        self.assertTrue(decision["acceptance"]["targets_file_backed_cache_policy"])
        self.assertTrue(decision["acceptance"]["keeps_tool_and_streaming_replay_blocked"])
        self.assertTrue(decision["acceptance"]["emits_explicit_promotion_readiness"])
        rendered = json.dumps(decision, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
            rule["id"],
            rule["candidate_id"],
        ):
            self.assertNotIn(forbidden, rendered)

    def test_cache_replay_policy_decision_keeps_staged_for_insufficient_holdout(self) -> None:
        evidence = {
            "schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 1,
                "applied_count": 1,
                "holdout_count": 0,
                "exact_hit_count": 1,
                "miss_count": 0,
                "observed_hits": 1,
                "projected_hits": 2,
                "observed_savings_usd": 0.03,
                "projected_savings_usd": 0.06,
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "stale_evidence": {"stale": False, "age_hours": 1.0},
            "blocker_breakdown": [],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["decision"], "keep-staged")
        self.assertTrue(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["keep_blocked"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["top_decision"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertIn("missing-holdout-coverage", decision["reason_codes"])
        self.assertIsNone(decision["top_decision"]["local_policy_patch"])
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])

    def test_cache_replay_policy_decision_keeps_blocked_for_applied_miss_with_holdout(self) -> None:
        evidence = {
            "schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 5,
                "applied_count": 1,
                "holdout_count": 4,
                "exact_hit_count": 0,
                "miss_count": 1,
                "observed_hits": 0,
                "projected_hits": 35,
                "observed_savings_usd": 0.0,
                "projected_savings_usd": 0.075373,
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "stale_evidence": {"stale": False, "age_hours": 1.0},
            "blocker_breakdown": [],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["decision"], "keep-blocked")
        self.assertEqual(decision["next_action"], "keep-cache-replay-blocked")
        self.assertEqual(decision["summary"]["next_action"], "keep-cache-replay-blocked")
        self.assertTrue(decision["summary"]["keep_blocked"])
        self.assertFalse(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["summary"]["applied_count"], 1)
        self.assertEqual(decision["summary"]["holdout_count"], 4)
        self.assertEqual(decision["summary"]["miss_count"], 1)
        self.assertEqual(decision["summary"]["projected_hits"], 35)
        self.assertIn("missing-observed-cache-hits", decision["reason_codes"])
        self.assertIn("missing-observed-cache-savings", decision["reason_codes"])
        self.assertIn("applied-cache-replay-miss-observed", decision["reason_codes"])
        self.assertEqual(decision["top_decision"]["reason"], "applied-cache-replay-miss-observed")
        self.assertEqual(decision["top_decision"]["recommended_next_action"], "keep-cache-replay-blocked")
        self.assertEqual(decision["top_decision"]["next_action"], "keep-cache-replay-blocked")
        self.assertEqual(decision["top_decision"]["target_local_rule_file"], "cache_rules.yaml")
        self.assertEqual(decision["top_decision"]["target_local_policy_section"], "cache.pattern_rules")
        self.assertIsNone(decision["top_decision"]["local_policy_patch"])
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])

    def test_cache_replay_policy_decision_retires_no_repeat_warmup_after_elapsed_window(self) -> None:
        evidence = {
            "schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 40,
                "applied_count": 24,
                "holdout_count": 16,
                "exact_hit_count": 0,
                "miss_count": 24,
                "observed_hits": 0,
                "projected_hits": 35,
                "observed_savings_usd": 0.0,
                "projected_savings_usd": 0.075373,
                "top_applied_miss_blocker": "first-seen-cache-warmup",
                "invalidation_skipped_count": 0,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "applied_miss_blocker_breakdown": [{"value": "first-seen-cache-warmup", "count": 24}],
            "warmup_analysis": {
                "schema": "tokenclaw.request_shape_cache_replay_warmup_analysis.v1",
                "status": "repeat-window-elapsed-no-live-repeat",
                "classification": "first-seen-warmup-no-later-repeat-yet",
                "next_action": "keep-staged-until-live-repeat-or-blocker",
                "warmup_only_applied_misses": True,
                "warmup_miss_count": 24,
                "applied_miss_count": 24,
                "non_warmup_miss_count": 0,
                "observed_hit_blocker": "first-seen-cache-warmup",
                "first_warmup_age_hours": 2.5,
                "latest_warmup_age_hours": 0.1,
                "repeat_window": {
                    "schema": "tokenclaw.request_shape_cache_replay_repeat_window.v1",
                    "ttl_seconds": 3600,
                    "ttl_hours": 1.0,
                    "eligible": True,
                    "elapsed": True,
                    "projected_hits": 35,
                    "observed_hits": 0,
                    "projected_savings_usd": 0.075373,
                    "later_exact_repeat_expected": True,
                    "later_exact_repeat_absent": True,
                    "reason": "repeat-window-elapsed-no-live-repeat",
                    "metadata_only": True,
                    "aggregate_only": True,
                },
                "provider_calls_made": False,
                "managed_server_calls_made": False,
                "policy_files_written": False,
                "cache_entries_written": False,
                "cache_keys_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "file_paths_included": False,
                "metadata_only": True,
                "aggregate_only": True,
            },
            "stale_evidence": {"stale": False, "age_hours": 2.5},
            "blocker_breakdown": [],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(
            evidence,
            hit_recovery_report=self._cache_replay_hit_recovery_smoke(),
        )

        self.assertEqual(decision["decision"], "retire-staged-no-repeat")
        self.assertEqual(decision["promotion_decision"], "retire-staged-no-repeat")
        self.assertEqual(decision["promotion_readiness"], "retire-staged-no-repeat")
        self.assertEqual(decision["impact_recommendation"], "retire-staged-no-repeat")
        self.assertEqual(decision["reason"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["promotion_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["observed_hit_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["next_action"], "retire-cache-replay-canary-no-repeat")
        self.assertFalse(decision["summary"]["keep_staged_warmup"])
        self.assertTrue(decision["summary"]["retire_staged_no_repeat"])
        self.assertTrue(decision["summary"]["retirement_required"])
        self.assertEqual(decision["summary"]["promotion_readiness"], "retire-staged-no-repeat")
        self.assertFalse(decision["summary"]["promotion_ready"])
        self.assertFalse(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["keep_blocked"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["summary"]["applied_count"], 24)
        self.assertEqual(decision["summary"]["holdout_count"], 16)
        self.assertEqual(decision["summary"]["miss_count"], 24)
        self.assertEqual(decision["summary"]["observed_hits"], 0)
        self.assertEqual(decision["summary"]["projected_hits"], 35)
        self.assertTrue(decision["summary"]["hit_recovery_demonstrated"])
        self.assertEqual(decision["summary"]["synthetic_hit_recovery_exact_hit_count"], 1)
        self.assertEqual(decision["summary"]["synthetic_hit_recovery_status"], "hit-recovered")
        self.assertTrue(decision["summary"]["target_matches_hit_recovery_shape"])
        self.assertEqual(decision["summary"]["warmup_status"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["summary"]["warmup_classification"], "first-seen-warmup-no-later-repeat-yet")
        self.assertEqual(decision["summary"]["warmup_miss_count"], 24)
        self.assertEqual(decision["summary"]["non_warmup_miss_count"], 0)
        self.assertEqual(decision["summary"]["first_warmup_age_hours"], 2.5)
        self.assertTrue(decision["summary"]["repeat_window_elapsed"])
        self.assertTrue(decision["summary"]["later_exact_repeat_expected"])
        self.assertTrue(decision["summary"]["later_exact_repeat_absent"])
        self.assertEqual(decision["summary"]["top_applied_miss_blocker"], "first-seen-cache-warmup")
        self.assertEqual(decision["summary"]["promotion_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["summary"]["observed_hit_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertIn("retire-staged-no-repeat", decision["reason_codes"])
        self.assertIn("repeat-window-elapsed-no-live-repeat", decision["reason_codes"])
        self.assertIn("first-seen-cache-warmup", decision["reason_codes"])
        self.assertIn("applied-miss:first-seen-cache-warmup", decision["reason_codes"])
        self.assertEqual(decision["hit_recovery_metrics"]["source_schema"], "tokenclaw.cache_replay_hit_recovery_smoke.v1")
        self.assertTrue(decision["hit_recovery_metrics"]["hit_recovery_demonstrated"])
        self.assertEqual(decision["hit_recovery_metrics"]["synthetic_exact_hit_count"], 1)
        self.assertFalse(decision["hit_recovery_metrics"]["provider_calls_made"])
        self.assertTrue(decision["hit_recovery_metrics"]["metadata_only"])
        self.assertTrue(decision["hit_recovery_metrics"]["synthetic_only"])
        self.assertFalse(decision["hit_recovery_metrics"]["target_rule_id_included"])
        self.assertEqual(decision["warmup_analysis"]["status"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["warmup_analysis"]["repeat_window"]["ttl_seconds"], 3600)
        self.assertTrue(decision["warmup_analysis"]["repeat_window"]["later_exact_repeat_absent"])
        self.assertFalse(decision["warmup_analysis"]["provider_calls_made"])
        self.assertFalse(decision["warmup_analysis"]["managed_server_calls_made"])
        self.assertFalse(decision["warmup_analysis"]["policy_files_written"])
        self.assertFalse(decision["warmup_analysis"]["cache_entries_written"])
        self.assertFalse(decision["warmup_analysis"]["cache_keys_included"])
        self.assertFalse(decision["warmup_analysis"]["request_ids_included"])
        self.assertFalse(decision["warmup_analysis"]["session_ids_included"])
        self.assertTrue(decision["warmup_analysis"]["metadata_only"])
        self.assertTrue(decision["warmup_analysis"]["aggregate_only"])
        self.assertEqual(
            decision["duplicate_suppression"]["reason"],
            "synthetic-hit-recovery-proven-live-traffic-no-repeat-retired",
        )
        self.assertTrue(decision["duplicate_suppression"]["suppresses_generic_replay_ready_issue"])
        self.assertTrue(decision["duplicate_suppression"]["suppresses_new_cache_replay_stage_issue"])
        self.assertEqual(decision["top_decision"]["promotion_decision"], "retire-staged-no-repeat")
        self.assertEqual(decision["top_decision"]["promotion_readiness"], "retire-staged-no-repeat")
        self.assertEqual(
            decision["top_decision"]["post_rollback_observation"]["successor_resolution"],
            "retire-staged-no-repeat",
        )
        self.assertEqual(
            decision["top_decision"]["promotion_decision_options"],
            ["promote", "keep-staged-warmup", "retire-staged-no-repeat", "keep-blocked"],
        )
        self.assertIn("retire-staged-no-repeat", decision["top_decision"]["decision_options"])
        self.assertEqual(decision["top_decision"]["reason"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["top_decision"]["promotion_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["top_decision"]["observed_hit_blocker"], "repeat-window-elapsed-no-live-repeat")
        self.assertEqual(decision["top_decision"]["recommended_next_action"], "retire-cache-replay-canary-no-repeat")
        self.assertEqual(decision["top_decision"]["warmup_analysis"]["status"], "repeat-window-elapsed-no-live-repeat")
        self.assertTrue(decision["top_decision"]["warmup_analysis"]["repeat_window"]["elapsed"])
        self.assertEqual(decision["top_decision"]["coverage"]["applied_count"], 24)
        self.assertEqual(decision["top_decision"]["coverage"]["holdout_count"], 16)
        self.assertEqual(decision["top_decision"]["coverage"]["miss_count"], 24)
        self.assertEqual(decision["top_decision"]["coverage"]["observed_hits"], 0)
        self.assertIsNotNone(decision["top_decision"]["local_policy_patch"])
        self.assertEqual(
            decision["top_decision"]["local_policy_patch"]["patch_type"],
            "retire_openai_exact_cache_replay_canary",
        )
        self.assertFalse(decision["top_decision"]["local_policy_patch"]["pattern_rules"][0]["enabled"])
        self.assertEqual(
            decision["top_decision"]["local_policy_patch"]["pattern_rules"][0]["disabled_reason"],
            "repeat-window-elapsed-no-live-repeat",
        )
        self.assertEqual(
            decision["top_decision"]["rollback_metadata"]["rollback_action_type"],
            "disable_openai_exact_cache_replay_policy",
        )
        self.assertEqual(
            decision["top_decision"]["rollback_metadata"]["disable_patch"]["pattern_rules"][0]["disabled_reason"],
            "repeat-window-elapsed-no-live-repeat",
        )
        self.assertTrue(decision["acceptance"]["records_durable_decision"])
        self.assertTrue(decision["acceptance"]["single_durable_decision"])
        self.assertTrue(decision["acceptance"]["emits_explicit_canary_promotion_decision"])
        self.assertTrue(decision["acceptance"]["emits_explicit_promotion_readiness"])
        self.assertTrue(decision["acceptance"]["reports_synthetic_hit_recovery_smoke"])
        self.assertTrue(decision["acceptance"]["reports_applied_miss_blocker_breakdown"])
        self.assertTrue(decision["acceptance"]["reports_observed_hit_blocker"])
        self.assertTrue(decision["acceptance"]["reports_warmup_analysis"])
        self.assertTrue(decision["acceptance"]["reports_repeat_window_metadata"])
        self.assertTrue(decision["acceptance"]["distinguishes_first_seen_warmup_from_ineffective_replay"])
        self.assertTrue(decision["acceptance"]["suppresses_generic_replay_ready_issue_recreation"])
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])
        rendered = json.dumps(decision, sort_keys=True)
        self.assertNotIn("local-openai-cache-replay-canary-ae8404ee817f89f4", rendered)
        self.assertNotIn("raw-request-fingerprint-must-not-leak", rendered)

        evidence["warmup_analysis"]["status"] = "repeat-window-active"
        evidence["warmup_analysis"]["repeat_window"]["elapsed"] = False
        evidence["warmup_analysis"]["repeat_window"]["later_exact_repeat_absent"] = False
        keep_staged = build_request_shape_cache_replay_policy_decision_report(
            evidence,
            hit_recovery_report=self._cache_replay_hit_recovery_smoke(),
        )
        self.assertEqual(keep_staged["decision"], "keep-staged")
        self.assertEqual(keep_staged["promotion_decision"], "keep-staged-warmup")
        self.assertEqual(keep_staged["next_action"], "keep-cache-replay-canary-staged")
        self.assertFalse(keep_staged["summary"]["retire_staged_no_repeat"])

    def test_cache_replay_policy_decision_keeps_blocked_for_invalidation_risk(self) -> None:
        evidence = {
            "schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 3,
                "applied_count": 1,
                "holdout_count": 1,
                "exact_hit_count": 1,
                "miss_count": 0,
                "observed_hits": 1,
                "projected_hits": 2,
                "observed_savings_usd": 0.03,
                "projected_savings_usd": 0.06,
                "invalidation_skipped_count": 1,
                "unsupported_shape_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "error_count": 0,
            },
            "stale_evidence": {"stale": False, "age_hours": 1.0},
            "blocker_breakdown": [{"value": "dependency-changed", "count": 1}],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["decision"], "keep-blocked")
        self.assertTrue(decision["summary"]["keep_blocked"])
        self.assertFalse(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertIn("invalidation-or-stale-risk-observed", decision["reason_codes"])
        self.assertIn("dependency-changed", decision["reason_codes"])
        self.assertEqual(decision["top_decision"]["coverage"]["has_no_invalidation_skips"], False)
        self.assertIsNone(decision["top_decision"]["local_policy_patch"])

    def test_cache_replay_policy_decision_rolls_back_stale_evidence(self) -> None:
        evidence = {
            "schema": "tokenclaw.request_shape_cache_replay_evidence.v1",
            "status": "observed",
            "staged_canary_count": 1,
            "staged_canaries": [
                {
                    "shape": {
                        "source_surface": "openai_responses",
                        "endpoint": "responses",
                        "category": "chat",
                        "workflow_phase": "chat",
                        "text_bucket": "2k_8k_chars",
                        "token_bucket": "500_2k_tokens",
                        "stream": False,
                        "has_tools": False,
                    },
                    "ttl_seconds": 3600,
                }
            ],
            "summary": {
                "observed_row_count": 2,
                "applied_count": 1,
                "holdout_count": 1,
                "exact_hit_count": 1,
                "observed_hits": 1,
                "projected_hits": 2,
                "observed_savings_usd": 0.03,
                "projected_savings_usd": 0.06,
                "invalidation_skipped_count": 0,
            },
            "stale_evidence": {"stale": True, "age_hours": 96.0},
            "blocker_breakdown": [],
        }

        decision = build_request_shape_cache_replay_policy_decision_report(evidence)

        self.assertEqual(decision["decision"], "rollback")
        self.assertTrue(decision["summary"]["rollback_required"])
        self.assertFalse(decision["summary"]["promotion_allowed"])
        self.assertEqual(decision["top_decision"]["reason"], "stale-cache-replay-evidence")
        self.assertEqual(decision["top_decision"]["local_policy_patch"]["patch_type"], "rollback_openai_exact_cache_replay_policy")
        self.assertIn("stale-cache-replay-evidence", decision["top_decision"]["reason_codes"])
        reobserve = decision["top_decision"]["post_rollback_observation"]
        self.assertEqual(reobserve["schema"], "tokenclaw.request_shape_cache_replay_post_rollback_reobserve_window.v1")
        self.assertEqual(reobserve["decision"], "reobserve-after-rollback")
        self.assertEqual(reobserve["state"], "rollback-required")
        self.assertEqual(reobserve["successor_resolution"], "rollback-required")
        recorded = reobserve["recorded_evidence"]
        self.assertEqual(
            recorded["schema"],
            "tokenclaw.request_shape_cache_replay_reobserve_recorded_evidence.v1",
        )
        self.assertEqual(recorded["age_hours"], 96.0)
        self.assertTrue(recorded["rollback_required"])
        self.assertFalse(recorded["retirement_required"])
        self.assertTrue(recorded["metadata_only"])
        self.assertTrue(recorded["aggregate_only"])
        self.assertEqual(
            decision["summary"]["post_rollback_observation_successor_resolution"],
            "rollback-required",
        )
        self.assertTrue(decision["acceptance"]["reports_reobserve_recorded_evidence"])
        self.assertTrue(decision["acceptance"]["resolves_stale_successor_beyond_evidence_age"])
        self.assertEqual(reobserve["next_state"], "reobserve-window-open")
        self.assertEqual(reobserve["next_action"], "apply-cache-replay-rollback-before-reobserve")
        self.assertEqual(reobserve["traffic_floor"]["minimum_observed_rows"], 10)
        self.assertEqual(reobserve["traffic_floor"]["minimum_applied_count"], 1)
        self.assertEqual(reobserve["traffic_floor"]["minimum_holdout_count"], 1)
        self.assertEqual(reobserve["traffic_floor"]["minimum_repeat_window_seconds"], 3600)
        self.assertEqual(reobserve["expiry"]["max_age_hours"], 72.0)
        self.assertEqual(reobserve["expiry"]["reference"], "rollback_applied_at")
        self.assertFalse(reobserve["expiry"]["expires_at_included"])
        self.assertEqual(reobserve["cache_apply_action_count"], 0)
        self.assertEqual(reobserve["cache_entries_written"], 0)
        self.assertFalse(reobserve["policy_files_written"])
        self.assertFalse(reobserve["provider_calls_made"])
        self.assertFalse(reobserve["managed_server_calls_made"])
        self.assertFalse(reobserve["cache_keys_included"])
        self.assertFalse(reobserve["request_ids_included"])
        self.assertFalse(reobserve["session_ids_included"])
        self.assertFalse(reobserve["file_paths_included"])
        self.assertEqual(decision["post_rollback_observation"], reobserve)
        self.assertEqual(decision["summary"]["post_rollback_observation_state"], "rollback-required")
        self.assertEqual(decision["summary"]["post_rollback_observation_next_state"], "reobserve-window-open")
        self.assertEqual(decision["summary"]["cache_apply_action_count"], 0)
        self.assertFalse(decision["summary"]["cache_entries_written"])
        self.assertTrue(decision["acceptance"]["emits_post_rollback_reobserve_window"])
        self.assertTrue(decision["acceptance"]["reports_reobserve_traffic_floor"])
        self.assertTrue(decision["acceptance"]["reports_reobserve_expiry_metadata"])
        self.assertTrue(decision["acceptance"]["reports_reobserve_next_state"])
        self.assertTrue(decision["acceptance"]["reobserve_window_writes_no_cache_entries"])
        self.assertTrue(decision["acceptance"]["reobserve_window_emits_no_cache_apply_actions"])

    def test_cache_replay_policy_decision_cli_keeps_blocked_without_canary_policy(self) -> None:
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=6_000,
            cost=0.01,
            baseline=0.01,
        )
        missing_policy_path = Path(self.tmpdir.name) / "missing" / "cache_canary_policy.yaml"

        stdout = io.StringIO()
        code = cli.request_shape_cache_replay_policy_decision_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--rules-path",
                str(missing_policy_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_cache_replay_policy_decision.v1")
        self.assertEqual(payload["decision"], "keep-blocked")
        self.assertTrue(payload["summary"]["keep_blocked"])
        self.assertEqual(payload["summary"]["staged_canary_count"], 0)
        self.assertIn("missing-cache-replay-canary-policy", payload["reason_codes"])
        self.assertTrue(payload["acceptance"]["drafts_local_policy_patch_or_blocker"])
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertNotIn(str(missing_policy_path), stdout.getvalue())
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_managed_recommendation_handoff_cli_ranks_local_policy_handoffs(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        for cost in (0.01, 0.03, 0.02):
            self._log_call(
                provider="openai",
                path="/v1/responses",
                source_surface="openai_responses",
                endpoint="responses",
                requested_model="gpt-5.4-mini",
                routed_model="gpt-5.4-mini",
                requested_model_family="gpt-5",
                routed_model_family="gpt-5",
                category="chat",
                workflow_phase="chat",
                stream=0,
                has_tools=False,
                cache_status="miss",
                cache_reason="exact-miss",
                text_chars=6_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.managed_recommendation_handoff_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--request-shape-limit",
                "20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.managed_recommendation_handoff_health.v1")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["provider_calls_made"])
        self.assertFalse(payload["managed_server_calls_made"])
        self.assertEqual(payload["managed_dependency"], "optional")

        by_family = {row["local_action_family"]: row for row in payload["omissions"]}
        for family, rule_file in {
            "crunch": "crunch_rules.yaml",
            "cache": "cache_rules.yaml",
        }.items():
            self.assertIn(family, by_family)
            row = by_family[family]
            self.assertEqual(row["follow_up_owner"], "local-policy")
            self.assertTrue(row["local_file_backed_representation"]["exists"])
            self.assertEqual(row["local_file_backed_representation"]["rule_file"], rule_file)
            self.assertIn(rule_file, row["local_handoff_reason"])
        self.assertEqual(by_family["crunch"]["next_action"], "stage-repeated-context-crunch-canary")

        rendered = stdout.getvalue()
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertTrue(payload["privacy"]["aggregate_only"])

    def test_crunch_opportunity_dry_run_projects_repeated_context_savings(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        self._log_call(
            provider="openai",
            path="/v1/responses",
            source_surface="openai_responses",
            endpoint="responses",
            requested_model="gpt-5.4-mini",
            routed_model="gpt-5.4-mini",
            requested_model_family="gpt-5",
            routed_model_family="gpt-5",
            category="chat",
            workflow_phase="chat",
            stream=0,
            has_tools=False,
            cache_status="miss",
            cache_reason="exact-miss",
            text_chars=1_200,
            cost=0.004,
            baseline=0.004,
        )

        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-dry-run")
        dry_run = report["crunch_opportunity_dry_run"]

        self.assertEqual(dry_run["schema"], "tokenclaw.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(dry_run["status"], "ranked")
        self.assertEqual(dry_run["summary"]["measurement_ready_cohort_count"], 1)
        self.assertGreater(dry_run["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(dry_run["summary"]["projected_saved_usd"], 0)
        self.assertEqual(dry_run["summary"]["activation_state"], "activation-ready")
        self.assertEqual(dry_run["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        follow_up = dry_run["activation_follow_up"]
        self.assertEqual(follow_up["schema"], "tokenclaw.request_shape_crunch_activation_follow_up.v1")
        self.assertEqual(follow_up["activation_state"], "activation-ready")
        self.assertEqual(follow_up["activation_mode"], "canary-candidate")
        self.assertEqual(follow_up["savings_status"], "projected-savings-ranked")
        self.assertEqual(follow_up["report_key"], "request_shape_crunch_opportunity")
        self.assertEqual(follow_up["evidence_schema"], "tokenclaw.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(follow_up["candidate_count"], dry_run["summary"]["candidate_count"])
        self.assertEqual(follow_up["matched_count"], dry_run["summary"]["matched_count"])
        self.assertEqual(follow_up["rows_considered"], dry_run["summary"]["rows_considered"])
        self.assertEqual(follow_up["projected_saved_chars"], dry_run["summary"]["projected_saved_chars"])
        self.assertEqual(follow_up["projected_saved_tokens"], dry_run["summary"]["projected_saved_tokens"])
        self.assertEqual(follow_up["projected_saved_usd"], dry_run["summary"]["projected_saved_usd"])
        self.assertEqual(follow_up["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(follow_up["target_local_policy"], "crunch_rules")
        self.assertEqual(follow_up["recommended_action_count"], 1)
        self.assertFalse(follow_up["canary_already_staged"])
        self.assertFalse(follow_up["canary_already_applied"])
        self.assertIsNone(follow_up["no_op_reason"])
        self.assertFalse(follow_up["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertEqual(follow_up["missing_measurements"], [])
        self.assertTrue(follow_up["privacy"]["metadata_only"])
        self.assertTrue(follow_up["privacy"]["aggregate_only"])
        top = dry_run["cohorts"][0]
        self.assertEqual(top["readiness"], "measurement-ready")
        self.assertEqual(top["candidate_status"], "candidate")
        self.assertEqual(top["policy_write_status"], "policy-write-required")
        self.assertTrue(top["policy_write_required"])
        self.assertEqual(top["reason"], "repeated-context-crunch-opportunity")
        self.assertEqual(top["sample_count"], 3)
        self.assertEqual(top["target_local_policy_section"], "crunch.rules")
        self.assertEqual(top["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(top["blocker_codes"], [])
        self.assertIn("duplicate_suppression", top)
        self.assertIn("repeated_context", top["work_classes"])
        self.assertIn("crunch", top["work_classes"])
        self.assertEqual(top["candidate_rule"], "repeated-context-conservative-dry-run")
        status_counts = {item["value"]: item["count"] for item in dry_run["candidate_status_breakdown"]}
        self.assertEqual(status_counts["candidate"], 3)
        policy_counts = {item["value"]: item["count"] for item in dry_run["policy_write_status_breakdown"]}
        self.assertEqual(policy_counts["policy-write-required"], 3)
        self.assertEqual(dry_run["summary"]["target_local_policy_section"], "crunch.rules")
        self.assertEqual(dry_run["summary"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(dry_run["summary"]["recommended_action_count"], 1)
        action = dry_run["recommended_actions"][0]
        self.assertEqual(action["schema"], "tokenclaw.request_shape_crunch_canary_action.v1")
        self.assertEqual(action["target_local_policy"], "crunch_rules")
        self.assertEqual(action["rollout_fraction"], 0.1)
        self.assertEqual(action["holdout_fraction"], 0.1)
        self.assertEqual(action["candidate_count"], dry_run["summary"]["candidate_count"])
        self.assertEqual(action["projected_saved_chars"], top["projected_saved_chars"])
        self.assertEqual(action["projected_saved_tokens"], top["projected_saved_tokens"])
        self.assertEqual(action["projected_saved_usd"], top["projected_saved_usd"])
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["holdout_required"])
        class_counts = {item["value"]: item["count"] for item in dry_run["work_class_breakdown"]}
        self.assertEqual(class_counts["repeated_context"], 3)
        rendered = json.dumps(dry_run, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(dry_run["privacy"]["metadata_only"])
        self.assertTrue(dry_run["privacy"]["aggregate_only"])
        self.assertFalse(dry_run["privacy"]["raw_request_bodies_included"])
        self.assertFalse(dry_run["privacy"]["provider_bodies_included"])
        self.assertFalse(dry_run["privacy"]["session_ids_included"])
        self.assertFalse(dry_run["privacy"]["cache_keys_included"])

    def test_crunch_opportunity_dry_run_consumes_ranked_activation_candidate_queue(self) -> None:
        raw_request_id = "raw-crunch-queue-request-id-must-not-leak"
        raw_session_id = "raw-crunch-queue-session-id-must-not-leak"
        raw_path = "/tmp/private/crunch-queue.py"
        rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "provider_family": "openai",
                "source_surface": "openai_responses",
                "endpoint": "responses",
                "app_family": "generic_openai",
                "requested_model_family": "gpt-5",
                "routed_model_family": "gpt-5",
                "category": "tool-result",
                "workflow_phase": "tool-execution",
                "stream": True,
                "has_tools": True,
                "text_bucket": "32k_128k_chars",
                "token_bucket": "8k_32k_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "candidate_families": ["crunch_candidate"],
                "blocker_codes": [],
                "row_count": 18,
                "sample_count": 18,
                "successful_input_tokens": 180_000,
                "input_tokens": 180_000,
                "projected_crunch_tokens_saved": 9_000,
                "projected_crunch_chars_saved": 36_000,
                "projected_crunch_savings_usd": 0.027,
                "request_id": raw_request_id,
                "session_id": raw_session_id,
                "file_path": raw_path,
            }
        ]
        follow_up = build_request_shape_follow_up_candidates(rollups, limit=10)
        queue = follow_up["activation_candidate_queue"]

        dry_run = build_request_shape_crunch_opportunity_dry_run(queue)
        repeat = build_request_shape_crunch_opportunity_dry_run(queue)

        self.assertEqual(dry_run["source_schema"], "tokenclaw.request_shape_local_activation_candidate_queue.v1")
        self.assertEqual(dry_run["source_queue_status"], "ranked")
        self.assertEqual(dry_run["source_queue_entry_count"], 1)
        self.assertEqual(dry_run["source_queue_crunch_entry_count"], 1)
        self.assertEqual(dry_run["status"], "ranked")
        self.assertEqual(dry_run["summary"]["source_queue_crunch_entry_count"], 1)
        self.assertEqual(dry_run["summary"]["measurement_ready_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["projected_saved_tokens"], 9_000)
        self.assertEqual(dry_run["summary"]["projected_saved_chars"], 36_000)
        self.assertEqual(dry_run["summary"]["projected_saved_usd"], 0.027)
        self.assertEqual(dry_run["summary"]["top_next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(dry_run["missing_measurements"], [])
        top = dry_run["cohorts"][0]
        self.assertEqual(top["source_evidence_schema"], "tokenclaw.request_shape_local_activation_candidate_queue_entry.v1")
        self.assertEqual(top["source_activation_fingerprint"], queue["entries"][0]["fingerprint"])
        self.assertTrue(top["fingerprint"].startswith("crunch-opportunity:"))
        self.assertEqual(top["fingerprint"], repeat["cohorts"][0]["fingerprint"])
        self.assertEqual(top["sample_count"], 18)
        self.assertEqual(top["projected_saved_tokens"], 9_000)
        self.assertEqual(top["projected_saved_chars"], 36_000)
        self.assertEqual(top["projected_saved_usd"], 0.027)
        self.assertEqual(top["readiness"], "measurement-ready")
        self.assertEqual(top["reason"], "repeated-context-crunch-opportunity")
        self.assertIn("duplicate_suppression", top)
        self.assertTrue(dry_run["privacy"]["metadata_only"])
        self.assertTrue(dry_run["privacy"]["aggregate_only"])
        self.assertFalse(dry_run["privacy"]["provider_calls_made"])
        self.assertFalse(dry_run["privacy"]["managed_server_calls_made"])
        self.assertFalse(dry_run["summary"]["policy_files_written"])
        rendered = json.dumps(dry_run, sort_keys=True)
        self.assertNotIn(raw_request_id, rendered)
        self.assertNotIn(raw_session_id, rendered)
        self.assertNotIn(raw_path, rendered)

    def test_crunch_opportunity_dry_run_noops_for_empty_activation_candidate_queue(self) -> None:
        empty_queue = {
            "schema": "tokenclaw.request_shape_local_activation_candidate_queue.v1",
            "status": "empty",
            "entries": [],
            "privacy": {"metadata_only": True, "aggregate_only": True},
        }

        dry_run = build_request_shape_crunch_opportunity_dry_run(empty_queue)

        self.assertEqual(dry_run["status"], "no-repeated-context-crunch-cohorts")
        self.assertEqual(dry_run["source_schema"], "tokenclaw.request_shape_local_activation_candidate_queue.v1")
        self.assertEqual(dry_run["source_queue_status"], "empty")
        self.assertEqual(dry_run["source_queue_entry_count"], 0)
        self.assertEqual(dry_run["source_queue_crunch_entry_count"], 0)
        self.assertEqual(dry_run["summary"]["top_next_action"], "rank-repeated-context-crunch-dry-run")
        self.assertIn("repeated-context-crunch-cohorts", dry_run["missing_measurements"])
        self.assertEqual(dry_run["repeated_context_drill"]["state"], "no-source")
        self.assertEqual(dry_run["cohorts"], [])
        self.assertTrue(dry_run["privacy"]["metadata_only"])
        self.assertTrue(dry_run["privacy"]["aggregate_only"])

    def test_crunch_opportunity_dry_run_noops_for_small_or_one_off_rollups(self) -> None:
        small_rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "source_schema": "tokenclaw.request_shape_rollup_row.v1",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "provider_family": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "chat",
                "workflow_phase": "chat",
                "stream": False,
                "has_tools": False,
                "cache_status": "miss",
                "routing_status": "passthrough",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "row_count": 1,
                "successful_input_tokens": 1_200,
                "input_tokens": 1_200,
                "input_token_cost_usd": 0.0036,
                "projected_crunch_tokens_saved": 0,
                "projected_crunch_chars_saved": 0,
                "projected_crunch_savings_usd": 0.0,
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "blocker_codes": ["exact-cache-miss"],
            }
        ]

        dry_run = build_request_shape_crunch_opportunity_dry_run(small_rollups)

        self.assertEqual(dry_run["status"], "no-positive-crunch-opportunity")
        self.assertEqual(dry_run["summary"]["recommended_action_count"], 0)
        self.assertEqual(dry_run["summary"]["no_op_reason"], "insufficient-repeat-evidence")
        self.assertEqual(dry_run["no_op_reason"], "insufficient-repeat-evidence")
        self.assertIn("positive-observed-or-projected-savings", dry_run["missing_measurements"])
        self.assertEqual(dry_run["cohorts"][0]["candidate_status"], "too-small")
        self.assertEqual(dry_run["cohorts"][0]["policy_write_status"], "no-policy-write")
        self.assertFalse(dry_run["cohorts"][0]["policy_write_required"])
        self.assertEqual(dry_run["cohorts"][0]["sample_count"], 1)
        self.assertEqual(dry_run["cohorts"][0]["target_local_policy_section"], "crunch.rules")
        self.assertEqual(dry_run["cohorts"][0]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertIn("insufficient-repeat-evidence", dry_run["cohorts"][0]["blocker_codes"])
        status_counts = {item["value"]: item["count"] for item in dry_run["candidate_status_breakdown"]}
        self.assertEqual(status_counts["too-small"], 1)
        self.assertTrue(dry_run["privacy"]["metadata_only"])
        self.assertTrue(dry_run["privacy"]["aggregate_only"])
        self.assertFalse(dry_run["privacy"]["provider_calls_made"])
        self.assertFalse(dry_run["privacy"]["managed_server_calls_made"])
        drill = dry_run["repeated_context_drill"]
        self.assertEqual(drill["schema"], "tokenclaw.request_shape_repeated_context_crunch_drill.v1")
        self.assertEqual(drill["state"], "no-repeat")
        self.assertEqual(drill["missing_threshold"], "repeated-context-min-samples-2")
        self.assertEqual(drill["next_action"], "collect-repeated-context-samples")
        self.assertEqual(drill["ranked_cohort_count"], 0)
        self.assertEqual(drill["no_repeat_cohort_count"], 1)
        self.assertEqual(drill["ranked_cohorts"], [])
        self.assertFalse(drill["policy_files_written"])
        self.assertEqual(dry_run["summary"]["repeated_context_drill_state"], "no-repeat")

    def test_crunch_opportunity_dry_run_drill_ranks_repeated_large_context_cohorts(self) -> None:
        rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "source_schema": "tokenclaw.request_shape_rollup_row.v1",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "provider_family": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "chat",
                "workflow_phase": "thinking",
                "stream": False,
                "has_tools": False,
                "cache_status": "miss",
                "routing_status": "passthrough",
                "text_bucket": "gte_128k_chars",
                "token_bucket": "gte_32k_tokens",
                "row_count": 8,
                "successful_input_tokens": 1_600_000,
                "input_tokens": 1_600_000,
                "input_token_cost_usd": 4.8,
                "projected_crunch_tokens_saved": 70_000,
                "projected_crunch_chars_saved": 280_000,
                "projected_crunch_savings_usd": 0.21,
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "blocker_codes": [],
            },
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "source_schema": "tokenclaw.request_shape_rollup_row.v1",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "provider_family": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "chat",
                "workflow_phase": "chat",
                "stream": False,
                "has_tools": True,
                "cache_status": "miss",
                "routing_status": "passthrough",
                "text_bucket": "8k_32k_chars",
                "token_bucket": "8k_32k_tokens",
                "row_count": 3,
                "successful_input_tokens": 45_000,
                "input_tokens": 45_000,
                "input_token_cost_usd": 0.135,
                "projected_crunch_tokens_saved": 1_500,
                "projected_crunch_chars_saved": 6_000,
                "projected_crunch_savings_usd": 0.0045,
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "blocker_codes": [],
            },
        ]

        dry_run = build_request_shape_crunch_opportunity_dry_run(rollups)

        drill = dry_run["repeated_context_drill"]
        self.assertEqual(drill["state"], "ranked")
        self.assertIsNone(drill["missing_threshold"])
        self.assertEqual(drill["next_action"], "rank-repeated-context-crunch-cohorts")
        self.assertEqual(drill["ranked_cohort_count"], 2)
        self.assertEqual(drill["repeat_evidence_cohort_count"], 2)
        self.assertEqual(drill["large_context_cohort_count"], 2)
        self.assertEqual(drill["min_sample_threshold"], 2)
        self.assertEqual(drill["projected_saved_tokens"], 71_500)
        self.assertGreater(drill["projected_saved_usd"], 0)
        self.assertEqual(dry_run["summary"]["repeated_context_drill_state"], "ranked")
        self.assertEqual(dry_run["summary"]["repeated_context_ranked_cohort_count"], 2)
        ranks = drill["ranked_cohorts"]
        self.assertEqual([item["rank"] for item in ranks], [1, 2])
        # Larger sample count + larger median text size + higher projection ranks first.
        self.assertEqual(ranks[0]["sample_count"], 8)
        self.assertEqual(ranks[0]["text_bucket"], "gte_128k_chars")
        self.assertEqual(ranks[0]["median_sample_input_tokens"], 200_000)
        self.assertEqual(ranks[0]["projected_saved_tokens"], 70_000)
        self.assertFalse(ranks[0]["safety_blocked"])
        self.assertGreater(ranks[0]["repetition_signal"], ranks[1]["repetition_signal"])
        self.assertEqual(ranks[1]["sample_count"], 3)
        rendered = json.dumps(drill, sort_keys=True)
        for forbidden in ("raw prompt", "/tmp/", "req_", "sess_", "cache-key-"):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(drill["privacy"]["metadata_only"])
        self.assertTrue(drill["privacy"]["aggregate_only"])
        self.assertFalse(drill["policy_files_written"])

    def test_crunch_opportunity_dry_run_drill_reports_too_small_when_repeats_are_small(self) -> None:
        rollups = [
            {
                "schema": "tokenclaw.request_shape_rollup_row.v1",
                "source_schema": "tokenclaw.request_shape_rollup_row.v1",
                "candidate_work_classes": ["repeated_context", "crunch"],
                "provider_family": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "chat",
                "workflow_phase": "chat",
                "stream": False,
                "has_tools": False,
                "cache_status": "miss",
                "routing_status": "passthrough",
                "text_bucket": "2k_8k_chars",
                "token_bucket": "500_2k_tokens",
                "row_count": 5,
                "successful_input_tokens": 6_000,
                "input_tokens": 6_000,
                "input_token_cost_usd": 0.018,
                "projected_crunch_tokens_saved": 0,
                "projected_crunch_chars_saved": 0,
                "projected_crunch_savings_usd": 0.0,
                "current_crunch_tokens_saved": 0,
                "current_crunch_chars_saved": 0,
                "current_crunch_savings_usd": 0.0,
                "blocker_codes": [],
            }
        ]

        dry_run = build_request_shape_crunch_opportunity_dry_run(rollups)

        drill = dry_run["repeated_context_drill"]
        self.assertEqual(drill["state"], "too-small")
        self.assertEqual(drill["missing_threshold"], "repeated-context-large-context-size")
        self.assertEqual(drill["next_action"], "collect-large-context-samples")
        self.assertEqual(drill["ranked_cohort_count"], 0)
        self.assertEqual(drill["too_small_cohort_count"], 1)
        self.assertEqual(drill["repeat_evidence_cohort_count"], 1)
        self.assertEqual(drill["large_context_cohort_count"], 0)
        self.assertEqual(dry_run["summary"]["repeated_context_drill_state"], "too-small")

    def test_crunch_opportunity_dry_run_drill_reports_no_source_without_crunch_rows(self) -> None:
        dry_run = build_request_shape_crunch_opportunity_dry_run([])

        self.assertEqual(dry_run["status"], "no-repeated-context-crunch-cohorts")
        drill = dry_run["repeated_context_drill"]
        self.assertEqual(drill["state"], "no-source")
        self.assertEqual(drill["missing_threshold"], "repeated-context-source-traffic")
        self.assertEqual(drill["next_action"], "collect-source-traffic")
        self.assertEqual(drill["crunch_row_count"], 0)
        self.assertEqual(drill["matched_count"], 0)
        self.assertEqual(drill["candidate_cohort_count"], 0)
        self.assertEqual(drill["ranked_cohorts"], [])
        self.assertEqual(dry_run["summary"]["repeated_context_drill_state"], "no-source")

    def test_crunch_canary_stage_report_targets_anthropic_thinking_tool_result_cohort(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-crunch-canary-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_repeated_context_crunch_canary_stage.v1")
        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertTrue(report["dry_run"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertTrue(report["acceptance"]["stages_one_repeated_context_crunch_canary"])
        self.assertTrue(report["acceptance"]["has_projected_tokens"])
        self.assertTrue(report["acceptance"]["has_projected_savings"])
        self.assertTrue(report["acceptance"]["has_holdout_metadata"])
        self.assertTrue(report["acceptance"]["has_projected_lifecycle_split"])
        self.assertTrue(report["acceptance"]["has_safety_stop_metadata"])
        self.assertTrue(report["acceptance"]["has_cohort_selector"])
        self.assertTrue(report["acceptance"]["has_rollback_threshold"])
        self.assertTrue(report["acceptance"]["has_duplicate_suppression"])
        self.assertTrue(report["acceptance"]["unsafe_or_stale_cohorts_remain_skipped"])

        action = report["top_stage_action"]
        self.assertEqual(action["schema"], "tokenclaw.request_shape_crunch_canary_action.v1")
        self.assertEqual(action["action_type"], "stage-local-repeated-context-crunch-canary")
        self.assertEqual(action["target_local_policy"], "crunch_rules")
        self.assertEqual(action["conditions"]["provider_family"], "anthropic")
        self.assertEqual(action["conditions"]["source_surface"], "anthropic_messages")
        self.assertEqual(action["conditions"]["endpoint"], "messages")
        self.assertEqual(action["conditions"]["category"], "tool-result")
        self.assertEqual(action["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(action["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertTrue(action["conditions"]["stream"])
        self.assertTrue(action["conditions"]["has_tools"])
        self.assertEqual(action["cohort_selector"]["schema"], "tokenclaw.request_shape_crunch_canary_cohort_selector.v1")
        self.assertEqual(action["cohort_selector"]["workflow_phase"], "thinking")
        self.assertEqual(action["rollout_fraction"], 0.05)
        self.assertEqual(action["holdout_fraction"], 0.2)
        self.assertEqual(action["rollback_threshold"], 0.2)
        self.assertEqual(action["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertEqual(action["rollback_metadata"]["target_policy_id"], action["policy_id"])
        self.assertEqual(action["rollback_metadata"]["target_cohort_id"], action["cohort_id"])
        self.assertTrue(action["duplicate_suppression"]["metadata_only"])
        self.assertGreater(action["projected_saved_tokens"], 0)
        self.assertGreater(action["projected_saved_usd"], 0)
        self.assertEqual(action["projected_lifecycle"]["schema"], "tokenclaw.request_shape_crunch_canary_projected_lifecycle.v1")
        self.assertEqual(action["projected_lifecycle"]["matched_count"], 3)
        self.assertEqual(action["projected_lifecycle"]["projected_canary_applied_count"], 1)
        self.assertEqual(action["projected_lifecycle"]["projected_canary_holdout_count"], 1)
        self.assertEqual(action["projected_lifecycle"]["projected_skipped_count"], 1)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_saved_tokens"], 0)
        self.assertGreater(action["projected_lifecycle"]["projected_applied_saved_usd"], 0)
        self.assertEqual(action["source_evidence_schema"], "tokenclaw.request_shape_rollup_row.v1")
        self.assertEqual(
            action["source_evidence_schemas"],
            [
                "tokenclaw.request_shape_follow_up_candidates.v1",
                "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            ],
        )
        self.assertEqual(action["local_only_reason"], "file-backed-local-policy-no-managed-dependency")
        self.assertIn("thinking-routing-guard", action["evidence_blocker_codes"])
        self.assertIn("tool-call-cache-disabled", action["evidence_blocker_codes"])
        self.assertIn("unsupported-streaming-shape", action["evidence_blocker_codes"])
        self.assertEqual(
            action["projected_lifecycle"]["evidence_blocker_codes"],
            action["evidence_blocker_codes"],
        )
        self.assertTrue(action["projected_lifecycle"]["privacy"]["metadata_only"])
        self.assertTrue(action["projected_lifecycle"]["privacy"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["metadata_only"])
        self.assertTrue(action["safety_gates"]["aggregate_only"])
        self.assertTrue(action["safety_gates"]["local_only"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(action["safety_gates"]["tool_call_cache_enablement_allowed"])
        self.assertFalse(action["safety_gates"]["request_ids_included"])
        self.assertFalse(action["safety_gates"]["session_ids_included"])
        self.assertFalse(action["safety_gates"]["cache_keys_included"])
        self.assertFalse(action["safety_gates"]["file_paths_included"])
        self.assertFalse(action["safety_gates"]["tool_payloads_included"])
        self.assertTrue(action["safety_gates"]["holdout_required"])
        self.assertTrue(action["safety_gates"]["records_applied_holdout_skipped_safety_stopped_fallback_rollback"])
        self.assertTrue(action["lifecycle_metadata"]["emits_applied"])
        self.assertTrue(action["lifecycle_metadata"]["emits_holdout"])
        self.assertTrue(action["lifecycle_metadata"]["emits_skipped"])
        self.assertTrue(action["lifecycle_metadata"]["emits_safety_stopped"])
        self.assertEqual(action["lifecycle_metadata"]["evidence_blocker_codes"], action["evidence_blocker_codes"])
        self.assertEqual(action["lifecycle_metadata"]["projected_canary_applied_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["projected_canary_holdout_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["projected_skipped_count"], 1)
        self.assertEqual(action["lifecycle_metadata"]["impact_report"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        projection = report["stage_lifecycle_projection"]
        self.assertEqual(projection["schema"], "tokenclaw.request_shape_crunch_canary_stage_lifecycle_projection.v1")
        self.assertEqual(projection["matched_count"], 3)
        self.assertEqual(projection["projected_canary_applied_count"], 1)
        self.assertEqual(projection["projected_canary_holdout_count"], 1)
        self.assertEqual(projection["projected_skipped_count"], 1)
        self.assertGreater(projection["projected_applied_saved_tokens"], 0)
        self.assertGreater(projection["projected_applied_saved_usd"], 0)
        self.assertTrue(projection["privacy"]["metadata_only"])
        self.assertTrue(projection["privacy"]["aggregate_only"])
        self.assertEqual(report["source_report"]["activation_follow_up"]["next_action"], "stage-repeated-context-crunch-canary")
        self.assertEqual(report["duplicate_suppression"]["schema"], "tokenclaw.request_shape_crunch_stage_duplicate_suppression_summary.v1")
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_cli_emits_direct_stage_payload(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-stage",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_repeated_context_crunch_canary_stage.v1")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertEqual(payload["top_stage_action"]["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(payload["top_stage_action"]["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertGreater(payload["top_stage_action"]["projected_saved_tokens"], 0)
        self.assertEqual(payload["top_stage_action"]["holdout_fraction"], 0.2)
        self.assertEqual(payload["top_stage_action"]["rollback_threshold"], 0.2)
        self.assertEqual(payload["top_stage_action"]["cohort_selector"]["text_bucket"], "gte_128k_chars")
        self.assertTrue(payload["acceptance"]["has_duplicate_suppression"])
        self.assertEqual(payload["top_stage_action"]["projected_lifecycle"]["projected_canary_applied_count"], 1)
        self.assertEqual(payload["top_stage_action"]["projected_lifecycle"]["projected_canary_holdout_count"], 1)
        self.assertTrue(payload["top_stage_action"]["lifecycle_metadata"]["emits_safety_stopped"])
        self.assertEqual(payload["stage_lifecycle_projection"]["projected_canary_applied_count"], 1)
        self.assertEqual(payload["stage_lifecycle_projection"]["projected_canary_holdout_count"], 1)
        self.assertTrue(payload["privacy"]["aggregate_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_crunch_canary_stage_cli_accepts_context_plateau_stats_source(self) -> None:
        stats_path = Path(self.tmpdir.name) / "context-plateaus.json"
        stats_path.write_text(
            json.dumps(
                {
                    "context_plateaus": [
                        {
                            "session_id": "raw-session-id-must-not-leak",
                            "source_surface": "anthropic_messages",
                            "app_family": "claude-code",
                            "category": "tool-result",
                            "workflow_phase": "tool-execution",
                            "calls": 24,
                            "plateau_pairs": 18,
                            "median_text_chars": 96_000,
                            "p90_text_chars": 118_000,
                            "cost_usd": 0.72,
                            "cache_read_savings_usd": 0.18,
                            "crunch_saved_chars": 4_000,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--context-plateau-stats-json",
                str(stats_path),
                "--rollout-fraction",
                "0.10",
                "--holdout-fraction",
                "0.20",
                "--run-id",
                "plateau-stage-cli",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_repeated_context_crunch_canary_stage.v1")
        self.assertEqual(payload["status"], "staged")
        self.assertEqual(payload["source_report"]["schema"], "tokenclaw.context_plateau_crunch_rollups.v1")
        self.assertEqual(payload["source_report"]["window"]["source"], "context_plateaus")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertTrue(payload["acceptance"]["stages_one_repeated_context_crunch_canary"])
        self.assertTrue(payload["acceptance"]["has_projected_lifecycle_split"])
        self.assertTrue(payload["acceptance"]["has_holdout_metadata"])
        self.assertEqual(payload["top_stage_action"]["source_evidence_schema"], "tokenclaw.context_plateau_crunch_rollup_row.v1")
        self.assertEqual(payload["top_stage_action"]["conditions"]["source_surface"], "anthropic_messages")
        self.assertEqual(payload["top_stage_action"]["conditions"]["category"], "tool-result")
        self.assertEqual(payload["top_stage_action"]["target_local_policy_section"], "crunch.rules")
        self.assertEqual(payload["top_stage_action"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertFalse(payload["privacy"]["provider_calls_made"])
        self.assertFalse(payload["privacy"]["managed_server_calls_made"])
        self.assertNotIn("raw-session-id-must-not-leak", stdout.getvalue())
        self.assertNotIn(str(stats_path), stdout.getvalue())

    def test_context_plateau_crunch_canary_apply_and_impact_have_coverage_and_suppression(self) -> None:
        source_report = build_context_plateau_crunch_rollup_report(
            {
                "context_plateaus": [
                    {
                        "session_id": "raw-session-id-must-not-leak",
                        "source_surface": "anthropic_messages",
                        "app_family": "claude-code",
                        "category": "tool-result",
                        "workflow_phase": "tool-execution",
                        "calls": 24,
                        "plateau_pairs": 18,
                        "median_text_chars": 96_000,
                        "p90_text_chars": 118_000,
                        "cost_usd": 0.72,
                        "crunch_saved_chars": 4_000,
                    }
                ],
            }
        )
        self.assertIsNotNone(source_report)
        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        stage = build_request_shape_crunch_canary_stage_report(
            self.store,
            source_rollup_report=source_report,
            rules_path=rules_path,
            rollout_fraction=0.10,
            holdout_fraction=0.20,
            run_id="plateau-source-stage",
        )
        self.assertEqual(stage["status"], "staged")
        self.assertEqual(stage["staged_canary_count"], 1)
        self.assertTrue(stage["acceptance"]["has_projected_lifecycle_split"])
        action = stage["top_stage_action"]

        apply_result = apply_request_shape_crunch_canary_action(action, rules_path=rules_path)
        self.assertTrue(apply_result["ok"])
        self.assertTrue(apply_result["wrote_policy_files"])

        restage = build_request_shape_crunch_canary_stage_report(
            self.store,
            source_rollup_report=source_report,
            rules_path=rules_path,
            rollout_fraction=0.10,
            holdout_fraction=0.20,
            run_id="plateau-source-restage",
        )
        self.assertEqual(restage["status"], "already-staged")
        self.assertEqual(restage["duplicate_suppression"]["suppressed_existing_cohort_count"], 1)
        self.assertTrue(restage["duplicate_suppression"]["suppresses_new_stage_action"])

        features = dict(action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"plateau-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            category="tool-result",
            workflow_phase="tool-execution",
            text_chars=96_000,
            cost=0.60,
            baseline=0.72,
            crunch_extra={
                "changed": True,
                "before_chars": 96_000,
                "after_chars": 86_000,
                "saved_chars": 10_000,
                "tokens_saved_est": 2_500,
                "request_shape_repeated_context_canary": selected["applied"],
            },
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            category="tool-result",
            workflow_phase="tool-execution",
            text_chars=96_000,
            cost=0.72,
            baseline=0.72,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )

        impact = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="plateau-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertEqual(impact["summary"]["applied_count"], 1)
        self.assertEqual(impact["summary"]["holdout_count"], 1)
        self.assertEqual(impact["summary"]["safety_stop_count"], 0)
        self.assertEqual(impact["summary"]["rollback_count"], 0)
        self.assertGreater(impact["summary"]["saved_tokens"], 0)
        self.assertGreater(impact["summary"]["saved_usd"], 0)
        self.assertFalse(impact["privacy"]["provider_calls_made"])
        self.assertFalse(impact["privacy"]["managed_server_calls_made"])
        rendered = json.dumps({"stage": stage, "restage": restage, "impact": impact}, sort_keys=True)
        self.assertNotIn("raw-session-id-must-not-leak", rendered)
        self.assertNotIn(str(rules_path), rendered)

    def test_crunch_canary_stage_reports_activation_ready_rollup_selection_and_skips(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                actual_input_tokens=100,
                cost=0.10,
                baseline=0.10,
            )
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                category="tool-heavy",
                workflow_phase="summary",
                text_chars=132_000,
                actual_input_tokens=9_000,
                cost=0.10,
                baseline=0.10,
            )

        rules_path = Path(self.tmpdir.name) / "missing-crunch-rules.yaml"
        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="activation-ready-selection",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=rules_path,
            max_new_canaries=1,
        )

        self.assertEqual(report["status"], "staged")
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertFalse(rules_path.exists())
        self.assertTrue(report["acceptance"]["has_activation_ready_rollup_selection"])
        self.assertTrue(report["acceptance"]["drafts_only_activation_ready_rollups"])
        self.assertTrue(report["acceptance"]["reports_skipped_rollup_reasons"])

        selection = report["activation_ready_rollup_selection"]
        self.assertEqual(selection["schema"], "tokenclaw.request_shape_crunch_canary_stage_rollup_selection.v1")
        self.assertEqual(selection["activation_ready_cohort_count"], 2)
        self.assertEqual(selection["drafted_count"], 1)
        self.assertEqual(selection["skipped_count"], 1)
        self.assertEqual(selection["target_local_policy_section"], "crunch.rules")
        self.assertEqual(selection["target_local_rule_file"], "crunch_rules.yaml")
        self.assertFalse(selection["policy_files_written"])
        skipped_reasons = {item["value"]: item["count"] for item in selection["skipped_reason_breakdown"]}
        self.assertIn("stage-action-limit-reached", skipped_reasons)

        drafted = next(row for row in selection["rows"] if row["state"] == "drafted")
        skipped = next(row for row in selection["rows"] if row["state"] == "skipped")
        self.assertTrue(drafted["selected_for_stage"])
        self.assertEqual(drafted["activation_readiness"], "activation-ready")
        self.assertEqual(drafted["canary_fraction"], 0.05)
        self.assertEqual(drafted["holdout_fraction"], 0.2)
        self.assertEqual(drafted["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertEqual(drafted["source_evidence_schema"], "tokenclaw.request_shape_rollup_row.v1")
        self.assertFalse(skipped["selected_for_stage"])
        self.assertEqual(skipped["activation_readiness"], "activation-ready")
        self.assertEqual(skipped["skip_reason"], "stage-action-limit-reached")

        action = report["top_stage_action"]
        self.assertEqual(action["source_readiness"], "activation-ready")
        self.assertIn("measurement-ready", action["source_readiness_aliases"])
        self.assertEqual(action["target_local_policy_section"], "crunch.rules")
        self.assertEqual(action["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(action["source_evidence_schema"], "tokenclaw.request_shape_rollup_row.v1")
        self.assertTrue(action["privacy"]["metadata_only"])
        self.assertTrue(action["privacy"]["aggregate_only"])
        self.assertFalse(action["privacy"]["raw_prompts_included"])
        self.assertFalse(action["privacy"]["provider_bodies_included"])
        self.assertFalse(action["privacy"]["request_ids_included"])
        self.assertFalse(action["privacy"]["session_ids_included"])
        self.assertFalse(action["privacy"]["file_paths_included"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            str(rules_path),
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_cli_apply_writes_file_backed_local_rule(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=cost,
                baseline=cost,
            )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "cli-2026-06-15-stage-apply",
                "--rollout-fraction",
                "0.05",
                "--holdout-fraction",
                "0.20",
                "--apply",
                "--rules-path",
                str(rules_path),
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "staged-and-applied")
        self.assertFalse(payload["dry_run"])
        self.assertFalse(payload["read_only"])
        self.assertEqual(payload["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(payload["staged_canary_count"], 1)
        self.assertTrue(payload["acceptance"]["stages_one_repeated_context_crunch_canary"])
        self.assertTrue(payload["acceptance"]["has_projected_lifecycle_split"])
        apply_result = payload["apply_result"]
        self.assertTrue(apply_result["ok"])
        self.assertTrue(apply_result["wrote_policy_files"])
        self.assertFalse(apply_result["rules_path_included"])
        self.assertEqual(apply_result["target_local_policy"], "crunch_rules")
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        canary = rules["request_shape_repeated_context_canaries"]["rules"][0]
        self.assertEqual(canary["id"], payload["top_stage_action"]["policy_id"])
        self.assertEqual(canary["cohort_id"], payload["top_stage_action"]["cohort_id"])
        self.assertEqual(canary["conditions"]["provider_family"], "anthropic")
        self.assertEqual(canary["conditions"]["category"], "tool-result")
        self.assertEqual(canary["conditions"]["workflow_phase"], "thinking")
        self.assertEqual(canary["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(canary["rollout"]["canary_fraction"], 0.05)
        self.assertEqual(canary["rollout"]["holdout_fraction"], 0.2)
        self.assertEqual(canary["source_evidence_schema"], "tokenclaw.request_shape_rollup_row.v1")
        self.assertIn("tokenclaw.request_shape_follow_up_candidates.v1", canary["source_evidence_schemas"])
        self.assertIn("tokenclaw.request_shape_crunch_opportunity_dry_run.v1", canary["source_evidence_schemas"])
        self.assertEqual(canary["local_only_reason"], "file-backed-local-policy-no-managed-dependency")
        self.assertIn("thinking-routing-guard", canary["evidence_blocker_codes"])
        self.assertIn("tool-call-cache-disabled", canary["evidence_blocker_codes"])
        self.assertIn("unsupported-streaming-shape", canary["evidence_blocker_codes"])
        self.assertEqual(canary["projected_saved_tokens"], payload["top_stage_action"]["projected_saved_tokens"])
        self.assertEqual(canary["projected_saved_usd"], payload["top_stage_action"]["projected_saved_usd"])
        self.assertEqual(canary["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertEqual(canary["rollback_metadata"]["target_policy_id"], payload["top_stage_action"]["policy_id"])
        self.assertEqual(apply_result["rollback_metadata"]["target_cohort_id"], payload["top_stage_action"]["cohort_id"])
        self.assertTrue(canary["safety_gates"]["local_file_backed"])
        self.assertTrue(canary["safety_gates"]["local_only"])
        self.assertFalse(canary["safety_gates"]["tool_call_cache_enabled"])
        self.assertFalse(canary["safety_gates"]["tool_call_cache_enablement_allowed"])
        self.assertFalse(canary["safety_gates"]["request_ids_included"])
        self.assertFalse(canary["safety_gates"]["session_ids_included"])
        self.assertFalse(canary["safety_gates"]["cache_keys_included"])
        self.assertFalse(canary["safety_gates"]["file_paths_included"])
        self.assertFalse(canary["safety_gates"]["tool_payloads_included"])
        self.assertTrue(canary["privacy"]["metadata_only"])
        self.assertTrue(canary["privacy"]["aggregate_only"])
        self.assertFalse(canary["privacy"]["raw_prompts_included"])
        self.assertFalse(canary["privacy"]["provider_bodies_included"])
        self.assertFalse(canary["privacy"]["request_ids_included"])
        self.assertFalse(canary["privacy"]["session_ids_included"])
        self.assertFalse(canary["privacy"]["cache_keys_included"])
        self.assertFalse(canary["privacy"]["file_paths_included"])
        self.assertFalse(canary["privacy"]["tool_payloads_included"])

        rendered = stdout.getvalue()
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            str(rules_path),
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_filter_reports_already_staged_anthropic_messages_cohort(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                actual_input_tokens=100,
                cost=0.10,
                baseline=0.10,
            )
        for _ in range(2):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=80_000,
                actual_input_tokens=100,
                cost=0.08,
                baseline=0.08,
            )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        first_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="targeted-stage-before-rule",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
            rules_path=rules_path,
        )
        target_action = next(
            action
            for action in first_report["stage_actions"]
            if action["conditions"]["endpoint"] == "messages"
            and action["conditions"]["category"] == "tool-result"
            and action["conditions"]["workflow_phase"] == "thinking"
            and action["conditions"]["text_bucket"] == "gte_128k_chars"
            and action["conditions"]["token_bucket"] == "lt_500_tokens"
        )
        apply_result = apply_request_shape_crunch_canary_action(target_action, rules_path=rules_path)
        self.assertTrue(apply_result["ok"])

        cohort_filter = {
            "provider_family": "anthropic",
            "source_surface": "anthropic_messages",
            "endpoint": "messages",
            "category": "tool-result",
            "workflow_phase": "thinking",
            "text_bucket": "gte_128k_chars",
            "token_bucket": "lt_500_tokens",
            "cache_status": "skipped",
            "routing_status": "passthrough",
            "stream": True,
            "has_tools": True,
        }
        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="targeted-stage-after-rule",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
            rules_path=rules_path,
            cohort_filter=cohort_filter,
        )

        self.assertEqual(report["status"], "already-staged")
        self.assertTrue(report["ok"])
        self.assertEqual(report["staged_canary_count"], 0)
        self.assertEqual(report["already_staged_canary_count"], 1)
        self.assertEqual(report["reported_canary_count"], 1)
        self.assertEqual(report["stage_actions"], [])
        self.assertEqual(report["target_local_policy_section"], "crunch.rules")
        self.assertEqual(report["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(report["top_reported_canary"]["cohort_id"], target_action["cohort_id"])
        self.assertEqual(report["top_reported_canary"]["row_count"], 3)
        self.assertEqual(report["top_reported_canary"]["token_bucket"], "lt_500_tokens")
        self.assertEqual(
            report["top_reported_canary"]["duplicate_suppression"]["reason"],
            "matching-repeated-context-crunch-canary-already-staged-in-local-policy",
        )
        self.assertEqual(report["duplicate_suppression"]["suppressed_existing_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 0)
        self.assertEqual(report["duplicate_suppression"]["target_local_policy_section"], "crunch.rules")
        self.assertEqual(report["duplicate_suppression"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(report["acceptance"]["reports_one_new_or_existing_repeated_context_crunch_canary"])
        self.assertTrue(report["acceptance"]["has_holdout_metadata"])
        self.assertTrue(report["acceptance"]["has_file_backed_target"])
        self.assertTrue(report["acceptance"]["has_projected_lifecycle_split"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertNotIn(str(rules_path), json.dumps(report, sort_keys=True))

    def test_crunch_canary_stage_cli_filter_reports_already_staged_target(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                actual_input_tokens=100,
                cost=0.10,
                baseline=0.10,
            )
        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        first_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="targeted-cli-before-rule",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
            rules_path=rules_path,
        )
        apply_result = apply_request_shape_crunch_canary_action(first_report["top_stage_action"], rules_path=rules_path)
        self.assertTrue(apply_result["ok"])

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_stage_cli(
            [
                "--db",
                self.db_path,
                "--limit",
                "20",
                "--run-id",
                "targeted-cli-after-rule",
                "--rules-path",
                str(rules_path),
                "--cohort-provider-family",
                "anthropic",
                "--cohort-source-surface",
                "anthropic_messages",
                "--cohort-endpoint",
                "messages",
                "--cohort-category",
                "tool-result",
                "--cohort-workflow-phase",
                "thinking",
                "--cohort-text-bucket",
                "gte_128k_chars",
                "--cohort-token-bucket",
                "lt_500_tokens",
                "--cohort-cache-status",
                "skipped",
                "--cohort-routing-status",
                "passthrough",
                "--cohort-stream",
                "true",
                "--cohort-has-tools",
                "true",
            ],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "already-staged")
        self.assertEqual(payload["staged_canary_count"], 0)
        self.assertEqual(payload["already_staged_canary_count"], 1)
        self.assertEqual(payload["reported_canary_count"], 1)
        self.assertEqual(payload["target_local_policy_section"], "crunch.rules")
        self.assertEqual(payload["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(payload["top_reported_canary"]["endpoint"], "messages")
        self.assertEqual(payload["top_reported_canary"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(payload["top_reported_canary"]["token_bucket"], "lt_500_tokens")
        self.assertTrue(payload["acceptance"]["reports_one_new_or_existing_repeated_context_crunch_canary"])
        self.assertTrue(payload["acceptance"]["has_holdout_metadata"])
        self.assertTrue(payload["acceptance"]["has_file_backed_target"])
        self.assertNotIn(str(rules_path), stdout.getvalue())

    def test_crunch_canary_stage_skips_existing_local_rule_and_stages_next_cohort(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=0.20,
                baseline=0.20,
            )
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=80_000,
                cost=0.08,
                baseline=0.08,
            )

        first_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="first-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=Path(self.tmpdir.name) / "missing-crunch-rules.yaml",
        )
        covered_action = first_report["top_stage_action"]
        self.assertEqual(covered_action["conditions"]["text_bucket"], "gte_128k_chars")

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "request_shape_repeated_context_canaries": {
                        "enabled": True,
                        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
                        "rules": [
                            {
                                "id": covered_action["policy_id"],
                                "enabled": True,
                                "policy_source": "local-manual",
                                "cohort_id": covered_action["cohort_id"],
                                "conditions": covered_action["conditions"],
                                "rollout": {
                                    "canary_enabled": True,
                                    "canary_fraction": 0.05,
                                    "holdout_fraction": 0.20,
                                },
                            }
                        ],
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="second-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=rules_path,
        )

        self.assertEqual(report["status"], "staged")
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertEqual(report["top_cohort"]["text_bucket"], "gte_128k_chars")
        self.assertEqual(report["top_cohort"]["duplicate_suppression"]["reason"], "matching-repeated-context-crunch-canary-already-staged-in-local-policy")
        self.assertEqual(report["top_stage_action"]["conditions"]["text_bucket"], "32k_128k_chars")
        self.assertEqual(report["top_stage_cohort"]["cohort_id"], report["top_stage_action"]["cohort_id"])
        self.assertNotEqual(report["top_stage_action"]["cohort_id"], covered_action["cohort_id"])
        self.assertEqual(report["duplicate_suppression"]["suppressed_existing_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 1)
        self.assertFalse(report["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertNotIn(str(rules_path), json.dumps(report, sort_keys=True))

    def test_crunch_canary_stage_suppresses_active_max_rollout_and_stages_unsuppressed(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=0.20,
                baseline=0.20,
            )
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=80_000,
                cost=0.08,
                baseline=0.08,
            )

        first_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="active-max-stage-first",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
            rules_path=Path(self.tmpdir.name) / "missing-crunch-rules.yaml",
        )
        active_action = first_report["top_stage_action"]
        self.assertEqual(active_action["conditions"]["text_bucket"], "gte_128k_chars")

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump(
                {
                    "request_shape_repeated_context_canaries": {
                        "enabled": True,
                        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
                        "rules": [
                            {
                                "id": active_action["policy_id"],
                                "enabled": True,
                                "policy_source": "local-manual",
                                "cohort_id": active_action["cohort_id"],
                                "source_evidence_schema": active_action["source_evidence_schema"],
                                "conditions": active_action["conditions"],
                                "rollout": {
                                    "canary_enabled": True,
                                    "canary_fraction": 0.30,
                                    "holdout_fraction": 0.10,
                                },
                                "safety_gates": {"max_rollout_fraction": 0.30},
                                "policy_decision": {
                                    "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                    "decision_id": "request-shape-crunch-policy-decision:test-max-rollout",
                                    "source_evidence_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                    "decision": "widen",
                                    "graduation_decision": "widen",
                                    "applied_count": 107,
                                    "holdout_count": 40,
                                    "observed_saved_tokens": 8606129,
                                    "observed_saved_usd": 25.818387,
                                    "error_rate_delta": 0.0,
                                    "retry_rate_delta": 0.0,
                                    "fallback_rate_delta": 0.0,
                                    "safety_stop_state": "none",
                                    "previous_canary_fraction": 0.20,
                                    "widened_canary_fraction": 0.30,
                                    "holdout_fraction": 0.10,
                                    "metadata_only": True,
                                    "aggregate_only": True,
                                },
                                "projected_saved_chars": active_action["projected_saved_chars"],
                                "projected_saved_tokens": active_action["projected_saved_tokens"],
                                "projected_saved_usd": active_action["projected_saved_usd"],
                            }
                        ],
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="active-max-stage-second",
            rollout_fraction=0.10,
            holdout_fraction=0.10,
            rules_path=rules_path,
        )

        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertEqual(report["top_stage_action"]["conditions"]["text_bucket"], "32k_128k_chars")
        self.assertNotEqual(report["top_stage_action"]["cohort_id"], active_action["cohort_id"])
        self.assertEqual(report["top_cohort"]["duplicate_suppression"]["reason"], "repeated-context-crunch-active-at-max-rollout")
        self.assertTrue(report["top_cohort"]["duplicate_suppression"]["active_at_max_rollout"])
        self.assertEqual(report["top_cohort"]["duplicate_suppression"]["matching_max_rollout_fraction"], 0.30)
        self.assertEqual(
            report["top_cohort"]["duplicate_suppression"]["matching_policy_decision"]["source_evidence_schema"],
            "tokenclaw.request_shape_crunch_policy_decision.v1",
        )
        self.assertEqual(report["duplicate_suppression"]["suppressed_existing_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["active_max_rollout_suppressed_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["newly_staged_cohort_count"], 1)
        self.assertFalse(report["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertTrue(report["acceptance"]["does_not_restage_suppressed_or_existing_widened_cohorts"])
        self.assertTrue(report["privacy"]["metadata_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn(str(rules_path), rendered)

    def test_crunch_canary_stage_stages_all_remaining_unsuppressed_cohorts_in_one_run(self) -> None:
        for text_chars, workflow_phase in (
            (20_000, "thinking"),
            (80_000, "thinking"),
            (132_000, "thinking"),
        ):
            for _ in range(3):
                self._log_call(
                    stream=1,
                    has_tools=True,
                    cache_status="skipped",
                    cache_reason="streaming tools-disabled",
                    routing_reason="keep requested model for thinking request",
                    workflow_phase=workflow_phase,
                    text_chars=text_chars,
                    cost=0.10,
                    baseline=0.10,
                )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=50,
            run_id="multi-cohort-stage-first",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=rules_path,
        )

        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertEqual(report["staged_canary_count"], 3)
        self.assertEqual(len(report["stage_actions"]), 3)
        self.assertEqual(report["duplicate_suppression"]["suppressed_existing_cohort_count"], 0)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 3)
        self.assertEqual(report["duplicate_suppression"]["newly_staged_cohort_count"], 3)
        self.assertFalse(report["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertTrue(report["acceptance"]["stages_all_unsuppressed_cohorts_within_bound"])
        self.assertTrue(report["acceptance"]["does_not_restage_suppressed_or_existing_widened_cohorts"])
        cohort_ids = {action["cohort_id"] for action in report["stage_actions"]}
        self.assertEqual(len(cohort_ids), 3)
        text_buckets = {action["conditions"]["text_bucket"] for action in report["stage_actions"]}
        self.assertEqual(text_buckets, {"8k_32k_chars", "32k_128k_chars", "gte_128k_chars"})
        for action in report["stage_actions"]:
            self.assertGreater(action["projected_saved_tokens"], 0)
            self.assertEqual(action["target_local_policy"], "crunch_rules")

        apply_result = apply_request_shape_crunch_canary_actions(report["stage_actions"], rules_path=rules_path)
        self.assertTrue(apply_result["ok"])
        self.assertEqual(apply_result["applied_count"], 3)
        self.assertEqual(apply_result["failed_count"], 0)
        self.assertTrue(apply_result["wrote_policy_files"])
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        staged_rules = rules["request_shape_repeated_context_canaries"]["rules"]
        self.assertEqual(len(staged_rules), 3)
        self.assertEqual({rule["id"] for rule in staged_rules}, {action["policy_id"] for action in report["stage_actions"]})

        second_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=50,
            run_id="multi-cohort-stage-second",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=rules_path,
        )

        self.assertEqual(second_report["status"], "already-staged")
        self.assertTrue(second_report["ok"])
        self.assertEqual(second_report["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(second_report["staged_canary_count"], 0)
        self.assertEqual(second_report["stage_actions"], [])
        self.assertEqual(second_report["duplicate_suppression"]["suppressed_existing_cohort_count"], 3)
        self.assertEqual(second_report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 0)
        self.assertEqual(second_report["duplicate_suppression"]["newly_staged_cohort_count"], 0)
        self.assertTrue(second_report["duplicate_suppression"]["suppresses_new_stage_action"])

        second_apply_result = apply_request_shape_crunch_canary_actions(second_report["stage_actions"], rules_path=rules_path)
        self.assertEqual(second_apply_result["applied_count"], 0)
        rules_after_idempotent_rerun = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rules_after_idempotent_rerun["request_shape_repeated_context_canaries"]["rules"]), 3)

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_bounds_new_canaries_per_run(self) -> None:
        for text_chars, workflow_phase in (
            (20_000, "thinking"),
            (80_000, "thinking"),
            (132_000, "thinking"),
        ):
            for _ in range(3):
                self._log_call(
                    stream=1,
                    has_tools=True,
                    cache_status="skipped",
                    cache_reason="streaming tools-disabled",
                    routing_reason="keep requested model for thinking request",
                    workflow_phase=workflow_phase,
                    text_chars=text_chars,
                    cost=0.10,
                    baseline=0.10,
                )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=50,
            run_id="bounded-multi-cohort-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=Path(self.tmpdir.name) / "missing-crunch-rules.yaml",
            max_new_canaries=2,
        )

        self.assertEqual(report["status"], "staged")
        self.assertEqual(report["staged_canary_count"], 2)
        self.assertEqual(len(report["stage_actions"]), 2)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 3)
        self.assertEqual(report["duplicate_suppression"]["newly_staged_cohort_count"], 2)
        self.assertEqual(report["duplicate_suppression"]["stage_action_limit"], 2)
        self.assertTrue(report["acceptance"]["stages_all_unsuppressed_cohorts_within_bound"])

    def test_crunch_canary_stage_applies_ten_unsuppressed_with_existing_suppression(self) -> None:
        shapes = [
            ("anthropic", "anthropic_messages", "messages", "tool-result", "thinking", True, True, 132_000, 100, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "messages", "tool-result", "thinking", True, True, 80_000, 100, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "messages", "tool-heavy", "thinking", True, True, 132_000, 100, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "messages", "tool-heavy", "summary", True, True, 132_000, 9_000, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "unknown", "tool-heavy", "thinking", True, True, 132_000, 100, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "unknown", "tool-heavy", "summary", True, True, 132_000, 9_000, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "unknown", "tool-heavy", "thinking", True, True, 80_000, 100, "skipped", "passthrough"),
            ("anthropic", "anthropic_messages", "messages", "tool-heavy", "thinking", True, True, 80_000, 100, "skipped", "passthrough"),
            ("openai", "openai_responses", "responses", "tool-light", "tool-light", False, True, 20_000, 3_000, "skipped", "passthrough"),
            ("openai", "openai_responses", "responses", "chat", "chat", False, False, 20_000, 3_000, "miss", "passthrough"),
            ("openai", "openai_responses", "responses", "tool-light", "tool-light", False, True, 20_000, 3_000, "skipped", "routed"),
        ]
        for provider, surface, endpoint, category, phase, stream, has_tools, text_chars, input_tokens, cache_status, routing_status in shapes:
            for _ in range(2):
                self._log_call(
                    provider=provider,
                    path="/v1/responses" if provider == "openai" else "/v1/messages",
                    source_surface=surface,
                    endpoint=endpoint,
                    requested_model="gpt-5.4" if provider == "openai" else "claude-sonnet-4-6",
                    routed_model="gpt-5.4-mini" if routing_status == "routed" else ("gpt-5.4" if provider == "openai" else "claude-sonnet-4-6"),
                    requested_model_family="gpt-5" if provider == "openai" else "claude-sonnet",
                    routed_model_family="gpt-5" if provider == "openai" else "claude-sonnet",
                    category=category,
                    workflow_phase=phase,
                    stream=1 if stream else 0,
                    has_tools=has_tools,
                    cache_status=cache_status,
                    cache_reason="exact-miss" if cache_status == "miss" else "streaming tools-disabled",
                    routing_reason="routed" if routing_status == "routed" else "keep requested model",
                    routing_extra={"routing_status": routing_status},
                    text_chars=text_chars,
                    actual_input_tokens=input_tokens,
                    cost=0.10,
                    baseline=0.10,
                )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        initial = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=100,
            run_id="stage-eleven-before-suppression",
            rules_path=rules_path,
            max_new_canaries=11,
        )
        self.assertEqual(initial["staged_canary_count"], 11)

        existing = initial["stage_actions"][0]
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump(
                {
                    "request_shape_repeated_context_canaries": {
                        "enabled": True,
                        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
                        "rules": [
                            {
                                "id": existing["policy_id"],
                                "enabled": True,
                                "policy_source": "local-manual",
                                "cohort_id": existing["cohort_id"],
                                "source_evidence_schema": existing["source_evidence_schema"],
                                "source_evidence_schemas": existing["source_evidence_schemas"],
                                "local_only_reason": existing["local_only_reason"],
                                "evidence_blocker_codes": existing["evidence_blocker_codes"],
                                "conditions": existing["conditions"],
                                "rollout": {
                                    "canary_enabled": True,
                                    "canary_fraction": 0.10,
                                    "holdout_fraction": 0.10,
                                },
                                "projected_saved_chars": existing["projected_saved_chars"],
                                "projected_saved_tokens": existing["projected_saved_tokens"],
                                "projected_saved_usd": existing["projected_saved_usd"],
                            }
                        ],
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=100,
            run_id="stage-ten-after-suppression",
            rules_path=rules_path,
            max_new_canaries=10,
        )

        self.assertEqual(report["status"], "staged")
        self.assertEqual(report["staged_canary_count"], 10)
        self.assertEqual(len(report["stage_actions"]), 10)
        self.assertEqual(report["duplicate_suppression"]["suppressed_existing_cohort_count"], 1)
        self.assertEqual(report["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 10)
        self.assertEqual(report["duplicate_suppression"]["newly_staged_cohort_count"], 10)
        self.assertEqual(report["duplicate_suppression"]["stage_action_limit"], 10)
        self.assertTrue(report["acceptance"]["stages_all_unsuppressed_cohorts_within_bound"])
        self.assertTrue(report["acceptance"]["does_not_restage_suppressed_or_existing_widened_cohorts"])
        self.assertNotIn(existing["cohort_id"], {action["cohort_id"] for action in report["stage_actions"]})
        self.assertGreater(report["source_report"]["crunch_opportunity_summary"]["projected_saved_tokens"], 0)
        self.assertGreater(report["source_report"]["crunch_opportunity_summary"]["projected_saved_usd"], 0)

        apply_result = apply_request_shape_crunch_canary_actions(report["stage_actions"], rules_path=rules_path)
        self.assertTrue(apply_result["ok"])
        self.assertEqual(apply_result["schema"], "tokenclaw.request_shape_crunch_canary_apply_batch.v1")
        self.assertEqual(apply_result["applied_count"], 10)
        self.assertEqual(apply_result["failed_count"], 0)
        self.assertTrue(apply_result["wrote_policy_files"])
        self.assertFalse(apply_result["rules_path_included"])
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        staged_rules = rules["request_shape_repeated_context_canaries"]["rules"]
        self.assertEqual(len(staged_rules), 11)
        self.assertEqual({rule["id"] for rule in staged_rules}, {existing["policy_id"], *apply_result["policy_ids"]})
        for rule in staged_rules:
            self.assertEqual(rule["policy_source"], "local-manual")
            self.assertEqual(rule["source_evidence_schema"], "tokenclaw.request_shape_rollup_row.v1")
            self.assertGreater(rule["projected_saved_tokens"], 0)
            self.assertGreater(rule["projected_saved_usd"], 0)
            self.assertEqual(rule["rollout"]["holdout_fraction"], 0.1)

        review = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=100,
            run_id="stage-ten-after-apply-review",
            rules_path=rules_path,
            max_new_canaries=10,
        )
        self.assertEqual(review["status"], "already-staged")
        self.assertTrue(review["ok"])
        self.assertEqual(review["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(review["staged_canary_count"], 0)
        self.assertEqual(review["duplicate_suppression"]["suppressed_existing_cohort_count"], 11)
        self.assertEqual(review["duplicate_suppression"]["stageable_unsuppressed_cohort_count"], 0)
        self.assertTrue(review["duplicate_suppression"]["suppresses_new_stage_action"])

        rendered = json.dumps({"report": report, "apply": apply_result, "rules": staged_rules}, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            str(rules_path),
        ):
            self.assertNotIn(forbidden, rendered)

    def test_crunch_canary_stage_prefers_unsuppressed_stage_action_over_existing_measurement(self) -> None:
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=80_000,
                cost=0.08,
                baseline=0.08,
            )
        first_report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="measurement-and-stage-first",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=Path(self.tmpdir.name) / "missing-crunch-rules.yaml",
        )
        existing_action = first_report["top_stage_action"]
        features = dict(existing_action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(
                existing_action,
                {**features, "cohort_sample_id": f"sample-{index}"},
            )
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming tools-disabled",
            routing_reason="keep requested model for thinking request",
            workflow_phase="thinking",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            crunch_extra={"request_shape_repeated_context_canary": selected["applied"]},
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming tools-disabled",
            routing_reason="keep requested model for thinking request",
            workflow_phase="thinking",
            text_chars=80_000,
            cost=0.07,
            baseline=0.07,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )
        for _ in range(3):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=0.20,
                baseline=0.20,
            )

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        apply_result = apply_request_shape_crunch_canary_action(existing_action, rules_path=rules_path)
        self.assertTrue(apply_result["ok"])

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="measurement-and-stage-second",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
            rules_path=rules_path,
        )

        self.assertEqual(report["status"], "staged")
        self.assertTrue(report["ok"])
        self.assertEqual(report["staged_canary_count"], 1)
        self.assertEqual(report["top_stage_action"]["conditions"]["text_bucket"], "gte_128k_chars")
        self.assertNotEqual(report["top_stage_action"]["cohort_id"], existing_action["cohort_id"])
        source = report["source_report"]["activation_follow_up"]
        self.assertEqual(source["activation_state"], "activation-ready")
        self.assertEqual(source["next_action"], "stage-repeated-context-crunch-canary")
        self.assertTrue(source["canary_already_staged"])
        self.assertTrue(source["canary_already_applied"])
        self.assertFalse(source["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertGreaterEqual(
            source["duplicate_suppression"]["suppressed_existing_cohort_count"],
            1,
        )
        self.assertGreaterEqual(
            source["duplicate_suppression"]["stageable_unsuppressed_cohort_count"],
            1,
        )
        self.assertEqual(
            report["duplicate_suppression"]["schema"],
            "tokenclaw.request_shape_crunch_stage_duplicate_suppression_summary.v1",
        )
        self.assertFalse(report["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertNotIn(str(rules_path), json.dumps(report, sort_keys=True))

    def test_crunch_canary_stage_keeps_safety_stopped_cohort_out_of_applied_holdout_split(self) -> None:
        safety_lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "local-repeated-context-crunch-canary-safety",
            "cohort_id": "request-shape-crunch:safety",
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }
        for _ in range(2):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming tools-disabled",
                routing_reason="keep requested model for thinking request",
                workflow_phase="thinking",
                text_chars=132_000,
                cost=0.08,
                baseline=0.08,
                crunch_extra={"request_shape_repeated_context_canary": safety_lifecycle},
            )

        report = build_request_shape_crunch_canary_stage_report(
            self.store,
            limit=20,
            run_id="2026-06-15-crunch-canary-safety-stage",
            rollout_fraction=0.05,
            holdout_fraction=0.20,
        )

        self.assertEqual(report["status"], "no-stageable-cohort")
        self.assertFalse(report["ok"])
        self.assertEqual(report["staged_canary_count"], 0)
        self.assertTrue(report["acceptance"]["unsafe_or_stale_cohorts_remain_skipped"])
        projection = report["stage_lifecycle_projection"]
        self.assertEqual(projection["projected_canary_applied_count"], 0)
        self.assertEqual(projection["projected_canary_holdout_count"], 0)
        self.assertEqual(projection["projected_safety_stopped_count"], 2)
        self.assertEqual(projection["skipped_or_safety_reasons"][0]["value"], "repeated-context-crunch-canary-safety-stopped")
        self.assertEqual(report["cohort_lifecycle_projections"][0]["status"], "safety-stopped")
        self.assertEqual(report["cohort_lifecycle_projections"][0]["reason"], "repeated-context-crunch-canary-safety-stopped")
        self.assertTrue(report["cohort_lifecycle_projections"][0]["privacy"]["metadata_only"])
        self.assertTrue(report["cohort_lifecycle_projections"][0]["privacy"]["aggregate_only"])

    def test_request_shape_crunch_canary_apply_writes_local_rule_and_lifecycle_metadata(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-canary-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]

        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump({"enabled": True, "unrelated_section": {"keep": True}}),
            encoding="utf-8",
        )

        dry_apply = apply_request_shape_crunch_canary_action(action, rules_path=rules_path, dry_run=True)
        self.assertTrue(dry_apply["ok"])
        self.assertFalse(dry_apply["wrote_policy_files"])
        self.assertNotIn("request_shape_repeated_context_canaries", yaml.safe_load(rules_path.read_text(encoding="utf-8")))

        applied = apply_request_shape_crunch_canary_action(action, rules_path=rules_path)
        self.assertTrue(applied["ok"])
        self.assertTrue(applied["wrote_policy_files"])
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.assertEqual(rules["unrelated_section"], {"keep": True})
        canaries = rules["request_shape_repeated_context_canaries"]
        self.assertTrue(canaries["enabled"])
        self.assertEqual(canaries["rules"][0]["id"], action["policy_id"])
        self.assertEqual(canaries["rules"][0]["rollout"]["canary_fraction"], 0.1)
        self.assertEqual(canaries["rules"][0]["rollout"]["holdout_fraction"], 0.1)

        selected: dict[str, dict[str, object]] = {}
        features = dict(action["conditions"])
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)
        self.assertEqual(selected["applied"]["cohort"], "canary_applied")
        self.assertEqual(selected["holdout"]["cohort"], "canary_holdout")

        unrelated = request_shape_crunch_canary_lifecycle(action, {**features, "category": "chat", "cohort_sample_id": "unrelated"})
        self.assertEqual(unrelated["status"], "skipped")
        self.assertEqual(unrelated["reason"], "cohort-mismatch")

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            crunch_extra={"request_shape_repeated_context_canary": selected["applied"]},
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.07,
            baseline=0.07,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )
        lifecycle_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-canary-lifecycle")
        dry_run = lifecycle_report["crunch_opportunity_dry_run"]
        self.assertEqual(dry_run["status"], "canary-staged")
        self.assertEqual(dry_run["summary"]["canary_staged_cohort_count"], 1)
        self.assertEqual(dry_run["summary"]["canary_applied_rows"], 1)
        self.assertEqual(dry_run["summary"]["canary_holdout_rows"], 1)
        self.assertEqual(dry_run["summary"]["activation_state"], "measurement-required")
        self.assertEqual(dry_run["summary"]["top_next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(dry_run["activation_follow_up"]["activation_state"], "measurement-required")
        self.assertEqual(dry_run["activation_follow_up"]["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(
            dry_run["activation_follow_up"]["no_op_reason"],
            "matching-repeated-context-crunch-canary-already-staged",
        )
        self.assertTrue(dry_run["activation_follow_up"]["canary_already_staged"])
        self.assertTrue(dry_run["activation_follow_up"]["canary_already_applied"])
        self.assertTrue(dry_run["activation_follow_up"]["duplicate_suppression"]["suppresses_new_stage_action"])
        self.assertEqual(
            dry_run["activation_follow_up"]["duplicate_suppression"]["matching_local_policy"],
            "crunch_rules",
        )
        self.assertIn("missing-crunch-canary-impact-measurement", dry_run["activation_follow_up"]["missing_measurements"])
        self.assertEqual(dry_run["cohorts"][0]["readiness"], "canary-staged")
        self.assertEqual(dry_run["cohorts"][0]["reason"], "repeated-context-crunch-canary-applied-and-holdout")
        self.assertEqual(dry_run["recommended_actions"], [])

    def test_crunch_canary_impact_reports_positive_applied_against_holdout(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-impact-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        features = dict(action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.06,
            baseline=0.08,
            crunch_extra={
                "changed": True,
                "before_chars": 80_000,
                "after_chars": 72_000,
                "saved_chars": 8_000,
                "tokens_saved_est": 2_000,
                "request_shape_repeated_context_canary": selected["applied"],
            },
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertEqual(impact_report["status"], "widen-ready")
        self.assertEqual(impact_report["next_action"], "widen")
        self.assertEqual(impact_report["graduation_decision"], "widen")
        self.assertEqual(impact_report["summary"]["applied_count"], 1)
        self.assertEqual(impact_report["summary"]["holdout_count"], 1)
        self.assertEqual(impact_report["summary"]["saved_chars"], 8_000)
        self.assertEqual(impact_report["summary"]["saved_tokens"], 2_000)
        self.assertEqual(impact_report["summary"]["estimated_saved_tokens"], 2_000)
        self.assertEqual(impact_report["summary"]["captured_saved_tokens"], 2_000)
        self.assertAlmostEqual(impact_report["summary"]["captured_saved_usd"], 0.02)
        self.assertGreater(impact_report["summary"]["projected_saved_tokens"], 0)
        self.assertGreater(impact_report["summary"]["projected_saved_usd"], 0)
        self.assertEqual(impact_report["captured_savings"]["status"], "captured")
        self.assertEqual(impact_report["captured_savings"]["applied_count"], 1)
        self.assertEqual(impact_report["captured_savings"]["holdout_count"], 1)
        self.assertGreater(impact_report["summary"]["saved_usd"], 0)
        self.assertEqual(impact_report["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(impact_report["summary"]["latency_avg_delta_ms"], 0.0)
        self.assertEqual(impact_report["summary"]["fallback_count"], 0)
        self.assertEqual(impact_report["summary"]["safety_stop_count"], 0)
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "promotion-ready")
        self.assertEqual(impact_report["summary"]["promotion_ready_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "widen")
        self.assertEqual(impact_report["summary"]["top_next_action"], "widen")
        self.assertEqual(impact_report["summary"]["graduation_decision"], "widen")
        self.assertEqual(impact_report["summary"]["top_graduation_decision"], "widen")
        self.assertEqual(impact_report["summary"]["recommended_next_action"], "widen-repeated-context-crunch-canary")
        self.assertTrue(impact_report["summary"]["applied_vs_holdout_coverage"]["has_applied_coverage"])
        self.assertTrue(impact_report["summary"]["applied_vs_holdout_coverage"]["has_holdout_coverage"])
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "widen-ready")
        self.assertEqual(candidate["impact_recommendation"], "promotion-ready")
        self.assertEqual(candidate["promotion_recommendation"], "promotion-ready")
        self.assertEqual(candidate["graduation_decision"], "widen")
        self.assertEqual(candidate["recommended_next_action"], "widen-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "widen")
        self.assertIsNone(candidate["top_blocker"])
        self.assertEqual(candidate["cohorts"]["canary_applied"]["saved_chars"], 8_000)
        self.assertEqual(candidate["cohorts"]["canary_holdout"]["saved_chars"], 0)
        self.assertEqual(candidate["captured_savings"]["rule_group"], "repeated-context-conservative-dry-run")
        self.assertEqual(candidate["captured_savings"]["applied_cost_delta_usd"], 0.02)
        self.assertEqual(candidate["captured_savings"]["holdout_avg_cost_delta_usd"], 0.0)
        self.assertEqual(candidate["captured_savings"]["captured_saved_tokens"], 2_000)
        self.assertAlmostEqual(candidate["captured_savings"]["captured_saved_usd"], 0.02)
        self.assertEqual(candidate["cohorts"]["canary_applied"]["latency_avg_ms"], 125.0)
        self.assertEqual(candidate["latency_avg_delta_ms"], 0.0)
        self.assertEqual(candidate["estimated_saved_tokens"], 2_000)
        self.assertEqual(candidate["coverage"]["applied_count"], 1)
        self.assertEqual(candidate["coverage"]["holdout_count"], 1)
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "promotion-ready")
        self.assertEqual(candidate["promotion_metadata"]["graduation_decision"], "widen")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "widen")
        self.assertEqual(candidate["promotion_metadata"]["observed_saved_tokens"], 2_000)
        self.assertEqual(candidate["promotion_metadata"]["captured_saved_tokens"], 2_000)
        self.assertAlmostEqual(candidate["promotion_metadata"]["captured_saved_usd"], 0.02)
        feedback = impact_report["activation_lifecycle_feedback"]
        self.assertEqual(feedback["schema"], "tokenclaw.activation_staged_lifecycle_feedback_summary.v1")
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["action_family"], "crunch")
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["applied_count"], 1)
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["holdout_count"], 1)

        rendered = json.dumps(impact_report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-fingerprint-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(impact_report["privacy"]["metadata_only"])
        self.assertTrue(impact_report["privacy"]["aggregate_only"])
        self.assertFalse(impact_report["privacy"]["raw_request_bodies_included"])
        self.assertFalse(impact_report["privacy"]["provider_bodies_included"])
        self.assertFalse(impact_report["privacy"]["request_ids_included"])
        self.assertFalse(impact_report["privacy"]["session_ids_included"])
        self.assertFalse(impact_report["privacy"]["individual_candidate_ids_included"])

    def test_crunch_canary_impact_reports_no_applied_coverage_without_failing(self) -> None:
        report = build_request_shape_crunch_canary_impact_report([])

        self.assertEqual(report["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertTrue(report["ok"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["status"], "no-applied-coverage")
        self.assertEqual(report["next_action"], "stage-canary-first")
        self.assertEqual(report["graduation_decision"], "keep-staged")
        self.assertEqual(report["summary"]["next_action"], "stage-canary-first")
        self.assertEqual(report["summary"]["graduation_decision"], "keep-staged")
        self.assertEqual(report["summary"]["candidate_count"], 0)
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["holdout_count"], 0)
        self.assertEqual(report["summary"]["estimated_saved_tokens"], 0)
        self.assertEqual(report["summary"]["captured_saved_tokens"], 0)
        self.assertEqual(report["summary"]["captured_saved_usd"], 0.0)
        self.assertEqual(report["captured_savings"]["status"], "no-captured-savings")
        self.assertFalse(report["summary"]["applied_vs_holdout_coverage"]["has_applied_coverage"])
        self.assertIn("missing-applied-or-holdout-coverage", report["missing_measurements"])
        self.assertIn("applied-crunch-canary-coverage", report["missing_measurements"])
        self.assertIn("crunch-canary-lifecycle-metadata", report["missing_measurements"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])

    def test_crunch_canary_impact_rows_rank_metadata_only_repeated_context_measurements(self) -> None:
        impact_candidates = [
            {
                "schema": "tokenclaw.request_shape_crunch_canary_impact_candidate.v1",
                "policy_id": "raw-session-id-must-not-leak",
                "cohort_id": "/tmp/private/source.py",
                "cohort_metadata": {
                    "provider_family": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                },
                "verdict": "widen-ready",
                "impact_recommendation": "promotion-ready",
                "durable_next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
                "reason_codes": ["applied-savings-with-holdout-no-regression"],
                "observed_count": 10,
                "applied_count": 7,
                "holdout_count": 3,
                "fallback_count": 0,
                "rollback_count": 0,
                "safety_stop_count": 0,
                "saved_tokens": 12_000,
                "saved_usd": 0.036,
                "projected_saved_tokens": 12_000,
                "projected_saved_usd": 0.036,
                "coverage": {"skipped_count": 2},
            }
        ]
        activation_ready_measurements = {
            "schema": "tokenclaw.request_shape_crunch_activation_ready_measurements.v1",
            "cohorts": [
                {
                    "schema": "tokenclaw.request_shape_crunch_activation_ready_cohort_measurement.v1",
                    "rank": 1,
                    "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:measurement-required",
                    "policy_id": "local-repeated-context-crunch-canary",
                    "state": "keep-staged",
                    "readiness": "canary-staged",
                    "next_action": "measure-repeated-context-crunch-canary-impact",
                    "provider_family": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "sample_count": 42,
                    "row_count": 42,
                    "projected_saved_tokens": 8_000,
                    "projected_saved_usd": 0.024,
                    "observed_saved_tokens": 0,
                    "observed_saved_usd": 0.0,
                    "applied_count": 0,
                    "holdout_count": 0,
                    "skipped_count": 42,
                    "fallback_count": 0,
                    "error_count": 0,
                    "retry_count": 0,
                    "rollback_count": 0,
                    "safety_stop_count": 0,
                    "reason_codes": ["matching-repeated-context-crunch-canary-already-staged"],
                }
            ],
        }
        follow_up_candidates = {
            "schema": "tokenclaw.request_shape_follow_up_candidates.v1",
            "candidates": [
                {
                    "schema": "tokenclaw.request_shape_blocker_cohort.v1",
                    "rank": 1,
                    "local_action_family": "crunch",
                    "readiness_state": "measurement-required",
                    "next_action": "measure-repeated-context-crunch-canary-impact",
                    "blocker_codes": ["tool-call-cache-disabled"],
                    "sample_count": 437,
                    "row_count": 437,
                    "projected_saved_tokens": 1_826_069,
                    "projected_savings_usd": 5.478208,
                    "observed_savings_usd": 287.030118,
                    "provider_family": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "unknown",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                },
                {
                    "schema": "tokenclaw.request_shape_blocker_cohort.v1",
                    "rank": 2,
                    "local_action_family": "crunch",
                    "readiness_state": "blocked",
                    "next_action": "measure-repeated-context-crunch-canary-impact",
                    "blocker_codes": ["insufficient-repeat-evidence"],
                    "sample_count": 1,
                    "row_count": 1,
                    "provider_family": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                },
            ],
        }
        activation_evidence = {
            "schema": "tokenclaw.request_shape_crunch_activation_evidence.v1",
            "status": "active-rule-evidence-observed",
            "decision_id": "request-shape-crunch-policy-decision:raw-request-id-must-not-leak",
            "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            "duplicate_suppression": {
                "suppresses_new_activation_issue": True,
                "suppresses_generic_crunch_activation_issue": True,
                "reason": "repeated-context-crunch-full-rollout-active",
            },
            "summary": {
                "applied_count": 107,
                "holdout_count": 40,
                "skipped_count": 280,
                "fallback_count": 0,
                "rollback_count": 0,
                "safety_stop_count": 0,
                "observed_saved_tokens": 8_606_129,
                "observed_saved_usd": 25.818387,
                "post_widening_status": "post-widening-active-at-max-rollout",
                "next_action": "measure-full-rollout-repeated-context-crunch-outcomes",
            },
        }

        report = build_request_shape_crunch_canary_impact_rows_report(
            impact_candidates=impact_candidates,
            activation_ready_measurements=activation_ready_measurements,
            follow_up_candidates=follow_up_candidates,
            activation_evidence=activation_evidence,
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_crunch_canary_impact_rows.v1")
        self.assertEqual(report["status"], "ranked")
        self.assertTrue(report["acceptance"]["has_ranked_repeated_context_crunch_impact_rows"])
        self.assertTrue(report["acceptance"]["has_blocker_codes"])
        self.assertTrue(report["acceptance"]["has_local_action_family"])
        self.assertTrue(report["acceptance"]["has_readiness_state"])
        self.assertTrue(report["acceptance"]["has_next_action"])
        self.assertTrue(report["acceptance"]["has_sample_count"])
        self.assertTrue(report["acceptance"]["has_canary_counts"])
        self.assertTrue(report["acceptance"]["has_projected_and_observed_savings"])
        self.assertTrue(report["acceptance"]["emits_durable_measurement_state"])
        states = {row["measurement_state"] for row in report["rows"]}
        self.assertIn("measured", states)
        self.assertIn("measurement-required", states)
        self.assertIn("blocked", states)
        self.assertIn("superseded", states)
        top = report["top_row"]
        self.assertEqual(top["local_action_family"], "crunch")
        self.assertEqual(top["next_action"], "measure-full-rollout-repeated-context-crunch-outcomes")
        self.assertEqual(top["applied_count"], 107)
        self.assertEqual(top["holdout_count"], 40)
        self.assertEqual(top["skipped_count"], 280)
        self.assertGreater(top["observed_saved_usd"], 0)
        for row in report["rows"]:
            self.assertIn("rank", row)
            self.assertIn("blocker_codes", row)
            self.assertIn("sample_count", row)
            self.assertIn("projected_saved_usd", row)
            self.assertIn("observed_saved_usd", row)
            self.assertTrue(row["privacy"]["metadata_only"])
            self.assertTrue(row["privacy"]["aggregate_only"])

        rendered = json.dumps(report, sort_keys=True)
        for forbidden in (
            "raw prompt must not leak",
            "provider body must not leak",
            "raw response must not leak",
            "raw-session-id-must-not-leak",
            "raw-cache-key-must-not-leak",
            "raw-request-id-must-not-leak",
            "/tmp/private/source.py",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        self.assertFalse(report["privacy"]["cache_keys_included"])
        self.assertFalse(report["privacy"]["file_paths_included"])
        self.assertFalse(report["privacy"]["individual_candidate_ids_included"])

    def test_crunch_canary_impact_classifies_remaining_activation_ready_cohorts(self) -> None:
        measured_lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "local-repeated-context-crunch-canary-measured",
            "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:measured",
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "selected-canary",
            "policy_source": "local-manual",
            "source_evidence_schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            "projected_saved_tokens": 10_000,
            "projected_saved_usd": 0.03,
            "metadata_only": True,
            "aggregate_only": True,
        }
        holdout_lifecycle = dict(measured_lifecycle, status="holdout", cohort="canary_holdout")
        opportunity_report = {
            "schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            "cohorts": [
                {
                    "rank": 1,
                    "cohort_id": measured_lifecycle["cohort_id"],
                    "policy_id": measured_lifecycle["policy_id"],
                    "readiness": "canary-staged",
                    "row_count": 2,
                    "projected_saved_tokens": 10_000,
                    "projected_saved_usd": 0.03,
                    "crunch_canary_lifecycle": {"applied_count": 1, "holdout_count": 1},
                },
                {
                    "rank": 2,
                    "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:stageable",
                    "policy_id": "local-repeated-context-crunch-canary-stageable",
                    "readiness": "measurement-ready",
                    "row_count": 4,
                    "projected_saved_tokens": 8_000,
                    "projected_saved_usd": 0.024,
                    "duplicate_suppression": {"suppresses_new_stage_action": False, "suppressed": False},
                },
                {
                    "rank": 3,
                    "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:staged",
                    "policy_id": "local-repeated-context-crunch-canary-staged",
                    "readiness": "measurement-ready",
                    "row_count": 3,
                    "projected_saved_tokens": 6_000,
                    "projected_saved_usd": 0.018,
                    "duplicate_suppression": {
                        "suppresses_new_stage_action": True,
                        "suppressed": True,
                        "reason": "matching-repeated-context-crunch-canary-already-staged-in-local-policy",
                        "matching_local_policy": "crunch_rules",
                    },
                },
                {
                    "rank": 4,
                    "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:blocked",
                    "policy_id": "local-repeated-context-crunch-canary-blocked",
                    "readiness": "skipped",
                    "reason": "insufficient-repeat-evidence",
                    "blockers": ["insufficient-repeat-evidence"],
                    "row_count": 1,
                    "projected_saved_tokens": 0,
                    "projected_saved_usd": 0.0,
                },
            ],
            "recommended_actions": [
                {
                    "action_type": "stage-local-repeated-context-crunch-canary",
                    "policy_id": "local-repeated-context-crunch-canary-stageable",
                    "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:stageable",
                    "conditions": {
                        "provider_family": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "stream": True,
                        "has_tools": True,
                    },
                    "rollout_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "projected_saved_tokens": 8_000,
                    "projected_saved_usd": 0.024,
                    "source_evidence_schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
                }
            ],
        }
        base_row = {
            "created_at": utc_now(),
            "provider_family": "anthropic",
            "provider": "anthropic",
            "source_surface": "anthropic_messages",
            "endpoint": "messages",
            "category": "tool-result",
            "workflow_phase": "thinking",
            "stream": True,
            "has_tools": True,
            "text_bucket": "gte_128k_chars",
            "token_bucket": "lt_500_tokens",
            "actual_input_tokens": 20_000,
            "input_tokens_est": 20_000,
            "text_chars": 80_000,
            "cost_est_usd": 0.08,
            "status_code": 200,
            "retry_count": 0,
            "latency_ms": 125,
        }
        report = build_request_shape_crunch_canary_impact_report(
            [
                {
                    **base_row,
                    "crunch_json": stable_json(
                        {
                            "changed": True,
                            "before_chars": 80_000,
                            "after_chars": 72_000,
                            "saved_chars": 8_000,
                            "tokens_saved_est": 2_000,
                            "request_shape_repeated_context_canary": measured_lifecycle,
                        }
                    ),
                },
                {
                    **base_row,
                    "crunch_json": stable_json({"request_shape_repeated_context_canary": holdout_lifecycle}),
                },
            ],
            opportunity_report=opportunity_report,
        )

        measurements = report["activation_ready_measurements"]
        self.assertEqual(measurements["schema"], "tokenclaw.request_shape_crunch_activation_ready_measurements.v1")
        self.assertEqual(measurements["status"], "classified")
        self.assertEqual(measurements["measured_count"], 1)
        self.assertEqual(measurements["keep_staged_count"], 1)
        self.assertEqual(measurements["stageable_count"], 1)
        self.assertEqual(measurements["blocked_count"], 1)
        self.assertEqual(measurements["bounded_stage_recommendation_count"], 1)
        self.assertEqual(report["summary"]["measured_cohort_count"], 1)
        self.assertEqual(report["summary"]["keep_staged_cohort_count"], 1)
        self.assertEqual(report["summary"]["stageable_cohort_count"], 1)
        self.assertEqual(report["summary"]["blocked_cohort_count"], 1)
        states = {item["cohort_id"]: item["state"] for item in measurements["cohorts"]}
        self.assertEqual(states[measured_lifecycle["cohort_id"]], "measured")
        self.assertEqual(states["request-shape-crunch:anthropic:messages:tool-result:stageable"], "stageable")
        self.assertEqual(states["request-shape-crunch:anthropic:messages:tool-result:staged"], "keep-staged")
        self.assertEqual(states["request-shape-crunch:anthropic:messages:tool-result:blocked"], "blocked")
        staged_row = next(
            item
            for item in measurements["cohorts"]
            if item["cohort_id"] == "request-shape-crunch:anthropic:messages:tool-result:staged"
        )
        self.assertEqual(staged_row["sample_count"], 3)
        self.assertEqual(staged_row["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(staged_row["applied_count"], 0)
        self.assertEqual(staged_row["holdout_count"], 0)
        self.assertEqual(staged_row["safety_stop_count"], 0)
        self.assertIn("applied-crunch-canary-coverage", staged_row["missing_measurements"])
        stage_follow_up = measurements["bounded_stage_recommendations"][0]
        self.assertEqual(stage_follow_up["target_local_policy"], "crunch_rules")
        self.assertEqual(stage_follow_up["conditions"]["category"], "tool-result")
        self.assertTrue(stage_follow_up["privacy"]["metadata_only"])
        self.assertTrue(measurements["privacy"]["aggregate_only"])
        self.assertFalse(json.loads(json.dumps(report))["privacy"]["raw_prompts_included"])

    def test_crunch_canary_impact_measures_newly_staged_after_max_rollout_suppression(self) -> None:
        active_cohort_id = "request-shape-crunch:anthropic:messages:tool-result:active-max"
        staged_cohort_id = "request-shape-crunch:anthropic:messages:tool-result:newly-staged"
        opportunity_report = {
            "schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            "cohorts": [
                {
                    "rank": 1,
                    "cohort_id": active_cohort_id,
                    "policy_id": "local-repeated-context-crunch-canary-active-max",
                    "readiness": "canary-staged",
                    "row_count": 147,
                    "projected_saved_tokens": 8606129,
                    "projected_saved_usd": 25.818387,
                    "crunch_canary_lifecycle": {"applied_count": 107, "holdout_count": 40, "skipped_count": 280},
                    "duplicate_suppression": {
                        "suppressed": True,
                        "suppresses_new_stage_action": True,
                        "active_at_max_rollout": True,
                        "reason": "repeated-context-crunch-active-at-max-rollout",
                        "matching_local_policy": "crunch_rules",
                        "matching_policy_id": "local-repeated-context-crunch-canary-active-max",
                        "matching_max_rollout_fraction": 0.3,
                    },
                },
                {
                    "rank": 2,
                    "cohort_id": staged_cohort_id,
                    "policy_id": "local-repeated-context-crunch-canary-newly-staged",
                    "readiness": "canary-staged",
                    "row_count": 44,
                    "projected_saved_tokens": 238878,
                    "projected_saved_usd": 0.716636,
                    "crunch_canary_lifecycle": {
                        "applied_count": 7,
                        "holdout_count": 5,
                        "skipped_count": 32,
                        "fallback_count": 0,
                        "retry_count": 1,
                        "rollback_count": 0,
                        "safety_stopped_count": 0,
                    },
                    "duplicate_suppression": {
                        "suppressed": True,
                        "suppresses_new_stage_action": True,
                        "active_at_max_rollout": False,
                        "reason": "matching-repeated-context-crunch-canary-already-staged-in-local-policy",
                        "matching_local_policy": "crunch_rules",
                        "matching_policy_id": "local-repeated-context-crunch-canary-newly-staged",
                    },
                },
            ],
            "recommended_actions": [
                {
                    "action_type": "stage-local-repeated-context-crunch-canary",
                    "policy_id": "local-repeated-context-crunch-canary-active-max",
                    "cohort_id": active_cohort_id,
                    "conditions": {"category": "tool-result"},
                    "rollout_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "projected_saved_tokens": 8606129,
                    "projected_saved_usd": 25.818387,
                },
                {
                    "action_type": "stage-local-repeated-context-crunch-canary",
                    "policy_id": "local-repeated-context-crunch-canary-newly-staged",
                    "cohort_id": staged_cohort_id,
                    "conditions": {"category": "tool-result"},
                    "rollout_fraction": 0.1,
                    "holdout_fraction": 0.1,
                    "projected_saved_tokens": 238878,
                    "projected_saved_usd": 0.716636,
                },
            ],
        }

        report = build_request_shape_crunch_canary_impact_report([], opportunity_report=opportunity_report)

        self.assertEqual(report["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertEqual(report["newly_staged_measurement"]["schema"], "tokenclaw.request_shape_crunch_newly_staged_measurement.v1")
        self.assertEqual(report["newly_staged_measurement"]["status"], "measured")
        self.assertEqual(report["newly_staged_measurement"]["cohort_count"], 1)
        self.assertEqual(report["newly_staged_measurement"]["applied_count"], 7)
        self.assertEqual(report["newly_staged_measurement"]["holdout_count"], 5)
        self.assertEqual(report["newly_staged_measurement"]["skipped_count"], 32)
        self.assertEqual(report["newly_staged_measurement"]["retry_count"], 1)
        self.assertEqual(report["newly_staged_measurement"]["fallback_count"], 0)
        self.assertEqual(report["newly_staged_measurement"]["rollback_count"], 0)
        self.assertEqual(report["newly_staged_measurement"]["safety_stop_count"], 0)
        self.assertEqual(report["newly_staged_measurement"]["active_max_rollout_suppressed_count"], 1)
        self.assertEqual(report["summary"]["newly_staged_measurement_cohort_count"], 1)
        self.assertEqual(report["summary"]["newly_staged_applied_count"], 7)
        self.assertEqual(report["summary"]["newly_staged_holdout_count"], 5)
        self.assertEqual(report["activation_ready_measurements"]["bounded_stage_recommendation_count"], 0)
        self.assertEqual(report["activation_ready_measurements"]["cohorts"][0]["duplicate_suppression"]["active_at_max_rollout"], True)
        rendered = json.dumps(report, sort_keys=True)
        self.assertIn("repeated-context-crunch-active-at-max-rollout", rendered)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])

    def test_crunch_canary_impact_names_high_cost_thinking_measurement_row(self) -> None:
        cohort_id = "request-shape-crunch:anthropic:unknown:tool-result:high-cost-thinking"
        opportunity_report = {
            "schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            "cohorts": [
                {
                    "rank": 1,
                    "cohort_id": cohort_id,
                    "policy_id": "local-repeated-context-crunch-canary-thinking",
                    "readiness": "canary-staged",
                    "reason": "repeated-context-crunch-canary-applied-and-holdout",
                    "provider_family": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "unknown",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "row_count": 446,
                    "projected_saved_tokens": 1_860_651,
                    "projected_saved_chars": 7_442_604,
                    "projected_saved_usd": 5.581954,
                    "current_conservative_tokens_saved": 0,
                    "current_conservative_chars_saved": 0,
                    "current_conservative_savings_usd": 0.0,
                    "evidence_blocker_codes": [
                        "tool-call-cache-disabled",
                        "unsupported-streaming-shape",
                    ],
                    "crunch_canary_lifecycle": {
                        "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
                        "policy_id": "local-repeated-context-crunch-canary-thinking",
                        "cohort_id": cohort_id,
                        "applied_count": 99,
                        "holdout_count": 39,
                        "skipped_count": 308,
                        "fallback_count": 0,
                        "rollback_count": 0,
                        "safety_stopped_count": 0,
                        "metadata_only": True,
                        "aggregate_only": True,
                    },
                    "duplicate_suppression": {
                        "suppresses_new_stage_action": True,
                        "suppressed": True,
                        "reason": "matching-repeated-context-crunch-canary-canary-staged",
                        "matching_local_policy": "crunch_rules",
                    },
                }
            ],
            "recommended_actions": [],
        }

        report = build_request_shape_crunch_canary_impact_report([], opportunity_report=opportunity_report)

        rows = report["activation_ready_measurements"]["cohorts"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["schema"], "tokenclaw.request_shape_crunch_activation_ready_cohort_measurement.v1")
        self.assertEqual(row["cohort_id"], cohort_id)
        self.assertEqual(row["state"], "keep-staged")
        self.assertEqual(row["next_action"], "measure-repeated-context-crunch-canary-impact")
        self.assertEqual(row["provider_family"], "anthropic")
        self.assertEqual(row["source_surface"], "anthropic_messages")
        self.assertEqual(row["category"], "tool-result")
        self.assertEqual(row["workflow_phase"], "thinking")
        self.assertEqual(row["text_bucket"], "gte_128k_chars")
        self.assertEqual(row["token_bucket"], "lt_500_tokens")
        self.assertEqual(row["sample_count"], 446)
        self.assertEqual(row["applied_count"], 99)
        self.assertEqual(row["holdout_count"], 39)
        self.assertEqual(row["skipped_count"], 308)
        self.assertEqual(row["fallback_count"], 0)
        self.assertEqual(row["retry_count"], 0)
        self.assertEqual(row["rollback_count"], 0)
        self.assertEqual(row["safety_stop_count"], 0)
        self.assertEqual(row["projected_saved_tokens"], 1_860_651)
        self.assertEqual(row["projected_saved_usd"], 5.581954)
        self.assertIn("crunch-canary-impact-observed-savings", row["missing_measurements"])
        self.assertIn("tool-call-cache-disabled", row["evidence_blocker_codes"])
        self.assertTrue(row["privacy"]["metadata_only"])
        self.assertTrue(row["privacy"]["aggregate_only"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw prompt must not leak", rendered)
        self.assertFalse(report["privacy"]["request_ids_included"])

    def test_crunch_canary_impact_emits_durable_action_for_fresh_staged_holdout(self) -> None:
        lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "local-repeated-context-crunch-canary-fresh",
            "cohort_id": "request-shape-crunch:anthropic:messages:tool-result:fresh",
            "status": "holdout",
            "cohort": "canary_holdout",
            "reason": "selected-holdout",
            "policy_source": "local-manual",
            "source_evidence_schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
            "source_evidence_schemas": [
                "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
                "tokenclaw.request_shape_follow_up_candidates.v1",
            ],
            "staged_at": "2026-06-16T18:00:00+00:00",
            "projected_saved_tokens": 12_000,
            "projected_saved_usd": 0.036,
            "rollback_metadata_present": True,
            "metadata_only": True,
            "aggregate_only": True,
        }
        report = build_request_shape_crunch_canary_impact_report(
            [
                {
                    "created_at": utc_now(),
                    "provider_family": "anthropic",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "actual_input_tokens": 20_000,
                    "input_tokens_est": 20_000,
                    "text_chars": 80_000,
                    "cost_est_usd": 0.08,
                    "status_code": 200,
                    "retry_count": 0,
                    "latency_ms": 125,
                    "crunch_json": stable_json(
                        {
                            "changed": False,
                            "tokens_saved_est": 0,
                            "request_shape_repeated_context_canary": lifecycle,
                        }
                    ),
                }
            ]
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertEqual(report["summary"]["candidate_count"], 1)
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["holdout_count"], 1)
        self.assertEqual(report["summary"]["cohort_family_action_count"], 1)
        self.assertEqual(report["summary"]["freshly_staged_cohort_count"], 1)
        self.assertEqual(report["summary"]["top_durable_next_action"], "measure-more")
        action = report["cohort_family_actions"][0]
        self.assertEqual(action["schema"], "tokenclaw.request_shape_crunch_canary_cohort_family_action.v1")
        self.assertEqual(action["durable_next_action"], "measure-more")
        self.assertEqual(action["applied_count"], 0)
        self.assertEqual(action["holdout_count"], 1)
        self.assertEqual(action["projected_saved_tokens"], 12_000)
        self.assertEqual(action["projected_saved_usd"], 0.036)
        self.assertEqual(action["source_evidence_schema"], "tokenclaw.request_shape_crunch_opportunity_dry_run.v1")
        self.assertEqual(action["staged_at"], "2026-06-16T18:00:00+00:00")
        self.assertIn("applied-crunch-canary-coverage", action["missing_measurements"])
        candidate = report["candidates"][0]
        self.assertEqual(candidate["durable_next_action"], "measure-more")
        self.assertEqual(candidate["missing_measurements"], ["applied-crunch-canary-coverage"])
        self.assertTrue(candidate["rollback_metadata_present"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])

    def test_crunch_canary_impact_cli_returns_no_applied_coverage_status(self) -> None:
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_crunch_canary_impact_cli(
            ["--db", self.db_path, "--limit", "10"],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_crunch_canary_impact.v1")
        self.assertEqual(payload["status"], "no-applied-coverage")
        self.assertEqual(payload["next_action"], "stage-canary-first")
        self.assertEqual(payload["summary"]["applied_count"], 0)
        self.assertEqual(payload["summary"]["holdout_count"], 0)
        self.assertTrue(payload["source_report"]["privacy"]["metadata_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())

    def test_crunch_canary_impact_blocks_widening_on_safety_stop(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-safety-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        safety_lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": action["policy_id"],
            "cohort_id": action["cohort_id"],
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            status_code=500,
            retry_count=2,
            crunch_extra={"request_shape_repeated_context_canary": safety_lifecycle},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-safety-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["status"], "no-widen")
        self.assertEqual(impact_report["summary"]["candidate_count"], 1)
        self.assertEqual(impact_report["summary"]["applied_count"], 0)
        self.assertEqual(impact_report["summary"]["holdout_count"], 0)
        self.assertEqual(impact_report["summary"]["safety_stop_count"], 1)
        self.assertEqual(impact_report["summary"]["top_blocker_code"], "canary-safety-stopped")
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "rollback")
        self.assertEqual(impact_report["summary"]["rollback_recommended_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "rollback")
        self.assertEqual(impact_report["summary"]["graduation_decision"], "rollback")
        self.assertEqual(impact_report["summary"]["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "no-widen")
        self.assertEqual(candidate["impact_recommendation"], "rollback")
        self.assertEqual(candidate["graduation_decision"], "rollback")
        self.assertEqual(candidate["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "rollback")
        self.assertIn("canary-safety-stopped", candidate["reason_codes"])
        self.assertIn("missing-applied-coverage", candidate["reason_codes"])
        self.assertIn("missing-holdout-coverage", candidate["reason_codes"])
        self.assertEqual(candidate["top_blocker"], "canary-safety-stopped")
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "rollback")
        feedback = impact_report["activation_lifecycle_feedback"]
        self.assertEqual(feedback["cohort_lifecycle_metadata"][0]["safety_stop_count"], 1)
        self.assertIn("canary-safety-stopped", feedback["cohort_lifecycle_metadata"][0]["reason_codes"])

    def test_crunch_canary_impact_recommends_rollback_on_regressed_applied_cohort(self) -> None:
        for cost in (0.08, 0.07, 0.09):
            self._log_call(
                stream=1,
                has_tools=True,
                cache_status="skipped",
                cache_reason="streaming",
                text_chars=80_000,
                cost=cost,
                baseline=cost,
            )
        report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-regression-stage")
        action = report["crunch_opportunity_dry_run"]["recommended_actions"][0]
        features = dict(action["conditions"])
        selected: dict[str, dict[str, object]] = {}
        for index in range(5000):
            lifecycle = request_shape_crunch_canary_lifecycle(action, {**features, "cohort_sample_id": f"sample-{index}"})
            if lifecycle["status"] in {"applied", "holdout"}:
                selected.setdefault(str(lifecycle["status"]), lifecycle)
            if {"applied", "holdout"} <= set(selected):
                break
        self.assertIn("applied", selected)
        self.assertIn("holdout", selected)

        self.store.conn.execute("delete from calls")
        self.store.conn.commit()
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            status_code=500,
            latency_ms=900,
            retry_count=2,
            crunch_extra={
                "changed": True,
                "before_chars": 80_000,
                "after_chars": 72_000,
                "saved_chars": 8_000,
                "tokens_saved_est": 2_000,
                "request_shape_repeated_context_canary": selected["applied"],
            },
        )
        self._log_call(
            stream=1,
            has_tools=True,
            cache_status="skipped",
            cache_reason="streaming",
            text_chars=80_000,
            cost=0.08,
            baseline=0.08,
            latency_ms=100,
            crunch_extra={"request_shape_repeated_context_canary": selected["holdout"]},
        )

        impact_report = build_request_shape_rollups_report(self.store, limit=20, persist=False, run_id="crunch-regression-impact")[
            "crunch_canary_impact"
        ]
        self.assertEqual(impact_report["status"], "no-widen")
        self.assertEqual(impact_report["summary"]["top_impact_recommendation"], "rollback")
        self.assertEqual(impact_report["summary"]["rollback_recommended_count"], 1)
        self.assertEqual(impact_report["summary"]["next_action"], "rollback")
        self.assertEqual(impact_report["summary"]["error_rate_delta"], 1.0)
        self.assertEqual(impact_report["summary"]["retry_rate_delta"], 2.0)
        self.assertEqual(impact_report["summary"]["latency_avg_delta_ms"], 800.0)

        candidate = impact_report["candidates"][0]
        self.assertEqual(candidate["verdict"], "no-widen")
        self.assertEqual(candidate["impact_recommendation"], "rollback")
        self.assertEqual(candidate["recommended_next_action"], "rollback-repeated-context-crunch-canary")
        self.assertEqual(candidate["next_action"], "rollback")
        self.assertEqual(candidate["latency_avg_delta_ms"], 800.0)
        self.assertIn("error-rate-regression", candidate["reason_codes"])
        self.assertIn("retry-rate-regression", candidate["reason_codes"])
        self.assertEqual(candidate["promotion_metadata"]["impact_recommendation"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["next_action"], "rollback")
        self.assertEqual(candidate["promotion_metadata"]["error_rate_delta"], 1.0)
        self.assertEqual(candidate["promotion_metadata"]["retry_rate_delta"], 2.0)
        self.assertEqual(candidate["promotion_metadata"]["latency_avg_delta_ms"], 800.0)

    def test_crunch_canary_impact_emits_collect_and_keep_blocked_recommendations(self) -> None:
        def impact_row(
            *,
            policy_id: str,
            cohort_id: str,
            status: str,
            saved_tokens: int = 0,
            saved_chars: int = 0,
        ) -> dict[str, object]:
            lifecycle = {
                "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
                "policy_id": policy_id,
                "cohort_id": cohort_id,
                "status": status,
                "cohort": "canary_applied" if status == "applied" else "canary_holdout",
                "reason": status,
                "metadata_only": True,
            }
            return {
                "created_at": utc_now(),
                "provider_family": "anthropic",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "stream": True,
                "has_tools": True,
                "text_bucket": "gte_128k_chars",
                "token_bucket": "lt_500_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "actual_input_tokens": 20_000,
                "input_tokens_est": 20_000,
                "text_chars": 80_000,
                "cost_est_usd": 0.08,
                "status_code": 200,
                "retry_count": 0,
                "latency_ms": 125,
                "crunch_json": stable_json(
                    {
                        "changed": saved_tokens > 0,
                        "before_chars": 80_000,
                        "after_chars": max(0, 80_000 - saved_chars),
                        "saved_chars": saved_chars,
                        "tokens_saved_est": saved_tokens,
                        "request_shape_repeated_context_canary": lifecycle,
                    }
                ),
            }

        report = build_request_shape_crunch_canary_impact_report(
            [
                impact_row(policy_id="collect-policy", cohort_id="collect-cohort", status="applied", saved_tokens=2_000, saved_chars=8_000),
                impact_row(policy_id="blocked-policy", cohort_id="blocked-cohort", status="applied"),
                impact_row(policy_id="blocked-policy", cohort_id="blocked-cohort", status="holdout"),
            ]
        )

        by_policy = {candidate["policy_id"]: candidate for candidate in report["candidates"]}
        self.assertEqual(by_policy["collect-policy"]["impact_recommendation"], "collect-more-evidence")
        self.assertEqual(
            by_policy["collect-policy"]["recommended_next_action"],
            "collect-repeated-context-crunch-canary-impact-evidence",
        )
        self.assertEqual(by_policy["collect-policy"]["next_action"], "keep-observing")
        self.assertIn("missing-holdout-coverage", by_policy["collect-policy"]["reason_codes"])

        self.assertEqual(by_policy["blocked-policy"]["impact_recommendation"], "keep-blocked")
        self.assertEqual(
            by_policy["blocked-policy"]["recommended_next_action"],
            "keep-repeated-context-crunch-canary-blocked",
        )
        self.assertEqual(by_policy["blocked-policy"]["next_action"], "keep-observing")
        self.assertIn("no-applied-savings", by_policy["blocked-policy"]["reason_codes"])
        self.assertEqual(report["summary"]["collect_more_evidence_count"], 1)
        self.assertEqual(report["summary"]["keep_blocked_count"], 1)
        self.assertFalse(report["privacy"]["provider_bodies_included"])

    def test_crunch_policy_decision_promotes_positive_canary_with_rollback_metadata(self) -> None:
        def impact_row(status: str, *, saved_tokens: int = 0, saved_chars: int = 0, saved_usd: float = 0.0) -> dict[str, object]:
            lifecycle = {
                "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
                "policy_id": "policy-promote",
                "cohort_id": "cohort-promote",
                "status": status,
                "cohort": "canary_applied" if status == "applied" else "canary_holdout",
                "reason": status,
                "policy_source": "local-manual",
                "metadata_only": True,
            }
            crunch_json = {
                "changed": saved_tokens > 0,
                "before_chars": 80_000,
                "after_chars": max(0, 80_000 - saved_chars),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "request_shape_repeated_context_canary": lifecycle,
            }
            if saved_usd:
                crunch_json["savings_usd"] = saved_usd
            return {
                "created_at": utc_now(),
                "provider_family": "anthropic",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "stream": True,
                "has_tools": True,
                "text_bucket": "gte_128k_chars",
                "token_bucket": "lt_500_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "actual_input_tokens": 20_000,
                "input_tokens_est": 20_000,
                "text_chars": 80_000,
                "cost_est_usd": 0.08,
                "status_code": 200,
                "retry_count": 0,
                "latency_ms": 125,
                "crunch_json": stable_json(crunch_json),
            }

        impact_report = build_request_shape_crunch_canary_impact_report(
            [
                impact_row("applied", saved_tokens=2_000, saved_chars=8_000, saved_usd=0.0125),
                impact_row("holdout"),
            ]
        )
        decision = build_request_shape_crunch_policy_decision_report(impact_report)

        self.assertEqual(decision["schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertEqual(decision["decision"], "widen")
        self.assertEqual(decision["graduation_decision"], "widen")
        self.assertEqual(decision["summary"]["decision"], "widen")
        self.assertEqual(decision["summary"]["graduation_decision"], "widen")
        self.assertTrue(decision["summary"]["promotion_allowed"])
        self.assertFalse(decision["summary"]["keep_staged"])
        self.assertEqual(decision["summary"]["applied_count"], 1)
        self.assertEqual(decision["summary"]["holdout_count"], 1)
        self.assertEqual(decision["summary"]["observed_saved_tokens"], 2_000)
        self.assertEqual(decision["summary"]["observed_saved_usd"], 0.0125)
        self.assertEqual(decision["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["fallback_rate_delta"], 0.0)
        self.assertEqual(decision["summary"]["policy_source"], "local-manual")
        self.assertFalse(decision["summary"]["policy_files_written"])
        top = decision["top_decision"]
        self.assertEqual(top["decision_options"], ["widen", "rollback", "keep-staged", "blocked"])
        self.assertEqual(top["local_policy_patch"]["patch_type"], "widen_repeated_context_crunch_canary")
        self.assertEqual(top["local_policy_patch"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(top["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertTrue(top["rollback_metadata"]["required_for_promotion"])
        self.assertTrue(top["coverage"]["has_applied_coverage"])
        self.assertTrue(top["coverage"]["has_holdout_coverage"])
        ledger = build_request_shape_crunch_policy_decision_ledger(
            decision,
            recorded_at="2026-06-15T18:00:00+00:00",
        )
        self.assertEqual(ledger["schema"], "tokenclaw.request_shape_crunch_policy_decision_ledger.v1")
        self.assertEqual(ledger["status"], "recordable")
        self.assertEqual(ledger["entries"][0]["status"], "positive")
        self.assertEqual(ledger["entries"][0]["recommendation"], "widen")
        self.assertEqual(ledger["entries"][0]["applied_count"], 1)
        self.assertEqual(ledger["entries"][0]["holdout_count"], 1)
        first = record_request_shape_crunch_policy_decision_ledger(
            decision,
            store_obj=self.store,
            recorded_at="2026-06-15T18:00:01+00:00",
        )
        rows = self.store.promotion_outcome_feedback_rows(action_family="crunch", limit=10)
        self.assertTrue(first["wrote_store"])
        self.assertEqual(first["summary"]["rows_written"], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_evidence_schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertEqual(rows[0]["status"], "positive")
        self.assertEqual(rows[0]["recommendation"], "widen")
        self.assertEqual(rows[0]["applied_count"], 1)
        self.assertEqual(rows[0]["holdout_count"], 1)
        self.assertTrue(decision["privacy"]["metadata_only"])
        self.assertTrue(decision["privacy"]["aggregate_only"])
        self.assertFalse(decision["privacy"]["raw_prompts_included"])
        self.assertFalse(decision["privacy"]["provider_bodies_included"])
        self.assertFalse(decision["privacy"]["request_ids_included"])
        self.assertFalse(decision["privacy"]["session_ids_included"])

    def test_crunch_policy_decision_apply_widens_local_rule_with_rollback_metadata(self) -> None:
        def impact_row(status: str, *, saved_tokens: int = 0, saved_chars: int = 0, saved_usd: float = 0.0) -> dict[str, object]:
            lifecycle = {
                "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
                "policy_id": "policy-promote",
                "cohort_id": "cohort-promote",
                "status": status,
                "cohort": "canary_applied" if status == "applied" else "canary_holdout",
                "reason": status,
                "policy_source": "local-manual",
                "metadata_only": True,
            }
            crunch_json = {
                "changed": saved_tokens > 0,
                "before_chars": 80_000,
                "after_chars": max(0, 80_000 - saved_chars),
                "saved_chars": saved_chars,
                "tokens_saved_est": saved_tokens,
                "request_shape_repeated_context_canary": lifecycle,
            }
            if saved_usd:
                crunch_json["savings_usd"] = saved_usd
            return {
                "created_at": utc_now(),
                "provider_family": "anthropic",
                "provider": "anthropic",
                "source_surface": "anthropic_messages",
                "endpoint": "messages",
                "category": "tool-result",
                "workflow_phase": "thinking",
                "stream": True,
                "has_tools": True,
                "text_bucket": "gte_128k_chars",
                "token_bucket": "lt_500_tokens",
                "cache_status": "skipped",
                "routing_status": "passthrough",
                "requested_model": "claude-sonnet-4-6",
                "routed_model": "claude-sonnet-4-6",
                "actual_input_tokens": 20_000,
                "input_tokens_est": 20_000,
                "text_chars": 80_000,
                "cost_est_usd": 0.08,
                "status_code": 200,
                "retry_count": 0,
                "latency_ms": 125,
                "crunch_json": stable_json(crunch_json),
            }

        decision = build_request_shape_crunch_policy_decision_report(
            build_request_shape_crunch_canary_impact_report(
                [
                    impact_row("applied", saved_tokens=2_000, saved_chars=8_000, saved_usd=0.0125),
                    impact_row("holdout"),
                ]
            )
        )
        decision_id = decision["decision_id"]
        rules_path = Path(self.tmpdir.name) / "config" / "crunch_rules.yaml"
        rules_path.parent.mkdir()
        rules_path.write_text(
            yaml.safe_dump(
                {
                    "enabled": True,
                    "request_shape_repeated_context_canaries": {
                        "enabled": True,
                        "schema": "tokenclaw.request_shape_repeated_context_canaries.v1",
                        "rules": [
                            {
                                "id": "policy-promote",
                                "enabled": True,
                                "policy_source": "local-manual",
                                "cohort_id": "cohort-promote",
                                "conditions": {
                                    "provider_family": "anthropic",
                                    "source_surface": "anthropic_messages",
                                    "endpoint": "messages",
                                    "category": "tool-result",
                                    "workflow_phase": "thinking",
                                    "stream": True,
                                    "has_tools": True,
                                    "text_bucket": "gte_128k_chars",
                                    "token_bucket": "lt_500_tokens",
                                    "cache_status": "skipped",
                                    "routing_status": "passthrough",
                                },
                                "rollout": {
                                    "canary_enabled": True,
                                    "canary_fraction": 0.10,
                                    "holdout_fraction": 0.10,
                                    "canary_salt": "policy-promote",
                                    "canary_unit": "request_shape_cohort",
                                },
                                "safety_gates": {
                                    "metadata_only": True,
                                    "aggregate_only": True,
                                    "raw_prompts_included": False,
                                    "provider_bodies_included": False,
                                    "request_ids_included": False,
                                    "session_ids_included": False,
                                    "cache_keys_included": False,
                                },
                            }
                        ],
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        dry_apply = apply_request_shape_crunch_policy_decision(
            decision,
            rules_path=rules_path,
            dry_run=True,
            decision_id=decision_id,
        )
        self.assertTrue(dry_apply["ok"])
        self.assertFalse(dry_apply["wrote_policy_files"])
        self.assertEqual(dry_apply["decision_id"], decision_id)
        self.assertEqual(dry_apply["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(dry_apply["previous_canary_fraction"], 0.1)
        self.assertEqual(dry_apply["canary_fraction"], 0.2)
        self.assertEqual(dry_apply["widened_cohort"]["cohort_id"], "cohort-promote")
        self.assertEqual(dry_apply["rollback_metadata"]["rollback_action_type"], "disable_repeated_context_crunch_canary")
        self.assertTrue(dry_apply["privacy"]["metadata_only"])
        self.assertTrue(dry_apply["privacy"]["aggregate_only"])
        self.assertFalse(dry_apply["privacy"]["raw_prompts_included"])
        self.assertFalse(dry_apply["privacy"]["provider_bodies_included"])
        self.assertFalse(dry_apply["privacy"]["request_ids_included"])
        self.assertFalse(dry_apply["privacy"]["session_ids_included"])
        self.assertEqual(
            yaml.safe_load(rules_path.read_text(encoding="utf-8"))["request_shape_repeated_context_canaries"]["rules"][0]["rollout"]["canary_fraction"],
            0.10,
        )

        applied = apply_request_shape_crunch_policy_decision(
            decision,
            rules_path=rules_path,
            decision_id=decision_id,
        )
        self.assertTrue(applied["ok"])
        self.assertTrue(applied["wrote_policy_files"])
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        widened = rules["request_shape_repeated_context_canaries"]["rules"][0]
        self.assertEqual(widened["rollout"]["canary_fraction"], 0.2)
        self.assertEqual(widened["rollout"]["holdout_fraction"], 0.1)
        self.assertEqual(widened["policy_decision"]["decision_id"], decision_id)
        self.assertEqual(widened["policy_decision"]["source_evidence_schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertEqual(widened["policy_decision"]["observed_saved_tokens"], 2_000)
        self.assertEqual(widened["rollback_metadata"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(widened["rollback_metadata"]["required_for_promotion"])
        self.assertTrue(widened["privacy"]["metadata_only"])
        self.assertFalse(widened["privacy"]["raw_prompts_included"])

        applied_again = apply_request_shape_crunch_policy_decision(
            decision,
            rules_path=rules_path,
            decision_id=decision_id,
        )
        self.assertTrue(applied_again["ok"])
        self.assertTrue(applied_again["already_applied"])
        self.assertEqual(applied_again["status"], "already-applied")
        self.assertFalse(applied_again["wrote_policy_files"])
        self.assertEqual(applied_again["canary_fraction"], 0.2)

        fresh_decision = build_request_shape_crunch_policy_decision_report(
            build_request_shape_crunch_canary_impact_report(
                [
                    impact_row("applied", saved_tokens=2_000, saved_chars=8_000, saved_usd=0.0125),
                    impact_row("applied", saved_tokens=3_000, saved_chars=12_000, saved_usd=0.0185),
                    impact_row("holdout"),
                ]
            )
        )
        self.assertEqual(fresh_decision["decision_id"], decision_id)
        fresh_apply = apply_request_shape_crunch_policy_decision(
            fresh_decision,
            rules_path=rules_path,
            decision_id=decision_id,
        )
        self.assertTrue(fresh_apply["ok"])
        self.assertFalse(fresh_apply["already_applied"])
        self.assertTrue(fresh_apply["wrote_policy_files"])
        self.assertEqual(fresh_apply["previous_canary_fraction"], 0.2)
        self.assertEqual(fresh_apply["canary_fraction"], 0.3)
        fresh_rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        fresh_widened = fresh_rules["request_shape_repeated_context_canaries"]["rules"][0]
        self.assertEqual(fresh_widened["rollout"]["canary_fraction"], 0.3)
        self.assertEqual(fresh_widened["policy_decision"]["decision_id"], decision_id)
        self.assertEqual(fresh_widened["policy_decision"]["observed_saved_tokens"], 5_000)
        self.assertEqual(fresh_widened["policy_decision"]["applied_count"], 2)
        self.assertIn("application_fingerprint", fresh_widened["policy_decision"])

        fresh_apply_again = apply_request_shape_crunch_policy_decision(
            fresh_decision,
            rules_path=rules_path,
            decision_id=decision_id,
        )
        self.assertTrue(fresh_apply_again["ok"])
        self.assertTrue(fresh_apply_again["already_applied"])
        self.assertFalse(fresh_apply_again["wrote_policy_files"])
        self.assertEqual(fresh_apply_again["canary_fraction"], 0.3)

        full_rollout = apply_request_shape_crunch_policy_decision(
            fresh_decision,
            rules_path=rules_path,
            decision_id=decision_id,
            promote_full_rollout=True,
        )
        self.assertTrue(full_rollout["ok"])
        self.assertEqual(full_rollout["status"], "full-rollout-applied")
        self.assertTrue(full_rollout["wrote_policy_files"])
        self.assertTrue(full_rollout["full_rollout_ready"])
        self.assertTrue(full_rollout["full_rollout_applied"])
        self.assertFalse(full_rollout["canary_enabled"])
        self.assertEqual(full_rollout["previous_canary_fraction"], 0.3)
        self.assertEqual(full_rollout["canary_fraction"], 1.0)
        self.assertEqual(full_rollout["full_rollout_fraction"], 1.0)
        self.assertEqual(full_rollout["holdout_fraction"], 0.0)
        full_rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        full_rule = full_rules["request_shape_repeated_context_canaries"]["rules"][0]
        self.assertFalse(full_rule["rollout"]["canary_enabled"])
        self.assertTrue(full_rule["rollout"]["full_rollout_enabled"])
        self.assertEqual(full_rule["rollout"]["full_rollout_fraction"], 1.0)
        self.assertEqual(full_rule["rollout"]["holdout_fraction"], 0.0)
        self.assertEqual(full_rule["policy_decision"]["decision"], "promote-full")
        self.assertEqual(full_rule["policy_decision"]["graduation_decision"], "promote-full")
        self.assertEqual(full_rule["policy_decision"]["source_evidence_schema"], "tokenclaw.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(full_rule["policy_decision"]["source_policy_decision_schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertIn("full_rollout_fingerprint", full_rule["policy_decision"])
        self.assertEqual(full_rule["rollback_metadata"]["selected_decision"], "promote-full")

        full_rollout_again = apply_request_shape_crunch_policy_decision(
            fresh_decision,
            rules_path=rules_path,
            decision_id=decision_id,
            promote_full_rollout=True,
        )
        self.assertTrue(full_rollout_again["ok"])
        self.assertEqual(full_rollout_again["status"], "already-full-rollout")
        self.assertTrue(full_rollout_again["already_applied"])
        self.assertFalse(full_rollout_again["wrote_policy_files"])

    def test_crunch_policy_decision_rolls_back_on_safety_stop(self) -> None:
        lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "policy-safety",
            "cohort_id": "cohort-safety",
            "status": "safety-stopped",
            "cohort": "safety_stopped",
            "reason": "error-rate-regression",
            "safety_stop": True,
            "metadata_only": True,
        }
        impact_report = build_request_shape_crunch_canary_impact_report(
            [
                {
                    "created_at": utc_now(),
                    "provider_family": "anthropic",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "actual_input_tokens": 20_000,
                    "input_tokens_est": 20_000,
                    "text_chars": 80_000,
                    "cost_est_usd": 0.08,
                    "status_code": 500,
                    "retry_count": 2,
                    "latency_ms": 125,
                    "crunch_json": stable_json({"request_shape_repeated_context_canary": lifecycle}),
                }
            ]
        )
        decision = build_request_shape_crunch_policy_decision_report(impact_report)

        self.assertEqual(decision["decision"], "rollback")
        self.assertEqual(decision["graduation_decision"], "rollback")
        self.assertTrue(decision["summary"]["rollback_required"])
        self.assertEqual(decision["summary"]["safety_stop_state"], "observed")
        self.assertEqual(decision["top_decision"]["reason"], "canary-safety-stopped")
        self.assertFalse(decision["top_decision"]["promotion_allowed"])
        self.assertEqual(decision["top_decision"]["local_policy_patch"]["patch_type"], "rollback_repeated_context_crunch_canary")
        self.assertIn("canary-safety-stopped", decision["top_decision"]["reason_codes"])
        self.assertEqual(decision["top_decision"]["metrics"]["safety_stop_count"], 1)

    def test_crunch_policy_decision_keeps_staged_for_missing_holdout(self) -> None:
        lifecycle = {
            "schema": "tokenclaw.request_shape_crunch_canary_lifecycle.v1",
            "policy_id": "policy-staged",
            "cohort_id": "cohort-staged",
            "status": "applied",
            "cohort": "canary_applied",
            "reason": "applied",
            "metadata_only": True,
        }
        impact_report = build_request_shape_crunch_canary_impact_report(
            [
                {
                    "created_at": utc_now(),
                    "provider_family": "anthropic",
                    "provider": "anthropic",
                    "source_surface": "anthropic_messages",
                    "endpoint": "messages",
                    "category": "tool-result",
                    "workflow_phase": "thinking",
                    "stream": True,
                    "has_tools": True,
                    "text_bucket": "gte_128k_chars",
                    "token_bucket": "lt_500_tokens",
                    "cache_status": "skipped",
                    "routing_status": "passthrough",
                    "requested_model": "claude-sonnet-4-6",
                    "routed_model": "claude-sonnet-4-6",
                    "actual_input_tokens": 20_000,
                    "input_tokens_est": 20_000,
                    "text_chars": 80_000,
                    "cost_est_usd": 0.08,
                    "status_code": 200,
                    "retry_count": 0,
                    "latency_ms": 125,
                    "crunch_json": stable_json(
                        {
                            "changed": True,
                            "saved_chars": 8_000,
                            "tokens_saved_est": 2_000,
                            "request_shape_repeated_context_canary": lifecycle,
                        }
                    ),
                }
            ]
        )
        decision = build_request_shape_crunch_policy_decision_report(impact_report)

        self.assertEqual(decision["decision"], "keep-staged")
        self.assertEqual(decision["graduation_decision"], "keep-staged")
        self.assertTrue(decision["summary"]["keep_staged"])
        self.assertFalse(decision["summary"]["keep_blocked"])
        self.assertEqual(decision["top_decision"]["local_policy_patch"]["patch_type"], "keep_repeated_context_crunch_canary_staged")
        self.assertIn("missing-holdout-coverage", decision["top_decision"]["reason_codes"])
        ledger = build_request_shape_crunch_policy_decision_ledger(decision, recorded_at="2026-06-15T18:01:00+00:00")
        self.assertEqual(ledger["entries"][0]["status"], "needs-more-samples")
        self.assertEqual(ledger["entries"][0]["recommendation"], "keep-staged")

    def test_crunch_policy_decision_fixture_emits_four_promotion_outcomes(self) -> None:
        def candidate(
            policy_id: str,
            recommendation: str,
            *,
            applied_count: int,
            holdout_count: int,
            saved_tokens: int = 0,
            saved_usd: float = 0.0,
            safety_stop_count: int = 0,
            rollback_count: int = 0,
            reason_codes: list[str] | None = None,
        ) -> dict[str, object]:
            return {
                "schema": "tokenclaw.request_shape_crunch_canary_impact_candidate.v1",
                "policy_id": policy_id,
                "cohort_id": policy_id.replace("policy", "cohort"),
                "policy_source": "local-manual",
                "impact_recommendation": recommendation,
                "applied_count": applied_count,
                "holdout_count": holdout_count,
                "saved_tokens": saved_tokens,
                "saved_chars": saved_tokens * 4,
                "saved_usd": saved_usd,
                "projected_saved_tokens": max(saved_tokens, 1000),
                "projected_saved_usd": max(saved_usd, 0.003),
                "safety_stop_count": safety_stop_count,
                "rollback_count": rollback_count,
                "fallback_count": 0,
                "error_rate_delta": 0.0,
                "retry_rate_delta": 0.0,
                "fallback_rate_delta": 0.0,
                "reason_codes": reason_codes or [],
                "coverage": {
                    "schema": "tokenclaw.request_shape_crunch_canary_coverage.v1",
                    "applied_count": applied_count,
                    "holdout_count": holdout_count,
                    "has_applied_coverage": applied_count > 0,
                    "has_holdout_coverage": holdout_count > 0,
                    "safety_stop_count": safety_stop_count,
                    "rollback_count": rollback_count,
                    "metadata_only": True,
                    "aggregate_only": True,
                },
            }

        report = build_request_shape_crunch_policy_decision_report(
            {
                "schema": "tokenclaw.request_shape_crunch_canary_impact.v1",
                "status": "observed",
                "summary": {
                    "applied_count": 7,
                    "holdout_count": 3,
                    "saved_tokens": 3200,
                    "provider_calls_made": 0,
                    "managed_server_calls_made": 0,
                    "policy_files_written": False,
                },
                "candidates": [
                    candidate("policy-promote", "promotion-ready", applied_count=4, holdout_count=2, saved_tokens=3200, saved_usd=0.0096),
                    candidate("policy-staged", "collect-more-evidence", applied_count=2, holdout_count=0, saved_tokens=900, saved_usd=0.0027, reason_codes=["missing-holdout-coverage"]),
                    candidate("policy-rollback", "rollback", applied_count=1, holdout_count=1, safety_stop_count=1, rollback_count=1, reason_codes=["canary-safety-stopped"]),
                    candidate("policy-blocked", "keep-blocked", applied_count=0, holdout_count=0, reason_codes=["no-applied-savings"]),
                ],
            }
        )

        self.assertEqual(report["schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertEqual(report["promotion_decision"], "promote")
        self.assertEqual(report["promotion_readiness"], "promotion-ready")
        self.assertEqual(report["decision"], "widen")
        self.assertFalse(report["summary"]["policy_files_written"])
        self.assertEqual(report["summary"]["provider_calls_made"], 0)
        self.assertEqual(report["summary"]["managed_server_calls_made"], 0)
        decisions_by_policy = {item["policy_id"]: item for item in report["decisions"]}
        self.assertEqual(decisions_by_policy["policy-promote"]["promotion_decision"], "promote")
        self.assertEqual(decisions_by_policy["policy-staged"]["promotion_decision"], "keep-staged")
        self.assertEqual(decisions_by_policy["policy-rollback"]["promotion_decision"], "rollback")
        self.assertEqual(decisions_by_policy["policy-blocked"]["promotion_decision"], "keep-blocked")
        self.assertEqual(decisions_by_policy["policy-promote"]["promotion_decision_options"], ["promote", "keep-staged", "rollback", "keep-blocked"])
        self.assertTrue(decisions_by_policy["policy-promote"]["coverage"]["has_applied_coverage"])
        self.assertTrue(decisions_by_policy["policy-promote"]["coverage"]["has_holdout_coverage"])
        self.assertEqual(decisions_by_policy["policy-promote"]["observed_saved_tokens"], 3200)
        self.assertEqual(decisions_by_policy["policy-rollback"]["metrics"]["safety_stop_count"], 1)
        self.assertEqual(decisions_by_policy["policy-rollback"]["metrics"]["rollback_count"], 1)
        duplicate_suppression = decisions_by_policy["policy-promote"]["duplicate_suppression"]
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertTrue(duplicate_suppression["suppresses_generic_crunch_activation_issue"])
        self.assertEqual(duplicate_suppression["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(str(duplicate_suppression["fingerprint"]).startswith("activation:"))
        self.assertTrue(report["summary"]["duplicate_activation_issue_suppressed"])
        self.assertTrue(report["privacy"]["metadata_only"])
        self.assertTrue(report["privacy"]["aggregate_only"])
        self.assertFalse(report["privacy"]["raw_prompts_included"])
        self.assertFalse(report["privacy"]["provider_bodies_included"])
        self.assertFalse(report["privacy"]["request_ids_included"])
        self.assertFalse(report["privacy"]["session_ids_included"])
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("raw prompt", rendered)

    def test_crunch_activation_evidence_joins_current_decision_and_active_rule(self) -> None:
        decision_id = "request-shape-crunch-policy-decision:9db327d1abdec766"
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                yaml.safe_dump(
                    {
                        "request_shape_repeated_context_canaries": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "raw-policy-secret-should-not-leak",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "policy_decision": {
                                        "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                        "decision_id": decision_id,
                                        "source_evidence_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                        "decision": "widen",
                                        "graduation_decision": "widen",
                                        "applied_count": 26,
                                        "holdout_count": 17,
                                        "observed_saved_tokens": 1647683,
                                        "observed_saved_usd": 4.943049,
                                        "error_rate_delta": 0.0,
                                        "retry_rate_delta": 0.0,
                                        "fallback_rate_delta": 0.0,
                                        "safety_stop_state": "none",
                                        "previous_canary_fraction": 0.20,
                                        "widened_canary_fraction": 0.30,
                                        "holdout_fraction": 0.10,
                                    },
                                    "rollout": {
                                        "schema": "tokenclaw.request_shape_crunch_canary_rollout.v1",
                                        "canary_fraction": 0.30,
                                        "holdout_fraction": 0.10,
                                    },
                                    "safety_gates": {
                                        "metadata_only": True,
                                        "aggregate_only": True,
                                        "max_rollout_fraction": 0.50,
                                    },
                                }
                            ],
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            payload = build_request_shape_crunch_activation_evidence_report(
                crunch_policy_decision={
                    "schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                    "decision": "widen",
                    "graduation_decision": "widen",
                    "decision_id": decision_id,
                    "summary": {
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "applied_count": 77,
                        "holdout_count": 30,
                        "observed_saved_tokens": 5965139,
                        "observed_saved_usd": 17.895417,
                        "coverage": {
                            "skipped_count": 218,
                            "fallback_count": 0,
                            "safety_stop_count": 0,
                            "rollback_count": 0,
                        },
                    },
                },
                crunch_canary_impact={
                    "schema": "tokenclaw.request_shape_crunch_canary_impact.v1",
                    "summary": {
                        "applied_count": 77,
                        "holdout_count": 30,
                        "saved_tokens": 5965139,
                        "saved_usd": 17.895417,
                    },
                },
                rules_path=rules_path,
            )

        self.assertEqual(payload["schema"], "tokenclaw.request_shape_crunch_activation_evidence.v1")
        self.assertEqual(payload["status"], "active-rule-evidence-observed")
        self.assertEqual(payload["decision_id"], decision_id)
        self.assertEqual(payload["summary"]["active_rule_count"], 1)
        self.assertEqual(payload["summary"]["matching_active_rule_count"], 1)
        self.assertEqual(payload["summary"]["widened_rule_count"], 1)
        self.assertEqual(payload["next_action"], "widen-further")
        self.assertEqual(payload["summary"]["post_widening_status"], "post-widening-widen-ready")
        self.assertEqual(payload["summary"]["post_widening_next_action"], "widen-further")
        self.assertEqual(payload["summary"]["applied_count"], 77)
        self.assertEqual(payload["summary"]["holdout_count"], 30)
        self.assertEqual(payload["summary"]["skipped_count"], 218)
        self.assertEqual(payload["summary"]["observed_saved_tokens"], 5965139)
        self.assertAlmostEqual(payload["summary"]["observed_saved_usd"], 17.895417)
        self.assertEqual(payload["summary"]["error_rate_delta"], 0.0)
        self.assertEqual(payload["summary"]["retry_rate_delta"], 0.0)
        self.assertEqual(payload["summary"]["fallback_rate_delta"], 0.0)
        self.assertEqual(payload["summary"]["safety_stop_count"], 0)
        self.assertEqual(payload["summary"]["canary_fraction"], 0.3)
        self.assertEqual(payload["summary"]["max_rollout_fraction"], 0.5)
        self.assertEqual(payload["summary"]["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(payload["privacy"]["metadata_only"])
        self.assertTrue(payload["privacy"]["aggregate_only"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-policy-secret-should-not-leak", rendered)
        self.assertNotIn(str(rules_path), rendered)

    def test_crunch_activation_evidence_emits_keep_active_duplicate_suppression(self) -> None:
        decision_id = "request-shape-crunch-policy-decision:keep-active"
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                yaml.safe_dump(
                    {
                        "request_shape_repeated_context_canaries": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "raw-keep-active-policy-secret-should-not-leak",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "policy_decision": {
                                        "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                        "decision_id": decision_id,
                                        "source_evidence_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                        "decision": "widen",
                                        "graduation_decision": "widen",
                                        "applied_count": 107,
                                        "holdout_count": 40,
                                        "observed_saved_tokens": 8606129,
                                        "observed_saved_usd": 25.818387,
                                        "error_rate_delta": 0.0,
                                        "retry_rate_delta": 0.0,
                                        "fallback_rate_delta": 0.0,
                                        "safety_stop_state": "none",
                                        "widened_canary_fraction": 0.30,
                                        "holdout_fraction": 0.10,
                                    },
                                    "rollout": {"canary_fraction": 0.30, "holdout_fraction": 0.10},
                                    "safety_gates": {"max_rollout_fraction": 0.30},
                                    "rollback_metadata": {
                                        "rollback_action_type": "disable_repeated_context_crunch_canary",
                                        "required_for_promotion": True,
                                    },
                                }
                            ],
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_request_shape_crunch_activation_evidence_report(
                crunch_policy_decision={
                    "schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                    "decision": "widen",
                    "graduation_decision": "widen",
                    "decision_id": decision_id,
                    "summary": {
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "applied_count": 107,
                        "holdout_count": 40,
                        "observed_saved_tokens": 8606129,
                        "observed_saved_usd": 25.818387,
                        "coverage": {
                            "skipped_count": 280,
                            "fallback_count": 0,
                            "safety_stop_count": 0,
                            "rollback_count": 0,
                        },
                    },
                },
                crunch_canary_impact={"schema": "tokenclaw.request_shape_crunch_canary_impact.v1", "summary": {}},
                rules_path=rules_path,
            )

        self.assertEqual(payload["status"], "active-rule-evidence-observed")
        self.assertEqual(payload["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(payload["summary"]["post_widening_status"], "post-widening-active-at-max-rollout")
        self.assertEqual(payload["summary"]["post_widening_next_action"], "keep-active")
        self.assertEqual(payload["summary"]["post_max_rollout_status"], "post-max-rollout-full-rollout-ready")
        self.assertEqual(payload["summary"]["post_max_rollout_decision"], "promote-full")
        self.assertEqual(payload["summary"]["post_max_rollout_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(payload["summary"]["post_max_rollout_promotion_allowed"])
        post_max = payload["post_max_rollout_decision"]
        self.assertEqual(post_max["schema"], "tokenclaw.request_shape_crunch_post_max_rollout_decision.v1")
        self.assertEqual(post_max["decision"], "promote-full")
        self.assertEqual(post_max["target_local_rule_file"], "crunch_rules.yaml")
        self.assertEqual(post_max["target_local_policy_section"], "crunch.rules")
        self.assertEqual(post_max["local_policy_patch"]["patch_type"], "promote_repeated_context_crunch_rule_full_rollout")
        self.assertEqual(post_max["local_policy_patch"]["rollout_update"]["full_rollout_fraction"], 1.0)
        self.assertFalse(post_max["local_policy_patch"]["policy_file_contents_included"])
        self.assertTrue(post_max["rollback_metadata"]["present"])
        self.assertFalse(post_max["privacy"]["raw_prompts_included"])
        self.assertFalse(post_max["privacy"]["provider_bodies_included"])
        self.assertFalse(post_max["privacy"]["request_ids_included"])
        self.assertFalse(post_max["privacy"]["session_ids_included"])
        self.assertFalse(post_max["privacy"]["cache_keys_included"])
        follow_up = payload["activation_follow_up"]
        self.assertEqual(follow_up["activation_state"], "measured-active")
        self.assertEqual(follow_up["next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertEqual(follow_up["post_max_rollout_decision"]["decision"], "promote-full")
        duplicate_suppression = payload["duplicate_suppression"]
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertTrue(duplicate_suppression["suppresses_generic_crunch_activation_issue"])
        self.assertEqual(duplicate_suppression["matching_local_policy"], "crunch_rules")
        self.assertEqual(duplicate_suppression["target_local_rule_file"], "crunch_rules.yaml")
        self.assertTrue(str(duplicate_suppression["fingerprint"]).startswith("activation:"))
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-keep-active-policy-secret-should-not-leak", rendered)
        self.assertNotIn(str(rules_path), rendered)

    def test_crunch_activation_evidence_reports_full_rollout_applied_rule(self) -> None:
        decision_id = "request-shape-crunch-policy-decision:full-rollout"
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                yaml.safe_dump(
                    {
                        "request_shape_repeated_context_canaries": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "raw-full-rollout-policy-secret-should-not-leak",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "policy_decision": {
                                        "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                        "decision_id": decision_id,
                                        "source_evidence_schema": "tokenclaw.request_shape_crunch_activation_evidence.v1",
                                        "source_policy_decision_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                        "decision": "promote-full",
                                        "graduation_decision": "promote-full",
                                        "applied_count": 107,
                                        "holdout_count": 40,
                                        "observed_saved_tokens": 8606129,
                                        "observed_saved_usd": 25.818387,
                                        "error_rate_delta": 0.0,
                                        "retry_rate_delta": 0.0,
                                        "fallback_rate_delta": 0.0,
                                        "safety_stop_state": "none",
                                        "previous_canary_fraction": 0.30,
                                        "widened_canary_fraction": 1.0,
                                        "full_rollout_fraction": 1.0,
                                        "holdout_fraction": 0.0,
                                    },
                                    "rollout": {
                                        "canary_enabled": False,
                                        "full_rollout_enabled": True,
                                        "full_rollout_fraction": 1.0,
                                        "canary_fraction": 1.0,
                                        "holdout_fraction": 0.0,
                                    },
                                    "safety_gates": {
                                        "max_rollout_fraction": 1.0,
                                        "previous_max_rollout_fraction": 0.30,
                                    },
                                    "rollback_metadata": {
                                        "rollback_action_type": "disable_repeated_context_crunch_canary",
                                        "required_for_promotion": True,
                                    },
                                }
                            ],
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_request_shape_crunch_activation_evidence_report(
                crunch_policy_decision={
                    "schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                    "decision": "widen",
                    "graduation_decision": "widen",
                    "decision_id": decision_id,
                    "summary": {
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "applied_count": 107,
                        "holdout_count": 40,
                        "observed_saved_tokens": 8606129,
                        "observed_saved_usd": 25.818387,
                        "coverage": {
                            "skipped_count": 280,
                            "fallback_count": 0,
                            "safety_stop_count": 0,
                            "rollback_count": 0,
                        },
                    },
                },
                crunch_canary_impact={"schema": "tokenclaw.request_shape_crunch_canary_impact.v1", "summary": {}},
                rules_path=rules_path,
            )

        self.assertEqual(payload["status"], "active-rule-evidence-observed")
        self.assertEqual(payload["next_action"], "measure-full-rollout-repeated-context-crunch-outcomes")
        self.assertEqual(payload["summary"]["full_rollout_rule_count"], 1)
        self.assertEqual(payload["summary"]["matching_full_rollout_rule_count"], 1)
        self.assertTrue(payload["summary"]["full_rollout_active"])
        self.assertEqual(payload["summary"]["full_rollout_fraction"], 1.0)
        self.assertEqual(payload["summary"]["post_max_rollout_status"], "post-max-rollout-full-rollout-applied")
        self.assertEqual(payload["summary"]["post_max_rollout_decision"], "full-rollout-applied")
        self.assertEqual(payload["summary"]["post_max_rollout_next_action"], "measure-full-rollout-repeated-context-crunch-outcomes")
        self.assertFalse(payload["summary"]["post_max_rollout_promotion_allowed"])
        post_max = payload["post_max_rollout_decision"]
        self.assertEqual(post_max["decision"], "full-rollout-applied")
        self.assertTrue(post_max["full_rollout_active"])
        self.assertNotIn("local_policy_patch", post_max)
        follow_up = payload["activation_follow_up"]
        self.assertEqual(follow_up["status"], "full-rollout-outcome-recorded")
        self.assertEqual(follow_up["activation_state"], "full-rollout-active")
        self.assertEqual(follow_up["no_op_reason"], "repeated-context-crunch-full-rollout-active")
        duplicate_suppression = payload["duplicate_suppression"]
        self.assertTrue(duplicate_suppression["suppresses_new_activation_issue"])
        self.assertEqual(duplicate_suppression["reason"], "repeated-context-crunch-full-rollout-active")
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-full-rollout-policy-secret-should-not-leak", rendered)
        self.assertNotIn(str(rules_path), rendered)

    def test_remaining_crunch_measurements_skip_keep_active_rule_coverage(self) -> None:
        decision_id = "request-shape-crunch-policy-decision:keep-active"
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                yaml.safe_dump(
                    {
                        "request_shape_repeated_context_canaries": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "raw-active-rule-secret-should-not-leak",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "cohort_id": "request-shape-crunch:covered",
                                    "conditions": {
                                        "provider_family": "anthropic",
                                        "source_surface": "anthropic_messages",
                                        "endpoint": "messages",
                                        "category": "tool-result",
                                        "workflow_phase": "thinking",
                                        "stream": True,
                                        "has_tools": True,
                                        "cache_status": "skipped",
                                        "routing_status": "passthrough",
                                        "text_bucket": "gte_128k_chars",
                                        "token_bucket": "lt_500_tokens",
                                    },
                                    "rollout": {"canary_enabled": True, "canary_fraction": 0.30, "holdout_fraction": 0.10},
                                    "safety_gates": {"max_rollout_fraction": 0.30},
                                    "policy_decision": {
                                        "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                        "decision_id": decision_id,
                                        "source_evidence_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                        "decision": "widen",
                                        "graduation_decision": "widen",
                                        "applied_count": 107,
                                        "holdout_count": 40,
                                        "observed_saved_tokens": 8606129,
                                        "observed_saved_usd": 25.818387,
                                        "widened_canary_fraction": 0.30,
                                        "holdout_fraction": 0.10,
                                    },
                                    "rollback_metadata": {
                                        "rollback_action_type": "disable_repeated_context_crunch_canary",
                                        "required_for_promotion": True,
                                    },
                                }
                            ],
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            activation = build_request_shape_crunch_activation_evidence_report(
                crunch_policy_decision={
                    "schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                    "decision": "widen",
                    "graduation_decision": "widen",
                    "decision_id": decision_id,
                    "summary": {
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "applied_count": 107,
                        "holdout_count": 40,
                        "observed_saved_tokens": 8606129,
                        "observed_saved_usd": 25.818387,
                        "coverage": {"skipped_count": 280, "fallback_count": 0, "safety_stop_count": 0, "rollback_count": 0},
                    },
                },
                crunch_canary_impact={"schema": "tokenclaw.request_shape_crunch_canary_impact.v1", "summary": {}},
                rules_path=rules_path,
            )
            follow_up = {
                "schema": "tokenclaw.request_shape_follow_up_candidates.v1",
                "candidates": [
                    {
                        "rank": 1,
                        "provider_family": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "stream": True,
                        "has_tools": True,
                        "cache_status": "skipped",
                        "routing_status": "passthrough",
                        "text_bucket": "gte_128k_chars",
                        "token_bucket": "lt_500_tokens",
                        "row_count": 107,
                        "sample_count": 107,
                        "projected_saved_tokens": 8606129,
                        "projected_savings_usd": 25.818387,
                        "projected_crunch_tokens_saved": 8606129,
                        "projected_crunch_savings_usd": 25.818387,
                        "blocker_codes": ["tool-call-cache-disabled", "unsupported-streaming-shape"],
                        "readiness_state": "measurement-required",
                        "next_action": "measure-repeated-context-crunch-canary-impact",
                        "local_action_family": "crunch",
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                    {
                        "rank": 2,
                        "provider_family": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "unknown",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "stream": True,
                        "has_tools": True,
                        "cache_status": "skipped",
                        "routing_status": "passthrough",
                        "text_bucket": "gte_128k_chars",
                        "token_bucket": "lt_500_tokens",
                        "row_count": 446,
                        "sample_count": 446,
                        "projected_saved_tokens": 1860651,
                        "projected_savings_usd": 5.581954,
                        "projected_crunch_tokens_saved": 1860651,
                        "projected_crunch_savings_usd": 5.581954,
                        "blocker_codes": ["tool-call-cache-disabled", "unsupported-streaming-shape"],
                        "readiness_state": "measurement-required",
                        "next_action": "measure-repeated-context-crunch-canary-impact",
                        "local_action_family": "crunch",
                        "privacy": {"metadata_only": True, "aggregate_only": True},
                    },
                ],
            }
            opportunity = {
                "schema": "tokenclaw.request_shape_crunch_opportunity_dry_run.v1",
                "cohorts": [
                    {
                        "provider_family": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "messages",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "stream": True,
                        "has_tools": True,
                        "cache_status": "skipped",
                        "routing_status": "passthrough",
                        "text_bucket": "gte_128k_chars",
                        "token_bucket": "lt_500_tokens",
                        "evidence_blocker_codes": ["tool-call-cache-disabled"],
                        "crunch_canary_lifecycle": {"applied_count": 107, "holdout_count": 40},
                        "duplicate_suppression": {"suppresses_new_stage_action": True, "matching_local_policy": "crunch_rules"},
                    },
                    {
                        "provider_family": "anthropic",
                        "source_surface": "anthropic_messages",
                        "endpoint": "unknown",
                        "category": "tool-result",
                        "workflow_phase": "thinking",
                        "stream": True,
                        "has_tools": True,
                        "cache_status": "skipped",
                        "routing_status": "passthrough",
                        "text_bucket": "gte_128k_chars",
                        "token_bucket": "lt_500_tokens",
                        "evidence_blocker_codes": ["tool-call-cache-disabled", "unsupported-streaming-shape"],
                        "crunch_canary_lifecycle": {
                            "applied_count": 24,
                            "holdout_count": 16,
                            "fallback_count": 1,
                            "retry_count": 2,
                            "rollback_count": 0,
                            "safety_stopped_count": 0,
                        },
                        "duplicate_suppression": {"suppresses_new_stage_action": False},
                    },
                ],
            }

            payload = build_request_shape_crunch_remaining_measurement_report(
                follow_up_candidates=follow_up,
                crunch_opportunity=opportunity,
                activation_evidence=activation,
                rules_path=rules_path,
            )

        self.assertEqual(payload["schema"], "tokenclaw.request_shape_crunch_remaining_measurement_cohorts.v1")
        self.assertEqual(payload["status"], "ranked")
        self.assertEqual(payload["summary"]["excluded_active_rule_covered_count"], 1)
        self.assertEqual(payload["summary"]["remaining_measurement_required_count"], 1)
        self.assertEqual(payload["summary"]["active_rule_next_action"], "promote-full-repeated-context-crunch-rule")
        self.assertTrue(payload["summary"]["active_rule_duplicate_suppresses_new_activation_issue"])
        cohort = payload["cohorts"][0]
        self.assertEqual(cohort["row_count"], 446)
        self.assertEqual(cohort["readiness_state"], "measurement-required")
        self.assertEqual(cohort["blocker_codes"], ["tool-call-cache-disabled", "unsupported-streaming-shape"])
        self.assertEqual(cohort["projected_saved_tokens"], 1860651)
        self.assertAlmostEqual(cohort["projected_saved_usd"], 5.581954)
        self.assertEqual(cohort["active_rule_coverage_status"], "not-covered-by-active-rule")
        self.assertEqual(cohort["applied_count"], 24)
        self.assertEqual(cohort["holdout_count"], 16)
        self.assertEqual(cohort["fallback_count"], 1)
        self.assertEqual(cohort["retry_count"], 2)
        self.assertEqual(cohort["rollback_count"], 0)
        self.assertEqual(cohort["safety_stop_count"], 0)
        self.assertTrue(cohort["privacy"]["metadata_only"])
        self.assertTrue(cohort["privacy"]["aggregate_only"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-active-rule-secret-should-not-leak", rendered)
        self.assertNotIn(str(rules_path), rendered)

    def test_crunch_activation_evidence_requests_rollback_for_post_widening_regression(self) -> None:
        decision_id = "request-shape-crunch-policy-decision:rollback-case"
        with tempfile.TemporaryDirectory() as tmp:
            rules_path = Path(tmp) / "crunch_rules.yaml"
            rules_path.write_text(
                yaml.safe_dump(
                    {
                        "request_shape_repeated_context_canaries": {
                            "enabled": True,
                            "rules": [
                                {
                                    "id": "raw-rollback-policy-secret-should-not-leak",
                                    "enabled": True,
                                    "policy_source": "local-manual",
                                    "policy_decision": {
                                        "schema": "tokenclaw.request_shape_crunch_policy_decision_rule_metadata.v1",
                                        "decision_id": decision_id,
                                        "source_evidence_schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                                        "decision": "widen",
                                        "graduation_decision": "widen",
                                        "applied_count": 10,
                                        "holdout_count": 5,
                                        "observed_saved_tokens": 1000,
                                        "observed_saved_usd": 0.003,
                                        "error_rate_delta": 0.25,
                                        "retry_rate_delta": 0.0,
                                        "fallback_rate_delta": 0.0,
                                        "safety_stop_state": "none",
                                        "widened_canary_fraction": 0.30,
                                    },
                                    "rollout": {"canary_fraction": 0.30, "holdout_fraction": 0.10},
                                    "safety_gates": {"max_rollout_fraction": 0.50},
                                }
                            ],
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_request_shape_crunch_activation_evidence_report(
                crunch_policy_decision={
                    "schema": "tokenclaw.request_shape_crunch_policy_decision.v1",
                    "decision": "widen",
                    "graduation_decision": "widen",
                    "decision_id": decision_id,
                    "summary": {
                        "decision": "widen",
                        "graduation_decision": "widen",
                        "decision_id": decision_id,
                        "applied_count": 10,
                        "holdout_count": 5,
                        "observed_saved_tokens": 1000,
                        "observed_saved_usd": 0.003,
                        "error_rate_delta": 0.25,
                        "retry_rate_delta": 0.0,
                        "fallback_rate_delta": 0.0,
                        "coverage": {"fallback_count": 0, "safety_stop_count": 0, "rollback_count": 0},
                    },
                },
                crunch_canary_impact={"schema": "tokenclaw.request_shape_crunch_canary_impact.v1", "summary": {}},
                rules_path=rules_path,
            )

        self.assertEqual(payload["status"], "active-rule-evidence-observed")
        self.assertEqual(payload["next_action"], "rollback")
        self.assertEqual(payload["summary"]["post_widening_status"], "post-widening-rollback-required")
        self.assertEqual(payload["summary"]["post_widening_next_action"], "rollback")
        self.assertEqual(payload["summary"]["post_widening_reason_codes"], ["error-rate-regression"])
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw-rollback-policy-secret-should-not-leak", rendered)

    def test_crunch_policy_decision_cli_keeps_blocked_without_canary_metadata(self) -> None:
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_crunch_policy_decision_cli(
            ["--db", self.db_path, "--limit", "10"],
            stdout=stdout,
        )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema"], "tokenclaw.request_shape_crunch_policy_decision.v1")
        self.assertEqual(payload["decision"], "blocked")
        self.assertEqual(payload["graduation_decision"], "blocked")
        self.assertTrue(payload["summary"]["keep_blocked"])
        self.assertEqual(payload["summary"]["applied_count"], 0)
        self.assertEqual(payload["summary"]["holdout_count"], 0)
        self.assertEqual(payload["top_decision"]["reason"], "missing-applied-or-holdout-coverage")
        self.assertEqual(payload["ledger_update"]["schema"], "tokenclaw.request_shape_crunch_policy_decision_ledger.v1")
        self.assertEqual(payload["ledger_update"]["status"], "recorded")
        self.assertEqual(payload["ledger_update"]["summary"]["rows_written"], 1)
        rows = self.store.promotion_outcome_feedback_rows(action_family="crunch", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "needs-more-samples")
        self.assertFalse(payload["privacy"]["raw_prompts_included"])
        self.assertTrue(payload["source_rollups"]["privacy"]["metadata_only"])
        self.assertNotIn("raw prompt must not leak", stdout.getvalue())
        self.assertNotIn("raw-session-id-must-not-leak", stdout.getvalue())

    def test_cli_persists_rollups_by_default_and_dry_run_skips_write(self) -> None:
        self._log_call()
        self._log_call()

        stdout = io.StringIO()
        code = cli.request_shape_rollups_cli(
            ["--db", self.db_path, "--limit", "10", "--run-id", "cli-rollup"],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["persisted"])
        self.assertEqual(payload["persisted_count"], 1)
        self.assertEqual(len(self.store.request_shape_rollup_rows(run_id="cli-rollup")), 1)

        stdout = io.StringIO()
        code = cli.request_shape_rollups_cli(
            ["--db", self.db_path, "--limit", "10", "--run-id", "cli-dry", "--dry-run"],
            stdout=stdout,
        )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["persisted"])
        self.assertEqual(payload["persisted_count"], 0)
        self.assertEqual(self.store.request_shape_rollup_rows(run_id="cli-dry"), [])
