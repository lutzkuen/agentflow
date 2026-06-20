import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.optimization import feedback
from tokenclaw.store import Store
from tokenclaw.terminal_compaction_feedback import (
    FEEDBACK_SCHEMA,
    SOURCE_SURFACE,
    build_terminal_output_compaction_lifecycle_feedback,
    queue_terminal_output_compaction_lifecycle_feedback,
)


RAW_STRINGS = (
    "RAW_TERMINAL_SECRET_LIFECYCLE",
    "raw-session-id-lifecycle",
    "raw-request-id-lifecycle",
    "raw-cache-key-lifecycle",
    "raw-rule-id-lifecycle",
    "raw-candidate-id-lifecycle",
    "/workspace/private/lifecycle.py",
)


def _candidate(*, verdict: str = "promote", applied: int = 1, holdout: int = 1, safety_stop: int = 0) -> dict:
    return {
        "candidate_id": "terminal-output-compaction-candidate",
        "rule_id": "local-terminal-output-compaction-canary",
        "policy_source": "managed-recommended",
        "provider": "anthropic",
        "source_surface": "anthropic_messages",
        "endpoint": "messages",
        "category": "tool-result",
        "workflow_phase": "tool-execution",
        "requested_model_family": "sonnet",
        "routed_model_family": "sonnet",
        "stream": True,
        "cohorts": {
            "applied": {"count": applied, "error_rate": 0.0, "retry_rate": 0.0},
            "holdout": {"count": holdout, "error_rate": 0.0, "retry_rate": 0.0},
            "safety_stop": {"count": safety_stop, "error_rate": 1.0, "retry_rate": 1.0},
        },
        "deltas": {
            "error_rate_delta": 0.0,
            "retry_rate_delta": 0.0,
            "latency_avg_ms_delta": -100,
            "cost_avg_usd_delta": -0.001,
        },
        "verdict": verdict,
        "reason_codes": ["impact-positive" if verdict != "rollback" else "rollback-error-rate-delta"],
        "net_savings_usd": 0.012,
        "projected_holdout_savings_usd": 0.006,
        "privacy": {"metadata_only": True, "raw_terminal_text_included": False},
    }


class TerminalOutputCompactionLifecycleFeedbackTests(unittest.TestCase):
    def assert_event(self, result: dict, *, command: str, event_type: str) -> dict:
        payload = build_terminal_output_compaction_lifecycle_feedback(result, command=command)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["event_type"], event_type)
        self.assertEqual(payload["metadata"]["schema"], FEEDBACK_SCHEMA)
        self.assertEqual(payload["metadata"]["lifecycle_kind"], "terminal_output_compaction")
        self.assertFalse(payload["metadata"]["privacy"]["raw_terminal_text_included"])
        self.assertFalse(payload["metadata"]["privacy"]["request_ids_included"])
        self.assertEqual(managed_egress_violations(payload), [])
        rendered = json.dumps(payload, sort_keys=True)
        for raw in RAW_STRINGS:
            self.assertNotIn(raw, rendered)
        return payload

    def test_reviewed_and_rejected_events_are_metadata_only(self):
        reviewed = {
            "schema": "agentflow.terminal_output_compaction_dry_run.v1",
            "ok": True,
            "dry_run": True,
            "read_only": True,
            "generated_at": "2026-06-12T19:00:00+00:00",
            "policy": {"rule_id": "local-terminal-output-compaction-dry-run", "policy_source": "local-default"},
            "summary": {"planned_call_count": 1, "projected_saved_tokens": 1200, "projected_saved_usd": 0.003},
            "plans": [
                {
                    "candidate_id": "terminal-output-compaction-dry-run:abc",
                    "status": "planned",
                    "source_surface": "anthropic_messages",
                    "category": "tool-result",
                    "model_family": "sonnet",
                    "stream": True,
                    "projected_saved_tokens": 1200,
                    "projected_saved_chars": 4800,
                    "target_count": 1,
                    "preservation_flags": {"tool_protocol_ids_preserved": True},
                }
            ],
            "privacy": {"metadata_only_output": True, "raw_terminal_text_included": False},
        }
        self.assert_event(reviewed, command="review", event_type="reviewed")

        rejected = {
            **reviewed,
            "summary": {"planned_call_count": 0, "blocked_call_count": 1},
            "plans": [
                {
                    "candidate_id": "terminal-output-compaction-dry-run:blocked",
                    "status": "blocked",
                    "blockers": ["request-body-unavailable", "raw-request-id-lifecycle"],
                    "projected_saved_tokens": 0,
                }
            ],
        }
        self.assert_event(rejected, command="review", event_type="rejected")

    def test_applied_holdout_canary_safety_stop_and_rollback_events_are_metadata_only(self):
        apply_result = {
            "schema": "agentflow.pattern_rollout_actions_apply.v1",
            "ok": True,
            "dry_run": False,
            "actions": [
                {
                    "action_id": "terminal-rollout-action",
                    "rule_collection": "terminal_output_compaction.rules",
                    "target_candidate_id": "terminal-compaction-candidate-123",
                    "target_rule_id": "managed-terminal-output-compaction-rule",
                    "status": "applied",
                    "action_type": "widen",
                    "proposed_edit": {
                        "changed": True,
                        "rule": {
                            "id": "managed-terminal-output-compaction-rule",
                            "candidate_id": "terminal-compaction-candidate-123",
                            "policy_source": "managed-recommended",
                            "action": {"type": "compact_terminal_output"},
                            "canary": {"canary_fraction": 0.25, "holdout_fraction": 0.75},
                            "safety_stop": {"enabled": True},
                        },
                    },
                }
            ],
        }
        self.assert_event(apply_result, command="apply", event_type="applied")

        for expected, summary, candidate in (
            ("holdout", {"holdout_count": 1}, _candidate(applied=0, holdout=1)),
            ("canary-applied", {"applied_count": 1, "holdout_count": 1}, _candidate(applied=1, holdout=1)),
            ("safety-stop", {"safety_stop_count": 1}, _candidate(verdict="hold", applied=0, holdout=0, safety_stop=1)),
            ("rollback", {"rollback_action_count": 1, "applied_count": 2, "holdout_count": 1}, _candidate(verdict="rollback", applied=2, holdout=1)),
        ):
            impact_result = {
                "schema": "agentflow.terminal_output_compaction_impact.v1",
                "ok": True,
                "read_only": True,
                "summary": summary,
                "candidates": [candidate],
                "privacy": {"metadata_only": True, "raw_terminal_text_included": False},
            }
            self.assert_event(impact_result, command="impact", event_type=expected)

    def test_lifecycle_feedback_queues_offline_and_status_is_payload_free(self):
        result = {
            "schema": "agentflow.terminal_output_compaction_impact.v1",
            "ok": True,
            "read_only": True,
            "summary": {"applied_count": 1, "holdout_count": 1},
            "candidates": [_candidate()],
            "privacy": {"metadata_only": True, "raw_terminal_text_included": False},
        }

        with TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                with patch.dict(os.environ, {"AGENTFLOW_RECOMMENDATION_ENABLED": "0"}, clear=False):
                    meta = asyncio.run(
                        queue_terminal_output_compaction_lifecycle_feedback(
                            store,
                            result,
                            command="impact",
                            flush_immediately=False,
                        )
                    )
                status = feedback.managed_feedback_status_result(
                    store,
                    source_surface=SOURCE_SURFACE,
                    sample_limit=5,
                )
            finally:
                store.conn.close()

        self.assertEqual(meta["status"], "queued")
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        lifecycle = status["terminal_output_compaction_lifecycle"]
        self.assertEqual(lifecycle["schema"], "agentflow.terminal_output_compaction_lifecycle_queue_status.v1")
        self.assertEqual(lifecycle["queue_rows"], 1)
        self.assertEqual(lifecycle["event_type_breakdown"], [{"value": "canary-applied", "count": 1}])
        self.assertFalse(lifecycle["payload_json_included"])
        rendered = json.dumps(status, sort_keys=True)
        for raw in RAW_STRINGS:
            self.assertNotIn(raw, rendered)


if __name__ == "__main__":
    unittest.main()
