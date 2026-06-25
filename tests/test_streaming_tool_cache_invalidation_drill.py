from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

from tokenclaw.managed_egress import managed_egress_violations
from tokenclaw.store import Store, stable_json, utc_now
from tokenclaw.streaming_tool_cache_invalidation_drill import (
    BLOCKER_MISSING,
    BLOCKER_NO_TOOL_REPEAT,
    BLOCKER_STABLE,
    BLOCKER_STALE,
    BLOCKER_UNSAFE,
    DRILL_SCHEMA,
    DRILL_SOURCE_SURFACE,
    build_streaming_tool_cache_invalidation_drill,
    record_streaming_tool_cache_invalidation_drill_feedback,
)


class StreamingToolCacheInvalidationDrillTests(unittest.TestCase):
    ENV_KEYS = (
        "TOKENCLAW_RECOMMENDATION_ENABLED",
        "TOKENCLAW_RECOMMENDATION_SERVER_URL",
        "TOKENCLAW_MANAGED",
        "TOKENCLAW_MANAGED_MODE",
    )

    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _log_call(
        self,
        store: Store,
        *,
        call_id: str,
        category: str,
        cache_json: dict,
        stream: bool = True,
        cache_hit: int = 0,
        raw_marker: str = "RAW_MUST_STAY_LOCAL",
    ) -> None:
        request_json = stable_json({
            "messages": [{"content": raw_marker}],
            "file_path": "/home/lutz/private/project/secret.py",
            "cache_key": "secret-cache-key",
        })
        response_json = stable_json({"content": raw_marker})
        store.log_call(
            id=call_id,
            created_at=utc_now(),
            path="/v1/messages",
            requested_model="claude-sonnet-4-5",
            routed_model="claude-sonnet-4-5",
            stream=1 if stream else 0,
            cache_hit=cache_hit,
            status_code=200,
            latency_ms=1500,
            input_tokens_est=2000,
            output_tokens_est=150,
            actual_input_tokens=1900,
            actual_output_tokens=130,
            cost_est_usd=0.5,
            cost_baseline_usd=0.5,
            crunch_json=stable_json({"status": "unchanged"}),
            routing_json=stable_json({"status": "passthrough", "reason": "no-rule"}),
            cache_json=stable_json(cache_json),
            error=None,
            request_json=request_json,
            response_json=response_json,
            session_id="local-session-id-must-stay-local",
            category=category,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=200,
            retry_count=0,
            thinking_output_tokens=0,
            provider="anthropic",
            source_surface="anthropic_messages",
            endpoint="/v1/messages",
            requested_model_family="claude-sonnet",
            routed_model_family="claude-sonnet",
            routing_outcome_label="passthrough",
        )

    def _seed_all_fixtures(self, store: Store) -> None:
        # 1) stable dependency evidence
        self._log_call(
            store,
            call_id="stable-evidence",
            category="tool-result",
            raw_marker="RAW_STABLE_SECRET",
            cache_json={
                "status": "skipped",
                "reason": "streaming-tools-disabled",
                "file_dependency_audit": {
                    "file_watch_enabled": True,
                    "snapshot_count": 3,
                    "snapshot_count_bucket": "few",
                    "safe_invalidation_evidence": True,
                    "invalidation_reason": None,
                    "paths_included": False,
                },
                "session_memory_replayability": {
                    "replayability_level": "local-exact-response-dry-run",
                },
            },
        )
        # 2) missing invalidation evidence
        self._log_call(
            store,
            call_id="missing-evidence",
            category="tool-heavy",
            raw_marker="RAW_MISSING_SECRET",
            cache_json={
                "status": "skipped",
                "reason": "streaming-tools-disabled",
                "file_dependency_audit": {
                    "file_watch_enabled": True,
                    "snapshot_count": 0,
                    "snapshot_count_bucket": "none",
                    "safe_invalidation_evidence": False,
                    "invalidation_reason": "file-dependency-missing",
                    "paths_included": False,
                },
                "session_memory_replayability": {"replayability_level": "features_only"},
            },
        )
        # 3) stale dependency evidence
        self._log_call(
            store,
            call_id="stale-evidence",
            category="tool-result",
            raw_marker="RAW_STALE_SECRET",
            cache_json={
                "status": "skipped",
                "reason": "streaming-tools-disabled",
                "file_dependency_audit": {
                    "file_watch_enabled": True,
                    "snapshot_count": 4,
                    "snapshot_count_bucket": "few",
                    "changed_path_count": 2,
                    "safe_invalidation_evidence": False,
                    "invalidation_reason": "dependency-changed",
                    "paths_included": False,
                },
                "session_memory_replayability": {"replayability_level": "features_only"},
            },
        )
        # 4) unsafe tool-call shape
        self._log_call(
            store,
            call_id="unsafe-shape",
            category="tool-heavy",
            raw_marker="RAW_UNSAFE_SECRET",
            cache_json={
                "status": "skipped",
                "reason": "unsafe-tool-calls-without-invalidation",
                "blockers": ["unsafe-tool-call-shape"],
                "file_dependency_audit": {
                    "file_watch_enabled": True,
                    "snapshot_count": 1,
                    "safe_invalidation_evidence": False,
                    "invalidation_reason": None,
                    "paths_included": False,
                },
                "session_memory_replayability": {"replayability_level": "features_only"},
            },
        )
        # 5) streaming no-tool repeat (two rows = repeat)
        for idx in range(2):
            self._log_call(
                store,
                call_id=f"no-tool-repeat-{idx}",
                category="chat",
                raw_marker="RAW_NO_TOOL_SECRET",
                cache_json={
                    "status": "skipped",
                    "reason": "miss",
                    "session_memory_replayability": {
                        "replayability_level": "local-exact-response-dry-run",
                    },
                },
            )
        # excluded: non-streaming tool-result must not be considered
        self._log_call(
            store,
            call_id="non-streaming-excluded",
            category="tool-result",
            stream=False,
            raw_marker="RAW_NON_STREAMING_SECRET",
            cache_json={"status": "skipped", "reason": "miss"},
        )

    def test_build_drill_classifies_all_blocker_codes_and_excludes_raw_content(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                self._seed_all_fixtures(store)
                rows = store.streaming_tool_cache_invalidation_drill_rows(
                    window_hours=24, limit=100
                )
                payload = build_streaming_tool_cache_invalidation_drill(rows, window_hours=24)
            finally:
                store.conn.close()

        self.assertEqual(payload["schema"], DRILL_SCHEMA)
        # 4 distinct tool fixtures + 2 no-tool repeats considered; non-streaming excluded.
        self.assertEqual(payload["rows_considered"], 6)

        blockers = {
            drill["stale_risk_blocker"] for drill in payload["drills"]
        }
        self.assertEqual(
            blockers,
            {
                BLOCKER_STABLE,
                BLOCKER_MISSING,
                BLOCKER_STALE,
                BLOCKER_UNSAFE,
                BLOCKER_NO_TOOL_REPEAT,
            },
        )

        by_blocker = {drill["stale_risk_blocker"]: drill for drill in payload["drills"]}
        # stable evidence cohort carries safe-invalidation evidence and stable stability.
        self.assertEqual(by_blocker[BLOCKER_STABLE]["dependency_stability"], "stable")
        self.assertEqual(by_blocker[BLOCKER_STABLE]["safe_invalidation_evidence_count"], 1)
        self.assertTrue(by_blocker[BLOCKER_STABLE]["tool_calls_present"])
        # no-tool repeat is observed twice and marked tool-free.
        self.assertEqual(by_blocker[BLOCKER_NO_TOOL_REPEAT]["observed_repeat_count"], 2)
        self.assertFalse(by_blocker[BLOCKER_NO_TOOL_REPEAT]["tool_calls_present"])
        self.assertEqual(by_blocker[BLOCKER_STALE]["dependency_stability"], "stale")
        self.assertEqual(by_blocker[BLOCKER_UNSAFE]["dependency_stability"], "unsafe")
        self.assertEqual(by_blocker[BLOCKER_MISSING]["dependency_stability"], "missing")

        # Acceptance: every evidence class classified, and nothing replay-eligible created.
        acceptance = payload["acceptance"]
        self.assertTrue(acceptance["classifies_stable_dependency_evidence"])
        self.assertTrue(acceptance["classifies_missing_invalidation_evidence"])
        self.assertTrue(acceptance["classifies_stale_dependency_evidence"])
        self.assertTrue(acceptance["classifies_unsafe_tool_call_shape"])
        self.assertTrue(acceptance["classifies_streaming_no_tool_repeat"])
        self.assertTrue(acceptance["no_cache_hit_created"])
        self.assertTrue(acceptance["no_replay_eligible_entry_created"])

        # The drill must never serve cache, create entries, or mutate requests.
        self.assertFalse(payload["serves_cached_responses"])
        self.assertFalse(payload["creates_cache_entry"])
        self.assertEqual(payload["replay_eligible_entries_created"], 0)
        self.assertFalse(payload["mutates_provider_requests"])
        self.assertFalse(payload["tool_cache_replay_enabled"])
        self.assertEqual(payload["observed_cache_hit_count"], 0)

        # Privacy: metadata only, no raw bodies / paths / cache keys / identifiers.
        self.assertTrue(payload["privacy_summary"]["metadata_only"])
        self.assertFalse(managed_egress_violations(payload))
        rendered = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "RAW_STABLE_SECRET",
            "RAW_MISSING_SECRET",
            "RAW_STALE_SECRET",
            "RAW_UNSAFE_SECRET",
            "RAW_NO_TOOL_SECRET",
            "RAW_NON_STREAMING_SECRET",
            "local-session-id-must-stay-local",
            "secret-cache-key",
            "/home/lutz/private/project/secret.py",
        ):
            self.assertNotIn(forbidden, rendered)

    def test_drill_creates_no_cache_entry_when_a_row_was_an_observed_hit(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                # Even a previously-observed cache hit must not become a drill-created
                # replay-eligible entry; the drill only counts, never serves/creates.
                self._log_call(
                    store,
                    call_id="observed-hit",
                    category="tool-result",
                    cache_hit=1,
                    cache_json={
                        "status": "hit",
                        "reason": "exact-hit",
                        "file_dependency_audit": {
                            "file_watch_enabled": True,
                            "snapshot_count": 2,
                            "safe_invalidation_evidence": True,
                            "invalidation_reason": None,
                            "paths_included": False,
                        },
                    },
                )
                before = store.conn.execute("select count(*) as c from calls").fetchone()["c"]
                rows = store.streaming_tool_cache_invalidation_drill_rows(window_hours=24, limit=100)
                payload = build_streaming_tool_cache_invalidation_drill(rows, window_hours=24)
                after = store.conn.execute("select count(*) as c from calls").fetchone()["c"]
            finally:
                store.conn.close()

        self.assertEqual(before, after)  # drill wrote nothing to the store
        self.assertEqual(payload["observed_cache_hit_count"], 1)
        self.assertEqual(payload["replay_eligible_entries_created"], 0)
        self.assertTrue(payload["acceptance"]["no_replay_eligible_entry_created"])
        self.assertFalse(payload["wrote_store"])

    def test_record_queues_drill_when_managed_is_disabled(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            store = Store(tmp.name)
            try:
                self._log_call(
                    store,
                    call_id="stable-evidence",
                    category="tool-result",
                    cache_json={
                        "status": "skipped",
                        "reason": "streaming-tools-disabled",
                        "file_dependency_audit": {
                            "file_watch_enabled": True,
                            "snapshot_count": 3,
                            "safe_invalidation_evidence": True,
                            "invalidation_reason": None,
                            "paths_included": False,
                        },
                    },
                )
                meta = asyncio.run(
                    record_streaming_tool_cache_invalidation_drill_feedback(
                        store,
                        window_hours=24,
                        max_rows=100,
                        max_cohorts=10,
                    )
                )
                row = store.conn.execute(
                    "select source_surface, endpoint, status, payload_json "
                    "from managed_outcome_feedback_queue"
                ).fetchone()
            finally:
                store.conn.close()

        self.assertFalse(meta["enabled"])
        self.assertEqual(meta["reason"], "queued-managed-disabled")
        self.assertEqual(meta["rows_considered"], 1)
        self.assertEqual(row["source_surface"], DRILL_SOURCE_SURFACE)
        self.assertEqual(row["endpoint"], "/v1/policy-events")
        self.assertEqual(row["status"], "queued")
        self.assertIn(DRILL_SCHEMA, row["payload_json"])


if __name__ == "__main__":
    unittest.main()
