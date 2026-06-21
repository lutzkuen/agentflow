import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from tokenclaw.cli_commands.onboarding import tokenclaw_cli
from tokenclaw.db_adoption import adopt_legacy_sqlite_evidence, detect_legacy_evidence_gap
from tokenclaw.store import PostgresConnection, SQLiteStore, Store, stable_json, utc_now


class StoreBackendTest(unittest.TestCase):
    def setUp(self):
        self.saved_database_url = os.environ.get("TOKENCLAW_DATABASE_URL")
        self.saved_busy_timeout = os.environ.get("TOKENCLAW_SQLITE_BUSY_TIMEOUT_MS")
        self.saved_wal = os.environ.get("TOKENCLAW_SQLITE_WAL")
        self.saved_retention_days = os.environ.get("TOKENCLAW_SQLITE_RETENTION_DAYS")
        self.saved_retention_enabled = os.environ.get("TOKENCLAW_SQLITE_RETENTION_ENABLED")
        self.saved_legacy_db_warning = os.environ.get("TOKENCLAW_LEGACY_DB_WARNING")
        os.environ.pop("TOKENCLAW_DATABASE_URL", None)
        os.environ.pop("TOKENCLAW_SQLITE_BUSY_TIMEOUT_MS", None)
        os.environ.pop("TOKENCLAW_SQLITE_WAL", None)
        os.environ.pop("TOKENCLAW_SQLITE_RETENTION_DAYS", None)
        os.environ.pop("TOKENCLAW_SQLITE_RETENTION_ENABLED", None)
        os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = "0"

    def tearDown(self):
        if self.saved_database_url is None:
            os.environ.pop("TOKENCLAW_DATABASE_URL", None)
        else:
            os.environ["TOKENCLAW_DATABASE_URL"] = self.saved_database_url
        if self.saved_busy_timeout is None:
            os.environ.pop("TOKENCLAW_SQLITE_BUSY_TIMEOUT_MS", None)
        else:
            os.environ["TOKENCLAW_SQLITE_BUSY_TIMEOUT_MS"] = self.saved_busy_timeout
        if self.saved_wal is None:
            os.environ.pop("TOKENCLAW_SQLITE_WAL", None)
        else:
            os.environ["TOKENCLAW_SQLITE_WAL"] = self.saved_wal
        if self.saved_retention_days is None:
            os.environ.pop("TOKENCLAW_SQLITE_RETENTION_DAYS", None)
        else:
            os.environ["TOKENCLAW_SQLITE_RETENTION_DAYS"] = self.saved_retention_days
        if self.saved_retention_enabled is None:
            os.environ.pop("TOKENCLAW_SQLITE_RETENTION_ENABLED", None)
        else:
            os.environ["TOKENCLAW_SQLITE_RETENTION_ENABLED"] = self.saved_retention_enabled
        if self.saved_legacy_db_warning is None:
            os.environ.pop("TOKENCLAW_LEGACY_DB_WARNING", None)
        else:
            os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = self.saved_legacy_db_warning

    def test_adopt_legacy_sqlite_evidence_is_idempotent_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_db = root / "tokenclaw.sqlite3"
            legacy_db = root / "agentflow.sqlite3"
            legacy = Store(str(legacy_db))
            canonical = Store(str(canonical_db))
            try:
                canonical.log_call(
                    id="shared-call",
                    created_at="2026-06-21T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="canonical-model",
                    routed_model="canonical-model",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1,
                    input_tokens_est=1,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "canonical"}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )
                legacy.log_call(
                    id="shared-call",
                    created_at="2026-06-20T10:00:00+00:00",
                    path="/v1/messages",
                    requested_model="legacy-model",
                    routed_model="legacy-model",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=2,
                    input_tokens_est=2,
                    output_tokens_est=2,
                    cost_est_usd=0.002,
                    cost_baseline_usd=0.002,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "legacy"}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )
                legacy.log_call(
                    id="legacy-only-call",
                    created_at="2026-06-21T11:00:00+00:00",
                    path="/v1/messages",
                    requested_model="legacy-only-model",
                    routed_model="legacy-only-model",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=3,
                    input_tokens_est=3,
                    output_tokens_est=3,
                    cost_est_usd=0.003,
                    cost_baseline_usd=0.003,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "legacy-only"}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )
                legacy.persist_request_shape_rollups(
                    run_id="legacy-run",
                    generated_at="2026-06-21T11:05:00+00:00",
                    rows=[
                        {
                            "id": "legacy-rollup",
                            "run_id": "legacy-run",
                            "generated_at": "2026-06-21T11:05:00+00:00",
                            "rollup_key": "surface:endpoint",
                            "candidate_id": "candidate-1",
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
                            "row_count": 7,
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
                        "snapshot_id": "legacy-snapshot",
                        "run_id": "legacy-run",
                        "generated_at": "2026-06-21T11:05:00+00:00",
                        "window": {"source": "test", "start": "2026-06-21T10:00:00+00:00", "end": "2026-06-21T11:00:00+00:00"},
                        "summary": {
                            "rows_considered": 2,
                            "rollup_count": 1,
                            "ranked_candidate_count": 1,
                            "top_next_action": "rank",
                            "top_local_action_family": "crunch",
                            "top_readiness_state": "ready",
                            "total_projected_savings_usd": 0.1,
                        },
                    }
                )
            finally:
                legacy.conn.close()
                canonical.conn.close()

            dry_run = adopt_legacy_sqlite_evidence(canonical_db=canonical_db, legacy_db=legacy_db, dry_run=True)
            self.assertTrue(dry_run["ok"])
            self.assertEqual(dry_run["status"], "dry-run")
            self.assertEqual(dry_run["summary"]["rows_inserted"], 3)
            with sqlite3.connect(str(canonical_db)) as conn:
                self.assertEqual(conn.execute("select count(*) from calls").fetchone()[0], 1)

            result = adopt_legacy_sqlite_evidence(canonical_db=canonical_db, legacy_db=legacy_db)
            self.assertTrue(result["ok"])
            self.assertEqual(result["legacy_open_mode"], "ro")
            self.assertEqual(result["summary"]["rows_inserted"], 3)
            self.assertEqual(result["summary"]["rows_skipped"], 1)

            second = adopt_legacy_sqlite_evidence(canonical_db=canonical_db, legacy_db=legacy_db)
            self.assertEqual(second["summary"]["rows_inserted"], 0)
            self.assertEqual(second["summary"]["rows_skipped"], 4)

            with sqlite3.connect(str(canonical_db)) as conn:
                self.assertEqual(conn.execute("select count(*) from calls").fetchone()[0], 2)
                self.assertEqual(conn.execute("select count(*) from request_shape_rollups").fetchone()[0], 1)
                self.assertEqual(conn.execute("select count(*) from request_shape_rollup_snapshots").fetchone()[0], 1)
                row = conn.execute("select requested_model from calls where id = 'shared-call'").fetchone()
                self.assertEqual(row[0], "canonical-model")

    def test_detect_legacy_evidence_gap_reports_richer_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_db = root / "tokenclaw.sqlite3"
            legacy_db = root / "agentflow.sqlite3"
            canonical = Store(str(canonical_db))
            legacy = Store(str(legacy_db))
            try:
                legacy.log_call(
                    id="newer-legacy-call",
                    created_at="2026-06-21T12:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-5",
                    routed_model="claude-sonnet-4-5",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1,
                    input_tokens_est=1,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "legacy"}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )
            finally:
                canonical.conn.close()
                legacy.conn.close()

            detection = detect_legacy_evidence_gap(canonical_db=canonical_db)
            self.assertTrue(detection["richer_legacy_detected"])
            self.assertEqual(detection["status"], "richer-legacy-detected")
            self.assertIn("legacy-has-more-calls", detection["reason_codes"])
            self.assertEqual(detection["bottleneck_signal"]["blocker_code"], "stranded-legacy-agentflow-sqlite-evidence")

    def test_tokenclaw_db_adopt_legacy_cli_outputs_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_db = root / "tokenclaw.sqlite3"
            legacy_db = root / "agentflow.sqlite3"
            legacy = Store(str(legacy_db))
            try:
                legacy.log_call(
                    id="legacy-cli-call",
                    created_at="2026-06-21T12:00:00+00:00",
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-5",
                    routed_model="claude-sonnet-4-5",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1,
                    input_tokens_est=1,
                    output_tokens_est=1,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "legacy"}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )
            finally:
                legacy.conn.close()

            stdout = StringIO()
            code = tokenclaw_cli(
                ["db", "adopt-legacy", "--db", str(canonical_db), "--from", str(legacy_db)],
                stdout=stdout,
                stderr=StringIO(),
            )
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["schema"], "tokenclaw.legacy_sqlite_evidence_adoption.v1")
            self.assertEqual(payload["summary"]["rows_inserted"], 1)

    def test_store_uses_sqlite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                self.assertIsInstance(store, SQLiteStore)
                self.assertEqual(store.backend, "sqlite")
                self.assertTrue(store.database_url.startswith("sqlite:///"))
            finally:
                store.conn.close()

    def test_invalid_database_url_is_rejected_before_driver_import(self):
        os.environ["TOKENCLAW_DATABASE_URL"] = "mysql://localhost/tokenclaw"

        with self.assertRaisesRegex(ValueError, "postgresql:// or postgres://"):
            Store()

    def test_concurrent_sqlite_reads_and_writes_use_store_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                def write_call(i):
                    store.log_call(
                        id=f"call-{i}",
                        created_at=utc_now(),
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-5",
                        routed_model="claude-haiku-4-5",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=i,
                        input_tokens_est=10,
                        output_tokens_est=5,
                        cost_est_usd=0.001,
                        cost_baseline_usd=0.002,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({"reason": "test"}),
                        cache_json=stable_json({"status": "miss"}),
                        session_id="session-a",
                        category="chat",
                    )
                    if i % 5 == 0:
                        store.set_cache(f"cache-{i}", "model", 10, {"i": i})
                    return store.conn.execute("select count(*) as c from calls").fetchone()["c"]

                with ThreadPoolExecutor(max_workers=8) as pool:
                    counts = list(pool.map(write_call, range(40)))

                self.assertEqual(max(counts), 40)
                self.assertEqual(store.conn.execute("select count(*) as c from calls").fetchone()["c"], 40)
                self.assertEqual(store.get_cache("cache-35"), {"i": 35})
            finally:
                store.conn.close()

    def test_log_call_records_policy_decision_blocker_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="missing-policy-decision",
                    created_at=utc_now(),
                    path="/v1/messages",
                    provider="anthropic",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=1,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=12,
                    input_tokens_est=10,
                    output_tokens_est=5,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.001,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=None,
                    cache_json=stable_json({"status": "skipped"}),
                    category="tool-result",
                )

                row = store.conn.execute(
                    "select source_surface, endpoint, routing_json from calls where id = ?",
                    ("missing-policy-decision",),
                ).fetchone()
                self.assertEqual(row["source_surface"], "anthropic_messages")
                managed = json.loads(row["routing_json"])["managed_recommendation"]
                self.assertEqual(managed["schema"], "tokenclaw.managed_policy_decision_evaluation.v1")
                self.assertEqual(managed["status"], "skipped-local-blocker")
                self.assertEqual(managed["reason"], "policy-decision-metadata-missing")
                self.assertTrue(managed["coverage_denominator_included"])
                self.assertTrue(managed["expected_evaluation"])
                self.assertEqual(managed["fallback"], "local-policy")
                self.assertTrue(managed["privacy"]["metadata_only"])
                self.assertFalse(managed["privacy"]["provider_bodies_included"])
            finally:
                store.conn.close()

    def test_log_call_defaults_routing_outcome_label_to_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                columns = {
                    row["name"]
                    for row in store.conn.execute("pragma table_info(calls)").fetchall()
                }
                self.assertIn("routing_outcome_label", columns)

                store.log_call(
                    id="label-default",
                    created_at=utc_now(),
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=12,
                    input_tokens_est=10,
                    output_tokens_est=5,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "test"}),
                    cache_json=stable_json({"status": "miss"}),
                    session_id="session-label",
                    category="chat",
                )

                row = store.conn.execute(
                    "select routing_outcome_label from calls where id = ?",
                    ("label-default",),
                ).fetchone()
                self.assertEqual(row["routing_outcome_label"], "unknown")
            finally:
                store.conn.close()

    def test_log_call_extracts_managed_routing_json_for_dashboard_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                columns = {
                    row["name"]
                    for row in store.conn.execute("pragma table_info(calls)").fetchall()
                }
                self.assertIn("managed_routing_json", columns)

                store.log_call(
                    id="managed-routing",
                    created_at=utc_now(),
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=12,
                    input_tokens_est=10,
                    output_tokens_est=5,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({
                        "reason": "managed route",
                        "managed_recommendation": {
                            "schema": "tokenclaw.managed_policy_decision_evaluation.v1",
                            "enabled": True,
                            "status": "received",
                            "policy_id": "managed-route-1",
                            "target_model": "claude-haiku-4-5-20251001",
                            "confidence": 0.91,
                            "applied": True,
                            "local_action_taken": "route_to",
                        },
                    }),
                    cache_json=stable_json({"status": "miss"}),
                    session_id="session-managed",
                    category="tool-result",
                )

                row = store.conn.execute(
                    "select managed_routing_json from calls where id = ?",
                    ("managed-routing",),
                ).fetchone()
                managed = json.loads(row["managed_routing_json"])
                self.assertEqual(managed["policy_id"], "managed-route-1")
                self.assertTrue(managed["applied"])
                self.assertEqual(managed["confidence"], 0.91)
            finally:
                store.conn.close()

    def test_finalize_outcome_labels_marks_clean_older_call_safe(self):
        now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
        older = (now - timedelta(seconds=120)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_call(
                    id="safe-call",
                    created_at=older,
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-haiku-4-5-20251001",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=12,
                    input_tokens_est=10,
                    output_tokens_est=5,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "test"}),
                    cache_json=stable_json({"status": "miss"}),
                    session_id="session-safe",
                    category="chat",
                )

                result = store.finalize_outcome_labels(now=now.isoformat())
                row = store.conn.execute(
                    "select routing_outcome_label from calls where id = ?",
                    ("safe-call",),
                ).fetchone()

                self.assertEqual(result["safe_count"], 1)
                self.assertEqual(row["routing_outcome_label"], "safe")
            finally:
                store.conn.close()

    def test_finalize_outcome_labels_marks_errors_fallbacks_and_retries_unsafe(self):
        now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
        base = now - timedelta(seconds=180)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                cases = [
                    ("http-error", "session-error", 500, 0, {"reason": "test"}, 0),
                    ("fallback", "session-fallback", 200, 0, {"fallback": True}, 5),
                    ("own-retry", "session-own-retry", 200, 1, {"reason": "test"}, 10),
                    ("followed-by-retry", "session-followed", 200, 0, {"reason": "test"}, 20),
                    ("retry-sibling", "session-followed", 200, 1, {"reason": "test"}, 40),
                ]
                for call_id, session_id, status_code, retry_count, routing, offset in cases:
                    store.log_call(
                        id=call_id,
                        created_at=(base + timedelta(seconds=offset)).isoformat(),
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001",
                        stream=0,
                        cache_hit=0,
                        status_code=status_code,
                        latency_ms=12,
                        input_tokens_est=10,
                        output_tokens_est=5,
                        cost_est_usd=0.001,
                        cost_baseline_usd=0.002,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json(routing),
                        cache_json=stable_json({"status": "miss"}),
                        session_id=session_id,
                        category="chat",
                        retry_count=retry_count,
                    )

                result = store.finalize_outcome_labels(now=now.isoformat())
                labels = {
                    row["id"]: row["routing_outcome_label"]
                    for row in store.conn.execute(
                        "select id, routing_outcome_label from calls"
                    ).fetchall()
                }

                self.assertGreaterEqual(result["unsafe_count"], 4)
                self.assertEqual(labels["http-error"], "unsafe")
                self.assertEqual(labels["fallback"], "unsafe")
                self.assertEqual(labels["own-retry"], "unsafe")
                self.assertEqual(labels["followed-by-retry"], "unsafe")
            finally:
                store.conn.close()

    def test_finalize_outcome_labels_keeps_recent_or_sessionless_calls_unknown(self):
        now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                for call_id, created_at, session_id in (
                    ("recent-call", now - timedelta(seconds=30), "session-recent"),
                    ("sessionless-call", now - timedelta(seconds=120), None),
                ):
                    store.log_call(
                        id=call_id,
                        created_at=created_at.isoformat(),
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-haiku-4-5-20251001",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=12,
                        input_tokens_est=10,
                        output_tokens_est=5,
                        cost_est_usd=0.001,
                        cost_baseline_usd=0.002,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({"reason": "test"}),
                        cache_json=stable_json({"status": "miss"}),
                        session_id=session_id,
                        category="chat",
                    )

                result = store.finalize_outcome_labels(now=now.isoformat())
                labels = {
                    row["id"]: row["routing_outcome_label"]
                    for row in store.conn.execute(
                        "select id, routing_outcome_label from calls"
                    ).fetchall()
                }

                self.assertEqual(result["safe_count"], 0)
                self.assertEqual(result["unknown_count"], 1)
                self.assertEqual(labels["recent-call"], "unknown")
                self.assertEqual(labels["sessionless-call"], "unknown")
            finally:
                store.conn.close()

    def test_log_call_preserves_existing_managed_policy_decision_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                managed = {
                    "schema": "tokenclaw.policy_decision.v1",
                    "status": "received",
                    "policy_id": "policy-1",
                    "applied": True,
                }
                store.log_call(
                    id="real-policy-decision",
                    created_at=utc_now(),
                    path="/v1/responses",
                    provider="openai",
                    source_surface="openai_responses",
                    endpoint="responses",
                    requested_model="gpt-5.4",
                    routed_model="gpt-5.4-mini",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=12,
                    input_tokens_est=10,
                    output_tokens_est=5,
                    cost_est_usd=0.001,
                    cost_baseline_usd=0.002,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"managed_recommendation": managed}),
                    cache_json=stable_json({"status": "miss"}),
                    category="chat",
                )

                row = store.conn.execute(
                    "select routing_json from calls where id = ?",
                    ("real-policy-decision",),
                ).fetchone()
                self.assertEqual(json.loads(row["routing_json"])["managed_recommendation"], managed)
            finally:
                store.conn.close()

    def test_codex_turn_start_records_policy_decision_blocker_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                store.log_codex_app_event(
                    id="codex-turn-start",
                    created_at=utc_now(),
                    direction="client_to_server",
                    method="turn/start",
                    routing_json=None,
                )
                store.log_codex_app_event(
                    id="codex-turn-completed",
                    created_at=utc_now(),
                    direction="server_to_client",
                    method="turn/completed",
                    routing_json=None,
                )

                rows = {
                    row["id"]: (
                        json.loads(row["routing_json"])["managed_recommendation"]
                        if row["routing_json"]
                        else None
                    )
                    for row in store.conn.execute(
                        "select id, routing_json from codex_app_events order by id"
                    ).fetchall()
                }
                self.assertEqual(rows["codex-turn-start"]["source_surface"], "codex_turn")
                self.assertEqual(rows["codex-turn-start"]["status"], "skipped-local-blocker")
                self.assertEqual(rows["codex-turn-start"]["reason"], "policy-decision-metadata-missing")
                self.assertTrue(rows["codex-turn-start"]["coverage_denominator_included"])
                self.assertIsNone(rows["codex-turn-completed"])
            finally:
                store.conn.close()

    def test_sqlite_store_enables_busy_timeout_wal_and_dashboard_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                busy_timeout = store.conn.execute("pragma busy_timeout").fetchone()[0]
                journal_mode = store.conn.execute("pragma journal_mode").fetchone()[0]
                indexes = {
                    row["name"]
                    for row in store.conn.execute("""
                        select name from sqlite_master
                        where type = 'index'
                    """).fetchall()
                }

                self.assertGreaterEqual(busy_timeout, 5000)
                self.assertEqual(str(journal_mode).lower(), "wal")
                self.assertIn("idx_calls_created_at", indexes)
                self.assertIn("idx_codex_app_events_created_at", indexes)
                self.assertIn("idx_codex_app_events_start_recent", indexes)
                self.assertIn("idx_codex_app_events_response_lookup", indexes)
                self.assertIn("idx_tokenclaw_sqlite_maintenance_runs_recent", indexes)
            finally:
                store.conn.close()

    def test_sqlite_retention_defaults_to_seven_days_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                status = store.sqlite_retention_status()
                self.assertTrue(status["enabled"])
                self.assertEqual(status["retention_days"], 7)
                self.assertEqual(status["configured_by"], "local-default")

                os.environ["TOKENCLAW_SQLITE_RETENTION_DAYS"] = "0"
                disabled = store.sqlite_retention_status()
                self.assertFalse(disabled["enabled"])
                self.assertIsNone(disabled["retention_days"])
            finally:
                store.conn.close()

    def test_sqlite_maintenance_purges_old_rows_and_keeps_recent_rows(self):
        now = datetime(2026, 6, 13, 8, 0, tzinfo=timezone.utc)
        old = (now - timedelta(days=9)).isoformat()
        recent = (now - timedelta(days=2)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                for call_id, created_at in (("old-call", old), ("recent-call", recent)):
                    store.log_call(
                        id=call_id,
                        created_at=created_at,
                        path="/v1/messages",
                        requested_model="claude-sonnet-4-6",
                        routed_model="claude-sonnet-4-6",
                        stream=0,
                        cache_hit=0,
                        status_code=200,
                        latency_ms=1,
                        input_tokens_est=1,
                        output_tokens_est=1,
                        cost_est_usd=0.001,
                        cost_baseline_usd=0.001,
                        crunch_json=stable_json({"changed": False}),
                        routing_json=stable_json({"reason": "test"}),
                        cache_json=stable_json({"status": "miss"}),
                    )
                    store.log_codex_app_event(
                        id=f"{call_id}-codex",
                        created_at=created_at,
                        direction="server_to_client",
                        method="turn/completed",
                    )
                    store.conn.execute(
                        """
                        insert into routing_experiments(id, call_id, created_at, requested_model, routed_model, primary_model, shadow_model)
                        values (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (f"{call_id}-experiment", call_id, created_at, "gpt-5", "gpt-5-mini", "gpt-5-mini", "gpt-5"),
                    )
                    store.log_provider_tool_adoption_window(
                        id=f"{call_id}-adoption",
                        created_at=created_at,
                        updated_at=created_at,
                        provider="openai",
                        status="fulfilled",
                    )
                    store.log_optimization_eval_result(
                        id=f"{call_id}-eval",
                        run_id="run",
                        created_at=created_at,
                        candidate_id="candidate",
                        status_class="pass",
                        result_json=stable_json({"ok": True}),
                    )

                store.set_cache("old-cache", "model", 10, {"old": True}, file_deps=[{"path": "/tmp/old", "exists": False}])
                store.conn.execute("update cache set created_at = ? where cache_key = ?", (old, "old-cache"))
                store.set_cache("recent-cache", "model", 10, {"recent": True}, file_deps=[{"path": "/tmp/recent", "exists": False}])
                store.conn.execute("update cache set created_at = ? where cache_key = ?", (recent, "recent-cache"))
                store.set_semantic_cache("old-semantic", "model", [1.0], {"old": True}, 10)
                store.conn.execute("update semantic_cache set created_at = ? where cache_key = ?", (old, "old-semantic"))
                store.set_semantic_cache("recent-semantic", "model", [1.0], {"recent": True}, 10)
                store.conn.execute("update semantic_cache set created_at = ? where cache_key = ?", (recent, "recent-semantic"))

                result = store.run_sqlite_maintenance(retention_days=7, now=now.isoformat())

                self.assertEqual(result["status"], "ok")
                self.assertGreaterEqual(result["total_deleted_rows"], 8)
                self.assertEqual(store.conn.execute("select count(*) as c from calls where id = 'old-call'").fetchone()["c"], 0)
                self.assertEqual(store.conn.execute("select count(*) as c from calls where id = 'recent-call'").fetchone()["c"], 1)
                self.assertEqual(store.conn.execute("select count(*) as c from cache where cache_key = 'old-cache'").fetchone()["c"], 0)
                self.assertEqual(store.conn.execute("select count(*) as c from cache_file_deps where cache_key = 'old-cache'").fetchone()["c"], 0)
                self.assertEqual(store.conn.execute("select count(*) as c from cache where cache_key = 'recent-cache'").fetchone()["c"], 1)
                self.assertEqual(store.conn.execute("select count(*) as c from codex_app_events where id = 'old-call-codex'").fetchone()["c"], 0)
                self.assertEqual(store.conn.execute("select count(*) as c from codex_app_events where id = 'recent-call-codex'").fetchone()["c"], 1)
                self.assertTrue(store.latest_sqlite_maintenance_run()["optimize_ran"])
            finally:
                store.conn.close()

    def test_sqlite_maintenance_preserves_unsent_feedback_rows(self):
        now = datetime(2026, 6, 13, 8, 0, tzinfo=timezone.utc)
        old = (now - timedelta(days=10)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                for status in ("queued", "retryable-error", "sending", "sent", "error", "dropped-after-limit"):
                    store.enqueue_managed_outcome_feedback(
                        id=f"feedback-{status}",
                        created_at=old,
                        updated_at=old,
                        source_surface="test",
                        endpoint="/v1/policy-events",
                        optimization_unit_id=1,
                        payload_json=stable_json({"schema": "test"}),
                        status=status,
                        next_attempt_at=old,
                    )

                result = store.run_sqlite_maintenance(retention_days=7, now=now.isoformat())

                remaining = {
                    row["id"]
                    for row in store.conn.execute("select id from managed_outcome_feedback_queue").fetchall()
                }
                self.assertEqual(result["deleted_rows"]["managed_outcome_feedback_queue"], 3)
                self.assertEqual(remaining, {"feedback-queued", "feedback-retryable-error", "feedback-sending"})
            finally:
                store.conn.close()

    def test_dashboard_codex_event_plans_use_created_at_and_turn_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "tokenclaw.sqlite3"))
            try:
                recent_plan = " ".join(
                    row["detail"]
                    for row in store.conn.execute("""
                        explain query plan
                        select created_at, direction
                        from codex_app_events
                        order by created_at desc
                        limit 50
                    """).fetchall()
                )
                today_plan = " ".join(
                    row["detail"]
                    for row in store.conn.execute("""
                        explain query plan
                        select count(*)
                        from codex_app_events
                        where created_at >= ?
                    """, ("2026-06-08T00:00:00+00:00",)).fetchall()
                )
                turn_plan = " ".join(
                    row["detail"]
                    for row in store.conn.execute("""
                        explain query plan
                        select s.id,
                               (
                                   select r.id from codex_app_events r
                                   where r.direction = 'server_to_client'
                                     and r.request_id = s.request_id
                                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                                   order by r.created_at desc
                                   limit 1
                               ) as response_event_id
                        from codex_app_events s
                        where s.direction = 'client_to_server'
                          and s.method = 'turn/start'
                    """).fetchall()
                )

                self.assertIn("idx_codex_app_events_created_at", recent_plan)
                self.assertIn("idx_codex_app_events_created_at", today_plan)
                self.assertIn("idx_codex_app_events_start_recent", turn_plan)
                self.assertIn("idx_codex_app_events_response_lookup", turn_plan)
            finally:
                store.conn.close()

    def test_sqlite_wal_allows_dashboard_reads_during_writer_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tokenclaw.sqlite3")
            writer_store = Store(db_path)
            reader_store = Store(db_path)
            raw_writer = sqlite3.connect(db_path, timeout=0.1)
            try:
                writer_store.log_call(
                    id="existing-call",
                    created_at=utc_now(),
                    path="/v1/messages",
                    requested_model="claude-sonnet-4-6",
                    routed_model="claude-sonnet-4-6",
                    stream=0,
                    cache_hit=0,
                    status_code=200,
                    latency_ms=1,
                    input_tokens_est=10,
                    output_tokens_est=1,
                    cost_est_usd=0.0,
                    cost_baseline_usd=0.0,
                    crunch_json=stable_json({"changed": False}),
                    routing_json=stable_json({"reason": "test"}),
                    cache_json=stable_json({"status": "miss"}),
                    session_id="session-lock",
                    category="chat",
                )
                raw_writer.execute("begin exclusive")
                raw_writer.execute("""
                    insert into cache(cache_key, created_at, model, response_json, request_chars, response_chars)
                    values (?, ?, ?, ?, ?, ?)
                """, ("pending-cache", utc_now(), "model", "{}", 2, 2))

                count = reader_store.conn.execute("select count(*) as c from calls").fetchone()["c"]

                self.assertEqual(count, 1)
            finally:
                try:
                    raw_writer.rollback()
                except sqlite3.Error:
                    pass
                raw_writer.close()
                reader_store.conn.close()
                writer_store.conn.close()

    def test_postgres_translation_covers_dashboard_sql_patterns(self):
        translated = PostgresConnection._translate_sql("""
            select date(created_at) as day,
                   sum(json_extract(crunch_json, '$.saved_chars')) as saved,
                   avg(json_extract(crunch_json, '$.crunch_ratio')) as ratio,
                   json_extract(crunch_json, '$.changed') as changed
            from calls
            where date(created_at) >= date('now', '-6 days')
              and json_extract(crunch_json, '$.changed') = 1
              and session_id = ?
            group by date(created_at)
        """)

        self.assertIn("created_at::date as day", translated)
        self.assertIn("CURRENT_DATE - interval '6 days'", translated)
        self.assertIn("sum((jsonb_extract_path_text(crunch_json::jsonb, 'saved_chars'))::numeric)", translated)
        self.assertIn("avg((jsonb_extract_path_text(crunch_json::jsonb, 'crunch_ratio'))::numeric)", translated)
        self.assertIn("jsonb_extract_path_text(crunch_json::jsonb, 'changed') in ('1', 'true')", translated)
        self.assertIn("session_id = %s", translated)

    def test_postgres_translation_covers_recent_summary_interval(self):
        translated = PostgresConnection._translate_sql(
            "SELECT * FROM calls WHERE datetime(created_at) >= datetime('now', ?)"
        )

        self.assertEqual(
            translated,
            "SELECT * FROM calls WHERE created_at >= (now() + %s::interval)",
        )


if __name__ == "__main__":
    unittest.main()
