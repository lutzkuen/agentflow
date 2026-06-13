import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentflow_proxy.store import PostgresConnection, SQLiteStore, Store, stable_json, utc_now


class StoreBackendTest(unittest.TestCase):
    def setUp(self):
        self.saved_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
        self.saved_busy_timeout = os.environ.get("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS")
        self.saved_wal = os.environ.get("AGENTFLOW_SQLITE_WAL")
        self.saved_retention_days = os.environ.get("AGENTFLOW_SQLITE_RETENTION_DAYS")
        self.saved_retention_enabled = os.environ.get("AGENTFLOW_SQLITE_RETENTION_ENABLED")
        os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        os.environ.pop("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS", None)
        os.environ.pop("AGENTFLOW_SQLITE_WAL", None)
        os.environ.pop("AGENTFLOW_SQLITE_RETENTION_DAYS", None)
        os.environ.pop("AGENTFLOW_SQLITE_RETENTION_ENABLED", None)

    def tearDown(self):
        if self.saved_database_url is None:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        else:
            os.environ["AGENTFLOW_DATABASE_URL"] = self.saved_database_url
        if self.saved_busy_timeout is None:
            os.environ.pop("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS", None)
        else:
            os.environ["AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS"] = self.saved_busy_timeout
        if self.saved_wal is None:
            os.environ.pop("AGENTFLOW_SQLITE_WAL", None)
        else:
            os.environ["AGENTFLOW_SQLITE_WAL"] = self.saved_wal
        if self.saved_retention_days is None:
            os.environ.pop("AGENTFLOW_SQLITE_RETENTION_DAYS", None)
        else:
            os.environ["AGENTFLOW_SQLITE_RETENTION_DAYS"] = self.saved_retention_days
        if self.saved_retention_enabled is None:
            os.environ.pop("AGENTFLOW_SQLITE_RETENTION_ENABLED", None)
        else:
            os.environ["AGENTFLOW_SQLITE_RETENTION_ENABLED"] = self.saved_retention_enabled

    def test_store_uses_sqlite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                self.assertIsInstance(store, SQLiteStore)
                self.assertEqual(store.backend, "sqlite")
                self.assertTrue(store.database_url.startswith("sqlite:///"))
            finally:
                store.conn.close()

    def test_invalid_database_url_is_rejected_before_driver_import(self):
        os.environ["AGENTFLOW_DATABASE_URL"] = "mysql://localhost/agentflow"

        with self.assertRaisesRegex(ValueError, "postgresql:// or postgres://"):
            Store()

    def test_concurrent_sqlite_reads_and_writes_use_store_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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

    def test_sqlite_store_enables_busy_timeout_wal_and_dashboard_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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
                self.assertIn("idx_agentflow_sqlite_maintenance_runs_recent", indexes)
            finally:
                store.conn.close()

    def test_sqlite_retention_defaults_to_seven_days_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
            try:
                status = store.sqlite_retention_status()
                self.assertTrue(status["enabled"])
                self.assertEqual(status["retention_days"], 7)
                self.assertEqual(status["configured_by"], "local-default")

                os.environ["AGENTFLOW_SQLITE_RETENTION_DAYS"] = "0"
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
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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
            store = Store(str(Path(tmp) / "agentflow.sqlite3"))
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
            db_path = str(Path(tmp) / "agentflow.sqlite3")
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
