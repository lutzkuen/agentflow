import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentflow_proxy.store import PostgresConnection, SQLiteStore, Store, stable_json, utc_now


class StoreBackendTest(unittest.TestCase):
    def setUp(self):
        self.saved_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
        self.saved_busy_timeout = os.environ.get("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS")
        self.saved_wal = os.environ.get("AGENTFLOW_SQLITE_WAL")
        os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        os.environ.pop("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS", None)
        os.environ.pop("AGENTFLOW_SQLITE_WAL", None)

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
