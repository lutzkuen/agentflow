import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentflow_proxy.store import PostgresConnection, SQLiteStore, Store, stable_json, utc_now


class StoreBackendTest(unittest.TestCase):
    def setUp(self):
        self.saved_database_url = os.environ.get("AGENTFLOW_DATABASE_URL")
        os.environ.pop("AGENTFLOW_DATABASE_URL", None)

    def tearDown(self):
        if self.saved_database_url is None:
            os.environ.pop("AGENTFLOW_DATABASE_URL", None)
        else:
            os.environ["AGENTFLOW_DATABASE_URL"] = self.saved_database_url

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
