from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _dependency_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2_5"
    if count <= 20:
        return "6_20"
    if count <= 128:
        return "21_128"
    return "128_plus"


def _cache_file_dependency_audit_from_rows(rows: list[Any]) -> dict[str, Any]:
    changed = 0
    deleted = 0
    created = 0
    missing = 0
    for row in rows:
        expected_exists = bool(row["exists_flag"])
        if not expected_exists:
            missing += 1
        path = Path(row["path"])
        try:
            stat = path.stat()
            current_exists = path.is_file()
        except OSError:
            stat = None
            current_exists = False
        if expected_exists and not current_exists:
            deleted += 1
        elif not expected_exists and current_exists:
            created += 1
        elif expected_exists and current_exists and stat is not None:
            if row["mtime_ns"] != stat.st_mtime_ns or row["size"] != stat.st_size:
                changed += 1
    count = len(rows)
    invalidation_reason = None
    if deleted:
        invalidation_reason = "dependency-deleted"
    elif changed or created:
        invalidation_reason = "dependency-changed"
    elif not rows:
        invalidation_reason = "file-dependency-missing"
    elif missing:
        invalidation_reason = "dependency-missing"
    safe = bool(count > 0 and not (changed or deleted or created or missing))
    return {
        "schema": "agentflow.cache_file_dependency_audit.v1",
        "file_watch_enabled": True,
        "snapshot_root_policy": "stored-local-paths",
        "root_path_included": False,
        "snapshot_count": count,
        "snapshot_count_bucket": _dependency_count_bucket(count),
        "candidate_path_count_bucket": _dependency_count_bucket(count),
        "raw_candidate_path_count_bucket": _dependency_count_bucket(count),
        "distinct_candidate_path_count_bucket": _dependency_count_bucket(count),
        "max_paths": None,
        "cap_exceeded": False,
        "cap_trimmed": False,
        "dependency_capture_reason": "complete",
        "present_path_count": max(0, count - missing),
        "missing_path_count": missing,
        "changed_path_count": changed + created,
        "deleted_path_count": deleted,
        "created_path_count": created,
        "invalidation_reason": invalidation_reason if not safe else None,
        "safe_invalidation_evidence": safe,
        "file_dependency_evidence_available": safe,
        "paths_included": False,
    }


def _cache_file_dependency_invalidation_reason(audit: dict[str, Any]) -> str | None:
    reason = audit.get("invalidation_reason")
    if reason in {"dependency-changed", "dependency-deleted"}:
        return str(reason)
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _configure_sqlite_connection(conn: sqlite3.Connection, path: str) -> None:
    busy_timeout_ms = max(0, _env_int("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS", 5000))
    conn.execute(f"pragma busy_timeout = {busy_timeout_ms}")
    if _env_bool("AGENTFLOW_SQLITE_WAL", True) and path not in {"", ":memory:"}:
        try:
            conn.execute("pragma journal_mode = WAL")
            conn.execute("pragma synchronous = NORMAL")
        except sqlite3.OperationalError:
            # Some special filesystems or read-only copies cannot switch modes.
            # Keep the busy timeout so dashboard reads still wait instead of
            # failing immediately under transient writer locks.
            pass


class CompatRow(dict[str, Any]):
    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class PostgresResult:
    def __init__(self, rows: list[CompatRow]):
        self._rows = rows

    def fetchone(self) -> CompatRow | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[CompatRow]:
        return self._rows


class SQLiteConnection:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> PostgresResult:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows: list[CompatRow] = []
            if cur.description:
                names = [col[0] for col in cur.description]
                rows = [CompatRow(zip(names, row)) for row in cur.fetchall()]
            else:
                self._conn.commit()
            return PostgresResult(rows)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class PostgresConnection:
    def __init__(self, pool: Any):
        self.pool = pool

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> PostgresResult:
        sql = self._translate_sql(sql)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows: list[CompatRow] = []
                if cur.description:
                    names = [col.name for col in cur.description]
                    rows = [CompatRow(zip(names, row)) for row in cur.fetchall()]
                else:
                    conn.commit()
                return PostgresResult(rows)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        self.pool.close()

    @staticmethod
    def _json_text_expr(column: str, key: str) -> str:
        return f"jsonb_extract_path_text({column}::jsonb, '{key}')"

    @classmethod
    def _translate_sql(cls, sql: str) -> str:
        sql = sql.replace("datetime(created_at) >= datetime('now', ?)", "created_at >= (now() + %s::interval)")
        sql = sql.replace("date('now', '-6 days')", "(CURRENT_DATE - interval '6 days')")
        sql = sql.replace("DATE('now')", "CURRENT_DATE")
        sql = sql.replace("date('now')", "CURRENT_DATE")
        sql = sql.replace("DATE(created_at)", "created_at::date")
        sql = sql.replace("date(created_at)", "created_at::date")

        def json_repl(match: re.Match[str]) -> str:
            return cls._json_text_expr(match.group("column"), match.group("key"))

        sql = re.sub(
            r"json_extract\((?P<column>[a-zA-Z_][a-zA-Z0-9_]*), '\$\.(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)'\)",
            json_repl,
            sql,
        )

        for column, key in (
            ("crunch_json", "saved_chars"),
            ("crunch_json", "tokens_saved_est"),
            ("crunch_json", "crunch_ratio"),
            ("routing_json", "text_chars"),
        ):
            expr = cls._json_text_expr(column, key)
            sql = sql.replace(f"sum({expr})", f"sum(({expr})::numeric)")
            sql = sql.replace(f"avg({expr})", f"avg(({expr})::numeric)")
            sql = sql.replace(f"coalesce({expr}, 0)", f"coalesce(({expr})::numeric, 0)")

        changed_expr = cls._json_text_expr("crunch_json", "changed")
        sql = sql.replace(f"{changed_expr} = 1", f"{changed_expr} in ('1', 'true')")

        text_chars_expr = cls._json_text_expr("routing_json", "text_chars")
        sql = sql.replace(
            "CAST(coalesce(\n"
            f"                   {text_chars_expr},\n"
            "                   coalesce(actual_input_tokens, input_tokens_est, 0) * 4,\n"
            "                   0\n"
            "               ) AS INTEGER)",
            "CAST(coalesce(\n"
            f"                   ({text_chars_expr})::numeric,\n"
            "                   coalesce(actual_input_tokens, input_tokens_est, 0) * 4,\n"
            "                   0\n"
            "               ) AS INTEGER)",
        )

        return sql.replace("?", "%s")


class SQLiteStore:
    backend = "sqlite"

    def __init__(self, path: str):
        self.path = path
        self.database_url = f"sqlite:///{path}"
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        timeout_s = max(0.0, _env_int("AGENTFLOW_SQLITE_BUSY_TIMEOUT_MS", 5000) / 1000.0)
        self._raw_conn = sqlite3.connect(path, check_same_thread=False, timeout=timeout_s)
        _configure_sqlite_connection(self._raw_conn, path)
        self._raw_conn.row_factory = sqlite3.Row
        self.conn: sqlite3.Connection | SQLiteConnection = self._raw_conn
        self._init()
        self.conn = SQLiteConnection(self._raw_conn, self._lock)

    def _init(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            create table if not exists cache (
              cache_key text primary key,
              created_at text not null,
              model text not null,
              response_json text not null,
              request_chars integer,
              response_chars integer
            )
            """)
            cur.execute("""
            create table if not exists semantic_cache (
              cache_key text primary key,
              created_at text not null,
              model text not null,
              embedding_json text not null,
              response_json text not null,
              request_chars integer
            )
            """)
            cur.execute("""
            create table if not exists cache_file_deps (
              cache_key text not null,
              path text not null,
              exists_flag integer not null,
              mtime_ns integer,
              size integer,
              primary key(cache_key, path)
            )
            """)
            cur.execute("""
            create table if not exists calls (
              id text primary key,
              created_at text not null,
              path text not null,
              requested_model text,
              routed_model text,
              stream integer,
              cache_hit integer,
              status_code integer,
              latency_ms integer,
              input_tokens_est integer,
              output_tokens_est integer,
              cost_est_usd real,
              crunch_json text,
              routing_json text,
              error text,
              request_json text,
              response_json text
            )
            """)
            self._ensure_column("calls", "actual_input_tokens", "integer")
            self._ensure_column("calls", "actual_output_tokens", "integer")
            self._ensure_column("calls", "session_id", "text")
            self._ensure_column("calls", "cost_baseline_usd", "real")
            self._ensure_column("calls", "category", "text")
            self._ensure_column("calls", "cache_creation_input_tokens", "integer")
            self._ensure_column("calls", "cache_read_input_tokens", "integer")
            self._ensure_column("calls", "retry_count", "integer")
            self._ensure_column("calls", "cache_json", "text")
            self._ensure_column("calls", "thinking_output_tokens", "integer")
            self._ensure_column("calls", "provider", "text")
            self._ensure_column("calls", "source_surface", "text")
            self._ensure_column("calls", "endpoint", "text")
            self._ensure_column("calls", "requested_model_family", "text")
            self._ensure_column("calls", "routed_model_family", "text")
            cur.execute("""
            create table if not exists routing_experiments (
              id text primary key,
              call_id text not null,
              created_at text not null,
              provider text,
              source_surface text,
              stream integer,
              requested_model text not null,
              routed_model text not null,
              primary_model text not null,
              shadow_model text not null,
              category text,
              routing_reason text,
              input_tokens_est integer,
              primary_status_code integer,
              shadow_status_code integer,
              primary_latency_ms integer,
              shadow_latency_ms integer,
              primary_output_chars integer,
              shadow_output_chars integer,
              primary_output_sha256 text,
              shadow_output_sha256 text,
              output_similarity real,
              passed_threshold integer,
              primary_cost_est_usd real,
              shadow_cost_est_usd real,
              budget_limit_usd real,
              budget_spent_before_usd real,
              budget_remaining_before_usd real,
              budget_spent_after_usd real,
              error text,
              routing_json text,
              experiment_json text,
              primary_response_json text,
              shadow_response_json text
            )
            """)
            self._ensure_column("routing_experiments", "provider", "text")
            self._ensure_column("routing_experiments", "source_surface", "text")
            self._ensure_column("routing_experiments", "stream", "integer")
            self._ensure_column("routing_experiments", "budget_limit_usd", "real")
            self._ensure_column("routing_experiments", "budget_spent_before_usd", "real")
            self._ensure_column("routing_experiments", "budget_remaining_before_usd", "real")
            self._ensure_column("routing_experiments", "budget_spent_after_usd", "real")
            cur.execute("""
            create table if not exists codex_app_events (
              id text primary key,
              created_at text not null,
              direction text not null,
              method text,
              request_id text,
              thread_id text,
              message_chars integer,
              params_chars integer,
              input_items integer,
              input_text_chars integer,
              result_chars integer,
              error_code integer,
              error_message text,
              latency_ms integer,
              session_id text
            )
            """)
            self._ensure_column("codex_app_events", "routing_json", "text")
            self._ensure_column("codex_app_events", "crunch_json", "text")
            self._ensure_column("codex_app_events", "cache_json", "text")
            self._ensure_column("codex_app_events", "event_window_json", "text")
            self._ensure_column("codex_app_events", "metadata_json", "text")
            cur.execute("""
            create table if not exists managed_outcome_feedback_queue (
              id text primary key,
              created_at text not null,
              updated_at text not null,
              source_surface text not null,
              endpoint text not null,
              optimization_unit_id integer not null,
              payload_json text not null,
              status text not null,
              attempts integer not null default 0,
              next_attempt_at text not null,
              last_error text,
              last_status_code integer,
              sent_at text
            )
            """)
            cur.execute("""
            create table if not exists optimization_eval_results (
              id text primary key,
              run_id text not null,
              created_at text not null,
              candidate_id text not null,
              source_surface text,
              optimization_family text,
              action_family text,
              status_class text not null,
              reason_codes_json text,
              score_json text,
              cost_json text,
              result_json text not null
            )
            """)
            cur.execute("""
            create index if not exists idx_codex_app_events_start_recent
            on codex_app_events(direction, method, created_at)
            """)
            cur.execute("""
            create index if not exists idx_codex_app_events_response_lookup
            on codex_app_events(direction, request_id, created_at)
            """)
            cur.execute("""
            create index if not exists idx_calls_created_at
            on calls(created_at)
            """)
            cur.execute("""
            create index if not exists idx_codex_app_events_created_at
            on codex_app_events(created_at)
            """)
            cur.execute("""
            create index if not exists idx_managed_outcome_feedback_due
            on managed_outcome_feedback_queue(status, next_attempt_at, created_at)
            """)
            cur.execute("""
            create index if not exists idx_optimization_eval_results_recent
            on optimization_eval_results(created_at, status_class)
            """)
            self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"pragma table_info({table})").fetchall()
        }
        if column not in existing:
            self.conn.execute(f"alter table {table} add column {column} {definition}")

    def get_cache_with_reason(self, key: str) -> tuple[Optional[dict[str, Any]], str | None]:
        with self._lock:
            audit = self.cache_file_dependency_audit(key)
            invalidation_reason = _cache_file_dependency_invalidation_reason(audit)
            if invalidation_reason:
                self.delete_cache(key)
                return None, invalidation_reason
            row = self.conn.execute("select response_json from cache where cache_key = ?", (key,)).fetchone()
            if not row:
                return None, None
            return json.loads(row["response_json"]), None

    def get_cache(self, key: str) -> Optional[dict[str, Any]]:
        response, _reason = self.get_cache_with_reason(key)
        return response

    def set_cache(
        self,
        key: str,
        model: str,
        request_chars: int,
        response: dict[str, Any],
        file_deps: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            response_json = stable_json(response)
            self.conn.execute(
                "insert or replace into cache(cache_key, created_at, model, response_json, request_chars, response_chars) values (?, ?, ?, ?, ?, ?)",
                (key, utc_now(), model, response_json, request_chars, len(response_json)),
            )
            self.conn.execute("delete from cache_file_deps where cache_key = ?", (key,))
            for dep in file_deps or []:
                path = dep.get("path")
                if not path:
                    continue
                self.conn.execute(
                    "insert or replace into cache_file_deps(cache_key, path, exists_flag, mtime_ns, size) values (?, ?, ?, ?, ?)",
                    (
                        key,
                        str(path),
                        1 if dep.get("exists") else 0,
                        dep.get("mtime_ns"),
                        dep.get("size"),
                    ),
                )
            self.conn.commit()

    def delete_cache(self, key: str) -> None:
        with self._lock:
            self.conn.execute("delete from cache where cache_key = ?", (key,))
            self.conn.execute("delete from cache_file_deps where cache_key = ?", (key,))
            self.conn.commit()

    def cache_file_dependency_audit(self, key: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "select path, exists_flag, mtime_ns, size from cache_file_deps where cache_key = ?",
            (key,),
        ).fetchall()
        return _cache_file_dependency_audit_from_rows(rows)

    def _cache_file_deps_changed(self, key: str) -> bool:
        audit = self.cache_file_dependency_audit(key)
        return _cache_file_dependency_invalidation_reason(audit) is not None

    def get_semantic_cache(self, embedding: list[float], model: str, threshold: float) -> Optional[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "select embedding_json, response_json from semantic_cache where model = ? limit 500",
                (model,),
            ).fetchall()
            if not rows:
                return None
            best_sim = -1.0
            best_resp: Optional[dict[str, Any]] = None
            for row in rows:
                stored = json.loads(row["embedding_json"])
                sim = cosine_similarity(embedding, stored)
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_resp = json.loads(row["response_json"])
            return best_resp

    def set_semantic_cache(self, key: str, model: str, embedding: list[float], response: dict[str, Any], request_chars: int) -> None:
        with self._lock:
            self.conn.execute(
                "insert or replace into semantic_cache(cache_key, created_at, model, embedding_json, response_json, request_chars) values (?, ?, ?, ?, ?, ?)",
                (key, utc_now(), model, json.dumps(embedding), stable_json(response), request_chars),
            )
            self.conn.commit()

    def log_call(self, **kwargs: Any) -> None:
        cols = [
            "id", "created_at", "path", "requested_model", "routed_model", "stream", "cache_hit", "status_code",
            "latency_ms", "input_tokens_est", "output_tokens_est", "actual_input_tokens", "actual_output_tokens",
            "cost_est_usd", "cost_baseline_usd", "crunch_json", "routing_json", "cache_json", "error", "request_json", "response_json", "session_id",
            "category", "cache_creation_input_tokens", "cache_read_input_tokens", "retry_count", "thinking_output_tokens", "provider",
            "source_surface", "endpoint", "requested_model_family", "routed_model_family",
        ]
        values = [kwargs.get(c, "anthropic") if c == "provider" else kwargs.get(c) for c in cols]
        with self._lock:
            self.conn.execute(
                f"insert into calls({','.join(cols)}) values ({','.join(['?']*len(cols))})",
                values,
            )
            self.conn.commit()

    def update_call_routing_json(self, call_id: str, routing_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "update calls set routing_json = ? where id = ?",
                (routing_json, call_id),
            )
            self.conn.commit()

    def update_call_cache_json(self, call_id: str, cache_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "update calls set cache_json = ? where id = ?",
                (cache_json, call_id),
            )
            self.conn.commit()

    def update_routing_experiment_json(self, experiment_id: str, experiment_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "update routing_experiments set experiment_json = ? where id = ?",
                (experiment_json, experiment_id),
            )
            self.conn.commit()

    def update_codex_app_event_routing_json(self, event_id: str, routing_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "update codex_app_events set routing_json = ? where id = ?",
                (routing_json, event_id),
            )
            self.conn.commit()

    def update_codex_app_event_window_json(self, event_id: str, event_window_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "update codex_app_events set event_window_json = ? where id = ?",
                (event_window_json, event_id),
            )
            self.conn.commit()

    def log_codex_app_event(self, **kwargs: Any) -> None:
        cols = [
            "id", "created_at", "direction", "method", "request_id", "thread_id",
            "message_chars", "params_chars", "input_items", "input_text_chars",
            "result_chars", "error_code", "error_message", "latency_ms", "session_id",
            "routing_json", "crunch_json", "cache_json", "event_window_json", "metadata_json",
        ]
        values = [kwargs.get(c) for c in cols]
        with self._lock:
            self.conn.execute(
                f"insert into codex_app_events({','.join(cols)}) values ({','.join(['?']*len(cols))})",
                values,
            )
            self.conn.commit()

    def enqueue_managed_outcome_feedback(self, **kwargs: Any) -> None:
        cols = [
            "id", "created_at", "updated_at", "source_surface", "endpoint",
            "optimization_unit_id", "payload_json", "status", "attempts",
            "next_attempt_at", "last_error", "last_status_code", "sent_at",
        ]
        now = utc_now()
        values = [
            kwargs.get("id"),
            kwargs.get("created_at") or now,
            kwargs.get("updated_at") or now,
            kwargs.get("source_surface"),
            kwargs.get("endpoint"),
            kwargs.get("optimization_unit_id"),
            kwargs.get("payload_json"),
            kwargs.get("status") or "queued",
            kwargs.get("attempts") or 0,
            kwargs.get("next_attempt_at") or now,
            kwargs.get("last_error"),
            kwargs.get("last_status_code"),
            kwargs.get("sent_at"),
        ]
        with self._lock:
            self.conn.execute(
                f"insert into managed_outcome_feedback_queue({','.join(cols)}) values ({','.join(['?']*len(cols))})",
                values,
            )
            self.conn.commit()

    def get_managed_outcome_feedback(self, queue_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                   payload_json, status, attempts, next_attempt_at, last_error,
                   last_status_code, sent_at
            from managed_outcome_feedback_queue
            where id = ?
            """,
            (queue_id,),
        ).fetchone()
        return dict(row) if row else None

    def claim_managed_outcome_feedback(self, queue_id: str, *, now: str | None = None) -> dict[str, Any] | None:
        now = now or utc_now()
        with self._lock:
            row = self.conn.execute(
                """
                select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                       payload_json, status, attempts, next_attempt_at, last_error,
                       last_status_code, sent_at
                from managed_outcome_feedback_queue
                where id = ?
                  and status in ('queued', 'retryable-error')
                  and next_attempt_at <= ?
                """,
                (queue_id, now),
            ).fetchone()
            if not row:
                return None
            attempts = int(row["attempts"] or 0) + 1
            self.conn.execute(
                """
                update managed_outcome_feedback_queue
                set status = ?, attempts = ?, updated_at = ?
                where id = ?
                """,
                ("sending", attempts, now, queue_id),
            )
            self.conn.commit()
            claimed = dict(row)
            claimed["attempts"] = attempts
            claimed["status"] = "sending"
            return claimed

    def claim_due_managed_outcome_feedback(
        self,
        *,
        limit: int,
        now: str | None = None,
        source_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        now = now or utc_now()
        capped = max(1, min(int(limit or 1), 100))
        source_clause = "and source_surface = ?" if source_surface else ""
        params: tuple[Any, ...]
        if source_surface:
            params = (now, source_surface, capped)
        else:
            params = (now, capped)
        with self._lock:
            rows = self.conn.execute(
                f"""
                select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                       payload_json, status, attempts, next_attempt_at, last_error,
                       last_status_code, sent_at
                from managed_outcome_feedback_queue
                where status in ('queued', 'retryable-error')
                  and next_attempt_at <= ?
                  {source_clause}
                order by created_at asc
                limit ?
                """,
                params,
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                attempts = int(row["attempts"] or 0) + 1
                self.conn.execute(
                    """
                    update managed_outcome_feedback_queue
                    set status = ?, attempts = ?, updated_at = ?
                    where id = ?
                    """,
                    ("sending", attempts, now, row["id"]),
                )
                item = dict(row)
                item["attempts"] = attempts
                item["status"] = "sending"
                claimed.append(item)
            self.conn.commit()
            return claimed

    def due_managed_outcome_feedback(
        self,
        *,
        limit: int,
        now: str | None = None,
        source_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        now = now or utc_now()
        capped = max(1, min(int(limit or 1), 1000))
        source_clause = "and source_surface = ?" if source_surface else ""
        params: tuple[Any, ...]
        if source_surface:
            params = (now, source_surface, capped)
        else:
            params = (now, capped)
        rows = self.conn.execute(
            f"""
            select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                   status, attempts, next_attempt_at, last_error, last_status_code, sent_at
            from managed_outcome_feedback_queue
            where status in ('queued', 'retryable-error')
              and next_attempt_at <= ?
              {source_clause}
            order by created_at asc
            limit ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_managed_outcome_feedback_sent(self, queue_id: str, *, status_code: int | None = None) -> None:
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                update managed_outcome_feedback_queue
                set status = ?, updated_at = ?, sent_at = ?, last_error = null, last_status_code = ?
                where id = ?
                """,
                ("sent", now, now, status_code, queue_id),
            )
            self.conn.commit()

    def mark_managed_outcome_feedback_retry(
        self,
        queue_id: str,
        *,
        status: str,
        error: str | None,
        status_code: int | None,
        next_attempt_at: str | None,
    ) -> None:
        now = utc_now()
        with self._lock:
            self.conn.execute(
                """
                update managed_outcome_feedback_queue
                set status = ?, updated_at = ?, last_error = ?, last_status_code = ?, next_attempt_at = ?
                where id = ?
                """,
                (status, now, error, status_code, next_attempt_at or now, queue_id),
            )
            self.conn.commit()

    def managed_outcome_feedback_summary(self, *, source_surface: str | None = None) -> list[dict[str, Any]]:
        if source_surface:
            rows = self.conn.execute(
                """
                select status, count(*) as count
                from managed_outcome_feedback_queue
                where source_surface = ?
                group by status
                order by count desc, status asc
                """,
                (source_surface,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select status, count(*) as count
                from managed_outcome_feedback_queue
                group by status
                order by count desc, status asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def managed_outcome_feedback_rows(
        self,
        *,
        source_surface: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit or 1), 10000))
        if source_surface:
            rows = self.conn.execute(
                """
                select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                       status, attempts, next_attempt_at, last_error, last_status_code, sent_at
                from managed_outcome_feedback_queue
                where source_surface = ?
                order by created_at asc
                limit ?
                """,
                (source_surface, capped),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select id, created_at, updated_at, source_surface, endpoint, optimization_unit_id,
                       status, attempts, next_attempt_at, last_error, last_status_code, sent_at
                from managed_outcome_feedback_queue
                order by created_at asc
                limit ?
                """,
                (capped,),
            ).fetchall()
        return [dict(row) for row in rows]

    def managed_outcome_feedback_payload_rows(
        self,
        *,
        source_surface: str | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit or 1), 10000))
        if source_surface:
            rows = self.conn.execute(
                """
                select id, source_surface, endpoint, status, payload_json
                from managed_outcome_feedback_queue
                where source_surface = ?
                order by created_at asc
                limit ?
                """,
                (source_surface, capped),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                select id, source_surface, endpoint, status, payload_json
                from managed_outcome_feedback_queue
                order by created_at asc
                limit ?
                """,
                (capped,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_routing_experiment(self, **kwargs: Any) -> None:
        cols = [
            "id", "call_id", "created_at", "provider", "source_surface", "stream", "requested_model", "routed_model",
            "primary_model", "shadow_model", "category", "routing_reason",
            "input_tokens_est", "primary_status_code", "shadow_status_code",
            "primary_latency_ms", "shadow_latency_ms", "primary_output_chars",
            "shadow_output_chars", "primary_output_sha256", "shadow_output_sha256",
            "output_similarity", "passed_threshold", "primary_cost_est_usd",
            "shadow_cost_est_usd", "budget_limit_usd", "budget_spent_before_usd",
            "budget_remaining_before_usd", "budget_spent_after_usd",
            "error", "routing_json", "experiment_json",
            "primary_response_json", "shadow_response_json",
        ]
        values = [kwargs.get(c) for c in cols]
        with self._lock:
            self.conn.execute(
                f"insert into routing_experiments({','.join(cols)}) values ({','.join(['?']*len(cols))})",
                values,
            )
            self.conn.commit()

    def log_optimization_eval_result(self, **kwargs: Any) -> None:
        cols = [
            "id", "run_id", "created_at", "candidate_id", "source_surface",
            "optimization_family", "action_family", "status_class",
            "reason_codes_json", "score_json", "cost_json", "result_json",
        ]
        values = [kwargs.get(c) for c in cols]
        with self._lock:
            self.conn.execute(
                f"insert into optimization_eval_results({','.join(cols)}) values ({','.join(['?']*len(cols))})",
                values,
            )
            self.conn.commit()


class PostgresStore(SQLiteStore):
    backend = "postgres"

    def __init__(self, database_url: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "AGENTFLOW_DATABASE_URL requires the psycopg pool extra. "
                "Install with: python -m pip install 'psycopg[binary,pool]>=3.2'"
            ) from exc
        self.path = database_url
        self.database_url = database_url
        min_size = int(os.getenv("AGENTFLOW_POSTGRES_POOL_MIN", "1"))
        max_size = int(os.getenv("AGENTFLOW_POSTGRES_POOL_MAX", "10"))
        self.pool = ConnectionPool(conninfo=database_url, min_size=min_size, max_size=max_size, open=True)
        self.conn = PostgresConnection(self.pool)
        self._lock = threading.RLock()
        self._init()

    def _init(self) -> None:
        for sql in (
            """
            create table if not exists cache (
              cache_key text primary key,
              created_at timestamptz not null,
              model text not null,
              response_json text not null,
              request_chars integer,
              response_chars integer
            )
            """,
            """
            create table if not exists semantic_cache (
              cache_key text primary key,
              created_at timestamptz not null,
              model text not null,
              embedding_json text not null,
              response_json text not null,
              request_chars integer
            )
            """,
            """
            create table if not exists cache_file_deps (
              cache_key text not null,
              path text not null,
              exists_flag integer not null,
              mtime_ns bigint,
              size bigint,
              primary key(cache_key, path)
            )
            """,
            """
            create table if not exists calls (
              id text primary key,
              created_at timestamptz not null,
              path text not null,
              requested_model text,
              routed_model text,
              stream integer,
              cache_hit integer,
              status_code integer,
              latency_ms integer,
              input_tokens_est integer,
              output_tokens_est integer,
              actual_input_tokens integer,
              actual_output_tokens integer,
              cost_est_usd numeric,
              cost_baseline_usd numeric,
              crunch_json text,
              routing_json text,
              cache_json text,
              error text,
              request_json text,
              response_json text,
              session_id text,
              category text,
              cache_creation_input_tokens integer,
              cache_read_input_tokens integer,
              retry_count integer,
              thinking_output_tokens integer,
              provider text,
              source_surface text,
              endpoint text,
              requested_model_family text,
              routed_model_family text
            )
            """,
            """
            create table if not exists routing_experiments (
              id text primary key,
              call_id text not null,
              created_at timestamptz not null,
              provider text,
              source_surface text,
              stream integer,
              requested_model text not null,
              routed_model text not null,
              primary_model text not null,
              shadow_model text not null,
              category text,
              routing_reason text,
              input_tokens_est integer,
              primary_status_code integer,
              shadow_status_code integer,
              primary_latency_ms integer,
              shadow_latency_ms integer,
              primary_output_chars integer,
              shadow_output_chars integer,
              primary_output_sha256 text,
              shadow_output_sha256 text,
              output_similarity numeric,
              passed_threshold integer,
              primary_cost_est_usd numeric,
              shadow_cost_est_usd numeric,
              budget_limit_usd numeric,
              budget_spent_before_usd numeric,
              budget_remaining_before_usd numeric,
              budget_spent_after_usd numeric,
              error text,
              routing_json text,
              experiment_json text,
              primary_response_json text,
              shadow_response_json text
            )
            """,
            """
            create table if not exists codex_app_events (
              id text primary key,
              created_at timestamptz not null,
              direction text not null,
              method text,
              request_id text,
              thread_id text,
              message_chars integer,
              params_chars integer,
              input_items integer,
              input_text_chars integer,
              result_chars integer,
              error_code integer,
              error_message text,
              latency_ms integer,
              session_id text,
              routing_json text,
              crunch_json text,
              cache_json text,
              event_window_json text,
              metadata_json text
            )
            """,
            """
            create table if not exists managed_outcome_feedback_queue (
              id text primary key,
              created_at timestamptz not null,
              updated_at timestamptz not null,
              source_surface text not null,
              endpoint text not null,
              optimization_unit_id integer not null,
              payload_json text not null,
              status text not null,
              attempts integer not null default 0,
              next_attempt_at timestamptz not null,
              last_error text,
              last_status_code integer,
              sent_at timestamptz
            )
            """,
            """
            create table if not exists optimization_eval_results (
              id text primary key,
              run_id text not null,
              created_at timestamptz not null,
              candidate_id text not null,
              source_surface text,
              optimization_family text,
              action_family text,
              status_class text not null,
              reason_codes_json text,
              score_json text,
              cost_json text,
              result_json text not null
            )
            """,
        ):
            self.conn.execute(sql)
        for column in ("routing_json", "crunch_json", "cache_json", "event_window_json", "metadata_json"):
            self.conn.execute(f"alter table codex_app_events add column if not exists {column} text")
        for column in ("source_surface", "endpoint", "requested_model_family", "routed_model_family"):
            self.conn.execute(f"alter table calls add column if not exists {column} text")
        for column, definition in (
            ("provider", "text"),
            ("source_surface", "text"),
            ("stream", "integer"),
            ("budget_limit_usd", "numeric"),
            ("budget_spent_before_usd", "numeric"),
            ("budget_remaining_before_usd", "numeric"),
            ("budget_spent_after_usd", "numeric"),
        ):
            self.conn.execute(f"alter table routing_experiments add column if not exists {column} {definition}")
        self.conn.execute("""
            create index if not exists idx_codex_app_events_start_recent
            on codex_app_events(direction, method, created_at)
        """)
        self.conn.execute("""
            create index if not exists idx_codex_app_events_response_lookup
            on codex_app_events(direction, request_id, created_at)
        """)
        self.conn.execute("""
            create index if not exists idx_calls_created_at
            on calls(created_at)
        """)
        self.conn.execute("""
            create index if not exists idx_codex_app_events_created_at
            on codex_app_events(created_at)
        """)
        self.conn.execute("""
            create index if not exists idx_managed_outcome_feedback_due
            on managed_outcome_feedback_queue(status, next_attempt_at, created_at)
        """)
        self.conn.execute("""
            create index if not exists idx_optimization_eval_results_recent
            on optimization_eval_results(created_at, status_class)
        """)

    def set_cache(
        self,
        key: str,
        model: str,
        request_chars: int,
        response: dict[str, Any],
        file_deps: list[dict[str, Any]] | None = None,
    ) -> None:
        response_json = stable_json(response)
        self.conn.execute(
            """
            insert into cache(cache_key, created_at, model, response_json, request_chars, response_chars)
            values (?, ?, ?, ?, ?, ?)
            on conflict (cache_key) do update set
              created_at = excluded.created_at,
              model = excluded.model,
              response_json = excluded.response_json,
              request_chars = excluded.request_chars,
              response_chars = excluded.response_chars
            """,
            (key, utc_now(), model, response_json, request_chars, len(response_json)),
        )
        self.conn.execute("delete from cache_file_deps where cache_key = ?", (key,))
        for dep in file_deps or []:
            path = dep.get("path")
            if not path:
                continue
            self.conn.execute(
                """
                insert into cache_file_deps(cache_key, path, exists_flag, mtime_ns, size)
                values (?, ?, ?, ?, ?)
                on conflict (cache_key, path) do update set
                  exists_flag = excluded.exists_flag,
                  mtime_ns = excluded.mtime_ns,
                  size = excluded.size
                """,
                (
                    key,
                    str(path),
                    1 if dep.get("exists") else 0,
                    dep.get("mtime_ns"),
                    dep.get("size"),
                ),
            )

    def cache_file_dependency_audit(self, key: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "select path, exists_flag, mtime_ns, size from cache_file_deps where cache_key = ?",
            (key,),
        ).fetchall()
        return _cache_file_dependency_audit_from_rows(rows)

    def get_cache_with_reason(self, key: str) -> tuple[Optional[dict[str, Any]], str | None]:
        audit = self.cache_file_dependency_audit(key)
        invalidation_reason = _cache_file_dependency_invalidation_reason(audit)
        if invalidation_reason:
            self.delete_cache(key)
            return None, invalidation_reason
        row = self.conn.execute("select response_json from cache where cache_key = ?", (key,)).fetchone()
        if not row:
            return None, None
        return json.loads(row["response_json"]), None

    def get_cache(self, key: str) -> Optional[dict[str, Any]]:
        response, _reason = self.get_cache_with_reason(key)
        return response

    def delete_cache(self, key: str) -> None:
        self.conn.execute("delete from cache where cache_key = ?", (key,))
        self.conn.execute("delete from cache_file_deps where cache_key = ?", (key,))

    def set_semantic_cache(self, key: str, model: str, embedding: list[float], response: dict[str, Any], request_chars: int) -> None:
        self.conn.execute(
            """
            insert into semantic_cache(cache_key, created_at, model, embedding_json, response_json, request_chars)
            values (?, ?, ?, ?, ?, ?)
            on conflict (cache_key) do update set
              created_at = excluded.created_at,
              model = excluded.model,
              embedding_json = excluded.embedding_json,
              response_json = excluded.response_json,
              request_chars = excluded.request_chars
            """,
            (key, utc_now(), model, json.dumps(embedding), stable_json(response), request_chars),
        )


def Store(path: str | None = None) -> SQLiteStore | PostgresStore:
    database_url = os.getenv("AGENTFLOW_DATABASE_URL", "").strip()
    if database_url:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("AGENTFLOW_DATABASE_URL must start with postgresql:// or postgres://")
        return PostgresStore(database_url)
    if path is None:
        path = str(Path.home() / ".agentflow" / "agentflow.sqlite3")
    return SQLiteStore(path)
