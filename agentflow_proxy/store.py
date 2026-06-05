from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
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
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"pragma table_info({table})").fetchall()
        }
        if column not in existing:
            self.conn.execute(f"alter table {table} add column {column} {definition}")

    def get_cache(self, key: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("select response_json from cache where cache_key = ?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row["response_json"])

    def set_cache(self, key: str, model: str, request_chars: int, response: dict[str, Any]) -> None:
        response_json = stable_json(response)
        self.conn.execute(
            "insert or replace into cache(cache_key, created_at, model, response_json, request_chars, response_chars) values (?, ?, ?, ?, ?, ?)",
            (key, utc_now(), model, response_json, request_chars, len(response_json)),
        )
        self.conn.commit()

    def get_semantic_cache(self, embedding: list[float], model: str, threshold: float) -> Optional[dict[str, Any]]:
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
            "category", "cache_creation_input_tokens", "cache_read_input_tokens", "retry_count", "thinking_output_tokens",
        ]
        values = [kwargs.get(c) for c in cols]
        self.conn.execute(
            f"insert into calls({','.join(cols)}) values ({','.join(['?']*len(cols))})",
            values,
        )
        self.conn.commit()
