from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenclaw.paths import default_db_path, safe_expanduser


ADOPTABLE_TABLES = ("calls", "request_shape_rollups", "request_shape_rollup_snapshots")
DETECTION_SCHEMA = "tokenclaw.legacy_sqlite_evidence_detection.v1"
ADOPTION_SCHEMA = "tokenclaw.legacy_sqlite_evidence_adoption.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_path(path: str | Path | None) -> Path:
    if path is None:
        return Path(default_db_path())
    return safe_expanduser(str(path).removeprefix("sqlite:///"))


def default_legacy_db_path(canonical_db: str | Path | None = None) -> Path:
    canonical = _sqlite_path(canonical_db)
    return canonical.with_name("agentflow.sqlite3")


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _connect_writable(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if not _table_exists(conn, table):
        return []
    return [str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()]


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(f"select count(*) from {table}").fetchone()
    return int(row[0] or 0) if row else 0


def _latest_created_at(conn: sqlite3.Connection, table: str) -> str | None:
    if not _table_exists(conn, table):
        return None
    columns = set(_table_columns(conn, table))
    if "created_at" not in columns:
        return None
    row = conn.execute(f"select max(created_at) from {table}").fetchone()
    return str(row[0]) if row and row[0] else None


def _latest_generated_at(conn: sqlite3.Connection, table: str) -> str | None:
    if not _table_exists(conn, table):
        return None
    columns = set(_table_columns(conn, table))
    if "generated_at" not in columns:
        return None
    row = conn.execute(f"select max(generated_at) from {table}").fetchone()
    return str(row[0]) if row and row[0] else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _db_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "calls": 0,
            "request_shape_rollups": 0,
            "request_shape_rollup_snapshots": 0,
            "latest_call_at": None,
            "latest_rollup_generated_at": None,
            "size_bytes": 0,
        }
    try:
        with _connect_readonly(path) as conn:
            return {
                "path": str(path),
                "exists": True,
                "calls": _row_count(conn, "calls"),
                "request_shape_rollups": _row_count(conn, "request_shape_rollups"),
                "request_shape_rollup_snapshots": _row_count(conn, "request_shape_rollup_snapshots"),
                "latest_call_at": _latest_created_at(conn, "calls"),
                "latest_rollup_generated_at": _latest_generated_at(conn, "request_shape_rollups"),
                "size_bytes": path.stat().st_size,
            }
    except sqlite3.Error as exc:
        return {
            "path": str(path),
            "exists": True,
            "error": type(exc).__name__,
            "calls": 0,
            "request_shape_rollups": 0,
            "request_shape_rollup_snapshots": 0,
            "latest_call_at": None,
            "latest_rollup_generated_at": None,
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }


def detect_legacy_evidence_gap(
    *,
    canonical_db: str | Path | None = None,
    legacy_db: str | Path | None = None,
) -> dict[str, Any]:
    canonical = _sqlite_path(canonical_db)
    legacy = _sqlite_path(legacy_db) if legacy_db is not None else default_legacy_db_path(canonical)
    canonical_summary = _db_summary(canonical)
    legacy_summary = _db_summary(legacy)

    reasons: list[str] = []
    if legacy_summary.get("exists") and not canonical_summary.get("exists"):
        reasons.append("canonical-db-missing")
    if int(legacy_summary.get("calls") or 0) > int(canonical_summary.get("calls") or 0):
        reasons.append("legacy-has-more-calls")
    if int(legacy_summary.get("request_shape_rollups") or 0) > int(canonical_summary.get("request_shape_rollups") or 0):
        reasons.append("legacy-has-more-request-shape-rollups")
    if int(legacy_summary.get("request_shape_rollup_snapshots") or 0) > int(canonical_summary.get("request_shape_rollup_snapshots") or 0):
        reasons.append("legacy-has-more-request-shape-rollup-snapshots")

    legacy_latest = _parse_dt(legacy_summary.get("latest_call_at"))
    canonical_latest = _parse_dt(canonical_summary.get("latest_call_at"))
    if legacy_latest is not None and (canonical_latest is None or legacy_latest > canonical_latest):
        reasons.append("legacy-has-newer-calls")

    status = "richer-legacy-detected" if reasons else "no-richer-legacy"
    if not legacy_summary.get("exists"):
        status = "legacy-db-missing"
    elif legacy_summary.get("error"):
        status = "legacy-db-unreadable"

    return {
        "schema": DETECTION_SCHEMA,
        "generated_at": _now(),
        "status": status,
        "richer_legacy_detected": status == "richer-legacy-detected",
        "canonical_db": canonical_summary,
        "legacy_db": legacy_summary,
        "reason_codes": reasons,
        "next_action": "run-tokenclaw-db-adopt-legacy" if status == "richer-legacy-detected" else "none",
        "bottleneck_signal": {
            "schema": "tokenclaw.savings_loop_bottleneck.v1",
            "status": "blocked" if status == "richer-legacy-detected" else "clear",
            "blocker_code": "stranded-legacy-agentflow-sqlite-evidence" if status == "richer-legacy-detected" else None,
            "local_action_family": "storage",
            "next_action": "adopt-legacy-sqlite-evidence" if status == "richer-legacy-detected" else "none",
        },
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "provider_bodies_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
        },
    }


def _copy_table(
    *,
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    dry_run: bool,
) -> dict[str, Any]:
    source_columns = _table_columns(source, table)
    target_columns = _table_columns(target, table)
    common_columns = [column for column in target_columns if column in source_columns]
    examined = _row_count(source, table)
    if not source_columns:
        return {
            "table": table,
            "status": "source-table-missing",
            "rows_examined": 0,
            "rows_inserted": 0,
            "rows_skipped": 0,
            "columns_copied": [],
        }
    if not target_columns:
        return {
            "table": table,
            "status": "target-table-missing",
            "rows_examined": examined,
            "rows_inserted": 0,
            "rows_skipped": examined,
            "columns_copied": [],
        }
    if "id" not in common_columns:
        return {
            "table": table,
            "status": "primary-key-column-missing",
            "rows_examined": examined,
            "rows_inserted": 0,
            "rows_skipped": examined,
            "columns_copied": common_columns,
        }

    ids = [str(row[0]) for row in source.execute(f"select id from {table}")]
    existing = set()
    if ids:
        for index in range(0, len(ids), 900):
            chunk = ids[index : index + 900]
            placeholders = ",".join("?" for _ in chunk)
            existing.update(
                str(row[0])
                for row in target.execute(
                    f"select id from {table} where id in ({placeholders})",
                    chunk,
                )
            )
    inserted = examined - len(existing)
    if dry_run:
        return {
            "table": table,
            "status": "dry-run",
            "rows_examined": examined,
            "rows_inserted": inserted,
            "rows_skipped": len(existing),
            "columns_copied": common_columns,
        }

    quoted_columns = ",".join(common_columns)
    placeholders = ",".join("?" for _ in common_columns)
    sql = f"insert or ignore into {table} ({quoted_columns}) values ({placeholders})"
    inserted_actual = 0
    skipped_actual = 0
    cursor = source.execute(f"select {quoted_columns} from {table}")
    for row in cursor:
        before = target.total_changes
        target.execute(sql, tuple(row))
        if target.total_changes > before:
            inserted_actual += 1
        else:
            skipped_actual += 1
    return {
        "table": table,
        "status": "copied",
        "rows_examined": examined,
        "rows_inserted": inserted_actual,
        "rows_skipped": skipped_actual,
        "columns_copied": common_columns,
    }


def adopt_legacy_sqlite_evidence(
    *,
    canonical_db: str | Path | None = None,
    legacy_db: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    canonical = _sqlite_path(canonical_db)
    legacy = _sqlite_path(legacy_db) if legacy_db is not None else default_legacy_db_path(canonical)
    detection = detect_legacy_evidence_gap(canonical_db=canonical, legacy_db=legacy)
    if not legacy.exists():
        return {
            "schema": ADOPTION_SCHEMA,
            "generated_at": _now(),
            "ok": False,
            "dry_run": bool(dry_run),
            "status": "legacy-db-missing",
            "canonical_db": str(canonical),
            "legacy_db": str(legacy),
            "legacy_open_mode": "ro",
            "table_results": [],
            "summary": {"rows_examined": 0, "rows_inserted": 0, "rows_skipped": 0},
            "legacy_detection": detection,
        }

    if not dry_run:
        from tokenclaw.store import Store

        Store(str(canonical)).conn.close()
    table_results: list[dict[str, Any]] = []
    with _connect_readonly(legacy) as source:
        if dry_run and not canonical.exists():
            from tokenclaw.store import Store

            with tempfile.TemporaryDirectory() as tmp:
                target_path = Path(tmp) / "tokenclaw-dry-run.sqlite3"
                Store(str(target_path)).conn.close()
                with _connect_readonly(target_path) as target:
                    for table in ADOPTABLE_TABLES:
                        table_results.append(_copy_table(source=source, target=target, table=table, dry_run=True))
        elif dry_run:
            with _connect_readonly(canonical) as target:
                for table in ADOPTABLE_TABLES:
                    table_results.append(_copy_table(source=source, target=target, table=table, dry_run=True))
        else:
            with _connect_writable(canonical) as target:
                try:
                    target.execute("begin")
                    for table in ADOPTABLE_TABLES:
                        table_results.append(_copy_table(source=source, target=target, table=table, dry_run=False))
                    target.commit()
                except Exception:
                    target.rollback()
                    raise

    summary = {
        "rows_examined": sum(int(item.get("rows_examined") or 0) for item in table_results),
        "rows_inserted": sum(int(item.get("rows_inserted") or 0) for item in table_results),
        "rows_skipped": sum(int(item.get("rows_skipped") or 0) for item in table_results),
    }
    return {
        "schema": ADOPTION_SCHEMA,
        "generated_at": _now(),
        "ok": True,
        "dry_run": bool(dry_run),
        "status": "dry-run" if dry_run else "adopted",
        "canonical_db": str(canonical),
        "legacy_db": str(legacy),
        "legacy_open_mode": "ro",
        "table_results": table_results,
        "summary": summary,
        "legacy_detection": detection,
        "privacy": {
            "metadata_only": True,
            "aggregate_only": True,
            "provider_bodies_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
        },
    }


def maybe_warn_about_legacy_evidence(canonical_db: str | Path, *, stream: Any | None = None) -> dict[str, Any] | None:
    legacy = default_legacy_db_path(canonical_db)
    if not legacy.exists():
        return None
    detection = detect_legacy_evidence_gap(canonical_db=canonical_db)
    if not detection.get("richer_legacy_detected"):
        return None
    target = stream if stream is not None else sys.stderr
    target.write(json.dumps({"tokenclaw_warning": detection}, sort_keys=True) + "\n")
    return detection
