from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any

from tokenclaw.db_adoption import adopt_legacy_sqlite_evidence, detect_legacy_evidence_gap
from tokenclaw.paths import default_db_path
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.request_shape_rollups import (
    DEFAULT_CACHE_REPLAY_CANARY_MAX_EVIDENCE_AGE_HOURS,
    DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS,
    build_request_shape_cache_replay_evidence_report,
)
from tokenclaw.store import utc_now


SAVINGS_LOOP_BOTTLENECKS_SCHEMA = "tokenclaw.savings_loop_bottlenecks.v1"
DEFAULT_ACTIVE_WINDOW_HOURS = 24.0
DEFAULT_ACTIVATION_MIN_SOURCE_ROWS = 10


def _parse_utc(value: Any) -> datetime | None:
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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_privacy() -> dict[str, bool]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "raw_prompts_included": False,
        "provider_bodies_included": False,
        "raw_request_bodies_included": False,
        "raw_responses_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "absolute_paths_included": False,
        "tool_payloads_included": False,
        "managed_server_calls_made": False,
        "provider_calls_made": False,
        "policy_files_written": False,
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _safe_scalar(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> Any:
    try:
        row = conn.execute(sql, params).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def _source_traffic_row(
    conn: Any,
    *,
    since: str,
    activation_min_source_rows: int,
) -> dict[str, Any]:
    rows = _as_int(_safe_scalar(conn, "select count(*) from calls where created_at >= ?", (since,)))
    latest = _safe_scalar(conn, "select max(created_at) from calls")
    below_threshold = rows < activation_min_source_rows
    return {
        "kind": "source-traffic",
        "status": "blocked" if below_threshold else "clear",
        "blocker_code": "source-traffic-below-activation-threshold" if below_threshold else None,
        "why": (
            f"canonical DB has {rows} provider call rows in the active window; "
            f"activation threshold is {activation_min_source_rows}"
        ),
        "operator_action": "Generate local provider traffic or inspect activation health before ranking new cohorts."
        if below_threshold
        else "Source traffic is present for savings-loop analysis.",
        "command": "tokenclaw stats --json" if below_threshold else "none",
        "metrics": {
            "active_window_rows": rows,
            "activation_min_source_rows": activation_min_source_rows,
            "below_activation_threshold": below_threshold,
            "latest_call_at": latest,
        },
        "privacy": _metadata_privacy(),
    }


def _legacy_gap_row(*, canonical_db: str | Path | None, legacy_db: str | Path | None) -> dict[str, Any]:
    if canonical_db is None or str(canonical_db).startswith(("postgresql://", "postgres://")):
        return {
            "kind": "stranded-legacy-db",
            "status": "unavailable",
            "blocker_code": None,
            "why": "legacy SQLite adoption check is available only for local SQLite stores",
            "operator_action": "No legacy SQLite adoption action is available for this store backend.",
            "command": "none",
            "metrics": {"stranded_legacy_rows": 0, "richer_legacy_detected": False},
            "privacy": _metadata_privacy(),
        }
    detection = detect_legacy_evidence_gap(canonical_db=canonical_db, legacy_db=legacy_db)
    canonical = detection.get("canonical_db") if isinstance(detection.get("canonical_db"), dict) else {}
    legacy = detection.get("legacy_db") if isinstance(detection.get("legacy_db"), dict) else {}
    canonical_calls = _as_int(canonical.get("calls"))
    legacy_calls = _as_int(legacy.get("calls"))
    stranded_calls = max(0, legacy_calls - canonical_calls)
    blocked = bool(detection.get("richer_legacy_detected"))
    return {
        "kind": "stranded-legacy-db",
        "status": "blocked" if blocked else "clear",
        "blocker_code": "stranded-legacy-agentflow-sqlite-evidence" if blocked else None,
        "why": f"legacy SQLite has {stranded_calls} more call rows than the canonical DB"
        if blocked
        else "no richer legacy SQLite evidence was detected",
        "operator_action": "Adopt legacy SQLite evidence into the canonical TokenClaw DB."
        if blocked
        else "No legacy evidence adoption is needed.",
        "command": "tokenclaw db adopt-legacy" if blocked else "none",
        "metrics": {
            "canonical_calls": canonical_calls,
            "legacy_calls": legacy_calls,
            "stranded_legacy_rows": stranded_calls,
            "canonical_request_shape_rollups": _as_int(canonical.get("request_shape_rollups")),
            "legacy_request_shape_rollups": _as_int(legacy.get("request_shape_rollups")),
            "canonical_request_shape_rollup_snapshots": _as_int(canonical.get("request_shape_rollup_snapshots")),
            "legacy_request_shape_rollup_snapshots": _as_int(legacy.get("request_shape_rollup_snapshots")),
            "richer_legacy_detected": blocked,
            "reason_codes": [public_label(code, "unknown") for code in detection.get("reason_codes") or []],
        },
        "privacy": _metadata_privacy(),
    }


def _legacy_adoption_preflight(
    *,
    canonical_db: str | Path | None,
    legacy_db: str | Path | None,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "schema": "tokenclaw.legacy_sqlite_evidence_preflight.v1",
            "status": "not-run",
            "enabled": False,
            "ok": True,
            "rows_inserted": 0,
            "rows_skipped": 0,
            "gap_cleared": False,
            "privacy": _metadata_privacy(),
        }
    if canonical_db is None or str(canonical_db).startswith(("postgresql://", "postgres://")):
        return {
            "schema": "tokenclaw.legacy_sqlite_evidence_preflight.v1",
            "status": "unavailable",
            "enabled": True,
            "ok": False,
            "reason": "legacy SQLite adoption preflight is available only for local SQLite stores",
            "rows_inserted": 0,
            "rows_skipped": 0,
            "gap_cleared": False,
            "privacy": _metadata_privacy(),
        }

    before = detect_legacy_evidence_gap(canonical_db=canonical_db, legacy_db=legacy_db)
    if not before.get("richer_legacy_detected"):
        return {
            "schema": "tokenclaw.legacy_sqlite_evidence_preflight.v1",
            "status": "clear-before-adoption",
            "enabled": True,
            "ok": True,
            "rows_inserted": 0,
            "rows_skipped": 0,
            "gap_cleared": True,
            "preflight_reason_codes": [public_label(code, "unknown") for code in before.get("reason_codes") or []],
            "privacy": _metadata_privacy(),
        }

    saved_warning = os.environ.get("TOKENCLAW_LEGACY_DB_WARNING")
    os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = "0"
    try:
        result = adopt_legacy_sqlite_evidence(canonical_db=canonical_db, legacy_db=legacy_db, dry_run=False)
    except Exception as exc:
        return {
            "schema": "tokenclaw.legacy_sqlite_evidence_preflight.v1",
            "status": "failed",
            "enabled": True,
            "ok": False,
            "error": type(exc).__name__,
            "rows_inserted": 0,
            "rows_skipped": 0,
            "gap_cleared": False,
            "preflight_reason_codes": [public_label(code, "unknown") for code in before.get("reason_codes") or []],
            "privacy": _metadata_privacy(),
        }
    finally:
        if saved_warning is None:
            os.environ.pop("TOKENCLAW_LEGACY_DB_WARNING", None)
        else:
            os.environ["TOKENCLAW_LEGACY_DB_WARNING"] = saved_warning

    after = detect_legacy_evidence_gap(canonical_db=canonical_db, legacy_db=legacy_db)
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    table_results = []
    for item in result.get("table_results") or []:
        if not isinstance(item, dict):
            continue
        table_results.append({
            "table": public_label(item.get("table"), "unknown"),
            "status": public_label(item.get("status"), "unknown"),
            "rows_examined": _as_int(item.get("rows_examined")),
            "rows_inserted": _as_int(item.get("rows_inserted")),
            "rows_skipped": _as_int(item.get("rows_skipped")),
            "column_count": len(item.get("columns_copied") or []),
        })
    gap_cleared = not bool(after.get("richer_legacy_detected"))
    return {
        "schema": "tokenclaw.legacy_sqlite_evidence_preflight.v1",
        "status": "adopted-gap-cleared" if gap_cleared else "adopted-gap-remains",
        "enabled": True,
        "ok": bool(result.get("ok")) and gap_cleared,
        "adoption_status": public_label(result.get("status"), "unknown"),
        "legacy_open_mode": public_label(result.get("legacy_open_mode"), "unknown"),
        "rows_examined": _as_int(summary.get("rows_examined")),
        "rows_inserted": _as_int(summary.get("rows_inserted")),
        "rows_skipped": _as_int(summary.get("rows_skipped")),
        "table_results": table_results,
        "gap_cleared": gap_cleared,
        "post_adoption_detection_status": public_label(after.get("status"), "unknown"),
        "preflight_reason_codes": [public_label(code, "unknown") for code in before.get("reason_codes") or []],
        "post_adoption_reason_codes": [public_label(code, "unknown") for code in after.get("reason_codes") or []],
        "privacy": _metadata_privacy(),
    }


def _rollup_freshness_row(
    conn: Any,
    *,
    now: datetime,
    max_age_hours: float,
) -> dict[str, Any]:
    rollup_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollups"))
    snapshot_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollup_snapshots"))
    newest_rollup_at = _safe_scalar(conn, "select max(generated_at) from request_shape_rollups")
    newest_snapshot_at = _safe_scalar(conn, "select max(generated_at) from request_shape_rollup_snapshots")
    newest_candidates = [_parse_utc(newest_rollup_at), _parse_utc(newest_snapshot_at)]
    newest = max((candidate for candidate in newest_candidates if candidate is not None), default=None)
    age_hours = round((now - newest).total_seconds() / 3600.0, 3) if newest else None
    stale = bool(age_hours is None or (max_age_hours > 0 and age_hours > max_age_hours))
    if rollup_count <= 0 and snapshot_count <= 0:
        blocker = "no-request-shape-rollups"
        why = "canonical DB has no request-shape rollups or snapshots"
        action = "Emit request-shape rollups from local metadata."
        command = "tokenclaw request-shape-rollups --dry-run"
    elif stale:
        blocker = "request-shape-rollups-stale"
        why = f"newest request-shape rollup/snapshot is {age_hours}h old; max age is {max_age_hours}h"
        action = "Refresh request-shape rollups before ranking new activation cohorts."
        command = "tokenclaw request-shape-rollups --dry-run"
    else:
        blocker = None
        why = "request-shape rollup evidence is present and fresh"
        action = "No rollup refresh action is needed."
        command = "none"
    return {
        "kind": "rollup-freshness",
        "status": "blocked" if blocker else "clear",
        "blocker_code": blocker,
        "why": why,
        "operator_action": action,
        "command": command,
        "metrics": {
            "request_shape_rollup_count": rollup_count,
            "request_shape_rollup_snapshot_count": snapshot_count,
            "newest_rollup_generated_at": newest_rollup_at,
            "newest_snapshot_generated_at": newest_snapshot_at,
            "newest_evidence_age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "stale": stale,
        },
        "privacy": _metadata_privacy(),
    }


def _crunch_dry_run_row(conn: Any) -> dict[str, Any]:
    rollup_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollups"))
    snapshot_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollup_snapshots"))
    rows_considered = rollup_count if rollup_count > 0 else snapshot_count
    blocked = rows_considered <= 0
    return {
        "kind": "crunch-dry-run",
        "status": "blocked" if blocked else "clear",
        "blocker_code": "zero-row-crunch-dry-run" if blocked else None,
        "why": "crunch dry-run has zero rehydrated request-shape rows"
        if blocked
        else f"crunch dry-run can rehydrate {rows_considered} persisted request-shape evidence rows",
        "operator_action": "Refresh request-shape rollups and rerun repeated-context crunch dry-run ranking."
        if blocked
        else "Crunch dry-run source rows are present.",
        "command": "tokenclaw request-shape-rollups --dry-run" if blocked else "tokenclaw request-shape-crunch-canary-stage",
        "metrics": {
            "local_action_family": "crunch",
            "target_local_rule_file": "crunch_rules.yaml",
            "rows_considered": rows_considered,
            "request_shape_rollup_count": rollup_count,
            "request_shape_rollup_snapshot_count": snapshot_count,
            "zero_row_dry_run": blocked,
        },
        "privacy": _metadata_privacy(),
    }


def _activation_queue_policy_rows(activation_burndown: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(activation_burndown, dict):
        return []
    rows: list[dict[str, Any]] = []
    for entry in activation_burndown.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        rule_file = public_label(entry.get("target_local_rule_file"), "unknown")
        if rule_file not in {"routing_rules.yaml", "crunch_rules.yaml", "cache_rules.yaml", "cache_canary_policy.yaml"}:
            continue
        blocker_codes = [public_label(code, "unknown") for code in entry.get("blocker_codes") or []]
        freshness_state = public_label(entry.get("freshness_state"), "unknown")
        blocking_reason = public_label(entry.get("blocking_reason") or entry.get("unblock_reason"), "unknown")
        signal_text = " ".join([freshness_state, blocking_reason, *blocker_codes])
        if not any(token in signal_text for token in ("stale", "zero-traffic", "evidence-older-than-max-age", "rollback")):
            continue
        rows.append({
            "kind": "stale-policy-rule",
            "status": "blocked",
            "blocker_code": blocking_reason if blocking_reason != "unknown" else "stale-local-policy-rule",
            "why": f"{rule_file} has a stale or rollback-required activation successor",
            "operator_action": "Review the local activation successor and apply the recommended rollback or refresh action.",
            "command": public_label(entry.get("next_action"), "inspect-local-activation-next-action-queue"),
            "metrics": {
                "local_action_family": public_label(entry.get("local_action_family") or entry.get("lever"), "unknown"),
                "target_local_rule_file": rule_file,
                "sample_count": _as_int(entry.get("sample_count")),
                "freshness_state": freshness_state,
                "projected_savings_usd": round(_as_float(entry.get("projected_savings_usd")), 6),
                "blocker_codes": blocker_codes,
            },
            "privacy": _metadata_privacy(),
        })
    return rows


def _cache_policy_rows(
    store_obj: Any,
    *,
    config_dir: str | Path | None,
    policy_scan_limit: int,
    max_age_hours: float,
) -> list[dict[str, Any]]:
    config = Path(config_dir or os.getenv("TOKENCLAW_CONFIG_DIR") or os.getenv("TOKENCLAW_POLICY_CONFIG_DIR") or Path.home() / ".tokenclaw")
    rules_path = config.expanduser() / "cache_canary_policy.yaml"
    try:
        evidence = build_request_shape_cache_replay_evidence_report(
            store_obj,
            rules_path=rules_path,
            limit=policy_scan_limit,
            max_age_hours=max_age_hours,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    stale_rules = evidence.get("stale_zero_traffic_rules") if isinstance(evidence.get("stale_zero_traffic_rules"), list) else []
    for rule in stale_rules:
        if not isinstance(rule, dict):
            continue
        rows.append({
            "kind": "stale-policy-rule",
            "status": "blocked",
            "blocker_code": "stale-no-canary-traffic",
            "why": "staged cache replay rule has no observed canary traffic before the freshness deadline",
            "operator_action": "Roll back the stale cache replay rule or refresh evidence before promotion.",
            "command": "tokenclaw request-shape-cache-replay-policy-decision --apply",
            "metrics": {
                "local_action_family": "cache",
                "target_local_rule_file": "cache_canary_policy.yaml",
                "rule_ref": public_id(rule.get("rule_id"), prefix="rule") if rule.get("rule_id") else None,
                "age_hours": rule.get("age_hours"),
                "max_age_hours": rule.get("max_age_hours"),
                "reason": public_label(rule.get("reason"), "stale-no-canary-traffic"),
            },
            "privacy": _metadata_privacy(),
        })
    return rows


def _status_order(status: str) -> int:
    return {"blocked": 0, "unavailable": 1, "clear": 2}.get(status, 3)


def build_savings_loop_bottlenecks_report(
    store_obj: Any,
    *,
    db_path: str | Path | None = None,
    legacy_db: str | Path | None = None,
    config_dir: str | Path | None = None,
    activation_burndown: dict[str, Any] | None = None,
    active_window_hours: float = DEFAULT_ACTIVE_WINDOW_HOURS,
    activation_min_source_rows: int = DEFAULT_ACTIVATION_MIN_SOURCE_ROWS,
    rollup_max_age_hours: float = DEFAULT_ROLLUP_SNAPSHOT_MAX_AGE_HOURS,
    policy_max_age_hours: float = DEFAULT_CACHE_REPLAY_CANARY_MAX_EVIDENCE_AGE_HOURS,
    policy_scan_limit: int = 1000,
    adopt_legacy_preflight: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a compact operator report for conditions that stall savings activation."""

    now_dt = (now or _parse_utc(utc_now()) or datetime.now(timezone.utc)).astimezone(timezone.utc)
    active_hours = max(0.1, float(active_window_hours or DEFAULT_ACTIVE_WINDOW_HOURS))
    since = (now_dt - timedelta(hours=active_hours)).isoformat()
    conn = store_obj.conn
    canonical_db = db_path or getattr(store_obj, "path", None) or default_db_path()
    adoption_preflight = _legacy_adoption_preflight(
        canonical_db=canonical_db,
        legacy_db=legacy_db,
        enabled=bool(adopt_legacy_preflight),
    )
    rows = [
        _source_traffic_row(
            conn,
            since=since,
            activation_min_source_rows=max(0, int(activation_min_source_rows)),
        ),
        _legacy_gap_row(canonical_db=canonical_db, legacy_db=legacy_db),
        _rollup_freshness_row(conn, now=now_dt, max_age_hours=max(0.0, float(rollup_max_age_hours))),
        _crunch_dry_run_row(conn),
    ]
    policy_rows = _cache_policy_rows(
        store_obj,
        config_dir=config_dir,
        policy_scan_limit=max(1, min(int(policy_scan_limit or 1), 10000)),
        max_age_hours=max(0.0, float(policy_max_age_hours)),
    )
    policy_rows.extend(_activation_queue_policy_rows(activation_burndown))
    if policy_rows:
        rows.extend(policy_rows)
    else:
        rows.append({
            "kind": "stale-policy-rule",
            "status": "clear",
            "blocker_code": None,
            "why": "no stale zero-traffic staged policy rules were detected in bounded local metadata",
            "operator_action": "No stale policy rollback action is needed.",
            "command": "none",
            "metrics": {
                "stale_zero_traffic_rule_count": 0,
                "checked_rule_files": ["cache_canary_policy.yaml", "cache_rules.yaml", "crunch_rules.yaml", "routing_rules.yaml"],
                "policy_scan_limit": max(1, min(int(policy_scan_limit or 1), 10000)),
            },
            "privacy": _metadata_privacy(),
        })

    rows.sort(key=lambda row: (_status_order(str(row.get("status") or "")), str(row.get("kind") or "")))
    blocked = [row for row in rows if row.get("status") == "blocked"]
    top = blocked[0] if blocked else rows[0] if rows else {}
    status = "stalled" if blocked else "alive"
    source_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "source-traffic"), {}) or {}
    legacy_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "stranded-legacy-db"), {}) or {}
    rollup_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "rollup-freshness"), {}) or {}
    crunch_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "crunch-dry-run"), {}) or {}
    stale_policy_count = sum(1 for row in rows if row.get("kind") == "stale-policy-rule" and row.get("status") == "blocked")
    return {
        "schema": SAVINGS_LOOP_BOTTLENECKS_SCHEMA,
        "generated_at": now_dt.isoformat(),
        "status": status,
        "read_only": not bool(adopt_legacy_preflight),
        "summary": {
            "blocked_count": len(blocked),
            "row_count": len(rows),
            "top_kind": top.get("kind"),
            "top_status": top.get("status"),
            "top_blocker_code": top.get("blocker_code"),
            "top_next_action": top.get("operator_action"),
            "top_command": top.get("command"),
            "active_window_hours": active_hours,
            "source_traffic_rows": _as_int(source_metrics.get("active_window_rows")),
            "below_activation_threshold": bool(source_metrics.get("below_activation_threshold")),
            "stranded_legacy_rows": _as_int(legacy_metrics.get("stranded_legacy_rows")),
            "request_shape_rollup_count": _as_int(rollup_metrics.get("request_shape_rollup_count")),
            "request_shape_rollup_snapshot_count": _as_int(rollup_metrics.get("request_shape_rollup_snapshot_count")),
            "newest_rollup_age_hours": rollup_metrics.get("newest_evidence_age_hours"),
            "crunch_dry_run_rows_considered": _as_int(crunch_metrics.get("rows_considered")),
            "zero_row_crunch_dry_run": bool(crunch_metrics.get("zero_row_dry_run")),
            "stale_policy_rule_count": stale_policy_count,
            "legacy_adoption_preflight_status": adoption_preflight.get("status"),
            "legacy_adoption_rows_inserted": _as_int(adoption_preflight.get("rows_inserted")),
            "legacy_adoption_gap_cleared": bool(adoption_preflight.get("gap_cleared")),
        },
        "rows": rows,
        "legacy_adoption_preflight": adoption_preflight,
        "privacy": _metadata_privacy(),
    }
