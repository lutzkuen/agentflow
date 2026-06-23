from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
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
    build_request_shape_rollups_report,
)
from tokenclaw.store import stable_json, utc_now


SAVINGS_LOOP_BOTTLENECKS_SCHEMA = "tokenclaw.savings_loop_bottlenecks.v1"
ROLLUP_REFRESH_PREFLIGHT_SCHEMA = "tokenclaw.request_shape_rollup_refresh_preflight.v1"
CAPTURED_AVAILABLE_SCHEMA = "tokenclaw.savings_loop_captured_vs_available.v1"
OUTCOME_RECONCILIATION_SCHEMA = "tokenclaw.savings_loop_outcome_reconciliation.v1"
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


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _row_exists(conn: Any, table: str, row_id: str) -> bool:
    try:
        row = conn.execute(f"select 1 from {table} where id = ? limit 1", (row_id,)).fetchone()
    except Exception:
        return False
    return row is not None


def _new_cohort_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "cost_est_usd": 0.0,
        "baseline_cost_usd": 0.0,
        "error_count": 0,
        "retry_rows": 0,
        "latency_ms_total": 0.0,
        "latency_sample_count": 0,
    }


def _add_cohort_row(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["count"] += 1
    bucket["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
    bucket["baseline_cost_usd"] += _as_float(row.get("cost_baseline_usd"))
    bucket["error_count"] += int(_as_int(row.get("status_code")) >= 400)
    bucket["retry_rows"] += int(_as_int(row.get("retry_count")) > 0)
    latency = row.get("latency_ms")
    if latency is not None:
        bucket["latency_ms_total"] += _as_float(latency)
        bucket["latency_sample_count"] += 1


def _finalize_cohort_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(bucket.get("count"))
    return {
        "count": count,
        "cost_est_usd": round(_as_float(bucket.get("cost_est_usd")), 8),
        "baseline_cost_usd": round(_as_float(bucket.get("baseline_cost_usd")), 8),
        "observed_savings_usd": round(max(_as_float(bucket.get("baseline_cost_usd")) - _as_float(bucket.get("cost_est_usd")), 0.0), 8),
        "error_count": _as_int(bucket.get("error_count")),
        "error_rate": round(_as_int(bucket.get("error_count")) / count, 6) if count else 0.0,
        "retry_rows": _as_int(bucket.get("retry_rows")),
        "retry_rate": round(_as_int(bucket.get("retry_rows")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_float(bucket.get("latency_ms_total")) / _as_int(bucket.get("latency_sample_count")), 3)
        if _as_int(bucket.get("latency_sample_count"))
        else None,
    }


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
    refresh_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rollup_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollups"))
    snapshot_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollup_snapshots"))
    newest_rollup_at = _safe_scalar(conn, "select max(generated_at) from request_shape_rollups")
    newest_snapshot_at = _safe_scalar(conn, "select max(generated_at) from request_shape_rollup_snapshots")
    newest_candidates = [_parse_utc(newest_rollup_at), _parse_utc(newest_snapshot_at)]
    newest = max((candidate for candidate in newest_candidates if candidate is not None), default=None)
    refresh = refresh_preflight if isinstance(refresh_preflight, dict) else {}
    refresh_rollup_count = _as_int(refresh.get("rollup_count"))
    refresh_newest_at = refresh.get("newest_evidence_at") if isinstance(refresh.get("newest_evidence_at"), str) else None
    refresh_newest = _parse_utc(refresh_newest_at)
    if refresh_rollup_count > 0 and (newest is None or (refresh_newest is not None and refresh_newest >= newest)):
        newest = refresh_newest or now
    age_hours = round((now - newest).total_seconds() / 3600.0, 3) if newest else None
    stale = bool(age_hours is None or (max_age_hours > 0 and age_hours > max_age_hours))
    refreshed = refresh_rollup_count > 0 and not stale
    if rollup_count <= 0 and snapshot_count <= 0 and not refreshed:
        blocker = "no-request-shape-rollups"
        why = "canonical DB has no request-shape rollups or snapshots"
        action = "Emit request-shape rollups from local metadata."
        command = "tokenclaw request-shape-rollups --dry-run"
    elif stale and not refreshed:
        blocker = "request-shape-rollups-stale"
        why = f"newest request-shape rollup/snapshot is {age_hours}h old; max age is {max_age_hours}h"
        action = "Refresh request-shape rollups before ranking new activation cohorts."
        command = "tokenclaw request-shape-rollups --dry-run"
    else:
        blocker = None
        why = (
            "canonical traffic refresh produced fresh request-shape rollup evidence"
            if refreshed and rollup_count <= 0 and snapshot_count <= 0
            else "request-shape rollup evidence is present and fresh"
        )
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
            "canonical_refresh_status": public_label(refresh.get("status"), "not-run"),
            "canonical_refresh_rows_considered": _as_int(refresh.get("rows_considered")),
            "canonical_refresh_rollup_count": refresh_rollup_count,
            "canonical_refresh_ranked_candidate_count": _as_int(refresh.get("ranked_candidate_count")),
            "canonical_refresh_newest_evidence_at": refresh_newest_at,
            "newest_evidence_age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "stale": stale,
        },
        "privacy": _metadata_privacy(),
    }


def _crunch_dry_run_row(conn: Any, *, refresh_preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    rollup_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollups"))
    snapshot_count = _as_int(_safe_scalar(conn, "select count(*) from request_shape_rollup_snapshots"))
    refresh = refresh_preflight if isinstance(refresh_preflight, dict) else {}
    refresh_rows = _as_int(refresh.get("crunch_dry_run_rows_considered"))
    rows_considered = rollup_count if rollup_count > 0 else snapshot_count if snapshot_count > 0 else refresh_rows
    evidence_source = (
        "persisted-rollups"
        if rollup_count > 0
        else "persisted-snapshots"
        if snapshot_count > 0
        else "canonical-refresh-dry-run"
        if refresh_rows > 0
        else "none"
    )
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
            "canonical_refresh_status": public_label(refresh.get("status"), "not-run"),
            "canonical_refresh_crunch_dry_run_rows_considered": refresh_rows,
            "evidence_source": evidence_source,
            "zero_row_dry_run": blocked,
        },
        "privacy": _metadata_privacy(),
    }


def _request_shape_rollup_refresh_preflight(
    store_obj: Any,
    *,
    source_metrics: dict[str, Any],
    activation_min_source_rows: int,
    limit: int,
) -> dict[str, Any]:
    source_rows = _as_int(source_metrics.get("active_window_rows"))
    if source_rows < max(0, int(activation_min_source_rows)):
        return {
            "schema": ROLLUP_REFRESH_PREFLIGHT_SCHEMA,
            "status": "skipped-below-source-threshold",
            "enabled": False,
            "persisted": False,
            "source_traffic_rows": source_rows,
            "activation_min_source_rows": max(0, int(activation_min_source_rows)),
            "rows_considered": 0,
            "rollup_count": 0,
            "ranked_candidate_count": 0,
            "crunch_dry_run_rows_considered": 0,
            "newest_evidence_at": None,
            "privacy": _metadata_privacy(),
        }
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    try:
        report = build_request_shape_rollups_report(
            store_obj,
            limit=capped_limit,
            persist=False,
            run_id="savings-loop-canonical-refresh-dry-run",
        )
    except Exception as exc:
        return {
            "schema": ROLLUP_REFRESH_PREFLIGHT_SCHEMA,
            "status": "unavailable",
            "enabled": True,
            "persisted": False,
            "source_traffic_rows": source_rows,
            "activation_min_source_rows": max(0, int(activation_min_source_rows)),
            "error_type": public_label(exc.__class__.__name__, "Exception"),
            "rows_considered": 0,
            "rollup_count": 0,
            "ranked_candidate_count": 0,
            "crunch_dry_run_rows_considered": 0,
            "newest_evidence_at": None,
            "privacy": _metadata_privacy(),
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    follow_up = report.get("follow_up_candidates") if isinstance(report.get("follow_up_candidates"), dict) else {}
    follow_up_summary = follow_up.get("summary") if isinstance(follow_up.get("summary"), dict) else {}
    crunch = report.get("crunch_opportunity_dry_run") if isinstance(report.get("crunch_opportunity_dry_run"), dict) else {}
    crunch_summary = crunch.get("summary") if isinstance(crunch.get("summary"), dict) else {}
    window = report.get("window") if isinstance(report.get("window"), dict) else {}
    rollup_count = _as_int(summary.get("rollup_count"))
    rows_considered = _as_int(summary.get("rows_considered"))
    ranked_candidate_count = _as_int(follow_up_summary.get("ranked_candidate_count"))
    crunch_rows = _as_int(crunch_summary.get("rows_considered"))
    status = "refreshed" if rollup_count > 0 else "no-rollups"
    return {
        "schema": ROLLUP_REFRESH_PREFLIGHT_SCHEMA,
        "status": status,
        "enabled": True,
        "persisted": False,
        "source_traffic_rows": source_rows,
        "activation_min_source_rows": max(0, int(activation_min_source_rows)),
        "rows_considered": rows_considered,
        "rollup_count": rollup_count,
        "ranked_candidate_count": ranked_candidate_count,
        "crunch_dry_run_rows_considered": crunch_rows,
        "newest_evidence_at": window.get("end") if isinstance(window.get("end"), str) else report.get("generated_at"),
        "generated_at": report.get("generated_at"),
        "top_next_action": public_label(follow_up_summary.get("top_next_action"), "unknown"),
        "top_local_action_family": public_label(follow_up_summary.get("top_local_action_family"), "unknown"),
        "top_readiness_state": public_label(follow_up_summary.get("top_readiness_state"), "unknown"),
        "crunch_dry_run_status": public_label(crunch.get("status"), "unknown"),
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


def _routing_canary_group(row: dict[str, Any]) -> tuple[tuple[str, str], dict[str, Any], str] | None:
    routing = _json_obj(row.get("routing_json"))
    canary = routing.get("phase_canary") if isinstance(routing.get("phase_canary"), dict) else {}
    if not canary or not canary.get("enabled"):
        return None
    status = str(canary.get("status") or "")
    cohort = str(canary.get("cohort") or "")
    if status not in {"applied", "holdout", "safety_stopped"} and cohort not in {"canary_applied", "canary_holdout", "safety_stopped"}:
        return None
    promotion = canary.get("promotion") if isinstance(canary.get("promotion"), dict) else {}
    policy_id = str(canary.get("policy_id") or canary.get("rule_id") or "local-phase-sonnet-haiku-canary-v1")
    group = {
        "action_family": "routing",
        "policy_section": "routing.phase_canary",
        "policy_id": policy_id,
        "rule_id": canary.get("rule_id") or policy_id,
        "candidate_id": canary.get("candidate_id") or canary.get("target_candidate_id"),
        "action_id": canary.get("promotion_action_id"),
        "rule_source": canary.get("policy_source") or "local-manual",
        "source_surface": canary.get("source_surface") or row.get("source_surface") or "anthropic_messages",
        "source_evidence_schema": promotion.get("source_report_schema") or "tokenclaw.phase_routing_outcome_feedback.v1",
        "projected_savings_usd": _as_float(promotion.get("projected_savings_usd")),
    }
    normalized_cohort = "safety_stopped" if status == "safety_stopped" or cohort == "safety_stopped" else cohort
    if normalized_cohort == "canary_applied" or status == "applied":
        normalized_cohort = "canary_applied"
    elif normalized_cohort == "canary_holdout" or status == "holdout":
        normalized_cohort = "canary_holdout"
    elif normalized_cohort not in {"safety_stopped", "bypassed_or_disabled"}:
        normalized_cohort = "skipped"
    return ("routing", policy_id), group, normalized_cohort


def _cache_canary_group(row: dict[str, Any]) -> tuple[tuple[str, str], dict[str, Any], str] | None:
    cache = _json_obj(row.get("cache_json"))
    canary = cache.get("cache_replay_canary") if isinstance(cache.get("cache_replay_canary"), dict) else {}
    pattern_rule = cache.get("pattern_rule") if isinstance(cache.get("pattern_rule"), dict) else {}
    if not canary and not pattern_rule:
        return None
    canary_status = str(canary.get("status") or "")
    canary_obj = canary.get("canary") if isinstance(canary.get("canary"), dict) else {}
    cohort = str(canary.get("canary_cohort") or canary_obj.get("cohort") or "")
    if cohort not in {"canary_applied", "canary_holdout"} and canary_status not in {"applied", "holdout", "safety_stopped"}:
        return None
    graduation = pattern_rule.get("graduation") if isinstance(pattern_rule.get("graduation"), dict) else {}
    policy_id = str(canary.get("rule_id") or pattern_rule.get("rule_id") or canary.get("candidate_id") or "cache-replay-canary")
    projected = (
        _as_float(canary.get("projected_input_savings_usd"))
        or _as_float(cache.get("estimated_saved_cost_usd"))
        or _as_float(graduation.get("projected_savings_usd"))
    )
    group = {
        "action_family": "cache",
        "policy_section": "cache.replay_canary",
        "policy_id": policy_id,
        "rule_id": canary.get("rule_id") or pattern_rule.get("rule_id") or policy_id,
        "candidate_id": canary.get("candidate_id") or pattern_rule.get("candidate_id"),
        "action_id": None,
        "rule_source": canary.get("policy_source") or pattern_rule.get("policy_source") or "local-manual",
        "source_surface": row.get("source_surface") or "openai_responses",
        "source_evidence_schema": canary.get("schema") or graduation.get("source_schema") or "tokenclaw.cache_replay_canary_decision.v1",
        "projected_savings_usd": projected,
    }
    if cohort == "canary_applied" or canary_status == "applied":
        normalized_cohort = "canary_applied"
    elif cohort == "canary_holdout" or canary_status == "holdout":
        normalized_cohort = "canary_holdout"
    elif canary_status == "safety_stopped":
        normalized_cohort = "safety_stopped"
    else:
        normalized_cohort = "skipped"
    return ("cache", policy_id), group, normalized_cohort


def _reconcile_applied_canary_outcomes(
    store_obj: Any,
    *,
    since: str,
    limit: int,
    generated_at: str,
    persist: bool = True,
) -> dict[str, Any]:
    conn = store_obj.conn
    capped = max(1, min(int(limit or 1), 10000))
    result = {
        "schema": OUTCOME_RECONCILIATION_SCHEMA,
        "status": "no-canary-outcomes",
        "rows_scanned": 0,
        "group_count": 0,
        "promotion_rows_written": 0,
        "optimization_eval_rows_written": 0,
        "persisted": bool(persist),
        "groups": [],
        "privacy": _metadata_privacy(),
    }
    if not (_table_exists(conn, "promotion_outcome_feedback") and _table_exists(conn, "optimization_eval_results")):
        result["status"] = "tables-unavailable"
        return result
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select id, created_at, status_code, latency_ms, retry_count,
                       cost_est_usd, cost_baseline_usd, routing_json, cache_json,
                       source_surface, provider
                from calls
                where created_at >= ?
                  and (routing_json is not null or cache_json is not null)
                order by created_at desc
                limit ?
                """,
                (since, capped),
            ).fetchall()
        ]
    except Exception as exc:
        result.update({"status": "unavailable", "error_type": public_label(exc.__class__.__name__, "Exception")})
        return result
    result["rows_scanned"] = len(rows)

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        for extractor in (_routing_canary_group, _cache_canary_group):
            extracted = extractor(row)
            if extracted is None:
                continue
            key, meta, cohort = extracted
            group = groups.setdefault(key, {
                **meta,
                "cohorts": {
                    "canary_applied": _new_cohort_bucket(),
                    "canary_holdout": _new_cohort_bucket(),
                    "skipped": _new_cohort_bucket(),
                    "bypassed_or_disabled": _new_cohort_bucket(),
                    "safety_stopped": _new_cohort_bucket(),
                },
                "newest_call_at": row.get("created_at"),
            })
            group["projected_savings_usd"] = max(_as_float(group.get("projected_savings_usd")), _as_float(meta.get("projected_savings_usd")))
            if str(row.get("created_at") or "") > str(group.get("newest_call_at") or ""):
                group["newest_call_at"] = row.get("created_at")
            _add_cohort_row(group["cohorts"].setdefault(cohort, _new_cohort_bucket()), row)

    public_groups: list[dict[str, Any]] = []
    promotion_written = 0
    eval_written = 0
    for (_family, policy_id), group in sorted(groups.items()):
        cohorts = {name: _finalize_cohort_bucket(bucket) for name, bucket in group["cohorts"].items()}
        applied = cohorts["canary_applied"]
        holdout = cohorts["canary_holdout"]
        safety = cohorts["safety_stopped"]
        applied_count = _as_int(applied.get("count"))
        holdout_count = _as_int(holdout.get("count"))
        if applied_count + holdout_count + _as_int(safety.get("count")) <= 0:
            continue
        observed = _as_float(applied.get("observed_savings_usd"))
        projected = _as_float(group.get("projected_savings_usd"))
        error_delta = _as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate"))
        retry_delta = _as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate"))
        latency_delta = None
        if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
            latency_delta = round(_as_float(applied.get("latency_avg_ms")) - _as_float(holdout.get("latency_avg_ms")), 3)
        rollback_needed = _as_int(safety.get("count")) > 0 or error_delta > 0.05
        recommendation = "rollback" if rollback_needed else "widen" if observed > 0 and applied_count > 0 else "keep-canary"
        status = "rollback-needed" if rollback_needed else "positive" if observed > 0 and applied_count > 0 else "needs-more-samples"
        reason_codes: list[str] = []
        if rollback_needed:
            reason_codes.append("safety-or-error-regression")
        if applied_count <= 0:
            reason_codes.append("no-applied-cohort")
        if holdout_count <= 0:
            reason_codes.append("no-holdout-cohort")
        entry_id = _stable_id(
            "savings-loop-outcome",
            group.get("action_family"),
            policy_id,
            group.get("newest_call_at"),
            applied_count,
            holdout_count,
            _as_int(safety.get("count")),
            round(observed, 8),
        )
        entry = {
            "schema": "tokenclaw.promotion_outcome_feedback_entry.v1",
            "id": entry_id,
            "created_at": generated_at,
            "impact_generated_at": generated_at,
            "policy_id": policy_id,
            "action_family": group.get("action_family"),
            "policy_section": group.get("policy_section"),
            "rule_source": group.get("rule_source"),
            "rule_id": group.get("rule_id"),
            "candidate_id": group.get("candidate_id"),
            "action_id": group.get("action_id"),
            "source_evidence_schema": group.get("source_evidence_schema"),
            "source_surface": group.get("source_surface"),
            "status": status,
            "recommendation": recommendation,
            "rollback_needed": rollback_needed,
            "reason_codes": reason_codes,
            "observed_savings_usd": round(observed, 8),
            "projected_savings_usd": round(projected, 8),
            "projection_realization_ratio": round(observed / projected, 6) if projected > 0 else None,
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "skipped_count": _as_int(cohorts["skipped"].get("count")),
            "bypassed_count": _as_int(cohorts["bypassed_or_disabled"].get("count")),
            "safety_stop_count": _as_int(safety.get("count")),
            "error_rate_delta": round(error_delta, 6),
            "retry_rate_delta": round(retry_delta, 6),
            "latency_delta_ms": latency_delta,
            "cohort_metrics": cohorts,
            "privacy": _metadata_privacy(),
        }
        if persist and not _row_exists(conn, "promotion_outcome_feedback", entry_id):
            store_obj.log_promotion_outcome_feedback(
                id=entry_id,
                created_at=generated_at,
                impact_generated_at=generated_at,
                policy_id=policy_id,
                action_family=group.get("action_family"),
                policy_section=group.get("policy_section"),
                rule_source=group.get("rule_source"),
                rule_id=group.get("rule_id"),
                candidate_id=group.get("candidate_id"),
                action_id=group.get("action_id"),
                source_evidence_schema=group.get("source_evidence_schema"),
                status=status,
                recommendation=recommendation,
                rollback_needed=1 if rollback_needed else 0,
                observed_savings_usd=round(observed, 8),
                projected_savings_usd=round(projected, 8),
                projection_realization_ratio=entry.get("projection_realization_ratio"),
                applied_count=applied_count,
                holdout_count=holdout_count,
                skipped_count=entry["skipped_count"],
                bypassed_count=entry["bypassed_count"],
                safety_stop_count=entry["safety_stop_count"],
                error_rate_delta=entry["error_rate_delta"],
                retry_rate_delta=entry["retry_rate_delta"],
                latency_delta_ms=latency_delta,
                feedback_json=stable_json(entry),
            )
            promotion_written += 1
        eval_id = _stable_id("savings-loop-eval", entry_id)
        if persist and not _row_exists(conn, "optimization_eval_results", eval_id):
            store_obj.log_optimization_eval_result(
                id=eval_id,
                run_id="savings-loop-outcome-reconciliation",
                created_at=generated_at,
                candidate_id=str(group.get("candidate_id") or policy_id),
                source_surface=group.get("source_surface"),
                optimization_family=f"{group.get('action_family')}_canary",
                action_family=group.get("action_family"),
                status_class=status,
                reason_codes_json=stable_json(reason_codes),
                score_json=stable_json({
                    "observed_savings_usd": round(observed, 8),
                    "projected_savings_usd": round(projected, 8),
                    "projection_realization_ratio": entry.get("projection_realization_ratio"),
                }),
                cost_json=stable_json({
                    "applied": applied,
                    "holdout": holdout,
                    "safety_stopped": safety,
                }),
                result_json=stable_json(entry),
            )
            eval_written += 1
        public_groups.append({
            "action_family": group.get("action_family"),
            "policy_id": public_id(policy_id, prefix="policy"),
            "candidate_id": public_id(group.get("candidate_id"), prefix="candidate") if group.get("candidate_id") else None,
            "status": status,
            "recommendation": recommendation,
            "observed_savings_usd": round(observed, 8),
            "projected_savings_usd": round(projected, 8),
            "applied_count": applied_count,
            "holdout_count": holdout_count,
            "top_blocker_code": reason_codes[0] if reason_codes else None,
        })

    result.update({
        "status": "recorded"
        if promotion_written or eval_written
        else "dry-run"
        if public_groups and not persist
        else "up-to-date"
        if public_groups
        else "no-canary-outcomes",
        "group_count": len(public_groups),
        "promotion_rows_written": promotion_written,
        "optimization_eval_rows_written": eval_written,
        "groups": public_groups,
    })
    return result


def _blocked_savings_from_rollups(conn: Any, *, since: str, limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "request_shape_rollups"):
        return []
    capped = max(1, min(int(limit or 1), 10000))
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                select generated_at, blocker_codes_json, baseline_cost_usd, observed_savings_usd, cost_est_usd, row_count
                from request_shape_rollups
                where generated_at >= ?
                order by generated_at desc
                limit ?
                """,
                (since, capped),
            ).fetchall()
        ]
    except Exception:
        return []
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        codes = []
        try:
            parsed = json.loads(row.get("blocker_codes_json") or "[]")
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            codes = [public_label(code, "unknown") for code in parsed if code]
        if not codes:
            continue
        for code in codes:
            bucket = buckets.setdefault(code, {
                "blocker_code": code,
                "blocked_baseline_usd": 0.0,
                "available_savings_usd": 0.0,
                "row_count": 0,
                "source_rollup_count": 0,
            })
            bucket["blocked_baseline_usd"] += _as_float(row.get("baseline_cost_usd"))
            bucket["available_savings_usd"] += max(
                _as_float(row.get("observed_savings_usd")),
                _as_float(row.get("baseline_cost_usd")) - _as_float(row.get("cost_est_usd")),
                0.0,
            )
            bucket["row_count"] += _as_int(row.get("row_count"), 1)
            bucket["source_rollup_count"] += 1
    result = []
    for bucket in buckets.values():
        result.append({
            **bucket,
            "blocked_baseline_usd": round(_as_float(bucket.get("blocked_baseline_usd")), 8),
            "available_savings_usd": round(_as_float(bucket.get("available_savings_usd")), 8),
        })
    result.sort(key=lambda item: (_as_float(item.get("available_savings_usd")), _as_float(item.get("blocked_baseline_usd"))), reverse=True)
    return result


def _captured_savings_from_feedback(store_obj: Any, *, since: str, limit: int) -> list[dict[str, Any]]:
    if not hasattr(store_obj, "promotion_outcome_feedback_rows"):
        return []
    since_dt = _parse_utc(since)
    buckets: dict[str, dict[str, Any]] = {}
    for row in store_obj.promotion_outcome_feedback_rows(limit=max(1, min(int(limit or 1), 10000))):
        created_at = _parse_utc(row.get("created_at"))
        if since_dt is not None and created_at is not None and created_at < since_dt:
            continue
        family = public_label(row.get("action_family"), "unknown")
        status = public_label(row.get("status"), "unknown")
        key = f"{family}:{status}"
        bucket = buckets.setdefault(key, {
            "action_family": family,
            "status": status,
            "captured_savings_usd": 0.0,
            "projected_savings_usd": 0.0,
            "entry_count": 0,
            "applied_count": 0,
            "holdout_count": 0,
            "top_blocker_code": None,
        })
        bucket["captured_savings_usd"] += _as_float(row.get("observed_savings_usd"))
        bucket["projected_savings_usd"] += _as_float(row.get("projected_savings_usd"))
        bucket["entry_count"] += 1
        bucket["applied_count"] += _as_int(row.get("applied_count"))
        bucket["holdout_count"] += _as_int(row.get("holdout_count"))
        feedback = _json_obj(row.get("feedback_json"))
        reasons = feedback.get("reason_codes") if isinstance(feedback.get("reason_codes"), list) else []
        if reasons and not bucket.get("top_blocker_code"):
            bucket["top_blocker_code"] = public_label(reasons[0], "unknown")
    result = []
    for bucket in buckets.values():
        result.append({
            **bucket,
            "captured_savings_usd": round(_as_float(bucket.get("captured_savings_usd")), 8),
            "projected_savings_usd": round(_as_float(bucket.get("projected_savings_usd")), 8),
        })
    result.sort(key=lambda item: _as_float(item.get("captured_savings_usd")), reverse=True)
    return result


def _captured_vs_available_report(
    store_obj: Any,
    *,
    since: str,
    limit: int,
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    captured = _captured_savings_from_feedback(store_obj, since=since, limit=limit)
    available = _blocked_savings_from_rollups(store_obj.conn, since=since, limit=limit)
    top_captured = captured[0] if captured else {}
    top_available = available[0] if available else {}
    return {
        "schema": CAPTURED_AVAILABLE_SCHEMA,
        "status": "available" if captured or available else "empty",
        "captured_savings_usd": round(sum(_as_float(row.get("captured_savings_usd")) for row in captured), 8),
        "available_blocked_savings_usd": round(sum(_as_float(row.get("available_savings_usd")) for row in available), 8),
        "blocked_baseline_usd": round(sum(_as_float(row.get("blocked_baseline_usd")) for row in available), 8),
        "top_captured_bucket": {
            "action_family": top_captured.get("action_family"),
            "status": top_captured.get("status"),
            "captured_savings_usd": top_captured.get("captured_savings_usd", 0.0),
            "top_blocker_code": top_captured.get("top_blocker_code"),
        } if top_captured else None,
        "top_available_blocker": {
            "blocker_code": top_available.get("blocker_code"),
            "available_savings_usd": top_available.get("available_savings_usd", 0.0),
            "blocked_baseline_usd": top_available.get("blocked_baseline_usd", 0.0),
        } if top_available else None,
        "captured_buckets": captured[:10],
        "available_blocker_buckets": available[:10],
        "outcome_feedback_reconciliation": {
            "status": reconciliation.get("status"),
            "promotion_rows_written": _as_int(reconciliation.get("promotion_rows_written")),
            "optimization_eval_rows_written": _as_int(reconciliation.get("optimization_eval_rows_written")),
            "group_count": _as_int(reconciliation.get("group_count")),
        },
        "privacy": _metadata_privacy(),
    }


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
    persist_outcome_feedback: bool = True,
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
    source_row = _source_traffic_row(
        conn,
        since=since,
        activation_min_source_rows=max(0, int(activation_min_source_rows)),
    )
    source_metrics = source_row.get("metrics") if isinstance(source_row.get("metrics"), dict) else {}
    refresh_preflight = _request_shape_rollup_refresh_preflight(
        store_obj,
        source_metrics=source_metrics,
        activation_min_source_rows=max(0, int(activation_min_source_rows)),
        limit=max(1, min(int(policy_scan_limit or 1), 10000)),
    )
    rows = [
        source_row,
        _legacy_gap_row(canonical_db=canonical_db, legacy_db=legacy_db),
        _rollup_freshness_row(
            conn,
            now=now_dt,
            max_age_hours=max(0.0, float(rollup_max_age_hours)),
            refresh_preflight=refresh_preflight,
        ),
        _crunch_dry_run_row(conn, refresh_preflight=refresh_preflight),
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
    generated_at = now_dt.isoformat()
    outcome_reconciliation = _reconcile_applied_canary_outcomes(
        store_obj,
        since=since,
        limit=max(1, min(int(policy_scan_limit or 1), 10000)),
        generated_at=generated_at,
        persist=bool(persist_outcome_feedback),
    )
    captured_vs_available = _captured_vs_available_report(
        store_obj,
        since=since,
        limit=max(1, min(int(policy_scan_limit or 1), 10000)),
        reconciliation=outcome_reconciliation,
    )
    source_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "source-traffic"), {}) or {}
    legacy_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "stranded-legacy-db"), {}) or {}
    rollup_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "rollup-freshness"), {}) or {}
    crunch_metrics = next((row.get("metrics") for row in rows if row.get("kind") == "crunch-dry-run"), {}) or {}
    stale_policy_count = sum(1 for row in rows if row.get("kind") == "stale-policy-rule" and row.get("status") == "blocked")
    return {
        "schema": SAVINGS_LOOP_BOTTLENECKS_SCHEMA,
        "generated_at": generated_at,
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
            "request_shape_rollup_refresh_status": refresh_preflight.get("status"),
            "request_shape_rollup_refresh_rows_considered": _as_int(refresh_preflight.get("rows_considered")),
            "request_shape_rollup_refresh_rollup_count": _as_int(refresh_preflight.get("rollup_count")),
            "request_shape_rollup_refresh_ranked_candidate_count": _as_int(refresh_preflight.get("ranked_candidate_count")),
            "request_shape_rollup_refresh_crunch_dry_run_rows_considered": _as_int(
                refresh_preflight.get("crunch_dry_run_rows_considered")
            ),
            "captured_savings_usd": _as_float(captured_vs_available.get("captured_savings_usd")),
            "available_blocked_savings_usd": _as_float(captured_vs_available.get("available_blocked_savings_usd")),
            "blocked_baseline_usd": _as_float(captured_vs_available.get("blocked_baseline_usd")),
            "top_available_blocker_code": (
                (captured_vs_available.get("top_available_blocker") or {}).get("blocker_code")
                if isinstance(captured_vs_available.get("top_available_blocker"), dict)
                else None
            ),
            "outcome_reconciliation_status": outcome_reconciliation.get("status"),
            "promotion_outcome_rows_written": _as_int(outcome_reconciliation.get("promotion_rows_written")),
            "optimization_eval_rows_written": _as_int(outcome_reconciliation.get("optimization_eval_rows_written")),
        },
        "rows": rows,
        "captured_vs_available": captured_vs_available,
        "outcome_feedback_reconciliation": outcome_reconciliation,
        "legacy_adoption_preflight": adoption_preflight,
        "request_shape_rollup_refresh": refresh_preflight,
        "privacy": _metadata_privacy(),
    }
