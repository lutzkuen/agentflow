from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Sequence

from agentflow_proxy.optimization.cli_support import (
    default_db_path,
    open_store_for_db,
    redact_secret,
    redact_url,
    write_json,
)


MANAGED_POLICY_API_KEY_ENV = "AGENTFLOW_MANAGED_API_KEY"


def parse_utc_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        value = str(raw)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def seconds_since(raw: Any, now: datetime) -> int | None:
    parsed = parse_utc_iso(raw)
    if parsed is None:
        return None
    return max(0, int((now - parsed).total_seconds()))


def breakdown_from_counts(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def public_feedback_row(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    optimization_unit_id = row.get("optimization_unit_id")
    if optimization_unit_id in (0, "0"):
        optimization_unit_id = None
    return {
        "queue_id": row.get("id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "source_surface": row.get("source_surface"),
        "endpoint": row.get("endpoint"),
        "optimization_unit_id": optimization_unit_id,
        "status": row.get("status"),
        "attempts": row.get("attempts") or 0,
        "next_attempt_at": row.get("next_attempt_at"),
        "last_status_code": row.get("last_status_code"),
        "sent_at": row.get("sent_at"),
        "age_seconds": seconds_since(row.get("created_at"), now),
        "payload_included": False,
    }


def managed_feedback_config() -> dict[str, Any]:
    from agentflow_proxy import recommendations

    return {
        "enabled": recommendations.recommendations_enabled(),
        "server_url": redact_url(recommendations.recommendation_server_url()),
        "server_configured": recommendations.recommendation_server_configured(),
        "timeout_seconds": recommendations.recommendation_timeout_seconds(),
        "failure_mode": recommendations.recommendation_failure_mode(),
        "queue_max_attempts": recommendations.outcome_feedback_queue_max_attempts(),
        "queue_retry_delay_seconds": recommendations.outcome_feedback_queue_retry_delay_seconds(),
        "auth_configured": recommendations.managed_auth_configured(),
        "api_key_value_included": False,
    }


def managed_feedback_status_result(
    store: Any,
    *,
    source_surface: str | None,
    sample_limit: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    generated_at = now.isoformat()
    status_counts = {
        str(row.get("status") or "unknown"): int(row.get("count") or 0)
        for row in store.managed_outcome_feedback_summary(source_surface=source_surface)
    } if hasattr(store, "managed_outcome_feedback_summary") else {}
    rows = (
        store.managed_outcome_feedback_rows(source_surface=source_surface, limit=10000)
        if hasattr(store, "managed_outcome_feedback_rows")
        else []
    )
    due_rows = (
        store.due_managed_outcome_feedback(limit=max(1, sample_limit), source_surface=source_surface)
        if hasattr(store, "due_managed_outcome_feedback")
        else []
    )
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_surface") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    pending_rows = [
        row
        for row in rows
        if row.get("status") in {"queued", "retryable-error"}
    ]
    oldest_pending = min(
        pending_rows,
        key=lambda row: parse_utc_iso(row.get("created_at")) or now,
        default=None,
    )
    dropped = status_counts.get("dropped-after-limit", 0)
    summary = {
        "total": sum(status_counts.values()),
        "queued": status_counts.get("queued", 0),
        "retryable_error": status_counts.get("retryable-error", 0),
        "sending": status_counts.get("sending", 0),
        "sent": status_counts.get("sent", 0),
        "dropped_after_limit": dropped,
        "error": status_counts.get("error", 0),
        "due": len(due_rows),
        "oldest_pending_age_seconds": seconds_since(oldest_pending.get("created_at"), now) if oldest_pending else None,
        "retry_limit_drops": dropped,
    }
    return {
        "schema": "agentflow.managed_feedback_status.v1",
        "ok": True,
        "generated_at": generated_at,
        "source_surface": source_surface,
        "managed_feedback": managed_feedback_config(),
        "summary": summary,
        "status_breakdown": breakdown_from_counts(status_counts),
        "source_surface_breakdown": breakdown_from_counts(source_counts),
        "oldest_pending": public_feedback_row(oldest_pending, now=now) if oldest_pending else None,
        "due_samples": [
            public_feedback_row(row, now=now)
            for row in due_rows[:max(0, sample_limit)]
        ],
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }


def safe_managed_feedback_flush_result(result: dict[str, Any]) -> dict[str, Any]:
    secret = os.getenv(MANAGED_POLICY_API_KEY_ENV)
    safe = redact_secret(result, secret)
    if isinstance(safe.get("managed_feedback"), dict):
        safe["managed_feedback"]["server_url"] = redact_url(safe["managed_feedback"].get("server_url"))
    for item in safe.get("results", []) if isinstance(safe.get("results"), list) else []:
        if isinstance(item, dict) and "server_url" in item:
            item["server_url"] = redact_url(item.get("server_url"))
    return safe


def managed_feedback_status_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Report local managed outcome feedback queue status")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum due queue samples to include, default: 20.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    store = open_store_for_db(str(args.db))
    try:
        result = managed_feedback_status_result(
            store,
            source_surface=args.source_surface,
            sample_limit=max(0, min(args.limit, 100)),
        )
    finally:
        store.conn.close()
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0


def managed_feedback_flush_cli(argv: Sequence[str] | None = None, *, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Flush due local managed outcome feedback queue rows in bounded batches")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help="AgentFlow database URL or SQLite path, default: AGENTFLOW_DB or ~/.agentflow/agentflow.sqlite3",
    )
    parser.add_argument("--source-surface", help="Optional queue source surface filter, for example codex_turn.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum due rows to flush, default: 5, max: 100.")
    parser.add_argument("--dry-run", action="store_true", help="Report due rows without claiming or sending them.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON instead of emitting one compact line.")
    args = parser.parse_args(argv)

    stdout = stdout if stdout is not None else sys.stdout
    limit = max(1, min(args.limit, 100))
    store = open_store_for_db(str(args.db))
    try:
        before = managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
        if args.dry_run:
            results = [
                {**row, "status": "would-send"}
                for row in before.get("due_samples", [])
            ]
            flush_status = "dry-run"
            reason = "dry-run"
        else:
            from agentflow_proxy import recommendations

            if recommendations.recommendations_enabled():
                results = asyncio.run(
                    recommendations.flush_queued_outcome_feedback(
                        store,
                        limit=limit,
                        source_surface=args.source_surface,
                    )
                )
                flush_status = "completed"
                reason = "ok"
            else:
                results = []
                flush_status = "skipped"
                reason = "managed-feedback-disabled"
        after = managed_feedback_status_result(store, source_surface=args.source_surface, sample_limit=limit)
    finally:
        store.conn.close()

    result_counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        result_counts[status] = result_counts.get(status, 0) + 1
    result = {
        "schema": "agentflow.managed_feedback_flush.v1",
        "ok": True,
        "dry_run": bool(args.dry_run),
        "source_surface": args.source_surface,
        "limit": limit,
        "flush": {
            "status": flush_status,
            "reason": reason,
            "attempted": len(results) if not args.dry_run else 0,
            "would_attempt": len(results) if args.dry_run else 0,
            "sent": result_counts.get("sent", 0),
            "retryable_error": result_counts.get("retryable-error", 0),
            "dropped_after_limit": result_counts.get("dropped-after-limit", 0),
        },
        "managed_feedback": managed_feedback_config(),
        "before": before["summary"],
        "after": after["summary"],
        "result_breakdown": breakdown_from_counts(result_counts),
        "results": results,
        "privacy": {
            "metadata_only": True,
            "payload_json_included": False,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "secrets_included": False,
        },
    }
    result = safe_managed_feedback_flush_result(result)
    if args.pretty:
        stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        write_json(stdout, result)
    return 0
