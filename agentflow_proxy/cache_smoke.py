from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentflow_proxy.cache import cache_key_for
from agentflow_proxy.policy_files import utc_now
from agentflow_proxy.public_metadata import public_label


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bucket_count(count: int) -> str:
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


def _char_bucket(count: Any) -> str:
    value = _as_int(count)
    if value <= 0:
        return "0"
    if value < 2_000:
        return "lt_2k_chars"
    if value < 8_000:
        return "2k_8k_chars"
    if value < 32_000:
        return "8k_32k_chars"
    return "32k_plus_chars"


def _breakdown(counter: Counter[str], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _cache_rows(store_obj: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = store_obj.conn.execute(
        """
        select c.cache_key, c.created_at, c.model, c.request_chars, c.response_chars,
               count(d.path) as dependency_count
        from cache c
        left join cache_file_deps d on d.cache_key = c.cache_key
        group by c.cache_key, c.created_at, c.model, c.request_chars, c.response_chars
        order by c.created_at desc
        limit ?
        """,
        (max(1, int(limit or 10)),),
    ).fetchall()
    return [dict(row) for row in rows]


def _count_rows(store_obj: Any, table: str) -> int:
    try:
        row = store_obj.conn.execute(f"select count(*) as c from {table}").fetchone()
    except Exception:
        return 0
    return _as_int(row["c"] if row else 0)


def _model_breakdown(store_obj: Any) -> list[dict[str, Any]]:
    rows = store_obj.conn.execute(
        """
        select model, count(*) as count
        from cache
        group by model
        order by count desc, model asc
        """
    ).fetchall()
    return [{"model": public_label(row["model"], "unknown"), "count": _as_int(row["count"])} for row in rows]


def _cache_row_diagnostic(store_obj: Any, row: dict[str, Any]) -> dict[str, Any]:
    audit = store_obj.cache_file_dependency_audit(str(row["cache_key"]))
    invalidation_reason = audit.get("invalidation_reason")
    return {
        "created_at": row.get("created_at"),
        "model": row.get("model"),
        "request_chars_bucket": _char_bucket(row.get("request_chars")),
        "response_chars_bucket": _char_bucket(row.get("response_chars")),
        "file_dependency_count_bucket": _bucket_count(_as_int(row.get("dependency_count"))),
        "file_dependency_audit": {
            "snapshot_count_bucket": audit.get("snapshot_count_bucket"),
            "changed_path_count": _as_int(audit.get("changed_path_count")),
            "deleted_path_count": _as_int(audit.get("deleted_path_count")),
            "created_path_count": _as_int(audit.get("created_path_count")),
            "missing_path_count": _as_int(audit.get("missing_path_count")),
            "invalidation_reason": invalidation_reason,
            "safe_invalidation_evidence": bool(audit.get("safe_invalidation_evidence")),
            "paths_included": False,
        },
        "invalidated": bool(invalidation_reason in {"dependency-changed", "dependency-deleted"}),
    }


def _call_rows(store_obj: Any, *, scan_limit: int) -> list[dict[str, Any]]:
    rows = store_obj.conn.execute(
        """
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               requested_model, routed_model, stream, cache_hit, cache_json,
               routing_json, request_json, category
        from calls
        order by created_at desc
        limit ?
        """,
        (max(1, int(scan_limit or 5000)),),
    ).fetchall()
    return [dict(row) for row in rows]


def _status_reason(cache_meta: dict[str, Any]) -> tuple[str, str]:
    return str(cache_meta.get("status") or "missing"), str(cache_meta.get("reason") or "missing")


def _exact_lookup_attempted(cache_meta: dict[str, Any]) -> bool:
    status, reason = _status_reason(cache_meta)
    if bool(cache_meta.get("exact_enabled")):
        return status in {"miss", "hit"}
    return reason in {
        "exact-miss",
        "exact-pattern-miss",
        "exact-and-semantic-miss",
        "exact-match",
        "streaming-exact-pattern-miss",
        "streaming-exact-match",
    }


def _call_shape(row: dict[str, Any], cache_meta: dict[str, Any]) -> str:
    routing = _json_obj(row.get("routing_json"))
    text_chars = routing.get("text_chars")
    if text_chars is None:
        text_chars = routing.get("input_text_chars")
    status, reason = _status_reason(cache_meta)
    return json.dumps(
        {
            "provider": public_label(row.get("provider") or "anthropic", "unknown"),
            "path": public_label(row.get("path"), "provider-api-path"),
            "model": public_label(row.get("routed_model") or row.get("requested_model"), "unknown"),
            "stream": _as_int(row.get("stream")),
            "category": public_label(row.get("category") or routing.get("category"), "unknown"),
            "has_tools": routing.get("has_tools"),
            "text_bucket": _char_bucket(text_chars),
            "cache_status": public_label(status, "unknown"),
            "cache_reason": public_label(reason, "unknown"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _key_stability_sample(store_obj: Any, call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cache_keys = {
        str(row["cache_key"])
        for row in store_obj.conn.execute("select cache_key from cache").fetchall()
        if row["cache_key"]
    }
    if not cache_keys:
        return {
            "status": "unavailable",
            "reason": "no-cache-rows",
            "raw_request_bodies_included": False,
            "cache_keys_included": False,
        }

    missing_request_json = 0
    for row in call_rows:
        raw_request = row.get("request_json")
        if not raw_request:
            missing_request_json += 1
            continue
        body = _json_obj(raw_request)
        if not body:
            continue
        provider = str(row.get("provider") or "anthropic")
        computed = cache_key_for(body, str(row.get("path") or ""), provider=provider)
        if computed in cache_keys:
            return {
                "status": "matched",
                "reason": "stored-request-recomputes-existing-cache-key",
                "call_created_at": row.get("created_at"),
                "provider": public_label(provider, "unknown"),
                "path": public_label(row.get("path"), "provider-api-path"),
                "model": public_label(row.get("routed_model") or row.get("requested_model"), "unknown"),
                "raw_request_bodies_included": False,
                "cache_keys_included": False,
            }

    return {
        "status": "unavailable",
        "reason": "cache-table-does-not-store-request-basis-and-recent-calls-lack-matching-log-body",
        "calls_without_request_json": missing_request_json,
        "calls_scanned": len(call_rows),
        "raw_request_bodies_included": False,
        "cache_keys_included": False,
    }


def build_cache_smoke_diagnostic(
    store_obj: Any,
    *,
    limit: int = 10,
    scan_limit: int = 5000,
) -> dict[str, Any]:
    cache_count = _count_rows(store_obj, "cache")
    semantic_count = _count_rows(store_obj, "semantic_cache")
    dependency_count = _count_rows(store_obj, "cache_file_deps")
    newest_rows = _cache_rows(store_obj, limit=limit) if cache_count else []
    newest = [_cache_row_diagnostic(store_obj, row) for row in newest_rows]

    all_invalidations = Counter[str]()
    for row in store_obj.conn.execute("select cache_key from cache").fetchall():
        audit = store_obj.cache_file_dependency_audit(str(row["cache_key"]))
        reason = str(audit.get("invalidation_reason") or "")
        if reason in {"dependency-changed", "dependency-deleted"}:
            all_invalidations[reason] += 1

    call_rows = _call_rows(store_obj, scan_limit=scan_limit)
    status_reason_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    exact_misses = 0
    exact_hits = 0
    semantic_hits = 0
    semantic_misses = 0
    exact_eligible = 0
    invalidated_lookups = 0
    streaming_skip_count = 0
    tools_disabled_skip_count = 0
    file_dependency_blocked_count = 0
    shape_counts: Counter[str] = Counter()
    missing_cache_json = 0

    for row in call_rows:
        cache_meta = _json_obj(row.get("cache_json"))
        if not cache_meta:
            missing_cache_json += 1
            cache_meta = {"status": "missing", "reason": "missing-cache-json"}
        status, reason = _status_reason(cache_meta)
        public_status = public_label(status, "unknown")
        public_reason = public_label(reason, "unknown")
        status_reason_counts[f"{public_status}:{public_reason}"] += 1
        if status == "skipped":
            skip_reason_counts[public_reason] += 1
            if "streaming" in reason or _as_int(row.get("stream")):
                streaming_skip_count += 1
            if "tools-disabled" in reason or "tool-cache-disabled" in reason:
                tools_disabled_skip_count += 1
        if _exact_lookup_attempted(cache_meta):
            exact_eligible += 1
            shape_counts[_call_shape(row, cache_meta)] += 1
        if status == "hit" and str(cache_meta.get("hit_type") or "") == "semantic":
            semantic_hits += 1
        elif status == "hit" or _as_int(row.get("cache_hit")):
            exact_hits += 1
        if status == "miss" and "exact" in reason:
            exact_misses += 1
        if status == "miss" and "semantic" in reason:
            semantic_misses += 1
        file_dependency_blocked = bool("file-dependency" in reason or "dependency-" in reason)
        blocker_reasons = cache_meta.get("cache_replay_blocker_reasons")
        if isinstance(blocker_reasons, list) and any(
            "file-dependency" in str(item) or "dependency-" in str(item)
            for item in blocker_reasons
        ):
            file_dependency_blocked = True
        if cache_meta.get("invalidated") or cache_meta.get("invalidation_reason"):
            invalidated_lookups += 1
            file_dependency_blocked = True
        if file_dependency_blocked:
            file_dependency_blocked_count += 1

    duplicate_shapes = {shape: count for shape, count in shape_counts.items() if count > 1}
    duplicate_candidate_rows = sum(duplicate_shapes.values())

    no_hit_reasons: list[str] = []
    if exact_hits == 0:
        if exact_eligible == 0:
            no_hit_reasons.append("no exact-cache-eligible lookups were recorded in scanned calls")
        if skip_reason_counts:
            top_skip = skip_reason_counts.most_common(1)[0][0]
            no_hit_reasons.append(f"most scanned cache decisions skipped lookup because {top_skip}")
        if exact_misses:
            no_hit_reasons.append("exact lookups occurred but missed the stored cache rows")
        if invalidated_lookups or all_invalidations:
            no_hit_reasons.append("some cache entries or lookups were invalidated by file dependency evidence")
        if not no_hit_reasons and cache_count:
            no_hit_reasons.append("cache rows exist but scanned calls did not recompute a matching exact key")

    return {
        "schema": "agentflow.cache_smoke_diagnostic.v1",
        "generated_at": utc_now(),
        "summary": {
            "cache_rows": cache_count,
            "exact_cache_rows": cache_count,
            "semantic_cache_rows": semantic_count,
            "cache_file_dependency_rows": dependency_count,
            "calls_scanned": len(call_rows),
            "calls_missing_cache_json": missing_cache_json,
            "eligible_lookup_count": exact_eligible,
            "exact_lookup_count": exact_eligible,
            "exact_miss_count": exact_misses,
            "semantic_miss_count": semantic_misses,
            "cache_hit_count": exact_hits,
            "exact_hit_count": exact_hits,
            "semantic_hit_count": semantic_hits,
            "skip_streaming_count": streaming_skip_count,
            "tools_disabled_skip_count": tools_disabled_skip_count,
            "invalidated_lookup_count": invalidated_lookups,
            "invalidated_cache_row_count": sum(all_invalidations.values()),
            "file_dependency_blocked_count": file_dependency_blocked_count,
            "duplicate_shape_group_count": len(duplicate_shapes),
            "duplicate_shape_candidate_rows": duplicate_candidate_rows,
            "duplicate_key_opportunity_count": duplicate_candidate_rows,
            "explains_zero_hits": exact_hits == 0,
            "no_hit_reasons": no_hit_reasons,
        },
        "cache_rows_by_model": _model_breakdown(store_obj),
        "newest_cache_rows": newest,
        "lookup_breakdown": _breakdown(status_reason_counts),
        "skip_reason_breakdown": _breakdown(skip_reason_counts),
        "invalidation_breakdown": _breakdown(all_invalidations),
        "duplicate_key_opportunity": {
            "basis": "metadata-only lookup shape; raw request bodies and cache keys are not emitted",
            "exact_key_confirmed": False,
            "candidate_group_count": len(duplicate_shapes),
            "candidate_row_count": duplicate_candidate_rows,
            "reconstruction_required_for_exact_confirmation": True,
        },
        "selected_cache_row_reconstruction": _key_stability_sample(store_obj, call_rows),
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "cache_keys_included": False,
            "raw_session_ids_included": False,
            "database_path_included": False,
        },
    }
