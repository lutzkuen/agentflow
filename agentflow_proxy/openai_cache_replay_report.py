from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_opportunity.v1"


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _increment(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = str(key or "unknown")
    counter[text] = counter.get(text, 0) + amount


def _breakdown(counter: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"value": key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _cost_bucket(cost: float) -> str:
    if cost <= 0:
        return "unknown"
    if cost < 0.001:
        return "lt_0_001_usd"
    if cost < 0.01:
        return "0_001_0_01_usd"
    if cost < 0.05:
        return "0_01_0_05_usd"
    if cost < 0.25:
        return "0_05_0_25_usd"
    return "gte_0_25_usd"


def _text_bucket(chars: int) -> str:
    if chars <= 0:
        return "unknown"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _status_bucket(status_code: Any) -> str:
    code = _as_int(status_code, -1)
    if code < 0:
        return "unknown"
    if code < 300:
        return "2xx"
    if code < 400:
        return "3xx"
    if code < 500:
        return "4xx"
    return "5xx"


def _source_surface(row: dict[str, Any]) -> str:
    return str(row.get("source_surface") or openai_source_surface(str(row.get("path") or "")))


def _endpoint(row: dict[str, Any]) -> str:
    return str(row.get("endpoint") or openai_endpoint(str(row.get("path") or "")))


def _feature_unit(routing: dict[str, Any]) -> dict[str, Any]:
    for key in ("openai_feature_unit", "openai_preflight_unit", "openai_local_feature_unit"):
        value = routing.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _row_text_chars(row: dict[str, Any], routing: dict[str, Any], feature: dict[str, Any]) -> int:
    chars = _as_int(routing.get("text_chars"))
    if chars > 0:
        return chars
    input_features = feature.get("input_features") if isinstance(feature.get("input_features"), dict) else {}
    chars = _as_int(input_features.get("text_chars"))
    if chars > 0:
        return chars
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * 4)


def _has_tools(row: dict[str, Any], routing: dict[str, Any], cache: dict[str, Any], feature: dict[str, Any]) -> bool:
    if routing.get("has_tools") is not None:
        return bool(routing.get("has_tools"))
    tool_features = feature.get("tool_features") if isinstance(feature.get("tool_features"), dict) else {}
    if tool_features.get("has_tools") is not None:
        return bool(tool_features.get("has_tools"))
    category = str(row.get("category") or routing.get("category") or "").lower()
    reason = str(cache.get("reason") or "").lower()
    return category.startswith("tool") or "tool" in reason


def _cache_status(row: dict[str, Any], cache: dict[str, Any]) -> str:
    status = str(cache.get("status") or "").strip().lower()
    if status:
        return status
    return "hit" if _as_int(row.get("cache_hit")) else "missing"


def _cache_reason(cache: dict[str, Any], status: str) -> str:
    reason = str(cache.get("reason") or "").strip().lower()
    if reason:
        return reason
    if status == "hit":
        return "cache-hit"
    if status == "miss":
        return "exact-miss"
    return "unknown"


def _replayability_level(cache: dict[str, Any], feature: dict[str, Any]) -> str:
    value = cache.get("replayability_level") or feature.get("replayability_level")
    return str(value or "metadata_shape")


def _fingerprint_value(cache: dict[str, Any], routing: dict[str, Any], feature: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        cache.get("request_fingerprint"),
        cache.get("request_fingerprint_sha256"),
        routing.get("request_fingerprint"),
    ]
    for container in (
        cache.get("pattern_features"),
        cache.get("managed_pattern_features"),
        routing.get("managed_pattern_features"),
        feature.get("grouping_identifiers"),
    ):
        if isinstance(container, dict):
            candidates.extend(
                [
                    container.get("request_fingerprint"),
                    container.get("request_fingerprint_sha256"),
                    container.get("request_hash"),
                ]
            )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dependency_fingerprint_available(cache: dict[str, Any]) -> bool:
    fingerprint = cache.get("file_dependency_fingerprint")
    if isinstance(fingerprint, dict):
        return bool(fingerprint.get("fingerprint_available") or fingerprint.get("fingerprint_sha256"))
    return bool(cache.get("file_dependency_fingerprint_available") or cache.get("file_dependency_fingerprint_sha256"))


def _sanitized_dependency_audit(cache: dict[str, Any]) -> dict[str, Any]:
    audit = cache.get("file_dependency_audit")
    if isinstance(audit, dict):
        safe = bool(audit.get("safe_invalidation_evidence"))
        return {
            "schema": str(audit.get("schema") or "agentflow.cache_file_dependency_audit.v1"),
            "file_watch_enabled": bool(audit.get("file_watch_enabled")),
            "snapshot_root_policy": str(audit.get("snapshot_root_policy") or "unknown"),
            "root_path_included": False,
            "snapshot_count": _as_int(audit.get("snapshot_count")),
            "snapshot_count_bucket": str(audit.get("snapshot_count_bucket") or "unknown"),
            "candidate_path_count_bucket": str(audit.get("candidate_path_count_bucket") or "unknown"),
            "max_paths": audit.get("max_paths"),
            "cap_exceeded": bool(audit.get("cap_exceeded")),
            "present_path_count": _as_int(audit.get("present_path_count")),
            "missing_path_count": _as_int(audit.get("missing_path_count")),
            "changed_path_count": _as_int(audit.get("changed_path_count")),
            "deleted_path_count": _as_int(audit.get("deleted_path_count")),
            "created_path_count": _as_int(audit.get("created_path_count")),
            "invalidation_reason": audit.get("invalidation_reason"),
            "safe_invalidation_evidence": safe,
            "file_dependency_evidence_available": bool(audit.get("file_dependency_evidence_available") or safe),
            "paths_included": False,
        }
    evidence = bool(cache.get("file_dependency_evidence_available") or cache.get("safe_invalidation_evidence"))
    return {
        "schema": "agentflow.cache_file_dependency_audit.v1",
        "file_watch_enabled": bool(cache.get("file_watch_enabled")),
        "snapshot_root_policy": "unknown",
        "root_path_included": False,
        "snapshot_count": _as_int(cache.get("file_dependency_count")),
        "snapshot_count_bucket": "unknown",
        "candidate_path_count_bucket": "unknown",
        "max_paths": None,
        "cap_exceeded": False,
        "present_path_count": _as_int(cache.get("file_dependency_count")),
        "missing_path_count": 0,
        "changed_path_count": 0,
        "deleted_path_count": 0,
        "created_path_count": 0,
        "invalidation_reason": cache.get("invalidation_reason"),
        "safe_invalidation_evidence": bool(cache.get("safe_invalidation_evidence")),
        "file_dependency_evidence_available": evidence,
        "paths_included": False,
    }


def _dependency_status(audit: dict[str, Any]) -> str:
    reason = str(audit.get("invalidation_reason") or "")
    if audit.get("cap_exceeded"):
        return "invalidated"
    if reason in {"dependency-changed", "dependency-deleted", "dependency-created", "dependency-cap-exceeded"}:
        return "invalidated"
    if audit.get("safe_invalidation_evidence"):
        return "stable"
    if reason in {"file-dependency-missing", "dependency-missing", "file-watch-disabled"}:
        return "missing"
    if not audit.get("file_dependency_evidence_available"):
        return "missing"
    return "unknown"


def _row_blockers(
    *,
    cache_status: str,
    cache_reason: str,
    stream: bool,
    has_tools: bool,
    dependency_status: str,
) -> list[str]:
    blockers: set[str] = set()
    if cache_status == "hit":
        blockers.add("already-cache-hit")
    if stream:
        blockers.add("unsupported-streaming-shape")
    if "tools-disabled" in cache_reason or "tool-cache-disabled" in cache_reason:
        blockers.add("tool-call-cache-disabled")
    if has_tools and dependency_status == "missing":
        blockers.add("file-dependency-missing")
    if has_tools and dependency_status == "invalidated":
        blockers.add("file-dependency-invalidated")
    if cache_status == "miss" and cache_reason in {"exact-miss", "miss"}:
        blockers.add("exact-miss")
    if not blockers or blockers == {"exact-miss"}:
        blockers.add("replay-rule-required")
    return sorted(blockers)


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    endpoint = str(basis.get("endpoint") or "unknown").replace("_", "-")
    category = str(basis.get("category") or "unknown").replace("_", "-")
    return f"openai-cache-replay:{endpoint}:{category}:{digest}"


def _new_group(basis: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_surface": basis["source_surface"],
        "endpoint": basis["endpoint"],
        "category": basis["category"],
        "workflow_phase": basis["workflow_phase"],
        "requested_model_family": basis["requested_model_family"],
        "stream": basis["stream"],
        "has_tools": basis["has_tools"],
        "cache_status": basis["cache_status"],
        "cache_reason": basis["cache_reason"],
        "replayability_level": basis["replayability_level"],
        "request_fingerprint_available": basis["request_fingerprint_available"],
        "file_dependency_fingerprint_available": basis["file_dependency_fingerprint_available"],
        "file_dependency_status": basis["file_dependency_status"],
        "text_bucket": basis["text_bucket"],
        "cost_bucket": basis["cost_bucket"],
        "matched_count": 0,
        "blocked_count": 0,
        "safety_eligible_count": 0,
        "already_cache_hit_count": 0,
        "duplicate_fingerprint_groups": 0,
        "duplicate_fingerprint_rows": 0,
        "session_pattern_repeated_rows": 0,
        "estimated_cost_usd": 0.0,
        "projected_savings_usd": 0.0,
        "input_tokens": 0,
        "status_counts": {},
        "blocker_counts": {},
        "cache_outcome_counts": {},
        "invalidation_counts": {},
        "file_dependency_audit": None,
        "_rows": [],
        "_fingerprints": {},
        "_session_shape_costs": {},
    }


def _merge_audit(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return {**right, "paths_included": False, "root_path_included": False}
    merged = {**left}
    for key in (
        "snapshot_count",
        "present_path_count",
        "missing_path_count",
        "changed_path_count",
        "deleted_path_count",
        "created_path_count",
    ):
        merged[key] = _as_int(merged.get(key)) + _as_int(right.get(key))
    merged["cap_exceeded"] = bool(merged.get("cap_exceeded") or right.get("cap_exceeded"))
    merged["safe_invalidation_evidence"] = bool(
        merged.get("safe_invalidation_evidence") or right.get("safe_invalidation_evidence")
    )
    merged["file_dependency_evidence_available"] = bool(
        merged.get("file_dependency_evidence_available") or right.get("file_dependency_evidence_available")
    )
    if not merged.get("invalidation_reason"):
        merged["invalidation_reason"] = right.get("invalidation_reason")
    merged["paths_included"] = False
    merged["root_path_included"] = False
    return merged


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    rows = group.pop("_rows", [])
    fingerprints = group.pop("_fingerprints", {})
    session_shape_costs = group.pop("_session_shape_costs", {})

    projected = 0.0
    duplicate_groups = 0
    duplicate_rows = 0
    for fingerprint_rows in fingerprints.values():
        if len(fingerprint_rows) <= 1:
            continue
        duplicate_groups += 1
        duplicate_rows += len(fingerprint_rows)
        costs = [_as_float(row.get("cost")) for row in fingerprint_rows]
        projected += max(0.0, sum(costs) - min(costs or [0.0]))

    session_repeated_rows = 0
    for session_rows in session_shape_costs.values():
        if len(session_rows) <= 1:
            continue
        session_repeated_rows += len(session_rows)
        if duplicate_groups == 0:
            costs = [_as_float(row.get("cost")) for row in session_rows]
            projected += max(0.0, sum(costs) - min(costs or [0.0]))

    group["duplicate_fingerprint_groups"] = duplicate_groups
    group["duplicate_fingerprint_rows"] = duplicate_rows
    group["session_pattern_repeated_rows"] = session_repeated_rows
    group["projected_savings_usd"] = round(projected, 6)
    group["estimated_cost_usd"] = round(_as_float(group.get("estimated_cost_usd")), 6)
    group["blocked_count"] = sum(1 for row in rows if row.get("safety_blocked"))
    group["safety_eligible_count"] = max(0, _as_int(group.get("matched_count")) - _as_int(group.get("blocked_count")))
    group["status_breakdown"] = _breakdown(group.pop("status_counts", {}))
    group["blocker_reason_breakdown"] = _breakdown(group.pop("blocker_counts", {}))
    group["cache_outcome_breakdown"] = _breakdown(group.pop("cache_outcome_counts", {}))
    group["invalidation_reason_breakdown"] = _breakdown(group.pop("invalidation_counts", {}))
    group["blockers"] = [row["value"] for row in group["blocker_reason_breakdown"]]
    group["privacy"] = {
        "metadata_only": True,
        "request_fingerprint_included": False,
        "file_dependency_fingerprint_included": False,
        "raw_session_ids_included": False,
        "file_paths_included": False,
        "cache_keys_included": False,
    }
    return group


def build_openai_cache_replay_report(store_obj: Any, limit: int = 1000) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, latency_ms, input_tokens_est, output_tokens_est,
                   actual_input_tokens, actual_output_tokens, cost_est_usd,
                   cost_baseline_usd, retry_count, category, crunch_json, routing_json,
                   cache_json, request_json, response_json, session_id
            from calls
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    blocker_totals: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    cache_outcome_counts: dict[str, int] = {}
    replayability_counts: dict[str, int] = {}
    fingerprint_availability_counts: dict[str, int] = {}
    dependency_fingerprint_availability_counts: dict[str, int] = {}
    dependency_status_counts: dict[str, int] = {}
    openai_count = 0
    fingerprint_rows = 0
    dependency_fingerprint_rows = 0
    body_rows_present = 0

    for row in rows:
        if str(row.get("provider") or "").lower() != "openai":
            continue
        openai_count += 1
        if row.get("request_json") or row.get("response_json"):
            body_rows_present += 1
        routing = _json_obj(row.get("routing_json"))
        cache = _json_obj(row.get("cache_json"))
        feature = _feature_unit(routing)
        endpoint = str(feature.get("endpoint") or _endpoint(row))
        source_surface = str(feature.get("source_surface") or _source_surface(row))
        requested_model = str(row.get("requested_model") or routing.get("requested_model") or "")
        model_family = str(
            feature.get("requested_model_family")
            or row.get("requested_model_family")
            or openai_model_family(requested_model)
            or "unknown"
        )
        category = str(row.get("category") or feature.get("category") or routing.get("category") or "unknown")
        workflow_phase = str(feature.get("workflow_phase") or routing.get("workflow_phase") or category or "unknown")
        stream = bool(_as_int(row.get("stream")) or feature.get("stream"))
        has_tools = _has_tools(row, routing, cache, feature)
        text_chars = _row_text_chars(row, routing, feature)
        input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est")) or max(0, text_chars // 4)
        cost = _as_float(row.get("cost_est_usd")) or _as_float(row.get("cost_baseline_usd"))
        cache_status = _cache_status(row, cache)
        cache_reason = _cache_reason(cache, cache_status)
        replayability = _replayability_level(cache, feature)
        fingerprint = _fingerprint_value(cache, routing, feature)
        fingerprint_state = "available" if fingerprint else "missing"
        if fingerprint:
            fingerprint_rows += 1
        dep_fingerprint_available = _dependency_fingerprint_available(cache)
        dep_fingerprint_state = "available" if dep_fingerprint_available else "missing"
        if dep_fingerprint_available:
            dependency_fingerprint_rows += 1
        audit = _sanitized_dependency_audit(cache)
        dep_status = _dependency_status(audit)
        blockers = _row_blockers(
            cache_status=cache_status,
            cache_reason=cache_reason,
            stream=stream,
            has_tools=has_tools,
            dependency_status=dep_status,
        )
        safety_blocked = bool(
            set(blockers)
            - {"exact-miss", "replay-rule-required", "already-cache-hit"}
        ) or cache_status == "hit"

        _increment(endpoint_counts, endpoint)
        _increment(category_counts, category)
        _increment(cache_outcome_counts, f"{cache_status}:{cache_reason}")
        _increment(replayability_counts, replayability)
        _increment(fingerprint_availability_counts, fingerprint_state)
        _increment(dependency_fingerprint_availability_counts, dep_fingerprint_state)
        _increment(dependency_status_counts, dep_status)
        for blocker in blockers:
            _increment(blocker_totals, blocker)

        basis = {
            "source_surface": source_surface,
            "endpoint": endpoint,
            "category": category,
            "workflow_phase": workflow_phase,
            "requested_model_family": model_family,
            "stream": stream,
            "has_tools": has_tools,
            "cache_status": cache_status,
            "cache_reason": cache_reason,
            "replayability_level": replayability,
            "request_fingerprint_available": bool(fingerprint),
            "file_dependency_fingerprint_available": dep_fingerprint_available,
            "file_dependency_status": dep_status,
            "text_bucket": _text_bucket(text_chars),
            "cost_bucket": _cost_bucket(cost),
        }
        cid = _candidate_id(basis)
        group = groups.setdefault(cid, _new_group(basis, cid))
        group["matched_count"] += 1
        group["already_cache_hit_count"] += int(cache_status == "hit")
        group["estimated_cost_usd"] += cost
        group["input_tokens"] += input_tokens
        group["file_dependency_audit"] = _merge_audit(group.get("file_dependency_audit"), audit)
        _increment(group["status_counts"], _status_bucket(row.get("status_code")))
        _increment(group["cache_outcome_counts"], f"{cache_status}:{cache_reason}")
        if audit.get("invalidation_reason"):
            _increment(group["invalidation_counts"], audit.get("invalidation_reason"))
        for blocker in blockers:
            _increment(group["blocker_counts"], blocker)
        safe_row = {"cost": cost, "safety_blocked": safety_blocked}
        group["_rows"].append(safe_row)
        if fingerprint and not safety_blocked:
            group["_fingerprints"].setdefault(fingerprint, []).append(safe_row)
        session_id = str(row.get("session_id") or "")
        if session_id and not fingerprint and not safety_blocked:
            group["_session_shape_costs"].setdefault(session_id, []).append(safe_row)

    candidates = [_finalize_group(group) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("projected_savings_usd")),
            _as_int(item.get("safety_eligible_count")),
            _as_float(item.get("estimated_cost_usd")),
            _as_int(item.get("matched_count")),
        ),
        reverse=True,
    )
    matched_count = sum(_as_int(item.get("matched_count")) for item in candidates)
    blocked_count = sum(_as_int(item.get("blocked_count")) for item in candidates)
    safety_eligible_count = sum(_as_int(item.get("safety_eligible_count")) for item in candidates)
    projected_savings = sum(_as_float(item.get("projected_savings_usd")) for item in candidates)
    estimated_cost = sum(_as_float(item.get("estimated_cost_usd")) for item in candidates)

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "summary": {
            "openai_call_count": openai_count,
            "candidate_count": len(candidates),
            "matched_count": matched_count,
            "blocked_count": blocked_count,
            "safety_eligible_count": safety_eligible_count,
            "already_cache_hit_count": blocker_totals.get("already-cache-hit", 0),
            "request_fingerprint_rows": fingerprint_rows,
            "file_dependency_fingerprint_rows": dependency_fingerprint_rows,
            "request_body_rows_present_but_not_read": body_rows_present,
            "estimated_cost_usd": round(estimated_cost, 6),
            "projected_savings_usd": round(projected_savings, 6),
        },
        "projection_policy": {
            "schema": "agentflow.openai_cache_replay_projection_policy.v1",
            "provider_calls_made": False,
            "raw_body_required": False,
            "method": "metadata-only repeated request fingerprints, then same-session shape repeats when fingerprints are unavailable",
            "conservative_rule": "project only repeated observed local cost beyond the first matching call",
            "default_apply": False,
        },
        "endpoint_breakdown": _breakdown(endpoint_counts),
        "category_breakdown": _breakdown(category_counts),
        "cache_outcome_breakdown": _breakdown(cache_outcome_counts),
        "replayability_breakdown": _breakdown(replayability_counts),
        "request_fingerprint_availability_breakdown": _breakdown(fingerprint_availability_counts),
        "file_dependency_fingerprint_availability_breakdown": _breakdown(dependency_fingerprint_availability_counts),
        "file_dependency_status_breakdown": _breakdown(dependency_status_counts),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "file_dependency_fingerprints_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "basis": "local calls table metadata plus sanitized routing/cache decision summaries only",
        },
    }
