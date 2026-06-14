from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.cache import cache_pattern_hashes_from_features, cache_pattern_rules_from_policy_payload
from agentflow_proxy.openai_cache_replay_report import (
    _as_float,
    _as_int,
    _breakdown,
    _cache_reason,
    _cache_status,
    _dependency_status,
    _endpoint,
    _feature_unit,
    _fingerprint_value,
    _has_tools,
    _increment,
    _json_obj,
    _replayability_level,
    _row_text_chars,
    _sanitized_dependency_audit,
    _source_surface,
    _text_bucket,
)
from agentflow_proxy.optimization.openai_features import openai_model_family
from agentflow_proxy.pattern_rollout import pattern_canary_decision, pattern_rollout_public_meta
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.openai_cache_replay_dry_run.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _public_hash(value: Any) -> str:
    return "sha256:" + _hash(value)[:16]


def _values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float, bool))]
    return []


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return False


def _condition_matches(conditions: dict[str, Any], key: str, actual: Any) -> bool:
    if key not in conditions:
        return True
    expected = {item.lower() for item in _values(conditions.get(key))}
    if not expected:
        return True
    return str(actual or "").lower() in expected


def _merge_features(cache: dict[str, Any], routing: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in (
        routing.get("managed_pattern_features"),
        cache.get("managed_pattern_features"),
        cache.get("pattern_features"),
        feature,
    ):
        if isinstance(item, dict):
            merged.update(item)
    return merged


def _public_canary(canary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(canary, dict):
        return None
    public = {
        key: value
        for key, value in canary.items()
        if key not in {"pattern_hashes", "rule_id", "candidate_id"}
    }
    public["pattern_hash_count"] = len(canary.get("pattern_hashes") or []) if isinstance(canary.get("pattern_hashes"), list) else 0
    public["pattern_hashes_included"] = False
    return public


def _projection_meta(rule: dict[str, Any]) -> dict[str, Any] | None:
    graduation = rule.get("graduation") if isinstance(rule.get("graduation"), dict) else None
    if not isinstance(graduation, dict):
        return None
    projection: dict[str, Any] = {}
    for key in (
        "schema",
        "source_schema",
        "source_reason",
        "source_verdict",
        "cohort_bucket",
        "source_surface",
        "endpoint",
        "category",
        "workflow_phase",
        "text_bucket",
        "token_bucket",
    ):
        if graduation.get(key) is not None:
            projection[key] = graduation.get(key)
    for key in (
        "projected_hits",
        "projected_hit_count",
        "sample_count",
        "applied_count",
        "holdout_count",
    ):
        if graduation.get(key) is not None:
            projection[key] = _as_int(graduation.get(key))
    for key in (
        "projected_savings_usd",
        "observed_savings_usd",
        "projected_saved_cost_usd",
    ):
        if graduation.get(key) is not None:
            projection[key] = round(_as_float(graduation.get(key)), 9)
    if graduation.get("aggregate_only") is not None:
        projection["aggregate_only"] = bool(graduation.get("aggregate_only"))
    if not projection:
        return None
    projection["metadata_only"] = True
    projection["raw_prompts_included"] = False
    projection["raw_request_bodies_included"] = False
    projection["raw_responses_included"] = False
    projection["cache_keys_included"] = False
    projection["request_ids_included"] = False
    projection["session_ids_included"] = False
    return projection


def _row_unit(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("provider") or "").lower() != "openai":
        return None
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
    request_fingerprint = _fingerprint_value(cache, routing, feature)
    pattern_features = _merge_features(cache, routing, feature)
    pattern_hashes = cache_pattern_hashes_from_features(pattern_features)
    audit = _sanitized_dependency_audit(cache)
    dep_status = _dependency_status(audit)
    session_id = str(row.get("session_id") or "")
    session_id_hash = _public_hash({"session_id": session_id}) if session_id else None
    return {
        "created_at": row.get("created_at"),
        "source_surface": source_surface,
        "endpoint": endpoint,
        "category": category,
        "workflow_phase": workflow_phase,
        "requested_model": requested_model,
        "requested_model_family": model_family,
        "stream": stream,
        "has_tools": has_tools,
        "cache_status": cache_status,
        "cache_reason": cache_reason,
        "replayability_level": replayability,
        "request_fingerprint_available": bool(request_fingerprint),
        "request_fingerprint": request_fingerprint,
        "pattern_features": {
            **pattern_features,
            "source_surface": source_surface,
            "category": category,
            "workflow_phase": workflow_phase,
            "text_bucket": _text_bucket(text_chars),
            "requested_model": requested_model,
            "candidate_target_model": str(row.get("routed_model") or requested_model),
            "replayability_level": replayability,
            "has_tools": has_tools,
            "stream": stream,
            "session_id_hash": session_id_hash,
        },
        "pattern_hashes": pattern_hashes,
        "session_id_available": bool(session_id),
        "session_id_hash": session_id_hash,
        "file_dependency_status": dep_status,
        "file_dependency_audit": audit,
        "file_dependency_evidence_available": bool(audit.get("file_dependency_evidence_available")),
        "safe_invalidation_evidence": bool(audit.get("safe_invalidation_evidence")),
        "text_bucket": _text_bucket(text_chars),
        "input_tokens": input_tokens,
        "cost_est_usd": cost,
    }


def _read_openai_units(store_obj: Any, limit: int) -> list[dict[str, Any]]:
    capped = max(1, min(_as_int(limit) or 1000, 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, cache_hit,
                   status_code, input_tokens_est, actual_input_tokens, cost_est_usd,
                   cost_baseline_usd, category, routing_json, cache_json, session_id
            from calls
            order by created_at desc
            limit ?
            """,
            (capped,),
        ).fetchall()
    ]
    units: list[dict[str, Any]] = []
    for row in rows:
        unit = _row_unit(row)
        if unit is not None:
            units.append(unit)
    return units


def _rule_decision(unit: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    last_reason = "no-matching-rule"
    feature_hashes = set(unit.get("pattern_hashes") or [])
    if not feature_hashes:
        last_reason = "pattern-features-missing"
    for rule in rules:
        rule_id = str(rule.get("id") or "openai-cache-replay-rule")
        candidate_id = rule.get("candidate_id")
        policy_source = str(rule.get("policy_source") or "managed-recommended")
        if not rule.get("enabled", True):
            last_reason = "disabled"
            continue
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        rule_hashes = {str(item) for item in conditions.get("pattern_hashes") or []}
        wildcard_rule = "sha256:*" in rule_hashes
        matched_hashes = sorted(feature_hashes) if wildcard_rule else sorted(feature_hashes.intersection(rule_hashes))
        if not matched_hashes and not wildcard_rule:
            last_reason = "pattern-hash-mismatch"
            continue
        for key in ("source_surface", "endpoint", "category", "workflow_phase", "text_bucket"):
            if not _condition_matches(conditions, key, unit.get(key)):
                last_reason = f"{key}-mismatch"
                break
        else:
            excluded_categories = {item.lower() for item in _values(conditions.get("category_not_in"))}
            if excluded_categories and str(unit.get("category") or "").lower() in excluded_categories:
                last_reason = "category-excluded"
                continue
            if "has_tools" in conditions and _bool_value(conditions.get("has_tools")) != bool(unit.get("has_tools")):
                last_reason = "has-tools-mismatch"
                continue
            if "stream" in conditions and _bool_value(conditions.get("stream")) != bool(unit.get("stream")):
                last_reason = "stream-mismatch"
                continue
            replayability_levels = {item.lower() for item in _values(conditions.get("replayability_levels"))}
            if replayability_levels and str(unit.get("replayability_level") or "").lower() not in replayability_levels:
                last_reason = "replayability-gate-mismatch"
                continue
            model_pattern = str(conditions.get("model_pattern") or "").strip().lower()
            if model_pattern and model_pattern not in str(unit.get("requested_model") or "").lower():
                last_reason = "model-pattern-mismatch"
                continue

            action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            base = {
                "rule_id": rule_id,
                "candidate_id": candidate_id,
                "policy_source": policy_source,
                "matched_pattern_hash_count": len(matched_hashes),
                "matched_pattern_hashes_included": False,
                "rollout": pattern_rollout_public_meta(rule.get("rollout")),
            }
            projection = _projection_meta(rule)
            if projection:
                base["projection"] = projection
                if projection.get("cohort_bucket") is not None:
                    base["cohort_bucket"] = projection.get("cohort_bucket")
                if projection.get("projected_hits") is not None:
                    base["projected_hits"] = projection.get("projected_hits")
                if projection.get("projected_savings_usd") is not None:
                    base["projected_savings_usd"] = projection.get("projected_savings_usd")
            if action.get("type") not in {"exact_cache", "exact_cache_pattern"}:
                return {**base, "status": "blocked", "reason": "unsupported-action", "blockers": ["unsupported-action"]}
            if bool(unit.get("stream")) and not bool(action.get("streaming")):
                return {**base, "status": "blocked", "reason": "streaming-not-allowed", "blockers": ["streaming-not-allowed"]}
            if str(action.get("scope") or "session") != "session":
                return {**base, "status": "blocked", "reason": "unsupported-scope", "blockers": ["unsupported-scope"]}
            if not unit.get("session_id_available"):
                return {**base, "status": "blocked", "reason": "session-scope-missing", "blockers": ["session-scope-missing"]}
            if bool(unit.get("has_tools")) and not bool(action.get("allow_tool_calls")):
                return {
                    **base,
                    "status": "invalidation-required",
                    "reason": "tool-cache-rule-requires-safe-invalidation",
                    "blockers": ["tool-call-disabled", "safe-invalidation-required"],
                    "requires_file_dependency_evidence": True,
                }
            if bool(unit.get("has_tools")) and not bool(action.get("safe_invalidation")):
                return {
                    **base,
                    "status": "invalidation-required",
                    "reason": "tool-cache-rule-missing-safe-invalidation",
                    "blockers": ["safe-invalidation-required"],
                    "requires_file_dependency_evidence": True,
                }
            if bool(unit.get("has_tools")):
                audit = unit.get("file_dependency_audit") if isinstance(unit.get("file_dependency_audit"), dict) else {}
                reason = str(audit.get("invalidation_reason") or "")
                audit_blocker = None
                if audit.get("cap_exceeded"):
                    audit_blocker = "dependency-cap-exceeded"
                elif reason in {
                    "file-dependency-missing",
                    "dependency-missing",
                    "dependency-changed",
                    "dependency-deleted",
                    "dependency-created",
                    "file-watch-disabled",
                }:
                    audit_blocker = reason
                elif not bool(unit.get("safe_invalidation_evidence")):
                    audit_blocker = "file-dependency-missing"
                if audit_blocker:
                    return {
                        **base,
                        "status": "invalidation-required",
                        "reason": audit_blocker,
                        "blockers": [audit_blocker],
                        "requires_file_dependency_evidence": True,
                        "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                        "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                    }
            key_basis = {
                "provider": "openai",
                "scope": "session",
                "session_id_hash": unit.get("session_id_hash"),
                "source_surface": unit.get("source_surface"),
                "endpoint": unit.get("endpoint"),
                "rule_id": rule_id,
                "candidate_id": str(candidate_id or ""),
                "matched_pattern_hashes": matched_hashes,
                "request_fingerprint_hash": _public_hash(unit.get("request_fingerprint")) if unit.get("request_fingerprint") else None,
                "category": unit.get("category"),
                "workflow_phase": unit.get("workflow_phase"),
            }
            replay_key = _public_hash(key_basis)
            canary_features = {
                "source_surface": unit.get("source_surface"),
                "category": unit.get("category"),
                "workflow_phase": unit.get("workflow_phase"),
                "text_bucket": unit.get("text_bucket"),
                "requested_model": unit.get("requested_model"),
                "replayability_level": unit.get("replayability_level"),
                "has_tools": bool(unit.get("has_tools")),
                "stream": bool(unit.get("stream")),
                "session_id_hash": unit.get("session_id_hash"),
                "request_fingerprint": replay_key,
            }
            canary = pattern_canary_decision(
                rollout=rule.get("rollout"),
                rule_id=rule_id,
                candidate_id=candidate_id,
                pattern_hashes=matched_hashes,
                features=canary_features,
            )
            if canary.get("enabled") and not canary.get("selected", True):
                return {
                    **base,
                    "status": "holdout",
                    "reason": "canary_holdout",
                    "blockers": ["canary_holdout"],
                    "session_scoped_key_available": True,
                    "session_scoped_key_fingerprint": replay_key,
                    "canary": _public_canary(canary),
                }
            return {
                **base,
                "status": "projected-applied",
                "reason": "rule-match",
                "blockers": [],
                "session_scoped_key_available": True,
                "session_scoped_key_fingerprint": replay_key,
                "canary": _public_canary(canary) if canary.get("enabled") else None,
                "requires_file_dependency_evidence": bool(unit.get("has_tools")),
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
            }
    return {"status": "unmatched", "reason": last_reason, "blockers": [last_reason] if last_reason else []}


def _add_breakdown_counts(unit: dict[str, Any], decision: dict[str, Any], counts: dict[str, dict[str, int]]) -> None:
    _increment(counts["status"], decision.get("status") or "unknown")
    _increment(counts["reason"], decision.get("reason") or "unknown")
    _increment(counts["endpoint"], unit.get("endpoint") or "unknown")
    _increment(counts["source_surface"], unit.get("source_surface") or "unknown")
    _increment(counts["file_dependency_status"], unit.get("file_dependency_status") or "unknown")
    for blocker in decision.get("blockers") or []:
        _increment(counts["blocker"], blocker)


def build_openai_cache_replay_dry_run(store_obj: Any, proposed_policy: Any, *, limit: int = 1000) -> dict[str, Any]:
    rules = cache_pattern_rules_from_policy_payload(proposed_policy)
    units = _read_openai_units(store_obj, limit)
    cache_rows_before = _cache_row_count(store_obj)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {
        "status": {},
        "reason": {},
        "blocker": {},
        "endpoint": {},
        "source_surface": {},
        "file_dependency_status": {},
    }
    for unit in units:
        decision = _rule_decision(unit, rules)
        _add_breakdown_counts(unit, decision, counts)
        status = str(decision.get("status") or "unknown")
        key = (
            status,
            str(decision.get("rule_id") or ""),
            str(decision.get("session_scoped_key_fingerprint") or _public_hash({
                "source_surface": unit.get("source_surface"),
                "endpoint": unit.get("endpoint"),
                "category": unit.get("category"),
                "reason": decision.get("reason"),
            })),
        )
        bucket = grouped.setdefault(
            key,
            {
                "status": status,
                "reason": decision.get("reason"),
                "rule_id": decision.get("rule_id"),
                "candidate_id": decision.get("candidate_id"),
                "policy_source": decision.get("policy_source"),
                "source_surface": unit.get("source_surface"),
                "endpoint": unit.get("endpoint"),
                "category": unit.get("category"),
                "workflow_phase": unit.get("workflow_phase"),
                "requested_model_family": unit.get("requested_model_family"),
                "stream": bool(unit.get("stream")),
                "has_tools": bool(unit.get("has_tools")),
                "replayability_level": unit.get("replayability_level"),
                "session_scoped_key_available": bool(decision.get("session_scoped_key_available")),
                "session_scoped_key_fingerprint": decision.get("session_scoped_key_fingerprint"),
                "session_scoped_key_fingerprint_included": bool(decision.get("session_scoped_key_fingerprint")),
                "request_fingerprint_available": bool(unit.get("request_fingerprint_available")),
                "file_dependency_status": unit.get("file_dependency_status"),
                "file_dependency_evidence_available": bool(unit.get("file_dependency_evidence_available")),
                "safe_invalidation_evidence": bool(unit.get("safe_invalidation_evidence")),
                "requires_file_dependency_evidence": bool(decision.get("requires_file_dependency_evidence")),
                "matched_pattern_hash_count": _as_int(decision.get("matched_pattern_hash_count")),
                "matched_pattern_hashes_included": False,
                "cohort_bucket": decision.get("cohort_bucket"),
                "projection": decision.get("projection"),
                "count": 0,
                "input_tokens": 0,
                "estimated_cost_usd": 0.0,
                "projected_hits": 0,
                "projected_savings_usd": 0.0,
                "first_seen_at": unit.get("created_at"),
                "last_seen_at": unit.get("created_at"),
                "_blockers": set(decision.get("blockers") or []),
            },
        )
        bucket["count"] += 1
        bucket["input_tokens"] += _as_int(unit.get("input_tokens"))
        bucket["estimated_cost_usd"] += _as_float(unit.get("cost_est_usd"))
        if decision.get("canary") and not bucket.get("canary"):
            bucket["canary"] = decision.get("canary")
        if decision.get("rollout") and not bucket.get("rollout"):
            bucket["rollout"] = decision.get("rollout")
        for blocker in decision.get("blockers") or []:
            bucket["_blockers"].add(str(blocker))
        if str(unit.get("created_at") or "") < str(bucket.get("first_seen_at") or unit.get("created_at") or ""):
            bucket["first_seen_at"] = unit.get("created_at")
        if str(unit.get("created_at") or "") > str(bucket.get("last_seen_at") or ""):
            bucket["last_seen_at"] = unit.get("created_at")

    rows: list[dict[str, Any]] = []
    projected_hits = 0
    projected_savings = 0.0
    for bucket in grouped.values():
        if bucket["status"] == "projected-applied" and _as_int(bucket.get("count")) > 1:
            avg_cost = float(bucket["estimated_cost_usd"]) / _as_int(bucket.get("count"))
            bucket["projected_hits"] = _as_int(bucket.get("count")) - 1
            bucket["projected_savings_usd"] = max(0.0, float(bucket["estimated_cost_usd"]) - avg_cost)
            projected_hits += bucket["projected_hits"]
            projected_savings += bucket["projected_savings_usd"]
        finalized = {key: value for key, value in bucket.items() if key != "_blockers"}
        finalized["blockers"] = sorted(bucket["_blockers"])
        finalized["estimated_cost_usd"] = round(float(finalized["estimated_cost_usd"]), 6)
        finalized["projected_savings_usd"] = round(float(finalized["projected_savings_usd"]), 6)
        rows.append(finalized)
    rows.sort(
        key=lambda row: (
            _as_float(row.get("projected_savings_usd")),
            _as_int(row.get("projected_hits")),
            _as_int(row.get("count")),
            _as_float(row.get("estimated_cost_usd")),
        ),
        reverse=True,
    )
    cache_rows_after = _cache_row_count(store_obj)
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "summary": {
            "openai_rows_considered": len(units),
            "policy_rule_count": len(rules),
            "matched_rows": len(units) - counts["status"].get("unmatched", 0),
            "projected_applied_rows": counts["status"].get("projected-applied", 0),
            "holdout_rows": counts["status"].get("holdout", 0),
            "blocked_rows": sum(
                value
                for key, value in counts["status"].items()
                if key in {"blocked", "invalidation-required"}
            ),
            "invalidation_required_rows": counts["status"].get("invalidation-required", 0),
            "session_scoped_key_available_rows": sum(row["count"] for row in rows if row.get("session_scoped_key_available")),
            "projected_hits": projected_hits,
            "projected_savings_usd": round(projected_savings, 6),
            "cache_rows_before": cache_rows_before,
            "cache_rows_after": cache_rows_after,
            "cache_table_mutated": cache_rows_before != cache_rows_after,
            "provider_calls_made": 0,
            "cache_entries_written": 0,
        },
        "status_breakdown": _breakdown(counts["status"]),
        "reason_breakdown": _breakdown(counts["reason"]),
        "blocker_breakdown": _breakdown(counts["blocker"]),
        "endpoint_breakdown": _breakdown(counts["endpoint"]),
        "source_surface_breakdown": _breakdown(counts["source_surface"]),
        "file_dependency_status_breakdown": _breakdown(counts["file_dependency_status"]),
        "rows": rows[: max(1, min(_as_int(limit) or 50, 1000))],
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_responses_included": False,
            "raw_provider_bodies_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "raw_session_ids_included": False,
            "cache_keys_included": False,
            "request_fingerprints_included": False,
            "pattern_hashes_included": False,
            "secrets_included": False,
            "provider_calls_made": False,
            "basis": "local OpenAI call metadata plus sanitized routing/cache/dependency decisions only",
        },
    }


def _cache_row_count(store_obj: Any) -> int | None:
    try:
        row = store_obj.conn.execute("select count(*) as c from cache").fetchone()
        return _as_int(row["c"] if row is not None else 0)
    except Exception:
        return None
