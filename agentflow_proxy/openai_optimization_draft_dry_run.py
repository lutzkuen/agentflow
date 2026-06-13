from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from agentflow_proxy.openai_optimization_governor import (
    LIFECYCLE_SOURCE_SURFACE,
    build_openai_optimization_governor,
)
from agentflow_proxy.optimization.openai_features import openai_endpoint, openai_model_family, openai_source_surface
from agentflow_proxy.policy_bundle import validate_policy_bundle
from agentflow_proxy.policy_workbench import load_staged_policy_draft
from agentflow_proxy.pricing import estimate_cost, pricing_basis
from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.openai_optimization_draft_dry_run.v1"
LIFECYCLE_SCHEMA = "agentflow.openai_optimization_draft_dry_run_lifecycle_feedback.v1"
RAW_FIELD_NAMES = {
    "authorization",
    "body",
    "cache_key",
    "cache_keys",
    "content",
    "file_path",
    "file_paths",
    "messages",
    "prompt",
    "provider_body",
    "raw_body",
    "raw_messages",
    "raw_prompt",
    "raw_request",
    "raw_response",
    "request_id",
    "response",
    "secret",
    "session_id",
    "tenant_id",
    "tool_payload",
    "transcript",
}
SAFE_RAW_FLAG_KEYS = {
    "raw_commands_included",
    "raw_paths_included",
    "raw_request_ids_included",
    "raw_terminal_text_included",
}
FAMILY_ALIASES = {
    "routing": "routing",
    "old_context_summary": "old_context_summary",
    "old_context_summarization": "old_context_summary",
    "summary": "old_context_summary",
    "summarization": "old_context_summary",
    "cache": "cache_replay",
    "cache_replay": "cache_replay",
}
FAMILY_TO_ACTION = {
    "routing": "routing",
    "old_context_summary": "old_context_summarization",
    "cache_replay": "cache",
}
PRIVACY = {
    "metadata_only": True,
    "local_only": True,
    "raw_prompts_included": False,
    "raw_messages_included": False,
    "raw_request_bodies_included": False,
    "raw_responses_included": False,
    "provider_bodies_included": False,
    "tool_payloads_included": False,
    "file_paths_included": False,
    "request_ids_included": False,
    "session_ids_included": False,
    "cache_keys_included": False,
    "provider_calls_made": False,
    "managed_server_calls_made": False,
    "active_policy_files_written": False,
}


def _family(value: Any) -> str:
    return FAMILY_ALIASES.get(str(value or "").strip().lower().replace("-", "_"), str(value or "unknown").strip().lower())


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


def _text_bucket(chars: Any) -> str:
    value = _as_int(chars)
    if value < 1500:
        return "lt-1_5k"
    if value < 8000:
        return "1_5k-8k"
    if value < 30000:
        return "8k-30k"
    return "gte-30k"


def _hash_public(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _public_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    lowered = text.lower()
    if len(text) > 128 or any(char.isspace() for char in text) or any(term in lowered for term in RAW_FIELD_NAMES):
        return _hash_public(text)
    return text


def _scan_raw_fields(value: Any, errors: list[dict[str, str]], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            child_path = f"{path}.{key}"
            if lowered in SAFE_RAW_FLAG_KEYS:
                if bool(child):
                    errors.append({"path": child_path, "message": "OpenAI optimization draft dry-runs require local-only metadata and no raw payload flags"})
                    continue
                _scan_raw_fields(child, errors, child_path)
                continue
            if lowered in RAW_FIELD_NAMES or lowered.startswith("raw_"):
                errors.append({"path": child_path, "message": "raw or local-identifier fields are not accepted in OpenAI optimization draft dry-runs"})
                continue
            if lowered in {"provider_forwarding", "server_content_processing", "managed_enforced"} and bool(child):
                errors.append({"path": child_path, "message": "OpenAI optimization draft dry-runs require local-only managed-recommended actions"})
                continue
            _scan_raw_fields(child, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_raw_fields(child, errors, f"{path}[{index}]")


def _route_policy(bundle: dict[str, Any]) -> dict[str, Any] | None:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    routing = policies.get("routing") if isinstance(policies.get("routing"), dict) else {}
    openai = routing.get("openai") if isinstance(routing.get("openai"), dict) else {}
    canary = openai.get("canary") if isinstance(openai.get("canary"), dict) else routing.get("openai_canary")
    return canary if isinstance(canary, dict) else None


def _summary_policy(bundle: dict[str, Any]) -> dict[str, Any] | None:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    crunch = policies.get("crunch") if isinstance(policies.get("crunch"), dict) else {}
    summary = crunch.get("old_context_summarization") if isinstance(crunch.get("old_context_summarization"), dict) else None
    return summary


def _cache_rules(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    policies = bundle.get("policies") if isinstance(bundle.get("policies"), dict) else {}
    cache = policies.get("cache") if isinstance(policies.get("cache"), dict) else {}
    rules = cache.get("pattern_rules") if isinstance(cache.get("pattern_rules"), list) else []
    return [rule for rule in rules if isinstance(rule, dict)]


def _actions_from_manifest(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    metadata = (manifest or {}).get("metadata") if isinstance((manifest or {}).get("metadata"), dict) else {}
    review = metadata.get("openai_optimization_review") if isinstance(metadata.get("openai_optimization_review"), dict) else {}
    actions: dict[str, dict[str, Any]] = {}
    for item in review.get("selected_actions") if isinstance(review.get("selected_actions"), list) else []:
        if not isinstance(item, dict):
            continue
        family = _family(item.get("action_family"))
        actions[family] = item
    return actions


def _read_rows(store_obj: Any, limit: int) -> list[dict[str, Any]]:
    capped = max(1, min(_as_int(limit, 1000), 10_000))
    rows = store_obj.conn.execute(
        """
        select id, created_at, path, coalesce(provider, 'anthropic') as provider,
               source_surface, endpoint, requested_model, routed_model,
               requested_model_family, routed_model_family, stream, cache_hit,
               status_code, latency_ms, retry_count, input_tokens_est,
               output_tokens_est, actual_input_tokens, actual_output_tokens,
               cost_est_usd, cost_baseline_usd, category, routing_json,
               crunch_json, cache_json, session_id
        from calls
        order by created_at desc
        limit ?
        """,
        (capped,),
    ).fetchall()
    return [dict(row) for row in rows if str(row["provider"] or "").lower() == "openai"]


def _row_unit(row: dict[str, Any]) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    path = str(row.get("path") or "")
    endpoint = str(row.get("endpoint") or routing.get("endpoint") or openai_endpoint(path))
    source_surface = str(row.get("source_surface") or routing.get("source_surface") or openai_source_surface(path))
    requested = str(row.get("requested_model") or routing.get("requested_model") or "")
    text_chars = _as_int(routing.get("text_chars")) or max(0, (_as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))) * 4)
    input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est")) or max(0, text_chars // 4)
    output_tokens = _as_int(row.get("actual_output_tokens")) or _as_int(row.get("output_tokens_est"))
    session = str(row.get("session_id") or "")
    return {
        "row_hash": _hash_public({"id": row.get("id"), "created_at": row.get("created_at")}),
        "created_at": row.get("created_at"),
        "source_surface": source_surface,
        "endpoint": endpoint,
        "path": path or ("/v1/chat/completions" if endpoint == "chat_completions" else "/v1/responses"),
        "requested_model": requested,
        "requested_model_family": str(row.get("requested_model_family") or openai_model_family(requested) or "unknown"),
        "category": str(row.get("category") or routing.get("category") or "unknown"),
        "workflow_phase": str(routing.get("workflow_phase") or "unknown"),
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "has_tools": bool(routing.get("has_tools") or cache.get("has_tools")),
        "stream": bool(_as_int(row.get("stream")) or routing.get("stream")),
        "status_code": row.get("status_code"),
        "latency_ms": row.get("latency_ms"),
        "retry_count": row.get("retry_count"),
        "cost_est_usd": _as_float(row.get("cost_est_usd")),
        "cost_baseline_usd": _as_float(row.get("cost_baseline_usd")) or _as_float(row.get("cost_est_usd")),
        "session_hash": _hash_public({"session_id": session}) if session else None,
        "routing": routing,
        "crunch": crunch,
        "cache": cache,
    }


def _values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _matches_value(allowed: Any, actual: Any) -> bool:
    values = {item.lower() for item in _values(allowed)}
    return not values or str(actual or "").lower() in values


def _cohort(unit: dict[str, Any], *, family: str, candidate_id: str, canary_fraction: float, holdout_fraction: float, salt: str) -> tuple[str, bool]:
    basis = {
        "family": family,
        "candidate_id": candidate_id,
        "endpoint": unit["endpoint"],
        "model": unit["requested_model_family"],
        "category": unit["category"],
        "text_bucket": unit["text_bucket"],
        "session_hash": unit.get("session_hash"),
    }
    score = int(hashlib.sha256((salt + ":" + stable_json(basis)).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if score < holdout_fraction:
        return "holdout", False
    if score < holdout_fraction + canary_fraction:
        return "applied", True
    return "not_selected", False


def _evidence_blocker(action: dict[str, Any] | None, policy: dict[str, Any] | None) -> str | None:
    evidence = {}
    if isinstance(action, dict) and isinstance(action.get("evidence_freshness"), dict):
        evidence = action["evidence_freshness"]
    elif isinstance(policy, dict):
        managed = policy.get("managed_recommendation") if isinstance(policy.get("managed_recommendation"), dict) else {}
        if isinstance(managed.get("evidence_freshness"), dict):
            evidence = managed["evidence_freshness"]
    if evidence.get("stale") is True:
        return "stale-evidence"
    if evidence and _as_float(evidence.get("sample_count"), 1.0) <= 0:
        return "missing-evidence"
    if not evidence and not (isinstance(action, dict) and isinstance(action.get("expected_impact"), dict)):
        return "missing-evidence"
    return None


def _routing_candidate(unit: dict[str, Any], policy: dict[str, Any] | None, action: dict[str, Any] | None, *, canary_fraction: float, holdout_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(policy, dict):
        return {}, {"family": "routing", "status": "unsupported", "eligible": False, "reason": "missing-routing-policy", "savings": 0.0}
    target_model = str(policy.get("target_model") or "")
    meta = {
        "enabled": True,
        "policy_source": "managed-recommended",
        "policy_id": policy.get("policy_id") or policy.get("rule_id") or "openai-optimization-draft-routing",
        "rule_id": policy.get("policy_id") or policy.get("rule_id") or "openai-optimization-draft-routing",
        "target_candidate_id": policy.get("target_candidate_id") or policy.get("candidate_id"),
        "candidate_id": policy.get("candidate_id") or policy.get("target_candidate_id"),
        "requested_model": unit["requested_model"],
        "target_model": target_model,
        "actual_forwarded_model": unit["requested_model"],
        "category": unit["category"],
        "text_chars": unit["text_chars"],
        "has_tools": unit["has_tools"],
        "stream": unit["stream"],
        "canary_fraction": canary_fraction,
        "holdout_fraction": holdout_fraction,
    }
    reason = _evidence_blocker(action, policy)
    if reason:
        meta.update({"status": "safety_stopped" if reason == "stale-evidence" else "ineligible", "reason": reason, "safety_stop": {"tripped": True, "reason_codes": [reason]}})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": reason, "eligible": True, "reason": reason, "savings": 0.0}
    model_pattern = str(policy.get("model_pattern") or "").lower()
    if model_pattern and model_pattern not in unit["requested_model"].lower():
        meta.update({"status": "ineligible", "reason": "requested-model-not-enabled"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "unsupported", "eligible": False, "reason": "requested-model-not-enabled", "savings": 0.0}
    if unit["has_tools"] and not bool(policy.get("allow_tools")):
        meta.update({"status": "ineligible", "reason": "tool-request-not-enabled"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "unsupported", "eligible": False, "reason": "tool-request-not-enabled", "savings": 0.0}
    if unit["stream"] and not bool(policy.get("allow_stream")):
        meta.update({"status": "ineligible", "reason": "streaming-not-enabled"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "unsupported", "eligible": False, "reason": "streaming-not-enabled", "savings": 0.0}
    if not _matches_value(policy.get("eligible_categories"), unit["category"]) or unit["category"] in set(_values(policy.get("excluded_categories"))):
        meta.update({"status": "ineligible", "reason": "category-not-enabled"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "no_match", "eligible": False, "reason": "category-not-enabled", "savings": 0.0}
    if unit["text_chars"] < _as_int(policy.get("min_text_chars")) or (_as_int(policy.get("max_text_chars")) and unit["text_chars"] > _as_int(policy.get("max_text_chars"))):
        meta.update({"status": "ineligible", "reason": "request-size-not-enabled"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "no_match", "eligible": False, "reason": "request-size-not-enabled", "savings": 0.0}
    cohort, selected = _cohort(
        unit,
        family="routing",
        candidate_id=str(meta.get("candidate_id") or meta["policy_id"]),
        canary_fraction=canary_fraction,
        holdout_fraction=holdout_fraction,
        salt=str(policy.get("salt") or "openai-optimization-draft-dry-run"),
    )
    current_cost = estimate_cost(unit["requested_model"], unit["input_tokens"], unit["output_tokens"], provider="openai") or 0.0
    target_cost = estimate_cost(target_model, unit["input_tokens"], unit["output_tokens"], provider="openai") or 0.0
    savings = max(0.0, current_cost - target_cost)
    if cohort == "holdout":
        meta.update({"status": "holdout", "cohort": "canary_holdout", "reason": "selected-holdout"})
        return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "holdout", "eligible": True, "reason": "selected-holdout", "savings": savings}
    meta.update({"status": "applied" if selected else "not_selected", "cohort": "canary_applied" if selected else "skipped", "reason": "selected-canary" if selected else "outside-canary-fraction", "actual_forwarded_model": target_model if selected else unit["requested_model"], "projected_input_savings_usd": savings})
    return {"openai_canary": meta, "requested_model": unit["requested_model"], "routed_model": target_model if selected else unit["requested_model"], "category": unit["category"], "text_chars": unit["text_chars"], "has_tools": unit["has_tools"], "stream": unit["stream"]}, {"family": "routing", "status": "applied_if_enabled" if selected else "not_selected", "eligible": True, "reason": meta["reason"], "savings": savings if selected else 0.0}


def _summary_candidate(unit: dict[str, Any], policy: dict[str, Any] | None, action: dict[str, Any] | None, *, canary_fraction: float, holdout_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(policy, dict):
        return {}, {"family": "old_context_summary", "status": "unsupported", "eligible": False, "reason": "missing-summary-policy", "savings": 0.0}
    meta = {
        "enabled": True,
        "policy_source": "managed-recommended",
        "rule_id": policy.get("rule_id") or "openai-optimization-draft-summary",
        "candidate_id": policy.get("candidate_id"),
        "status": "not_evaluated",
        "reason_codes": [],
    }
    reason = _evidence_blocker(action, policy)
    if reason:
        meta.update({"status": "safety_stopped" if reason == "stale-evidence" else "skipped", "reason_codes": [reason], "applied": False})
        return {"old_context_summarization": meta}, {"family": "old_context_summary", "status": reason, "eligible": True, "reason": reason, "savings": 0.0}
    excluded = set(_values(policy.get("excluded_categories"))) or {"tool-heavy", "tool-result"}
    if unit["category"] in excluded:
        meta.update({"status": "skipped", "reason_codes": ["blocked_category"], "applied": False})
        return {"old_context_summarization": meta}, {"family": "old_context_summary", "status": "no_match", "eligible": False, "reason": "blocked_category", "savings": 0.0}
    min_chars = _as_int(policy.get("min_request_chars") or policy.get("min_source_chars"), 32000)
    if unit["text_chars"] < min_chars:
        meta.update({"status": "skipped", "reason_codes": ["request_below_min_chars"], "applied": False})
        return {"old_context_summarization": meta}, {"family": "old_context_summary", "status": "no_match", "eligible": False, "reason": "request_below_min_chars", "savings": 0.0}
    cohort, selected = _cohort(
        unit,
        family="old_context_summary",
        candidate_id=str(meta.get("candidate_id") or meta["rule_id"]),
        canary_fraction=canary_fraction,
        holdout_fraction=holdout_fraction,
        salt=str((policy.get("canary") or {}).get("salt") if isinstance(policy.get("canary"), dict) else "openai-optimization-draft-dry-run"),
    )
    saved_tokens = max(0, int((unit["text_chars"] * (1.0 - _as_float(policy.get("summary_compression_ratio"), 0.125))) // 4))
    basis = pricing_basis(unit["requested_model"], provider="openai")
    savings = (saved_tokens / 1_000_000.0) * _as_float(basis.get("input_usd_per_million"))
    if cohort == "holdout":
        meta.update({"status": "holdout", "reason_codes": ["holdout"], "applied": False, "canary": {"cohort": "canary_holdout"}})
        return {"old_context_summarization": meta}, {"family": "old_context_summary", "status": "holdout", "eligible": True, "reason": "holdout", "savings": savings}
    meta.update({"status": "applied" if selected else "not_evaluated", "reason_codes": ["summary-created" if selected else "outside-canary-fraction"], "applied": bool(selected), "canary": {"cohort": "canary_applied" if selected else "skipped"}, "tokens_saved_est": saved_tokens, "projected_net_savings_usd": round(savings, 8)})
    return {"old_context_summarization": meta}, {"family": "old_context_summary", "status": "applied_if_enabled" if selected else "not_selected", "eligible": True, "reason": meta["reason_codes"][0], "savings": savings if selected else 0.0}


def _cache_candidate(unit: dict[str, Any], rules: list[dict[str, Any]], action: dict[str, Any] | None, *, canary_fraction: float, holdout_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rules:
        return {}, {"family": "cache_replay", "status": "unsupported", "eligible": False, "reason": "missing-cache-rule", "savings": 0.0}
    feature_hashes = set(_values(unit["cache"].get("pattern_hashes"))) | set(_values(unit["routing"].get("pattern_hashes")))
    for key in ("pattern_hash", "pattern_hashes"):
        feature_hashes.update(_values(unit["cache"].get(key)))
    for rule in rules:
        conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        expected = set(_values(conditions.get("pattern_hash"))) | set(_values(conditions.get("pattern_hashes")))
        if expected and not feature_hashes.intersection(expected):
            continue
        reason = _evidence_blocker(action, rule)
        rule_id = str(rule.get("id") or "openai-optimization-draft-cache")
        candidate_id = str(rule.get("candidate_id") or rule_id)
        pattern_rule = {"rule_id": rule_id, "candidate_id": candidate_id, "policy_source": "managed-recommended"}
        replay = {"rule_id": rule_id, "candidate_id": candidate_id, "policy_source": "managed-recommended"}
        if reason:
            replay.update({"status": "safety_stopped" if reason == "stale-evidence" else "bypassed", "reason": reason})
            return {"status": "skipped", "reason": reason, "pattern_rule": pattern_rule, "cache_replay_canary": replay}, {"family": "cache_replay", "status": reason, "eligible": True, "reason": reason, "savings": 0.0}
        action_payload = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        if unit["stream"] and not bool(action_payload.get("streaming")):
            replay.update({"status": "bypassed", "reason": "unsupported-streaming-shape"})
            return {"status": "skipped", "reason": "unsupported-streaming-shape", "pattern_rule": pattern_rule, "cache_replay_canary": replay}, {"family": "cache_replay", "status": "unsupported", "eligible": False, "reason": "unsupported-streaming-shape", "savings": 0.0}
        if unit["has_tools"] and not bool(action_payload.get("safe_invalidation")):
            replay.update({"status": "bypassed", "reason": "cache-replay-invalidation-missing"})
            return {"status": "skipped", "reason": "cache-replay-invalidation-missing", "pattern_rule": pattern_rule, "cache_replay_canary": replay}, {"family": "cache_replay", "status": "unsupported", "eligible": False, "reason": "cache-replay-invalidation-missing", "savings": 0.0}
        cohort, selected = _cohort(
            unit,
            family="cache_replay",
            candidate_id=candidate_id,
            canary_fraction=canary_fraction,
            holdout_fraction=holdout_fraction,
            salt=str((rule.get("canary") or {}).get("salt") if isinstance(rule.get("canary"), dict) else "openai-optimization-draft-dry-run"),
        )
        if cohort == "holdout":
            replay.update({"status": "holdout", "reason": "canary_holdout", "canary_cohort": "canary_holdout"})
            return {"status": "skipped", "reason": "canary_holdout", "pattern_rule": pattern_rule, "cache_replay_canary": replay}, {"family": "cache_replay", "status": "holdout", "eligible": True, "reason": "canary_holdout", "savings": unit["cost_est_usd"]}
        replay.update({"status": "applied" if selected else "bypassed", "reason": "no-dependency-required" if selected else "outside-canary-fraction", "canary_cohort": "canary_applied" if selected else "skipped"})
        return {"status": "hit" if selected else "skipped", "reason": "dry-run-cache-replay" if selected else "outside-canary-fraction", "pattern_rule": pattern_rule, "cache_replay_canary": replay}, {"family": "cache_replay", "status": "applied_if_enabled" if selected else "not_selected", "eligible": True, "reason": replay["reason"], "savings": unit["cost_est_usd"] if selected else 0.0}
    return {}, {"family": "cache_replay", "status": "no_match", "eligible": False, "reason": "pattern-hash-mismatch", "savings": 0.0}


def _empty_family(family: str) -> dict[str, Any]:
    return {
        "action_family": FAMILY_TO_ACTION[family],
        "eligible": 0,
        "applied_if_enabled": 0,
        "holdout": 0,
        "suppressed": 0,
        "conflict": 0,
        "safety_stop": 0,
        "unsupported": 0,
        "stale_evidence": 0,
        "missing_evidence": 0,
        "no_match": 0,
        "expected_net_savings_usd": 0.0,
        "reason_counts": {},
    }


def _add_family_count(families: dict[str, dict[str, Any]], family: str, status: str, reason: str, savings: float) -> None:
    row = families.setdefault(family, _empty_family(family))
    if status == "applied_if_enabled":
        row["eligible"] += 1
        row["applied_if_enabled"] += 1
    elif status == "holdout":
        row["eligible"] += 1
        row["holdout"] += 1
    elif status == "unsupported":
        row["unsupported"] += 1
    elif status == "stale-evidence":
        row["eligible"] += 1
        row["safety_stop"] += 1
        row["stale_evidence"] += 1
    elif status == "missing-evidence":
        row["eligible"] += 1
        row["missing_evidence"] += 1
    elif status == "no_match":
        row["no_match"] += 1
    elif status == "not_selected":
        row["eligible"] += 1
    row["expected_net_savings_usd"] = round(float(row["expected_net_savings_usd"]) + max(0.0, float(savings or 0.0)), 8)
    reasons = row["reason_counts"]
    reasons[reason or status or "unknown"] = int(reasons.get(reason or status or "unknown", 0)) + 1


def _example(unit: dict[str, Any], family: str, status: str, reason: str, governor: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_hash": unit["row_hash"],
        "action_family": FAMILY_TO_ACTION[family],
        "endpoint": unit["endpoint"],
        "source_surface": unit["source_surface"],
        "requested_model_family": unit["requested_model_family"],
        "category": unit["category"],
        "workflow_phase": unit["workflow_phase"],
        "text_bucket": unit["text_bucket"],
        "stream": bool(unit["stream"]),
        "has_tools": bool(unit["has_tools"]),
        "status_bucket": _status_bucket(unit.get("status_code")),
        "decision_status": status,
        "reason": reason,
        "governor_selected_action_family": governor.get("selected_action_family"),
        "raw_fields_included": False,
    }


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


def _error_result(draft: str, error: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ok": False,
        "dry_run": True,
        "draft_id": draft,
        "generated_at": utc_now(),
        "summary": {"openai_rows_considered": 0},
        "families": {},
        "decision_examples": [],
        "feedback": None,
        "privacy": PRIVACY,
        "error": error,
    }


def _feedback_event(result: dict[str, Any]) -> dict[str, Any]:
    families = result.get("families") if isinstance(result.get("families"), dict) else {}
    return {
        "schema": LIFECYCLE_SCHEMA,
        "event_type": "openai_optimization_draft_dry_run",
        "occurred_at": result.get("generated_at"),
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "lifecycle_phase": "dry_run",
        "draft_id": _public_id(result.get("draft_id")),
        "summary": result.get("summary"),
        "family_results": [
            {
                "action_family": row.get("action_family"),
                "eligible": row.get("eligible"),
                "applied_if_enabled": row.get("applied_if_enabled"),
                "holdout": row.get("holdout"),
                "suppressed": row.get("suppressed"),
                "conflict": row.get("conflict"),
                "safety_stop": row.get("safety_stop"),
                "unsupported": row.get("unsupported"),
                "stale_evidence": row.get("stale_evidence"),
                "missing_evidence": row.get("missing_evidence"),
                "expected_net_savings_usd": row.get("expected_net_savings_usd"),
            }
            for row in families.values()
            if isinstance(row, dict)
        ],
        "privacy": PRIVACY,
    }


async def queue_openai_optimization_draft_dry_run_feedback(store_obj: Any, result: dict[str, Any]) -> dict[str, Any]:
    from agentflow_proxy.recommendations import queue_policy_event_feedback

    meta = await queue_policy_event_feedback(
        store_obj,
        _feedback_event(result),
        source_surface=LIFECYCLE_SOURCE_SURFACE,
        queue_when_disabled=True,
        flush_immediately=False,
    )
    return {
        "schema": "agentflow.openai_optimization_draft_dry_run_feedback_queue.v1",
        "source_surface": LIFECYCLE_SOURCE_SURFACE,
        "status": meta.get("status"),
        "reason": meta.get("reason"),
        "endpoint": meta.get("endpoint"),
        "queue_id": meta.get("queue_id"),
        "attempts": meta.get("attempts"),
        "payload_included": False,
    }


async def dry_run_openai_optimization_draft(
    draft: str,
    *,
    workspace: str | Path | None = None,
    store_obj: Any | None = None,
    limit: int = 1000,
    canary_fraction: float = 1.0,
    holdout_fraction: float = 0.0,
    queue_feedback: bool = False,
) -> dict[str, Any]:
    loaded = load_staged_policy_draft(draft, workspace=workspace)
    if not loaded.get("ok"):
        return _error_result(draft, loaded.get("error") if isinstance(loaded.get("error"), dict) else {"type": "draft_not_found", "message": "staged draft could not be loaded"})
    bundle = loaded["bundle"]
    validation = validate_policy_bundle(bundle)
    raw_errors: list[dict[str, str]] = []
    _scan_raw_fields(bundle, raw_errors)
    if not validation.get("ok") or raw_errors:
        return _error_result(
            draft,
            {
                "type": "validation_failed",
                "message": "staged OpenAI optimization draft is not safe to dry-run",
                "errors": [*(validation.get("errors") or []), *raw_errors],
            },
        )
    metadata = (loaded.get("manifest") or {}).get("metadata") if isinstance((loaded.get("manifest") or {}).get("metadata"), dict) else {}
    if "openai_optimization_review" not in metadata:
        return _error_result(draft, {"type": "not_openai_optimization_draft", "message": "staged draft does not include OpenAI optimization review metadata"})
    if store_obj is None:
        return _error_result(draft, {"type": "store_required", "message": "OpenAI optimization draft dry-run requires a local metadata store"})

    actions = _actions_from_manifest(loaded.get("manifest") if isinstance(loaded.get("manifest"), dict) else None)
    route_policy = _route_policy(bundle)
    summary_policy = _summary_policy(bundle)
    cache_rules = _cache_rules(bundle)
    rows = [_row_unit(row) for row in _read_rows(store_obj, limit)]
    families = {family: _empty_family(family) for family in ("routing", "old_context_summary", "cache_replay")}
    examples: list[dict[str, Any]] = []
    governor_selected = Counter()
    governor_suppressed = Counter()

    previous_env = {
        "AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_CANARY_FRACTION": os.environ.get("AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_CANARY_FRACTION"),
        "AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_HOLDOUT_FRACTION": os.environ.get("AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_HOLDOUT_FRACTION"),
    }
    os.environ["AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_CANARY_FRACTION"] = str(max(0.0, min(1.0, canary_fraction)))
    os.environ["AGENTFLOW_OPENAI_OPTIMIZATION_GOVERNOR_HOLDOUT_FRACTION"] = str(max(0.0, min(1.0, holdout_fraction)))
    try:
        for unit in rows:
            routing_meta, routing_decision = _routing_candidate(unit, route_policy, actions.get("routing"), canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
            crunch_meta, summary_decision = _summary_candidate(unit, summary_policy, actions.get("old_context_summary"), canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
            cache_meta, cache_decision = _cache_candidate(unit, cache_rules, actions.get("cache_replay"), canary_fraction=canary_fraction, holdout_fraction=holdout_fraction)
            governor = build_openai_optimization_governor(
                routing_meta=routing_meta,
                crunch_meta=crunch_meta,
                cache_meta=cache_meta,
                path=unit["path"],
                requested_model=unit["requested_model"],
                category=unit["category"],
                stream=unit["stream"],
                session_id=None,
            )
            governor_selected[str(governor.get("selected_action_family") or "none")] += 1
            suppressed = governor.get("suppressed_families") if isinstance(governor.get("suppressed_families"), list) else []
            suppressed_by_family = {item.get("family"): item for item in suppressed if isinstance(item, dict)}
            for family, decision in (
                ("routing", routing_decision),
                ("old_context_summary", summary_decision),
                ("cache_replay", cache_decision),
            ):
                status = str(decision.get("status") or "unknown")
                reason = str(decision.get("reason") or status)
                _add_family_count(families, family, status, reason, _as_float(decision.get("savings")))
                suppressed_item = suppressed_by_family.get(family)
                if isinstance(suppressed_item, dict):
                    families[family]["suppressed"] += 1
                    reason_codes = [str(code) for code in suppressed_item.get("reason_codes") or []]
                    for code in reason_codes:
                        families[family]["reason_counts"][code] = int(families[family]["reason_counts"].get(code, 0)) + 1
                    if "conflicts-with-selected-family" in reason_codes:
                        families[family]["conflict"] += 1
                        governor_suppressed[family] += 1
                if len(examples) < 12 and (status == "applied_if_enabled" or suppressed_item or status in {"holdout", "stale-evidence", "missing-evidence", "unsupported"}):
                    examples.append(_example(unit, family, status, reason, governor))
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    for row in families.values():
        row["reason_counts"] = dict(sorted(row["reason_counts"].items(), key=lambda item: (-item[1], item[0])))
    summary = {
        "openai_rows_considered": len(rows),
        "staged_action_count": len(actions),
        "candidate_action_families": sorted(FAMILY_TO_ACTION[family] for family, row in families.items() if row["eligible"] or row["unsupported"] or row["no_match"]),
        "governor_selected_counts": dict(sorted(governor_selected.items())),
        "governor_suppressed_counts": dict(sorted(governor_suppressed.items())),
        "eligible_total": sum(_as_int(row.get("eligible")) for row in families.values()),
        "applied_if_enabled_total": sum(_as_int(row.get("applied_if_enabled")) for row in families.values()),
        "holdout_total": sum(_as_int(row.get("holdout")) for row in families.values()),
        "suppressed_total": sum(_as_int(row.get("suppressed")) for row in families.values()),
        "conflict_total": sum(_as_int(row.get("conflict")) for row in families.values()),
        "expected_net_savings_usd": round(sum(_as_float(row.get("expected_net_savings_usd")) for row in families.values()), 8),
        "active_policy_files_written": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
    }
    result = {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "draft_id": (loaded.get("manifest") or {}).get("draft_id") or draft,
        "generated_at": utc_now(),
        "draft": {
            "workspace": loaded.get("workspace"),
            "manifest_path": loaded.get("manifest_path"),
            "bundle_path": loaded.get("bundle_path"),
            "openai_optimization_review": metadata.get("openai_optimization_review"),
        },
        "summary": summary,
        "families": families,
        "decision_examples": examples,
        "feedback": None,
        "privacy": PRIVACY,
        "error": None,
    }
    if queue_feedback:
        result["feedback"] = await queue_openai_optimization_draft_dry_run_feedback(store_obj, result)
    return result
