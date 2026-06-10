from __future__ import annotations

from typing import Any

from agentflow_proxy.session_phase_memory import build_session_phase_memory_for_session

SCHEMA = "agentflow.session_memory_optimization_hints.v1"
TOKEN_CHARS = 4
SAFE_MEMORY_BLOCKERS = {"context_plateau_active"}
UNSAFE_MEMORY_BLOCKERS = {
    "recent_errors",
    "recent_retries",
    "recent_routing_fallback",
    "thinking_phase_present",
    "small_sample",
}


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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return default


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, list):
        return {str(item).lower() for item in value}
    return set()


def _savings_bucket(tokens: int) -> str:
    if tokens <= 0:
        return "none"
    if tokens < 1_000:
        return "lt_1k_tokens"
    if tokens < 10_000:
        return "1k_10k_tokens"
    if tokens < 100_000:
        return "10k_100k_tokens"
    return "gte_100k_tokens"


def _memory_public_view(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {
            "available": False,
            "raw_session_id_included": False,
        }
    plateau = memory.get("context_plateau") if isinstance(memory.get("context_plateau"), dict) else {}
    window = memory.get("window") if isinstance(memory.get("window"), dict) else {}
    return {
        "available": True,
        "session_key": memory.get("session_key"),
        "session_key_kind": memory.get("session_key_kind"),
        "raw_session_id_included": False,
        "dominant_phase": memory.get("dominant_phase"),
        "classifications": list(memory.get("classifications") or []),
        "context_plateau": {
            "active": bool(plateau.get("active")),
            "pairs": _as_int(plateau.get("pairs")),
            "min_text_chars": _as_int(plateau.get("min_text_chars")),
            "max_delta_ratio": _as_float(plateau.get("max_delta_ratio")),
        },
        "window": {
            "call_count": _as_int(window.get("call_count")),
            "window_size": _as_int(window.get("window_size")),
        },
        "error_rate": _as_float(memory.get("error_rate")),
        "blocker_reasons": list(memory.get("blocker_reasons") or []),
        "text_bucket_counts": list(memory.get("text_bucket_counts") or []),
        "cache_status_counts": list(memory.get("cache_status_counts") or []),
    }


def _current_thinking_present(*, current_thinking: bool, routing_meta: dict[str, Any] | None) -> bool:
    if current_thinking:
        return True
    routing = routing_meta if isinstance(routing_meta, dict) else {}
    text = " ".join(str(routing.get(key) or "") for key in ("reason", "workflow_phase", "category"))
    return "thinking" in text.lower()


def _base_blockers(
    *,
    memory: dict[str, Any] | None,
    policy: dict[str, Any],
    category: str | None,
    has_tool_blocks: bool,
    current_thinking: bool,
    routing_meta: dict[str, Any] | None,
    text_chars: int,
) -> list[str]:
    blockers: list[str] = []
    if not isinstance(memory, dict):
        return ["no_session_memory"]
    plateau = memory.get("context_plateau") if isinstance(memory.get("context_plateau"), dict) else {}
    window = memory.get("window") if isinstance(memory.get("window"), dict) else {}
    plateau_pairs = _as_int(plateau.get("pairs"))
    if not bool(plateau.get("active")):
        blockers.append("no_context_plateau")
    if plateau_pairs < _as_int(policy.get("min_plateau_pairs"), 3):
        blockers.append("plateau_count_below_threshold")
    if _as_int(window.get("call_count")) < _as_int(policy.get("min_call_count"), 4):
        blockers.append("recent_call_count_below_threshold")
    if text_chars < _as_int(policy.get("min_text_chars"), 8000):
        blockers.append("request_too_small")
    allowed_phases = _string_set(policy.get("allowed_phases"))
    phase = str(memory.get("dominant_phase") or "").lower()
    if allowed_phases and phase not in allowed_phases:
        blockers.append("phase_not_allowed")
    if _as_float(memory.get("error_rate")) > _as_float(policy.get("max_error_rate"), 0.0):
        blockers.append("recent_error_rate_exceeded")
    for reason in memory.get("blocker_reasons") or []:
        if reason in UNSAFE_MEMORY_BLOCKERS:
            blockers.append(str(reason))
    if _as_bool(policy.get("block_tool_results"), True) and str(category or "").lower() == "tool-result":
        blockers.append("tool_result_state_dependence")
    if _as_bool(policy.get("block_thinking"), True) and _current_thinking_present(
        current_thinking=current_thinking,
        routing_meta=routing_meta,
    ):
        blockers.append("thinking_blocks_present")
    return sorted(set(blockers))


def _projected_tokens(text_chars: int, plateau_pairs: int, ratio: float) -> int:
    bounded_ratio = min(0.50, max(0.0, ratio))
    plateau_factor = min(1.0, max(0.25, plateau_pairs / 10.0))
    return max(0, int((max(0, text_chars) // TOKEN_CHARS) * bounded_ratio * plateau_factor))


def build_session_memory_optimization_hints(
    *,
    store_obj: Any,
    session_id: str | None,
    stream: bool,
    has_tool_blocks: bool,
    category: str | None,
    text_chars: int,
    routing_meta: dict[str, Any] | None,
    crunch_policy: dict[str, Any],
    crunch_policy_source: str,
    crunch_rule_path: str,
    cache_policy: dict[str, Any],
    cache_policy_source: str,
    cache_rule_path: str,
    safe_invalidation_evidence: bool = False,
    reviewed_cache_pattern_rule: bool = False,
    current_thinking: bool = False,
) -> dict[str, Any]:
    session_id = str(session_id or "").strip()
    memory = build_session_phase_memory_for_session(store_obj, session_id) if session_id else None
    memory_view = _memory_public_view(memory)
    crunch_hint_policy = (
        crunch_policy.get("session_memory_hints")
        if isinstance(crunch_policy.get("session_memory_hints"), dict)
        else {}
    )
    cache_hint_policy = (
        cache_policy.get("session_memory_hints")
        if isinstance(cache_policy.get("session_memory_hints"), dict)
        else {}
    )
    plateau_pairs = _as_int((memory or {}).get("context_plateau", {}).get("pairs") if isinstance(memory, dict) else 0)

    crunch_blockers = _base_blockers(
        memory=memory,
        policy=crunch_hint_policy,
        category=category,
        has_tool_blocks=has_tool_blocks,
        current_thinking=current_thinking,
        routing_meta=routing_meta,
        text_chars=text_chars,
    )
    crunch_enabled = _as_bool(crunch_hint_policy.get("enabled"), False)
    crunch_tokens = _projected_tokens(
        text_chars,
        plateau_pairs,
        _as_float(crunch_hint_policy.get("projected_savings_ratio"), 0.10),
    )
    crunch_status = "eligible" if crunch_enabled and not crunch_blockers else "blocked"
    crunch_reason = "session-plateau-eligible" if crunch_status == "eligible" else "session-plateau-blocked"
    if not crunch_enabled:
        crunch_status = "skipped"
        crunch_reason = "policy-disabled"
    crunch_hint = {
        "schema": SCHEMA,
        "enabled": crunch_enabled,
        "status": crunch_status,
        "reason": crunch_reason,
        "rule_id": str(crunch_hint_policy.get("rule_id") or "local-session-plateau-crunch-hint"),
        "policy_source": crunch_policy_source,
        "rule_path": crunch_rule_path,
        "memory": memory_view,
        "blockers": crunch_blockers,
        "crunch_profile": str(crunch_hint_policy.get("crunch_profile") or "plateau-repeated-context-review"),
        "old_context_summary_canary_candidate": bool(
            crunch_enabled
            and not crunch_blockers
            and _as_bool(crunch_hint_policy.get("old_context_summary_canary"), False)
        ),
        "projected_tokens_saved_est": crunch_tokens if crunch_enabled and not crunch_blockers else 0,
        "projected_savings_bucket": _savings_bucket(crunch_tokens if crunch_enabled and not crunch_blockers else 0),
        "mutation_applied": False,
        "request_mutation": "none",
        "dry_run_projection": {
            "eligible": bool(crunch_enabled and not crunch_blockers),
            "profile": str(crunch_hint_policy.get("crunch_profile") or "plateau-repeated-context-review"),
            "requires_review_before_mutation": True,
        },
    }

    cache_blockers = _base_blockers(
        memory=memory,
        policy=cache_hint_policy,
        category=category,
        has_tool_blocks=has_tool_blocks,
        current_thinking=current_thinking,
        routing_meta=routing_meta,
        text_chars=text_chars,
    )
    cache_enabled = _as_bool(cache_hint_policy.get("enabled"), False)
    if has_tool_blocks and not _as_bool(cache_hint_policy.get("allow_tool_calls"), False):
        cache_blockers.append("tool_call_cache_disabled")
    if stream and not _as_bool(cache_hint_policy.get("allow_streaming_replay"), False):
        cache_blockers.append("streaming_replay_reviewed_rule_required")
    if _as_bool(cache_hint_policy.get("require_safe_invalidation"), True) and not safe_invalidation_evidence:
        cache_blockers.append("missing_invalidation_evidence")
    if _as_bool(cache_hint_policy.get("require_reviewed_pattern_rule"), True) and not reviewed_cache_pattern_rule:
        cache_blockers.append("reviewed_pattern_rule_required")
    cache_blockers = sorted(set(cache_blockers))
    cache_status = "eligible" if cache_enabled and not cache_blockers else "blocked"
    cache_reason = "session-plateau-dry-run-eligible" if cache_status == "eligible" else "session-plateau-blocked"
    if not cache_enabled:
        cache_status = "skipped"
        cache_reason = "policy-disabled"
    cache_hint = {
        "schema": SCHEMA,
        "enabled": cache_enabled,
        "status": cache_status,
        "reason": cache_reason,
        "rule_id": str(cache_hint_policy.get("rule_id") or "local-session-plateau-cache-hint"),
        "policy_source": cache_policy_source,
        "rule_path": cache_rule_path,
        "memory": memory_view,
        "blockers": cache_blockers,
        "cacheability_hint": "dry-run-exact-replay-group" if cache_enabled and not cache_blockers else "not-cacheable",
        "replayability_level": "local-exact-response-dry-run" if cache_enabled and not cache_blockers else "features_only",
        "cache_mutation": False,
        "request_mutation": "none",
        "dry_run_projection": {
            "eligible": bool(cache_enabled and not cache_blockers),
            "exact_replay_grouping_candidate": bool(cache_enabled and not cache_blockers),
            "safe_invalidation_evidence": bool(safe_invalidation_evidence),
            "reviewed_pattern_rule": bool(reviewed_cache_pattern_rule),
            "requires_reviewed_pattern_rule": _as_bool(cache_hint_policy.get("require_reviewed_pattern_rule"), True),
            "requires_safe_invalidation": _as_bool(cache_hint_policy.get("require_safe_invalidation"), True),
        },
    }

    return {
        "schema": SCHEMA,
        "crunch": crunch_hint,
        "cache": cache_hint,
        "privacy": {
            "metadata_only": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "raw_session_ids_included": False,
            "session_ids_hashed": True,
            "request_json_read": False,
            "response_json_read": False,
            "cache_mutation": False,
            "request_mutation": False,
        },
    }
