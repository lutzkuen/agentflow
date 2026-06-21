from __future__ import annotations

import hashlib
import json
from typing import Any

from tokenclaw.anthropic_thinking_compaction_report import (
    _as_float,
    _as_int,
    _breakdown,
    _count_bucket,
    _endpoint,
    _json_obj,
    _model_family,
    _source_surface,
    _text_bucket,
    _token_bucket,
)
from tokenclaw.crunch import TOKEN_CHARS, anthropic_thinking_compaction_effective_policy, normalize_text
from tokenclaw.pricing import estimate_blended_input_savings
from tokenclaw.public_metadata import public_label
from tokenclaw.store import stable_json, utc_now


SCHEMA = "tokenclaw.anthropic_thinking_compaction_dry_run.v1"
PLAN_SCHEMA = "tokenclaw.anthropic_thinking_compaction_plan.v1"
ACTION_FAMILY = "anthropic_thinking_history_compaction"
_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def _staged_policy_metadata() -> dict[str, Any]:
    policy = anthropic_thinking_compaction_effective_policy()
    rules = [
        {
            "rule_id": public_label(rule.get("rule_id"), "unknown"),
            "candidate_id": public_label(rule.get("candidate_id"), "none"),
            "enabled": bool(rule.get("enabled")),
            "policy_source": public_label(rule.get("policy_source"), "unknown"),
            "conditions": rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {},
            "min_text_chars": _as_int(rule.get("min_text_chars")),
            "min_block_chars": _as_int(rule.get("min_block_chars")),
            "similarity_threshold": _as_float(rule.get("similarity_threshold")),
            "canary": {
                "enabled": bool((rule.get("canary") or {}).get("enabled")) if isinstance(rule.get("canary"), dict) else False,
                "fraction": _as_float((rule.get("canary") or {}).get("fraction")) if isinstance(rule.get("canary"), dict) else 0.0,
                "holdout_fraction": _as_float((rule.get("canary") or {}).get("holdout_fraction")) if isinstance(rule.get("canary"), dict) else 0.0,
                "salt_included": False,
                "fingerprint_included": False,
            },
            "action": {
                "type": "compact_thinking_history_block",
                "preserve_tool_protocol": True,
                "preserve_assistant_text_fallback": True,
                "provider_calls_made": False,
                "managed_server_calls_made": False,
            },
            "privacy": {
                "metadata_only_output": True,
                "raw_thinking_text_included": False,
                "thinking_block_fingerprints_included": False,
                "raw_request_bodies_included": False,
                "request_ids_included": False,
                "session_ids_included": False,
                "file_paths_included": False,
            },
        }
        for rule in policy.get("rules") or []
        if isinstance(rule, dict)
    ]
    return {
        "schema": "tokenclaw.anthropic_thinking_compaction_staged_local_canary.v1",
        "enabled": bool(policy.get("enabled")),
        "runtime_mutation_enabled": bool(policy.get("enabled")),
        "default_apply": False,
        "policy_source": public_label(policy.get("policy_source"), "unknown"),
        "rule_path": public_label(policy.get("rule_path"), "unknown"),
        "rule_path_included": False,
        "configured_rule_count": len(rules),
        "rules": rules,
        "next_action": "enable-local-canary-fraction-after-dry-run-review" if rules else "stage-local-canary-rule",
        "lifecycle_metadata": {
            "emits_applied": True,
            "emits_holdout": True,
            "emits_safety_stop": True,
            "impact_report": "tokenclaw.anthropic_thinking_compaction_impact.v1",
        },
        "privacy": {
            "metadata_only_output": True,
            "raw_policy_file_contents_included": False,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprints_included": False,
            "raw_request_bodies_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
        },
    }


def _hash_basis(value: Any, *, length: int = 16) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _public_candidate_id(prefix: str, basis: dict[str, Any]) -> str:
    return f"{prefix}:{_hash_basis(basis, length=20)}"


def _message_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _thinking_text(block: dict[str, Any]) -> str:
    for key in ("thinking", "text", "data"):
        value = block.get(key)
        if isinstance(value, str):
            return value
    return ""


def _has_assistant_text_fallback(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, str) and block.strip():
            return True
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].strip():
            return True
    return False


def _tool_use_ids(message: Any) -> set[str]:
    ids: set[str] = set()
    for block in _message_blocks(message):
        if block.get("type") == "tool_use" and block.get("id"):
            ids.add(str(block["id"]))
    return ids


def _tool_result_ids(message: Any) -> set[str]:
    ids: set[str] = set()
    for block in _message_blocks(message):
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))
    return ids


def _has_tool_result(message: Any) -> bool:
    return any(block.get("type") == "tool_result" for block in _message_blocks(message))


def _top_level_thinking_active(body: dict[str, Any]) -> bool:
    thinking = body.get("thinking")
    if not thinking:
        return False
    if isinstance(thinking, dict) and str(thinking.get("type") or "").strip().lower() == "disabled":
        return False
    return True


def _assistant_age_bucket(age: int) -> str:
    if age <= 0:
        return "latest_assistant"
    if age == 1:
        return "previous_assistant"
    if age <= 5:
        return "assistant_age_2_5"
    return "assistant_age_gt_5"


def _block_size_bucket(chars: int) -> str:
    if chars <= 0:
        return "0_chars"
    if chars < 2_000:
        return "lt_2k_chars"
    if chars < 8_000:
        return "2k_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _shingles(text: str, n: int = 4) -> frozenset[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return frozenset(words)
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cohort(candidate_id: str, *, local_salt: str | None, fraction: float, holdout_fraction: float) -> dict[str, Any]:
    fraction = min(max(float(fraction), 0.0), 1.0)
    holdout_fraction = min(max(float(holdout_fraction), 0.0), 1.0)
    if fraction + holdout_fraction > 1.0:
        fraction = max(0.0, 1.0 - holdout_fraction)
    digest = hashlib.sha256(
        stable_json(
            {
                "candidate_id": candidate_id,
                "salt": local_salt or "anthropic-thinking-compaction-dry-run",
                "unit": "thinking_block_local_fingerprint",
            }
        ).encode("utf-8")
    ).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if score < holdout_fraction:
        cohort = "holdout"
        selected = False
    elif score < holdout_fraction + fraction:
        cohort = "canary"
        selected = True
    else:
        cohort = "not_selected"
        selected = False
    return {
        "cohort": cohort,
        "selected": selected,
        "canary_fraction": fraction,
        "holdout_fraction": holdout_fraction,
        "cohort_key_hash": digest[:16],
        "cohort_score": round(score, 12),
        "cohort_basis": "local-only-salted-hidden-thinking-fingerprint",
        "salt_included": False,
        "fingerprint_included": False,
    }


def _row_basis(row: dict[str, Any], body: dict[str, Any] | None) -> dict[str, Any]:
    provider = str(row.get("provider") or "anthropic").lower()
    path = str(row.get("path") or "")
    routing = _json_obj(row.get("routing_json"))
    requested_model = row.get("requested_model") or (body or {}).get("model")
    routed_model = row.get("routed_model") or requested_model
    category = str(row.get("category") or routing.get("category") or "unknown")
    source_surface = str(row.get("source_surface") or routing.get("source_surface") or _source_surface(provider, path))
    endpoint = str(row.get("endpoint") or routing.get("endpoint") or _endpoint(provider, path))
    actual_input_tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    cache_creation_tokens = _as_int(row.get("cache_creation_input_tokens"))
    cache_read_tokens = _as_int(row.get("cache_read_input_tokens"))
    text_chars = _as_int(routing.get("text_chars")) or (actual_input_tokens + cache_creation_tokens + cache_read_tokens) * TOKEN_CHARS
    return {
        "provider": provider,
        "path": path,
        "source_surface": source_surface,
        "endpoint": endpoint,
        "category": category,
        "workflow_phase": str(routing.get("workflow_phase") or routing.get("phase") or category or "unknown"),
        "routing_reason": str(routing.get("reason") or "unknown"),
        "requested_model": requested_model,
        "routed_model": routed_model,
        "requested_model_family": str(row.get("requested_model_family") or _model_family(requested_model)),
        "routed_model_family": str(row.get("routed_model_family") or _model_family(routed_model)),
        "stream": bool(_as_int(row.get("stream"))),
        "status_code": _as_int(row.get("status_code")),
        "actual_input_tokens": actual_input_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "text_chars": text_chars,
        "text_bucket": _text_bucket(text_chars),
    }


def _base_blockers(row: dict[str, Any], basis: dict[str, Any], body: dict[str, Any] | None, *, min_text_chars: int) -> set[str]:
    blockers: set[str] = set()
    if basis["provider"] != "anthropic":
        blockers.add("non-anthropic-provider")
    if basis["source_surface"] != "anthropic_messages":
        blockers.add("non-anthropic-source-surface")
    if basis["endpoint"] != "messages":
        blockers.add("non-anthropic-messages-endpoint")
    if basis["category"] != "tool-result":
        blockers.add("non-tool-result-category")
    if basis["text_chars"] < min_text_chars:
        blockers.add("below-min-text-size")
    if basis["status_code"] >= 400:
        blockers.add("error-response")
    if body is None:
        blockers.add("request-body-unavailable")
        blockers.add("metadata-insufficient")
    elif _top_level_thinking_active(body):
        blockers.add("active-top-level-thinking-request")
    return blockers


def _source_thinking_diagnosis(meta: dict[str, Any]) -> dict[str, Any]:
    diagnosis = meta.get("diagnosis") if isinstance(meta.get("diagnosis"), dict) else {}
    fallback = (
        diagnosis.get("old_context_summarization_fallback")
        if isinstance(diagnosis.get("old_context_summarization_fallback"), dict)
        else {}
    )
    return {
        "schema": public_label(diagnosis.get("schema"), "unknown"),
        "status": public_label(diagnosis.get("status"), "unknown"),
        "reason": public_label(diagnosis.get("reason"), "unknown"),
        "recommended_strategy": public_label(diagnosis.get("recommended_strategy"), "unknown"),
        "route_crunch_mismatch_explained": bool(diagnosis.get("route_crunch_mismatch_explained")),
        "old_context_summarization_fallback": {
            "recommended": bool(fallback.get("recommended")),
            "reason": public_label(fallback.get("reason"), "unknown"),
            "policy_enabled": bool(fallback.get("policy_enabled")),
        },
    }


def _source_thinking_metadata(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    meta = crunch.get("anthropic_thinking_history") if isinstance(crunch.get("anthropic_thinking_history"), dict) else {}
    if not meta:
        return {
            "available": False,
            "schema": None,
            "fingerprints_included": False,
            "raw_text_included": False,
        }
    return {
        "available": True,
        "schema": public_label(meta.get("schema"), "unknown"),
        "status": public_label(meta.get("status"), "unknown"),
        "reason": public_label(meta.get("reason"), "unknown"),
        "privacy_mode": public_label(meta.get("privacy_mode"), "unknown"),
        "body_available": bool(meta.get("body_available")),
        "thinking_block_count": _as_int(meta.get("thinking_block_count")),
        "redacted_thinking_block_count": _as_int(meta.get("redacted_thinking_block_count")),
        "top_level_thinking_active": bool(meta.get("top_level_thinking_active")),
        "thinking_signal_kind": public_label(meta.get("thinking_signal_kind"), "unknown"),
        "history_block_absence_reason": public_label(meta.get("history_block_absence_reason"), "none"),
        "route_crunch_mismatch_explained": bool(meta.get("route_crunch_mismatch_explained")),
        "diagnosis": _source_thinking_diagnosis(meta),
        "thinking_history_size_bucket": public_label(meta.get("thinking_history_size_bucket"), "unknown"),
        "thinking_block_count_bucket": public_label(meta.get("thinking_block_count_bucket"), "unknown"),
        "unique_local_thinking_block_fingerprint_count": _as_int(meta.get("unique_local_thinking_block_fingerprint_count")),
        "exact_duplicate_thinking_block_count": _as_int(meta.get("exact_duplicate_thinking_block_count")),
        "near_duplicate_thinking_block_count": _as_int(meta.get("near_duplicate_thinking_block_count")),
        "assistant_message_with_thinking_count": _as_int(meta.get("assistant_message_with_thinking_count")),
        "missing_assistant_text_fallback_count": _as_int(meta.get("missing_assistant_text_fallback_count")),
        "adjacent_tool_use_dependency_count": _as_int(meta.get("adjacent_tool_use_dependency_count")),
        "unsupported_content_block_shape_count": _as_int(meta.get("unsupported_content_block_shape_count")),
        "policy_source": public_label(meta.get("policy_source"), "unknown"),
        "rule_path_included": False,
        "fingerprints_included": False,
        "raw_text_included": False,
    }


def _thinking_blocks(body: dict[str, Any], *, min_block_chars: int, similarity_threshold: float) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    assistant_indexes = [idx for idx, msg in enumerate(messages) if isinstance(msg, dict) and msg.get("role") == "assistant"]
    assistant_age_by_index = {
        msg_idx: len(assistant_indexes) - 1 - order
        for order, msg_idx in enumerate(assistant_indexes)
    }
    entries: list[dict[str, Any]] = []
    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            entries.append({
                "message_index": msg_idx,
                "block_index": None,
                "type": "unsupported",
                "text": "",
                "blockers": {"unsupported-content-block-shape"},
                "assistant_age": assistant_age_by_index.get(msg_idx, 0),
                "assistant_text_fallback": False,
            })
            continue
        assistant_text_fallback = _has_assistant_text_fallback(message)
        tool_use_ids = _tool_use_ids(message)
        next_message = messages[msg_idx + 1] if msg_idx + 1 < len(messages) else None
        unresolved_tool_use = bool(tool_use_ids and not (isinstance(next_message, dict) and tool_use_ids <= _tool_result_ids(next_message)))
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") not in _THINKING_BLOCK_TYPES:
                continue
            text = _thinking_text(block)
            normalized = normalize_text(text)
            local_fingerprint = "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
            blockers: set[str] = set()
            if block.get("type") == "redacted_thinking":
                blockers.add("redacted-thinking-block")
            if block.get("type") == "thinking" and not text:
                blockers.add("unsupported-content-block-shape")
            if not assistant_text_fallback:
                blockers.add("missing-assistant-text-fallback")
            if msg_idx >= len(messages) - 1:
                blockers.add("latest-message-not-old-history")
            if unresolved_tool_use:
                blockers.add("unresolved-tool-use-dependency")
            chars = len(text)
            if chars < min_block_chars:
                blockers.add("thinking-block-below-min-chars")
            entries.append({
                "message_index": msg_idx,
                "block_index": block_index,
                "type": str(block.get("type") or "unknown"),
                "text": text,
                "chars": chars,
                "local_fingerprint": local_fingerprint,
                "shingles": _shingles(text) if chars >= min_block_chars else frozenset(),
                "assistant_age": assistant_age_by_index.get(msg_idx, 0),
                "assistant_text_fallback": assistant_text_fallback,
                "blockers": blockers,
            })
    for index, entry in enumerate(entries):
        if entry.get("type") != "thinking" or not entry.get("local_fingerprint"):
            entry["duplicate_kind"] = "none"
            continue
        duplicate_kind = "none"
        for newer in entries[index + 1 :]:
            if newer.get("type") != "thinking":
                continue
            if newer.get("local_fingerprint") and newer.get("local_fingerprint") == entry.get("local_fingerprint"):
                duplicate_kind = "exact"
                break
            if entry.get("shingles") and newer.get("shingles"):
                if _jaccard(entry["shingles"], newer["shingles"]) >= similarity_threshold:
                    duplicate_kind = "near"
                    break
        entry["duplicate_kind"] = duplicate_kind
        if duplicate_kind == "none":
            entry["blockers"].add("no-newer-duplicate-thinking-block")
    return entries


def _plan_for_block(
    *,
    row: dict[str, Any],
    basis: dict[str, Any],
    block: dict[str, Any],
    base_blockers: set[str],
    local_salt: str | None,
    canary_fraction: float,
    holdout_fraction: float,
) -> dict[str, Any]:
    candidate_basis = {
        "row": row.get("id"),
        "source_surface": basis["source_surface"],
        "category": basis["category"],
        "workflow_phase": basis["workflow_phase"],
        "text_bucket": basis["text_bucket"],
        "fingerprint": block.get("local_fingerprint"),
        "message_index": block.get("message_index"),
        "block_index": block.get("block_index"),
        "duplicate_kind": block.get("duplicate_kind"),
    }
    candidate_id = _public_candidate_id("anthropic-thinking-compaction-plan", candidate_basis)
    cohort = _cohort(
        candidate_id,
        local_salt=local_salt,
        fraction=canary_fraction,
        holdout_fraction=holdout_fraction,
    )
    projected_saved_chars = max(0, _as_int(block.get("chars")) - len("[thinking history compacted by AgentFlow dry-run]"))
    projected_saved_tokens = projected_saved_chars // TOKEN_CHARS
    projected_saved_usd = estimate_blended_input_savings(
        str(basis.get("routed_model") or basis.get("requested_model") or ""),
        tokens_saved=projected_saved_tokens,
        input_tokens=_as_int(basis.get("actual_input_tokens")),
        cache_read_tokens=_as_int(basis.get("cache_read_tokens")),
        provider="anthropic",
    ) or 0.0
    blockers = set(base_blockers)
    blockers.update(str(item) for item in block.get("blockers") or [])
    if block.get("duplicate_kind") not in {"exact", "near"}:
        blockers.add("no-newer-duplicate-thinking-block")
    if projected_saved_chars <= 0:
        blockers.add("no-thinking-compaction-savings-projected")
    if blockers:
        status = "blocked"
        reason = sorted(blockers)[0]
        saved_chars = 0
        saved_tokens = 0
        emitted_projected_chars = 0
        emitted_projected_tokens = 0
        emitted_projected_usd = 0.0
        no_op_reason = reason
    elif cohort["cohort"] == "holdout":
        status = "holdout"
        reason = "thinking-compaction-holdout"
        saved_chars = 0
        saved_tokens = 0
        emitted_projected_chars = projected_saved_chars
        emitted_projected_tokens = projected_saved_tokens
        emitted_projected_usd = projected_saved_usd
        no_op_reason = "canary-holdout-forward-original"
    elif not cohort["selected"]:
        status = "blocked"
        reason = "thinking-compaction-canary-not-selected"
        blockers.add(reason)
        saved_chars = 0
        saved_tokens = 0
        emitted_projected_chars = 0
        emitted_projected_tokens = 0
        emitted_projected_usd = 0.0
        no_op_reason = reason
    else:
        status = "planned"
        reason = "planned-thinking-history-compaction"
        saved_chars = projected_saved_chars
        saved_tokens = projected_saved_tokens
        emitted_projected_chars = projected_saved_chars
        emitted_projected_tokens = projected_saved_tokens
        emitted_projected_usd = projected_saved_usd
        no_op_reason = None
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": candidate_id,
        "candidate_id": candidate_id,
        "status": status,
        "action_family": ACTION_FAMILY,
        "action": {
            "type": "compact_thinking_history_block",
            "dry_run_only": True,
            "replacement_notice_included": False,
            "fallback_behavior": "forward_original_request_body",
        },
        "reason": reason,
        "no_op_reason": no_op_reason,
        "blockers": sorted(blockers),
        "provider": public_label(basis["provider"], "unknown"),
        "source_surface": public_label(basis["source_surface"], "unknown"),
        "endpoint": public_label(basis["endpoint"], "unknown"),
        "category": public_label(basis["category"], "unknown"),
        "workflow_phase": public_label(basis["workflow_phase"], "unknown"),
        "requested_model_family": public_label(basis["requested_model_family"], "unknown"),
        "routed_model_family": public_label(basis["routed_model_family"], "unknown"),
        "stream": bool(basis["stream"]),
        "text_bucket": basis["text_bucket"],
        "thinking_block": {
            "kind": "assistant_thinking_history",
            "duplicate_kind": str(block.get("duplicate_kind") or "none"),
            "assistant_age_bucket": _assistant_age_bucket(_as_int(block.get("assistant_age"))),
            "size_bucket": _block_size_bucket(_as_int(block.get("chars"))),
            "block_count_bucket": _count_bucket(1),
            "fingerprint_present": bool(block.get("local_fingerprint")),
            "fingerprint_included": False,
            "raw_text_included": False,
            "assistant_text_fallback_present": bool(block.get("assistant_text_fallback")),
        },
        "source_metadata": _source_thinking_metadata(row),
        "counts": {
            "before_chars": _as_int(block.get("chars")),
            "after_chars": max(0, _as_int(block.get("chars")) - saved_chars),
            "saved_chars": saved_chars,
            "saved_tokens_est": saved_tokens,
            "projected_before_chars": _as_int(block.get("chars")),
            "projected_after_chars": max(0, _as_int(block.get("chars")) - projected_saved_chars),
            "projected_saved_chars": emitted_projected_chars,
            "projected_saved_tokens_est": emitted_projected_tokens,
            "projected_saved_usd": round(float(emitted_projected_usd), 8),
        },
        "cohort": cohort,
        "fallback": {
            "strategy": "preserve-provider-request-body",
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "on_apply_error": "forward-original-request",
        },
        "mutation": {
            "dry_run_only": True,
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "policy_file_changed": False,
            "eligible_for_apply": status in {"planned", "holdout"},
            "would_change_request_body_if_applied": status in {"planned", "holdout"},
        },
        "privacy": {
            "metadata_only_output": True,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprint_included": False,
            "raw_prompt_text_included": False,
            "raw_messages_included": False,
            "raw_tool_payloads_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
    }


def _blocked_row_plan(row: dict[str, Any], basis: dict[str, Any], blockers: set[str]) -> dict[str, Any]:
    candidate_id = _public_candidate_id(
        "anthropic-thinking-compaction-plan",
        {
            "row": row.get("id"),
            "source_surface": basis["source_surface"],
            "category": basis["category"],
            "body": "unavailable-or-insufficient",
        },
    )
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": candidate_id,
        "candidate_id": candidate_id,
        "status": "blocked",
        "action_family": ACTION_FAMILY,
        "action": {
            "type": "compact_thinking_history_block",
            "dry_run_only": True,
            "replacement_notice_included": False,
            "fallback_behavior": "forward_original_request_body",
        },
        "reason": sorted(blockers)[0] if blockers else "metadata-insufficient",
        "blockers": sorted(blockers or {"metadata-insufficient"}),
        "provider": public_label(basis["provider"], "unknown"),
        "source_surface": public_label(basis["source_surface"], "unknown"),
        "endpoint": public_label(basis["endpoint"], "unknown"),
        "category": public_label(basis["category"], "unknown"),
        "workflow_phase": public_label(basis["workflow_phase"], "unknown"),
        "requested_model_family": public_label(basis["requested_model_family"], "unknown"),
        "routed_model_family": public_label(basis["routed_model_family"], "unknown"),
        "stream": bool(basis["stream"]),
        "text_bucket": basis["text_bucket"],
        "thinking_block": {
            "kind": "assistant_thinking_history",
            "duplicate_kind": "unknown",
            "assistant_age_bucket": "unknown",
            "size_bucket": "unknown",
            "block_count_bucket": "0",
            "fingerprint_present": False,
            "fingerprint_included": False,
            "raw_text_included": False,
            "assistant_text_fallback_present": False,
        },
        "source_metadata": _source_thinking_metadata(row),
        "counts": {
            "before_chars": 0,
            "after_chars": 0,
            "saved_chars": 0,
            "saved_tokens_est": 0,
            "projected_saved_usd": 0.0,
        },
        "cohort": {
            "cohort": "blocked",
            "selected": False,
            "canary_fraction": None,
            "holdout_fraction": None,
            "cohort_key_hash": None,
            "cohort_score": None,
            "cohort_basis": "not-applicable",
            "salt_included": False,
            "fingerprint_included": False,
        },
        "fallback": {
            "strategy": "preserve-provider-request-body",
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "on_apply_error": "forward-original-request",
        },
        "mutation": {
            "dry_run_only": True,
            "request_body_changed": False,
            "provider_call_made": False,
            "managed_server_call_made": False,
            "policy_file_changed": False,
        },
        "privacy": {
            "metadata_only_output": True,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprint_included": False,
            "raw_prompt_text_included": False,
            "raw_messages_included": False,
            "raw_tool_payloads_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "file_paths_included": False,
        },
    }


def build_anthropic_thinking_compaction_dry_run(
    store_obj: Any,
    *,
    limit: int = 500,
    examples: int = 50,
    min_text_chars: int = 8_000,
    min_block_chars: int = 2_000,
    similarity_threshold: float = 0.95,
    canary_fraction: float = 1.0,
    holdout_fraction: float = 0.0,
    local_salt: str | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 10_000))
    sample_limit = max(1, min(int(examples or 50), 500))
    text_floor = max(1, int(min_text_chars or 8_000))
    block_floor = max(1, int(min_block_chars or 2_000))
    similarity = min(max(float(similarity_threshold), 0.0), 1.0)
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select * from (
                select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                       source_surface, endpoint, requested_model, routed_model,
                       requested_model_family, routed_model_family, stream, status_code,
                       input_tokens_est, actual_input_tokens, actual_output_tokens,
                       cache_creation_input_tokens, cache_read_input_tokens,
                       thinking_output_tokens, cost_est_usd, cost_baseline_usd,
                       category, crunch_json, routing_json, cache_json, request_json, session_id
                from calls
                order by created_at desc
                limit ?
            ) recent_calls
            order by created_at asc
            """,
            (capped_limit,),
        ).fetchall()
    ]

    plans: list[dict[str, Any]] = []
    scanned_body_rows = 0
    bodyless_rows = 0
    for row in rows:
        body = _json_obj(row.get("request_json"))
        basis = _row_basis(row, body if body else None)
        blockers = _base_blockers(row, basis, body if body else None, min_text_chars=text_floor)
        if not body:
            bodyless_rows += 1
            if basis["provider"] == "anthropic" and (basis["category"] == "tool-result" or "thinking" in basis["routing_reason"]):
                plans.append(_blocked_row_plan(row, basis, blockers))
            continue
        scanned_body_rows += 1
        messages = body.get("messages")
        if not isinstance(messages, list):
            blockers.add("unsupported-content-block-shape")
            plans.append(_blocked_row_plan(row, basis, blockers))
            continue
        if basis["category"] == "tool-result" and not any(_has_tool_result(message) for message in messages if isinstance(message, dict)):
            blockers.add("tool-result-content-unverified")
        blocks = _thinking_blocks(body, min_block_chars=block_floor, similarity_threshold=similarity)
        if not blocks:
            row_blockers = set(blockers)
            row_blockers.add("no-thinking-history-blocks")
            plans.append(_blocked_row_plan(row, basis, row_blockers))
            continue
        for block in blocks:
            plans.append(
                _plan_for_block(
                    row=row,
                    basis=basis,
                    block=block,
                    base_blockers=blockers,
                    local_salt=local_salt,
                    canary_fraction=canary_fraction,
                    holdout_fraction=holdout_fraction,
                )
            )

    plans.sort(
        key=lambda item: (
            {"planned": 2, "holdout": 1}.get(str(item.get("status") or ""), 0),
            _as_float((item.get("counts") or {}).get("projected_saved_usd")),
            _as_int((item.get("counts") or {}).get("projected_saved_chars")),
        ),
        reverse=True,
    )
    status_counts: dict[str, int] = {}
    blocker_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    age_counts: dict[str, int] = {}
    size_counts: dict[str, int] = {}
    thinking_token_bucket_counts: dict[str, int] = {}
    for row in rows:
        bucket = _token_bucket(_as_int(row.get("thinking_output_tokens")))
        thinking_token_bucket_counts[bucket] = thinking_token_bucket_counts.get(bucket, 0) + 1
    for plan in plans:
        status = str(plan.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        block = plan.get("thinking_block") if isinstance(plan.get("thinking_block"), dict) else {}
        for counter, key in (
            (duplicate_counts, block.get("duplicate_kind") or "unknown"),
            (age_counts, block.get("assistant_age_bucket") or "unknown"),
            (size_counts, block.get("size_bucket") or "unknown"),
            (cohort_counts, (plan.get("cohort") or {}).get("cohort") or "unknown"),
        ):
            label = str(key)
            counter[label] = counter.get(label, 0) + 1
        for blocker in plan.get("blockers") or []:
            blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1
    planned = [plan for plan in plans if plan.get("status") == "planned"]
    holdout = [plan for plan in plans if plan.get("status") == "holdout"]
    eligible = planned + holdout
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "lookback_call_limit": capped_limit,
        "summary": {
            "scanned_call_count": len(rows),
            "body_on_row_count": scanned_body_rows,
            "body_off_row_count": bodyless_rows,
            "plan_count": len(plans),
            "planned_candidate_count": len(planned),
            "holdout_candidate_count": len(holdout),
            "eligible_candidate_count": len(eligible),
            "blocked_candidate_count": len(plans) - len(eligible),
            "applied_projected_saved_chars": sum(_as_int((plan.get("counts") or {}).get("saved_chars")) for plan in planned),
            "applied_projected_saved_tokens": sum(_as_int((plan.get("counts") or {}).get("saved_tokens_est")) for plan in planned),
            "applied_projected_saved_usd": round(
                sum(_as_float((plan.get("counts") or {}).get("projected_saved_usd")) for plan in planned),
                8,
            ),
            "holdout_projected_saved_chars": sum(_as_int((plan.get("counts") or {}).get("projected_saved_chars")) for plan in holdout),
            "holdout_projected_saved_tokens": sum(_as_int((plan.get("counts") or {}).get("projected_saved_tokens_est")) for plan in holdout),
            "holdout_projected_saved_usd": round(
                sum(_as_float((plan.get("counts") or {}).get("projected_saved_usd")) for plan in holdout),
                8,
            ),
            "projected_saved_chars": sum(_as_int((plan.get("counts") or {}).get("projected_saved_chars")) for plan in eligible),
            "projected_saved_tokens": sum(_as_int((plan.get("counts") or {}).get("projected_saved_tokens_est")) for plan in eligible),
            "projected_saved_usd": round(sum(_as_float((plan.get("counts") or {}).get("projected_saved_usd")) for plan in eligible), 8),
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "request_bodies_modified": False,
            "policy_files_changed": False,
        },
        "policy": {
            "schema": "tokenclaw.anthropic_thinking_compaction_dry_run_policy.v1",
            "enabled": True,
            "runtime_mutation_enabled": False,
            "default_apply": False,
            "review_only": True,
            "raw_policy_file_contents_included": False,
            "min_text_chars": text_floor,
            "min_block_chars": block_floor,
            "similarity_threshold": similarity,
            "candidate_rule": "older duplicate or near-duplicate assistant thinking blocks with assistant text fallback",
            "canary": {
                "fraction": min(max(float(canary_fraction), 0.0), 1.0),
                "holdout_fraction": min(max(float(holdout_fraction), 0.0), 1.0),
                "salt_included": False,
            },
            "savings_pricing": "cache-read blended input price for the routed/requested Anthropic model",
            "staged_local_canary": _staged_policy_metadata(),
        },
        "status_breakdown": _breakdown(status_counts, label="status"),
        "cohort_breakdown": _breakdown(cohort_counts, label="cohort"),
        "duplicate_kind_breakdown": _breakdown(duplicate_counts, label="kind"),
        "assistant_age_bucket_breakdown": _breakdown(age_counts, label="bucket"),
        "block_size_bucket_breakdown": _breakdown(size_counts, label="bucket"),
        "blocker_reason_breakdown": _breakdown(blocker_counts),
        "thinking_output_token_bucket_breakdown": _breakdown(thinking_token_bucket_counts, label="bucket"),
        "plans": plans[:sample_limit],
        "privacy": {
            "metadata_only_output": True,
            "content_free": True,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_tool_payloads_included": False,
            "raw_thinking_text_included": False,
            "thinking_block_fingerprints_included": False,
            "raw_responses_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local Anthropic call metadata plus local request bodies when body logging is enabled; raw content is never emitted",
        },
    }
