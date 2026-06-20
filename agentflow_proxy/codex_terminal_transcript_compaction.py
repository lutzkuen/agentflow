from __future__ import annotations

import copy
import hashlib
from typing import Any

from agentflow_proxy.codex_turn_policy import (
    CODEX_ACTION_KEY_HINTS,
    CODEX_ACTION_VALUE_HINTS,
    CODEX_TEXT_INPUT_TYPES,
)
from agentflow_proxy.store import stable_json
from agentflow_proxy.terminal_features import _line_classes


FAMILY = "codex_terminal_transcript_compaction"
SCHEMA = "agentflow.codex_terminal_transcript_compaction_canary.v1"
_DEFAULT_CANARY_SALT = "codex-terminal-transcript-compaction"
_INCOMPATIBLE_FAMILIES = frozenset({"exact_cache_replay", "managed_recommendation"})
_MIN_TERMINAL_LINE_FRACTION = 0.05


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_fraction(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = default
    if f != f:
        f = default
    return min(max(f, 0.0), 1.0)


def _action_hint_in_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_l = str(key).replace("-", "_").lower()
            if key_l in CODEX_ACTION_KEY_HINTS:
                return True
            if key_l == "type" and isinstance(nested, str) and nested.strip().lower() in CODEX_ACTION_VALUE_HINTS:
                return True
            if isinstance(nested, (dict, list)) and _action_hint_in_value(nested):
                return True
    elif isinstance(value, list):
        return any(_action_hint_in_value(item) for item in value)
    return False


def _has_action_like_params(params: dict[str, Any]) -> bool:
    return _action_hint_in_value(params)


def _text_input_entries(input_value: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(input_value, str):
        entries.append({"container": None, "key": None, "text": input_value})
        return entries
    if isinstance(input_value, dict):
        block_type = str(input_value.get("type") or "text").strip().lower()
        if block_type in CODEX_TEXT_INPUT_TYPES:
            for key in ("text", "input_text", "value"):
                if isinstance(input_value.get(key), str):
                    entries.append({"container": input_value, "key": key, "text": input_value[key]})
                    break
        return entries
    if isinstance(input_value, list):
        for idx, item in enumerate(input_value):
            if isinstance(item, str):
                entries.append({"container": input_value, "key": idx, "text": item})
            elif isinstance(item, dict):
                block_type = str(item.get("type") or "text").strip().lower()
                if block_type in CODEX_TEXT_INPUT_TYPES:
                    for key in ("text", "input_text", "value"):
                        if isinstance(item.get(key), str):
                            entries.append({"container": item, "key": key, "text": item[key]})
                            break
    return entries


def _set_text_entry(entry: dict[str, Any], text: str) -> None:
    container = entry.get("container")
    key = entry.get("key")
    if container is None:
        entry["text"] = text
    elif isinstance(container, list) and isinstance(key, int):
        container[key] = text
    elif isinstance(container, dict) and isinstance(key, str):
        container[key] = text


def _is_terminal_block(text: str, *, min_chars: int = 0) -> bool:
    if len(text) < min_chars:
        return False
    lines = text.splitlines()
    if not lines:
        return False
    terminal_count = sum(1 for line in lines if _line_classes(line))
    return terminal_count >= max(1, int(len(lines) * _MIN_TERMINAL_LINE_FRACTION))


def _compact_text_block(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_evidence_lines: int,
    preserve_diagnostics: bool,
    preserve_error_lines: bool,
    min_block_chars: int,
) -> tuple[str, dict[str, Any]]:
    before_chars = len(text)
    no_change: dict[str, Any] = {
        "changed": False,
        "before_chars": before_chars,
        "after_chars": before_chars,
        "saved_chars": 0,
    }

    if before_chars < min_block_chars:
        return text, {**no_change, "reason": "below-min-block-chars"}

    if not _is_terminal_block(text, min_chars=min_block_chars):
        return text, {**no_change, "reason": "not-terminal-block"}

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines:
        return text, {**no_change, "reason": "too-few-lines"}

    head_count = min(head_lines, total_lines)
    tail_start = max(head_count, total_lines - tail_lines)
    tail_count = total_lines - tail_start

    head = lines[:head_count]
    middle = lines[head_count:tail_start]
    tail = lines[tail_start:]

    diagnostic_preserved: list[str] = []
    if (preserve_diagnostics or preserve_error_lines) and max_evidence_lines > 0:
        for line in middle:
            stripped = line.rstrip("\n\r")
            classes = _line_classes(stripped)
            if classes and (
                (preserve_error_lines and ("error_line" in classes or "stack_trace" in classes or "test_output" in classes))
                or (preserve_diagnostics and classes)
            ):
                diagnostic_preserved.append(line)
                if len(diagnostic_preserved) >= max_evidence_lines:
                    break

    omitted_count = len(middle) - len(diagnostic_preserved)

    if omitted_count <= 0:
        return text, {**no_change, "reason": "no-omittable-lines"}

    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    notice = (
        f"[AgentFlow: middle of long terminal-transcript block omitted; "
        f"hash={h}; original_chars={before_chars}]\n"
    )

    compacted_lines = head + [notice] + diagnostic_preserved + tail
    compacted = "".join(compacted_lines)

    if len(compacted) >= before_chars:
        return text, {**no_change, "reason": "no-savings"}

    return compacted, {
        "changed": True,
        "reason": "terminal-transcript-compacted",
        "before_chars": before_chars,
        "after_chars": len(compacted),
        "saved_chars": before_chars - len(compacted),
        "original_line_count": total_lines,
        "head_lines_kept": head_count,
        "tail_lines_kept": tail_count,
        "middle_lines": len(middle),
        "diagnostic_lines_kept": len(diagnostic_preserved),
        "omitted_lines": omitted_count,
        "raw_text_included": False,
        "raw_lines_included": False,
    }


def _canary_sample(params: dict[str, Any], *, canary: dict[str, Any]) -> dict[str, Any]:
    fraction = _bounded_fraction(canary.get("canary_fraction", canary.get("fraction")), 0.0)
    holdout = _bounded_fraction(canary.get("holdout_fraction"), max(0.0, 1.0 - fraction))
    salt = str(canary.get("canary_salt") or canary.get("salt") or _DEFAULT_CANARY_SALT).strip() or _DEFAULT_CANARY_SALT
    unit_raw = str(canary.get("canary_unit") or canary.get("unit") or "source_hash").strip().lower().replace("-", "_")
    unit = unit_raw if unit_raw in {"source_hash", "thread_id", "model_and_size"} else "source_hash"

    if unit == "thread_id":
        thread_id = params.get("threadId") or params.get("thread_id")
        material = str(thread_id or "") or stable_json(params)
    elif unit == "model_and_size":
        model = params.get("model") or ""
        input_chars = len(stable_json(params.get("input") or ""))
        material = f"{model}:{input_chars}"
    else:
        unit = "source_hash"
        material = stable_json(params)

    candidate_id = f"{FAMILY}:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:12]}"
    digest = hashlib.sha256(f"{salt}\0{unit}\0{candidate_id}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    applied_cutoff = min(1.0, holdout + fraction)

    if sample < holdout:
        cohort, status, reason = "canary_holdout", "holdout", "terminal-compaction-holdout"
    elif sample < applied_cutoff:
        cohort, status, reason = "canary_applied", "applied", "terminal-compaction-applied"
    else:
        cohort, status, reason = "not_selected", "skipped", "terminal-compaction-not-selected"

    return {
        "enabled": True,
        "cohort": cohort,
        "status": status,
        "reason": reason,
        "candidate_id": candidate_id,
        "fraction": fraction,
        "holdout_fraction": holdout,
        "sample_unit": unit,
        "sample_bucket": round(sample, 6),
        "hash_basis": "local-only-salted-policy-sample",
        "raw_basis_included": False,
    }


def _apply_compaction(params: dict[str, Any], *, action: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    keep_recent_turns = max(0, _as_int(action.get("keep_recent_turns"), 2))
    head_lines = max(1, _as_int(action.get("head_lines"), 12))
    tail_lines = max(1, _as_int(action.get("tail_lines"), 16))
    max_evidence_lines = max(0, _as_int(action.get("max_evidence_lines"), 80))
    min_block_chars = max(0, _as_int(action.get("min_block_chars"), 2000))
    min_saved_chars = max(0, _as_int(action.get("min_saved_chars"), 500))
    preserve_diagnostics = bool(action.get("preserve_diagnostics", True))
    preserve_error_lines = bool(action.get("preserve_error_lines", True))

    new_params = copy.deepcopy(params)
    entries = _text_input_entries(new_params.get("input"))
    before_chars = len(stable_json(params))

    if not entries:
        return params, {
            "status": "skipped",
            "reason": "no-text-input",
            "applied": False,
            "changed": False,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "saved_chars": 0,
            "blocks_examined": 0,
            "blocks_compacted": 0,
        }

    older_until = max(0, len(entries) - keep_recent_turns)
    blocks_examined = 0
    blocks_compacted = 0

    for idx, entry in enumerate(entries):
        if idx >= older_until:
            continue
        blocks_examined += 1
        text = str(entry["text"])
        compacted_text, _block_meta = _compact_text_block(
            text,
            head_lines=head_lines,
            tail_lines=tail_lines,
            max_evidence_lines=max_evidence_lines,
            preserve_diagnostics=preserve_diagnostics,
            preserve_error_lines=preserve_error_lines,
            min_block_chars=min_block_chars,
        )
        if _block_meta.get("changed"):
            _set_text_entry(entry, compacted_text)
            blocks_compacted += 1

    if isinstance(params.get("input"), str) and entries:
        new_params["input"] = entries[0]["text"]

    after_chars = len(stable_json(new_params))
    saved_chars = before_chars - after_chars

    if blocks_compacted == 0 or saved_chars < min_saved_chars:
        reason = "below-min-saved-chars" if blocks_compacted > 0 else "no-eligible-blocks"
        return params, {
            "status": "skipped",
            "reason": reason,
            "applied": False,
            "changed": False,
            "before_chars": before_chars,
            "after_chars": before_chars,
            "saved_chars": 0,
            "blocks_examined": blocks_examined,
            "blocks_compacted": 0,
        }

    return new_params, {
        "status": "applied",
        "reason": "terminal-transcript-compacted",
        "applied": True,
        "changed": True,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "saved_chars": saved_chars,
        "blocks_examined": blocks_examined,
        "blocks_compacted": blocks_compacted,
        "raw_text_included": False,
        "raw_commands_included": False,
    }


def codex_terminal_transcript_compaction_decision(
    params: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
    coordinator_ledger: dict[str, Any] | None = None,
    before_chars: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from agentflow_proxy.codex_turn_policy import CODEX_APP_POLICY  # late-bind for reload safety
    effective_policy = policy if policy is not None else (CODEX_APP_POLICY.get("terminal_transcript_compaction") or {})

    _before_chars = before_chars if before_chars is not None else len(stable_json(params))

    def _skip(reason: str, *, suppressed_by: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return params, {
            FAMILY: {
                "schema": SCHEMA,
                "optimization_family": FAMILY,
                "status": "skipped",
                "reason": reason,
                "applied": False,
                "changed": False,
                "before_chars": _before_chars,
                "after_chars": _before_chars,
                "saved_chars": 0,
                "raw_text_included": False,
                "raw_commands_included": False,
                "coordinator": {
                    "selected_families": [],
                    "suppressed_families": [FAMILY] if suppressed_by else [],
                    "suppressed_by": suppressed_by,
                },
            }
        }

    if not bool(effective_policy.get("enabled")):
        return _skip("disabled")

    ledger = coordinator_ledger or {}
    for incompatible in _INCOMPATIBLE_FAMILIES:
        if ledger.get(incompatible):
            return _skip(f"suppressed-by-{incompatible}", suppressed_by=incompatible)

    if _has_action_like_params(params):
        return _skip("action-like-params")

    if not _text_input_entries(params.get("input")):
        return _skip("no-text-input")

    canary_policy = effective_policy.get("canary") if isinstance(effective_policy.get("canary"), dict) else {}
    sample = _canary_sample(params, canary=canary_policy)

    if sample["status"] == "holdout":
        return params, {
            FAMILY: {
                "schema": SCHEMA,
                "optimization_family": FAMILY,
                "status": "holdout",
                "reason": sample["reason"],
                "applied": False,
                "changed": False,
                "before_chars": _before_chars,
                "after_chars": _before_chars,
                "saved_chars": 0,
                "canary": sample,
                "raw_text_included": False,
                "raw_commands_included": False,
                "coordinator": {
                    "selected_families": [],
                    "suppressed_families": [],
                    "suppressed_by": None,
                },
            }
        }

    if sample["status"] == "skipped":
        return _skip(sample["reason"])

    action = effective_policy.get("action") if isinstance(effective_policy.get("action"), dict) else {}
    compacted_params, result = _apply_compaction(params, action=action)

    applied = result.get("applied", False)
    meta: dict[str, Any] = {
        "schema": SCHEMA,
        "optimization_family": FAMILY,
        "status": result.get("status", "skipped"),
        "reason": result.get("reason", "unknown"),
        "applied": applied,
        "changed": result.get("changed", False),
        "before_chars": result.get("before_chars", _before_chars),
        "after_chars": result.get("after_chars", _before_chars),
        "saved_chars": result.get("saved_chars", 0),
        "canary": sample,
        "blocks_examined": result.get("blocks_examined", 0),
        "blocks_compacted": result.get("blocks_compacted", 0),
        "raw_text_included": False,
        "raw_commands_included": False,
        "coordinator": {
            "selected_families": [FAMILY] if applied else [],
            "suppressed_families": [],
            "suppressed_by": None,
        },
    }

    return (compacted_params if applied else params), {FAMILY: meta}
