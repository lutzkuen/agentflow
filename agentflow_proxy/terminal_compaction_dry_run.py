from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from agentflow_proxy.pricing import estimate_cost
from agentflow_proxy.store import stable_json, utc_now
from agentflow_proxy.terminal_features import terminal_log_features_from_text


TERMINAL_OUTPUT_COMPACTION_DRY_RUN_SCHEMA = "agentflow.terminal_output_compaction_dry_run.v1"
TERMINAL_OUTPUT_COMPACTION_PLAN_SCHEMA = "agentflow.terminal_output_compaction_plan.v1"
TOKEN_CHARS = 4
DEFAULT_KEEP_RECENT_TURNS = 2
DEFAULT_MIN_BLOCK_CHARS = 2_000
DEFAULT_HEAD_LINES = 12
DEFAULT_TAIL_LINES = 16
DEFAULT_MAX_EVIDENCE_LINES = 80
DEFAULT_MIN_SAVED_CHARS = 500
PLATEAU_MIN_TEXT_CHARS = 8_000
PLATEAU_MAX_DELTA_RATIO = 0.03

_COMMAND_RE = re.compile(r"^\s*(?:[$#>]\s+\S|[+\-]\s+\S|[A-Za-z]:\\[^>]+>\s+\S)")
_ERROR_RE = re.compile(r"\b(?:ERROR|ERR|FATAL|CRITICAL|Exception|Traceback|AssertionError|failed|failure|panic)\b", re.IGNORECASE)
_STACK_RE = re.compile(r"^\s*(?:Traceback \(most recent call last\):|File \".+\", line \d+|at\s+.+:\d+(?::\d+)?|Caused by:)")
_FAILURE_RE = re.compile(r"^\s*(?:FAILED\s+\S+|FAILURES|FAIL:|ERROR:|AssertionError|E\s+AssertionError|\d+\s+failed\b)", re.IGNORECASE)
_EXIT_RE = re.compile(r"\b(?:exit(?:ed)?(?: status| code)?|return code|status)\s*[=:]?\s*(?:[1-9]\d*|failed)\b", re.IGNORECASE)
_FILE_CHANGE_RE = re.compile(
    r"^\s*(?:(?:modified|created|deleted|renamed|updated|wrote|changed)\s+\S+|[MADRCU?]{1,2}\s+\S+|"
    r"\S+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|md|yaml|yml|toml|json):\d+)",
    re.IGNORECASE,
)


def _json_obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
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


def _hash_basis(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _text_bucket(chars: int) -> str:
    if chars < 8_000:
        return "lt_8k_chars"
    if chars < 32_000:
        return "8k_32k_chars"
    if chars < 128_000:
        return "32k_128k_chars"
    return "gte_128k_chars"


def _model_family(model: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    if text:
        return text.split("-", 2)[0]
    return "unknown"


def _source_surface(provider: Any, path: Any) -> str:
    if str(provider or "anthropic") == "anthropic" and str(path or "").endswith("/v1/messages"):
        return "anthropic_messages"
    return str(provider or "unknown")


def _category(row: dict[str, Any]) -> str:
    routing = _json_obj(row.get("routing_json"))
    return str(row.get("category") or routing.get("category") or "unknown")


def _row_text_chars(row: dict[str, Any]) -> int:
    routing = _json_obj(row.get("routing_json"))
    text_chars = _as_int(routing.get("text_chars"))
    if text_chars > 0:
        return text_chars
    tokens = _as_int(row.get("actual_input_tokens")) or _as_int(row.get("input_tokens_est"))
    return max(0, tokens * TOKEN_CHARS)


def _plateau_row_ids(rows: list[dict[str, Any]], *, min_text_chars: int, max_delta_ratio: float) -> set[str]:
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        session = row.get("session_id")
        row_id = row.get("id")
        if session in (None, "") or row_id in (None, ""):
            continue
        by_session.setdefault(str(session), []).append(row)

    plateau_ids: set[str] = set()
    for session_rows in by_session.values():
        previous: dict[str, Any] | None = None
        for row in sorted(session_rows, key=lambda item: str(item.get("created_at") or "")):
            text_chars = _row_text_chars(row)
            if previous is not None:
                previous_chars = _row_text_chars(previous)
                if (
                    previous_chars >= min_text_chars
                    and text_chars >= min_text_chars
                    and abs(text_chars - previous_chars) / max(previous_chars, 1) <= max_delta_ratio
                ):
                    plateau_ids.add(str(previous.get("id")))
                    plateau_ids.add(str(row.get("id")))
            previous = row
    return plateau_ids


def _plateau_status(row: dict[str, Any], plateau_ids: set[str], *, min_text_chars: int) -> str:
    if row.get("session_id") in (None, ""):
        return "no-session"
    if str(row.get("id") or "") in plateau_ids:
        return "plateau-adjacent"
    if _row_text_chars(row) >= min_text_chars:
        return "large-not-plateaued"
    return "below-plateau-threshold"


def _line_classes(line: str) -> set[str]:
    classes: set[str] = set()
    if _COMMAND_RE.search(line):
        classes.add("command_summary")
    if _STACK_RE.search(line):
        classes.add("stack_trace")
    if _FAILURE_RE.search(line):
        classes.add("failure_line")
    if _EXIT_RE.search(line):
        classes.add("exit_status")
    if _FILE_CHANGE_RE.search(line):
        classes.add("file_change_hint")
    if _ERROR_RE.search(line):
        classes.add("error_line")
    return classes


def _evidence_counts(lines: list[str]) -> dict[str, int]:
    counts = {
        "command_summary": 0,
        "error_line": 0,
        "stack_trace": 0,
        "failure_line": 0,
        "exit_status": 0,
        "file_change_hint": 0,
    }
    for line in lines:
        for class_name in _line_classes(line):
            counts[class_name] += 1
    return counts


def _terminal_fraction_bucket(features: dict[str, Any]) -> str:
    return str(features.get("terminal_output_char_fraction_bucket") or "none")


def _terminal_candidate(text: str, *, min_block_chars: int) -> tuple[bool, dict[str, Any], str]:
    if len(text) < min_block_chars:
        return False, terminal_log_features_from_text(text), "below-min-block-chars"
    features = terminal_log_features_from_text(text)
    if _terminal_fraction_bucket(features) == "none":
        return False, features, "terminal-output-signal-missing"
    return True, features, "eligible"


def _compact_terminal_text(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_evidence_lines: int,
    min_saved_chars: int,
) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    source_counts = _evidence_counts(lines)
    selected: set[int] = set(range(min(max(0, head_lines), len(lines))))
    if tail_lines > 0:
        selected.update(range(max(0, len(lines) - tail_lines), len(lines)))

    evidence_seen: set[tuple[str, str]] = set()
    evidence_added = 0
    for index, line in enumerate(lines):
        classes = _line_classes(line)
        if not classes:
            continue
        normalized = re.sub(r"\d+", "N", line.strip())[:180]
        key = ("+".join(sorted(classes)), normalized)
        if key in evidence_seen:
            continue
        evidence_seen.add(key)
        selected.add(index)
        evidence_added += 1
        if evidence_added >= max(0, max_evidence_lines):
            break

    ordered = sorted(selected)
    preserved_lines = [lines[index] for index in ordered]
    omitted_lines = max(0, len(lines) - len(ordered))
    marker = (
        "[agentflow terminal-output compaction dry-run: "
        f"preserved {len(ordered)} of {len(lines)} lines; omitted {omitted_lines} repetitive terminal/log lines]"
    )
    replacement_lines = [marker, *preserved_lines]
    replacement = "\n".join(replacement_lines)
    if text.endswith("\n"):
        replacement += "\n"
    saved_chars = max(0, len(text) - len(replacement))
    after_counts = _evidence_counts(replacement.splitlines())
    flags = {
        "command_summaries_preserved": source_counts["command_summary"] == 0 or after_counts["command_summary"] > 0,
        "error_lines_preserved": source_counts["error_line"] == 0 or after_counts["error_line"] > 0,
        "stack_traces_preserved": source_counts["stack_trace"] == 0 or after_counts["stack_trace"] > 0,
        "failure_lines_preserved": source_counts["failure_line"] == 0 or after_counts["failure_line"] > 0,
        "exit_status_preserved": source_counts["exit_status"] == 0 or after_counts["exit_status"] > 0,
        "file_change_hints_preserved": source_counts["file_change_hint"] == 0 or after_counts["file_change_hint"] > 0,
    }
    if saved_chars < min_saved_chars:
        return text, {
            "status": "blocked",
            "reason": "no-compaction-savings-projected",
            "before_chars": len(text),
            "after_chars": len(text),
            "saved_chars": 0,
            "line_count": len(lines),
            "omitted_line_count": 0,
            "source_evidence_counts": source_counts,
            "preserved_evidence_counts": source_counts,
            "preservation_flags": flags,
        }
    return replacement, {
        "status": "planned",
        "reason": "terminal-output-compaction-planned",
        "before_chars": len(text),
        "after_chars": len(replacement),
        "saved_chars": saved_chars,
        "line_count": len(lines),
        "preserved_line_count": len(ordered),
        "omitted_line_count": omitted_lines,
        "source_evidence_counts": source_counts,
        "preserved_evidence_counts": after_counts,
        "preservation_flags": flags,
    }


def _target_id(message_index: int, path: list[Any]) -> str:
    basis = {"message_index": message_index, "path": path}
    return "terminal-block:" + _hash_basis(basis)


def _iter_message_text_targets(message: dict[str, Any], message_index: int) -> list[dict[str, Any]]:
    content = message.get("content")
    targets: list[dict[str, Any]] = []
    if isinstance(content, str):
        targets.append({"message_index": message_index, "path": ["content"], "text": content, "kind": "message_text"})
        return targets
    if not isinstance(content, list):
        return targets

    for content_index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            nested = block.get("content")
            if isinstance(nested, str):
                targets.append(
                    {
                        "message_index": message_index,
                        "path": ["content", content_index, "content"],
                        "text": nested,
                        "kind": "tool_result",
                        "tool_use_id_present": bool(tool_use_id),
                    }
                )
            elif isinstance(nested, list):
                for nested_index, nested_block in enumerate(nested):
                    if isinstance(nested_block, dict) and nested_block.get("type") in {"text", "input_text"} and isinstance(nested_block.get("text"), str):
                        targets.append(
                            {
                                "message_index": message_index,
                                "path": ["content", content_index, "content", nested_index, "text"],
                                "text": nested_block["text"],
                                "kind": "tool_result_text",
                                "tool_use_id_present": bool(tool_use_id),
                            }
                        )
            continue
        if block_type in {"text", "input_text"} and isinstance(block.get("text"), str):
            targets.append(
                {
                    "message_index": message_index,
                    "path": ["content", content_index, "text"],
                    "text": block["text"],
                    "kind": "adjacent_text",
                }
            )
    return targets


def _set_path(root: dict[str, Any], message_index: int, path: list[Any], value: str) -> None:
    current: Any = root["messages"][message_index]
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _tool_result_ids(body: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ids
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                ids.append(str(block["tool_use_id"]))
    return ids


def plan_terminal_output_compaction(
    body: dict[str, Any],
    *,
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    max_evidence_lines: int = DEFAULT_MAX_EVIDENCE_LINES,
    min_saved_chars: int = DEFAULT_MIN_SAVED_CHARS,
    policy_source: str = "local-default",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(body, dict):
        return None, {"status": "blocked", "reason": "invalid-request-body"}
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None, {"status": "blocked", "reason": "messages-missing"}
    old_limit = max(0, len(messages) - max(0, keep_recent_turns))
    if old_limit <= 0:
        return None, {"status": "blocked", "reason": "recent-turns-only"}

    before = stable_json(body)
    planned_targets: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    replacement_by_path: dict[tuple[Any, ...], str] = {}
    for message_index, message in enumerate(messages[:old_limit]):
        if not isinstance(message, dict):
            continue
        for target in _iter_message_text_targets(message, message_index):
            ok, features, reason = _terminal_candidate(str(target.get("text") or ""), min_block_chars=min_block_chars)
            if not ok:
                blockers[reason] = blockers.get(reason, 0) + 1
                continue
            if str(target.get("kind") or "").startswith("tool_result") and not target.get("tool_use_id_present"):
                blockers["tool-result-id-missing"] = blockers.get("tool-result-id-missing", 0) + 1
                continue
            replacement, stats = _compact_terminal_text(
                str(target.get("text") or ""),
                head_lines=head_lines,
                tail_lines=tail_lines,
                max_evidence_lines=max_evidence_lines,
                min_saved_chars=min_saved_chars,
            )
            if stats["status"] != "planned":
                blockers[str(stats.get("reason") or "not-planned")] = blockers.get(str(stats.get("reason") or "not-planned"), 0) + 1
                continue
            path = list(target["path"])
            target_plan = {
                "target_id": _target_id(message_index, path),
                "message_index": message_index,
                "kind": target.get("kind"),
                "terminal_output_char_fraction_bucket": _terminal_fraction_bucket(features),
                "before_chars": stats["before_chars"],
                "after_chars": stats["after_chars"],
                "saved_chars": stats["saved_chars"],
                "estimated_saved_tokens": stats["saved_chars"] // TOKEN_CHARS,
                "line_count": stats["line_count"],
                "preserved_line_count": stats["preserved_line_count"],
                "omitted_line_count": stats["omitted_line_count"],
                "source_evidence_counts": stats["source_evidence_counts"],
                "preserved_evidence_counts": stats["preserved_evidence_counts"],
                "preservation_flags": stats["preservation_flags"],
            }
            planned_targets.append(target_plan)
            replacement_by_path[(message_index, *path)] = replacement

    if not planned_targets:
        reason = "no-terminal-output-candidates"
        if blockers:
            reason = sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return None, {
            "status": "blocked",
            "reason": reason,
            "blocker_counts": [{"value": key, "count": value} for key, value in sorted(blockers.items())],
            "old_turn_count": old_limit,
            "recent_turn_count": len(messages) - old_limit,
        }

    planned_body = copy.deepcopy(body)
    for path_key, replacement in replacement_by_path.items():
        message_index = int(path_key[0])
        path = list(path_key[1:])
        _set_path(planned_body, message_index, path, replacement)

    after = stable_json(planned_body)
    before_ids = _tool_result_ids(body)
    after_ids = _tool_result_ids(planned_body)
    before_recent = stable_json({"messages": messages[old_limit:]})
    after_recent = stable_json({"messages": planned_body.get("messages", [])[old_limit:]})
    saved_chars = max(0, len(before) - len(after))
    all_flags = {
        "tool_protocol_ids_preserved": before_ids == after_ids,
        "recent_turns_preserved": before_recent == after_recent,
    }
    for key in (
        "command_summaries_preserved",
        "error_lines_preserved",
        "stack_traces_preserved",
        "failure_lines_preserved",
        "exit_status_preserved",
        "file_change_hints_preserved",
    ):
        all_flags[key] = all(bool(target["preservation_flags"].get(key)) for target in planned_targets)
    plan = {
        "schema": TERMINAL_OUTPUT_COMPACTION_PLAN_SCHEMA,
        "status": "planned",
        "reason": "terminal-output-compaction-planned",
        "policy_source": policy_source,
        "rule_id": "local-terminal-output-compaction-dry-run",
        "mutation_applied": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "keep_recent_turns": max(0, keep_recent_turns),
        "min_block_chars": max(0, min_block_chars),
        "head_lines": max(0, head_lines),
        "tail_lines": max(0, tail_lines),
        "target_count": len(planned_targets),
        "before_chars": len(before),
        "after_chars": len(after),
        "saved_chars": saved_chars,
        "estimated_saved_tokens": saved_chars // TOKEN_CHARS,
        "targets": planned_targets,
        "preservation_flags": all_flags,
    }
    return plan, {"status": "planned", "reason": "terminal-output-compaction-planned", "planned_body": planned_body}


def apply_terminal_output_compaction_plan(body: dict[str, Any], plan: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Apply a freshly computed terminal compaction plan to a copy of body.

    This helper intentionally recomputes the deterministic plan from body settings instead of
    trusting serialized raw replacements. Dry-run reports stay metadata-only.
    """
    if strict and not isinstance(plan, dict):
        raise ValueError("terminal compaction plan is required")
    settings = plan if isinstance(plan, dict) else {}
    fresh, meta = plan_terminal_output_compaction(
        copy.deepcopy(body),
        keep_recent_turns=_as_int(settings.get("keep_recent_turns"), DEFAULT_KEEP_RECENT_TURNS),
        min_block_chars=_as_int(settings.get("min_block_chars"), DEFAULT_MIN_BLOCK_CHARS),
        head_lines=_as_int(settings.get("head_lines"), DEFAULT_HEAD_LINES),
        tail_lines=_as_int(settings.get("tail_lines"), DEFAULT_TAIL_LINES),
    )
    if fresh is None:
        if strict:
            raise ValueError(str(meta.get("reason") or "terminal compaction plan is not applicable"))
        return copy.deepcopy(body)
    planned_body = meta.get("planned_body")
    return planned_body if isinstance(planned_body, dict) else copy.deepcopy(body)


def build_terminal_output_compaction_dry_run(
    store_obj: Any,
    *,
    limit: int = 500,
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    min_block_chars: int = DEFAULT_MIN_BLOCK_CHARS,
    min_text_chars: int = PLATEAU_MIN_TEXT_CHARS,
    max_plateau_delta_ratio: float = PLATEAU_MAX_DELTA_RATIO,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 10_000))
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select * from (
                select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                       requested_model, routed_model, stream, status_code, input_tokens_est,
                       actual_input_tokens, cost_est_usd, cost_baseline_usd, category,
                       crunch_json, routing_json, cache_json, request_json, session_id
                from calls
                order by created_at desc
                limit ?
            ) recent_calls
            order by created_at asc
            """,
            (capped_limit,),
        ).fetchall()
    ]
    plateau_ids = _plateau_row_ids(rows, min_text_chars=max(1, int(min_text_chars)), max_delta_ratio=max(0.0, float(max_plateau_delta_ratio)))

    plans: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    provider_rows = 0
    body_rows = 0
    raw_used = 0
    for row in rows:
        provider = str(row.get("provider") or "anthropic")
        category = _category(row)
        body = _json_obj(row.get("request_json"))
        if provider == "anthropic" and str(row.get("path") or "").endswith("/v1/messages"):
            provider_rows += 1
        if body:
            body_rows += 1
        blockers: list[str] = []
        if provider != "anthropic" or not str(row.get("path") or "").endswith("/v1/messages"):
            blockers.append("unsupported-source-surface")
        if category != "tool-result":
            blockers.append("non-tool-result-category")
        plateau_status = _plateau_status(row, plateau_ids, min_text_chars=max(1, int(min_text_chars)))
        if plateau_status != "plateau-adjacent":
            blockers.append(plateau_status)
        if not body:
            blockers.append("request-body-unavailable")
        if _as_int(row.get("status_code")) >= 400:
            blockers.append("error-response")
        plan: dict[str, Any] | None = None
        meta: dict[str, Any] = {}
        if not blockers:
            raw_used += 1
            plan, meta = plan_terminal_output_compaction(
                body,
                keep_recent_turns=keep_recent_turns,
                min_block_chars=min_block_chars,
            )
            if plan is None:
                blockers.append(str(meta.get("reason") or "not-eligible"))

        for blocker in blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

        if plan is None and not blockers:
            continue
        basis = {
            "provider": provider,
            "source_surface": _source_surface(provider, row.get("path")),
            "category": category,
            "model_family": _model_family(row.get("routed_model") or row.get("requested_model")),
            "text_bucket": _text_bucket(_row_text_chars(row)),
            "plateau_status": plateau_status,
            "created_bucket": str(row.get("created_at") or "")[:13],
            "call_hash": _hash_basis({"id": row.get("id")}),
        }
        entry = {
            "candidate_id": "terminal-output-compaction-dry-run:" + _hash_basis(basis),
            **basis,
            "stream": bool(_as_int(row.get("stream"))),
            "status_code_bucket": "2xx" if _as_int(row.get("status_code")) < 300 else ("4xx" if _as_int(row.get("status_code")) < 500 else "5xx"),
            "blockers": sorted(set(blockers)),
            "policy_source": "local-default",
            "dry_run": True,
            "mutation_applied": False,
        }
        if plan is not None:
            saved_tokens = _as_int(plan.get("estimated_saved_tokens"))
            saved_usd = estimate_cost(str(row.get("routed_model") or row.get("requested_model") or ""), saved_tokens, 0, provider="anthropic") or 0.0
            entry.update(
                {
                    "status": "planned",
                    "before_chars": plan["before_chars"],
                    "after_chars": plan["after_chars"],
                    "projected_saved_chars": plan["saved_chars"],
                    "projected_saved_tokens": saved_tokens,
                    "projected_saved_usd": round(saved_usd, 8),
                    "target_count": plan["target_count"],
                    "preservation_flags": plan["preservation_flags"],
                    "target_summaries": [
                        {
                            "target_id": target["target_id"],
                            "kind": target["kind"],
                            "before_chars": target["before_chars"],
                            "after_chars": target["after_chars"],
                            "saved_chars": target["saved_chars"],
                            "estimated_saved_tokens": target["estimated_saved_tokens"],
                            "line_count": target["line_count"],
                            "preserved_line_count": target["preserved_line_count"],
                            "omitted_line_count": target["omitted_line_count"],
                            "source_evidence_counts": target["source_evidence_counts"],
                            "preserved_evidence_counts": target["preserved_evidence_counts"],
                            "preservation_flags": target["preservation_flags"],
                        }
                        for target in plan["targets"]
                    ],
                }
            )
        else:
            entry.update(
                {
                    "status": "blocked",
                    "before_chars": _row_text_chars(row),
                    "after_chars": _row_text_chars(row),
                    "projected_saved_chars": 0,
                    "projected_saved_tokens": 0,
                    "projected_saved_usd": 0.0,
                    "target_count": 0,
                    "preservation_flags": {},
                    "target_summaries": [],
                }
            )
        plans.append(entry)

    plans.sort(
        key=lambda item: (
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("projected_saved_chars")),
            1 if item.get("status") == "planned" else 0,
        ),
        reverse=True,
    )
    planned = [item for item in plans if item.get("status") == "planned"]
    return {
        "schema": TERMINAL_OUTPUT_COMPACTION_DRY_RUN_SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "lookback_call_limit": capped_limit,
        "policy": {
            "schema": "agentflow.terminal_output_compaction_policy.v1",
            "policy_source": "local-default",
            "rule_id": "local-terminal-output-compaction-dry-run",
            "default_apply": False,
            "keep_recent_turns": max(0, int(keep_recent_turns)),
            "min_block_chars": max(0, int(min_block_chars)),
            "min_text_chars": max(1, int(min_text_chars)),
            "max_plateau_delta_ratio": max(0.0, float(max_plateau_delta_ratio)),
        },
        "summary": {
            "scanned_call_count": len(rows),
            "provider_call_count": provider_rows,
            "body_rows": body_rows,
            "raw_bodies_read_locally": raw_used,
            "planned_call_count": len(planned),
            "blocked_call_count": len(plans) - len(planned),
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in planned),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in planned),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in planned), 8),
        },
        "blocker_reason_breakdown": [
            {"value": key, "count": value}
            for key, value in sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "plans": plans,
        "privacy": {
            "metadata_only_output": True,
            "raw_bodies_read_locally": raw_used > 0,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_tool_payloads_included": False,
            "raw_terminal_text_included": False,
            "raw_session_ids_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
        },
    }
