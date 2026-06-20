from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.codex_turn_policy import CODEX_APP_SOURCE_SURFACE
from agentflow_proxy.crunch import TOKEN_CHARS
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.store import utc_now


SCHEMA = "agentflow.codex_terminal_transcript_opportunity.v1"

_FRACTION_MIDPOINTS = {
    "none": 0.0,
    "lt_10pct": 0.05,
    "10_25pct": 0.17,
    "25_50pct": 0.37,
    "50_75pct": 0.62,
    "gte_75pct": 0.82,
}
_STALE_REASONS = {
    "dependency-missing",
    "dependency-cap-exceeded",
    "dependency-changed",
    "dependency-deleted",
    "file-dependency-missing",
    "file-watch-disabled",
    "stale-risk-blockers",
    "stale-risk-reference",
}


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
    value = str(key or "unknown")
    counter[value] = counter.get(value, 0) + amount


def _breakdown(counter: dict[str, int], *, label: str = "value") -> list[dict[str, Any]]:
    return [
        {label: key, "count": value}
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


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


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 5:
        return "2_5"
    if value <= 20:
        return "6_20"
    if value <= 100:
        return "21_100"
    if value <= 1000:
        return "101_1000"
    return "1000_plus"


def _fraction_bucket(numerator: int, denominator: int) -> str:
    if numerator <= 0 or denominator <= 0:
        return "none"
    ratio = numerator / denominator
    if ratio < 0.10:
        return "lt_10pct"
    if ratio < 0.25:
        return "10_25pct"
    if ratio < 0.50:
        return "25_50pct"
    if ratio < 0.75:
        return "50_75pct"
    return "gte_75pct"


def _terminal_fraction_value(bucket: Any) -> float:
    return _FRACTION_MIDPOINTS.get(str(bucket or "none"), 0.0)


def _terminal_feature_meta(routing: dict[str, Any]) -> dict[str, Any]:
    meta = routing.get("terminal_log_features")
    return meta if isinstance(meta, dict) else {}


def _workflow_phase(window: dict[str, Any], routing: dict[str, Any], crunch: dict[str, Any], cache: dict[str, Any]) -> str:
    for source in (window, cache, crunch, routing):
        value = source.get("workflow_phase") if isinstance(source, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _status_from_cache(cache: dict[str, Any]) -> str:
    return str(cache.get("status") or "unknown")


def _has_reason(sources: list[dict[str, Any]], reasons: set[str]) -> bool:
    for source in sources:
        if str(source.get("reason") or "") in reasons:
            return True
        for value in source.values():
            if isinstance(value, dict) and _has_reason([value], reasons):
                return True
            if isinstance(value, list) and any(isinstance(item, dict) and _has_reason([item], reasons) for item in value):
                return True
    return False


def _already_crunched_scaffold_chars(crunch: dict[str, Any]) -> int:
    repeated = crunch.get("codex_repeated_scaffolding")
    if isinstance(repeated, dict):
        return _as_int(repeated.get("saved_chars"))
    if str(crunch.get("reason") or "") == "codex-repeated-scaffolding-crunched":
        return _as_int(crunch.get("saved_chars"))
    return 0


def _safety_preserve_diagnostics(features: dict[str, Any]) -> bool:
    if not features:
        return False
    if features.get("stack_trace_present") or features.get("test_output_present"):
        return True
    classes = features.get("class_count_buckets") if isinstance(features.get("class_count_buckets"), dict) else {}
    for key in ("build_output", "test_output", "stack_trace"):
        if str(classes.get(key) or "zero") not in {"", "zero", "0"}:
            return True
    return str(features.get("error_line_count_bucket") or "zero") not in {"", "zero", "0"}


def _event_is_terminal_transcript(method: Any) -> bool:
    method_l = str(method or "").replace("_", "").replace("-", "").lower()
    return (
        "commandexecution" in method_l
        or "outputdelta" in method_l
        or "toolresult" in method_l
        or "terminal" in method_l
    )


def _event_aggregates_for_request(store_obj: Any, request_id: Any) -> dict[str, int]:
    if request_id is None:
        return {"event_count": 0, "message_chars": 0, "error_count": 0}
    rows = store_obj.conn.execute(
        """
        select method, message_chars, error_code
        from codex_app_events
        where direction = 'server_to_client'
          and request_id = ?
        """,
        (request_id,),
    ).fetchall()
    event_count = 0
    message_chars = 0
    error_count = 0
    for row in rows:
        if row["error_code"] is not None:
            error_count += 1
        if not _event_is_terminal_transcript(row["method"]):
            continue
        event_count += 1
        message_chars += _as_int(row["message_chars"])
    return {"event_count": event_count, "message_chars": message_chars, "error_count": error_count}


def _candidate_id(basis: dict[str, Any]) -> str:
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"codex-terminal-transcript:{digest}"


def _new_group(public_basis: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(public_basis),
        **public_basis,
        "matched_turn_count": 0,
        "candidate_turn_count": 0,
        "blocked_turn_count": 0,
        "successful_turn_count": 0,
        "error_turn_count": 0,
        "estimated_input_chars": 0,
        "estimated_input_tokens": 0,
        "terminal_event_window_count": 0,
        "terminal_event_message_chars": 0,
        "estimated_terminal_transcript_chars": 0,
        "gross_projected_saved_chars": 0,
        "already_crunched_repeated_scaffold_chars": 0,
        "projected_saved_chars": 0,
        "projected_saved_tokens": 0,
        "projected_saved_usd": 0.0,
        "blocker_counts": {},
    }


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    group["projected_saved_tokens"] = group["projected_saved_chars"] // TOKEN_CHARS
    group["projected_saved_usd"] = round(_as_float(group.get("projected_saved_usd")), 6)
    blockers = group.pop("blocker_counts", {})
    group["blocker_reason_breakdown"] = _breakdown(blockers)
    group["blockers"] = [item["value"] for item in group["blocker_reason_breakdown"]]
    group["candidate_status"] = "candidate" if group["candidate_turn_count"] else "blocked"
    group["sample_limits"] = {
        "raw_samples_included": False,
        "sampled_turn_count": group["matched_turn_count"],
    }
    group["privacy"] = {
        "metadata_only": True,
        "raw_terminal_text_included": False,
        "raw_prompts_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_commands_included": False,
        "file_paths_included": False,
        "request_ids_included": False,
        "thread_ids_included": False,
        "session_ids_included": False,
        "cache_keys_included": False,
        "event_method_names_included": False,
    }
    return group


def build_codex_terminal_transcript_opportunity_report(
    store_obj: Any,
    *,
    limit: int = 1000,
    min_input_chars: int = 8000,
    min_terminal_chars: int = 2000,
    compaction_ratio: float = 0.65,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 1000), 10_000))
    input_floor = max(1, int(min_input_chars or 8000))
    terminal_floor = max(1, int(min_terminal_chars or 2000))
    ratio = min(max(float(compaction_ratio), 0.0), 0.95)
    rows = [
        dict(row)
        for row in store_obj.conn.execute(
            """
            select id, created_at, request_id, thread_id, session_id,
                   message_chars, params_chars, input_items, input_text_chars,
                   result_chars, error_code, routing_json, crunch_json,
                   cache_json, event_window_json
            from codex_app_events
            where direction = 'client_to_server'
              and method = 'turn/start'
            order by created_at desc
            limit ?
            """,
            (capped_limit,),
        ).fetchall()
    ]

    groups: dict[str, dict[str, Any]] = {}
    phase_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    text_bucket_counts: dict[str, int] = {}
    terminal_fraction_counts: dict[str, int] = {}
    cache_status_counts: dict[str, int] = {}
    blocker_totals: dict[str, int] = {}
    scanned_turns = 0
    terminal_signal_turns = 0
    candidate_turns = 0
    blocked_turns = 0
    terminal_event_windows = 0
    terminal_event_chars_total = 0

    for row in rows:
        scanned_turns += 1
        routing = _json_obj(row.get("routing_json"))
        crunch = _json_obj(row.get("crunch_json"))
        cache = _json_obj(row.get("cache_json"))
        window = _json_obj(row.get("event_window_json"))
        terminal_features = _terminal_feature_meta(routing)
        event_aggs = _event_aggregates_for_request(store_obj, row.get("request_id"))

        input_chars = _as_int(row.get("input_text_chars")) or _as_int(window.get("input_text_chars"))
        input_items = _as_int(row.get("input_items")) or _as_int(window.get("input_items"))
        feature_fraction_bucket = str(terminal_features.get("terminal_output_char_fraction_bucket") or "none")
        feature_terminal_chars = int(input_chars * _terminal_fraction_value(feature_fraction_bucket))
        event_terminal_chars = event_aggs["message_chars"]
        event_terminal_count = event_aggs["event_count"]
        event_fraction_bucket = _fraction_bucket(event_terminal_chars, max(input_chars, 1))
        terminal_fraction_bucket = (
            event_fraction_bucket
            if _terminal_fraction_value(event_fraction_bucket) > _terminal_fraction_value(feature_fraction_bucket)
            else feature_fraction_bucket
        )
        estimated_terminal_chars = max(feature_terminal_chars, event_terminal_chars)
        has_terminal_signal = estimated_terminal_chars >= terminal_floor
        if has_terminal_signal:
            terminal_signal_turns += 1
        if event_terminal_count:
            terminal_event_windows += 1
            terminal_event_chars_total += event_terminal_chars

        phase = _workflow_phase(window, routing, crunch, cache)
        cache_status = _status_from_cache(cache)
        repeated_scaffold_saved = _already_crunched_scaffold_chars(crunch)
        non_text = (
            _has_reason([routing, crunch, cache], {"non-text-input"})
            or (input_items > 0 and input_chars <= 0)
        )
        action_like = _has_reason([routing, crunch, cache], {"action-like-params"})
        stale_risk = _has_reason([routing, crunch, cache], _STALE_REASONS)
        safety_diagnostics = _safety_preserve_diagnostics(terminal_features)
        already_crunched = repeated_scaffold_saved > 0

        blockers: list[str] = []
        if non_text:
            blockers.append("non-text-input")
        if action_like:
            blockers.append("action-like-params")
        if phase == "unknown":
            blockers.append("missing-phase")
        elif phase != "tool_execution":
            blockers.append("non-tool-execution-phase")
        if not has_terminal_signal:
            blockers.append("no-terminal-signal")
        if input_chars < input_floor:
            blockers.append("below-min-size")
        if stale_risk:
            blockers.append("stale-risk-blockers")
        if event_aggs["error_count"] or _as_int(row.get("error_code")):
            blockers.append("error-window")
        if safety_diagnostics:
            blockers.append("safety-preserve-diagnostics")
        if already_crunched:
            blockers.append("already-crunched-repeated-scaffolding")

        hard_blockers = {
            "non-text-input",
            "action-like-params",
            "missing-phase",
            "non-tool-execution-phase",
            "no-terminal-signal",
            "below-min-size",
            "stale-risk-blockers",
            "error-window",
        }
        is_candidate = not (set(blockers) & hard_blockers)
        if is_candidate:
            candidate_turns += 1
        else:
            blocked_turns += 1

        gross_saved_chars = int(estimated_terminal_chars * ratio) if is_candidate else 0
        projected_saved_chars = max(0, gross_saved_chars - min(gross_saved_chars, repeated_scaffold_saved))
        projected_saved_tokens = projected_saved_chars // TOKEN_CHARS
        projected_saved_usd = estimate_cost(
            str(routing.get("requested_model") or codex_app_model()),
            projected_saved_tokens,
            0,
            provider="openai",
            processing_mode=codex_app_processing_mode(),
        ) or 0.0
        signal_source = "none"
        if feature_terminal_chars >= terminal_floor and event_terminal_count:
            signal_source = "input-terminal-features+event-window"
        elif feature_terminal_chars >= terminal_floor:
            signal_source = "input-terminal-features"
        elif event_terminal_count:
            signal_source = "event-window-terminal-events"

        public_basis = {
            "source_surface": CODEX_APP_SOURCE_SURFACE,
            "app_family": "codex",
            "granularity": "agent_turn",
            "workflow_phase": phase,
            "text_bucket": _text_bucket(input_chars),
            "terminal_fraction_bucket": terminal_fraction_bucket,
            "terminal_event_count_bucket": _count_bucket(event_terminal_count),
            "terminal_signal_source": signal_source,
            "cache_status": cache_status,
            "already_crunched_repeated_scaffold": already_crunched,
            "safety_preserve_diagnostics": safety_diagnostics,
        }
        group_key = _candidate_id(public_basis)
        group = groups.setdefault(group_key, _new_group(public_basis))
        group["matched_turn_count"] += 1
        group["candidate_turn_count"] += int(is_candidate)
        group["blocked_turn_count"] += int(not is_candidate)
        group["successful_turn_count"] += int(not (event_aggs["error_count"] or _as_int(row.get("error_code"))))
        group["error_turn_count"] += int(bool(event_aggs["error_count"] or _as_int(row.get("error_code"))))
        group["estimated_input_chars"] += input_chars
        group["estimated_input_tokens"] += max(0, input_chars // TOKEN_CHARS)
        group["terminal_event_window_count"] += int(event_terminal_count > 0)
        group["terminal_event_message_chars"] += event_terminal_chars
        group["estimated_terminal_transcript_chars"] += estimated_terminal_chars
        group["gross_projected_saved_chars"] += gross_saved_chars
        group["already_crunched_repeated_scaffold_chars"] += repeated_scaffold_saved
        group["projected_saved_chars"] += projected_saved_chars
        group["projected_saved_tokens"] += projected_saved_tokens
        group["projected_saved_usd"] += float(projected_saved_usd)
        for blocker in blockers or ["ready-for-terminal-transcript-review"]:
            _increment(group["blocker_counts"], blocker)
            _increment(blocker_totals, blocker)
        _increment(phase_counts, phase)
        _increment(source_counts, signal_source)
        _increment(text_bucket_counts, _text_bucket(input_chars))
        _increment(terminal_fraction_counts, terminal_fraction_bucket)
        _increment(cache_status_counts, cache_status)

    candidates = [_finalize_group(group) for group in groups.values()]
    candidates.sort(
        key=lambda item: (
            _as_int(item.get("candidate_turn_count")),
            _as_int(item.get("projected_saved_chars")),
            _as_float(item.get("projected_saved_usd")),
            _as_int(item.get("matched_turn_count")),
        ),
        reverse=True,
    )

    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "limit": capped_limit,
        "min_input_chars": input_floor,
        "min_terminal_chars": terminal_floor,
        "projection_policy": {
            "schema": "agentflow.codex_terminal_transcript_projection_policy.v1",
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "default_apply": False,
            "method": "local Codex turn/start metadata, event-window terminal event counts, and terminal-log feature buckets only",
            "compaction_ratio": ratio,
            "raw_text_required": False,
            "already_crunched_policy": "subtract repeated-scaffolding saved chars from gross terminal-transcript projection to avoid double counting",
        },
        "sample_policy": {
            "raw_samples_included": False,
            "sampled_turn_limit": capped_limit,
            "sampled_candidate_limit": len(candidates),
        },
        "summary": {
            "scanned_turn_count": scanned_turns,
            "terminal_signal_turn_count": terminal_signal_turns,
            "candidate_count": sum(1 for item in candidates if _as_int(item.get("candidate_turn_count")) > 0),
            "candidate_turn_count": candidate_turns,
            "blocked_turn_count": blocked_turns,
            "terminal_event_window_count": terminal_event_windows,
            "terminal_event_message_chars": terminal_event_chars_total,
            "estimated_terminal_transcript_chars": sum(_as_int(item.get("estimated_terminal_transcript_chars")) for item in candidates),
            "gross_projected_saved_chars": sum(_as_int(item.get("gross_projected_saved_chars")) for item in candidates),
            "already_crunched_repeated_scaffold_chars": sum(_as_int(item.get("already_crunched_repeated_scaffold_chars")) for item in candidates),
            "projected_saved_chars": sum(_as_int(item.get("projected_saved_chars")) for item in candidates),
            "projected_saved_tokens": sum(_as_int(item.get("projected_saved_tokens")) for item in candidates),
            "projected_saved_usd": round(sum(_as_float(item.get("projected_saved_usd")) for item in candidates), 6),
        },
        "workflow_phase_breakdown": _breakdown(phase_counts, label="phase"),
        "terminal_signal_source_breakdown": _breakdown(source_counts, label="source"),
        "text_bucket_breakdown": _breakdown(text_bucket_counts, label="bucket"),
        "terminal_fraction_bucket_breakdown": _breakdown(terminal_fraction_counts, label="bucket"),
        "cache_status_breakdown": _breakdown(cache_status_counts, label="status"),
        "blocker_reason_breakdown": _breakdown(blocker_totals),
        "candidates": candidates,
        "privacy": {
            "metadata_only": True,
            "raw_terminal_text_included": False,
            "raw_terminal_lines_included": False,
            "raw_prompts_included": False,
            "raw_messages_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "raw_commands_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "request_ids_included": False,
            "thread_ids_included": False,
            "session_ids_included": False,
            "cache_keys_included": False,
            "event_method_names_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local Codex app event metadata and derived feature buckets; raw text, ids, paths, and commands are never emitted",
        },
    }
