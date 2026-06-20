from __future__ import annotations

import hashlib
import json
from typing import Any

from agentflow_proxy.codex_turn_policy import (
    CODEX_APP_POLICY,
    CODEX_APP_SOURCE_SURFACE,
    CODEX_TERMINAL_TRANSCRIPT_COMPACTION_CONDITION_KEYS,
)
from agentflow_proxy.codex_terminal_compaction_report import (
    _already_crunched_scaffold_chars,
    _as_float,
    _as_int,
    _breakdown,
    _count_bucket,
    _event_aggregates_for_request,
    _fraction_bucket,
    _has_reason,
    _json_obj,
    _safety_preserve_diagnostics,
    _status_from_cache,
    _terminal_feature_meta,
    _terminal_fraction_value,
    _text_bucket,
    _workflow_phase,
)
from agentflow_proxy.crunch import TOKEN_CHARS
from agentflow_proxy.pricing import codex_app_model, codex_app_processing_mode, estimate_cost
from agentflow_proxy.public_metadata import public_id, public_label
from agentflow_proxy.store import stable_json, utc_now


SCHEMA = "agentflow.codex_terminal_transcript_compaction_dry_run.v1"
PLAN_SCHEMA = "agentflow.codex_terminal_transcript_compaction_plan.v1"
DEFAULT_AVG_TERMINAL_LINE_CHARS = 100

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
_SUPPORTED_CONDITIONS = set(CODEX_TERMINAL_TRANSCRIPT_COMPACTION_CONDITION_KEYS)


def _hash_basis(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _bounded_fraction(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed != parsed:
        parsed = default
    return min(max(parsed, 0.0), 1.0)


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        return any(_value_matches(item, actual) for item in expected)
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return _normalized(expected) == _normalized(actual)


def _rule_id(rule: dict[str, Any], index: int) -> str:
    return public_id(
        rule.get("rule_id") or rule.get("id") or f"codex-terminal-transcript-rule-{index + 1}",
        prefix="codex-terminal-transcript-rule",
        fallback=f"codex-terminal-transcript-rule-{index + 1}",
    )


def _candidate_policy_id(rule: dict[str, Any], index: int) -> str:
    return public_id(
        rule.get("candidate_id") or rule.get("rule_id") or rule.get("id") or f"codex-terminal-transcript-candidate-{index + 1}",
        prefix="codex-terminal-transcript-candidate",
        fallback=f"codex-terminal-transcript-candidate-{index + 1}",
    )


def _action_id(rule: dict[str, Any]) -> str | None:
    value = rule.get("action_id")
    if value is None:
        return None
    return public_id(value, prefix="codex-terminal-transcript-action")


def _policy_rules(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    source = policy if isinstance(policy, dict) else {}
    terminal = source.get("terminal_transcript_compaction") if "terminal_transcript_compaction" in source else source
    if not isinstance(terminal, dict) or not terminal:
        terminal = CODEX_APP_POLICY.get("terminal_transcript_compaction")
    if not isinstance(terminal, dict):
        return []
    rules: list[dict[str, Any]] = []
    base = {key: value for key, value in terminal.items() if key != "rules"}
    if base:
        rules.append(base)
    for item in terminal.get("rules") or []:
        if not isinstance(item, dict):
            continue
        merged = {key: value for key, value in base.items() if key not in {"rule_id", "candidate_id", "action_id"}}
        merged.update(item)
        if "id" in item and "rule_id" not in merged:
            merged["rule_id"] = item["id"]
        rules.append(merged)
    return rules


def _method_is_terminal_transcript(method: Any) -> bool:
    method_l = str(method or "").replace("_", "").replace("-", "").lower()
    return (
        "commandexecution" in method_l
        or "outputdelta" in method_l
        or "toolresult" in method_l
        or "terminal" in method_l
    )


def _window_terminal_event_count(window: dict[str, Any]) -> int:
    counts = window.get("method_counts") if isinstance(window.get("method_counts"), dict) else {}
    total = 0
    for method, count in counts.items():
        if _method_is_terminal_transcript(method):
            total += _as_int(count)
    return total


def _event_signal(store_obj: Any, row: dict[str, Any], input_chars: int, terminal_features: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    event_aggs = _event_aggregates_for_request(store_obj, row.get("request_id"))
    feature_fraction_bucket = str(terminal_features.get("terminal_output_char_fraction_bucket") or "none")
    feature_terminal_chars = int(input_chars * _terminal_fraction_value(feature_fraction_bucket))
    event_terminal_chars = _as_int(event_aggs.get("message_chars"))
    event_terminal_count = max(_as_int(event_aggs.get("event_count")), _window_terminal_event_count(window))
    event_fraction_bucket = _fraction_bucket(event_terminal_chars, max(input_chars, 1))
    terminal_fraction_bucket = (
        event_fraction_bucket
        if _terminal_fraction_value(event_fraction_bucket) > _terminal_fraction_value(feature_fraction_bucket)
        else feature_fraction_bucket
    )
    signal_source = "none"
    if feature_terminal_chars > 0 and event_terminal_count:
        signal_source = "input-terminal-features+event-window"
    elif feature_terminal_chars > 0:
        signal_source = "input-terminal-features"
    elif event_terminal_count:
        signal_source = "event-window-terminal-events"
    return {
        **event_aggs,
        "feature_terminal_chars": feature_terminal_chars,
        "event_terminal_chars": event_terminal_chars,
        "event_terminal_count": event_terminal_count,
        "terminal_fraction_bucket": terminal_fraction_bucket,
        "terminal_signal_source": signal_source,
        "estimated_terminal_chars": max(feature_terminal_chars, event_terminal_chars),
    }


def _row_features(store_obj: Any, row: dict[str, Any], index: int) -> dict[str, Any]:
    routing = _json_obj(row.get("routing_json"))
    crunch = _json_obj(row.get("crunch_json"))
    cache = _json_obj(row.get("cache_json"))
    window = _json_obj(row.get("event_window_json"))
    terminal_features = _terminal_feature_meta(routing)
    input_chars = _as_int(row.get("input_text_chars")) or _as_int(window.get("input_text_chars"))
    input_items = _as_int(row.get("input_items")) or _as_int(window.get("input_items"))
    signal = _event_signal(store_obj, row, input_chars, terminal_features, window)
    repeated_scaffold_saved = _already_crunched_scaffold_chars(crunch)
    safety_diagnostics = _safety_preserve_diagnostics(terminal_features)
    workflow_phase = _workflow_phase(window, routing, crunch, cache)
    cache_status = _status_from_cache(cache)
    non_text = _has_reason([routing, crunch, cache], {"non-text-input"}) or (input_items > 0 and input_chars <= 0)
    action_like = _has_reason([routing, crunch, cache], {"action-like-params"})
    stale_risk = _has_reason([routing, crunch, cache], _STALE_REASONS)
    error_window = bool(_as_int(row.get("error_code")) or _as_int(signal.get("error_count")))
    text_bucket = _text_bucket(input_chars)
    return {
        "row_index": index,
        "created_bucket": str(row.get("created_at") or "")[:13],
        "source_surface": CODEX_APP_SOURCE_SURFACE,
        "app_family": "codex",
        "granularity": "agent_turn",
        "workflow_phase": workflow_phase,
        "text_bucket": text_bucket,
        "input_size_bucket": text_bucket,
        "input_chars": input_chars,
        "input_tokens": max(0, input_chars // TOKEN_CHARS),
        "input_items": input_items,
        "terminal_fraction_bucket": signal["terminal_fraction_bucket"],
        "terminal_output_char_fraction_bucket": signal["terminal_fraction_bucket"],
        "terminal_event_count_bucket": _count_bucket(signal["event_terminal_count"]),
        "terminal_event_count": signal["event_terminal_count"],
        "terminal_event_message_chars": signal["event_terminal_chars"],
        "terminal_signal_source": signal["terminal_signal_source"],
        "estimated_terminal_chars": signal["estimated_terminal_chars"],
        "cache_status": cache_status,
        "already_crunched_repeated_scaffold": repeated_scaffold_saved > 0,
        "already_crunched_repeated_scaffold_chars": repeated_scaffold_saved,
        "safety_preserve_diagnostics": safety_diagnostics,
        "requested_model": routing.get("requested_model") or routing.get("routed_model") or codex_app_model(),
        "terminal_features": terminal_features,
        "base_blockers": {
            "non-text-input": non_text,
            "action-like-params": action_like,
            "missing-phase": workflow_phase == "unknown",
            "non-tool-execution-phase": workflow_phase not in {"unknown", "tool_execution"},
            "no-terminal-signal": signal["estimated_terminal_chars"] <= 0 or signal["terminal_signal_source"] == "none",
            "stale-risk-blockers": stale_risk,
            "error-window": error_window,
        },
    }


def _condition_blockers(rule: dict[str, Any], features: dict[str, Any], projected_saved_chars: int) -> list[str]:
    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    blockers: list[str] = []
    for key in sorted(set(conditions) - _SUPPORTED_CONDITIONS):
        blockers.append(f"unsupported-condition:{public_label(key, 'unknown')}")
    for key, expected in conditions.items():
        if key not in _SUPPORTED_CONDITIONS:
            continue
        if key == "min_input_chars":
            if features["input_chars"] < _as_int(expected):
                blockers.append("below-min-input-chars")
            continue
        if key == "min_terminal_chars":
            if features["estimated_terminal_chars"] < _as_int(expected):
                blockers.append("below-min-terminal-chars")
            continue
        if key == "min_projected_saved_chars":
            if projected_saved_chars < _as_int(expected):
                blockers.append("below-min-projected-savings")
            continue
        actual = features.get(key)
        if actual is None or actual == "":
            blockers.append(f"insufficient-metadata:{key}")
            continue
        if not _value_matches(expected, actual):
            blockers.append(f"condition-mismatch:{key}")
    return blockers


def _bucket_midpoint(bucket: Any) -> int:
    text = str(bucket or "zero")
    if text in {"", "zero", "none", "0"}:
        return 0
    if text in {"one", "1"}:
        return 1
    if text in {"2_5", "two_to_five"}:
        return 3
    if text in {"6_10", "six_to_ten"}:
        return 8
    if text in {"gte_11", "11_plus", "1000_plus"}:
        return 12
    return 1


def _diagnostic_line_estimate(features: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    terminal = features.get("terminal_features") if isinstance(features.get("terminal_features"), dict) else {}
    classes = terminal.get("class_count_buckets") if isinstance(terminal.get("class_count_buckets"), dict) else {}
    counts = {
        "stack_trace": _bucket_midpoint(classes.get("stack_trace")) + int(bool(terminal.get("stack_trace_present"))),
        "test_output": _bucket_midpoint(classes.get("test_output")) + int(bool(terminal.get("test_output_present"))),
        "build_output": _bucket_midpoint(classes.get("build_output")),
        "error_line": _bucket_midpoint(terminal.get("error_line_count_bucket")),
    }
    return sum(counts.values()), counts


def _estimate_plan(rule: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    before_terminal_chars = max(0, _as_int(features.get("estimated_terminal_chars")))
    event_count = max(0, _as_int(features.get("terminal_event_count")))
    line_count = max(event_count, before_terminal_chars // DEFAULT_AVG_TERMINAL_LINE_CHARS)
    head_lines = _as_int(action.get("head_lines"), 12)
    tail_lines = _as_int(action.get("tail_lines"), 16)
    diagnostic_lines, diagnostic_counts = _diagnostic_line_estimate(features)
    max_evidence = _as_int(action.get("max_evidence_lines"), 80)
    preserved_diagnostic_lines = min(max(0, max_evidence), diagnostic_lines)
    preserved_line_count = min(max(0, line_count), max(0, head_lines) + max(0, tail_lines) + preserved_diagnostic_lines)
    after_terminal_chars = min(
        before_terminal_chars,
        preserved_line_count * DEFAULT_AVG_TERMINAL_LINE_CHARS + 180,
    )
    target_saved_chars = max(0, before_terminal_chars - after_terminal_chars)
    repeated_saved = _as_int(features.get("already_crunched_repeated_scaffold_chars"))
    saved_chars = max(0, target_saved_chars - min(target_saved_chars, repeated_saved))
    after_input_chars = max(0, _as_int(features.get("input_chars")) - saved_chars)
    saved_tokens = saved_chars // TOKEN_CHARS
    saved_usd = estimate_cost(
        str(features.get("requested_model") or codex_app_model()),
        saved_tokens,
        0,
        provider="openai",
        processing_mode=codex_app_processing_mode(),
    ) or 0.0
    return {
        "before_chars": _as_int(features.get("input_chars")),
        "after_chars": after_input_chars,
        "target_before_chars": before_terminal_chars,
        "target_after_chars": max(0, before_terminal_chars - saved_chars),
        "projected_saved_chars": saved_chars,
        "projected_saved_tokens": saved_tokens,
        "projected_saved_usd": round(float(saved_usd), 8),
        "line_count_estimate": line_count,
        "preserved_head_line_count": min(max(0, head_lines), line_count),
        "preserved_tail_line_count": min(max(0, tail_lines), line_count),
        "preserved_diagnostic_line_count": preserved_diagnostic_lines,
        "preserved_line_count_estimate": preserved_line_count,
        "omitted_line_count_estimate": max(0, line_count - preserved_line_count),
        "source_diagnostic_counts_estimate": diagnostic_counts,
    }


def _cohort(rule: dict[str, Any], features: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    canary = rule.get("canary") if isinstance(rule.get("canary"), dict) else {}
    enabled = canary.get("enabled")
    if enabled is False:
        return {
            "cohort": "applied",
            "status": "planned",
            "reason": "canary-disabled",
            "fraction": 1.0,
            "holdout_fraction": 0.0,
            "sample_unit": "none",
            "sample_bucket": None,
            "hash_basis": "not-used",
            "raw_basis_included": False,
        }
    fraction = _bounded_fraction(canary.get("fraction", canary.get("canary_fraction")), 1.0)
    holdout = _bounded_fraction(canary.get("holdout_fraction"), 0.0)
    salt = str(canary.get("salt") or "codex-terminal-transcript-compaction-dry-run")
    unit = _normalized(canary.get("unit") or canary.get("canary_unit") or "source_hash")
    if unit not in {"source_hash", "thread_id", "model_and_size"}:
        unit = "source_hash"
    material = stable_json({
        "candidate_id": candidate_id,
        "unit": unit,
        "features": {
            "row_index": features.get("row_index"),
            "created_bucket": features.get("created_bucket"),
            "source_surface": features.get("source_surface"),
            "workflow_phase": features.get("workflow_phase"),
            "text_bucket": features.get("text_bucket"),
            "terminal_fraction_bucket": features.get("terminal_fraction_bucket"),
            "terminal_event_count_bucket": features.get("terminal_event_count_bucket"),
        },
    })
    digest = hashlib.sha256(f"{salt}\0{unit}\0{material}".encode("utf-8")).hexdigest()
    sample = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if sample < holdout:
        cohort = "holdout"
        status = "holdout"
        reason = "canary-holdout"
    elif sample < min(1.0, holdout + fraction):
        cohort = "canary_applied"
        status = "planned"
        reason = "canary-applied"
    else:
        cohort = "not_selected"
        status = "skipped"
        reason = "canary-not-selected"
    return {
        "cohort": cohort,
        "status": status,
        "reason": reason,
        "fraction": fraction,
        "holdout_fraction": holdout,
        "sample_unit": unit,
        "sample_bucket": round(sample, 6),
        "hash_basis": "local-only-salted-policy-sample",
        "raw_basis_included": False,
    }


def _plan_entry(rule: dict[str, Any], rule_index: int, features: dict[str, Any], blockers: list[str], plan_stats: dict[str, Any]) -> dict[str, Any]:
    candidate_policy_id = _candidate_policy_id(rule, rule_index)
    basis = {
        "policy_candidate_id": candidate_policy_id,
        "row_index": features.get("row_index"),
        "source_surface": features.get("source_surface"),
        "workflow_phase": features.get("workflow_phase"),
        "text_bucket": features.get("text_bucket"),
        "terminal_fraction_bucket": features.get("terminal_fraction_bucket"),
        "terminal_event_count_bucket": features.get("terminal_event_count_bucket"),
        "terminal_signal_source": features.get("terminal_signal_source"),
    }
    candidate_id = "codex-terminal-transcript-dry-run:" + _hash_basis(basis)
    cohort = _cohort(rule, features, candidate_id) if not blockers else {
        "cohort": "blocked",
        "status": "blocked",
        "reason": "blocked",
        "fraction": None,
        "holdout_fraction": None,
        "sample_unit": None,
        "sample_bucket": None,
        "hash_basis": "not-applicable",
        "raw_basis_included": False,
    }
    status = cohort["status"] if not blockers else "blocked"
    if status == "skipped":
        blockers = [cohort["reason"]]
    target_id = "codex-terminal-transcript-target:" + _hash_basis({"candidate_id": candidate_id, "target": "terminal-window"})
    return {
        "schema": PLAN_SCHEMA,
        "candidate_id": candidate_id,
        "policy_candidate_id": candidate_policy_id,
        "rule_id": _rule_id(rule, rule_index),
        "action_id": _action_id(rule),
        "policy_source": public_label(rule.get("policy_source") or "local-default", "unknown"),
        "status": status,
        "reason": cohort["reason"] if not blockers else blockers[0],
        "blockers": sorted(set(blockers)),
        "dry_run": True,
        "mutation_applied": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "source_surface": features["source_surface"],
        "app_family": "codex",
        "granularity": "agent_turn",
        "workflow_phase": features["workflow_phase"],
        "text_bucket": features["text_bucket"],
        "terminal_fraction_bucket": features["terminal_fraction_bucket"],
        "terminal_event_count_bucket": features["terminal_event_count_bucket"],
        "terminal_signal_source": features["terminal_signal_source"],
        "cache_status": features["cache_status"],
        "target_count": 1 if not blockers else 0,
        "terminal_event_count": features["terminal_event_count"],
        "terminal_event_message_chars": features["terminal_event_message_chars"],
        "before_chars": plan_stats["before_chars"],
        "after_chars": plan_stats["after_chars"] if not blockers else plan_stats["before_chars"],
        "projected_saved_chars": plan_stats["projected_saved_chars"] if not blockers else 0,
        "projected_saved_tokens": plan_stats["projected_saved_tokens"] if not blockers else 0,
        "projected_saved_usd": plan_stats["projected_saved_usd"] if not blockers else 0.0,
        "canary": cohort,
        "target_summaries": [] if blockers else [
            {
                "target_id": target_id,
                "kind": "terminal_transcript_window",
                "before_chars": plan_stats["target_before_chars"],
                "after_chars": plan_stats["target_after_chars"],
                "saved_chars": plan_stats["projected_saved_chars"],
                "estimated_saved_tokens": plan_stats["projected_saved_tokens"],
                "line_count_estimate": plan_stats["line_count_estimate"],
                "preserved_head_line_count": plan_stats["preserved_head_line_count"],
                "preserved_tail_line_count": plan_stats["preserved_tail_line_count"],
                "preserved_diagnostic_line_count": plan_stats["preserved_diagnostic_line_count"],
                "preserved_line_count_estimate": plan_stats["preserved_line_count_estimate"],
                "omitted_line_count_estimate": plan_stats["omitted_line_count_estimate"],
                "source_diagnostic_counts_estimate": plan_stats["source_diagnostic_counts_estimate"],
                "preservation_flags": {
                    "tool_protocol_preserved": True,
                    "recent_turns_preserved": True,
                    "diagnostics_preserved": bool(rule.get("action", {}).get("preserve_diagnostics", True)),
                    "error_lines_preserved": bool(rule.get("action", {}).get("preserve_error_lines", True)),
                },
            }
        ],
    }


def build_codex_terminal_transcript_compaction_dry_run(
    store_obj: Any,
    *,
    limit: int = 500,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capped_limit = max(1, min(int(limit or 500), 10_000))
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
    rules = _policy_rules(policy)
    plans: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    planned_saved_chars = 0
    planned_saved_tokens = 0
    planned_saved_usd = 0.0

    if not rules:
        blocker_counts["policy-unavailable"] = len(rows)

    for row_index, row in enumerate(rows):
        features = _row_features(store_obj, row, row_index)
        base_blockers = [key for key, active in features["base_blockers"].items() if active]
        if not rules:
            plan_stats = {
                "before_chars": features["input_chars"],
                "after_chars": features["input_chars"],
                "target_before_chars": features["estimated_terminal_chars"],
                "target_after_chars": features["estimated_terminal_chars"],
                "projected_saved_chars": 0,
                "projected_saved_tokens": 0,
                "projected_saved_usd": 0.0,
                "line_count_estimate": 0,
                "preserved_head_line_count": 0,
                "preserved_tail_line_count": 0,
                "preserved_diagnostic_line_count": 0,
                "preserved_line_count_estimate": 0,
                "omitted_line_count_estimate": 0,
                "source_diagnostic_counts_estimate": {},
            }
            continue
        for rule_index, rule in enumerate(rules):
            plan_stats = _estimate_plan(rule, features)
            blockers = list(base_blockers)
            action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            if str(action.get("type") or "compact_terminal_transcript") != "compact_terminal_transcript":
                blockers.append("unsupported-action")
            if features["safety_preserve_diagnostics"] and not bool(action.get("preserve_diagnostics", True)):
                blockers.append("diagnostics-preservation-disabled")
            if plan_stats["projected_saved_chars"] < _as_int(action.get("min_saved_chars"), 0):
                blockers.append("below-min-saved-chars")
            blockers.extend(_condition_blockers(rule, features, plan_stats["projected_saved_chars"]))
            plan = _plan_entry(rule, rule_index, features, blockers, plan_stats)
            plans.append(plan)
            for blocker in plan["blockers"]:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
            cohort = (plan.get("canary") or {}).get("cohort") or "unknown"
            cohort_counts[str(cohort)] = cohort_counts.get(str(cohort), 0) + 1
            status = str(plan.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in {"planned", "holdout"}:
                planned_saved_chars += _as_int(plan.get("projected_saved_chars"))
                planned_saved_tokens += _as_int(plan.get("projected_saved_tokens"))
                planned_saved_usd += _as_float(plan.get("projected_saved_usd"))

    plans.sort(
        key=lambda item: (
            1 if item.get("status") == "planned" else 0,
            1 if item.get("status") == "holdout" else 0,
            _as_int(item.get("projected_saved_chars")),
        ),
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": True,
        "read_only": True,
        "generated_at": utc_now(),
        "lookback_turn_limit": capped_limit,
        "policy": {
            "schema": "agentflow.codex_terminal_transcript_compaction_policy.v1",
            "rule_count": len(rules),
            "runtime_mutation_enabled": False,
            "default_apply": False,
            "review_only": True,
            "raw_policy_file_contents_included": False,
        },
        "summary": {
            "scanned_turn_count": len(rows),
            "evaluated_policy_rule_count": len(rules),
            "plan_count": len(plans),
            "planned_candidate_count": sum(1 for item in plans if item.get("status") in {"planned", "holdout"}),
            "applied_candidate_count": status_counts.get("planned", 0),
            "holdout_candidate_count": status_counts.get("holdout", 0),
            "blocked_candidate_count": status_counts.get("blocked", 0),
            "skipped_candidate_count": status_counts.get("skipped", 0),
            "projected_saved_chars": planned_saved_chars,
            "projected_saved_tokens": planned_saved_tokens,
            "projected_saved_usd": round(planned_saved_usd, 8),
        },
        "status_breakdown": _breakdown(status_counts, label="status"),
        "cohort_breakdown": _breakdown(cohort_counts, label="cohort"),
        "blocker_reason_breakdown": _breakdown(blocker_counts),
        "plans": plans,
        "privacy": {
            "metadata_only_output": True,
            "content_free": True,
            "raw_terminal_text_included": False,
            "raw_terminal_lines_included": False,
            "raw_commands_included": False,
            "raw_command_text_included": False,
            "raw_prompts_included": False,
            "raw_request_bodies_included": False,
            "raw_provider_bodies_included": False,
            "raw_responses_included": False,
            "tool_payloads_included": False,
            "file_paths_included": False,
            "cache_keys_included": False,
            "request_ids_included": False,
            "raw_request_ids_included": False,
            "thread_ids_included": False,
            "session_ids_included": False,
            "provider_calls_made": False,
            "managed_server_calls_made": False,
            "basis": "local Codex turn/start metadata, terminal feature buckets, and server-to-client event counts only",
        },
    }
