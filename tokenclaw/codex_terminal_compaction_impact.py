from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.codex_terminal_transcript_compaction import FAMILY
from tokenclaw.crunch import TOKEN_CHARS
from tokenclaw.pricing import codex_app_model, estimate_cost
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "tokenclaw.codex_terminal_transcript_compaction_impact.v1"
ACTION_SCHEMA = "tokenclaw.codex_terminal_transcript_compaction_rollback_action.v1"


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


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _status_bucket(error_code: Any) -> str:
    code = _as_int(error_code, 0)
    if code == 0:
        return "completed"
    if code < 0:
        return "jsonrpc-error"
    return "error"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_terminal_text_included": False,
        "raw_terminal_lines_included": False,
        "raw_commands_included": False,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_tool_payloads_included": False,
        "tool_payloads_included": False,
        "request_ids_included": False,
        "thread_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "cache_keys_included": False,
        "file_paths_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_files": False,
        "wrote_store": False,
    }


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "completed_count": 0,
        "error_count": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "input_tokens_est": 0,
        "output_tokens_est": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "saved_chars": 0,
        "saved_tokens_est": 0,
        "planned_saved_chars": 0,
        "planned_saved_tokens": 0,
        "gross_savings_usd": 0.0,
        "net_savings_usd": 0.0,
        "negative_savings_count": 0,
        "coordinator_suppressed_count": 0,
        "status_counts": Counter(),
        "reason_counts": Counter(),
    }


def _finalize_cohort(raw: dict[str, Any]) -> dict[str, Any]:
    count = _as_int(raw.get("count"))
    latency_samples = _as_int(raw.get("latency_sample_count"))
    status_counts = raw.get("status_counts") if isinstance(raw.get("status_counts"), Counter) else Counter()
    reason_counts = raw.get("reason_counts") if isinstance(raw.get("reason_counts"), Counter) else Counter()
    return {
        "count": count,
        "completed_count": _as_int(raw.get("completed_count")),
        "error_count": _as_int(raw.get("error_count")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "cost_avg_usd": round(_as_float(raw.get("cost_est_usd")) / count, 8) if count else 0.0,
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd")), 8),
        "cost_baseline_usd": round(_as_float(raw.get("cost_baseline_usd")), 8),
        "input_tokens_est": _as_int(raw.get("input_tokens_est")),
        "output_tokens_est": _as_int(raw.get("output_tokens_est")),
        "saved_chars": _as_int(raw.get("saved_chars")),
        "saved_tokens_est": _as_int(raw.get("saved_tokens_est")),
        "planned_saved_chars": _as_int(raw.get("planned_saved_chars")),
        "planned_saved_tokens": _as_int(raw.get("planned_saved_tokens")),
        "gross_savings_usd": round(_as_float(raw.get("gross_savings_usd")), 8),
        "net_savings_usd": round(_as_float(raw.get("net_savings_usd")), 8),
        "net_savings_per_call_usd": round(_as_float(raw.get("net_savings_usd")) / count, 8) if count else 0.0,
        "negative_savings_count": _as_int(raw.get("negative_savings_count")),
        "negative_savings_rate": round(_as_int(raw.get("negative_savings_count")) / count, 6) if count else 0.0,
        "coordinator_suppressed_count": _as_int(raw.get("coordinator_suppressed_count")),
        "status_breakdown": _counter_rows(status_counts),
        "reason_breakdown": _counter_rows(reason_counts),
    }


def _terminal_meta(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    meta = crunch.get(FAMILY)
    return meta if isinstance(meta, dict) else {}


def _cohort(meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    cohort = str(canary.get("cohort") or canary.get("status") or "").strip().lower()
    status = str(meta.get("status") or "").strip().lower()
    reason = str(meta.get("reason") or "").strip().lower()
    if cohort == "canary_applied" or status == "applied" or bool(meta.get("applied")):
        return "applied"
    if cohort == "canary_holdout" or status == "holdout" or "holdout" in reason:
        return "holdout"
    if status in {"safety_stop", "safety-stopped"} or "safety-stop" in reason:
        return "safety_stop"
    if str(((meta.get("coordinator") or {}) if isinstance(meta.get("coordinator"), dict) else {}).get("suppressed_by") or ""):
        return "blocked"
    return "skipped"


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    where = "where s.direction = 'client_to_server' and s.method = 'turn/start'"
    params: list[Any] = []
    if since:
        where += " and s.created_at >= ?"
        params.append(since)
    params.append(max(1, min(int(limit or 500), 10_000)))
    rows = store_obj.conn.execute(
        f"""
        select s.id, s.created_at, s.message_chars, s.params_chars, s.input_items,
               s.input_text_chars, s.result_chars, s.error_code, s.latency_ms,
               s.routing_json, s.crunch_json, s.cache_json, s.event_window_json,
               (
                   select r.error_code from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_error_code,
               (
                   select r.latency_ms from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_latency_ms,
               (
                   select r.result_chars from codex_app_events r
                   where r.direction = 'server_to_client'
                     and r.request_id = s.request_id
                     and coalesce(r.session_id, '') = coalesce(s.session_id, '')
                   order by r.created_at desc
                   limit 1
               ) as response_result_chars
        from codex_app_events s
        {where}
        order by s.created_at desc
        limit ?
        """,
        tuple(params),
    ).fetchall()
    return list(reversed([dict(row) for row in rows]))


def _group_parts(row: dict[str, Any], routing: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    rule_id = public_label(meta.get("rule_id") or "local-codex-terminal-transcript-compaction", "unknown")
    candidate_id = public_id(
        meta.get("candidate_id") or canary.get("candidate_id") or rule_id,
        prefix="codex-terminal-transcript-candidate",
        fallback=rule_id,
    )
    workflow_phase = (
        routing.get("workflow_phase")
        or routing.get("phase")
        or _json_obj(row.get("event_window_json")).get("workflow_phase")
        or "unknown"
    )
    model = routing.get("routed_model") or routing.get("requested_model") or codex_app_model()
    return {
        "source_surface": "codex_turn",
        "app_family": "codex",
        "granularity": "agent_turn",
        "workflow_phase": public_label(workflow_phase, "unknown"),
        "category": public_label(routing.get("category") or "tool-execution", "unknown"),
        "requested_model": public_label(routing.get("requested_model") or model, "unknown"),
        "routed_model": public_label(model, "unknown"),
        "policy_source": public_label(meta.get("policy_source") or "local-manual", "unknown"),
        "rule_id": rule_id,
        "candidate_id": candidate_id or rule_id,
    }


def _input_savings_usd(model: str, saved_tokens: int) -> float:
    return estimate_cost(model, max(0, saved_tokens), 0, provider="openai") or 0.0


def _row_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return estimate_cost(model, max(0, input_tokens), max(0, output_tokens), provider="openai") or 0.0


def _add_row(aggregate: dict[str, Any], row: dict[str, Any], meta: dict[str, Any], cohort_name: str) -> None:
    cohort = aggregate["cohorts"].setdefault(cohort_name, _empty_cohort())
    cohort["count"] += 1
    error_code = row.get("response_error_code")
    errored = error_code is not None
    cohort["completed_count"] += int(not errored)
    cohort["error_count"] += int(errored)
    latency_ms = _as_int(row.get("response_latency_ms"), _as_int(row.get("latency_ms"), -1))
    if latency_ms >= 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
    input_tokens = max(0, (_as_int(meta.get("after_chars")) or _as_int(row.get("input_text_chars"))) // TOKEN_CHARS)
    output_tokens = max(0, _as_int(row.get("response_result_chars")) // TOKEN_CHARS)
    before_tokens = max(input_tokens, (_as_int(meta.get("before_chars")) or _as_int(row.get("input_text_chars"))) // TOKEN_CHARS)
    saved_chars = _as_int(meta.get("saved_chars"))
    saved_tokens = max(0, saved_chars // TOKEN_CHARS)
    planned_chars = _as_int(meta.get("planned_saved_chars"), saved_chars if cohort_name == "applied" else 0)
    planned_tokens = max(0, planned_chars // TOKEN_CHARS)
    model = str(aggregate.get("routed_model") or aggregate.get("requested_model") or codex_app_model())
    cost = _row_cost(model, input_tokens, output_tokens)
    baseline = _row_cost(model, before_tokens, output_tokens)
    gross = _input_savings_usd(model, saved_tokens if cohort_name == "applied" else planned_tokens)
    net = gross
    if cohort_name == "applied" and (saved_chars < 0 or net < 0):
        cohort["negative_savings_count"] += 1
    coordinator = meta.get("coordinator") if isinstance(meta.get("coordinator"), dict) else {}
    if coordinator.get("suppressed_by"):
        cohort["coordinator_suppressed_count"] += 1
    cohort["input_tokens_est"] += input_tokens
    cohort["output_tokens_est"] += output_tokens
    cohort["cost_est_usd"] += cost
    cohort["cost_baseline_usd"] += baseline
    cohort["saved_chars"] += saved_chars
    cohort["saved_tokens_est"] += saved_tokens
    cohort["planned_saved_chars"] += planned_chars
    cohort["planned_saved_tokens"] += planned_tokens
    cohort["gross_savings_usd"] += gross
    cohort["net_savings_usd"] += net
    cohort["status_counts"][_status_bucket(error_code)] += 1
    cohort["reason_counts"][public_label(meta.get("reason") or "unknown", "unknown")] += 1

    observed_at = _parse_time(row.get("created_at"))
    if observed_at is not None:
        first = aggregate.get("first_observed_at")
        last = aggregate.get("last_observed_at")
        aggregate["first_observed_at"] = observed_at if first is None or observed_at < first else first
        aggregate["last_observed_at"] = observed_at if last is None or observed_at > last else last


def _new_aggregate(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        **parts,
        "group_key": "|".join(str(parts.get(key) or "unknown") for key in (
            "source_surface",
            "workflow_phase",
            "requested_model",
            "routed_model",
            "rule_id",
            "candidate_id",
        )),
        "cohorts": {
            "applied": _empty_cohort(),
            "holdout": _empty_cohort(),
            "blocked": _empty_cohort(),
            "skipped": _empty_cohort(),
            "safety_stop": _empty_cohort(),
        },
        "first_observed_at": None,
        "last_observed_at": None,
    }


def _delta(applied: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    latency_delta = None
    if applied.get("latency_avg_ms") is not None and holdout.get("latency_avg_ms") is not None:
        latency_delta = round(float(applied["latency_avg_ms"]) - float(holdout["latency_avg_ms"]), 2)
    return {
        "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
        "latency_avg_ms_delta": latency_delta,
        "cost_avg_usd_delta": round(_as_float(applied.get("cost_avg_usd")) - _as_float(holdout.get("cost_avg_usd")), 8),
        "net_savings_per_call_usd_delta": round(_as_float(applied.get("net_savings_per_call_usd")), 8),
    }


def _rollback_action(candidate: dict[str, Any], reason_codes: list[str]) -> dict[str, Any]:
    basis = {
        "rule_id": candidate.get("rule_id"),
        "candidate_id": candidate.get("candidate_id"),
        "reason_codes": reason_codes,
    }
    action_id = "codex-terminal-transcript-rollback:" + hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema": ACTION_SCHEMA,
        "action_id": action_id,
        "action_type": "rollback-local-codex-terminal-transcript-compaction-canary",
        "review_only": True,
        "wrote_local_files": False,
        "target_policy_family": "codex_app",
        "target_rule_id": candidate.get("rule_id"),
        "target_candidate_id": candidate.get("candidate_id"),
        "recommended_local_policy_patch": {
            "terminal_transcript_compaction": {
                "enabled": True,
                "canary": {
                    "fraction": 0.0,
                    "canary_fraction": 0.0,
                    "holdout_fraction": 1.0,
                },
            }
        },
        "reason_codes": reason_codes,
        "trigger_metrics": candidate.get("deltas", {}),
        "privacy": _privacy_summary(),
    }


def _verdict(candidate: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    applied = candidate["cohorts"]["applied"]
    holdout = candidate["cohorts"]["holdout"]
    deltas = candidate["deltas"]
    reasons: list[str] = []
    if applied["count"] < thresholds["min_applied_samples"]:
        reasons.append("insufficient-applied-samples")
    if holdout["count"] < thresholds["min_holdout_samples"]:
        reasons.append("insufficient-holdout-samples")
    if reasons:
        return "insufficient-evidence", reasons, []

    rollback_reasons: list[str] = []
    if applied["error_rate"] >= thresholds["rollback_error_rate"]:
        rollback_reasons.append("rollback-absolute-error-rate")
    if deltas["error_rate_delta"] >= thresholds["max_error_rate_delta"]:
        rollback_reasons.append("rollback-error-rate-delta")
    if rollback_reasons:
        candidate["reason_codes"] = rollback_reasons
        return "rollback", rollback_reasons, [_rollback_action(candidate, rollback_reasons)]

    if applied["error_rate"] > thresholds["max_error_rate"]:
        reasons.append("hold-applied-error-rate")
    latency_delta = deltas.get("latency_avg_ms_delta")
    if latency_delta is not None and latency_delta > thresholds["max_latency_regression_ms"]:
        reasons.append("hold-latency-regression")
    if applied["net_savings_usd"] < thresholds["min_net_savings_usd"]:
        reasons.append("hold-minimum-net-savings")
    if applied["negative_savings_rate"] > thresholds["max_negative_savings_rate"]:
        reasons.append("hold-negative-savings")
    if reasons:
        return "hold", reasons, []
    return "promote", ["impact-positive"], []


def _finalize_candidate(aggregate: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {name: _finalize_cohort(raw) for name, raw in aggregate["cohorts"].items()}
    candidate = {
        "candidate_id": aggregate["candidate_id"],
        "rule_id": aggregate["rule_id"],
        "policy_source": aggregate["policy_source"],
        "source_surface": aggregate["source_surface"],
        "app_family": aggregate["app_family"],
        "granularity": aggregate["granularity"],
        "category": aggregate["category"],
        "workflow_phase": aggregate["workflow_phase"],
        "requested_model": aggregate["requested_model"],
        "routed_model": aggregate["routed_model"],
        "first_observed_at": aggregate["first_observed_at"].isoformat() if aggregate.get("first_observed_at") else None,
        "last_observed_at": aggregate["last_observed_at"].isoformat() if aggregate.get("last_observed_at") else None,
        "cohorts": cohorts,
        "deltas": _delta(cohorts["applied"], cohorts["holdout"]),
        "net_savings_usd": cohorts["applied"]["net_savings_usd"],
        "projected_holdout_savings_usd": cohorts["holdout"]["gross_savings_usd"],
        "coordinator_suppression_count": cohorts["blocked"]["coordinator_suppressed_count"],
        "privacy": _privacy_summary(),
    }
    verdict, reasons, actions = _verdict(candidate, thresholds)
    candidate["verdict"] = verdict
    candidate["reason_codes"] = reasons
    candidate["rollback_actions"] = actions
    return candidate


def build_codex_terminal_transcript_compaction_impact_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_error_rate: float = 0.05,
    max_error_rate_delta: float = 0.05,
    max_latency_regression_ms: int = 2_000,
    min_net_savings_usd: float = 0.0,
    max_negative_savings_rate: float = 0.0,
    rollback_error_rate: float = 0.20,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    thresholds = {
        "min_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "min_net_savings_usd": round(float(min_net_savings_usd), 8),
        "max_negative_savings_rate": round(float(max_negative_savings_rate), 6),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
    }
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    aggregates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    cohort_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for row in rows:
        meta = _terminal_meta(row)
        if not meta:
            continue
        routing = _json_obj(row.get("routing_json"))
        parts = _group_parts(row, routing, meta)
        group_key = "|".join(str(parts.get(key) or "unknown") for key in (
            "source_surface",
            "workflow_phase",
            "requested_model",
            "routed_model",
            "rule_id",
            "candidate_id",
        ))
        aggregate = aggregates.setdefault(group_key, _new_aggregate(parts))
        cohort = _cohort(meta)
        _add_row(aggregate, row, meta, cohort)
        observed_rows += 1
        cohort_counts[cohort] += 1
        status_counts[_status_bucket(row.get("response_error_code"))] += 1

    candidates = [_finalize_candidate(aggregate, thresholds) for aggregate in aggregates.values()]
    candidates.sort(
        key=lambda item: (
            {"rollback": 0, "hold": 1, "insufficient-evidence": 2, "promote": 3}.get(str(item.get("verdict")), 4),
            str(item.get("group_key") or item.get("candidate_id") or ""),
        )
    )
    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in candidates)
    reason_counts: Counter[str] = Counter()
    rollback_actions: list[dict[str, Any]] = []
    for item in candidates:
        for reason in item.get("reason_codes") or []:
            reason_counts[str(reason)] += 1
        rollback_actions.extend([action for action in item.get("rollback_actions") or [] if isinstance(action, dict)])

    return {
        "schema": SCHEMA,
        "ok": True,
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "since": since,
        "thresholds": thresholds,
        "status": "matched" if observed_rows else "no-codex-terminal-transcript-compaction-canary-metadata",
        "summary": {
            "sampled_turn_count": len(rows),
            "observed_codex_terminal_transcript_compaction_metadata_row_count": observed_rows,
            "candidate_group_count": len(candidates),
            "applied_count": cohort_counts.get("applied", 0),
            "holdout_count": cohort_counts.get("holdout", 0),
            "blocked_count": cohort_counts.get("blocked", 0),
            "skipped_count": cohort_counts.get("skipped", 0),
            "safety_stop_count": cohort_counts.get("safety_stop", 0),
            "net_savings_usd": round(sum(_as_float(item.get("net_savings_usd")) for item in candidates), 8),
            "projected_holdout_savings_usd": round(sum(_as_float(item.get("projected_holdout_savings_usd")) for item in candidates), 8),
            "rollback_action_count": len(rollback_actions),
            "verdict_counts": _counter_rows(verdict_counts),
            "reason_code_counts": _counter_rows(reason_counts),
            "status_breakdown": _counter_rows(status_counts),
        },
        "rollback_actions": rollback_actions,
        "candidates": candidates,
        "privacy": _privacy_summary(),
    }
