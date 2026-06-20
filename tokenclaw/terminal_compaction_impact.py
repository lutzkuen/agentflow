from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from tokenclaw.pricing import estimate_cost
from tokenclaw.public_metadata import public_id, public_label
from tokenclaw.store import utc_now


SCHEMA = "agentflow.terminal_output_compaction_impact.v1"
ACTION_SCHEMA = "agentflow.terminal_output_compaction_rollback_action.v1"
TOKEN_CHARS = 4

_REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")


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


def _reason_code(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return text if _REASON_CODE_RE.match(text) else fallback


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


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


def _model_family(model: Any, provider: Any) -> str:
    text = str(model or "").lower()
    if "haiku" in text:
        return "haiku"
    if "sonnet" in text:
        return "sonnet"
    if "opus" in text:
        return "opus"
    if str(provider or "").lower() == "openai":
        if "mini" in text:
            return "mini"
        if text:
            return text.split("-", 2)[0]
    return "unknown"


def _source_surface(row: dict[str, Any], routing: dict[str, Any]) -> str:
    value = row.get("source_surface") or routing.get("source_surface")
    if value:
        return str(value)
    provider = str(row.get("provider") or "anthropic").lower()
    if provider == "anthropic":
        return "anthropic_messages"
    if provider == "openai":
        path = str(row.get("path") or "")
        if "chat/completions" in path:
            return "openai_chat_completions"
        if "responses" in path:
            return "openai_responses"
        return "openai"
    return "unknown"


def _privacy_summary() -> dict[str, Any]:
    return {
        "metadata_only": True,
        "aggregate_only": True,
        "content_free": True,
        "raw_prompts_included": False,
        "raw_messages_included": False,
        "raw_request_bodies_included": False,
        "raw_provider_bodies_included": False,
        "raw_responses_included": False,
        "raw_terminal_text_included": False,
        "raw_tool_payloads_included": False,
        "tool_payloads_included": False,
        "file_paths_included": False,
        "filesystem_paths_included": False,
        "cache_keys_included": False,
        "request_ids_included": False,
        "session_ids_included": False,
        "raw_session_ids_included": False,
        "request_fingerprints_included": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "wrote_local_files": False,
        "wrote_store": False,
    }


def _empty_cohort() -> dict[str, Any]:
    return {
        "count": 0,
        "error_count": 0,
        "retry_rows": 0,
        "retry_attempts": 0,
        "latency_ms_total": 0,
        "latency_sample_count": 0,
        "cost_est_usd": 0.0,
        "cost_baseline_usd": 0.0,
        "tokens_saved_est": 0,
        "planned_saved_tokens": 0,
        "gross_savings_usd": 0.0,
        "compaction_cost_usd": 0.0,
        "net_savings_usd": 0.0,
        "non_positive_savings_count": 0,
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
        "error_count": _as_int(raw.get("error_count")),
        "retry_rows": _as_int(raw.get("retry_rows")),
        "retry_attempts": _as_int(raw.get("retry_attempts")),
        "error_rate": round(_as_int(raw.get("error_count")) / count, 6) if count else 0.0,
        "retry_rate": round(_as_int(raw.get("retry_rows")) / count, 6) if count else 0.0,
        "latency_avg_ms": round(_as_int(raw.get("latency_ms_total")) / latency_samples, 2) if latency_samples else None,
        "cost_avg_usd": round(_as_float(raw.get("cost_est_usd")) / count, 8) if count else 0.0,
        "cost_est_usd": round(_as_float(raw.get("cost_est_usd")), 8),
        "cost_baseline_usd": round(_as_float(raw.get("cost_baseline_usd")), 8),
        "tokens_saved_est": _as_int(raw.get("tokens_saved_est")),
        "planned_saved_tokens": _as_int(raw.get("planned_saved_tokens")),
        "gross_savings_usd": round(_as_float(raw.get("gross_savings_usd")), 8),
        "compaction_cost_usd": round(_as_float(raw.get("compaction_cost_usd")), 8),
        "net_savings_usd": round(_as_float(raw.get("net_savings_usd")), 8),
        "net_savings_per_call_usd": round(_as_float(raw.get("net_savings_usd")) / count, 8) if count else 0.0,
        "non_positive_savings_count": _as_int(raw.get("non_positive_savings_count")),
        "non_positive_savings_rate": round(_as_int(raw.get("non_positive_savings_count")) / count, 6) if count else 0.0,
        "status_breakdown": _counter_rows(status_counts),
        "reason_breakdown": _counter_rows(reason_counts),
    }


def _row_terminal_meta(row: dict[str, Any]) -> dict[str, Any]:
    crunch = _json_obj(row.get("crunch_json"))
    meta = crunch.get("terminal_output_compaction")
    return meta if isinstance(meta, dict) else {}


def _cohort(meta: dict[str, Any]) -> str:
    canary = meta.get("canary") if isinstance(meta.get("canary"), dict) else {}
    cohort = str(canary.get("cohort") or canary.get("status") or "").strip().lower()
    status = str(meta.get("status") or "").strip().lower()
    reason = str(meta.get("reason") or "").strip().lower()
    if cohort == "canary_applied" or status == "applied" or bool(meta.get("applied")):
        return "applied"
    if cohort == "canary_holdout" or status == "holdout" or reason == "canary_holdout":
        return "holdout"
    if "safety-stop" in reason or status == "safety_stop" or meta.get("safety_stop_state") == "stopped":
        return "safety_stop"
    return "skipped"


def _call_rows(store_obj: Any, *, limit: int, since: str | None) -> list[dict[str, Any]]:
    if since:
        sql = """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, status_code,
                   latency_ms, cost_est_usd, cost_baseline_usd, category, retry_count,
                   actual_input_tokens, actual_output_tokens, input_tokens_est,
                   crunch_json, routing_json, cache_json
            from calls
            where created_at >= ?
            order by created_at desc
            limit ?
        """
        params: tuple[Any, ...] = (since, max(1, min(int(limit or 500), 10_000)))
    else:
        sql = """
            select id, created_at, path, coalesce(provider, 'anthropic') as provider,
                   source_surface, endpoint, requested_model, routed_model,
                   requested_model_family, routed_model_family, stream, status_code,
                   latency_ms, cost_est_usd, cost_baseline_usd, category, retry_count,
                   actual_input_tokens, actual_output_tokens, input_tokens_est,
                   crunch_json, routing_json, cache_json
            from calls
            order by created_at desc
            limit ?
        """
        params = (max(1, min(int(limit or 500), 10_000)),)
    rows = [dict(row) for row in store_obj.conn.execute(sql, params).fetchall()]
    return list(reversed(rows))


def _new_aggregate(parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_key": "|".join(_reason_code(parts.get(key), "unknown") for key in (
            "provider",
            "source_surface",
            "category",
            "workflow_phase",
            "requested_model_family",
            "routed_model_family",
            "rule_id",
            "candidate_id",
        )),
        **parts,
        "cohorts": {
            "applied": _empty_cohort(),
            "holdout": _empty_cohort(),
            "skipped": _empty_cohort(),
            "safety_stop": _empty_cohort(),
        },
        "first_observed_at": None,
        "last_observed_at": None,
    }


def _group_parts(row: dict[str, Any], routing: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    provider = str(row.get("provider") or "anthropic").lower()
    source_surface = _source_surface(row, routing)
    requested_model = row.get("requested_model")
    routed_model = row.get("routed_model") or requested_model
    rule_id = public_label(meta.get("rule_id") or "local-terminal-output-compaction-canary", "unknown")
    candidate = public_id(meta.get("candidate_id") or rule_id, prefix="terminal-compaction-candidate", fallback=rule_id)
    return {
        "provider": public_label(provider, "unknown"),
        "source_surface": public_label(source_surface, "unknown"),
        "endpoint": public_label(row.get("endpoint") or "messages", "unknown"),
        "category": public_label(meta.get("category") or row.get("category") or routing.get("category") or "unknown", "unknown"),
        "workflow_phase": public_label(routing.get("workflow_phase") or routing.get("phase") or row.get("category") or "unknown", "unknown"),
        "requested_model_family": public_label(row.get("requested_model_family") or _model_family(requested_model, provider), "unknown"),
        "routed_model_family": public_label(row.get("routed_model_family") or _model_family(routed_model, provider), "unknown"),
        "stream": bool(_as_int(row.get("stream"))),
        "policy_source": public_label(meta.get("policy_source") or "unknown", "unknown"),
        "rule_id": rule_id,
        "candidate_id": candidate or rule_id,
    }


def _input_savings_usd(row: dict[str, Any], tokens_saved: int) -> float:
    provider = str(row.get("provider") or "anthropic").lower()
    model = str(row.get("routed_model") or row.get("requested_model") or "")
    return estimate_cost(model, max(0, tokens_saved), 0, provider=provider) or 0.0


def _add_row(aggregate: dict[str, Any], row: dict[str, Any], meta: dict[str, Any], cohort_name: str) -> None:
    cohort = aggregate["cohorts"].setdefault(cohort_name, _empty_cohort())
    cohort["count"] += 1
    status_code = _as_int(row.get("status_code"), -1)
    retry_count = _as_int(row.get("retry_count"))
    latency_ms = _as_int(row.get("latency_ms"), -1)
    errored = status_code >= 400 if status_code >= 0 else False
    cohort["error_count"] += int(errored)
    cohort["retry_rows"] += int(retry_count > 0)
    cohort["retry_attempts"] += retry_count
    if latency_ms >= 0:
        cohort["latency_ms_total"] += latency_ms
        cohort["latency_sample_count"] += 1
    cohort["cost_est_usd"] += _as_float(row.get("cost_est_usd"))
    cohort["cost_baseline_usd"] += _as_float(row.get("cost_baseline_usd"))
    cohort["status_counts"][_status_bucket(status_code)] += 1
    cohort["reason_counts"][_reason_code(meta.get("reason"), "unknown")] += 1

    tokens_saved = _as_int(meta.get("tokens_saved_est"))
    planned_tokens = _as_int(meta.get("planned_saved_tokens"))
    if tokens_saved <= 0 and _as_int(meta.get("saved_chars")) > 0:
        tokens_saved = max(0, _as_int(meta.get("saved_chars")) // TOKEN_CHARS)
    if planned_tokens <= 0 and _as_int(meta.get("planned_saved_chars")) > 0:
        planned_tokens = max(0, _as_int(meta.get("planned_saved_chars")) // TOKEN_CHARS)
    if cohort_name == "applied" and planned_tokens <= 0:
        planned_tokens = tokens_saved

    gross = _input_savings_usd(row, tokens_saved if cohort_name == "applied" else planned_tokens)
    compaction_cost = _as_float(meta.get("compaction_cost_usd"))
    net = gross - compaction_cost if cohort_name == "applied" else 0.0
    cohort["tokens_saved_est"] += tokens_saved
    cohort["planned_saved_tokens"] += planned_tokens
    cohort["gross_savings_usd"] += gross
    cohort["compaction_cost_usd"] += compaction_cost
    cohort["net_savings_usd"] += net
    if cohort_name == "applied" and net <= 0:
        cohort["non_positive_savings_count"] += 1

    observed_at = _parse_time(row.get("created_at"))
    if observed_at is not None:
        first = aggregate.get("first_observed_at")
        last = aggregate.get("last_observed_at")
        aggregate["first_observed_at"] = observed_at if first is None or observed_at < first else first
        aggregate["last_observed_at"] = observed_at if last is None or observed_at > last else last


def _delta(applied: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    applied_latency = applied.get("latency_avg_ms")
    holdout_latency = holdout.get("latency_avg_ms")
    latency_delta = None
    if applied_latency is not None and holdout_latency is not None:
        latency_delta = round(float(applied_latency) - float(holdout_latency), 2)
    return {
        "error_rate_delta": round(_as_float(applied.get("error_rate")) - _as_float(holdout.get("error_rate")), 6),
        "retry_rate_delta": round(_as_float(applied.get("retry_rate")) - _as_float(holdout.get("retry_rate")), 6),
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
    action_id = "terminal-output-compaction-rollback:" + hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema": ACTION_SCHEMA,
        "action_id": action_id,
        "action_type": "rollback-local-terminal-output-compaction-canary",
        "review_only": True,
        "wrote_local_files": False,
        "target_policy_family": "crunch",
        "target_rule_id": candidate.get("rule_id"),
        "target_candidate_id": candidate.get("candidate_id"),
        "recommended_local_policy_patch": {
            "terminal_output_compaction": {
                "enabled": True,
                "canary": {
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
    actions: list[dict[str, Any]] = []
    if applied["count"] < thresholds["min_applied_samples"]:
        reasons.append("insufficient-applied-samples")
    if holdout["count"] < thresholds["min_holdout_samples"]:
        reasons.append("insufficient-holdout-samples")
    if reasons:
        return "insufficient-evidence", reasons, actions

    rollback_reasons: list[str] = []
    if applied["error_rate"] >= thresholds["rollback_error_rate"]:
        rollback_reasons.append("rollback-absolute-error-rate")
    if deltas["error_rate_delta"] >= thresholds["max_error_rate_delta"]:
        rollback_reasons.append("rollback-error-rate-delta")
    if deltas["retry_rate_delta"] >= thresholds["max_retry_rate_delta"]:
        rollback_reasons.append("rollback-retry-rate-delta")
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
    if applied["non_positive_savings_rate"] > thresholds["max_non_positive_savings_rate"]:
        reasons.append("hold-non-positive-savings")
    if reasons:
        return "hold", reasons, actions
    return "promote", ["impact-positive"], actions


def _finalize_candidate(aggregate: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    cohorts = {
        name: _finalize_cohort(raw)
        for name, raw in aggregate["cohorts"].items()
    }
    candidate = {
        "candidate_id": aggregate["candidate_id"],
        "rule_id": aggregate["rule_id"],
        "policy_source": aggregate["policy_source"],
        "provider": aggregate["provider"],
        "source_surface": aggregate["source_surface"],
        "endpoint": aggregate["endpoint"],
        "category": aggregate["category"],
        "workflow_phase": aggregate["workflow_phase"],
        "requested_model_family": aggregate["requested_model_family"],
        "routed_model_family": aggregate["routed_model_family"],
        "stream": aggregate["stream"],
        "first_observed_at": aggregate["first_observed_at"].isoformat() if aggregate.get("first_observed_at") else None,
        "last_observed_at": aggregate["last_observed_at"].isoformat() if aggregate.get("last_observed_at") else None,
        "cohorts": cohorts,
        "deltas": _delta(cohorts["applied"], cohorts["holdout"]),
        "net_savings_usd": cohorts["applied"]["net_savings_usd"],
        "projected_holdout_savings_usd": cohorts["holdout"]["gross_savings_usd"],
        "privacy": _privacy_summary(),
    }
    verdict, reasons, actions = _verdict(candidate, thresholds)
    candidate["verdict"] = verdict
    candidate["reason_codes"] = reasons
    candidate["rollback_actions"] = actions
    return candidate


def build_terminal_output_compaction_impact_report(
    store_obj: Any,
    *,
    limit: int = 500,
    since: str | None = None,
    min_applied_samples: int = 2,
    min_holdout_samples: int = 1,
    max_error_rate: float = 0.05,
    max_error_rate_delta: float = 0.05,
    max_retry_rate_delta: float = 0.10,
    max_latency_regression_ms: int = 2_000,
    min_net_savings_usd: float = 0.0,
    max_non_positive_savings_rate: float = 0.0,
    rollback_error_rate: float = 0.20,
) -> dict[str, Any]:
    lookback_limit = max(1, min(int(limit or 500), 10_000))
    thresholds = {
        "min_applied_samples": max(0, _as_int(min_applied_samples)),
        "min_holdout_samples": max(0, _as_int(min_holdout_samples)),
        "max_error_rate": round(float(max_error_rate), 6),
        "max_error_rate_delta": round(float(max_error_rate_delta), 6),
        "max_retry_rate_delta": round(float(max_retry_rate_delta), 6),
        "max_latency_regression_ms": _as_int(max_latency_regression_ms),
        "min_net_savings_usd": round(float(min_net_savings_usd), 8),
        "max_non_positive_savings_rate": round(float(max_non_positive_savings_rate), 6),
        "rollback_error_rate": round(float(rollback_error_rate), 6),
    }
    rows = _call_rows(store_obj, limit=lookback_limit, since=since)
    aggregates: dict[str, dict[str, Any]] = {}
    observed_rows = 0
    cohort_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for row in rows:
        meta = _row_terminal_meta(row)
        if not meta:
            continue
        routing = _json_obj(row.get("routing_json"))
        parts = _group_parts(row, routing, meta)
        group_key = "|".join(_reason_code(parts.get(key), "unknown") for key in (
            "provider",
            "source_surface",
            "category",
            "workflow_phase",
            "requested_model_family",
            "routed_model_family",
            "rule_id",
            "candidate_id",
        ))
        aggregate = aggregates.setdefault(group_key, _new_aggregate(parts))
        cohort = _cohort(meta)
        _add_row(aggregate, row, meta, cohort)
        observed_rows += 1
        cohort_counts[cohort] += 1
        status_counts[_status_bucket(row.get("status_code"))] += 1

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
        "generated_at": utc_now(),
        "read_only": True,
        "wrote_local_files": False,
        "wrote_store": False,
        "provider_calls_made": False,
        "managed_server_calls_made": False,
        "lookback_limit": lookback_limit,
        "since": since,
        "thresholds": thresholds,
        "status": "matched" if observed_rows else "no-terminal-output-compaction-canary-metadata",
        "summary": {
            "sampled_call_count": len(rows),
            "observed_terminal_output_compaction_metadata_row_count": observed_rows,
            "candidate_group_count": len(candidates),
            "applied_count": cohort_counts.get("applied", 0),
            "holdout_count": cohort_counts.get("holdout", 0),
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
